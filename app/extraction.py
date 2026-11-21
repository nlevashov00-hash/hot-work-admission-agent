"""Bounded extraction through an OpenAI-compatible chat completions endpoint."""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import os
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.contracts import parse_facts, validate_facts, validate_result
from app.prompt import PROMPT_VERSION, SYSTEM_PROMPT

MAX_TEXT_CHARS = 12000
TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 2
MAX_RETRY_WAIT_SECONDS = 30.0
MAX_RESPONSE_BYTES = 1_000_000
USER_AGENT = "HotWorkAdmissionAgent/1.0"


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str = field(repr=False)
    model: str

    @classmethod
    def from_env(cls) -> Config:
        values = [os.getenv(k, "").strip() for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")]
        if not all(values):
            raise ValueError("LLM configuration is incomplete")
        return cls(*values)


@dataclass
class Response:
    content: str
    usage: dict[str, Any] | None = None
    model: str | None = None


class Client(Protocol):
    def complete(self, text: str) -> Response: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


class OpenAIClient:
    def __init__(self, config: Config):
        parsed = urllib.parse.urlsplit(config.base_url)
        try:
            port = parsed.port
            host = parsed.hostname
        except ValueError as exc:
            raise ValueError("Invalid LLM base URL") from exc
        if (parsed.scheme not in ("http", "https") or not host or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or any(c.isspace() or ord(c) < 32 for c in config.base_url)
                or (parsed.scheme == "http" and host not in ("localhost", "127.0.0.1", "::1"))
                or (port is not None and not 1 <= port <= 65535)):
            raise ValueError("Invalid LLM base URL")
        if not config.api_key or not config.model:
            raise ValueError("LLM configuration is incomplete")
        self.config = config
        self.url = config.base_url.rstrip("/") + "/chat/completions"

    def complete(self, text: str) -> Response:
        payload = {"model": self.config.model, "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}], "temperature": 0}
        request = urllib.request.Request(self.url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + self.config.api_key, "Content-Type": "application/json", "User-Agent": USER_AGENT}, method="POST")
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Response too large")
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Invalid response content")
        usage = data.get("usage")
        return Response(content, usage if isinstance(usage, dict) else None, data.get("model") if isinstance(data.get("model"), str) else None)


@dataclass
class Extraction:
    facts: dict[str, Any] | None
    attempts: list[dict[str, Any]]
    elapsed_seconds: float
    prompt_version: str = PROMPT_VERSION
    error: str | None = None

    def failure_result(self, identifier: Any) -> dict[str, Any]:
        if self.facts is not None:
            raise ValueError("Extraction succeeded")
        return validate_result({"id": identifier, "decision": "ЭСКАЛИРОВАТЬ", "rules": ["П7"],
            "explanation": "Извлечение фактов не выполнено; требуется проверка человеком.",
            "missing": ["Достоверные сведения заявки"]})


def _retry_delay(exc: urllib.error.HTTPError, attempt: int) -> float:
    value = exc.headers.get('Retry-After') if exc.headers else None
    if value:
        try:
            seconds = float(value)
            if math.isfinite(seconds) and seconds >= 0:
                return seconds
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
                if when.tzinfo is not None:
                    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return (2 ** (attempt - 1)) + random.uniform(0, 1)


def _derive_proximity(facts: dict[str, Any]) -> dict[str, Any]:
    """Derive only missing proximity from a validated, unambiguous distance."""
    import copy
    normalized = copy.deepcopy(facts)
    for prefix in ("coal", "fuel"):
        near = normalized[f"{prefix}_store_near"]
        distance = normalized[f"{prefix}_store_distance_m"]
        if near["state"] != "unknown" or distance["state"] != "known":
            continue
        interval = distance["value"]
        if interval["max"] < 10:
            value = True
        elif interval["min"] >= 10:
            value = False
        else:
            continue
        near.update(state="known", value=value, evidence=distance["evidence"].copy())
    return validate_facts(normalized)


def extract(text: str, client: Client) -> Extraction:
    start = time.monotonic()
    attempts: list[dict[str, Any]] = []
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        return Extraction(None, attempts, time.monotonic() - start, error="invalid_text")
    waited = 0.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        began = time.monotonic()
        try:
            response = client.complete(text)
            if not isinstance(response, Response) or not isinstance(response.content, str):
                raise ValueError("Invalid client response")
            # Do not retain credentials even if echoed by a provider.
            secret = getattr(getattr(client, "config", None), "api_key", "")
            safe_raw = response.content.replace(secret, "[REDACTED]") if secret else response.content
            usage = response.usage if isinstance(response.usage, dict) else {}
            safe_usage = {k: v for k, v in usage.items()
                          if k in ("prompt_tokens", "completion_tokens", "total_tokens")
                          and type(v) is int and v >= 0}
            record = {"attempt": attempt, "raw_response": safe_raw, "usage": safe_usage or None, "model": response.model if response.model and (not secret or secret not in response.model) else None, "error": None}
            attempts.append(record)
            if secret and secret in response.content:
                raise ValueError("Response contains credential")
            facts = parse_facts(response.content)
            if any(fragment not in text for fact in facts.values() for fragment in fact["evidence"]):
                raise ValueError("Evidence is not in application")
            facts = _derive_proximity(facts)
            record["elapsed_seconds"] = time.monotonic() - began
            return Extraction(facts, attempts, time.monotonic() - start)
        except Exception as exc:
            # Only fixed categories and an HTTP integer are exposed, never exception text.
            status = exc.code if isinstance(exc, urllib.error.HTTPError) and type(exc.code) is int else None
            reason = exc.reason if isinstance(exc, urllib.error.URLError) and status is None else exc
            if status is not None:
                category = "http_error"
            elif isinstance(reason, (TimeoutError, socket.timeout)):
                category = "timeout"
            elif isinstance(exc, urllib.error.URLError) or isinstance(reason, (OSError, ssl.SSLError)):
                category = "network_error"
            else:
                category = "invalid_response" if len(attempts) and "elapsed_seconds" not in attempts[-1] else "unexpected_error"
            network_type = None
            for cls, label in ((socket.gaierror, "dns"), (ssl.SSLError, "tls"),
                               (PermissionError, "permission"), (ConnectionError, "connection"),
                               (TimeoutError, "timeout")):
                if isinstance(reason, cls):
                    network_type = label
                    break
            if category == "network_error" and network_type is None:
                network_type = "other"
            if len(attempts) == 0 or "elapsed_seconds" in attempts[-1]:
                attempts.append({"attempt": attempt, "raw_response": None, "usage": None, "model": None, "error": "request_failed"})
            else:
                attempts[-1]["error"] = "invalid_response"
            attempts[-1].update({"category": category, "http_status": status, "network_type": network_type})
            attempts[-1]["elapsed_seconds"] = time.monotonic() - began
            if status in (401, 403):
                break
            if status == 429 and attempt < MAX_ATTEMPTS:
                delay = _retry_delay(exc, attempt)
                if waited + delay > MAX_RETRY_WAIT_SECONDS:
                    attempts[-1].update({"retry_reason": "rate_limit_wait_exceeded", "wait_seconds": 0.0})
                    break
                attempts[-1].update({"retry_reason": "rate_limit", "wait_seconds": delay})
                time.sleep(delay)
                waited += delay
    return Extraction(None, attempts, time.monotonic() - start, error="extraction_failed")
