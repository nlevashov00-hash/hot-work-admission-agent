"""Validated data contracts for the admission workflow; no rule decisions here."""
from __future__ import annotations

import copy
import json
import math
from datetime import date
from typing import Any


class ContractError(ValueError):
    """Data does not meet the contract."""


def _object(value: Any, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContractError(f"Expected object with fields {sorted(keys)}")
    return value


def _string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _id(value: Any) -> bool:
    return (type(value) is int) or _string(value)


def _strings(value: Any, *, nonempty: bool = False) -> bool:
    return isinstance(value, list) and (not nonempty or bool(value)) and all(_string(x) for x in value)


def validate_batch(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list) or not data:
        raise ContractError("Batch must be a nonempty array")
    seen: set[tuple[type, Any]] = set()
    for item in data:
        _object(item, {"id", "text"})
        if not _id(item["id"]) or not isinstance(item["text"], str):
            raise ContractError("Invalid id or text")
        key = (type(item["id"]), item["id"])
        if key in seen:
            raise ContractError("Duplicate id")
        seen.add(key)
    return data


def parse_batch(raw: str) -> list[dict[str, Any]]:
    try:
        return validate_batch(json.loads(raw))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ContractError("Invalid batch JSON") from exc


# Negative facts use known + false. Unknown and contradictory facts have no value.
FACT_TYPES = {
    "permit_present": "bool", "permit_number": "string",
    "briefing_present": "bool", "briefing_date": "date",
    "coal_store_distance_m": "number", "fuel_store_distance_m": "number",
    "coal_store_near": "bool", "fuel_store_near": "bool",
    "written_approval": "bool", "extinguisher": "bool",
    "observer_present": "bool", "observer_independent": "bool",
    "location": "location", "wind_m_s": "number", "wind_source": "wind_source",
    "height_m": "number", "height_permit": "bool",
}


def _valid_value(value: Any, kind: str) -> bool:
    if kind == "bool":
        return type(value) is bool
    if kind == "string":
        return _string(value)
    if kind == "date":
        if not isinstance(value, str):
            return False
        try:
            return date.fromisoformat(value).isoformat() == value
        except ValueError:
            return False
    if kind == "location":
        return value in ("indoor", "outdoor")
    if kind == "wind_source":
        return value in ("forecast", "measured", "stated")
    if kind == "number":
        if not isinstance(value, dict) or set(value) != {"min", "max", "approximate"}:
            return False
        lo, hi = value["min"], value["max"]
        try:
            return (type(lo) in (int, float) and type(hi) in (int, float)
                    and math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi
                    and type(value["approximate"]) is bool
                    and (not value["approximate"] or lo < hi))
        except OverflowError:
            return False
    return False


def validate_facts(data: Any) -> dict[str, Any]:
    _object(data, set(FACT_TYPES))
    for name, kind in FACT_TYPES.items():
        fact = _object(data[name], {"state", "value", "evidence"})
        state, value, evidence = fact["state"], fact["value"], fact["evidence"]
        if state not in ("known", "unknown", "contradictory") or not _strings(evidence):
            raise ContractError(f"Invalid state or evidence: {name}")
        if state == "known":
            if not _valid_value(value, kind) or not evidence:
                raise ContractError(f"Invalid known fact: {name}")
        elif value is not None or (state == "contradictory" and len(evidence) < 2):
            raise ContractError(f"Invalid uncertain fact: {name}")
    return _reconcile_facts(data)


def _reconcile_facts(data: dict[str, Any]) -> dict[str, Any]:
    """Mark incompatible known claims as contradictory without changing input.

    Missing values and intervals crossing the proximity boundary are not conflicts.
    All conflicting claims contribute their evidence before any state is changed.
    """
    conflicts: list[tuple[str, str]] = []

    def known(name: str) -> bool:
        return data[name]["state"] == "known"

    for presence, detail in (("permit_present", "permit_number"),
                             ("briefing_present", "briefing_date")):
        if known(presence) and data[presence]["value"] is False and known(detail):
            conflicts.append((presence, detail))
    if (known("observer_present") and data["observer_present"]["value"] is False
            and known("observer_independent") and data["observer_independent"]["value"] is True):
        conflicts.append(("observer_present", "observer_independent"))
    for prefix in ("coal", "fuel"):
        near, distance = f"{prefix}_store_near", f"{prefix}_store_distance_m"
        if known(near) and known(distance):
            interval = data[distance]["value"]
            if ((data[near]["value"] is False and interval["max"] < 10)
                    or (data[near]["value"] is True and interval["min"] >= 10)):
                conflicts.append((near, distance))
    if not conflicts:
        return data
    reconciled = copy.deepcopy(data)
    for left, right in conflicts:
        evidence = data[left]["evidence"] + data[right]["evidence"]
        for name in (left, right):
            fact = reconciled[name]
            fact["state"] = "contradictory"
            fact["value"] = None
            fact["evidence"] = evidence.copy()
    return reconciled


def parse_facts(raw: str) -> dict[str, Any]:
    try:
        return validate_facts(json.loads(raw))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ContractError("Invalid facts JSON") from exc


def validate_rule_status(data: Any) -> dict[str, Any]:
    _object(data, {"rule", "status", "explanation", "missing"})
    if (data["rule"] not in [f"П{i}" for i in range(1, 8)]
        or data["status"] not in ("PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE", "HUMAN")
        or not _string(data["explanation"]) or not _strings(data["missing"])):
        raise ContractError("Invalid rule status")
    return data


def validate_result(data: Any) -> dict[str, Any]:
    _object(data, {"id", "decision", "rules", "explanation", "missing"})
    if (not _id(data["id"]) or data["decision"] not in ("ДОПУСТИТЬ", "ОТКАЗАТЬ", "ЭСКАЛИРОВАТЬ")
        or not _strings(data["rules"], nonempty=True)
        or any(rule not in [f"П{i}" for i in range(1, 8)] for rule in data["rules"])
        or len(set(data["rules"])) != len(data["rules"])
        or not _string(data["explanation"]) or not _strings(data["missing"])):
        raise ContractError("Invalid result")
    return data


def validate_results(batch: Any, results: Any) -> list[dict[str, Any]]:
    validate_batch(batch)
    if not isinstance(results, list) or len(results) != len(batch):
        raise ContractError("Results cannot be linked to batch")
    for item, result in zip(batch, results):
        validate_result(result)
        if type(item["id"]) is not type(result["id"]) or item["id"] != result["id"]:
            raise ContractError("Result id does not match input")
    return results
