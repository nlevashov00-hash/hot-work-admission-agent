"""Batch pipeline and mandatory per-request audit."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.contracts import parse_batch, validate_results
from app.extraction import Client, Config, Extraction, OpenAIClient, extract
from app.rules import evaluate


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, '[REDACTED]') if secret else value
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {_redact(k, secret): _redact(v, secret) for k, v in value.items()}
    return value


def process_batch(batch: Any, client: Client | None, model: str, journal: Path, secret: str = '') -> list[dict[str, Any]]:
    # Validate before any request or journal is created.
    from app.contracts import validate_batch
    validate_batch(batch)
    journal.mkdir(parents=True, exist_ok=True)
    results = []
    for item in batch:
        started = datetime.now(timezone.utc).isoformat()
        extraction = (extract(item['text'], client) if client is not None
                      else Extraction(None, [], 0.0, error='configuration_error'))
        statuses = []
        if extraction.facts is None:
            result = extraction.failure_result(item['id'])
        else:
            statuses, result = evaluate(item['id'], extraction.facts)
        record = {'input': item, 'timestamp_utc': started, 'model': model,
                  'prompt_version': extraction.prompt_version, 'attempts': extraction.attempts,
                  'facts': extraction.facts, 'rule_statuses': statuses, 'result': result,
                  'extraction_error': extraction.error, 'elapsed_seconds': extraction.elapsed_seconds}
        data = json.dumps(_redact(record, secret), ensure_ascii=False, indent=2, allow_nan=False)
        # Exclusive creation protects previous runs even on an unlikely collision.
        with (journal / (uuid.uuid4().hex + '.json')).open('x', encoding='utf-8') as stream:
            stream.write(data)
        results.append(result)
    return validate_results(batch, results)


def run(input_path: Path, output_path: Path, journal: Path, client: Client | None = None, model: str = '', secret: str = '') -> None:
    batch = parse_batch(input_path.read_text(encoding='utf-8-sig'))
    if client is None:
        secret = os.getenv('LLM_API_KEY', '').strip()
        try:
            config = Config.from_env()
            model = config.model
            client = OpenAIClient(config)
        except Exception:
            client = None
    results = process_batch(batch, client, model, journal, secret)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output_path.parent, delete=False) as stream:
            temp_name = stream.name
            json.dump(results, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temp_name, output_path)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def main() -> int:
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Process fire-work applications')
    parser.add_argument('--data-dir', type=Path, default=Path('/data'))
    args = parser.parse_args()
    try:
        run(args.data_dir / 'input.json', args.data_dir / 'output.json', args.data_dir / 'journal')
    except Exception:
        print('Ошибка обработки пакета или записи аудита.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
