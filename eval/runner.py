"""Evaluate the existing pipeline against separately labelled cases.

Matrix rows are reference decisions; columns are actual decisions or TECHNICAL_ERROR.
Technical failures occupy that column regardless of the returned decision. Accuracy
is correct decisions without technical failure divided by all labelled cases in the
group; unlabelled cases are excluded. Decision distribution and false admissions
use the actual returned decision; reference-rule comparisons are counted separately.
"""
import argparse
import json
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from app.contracts import validate_batch, validate_results
from app.extraction import Config, OpenAIClient
from app.pipeline import process_batch

DECISIONS = ('ДОПУСТИТЬ', 'ОТКАЗАТЬ', 'ЭСКАЛИРОВАТЬ')


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, list) or not data:
        raise ValueError('Dataset must be a nonempty array')
    for case in data:
        if not isinstance(case, dict) or set(case) != {'id', 'text', 'group', 'expected', 'rationale'}:
            raise ValueError('Invalid case fields')
        group, expected, rationale = case['group'], case['expected'], case['rationale']
        if group not in ('official', 'custom', 'unlabeled'):
            raise ValueError('Invalid group')
        if group == 'unlabeled':
            if expected is not None or rationale is not None:
                raise ValueError('Unlabeled case cannot have an expected result')
        elif not isinstance(expected, str) or not any(expected == d or expected.startswith(d + ' (') and expected.endswith(')') for d in DECISIONS):
            raise ValueError('Invalid expected decision')
        if group == 'custom' and (not isinstance(rationale, str) or not rationale.strip() or expected not in DECISIONS):
            raise ValueError('Custom rationale and decision required')
    validate_batch([{'id': c['id'], 'text': c['text']} for c in data])
    return data


def evaluate_cases(cases: list[dict[str, Any]], results: Any, errors: list[str | None]) -> dict[str, Any]:
    validate_results([{'id': c['id'], 'text': c['text']} for c in cases], results)
    if len(errors) != len(cases):
        raise ValueError('Missing extraction status')
    details = []
    for case, result, error in zip(cases, results, errors):
        expected = case['expected'].split(' (', 1)[0] if case['expected'] else None
        reference_rules = re.findall(r'П[1-7]', case['expected'] or '') if case['group'] == 'official' else []
        missing_rules = [r for r in reference_rules if r not in result['rules']]
        details.append({'id': case['id'], 'group': case['group'], 'expected': expected,
                        'reference': case['expected'], 'rationale': case['rationale'],
                        'actual': result, 'technical_error': error, 'expected_rules': reference_rules,
                        'missing_rules': missing_rules, 'rules_match': (not missing_rules and error is None) if reference_rules else None,
                        'match': (result['decision'] == expected and error is None) if expected else None})
    metrics = {}
    for group in ('official', 'custom'):
        subset = [d for d in details if d['group'] == group]
        matrix = {expected: {actual: 0 for actual in (*DECISIONS, 'TECHNICAL_ERROR')} for expected in DECISIONS}
        counts = Counter(d['actual']['decision'] for d in subset)
        for d in subset:
            column = 'TECHNICAL_ERROR' if d['technical_error'] is not None else d['actual']['decision']
            matrix[d['expected']][column] += 1
        correct = sum(d['match'] for d in subset)
        metrics[group] = {'total': len(subset), 'correct': correct, 'incorrect': len(subset)-correct,
                          'accuracy': correct / len(subset) if subset else None,
                          'technical_failures': sum(d['technical_error'] is not None for d in subset),
                          'rule_reference_total': sum(d['rules_match'] is not None for d in subset),
                          'rule_reference_mismatches': sum(d['rules_match'] is False for d in subset),
                          'distribution': {d: counts[d] for d in DECISIONS}, 'confusion_matrix': matrix,
                          'false_admission': sum(d['actual']['decision'] == 'ДОПУСТИТЬ' and d['expected'] != 'ДОПУСТИТЬ' for d in subset)}
    return {'metrics': metrics, 'details': details}


def run_eval(cases: list[dict[str, Any]], client: Any, model: str, journal: Path, output: Path, secret: str = '') -> dict[str, Any]:
    run_journal = journal / uuid.uuid4().hex
    results = process_batch([{'id': c['id'], 'text': c['text']} for c in cases], client, model, run_journal, secret)
    records = [json.loads(path.read_text(encoding='utf-8')) for path in run_journal.glob('*.json')]
    by_id = {(type(r['input']['id']), r['input']['id']): r for r in records}
    if len(records) != len(cases) or len(by_id) != len(cases):
        raise ValueError('Audit records cannot be linked to cases')
    errors = []
    for case in cases:
        record = by_id.get((type(case['id']), case['id']))
        if record is None or 'extraction_error' not in record:
            raise ValueError('Missing audit status')
        errors.append(record['extraction_error'])
    report = evaluate_cases(cases, results, errors)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('eval/dataset.json'))
    parser.add_argument('--output', type=Path, default=Path('eval/run_results.json'))
    parser.add_argument('--journal', type=Path, default=Path('eval/journal'))
    args = parser.parse_args()
    try:
        cases = load_dataset(args.dataset)
        config = Config.from_env()
        report = run_eval(cases, OpenAIClient(config), config.model, args.journal, args.output, config.api_key)
    except Exception:
        print('Ошибка загрузки, обработки или сохранения eval.')
        return 1
    print(json.dumps(report['metrics'], ensure_ascii=False, indent=2))
    for d in report['details']:
        if d['match'] is False or d['match'] is None or d['rules_match'] is False:
            print(f"{d['id']}: ожидание={d['expected'] or 'нет'}, результат={d['actual']['decision']}, отсутствуют правила={d['missing_rules']}, техническая ошибка={d['technical_error']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
