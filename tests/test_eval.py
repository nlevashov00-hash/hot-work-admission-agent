import copy
import json
import tempfile
import subprocess
import unittest
from pathlib import Path

from app.contracts import FACT_TYPES
from app.extraction import Response
from eval.runner import DECISIONS, evaluate_cases, load_dataset, run_eval


class Fake:
    def complete(self, text):
        return Response(json.dumps({k: {'state': 'unknown', 'value': None, 'evidence': []} for k in FACT_TYPES}))


def result(identifier, decision):
    return {'id': identifier, 'decision': decision, 'rules': ['П7'], 'explanation': 'тест', 'missing': []}


class EvalTests(unittest.TestCase):
    def test_dataset_sources(self):
        cases = load_dataset(Path('eval/dataset.json'))
        self.assertEqual([sum(c['group'] == group for c in cases) for group in ('official', 'custom', 'unlabeled')], [8, 13, 3])
        source = []
        for line in Path('REQUIREMENTS.md').read_text(encoding='utf-8-sig').splitlines():
            parts = [p.strip() for p in line.strip().strip('|').split('|')]
            if len(parts) == 3 and parts[0].isdigit() and 1 <= int(parts[0]) <= 11:
                source.append(parts)
        for case, row in zip(cases[:11], source):
            self.assertEqual((case['id'], case['text']), (int(row[0]), row[1]))
            self.assertEqual(case['expected'], row[2] if case['id'] <= 8 else None)

    def test_invalid_dataset(self):
        cases = load_dataset(Path('eval/dataset.json'))
        for mutation in ('duplicate', 'missing', 'unlabeled', 'rationale'):
            data = copy.deepcopy(cases)
            if mutation == 'duplicate': data[1]['id'] = data[0]['id']
            elif mutation == 'missing': del data[0]['text']
            elif mutation == 'unlabeled': data[8]['expected'] = 'ДОПУСТИТЬ'
            else: data[11]['rationale'] = ''
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'data.json'; path.write_text(json.dumps(data), encoding='utf-8')
                with self.subTest(mutation=mutation), self.assertRaises(ValueError): load_dataset(path)

    def test_metrics_and_unlabeled(self):
        cases = load_dataset(Path('eval/dataset.json'))
        results = [result(c['id'], 'ДОПУСТИТЬ') for c in cases]
        report = evaluate_cases(cases, results, [None] * len(cases))
        official = report['metrics']['official']
        self.assertEqual((official['total'], official['correct'], official['incorrect'], official['false_admission']), (8, 1, 7, 7))
        self.assertEqual(official['accuracy'], 1/8)
        self.assertEqual(sum(sum(row.values()) for row in official['confusion_matrix'].values()), 8)
        self.assertEqual(sum(d['match'] is None for d in report['details']), 3)
        with self.assertRaises(ValueError): evaluate_cases(cases, results[:-1], [None] * len(cases))

    def test_mixed_failure_and_rule_mismatch(self):
        cases = load_dataset(Path('eval/dataset.json'))[:3]
        results = [result(1, 'ЭСКАЛИРОВАТЬ'), result(2, 'ОТКАЗАТЬ'), result(3, 'ДОПУСТИТЬ')]
        results[1]['rules'] = ['П5']
        report = evaluate_cases(cases, results, ['extraction_failed', None, None])
        metrics = report['metrics']['official']
        self.assertEqual((metrics['correct'], metrics['incorrect'], metrics['technical_failures']), (1, 2, 1))
        self.assertEqual(metrics['false_admission'], 1)
        self.assertEqual(sum(metrics['confusion_matrix'][d][d] for d in DECISIONS), metrics['correct'])
        self.assertEqual(sum(sum(row.values()) for row in metrics['confusion_matrix'].values()), metrics['total'])
        self.assertEqual(metrics['confusion_matrix']['ДОПУСТИТЬ']['TECHNICAL_ERROR'], 1)
        self.assertEqual(metrics['confusion_matrix']['ОТКАЗАТЬ']['ОТКАЗАТЬ'], 1)
        self.assertEqual(metrics['confusion_matrix']['ОТКАЗАТЬ']['ДОПУСТИТЬ'], 1)
        self.assertFalse(report['details'][1]['rules_match'])
        self.assertEqual(report['details'][1]['missing_rules'], ['П1'])
        cases = [load_dataset(Path('eval/dataset.json'))[5]]
        report = evaluate_cases(cases, [result(6, 'ЭСКАЛИРОВАТЬ')], ['extraction_failed'])
        metrics = report['metrics']['official']
        matrix = metrics['confusion_matrix']
        self.assertEqual(metrics['correct'], 0)
        self.assertEqual(metrics['total'], 1)
        self.assertEqual(metrics['accuracy'], 0)
        self.assertEqual(matrix['ЭСКАЛИРОВАТЬ']['TECHNICAL_ERROR'], 1)
        self.assertEqual(matrix['ЭСКАЛИРОВАТЬ']['ЭСКАЛИРОВАТЬ'], 0)
        self.assertEqual(sum(matrix[d][d] for d in DECISIONS), metrics['correct'])
        self.assertEqual(sum(sum(row.values()) for row in matrix.values()), metrics['total'])
        self.assertEqual(report['details'][0]['actual']['decision'], 'ЭСКАЛИРОВАТЬ')
        self.assertFalse(report['details'][0]['match'])
        results[1]['rules'] = ['П1', 'П5']
        self.assertTrue(evaluate_cases(load_dataset(Path('eval/dataset.json'))[:3], results, [None]*3)['details'][1]['rules_match'])

    def test_generated_files_ignored(self):
        proc = subprocess.run(['git', 'check-ignore', 'eval/run_results.json', 'eval/journal/example.json'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(len(proc.stdout.splitlines()), 2)

    def test_runner_with_mixed_extraction_failure(self):
        class Mixed:
            def __init__(self): self.calls = 0
            def complete(self, text):
                self.calls += 1
                if self.calls <= 2: raise TimeoutError()
                return Fake().complete(text)
        cases = load_dataset(Path('eval/dataset.json'))[5:7]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_eval(cases, Mixed(), 'stub', root / 'journal', root / 'report.json')
            self.assertEqual(report['details'][0]['technical_error'], 'extraction_failed')
            self.assertFalse(report['details'][0]['match'])
            self.assertIsNone(report['details'][1]['technical_error'])
            self.assertEqual(report['metrics']['official']['technical_failures'], 1)

    def test_runner_saves_details(self):
        cases = load_dataset(Path('eval/dataset.json'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_eval(cases, Fake(), 'stub', root / 'journal', root / 'out.json')
            self.assertEqual(json.loads((root / 'out.json').read_text(encoding='utf-8')), report)
            self.assertEqual(len(report['details']), len(cases))
            self.assertEqual(len(list((root / 'journal').rglob('*.json'))), len(cases))


if __name__ == '__main__': unittest.main()
