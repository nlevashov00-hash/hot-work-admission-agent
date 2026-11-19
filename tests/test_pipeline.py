import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.contracts import FACT_TYPES, ContractError
from app.extraction import Response
from app.pipeline import run
from test_rules import base, put


def encode(f):
    for fact in f.values():
        if fact['state'] == 'known':
            fact['evidence'] = ['x']
    return Response(json.dumps(f, ensure_ascii=False), {'total_tokens': 10})


class Fake:
    def __init__(self, replies): self.replies = iter(replies)
    def complete(self, text):
        reply = next(self.replies)
        if isinstance(reply, Exception): raise reply
        return reply


class PipelineTests(unittest.TestCase):
    def test_batch_audit_and_repeated_run(self):
        good = base()
        bad = base(); put(bad, 'extinguisher', False)
        unknown = {k: {'state': 'unknown', 'value': None, 'evidence': []} for k in FACT_TYPES}
        batch = [{'id': i, 'text': 'x'} for i in (1, 2, 3, 4)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inp = root / 'input.json'; out = root / 'output.json'; journal = root / 'journal'
            inp.write_text(json.dumps(batch), encoding='utf-8')
            run(inp, out, journal, Fake([encode(good), encode(bad), encode(unknown), TimeoutError(), TimeoutError()]), 'test')
            results = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual([x['id'] for x in results], [1, 2, 3, 4])
            self.assertEqual([x['decision'] for x in results], ['ДОПУСТИТЬ', 'ОТКАЗАТЬ', 'ЭСКАЛИРОВАТЬ', 'ЭСКАЛИРОВАТЬ'])
            paths = set(journal.glob('*.json')); self.assertEqual(len(paths), 4)
            records = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
            self.assertEqual({r['input']['id'] for r in records}, {1, 2, 3, 4})
            failed = next(r for r in records if r['input']['id'] == 4)
            self.assertEqual(failed['extraction_error'], 'extraction_failed')
            self.assertEqual(len(failed['attempts']), 2)
            self.assertEqual(failed['rule_statuses'], [])
            self.assertEqual(next(r for r in records if r['input']['id'] == 1)['attempts'][0]['usage']['total_tokens'], 10)
            run(inp, out, journal, Fake([encode(good)] * 4), 'test')
            self.assertEqual(len(list(journal.glob('*.json'))), 8)
            self.assertTrue(paths.issubset(set(journal.glob('*.json'))))

    def test_invalid_input_and_audit_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inp = root / 'input.json'; out = root / 'output.json'
            inp.write_text('{', encoding='utf-8')
            with self.assertRaises(ContractError): run(inp, out, root / 'journal', Fake([]), 'test')
            self.assertFalse(out.exists())
            inp.write_text('[{"id":1,"text":"x"}]', encoding='utf-8')
            with patch('pathlib.Path.mkdir', side_effect=PermissionError):
                with self.assertRaises(PermissionError): run(inp, out, root / 'journal', Fake([]), 'test')
            self.assertFalse(out.exists())

    def test_configuration_errors_produce_results_and_audits(self):
        for env in ({'LLM_BASE_URL': 'https://example.test/v1', 'LLM_MODEL': 'test'},
                    {'LLM_BASE_URL': 'http://remote.example/v1', 'LLM_MODEL': 'test', 'LLM_API_KEY': 'private-key'}):
            with self.subTest(env_keys=list(env)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                batch = [{'id': 'a', 'text': 'private-key'}, {'id': 2, 'text': 'x'}]
                (root / 'input.json').write_text(json.dumps(batch), encoding='utf-8')
                with patch.dict('os.environ', env, clear=True):
                    run(root / 'input.json', root / 'output.json', root / 'journal')
                results = json.loads((root / 'output.json').read_text(encoding='utf-8'))
                self.assertEqual([x['id'] for x in results], ['a', 2])
                self.assertEqual([x['decision'] for x in results], ['ЭСКАЛИРОВАТЬ'] * 2)
                paths = list((root / 'journal').glob('*.json'))
                self.assertEqual(len(paths), 2)
                for path in paths:
                    record = json.loads(path.read_text(encoding='utf-8'))
                    self.assertEqual(record['extraction_error'], 'configuration_error')
                    self.assertEqual(record['attempts'], [])
                    if 'LLM_API_KEY' in env:
                        self.assertNotIn('private-key', path.read_text(encoding='utf-8'))

    def test_secret_redaction_and_failure_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inp = root / 'input.json'; out = root / 'output.json'
            inp.write_text(json.dumps([{'id': 1, 'text': 'secret'}, {'id': 2, 'text': 'x'}]), encoding='utf-8')
            run(inp, out, root / 'journal', Fake([ValueError('secret'), ValueError('secret'), encode(base())]), 'secret-model', 'secret')
            self.assertEqual(len(json.loads(out.read_text(encoding='utf-8'))), 2)
            for path in (root / 'journal').glob('*.json'):
                self.assertNotIn('secret', path.read_text(encoding='utf-8'))


if __name__ == '__main__': unittest.main()
