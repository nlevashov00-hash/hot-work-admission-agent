import copy
import unittest
from unittest.mock import patch
from app.contracts import FACT_TYPES
from app.extraction import Extraction
from app.smoke_test import EXPECTED, main, semantic_errors


def valid(identifier):
    facts = {name: {"state": "unknown", "value": None, "evidence": []} for name in FACT_TYPES}
    for name, (state, value) in EXPECTED[identifier].items():
        facts[name] = {"state": state, "value": value, "evidence": ["основание"] if state == "known" else []}
    return facts


class SmokeTests(unittest.TestCase):
    def test_semantics(self):
        for identifier in (1, 8):
            good = valid(identifier)
            self.assertEqual(semantic_errors(identifier, good), [])
            all_unknown = {k: {"state": "unknown", "value": None, "evidence": []} for k in FACT_TYPES}
            self.assertTrue(semantic_errors(identifier, all_unknown))
            wrong = copy.deepcopy(good)
            wrong['permit_number']['value'] = 'неверный'
            self.assertIn('permit_number', semantic_errors(identifier, wrong))
            self.assertEqual(set(EXPECTED[identifier]), set(FACT_TYPES))
            values = {"bool": True, "string": "НД-999", "date": "2026-01-01",
                      "number": {"min": 1, "max": 1, "approximate": False},
                      "location": "outdoor", "wind_source": "stated"}
            for name, (state, _) in EXPECTED[identifier].items():
                if state == "unknown":
                    with self.subTest(identifier=identifier, name=name):
                        invented = copy.deepcopy(good)
                        invented[name] = {"state": "known", "value": values[FACT_TYPES[name]], "evidence": ["x"]}
                        self.assertIn(name, semantic_errors(identifier, invented))

    def test_incomplete_expectations_fail_closed(self):
        incomplete = dict(EXPECTED[1])
        del incomplete["height_m"]
        with patch.dict(EXPECTED, {1: incomplete}):
            with self.assertRaises(ValueError):
                semantic_errors(1, valid(8))

    def test_main_requires_semantics(self):
        with patch('app.smoke_test.load_env'), patch('app.smoke_test.Config.from_env') as config, \
             patch('app.smoke_test.OpenAIClient') as client, patch('app.smoke_test.extract') as extract, \
             patch('builtins.print'):
            config.return_value.model = 'test'
            client.return_value.config.model = 'test'
            extract.side_effect = [Extraction(valid(1), [], 0), Extraction(valid(8), [], 0)]
            self.assertEqual(main(), 0)
            unknown = {k: {'state': 'unknown', 'value': None, 'evidence': []} for k in FACT_TYPES}
            extract.side_effect = [Extraction(unknown, [], 0), Extraction(valid(8), [], 0)]
            self.assertEqual(main(), 1)


if __name__ == '__main__': unittest.main()
