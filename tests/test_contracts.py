import copy
import json
import unittest

from app.contracts import (ContractError, FACT_TYPES, parse_batch, parse_facts,
                           validate_batch, validate_facts, validate_result,
                           validate_results, validate_rule_status)


def facts():
    return {name: {"state": "unknown", "value": None, "evidence": []}
            for name in FACT_TYPES}


def result(identifier=1):
    return {"id": identifier, "decision": "ЭСКАЛИРОВАТЬ", "rules": ["П7"],
            "explanation": "Недостаточно данных", "missing": ["Наряд"]}


class ContractTests(unittest.TestCase):
    def test_batch_and_identifiers(self):
        batch = parse_batch('[{"id": 1, "text": ""}, {"id": "1", "text": "данные"}]')
        self.assertEqual([x["id"] for x in batch], [1, "1"])
        for invalid in ('{', '{}', '[]', '[{"id": true, "text": "x"}]',
                        '[{"id": 1, "text": 2}]',
                        '[{"id": 1, "text": "x"}, {"id": 1, "text": "y"}]',
                        '[{"text": "x"}]'):
            with self.subTest(invalid=invalid), self.assertRaises(ContractError):
                parse_batch(invalid)

    def test_fact_states_and_values(self):
        data = facts()
        data["extinguisher"] = {"state": "known", "value": False, "evidence": ["нет огнетушителя"]}
        data["height_m"] = {"state": "known", "value": {"min": 2.5, "max": 3.5, "approximate": True}, "evidence": ["около 3 м"]}
        data["location"] = {"state": "contradictory", "value": None, "evidence": ["в помещении", "на улице"]}
        self.assertIs(validate_facts(data), data)
        invalid = []
        for name, value in [("extinguisher", {"state": "known", "value": None, "evidence": ["нет"]}),
                            ("height_m", {"state": "known", "value": {"min": 3, "max": 3, "approximate": True}, "evidence": ["около 3"]}),
                            ("location", {"state": "contradictory", "value": None, "evidence": ["одно"]}),
                            ("permit_present", {"state": "unknown", "value": False, "evidence": []})]:
            candidate = copy.deepcopy(data)
            candidate[name] = value
            invalid.append(candidate)
        invalid.append({})
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ContractError):
                validate_facts(candidate)
        with self.assertRaises(ContractError):
            parse_facts('not json')

    def test_rule_and_result(self):
        validate_rule_status({"rule": "П1", "status": "UNKNOWN", "explanation": "Нет данных", "missing": ["Наряд"]})
        with self.assertRaises(ContractError):
            validate_rule_status({"rule": "П1", "status": "OK", "explanation": "x", "missing": []})
        validate_result(result())
        for field, value in [("decision", "PASS"), ("rules", []), ("rules", ["П8"]),
                             ("missing", "нет"), ("id", False), ("explanation", "")]:
            item = result()
            item[field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_result(item)

    def test_linkage(self):
        batch = validate_batch([{"id": 1, "text": "x"}, {"id": "b", "text": "y"}])
        validate_results(batch, [result(1), result("b")])
        for results in ([result(1)], [result("b"), result(1)], [result("1"), result("b")]):
            with self.subTest(results=results), self.assertRaises(ContractError):
                validate_results(batch, results)


class ExtendedContractTests(unittest.TestCase):
    @staticmethod
    def known(value, evidence):
        return {"state": "known", "value": value, "evidence": [evidence]}

    def test_linked_conflicts_preserve_evidence_and_other_facts(self):
        pairs = [
            ("permit_present", False, "permit_number", "НД-1"),
            ("briefing_present", False, "briefing_date", "2026-10-01"),
            ("observer_present", False, "observer_independent", True),
        ]
        for prefix in ("coal", "fuel"):
            pairs += [(prefix + "_store_near", False, prefix + "_store_distance_m",
                       {"min": 5, "max": 5, "approximate": False}),
                      (prefix + "_store_near", True, prefix + "_store_distance_m",
                       {"min": 10, "max": 10, "approximate": False})]
        for left, lv, right, rv in pairs:
            with self.subTest(left=left, lv=lv):
                data = facts()
                data[left] = self.known(lv, "первое утверждение")
                data[right] = self.known(rv, "второе утверждение")
                data["extinguisher"] = self.known(True, "огнетушитель есть")
                before = copy.deepcopy(data)
                checked = validate_facts(data)
                for name in (left, right):
                    self.assertEqual(checked[name], {"state": "contradictory", "value": None,
                                                    "evidence": ["первое утверждение", "второе утверждение"]})
                self.assertEqual(checked["extinguisher"], before["extinguisher"])
                self.assertEqual(data, before)
                self.assertEqual(validate_facts(checked), checked)

    def test_incomplete_and_crossing_interval_are_not_conflicts(self):
        for near, lo, hi in [(False, 9, 10), (True, 9, 10), (False, 10, 10), (True, 9, 9)]:
            data = facts()
            data["coal_store_near"] = self.known(near, "рядом")
            data["coal_store_distance_m"] = self.known({"min": lo, "max": hi, "approximate": False}, "расстояние")
            self.assertEqual(validate_facts(data), data)
        data = facts()
        data["permit_present"] = self.known(False, "нет наряда")
        self.assertEqual(validate_facts(data), data)

    def test_required_fields_and_types(self):
        for name in FACT_TYPES:
            data = facts()
            del data[name]
            with self.subTest(missing=name), self.assertRaises(ContractError):
                validate_facts(data)
        for field in ("state", "value", "evidence"):
            data = facts()
            del data["extinguisher"][field]
            with self.subTest(missing=field), self.assertRaises(ContractError):
                validate_facts(data)
        for name, value in [("permit_present", 1), ("permit_number", 1),
                            ("briefing_date", "2026-02-30"), ("location", []),
                            ("wind_source", "other")]:
            data = facts()
            data[name] = self.known(value, "основание")
            with self.subTest(name=name), self.assertRaises(ContractError):
                validate_facts(data)
        for field in result():
            item = result()
            del item[field]
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_result(item)

    def test_invalid_states_and_evidence(self):
        for fact in ({"state": "other", "value": None, "evidence": []},
                     {"state": 1, "value": None, "evidence": []},
                     {"state": "known", "value": True, "evidence": []},
                     {"state": "unknown", "value": True, "evidence": []},
                     {"state": "contradictory", "value": None, "evidence": ["один"]},
                     {"state": "unknown", "value": None, "evidence": [1]}):
            data = facts()
            data["extinguisher"] = fact
            with self.subTest(fact=fact), self.assertRaises(ContractError):
                validate_facts(data)

    def test_nonfinite_json_and_rule_required_fields(self):
        for number in (float("nan"), float("inf"), float("-inf")):
            data = facts()
            data["wind_m_s"] = self.known({"min": number, "max": number, "approximate": False}, "ветер")
            with self.subTest(number=number), self.assertRaises(ContractError):
                parse_facts(json.dumps(data))
        rule = {"rule": "П1", "status": "PASS", "explanation": "есть", "missing": []}
        for field in rule:
            partial = dict(rule)
            del partial[field]
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_rule_status(partial)

    def test_numeric_intervals(self):
        for value in ({"min": -1, "max": 1, "approximate": False},
                      {"min": 2, "max": 1, "approximate": False},
                      {"min": 1, "max": 1, "approximate": True},
                      {"min": True, "max": 1, "approximate": False},
                      {"min": 0, "max": 1, "approximate": 0},
                      {"min": 0, "max": 1},
                      *({"min": x, "max": x, "approximate": False} for x in (float("nan"), float("inf"), float("-inf")))):
            data = facts()
            data["height_m"] = self.known(value, "высота")
            with self.subTest(value=value), self.assertRaises(ContractError):
                validate_facts(data)
        for value in ({"min": 0, "max": 0, "approximate": False},
                      {"min": 1.8, "max": 1.8, "approximate": False},
                      {"min": 9, "max": 11, "approximate": True}):
            data = facts()
            data["height_m"] = self.known(value, "высота")
            self.assertEqual(validate_facts(data), data)


if __name__ == "__main__":
    unittest.main()
