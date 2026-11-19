import unittest

from app.contracts import FACT_TYPES
from app.rules import evaluate


def base():
    f = {k: {"state": "unknown", "value": None, "evidence": []} for k in FACT_TYPES}
    for name, val in {"permit_number": "НД-1", "briefing_date": "2026-10-01",
                      "coal_store_near": False, "fuel_store_near": False,
                      "extinguisher": True, "observer_present": True,
                      "observer_independent": True, "location": "indoor",
                      "height_m": {"min": 0, "max": 0, "approximate": False}}.items():
        put(f, name, val)
    return f


def put(f, name, value):
    f[name] = {"state": "known", "value": value, "evidence": ["основание"]}


def interval(lo, hi=None):
    return {"min": lo, "max": lo if hi is None else hi, "approximate": hi is not None}


def status(f, n):
    return evaluate(1, f)[0][n - 1]["status"]


class RulesTests(unittest.TestCase):
    def test_baseline(self):
        statuses, result = evaluate("a", base())
        self.assertEqual(result["decision"], "ДОПУСТИТЬ")
        self.assertEqual(result["id"], "a")
        self.assertEqual(statuses[6]["status"], "NOT_APPLICABLE")

    def test_dates(self):
        for day, expected in [("2026-04-14", "FAIL"), ("2026-04-15", "PASS"),
                              ("2026-10-15", "PASS"), ("2026-10-16", "UNKNOWN")]:
            f = base(); put(f, "briefing_date", day)
            with self.subTest(day=day): self.assertEqual(status(f, 2), expected)

    def test_distance(self):
        for distance, expected in [(9.9, "FAIL"), (10, "NOT_APPLICABLE"), (10.1, "NOT_APPLICABLE")]:
            f = base(); f["coal_store_near"] = {"state": "unknown", "value": None, "evidence": []}
            put(f, "coal_store_distance_m", interval(distance)); put(f, "written_approval", False)
            with self.subTest(distance=distance): self.assertEqual(status(f, 3), expected)
        f = base(); f["coal_store_near"] = {"state": "unknown", "value": None, "evidence": []}
        put(f, "coal_store_distance_m", interval(9, 10))
        self.assertEqual(status(f, 3), "UNKNOWN")
        put(f, "coal_store_near", True); f["coal_store_distance_m"] = {"state": "unknown", "value": None, "evidence": []}
        self.assertEqual(status(f, 3), "UNKNOWN")
        put(f, "written_approval", True); self.assertEqual(status(f, 3), "PASS")

    def test_wind_height(self):
        for wind, expected in [(9.9, "PASS"), (10, "PASS"), (10.1, "FAIL")]:
            f = base(); put(f, "location", "outdoor"); put(f, "wind_m_s", interval(wind))
            self.assertEqual(status(f, 5), expected)
        f = base(); put(f, "location", "outdoor"); put(f, "wind_m_s", interval(9, 11))
        self.assertEqual(status(f, 5), "UNKNOWN")
        for height, expected in [(1.7, "NOT_APPLICABLE"), (1.8, "NOT_APPLICABLE"), (1.9, "HUMAN")]:
            f = base(); put(f, "height_m", interval(height)); put(f, "height_permit", False)
            self.assertEqual(status(f, 6), expected)
        put(f, "height_permit", True); self.assertEqual(status(f, 6), "PASS")
        put(f, "height_m", interval(1.7, 1.9)); self.assertEqual(status(f, 6), "UNKNOWN")
        self.assertEqual(status(base(), 5), "NOT_APPLICABLE")

    def test_p4_failure_with_missing_activates_p7(self):
        f = base()
        put(f, "observer_present", False)
        f["observer_independent"] = {"state": "unknown", "value": None, "evidence": []}
        f["extinguisher"] = {"state": "unknown", "value": None, "evidence": []}
        statuses, result = evaluate("case", f)
        self.assertEqual(statuses[3]["status"], "FAIL")
        self.assertIn("Наблюдающий отсутствует", statuses[3]["explanation"])
        self.assertIn("Огнетушитель", statuses[3]["missing"])
        self.assertEqual(statuses[6]["status"], "UNKNOWN")
        self.assertEqual(result["id"], "case")
        self.assertEqual(result["decision"], "ОТКАЗАТЬ")
        self.assertEqual(result["rules"], ["П4", "П7"])
        self.assertIn("Наблюдающий отсутствует", result["explanation"])
        self.assertIn("Огнетушитель", result["missing"])

    def test_p4_lists_each_confirmed_violation(self):
        f = base()
        put(f, "extinguisher", False)
        put(f, "observer_present", False)
        f["observer_independent"] = {"state": "unknown", "value": None, "evidence": []}
        statuses, result = evaluate(2, f)
        self.assertEqual(statuses[3]["status"], "FAIL")
        self.assertEqual(statuses[6]["status"], "UNKNOWN")
        self.assertEqual(result["decision"], "ОТКАЗАТЬ")
        self.assertEqual(result["rules"], ["П4", "П7"])
        for message in ("Огнетушитель отсутствует", "Наблюдающий отсутствует"):
            self.assertIn(message, statuses[3]["explanation"])
            self.assertIn(message, result["explanation"])
        self.assertIn("Независимость наблюдающего", result["missing"])

    def test_failure_unknown_and_contradiction(self):
        f = base(); put(f, "permit_present", False)
        statuses, result = evaluate(1, f)
        self.assertEqual(statuses[0]["status"], "UNKNOWN")  # number conflicts with absence
        f = base(); f["permit_number"] = {"state": "unknown", "value": None, "evidence": []}; put(f, "permit_present", False)
        f["height_m"] = {"state": "unknown", "value": None, "evidence": []}
        statuses, result = evaluate(1, f)
        self.assertEqual(result["decision"], "ОТКАЗАТЬ")
        self.assertIn("П1", result["rules"]); self.assertIn("П7", result["rules"])
        self.assertIn("Высота работ", result["missing"])
        f = base(); put(f, "observer_present", False); f["observer_independent"] = {"state": "unknown", "value": None, "evidence": []}
        f["extinguisher"] = {"state": "unknown", "value": None, "evidence": []}
        statuses, result = evaluate(1, f)
        self.assertEqual(statuses[3]["status"], "FAIL"); self.assertIn("Огнетушитель", result["missing"])
        f = base(); put(f, "height_m", interval(2)); self.assertEqual(evaluate(1, f)[1]["decision"], "ЭСКАЛИРОВАТЬ")


if __name__ == "__main__": unittest.main()
