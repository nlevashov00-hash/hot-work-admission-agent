"""Deterministic checks for the agreed fire-work rules."""
from datetime import date
from typing import Any

from app.contracts import validate_facts, validate_result, validate_rule_status


def evaluate(identifier: Any, raw_facts: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    f = validate_facts(raw_facts)
    statuses: list[dict[str, Any]] = []

    def value(name: str) -> Any:
        return f[name]["value"] if f[name]["state"] == "known" else None

    def add(n: int, status: str, message: str, missing: list[str] | None = None) -> None:
        statuses.append(validate_rule_status({"rule": f"П{n}", "status": status,
                                              "explanation": message, "missing": missing or []}))

    def threshold(name: str, boundary: float, *, strict_below: bool = False) -> bool | None:
        x = value(name)
        if x is None:
            return None
        if strict_below:
            return True if x["max"] < boundary else False if x["min"] >= boundary else None
        return True if x["min"] > boundary else False if x["max"] <= boundary else None

    present, number = value("permit_present"), value("permit_number")
    if present is False:
        add(1, "FAIL", "Подтверждено отсутствие наряда.")
    elif number is not None and f["permit_present"]["state"] != "contradictory":
        add(1, "PASS", "Номер наряда указан.")
    else:
        add(1, "UNKNOWN", "Наряд или его номер не установлены.", ["Наряд и его номер"])

    briefing, day = value("briefing_present"), value("briefing_date")
    if briefing is False:
        add(2, "FAIL", "Подтверждено отсутствие инструктажа.")
    elif day is not None and f["briefing_present"]["state"] != "contradictory":
        d = date.fromisoformat(day)
        if d < date(2026, 4, 15):
            add(2, "FAIL", "Инструктаж ранее допустимого периода.")
        elif d <= date(2026, 10, 15):
            add(2, "PASS", "Дата инструктажа в допустимом периоде.")
        else:
            add(2, "UNKNOWN", "Дата инструктажа после даты работ.", ["Дата актуального инструктажа"])
    else:
        add(2, "UNKNOWN", "Дата или наличие инструктажа не установлены.", ["Дата актуального инструктажа"])

    near = []
    for prefix in ("coal", "fuel"):
        name, dist = prefix + "_store_near", prefix + "_store_distance_m"
        if f[name]["state"] == "contradictory" or f[dist]["state"] == "contradictory":
            near.append(None)
        else:
            a, b = value(name), threshold(dist, 10, strict_below=True)
            near.append(a if a is not None else b)
    approval = value("written_approval")
    if True in near:
        if approval is True:
            add(3, "PASS", "Близкий склад и письменное согласование подтверждены.")
        elif approval is False:
            add(3, "FAIL", "Близкий склад и отсутствие письменного согласования подтверждены.")
        else:
            add(3, "UNKNOWN", "Для близкого склада не установлено письменное согласование.", ["Письменное согласование"])
    elif all(x is False for x in near):
        add(3, "NOT_APPLICABLE", "Оба типа складов не ближе 10 м.")
    else:
        add(3, "UNKNOWN", "Близость складов не установлена.", ["Близость складов угля и ГСМ"])

    names = [("extinguisher", "Огнетушитель"), ("observer_present", "Наблюдающий"),
             ("observer_independent", "Независимость наблюдающего")]
    missing = [label for name, label in names if value(name) is None]
    violations = [message for name, message in (
        ("extinguisher", "Огнетушитель отсутствует."),
        ("observer_present", "Наблюдающий отсутствует."),
        ("observer_independent", "Наблюдающий не является независимым."),
    ) if value(name) is False]
    if violations:
        add(4, "FAIL", " ".join(violations), missing)
    elif missing:
        add(4, "UNKNOWN", "Не установлены огнетушитель и независимый наблюдающий.", missing)
    else:
        add(4, "PASS", "Огнетушитель и независимый наблюдающий подтверждены.")

    location, wind = value("location"), threshold("wind_m_s", 10)
    if location == "indoor":
        add(5, "NOT_APPLICABLE", "Работы в помещении.")
    elif location is None:
        add(5, "UNKNOWN", "Тип места не установлен.", ["Тип места работ"])
    elif wind is None:
        add(5, "UNKNOWN", "Скорость ветра относительно порога не установлена.", ["Скорость ветра"])
    elif wind:
        add(5, "FAIL", "Ветер выше 10 м/с.")
    else:
        add(5, "PASS", "Ветер не превышает 10 м/с.")

    height = threshold("height_m", 1.8)
    if height is None:
        add(6, "UNKNOWN", "Высота относительно порога не установлена.", ["Высота работ"])
    elif not height:
        add(6, "NOT_APPLICABLE", "Высота не превышает 1,8 м.")
    elif value("height_permit") is True:
        add(6, "PASS", "Допуск на высоту указан.")
    else:
        add(6, "HUMAN", "При работе выше 1,8 м требуется решение человека.", ["Допуск на высоту или решение человека"])

    uncertain = [s for s in statuses if s["status"] in ("UNKNOWN", "HUMAN")]
    has_uncertainty = bool(uncertain) or any(s["missing"] for s in statuses)
    if has_uncertainty:
        add(7, "UNKNOWN", "Требуется уточнение или решение человека.")
    else:
        add(7, "NOT_APPLICABLE", "Неопределённость не выявлена.")
    failures = [s for s in statuses if s["status"] == "FAIL"]
    decision = "ОТКАЗАТЬ" if failures else "ЭСКАЛИРОВАТЬ" if has_uncertainty else "ДОПУСТИТЬ"
    relevant = failures + uncertain
    rules = [s["rule"] for s in statuses if s in relevant or (s["rule"] == "П7" and has_uncertainty)]
    if not rules:
        rules = [s["rule"] for s in statuses if s["rule"] != "П7"]
    result = {"id": identifier, "decision": decision, "rules": rules,
              "explanation": " ".join(f'{s["rule"]}: {s["explanation"]}' for s in (relevant or statuses[:-1])),
              "missing": list(dict.fromkeys(m for s in statuses for m in s["missing"]))}
    return statuses, validate_result(result)
