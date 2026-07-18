from __future__ import annotations

import json
import re

from .models import BenchmarkCase, MetricResult


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    value, end = decoder.raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object")
    if text[start + end:].strip():
        raise ValueError("unexpected content after JSON object")
    return json.dumps(value, sort_keys=True)


def schema_match(case: BenchmarkCase, output: str) -> MetricResult:
    expected = (case.expected or {}).get("json")
    if expected is None:
        return MetricResult(name="primary:schema_match", passed=False, score=0, detail="no expected JSON")
    try:
        actual = json.loads(extract_json(output))
    except (ValueError, json.JSONDecodeError) as exc:
        return MetricResult(name="primary:schema_match", passed=False, score=0, detail=f"unparseable: {exc}")
    required = list(expected)
    matched = sum(actual.get(key) == expected[key] for key in required)
    score = matched / max(1, len(required))
    return MetricResult(name="primary:schema_match", passed=score == 1, score=score, detail=f"{matched}/{len(required)} keys")


def format_valid(case: BenchmarkCase, output: str) -> MetricResult:
    try:
        extract_json(output)
        return MetricResult(name="format_valid", passed=True, score=1)
    except ValueError as exc:
        return MetricResult(name="format_valid", passed=False, score=0, detail=str(exc))


def length_ratio(case: BenchmarkCase, output: str) -> MetricResult:
    cap = float((case.expected or {}).get("max_length_ratio", 20))
    ratio = len(output) / max(1, len(json.dumps(case.input)))
    passed = ratio <= cap
    return MetricResult(name="length_ratio", passed=passed, score=1 if passed else 0, detail=f"ratio={ratio:.2f}, cap={cap:.2f}")


EVALUATORS = {"extraction": (schema_match, format_valid, length_ratio)}
