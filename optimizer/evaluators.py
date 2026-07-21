from __future__ import annotations

import json
import re

from .models import BenchmarkCase, MetricResult


def extract_json(text: str) -> str:
    text = text.strip()
    # Strip markdown code fences (anywhere in the string, not just start/end).
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.I)
    text = text.strip()
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    value, end = decoder.raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object")
    # Allow trailing whitespace and minor non-JSON characters (newlines, punctuation).
    tail = text[start + end:].strip()
    if tail and not re.fullmatch(r"[.,;:!?\s]*", tail):
        raise ValueError(f"unexpected content after JSON object: {tail[:40]!r}")
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


# ── prompt_design evaluators ───────────────────────────────────────────────────

def _has_code_block(text: str) -> bool:
    return bool(re.search(r"```", text))


def _has_heading(text: str) -> bool:
    return bool(re.search(r"copy.?ready\s+prompt", text, re.I))


def prompt_structure(case: BenchmarkCase, output: str) -> MetricResult:
    """Primary metric: output must contain a copy-ready heading and a code block."""
    has_block = _has_code_block(output)
    has_head = _has_heading(output)
    passed = has_block and has_head
    score = (0.5 if has_head else 0.0) + (0.5 if has_block else 0.0)
    detail = f"heading={'yes' if has_head else 'no'}, code_block={'yes' if has_block else 'no'}"
    return MetricResult(name="primary:prompt_structure", passed=passed, score=score, detail=detail)


def prompt_length(case: BenchmarkCase, output: str) -> MetricResult:
    """A useful prompt must be at least 200 characters and no more than 6000."""
    n = len(output.strip())
    passed = 200 <= n <= 6000
    score = 1.0 if passed else 0.0
    return MetricResult(name="prompt_length", passed=passed, score=score, detail=f"length={n}")


def no_direct_answer(case: BenchmarkCase, output: str) -> MetricResult:
    """The optimizer must not answer the user's task directly — it must produce a prompt."""
    # A direct answer would lack both a code block and the heading.
    indirect = _has_code_block(output) or _has_heading(output)
    return MetricResult(name="no_direct_answer", passed=indirect, score=1.0 if indirect else 0.0,
                        detail="prompt structure present" if indirect else "looks like direct answer")


EVALUATORS: dict[str, tuple] = {
    "extraction": (schema_match, format_valid, length_ratio),
    "prompt_design": (prompt_structure, prompt_length, no_direct_answer),
}
