from __future__ import annotations

import json
from statistics import fmean, quantiles, variance

from .db import conn
from .models import GateVerdict
from .runner import find_completed_run, load_cases, load_prompt, open_manifest, per_case_scores, run_benchmark
from .stats import confidence_sequence, minimum_detectable_effect, paired_deltas, sign_test

TRAIN_MARGIN = 0.03
FORMAT_SLACK = 0.01
ADVERSARIAL_SLACK = 0.02
P95_LATENCY_BUDGET_MS = 30_000
MIN_SEQUENTIAL_CASES = 20


def _average(scores: dict[str, float]) -> float:
    return fmean(scores.values()) if scores else 0.0


def _fence_stats(run_id: str, prompt_id: str) -> tuple[float, float, float]:
    with conn() as connection:
        rows = connection.execute("""SELECT e.metrics,e.latency_ms,b.difficulty FROM eval_results e
            JOIN benchmark_cases b ON b.case_id=e.case_id WHERE e.run_id=? AND e.prompt_id=?""", (run_id, prompt_id)).fetchall()
    formats: list[float] = []
    adversarial: list[float] = []
    latencies: list[float] = []
    for row in rows:
        latencies.append(row["latency_ms"])
        metrics = json.loads(row["metrics"])
        formats.extend(item["score"] for item in metrics if item["name"] == "format_valid")
        if row["difficulty"] == "adversarial":
            adversarial.extend(item["score"] for item in metrics if item["name"].startswith("primary:"))
    p95 = quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else 0)
    return _average({str(i): score for i, score in enumerate(formats)}), _average({str(i): score for i, score in enumerate(adversarial)}), p95


async def run_gate(backend, category: str, champion_id: str, candidate_ids: list[str], train_scores: dict[str, dict[str, float]], champion_train: dict[str, float]) -> GateVerdict:
    if not candidate_ids:
        return GateVerdict(challenger_id="-", champion_id=champion_id, stage="no_candidate", promote=False, train_delta=0, note="No valid candidates generated")
    challenger_id = max(candidate_ids, key=lambda item: _average(train_scores[item]) - _average(champion_train))
    train_delta = _average(train_scores[challenger_id]) - _average(champion_train)
    if train_delta < TRAIN_MARGIN:
        return GateVerdict(challenger_id=challenger_id, champion_id=champion_id, stage="train_margin", promote=False, train_delta=train_delta,
                           note=f"Best train delta {train_delta:+.3f} did not earn a holdout evaluation")

    champion, challenger = load_prompt(champion_id), load_prompt(challenger_id)
    holdout = load_cases(category, "holdout")
    champion_run = find_completed_run(champion_id, backend.model_tag, category, "holdout")
    if champion_run is None:
        manifest = open_manifest(backend.model_tag, category, "holdout")
        await run_benchmark(backend, manifest.run_id, champion, holdout)
        champion_run = manifest.run_id
    challenger_manifest = open_manifest(backend.model_tag, category, "holdout")
    champion_scores = per_case_scores(champion_run, champion_id)
    observed: list[float] = []
    stop, low, high = False, -1.0, 1.0
    for case in holdout:
        await run_benchmark(backend, challenger_manifest.run_id, challenger, [case])
        challenger_scores = per_case_scores(challenger_manifest.run_id, challenger_id)
        if case.case_id in champion_scores and case.case_id in challenger_scores:
            observed.append(challenger_scores[case.case_id] - champion_scores[case.case_id])
        stop, low, high = confidence_sequence(observed)
        if stop and len(observed) >= MIN_SEQUENTIAL_CASES:
            break
    challenger_scores = per_case_scores(challenger_manifest.run_id, challenger_id)
    deltas, wins, losses, ties = paired_deltas(champion_scores, challenger_scores)
    holdout_delta = _average({str(i): value for i, value in enumerate(deltas)})
    sample_var = variance(deltas) if len(deltas) > 1 else 0.25
    champion_fmt, champion_adv, _ = _fence_stats(champion_run, champion_id)
    challenger_fmt, challenger_adv, p95 = _fence_stats(challenger_manifest.run_id, challenger_id)
    fences: list[str] = []
    if challenger_fmt < champion_fmt - FORMAT_SLACK:
        fences.append(f"format validity fell from {champion_fmt:.3f} to {challenger_fmt:.3f}")
    if challenger_adv < champion_adv - ADVERSARIAL_SLACK:
        fences.append(f"adversarial score fell from {champion_adv:.3f} to {challenger_adv:.3f}")
    if p95 > P95_LATENCY_BUDGET_MS:
        fences.append(f"p95 latency {p95:.0f}ms exceeds budget")
    promote = not fences and stop and low > 0
    return GateVerdict(challenger_id=challenger_id, champion_id=champion_id, stage="fence" if fences else "sequential", promote=promote,
                       train_delta=train_delta, holdout_delta=holdout_delta, ci_low=low, ci_high=high, n_holdout=len(deltas), wins=wins,
                       losses=losses, ties=ties, sign_test_p=sign_test(wins, losses), mde=minimum_detectable_effect(len(deltas), sample_var),
                       fence_failures=fences, note="Eligible for human review" if promote else ("Hard fence failed" if fences else "Confidence sequence did not establish improvement"))
