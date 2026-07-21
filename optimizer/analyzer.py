from __future__ import annotations

import json
from collections import defaultdict

from .db import conn, new_id
from .models import FailureTheme

MIN_FAILURES_FOR_THEME = 2  # Don't generate variants targeting a single isolated failure.


def analyze_failures(category: str, champion_id: str, run_id: str) -> tuple[list[FailureTheme], list[dict]]:
    """Analyze only train rows. Holdout and vault data are structurally unreachable here."""
    with conn() as connection:
        rows = connection.execute("""SELECT e.case_id,e.metrics,b.input,b.expected FROM eval_results e
            JOIN benchmark_cases b ON b.case_id=e.case_id WHERE e.run_id=? AND e.prompt_id=?
            AND b.category=? AND b.split='train'""", (run_id, champion_id, category)).fetchall()

    buckets: dict[str, list[str]] = defaultdict(list)
    # Deduplicate exemplars by case_id — one entry per case, not per metric.
    seen_exemplar_ids: set[str] = set()
    exemplars: list[dict] = []

    for row in rows:
        metrics = json.loads(row["metrics"])
        failed_names = [item["name"] for item in metrics if not item["passed"]]
        if not failed_names:
            continue

        # Bucket ALL failed metrics, not just the first one.
        for name in failed_names:
            buckets[name].append(row["case_id"])

        # Add this case as an exemplar (deduplicated by case_id).
        if row["case_id"] not in seen_exemplar_ids:
            seen_exemplar_ids.add(row["case_id"])
            exemplars.append({
                "input": json.loads(row["input"]),
                "expected": (json.loads(row["expected"]) or {}).get("json", {}),
            })

    # Only surface themes with enough failures to be meaningful.
    significant = {label: ids for label, ids in buckets.items() if len(ids) >= MIN_FAILURES_FOR_THEME}
    themes = [
        FailureTheme(theme_id=new_id("theme"), category=category, label=label, exemplar_case_ids=case_ids[:5])
        for label, case_ids in sorted(significant.items(), key=lambda item: -len(item[1]))[:4]
    ]
    return themes, exemplars[:5]
