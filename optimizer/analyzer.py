from __future__ import annotations

import json
from collections import defaultdict

from .db import conn, new_id
from .models import FailureTheme


def analyze_failures(category: str, champion_id: str, run_id: str) -> tuple[list[FailureTheme], list[dict]]:
    """Analyze only train rows. Holdout and vault data are structurally unreachable here."""
    with conn() as connection:
        rows = connection.execute("""SELECT e.case_id,e.metrics,b.input,b.expected FROM eval_results e
            JOIN benchmark_cases b ON b.case_id=e.case_id WHERE e.run_id=? AND e.prompt_id=?
            AND b.category=? AND b.split='train'""", (run_id, champion_id, category)).fetchall()
    buckets: dict[str, list[str]] = defaultdict(list)
    exemplars: list[dict] = []
    for row in rows:
        failed = [item["name"] for item in json.loads(row["metrics"]) if not item["passed"]]
        if failed:
            buckets[failed[0]].append(row["case_id"])
            exemplars.append({"input": json.loads(row["input"]), "expected": (json.loads(row["expected"]) or {}).get("json", {})})
    themes = [FailureTheme(theme_id=new_id("theme"), category=category, label=label, exemplar_case_ids=case_ids[:5])
              for label, case_ids in sorted(buckets.items(), key=lambda item: -len(item[1]))[:4]]
    return themes, exemplars[:5]
