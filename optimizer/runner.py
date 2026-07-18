from __future__ import annotations

import asyncio
import hashlib
import json
import time

from .compiler import code_sha, compile_prompt
from .db import conn, dumps, new_id
from .evaluators import EVALUATORS
from .models import BenchmarkCase, GenParams, Prompt, PromptModules, RunManifest

CONCURRENCY = 4
CASE_TIMEOUT_SECONDS = 90


def dataset_snapshot_id(category: str, split: str) -> str:
    with conn() as connection:
        rows = connection.execute("SELECT case_id FROM benchmark_cases WHERE category=? AND split=? ORDER BY case_id", (category, split)).fetchall()
    return hashlib.sha256(",".join(row["case_id"] for row in rows).encode()).hexdigest()[:16]


def open_manifest(model_tag: str, category: str, split: str) -> RunManifest:
    manifest = RunManifest(run_id=new_id("run"), model_tag=model_tag, dataset_snapshot_id=dataset_snapshot_id(category, split),
                           compiler_sha=code_sha(), evaluator_sha=code_sha())
    with conn() as connection:
        connection.execute("""INSERT INTO run_manifests
            (run_id,model_tag,judge_tag,rubric_version,dataset_snapshot_id,compiler_sha,evaluator_sha,split,category)
            VALUES (?,?,?,?,?,?,?,?,?)""", (manifest.run_id, manifest.model_tag, manifest.judge_tag,
             manifest.rubric_version, manifest.dataset_snapshot_id, manifest.compiler_sha, manifest.evaluator_sha, split, category))
    return manifest


def load_cases(category: str, split: str) -> list[BenchmarkCase]:
    with conn() as connection:
        rows = connection.execute("SELECT * FROM benchmark_cases WHERE category=? AND split=? ORDER BY difficulty,case_id", (category, split)).fetchall()
    return [BenchmarkCase(case_id=row["case_id"], category=row["category"], input=json.loads(row["input"]),
                          expected=json.loads(row["expected"]) if row["expected"] else None, difficulty=row["difficulty"],
                          split=row["split"], source=row["source"], label_confidence=row["label_confidence"]) for row in rows]


def load_prompt(prompt_id: str) -> Prompt:
    with conn() as connection:
        row = connection.execute("SELECT * FROM prompts WHERE prompt_id=?", (prompt_id,)).fetchone()
    if row is None:
        raise KeyError(f"prompt not found: {prompt_id}")
    return Prompt(prompt_id=row["prompt_id"], category=row["category"], modules=PromptModules.model_validate_json(row["modules"]),
                  gen_params=GenParams.model_validate_json(row["gen_params"]), parent_id=row["parent_id"],
                  lineage_depth=row["lineage_depth"], mutation_note=row["mutation_note"], compiled_hash=row["compiled_hash"], status=row["status"])


async def run_benchmark(backend, run_id: str, prompt: Prompt, cases: list[BenchmarkCase]) -> None:
    """Run missing case records only; the composite primary key makes reruns safe."""
    with conn() as connection:
        done = {row["case_id"] for row in connection.execute("SELECT case_id FROM eval_results WHERE run_id=? AND prompt_id=?", (run_id, prompt.prompt_id))}
    limiter = asyncio.Semaphore(CONCURRENCY)

    async def evaluate(case: BenchmarkCase) -> None:
        async with limiter:
            rendered = compile_prompt(prompt.modules, {"user_input": json.dumps(case.input, sort_keys=True)})
            start = time.perf_counter()
            try:
                output = await asyncio.wait_for(backend.generate(rendered, case.input, case.expected, prompt.gen_params), timeout=CASE_TIMEOUT_SECONDS)
            except Exception as exc:
                output = f"ERROR: {type(exc).__name__}: {exc}"
            latency_ms = int((time.perf_counter() - start) * 1000)
            metrics = [metric(case, output).model_dump() for metric in EVALUATORS[case.category]]
            with conn() as connection:
                connection.execute("INSERT OR IGNORE INTO eval_results VALUES (?,?,?,?,?,?)", (run_id, prompt.prompt_id, case.case_id, output, dumps(metrics), latency_ms))

    await asyncio.gather(*(evaluate(case) for case in cases if case.case_id not in done))
    with conn() as connection:
        connection.execute("UPDATE run_manifests SET completed=1 WHERE run_id=?", (run_id,))


def per_case_scores(run_id: str, prompt_id: str) -> dict[str, float]:
    with conn() as connection:
        rows = connection.execute("SELECT case_id,metrics FROM eval_results WHERE run_id=? AND prompt_id=?", (run_id, prompt_id)).fetchall()
    result: dict[str, float] = {}
    for row in rows:
        primary = next((metric["score"] for metric in json.loads(row["metrics"]) if metric["name"].startswith("primary:")), None)
        if primary is not None:
            result[row["case_id"]] = float(primary)
    return result


def find_completed_run(prompt_id: str, model_tag: str, category: str, split: str) -> str | None:
    snapshot = dataset_snapshot_id(category, split)
    with conn() as connection:
        row = connection.execute("""SELECT r.run_id FROM run_manifests r JOIN eval_results e ON e.run_id=r.run_id
            WHERE e.prompt_id=? AND r.model_tag=? AND r.category=? AND r.split=? AND r.dataset_snapshot_id=?
            AND r.completed=1 GROUP BY r.run_id HAVING COUNT(*)=(SELECT COUNT(*) FROM benchmark_cases WHERE category=? AND split=?)
            ORDER BY r.started_at DESC LIMIT 1""", (prompt_id, model_tag, category, split, snapshot, category, split)).fetchone()
    return row["run_id"] if row else None
