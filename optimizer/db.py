from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Use an absolute path so the DB is always found regardless of CWD.
_DEFAULT_DB = Path(__file__).resolve().parent.parent / "optimizer.db"
DB_PATH = Path(os.environ.get("OPTIMIZER_DB", str(_DEFAULT_DB)))

_TABLES = [
    """CREATE TABLE IF NOT EXISTS prompts (
        prompt_id TEXT PRIMARY KEY, category TEXT NOT NULL, modules TEXT NOT NULL,
        gen_params TEXT NOT NULL, parent_id TEXT, lineage_depth INTEGER NOT NULL DEFAULT 0,
        mutation_note TEXT NOT NULL DEFAULT '', compiled_hash TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('champion','candidate','retired')),
        cooldown_until TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
    "CREATE INDEX IF NOT EXISTS idx_prompts_category_status ON prompts(category, status)",
    """CREATE TABLE IF NOT EXISTS benchmark_cases (
        case_id TEXT PRIMARY KEY, category TEXT NOT NULL, input TEXT NOT NULL, expected TEXT,
        difficulty TEXT NOT NULL CHECK(difficulty IN ('routine','hard','adversarial')),
        split TEXT NOT NULL CHECK(split IN ('train','holdout','vault')), source TEXT NOT NULL DEFAULT 'seed',
        label_confidence REAL NOT NULL DEFAULT 1.0
    )""",
    "CREATE INDEX IF NOT EXISTS idx_cases_category_split ON benchmark_cases(category, split)",
    """CREATE TABLE IF NOT EXISTS run_manifests (
        run_id TEXT PRIMARY KEY, model_tag TEXT NOT NULL, judge_tag TEXT NOT NULL,
        rubric_version TEXT NOT NULL, dataset_snapshot_id TEXT NOT NULL, compiler_sha TEXT NOT NULL,
        evaluator_sha TEXT NOT NULL, split TEXT NOT NULL, category TEXT NOT NULL,
        started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, completed INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS eval_results (
        run_id TEXT NOT NULL, prompt_id TEXT NOT NULL, case_id TEXT NOT NULL, output TEXT NOT NULL,
        metrics TEXT NOT NULL, latency_ms INTEGER NOT NULL, PRIMARY KEY(run_id, prompt_id, case_id),
        FOREIGN KEY(run_id) REFERENCES run_manifests(run_id)
    )""",
    """CREATE TABLE IF NOT EXISTS mutation_log (
        attempt_id TEXT PRIMARY KEY, category TEXT NOT NULL, failure_theme TEXT NOT NULL,
        strategy TEXT NOT NULL, module_touched TEXT NOT NULL, train_delta REAL,
        gate_outcome TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS promotions (
        promotion_id TEXT PRIMARY KEY, category TEXT NOT NULL, old_champion TEXT NOT NULL,
        new_champion TEXT NOT NULL, holdout_delta REAL, ci_low REAL, ci_high REAL,
        vault_confirmed INTEGER, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS traces (
        trace_id TEXT PRIMARY KEY, query TEXT NOT NULL, category TEXT NOT NULL, confidence REAL NOT NULL,
        prompt_id TEXT NOT NULL, compiled_hash TEXT NOT NULL, output TEXT NOT NULL, latency_ms INTEGER NOT NULL,
        model_tag TEXT NOT NULL, conversation_id TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS conversations (
        conversation_id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS feedback (
        feedback_id TEXT PRIMARY KEY,
        trace_id TEXT NOT NULL, prompt_id TEXT NOT NULL, signal TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""",
    "CREATE INDEX IF NOT EXISTS idx_feedback_trace ON feedback(trace_id)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_prompt ON feedback(prompt_id)",
]


@contextmanager
def conn() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    """Idempotent schema initialisation. Safe to call once at startup."""
    with conn() as connection:
        for stmt in _TABLES:
            connection.execute(stmt)
        # Migrate: add conversation_id column to traces if it doesn't exist yet.
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(traces)")}
        if "conversation_id" not in columns:
            connection.execute("ALTER TABLE traces ADD COLUMN conversation_id TEXT")
        # Migrate: feedback table now uses feedback_id PK (was trace_id).
        # If the old schema exists (trace_id PK with no feedback_id column), recreate.
        fb_cols = {row["name"] for row in connection.execute("PRAGMA table_info(feedback)")}
        if "feedback_id" not in fb_cols:
            connection.execute("ALTER TABLE feedback RENAME TO feedback_old")
            connection.execute("""CREATE TABLE feedback (
                feedback_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL, prompt_id TEXT NOT NULL, signal TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_feedback_trace ON feedback(trace_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_feedback_prompt ON feedback(prompt_id)")
            # Migrate old rows
            connection.execute("""INSERT OR IGNORE INTO feedback(feedback_id, trace_id, prompt_id, signal, created_at)
                SELECT hex(randomblob(6)), trace_id, prompt_id, signal, created_at FROM feedback_old""")
            connection.execute("DROP TABLE feedback_old")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
