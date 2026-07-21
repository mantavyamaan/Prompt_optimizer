import asyncio
import time
from optimizer.cycle import nightly_cycle, promote
from optimizer.serve import build_backend
from optimizer.db import conn

# Categories that the evolution loop targets.
EVOLVE_CATEGORIES = ["extraction", "prompt_design"]
SYNTHETIC_DATA_THRESHOLD = 150  # Generate more data when below this count.
SYNTHETIC_BATCH_SIZE = 5
CYCLE_SLEEP_SECONDS = 300  # 5 minutes between cycles.


async def check_ab_tests(backend) -> None:
    """Promote candidates that have enough positive user feedback from shadow traffic."""
    with conn() as connection:
        pending = connection.execute(
            "SELECT category, new_champion FROM promotions WHERE vault_confirmed IS NULL"
        ).fetchall()

    for p in pending:
        category = p["category"]
        candidate_id = p["new_champion"]
        with conn() as connection:
            rows = connection.execute(
                "SELECT signal FROM feedback WHERE prompt_id=? AND signal LIKE 'score:%'",
                (candidate_id,),
            ).fetchall()

        if len(rows) < 5:
            continue

        scores = []
        for r in rows:
            try:
                score = int(r["signal"].split(":")[1])
                if 0 <= score <= 100:
                    scores.append(score)
            except (ValueError, IndexError):
                # Malformed signal — skip rather than crashing the loop.
                pass

        if not scores:
            continue

        avg = sum(scores) / len(scores)
        if avg >= 85:
            print(f"A/B Test Winner! Candidate {candidate_id} achieved {avg:.1f}/100 over {len(scores)} interactions. Promoting!")
            # Wrap both promote() and vault_confirmed UPDATE in one atomic operation.
            try:
                with conn() as connection:
                    # Retire old champion, set new champion.
                    candidate_row = connection.execute(
                        "SELECT category FROM prompts WHERE prompt_id=?", (candidate_id,)
                    ).fetchone()
                    if candidate_row is None or candidate_row["category"] != category:
                        print(f"Candidate {candidate_id} not found in category {category} — skipping.")
                        continue
                    connection.execute(
                        "UPDATE prompts SET status='retired' WHERE category=? AND status='champion'",
                        (category,),
                    )
                    connection.execute(
                        "UPDATE prompts SET status='champion' WHERE prompt_id=?",
                        (candidate_id,),
                    )
                    connection.execute(
                        "UPDATE promotions SET vault_confirmed=1 WHERE new_champion=?",
                        (candidate_id,),
                    )
                # Invalidate cache outside the transaction (import is safe here).
                from optimizer.serve import CHAMPION_CACHE
                CHAMPION_CACHE.pop(category, None)
                print(f"Successfully promoted {candidate_id} to champion in '{category}'.")
            except Exception as e:
                print(f"Error during A/B promotion of {candidate_id}: {e}")


async def run_loop(backend=None) -> None:
    """Continuous evolution loop. Accepts a shared backend to avoid duplicate clients."""
    print("Starting continuous evolution loop...")
    if backend is None:
        backend = build_backend()

    while True:
        try:
            # ── Synthetic data generation ──────────────────────────────────────────
            from optimizer.synthetic import _run_generation
            for category in EVOLVE_CATEGORIES:
                with conn() as connection:
                    case_count = connection.execute(
                        "SELECT COUNT(*) as c FROM benchmark_cases WHERE category=?", (category,)
                    ).fetchone()["c"]
                if case_count < SYNTHETIC_DATA_THRESHOLD:
                    print(f"[{category}] Dataset has {case_count} cases. Generating {SYNTHETIC_BATCH_SIZE} more...")
                    await _run_generation(category, SYNTHETIC_BATCH_SIZE, backend)

            # ── A/B test promotions ────────────────────────────────────────────────
            await check_ab_tests(backend)

            # ── Optimization cycles for all categories ─────────────────────────────
            for category in EVOLVE_CATEGORIES:
                print(f"Running optimization cycle for '{category}'...")
                verdict = await nightly_cycle(backend, category)
                if verdict is None:
                    print(f"[{category}] No champion or training data found. Skipping.")
                    continue
                print(f"[{category}] Cycle completed. Verdict: promote={verdict.promote}, note='{verdict.note}'")
                if verdict.promote:
                    print(f"[{category}] Candidate {verdict.challenger_id} eligible for review.")
                else:
                    print(f"[{category}] Candidate {verdict.challenger_id} rejected. Cooldown applied.")

        except Exception as e:
            print(f"Error during evolution cycle: {type(e).__name__}: {e}")

        print(f"Sleeping for {CYCLE_SLEEP_SECONDS // 60} minutes before next cycle...")
        await asyncio.sleep(CYCLE_SLEEP_SECONDS)


if __name__ == "__main__":
    asyncio.run(run_loop())
