import asyncio
import time
from optimizer.cycle import nightly_cycle, promote
from optimizer.serve import build_backend
from optimizer.db import conn

async def check_ab_tests():
    with conn() as connection:
        pending = connection.execute("SELECT category, new_champion FROM promotions WHERE vault_confirmed IS NULL").fetchall()
        for p in pending:
            category = p["category"]
            candidate_id = p["new_champion"]
            rows = connection.execute("SELECT signal FROM feedback WHERE prompt_id=? AND signal LIKE 'score:%'", (candidate_id,)).fetchall()
            if len(rows) >= 5:
                scores = [int(r["signal"].split(":")[1]) for r in rows]
                avg = sum(scores) / len(scores)
                if avg >= 85:
                    print(f"A/B Test Winner! Candidate {candidate_id} achieved {avg:.1f}/100 in production over {len(scores)} interactions. Promoting!")
                    promote(category, candidate_id)
                    connection.execute("UPDATE promotions SET vault_confirmed=1 WHERE new_champion=?", (candidate_id,))

async def run_loop():
    print("Starting continuous evolution loop...")
    backend = build_backend()
    while True:
        try:
            # Automate Dataset Generation
            with conn() as connection:
                case_count = connection.execute("SELECT COUNT(*) as c FROM benchmark_cases WHERE category='extraction'").fetchone()["c"]
            
            # If the dataset is small, automatically generate 5 new adversarial cases before optimizing
            if case_count < 150:
                print(f"Dataset currently has {case_count} cases. Automatically generating 5 more synthetic edge cases...")
                from optimizer.synthetic import _run_generation
                await _run_generation("extraction", 5)

            await check_ab_tests()
            print("Running optimization cycle...")
            verdict = await nightly_cycle(backend, "extraction")
            if verdict is None:
                print("No champion or training data found. Waiting 60s.")
                await asyncio.sleep(60)
                continue
            
            print(f"Cycle completed. Verdict: promote={verdict.promote}, note='{verdict.note}'")
            if verdict.promote:
                print(f"Candidate {verdict.challenger_id} passed fences! Auto-promoting...")
                promote("extraction", verdict.challenger_id)
                print(f"Successfully promoted {verdict.challenger_id} to champion.")
            else:
                print(f"Candidate {verdict.challenger_id} rejected. Cooldown applied.")
                
        except Exception as e:
            print(f"Error during evolution cycle: {e}")
            
        print("Sleeping for 5 minutes before next cycle...")
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(run_loop())
