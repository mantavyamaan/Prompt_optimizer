from __future__ import annotations

import asyncio
import json

from .backends import MockLLM
from .compiler import compiled_hash
from .db import conn, dumps, new_id
from .gate import run_gate
from .models import Prompt
from .runner import load_cases, load_prompt, open_manifest, per_case_scores, run_benchmark
from .seed import seed_all


async def aa_null_test(repetitions: int = 10) -> float:
    """Champion versus itself must not be promoted; validates the gate's false-positive handling."""
    seed_all()
    with conn() as connection:
        champion_id = connection.execute("SELECT prompt_id FROM prompts WHERE status='champion' AND category='extraction'").fetchone()["prompt_id"]
    champion = load_prompt(champion_id)
    train = load_cases("extraction", "train")
    false_promotions = 0
    for _ in range(repetitions):
        manifest = open_manifest("mock-v1", "extraction", "train")
        await run_benchmark(MockLLM(), manifest.run_id, champion, train)
        scores = per_case_scores(manifest.run_id, champion_id)
        verdict = await run_gate(MockLLM(), "extraction", champion_id, [champion_id], {champion_id: scores}, scores)
        false_promotions += verdict.promote
    rate = false_promotions / repetitions
    print(json.dumps({"test": "A/A null", "promotions": false_promotions, "repetitions": repetitions, "rate": rate, "pass": rate <= .10}))
    return rate


async def planted_winner_test() -> bool:
    """Create a deliberately better candidate and confirm it reaches a review-eligible verdict."""
    seed_all()
    with conn() as connection:
        champion_id = connection.execute("SELECT prompt_id FROM prompts WHERE status='champion' AND category='extraction'").fetchone()["prompt_id"]
    champion = load_prompt(champion_id)
    modules = champion.modules.model_copy(deep=True)
    modules.constraints += "\nPLANTED_WINNER"
    candidate = Prompt(prompt_id=new_id("planted"), category="extraction", modules=modules, gen_params=champion.gen_params,
                       parent_id=champion_id, lineage_depth=1, mutation_note="self-test planted winner", compiled_hash=compiled_hash(modules))
    with conn() as connection:
        connection.execute("""INSERT OR IGNORE INTO prompts(prompt_id,category,modules,gen_params,parent_id,lineage_depth,mutation_note,compiled_hash,status)
            VALUES(?,?,?,?,?,?,?,?,?)""", (candidate.prompt_id, candidate.category, dumps(candidate.modules.model_dump()), dumps(candidate.gen_params.model_dump()),
             candidate.parent_id, candidate.lineage_depth, candidate.mutation_note, candidate.compiled_hash, "candidate"))
    train = load_cases("extraction", "train")
    old_manifest, new_manifest = open_manifest("mock-v1", "extraction", "train"), open_manifest("mock-v1", "extraction", "train")
    await run_benchmark(MockLLM(), old_manifest.run_id, champion, train)
    await run_benchmark(MockLLM(), new_manifest.run_id, candidate, train)
    verdict = await run_gate(MockLLM(), "extraction", champion_id, [candidate.prompt_id],
                             {candidate.prompt_id: per_case_scores(new_manifest.run_id, candidate.prompt_id)}, per_case_scores(old_manifest.run_id, champion_id))
    print(json.dumps({"test": "planted winner", "stage": verdict.stage, "review_eligible": verdict.promote, "note": verdict.note}))
    return verdict.promote


async def main() -> None:
    await aa_null_test()
    await planted_winner_test()


if __name__ == "__main__":
    asyncio.run(main())
