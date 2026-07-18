from __future__ import annotations

from .analyzer import analyze_failures
from .db import conn, new_id
from .gate import run_gate
from .generator import apply_cooldown, generate_variants, log
from .models import GateVerdict
from .runner import load_cases, load_prompt, open_manifest, per_case_scores, run_benchmark


async def nightly_cycle(backend, category: str = "extraction") -> GateVerdict | None:
    with conn() as connection:
        row = connection.execute("SELECT prompt_id FROM prompts WHERE category=? AND status='champion'", (category,)).fetchone()
    if row is None:
        return None
    champion = load_prompt(row["prompt_id"])
    train = load_cases(category, "train")
    if not train:
        return None
    champion_manifest = open_manifest(backend.model_tag, category, "train")
    await run_benchmark(backend, champion_manifest.run_id, champion, train)
    champion_scores = per_case_scores(champion_manifest.run_id, champion.prompt_id)
    themes, exemplars = analyze_failures(category, champion.prompt_id, champion_manifest.run_id)
    variants = generate_variants(champion, themes, exemplars)
    if not variants:
        return GateVerdict(challenger_id="-", champion_id=champion.prompt_id, stage="generation", promote=False, train_delta=0, note="No novel variants were generated")
    train_scores: dict[str, dict[str, float]] = {}
    for variant in variants:
        manifest = open_manifest(backend.model_tag, category, "train")
        await run_benchmark(backend, manifest.run_id, variant, train)
        train_scores[variant.prompt_id] = per_case_scores(manifest.run_id, variant.prompt_id)
    verdict = await run_gate(backend, category, champion.prompt_id, [variant.prompt_id for variant in variants], train_scores, champion_scores)
    if verdict.promote:
        with conn() as connection:
            connection.execute("""INSERT INTO promotions(promotion_id,category,old_champion,new_champion,holdout_delta,ci_low,ci_high,vault_confirmed)
                VALUES(?,?,?,?,?,?,?,NULL)""", (new_id("proposal"), category, champion.prompt_id, verdict.challenger_id,
                 verdict.holdout_delta, verdict.ci_low, verdict.ci_high))
        log(category, "-", "-", "-", "review_pending", verdict.train_delta)
    else:
        apply_cooldown(verdict.challenger_id)
        log(category, "-", "-", "-", "rejected", verdict.train_delta)
    return verdict


def promote(category: str, prompt_id: str) -> None:
    """Deploy only after a human review. Promotion invalidates the serving cache."""
    with conn() as connection:
        candidate = connection.execute("SELECT category FROM prompts WHERE prompt_id=?", (prompt_id,)).fetchone()
        if candidate is None or candidate["category"] != category:
            raise KeyError("candidate does not exist in the requested category")
        connection.execute("UPDATE prompts SET status='retired' WHERE category=? AND status='champion'", (category,))
        connection.execute("UPDATE prompts SET status='champion' WHERE prompt_id=?", (prompt_id,))
    from .serve import CHAMPION_CACHE
    CHAMPION_CACHE.pop(category, None)


async def vault_check(backend, category: str, champion_id: str, candidate_id: str) -> GateVerdict:
    """A post-review guardrail. Vault data is never used to choose a candidate."""
    champion, candidate = load_prompt(champion_id), load_prompt(candidate_id)
    vault = load_cases(category, "vault")
    champion_manifest, candidate_manifest = open_manifest(backend.model_tag, category, "vault"), open_manifest(backend.model_tag, category, "vault")
    await run_benchmark(backend, champion_manifest.run_id, champion, vault)
    await run_benchmark(backend, candidate_manifest.run_id, candidate, vault)
    old, new = per_case_scores(champion_manifest.run_id, champion_id), per_case_scores(candidate_manifest.run_id, candidate_id)
    from .stats import paired_deltas
    deltas, wins, losses, ties = paired_deltas(old, new)
    delta = sum(deltas) / len(deltas) if deltas else 0
    with conn() as connection:
        connection.execute("UPDATE promotions SET vault_confirmed=? WHERE category=? AND old_champion=? AND new_champion=?", (int(delta >= 0), category, champion_id, candidate_id))
    return GateVerdict(challenger_id=candidate_id, champion_id=champion_id, stage="vault", promote=delta >= 0, train_delta=0,
                       holdout_delta=delta, n_holdout=len(deltas), wins=wins, losses=losses, ties=ties,
                       note="Vault regression check passed" if delta >= 0 else "Vault regression detected; roll back")
