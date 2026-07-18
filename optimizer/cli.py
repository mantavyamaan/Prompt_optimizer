from __future__ import annotations

import asyncio
import json

import typer

from .backends import MockLLM
from .cycle import nightly_cycle, promote as deploy_prompt, vault_check
from .db import conn, init_db

app = typer.Typer(help="Evidence-driven prompt optimization commands.")


@app.command()
def seed() -> None:
    """Create the local database, benchmark corpus, and serving champions."""
    from .seed import seed_all
    seed_all()
    typer.echo("Seeded extraction corpus, baseline extractor, and Prompt Architect champion.")


@app.command()
def run(category: str = "extraction") -> None:
    """Run one full train-to-holdout optimization cycle using the mock backend."""
    init_db()
    verdict = asyncio.run(nightly_cycle(MockLLM(), category))
    if verdict is None:
        raise typer.BadParameter(f"No champion or train data for category {category!r}; run `seed` first.")
    typer.echo(json.dumps(verdict.model_dump(), indent=2))


@app.command()
def review(limit: int = 10) -> None:
    """List generated candidates and pending, evidence-backed promotion proposals."""
    init_db()
    with conn() as connection:
        proposals = connection.execute("""SELECT category,old_champion,new_champion,holdout_delta,ci_low,ci_high,created_at
            FROM promotions WHERE vault_confirmed IS NULL ORDER BY created_at DESC LIMIT ?""", (limit,)).fetchall()
        candidates = connection.execute("""SELECT prompt_id,parent_id,mutation_note,created_at FROM prompts
            WHERE status='candidate' ORDER BY created_at DESC LIMIT ?""", (limit,)).fetchall()
    if proposals:
        typer.echo("Pending review proposals:")
        for row in proposals:
            typer.echo(f"  {row['new_champion']} <- {row['old_champion']} | holdout {row['holdout_delta']:+.3f} | CI [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]")
    if candidates:
        typer.echo("Candidates:")
        for row in candidates:
            typer.echo(f"  {row['prompt_id']} <- {row['parent_id']}: {row['mutation_note']}")
    if not proposals and not candidates:
        typer.echo("Nothing awaiting review.")


@app.command("promote")
def promote_command(category: str, prompt_id: str) -> None:
    """Human-controlled deployment; run vault-check afterward."""
    deploy_prompt(category, prompt_id)
    typer.echo(f"Promoted {prompt_id}; serving cache invalidated. Run vault-check for a post-deployment regression guardrail.")


@app.command("rollback")
def rollback(category: str, prompt_id: str) -> None:
    """Roll back by promoting a known previous champion."""
    deploy_prompt(category, prompt_id)
    typer.echo(f"Rolled back {category} to {prompt_id}.")


@app.command("vault-check")
def vault_check_command(category: str, old_champion: str, candidate: str) -> None:
    """Compare a reviewed candidate with its prior champion on untouched vault data."""
    verdict = asyncio.run(vault_check(MockLLM(), category, old_champion, candidate))
    typer.echo(json.dumps(verdict.model_dump(), indent=2))


@app.command("generate-data")
def generate_data(category: str = "extraction", count: int = 20) -> None:
    """Dynamically generate massive synthetic edge cases using the LLM."""
    from .synthetic import generate_synthetic_cases
    init_db()
    typer.echo(f"Starting synthetic generation for {category}...")
    generate_synthetic_cases(category, count)


if __name__ == "__main__":
    app()
