from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Difficulty(StrEnum):
    ROUTINE = "routine"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class Split(StrEnum):
    TRAIN = "train"
    HOLDOUT = "holdout"
    VAULT = "vault"


class Status(StrEnum):
    CHAMPION = "champion"
    CANDIDATE = "candidate"
    RETIRED = "retired"


MODULE_NAMES = ("role", "context_rules", "format_instructions", "few_shot_examples", "constraints")


class PromptModules(BaseModel):
    role: str = ""
    context_rules: str = ""
    format_instructions: str = ""
    few_shot_examples: str = ""
    constraints: str = ""

    def diff(self, other: "PromptModules") -> list[str]:
        return [name for name in MODULE_NAMES if getattr(self, name) != getattr(other, name)]


class GenParams(BaseModel):
    temperature: float = Field(default=0.0, ge=0, le=2)
    seed: int = 7
    max_tokens: int = Field(default=1024, ge=1)


class Prompt(BaseModel):
    prompt_id: str
    category: str
    modules: PromptModules
    gen_params: GenParams = Field(default_factory=GenParams)
    parent_id: str | None = None
    lineage_depth: int = 0
    mutation_note: str = ""
    compiled_hash: str = ""
    status: Status = Status.CANDIDATE


class BenchmarkCase(BaseModel):
    case_id: str
    category: str
    input: dict[str, Any]
    expected: dict[str, Any] | None = None
    difficulty: Difficulty
    split: Split
    source: str = "seed"
    label_confidence: float = Field(default=1.0, ge=0, le=1)


class MetricResult(BaseModel):
    name: str
    passed: bool
    score: float = Field(ge=0, le=1)
    detail: str = ""


class RunManifest(BaseModel):
    run_id: str
    model_tag: str
    judge_tag: str = "none"
    rubric_version: str = "v0"
    dataset_snapshot_id: str
    compiler_sha: str
    evaluator_sha: str
    completed: bool = False


class FailureTheme(BaseModel):
    theme_id: str
    category: str
    label: str
    exemplar_case_ids: list[str]


class GateVerdict(BaseModel):
    challenger_id: str
    champion_id: str
    stage: str
    promote: bool
    train_delta: float
    holdout_delta: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_holdout: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    sign_test_p: float | None = None
    mde: float | None = None
    fence_failures: list[str] = Field(default_factory=list)
    note: str = ""

    @model_validator(mode="after")
    def promotion_needs_evidence(self) -> "GateVerdict":
        """
        A promotion via the sequential gate requires a positive CI lower bound.
        Vault confirmations (stage='vault') are allowed to promote based on delta alone.
        """
        if self.promote and self.stage == "sequential":
            if self.ci_low is None or self.ci_low <= 0:
                raise ValueError("a sequential promotion requires a positive sequential confidence bound")
        return self
