from __future__ import annotations

import asyncio
import os
import sys
import time
import random
from pathlib import Path

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .backends import HTTPBackend
from .compiler import compile_prompt
from .db import conn, init_db, new_id
from .runner import load_prompt


# ── Backend singleton ──────────────────────────────────────────────────────────
# Initialised in lifespan — never at import time — so the httpx.AsyncClient is
# always created inside a running event loop and environment variables are fully
# resolved by the time the process starts.
BACKEND: HTTPBackend | None = None


def build_backend() -> HTTPBackend:
    """Use local Ollama by default; server configuration can override it."""
    if os.getenv("OPTIMIZER_BACKEND", "ollama").lower() == "http":
        base_url = os.getenv("OPTIMIZER_BASE_URL")
        model = os.getenv("OPTIMIZER_MODEL")
        if not base_url or not model:
            raise RuntimeError("OPTIMIZER_BASE_URL and OPTIMIZER_MODEL are required for the HTTP backend")
        return HTTPBackend(base_url, model, os.getenv("OPTIMIZER_API_KEY", "x"))
    return HTTPBackend(
        os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global BACKEND
    # ── One-time startup ───────────────────────────────────────────────────────
    init_db()                       # Schema initialisation — only here, not per request.
    from .seed import seed_all
    seed_all()                      # Idempotent: no-op if already seeded.
    BACKEND = build_backend()

    # Add project root to sys.path for auto_evolve — deduplicated.
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

    from auto_evolve import run_loop
    # Pass the shared backend so auto_evolve doesn't create a second HTTPBackend.
    evolution_task = asyncio.create_task(run_loop(BACKEND))

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    evolution_task.cancel()
    try:
        await evolution_task
    except asyncio.CancelledError:
        pass
    if BACKEND is not None:
        await BACKEND.close()


app = FastAPI(title="Prompt Optimizer", version="0.1.0", lifespan=lifespan)
STATIC_DIR = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Cache: category → champion prompt_id. Invalidated on every promotion.
CHAMPION_CACHE: dict[str, str] = {}
_cache_lock: asyncio.Lock | None = None


def _get_cache_lock() -> asyncio.Lock:
    global _cache_lock
    if _cache_lock is None:
        _cache_lock = asyncio.Lock()
    return _cache_lock


def classify(text: str) -> tuple[str, float]:
    # The browser product is a prompt architect.
    # The extraction benchmark is retained for optimizer experiments only.
    return "prompt_design", 0.96


async def get_active_prompt_id(category: str) -> str | None:
    """Return the current champion prompt_id, with optional 10% shadow traffic."""
    async with _get_cache_lock():
        if category not in CHAMPION_CACHE:
            with conn() as connection:
                row = connection.execute(
                    "SELECT prompt_id FROM prompts WHERE category=? AND status='champion'",
                    (category,),
                ).fetchone()
            if row is None:
                return None
            CHAMPION_CACHE[category] = row["prompt_id"]

    champion = CHAMPION_CACHE[category]

    # Shadow Deployment: route 10% of traffic to the top pending candidate.
    if random.random() < 0.10:
        with conn() as connection:
            candidate = connection.execute(
                "SELECT new_champion FROM promotions WHERE category=? AND vault_confirmed IS NULL ORDER BY created_at DESC LIMIT 1",
                (category,),
            ).fetchone()
            if candidate:
                return candidate["new_champion"]

    return champion


# ── Request models ─────────────────────────────────────────────────────────────

class ProviderConfig(BaseModel):
    """Credentials are request-only: never persisted or logged by Prompt Optimizer."""
    base_url: str = Field(min_length=8, max_length=500)
    model: str = Field(min_length=1, max_length=200)
    api_key: str = Field(min_length=1, max_length=1000)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.rstrip("/")
        if not value.startswith(("https://", "http://")):
            raise ValueError("base_url must begin with http:// or https://")
        return value


class Query(BaseModel):
    text: str = Field(min_length=1)
    provider: ProviderConfig | None = None
    conversation_id: str | None = Field(default=None, max_length=100)


class Feedback(BaseModel):
    trace_id: str
    # Score is validated to 0–100 to prevent corrupt A/B promotion averages.
    signal: str = Field(pattern=r"^(thumbs_up|thumbs_down|format_fail|retry|score:(?:[0-9]|[1-9][0-9]|100))$")


class ConversationCreate(BaseModel):
    title: str = Field(default="New prompt", max_length=120)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/query")
async def query(request: Query):
    if BACKEND is None:
        raise HTTPException(status_code=503, detail="Server is still starting up — please retry in a moment.")

    started = time.perf_counter()
    category, confidence = classify(request.text)
    prompt_id = await get_active_prompt_id(category)
    if prompt_id is None:
        raise HTTPException(status_code=503, detail="No champion is deployed")

    prompt = load_prompt(prompt_id)

    if request.conversation_id is not None:
        with conn() as connection:
            known = connection.execute(
                "SELECT 1 FROM conversations WHERE conversation_id=?", (request.conversation_id,)
            ).fetchone()
        if known is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

    compiled = compile_prompt(prompt.modules, {"user_input": request.text})

    # An API key overrides Ollama for this request only.
    # The temporary backend is created inside the try block to guarantee cleanup.
    temporary_backend: HTTPBackend | None = None
    try:
        if request.provider is not None:
            temporary_backend = HTTPBackend(
                request.provider.base_url, request.provider.model, request.provider.api_key
            )
            active_backend = temporary_backend
        else:
            active_backend = BACKEND

        output = await active_backend.generate(
            compiled, {"text": request.text}, None, prompt.gen_params, priority=1
        )
    except Exception as exc:
        message = (
            "The selected API provider could not be reached."
            if request.provider
            else "Ollama is unavailable. Start Ollama and ensure llama3.1:latest is installed, then try again."
        )
        raise HTTPException(status_code=503, detail=message) from exc
    finally:
        if temporary_backend is not None:
            await temporary_backend.close()

    trace_id = new_id("trace")
    latency = int((time.perf_counter() - started) * 1000)
    with conn() as connection:
        connection.execute(
            """INSERT INTO traces(trace_id,query,category,confidence,prompt_id,compiled_hash,output,latency_ms,model_tag,conversation_id,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
            (
                trace_id, request.text, category, confidence, prompt_id,
                prompt.compiled_hash, output, latency,
                active_backend.model_tag, request.conversation_id,
            ),
        )
        if request.conversation_id is not None:
            connection.execute(
                "UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE conversation_id=?",
                (request.conversation_id,),
            )

    return {
        "trace_id": trace_id,
        "category": category,
        "confidence": confidence,
        "output": output,
        "backend": active_backend.model_tag,
        "source": "api_key" if request.provider else "ollama",
        "conversation_id": request.conversation_id,
    }


@app.post("/feedback")
async def feedback(request: Feedback):
    with conn() as connection:
        trace = connection.execute(
            "SELECT prompt_id FROM traces WHERE trace_id=?", (request.trace_id,)
        ).fetchone()
        if trace is None:
            raise HTTPException(status_code=404, detail="Unknown trace")
        # Each feedback submission gets its own row — history is preserved.
        connection.execute(
            "INSERT INTO feedback(feedback_id, trace_id, prompt_id, signal) VALUES(?,?,?,?)",
            (new_id("feedback"), request.trace_id, trace["prompt_id"], request.signal),
        )
    return {"ok": True}


@app.get("/traces")
async def recent_traces(limit: int = 12):
    """Recent local activity. prompt_id is omitted to preserve A/B test blindness."""
    limit = max(1, min(limit, 50))
    with conn() as connection:
        rows = connection.execute(
            """SELECT trace_id, category, latency_ms, model_tag, created_at
            FROM traces ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return {"traces": [dict(row) for row in rows]}


@app.get("/health")
async def health():
    return {
        "ok": True,
        "backend": BACKEND.model_tag if BACKEND else "not_ready",
        "mode": "ollama",
        "api_key_override": "available",
    }


@app.post("/conversations")
async def create_conversation(request: ConversationCreate):
    cid = new_id("conv")
    with conn() as connection:
        connection.execute(
            "INSERT INTO conversations(conversation_id, title) VALUES(?,?)",
            (cid, request.title),
        )
    return {"conversation_id": cid, "title": request.title}
