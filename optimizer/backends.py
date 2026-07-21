from __future__ import annotations

import asyncio
import hashlib
import json

import httpx


class MockLLM:
    """Deterministic backend for exercising the optimizer without an API key."""

    model_tag = "mock-v1"

    def __init__(self, skill_overrides: dict[str, float] | None = None) -> None:
        self.skill_overrides = skill_overrides or {}

    async def generate(self, prompt: str, case_input: dict, expected: dict | None, gen_params, priority: int = 0) -> str:
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        skill = self.skill_overrides.get(digest[:16], 0.55 + (int(digest[:8], 16) % 35) / 100)
        if "PLANTED_WINNER" in prompt:
            skill = 0.99
        case_hash = hashlib.sha256(json.dumps(case_input, sort_keys=True).encode()).hexdigest()
        roll = ((int(case_hash[:8], 16) ^ int(digest[8:16], 16)) % 1000) / 1000
        await asyncio.sleep(0.001)
        if expected and "json" in expected and roll < skill:
            return json.dumps(expected["json"], sort_keys=True)
        if expected is None:
            return json.dumps({
                "mode": "demo",
                "input": case_input.get("text", ""),
                "message": "Connect an OpenAI-compatible provider to enable live model responses.",
            }, indent=2)
        return '{"error":"malformed"}'  # intentionally invalid JSON for testing


# ── Shared priority state ──────────────────────────────────────────────────────
# These are initialised lazily inside an async context to ensure the event loop
# is already running when asyncio.Lock() is created.
_ui_lock: asyncio.Lock | None = None
_active_bg_requests: set[asyncio.Task] = set()
_lock_init_guard = asyncio.Lock.__new__(asyncio.Lock)  # placeholder, replaced lazily


def _get_ui_lock() -> asyncio.Lock:
    global _ui_lock
    if _ui_lock is None:
        _ui_lock = asyncio.Lock()
    return _ui_lock


class HTTPBackend:
    """OpenAI-compatible endpoint adapter (Ollama, vLLM, llama.cpp, etc.)."""

    def __init__(self, base_url: str, model: str, api_key: str = "x") -> None:
        self.base_url = base_url.rstrip("/")
        self.model_tag = model
        self._client = httpx.AsyncClient(
            timeout=600.0,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def _do_post(self, prompt: str, gen_params) -> str:
        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model_tag,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": gen_params.temperature,
                "seed": gen_params.seed,
                "max_tokens": gen_params.max_tokens,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def generate(
        self,
        prompt: str,
        case_input: dict,
        expected: dict | None,
        gen_params,
        priority: int = 0,
    ) -> str:
        ui_lock = _get_ui_lock()

        if priority == 1:
            # UI request: cancel all background tasks and jump the queue.
            async with ui_lock:
                for req in list(_active_bg_requests):
                    req.cancel()
                if _active_bg_requests:
                    # Brief pause to let cancellations propagate before we start.
                    await asyncio.sleep(0.3)

                last_exc: BaseException = RuntimeError("generation failed after retries")
                for attempt in range(4):
                    try:
                        return await self._do_post(prompt, gen_params)
                    except Exception as exc:
                        last_exc = exc
                        if attempt < 3:
                            await asyncio.sleep(2 ** attempt)
                raise last_exc
        else:
            # Background request: yield to UI and respect cancellations.
            max_cancellations = 20
            cancellation_count = 0
            last_exc: BaseException = RuntimeError("generation failed after retries")

            for attempt in range(4):
                while True:
                    # Block until no UI request is running.
                    async with ui_lock:
                        pass

                    req_task = asyncio.current_task()
                    if req_task is not None:
                        _active_bg_requests.add(req_task)
                    try:
                        return await self._do_post(prompt, gen_params)
                    except asyncio.CancelledError:
                        cancellation_count += 1
                        if cancellation_count >= max_cancellations:
                            raise RuntimeError(f"background task cancelled {max_cancellations} times; giving up")
                        # Wait for the UI request to finish, then retry.
                        await asyncio.sleep(0.1)
                        continue
                    except Exception as exc:
                        last_exc = exc
                        break
                    finally:
                        if req_task is not None:
                            _active_bg_requests.discard(req_task)

                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)

            raise last_exc

    async def close(self) -> None:
        await self._client.aclose()
