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
        return '{"error":"malformed"'  # intentionally invalid JSON


_ui_lock = asyncio.Lock()
_active_bg_requests = set()

class HTTPBackend:
    """OpenAI-compatible endpoint adapter (Ollama, vLLM, llama.cpp, etc.)."""

    def __init__(self, base_url: str, model: str, api_key: str = "x") -> None:
        self.base_url = base_url.rstrip("/")
        self.model_tag = model
        self._client = httpx.AsyncClient(timeout=600.0, headers={"Authorization": f"Bearer {api_key}"})

    async def _do_post(self, prompt: str, gen_params) -> str:
        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": self.model_tag, "messages": [{"role": "user", "content": prompt}],
                  "temperature": gen_params.temperature, "seed": gen_params.seed,
                  "max_tokens": gen_params.max_tokens},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def generate(self, prompt: str, case_input: dict, expected: dict | None, gen_params, priority: int = 0) -> str:
        if priority == 1:
            async with _ui_lock:
                # Cancel running background tasks immediately
                for req in list(_active_bg_requests):
                    req.cancel()
                if _active_bg_requests:
                    await asyncio.sleep(0.5)  # Wait for cancellations to propagate

                for attempt in range(4):
                    try:
                        return await self._do_post(prompt, gen_params)
                    except Exception:
                        if attempt == 3:
                            raise
                        await asyncio.sleep(2 ** attempt)
                raise RuntimeError("unreachable")
        else:
            for attempt in range(4):
                last_exc = None
                while True:
                    # Wait for UI requests to finish
                    async with _ui_lock:
                        pass
                        
                    req_task = asyncio.create_task(self._do_post(prompt, gen_params))
                    _active_bg_requests.add(req_task)
                    try:
                        return await req_task
                    except asyncio.CancelledError:
                        # Cancelled by a UI request. The while loop will restart and wait on _ui_lock!
                        continue
                    except Exception as e:
                        last_exc = e
                        break
                    finally:
                        _active_bg_requests.discard(req_task)
                
                if attempt == 3:
                    raise last_exc
                await asyncio.sleep(2 ** attempt)
            raise RuntimeError("unreachable")

    async def close(self) -> None:
        await self._client.aclose()
