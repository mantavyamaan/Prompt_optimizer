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

    async def generate(self, prompt: str, case_input: dict, expected: dict | None, gen_params) -> str:
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


class HTTPBackend:
    """OpenAI-compatible endpoint adapter (Ollama, vLLM, llama.cpp, etc.)."""

    def __init__(self, base_url: str, model: str, api_key: str = "x") -> None:
        self.base_url = base_url.rstrip("/")
        self.model_tag = model
        self._client = httpx.AsyncClient(timeout=60, headers={"Authorization": f"Bearer {api_key}"})

    async def generate(self, prompt: str, case_input: dict, expected: dict | None, gen_params) -> str:
        for attempt in range(4):
            try:
                response = await self._client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={"model": self.model_tag, "messages": [{"role": "user", "content": prompt}],
                          "temperature": gen_params.temperature, "seed": gen_params.seed,
                          "max_tokens": gen_params.max_tokens},
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception:
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    async def close(self) -> None:
        await self._client.aclose()
