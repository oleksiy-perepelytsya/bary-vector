from __future__ import annotations

from typing import Protocol

import httpx

from lib.config import Settings


class LLM(Protocol):
    def generate(self, prompt: str, max_tokens: int | None = None) -> str: ...


class OllamaLLM:
    def __init__(self, settings: Settings):
        self._url = settings.ollama_url.rstrip("/") + "/api/generate"
        self._model = settings.llm_model
        self._client = httpx.Client(timeout=settings.llm_timeout_seconds)
        self._default_max_tokens = settings.llm_max_tokens

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        if max_tokens is None:
            max_tokens = self._default_max_tokens
        resp = self._client.post(
            self._url,
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


class FakeLLM:
    def generate(self, prompt: str, max_tokens: int | None = None) -> str:  # noqa: ARG002
        return "placeholder summary"


def get_llm(settings: Settings) -> LLM:
    if settings.fake_llm:
        return FakeLLM()
    return OllamaLLM(settings)
