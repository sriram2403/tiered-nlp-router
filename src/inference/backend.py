"""
Inference Backend — Switchable GPT4All / Groq
----------------------------------------------
Single module that both Tier 2 and Tier 3 use for all generative inference.
Switch between local (GPT4All) and cloud (Groq) at any time — no restart needed.

Switch via API:
    POST /admin/switch-backend  {"provider": "gpt4all", "model": "llama3-8b"}
    POST /admin/switch-backend  {"provider": "groq",    "model": "llama3-8b"}

Switch via env var (overrides config at startup):
    INFERENCE_PROVIDER=gpt4all  INFERENCE_MODEL=phi3-mini

Available GPT4All models (must already be downloaded in GPT4All app):
    llama3-8b       → Meta-Llama-3-8B-Instruct.Q4_0.gguf
    deepseek-r1-8b  → deepseek-r1-distill-llama-8b.Q4_0.gguf
    llama3.2-3b     → Llama-3.2-3B-Instruct-Q4_0.gguf
    phi3-mini       → Phi-3-mini-4k-instruct.Q4_0.gguf

Available Groq models (requires GROQ_API_KEY):
    llama3-8b       → llama3-8b-8192
    llama3.1-8b     → llama-3.1-8b-instant
    llama3.3-70b    → llama-3.3-70b-versatile
    mixtral-8x7b    → mixtral-8x7b-32768
"""

from __future__ import annotations

import os
from typing import Optional
from loguru import logger


# ── Model registries ──────────────────────────────────────────────────────────

GPT4ALL_MODELS: dict[str, str] = {
    "llama3-8b":      "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    "deepseek-r1-8b": "deepseek-r1-distill-llama-8b.Q4_0.gguf",
    "llama3.2-3b":    "Llama-3.2-3B-Instruct-Q4_0.gguf",
    "phi3-mini":      "Phi-3-mini-4k-instruct.Q4_0.gguf",
}

GROQ_MODELS: dict[str, str] = {
    "llama3.1-8b":  "llama-3.1-8b-instant",
    "llama3.3-70b": "llama-3.3-70b-versatile",
    "llama4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "mixtral-8x7b": "mixtral-8x7b-32768",
}


# ── Backend implementations ───────────────────────────────────────────────────

class GPT4AllBackend:
    """Runs inference locally — no API key, no internet, zero cost."""

    def __init__(self, model_key: str = "phi3-mini"):
        self.model_key = model_key
        self.gguf_name = GPT4ALL_MODELS.get(model_key)
        if not self.gguf_name:
            raise ValueError(
                f"Unknown GPT4All model '{model_key}'. "
                f"Choose from: {list(GPT4ALL_MODELS)}"
            )
        self._model = None

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        model = self._load()
        try:
            with model.chat_session():
                return model.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                ).strip()
        except Exception as e:
            logger.error(f"GPT4All generation failed: {e}")
            raise

    def _load(self):
        if self._model is None:
            from gpt4all import GPT4All
            logger.info(f"Loading GPT4All model: {self.gguf_name}")
            self._model = GPT4All(self.gguf_name)
            logger.info("GPT4All model ready")
        return self._model

    def unload(self) -> None:
        """Free VRAM/RAM when switching away from this backend."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info(f"GPT4All model '{self.model_key}' unloaded")

    @property
    def info(self) -> dict:
        return {
            "provider": "gpt4all",
            "model_key": self.model_key,
            "gguf_file": self.gguf_name,
            "loaded": self._model is not None,
        }


class GroqBackend:
    """Calls the Groq cloud API — requires GROQ_API_KEY."""

    def __init__(self, model_key: str = "llama3-8b"):
        self.model_key = model_key
        self.groq_model = GROQ_MODELS.get(model_key)
        if not self.groq_model:
            raise ValueError(
                f"Unknown Groq model '{model_key}'. "
                f"Choose from: {list(GROQ_MODELS)}"
            )
        self._client = None

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY is not set")
            self._client = Groq(api_key=api_key)
            logger.info(f"Groq client initialised (model: {self.groq_model})")
        return self._client

    def unload(self) -> None:
        self._client = None

    @property
    def info(self) -> dict:
        return {
            "provider": "groq",
            "model_key": self.model_key,
            "groq_model": self.groq_model,
            "loaded": self._client is not None,
        }


# ── BackendManager singleton ──────────────────────────────────────────────────

class BackendManager:
    """
    Hot-swappable inference backend.
    Holds one active backend at a time; switching unloads the previous one.

    Usage
    -----
    Anywhere in the codebase:
        from src.inference.backend import get_backend
        answer = get_backend().generate(prompt)

    To switch provider at runtime (called by the API endpoint):
        get_backend().switch("groq", "llama3-8b")
        get_backend().switch("gpt4all", "deepseek-r1-8b")
    """

    def __init__(self, provider: str, model: str):
        self._backend: Optional[GPT4AllBackend | GroqBackend] = None
        self._provider = provider
        self._model = model
        self._init_backend(provider, model)

    # ── Core generate call ─────────────────────────────────────────────

    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.1
    ) -> str:
        return self._backend.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    # ── Hot-switch ─────────────────────────────────────────────────────

    def switch(self, provider: str, model: str) -> dict:
        """
        Switch to a different provider or model without restarting.

        Parameters
        ----------
        provider : "gpt4all" | "groq"
        model    : key from GPT4ALL_MODELS or GROQ_MODELS

        Returns current status dict.
        """
        if provider == self._provider and model == self._model:
            logger.info(f"Already on {provider}/{model} — no change")
            return self.status()

        logger.info(f"Switching backend: {self._provider}/{self._model} → {provider}/{model}")

        # Unload current to free memory before loading new
        if self._backend is not None:
            self._backend.unload()
            self._backend = None

        self._init_backend(provider, model)
        logger.info(f"Backend switched to {provider}/{model}")
        return self.status()

    # ── Status ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        base = self._backend.info if self._backend else {}
        base["available_gpt4all_models"] = list(GPT4ALL_MODELS.keys())
        base["available_groq_models"] = list(GROQ_MODELS.keys())
        return base

    # ── Internal ───────────────────────────────────────────────────────

    def _init_backend(self, provider: str, model: str) -> None:
        provider = provider.lower().strip()
        if provider == "gpt4all":
            self._backend = GPT4AllBackend(model)
        elif provider == "groq":
            self._backend = GroqBackend(model)
        else:
            raise ValueError(f"Unknown provider '{provider}'. Use 'gpt4all' or 'groq'.")
        self._provider = provider
        self._model = model


# ── Module-level singleton ────────────────────────────────────────────────────
# Initialised once on first import; both Tier 2 and Tier 3 share this instance.

_manager: Optional[BackendManager] = None


def init_backend(provider: str, model: str) -> BackendManager:
    """Call once at app startup (from RouterEngine or FastAPI lifespan)."""
    global _manager
    # Env vars override config so you can switch without editing files
    provider = os.environ.get("INFERENCE_PROVIDER", provider)
    model = os.environ.get("INFERENCE_MODEL", model)
    _manager = BackendManager(provider, model)
    logger.info(f"Inference backend initialised: {provider}/{model}")
    return _manager


def get_backend() -> BackendManager:
    """Get the shared BackendManager. Raises if init_backend() was never called."""
    if _manager is None:
        raise RuntimeError(
            "BackendManager not initialised. "
            "Call src.inference.backend.init_backend() at startup."
        )
    return _manager
