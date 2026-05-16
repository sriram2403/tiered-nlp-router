"""
Tier 3 — LLM (last resort)
---------------------------
Calls the active inference backend (GPT4All or Groq — switchable live).
Applies extractive prompt compression before each call to reduce token cost
even when an LLM call is unavoidable.
"""

from __future__ import annotations
from loguru import logger


class Tier3LLM:
    """
    Parameters
    ----------
    config : dict
        From router_config.yaml tier3 section
    """

    def __init__(self, config: dict):
        self.max_tokens = config.get("max_tokens", 512)
        self.temperature = config.get("temperature", 0.1)

    def handle(self, query: str, task: str = "generate") -> tuple[str, float]:
        compressed = self._compress_prompt(query)
        tokens_saved = len(query.split()) - len(compressed.split())
        if tokens_saved > 0:
            logger.debug(f"Prompt compressed: saved ~{tokens_saved} tokens")

        try:
            from src.inference.backend import get_backend
            answer = get_backend().generate(
                compressed,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return answer, 0.92
        except Exception as e:
            logger.error(f"Tier3 call failed: {e}")
            return f"Error: {e}", 0.0

    def _compress_prompt(self, query: str) -> str:
        """
        Token-budget enforcer: TF-IDF extractive sentence compression.
        For long prompts keeps only the highest-scoring sentences.
        """
        sentences = [s.strip() for s in query.split(".") if s.strip()]
        if len(sentences) <= 3:
            return query

        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer().fit_transform(sentences)
            scores = vec.toarray().sum(axis=1)
            top_idx = sorted(np.argsort(scores)[-3:])
            return ". ".join(sentences[i] for i in top_idx) + "."
        except Exception:
            return query
