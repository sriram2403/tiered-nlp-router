"""
Tier 2 — Small Model Inference
--------------------------------
Handles medium-complexity queries using:
  - Sentence-transformers for classification / QA (zero-shot cosine similarity)
  - Shared BackendManager for generative tasks (GPT4All or Groq, switchable live)
"""

from __future__ import annotations
from typing import Optional
from loguru import logger


class Tier2SmallModel:
    """
    Parameters
    ----------
    config : dict
        From router_config.yaml tier2 section
    """

    def __init__(self, config: dict):
        self.config = config
        self._st_model = None

    def handle(self, query: str, task: str = "classify") -> tuple[str, float]:
        if task in ("classify", "sentiment", "qa"):
            return self._classify_zeroshot(query, task)
        return self._generate(query)

    # ── Classification / QA via zero-shot similarity ──────────────────

    def _classify_zeroshot(self, query: str, task: str) -> tuple[str, float]:
        try:
            import numpy as np
            model = self._get_st_model()
            candidate_labels = ["question", "command", "statement", "opinion", "request"]
            query_emb = model.encode(query, normalize_embeddings=True)
            label_embs = model.encode(candidate_labels, normalize_embeddings=True)
            similarities = (label_embs @ query_emb).tolist()
            best_idx = int(np.argmax(similarities))
            confidence = float(similarities[best_idx])
            answer = f"Classified as: {candidate_labels[best_idx]}"
            logger.debug(f"Tier2 zeroshot → {candidate_labels[best_idx]} conf={confidence:.2f}")
            return answer, confidence
        except Exception as e:
            logger.warning(f"Tier2 classification failed: {e}")
            return "", 0.40

    # ── Generative path — delegates to shared BackendManager ─────────

    def _generate(self, query: str) -> tuple[str, float]:
        try:
            from src.inference.backend import get_backend
            max_tokens = self.config.get("max_tokens", 256)
            temperature = self.config.get("temperature", 0.1)
            answer = get_backend().generate(query, max_tokens=max_tokens, temperature=temperature)
            return answer, 0.78
        except Exception as e:
            logger.warning(f"Tier2 generation failed: {e}")
            return "", 0.30

    def _get_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            self._st_model = SentenceTransformer(model_name)
            logger.info(f"SentenceTransformer loaded: {model_name}")
        return self._st_model
