"""
Complexity Scorer
-----------------
Computes a [0,1] complexity score for an incoming NLP query using five signals:
Shannon entropy, OOV rate, n-gram perplexity, named-entity density, dep-tree depth.

Score near 0 → trivially simple (Tier 1).
Score near 1 → highly complex (Tier 3 / LLM).
"""

from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import spacy
from loguru import logger

_nlp: Optional[spacy.Language] = None


def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Run: python -m spacy download en_core_web_sm")
            raise
    return _nlp


@dataclass
class ComplexityBreakdown:
    entropy: float
    oov_rate: float
    perplexity_score: float
    entity_density: float
    dep_tree_depth: float
    overall: float


class ComplexityScorer:
    """Weighted complexity scorer. All signals normalised to [0,1]."""

    DEFAULT_WEIGHTS = {
        "entropy": 0.25,
        "oov_rate": 0.20,
        "perplexity": 0.25,
        "entity_density": 0.15,
        "dep_tree_depth": 0.15,
    }

    _NORM = {
        "entropy": 5.0,
        "oov_rate": 1.0,
        "perplexity": 500.0,
        "entity_density": 1.0,
        "dep_tree_depth": 15.0,
    }

    def __init__(self, weights: Optional[dict] = None, vocab: Optional[set] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"
        self._vocab = vocab

    def score(self, text: str) -> ComplexityBreakdown:
        nlp = _get_nlp()
        doc = nlp(text)
        tokens = [t.text.lower() for t in doc if not t.is_space]

        raw = {
            "entropy": self._entropy(tokens),
            "oov_rate": self._oov_rate(doc),
            "perplexity": self._perplexity_score(tokens),
            "entity_density": self._entity_density(doc),
            "dep_tree_depth": self._dep_tree_depth(doc),
        }

        norm = {k: min(v / self._NORM[k], 1.0) for k, v in raw.items()}
        overall = round(sum(norm[k] * self.weights[k] for k in self.weights), 4)

        return ComplexityBreakdown(
            entropy=norm["entropy"],
            oov_rate=norm["oov_rate"],
            perplexity_score=norm["perplexity"],
            entity_density=norm["entity_density"],
            dep_tree_depth=norm["dep_tree_depth"],
            overall=overall,
        )

    @staticmethod
    def _entropy(tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        n = len(tokens)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    def _oov_rate(self, doc) -> float:
        tokens = [t for t in doc if not t.is_space and not t.is_punct]
        if not tokens:
            return 0.0
        vocab = self._get_vocab()
        return sum(1 for t in tokens if t.text.lower() not in vocab) / len(tokens)

    @staticmethod
    def _perplexity_score(tokens: list[str]) -> float:
        if len(tokens) < 2:
            return 0.0
        counts = Counter(tokens)
        n = len(tokens)
        log_prob = sum(math.log(counts[t] / n) for t in tokens)
        return math.exp(-log_prob / n)

    @staticmethod
    def _entity_density(doc) -> float:
        tokens = [t for t in doc if not t.is_space]
        return len(doc.ents) / len(tokens) if tokens else 0.0

    @staticmethod
    def _dep_tree_depth(doc) -> float:
        max_depth = 0
        for sent in doc.sents:
            roots = [t for t in sent if t.head == t]
            if not roots:
                continue
            def _depth(token, visited: set) -> int:
                if token in visited:
                    return 0
                visited.add(token)
                children = list(token.children)
                return 0 if not children else 1 + max(_depth(c, visited) for c in children)
            max_depth = max(max_depth, _depth(roots[0], set()))
        return float(max_depth)

    def _get_vocab(self) -> set:
        if self._vocab is None:
            self._vocab = {lex.text.lower() for lex in _get_nlp().vocab if lex.is_alpha}
        return self._vocab
