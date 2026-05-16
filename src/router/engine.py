"""
Router Engine
-------------
Orchestrates the full routing pipeline:
  1. Semantic cache lookup (Supabase pgvector)
  2. Complexity scoring
  3. Tier selection (Tier 1 → 2 → 3)
  4. Confidence-based escalation
  5. Result logging to MLflow + Supabase
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml
from loguru import logger

from src.scoring.complexity import ComplexityScorer, ComplexityBreakdown
from src.tiers.tier1_classical import Tier1Classical
from src.tiers.tier2_small_model import Tier2SmallModel
from src.tiers.tier3_llm import Tier3LLM
from src.cache.semantic_cache import SemanticCache
from src.cache.unified_store import UnifiedKnowledgeStore
from src.monitoring.tracker import RoutingTracker
from src.router.classifier import AdaptiveRouter
from src.inference.backend import init_backend, get_backend


@dataclass
class RouteResult:
    answer: str
    tier_used: int                        # 0 = cache, 1, 2, or 3
    confidence: float
    complexity_score: float
    complexity_breakdown: ComplexityBreakdown
    latency_ms: float
    cache_hit: bool
    llm_calls_saved: bool                 # True if Tier 1 or 2 handled it
    store_hit: bool = False               # True if answered from unified knowledge store
    store_method: str = ""               # "extractive" | "synthesis" | ""
    metadata: dict = field(default_factory=dict)


class RouterEngine:
    """
    Main entry point. Instantiate once and call .route(query) per request.

    Parameters
    ----------
    config_path : str
        Path to configs/router_config.yaml
    """

    def __init__(self, config_path: str = "configs/router_config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        r = self.cfg["routing"]
        self.use_adaptive = r.get("use_adaptive_router", False)
        self.tier1_thresh = r["tier1_threshold"]
        self.tier2_thresh = r["tier2_threshold"]
        self.confidence_floor = r["confidence_floor"]

        logger.info("Initialising complexity scorer...")
        self.scorer = ComplexityScorer(
            weights=self.cfg["complexity"]["weights"]
        )

        logger.info("Initialising cache...")
        self.cache = SemanticCache(
            similarity_threshold=self.cfg["cache"]["similarity_threshold"],
            ttl_hours=self.cfg["cache"]["ttl_hours"],
        )

        # Initialise shared inference backend (GPT4All or Groq)
        inf = self.cfg.get("inference", {})
        init_backend(
            provider=inf.get("provider", "gpt4all"),
            model=inf.get("model", "phi3-mini"),
        )

        # Unified knowledge store — vector + keyword dual-indexed, no entry limit
        self.store: Optional[UnifiedKnowledgeStore] = None
        try:
            self.store = UnifiedKnowledgeStore()
            logger.info("Unified knowledge store initialised")
        except Exception as exc:
            logger.warning(f"Unified store disabled — init failed: {exc}")

        logger.info("Initialising tiers...")
        self.tier1 = Tier1Classical()
        self.tier2 = Tier2SmallModel(self.cfg["tier2"])
        self.tier3 = Tier3LLM(self.cfg["tier3"])

        if self.use_adaptive:
            logger.info("Initialising adaptive router...")
            self.adaptive_router = AdaptiveRouter()
        else:
            self.adaptive_router = None

        self.tracker = RoutingTracker(
            experiment=self.cfg["monitoring"]["mlflow_experiment"]
        )
        logger.info("RouterEngine ready.")

    def route(self, query: str, task: str = "classify") -> RouteResult:
        """
        Route a query through the tier system and return a RouteResult.

        Parameters
        ----------
        query : str
            The raw user query.
        task : str
            NLP task type hint: 'classify' | 'summarize' | 'qa' | 'generate'
        """
        t0 = time.perf_counter()

        # ── Step 1: Semantic cache lookup ─────────────────────────────
        cached = self.cache.lookup(query)
        if cached:
            latency = (time.perf_counter() - t0) * 1000
            result = RouteResult(
                answer=cached["answer"],
                tier_used=0,
                confidence=1.0,
                complexity_score=cached.get("complexity_score", 0.0),
                complexity_breakdown=cached.get("breakdown"),
                latency_ms=round(latency, 2),
                cache_hit=True,
                llm_calls_saved=True,
                metadata={"cache_key": cached.get("id")},
            )
            self.tracker.log(result, query, task)
            return result

        # ── Step 2: Unified knowledge store lookup ────────────────────
        # Combines vector-similarity and keyword-overlap into one score.
        # Extractive path (score ≥ 0.78): TF-IDF sentence ranking, zero model calls.
        # Synthesis path  (score ≥ 0.62): compressed LLM prompt, ~200 tokens.
        # Returns None if context doesn't answer the specific question — fall through.
        if self.store is not None:
            context, max_score = self.store.lookup(query)
            if context:
                answer, confidence = self.store.answer(query, context, max_score, get_backend())
                if answer is not None:
                    method = "extractive" if max_score >= 0.78 else "synthesis"
                    breakdown = self.scorer.score(query)
                    latency   = (time.perf_counter() - t0) * 1000
                    result = RouteResult(
                        answer=answer,
                        tier_used=2,
                        confidence=confidence,
                        complexity_score=breakdown.overall,
                        complexity_breakdown=breakdown,
                        latency_ms=round(latency, 2),
                        cache_hit=False,
                        llm_calls_saved=True,
                        store_hit=True,
                        store_method=method,
                    )
                    self.tracker.log(result, query, task)
                    return result
                logger.debug("Unified store: relevance miss — falling through to routing")

        breakdown = self.scorer.score(query)
        c = breakdown.overall
        logger.debug(f"Complexity={c:.3f} | query={query[:60]!r}")

        # ── Step 5: Route through tiers with confidence fallback ───────
        answer, confidence, tier_used = self._dispatch(query, task, c)

        # ── Step 6: Store result in all caches ─────────────────────────
        self.cache.store(
            query=query,
            answer=answer,
            complexity_score=c,
            breakdown=breakdown,
        )

        # After every Tier 3 LLM call: store answer in the unified knowledge
        # store so future similar/related queries can skip the LLM entirely.
        if tier_used == 3 and self.store is not None:
            self.store.store(query, answer)

        latency = (time.perf_counter() - t0) * 1000
        result = RouteResult(
            answer=answer,
            tier_used=tier_used,
            confidence=confidence,
            complexity_score=c,
            complexity_breakdown=breakdown,
            latency_ms=round(latency, 2),
            cache_hit=False,
            llm_calls_saved=(tier_used < 3),
            store_hit=False,
            store_method="",
        )
        self.tracker.log(result, query, task)
        return result

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, query: str, task: str, complexity: float
    ) -> tuple[str, float, int]:
        """Dispatch to tiers using adaptive router or complexity thresholds."""

        if self.adaptive_router:
            # Use learned model — get both tier and XGBoost confidence
            predicted_tier, router_confidence = self.adaptive_router.predict_with_confidence(
                complexity_score=complexity,
                task=task,
                query_length=len(query),
                query_text=query,
            )
            logger.debug(
                f"Adaptive router → tier {predicted_tier} (conf={router_confidence:.2f})"
            )
            # If the router itself is uncertain, start from tier 1 to be safe
            if self.adaptive_router.is_uncertain(complexity, task, len(query), query):
                logger.debug("Router uncertain — starting cascade from tier 1")
                predicted_tier = 1
        else:
            # Fallback to complexity thresholds
            if complexity <= self.tier1_thresh:
                predicted_tier = 1
            elif complexity <= self.tier2_thresh:
                predicted_tier = 2
            else:
                predicted_tier = 3

        # Try tiers starting from predicted, with confidence fallback
        if predicted_tier <= 1:
            answer, conf = self.tier1.handle(query, task)
            if conf >= self.confidence_floor:
                logger.debug(f"Tier 1 handled (conf={conf:.2f})")
                return answer, conf, 1
            logger.debug(f"Tier 1 conf={conf:.2f} < floor — escalating")

        if predicted_tier <= 2:
            answer, conf = self.tier2.handle(query, task)
            if conf >= self.confidence_floor:
                logger.debug(f"Tier 2 handled (conf={conf:.2f})")
                return answer, conf, 2
            logger.debug(f"Tier 2 conf={conf:.2f} < floor — escalating")

        # Tier 3: LLM — last resort
        answer, conf = self.tier3.handle(query, task)
        logger.debug(f"Tier 3 (LLM) handled (conf={conf:.2f})")
        return answer, conf, 3
