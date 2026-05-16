"""
Routing Tracker
----------------
Logs every routing decision to MLflow + Supabase for:
- Experiment tracking (which tier handled what)
- Cost-savings computation (LLM calls avoided)
- Dashboard metrics (FastAPI serves these to the UI)
"""

from __future__ import annotations
import os
import time
from loguru import logger

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


class RoutingTracker:
    def __init__(self, experiment: str = "tiered-nlp-router"):
        self.experiment = experiment
        self._supabase = None
        self._total_queries = 0
        self._llm_calls = 0
        self._cache_hits = 0

        if _MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(experiment)
                logger.info(f"MLflow experiment: {experiment}")
            except Exception as e:
                logger.warning(f"MLflow setup failed: {e}")

    def log(self, result, query: str, task: str = "classify") -> None:
        self._total_queries += 1
        if result.cache_hit:
            self._cache_hits += 1
        if result.tier_used == 3:
            self._llm_calls += 1

        # MLflow logging
        if _MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"query_{self._total_queries}", nested=True):
                    mlflow.log_metrics({
                        "tier_used": result.tier_used,
                        "confidence": result.confidence,
                        "complexity_score": result.complexity_score,
                        "latency_ms": result.latency_ms,
                        "cache_hit": int(result.cache_hit),
                        "llm_call_rate": self._llm_calls / self._total_queries,
                    })
                    mlflow.log_param("task_type", result.metadata.get("task", "unknown"))
            except Exception as e:
                logger.debug(f"MLflow log skipped: {e}")

        # Supabase logging
        self._log_supabase(result, query, task)

    def summary(self) -> dict:
        return {
            "total_queries": self._total_queries,
            "llm_calls": self._llm_calls,
            "cache_hits": self._cache_hits,
            "llm_call_rate": round(self._llm_calls / max(self._total_queries, 1), 3),
            "cache_hit_rate": round(self._cache_hits / max(self._total_queries, 1), 3),
            "estimated_savings_pct": round(
                (1 - self._llm_calls / max(self._total_queries, 1)) * 100, 1
            ),
        }

    def _log_supabase(self, result, query: str, task: str = "classify") -> None:
        try:
            client = self._get_supabase()
            client.table("routing_logs").insert({
                "query_preview": query[:120],
                "tier_used": result.tier_used,
                "confidence": result.confidence,
                "complexity_score": result.complexity_score,
                "latency_ms": result.latency_ms,
                "cache_hit": result.cache_hit,
                "task_type": task,
                "created_at": int(time.time()),
            }).execute()
        except Exception:
            pass  # Non-critical

    def _get_supabase(self):
        if self._supabase is None:
            from supabase import create_client
            self._supabase = create_client(
                os.getenv("SUPABASE_URL"),
                os.getenv("SUPABASE_SERVICE_KEY"),
            )
        return self._supabase
