"""
FastAPI Application
--------------------
Exposes the router engine as an HTTP API with:
  POST /route      — route a query
  GET  /metrics    — routing stats + LLM savings
  GET  /health     — health check
  GET  /docs       — auto-generated Swagger UI
"""

from __future__ import annotations
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(override=True)  # .env values must win over stale shell env vars

from fastapi import FastAPI, HTTPException
from loguru import logger

from src.router.engine import RouterEngine
from src.inference.backend import get_backend
from src.api.schemas import (
    QueryRequest, RouteResponse, MetricsSummary, ComplexityDetail,
    SwitchBackendRequest, BackendStatusResponse,
)

_engine: RouterEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    logger.info("Starting RouterEngine...")
    _engine = RouterEngine("configs/router_config.yaml")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Tiered NLP Router",
    description="LLM-call reduction engine with 3-tier NLP routing",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/route", response_model=RouteResponse)
async def route_query(request: QueryRequest):
    """Route a query through the tier system. Returns answer + full metadata."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Router not ready")
    try:
        result = _engine.route(request.query, request.task)
        breakdown = None
        if result.complexity_breakdown:
            b = result.complexity_breakdown
            breakdown = ComplexityDetail(
                entropy=b.entropy,
                oov_rate=b.oov_rate,
                perplexity_score=b.perplexity_score,
                entity_density=b.entity_density,
                dep_tree_depth=b.dep_tree_depth,
                overall=b.overall,
            )
        return RouteResponse(
            answer=result.answer,
            tier_used=result.tier_used,
            confidence=result.confidence,
            complexity_score=result.complexity_score,
            complexity_breakdown=breakdown,
            latency_ms=result.latency_ms,
            cache_hit=result.cache_hit,
            llm_calls_saved=result.llm_calls_saved,
            store_hit=result.store_hit,
            store_method=result.store_method,
        )
    except Exception as e:
        logger.error(f"Routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsSummary)
async def get_metrics():
    """Return routing statistics and LLM savings summary."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Router not ready")
    return _engine.tracker.summary()


@app.get("/health")
async def health():
    return {"status": "ok", "engine_ready": _engine is not None}


# ── Backend switching ──────────────────────────────────────────────────────────

@app.post("/admin/switch-backend", response_model=BackendStatusResponse)
async def switch_backend(request: SwitchBackendRequest):
    """
    Hot-swap the inference backend without restarting.

    Examples
    --------
    Switch to GPT4All — Llama 3 8B (local, free):
        {"provider": "gpt4all", "model": "llama3-8b"}

    Switch to GPT4All — DeepSeek R1 Distill 8B (local, free):
        {"provider": "gpt4all", "model": "deepseek-r1-8b"}

    Switch to GPT4All — Llama 3.2 3B (lightest, fastest local):
        {"provider": "gpt4all", "model": "llama3.2-3b"}

    Switch to GPT4All — Phi-3 Mini (default local model):
        {"provider": "gpt4all", "model": "phi3-mini"}

    Switch to Groq API — Llama 3 8B (cloud, needs GROQ_API_KEY):
        {"provider": "groq", "model": "llama3-8b"}
    """
    try:
        status = get_backend().switch(request.provider, request.model)
        return BackendStatusResponse(**status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Backend switch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/backend-status", response_model=BackendStatusResponse)
async def backend_status():
    """Return the currently active inference backend and available models."""
    status = get_backend().status()
    return BackendStatusResponse(**status)
