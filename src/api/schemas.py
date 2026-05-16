"""Pydantic schemas for the FastAPI router API."""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="The NLP query to route")
    task: str = Field("classify", description="Task type: classify | sentiment | qa | summarize | generate")


class ComplexityDetail(BaseModel):
    entropy: float
    oov_rate: float
    perplexity_score: float
    entity_density: float
    dep_tree_depth: float
    overall: float


class RouteResponse(BaseModel):
    answer: str
    tier_used: int = Field(..., description="0=cache, 1=classical, 2=small model, 3=LLM")
    confidence: float
    complexity_score: float
    complexity_breakdown: Optional[ComplexityDetail]
    latency_ms: float
    cache_hit: bool
    llm_calls_saved: bool
    store_hit: bool = False       # answered by unified knowledge store
    store_method: str = ""        # "extractive" | "synthesis" | ""


class MetricsSummary(BaseModel):
    total_queries: int
    llm_calls: int
    cache_hits: int
    llm_call_rate: float
    cache_hit_rate: float
    estimated_savings_pct: float


class SwitchBackendRequest(BaseModel):
    provider: str = Field(
        ...,
        description="'gpt4all' for local inference or 'groq' for cloud API",
        examples=["gpt4all", "groq"],
    )
    model: str = Field(
        ...,
        description=(
            "GPT4All keys: llama3-8b | deepseek-r1-8b | llama3.2-3b | phi3-mini  "
            "Groq keys: llama3-8b | llama3.1-8b | llama3.3-70b | mixtral-8x7b"
        ),
        examples=["llama3-8b", "phi3-mini"],
    )


class BackendStatusResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    provider: str
    model_key: str
    loaded: bool
    available_gpt4all_models: List[str]
    available_groq_models: List[str]
