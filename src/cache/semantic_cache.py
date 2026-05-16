"""
Semantic Cache (Supabase + pgvector)
--------------------------------------
Before any model call, embed the query and search Supabase for a
near-duplicate past answer using cosine similarity. On hit, return
instantly — zero model cost.

Supabase setup required:
  1. Enable pgvector extension in Supabase SQL editor:
       create extension if not exists vector;
  2. Create the cache table (see sql/create_cache_table.sql)
"""

from __future__ import annotations
import os
import time
from typing import Optional
from loguru import logger


# ── Module-level singletons shared by SemanticCache and KnowledgeCache ────────
# Loading the 90 MB embedding model twice would waste memory and startup time.

_shared_embedder = None
_shared_supabase  = None


def get_shared_embedder():
    """Return the shared SentenceTransformer instance (loads once)."""
    global _shared_embedder
    if _shared_embedder is None:
        from sentence_transformers import SentenceTransformer
        _shared_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Shared embedding model loaded (all-MiniLM-L6-v2)")
    return _shared_embedder


def get_shared_supabase():
    """Return the shared Supabase client (connects once)."""
    global _shared_supabase
    if _shared_supabase is None:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise EnvironmentError("Set SUPABASE_URL and SUPABASE_SERVICE_KEY env vars")
        _shared_supabase = create_client(url, key)
        logger.info("Shared Supabase client connected")
    return _shared_supabase


class SemanticCache:
    """
    Semantic query cache backed by Supabase pgvector.

    Parameters
    ----------
    similarity_threshold : float
        Cosine similarity above which a cached entry is returned.
    ttl_hours : int
        Entries older than this are ignored.
    """

    def __init__(self, similarity_threshold: float = 0.92, ttl_hours: int = 24):
        self.threshold = similarity_threshold
        self.ttl_seconds = ttl_hours * 3600
        logger.info(f"SemanticCache ready (threshold={similarity_threshold})")

    def lookup(self, query: str) -> Optional[dict]:
        """Return cached result if a similar query exists, else None."""
        try:
            embedding = self._embed(query)
            client = self._get_supabase()

            # pgvector cosine similarity search via Supabase RPC
            result = client.rpc(
                "match_cache",
                {
                    "query_embedding": embedding,
                    "match_threshold": self.threshold,
                    "match_count": 1,
                    "max_age_seconds": self.ttl_seconds,
                }
            ).execute()

            if result.data:
                logger.debug("Cache HIT")
                return result.data[0]
            logger.debug("Cache MISS")
            return None

        except Exception as e:
            logger.warning(f"Cache lookup failed (degrading gracefully): {e}")
            return None

    def store(self, query: str, answer: str, complexity_score: float, breakdown) -> None:
        """Store a query-answer pair in the cache."""
        try:
            embedding = self._embed(query)
            client = self._get_supabase()
            client.table("semantic_cache").insert({
                "query": query,
                "answer": answer,
                "embedding": embedding,
                "complexity_score": complexity_score,
                "created_at": int(time.time()),
            }).execute()
            logger.debug("Stored in cache")
        except Exception as e:
            logger.warning(f"Cache store failed: {e}")

    def _embed(self, text: str) -> list[float]:
        return get_shared_embedder().encode(text, normalize_embeddings=True).tolist()

    def _get_supabase(self):
        return get_shared_supabase()
