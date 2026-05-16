"""
Unified Knowledge Store
------------------------
Merges vector-similarity and keyword-graph lookup into a single system.

Every knowledge chunk is stored with BOTH:
  • its own dense embedding  → semantic similarity search (finds paraphrases)
  • keywords from its source query → keyword-overlap search (finds same-concept questions)

Lookup combines both signals with a weighted score:
    combined = 0.60 × vector_similarity + 0.40 × keyword_overlap_ratio

This means a chunk is ranked highly when:
  • The query is semantically close to it   (vector signal)
  • The query shares technical keywords     (keyword signal)
  Either signal alone can surface a chunk; both together gives high confidence.

Answer strategy (no hardcoded entry limit — Supabase handles millions of rows):
  combined ≥ HIGH (0.78) → extractive: TF-IDF sentence ranking, zero model calls
  combined ≥ LOW  (0.62) → synthesis:  compressed LLM prompt, ~200 tokens
  combined <  LOW        → miss:       fall through to normal Tier 1/2/3 routing

After every Tier 3 LLM call, the answer is stored so future questions on the
same topic — phrased differently, using the same terms, or both — can be
answered without touching the LLM again.
"""

from __future__ import annotations

import re
from typing import Optional
from loguru import logger

from src.cache.semantic_cache import get_shared_embedder, get_shared_supabase

# ── Scoring thresholds ────────────────────────────────────────────────────────
HIGH_THRESHOLD = 0.78   # extractive path: return best sentences, no model call
LOW_THRESHOLD  = 0.62   # synthesis path: compressed LLM prompt
VEC_WEIGHT     = 0.60   # weight of vector-similarity signal
KW_WEIGHT      = 0.40   # weight of keyword-overlap signal

# ── Vector search settings ────────────────────────────────────────────────────
VEC_SIMILARITY_FLOOR = 0.55   # minimum cosine sim to include a chunk in candidates
VEC_CANDIDATE_COUNT  = 20     # how many vector-similar chunks to retrieve

# ── Keyword search settings ───────────────────────────────────────────────────
KW_CANDIDATE_LIMIT = 40       # max rows returned by keyword overlap search
MIN_KW_LEN         = 3        # ignore keywords shorter than this

# ── Chunk storage settings ────────────────────────────────────────────────────
MIN_CHUNK_LEN    = 35          # minimum sentence length worth storing
MAX_STORE_CHUNKS = 10          # max sentences stored per LLM answer

# ── Extractive quality gate ───────────────────────────────────────────────────
MIN_TFIDF_SIM = 0.15           # minimum TF-IDF sim for a sentence to appear in answer

# ── Synthesis relevance signals ───────────────────────────────────────────────
_INSUFFICIENT = (
    "insufficient_context",
    "the knowledge does not",
    "i don't have",
    "i do not have",
    "insufficient",
    "not covered",
    "cannot answer",
    "not enough",
    "the context does not",
    "no relevant",
)


class UnifiedKnowledgeStore:
    """
    Dual-indexed knowledge store: vector search + keyword overlap, merged.

    Uses the same shared embedder and Supabase client as SemanticCache
    so the 90 MB model is only loaded into memory once.

    No entry limit — Supabase's HNSW and GIN indexes handle millions of rows.
    """

    def __init__(self):
        self._nlp = None

    # ── Main lookup ───────────────────────────────────────────────────────────

    def lookup(self, query: str) -> tuple[Optional[str], float]:
        """
        Retrieve relevant knowledge chunks using both vector similarity and
        keyword overlap, then combine into a weighted score.

        Returns
        -------
        (context, max_combined_score)
            context is None if no chunk clears the LOW_THRESHOLD.
        """
        keywords  = self.extract_keywords(query)
        embedding = self._embed(query)

        # ── 1. Vector search ──────────────────────────────────────────────────
        vec_rows = self._vector_search(embedding)

        # ── 2. Keyword search ─────────────────────────────────────────────────
        kw_rows = self._keyword_search(keywords) if keywords else []

        if not vec_rows and not kw_rows:
            return None, 0.0

        # ── 3. Merge and score ────────────────────────────────────────────────
        scored = self._merge_and_score(keywords, vec_rows, kw_rows)
        if not scored:
            return None, 0.0

        # Sort by combined score, descending
        ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        top    = ranked[0]["score"]

        if top < LOW_THRESHOLD:
            logger.debug(f"Unified store: best combined={top:.3f} < {LOW_THRESHOLD} — miss")
            return None, 0.0

        # Take the top chunks that are above the minimum relevance bar
        context = "\n\n".join(
            c["content"] for c in ranked[:5] if c["score"] >= LOW_THRESHOLD * 0.80
        )
        logger.info(
            f"Unified store: {len(ranked)} candidates, "
            f"top_score={top:.3f}, context_chunks={len(context.split(chr(10)*2))}"
        )
        return context, round(top, 4)

    def answer(
        self, query: str, context: str, max_score: float, backend
    ) -> tuple[Optional[str], float]:
        """
        Produce an answer from the retrieved context.

        High score → extractive (TF-IDF sentence ranking, zero model calls).
        Medium score → synthesis (compressed LLM prompt, ~200 tokens).

        Returns (None, 0.0) if the model signals the context is insufficient,
        so the caller can fall through to normal routing.
        """
        if max_score >= HIGH_THRESHOLD:
            return self._extractive(query, context)

        return self._synthesize(query, context, backend)

    # ── Storage ───────────────────────────────────────────────────────────────

    def store(self, query: str, answer_text: str) -> None:
        """
        Extract sentences from an LLM answer and store each chunk with:
          • its own embedding (for vector search)
          • keywords from the source query (for keyword overlap search)

        Called after every Tier 3 LLM response.
        No entry limit — the table grows indefinitely.
        """
        keywords = self.extract_keywords(query)
        chunks   = self._extract_chunks(answer_text)
        if not chunks:
            return

        try:
            rows = [
                {
                    "source_query": query[:500],
                    "content":      chunk,
                    "embedding":    self._embed(chunk),
                    "keywords":     keywords,
                }
                for chunk in chunks
            ]
            get_shared_supabase().table("knowledge_store").insert(rows).execute()
            logger.info(f"Unified store: stored {len(rows)} chunk(s) with {len(keywords)} keyword(s)")
        except Exception as exc:
            logger.warning(f"Unified store: store failed — {exc}")

    # ── Keyword extraction ────────────────────────────────────────────────────

    def extract_keywords(self, text: str) -> list[str]:
        """
        Extract meaningful, lemmatised keywords using spaCy.
        Returns lowercase, stop-word-free terms: nouns, entities, noun phrases.
        """
        nlp = self._get_nlp()
        doc = nlp(text.lower())
        kws = set()

        # Named entities (PDF, JSON, XML, Microsoft, etc.)
        for ent in doc.ents:
            kw = ent.lemma_.strip()
            if len(kw) >= MIN_KW_LEN:
                kws.add(kw)

        # Noun chunk roots + short multi-word phrases
        for chunk in doc.noun_chunks:
            root = chunk.root
            if not root.is_stop and len(root.lemma_) >= MIN_KW_LEN:
                kws.add(root.lemma_)
            phrase = " ".join(
                t.lemma_ for t in chunk
                if not t.is_stop and not t.is_punct and len(t.lemma_) >= MIN_KW_LEN
            )
            if len(phrase) >= MIN_KW_LEN and " " in phrase:
                kws.add(phrase)

        # Individual nouns and proper nouns
        for token in doc:
            if (
                token.pos_ in ("NOUN", "PROPN")
                and not token.is_stop
                and not token.is_punct
                and len(token.lemma_) >= MIN_KW_LEN
            ):
                kws.add(token.lemma_)

        return list(kws)

    # ── Private: search ───────────────────────────────────────────────────────

    def _vector_search(self, embedding: list[float]) -> list[dict]:
        """HNSW cosine-similarity search via Supabase RPC."""
        try:
            resp = get_shared_supabase().rpc("match_knowledge_store", {
                "query_embedding":    embedding,
                "similarity_threshold": VEC_SIMILARITY_FLOOR,
                "match_count":        VEC_CANDIDATE_COUNT,
            }).execute()
            return resp.data or []
        except Exception as exc:
            logger.warning(f"Unified store: vector search failed — {exc}")
            return []

    def _keyword_search(self, keywords: list[str]) -> list[dict]:
        """GIN array-overlap search: rows whose keywords array overlaps with query keywords."""
        if not keywords:
            return []
        try:
            pg_array = self._pg_array_literal(keywords)
            resp = (
                get_shared_supabase()
                .table("knowledge_store")
                .select("id, content, source_query, keywords")
                .filter("keywords", "ov", pg_array)
                .limit(KW_CANDIDATE_LIMIT)
                .execute()
            )
            return resp.data or []
        except Exception as exc:
            logger.warning(f"Unified store: keyword search failed — {exc}")
            return []

    def _merge_and_score(
        self,
        query_keywords: list[str],
        vec_rows: list[dict],
        kw_rows: list[dict],
    ) -> dict[str, dict]:
        """
        Merge vector and keyword results by row ID, compute combined score.

        combined = VEC_WEIGHT × vec_sim + KW_WEIGHT × keyword_overlap_ratio
        """
        total_kws = max(len(query_keywords), 1)
        scored: dict[str, dict] = {}

        # Seed from vector search
        for row in vec_rows:
            rid = row["id"]
            scored[rid] = {
                "content":  row["content"],
                "vec_sim":  row.get("similarity", 0.0),
                "kw_ratio": 0.0,
            }
            # Compute keyword overlap from stored keywords returned by vector search
            stored_kws = set(row.get("keywords") or [])
            if stored_kws and query_keywords:
                matched = stored_kws & set(query_keywords)
                scored[rid]["kw_ratio"] = len(matched) / total_kws

        # Add / update from keyword search
        for row in kw_rows:
            rid = row["id"]
            stored_kws = set(row.get("keywords") or [])
            matched_kws = stored_kws & set(query_keywords)
            kw_ratio = len(matched_kws) / total_kws

            if rid not in scored:
                scored[rid] = {
                    "content":  row["content"],
                    "vec_sim":  0.0,
                    "kw_ratio": kw_ratio,
                }
            else:
                # Update kw_ratio (keyword search is authoritative for this signal)
                scored[rid]["kw_ratio"] = max(scored[rid]["kw_ratio"], kw_ratio)

        # Compute final combined score for every candidate
        for s in scored.values():
            s["score"] = VEC_WEIGHT * s["vec_sim"] + KW_WEIGHT * s["kw_ratio"]

        return scored

    # ── Private: answer strategies ────────────────────────────────────────────

    def _extractive(self, query: str, context: str) -> tuple[Optional[str], float]:
        """
        TF-IDF sentence ranking — picks the most relevant sentences from
        context without calling any model.
        """
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", context)
            if len(s.strip()) >= 30
        ]
        if not sentences:
            return None, 0.0

        try:
            docs  = [query] + sentences
            vec   = TfidfVectorizer(stop_words="english", max_features=1000)
            tfidf = vec.fit_transform(docs)
            sims  = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

            if float(sims.max()) < MIN_TFIDF_SIM:
                return None, 0.0

            top_idx   = sims.argsort()[::-1]
            top_sents = [
                sentences[i] for i in top_idx[:3] if sims[i] >= MIN_TFIDF_SIM
            ]
            if not top_sents:
                return None, 0.0

            return " ".join(top_sents), round(float(sims[top_idx[0]]), 4)

        except Exception as exc:
            logger.warning(f"Unified store: extractive failed — {exc}")
            return None, 0.0

    def _synthesize(
        self, query: str, context: str, backend
    ) -> tuple[Optional[str], float]:
        """
        Compressed LLM prompt — uses ~200 tokens instead of the ~1000 a
        full Tier 3 call would use.  The model is instructed to signal
        INSUFFICIENT_CONTEXT rather than hallucinate.
        """
        prompt = (
            "You are a precise assistant. Answer the question using ONLY the "
            "knowledge chunks provided below.\n\n"
            "RULES:\n"
            "• If the chunks do not contain enough information to answer the "
            "specific question, respond with exactly: INSUFFICIENT_CONTEXT\n"
            "• Do not guess or add knowledge not found in the chunks.\n"
            "• If relevant, answer in 2–4 clear sentences.\n\n"
            f"Knowledge:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        try:
            raw = backend.generate(prompt, max_tokens=220, temperature=0.05)
        except Exception as exc:
            logger.warning(f"Unified store: synthesis failed — {exc}")
            return None, 0.0

        lower = raw.strip().lower()
        if "insufficient_context" in lower:
            return None, 0.0
        for phrase in _INSUFFICIENT:
            if lower.startswith(phrase):
                return None, 0.0
        if len(raw.strip()) < 30:
            return None, 0.0

        return raw.strip(), round(LOW_THRESHOLD * 0.92, 4)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        return get_shared_embedder().encode(text, normalize_embeddings=True).tolist()

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def _extract_chunks(self, text: str) -> list[str]:
        raw    = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks = [s.strip() for s in raw if len(s.strip()) >= MIN_CHUNK_LEN]
        return chunks[:MAX_STORE_CHUNKS]

    @staticmethod
    def _pg_array_literal(values: list[str]) -> str:
        """
        Format a Python list as a PostgreSQL array literal for the Supabase
        filter.  Multi-word keywords (containing spaces) are double-quoted.
        """
        parts = []
        for v in values:
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'"{escaped}"' if " " in escaped else escaped)
        return "{" + ",".join(parts) + "}"
