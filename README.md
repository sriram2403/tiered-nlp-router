# Tiered NLP Router

A production-ready system that intelligently decides how to answer a query — using the cheapest method that still gives a correct result. It combines classical NLP, small local models, a two-layer knowledge cache, and a large language model, routing each query to whichever tier can handle it. Typically **60–80% of queries never reach the LLM at all**.

---

## The Problem It Solves

Every query sent to an LLM costs money and takes time. Most queries don't need a full LLM — a repeat question should come from cache, a simple greeting needs no model at all, and a moderately complex question can often be answered from knowledge already stored. Only genuinely novel, hard questions should reach the LLM.

The router scores every query for complexity before doing anything else, then routes it to the appropriate tier. If a tier isn't confident enough, it escalates automatically.

---

## How It Works

### Query Journey

```text
Incoming query
      │
      ▼
┌─────────────────────────────┐
│  Tier 0 — Semantic Cache    │  Near-duplicate of a past query?
│  pgvector cosine sim ≥ 0.92 │  → Return stored answer instantly (0 model calls)
└─────────────────────────────┘
      │ miss
      ▼
┌─────────────────────────────┐
│  Knowledge Store            │  Topic covered by a past LLM answer?
│  vector + keyword dual index│  → Extract or synthesise from stored knowledge
└─────────────────────────────┘
      │ miss / below threshold
      ▼
┌─────────────────────────────┐
│  Complexity Scorer          │  Score 0–1 across 5 signals
│  Decides starting tier      │
└─────────────────────────────┘
      │
      ├── score ≤ 0.35 → Tier 1 (Classical NLP)
      ├── score ≤ 0.65 → Tier 2 (Small Local Model)
      └── score  > 0.65 → Tier 3 (LLM)
                │
                └── If confidence < 0.72 → escalate to next tier
      │
      ▼
┌─────────────────────────────┐
│  Store answer in both caches│  Cache + Knowledge Store updated after every
│  for future reuse           │  LLM call
└─────────────────────────────┘
```

---

## Complexity Scoring — How It Thinks

Before routing, every query is scored across five independent signals. The weighted sum becomes a single complexity score between 0 (simple) and 1 (hard).

| Signal | What it measures | Weight |
| --- | --- | --- |
| **Shannon entropy** | How unpredictable the token distribution is. High entropy = unusual or diverse vocabulary. | 0.25 |
| **OOV rate** | What fraction of words are outside the common vocabulary. Technical jargon scores high. | 0.20 |
| **Perplexity** | How surprising the word sequence is to a bigram language model trained on common text. | 0.25 |
| **Entity density** | Named entities per token. More people, places, organisations = more factual complexity. | 0.15 |
| **Dependency tree depth** | How deep the parse tree goes. Nested clauses and subordination score high. | 0.15 |

A query like *"hello"* scores near 0.0. *"Analyze the interaction between epsilon-differential privacy and dataset subsampling in stochastic gradient descent"* scores near 1.0.

The weights sum to 1.0 and are configurable in `configs/router_config.yaml`.

---

## Tier 1 — Classical NLP

Handles simple, deterministic cases using rule-based matching and statistical models. No neural network involved.

- **Intent detection** — Regex rules for greetings, farewells, time queries, weather, affirmations, and sentiment requests. Each rule carries a confidence score (0.88–0.98). If the rule fires above the 0.72 confidence floor, the response is immediate.
- **Sentiment analysis** — VADER polarity scoring via NLTK. Returns a label (positive / negative / neutral) with the compound score.
- **Named entity recognition** — spaCy `en_core_web_sm`. Extracts and labels entities (person, organisation, location, etc.).

If no rule matches, Tier 1 returns confidence 0.40, which is below the 0.72 floor, triggering escalation to Tier 2.

**Cost:** Zero. **Latency:** ~2 ms.

---

## Tier 2 — Small Local Model

Uses a quantised sentence-transformer for zero-shot classification. Runs entirely on CPU or local GPU — no API call.

The query is encoded by `all-MiniLM-L6-v2` and compared against label embeddings ("question", "statement", "command", "sentiment", "factual"). The best label's cosine similarity becomes the confidence score. If this is too low, Tier 2 also attempts a short generative answer via the active inference backend. The first response that clears 0.72 confidence is returned.

**Cost:** Zero (local). **Latency:** 50–200 ms (classification) or 1–3 s (generation).

---

## Tier 3 — LLM

The last resort for queries that neither classical NLP nor the small model can handle confidently.

Before the LLM is called, the query is passed through a **prompt compressor**: TF-IDF similarity identifies which sentences from the original query are most information-dense. These are reassembled into a shorter prompt, reducing token cost without losing meaning.

Supports two backends, switchable at runtime without restarting the server:

| Backend | Models | When to use |
| --- | --- | --- |
| **GPT4All** (local) | `phi3-mini`, `llama3-8b`, `llama3.2-3b`, `deepseek-r1-8b` | Free, private, no API key needed |
| **Groq** (cloud API) | `llama3.1-8b`, `llama3.3-70b`, `mixtral-8x7b` | Fast cloud inference, needs `GROQ_API_KEY` |

After the LLM answers, the response is stored in both the **semantic cache** and the **knowledge store** so future similar questions skip the LLM entirely.

**Cost:** API tokens (Groq) or RAM (GPT4All). **Latency:** 3–10 s.

---

## Two-Layer Caching

### Layer 1 — Semantic Cache (Exact Query Match)

Every answered query is embedded with `all-MiniLM-L6-v2` and stored in Supabase with a pgvector HNSW index. On the next request, the incoming query is embedded and compared. If cosine similarity ≥ 0.92 to any stored entry (and the entry is less than 24 hours old), the cached answer is returned immediately.

This catches re-asked questions, paraphrases, and near-duplicates without any model call.

### Layer 2 — Unified Knowledge Store (Topic-Level)

When an LLM answers a question, its response is split into individual sentences. Each sentence is stored as its own row with:

- **A dense embedding** (HNSW index) — finds semantically similar future questions
- **Keywords from the source query** (GIN index on `text[]`) — finds questions using the same technical terms

On lookup, both indexes are queried in parallel. Results are merged and scored:

```text
combined = 0.60 × vector_similarity + 0.40 × keyword_overlap_ratio
```

Chunks with the highest combined score are assembled into context. Depending on the score:

- **≥ 0.78 (extractive):** The most relevant sentences are selected using TF-IDF sentence ranking. Zero model calls.
- **0.62–0.78 (synthesis):** A compressed prompt (~200 tokens) is sent to the active backend. The model is instructed to answer strictly from the provided context or respond with `INSUFFICIENT_CONTEXT` if the knowledge doesn't cover the question. If it signals insufficient context, the store is treated as a miss and routing continues normally.
- **< 0.62:** Miss — falls through to normal tier routing.

---

## Adaptive Router (Optional)

An XGBoost classifier can learn which tier handles a given query best, based on historical routing outcomes stored in Supabase. Once enough routing logs accumulate, it is trained and replaces the complexity-threshold approach.

Features used: complexity score, query length, word count, average word length, digit ratio, question-mark presence, task type (one-hot encoded).

Train it with:

```bash
python scripts/train_classifier.py
```

Enable it in `configs/router_config.yaml`:

```yaml
routing:
  use_adaptive_router: true
```

---

## Project Structure

```text
tiered-nlp-router/
├── src/
│   ├── api/
│   │   ├── main.py             # FastAPI app — /route, /metrics, /health, /admin/*
│   │   └── schemas.py          # Pydantic request/response models
│   ├── router/
│   │   ├── engine.py           # Main orchestrator — ties every component together
│   │   └── classifier.py       # XGBoost adaptive router with Optuna HPO
│   ├── cache/
│   │   ├── semantic_cache.py   # Supabase pgvector exact-query cache
│   │   └── unified_store.py    # Dual-indexed knowledge store (vector + keyword)
│   ├── tiers/
│   │   ├── tier1_classical.py  # Regex rules, VADER sentiment, spaCy NER
│   │   ├── tier2_small_model.py  # Sentence-transformer zero-shot + local generation
│   │   └── tier3_llm.py        # LLM with TF-IDF prompt compression
│   ├── scoring/
│   │   └── complexity.py       # 5-signal complexity scorer
│   ├── inference/
│   │   └── backend.py          # Hot-swappable GPT4All / Groq backend
│   └── monitoring/
│       ├── tracker.py          # MLflow + Supabase routing logs
│       └── drift.py            # Distribution drift detection (KS test + PSI)
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── configs/
│   └── router_config.yaml      # Thresholds, weights, model config
├── sql/
│   └── create_tables.sql       # Supabase schema + RPC functions
├── scripts/
│   └── train_classifier.py     # Train adaptive XGBoost router
├── tests/
│   ├── test_router.py
│   ├── test_complexity.py
│   └── test_cache.py
├── .env.example                # Environment variable template
└── requirements.txt
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/tiered-nlp-router.git
cd tiered-nlp-router
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
GROQ_API_KEY=your-groq-api-key          # Only needed for Groq backend
```

### 3. Set up Supabase

Run `sql/create_tables.sql` in the Supabase SQL editor. This creates:

- `semantic_cache` — exact-query cache table with pgvector HNSW index
- `knowledge_store` — knowledge chunks with HNSW + GIN dual index
- `routing_logs` — per-request routing decisions for analytics
- `match_cache` and `match_knowledge_store` RPC functions

### 4. Start the API server

```bash
uvicorn src.api.main:app --reload
```

The server starts at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

### 5. Start the dashboard (optional)

```bash
streamlit run dashboard/app.py
```

---

## API Reference

### `POST /route`

Route a query through the tier system.

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "how does columnar storage improve query performance?", "task": "qa"}'
```

**Response:**

```json
{
  "answer": "...",
  "tier_used": 2,
  "confidence": 0.87,
  "complexity_score": 0.54,
  "complexity_breakdown": {
    "entropy": 0.72, "oov_rate": 0.18, "perplexity_score": 0.61,
    "entity_density": 0.09, "dep_tree_depth": 0.44, "overall": 0.54
  },
  "latency_ms": 312,
  "cache_hit": false,
  "store_hit": true,
  "store_method": "extractive",
  "llm_calls_saved": true
}
```

### `GET /metrics`

```json
{
  "total_queries": 500,
  "llm_calls": 87,
  "cache_hits": 203,
  "llm_call_rate": 0.174,
  "cache_hit_rate": 0.406,
  "estimated_savings_pct": 82.6
}
```

### `POST /admin/switch-backend`

Hot-swap the inference backend without restarting:

```bash
curl -X POST http://localhost:8000/admin/switch-backend \
  -H "Content-Type: application/json" \
  -d '{"provider": "groq", "model": "llama3.3-70b"}'
```

### `GET /health` — Returns `{"status": "ok", "engine_ready": true}`

---

## Configuration

`configs/router_config.yaml`:

```yaml
routing:
  use_adaptive_router: false   # true after training the XGBoost classifier
  tier1_threshold: 0.35
  tier2_threshold: 0.65
  confidence_floor: 0.72

complexity:
  weights:
    entropy: 0.25
    oov_rate: 0.20
    perplexity: 0.25
    entity_density: 0.15
    dep_tree_depth: 0.15

cache:
  similarity_threshold: 0.92
  ttl_hours: 24

knowledge_store:
  vec_similarity_floor: 0.55
  high_threshold: 0.78         # extractive path
  low_threshold: 0.62          # synthesis path

inference:
  provider: "gpt4all"          # or "groq"
  model: "phi3-mini"
```

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| API | FastAPI + Uvicorn |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Classical NLP | spaCy `en_core_web_sm`, NLTK VADER |
| Vector cache | Supabase + pgvector (HNSW index) |
| Keyword cache | Supabase PostgreSQL (`text[]` GIN index) |
| Local LLM | GPT4All (Phi-3, Llama 3, DeepSeek R1) |
| Cloud LLM | Groq API (Llama 3.1/3.3, Mixtral) |
| Adaptive router | XGBoost + Optuna HPO |
| Monitoring | MLflow + Supabase routing logs |
| Dashboard | Streamlit |

---

## License

MIT
