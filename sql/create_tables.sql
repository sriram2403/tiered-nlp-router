-- Enable pgvector extension
create extension if not exists vector;

-- Semantic cache table
create table if not exists semantic_cache (
  id          uuid primary key default gen_random_uuid(),
  query       text not null,
  answer      text not null,
  embedding   vector(384) not null,   -- all-MiniLM-L6-v2 dimension
  complexity_score float,
  created_at  bigint not null         -- unix timestamp
);

-- HNSW index for fast approximate nearest-neighbour search
create index if not exists cache_embedding_idx
  on semantic_cache
  using hnsw (embedding vector_cosine_ops);

-- Routing logs table
create table if not exists routing_logs (
  id              uuid primary key default gen_random_uuid(),
  query_preview   text,
  tier_used       int,
  confidence      float,
  complexity_score float,
  latency_ms      float,
  cache_hit       boolean,
  task_type       text default 'classify',   -- needed by adaptive router features
  created_at      bigint
);

-- Migration: add task_type to existing deployments (safe to run multiple times)
alter table routing_logs
  add column if not exists task_type text default 'classify';

-- ── Unified knowledge store ───────────────────────────────────────────────────
-- Dual-indexed: HNSW for vector search + GIN for keyword overlap.
-- Each row is one sentence from an LLM answer, stored with:
--   • its own dense embedding  (finds semantically similar future questions)
--   • keywords from the source query (finds questions with the same technical terms)
-- No row limit — HNSW and GIN indexes handle millions of entries efficiently.

create table if not exists knowledge_store (
  id           uuid primary key default gen_random_uuid(),
  source_query text not null,
  content      text not null,              -- one sentence / knowledge chunk
  embedding    vector(384) not null,       -- chunk embedding (all-MiniLM-L6-v2)
  keywords     text[] not null default '{}', -- lemmatised keywords from source query
  created_at   timestamptz default now()
);

-- HNSW index: sub-millisecond approximate nearest-neighbour search
create index if not exists ks_hnsw_idx
  on knowledge_store using hnsw (embedding vector_cosine_ops);

-- GIN index: fast array-overlap (&&) queries for keyword matching
create index if not exists ks_gin_idx
  on knowledge_store using gin (keywords);

-- RPC: vector search returning keywords column so Python can compute overlap
create or replace function match_knowledge_store(
  query_embedding      vector(384),
  similarity_threshold float,
  match_count          int
)
returns table (
  id           uuid,
  content      text,
  source_query text,
  keywords     text[],
  similarity   float
)
language sql stable
as $$
  select
    id,
    content,
    source_query,
    keywords,
    1 - (embedding <=> query_embedding) as similarity
  from knowledge_store
  where 1 - (embedding <=> query_embedding) > similarity_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- ── Knowledge base (topic-level knowledge cache — superseded by knowledge_store) ──
-- Stores knowledge chunks extracted from LLM answers, each embedded
-- individually so topic-adjacent queries can retrieve relevant facts without
-- making a new LLM call.

create table if not exists knowledge_base (
  id           uuid primary key default gen_random_uuid(),
  source_query text,                        -- original query that produced this chunk
  content      text not null,               -- one knowledge sentence / fact
  embedding    vector(384) not null,        -- chunk's own embedding (all-MiniLM-L6-v2)
  created_at   timestamptz default now()
);

-- HNSW index for fast nearest-neighbour lookup
create index if not exists knowledge_embedding_idx
  on knowledge_base
  using hnsw (embedding vector_cosine_ops);

-- RPC function for knowledge lookup
create or replace function match_knowledge(
  query_embedding    vector(384),
  similarity_threshold float,
  match_count        int
)
returns table (
  id          uuid,
  content     text,
  source_query text,
  similarity  float
)
language sql stable
as $$
  select
    id,
    content,
    source_query,
    1 - (embedding <=> query_embedding) as similarity
  from knowledge_base
  where 1 - (embedding <=> query_embedding) > similarity_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- ── Keyword graph (keyword-indexed knowledge links) ───────────────────────────
-- Every Tier 3 LLM answer is split into keywords and stored here.
-- Multiple rows with the same keyword form a "link" between all the queries
-- that share that keyword. New questions traverse these links to answer
-- without calling the LLM.
--
-- Link example:
--   keyword="format"  source_query="How does md differ from docx?"  answer="..."
--   keyword="format"  source_query="Why is PDF fixed-layout?"        answer="..."
--   → any future question containing "format" can draw on both answers

create table if not exists kw_query_links (
  id           uuid primary key default gen_random_uuid(),
  keyword      text not null,          -- lemmatised, lowercased (e.g. "file format")
  source_query text not null,          -- original question that produced this link
  answer       text not null,          -- LLM answer stored with this keyword
  created_at   timestamptz default now()
);

-- Index on keyword for fast IN(...) lookups
create index if not exists kw_links_keyword_idx
  on kw_query_links (keyword);

-- ── RPC function for cache lookup ─────────────────────────────────────────────
create or replace function match_cache(
  query_embedding vector(384),
  match_threshold float,
  match_count     int,
  max_age_seconds bigint
)
returns table (
  id            uuid,
  answer        text,
  complexity_score float,
  similarity    float
)
language sql stable
as $$
  select
    id,
    answer,
    complexity_score,
    1 - (embedding <=> query_embedding) as similarity
  from semantic_cache
  where
    1 - (embedding <=> query_embedding) > match_threshold
    and created_at > extract(epoch from now()) - max_age_seconds
  order by embedding <=> query_embedding
  limit match_count;
$$;
