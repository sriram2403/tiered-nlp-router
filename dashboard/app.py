"""
Hybrid NLP Router — Dashboard
------------------------------
Run with:  streamlit run dashboard/app.py
Requires the FastAPI server to be running at http://localhost:8000
"""

import time
import requests
import streamlit as st

API = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid NLP Router",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .tier-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 8px;
    }
    .tier-0 { background:#d1fae5; color:#065f46; }
    .tier-1 { background:#dbeafe; color:#1e40af; }
    .tier-2 { background:#fef3c7; color:#92400e; }
    .tier-3 { background:#fee2e2; color:#991b1b; }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .answer-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 16px;
        border-radius: 6px;
        font-size: 15px;
        line-height: 1.7;
    }
    .error-box {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path):
    try:
        r = requests.get(f"{API}{path}", timeout=10)
        return r.json()
    except Exception:
        return None

def api_post(path, body, timeout=300):
    try:
        r = requests.post(f"{API}{path}", json=body, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def tier_label(tier):
    labels = {0: "Cache Hit", 1: "Tier 1 — Classical NLP",
              2: "Tier 2 — Small Model", 3: "Tier 3 — LLM"}
    return labels.get(tier, f"Tier {tier}")

def tier_css(tier):
    return f"tier-{tier}"

def server_alive():
    try:
        requests.get(f"{API}/health", timeout=3)
        return True
    except Exception:
        return False


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 NLP Router")
    st.caption("LLM Call Reduction Engine")
    st.divider()

    # Server status
    alive = server_alive()
    if alive:
        st.success("API Server Online", icon="✅")
    else:
        st.error("API Server Offline", icon="🔴")
        st.info("Start with:\n```\nuvicorn src.api.main:app --reload\n```")

    st.divider()

    # Backend switcher
    st.subheader("🔀 Inference Backend")

    status = api_get("/admin/backend-status")
    if status:
        current_provider = status.get("provider", "gpt4all")
        current_model = status.get("model_key", "phi3-mini")
        loaded = status.get("loaded", False)

        st.caption(f"Active: **{current_provider} / {current_model}**")
        st.caption("🟢 Loaded" if loaded else "⚪ Not loaded yet")

        gpt4all_models = status.get("available_gpt4all_models", [])
        groq_models = status.get("available_groq_models", [])

        provider_choice = st.radio(
            "Provider",
            ["gpt4all", "groq"],
            index=0 if current_provider == "gpt4all" else 1,
            horizontal=True,
        )

        if provider_choice == "gpt4all":
            model_choice = st.selectbox(
                "Model",
                gpt4all_models,
                index=gpt4all_models.index(current_model) if current_model in gpt4all_models else 0,
            )
            st.caption("💻 Local inference — free, no API key")
        else:
            model_choice = st.selectbox(
                "Model",
                groq_models,
                index=groq_models.index(current_model) if current_model in groq_models else 0,
            )
            st.caption("☁️ Cloud API — fast, needs GROQ_API_KEY")

        if st.button("Apply", use_container_width=True, type="primary"):
            with st.spinner("Switching backend..."):
                result = api_post("/admin/switch-backend",
                                  {"provider": provider_choice, "model": model_choice})
            if "error" not in result:
                st.success(f"Switched to {provider_choice}/{model_choice}")
                st.rerun()
            else:
                st.error(result.get("error"))

    st.divider()

    # Task selector (persisted in session)
    st.subheader("⚙️ Task Type")
    task = st.selectbox(
        "Select task",
        ["classify", "qa", "summarize", "generate", "sentiment"],
        help="Tells the router what kind of NLP task you're performing",
    )

    st.divider()
    st.caption("Built with FastAPI + spaCy + XGBoost + GPT4All + Groq")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Hybrid NLP Router")
st.caption("Smart routing that decides when to call an LLM — and when not to.")

tabs = st.tabs(["💬 Query", "📊 Metrics", "📖 How It Works"])


# ── Tab 1: Query ──────────────────────────────────────────────────────────────

with tabs[0]:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Send a Query")

        query = st.text_area(
            "Your query",
            placeholder="Type anything — simple or complex...",
            height=140,
        )

        example_queries = {
            "Simple greeting (→ Tier 1)": "Hello, how are you?",
            "Sentiment (→ Tier 1)": "I love this product, it works great!",
            "Classification (→ Tier 2)": "Explain the difference between supervised and unsupervised learning",
            "Complex QA (→ Tier 3)": "Analyze the second-order effects of the US Inflation Reduction Act on European green energy subsidies",
        }

        st.caption("Or try an example:")
        for label, example in example_queries.items():
            if st.button(label, use_container_width=True):
                st.session_state["example_query"] = example
                st.rerun()

        if "example_query" in st.session_state:
            query = st.session_state.pop("example_query")

        submitted = st.button("Route Query →", type="primary",
                              use_container_width=True, disabled=not alive)

    with col_result:
        st.subheader("Result")

        if submitted and query.strip():
            with st.spinner("Routing query..."):
                t0 = time.time()
                result = api_post("/route", {"query": query, "task": task})
                elapsed = time.time() - t0

            if "error" in result:
                st.markdown(f'<div class="error-box">❌ {result["error"]}</div>',
                            unsafe_allow_html=True)
            else:
                tier = result.get("tier_used", 0)
                badge = tier_label(tier)
                css = tier_css(tier)

                # Tier badge
                st.markdown(
                    f'<div class="tier-badge {css}">{"🗄️" if tier == 0 else "⚡" if tier == 1 else "🤖" if tier == 2 else "🧠"} {badge}</div>',
                    unsafe_allow_html=True,
                )

                # Cache / savings indicator
                if result.get("cache_hit"):
                    st.success("Served from semantic cache — zero model calls")
                elif result.get("store_hit"):
                    method = result.get("store_method", "")
                    if method == "extractive":
                        st.success(
                            "Knowledge store hit — extractive answer "
                            "(TF-IDF sentence ranking, zero model calls)"
                        )
                    else:
                        st.success(
                            "Knowledge store hit — synthesised from stored knowledge "
                            "(compressed prompt, ~200 tokens vs ~1000)"
                        )
                elif result.get("llm_calls_saved"):
                    st.info("LLM call avoided — handled by classical or small model")
                else:
                    st.warning("LLM was called — answer stored for future questions on this topic")

                # Answer
                st.markdown("**Answer**")
                st.markdown(
                    f'<div class="answer-box">{result.get("answer", "")}</div>',
                    unsafe_allow_html=True,
                )

                st.divider()

                # Stats row
                c1, c2, c3 = st.columns(3)
                c1.metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")
                c2.metric("Confidence", f"{result.get('confidence', 0):.0%}")
                c3.metric("Complexity", f"{result.get('complexity_score', 0):.3f}")

                # Complexity breakdown chart
                breakdown = result.get("complexity_breakdown")
                if breakdown:
                    st.markdown("**Complexity Breakdown**")
                    signals = {
                        "Entropy": breakdown.get("entropy", 0),
                        "OOV Rate": breakdown.get("oov_rate", 0),
                        "Perplexity": breakdown.get("perplexity_score", 0),
                        "Entity Density": breakdown.get("entity_density", 0),
                        "Tree Depth": breakdown.get("dep_tree_depth", 0),
                    }
                    for name, val in signals.items():
                        col_a, col_b = st.columns([2, 3])
                        col_a.caption(name)
                        col_b.progress(min(float(val), 1.0), text=f"{val:.3f}")

        elif submitted:
            st.warning("Please enter a query first.")
        else:
            st.info("Enter a query on the left and click **Route Query →**")


# ── Tab 2: Metrics ────────────────────────────────────────────────────────────

with tabs[1]:
    st.subheader("Live Routing Metrics")

    if st.button("Refresh Metrics", type="secondary"):
        st.rerun()

    metrics = api_get("/metrics")

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Queries", metrics.get("total_queries", 0))
        c2.metric("LLM Calls", metrics.get("llm_calls", 0))
        c3.metric("Cache Hits", metrics.get("cache_hits", 0))
        c4.metric("Estimated Savings", f"{metrics.get('estimated_savings_pct', 0):.1f}%")

        st.divider()

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**LLM Call Rate**")
            llm_rate = metrics.get("llm_call_rate", 0)
            st.progress(llm_rate, text=f"{llm_rate:.1%} of queries hit the LLM")
            if llm_rate < 0.2:
                st.success("Excellent — well below 20% target")
            elif llm_rate < 0.5:
                st.warning("Good — getting there")
            else:
                st.error("High — more queries than expected hitting the LLM")

        with col_r:
            st.markdown("**Cache Hit Rate**")
            cache_rate = metrics.get("cache_hit_rate", 0)
            st.progress(cache_rate, text=f"{cache_rate:.1%} served from cache")
            if cache_rate > 0.3:
                st.success("Cache warming up well")
            else:
                st.info("Cache will improve with more repeated queries")
    else:
        st.info("No metrics yet — send some queries first.")


# ── Tab 3: How It Works ───────────────────────────────────────────────────────

with tabs[2]:
    st.subheader("How the Router Works")

    st.markdown("""
    Every query passes through a pipeline that decides the cheapest way to answer it correctly.
    """)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("### 🗄️ Cache")
        st.markdown("""
        **Tier 0**

        Before any model runs, the query is embedded and compared against past queries in Supabase.

        If a near-identical query exists (similarity > 0.92), the cached answer is returned instantly.

        **Cost:** Zero. **Latency:** ~50ms
        """)

    with c2:
        st.markdown("### ⚡ Tier 1")
        st.markdown("""
        **Classical NLP**

        Simple queries — greetings, sentiment, named entities — are handled by spaCy and NLTK rule-based models.

        No neural network, no API call.

        **Cost:** Zero. **Latency:** ~2ms
        """)

    with c3:
        st.markdown("### 🤖 Tier 2")
        st.markdown("""
        **Small Model**

        Medium-complexity queries use a sentence-transformer for zero-shot classification, or GPT4All for local generation.

        Runs entirely on your machine.

        **Cost:** Zero. **Latency:** ~1–2s
        """)

    with c4:
        st.markdown("### 🧠 Tier 3")
        st.markdown("""
        **LLM (Last Resort)**

        Only genuinely hard queries reach here. Uses GPT4All locally or Groq API in the cloud.

        The prompt is compressed before sending to reduce token cost.

        **Cost:** API tokens (Groq) or RAM (GPT4All). **Latency:** 3–8s
        """)

    st.divider()

    st.markdown("### Complexity Signals")
    st.markdown("""
    The router scores every query on 5 signals before deciding which tier to use:

    | Signal | What it measures |
    | --- | --- |
    | **Shannon Entropy** | How unpredictable the token distribution is |
    | **OOV Rate** | How many words are outside the common vocabulary |
    | **Perplexity** | How surprising the word sequence is |
    | **Entity Density** | How many named entities (people, places, orgs) appear |
    | **Tree Depth** | How syntactically complex the sentence structure is |

    A weighted combination of these five signals produces a score from 0 (simple) to 1 (complex).
    """)

    st.divider()

    st.markdown("### Expected LLM Call Reduction")
    st.markdown("""
    | Traffic type | Handled by |
    | --- | --- |
    | ~40% of queries | Tier 1 — free, instant |
    | ~25% of queries | Tier 2 — free, local |
    | ~10% of queries | Cache — free, ~50ms |
    | ~25% of queries | Tier 3 — LLM only when necessary |

    **Net result: ~75% fewer LLM calls** compared to sending everything to an LLM.
    """)
