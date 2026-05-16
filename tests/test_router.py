"""Integration-level tests for the router engine (mocked dependencies)."""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_engine(tmp_path):
    """Create a RouterEngine with all external deps mocked."""
    config = tmp_path / "router_config.yaml"
    config.write_text("""
routing:
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
  max_entries: 50000
tier2:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  onnx_quantized: true
  gpt4all_model: "test.gguf"
tier3:
  provider: "groq"
  model: "llama3-8b-8192"
  max_tokens: 512
  temperature: 0.1
monitoring:
  drift_check_interval: 100
  psi_threshold: 0.2
  mlflow_experiment: "test"
supabase:
  cache_table: "semantic_cache"
  logs_table: "routing_logs"
  metrics_table: "tier_metrics"
""")

    with patch("src.router.engine.ComplexityScorer"), \
         patch("src.router.engine.ConfidenceEvaluator"), \
         patch("src.router.engine.SemanticCache") as MockCache, \
         patch("src.router.engine.Tier1Classical") as MockT1, \
         patch("src.router.engine.Tier2SmallModel") as MockT2, \
         patch("src.router.engine.Tier3LLM") as MockT3, \
         patch("src.router.engine.RoutingTracker"):

        from src.router.engine import RouterEngine
        engine = RouterEngine(str(config))

        MockCache.return_value.lookup.return_value = None
        MockT1.return_value.handle.return_value = ("Simple answer", 0.95)

        yield engine, MockT1, MockT2, MockT3, MockCache


def test_tier1_handles_simple_query(mock_engine):
    engine, MockT1, MockT2, MockT3, MockCache = mock_engine
    engine.scorer.score.return_value = MagicMock(overall=0.20)
    result = engine.route("hello", "classify")
    assert result.tier_used == 1
    MockT2.return_value.handle.assert_not_called()
    MockT3.return_value.handle.assert_not_called()


def test_cache_hit_skips_all_tiers(mock_engine):
    engine, MockT1, MockT2, MockT3, MockCache = mock_engine
    MockCache.return_value.lookup.return_value = {
        "answer": "cached!", "complexity_score": 0.1
    }
    result = engine.route("hello", "classify")
    assert result.cache_hit is True
    assert result.tier_used == 0
    MockT1.return_value.handle.assert_not_called()


def test_low_confidence_escalates_to_tier2(mock_engine):
    engine, MockT1, MockT2, MockT3, MockCache = mock_engine
    engine.scorer.score.return_value = MagicMock(overall=0.30)
    MockT1.return_value.handle.return_value = ("", 0.50)  # below floor
    MockT2.return_value.handle.return_value = ("Medium answer", 0.85)
    result = engine.route("complex query here", "classify")
    assert result.tier_used == 2
