"""Tests for semantic cache logic."""
import pytest
from unittest.mock import MagicMock, patch


def test_cache_miss_returns_none():
    with patch("src.cache.semantic_cache.SemanticCache._get_supabase") as mock_sb, \
         patch("src.cache.semantic_cache.SemanticCache._embed", return_value=[0.1]*384):
        from src.cache.semantic_cache import SemanticCache
        cache = SemanticCache()
        mock_sb.return_value.rpc.return_value.execute.return_value = MagicMock(data=[])
        result = cache.lookup("some query")
        assert result is None


def test_cache_hit_returns_data():
    with patch("src.cache.semantic_cache.SemanticCache._get_supabase") as mock_sb, \
         patch("src.cache.semantic_cache.SemanticCache._embed", return_value=[0.1]*384):
        from src.cache.semantic_cache import SemanticCache
        cache = SemanticCache()
        mock_sb.return_value.rpc.return_value.execute.return_value = MagicMock(
            data=[{"answer": "cached answer", "complexity_score": 0.2}]
        )
        result = cache.lookup("some query")
        assert result is not None
        assert result["answer"] == "cached answer"
