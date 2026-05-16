"""Tests for the ComplexityScorer."""
import pytest
from unittest.mock import patch, MagicMock


def make_mock_doc(tokens, ents=None, sents=None):
    """Helper to mock a spaCy doc."""
    ents = ents or []
    doc = MagicMock()
    token_mocks = []
    for t in tokens:
        m = MagicMock()
        m.text = t
        m.text.lower.return_value = t.lower()
        m.is_space = False
        m.is_punct = False
        token_mocks.append(m)
    doc.__iter__ = lambda self: iter(token_mocks)
    doc.ents = ents
    doc.sents = sents or []
    return doc


class TestEntropySignal:
    def test_uniform_distribution_high_entropy(self):
        from src.scoring.complexity import ComplexityScorer
        scorer = ComplexityScorer()
        # All unique tokens → high entropy
        tokens = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
        entropy = scorer._entropy(tokens)
        assert entropy > 2.0

    def test_repeated_tokens_low_entropy(self):
        from src.scoring.complexity import ComplexityScorer
        scorer = ComplexityScorer()
        tokens = ["the", "the", "the", "the", "the"]
        entropy = scorer._entropy(tokens)
        assert entropy == 0.0

    def test_empty_tokens(self):
        from src.scoring.complexity import ComplexityScorer
        scorer = ComplexityScorer()
        assert scorer._entropy([]) == 0.0


class TestOOVRate:
    def test_all_known_words_zero_oov(self):
        from src.scoring.complexity import ComplexityScorer
        scorer = ComplexityScorer(vocab={"hello", "world", "test"})
        doc = make_mock_doc(["hello", "world", "test"])
        assert scorer._oov_rate(doc) == 0.0

    def test_all_unknown_words_full_oov(self):
        from src.scoring.complexity import ComplexityScorer
        scorer = ComplexityScorer(vocab={"the", "a", "is"})
        doc = make_mock_doc(["xyzabc123", "qwerty999"])
        assert scorer._oov_rate(doc) == 1.0


class TestComplexityBreakdown:
    def test_breakdown_fields_in_range(self):
        from src.scoring.complexity import ComplexityScorer
        scorer = ComplexityScorer(vocab={"hello", "world"})
        
        with patch("src.scoring.complexity._get_nlp") as mock_nlp_fn:
            mock_nlp = MagicMock()
            mock_nlp_fn.return_value = mock_nlp
            doc = make_mock_doc(["hello", "world"])
            mock_nlp.return_value = doc
            mock_nlp.vocab = []
            
            breakdown = scorer.score("hello world")
            assert 0.0 <= breakdown.overall <= 1.0
            assert 0.0 <= breakdown.entropy <= 1.0

    def test_weights_must_sum_to_one(self):
        from src.scoring.complexity import ComplexityScorer
        with pytest.raises(AssertionError):
            ComplexityScorer(weights={"entropy": 0.5, "oov_rate": 0.5,
                                      "perplexity": 0.5, "entity_density": 0.1,
                                      "dep_tree_depth": 0.1})
