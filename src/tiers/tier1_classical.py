"""
Tier 1 — Classical NLP
-----------------------
Handles simple, deterministic NLP tasks using:
- Rule-based intent detection (regex)
- VADER sentiment analysis
- TF-IDF + cosine similarity classification
- spaCy NER for basic entity extraction
"""

from __future__ import annotations
import re
from typing import Optional

import spacy
from loguru import logger

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    _vader = None

_nlp: Optional[spacy.Language] = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# Simple intent rules — extend freely
_INTENT_RULES = [
    (re.compile(r"\b(hello|hi|hey|greet)\b", re.I),       "greeting",    0.98),
    (re.compile(r"\b(bye|goodbye|see you)\b", re.I),       "farewell",    0.98),
    (re.compile(r"\bwhat time\b", re.I),                   "time_query",  0.95),
    (re.compile(r"\bweather\b", re.I),                     "weather",     0.90),
    (re.compile(r"\b(yes|no|maybe|sure|ok)\b", re.I),      "affirmation", 0.97),
    (re.compile(r"\bsentiment\b.*\bof\b", re.I),           "sentiment",   0.88),
    # "what is X?" / "who is X?" are knowledge questions — not handled here.
    # Removing this rule lets them fall through to Tier 2/3 for a real answer.
]


class Tier1Classical:
    """
    Handles simple queries using rule-based and classical statistical NLP.

    Returns
    -------
    (answer: str, confidence: float)
    """

    def handle(self, query: str, task: str = "classify") -> tuple[str, float]:
        # 1. Rule-based intent matching
        for pattern, intent, conf in _INTENT_RULES:
            if pattern.search(query):
                answer = self._intent_response(intent)
                logger.debug(f"Tier1 rule match: {intent} conf={conf}")
                return answer, conf

        # 2. Sentiment analysis
        if task == "sentiment" or "sentiment" in query.lower():
            return self._sentiment(query)

        # 3. NER extraction
        if task == "ner":
            return self._ner(query)

        # Default: no confident match — return low confidence to trigger escalation
        return "", 0.40

    def _sentiment(self, text: str) -> tuple[str, float]:
        if _vader is None:
            return "Sentiment analysis unavailable.", 0.5
        scores = _vader.polarity_scores(text)
        compound = scores["compound"]
        label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
        conf = min(0.5 + abs(compound) * 0.5, 0.99)
        return f"Sentiment: {label} (score={compound:.2f})", conf

    def _ner(self, text: str) -> tuple[str, float]:
        doc = _get_nlp()(text)
        if not doc.ents:
            return "No named entities found.", 0.85
        entities = [f"{e.text} ({e.label_})" for e in doc.ents]
        return "Entities: " + ", ".join(entities), 0.88

    @staticmethod
    def _intent_response(intent: str) -> str:
        responses = {
            "greeting": "Hello! How can I help you?",
            "farewell": "Goodbye! Have a great day.",
            "time_query": "I don't have access to real-time data.",
            "weather": "I don't have live weather data.",
            "affirmation": "Got it!",
            "sentiment": "I can analyse sentiment.",
        }
        return responses.get(intent, "Understood.")
