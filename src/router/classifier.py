"""
Adaptive Routing Classifier
---------------------------
Production-grade XGBoost classifier for intelligent tier routing.
Trains on Supabase routing logs and learns optimal tier-selection boundaries.

Key capabilities:
  - 10-feature engineering from query text + complexity signals
  - Optuna HPO (50 trials, 5-fold stratified CV)
  - SHAP explainability — mean |SHAP| per feature
  - predict_proba for confidence-weighted routing in the engine
  - Active learning: flags uncertain predictions near decision boundaries
  - Class-balanced training via sample weights
  - Bootstrap data generator for cold-start (no logs yet)
  - MLflow run tracking for every training job
  - needs_retraining() to auto-trigger retraining in production
"""

from __future__ import annotations

import os
import pickle
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from supabase import create_client

optuna.logging.set_verbosity(optuna.logging.WARNING)


class AdaptiveRouter:
    """
    Trains and serves an XGBoost classifier for tier prediction.

    Usage
    -----
    Cold-start (no logs yet):
        router = AdaptiveRouter()
        metrics = router.train()          # generates bootstrap data automatically

    After logs accumulate:
        router = AdaptiveRouter()
        if router.needs_retraining():
            metrics = router.train()

    Inference (called by RouterEngine):
        tier = router.predict(complexity_score=0.42, task="qa", query_length=80)
        tier, conf = router.predict_with_confidence(0.42, "qa", 80, query_text="...")
    """

    MODEL_VERSION = "2.0.0"
    MIN_TRAINING_SAMPLES = 100
    # Proba gap between top-2 classes below which a sample is flagged uncertain
    UNCERTAINTY_MARGIN = 0.15
    TASKS = ["classify", "generate", "qa", "summarize"]

    def __init__(
        self,
        model_path: str = "models/adaptive_router.pkl",
        mlflow_experiment: str = "tiered-nlp-router",
        n_optuna_trials: int = 50,
    ):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.mlflow_experiment = mlflow_experiment
        self.n_optuna_trials = n_optuna_trials

        self.model: Optional[xgb.XGBClassifier] = None
        self._supabase = None
        self._metadata: Dict[str, Any] = {}

        # Feature order must match at train and inference time
        self.feature_cols: List[str] = [
            "complexity_score",
            "query_length",
            "word_count",
            "avg_word_len",
            "digit_ratio",
            "question_mark",
            "task_classify",
            "task_generate",
            "task_qa",
            "task_summarize",
        ]

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def predict(
        self,
        complexity_score: float,
        task: str,
        query_length: int,
        query_text: str = "",
    ) -> int:
        """Return predicted tier (1, 2, or 3)."""
        tier, _ = self.predict_with_confidence(
            complexity_score, task, query_length, query_text
        )
        return tier

    def predict_with_confidence(
        self,
        complexity_score: float,
        task: str,
        query_length: int,
        query_text: str = "",
    ) -> Tuple[int, float]:
        """
        Return (tier, confidence) where confidence is the max predicted probability.
        Falls back to threshold routing if no model is loaded yet.
        """
        if not self.is_fitted:
            if self.model_path.exists():
                self._load_model()
            else:
                logger.warning(
                    "No trained model found — falling back to complexity thresholds"
                )
                return self._fallback_predict(complexity_score), 0.0

        features = self._build_inference_features(
            complexity_score, task, query_length, query_text
        )
        proba = self.model.predict_proba(features)[0]  # shape (3,)
        tier_idx = int(np.argmax(proba))
        confidence = float(proba[tier_idx])
        return tier_idx + 1, confidence

    def is_uncertain(
        self,
        complexity_score: float,
        task: str,
        query_length: int,
        query_text: str = "",
    ) -> bool:
        """
        True when the top-2 predicted probabilities are close.
        Use this to flag samples for active-learning review.
        """
        if not self.is_fitted:
            return False
        features = self._build_inference_features(
            complexity_score, task, query_length, query_text
        )
        proba = self.model.predict_proba(features)[0]
        sorted_proba = np.sort(proba)[::-1]
        return float(sorted_proba[0] - sorted_proba[1]) < self.UNCERTAINTY_MARGIN

    def train(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        tune_hyperparams: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training pipeline:
          1. Fetch routing logs from Supabase (bootstrap if < MIN_TRAINING_SAMPLES)
          2. Feature engineering
          3. Optuna HPO with 5-fold stratified CV (optional)
          4. Final XGBoost fit with early stopping + sample weighting
          5. SHAP explainability summary
          6. Persist model + log to MLflow
        """
        logger.info("Fetching routing logs from Supabase…")
        df = self._fetch_logs()

        if len(df) < self.MIN_TRAINING_SAMPLES:
            shortfall = self.MIN_TRAINING_SAMPLES - len(df)
            logger.warning(
                f"Only {len(df)} real samples — bootstrapping {shortfall} synthetic ones"
            )
            bootstrap = self._generate_bootstrap_data(shortfall)
            df = (
                pd.concat([df, bootstrap], ignore_index=True)
                if len(df) > 0
                else bootstrap
            )

        X, y = self._prepare_data(df)
        class_dist = np.bincount(y)
        logger.info(
            f"Dataset: {len(X)} samples | T1={class_dist[0]}  T2={class_dist[1]}  T3={class_dist[2]}"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # ── Hyperparameter optimisation ───────────────────────────────
        if tune_hyperparams:
            logger.info(f"Running Optuna HPO ({self.n_optuna_trials} trials)…")
            best_params = self._tune_hyperparams(X_train, y_train, random_state)
        else:
            best_params = self._default_params()

        # ── Final fit ────────────────────────────────────────────────
        sample_weights = compute_sample_weight("balanced", y_train)
        self.model = xgb.XGBClassifier(
            **best_params,
            objective="multi:softprob",
            num_class=3,
            random_state=random_state,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
            verbosity=0,
        )
        self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Evaluation ───────────────────────────────────────────────
        y_pred = self.model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        report = classification_report(
            y_test, y_pred, target_names=["Tier1", "Tier2", "Tier3"]
        )

        # ── Cross-validation ─────────────────────────────────────────
        cv_scores = self._cross_validate(X, y, best_params, random_state)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # ── SHAP feature importance ───────────────────────────────────
        shap_summary = self._compute_shap_summary(X_test[:min(200, len(X_test))])

        # ── Persist ──────────────────────────────────────────────────
        self._metadata = {
            "version": self.MODEL_VERSION,
            "trained_at": time.time(),
            "accuracy": accuracy,
            "f1_weighted": f1,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "n_samples": len(X),
            "params": best_params,
            "feature_cols": self.feature_cols,
        }
        self._save_model()

        # ── MLflow logging ────────────────────────────────────────────
        self._log_to_mlflow(best_params, accuracy, f1, cv_mean, cv_std, shap_summary)

        logger.info(
            f"Training complete | accuracy={accuracy:.3f} | cv={cv_mean:.3f}±{cv_std:.3f}"
        )
        logger.info(f"\n{report}")

        return {
            "accuracy": accuracy,
            "f1_weighted": f1,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "report": report,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "best_params": best_params,
            "shap_summary": shap_summary,
            "feature_importance": self.get_feature_importance(),
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Gain-based feature importances, normalised and sorted descending."""
        if not self.is_fitted:
            return {}
        raw = self.model.get_booster().get_score(importance_type="gain")
        total = sum(raw.values()) or 1.0
        return {
            k: round(v / total, 4)
            for k, v in sorted(raw.items(), key=lambda x: -x[1])
        }

    def needs_retraining(self, new_samples_threshold: int = 500) -> bool:
        """
        True when Supabase has accumulated enough new logs since last training.
        Lets the train_classifier script decide whether to re-run.
        """
        if not self._metadata:
            return True
        last_trained = self._metadata.get("trained_at", 0)
        try:
            client = self._get_supabase()
            resp = (
                client.table("routing_logs")
                .select("id", count="exact")
                .gt("created_at", int(last_trained))
                .execute()
            )
            new_count = resp.count or 0
            logger.info(f"{new_count} new samples since last training")
            return new_count >= new_samples_threshold
        except Exception as e:
            logger.warning(f"Could not check Supabase for new logs: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────
    # Feature engineering
    # ──────────────────────────────────────────────────────────────────

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = self._engineer_features(df)
        # tier_used is 1/2/3 in the DB; XGBoost needs 0-indexed classes
        y = df["tier_used"].values.astype(int) - 1
        return X, y

    def _engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract the 10-feature vector from a routing_logs dataframe."""
        text = df["query_preview"].fillna("").astype(str)
        task_col = df.get(
            "task_type",
            pd.Series(["classify"] * len(df), index=df.index),
        ).fillna("classify")

        feat = pd.DataFrame(index=df.index)
        feat["complexity_score"] = df["complexity_score"].fillna(0.5).clip(0, 1)
        feat["query_length"] = text.str.len()
        feat["word_count"] = text.str.split().str.len().fillna(1).clip(lower=1)
        feat["avg_word_len"] = text.apply(
            lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 4.5
        )
        feat["digit_ratio"] = text.apply(
            lambda t: sum(c.isdigit() for c in t) / max(len(t), 1)
        )
        feat["question_mark"] = text.str.contains(r"\?", regex=True).astype(int)

        for t in self.TASKS:
            feat[f"task_{t}"] = (task_col == t).astype(int)

        return feat[self.feature_cols].values.astype(np.float32)

    def _build_inference_features(
        self,
        complexity_score: float,
        task: str,
        query_length: int,
        query_text: str,
    ) -> np.ndarray:
        """Build the same 10-feature vector at inference time."""
        words = query_text.split() if query_text else []
        word_count = len(words) if words else max(query_length // 5, 1)
        avg_word_len = (
            float(np.mean([len(w) for w in words])) if words else 4.5
        )
        digit_ratio = (
            sum(c.isdigit() for c in query_text) / max(len(query_text), 1)
        )
        question_mark = int("?" in query_text)
        task_features = [1 if task == t else 0 for t in self.TASKS]

        vec = [
            complexity_score,
            query_length,
            word_count,
            avg_word_len,
            digit_ratio,
            question_mark,
            *task_features,
        ]
        return np.array([vec], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────
    # Hyperparameter tuning (Optuna)
    # ──────────────────────────────────────────────────────────────────

    def _tune_hyperparams(
        self, X: np.ndarray, y: np.ndarray, random_state: int
    ) -> Dict[str, Any]:
        """Optuna HPO: maximise 5-fold stratified CV accuracy."""

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            }
            scores = self._cross_validate(X, y, params, random_state)
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        study.optimize(
            objective,
            n_trials=self.n_optuna_trials,
            show_progress_bar=False,
            n_jobs=1,
        )

        logger.info(
            f"Best CV accuracy: {study.best_value:.3f} | params: {study.best_params}"
        )
        return study.best_params

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        random_state: int = 42,
    ) -> np.ndarray:
        """5-fold stratified CV — returns accuracy per fold."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            m = xgb.XGBClassifier(
                **params,
                objective="multi:softprob",
                num_class=3,
                verbosity=0,
            )
            w = compute_sample_weight("balanced", y[train_idx])
            m.fit(X[train_idx], y[train_idx], sample_weight=w)
            scores.append(accuracy_score(y[val_idx], m.predict(X[val_idx])))
        return np.array(scores)

    def _default_params(self) -> Dict[str, Any]:
        return {
            "max_depth": 4,
            "n_estimators": 150,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }

    # ──────────────────────────────────────────────────────────────────
    # SHAP explainability
    # ──────────────────────────────────────────────────────────────────

    def _compute_shap_summary(self, X_sample: np.ndarray) -> Dict[str, float]:
        """
        Compute mean absolute SHAP value per feature across all classes.
        Returns {} gracefully if shap is not installed.
        """
        try:
            import shap as shap_lib  # optional dependency

            explainer = shap_lib.TreeExplainer(self.model)
            raw = explainer.shap_values(X_sample)

            # raw is list[ndarray(n_samples, n_features)] for multiclass
            if isinstance(raw, list):
                stacked = np.stack(raw)          # (n_classes, n_samples, n_feats)
                mean_abs = np.abs(stacked).mean(axis=(0, 1))
            else:
                # newer shap may return (n_samples, n_feats, n_classes)
                mean_abs = np.abs(raw).mean(axis=(0, 2))

            summary = dict(zip(self.feature_cols, mean_abs.tolist()))
            top = sorted(summary.items(), key=lambda x: -x[1])[:5]
            logger.info("Top SHAP features: " + ", ".join(f"{k}={v:.4f}" for k, v in top))
            return {k: round(float(v), 5) for k, v in summary.items()}

        except ImportError:
            logger.warning("shap not installed — skipping SHAP analysis (pip install shap)")
            return {}
        except Exception as e:
            logger.warning(f"SHAP computation failed (non-fatal): {e}")
            return {}

    # ──────────────────────────────────────────────────────────────────
    # Bootstrap data for cold-start
    # ──────────────────────────────────────────────────────────────────

    def _generate_bootstrap_data(self, n: int) -> pd.DataFrame:
        """
        Generate synthetic training examples from routing heuristics.
        Seeded so the output is reproducible across runs.
        """
        rng = np.random.default_rng(42)
        records: List[Dict] = []

        tier_specs = [
            # (tier, complexity_range, confidence_range, tasks, sample_queries)
            (
                1,
                (0.05, 0.34),
                (0.80, 0.99),
                ["classify", "classify", "classify", "qa"],
                [
                    "hello how are you",
                    "what is the weather today",
                    "yes that is correct",
                    "good morning",
                    "what time is it",
                ],
            ),
            (
                2,
                (0.35, 0.64),
                (0.73, 0.92),
                ["classify", "summarize", "qa", "generate"],
                [
                    "explain deep learning versus machine learning",
                    "summarize this paragraph about climate change",
                    "what are the main benefits of renewable energy",
                    "describe the difference between supervised and unsupervised learning",
                    "classify this customer review as positive or negative",
                ],
            ),
            (
                3,
                (0.65, 0.99),
                (0.88, 0.99),
                ["generate", "qa", "summarize", "generate"],
                [
                    "analyze multi-dimensional trade-offs between transformer architectures for long-document understanding",
                    "write a comprehensive business plan for a sustainable tech startup with market analysis",
                    "compare and contrast the epistemological foundations of rationalism and empiricism",
                    "design an algorithm to solve the travelling salesman problem with time complexity analysis",
                    "explain quantum entanglement implications for distributed computing systems",
                ],
            ),
        ]

        n_per_tier = n // 3
        remainders = [n - 2 * n_per_tier, n_per_tier, n_per_tier]

        for (tier, c_range, conf_range, tasks, queries), count in zip(
            tier_specs, remainders
        ):
            for _ in range(count):
                c = float(rng.uniform(*c_range))
                task = str(rng.choice(tasks))
                q = str(rng.choice(queries))
                records.append(
                    {
                        "query_preview": q[:120],
                        "complexity_score": c,
                        "tier_used": tier,
                        "confidence": float(rng.uniform(*conf_range)),
                        "task_type": task,
                    }
                )

        df = pd.DataFrame(records)
        logger.info(
            f"Bootstrap: generated {len(df)} synthetic samples "
            f"(T1={remainders[0]}, T2={remainders[1]}, T3={remainders[2]})"
        )
        return df

    # ──────────────────────────────────────────────────────────────────
    # Data fetching
    # ──────────────────────────────────────────────────────────────────

    def _fetch_logs(self) -> pd.DataFrame:
        """Fetch and clean routing logs from Supabase."""
        try:
            client = self._get_supabase()
            resp = client.table("routing_logs").select("*").execute()
            df = pd.DataFrame(resp.data)
        except Exception as e:
            logger.error(f"Supabase fetch failed: {e}")
            return pd.DataFrame()

        if df.empty:
            return df

        # Only keep tier decisions (not cache hits) with sufficient confidence
        df = df[(df["tier_used"] > 0) & (df["confidence"] >= 0.70)].copy()

        if "task_type" not in df.columns:
            df["task_type"] = "classify"

        return df.reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────
    # Model persistence
    # ──────────────────────────────────────────────────────────────────

    def _save_model(self) -> None:
        with open(self.model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "metadata": self._metadata,
                    "feature_cols": self.feature_cols,
                    "tasks": self.TASKS,
                },
                f,
            )
        logger.info(f"Model v{self.MODEL_VERSION} saved → {self.model_path}")

    def _load_model(self) -> None:
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self._metadata = data.get("metadata", {})
        self.feature_cols = data.get("feature_cols", self.feature_cols)
        version = self._metadata.get("version", "?")
        accuracy = self._metadata.get("accuracy", 0)
        logger.info(
            f"Loaded model v{version} from {self.model_path} (accuracy={accuracy:.3f})"
        )

    # ──────────────────────────────────────────────────────────────────
    # MLflow logging
    # ──────────────────────────────────────────────────────────────────

    def _log_to_mlflow(
        self,
        params: Dict,
        accuracy: float,
        f1: float,
        cv_mean: float,
        cv_std: float,
        shap_summary: Dict,
    ) -> None:
        try:
            import mlflow  # optional at import time

            mlflow.set_experiment(self.mlflow_experiment)
            with mlflow.start_run(run_name="adaptive_router_training"):
                mlflow.log_params(params)
                mlflow.log_metrics(
                    {
                        "accuracy": accuracy,
                        "f1_weighted": f1,
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                    }
                )
                if shap_summary:
                    mlflow.log_metrics(
                        {f"shap_{k}": v for k, v in shap_summary.items()}
                    )
                mlflow.log_artifact(str(self.model_path))
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-fatal): {e}")

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _fallback_predict(self, complexity: float) -> int:
        """Complexity-threshold routing used before the model is trained."""
        if complexity <= 0.35:
            return 1
        elif complexity <= 0.65:
            return 2
        return 3

    def _get_supabase(self):
        if self._supabase is None:
            self._supabase = create_client(
                os.environ["SUPABASE_URL"],
                os.environ["SUPABASE_SERVICE_KEY"],
            )
        return self._supabase
