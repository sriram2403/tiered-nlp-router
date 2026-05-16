#!/usr/bin/env python3
"""
Train the Adaptive Routing Classifier
--------------------------------------
Run this script after accumulating routing logs to train (or retrain)
the XGBoost model that drives intelligent tier selection.

Usage
-----
Basic (with Optuna HPO):
    python scripts/train_classifier.py

Skip hyperparameter tuning (faster, uses sensible defaults):
    python scripts/train_classifier.py --no-tune

Only check whether retraining is needed without training:
    python scripts/train_classifier.py --check-only

Force retrain even if threshold not reached:
    python scripts/train_classifier.py --force

This will:
  1. Fetch routing logs from Supabase (bootstrap if not enough data yet)
  2. Run Optuna HPO with 5-fold cross-validation (unless --no-tune)
  3. Train final XGBoost model with early stopping + class balancing
  4. Compute SHAP feature importance summary
  5. Save model to models/adaptive_router.pkl
  6. Log metrics + artifact to MLflow
  7. Enable adaptive routing in configs/router_config.yaml
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from src.router.classifier import AdaptiveRouter  # noqa: E402


def _print_table(title: str, data: dict, top_n: int = 10) -> None:
    """Pretty-print a dict as a ranked table."""
    items = list(data.items())[:top_n]
    if not items:
        return
    col_w = max(len(k) for k, _ in items) + 2
    print(f"\n  {title}")
    print("  " + "─" * (col_w + 12))
    for rank, (k, v) in enumerate(items, 1):
        bar = "█" * int(v * 40)
        print(f"  {rank:>2}. {k:<{col_w}} {v:.4f}  {bar}")
    print()


def _enable_adaptive_routing(config_path: Path) -> None:
    content = config_path.read_text()
    if "use_adaptive_router: false" in content:
        config_path.write_text(
            content.replace("use_adaptive_router: false", "use_adaptive_router: true")
        )
        logger.info("Config updated: use_adaptive_router → true  (restart API to apply)")
    else:
        logger.info("Config already set to use_adaptive_router: true")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Hybrid NLP Router's adaptive XGBoost classifier"
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip Optuna HPO and use default hyperparameters (faster)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print whether retraining is needed, then exit without training",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Train even if the new-sample threshold hasn't been reached",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=500,
        help="Minimum new samples required to trigger retraining (default: 500)",
    )
    args = parser.parse_args()

    router = AdaptiveRouter(n_optuna_trials=args.trials)

    # ── Check-only mode ───────────────────────────────────────────────
    if args.check_only:
        needed = router.needs_retraining(new_samples_threshold=args.threshold)
        status = "YES — retraining recommended" if needed else "NO — model is up to date"
        print(f"\n  Retraining needed: {status}\n")
        sys.exit(0)

    # ── Retrain guard ─────────────────────────────────────────────────
    if not args.force and not router.needs_retraining(new_samples_threshold=args.threshold):
        logger.info(
            f"Fewer than {args.threshold} new samples since last training. "
            "Use --force to train anyway."
        )
        sys.exit(0)

    # ── Training ──────────────────────────────────────────────────────
    logger.info("Starting adaptive router training…")
    metrics = router.train(tune_hyperparams=not args.no_tune)

    if "error" in metrics:
        logger.error(f"Training failed: {metrics['error']}")
        sys.exit(1)

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  ADAPTIVE ROUTER — TRAINING RESULTS")
    print("═" * 55)
    print(f"  Accuracy (hold-out):   {metrics['accuracy']:.3f}")
    print(f"  F1 Weighted:           {metrics['f1_weighted']:.3f}")
    print(f"  CV Mean ± Std:         {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    print(f"  Train samples:         {metrics['train_samples']}")
    print(f"  Test samples:          {metrics['test_samples']}")
    print()
    print(metrics["report"])

    if metrics.get("best_params"):
        print("  Best hyperparameters:")
        for k, v in metrics["best_params"].items():
            print(f"    {k}: {v}")

    if metrics.get("feature_importance"):
        _print_table("Feature Importance (gain)", metrics["feature_importance"])

    if metrics.get("shap_summary"):
        _print_table("SHAP Feature Importance (mean |SHAP|)", metrics["shap_summary"])

    print("═" * 55 + "\n")

    # ── Enable adaptive routing in config ─────────────────────────────
    config_path = Path(__file__).parent.parent / "configs" / "router_config.yaml"
    if config_path.exists():
        _enable_adaptive_routing(config_path)
    else:
        logger.warning(f"Config not found at {config_path} — skipping auto-enable")


if __name__ == "__main__":
    main()
