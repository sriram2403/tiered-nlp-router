"""
Drift Detector
--------------
Monitors incoming query complexity distributions using:
- Kolmogorov-Smirnov test (feature distributions)
- Population Stability Index (PSI)

When drift is detected, raises the confidence_floor threshold so the
router becomes more conservative and escalates more queries to higher tiers.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from scipy import stats
from loguru import logger


def _psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    min_val = min(baseline.min(), current.min())
    max_val = max(baseline.max(), current.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    baseline_pct = np.histogram(baseline, bins=bin_edges)[0] / len(baseline)
    current_pct  = np.histogram(current,  bins=bin_edges)[0] / len(current)

    # Avoid division by zero
    baseline_pct = np.where(baseline_pct == 0, 1e-6, baseline_pct)
    current_pct  = np.where(current_pct  == 0, 1e-6, current_pct)

    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))


class DriftDetector:
    """
    Tracks complexity score distributions over a sliding window and
    alerts when the current distribution diverges from a baseline.

    Parameters
    ----------
    window_size : int
        Number of recent queries to compare against baseline.
    psi_threshold : float
        PSI above this value triggers a drift alert (0.2 = significant drift).
    baseline_size : int
        Number of initial queries used to establish the baseline.
    """

    def __init__(
        self,
        window_size: int = 200,
        psi_threshold: float = 0.2,
        baseline_size: int = 100,
    ):
        self.psi_threshold = psi_threshold
        self.baseline_size = baseline_size
        self._baseline: Optional[np.ndarray] = None
        self._window: deque = deque(maxlen=window_size)
        self._n_seen = 0
        self._drift_active = False

    def observe(self, complexity_score: float) -> dict:
        """
        Add a new observation. Returns drift status.

        Returns
        -------
        dict with keys: drift_detected, psi, ks_pvalue
        """
        self._window.append(complexity_score)
        self._n_seen += 1

        if self._n_seen == self.baseline_size:
            self._baseline = np.array(list(self._window))
            logger.info("Drift baseline established.")

        if self._baseline is None or len(self._window) < 50:
            return {"drift_detected": False, "psi": 0.0, "ks_pvalue": 1.0}

        current = np.array(list(self._window))
        psi = _psi(self._baseline, current)
        ks_stat, ks_pvalue = stats.ks_2samp(self._baseline, current)

        drift = psi > self.psi_threshold or ks_pvalue < 0.05
        if drift and not self._drift_active:
            logger.warning(f"Distribution drift detected! PSI={psi:.3f}, KS p={ks_pvalue:.3f}")
            self._drift_active = True
        elif not drift and self._drift_active:
            logger.info("Drift resolved.")
            self._drift_active = False

        return {"drift_detected": drift, "psi": round(psi, 4), "ks_pvalue": round(ks_pvalue, 4)}
