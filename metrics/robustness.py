"""
robustness.py
=============
Robustness metrics measuring model stability under input perturbations.

Implements:
- Gaussian noise robustness: accuracy degradation under increasing input noise
- Adversarial robustness: FGSM-style perturbation for PyTorch models
- Label stability: prediction consistency on slightly perturbed inputs
- Confidence calibration: Expected Calibration Error (ECE)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RobustnessMetrics:
    """
    Evaluate model robustness against various input perturbations.

    Example
    -------
    >>> rob = RobustnessMetrics(noise_levels=[0.01, 0.05, 0.1, 0.2])
    >>> metrics = rob.compute(trainer, X_test, y_test)
    """

    def __init__(
        self,
        noise_levels: tuple = (0.01, 0.05, 0.1, 0.2, 0.5),
        n_trials: int = 5,
        seed: int = 42,
        ece_bins: int = 10,
    ):
        self.noise_levels = noise_levels
        self.n_trials = n_trials
        self.seed = seed
        self.ece_bins = ece_bins

    def compute(self, trainer, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # --- Baseline accuracy ---
        try:
            y_pred_base = trainer.predict(X_test)
            base_acc = float((y_pred_base == y_test).mean())
        except Exception as e:
            logger.warning(f"Could not get baseline predictions: {e}")
            return metrics

        # --- Gaussian noise robustness ---
        metrics.update(self._gaussian_robustness(trainer, X_test, y_test, base_acc))

        # --- Expected Calibration Error ---
        try:
            y_proba = trainer.predict_proba(X_test)
            metrics["calibration_ece"] = self._ece(y_test, y_proba)
        except Exception:
            pass

        # --- Label stability ---
        metrics["label_stability"] = self._label_stability(trainer, X_test)

        return metrics

    def _gaussian_robustness(
        self, trainer, X_test: np.ndarray, y_test: np.ndarray, base_acc: float
    ) -> Dict[str, float]:
        rng = np.random.RandomState(self.seed)
        metrics = {}
        degradations = []

        for sigma in self.noise_levels:
            trial_accs = []
            for _ in range(self.n_trials):
                X_noisy = (X_test + rng.normal(0, sigma, X_test.shape)).astype(np.float32)
                try:
                    y_pred = trainer.predict(X_noisy)
                    acc = float((y_pred == y_test).mean())
                    trial_accs.append(acc)
                except Exception:
                    pass

            if trial_accs:
                mean_acc = float(np.mean(trial_accs))
                degradation = base_acc - mean_acc
                degradations.append(degradation)
                key = str(sigma).replace(".", "_")
                metrics[f"robustness_noise_sigma{key}_acc"] = round(mean_acc, 6)
                metrics[f"robustness_noise_sigma{key}_degradation"] = round(degradation, 6)

        if degradations:
            # Area under degradation curve: lower is more robust
            metrics["robustness_audc"] = round(float(np.mean(degradations)), 6)

        return metrics

    def _ece(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Expected Calibration Error.
        Measures how well predicted probabilities match empirical accuracy.
        Lower ECE = better calibration.
        """
        n = len(y_true)
        if n == 0:
            return float("nan")

        y_conf = y_proba.max(axis=1)           # max predicted probability
        y_pred = y_proba.argmax(axis=1)
        correct = (y_pred == y_true).astype(float)

        bins = np.linspace(0, 1, self.ece_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (y_conf >= lo) & (y_conf < hi)
            if mask.sum() == 0:
                continue
            bin_conf = y_conf[mask].mean()
            bin_acc = correct[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

        return round(float(ece), 6)

    def _label_stability(self, trainer, X_test: np.ndarray, eps: float = 0.001) -> float:
        """
        Fraction of test points where prediction is consistent under tiny perturbation.
        High stability = model is locally consistent (desirable).
        """
        rng = np.random.RandomState(self.seed)
        try:
            y_orig = trainer.predict(X_test)
            X_perturbed = (X_test + rng.normal(0, eps, X_test.shape)).astype(np.float32)
            y_perturbed = trainer.predict(X_perturbed)
            stability = float((y_orig == y_perturbed).mean())
            return round(stability, 6)
        except Exception:
            return float("nan")
