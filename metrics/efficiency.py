"""
efficiency.py
=============
Runtime and memory efficiency metrics.

Tracks:
- Training time
- Inference latency (mean, p50, p95, p99)
- Peak memory usage during training
- Throughput (samples/second)
- Model parameter count (for PyTorch/JAX)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EfficiencyMetrics:
    """
    Compute runtime efficiency metrics from training results and live inference profiling.

    Example
    -------
    >>> eff = EfficiencyMetrics()
    >>> metrics = eff.from_training_result(training_result)
    >>> metrics.update(eff.profile_inference(trainer, X_test, n_repeats=100))
    """

    def from_training_result(self, result) -> Dict[str, float]:
        """Extract efficiency metrics from a TrainingResult."""
        return {
            "efficiency_training_time_s": round(result.training_time_s, 4),
            "efficiency_peak_memory_mb": round(result.peak_memory_mb, 2),
            "efficiency_epochs_trained": float(result.epochs_trained),
            "efficiency_best_epoch": float(result.best_epoch),
        }

    def profile_inference(
        self,
        trainer,
        X: np.ndarray,
        n_repeats: int = 50,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Profile inference latency over n_repeats runs.

        Returns mean, p50, p95, p99 latencies and throughput.
        """
        latencies = []
        n = len(X)

        # Warm-up
        for _ in range(3):
            try:
                trainer.predict(X[:min(batch_size, n)])
            except Exception:
                return {}

        for _ in range(n_repeats):
            t0 = time.perf_counter()
            try:
                trainer.predict(X[:min(batch_size, n)])
            except Exception:
                break
            latencies.append(time.perf_counter() - t0)

        if not latencies:
            return {}

        lat_arr = np.array(latencies) * 1000  # ms
        batch_n = min(batch_size, n)

        return {
            "efficiency_inference_mean_ms": round(float(lat_arr.mean()), 4),
            "efficiency_inference_p50_ms": round(float(np.percentile(lat_arr, 50)), 4),
            "efficiency_inference_p95_ms": round(float(np.percentile(lat_arr, 95)), 4),
            "efficiency_inference_p99_ms": round(float(np.percentile(lat_arr, 99)), 4),
            "efficiency_throughput_sps": round(float(batch_n / (lat_arr.mean() / 1000)), 2),
        }

    def count_parameters(self, model) -> Dict[str, float]:
        """Count trainable parameters for PyTorch or sklearn models."""
        try:
            import torch.nn as nn
            if isinstance(model, nn.Module):
                total = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                return {
                    "efficiency_total_params": float(total),
                    "efficiency_trainable_params": float(trainable),
                }
        except ImportError:
            pass

        # sklearn: try to extract tree counts, etc.
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier
            if hasattr(model, "n_estimators"):
                return {"efficiency_n_estimators": float(model.n_estimators)}
        except Exception:
            pass

        return {}

    def profile_memory_during_inference(self, trainer, X: np.ndarray) -> Dict[str, float]:
        """Measure memory delta during a forward pass."""
        try:
            import psutil
            proc = psutil.Process()
            mem_before = proc.memory_info().rss / 1e6
            trainer.predict(X)
            mem_after = proc.memory_info().rss / 1e6
            return {"efficiency_inference_memory_mb": round(max(mem_after - mem_before, 0), 2)}
        except ImportError:
            return {}
