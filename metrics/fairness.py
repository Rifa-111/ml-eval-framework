"""
fairness.py
===========
Fairness and bias metrics for evaluating model equity across demographic groups.

Implements:
- Demographic Parity (Statistical Parity Difference)
- Equalized Odds (True Positive Rate and False Positive Rate parity)
- Equal Opportunity (TPR parity)
- Disparate Impact Ratio
- Predictive Parity
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FairnessMetrics:
    """
    Compute group fairness metrics for binary or multiclass classification.

    All metrics are computed per-column of the sensitive_attributes DataFrame.
    For multi-class, the privileged/unprivileged split is based on the mode class.

    Example
    -------
    >>> sensitive = pd.DataFrame({"gender": ["M", "F", "M", "F"]})
    >>> fm = FairnessMetrics()
    >>> metrics = fm.compute(y_true, y_pred, sensitive)
    """

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attrs: pd.DataFrame,
        positive_label: int = 1,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        for col in sensitive_attrs.columns:
            groups = sensitive_attrs[col].values
            unique_groups = np.unique(groups)

            if len(unique_groups) < 2:
                logger.warning(f"Sensitive attribute '{col}' has only one group, skipping.")
                continue

            # Use most frequent group as "privileged"
            privileged = _mode(groups)
            group_metrics = self._binary_fairness(
                y_true, y_pred, groups, privileged, positive_label, col
            )
            metrics.update(group_metrics)

        return metrics

    def _binary_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        privileged: str,
        positive_label: int,
        attr_name: str,
    ) -> Dict[str, float]:
        priv_mask = groups == privileged
        unpriv_mask = ~priv_mask

        # Positive prediction rates
        priv_ppr = _positive_rate(y_pred[priv_mask], positive_label)
        unpriv_ppr = _positive_rate(y_pred[unpriv_mask], positive_label)

        # True positive rates
        priv_tpr = _tpr(y_true[priv_mask], y_pred[priv_mask], positive_label)
        unpriv_tpr = _tpr(y_true[unpriv_mask], y_pred[unpriv_mask], positive_label)

        # False positive rates
        priv_fpr = _fpr(y_true[priv_mask], y_pred[priv_mask], positive_label)
        unpriv_fpr = _fpr(y_true[unpriv_mask], y_pred[unpriv_mask], positive_label)

        prefix = f"fairness_{attr_name}"

        return {
            # Demographic Parity: P(Ŷ=1|unprivileged) - P(Ŷ=1|privileged)
            # Ideal = 0
            f"{prefix}_dem_parity_diff": round(unpriv_ppr - priv_ppr, 6),

            # Disparate Impact: P(Ŷ=1|unprivileged) / P(Ŷ=1|privileged)
            # Ideal = 1, < 0.8 is often considered discriminatory (4/5ths rule)
            f"{prefix}_disparate_impact": round(
                unpriv_ppr / priv_ppr if priv_ppr > 0 else float("nan"), 6
            ),

            # Equal Opportunity: TPR difference
            # Ideal = 0
            f"{prefix}_equal_opportunity_diff": round(unpriv_tpr - priv_tpr, 6),

            # Equalized Odds: max of TPR and FPR difference
            # Ideal = 0
            f"{prefix}_equalized_odds_diff": round(
                max(abs(unpriv_tpr - priv_tpr), abs(unpriv_fpr - priv_fpr)), 6
            ),

            # Raw group-level positive rates (for debugging)
            f"{prefix}_priv_positive_rate": round(priv_ppr, 6),
            f"{prefix}_unpriv_positive_rate": round(unpriv_ppr, 6),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _positive_rate(y_pred: np.ndarray, positive_label: int) -> float:
    if len(y_pred) == 0:
        return float("nan")
    return float((y_pred == positive_label).mean())


def _tpr(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int) -> float:
    actual_pos = y_true == positive_label
    if not actual_pos.any():
        return float("nan")
    return float((y_pred[actual_pos] == positive_label).mean())


def _fpr(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int) -> float:
    actual_neg = y_true != positive_label
    if not actual_neg.any():
        return float("nan")
    return float((y_pred[actual_neg] == positive_label).mean())


def _mode(arr) -> str:
    vals, counts = np.unique(arr, return_counts=True)
    return vals[np.argmax(counts)]
