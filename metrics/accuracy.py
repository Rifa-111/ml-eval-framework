"""
accuracy.py
===========
Standard classification and regression accuracy metrics.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class AccuracyMetrics:
    """
    Computes a comprehensive suite of accuracy metrics.

    For classification: accuracy, balanced accuracy, F1, precision, recall,
                        ROC-AUC, log loss, MCC
    For regression:     MAE, MSE, RMSE, R², MAPE
    """

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        task: str = "classification",
        average: str = "weighted",
    ) -> Dict[str, float]:
        if task == "classification":
            return self._classification(y_true, y_pred, y_proba, average)
        elif task == "regression":
            return self._regression(y_true, y_pred)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _classification(self, y_true, y_pred, y_proba, average) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            log_loss,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        metrics: Dict[str, float] = {}

        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

        avg = average if len(np.unique(y_true)) > 2 else "binary"
        metrics["f1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        metrics["precision"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

        if y_proba is not None:
            try:
                n_classes = y_proba.shape[1]
                if n_classes > 2:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                    )
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            except Exception as e:
                pass

            try:
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
            except Exception:
                pass

        return metrics

    def _regression(self, y_true, y_pred) -> Dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        # MAPE — guard against zero denominators
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) if mask.any() else float("nan")

        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}
