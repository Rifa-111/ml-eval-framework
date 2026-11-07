"""
engine.py
=========
Orchestrates automated ablation studies.

An ablation study systematically removes or degrades one factor at a time
to measure its isolated contribution to model performance.

Supported ablation axes:
- Data quality   : label noise, feature corruption, missing values
- Feature sets   : removing individual or grouped features
- Sample size    : training set subsampling
- Class balance  : class oversampling / undersampling ratios
"""

from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.data_pipeline import DataSplit
from metrics.accuracy import AccuracyMetrics
from metrics.efficiency import EfficiencyMetrics
from metrics.fairness import FairnessMetrics
from metrics.robustness import RobustnessMetrics

logger = logging.getLogger(__name__)


@dataclass
class AblationVariant:
    """A single experimental condition within an ablation study."""
    name: str
    description: str
    axis: str                           # e.g. "data_quality", "feature_selection"
    perturbation: Callable              # f(DataSplit) -> DataSplit
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Results for one ablation variant × one model."""
    variant_name: str
    axis: str
    model_name: str
    run_id: str
    metrics: Dict[str, float]
    training_time_s: float
    params: Dict[str, Any]

    def to_row(self) -> Dict[str, Any]:
        return {
            "variant": self.variant_name,
            "axis": self.axis,
            "model": self.model_name,
            "run_id": self.run_id,
            "training_time_s": self.training_time_s,
            **self.metrics,
            **{f"param_{k}": v for k, v in self.params.items()},
        }


class AblationEngine:
    """
    Runs ablation studies over a grid of dataset variants and models.

    Usage
    -----
    >>> engine = AblationEngine(base_split=split, trainer_factory=make_trainer)
    >>> engine.add_axis("data_quality", [
    ...     AblationVariant("clean", "No noise", "data_quality", lambda s: s),
    ...     AblationVariant("5pct_noise", "5% label noise", "data_quality",
    ...                      label_noise_perturbation(0.05)),
    ... ])
    >>> results_df = engine.run(model_names=["random_forest", "mlp_pytorch"])
    """

    def __init__(
        self,
        base_split: DataSplit,
        trainer_factory: Callable,          # (model_name, n_features, n_classes) -> BaseTrainer
        metrics_config: Optional[Dict] = None,
    ):
        self.base_split = base_split
        self.trainer_factory = trainer_factory
        self.metrics_config = metrics_config or {}
        self._axes: Dict[str, List[AblationVariant]] = {}
        self._results: List[AblationResult] = []

    def add_axis(self, axis_name: str, variants: List[AblationVariant]):
        """Register a set of variants for one ablation axis."""
        self._axes[axis_name] = variants
        logger.info(f"Registered axis '{axis_name}' with {len(variants)} variants")

    def run(
        self,
        model_names: List[str],
        include_baseline: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """
        Execute the full ablation grid.

        Parameters
        ----------
        model_names : list of model names to evaluate on each variant
        include_baseline : if True, always include an unperturbed "baseline" run
        progress_callback : optional f(done, total, description) for progress reporting

        Returns
        -------
        pd.DataFrame with one row per (variant × model) combination
        """
        all_variants: List[AblationVariant] = []

        if include_baseline:
            all_variants.append(AblationVariant(
                name="baseline",
                description="Unperturbed data",
                axis="baseline",
                perturbation=lambda s: s,
                params={},
            ))

        for axis_name, variants in self._axes.items():
            all_variants.extend(variants)

        total = len(all_variants) * len(model_names)
        done = 0
        logger.info(f"Starting ablation: {len(all_variants)} variants × {len(model_names)} models = {total} runs")

        for variant in all_variants:
            perturbed_split = variant.perturbation(copy.deepcopy(self.base_split))

            for model_name in model_names:
                run_id = str(uuid.uuid4())[:8]
                logger.info(f"Running [{variant.name}] × [{model_name}] (run={run_id})")

                try:
                    result = self._single_run(perturbed_split, model_name, variant, run_id)
                    self._results.append(result)
                except Exception as e:
                    logger.error(f"Run failed [{variant.name}] × [{model_name}]: {e}", exc_info=True)

                done += 1
                if progress_callback:
                    progress_callback(done, total, f"{variant.name} × {model_name}")

        return self.results_dataframe()

    def _single_run(
        self,
        split: DataSplit,
        model_name: str,
        variant: AblationVariant,
        run_id: str,
    ) -> AblationResult:
        import time
        trainer = self.trainer_factory(
            model_name=model_name,
            n_features=split.n_features,
            n_classes=split.n_classes,
        )

        t0 = time.perf_counter()
        training_result = trainer.fit(
            split.X_train, split.y_train,
            split.X_val, split.y_val,
        )
        elapsed = time.perf_counter() - t0

        y_pred = trainer.predict(split.X_test)
        y_proba = None
        try:
            y_proba = trainer.predict_proba(split.X_test)
        except Exception:
            pass

        metrics = {}

        # Accuracy metrics
        acc = AccuracyMetrics()
        metrics.update(acc.compute(split.y_test, y_pred, y_proba))

        # Efficiency metrics
        eff = EfficiencyMetrics()
        metrics.update(eff.from_training_result(training_result))

        # Robustness metrics (optional)
        if self.metrics_config.get("robustness", True):
            rob = RobustnessMetrics()
            metrics.update(rob.compute(trainer, split.X_test, split.y_test))

        # Fairness metrics (if sensitive attributes available)
        if split.sensitive_test is not None and self.metrics_config.get("fairness", True):
            fair = FairnessMetrics()
            metrics.update(fair.compute(split.y_test, y_pred, split.sensitive_test))

        return AblationResult(
            variant_name=variant.name,
            axis=variant.axis,
            model_name=model_name,
            run_id=run_id,
            metrics=metrics,
            training_time_s=elapsed,
            params=variant.params,
        )

    def results_dataframe(self) -> pd.DataFrame:
        if not self._results:
            return pd.DataFrame()
        rows = [r.to_row() for r in self._results]
        return pd.DataFrame(rows)

    def summary_by_axis(self) -> pd.DataFrame:
        """Aggregate metrics by ablation axis for high-level analysis."""
        df = self.results_dataframe()
        if df.empty:
            return df
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        return (
            df.groupby(["axis", "variant", "model"])[numeric_cols]
            .mean()
            .reset_index()
        )
