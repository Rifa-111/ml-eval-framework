"""
wandb_tracker.py
================
Weights & Biases (W&B) integration for experiment tracking and visualization.

Provides a unified API consistent with the MLflow tracker, enabling
dual-tracking with both tools simultaneously.

Example
-------
>>> tracker = WandbTracker(project="ml-eval-ablation", entity="my-team")
>>> with tracker.start_run(name="rf_noise_study", config={"model": "rf"}):
...     tracker.log({"accuracy": 0.94, "f1": 0.93})
...     tracker.log_training_result(result)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


class WandbTracker:
    """
    Thin wrapper around wandb for the eval framework.

    Features:
    - Run grouping by experiment / ablation axis
    - Automatic config serialization
    - Table logging for ablation result grids
    - Summary metrics for easy comparison
    """

    def __init__(
        self,
        project: str = "ml-eval",
        entity: Optional[str] = None,
        group: Optional[str] = None,
        mode: str = "online",           # "online" | "offline" | "disabled"
    ):
        self.project = project
        self.entity = entity
        self.group = group
        self.mode = mode
        self._wandb = None
        self._run = None
        self._setup()

    def _setup(self):
        try:
            import wandb
            self._wandb = wandb
            logger.info(f"W&B tracking enabled: project={self.project}")
        except ImportError:
            logger.warning("wandb not installed. W&B tracking disabled. Run: pip install wandb")

    @contextmanager
    def start_run(
        self,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        job_type: Optional[str] = None,
    ) -> Generator:
        if self._wandb is None:
            yield None
            return

        self._run = self._wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config or {},
            group=self.group,
            tags=tags,
            job_type=job_type,
            mode=self.mode,
            reinit=True,
        )
        logger.info(f"W&B run started: {name} | url: {self._run.url}")
        try:
            yield self._run
        finally:
            self._wandb.finish()
            self._run = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        if self._wandb is None or self._run is None:
            return
        clean = {k: v for k, v in metrics.items() if _is_loggable(v)}
        self._wandb.log(clean, step=step, commit=commit)

    def log_training_result(self, result):
        """Log full TrainingResult including loss curves as W&B charts."""
        if self._wandb is None or self._run is None:
            return

        # Summary metrics
        self._wandb.summary.update(result.summary())

        # Training curves
        for step, (tl, vl) in enumerate(zip(result.train_losses, result.val_losses)):
            self._wandb.log({"train/loss": tl, "val/loss": vl}, step=step, commit=True)

    def log_table(self, df, table_name: str = "ablation_results"):
        """Log a pandas DataFrame as a W&B Table for interactive comparison."""
        if self._wandb is None or self._run is None:
            return
        table = self._wandb.Table(dataframe=df)
        self._wandb.log({table_name: table})

    def log_ablation_summary(self, df):
        """
        Log an ablation results DataFrame with automatic bar charts per metric.

        Creates W&B custom charts comparing models across ablation variants.
        """
        if self._wandb is None or self._run is None:
            return

        self.log_table(df, "ablation_results")

        # Log per-metric summaries
        metric_cols = [c for c in df.columns if c not in ("variant", "axis", "model", "run_id")]
        for col in metric_cols:
            try:
                data = [[row["variant"], row["model"], row[col]]
                        for _, row in df.iterrows()]
                table = self._wandb.Table(
                    data=data, columns=["variant", "model", col]
                )
                self._wandb.log({
                    f"chart/{col}": self._wandb.plot.bar(
                        table, "variant", col, title=f"{col} by variant"
                    )
                })
            except Exception:
                pass

    def update_config(self, config: Dict[str, Any]):
        if self._wandb and self._run:
            self._wandb.config.update(config, allow_val_change=True)

    def add_tags(self, tags: List[str]):
        if self._run:
            self._run.tags = self._run.tags + tuple(tags)

    @property
    def run_url(self) -> Optional[str]:
        return self._run.url if self._run else None


def _is_loggable(v) -> bool:
    """Check if a value can be logged to W&B."""
    if isinstance(v, (int, float, str, bool)):
        try:
            import math
            if isinstance(v, float) and not math.isfinite(v):
                return False
        except Exception:
            pass
        return True
    return False
