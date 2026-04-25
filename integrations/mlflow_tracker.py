"""
mlflow_tracker.py
=================
MLflow integration for experiment tracking, artifact logging, and model registry.

Provides a context-manager-based API for seamless run management.

Example
-------
>>> tracker = MLflowTracker(experiment_name="ablation_study_v1")
>>> with tracker.start_run(run_name="rf_baseline", tags={"model": "rf"}) as run:
...     tracker.log_params({"n_estimators": 100})
...     tracker.log_metrics({"accuracy": 0.94, "f1": 0.93})
...     tracker.log_training_result(training_result)
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Thin wrapper around MLflow for the eval framework.

    Features:
    - Auto-creates experiment if it doesn't exist
    - Logs training curves as metrics with step indices
    - Saves model artifacts and config files
    - Supports nested runs for ablation grids
    """

    def __init__(
        self,
        experiment_name: str = "ml_eval",
        tracking_uri: str = "mlruns",
        artifact_location: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._active_run = None
        self._setup(artifact_location)

    def _setup(self, artifact_location: Optional[str]):
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            try:
                mlflow.create_experiment(self.experiment_name, artifact_location=artifact_location)
            except Exception:
                pass  # already exists
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            logger.info(f"MLflow tracking at: {self.tracking_uri} | experiment: {self.experiment_name}")
        except ImportError:
            logger.warning("MLflow not installed. Tracking disabled. Run: pip install mlflow")
            self._mlflow = None

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> Generator:
        if self._mlflow is None:
            yield None
            return

        with self._mlflow.start_run(run_name=run_name, tags=tags, nested=nested) as run:
            self._active_run = run
            logger.info(f"MLflow run started: {run.info.run_id} ({run_name})")
            yield run
            self._active_run = None

    def log_params(self, params: Dict[str, Any]):
        if self._mlflow is None:
            return
        # MLflow requires string values
        flat = {k: str(v) for k, v in self._flatten(params).items()}
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self._mlflow is None:
            return
        clean = {k: float(v) for k, v in metrics.items() if _is_finite(v)}
        self._mlflow.log_metrics(clean, step=step)

    def log_training_result(self, result):
        """Log full TrainingResult including loss curves."""
        if self._mlflow is None:
            return

        # Log scalar summary
        self.log_metrics(result.summary())

        # Log train/val loss curves
        for step, (tl, vl) in enumerate(zip(result.train_losses, result.val_losses)):
            self._mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=step)

    def log_dataframe(self, df, artifact_name: str = "results.csv"):
        """Save a DataFrame as a CSV artifact."""
        if self._mlflow is None:
            return
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, artifact_name)
            df.to_csv(path, index=False)
            self._mlflow.log_artifact(path)

    def log_config(self, config: Dict[str, Any], name: str = "config.json"):
        """Save experiment config as a JSON artifact."""
        if self._mlflow is None:
            return
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, name)
            with open(path, "w") as f:
                json.dump(config, f, indent=2, default=str)
            self._mlflow.log_artifact(path)

    def log_figure(self, fig, artifact_name: str = "figure.png"):
        """Log a matplotlib or plotly figure."""
        if self._mlflow is None:
            return
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, artifact_name)
            try:
                fig.savefig(path, bbox_inches="tight", dpi=150)
            except AttributeError:
                fig.write_image(path)
            self._mlflow.log_artifact(path)

    def set_tags(self, tags: Dict[str, str]):
        if self._mlflow:
            self._mlflow.set_tags(tags)

    def get_run_url(self) -> Optional[str]:
        if self._active_run and self._mlflow:
            run_id = self._active_run.info.run_id
            return f"{self.tracking_uri}/#/experiments/{self._active_run.info.experiment_id}/runs/{run_id}"
        return None

    @staticmethod
    def _flatten(d: Dict, prefix: str = "") -> Dict:
        """Flatten nested dict for MLflow parameter logging."""
        result = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(MLflowTracker._flatten(v, key))
            else:
                result[key] = v
        return result


def _is_finite(v) -> bool:
    try:
        import math
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False
