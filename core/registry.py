"""
registry.py
===========
Central registry for models and experiments.
Models can be registered by name and instantiated from configs.
Experiments are tracked with unique IDs and metadata.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type


@dataclass
class ExperimentRecord:
    """Immutable record of a completed experiment."""
    experiment_id: str
    run_id: str
    model_name: str
    dataset_name: str
    dataset_variant: str
    backend: str
    metrics: Dict[str, float]
    training_summary: Dict[str, Any]
    config: Dict[str, Any]
    tags: Dict[str, str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRecord":
        return cls(**d)


class ModelRegistry:
    """
    Registry for model constructors, enabling lookup by name.

    Models can be registered with the @register decorator or
    using registry.register().

    Example
    -------
    >>> registry = ModelRegistry()
    >>>
    >>> @registry.register("random_forest")
    ... def _rf(**kwargs):
    ...     from sklearn.ensemble import RandomForestClassifier
    ...     return RandomForestClassifier(**kwargs)
    >>>
    >>> model = registry.build("random_forest", n_estimators=100)
    """

    def __init__(self):
        self._factories: Dict[str, Callable] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._register_defaults()

    def register(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Decorator to register a model factory."""
        def decorator(fn: Callable):
            self._factories[name] = fn
            self._metadata[name] = metadata or {}
            return fn
        return decorator

    def build(self, name: str, **kwargs) -> Any:
        if name not in self._factories:
            available = list(self._factories.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return self._factories[name](**kwargs)

    def list_models(self) -> List[str]:
        return sorted(self._factories.keys())

    def get_metadata(self, name: str) -> Dict[str, Any]:
        return self._metadata.get(name, {})

    def _register_defaults(self):
        """Register built-in sklearn and PyTorch models."""

        # --- scikit-learn ---
        @self.register("logistic_regression", {"backend": "sklearn", "task": "classification"})
        def _lr(**kwargs):
            from sklearn.linear_model import LogisticRegression
            kwargs.setdefault("max_iter", 1000)
            kwargs.setdefault("random_state", 42)
            return LogisticRegression(**kwargs)

        @self.register("random_forest", {"backend": "sklearn", "task": "classification"})
        def _rf(**kwargs):
            from sklearn.ensemble import RandomForestClassifier
            kwargs.setdefault("n_estimators", 100)
            kwargs.setdefault("random_state", 42)
            return RandomForestClassifier(**kwargs)

        @self.register("gradient_boosting", {"backend": "sklearn", "task": "classification"})
        def _gb(**kwargs):
            from sklearn.ensemble import GradientBoostingClassifier
            kwargs.setdefault("n_estimators", 100)
            kwargs.setdefault("random_state", 42)
            return GradientBoostingClassifier(**kwargs)

        @self.register("svm", {"backend": "sklearn", "task": "classification"})
        def _svm(**kwargs):
            from sklearn.svm import SVC
            kwargs.setdefault("probability", True)
            kwargs.setdefault("random_state", 42)
            return SVC(**kwargs)

        @self.register("knn", {"backend": "sklearn", "task": "classification"})
        def _knn(**kwargs):
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(**kwargs)

        @self.register("xgboost", {"backend": "sklearn", "task": "classification"})
        def _xgb(**kwargs):
            try:
                from xgboost import XGBClassifier
                kwargs.setdefault("random_state", 42)
                kwargs.setdefault("eval_metric", "mlogloss")
                return XGBClassifier(**kwargs)
            except ImportError:
                raise ImportError("xgboost is not installed. Run: pip install xgboost")

        # --- PyTorch ---
        @self.register("mlp_pytorch", {"backend": "pytorch", "task": "classification"})
        def _mlp(n_features: int, n_classes: int, hidden_sizes=(128, 64), dropout=0.2, **kwargs):
            import torch.nn as nn
            layers = []
            in_size = n_features
            for h in hidden_sizes:
                layers.extend([nn.Linear(in_size, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
                in_size = h
            layers.append(nn.Linear(in_size, n_classes))
            return nn.Sequential(*layers)

        @self.register("wide_mlp_pytorch", {"backend": "pytorch", "task": "classification"})
        def _wide_mlp(n_features: int, n_classes: int, **kwargs):
            return self.build("mlp_pytorch", n_features=n_features, n_classes=n_classes,
                              hidden_sizes=(512, 256, 128), dropout=0.3)


class ExperimentStore:
    """
    Persistent store for experiment records (JSON Lines format).

    Supports querying by model, dataset, tags, and metrics.

    Example
    -------
    >>> store = ExperimentStore(Path("experiments/results"))
    >>> store.add(record)
    >>> df = store.to_dataframe()
    >>> best = store.best_by_metric("accuracy", higher_is_better=True)
    """

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[ExperimentRecord] = []
        self._load_existing()

    def _load_existing(self):
        jsonl = self.store_dir / "records.jsonl"
        if jsonl.exists():
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._records.append(
                                ExperimentRecord.from_dict(json.loads(line))
                            )
                        except Exception:
                            pass

    def add(self, record: ExperimentRecord):
        self._records.append(record)
        with open(self.store_dir / "records.jsonl", "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def query(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[ExperimentRecord]:
        results = self._records
        if model_name:
            results = [r for r in results if r.model_name == model_name]
        if dataset_name:
            results = [r for r in results if r.dataset_name == dataset_name]
        if tags:
            results = [r for r in results if all(r.tags.get(k) == v for k, v in tags.items())]
        return results

    def best_by_metric(
        self,
        metric: str,
        higher_is_better: bool = True,
        filters: Optional[Dict[str, str]] = None,
    ) -> Optional[ExperimentRecord]:
        records = self.query(**(filters or {}))
        candidates = [r for r in records if metric in r.metrics]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.metrics[metric] * (1 if higher_is_better else -1))

    def to_dataframe(self):
        import pandas as pd
        rows = []
        for r in self._records:
            row = {
                "experiment_id": r.experiment_id,
                "run_id": r.run_id,
                "model_name": r.model_name,
                "dataset_name": r.dataset_name,
                "dataset_variant": r.dataset_variant,
                "backend": r.backend,
                "timestamp": r.timestamp,
                **{f"metric_{k}": v for k, v in r.metrics.items()},
                **{f"tag_{k}": v for k, v in r.tags.items()},
            }
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def new_run_id() -> str:
        return str(uuid.uuid4())[:8]


# Global singleton registry
default_registry = ModelRegistry()
