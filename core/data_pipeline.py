"""
data_pipeline.py
================
Dataset loading, preprocessing, and variant generation for ablation studies.
Supports tabular, image, and text datasets with a unified interface.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a dataset variant."""
    name: str
    source: str                         # path, sklearn dataset name, or HuggingFace id
    target_col: str = "label"
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed: int = 42
    feature_cols: Optional[List[str]] = None
    sensitive_cols: Optional[List[str]] = None   # for fairness evaluation
    max_samples: Optional[int] = None
    preprocessing: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable hash of config for caching."""
        raw = f"{self.name}{self.source}{self.target_col}{self.random_seed}"
        return hashlib.md5(raw.encode()).hexdigest()[:8]


@dataclass
class DataSplit:
    """Container for train/val/test splits."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    sensitive_train: Optional[pd.DataFrame] = None
    sensitive_test: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(np.unique(self.y_train))

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def class_balance(self) -> Dict[int, float]:
        unique, counts = np.unique(self.y_train, return_counts=True)
        total = len(self.y_train)
        return {int(c): round(cnt / total, 4) for c, cnt in zip(unique, counts)}

    def summary(self) -> Dict[str, Any]:
        return {
            "n_train": len(self.X_train),
            "n_val": len(self.X_val),
            "n_test": len(self.X_test),
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "class_balance": self.class_balance,
        }


class DataPipeline:
    """
    Unified data loading and preprocessing pipeline.

    Supports:
    - CSV / Parquet files
    - scikit-learn built-in datasets
    - Custom loading functions

    Example
    -------
    >>> cfg = DatasetConfig(name="breast_cancer", source="sklearn")
    >>> pipeline = DataPipeline(cfg)
    >>> split = pipeline.load()
    >>> print(split.summary())
    """

    _SKLEARN_DATASETS = {
        "iris": "load_iris",
        "breast_cancer": "load_breast_cancer",
        "wine": "load_wine",
        "digits": "load_digits",
        "diabetes": "load_diabetes",
        "california_housing": "fetch_california_housing",
    }

    def __init__(self, config: DatasetConfig, cache_dir: Optional[Path] = None):
        self.config = config
        self.cache_dir = cache_dir or Path(".cache/datasets")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._scaler: Optional[StandardScaler] = None
        self._label_encoder: Optional[LabelEncoder] = None

    def load(self) -> DataSplit:
        """Load, preprocess, and split the dataset."""
        logger.info(f"Loading dataset: {self.config.name} from {self.config.source}")
        X, y, feature_names, sensitive = self._load_raw()

        if self.config.max_samples and len(X) > self.config.max_samples:
            idx = np.random.RandomState(self.config.random_seed).choice(
                len(X), self.config.max_samples, replace=False
            )
            X, y = X[idx], y[idx]
            if sensitive is not None:
                sensitive = sensitive.iloc[idx].reset_index(drop=True)

        X = self._preprocess_features(X)
        y = self._encode_labels(y)

        split = self._split(X, y, sensitive, feature_names)
        logger.info(f"Dataset loaded. Summary: {split.summary()}")
        return split

    def _load_raw(self) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[pd.DataFrame]]:
        src = self.config.source.lower()

        if src == "sklearn" or src in self._SKLEARN_DATASETS:
            return self._load_sklearn()
        elif src.endswith(".csv"):
            return self._load_csv(Path(src))
        elif src.endswith(".parquet"):
            return self._load_parquet(Path(src))
        else:
            raise ValueError(f"Unsupported data source: {self.config.source}")

    def _load_sklearn(self) -> Tuple[np.ndarray, np.ndarray, List[str], None]:
        import sklearn.datasets as skd

        loader_name = self._SKLEARN_DATASETS.get(
            self.config.name, self._SKLEARN_DATASETS.get(self.config.source)
        )
        if loader_name is None:
            raise ValueError(f"Unknown sklearn dataset: {self.config.name}")

        loader = getattr(skd, loader_name)
        bunch = loader()
        X = bunch.data.astype(np.float32)
        y = bunch.target
        feature_names = list(bunch.feature_names) if hasattr(bunch, "feature_names") else [
            f"feature_{i}" for i in range(X.shape[1])
        ]
        return X, y, feature_names, None

    def _load_csv(self, path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[pd.DataFrame]]:
        df = pd.read_csv(path)
        return self._dataframe_to_arrays(df)

    def _load_parquet(self, path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[pd.DataFrame]]:
        df = pd.read_parquet(path)
        return self._dataframe_to_arrays(df)

    def _dataframe_to_arrays(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[pd.DataFrame]]:
        target = self.config.target_col
        feature_cols = self.config.feature_cols or [c for c in df.columns if c != target]

        sensitive = None
        if self.config.sensitive_cols:
            sensitive = df[self.config.sensitive_cols].copy()

        X = df[feature_cols].values.astype(np.float32)
        y = df[target].values
        return X, y, feature_cols, sensitive

    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        cfg = self.config.preprocessing
        if cfg.get("standardize", True):
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        if cfg.get("clip_outliers"):
            threshold = cfg.get("clip_threshold", 5.0)
            X = np.clip(X, -threshold, threshold)
        return X.astype(np.float32)

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        if y.dtype.kind in ("U", "O"):  # string labels
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)
        return y.astype(np.int64)

    def _split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive: Optional[pd.DataFrame],
        feature_names: List[str],
    ) -> DataSplit:
        seed = self.config.random_seed
        # First carve out test set
        X_trainval, X_test, y_trainval, y_test, idx_tv, idx_test = train_test_split(
            X, y, np.arange(len(X)),
            test_size=self.config.test_size,
            random_state=seed,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
        # Then carve out val from train+val
        val_ratio = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=seed,
            stratify=y_trainval if len(np.unique(y_trainval)) > 1 else None,
        )

        sens_train = sens_test = None
        if sensitive is not None:
            sens_train = sensitive.iloc[idx_tv].reset_index(drop=True)
            sens_test = sensitive.iloc[idx_test].reset_index(drop=True)

        return DataSplit(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            feature_names=feature_names,
            sensitive_train=sens_train,
            sensitive_test=sens_test,
            metadata={"config_fingerprint": self.config.fingerprint()},
        )

    def kfold_splits(self, n_splits: int = 5) -> Generator[DataSplit, None, None]:
        """Yield K-fold splits for cross-validation."""
        split = self.load()
        X_all = np.concatenate([split.X_train, split.X_val], axis=0)
        y_all = np.concatenate([split.y_train, split.y_val], axis=0)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            logger.info(f"Fold {fold_idx + 1}/{n_splits}")
            yield DataSplit(
                X_train=X_all[train_idx], y_train=y_all[train_idx],
                X_val=X_all[val_idx], y_val=y_all[val_idx],
                X_test=split.X_test, y_test=split.y_test,
                feature_names=split.feature_names,
                metadata={"fold": fold_idx, **split.metadata},
            )
