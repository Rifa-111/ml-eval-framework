"""
perturbations.py - Data perturbations for ablation studies.
"""
from __future__ import annotations
import copy
from typing import Callable, List
import numpy as np

from core.data_pipeline import DataSplit


def label_noise(noise_rate: float, seed: int = 42) -> Callable[[DataSplit], DataSplit]:
    """Randomly flip training labels with probability noise_rate."""
    def _perturb(split: DataSplit) -> DataSplit:
        split = copy.deepcopy(split)
        if noise_rate <= 0:
            return split
        rng = np.random.RandomState(seed)
        n = len(split.y_train)
        n_corrupt = int(n * noise_rate)
        indices = rng.choice(n, n_corrupt, replace=False)
        classes = np.unique(split.y_train)
        for idx in indices:
            original = split.y_train[idx]
            candidates = classes[classes != original]
            if len(candidates):
                split.y_train[idx] = rng.choice(candidates)
        return split
    _perturb.__name__ = f"label_noise_{noise_rate}"
    return _perturb


def feature_gaussian_noise(sigma: float = 0.1, seed: int = 42) -> Callable[[DataSplit], DataSplit]:
    """Add Gaussian noise to all training features."""
    def _perturb(split: DataSplit) -> DataSplit:
        split = copy.deepcopy(split)
        rng = np.random.RandomState(seed)
        split.X_train = (split.X_train + rng.normal(0, sigma, split.X_train.shape)).astype(np.float32)
        return split
    _perturb.__name__ = f"gaussian_noise_sigma{sigma}"
    return _perturb


def feature_masking(mask_rate: float = 0.1, seed: int = 42) -> Callable[[DataSplit], DataSplit]:
    """Zero out a random fraction of feature values."""
    def _perturb(split: DataSplit) -> DataSplit:
        split = copy.deepcopy(split)
        rng = np.random.RandomState(seed)
        mask = rng.random(split.X_train.shape) < mask_rate
        split.X_train = split.X_train.copy()
        split.X_train[mask] = 0.0
        return split
    _perturb.__name__ = f"feature_masking_{mask_rate}"
    return _perturb


def feature_dropout(feature_indices: List[int]) -> Callable[[DataSplit], DataSplit]:
    """Zero out specific features from train/val/test."""
    def _perturb(split: DataSplit) -> DataSplit:
        split = copy.deepcopy(split)
        for idx in feature_indices:
            split.X_train[:, idx] = 0.0
            split.X_val[:, idx] = 0.0
            split.X_test[:, idx] = 0.0
        return split
    return _perturb


def sample_subsetting(fraction: float, seed: int = 42) -> Callable[[DataSplit], DataSplit]:
    """Randomly subsample a fraction of training data."""
    def _perturb(split: DataSplit) -> DataSplit:
        split = copy.deepcopy(split)
        rng = np.random.RandomState(seed)
        n = len(split.X_train)
        k = max(1, int(n * fraction))
        idx = rng.choice(n, k, replace=False)
        split.X_train = split.X_train[idx]
        split.y_train = split.y_train[idx]
        return split
    _perturb.__name__ = f"subsample_{fraction}"
    return _perturb


def class_imbalance(imbalance_ratios: dict, seed: int = 42) -> Callable[[DataSplit], DataSplit]:
    """Downsample training classes to impose artificial imbalance."""
    def _perturb(split: DataSplit) -> DataSplit:
        split = copy.deepcopy(split)
        rng = np.random.RandomState(seed)
        keep_indices = []
        for cls, frac in imbalance_ratios.items():
            cls_idx = np.where(split.y_train == cls)[0]
            k = max(1, int(len(cls_idx) * frac))
            keep_indices.extend(rng.choice(cls_idx, k, replace=False).tolist())
        keep_indices = np.array(keep_indices)
        split.X_train = split.X_train[keep_indices]
        split.y_train = split.y_train[keep_indices]
        return split
    return _perturb


def outlier_injection(fraction: float = 0.05, magnitude: float = 5.0, seed: int = 42) -> Callable[[DataSplit], DataSplit]:
    """Inject extreme outlier values into a fraction of training samples."""
    def _perturb(split: DataSplit) -> DataSplit:
        split = copy.deepcopy(split)
        rng = np.random.RandomState(seed)
        n = len(split.X_train)
        k = max(1, int(n * fraction))
        rows = rng.choice(n, k, replace=False)
        cols = rng.randint(0, split.X_train.shape[1], k)
        signs = rng.choice([-1, 1], k)
        split.X_train = split.X_train.copy()
        split.X_train[rows, cols] += signs * magnitude
        return split
    return _perturb


def build_data_quality_axis():
    """Standard data quality ablation variants."""
    from ablation.engine import AblationVariant
    return [
        AblationVariant("dq_clean",      "No corruption",       "data_quality", lambda s: s,                    params={"noise": 0.0}),
        AblationVariant("dq_noise_5",    "5% label noise",      "data_quality", label_noise(0.05),               params={"noise": 0.05}),
        AblationVariant("dq_noise_10",   "10% label noise",     "data_quality", label_noise(0.10),               params={"noise": 0.10}),
        AblationVariant("dq_noise_20",   "20% label noise",     "data_quality", label_noise(0.20),               params={"noise": 0.20}),
        AblationVariant("dq_gauss_01",   "Gaussian σ=0.1",      "data_quality", feature_gaussian_noise(0.1),     params={"sigma": 0.1}),
        AblationVariant("dq_gauss_05",   "Gaussian σ=0.5",      "data_quality", feature_gaussian_noise(0.5),     params={"sigma": 0.5}),
        AblationVariant("dq_missing_10", "10% missing values",  "data_quality", feature_masking(0.10),           params={"missing_rate": 0.10}),
        AblationVariant("dq_missing_30", "30% missing values",  "data_quality", feature_masking(0.30),           params={"missing_rate": 0.30}),
        AblationVariant("dq_outliers_5", "5% outlier injection","data_quality", outlier_injection(0.05, 5.0),    params={"outlier_frac": 0.05}),
    ]


def build_sample_size_axis(fractions=(0.1, 0.25, 0.5, 0.75, 1.0)):
    """Sample-size scaling ablation variants."""
    from ablation.engine import AblationVariant
    variants = []
    for frac in fractions:
        fn = (lambda s: s) if frac >= 1.0 else sample_subsetting(frac)
        variants.append(AblationVariant(
            name=f"sample_{int(frac * 100)}pct",
            description=f"{int(frac * 100)}% of training data",
            axis="sample_size",
            perturbation=fn,
            params={"fraction": frac},
        ))
    return variants
