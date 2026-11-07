"""
feature_selector.py
===================
Feature selection strategies for ablation studies.

Supports:
- Variance threshold selection
- Mutual information ranking
- Permutation importance (post-hoc, requires a fitted model)
- Top-K / Bottom-K selection
- Random subsets
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np

from core.data_pipeline import DataSplit

logger = logging.getLogger(__name__)


def top_k_by_variance(k: int) -> Callable[[DataSplit], DataSplit]:
    """
    Keep the K features with highest variance on the training set.

    Useful as a simple unsupervised feature filter before ablation.
    """
    def _select(split: DataSplit) -> DataSplit:
        import copy
        split = copy.deepcopy(split)
        variances = split.X_train.var(axis=0)
        idx = np.argsort(variances)[::-1][:k]
        split.X_train = split.X_train[:, idx]
        split.X_val = split.X_val[:, idx]
        split.X_test = split.X_test[:, idx]
        split.feature_names = [split.feature_names[i] for i in idx]
        logger.info(f"top_k_variance: selected {k} features from {len(variances)}")
        return split
    _select.__name__ = f"top_{k}_variance"
    return _select


def top_k_by_mutual_info(k: int, seed: int = 42) -> Callable[[DataSplit], DataSplit]:
    """
    Keep the K features with highest mutual information with the target.

    Uses sklearn's mutual_info_classif under the hood.
    """
    def _select(split: DataSplit) -> DataSplit:
        import copy
        from sklearn.feature_selection import mutual_info_classif
        split = copy.deepcopy(split)
        scores = mutual_info_classif(split.X_train, split.y_train, random_state=seed)
        idx = np.argsort(scores)[::-1][:k]
        split.X_train = split.X_train[:, idx]
        split.X_val = split.X_val[:, idx]
        split.X_test = split.X_test[:, idx]
        split.feature_names = [split.feature_names[i] for i in idx]
        logger.info(f"top_k_mutual_info: selected {k} features")
        return split
    _select.__name__ = f"top_{k}_mi"
    return _select


def random_feature_subset(fraction: float, seed: int = 42) -> Callable[[DataSplit], DataSplit]:
    """
    Randomly keep a fraction of features. Useful for measuring
    the impact of feature count vs. feature quality.
    """
    def _select(split: DataSplit) -> DataSplit:
        import copy
        split = copy.deepcopy(split)
        rng = np.random.RandomState(seed)
        n_total = split.X_train.shape[1]
        k = max(1, int(n_total * fraction))
        idx = rng.choice(n_total, k, replace=False)
        idx = np.sort(idx)
        split.X_train = split.X_train[:, idx]
        split.X_val = split.X_val[:, idx]
        split.X_test = split.X_test[:, idx]
        split.feature_names = [split.feature_names[i] for i in idx]
        return split
    _select.__name__ = f"random_{int(fraction * 100)}pct"
    return _select


def permutation_importance_selector(
    fitted_trainer,
    k: int,
    n_repeats: int = 10,
    seed: int = 42,
) -> Callable[[DataSplit], DataSplit]:
    """
    Select features by permutation importance using a fitted trainer.

    Permutation importance is model-agnostic and measures the degradation
    in accuracy when each feature's values are randomly shuffled.

    Parameters
    ----------
    fitted_trainer : a trainer that has already been .fit()
    k              : number of top features to keep
    n_repeats      : how many times to shuffle each feature
    seed           : random seed
    """
    def _select(split: DataSplit) -> DataSplit:
        import copy
        split = copy.deepcopy(split)
        rng = np.random.RandomState(seed)
        n_features = split.X_test.shape[1]

        baseline_acc = (fitted_trainer.predict(split.X_test) == split.y_test).mean()
        importances = np.zeros(n_features)

        for feat_idx in range(n_features):
            trial_drops = []
            for _ in range(n_repeats):
                X_perm = split.X_test.copy()
                X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
                acc = (fitted_trainer.predict(X_perm) == split.y_test).mean()
                trial_drops.append(baseline_acc - acc)
            importances[feat_idx] = np.mean(trial_drops)

        top_idx = np.argsort(importances)[::-1][:k]
        split.X_train = split.X_train[:, top_idx]
        split.X_val = split.X_val[:, top_idx]
        split.X_test = split.X_test[:, top_idx]
        split.feature_names = [split.feature_names[i] for i in top_idx]
        split.metadata["permutation_importances"] = {
            split.feature_names[i]: float(importances[top_idx[i]]) for i in range(len(top_idx))
        }
        logger.info(
            f"permutation_importance_selector: kept top {k} features. "
            f"Baseline acc={baseline_acc:.4f}"
        )
        return split
    _select.__name__ = f"perm_importance_top_{k}"
    return _select


def build_feature_selection_axis(
    n_features: int,
    fractions: tuple = (0.25, 0.50, 0.75, 1.0),
    method: str = "variance",
):
    """
    Build a feature selection ablation axis.

    Parameters
    ----------
    n_features : total number of features in the dataset
    fractions  : fractions of features to include at each variant
    method     : "variance" | "mutual_info" | "random"

    Returns
    -------
    list of AblationVariant
    """
    from ablation.engine import AblationVariant

    selector_map = {
        "variance": top_k_by_variance,
        "mutual_info": top_k_by_mutual_info,
        "random": random_feature_subset,
    }
    selector_fn = selector_map.get(method, top_k_by_variance)

    variants = []
    for frac in fractions:
        k = max(1, int(n_features * frac))
        name = f"feat_{method}_{int(frac * 100)}pct"
        if frac >= 1.0:
            perturbation = lambda s: s
        else:
            perturbation = selector_fn(k)
        variants.append(AblationVariant(
            name=name,
            description=f"{method} top {int(frac*100)}% features (k={k})",
            axis="feature_selection",
            perturbation=perturbation,
            params={"method": method, "fraction": frac, "k": k},
        ))
    return variants
