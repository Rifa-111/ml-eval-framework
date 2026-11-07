"""
test_framework.py
=================
Unit and integration tests for the ML Eval Framework.

Run with: pytest tests/ -v --tb=short
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def binary_split():
    """Synthetic binary classification DataSplit."""
    from core.data_pipeline import DataSplit
    rng = np.random.RandomState(42)
    n_train, n_val, n_test, n_feat = 200, 40, 60, 10
    return DataSplit(
        X_train=rng.randn(n_train, n_feat).astype(np.float32),
        y_train=rng.randint(0, 2, n_train).astype(np.int64),
        X_val=rng.randn(n_val, n_feat).astype(np.float32),
        y_val=rng.randint(0, 2, n_val).astype(np.int64),
        X_test=rng.randn(n_test, n_feat).astype(np.float32),
        y_test=rng.randint(0, 2, n_test).astype(np.int64),
        feature_names=[f"f{i}" for i in range(n_feat)],
    )


@pytest.fixture
def multiclass_split():
    """Synthetic 3-class classification DataSplit."""
    from core.data_pipeline import DataSplit
    rng = np.random.RandomState(0)
    n_train, n_val, n_test, n_feat = 300, 60, 90, 15
    return DataSplit(
        X_train=rng.randn(n_train, n_feat).astype(np.float32),
        y_train=rng.randint(0, 3, n_train).astype(np.int64),
        X_val=rng.randn(n_val, n_feat).astype(np.float32),
        y_val=rng.randint(0, 3, n_val).astype(np.int64),
        X_test=rng.randn(n_test, n_feat).astype(np.float32),
        y_test=rng.randint(0, 3, n_test).astype(np.int64),
        feature_names=[f"f{i}" for i in range(n_feat)],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data Pipeline Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataPipeline:
    def test_load_sklearn_breast_cancer(self):
        from core.data_pipeline import DataPipeline, DatasetConfig
        cfg = DatasetConfig(name="breast_cancer", source="sklearn")
        split = DataPipeline(cfg).load()
        assert split.n_features == 30
        assert split.n_classes == 2
        assert len(split.X_train) > 0

    def test_load_sklearn_iris(self):
        from core.data_pipeline import DataPipeline, DatasetConfig
        cfg = DatasetConfig(name="iris", source="sklearn")
        split = DataPipeline(cfg).load()
        assert split.n_classes == 3

    def test_split_sizes(self):
        from core.data_pipeline import DataPipeline, DatasetConfig
        cfg = DatasetConfig(name="breast_cancer", source="sklearn", test_size=0.2, val_size=0.1)
        split = DataPipeline(cfg).load()
        total = len(split.X_train) + len(split.X_val) + len(split.X_test)
        assert total == 569  # known dataset size

    def test_max_samples(self):
        from core.data_pipeline import DataPipeline, DatasetConfig
        cfg = DatasetConfig(name="breast_cancer", source="sklearn", max_samples=100)
        split = DataPipeline(cfg).load()
        total = len(split.X_train) + len(split.X_val) + len(split.X_test)
        assert total == 100

    def test_reproducibility(self):
        from core.data_pipeline import DataPipeline, DatasetConfig
        cfg = DatasetConfig(name="breast_cancer", source="sklearn", random_seed=123)
        s1 = DataPipeline(cfg).load()
        s2 = DataPipeline(cfg).load()
        np.testing.assert_array_equal(s1.y_test, s2.y_test)

    def test_kfold(self):
        from core.data_pipeline import DataPipeline, DatasetConfig
        cfg = DatasetConfig(name="iris", source="sklearn")
        splits = list(DataPipeline(cfg).kfold_splits(n_splits=5))
        assert len(splits) == 5
        for s in splits:
            assert len(s.X_train) > 0
            assert len(s.X_val) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelRegistry:
    def test_default_models_registered(self):
        from core.registry import ModelRegistry
        reg = ModelRegistry()
        models = reg.list_models()
        assert "random_forest" in models
        assert "logistic_regression" in models
        assert "mlp_pytorch" in models

    def test_build_sklearn_model(self):
        from core.registry import ModelRegistry
        reg = ModelRegistry()
        model = reg.build("random_forest", n_estimators=50)
        assert hasattr(model, "fit")

    def test_custom_registration(self):
        from core.registry import ModelRegistry
        reg = ModelRegistry()

        @reg.register("my_custom_model")
        def _factory(**kwargs):
            from sklearn.dummy import DummyClassifier
            return DummyClassifier(**kwargs)

        model = reg.build("my_custom_model", strategy="most_frequent")
        assert model is not None

    def test_unknown_model_raises(self):
        from core.registry import ModelRegistry
        reg = ModelRegistry()
        with pytest.raises(KeyError):
            reg.build("does_not_exist")


# ─────────────────────────────────────────────────────────────────────────────
# Trainer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSklearnTrainer:
    def test_fit_predict(self, binary_split):
        from sklearn.ensemble import RandomForestClassifier
        from core.trainer import TrainingConfig, SklearnTrainer

        cfg = TrainingConfig(model_name="rf", backend="sklearn")
        trainer = SklearnTrainer(cfg, model_instance=RandomForestClassifier(n_estimators=10))
        result = trainer.fit(
            binary_split.X_train, binary_split.y_train,
            binary_split.X_val, binary_split.y_val,
        )
        assert result.backend == "sklearn"
        assert result.training_time_s > 0

        preds = trainer.predict(binary_split.X_test)
        assert preds.shape == (len(binary_split.X_test),)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, binary_split):
        from sklearn.ensemble import RandomForestClassifier
        from core.trainer import TrainingConfig, SklearnTrainer

        cfg = TrainingConfig(model_name="rf", backend="sklearn")
        trainer = SklearnTrainer(cfg, model_instance=RandomForestClassifier(n_estimators=10))
        trainer.fit(
            binary_split.X_train, binary_split.y_train,
            binary_split.X_val, binary_split.y_val,
        )
        proba = trainer.predict_proba(binary_split.X_test)
        assert proba.shape == (len(binary_split.X_test), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestPyTorchTrainer:
    def test_fit_predict(self, binary_split):
        pytest.importorskip("torch")
        from core.registry import ModelRegistry
        from core.trainer import TrainingConfig, PyTorchTrainer

        reg = ModelRegistry()
        model = reg.build("mlp_pytorch", n_features=10, n_classes=2)
        cfg = TrainingConfig(model_name="mlp", backend="pytorch", epochs=5, verbose=False)
        trainer = PyTorchTrainer(cfg, model=model)
        result = trainer.fit(
            binary_split.X_train, binary_split.y_train,
            binary_split.X_val, binary_split.y_val,
        )
        assert result.backend == "pytorch"
        assert len(result.train_losses) <= 5

        preds = trainer.predict(binary_split.X_test)
        assert preds.shape == (len(binary_split.X_test),)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAccuracyMetrics:
    def test_perfect_predictions(self):
        from metrics.accuracy import AccuracyMetrics
        y = np.array([0, 1, 0, 1, 0])
        metrics = AccuracyMetrics().compute(y, y)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_all_wrong(self):
        from metrics.accuracy import AccuracyMetrics
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        metrics = AccuracyMetrics().compute(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_multiclass_metrics_present(self):
        from metrics.accuracy import AccuracyMetrics
        rng = np.random.RandomState(0)
        y = rng.randint(0, 3, 100)
        metrics = AccuracyMetrics().compute(y, y)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "mcc" in metrics

    def test_with_proba_includes_auc(self):
        from metrics.accuracy import AccuracyMetrics
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X, y = load_breast_cancer(return_X_y=True)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        clf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        metrics = AccuracyMetrics().compute(y_te, clf.predict(X_te), proba)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] > 0.9


class TestRobustnessMetrics:
    def test_compute_returns_dict(self, binary_split):
        from sklearn.ensemble import RandomForestClassifier
        from core.trainer import TrainingConfig, SklearnTrainer
        from metrics.robustness import RobustnessMetrics

        cfg = TrainingConfig(model_name="rf", backend="sklearn")
        trainer = SklearnTrainer(cfg, model_instance=RandomForestClassifier(n_estimators=10))
        trainer.fit(
            binary_split.X_train, binary_split.y_train,
            binary_split.X_val, binary_split.y_val,
        )
        rob = RobustnessMetrics(noise_levels=(0.1,), n_trials=2)
        metrics = rob.compute(trainer, binary_split.X_test, binary_split.y_test)
        assert isinstance(metrics, dict)
        assert "robustness_audc" in metrics
        assert isinstance(metrics["robustness_audc"], float)


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPerturbations:
    def test_label_noise_changes_labels(self, binary_split):
        from ablation.perturbations import label_noise
        import copy
        original = copy.deepcopy(binary_split.y_train)
        perturbed = label_noise(0.3)(binary_split)
        changed = (original != perturbed.y_train).sum()
        # Roughly 30% should change
        assert changed > 0

    def test_label_noise_zero_is_identity(self, binary_split):
        from ablation.perturbations import label_noise
        import copy
        original = copy.deepcopy(binary_split.y_train)
        perturbed = label_noise(0.0)(binary_split)
        np.testing.assert_array_equal(original, perturbed.y_train)

    def test_sample_subsetting_reduces_size(self, binary_split):
        from ablation.perturbations import sample_subsetting
        n_before = len(binary_split.X_train)
        perturbed = sample_subsetting(0.5)(binary_split)
        n_after = len(perturbed.X_train)
        assert n_after == pytest.approx(n_before * 0.5, abs=2)

    def test_feature_masking_zeros(self, binary_split):
        from ablation.perturbations import feature_masking
        perturbed = feature_masking(0.5)(binary_split)
        zero_fraction = (perturbed.X_train == 0).mean()
        assert zero_fraction > 0.0  # some features should be zeroed

    def test_test_split_unchanged_by_perturbation(self, binary_split):
        """Test split should never be perturbed."""
        from ablation.perturbations import label_noise
        import copy
        original_test = copy.deepcopy(binary_split.y_test)
        perturbed = label_noise(0.5)(binary_split)
        np.testing.assert_array_equal(original_test, perturbed.y_test)


# ─────────────────────────────────────────────────────────────────────────────
# Integration Test
# ─────────────────────────────────────────────────────────────────────────────

class TestAblationIntegration:
    def test_mini_ablation_run(self, binary_split):
        from sklearn.ensemble import RandomForestClassifier
        from ablation.engine import AblationEngine, AblationVariant
        from core.trainer import TrainingConfig, SklearnTrainer

        def factory(model_name, n_features, n_classes):
            cfg = TrainingConfig(model_name=model_name, backend="sklearn")
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            return SklearnTrainer(cfg, model_instance=model)

        engine = AblationEngine(base_split=binary_split, trainer_factory=factory)
        engine.add_axis("test_axis", [
            AblationVariant("v1", "Identity", "test_axis", lambda s: s),
        ])

        df = engine.run(model_names=["rf"], include_baseline=True)
        assert len(df) == 2  # baseline + v1
        assert "accuracy" in df.columns
        assert "variant" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


class TestReleaseRegression:
    def test_scaler_is_fitted_to_training_samples_only(self):
        from core.data_pipeline import DataPipeline, DatasetConfig
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        cfg = DatasetConfig(name="iris", source="sklearn", random_seed=42)
        pipeline = DataPipeline(cfg)
        split = pipeline.load()
        raw = load_iris().data.astype(np.float32)
        labels = load_iris().target
        X_tv, _, y_tv, _ = train_test_split(
            raw, labels, test_size=cfg.test_size, random_state=cfg.random_seed,
            stratify=labels,
        )
        X_train, _, _, _ = train_test_split(
            X_tv, y_tv, test_size=cfg.val_size / (1 - cfg.test_size),
            random_state=cfg.random_seed, stratify=y_tv,
        )
        np.testing.assert_allclose(
            pipeline._scaler.mean_, X_train.mean(axis=0), rtol=1e-5, atol=1e-5,
        )
        assert len(split.X_train) > 0

    def test_sensitive_attributes_follow_actual_training_rows(self, tmp_path):
        import pandas as pd
        from core.data_pipeline import DataPipeline, DatasetConfig
        df = pd.DataFrame({
            "feature": np.arange(120, dtype=float),
            "label": np.tile([0, 1], 60),
            "cohort": [f"id_{i}" for i in range(120)],
        })
        path = tmp_path / "cohort.csv"
        df.to_csv(path, index=False)
        cfg = DatasetConfig(
            name="cohort", source=str(path), target_col="label",
            feature_cols=["feature"], sensitive_cols=["cohort"],
            preprocessing={"standardize": False},
        )
        split = DataPipeline(cfg).load()
        assert len(split.sensitive_train) == len(split.X_train)
        assert len(split.sensitive_test) == len(split.X_test)
        for row, cohort in zip(split.X_train, split.sensitive_train["cohort"]):
            assert cohort == f"id_{int(row[0])}"

    def test_ablation_imports_from_repository_root(self):
        from ablation.engine import AblationEngine
        from ablation.perturbations import build_data_quality_axis
        from ablation.feature_selector import top_k_by_variance
        assert AblationEngine is not None
        assert callable(build_data_quality_axis)
        assert callable(top_k_by_variance)
