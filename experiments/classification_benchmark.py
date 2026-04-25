"""
classification_benchmark.py
============================
End-to-end benchmark script: loads data, trains multiple models,
evaluates them across all metric dimensions, and logs to MLflow + W&B.

Usage
-----
python experiments/classification_benchmark.py
python experiments/classification_benchmark.py --dataset wine --models rf gb
python experiments/classification_benchmark.py --config configs/base_experiment.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from core.data_pipeline import DataPipeline, DatasetConfig
from core.registry import ModelRegistry, ExperimentStore, ExperimentRecord
from core.trainer import TrainingConfig, build_trainer
from metrics.accuracy import AccuracyMetrics
from metrics.efficiency import EfficiencyMetrics
from metrics.robustness import RobustnessMetrics
from integrations.mlflow_tracker import MLflowTracker

console = Console()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SKLEARN_DATASETS = ["iris", "breast_cancer", "wine", "digits"]

MODEL_CONFIGS = {
    "lr": ("logistic_regression", "sklearn", {}),
    "rf": ("random_forest", "sklearn", {"n_estimators": 100}),
    "gb": ("gradient_boosting", "sklearn", {"n_estimators": 100}),
    "svm": ("svm", "sklearn", {"C": 1.0}),
    "knn": ("knn", "sklearn", {"n_neighbors": 5}),
    "mlp": ("mlp_pytorch", "pytorch", {"hidden_sizes": (128, 64), "dropout": 0.3}),
}


def run_benchmark(dataset_name: str, model_keys: list, seed: int = 42):
    console.rule(f"[bold cyan]ML Eval — Benchmark: {dataset_name}[/bold cyan]")

    # ── Data ─────────────────────────────────────────────────────────────
    console.print(f"[dim]Loading dataset:[/dim] [bold]{dataset_name}[/bold]")
    cfg = DatasetConfig(name=dataset_name, source="sklearn", random_seed=seed)
    pipeline = DataPipeline(cfg)
    split = pipeline.load()

    info = split.summary()
    console.print(
        f"  [green]✓[/green] {info['n_train']} train | {info['n_val']} val | "
        f"{info['n_test']} test | {info['n_features']} features | "
        f"{info['n_classes']} classes"
    )

    # ── Setup ─────────────────────────────────────────────────────────────
    registry = ModelRegistry()
    store = ExperimentStore(Path("results/benchmark"))
    tracker = MLflowTracker(experiment_name=f"benchmark_{dataset_name}")

    acc_metrics = AccuracyMetrics()
    eff_metrics = EfficiencyMetrics()
    rob_metrics = RobustnessMetrics(noise_levels=(0.01, 0.05, 0.1, 0.5), n_trials=3)

    all_results = []

    # ── Training Loop ─────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training models...", total=len(model_keys))

        for key in model_keys:
            model_name, backend, hparams = MODEL_CONFIGS[key]
            progress.update(task, description=f"Training [bold]{model_name}[/bold] ({backend})")

            train_cfg = TrainingConfig(
                model_name=model_name,
                backend=backend,
                hyperparams=hparams,
                epochs=50,
                batch_size=32,
                learning_rate=1e-3,
                weight_decay=1e-4,
                random_seed=seed,
                verbose=False,
            )

            # Build trainer
            if backend == "sklearn":
                model_instance = registry.build(model_name, **hparams)
                trainer = build_trainer(train_cfg, model_instance=model_instance)
            elif backend == "pytorch":
                model_instance = registry.build(
                    "mlp_pytorch",
                    n_features=split.n_features,
                    n_classes=split.n_classes,
                    **hparams,
                )
                trainer = build_trainer(train_cfg, model=model_instance)
            else:
                progress.advance(task)
                continue

            # Train & evaluate
            with tracker.start_run(run_name=f"{model_name}_{dataset_name}",
                                   tags={"model": model_name, "dataset": dataset_name}):
                try:
                    training_result = trainer.fit(
                        split.X_train, split.y_train,
                        split.X_val, split.y_val,
                    )
                    y_pred = trainer.predict(split.X_test)
                    y_proba = None
                    try:
                        y_proba = trainer.predict_proba(split.X_test)
                    except Exception:
                        pass

                    metrics = {}
                    metrics.update(acc_metrics.compute(split.y_test, y_pred, y_proba))
                    metrics.update(eff_metrics.from_training_result(training_result))
                    metrics.update(rob_metrics.compute(trainer, split.X_test, split.y_test))

                    tracker.log_params({**train_cfg.hyperparams, "dataset": dataset_name})
                    tracker.log_metrics(metrics)
                    tracker.log_training_result(training_result)

                    record = ExperimentRecord(
                        experiment_id="benchmark",
                        run_id=ExperimentStore.new_run_id(),
                        model_name=model_name,
                        dataset_name=dataset_name,
                        dataset_variant="original",
                        backend=backend,
                        metrics=metrics,
                        training_summary=training_result.summary(),
                        config=train_cfg.hyperparams,
                        tags={"key": key},
                    )
                    store.add(record)
                    all_results.append({"model": model_name, **metrics})

                except Exception as e:
                    console.print(f"  [red]✗ {model_name} failed: {e}[/red]")

            progress.advance(task)

    # ── Results Table ─────────────────────────────────────────────────────
    if all_results:
        _print_results_table(all_results)

    console.print(f"\n[dim]MLflow results:[/dim] run [bold]mlflow ui --port 5000[/bold]")
    console.print(f"[dim]Results saved to:[/dim] [bold]results/benchmark/[/bold]")

    return store.to_dataframe()


def _print_results_table(results: list):
    console.print()
    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Accuracy", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("ROC-AUC", justify="right")
    table.add_column("Train Time (s)", justify="right")
    table.add_column("Noise Robustness", justify="right")

    # Sort by accuracy descending
    results_sorted = sorted(results, key=lambda r: r.get("accuracy", 0), reverse=True)

    for r in results_sorted:
        acc = r.get("accuracy", float("nan"))
        f1 = r.get("f1", float("nan"))
        auc = r.get("roc_auc", float("nan"))
        t = r.get("efficiency_training_time_s", float("nan"))
        rob = 1 - r.get("robustness_audc", float("nan"))  # higher = more robust

        table.add_row(
            r["model"],
            f"{acc:.4f}" if not _isnan(acc) else "—",
            f"{f1:.4f}" if not _isnan(f1) else "—",
            f"{auc:.4f}" if not _isnan(auc) else "—",
            f"{t:.3f}s" if not _isnan(t) else "—",
            f"{rob:.4f}" if not _isnan(rob) else "—",
        )

    console.print(table)


def _isnan(v):
    try:
        import math
        return math.isnan(v)
    except Exception:
        return True


def main():
    parser = argparse.ArgumentParser(description="ML Eval Benchmark")
    parser.add_argument("--dataset", default="breast_cancer", choices=SKLEARN_DATASETS)
    parser.add_argument("--models", nargs="+", default=["lr", "rf", "gb", "mlp"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_benchmark(dataset_name=args.dataset, model_keys=args.models, seed=args.seed)


if __name__ == "__main__":
    main()
