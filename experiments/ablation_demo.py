"""
ablation_demo.py
================
Demonstrates the full ablation study pipeline:
- Data quality degradation axis (label noise, feature corruption)
- Sample size scaling axis
- Dual tracking with MLflow + W&B
- Results visualization

Usage
-----
python experiments/ablation_demo.py
python experiments/ablation_demo.py --dataset wine --models rf gb
python experiments/ablation_demo.py --axes data_quality sample_size
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
from rich.panel import Panel

from ablation.engine import AblationEngine
from ablation.perturbations import build_data_quality_axis, build_sample_size_axis
from core.data_pipeline import DataPipeline, DatasetConfig
from core.registry import ModelRegistry
from core.trainer import TrainingConfig, build_trainer
from integrations.mlflow_tracker import MLflowTracker

console = Console()
logging.basicConfig(level=logging.WARNING)


def make_trainer_factory(registry: ModelRegistry, n_features: int, n_classes: int):
    """Returns a factory function for the ablation engine."""

    def factory(model_name: str, n_features: int, n_classes: int):
        # Map model names to their backend
        pytorch_models = {"mlp_pytorch", "wide_mlp_pytorch"}
        backend = "pytorch" if model_name in pytorch_models else "sklearn"

        cfg = TrainingConfig(
            model_name=model_name,
            backend=backend,
            epochs=40,
            batch_size=32,
            learning_rate=1e-3,
            early_stopping_patience=8,
            random_seed=42,
            verbose=False,
        )

        if backend == "sklearn":
            hyperparams = _default_hyperparams(model_name)
            model_instance = registry.build(model_name, **hyperparams)
            return build_trainer(cfg, model_instance=model_instance)
        else:
            model_instance = registry.build(
                model_name,
                n_features=n_features,
                n_classes=n_classes,
                hidden_sizes=(64, 32),
                dropout=0.2,
            )
            return build_trainer(cfg, model=model_instance)

    return factory


def _default_hyperparams(model_name: str) -> dict:
    defaults = {
        "logistic_regression": {"C": 1.0},
        "random_forest": {"n_estimators": 100, "max_depth": 8},
        "gradient_boosting": {"n_estimators": 100},
        "svm": {},
        "knn": {"n_neighbors": 5},
    }
    return defaults.get(model_name, {})


def run_ablation(
    dataset_name: str,
    model_names: list,
    axes: list,
    seed: int = 42,
    output_dir: Path = Path("results/ablation"),
):
    console.rule("[bold magenta]ML Eval — Ablation Study[/bold magenta]")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    console.print(f"[dim]Dataset:[/dim] [bold]{dataset_name}[/bold]")
    cfg = DatasetConfig(name=dataset_name, source="sklearn", random_seed=seed)
    split = DataPipeline(cfg).load()
    console.print(f"  [green]✓[/green] {split.summary()}")

    # ── Engine Setup ──────────────────────────────────────────────────────
    registry = ModelRegistry()
    factory = make_trainer_factory(registry, split.n_features, split.n_classes)
    tracker = MLflowTracker(experiment_name=f"ablation_{dataset_name}")

    engine = AblationEngine(base_split=split, trainer_factory=factory)

    # ── Register Axes ─────────────────────────────────────────────────────
    if "data_quality" in axes:
        dq_variants = build_data_quality_axis()
        engine.add_axis("data_quality", dq_variants)
        console.print(f"[green]✓[/green] Registered data_quality axis: {len(dq_variants)} variants")

    if "sample_size" in axes:
        ss_variants = build_sample_size_axis(fractions=(0.10, 0.25, 0.50, 0.75, 1.0))
        engine.add_axis("sample_size", ss_variants)
        console.print(f"[green]✓[/green] Registered sample_size axis: {len(ss_variants)} variants")

    # ── Progress Bar ──────────────────────────────────────────────────────
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running ablation...", total=None)

        def update_progress(done, total, desc):
            progress.update(task, total=total, completed=done, description=f"[cyan]{desc}")

        with tracker.start_run(run_name=f"ablation_{dataset_name}",
                               tags={"dataset": dataset_name, "axes": ",".join(axes)}):
            results_df = engine.run(
                model_names=model_names,
                include_baseline=True,
                progress_callback=update_progress,
            )

            if not results_df.empty:
                tracker.log_dataframe(results_df, "ablation_results.csv")
                tracker.log_config({
                    "dataset": dataset_name,
                    "models": model_names,
                    "axes": axes,
                    "seed": seed,
                })

    # ── Save & Display ────────────────────────────────────────────────────
    if not results_df.empty:
        csv_path = output_dir / f"ablation_{dataset_name}.csv"
        results_df.to_csv(csv_path, index=False)
        console.print(f"\n[green]Results saved:[/green] {csv_path}")

        _print_ablation_summary(results_df, engine)
        _generate_plots(results_df, output_dir, dataset_name)
    else:
        console.print("[yellow]No results to display.[/yellow]")

    console.print(f"\n[dim]Launch MLflow:[/dim] [bold]mlflow ui --port 5000[/bold]")
    return results_df


def _print_ablation_summary(df: pd.DataFrame, engine: AblationEngine):
    console.print()
    console.print(Panel.fit("[bold]Ablation Summary by Axis[/bold]", style="magenta"))
    summary = engine.summary_by_axis()

    if "accuracy" in summary.columns:
        for axis in summary["axis"].unique():
            ax_df = summary[summary["axis"] == axis].sort_values("accuracy", ascending=False)
            console.print(f"\n[cyan]Axis: {axis}[/cyan]")
            for _, row in ax_df.head(5).iterrows():
                acc = row.get("accuracy", float("nan"))
                f1 = row.get("f1", float("nan"))
                console.print(
                    f"  {row['variant']:<25} model={row['model']:<20} "
                    f"acc={acc:.4f}  f1={f1:.4f}"
                )


def _generate_plots(df: pd.DataFrame, output_dir: Path, dataset_name: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        metric_cols = [c for c in df.columns if c in ("accuracy", "f1", "roc_auc")]
        if not metric_cols:
            return

        fig, axes = plt.subplots(1, len(metric_cols), figsize=(6 * len(metric_cols), 5))
        if len(metric_cols) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metric_cols):
            for model in df["model"].unique():
                sub = df[df["model"] == model].copy()
                sub = sub.dropna(subset=[metric])
                if sub.empty:
                    continue
                ax.plot(
                    range(len(sub)),
                    sub[metric].values,
                    marker="o",
                    label=model,
                    linewidth=1.5,
                )
                ax.set_xticks(range(len(sub)))
                ax.set_xticklabels(sub["variant"].values, rotation=45, ha="right", fontsize=7)

            ax.set_title(f"{metric} by variant", fontweight="bold")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Ablation Study — {dataset_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        plot_path = output_dir / f"ablation_{dataset_name}_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"[green]Plot saved:[/green] {plot_path}")

    except ImportError:
        console.print("[yellow]matplotlib not available, skipping plots[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="ML Eval — Ablation Study")
    parser.add_argument("--dataset", default="breast_cancer",
                        choices=["iris", "breast_cancer", "wine", "digits"])
    parser.add_argument("--models", nargs="+",
                        default=["logistic_regression", "random_forest", "gradient_boosting"])
    parser.add_argument("--axes", nargs="+",
                        default=["data_quality", "sample_size"],
                        choices=["data_quality", "sample_size"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/ablation")
    args = parser.parse_args()

    run_ablation(
        dataset_name=args.dataset,
        model_names=args.models,
        axes=args.axes,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
