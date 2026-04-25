"""
reporting.py
============
Visualization and HTML report generation for ablation and benchmark results.

Generates:
- Per-axis metric heatmaps
- Model comparison radar charts
- Training curve overlays
- Self-contained HTML report with embedded plots
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_metric_by_variant(
    df: pd.DataFrame,
    metric: str = "accuracy",
    axis_filter: Optional[str] = None,
    figsize: tuple = (12, 5),
):
    """
    Line plot of a metric across ablation variants, one line per model.

    Parameters
    ----------
    df          : ablation results DataFrame
    metric      : column name to plot
    axis_filter : if set, only show variants from this ablation axis
    """
    import matplotlib.pyplot as plt

    if axis_filter:
        df = df[df["axis"] == axis_filter].copy()

    if metric not in df.columns:
        logger.warning(f"Metric '{metric}' not in DataFrame columns.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    models = df["model"].unique()
    palette = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for model, color in zip(models, palette):
        sub = df[df["model"] == model].dropna(subset=[metric])
        if sub.empty:
            continue
        ax.plot(
            range(len(sub)),
            sub[metric].values,
            marker="o",
            label=model,
            color=color,
            linewidth=2,
            markersize=6,
        )
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["variant"].values, rotation=35, ha="right", fontsize=8)

    title = f"{metric}" + (f" — axis: {axis_filter}" if axis_filter else "")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(metric, fontsize=11)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_heatmap(
    df: pd.DataFrame,
    metrics: List[str],
    index_col: str = "variant",
    figsize: tuple = (10, 6),
):
    """Heatmap of metric values across variants (rows) and metrics (columns)."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    pivot_data = df.groupby(index_col)[metrics].mean()

    # Normalize each column to [0, 1] for color mapping
    normalized = (pivot_data - pivot_data.min()) / (pivot_data.max() - pivot_data.min() + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(normalized.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot_data)))
    ax.set_yticklabels(pivot_data.index, fontsize=9)

    # Annotate cells with raw values
    for i in range(len(pivot_data)):
        for j, metric in enumerate(metrics):
            val = pivot_data.iloc[i][metric]
            text_color = "black" if 0.3 < normalized.iloc[i][metric] < 0.8 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Normalized score (per metric)")
    ax.set_title("Metric Heatmap by Variant", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_training_curves(
    training_results: List[Any],
    figsize: tuple = (10, 4),
):
    """Overlay training + validation loss curves for multiple models."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    palette = plt.cm.tab10(np.linspace(0, 1, len(training_results)))

    for result, color in zip(training_results, palette):
        if not result.train_losses:
            continue
        epochs = range(len(result.train_losses))
        axes[0].plot(epochs, result.train_losses, color=color,
                     label=result.model_name, linewidth=1.5)
        axes[1].plot(epochs, result.val_losses, color=color,
                     label=result.model_name, linewidth=1.5, linestyle="--")

    for ax, title in zip(axes, ["Train Loss", "Val Loss"]):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    return fig


def plot_radar(
    df: pd.DataFrame,
    metrics: List[str],
    model_col: str = "model",
    figsize: tuple = (7, 7),
):
    """
    Radar (spider) chart comparing models across multiple metrics.

    All metrics are normalized to [0, 1] for fair comparison.
    """
    import matplotlib.pyplot as plt

    df_agg = df.groupby(model_col)[metrics].mean()
    # Normalize
    df_norm = (df_agg - df_agg.min()) / (df_agg.max() - df_agg.min() + 1e-8)

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    palette = plt.cm.Set1(np.linspace(0, 0.8, len(df_norm)))

    for (model, row), color in zip(df_norm.iterrows(), palette):
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=model, color=color)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison (normalized)", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HTML Report
# ─────────────────────────────────────────────────────────────────────────────

def generate_html_report(
    df: pd.DataFrame,
    output_path: Path,
    experiment_name: str = "Ablation Study",
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Generate a self-contained HTML report from ablation results.

    Embeds all plots as base64 PNGs — no external dependencies needed to view.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = metadata or {}
    plots_b64: Dict[str, str] = {}

    key_metrics = [c for c in ["accuracy", "f1", "roc_auc", "robustness_audc",
                                "efficiency_training_time_s"] if c in df.columns]

    # Plot 1: Accuracy by variant (all axes)
    fig = plot_metric_by_variant(df, metric="accuracy")
    if fig:
        plots_b64["accuracy_by_variant"] = _fig_to_base64(fig)
        plt.close(fig)

    # Plot 2: Heatmap
    if key_metrics:
        fig = plot_heatmap(df, key_metrics[:5])
        if fig:
            plots_b64["metric_heatmap"] = _fig_to_base64(fig)
            plt.close(fig)

    # Plot 3: Radar (if enough metrics)
    radar_metrics = [m for m in ["accuracy", "f1", "label_stability", "calibration_ece"]
                     if m in df.columns]
    if len(radar_metrics) >= 3:
        fig = plot_radar(df, radar_metrics)
        if fig:
            plots_b64["radar_chart"] = _fig_to_base64(fig)
            plt.close(fig)

    # Summary table
    summary_df = df.groupby(["model", "variant"])[key_metrics].mean().reset_index()
    summary_html = summary_df.to_html(
        classes="results-table", index=False,
        float_format=lambda x: f"{x:.4f}",
        border=0,
    )

    html = _render_html(
        experiment_name=experiment_name,
        metadata=metadata,
        summary_html=summary_html,
        plots_b64=plots_b64,
        n_models=df["model"].nunique(),
        n_variants=df["variant"].nunique(),
        n_runs=len(df),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML report saved to: {output_path}")
    return output_path


def _render_html(
    experiment_name: str,
    metadata: Dict,
    summary_html: str,
    plots_b64: Dict[str, str],
    n_models: int,
    n_variants: int,
    n_runs: int,
) -> str:
    plot_sections = ""
    plot_titles = {
        "accuracy_by_variant": "Accuracy Across Ablation Variants",
        "metric_heatmap": "Metric Heatmap",
        "radar_chart": "Multi-metric Radar Comparison",
    }
    for key, b64 in plots_b64.items():
        title = plot_titles.get(key, key.replace("_", " ").title())
        plot_sections += f"""
        <section class="plot-section">
            <h2>{title}</h2>
            <img src="data:image/png;base64,{b64}" alt="{title}" />
        </section>
        """

    meta_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metadata.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{experiment_name} — ML Eval Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; line-height: 1.6; }}
  header {{ background: linear-gradient(135deg, #1a1f2e 0%, #161b27 100%);
            border-bottom: 1px solid #2a3050; padding: 2rem 3rem; }}
  header h1 {{ font-size: 1.8rem; font-weight: 700; color: #7eb8f7; letter-spacing: -0.02em; }}
  header p {{ color: #8b9bb4; margin-top: 0.4rem; font-size: 0.95rem; }}
  .stats {{ display: flex; gap: 1.5rem; margin-top: 1.5rem; }}
  .stat-card {{ background: #1e2535; border: 1px solid #2a3050; border-radius: 8px;
                padding: 1rem 1.5rem; min-width: 120px; }}
  .stat-card .value {{ font-size: 2rem; font-weight: 700; color: #7eb8f7; }}
  .stat-card .label {{ font-size: 0.75rem; color: #6b7a99; text-transform: uppercase; letter-spacing: 0.08em; }}
  main {{ max-width: 1200px; margin: 2rem auto; padding: 0 2rem; }}
  section {{ margin-bottom: 2.5rem; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; color: #a8b8d0; margin-bottom: 1rem;
        padding-bottom: 0.5rem; border-bottom: 1px solid #2a3050; }}
  .plot-section img {{ max-width: 100%; border-radius: 8px; border: 1px solid #2a3050; }}
  .results-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .results-table th {{ background: #1e2535; color: #7eb8f7; padding: 0.6rem 1rem;
                       text-align: left; border-bottom: 2px solid #2a3050; font-weight: 600; }}
  .results-table td {{ padding: 0.5rem 1rem; border-bottom: 1px solid #1e2535; color: #c8d4e0; }}
  .results-table tr:hover td {{ background: #1a2030; }}
  .meta-table td {{ padding: 0.35rem 0.8rem; border-bottom: 1px solid #1e2535; font-size: 0.85rem; }}
  .meta-table td:first-child {{ color: #6b7a99; font-weight: 500; width: 180px; }}
  footer {{ text-align: center; color: #4a5568; font-size: 0.8rem; padding: 2rem;
            border-top: 1px solid #1e2535; margin-top: 3rem; }}
</style>
</head>
<body>
<header>
  <h1>🔬 {experiment_name}</h1>
  <p>ML Eval — End-to-End Model Evaluation &amp; Ablation Framework</p>
  <div class="stats">
    <div class="stat-card"><div class="value">{n_models}</div><div class="label">Models</div></div>
    <div class="stat-card"><div class="value">{n_variants}</div><div class="label">Variants</div></div>
    <div class="stat-card"><div class="value">{n_runs}</div><div class="label">Total Runs</div></div>
  </div>
</header>
<main>
  {"<section><h2>Experiment Metadata</h2><table class='meta-table'>" + meta_rows + "</table></section>" if meta_rows else ""}
  {plot_sections}
  <section>
    <h2>Full Results Table</h2>
    {summary_html}
  </section>
</main>
<footer>Generated by ML Eval Framework · {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</footer>
</body>
</html>"""
