# ML Eval Framework

**Experimental Python toolkit for model benchmarking and data-ablation studies**

This public research repository contains scikit-learn and PyTorch model training utilities, tabular dataset processing, ablation experiments and selected predictive-performance, group-fairness, robustness and efficiency metrics. It is a **research prototype**, not verified production-ready software.

## Quick start

Python 3.10 or 3.11 recommended for the audited workflow.

```bash
git clone https://github.com/Rifa-111/ml-eval-framework.git
cd ml-eval-framework
python -m pip install -e ".[dev]"
python -m pytest -q tests/
python experiments/classification_benchmark.py --dataset iris --models lr rf
```

For an ablation example:

```bash
python experiments/ablation_demo.py --dataset iris --models logistic_regression --axes data_quality
```

The CLI accepts `--dataset`, `--models`, `--seed` and certain ablation-specific arguments. **The example YAML files in `configs/` are not connected to a `--config` CLI argument.** The default benchmark also includes a PyTorch MLP, which requires `pip install -e ".[torch]"`.

Optional integrations: `pip install -e ".[tracking]"` for MLflow and W&B, or `pip install -e ".[analysis]"` for plotting libraries. Installing dependencies does not automatically make tracking or a JAX trainer available. The inspected implementation does not establish a working JAX backend.

## Project layout

- `core/`: dataset splitting/preprocessing, model registry and scikit-learn/PyTorch trainers
- `ablation/`: data-quality and sample-size perturbations, experimental orchestration
- `metrics/`: classification, group fairness, noise robustness and runtime/memory calculations
- `integrations/`: optional MLflow and W&B wrappers
- `experiments/`: runnable benchmark and ablation scripts
- `tests/`: synthetic-data unit and integration checks
- `configs/`: illustrative YAML files, not a live CLI configuration system

## Important limitations

- The audit branch corrects a **data-leakage bug**: the previous data pipeline standardised features before train/test splitting. Historical outputs may therefore require regeneration before making performance claims.
- Some group fairness computations use heuristic group comparisons; no ethical or legal fairness guarantee is implied.
- Robustness tests use simulated perturbations. Results do not establish performance in deployment.
- `kfold_splits()` needs additional fold-specific preprocessing validation before it should be used to claim strictly leakage-free cross-validation.
- Generated MLflow data and Python caches are already present in repository history. New ignore rules do **not** erase previously committed files.
- Existing dataset and evaluation results are not an independently verified public benchmark.
- No external usage, third-party adoption or community impact is currently established by the repository alone.

## Reproducibility and contributing

See [RELEASE_AUDIT.md](RELEASE_AUDIT.md) and [CONTRIBUTING.md](CONTRIBUTING.md). The toolkit welcomes genuine issues, external reproducibility reports and contributions.

**Licence:** This repository currently has no verified top-level licence file. Public visibility is **not** itself an open-source licence; the owner should choose and add a licence before claiming MIT licensing or unrestricted reuse.
