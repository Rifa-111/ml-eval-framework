# 🔬 ML Eval — End-to-End Model Evaluation & Ablation Framework

A production-grade, reusable framework for training, benchmarking and comparing machine learning models across multiple dataset variants with automated ablation studies.

## Architecture

```
ml_eval_framework/
├── core/                    # Training engine & model registry
│   ├── trainer.py           # Unified trainer (PyTorch + sklearn + JAX)
│   ├── registry.py          # Model & experiment registry
│   └── data_pipeline.py     # Dataset loading & variant generation
├── ablation/                # Ablation study engine
│   ├── engine.py            # Orchestrates ablation runs
│   ├── perturbations.py     # Data quality perturbations
│   └── feature_selector.py  # Feature selection strategies
├── metrics/                 # Evaluation metrics suite
│   ├── accuracy.py          # Standard classification/regression metrics
│   ├── fairness.py          # Fairness & bias metrics
│   ├── robustness.py        # Robustness & adversarial metrics
│   └── efficiency.py        # Runtime & memory profiling
├── integrations/            # MLflow + W&B connectors
│   ├── mlflow_tracker.py    # MLflow experiment tracking
│   └── wandb_tracker.py     # Weights & Biases integration
├── configs/                 # YAML experiment configs
│   ├── base_experiment.yaml
│   └── ablation_study.yaml
├── experiments/             # Example experiment scripts
│   ├── classification_benchmark.py
│   └── ablation_demo.py
└── tests/                   # Unit & integration tests
```

## Quick Start

```bash
pip install -r requirements.txt

# Run a full benchmark
python experiments/classification_benchmark.py --config configs/base_experiment.yaml

# Run ablation study
python experiments/ablation_demo.py --config configs/ablation_study.yaml

# Launch MLflow UI
mlflow ui --port 5000
```

## Key Features

- **Multi-backend**: Unified API across PyTorch, scikit-learn, and JAX
- **Automated Ablations**: Data quality, feature selection, sample composition
- **Rich Metrics**: Accuracy, fairness (demographic parity, equalized odds), robustness, runtime
- **Dual Tracking**: MLflow + Weights & Biases with automatic run syncing
- **Reproducible**: Seeded experiments with full config serialization
- **Extensible**: Plugin architecture for custom models and metrics
