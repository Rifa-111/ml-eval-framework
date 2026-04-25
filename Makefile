# Makefile — ML Eval Framework
# ==============================
# Common development tasks. Run `make help` to see available commands.

.PHONY: help install test lint benchmark ablation report clean

PYTHON  := python
PYTEST  := pytest
DATASET := breast_cancer
MODELS  := lr rf gb mlp

help:
	@echo ""
	@echo "  ML Eval Framework — Development Commands"
	@echo "  ========================================="
	@echo "  make install      Install dependencies"
	@echo "  make test         Run unit tests with coverage"
	@echo "  make lint         Run ruff + mypy"
	@echo "  make benchmark    Run classification benchmark (breast_cancer)"
	@echo "  make ablation     Run ablation study (breast_cancer)"
	@echo "  make report       Generate HTML report from latest results"
	@echo "  make mlflow       Launch MLflow UI at port 5000"
	@echo "  make clean        Remove generated files"
	@echo ""

install:
	pip install -r requirements.txt
	pip install -e .

test:
	$(PYTEST) tests/ -v --tb=short --cov=. --cov-report=term-missing --cov-report=html:coverage_html

lint:
	@echo "Running ruff..."
	ruff check . --fix || true
	@echo "Running mypy..."
	mypy core/ metrics/ ablation/ --ignore-missing-imports || true

benchmark:
	$(PYTHON) experiments/classification_benchmark.py \
		--dataset $(DATASET) \
		--models $(MODELS)

ablation:
	$(PYTHON) experiments/ablation_demo.py \
		--dataset $(DATASET) \
		--axes data_quality sample_size \
		--models logistic_regression random_forest gradient_boosting

report:
	$(PYTHON) -c "\
import pandas as pd; \
from pathlib import Path; \
from scripts.reporting import generate_html_report; \
import sys; \
results_dir = Path('results/ablation'); \
csvs = list(results_dir.glob('*.csv')); \
if not csvs: sys.exit('No results CSVs found. Run make ablation first.'); \
df = pd.concat([pd.read_csv(f) for f in csvs]); \
out = generate_html_report(df, Path('results/report.html'), 'Ablation Study Report'); \
print(f'Report: {out}') \
"

mlflow:
	mlflow ui --port 5000

clean:
	rm -rf results/ mlruns/ wandb/ coverage_html/ .cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
