# Release and reproducibility audit

## Work reviewed

The project contains a modular implementation for model benchmarking, data-quality ablations, classification metrics and experiment tracking. This audit focuses on conservative correctness and documentation changes, not manufacturing evidence of independent community uptake.

## Findings and safeguards

1. **Data leakage (high):** The previous `DataPipeline.load()` fitted a `StandardScaler` on the full dataset before train/validation/test splitting. The review branch now fits preprocessing on training rows and transforms held-out rows with that fitted scaler. Earlier saved results are **not** recomputed.
2. **Sensitive-attribute alignment (high):** The previous `sensitive_train` used train+validation row indexes, not training-only indexes. This branch aligns it to the training subset.
3. **Evaluation caveat:** `kfold_splits()` uses a dataset already preprocessed under an initial train/validation split; it is not a fully fold-isolated preprocessing pipeline. Do not claim strict cross-validation leakage freedom until preprocessing is fitted within each fold.
4. **Claims:** “Production-grade”, full JAX backend coverage, automatic dual-tracking and YAML-configured CLI runs are not validated by the inspected scripts; README claims have been narrowed.
5. **Packaging:** Dependencies should be separated into a small core set and optional extras. Full PyTorch, JAX and tracking environments should be tested separately, not imposed on every pip installation.
6. **Repository hygiene:** The repository contains tracked `__pycache__/`, `.pyc`, `.DS_Store` and MLflow experiment records. Ignore rules prevent new additions but do not remove old files from history. Check any existing tracked logs for confidential paths, emails, data or credentials before wider promotion. A rewrite of Git history is not included.
7. **Licence:** Public GitHub visibility is not an open-source permission grant. Select a suitable licence explicitly before describing this project as MIT-licensed.
8. **Verification:** Passing unit tests is necessary but not a substitute for installing in a clean environment, running the CLI, validating metrics and obtaining genuine outside reproducibility review.

## Commands

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/
python experiments/classification_benchmark.py --dataset iris --models lr rf
```

Record the Git commit SHA, Python/pip versions, dataset version, command, console errors and output hashes when sharing independent reproduction results.

## UK Global Talent evidence

The repository, dated source commits, tests, descriptions and external PR/issue conversations can prove public accessibility and your technical work. They **cannot on their own** establish contribution to the wider technical community. Save genuine third-party review, independent usage, educational workshop adoption or upstream contributions if available. Do not create reciprocal/artificial stars or unsupported impact claims.
