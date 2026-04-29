# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the full DVC pipeline (download → prepare → train → evaluate → register)
dvc repro

# Run a single pipeline stage
dvc repro <stage_name>   # e.g. dvc repro train_model

# Run a stage script directly (from project root)
python src/download_data.py
python src/prepare_data.py
python src/train_model.py
python src/evaluate.py
python src/register_bentoml.py

# Serve the BentoML model
bentoml serve service:PenguinsService

# Build a Bento for deployment
bentoml build

# View MLflow experiments
mlflow ui
```

All scripts must be run from the project root — they reference `params.yaml`, `data/`, and `models/` via relative paths.

## Architecture

This is a 5-stage DVC pipeline for Palmer Penguins multi-class species classification (Adelie / Chinstrap / Gentoo).

**Pipeline stages** (`dvc.yaml`):
1. `download_data` — fetches raw dataset from OpenML (id `42585`) → `data/penguins.csv`
2. `prepare_data` — drops NaN rows, splits train/test, fits `OneHotEncoder(drop="first")` on train only to avoid leakage, saves encoded CSVs and `models/encoder.pkl`
3. `train_model` — Optuna (20 trials) tunes `RandomForestClassifier` hyperparams using 5-fold CV weighted F1; each trial is logged to MLflow via `MLflowCallback`; best model saved to `models/model.pkl`
4. `evaluate` — loads best model, evaluates on test set, logs metrics + model signature to MLflow, writes `metrics.json`
5. `register_bentoml` — saves model and encoder into the local BentoML model store as `penguins_classifier:latest` and `penguins_encoder:latest`

**Pipeline parameters** (`params.yaml`):
- `data.dataset_id` — OpenML dataset ID
- `prepare.test_size` / `prepare.random_state` — train/test split
- `experiment.name` — MLflow experiment name (also used as the Optuna study name)

**Serving** (`service.py`):
- `PenguinsService` loads both `penguins_classifier:latest` and `penguins_encoder:latest` from the BentoML store at startup
- `/predict` accepts a single `PenguinFeatures` Pydantic model; categorical columns (`island`, `sex`) are encoded with the stored encoder (using `toarray()` and `get_feature_names_out`), then concatenated with the four numeric columns before prediction
- The encoder registered in BentoML is the same object fitted in `prepare_data`, ensuring consistent feature engineering between training and inference

**Key constraint**: the encoder is fitted with `drop="first"` in `prepare_data.py`, but `service.py` calls `toarray()` directly without re-applying `drop`. The registered BentoML encoder preserves the fitted state including the drop setting, so inference is consistent with training.
