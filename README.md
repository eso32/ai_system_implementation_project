# 🐧 Palmer Penguins — End-to-End ML Pipeline

This project demonstrates a complete **end-to-end machine learning pipeline** for multi-class classification using the Palmer Penguins dataset. It covers the full lifecycle from data ingestion to model serving, with a strong focus on reproducibility, experiment tracking, and production readiness.

---

## 📊 Dataset

- **Source:** OpenML (ID: 42585)  
- **Task:** Classify penguin species (*Adelie, Chinstrap, Gentoo*)  

**Features:**
- Numerical: `culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g`
- Categorical: `island`, `sex`

---

## ⚙️ Pipeline Overview (DVC)

The project is structured as a reproducible pipeline using **DVC**, consisting of 5 stages:


download_data → prepare_data → train_model → evaluate → register_bentoml


### Outputs:
- Processed datasets (train/test split)
- Trained model (`models/model.pkl`)
- Encoder (`models/encoder.pkl`)
- Evaluation metrics (`metrics.json`)
- Registered models in BentoML

---

## 🧹 Data Preprocessing

- Missing values handled with `dropna()`
- Categorical encoding using `OneHotEncoder`
- Encoder persisted and reused across pipeline and inference
- Multi-class classification with **weighted F1-score**

---

## 🤖 Model Training

- Model: Scikit-learn (tree-based)
- Hyperparameter tuning with **Optuna**
- Evaluation using **5-fold cross-validation**
- Experiment tracking via **MLflow**
- ≥20 optimization trials

---

## 📈 Evaluation

- Metrics:
  - Accuracy
  - Weighted F1-score

- Results stored in:
  - `metrics.json` (DVC)
  - MLflow (best model run with parameters and metrics)

---

## 📦 Model Serving (BentoML)

Model and encoder are registered and served using **BentoML**.

### Endpoint: `/predict`

```json
{
  "culmen_length_mm": float,
  "culmen_depth_mm": float,
  "flipper_length_mm": float,
  "body_mass_g": float,
  "sex": "MALE | FEMALE",
  "island": "Biscoe | Dream | Torgersen"
}
```

Ensures consistent preprocessing between training and inference
Loads both model and encoder from BentoML store

## 🔁 Reproducibility & Experimentation
DVC — pipeline and data versioning
MLflow — experiment tracking and model registry
Optuna — hyperparameter optimization

## 🚀 How to Run
dvc repro
bentoml serve service:PenguinsService

## 📌 Highlights
End-to-end ML system (data → training → deployment)
Reproducible pipeline with DVC
Integrated experiment tracking (MLflow)
Production-ready model serving (BentoML)
Consistent feature engineering across all stages
