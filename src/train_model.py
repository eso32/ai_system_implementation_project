"""
Skrypt do trenowania modelu Random Forest na danych Penguins.

Wczytuje dane treningowe i testowe, trenuje klasyfikator lasu losowego
z parametrami z pliku params.yaml, zapisuje model oraz metryki.
"""

import json
import os
import pickle

import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from optuna.integration.mlflow import MLflowCallback
import mlflow
import optuna

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    experiment_name = params["experiment"]["name"]
    train_df = pd.read_csv("data/train.csv")

    target_col = "species"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=2),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_samples_split": trial.suggest_int("min_samples_split", 3, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
        }

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42,
        )
        
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
        return scores.mean()
        
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="f1",
    )

    study = optuna.create_study(study_name=experiment_name, direction="maximize")
    study.optimize(objective, n_trials=20, callbacks=[mlflow_callback])

    print(f"### BEST PARAMS: {study.best_params}")
    
    mlflow.set_experiment(experiment_name)
    best_model = RandomForestClassifier(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model zapisany do: {model_path}")






if __name__ == "__main__":
    main()
