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
from sklearn.metrics import accuracy_score, f1_score
import mlflow

mlflow.set_experiment("palmer_penguins")

def main():
    # Wczytanie parametrow modelu z pliku konfiguracyjnego
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    n_estimators = params["model"]["n_estimators"]
    max_depth = params["model"]["max_depth"]
    min_samples_split = params["model"]["min_samples_split"]
    min_samples_leaf = params["model"]["min_samples_leaf"]

    # Wczytanie zbiorow treningowego i testowego
    print("Wczytywanie danych treningowych i testowych...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Rozdzielenie cech (X) od zmiennej docelowej (y)
    target_col = "species"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(
        f"Zbior treningowy: {X_train.shape[0]} probek, "
        f"{X_train.shape[1]} cech."
    )
    print(f"Zbior testowy: {X_test.shape[0]} probek.")


    with mlflow.start_run():
        params = { "n_estimators": n_estimators }
        # Trenowanie klasyfikatora lasu losowego
        print(
            f"Trenowanie RandomForestClassifier "
            f"(n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf})..."
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        mlflow.log_params(params);
        model.fit(X_train, y_train)
        
        for features, target, d_name in [(X_train, y_train, "train"), (X_test, y_test, "test")]:
            # Obliczenie metryk na zbiorze testowym
            y_pred = model.predict(features)
            accuracy = accuracy_score(target, y_pred)
            f1 = f1_score(target, y_pred, average="weighted")
            mlflow.log_metric(f"{d_name}_f1", f1)
            mlflow.log_metric(f"{d_name}_accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            input_example=X_train
        )

        print("Model wytrenowany pomyslnie.")

    # Utworzenie katalogu na modele, jesli nie istnieje
    os.makedirs("models", exist_ok=True)

    # Zapis modelu do pliku za pomoca pickle
    model_path = "models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model zapisany do: {model_path}")



    # # Zapis metryk do pliku JSON
    # metrics = {
    #     "accuracy": round(accuracy, 4),
    #     "f1_score": round(f1, 4),
    # }

    # with open("metrics.json", "w") as f:
    #     json.dump(metrics, f, indent=2)
    # print("Metryki zapisane do: metrics.json")


if __name__ == "__main__":
    main()
