"""
Skrypt do trenowania modelu Random Forest na danych Titanic.

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


def main():
    # Wczytanie parametrow modelu z pliku konfiguracyjnego
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    n_estimators = params["model"]["n_estimators"]
    max_depth = params["model"]["max_depth"]
    random_state = params["model"]["random_state"]

    # Wczytanie zbiorow treningowego i testowego
    print("Wczytywanie danych treningowych i testowych...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Rozdzielenie cech (X) od zmiennej docelowej (y)
    target_col = "survived"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(
        f"Zbior treningowy: {X_train.shape[0]} probek, "
        f"{X_train.shape[1]} cech."
    )
    print(f"Zbior testowy: {X_test.shape[0]} probek.")

    # Trenowanie klasyfikatora lasu losowego
    print(
        f"Trenowanie RandomForestClassifier "
        f"(n_estimators={n_estimators}, max_depth={max_depth})..."
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    print("Model wytrenowany pomyslnie.")

    # Utworzenie katalogu na modele, jesli nie istnieje
    os.makedirs("models", exist_ok=True)

    # Zapis modelu do pliku za pomoca pickle
    model_path = "models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model zapisany do: {model_path}")

    # Obliczenie metryk na zbiorze testowym
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Dokladnosc (accuracy): {accuracy:.4f}")
    print(f"Miara F1 (f1_score):   {f1:.4f}")

    # Zapis metryk do pliku JSON
    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metryki zapisane do: metrics.json")


if __name__ == "__main__":
    main()
