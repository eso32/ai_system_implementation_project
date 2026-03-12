"""
Skrypt do przygotowania danych Penguins.

Wykonuje czyszczenie danych, uzupelnianie brakujacych wartosci,
kodowanie zmiennych kategorycznych oraz podzial na zbiory treningowy i testowy.
"""

import yaml
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def main():
    # Wczytanie parametrow podzialu danych
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    test_size = params["prepare"]["test_size"]
    random_state = params["prepare"]["random_state"]

    # Wczytanie surowych danych
    print("Wczytywanie danych z data/penguins.csv...")
    df = pd.read_csv("data/penguins.csv")
    print(f"Wczytano {len(df)} rekordow.")

    df = df.dropna();
    print("Zdropowano rekordy z brakującymi danymi.")

    # Oddzielenie kolumn kategorycznych i numerycznych
    categorical_cols = ["island", "sex"]
    num_cols = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]

    # 1️⃣ Fit i transformacja encodera
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore")
    X_cat = encoder.fit_transform(df[categorical_cols])

    # 2️⃣ Zamiana na DataFrame z odpowiednimi nazwami kolumn
    X_cat_df = pd.DataFrame(
        X_cat.toarray(),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # 3️⃣ Połączenie z kolumnami numerycznymi
    df_encoded = pd.concat([df[num_cols].reset_index(drop=True), X_cat_df], axis=1)

    # 4️⃣ Cel modelu (gatunek)
    y = df["species"]

    # 5️⃣ Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        df_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 6️⃣ Zapisanie danych i encodera
    X_train.to_csv("data/train.csv", index=False)
    X_test.to_csv("data/test.csv", index=False)
    print("Zapisano data/train.csv i data/test.csv.")


    # Utworzenie katalogu na dane, jesli nie istnieje
    os.makedirs("models", exist_ok=True)
    with open("models/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    print("Zapisano encoder.pkl")

if __name__ == "__main__":
    main()
