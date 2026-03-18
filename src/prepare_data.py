"""
Skrypt do przygotowania danych Penguins.

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

    # 1 Oddzielenie cech od zmiennej docelowej (gatunek)
    X = df.drop(columns=["species"])
    y = df["species"]

    # 2 Podział na zbiory treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3 Oddzielenie kolumn kategorycznych i numerycznych
    categorical_cols = ["island", "sex"]
    num_cols = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]

    # 4 Fit i transformacja encodera
    # do wykonania po podziale na train/test bo to leakage inaczej
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore")
    X_train_cat = encoder.fit_transform(X_train[categorical_cols])

    X_test_cat = encoder.transform(X_test[categorical_cols])

    # 5 Zamiana na DataFrame z odpowiednimi nazwami kolumn
    X_train_cat_df = pd.DataFrame(
        X_train_cat.toarray(),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    X_test_cat_df = pd.DataFrame(
        X_test_cat.toarray(),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # # 3️⃣ Połączenie z kolumnami numerycznymi
    df_train_encoded = pd.concat([X_train[num_cols].reset_index(drop=True), X_train_cat_df], axis=1)
    df_test_encoded = pd.concat([X_test[num_cols].reset_index(drop=True), X_test_cat_df], axis=1)

    # Zachowujemy target w plikach CSV, bo etap trenowania odczytuje go z train/test.csv.
    train_df = df_train_encoded.reset_index(drop=True)
    train_df["species"] = y_train.reset_index(drop=True)

    test_df = df_test_encoded.reset_index(drop=True)
    test_df["species"] = y_test.reset_index(drop=True)

    # 6️⃣ Zapisanie danych i encodera
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print("Zapisano data/train.csv i data/test.csv.")


    # Utworzenie katalogu na dane, jesli nie istnieje
    os.makedirs("models", exist_ok=True)
    with open("models/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    print("Zapisano encoder.pkl")

if __name__ == "__main__":
    main()
