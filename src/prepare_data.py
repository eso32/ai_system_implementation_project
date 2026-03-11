"""
Skrypt do przygotowania danych Titanic.

Wykonuje czyszczenie danych, uzupelnianie brakujacych wartosci,
kodowanie zmiennych kategorycznych oraz podzial na zbiory treningowy i testowy.
"""

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # Wczytanie parametrow podzialu danych
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    test_size = params["prepare"]["test_size"]
    random_state = params["prepare"]["random_state"]

    # Wczytanie surowych danych
    print("Wczytywanie danych z data/titanic.csv...")
    df = pd.read_csv("data/titanic.csv")
    print(f"Wczytano {len(df)} rekordow.")

    # Usuniecie kolumn, ktore nie sa przydatne do modelowania
    kolumny_do_usuniecia = ["name", "ticket", "cabin", "body", "boat", "home.dest"]
    df = df.drop(columns=kolumny_do_usuniecia)
    print(f"Usunieto kolumny: {kolumny_do_usuniecia}")

    # Uzupelnienie brakujacych wartosci
    # - wiek: mediana
    # - oplata: mediana
    # - port zaokretowania: wartosc najczestsza (moda)
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    print("Uzupelniono brakujace wartosci (age, fare, embarked).")

    # Kodowanie zmiennej plec: male=0, female=1
    df["sex"] = df["sex"].map({"male": 0, "female": 1}).astype(int)

    # Kodowanie one-hot dla portu zaokretowania (z usunieciem pierwszej kategorii)
    df = pd.get_dummies(df, columns=["embarked"], drop_first=True)
    print("Zakodowano zmienne kategoryczne (sex, embarked).")

    # Konwersja zmiennej docelowej na typ calkowity
    df["survived"] = df["survived"].astype(int)

    # Podzial na zbiory treningowy i testowy
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    print(
        f"Podzielono dane: {len(train_df)} treningowych, "
        f"{len(test_df)} testowych (test_size={test_size})."
    )

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print("Zapisano data/train.csv i data/test.csv.")


if __name__ == "__main__":
    main()
