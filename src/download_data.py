import os

import yaml
from sklearn.datasets import fetch_openml



def main():
    # Wczytanie parametrow z pliku konfiguracyjnego
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    dataset_id = params["data"]["dataset_id"]

    # Pobranie zbioru danych Titanic z OpenML
    print(f"Pobieranie zbioru danych o ID: {dataset_id} z OpenML...")
    data = fetch_openml(data_id=dataset_id, as_frame=True)
    df = data.frame

    print(f"Pobrano {len(df)} rekordow z {len(df.columns)} kolumnami.")

    # Utworzenie katalogu na dane, jesli nie istnieje
    os.makedirs("data", exist_ok=True)

    # Zapis surowych danych do pliku CSV
    output_path = "data/penguins.csv"
    df.to_csv(output_path, index=False)
    print(f"Dane zapisane do: {output_path}")


if __name__ == "__main__":
    main()
