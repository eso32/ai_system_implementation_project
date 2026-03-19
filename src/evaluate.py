import pickle
import mlflow
from sklearn.metrics import f1_score, accuracy_score
import json
import pandas as pd
from mlflow.models import infer_signature
import yaml

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    experiment_name = params["experiment"]["name"]
    mlflow.set_experiment(experiment_name)
    target_col = "species"
    test_df = pd.read_csv("data/test.csv")

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]


    with open("models/model.pkl", "rb") as f:
        best_model = pickle.load(f)

    with mlflow.start_run(run_name="Best model"):
        y_pred = best_model.predict(X_test)
        
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        mlflow.log_params(get_params(best_model))

        f1 = f1_score(y_test, y_pred, average="weighted")
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", accuracy)

        metrics = {
            "f1_score": f1,
            "accuracy_score": accuracy,
        }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metryki zapisane do: metrics.json")

def get_params(best_model):
    return {
        "n_estimators": best_model.n_estimators,
        "max_depth": best_model.max_depth,
        "min_samples_split": best_model.min_samples_split,
        "min_samples_leaf": best_model.min_samples_leaf,
        "random_state": best_model.random_state,
    }


if __name__ == "__main__":
    main()
