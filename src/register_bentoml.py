import bentoml
import pickle
import mlflow
import yaml

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    experiment_name = params["experiment"]["name"]
    mlflow.set_experiment(experiment_name)

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    model_tag = bentoml.sklearn.save_model(
        "penguins_classifier",
        model,
        signatures={
            "predict": {"batchable": True, "batch_dim": 0},
            "predict_proba": {"batchable": True, "batch_dim": 0},
        }
    )

    print(f"Model saved: {model_tag}")

    with open("models/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    encoder_tag = bentoml.sklearn.save_model(
        "penguins_encoder",
        model=encoder
    )

    with mlflow.start_run(run_name="register_bentoml"):
        mlflow.log_param("bentoml_model_name", "penguins_classifier")
        mlflow.log_param("bentoml_model_tag", str(model_tag))
        mlflow.log_param("bentoml_encoder_name", "penguins_encoder")
        mlflow.log_param("bentoml_encoder_tag", str(encoder_tag))

    print(f"Encoder saved: {encoder_tag}")


if __name__ == "__main__":
    main()
