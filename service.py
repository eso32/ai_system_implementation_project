import bentoml
from bentoml.models import BentoModel
from pydantic import BaseModel
import pandas as pd

class PenguinFeatures(BaseModel):
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str      # "MALE" / "FEMALE"
    island: str   # "Biscoe" / "Dream" / "Torgersen"

@bentoml.service(name="penguins_classifier_service")
class PenguinsService:
    penguins_classifier_model = BentoModel("penguins_classifier:latest")
    penguins_encoder = BentoModel("penguins_encoder:latest")
    num_cols = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
    cat_cols = ["island", "sex"]

    def __init__(self):
        self.penguins_classifier_model = bentoml.sklearn.load_model(self.penguins_classifier_model)
        self.penguins_encoder = bentoml.sklearn.load_model(self.penguins_encoder)

    @bentoml.api()
    def predict(self, features: PenguinFeatures):
        input_df = pd.DataFrame([{
            "culmen_length_mm": features.culmen_length_mm,
            "culmen_depth_mm": features.culmen_depth_mm,
            "flipper_length_mm": features.flipper_length_mm,
            "body_mass_g": features.body_mass_g,
            "sex": features.sex,
            "island": features.island,
        }])

        encoded = self.penguins_encoder.transform(input_df[self.cat_cols])
        encoded_df = pd.DataFrame(
            encoded.toarray(),
            columns=self.penguins_encoder.get_feature_names_out(self.cat_cols),
        )

        model_input = pd.concat([input_df[self.num_cols].reset_index(drop=True), encoded_df], axis=1)
        pred = self.penguins_classifier_model.predict(model_input)

        return {"prediction": pred[0]}

    @bentoml.api()
    def predict_batch(self, features_batch: pd.DataFrame):
        predictions = self.model.predict(features_batch)

        results = []
        for pred in predictions:
            results.append({"prediction": int(pred[0])})
        return results
