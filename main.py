from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
target_names = artifact["target_names"]

app = FastAPI(title="Iris ML API", version="1.0.0")


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: IrisFeatures):
    x = [[features.sepal_length, features.sepal_width,
          features.petal_length, features.petal_width]]
    class_id = int(model.predict(x)[0])
    probs = model.predict_proba(x)[0]
    return {
        "prediction": class_id,
        "class_name": target_names[class_id],
        "probabilities": {n: round(float(p), 3) for n, p in zip(target_names, probs)},
    }
