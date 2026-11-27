"""Train the Iris classifier, track the run in MLflow, and register the model."""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
EXPERIMENT_NAME = "Iris RandomForest"
REGISTERED_MODEL_NAME = "IrisClassifier"

# Use the MLflow server by default. Override this with MLFLOW_TRACKING_URI
# when using a different tracking server.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def main() -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    params = {
        "n_estimators": 100,
        "random_state": 42,
        "test_size": 0.2,
        "dataset": "Iris",
        "model_type": "RandomForestClassifier",
    }

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        random_state=params["random_state"],
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
        "precision_macro": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
    }

    # Keep the artifact format expected by the existing FastAPI application.
    joblib.dump(
        {"model": model, "target_names": list(iris.target_names)},
        MODEL_PATH,
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    # The same tracking server is used for the Model Registry.
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    signature = infer_signature(X_test, predictions)
    with mlflow.start_run(run_name="iris-random-forest") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="saved_model")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test[:5],
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"MLflow tracking URI: {TRACKING_URI}")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Registered model: {REGISTERED_MODEL_NAME}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        print(f"Saved FastAPI-compatible model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
