# Iris ML System: FastAPI, Streamlit, Docker and MLflow

A Random Forest classifier trained on the Iris dataset. The model is served through a FastAPI REST API, can be used from a Streamlit web interface, and is tracked and registered with MLflow.

## Project structure

~~~text
.
├── main.py                 # FastAPI prediction API
├── train.py                # trains the model and logs/registers it with MLflow
├── model.joblib            # model artifact used by FastAPI
├── streamlit_app.py        # Streamlit user interface
├── requirements.txt        # API and MLflow dependencies
├── requirements-ui.txt     # frontend dependencies
├── Dockerfile              # FastAPI image
├── Dockerfile.frontend     # Streamlit image
├── docker-compose.yml      # MLflow, API and frontend services
└── README.md
~~~

## 1. Install dependencies

Python 3.11 is recommended.

~~~bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-ui.txt
~~~

## 2. Start MLflow

MLflow uses a SQLite backend store so that experiment tracking and the Model Registry are both available. Start the tracking server in a separate terminal from the project directory:

~~~bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
~~~

Open the MLflow UI at [http://127.0.0.1:5000](http://127.0.0.1:5000).

By default, **train.py** connects to **http://127.0.0.1:5000**. To use another MLflow server, set **MLFLOW_TRACKING_URI** before running the script.

## 3. Train, track and register the model

With the MLflow server running, execute:

~~~bash
python train.py
~~~

The script:

- trains a **RandomForestClassifier** on the Iris dataset;
- evaluates it using accuracy, macro F1, macro precision and macro recall;
- logs the hyperparameters, evaluation metrics and saved **model.joblib** artifact to the **Iris RandomForest** experiment;
- logs the scikit-learn model with its input signature and registers it as **IrisClassifier**.

Each execution creates a new MLflow run and a new registered model version. The script also writes **model.joblib** in the project directory in the format expected by the existing FastAPI application.

In the MLflow UI, open the experiment to inspect runs and metrics, then open the Models section (or the registered model page) to inspect **IrisClassifier** and its versions.

## 4. Run the FastAPI backend

In a new terminal:

~~~bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
~~~

- API status: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Interactive API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Prediction endpoint: **POST /predict**

Example request:

~~~bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
~~~

## 5. Run the Streamlit frontend

In another terminal, with the FastAPI backend running:

~~~bash
streamlit run streamlit_app.py
~~~

Open [http://localhost:8501](http://localhost:8501), enter the four flower measurements and select **Predict species**. The frontend calls the FastAPI **/predict** endpoint and displays the predicted species and class probabilities.

If the API is running at a different address, set **API_URL** or change the FastAPI base URL in the Streamlit sidebar.

## 6. Run the services with Docker Compose (optional)

Build and start MLflow, FastAPI and Streamlit:

~~~bash
docker compose up --build -d
~~~

The services are exposed at:

- MLflow: [http://localhost:5000](http://localhost:5000)
- FastAPI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Streamlit: [http://localhost:8501](http://localhost:8501)

To create an MLflow run while using the Compose MLflow service, run the training script on the host after the MLflow service has started:

~~~bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python train.py
~~~

The MLflow database and artifacts are stored in the named **mlflow_data** Docker volume. Stop the services with:

~~~bash
docker compose down
~~~

To remove the persisted MLflow database and artifacts as well, use **docker compose down -v**.

## Notes

- FastAPI continues to load **model.joblib** locally, so the original **/predict** API contract is preserved.
- MLflow tracks and registers each newly trained model; the registered model can be promoted or used for deployment in a later extension.
- Do not commit the local MLflow database, artifact directory, virtual environment or secrets. They are excluded by **.gitignore**.
