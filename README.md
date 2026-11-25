# ML Model Deployment with FastAPI and Docker

A RandomForest classifier trained on the Iris dataset, served via a FastAPI REST API and containerized with Docker.

## Project structure
```
ml-fastapi-docker/
├── main.py            # FastAPI app
├── train.py           # trains and saves the model
├── model.joblib       # saved model
├── requirements.txt   # dependencies
├── Dockerfile         # container instructions
└── README.md
```

## 1. Train the model
```bash
pip install -r requirements.txt
python train.py
```

## 2. Run locally
```bash
uvicorn main:app --reload
```
- Root: http://127.0.0.1:8000/
- Swagger docs: http://127.0.0.1:8000/docs

## 3. Endpoints
**GET /** → `{"message": "ML API is running"}`

**POST /predict**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
Response:
```json
{"prediction": 0, "class_name": "setosa", "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}}
```

## 4. Docker
```bash
docker build -t ml-fastapi-app .
docker run -p 8000:8000 ml-fastapi-app
```
Then open http://localhost:8000/docs and test again.
