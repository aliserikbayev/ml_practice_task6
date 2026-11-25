"""Train a simple classifier on the Iris dataset and save it to model.joblib."""
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")

joblib.dump({"model": model, "target_names": list(iris.target_names)}, "model.joblib")
print("Saved model to model.joblib")
