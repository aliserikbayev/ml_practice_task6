"""Streamlit user interface for the Iris FastAPI prediction service."""

import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered",
)

st.title("🌸 Iris Flower Classifier")
st.write(
    "Enter the flower measurements below. The application sends them to the "
    "FastAPI model and displays its prediction and class probabilities."
)

api_url = st.sidebar.text_input(
    "FastAPI base URL",
    value=os.getenv("API_URL", "http://127.0.0.1:8000"),
).strip().rstrip("/")
st.sidebar.caption(f"Prediction endpoint: {api_url}/predict")

with st.form("iris_prediction_form"):
    st.subheader("Flower measurements (cm)")
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input(
            "Sepal length", min_value=0.01, value=5.1, step=0.1
        )
        petal_length = st.number_input(
            "Petal length", min_value=0.01, value=1.4, step=0.1
        )
    with col2:
        sepal_width = st.number_input(
            "Sepal width", min_value=0.01, value=3.5, step=0.1
        )
        petal_width = st.number_input(
            "Petal width", min_value=0.01, value=0.2, step=0.1
        )

    submitted = st.form_submit_button("Predict species", type="primary")

if submitted:
    if not api_url:
        st.error("Enter the URL of the FastAPI service in the sidebar.")
    else:
        payload = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        }
        try:
            response = requests.post(
                f"{api_url}/predict", json=payload, timeout=10
            )
            response.raise_for_status()
            result = response.json()

            st.success(f"Predicted species: **{result['class_name']}**")
            st.metric("Predicted class ID", result["prediction"])

            probabilities = result.get("probabilities", {})
            if probabilities:
                st.subheader("Prediction probabilities")
                probability_table = pd.DataFrame(
                    {
                        "Species": list(probabilities.keys()),
                        "Probability": list(probabilities.values()),
                    }
                )
                st.bar_chart(probability_table.set_index("Species"))
                probability_table["Probability"] = probability_table[
                    "Probability"
                ].map(lambda value: f"{value * 100:.1f}%")
                st.dataframe(
                    probability_table,
                    hide_index=True,
                    use_container_width=True,
                )
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to FastAPI. Make sure the API is running "
                "and the URL in the sidebar is correct."
            )
        except requests.exceptions.Timeout:
            st.error("The API request timed out. Please try again.")
        except requests.exceptions.HTTPError as error:
            st.error(f"The API returned an error: {error}")
        except (KeyError, ValueError) as error:
            st.error(f"The API response was not in the expected format: {error}")
