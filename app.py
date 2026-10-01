# app.py
# Iris Flower Classification — Streamlit Web App
# ------------------------------------------------
# Run with: streamlit run app.py
# Make sure you have run train.py first to generate model.pkl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered",
)

# ------------------------------------------------------------------
# Load the saved model pipeline
# ------------------------------------------------------------------
MODEL_PATH = "model.pkl"


@st.cache_resource
def load_model():
    """Load the trained pipeline from disk. Cached so it loads once."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model_data = load_model()

if model_data is None:
    st.error(
        "**model.pkl not found.**\n\n"
        "Please train the model first by running:\n"
        "```\npython train.py\n```"
    )
    st.stop()

pipeline = model_data["model"]
target_names = model_data["target_names"]

# ------------------------------------------------------------------
# Sidebar — user input sliders
# ------------------------------------------------------------------
st.sidebar.title("Flower Measurements")
st.sidebar.markdown("Adjust the sliders to enter the four measurements:")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.4, step=0.1
)
sepal_width = st.sidebar.slider(
    "Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.4, step=0.1
)
petal_length = st.sidebar.slider(
    "Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.7, step=0.1
)
petal_width = st.sidebar.slider(
    "Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.5, step=0.1
)

# Show selected values as a summary table in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Selected values:**")
input_summary = pd.DataFrame(
    {
        "Feature": [
            "Sepal Length (cm)",
            "Sepal Width (cm)",
            "Petal Length (cm)",
            "Petal Width (cm)",
        ],
        "Value": [sepal_length, sepal_width, petal_length, petal_width],
    }
)
st.sidebar.dataframe(input_summary, hide_index=True, use_container_width=True)

predict_button = st.sidebar.button("Predict", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# Main page — title and description
# ------------------------------------------------------------------
st.title("Iris Flower Classifier")
st.markdown(
    """
    This app uses a **Logistic Regression** model (with StandardScaler preprocessing)
    trained on the classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)
    to predict the species of an Iris flower from four measurements.

    Use the **sidebar sliders** to enter the measurements, then click **Predict**.
    """
)

st.markdown("---")

# ------------------------------------------------------------------
# Prediction section
# ------------------------------------------------------------------
st.subheader("Prediction")

if predict_button:
    # Prepare input as a 2D array for the pipeline
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict species and probabilities (pipeline handles scaling internally)
    prediction_index = pipeline.predict(input_features)[0]
    probabilities = pipeline.predict_proba(input_features)[0]
    predicted_species = target_names[prediction_index]

    # Display the predicted species
    st.success(f"**Predicted Species: {predicted_species.capitalize()}**")

    # Probability bar chart
    st.markdown("**Prediction Probabilities:**")
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.barh(
        [name.capitalize() for name in target_names],
        probabilities,
        color=colors,
        edgecolor="white",
        height=0.5,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Prediction Probabilities by Species")
    # Annotate bars with probability values
    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.2%}",
            va="center",
            fontsize=10,
        )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.info("Adjust the sliders in the sidebar and click **Predict** to see the result.")

st.markdown("---")

# ------------------------------------------------------------------
# Model Performance section (expandable)
# ------------------------------------------------------------------
with st.expander("Model Performance", expanded=False):
    st.markdown(
        "Metrics computed on the **test set** (20% of the Iris dataset, same split used during training)."
    )

    # Reproduce the same train/test split used in train.py
    iris = load_iris()
    X = iris.data
    y = iris.target
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Accuracy metric
    st.metric(label="Test Accuracy", value=f"{accuracy * 100:.2f}%")

    st.markdown("**Classification Report:**")
    # Parse classification_report output into a DataFrame for clean display
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.round(2)
    st.dataframe(report_df, use_container_width=True)

    st.markdown("**Confusion Matrix:**")
    cm_path = "confusion_matrix.png"
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix - Iris Classifier", use_container_width=True)
    else:
        st.warning(
            "confusion_matrix.png not found. Run `python train.py` to generate it."
        )
