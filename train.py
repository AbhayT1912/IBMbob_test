# train.py
# Iris Flower Classification — Training Script
# -----------------------------------------------
# This script:
#   1. Loads the built-in Iris dataset
#   2. Splits data into training and test sets (80/20)
#   3. Builds a Pipeline: StandardScaler + LogisticRegression
#   4. Trains and evaluates the model
#   5. Saves the confusion matrix plot as confusion_matrix.png
#   6. Saves the trained pipeline to model.pkl using Joblib

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# ------------------------------------------------------------------
# 1. Load the Iris dataset
# ------------------------------------------------------------------
print("=" * 55)
print("  Iris Flower Classification - Training Script")
print("=" * 55)

iris = load_iris()

# Wrap in a DataFrame so beginners can inspect it easily
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = [iris.target_names[t] for t in iris.target]

print("\n[INFO] Dataset preview (first 5 rows):")
print(df.head())
print(f"\nDataset shape: {df.shape[0]} samples, {len(iris.feature_names)} features")
print(f"Classes      : {list(iris.target_names)}")

# ------------------------------------------------------------------
# 2. Split into training and test sets (80/20)
# ------------------------------------------------------------------
X = iris.data
y = iris.target
target_names = list(iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain samples : {X_train.shape[0]}")
print(f"Test samples  : {X_test.shape[0]}")

# ------------------------------------------------------------------
# 3. Build Pipeline: StandardScaler → LogisticRegression
# ------------------------------------------------------------------
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=200, random_state=42)),
    ]
)

# ------------------------------------------------------------------
# 4. Train the pipeline
# ------------------------------------------------------------------
pipeline.fit(X_train, y_train)
print("\n[INFO] Model training complete.")

# ------------------------------------------------------------------
# 5. Evaluate the model
# ------------------------------------------------------------------
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(y_test, y_pred, target_names=target_names)
)

# ------------------------------------------------------------------
# 6. Confusion matrix — save as confusion_matrix.png
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=target_names,
    ax=ax,
    colorbar=False,
    cmap="Blues",
)
ax.set_title("Confusion Matrix - Iris Classifier", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.close()
print("[INFO] Confusion matrix saved to confusion_matrix.png")

# ------------------------------------------------------------------
# 7. Save the trained pipeline + label names using Joblib
# ------------------------------------------------------------------
model_data = {
    "model": pipeline,
    "target_names": target_names,
}
joblib.dump(model_data, "model.pkl")
print("[INFO] Model saved to model.pkl")
print("\nAll done! Run 'streamlit run app.py' to launch the app.")
print("=" * 55)
