# 🌸 Iris Flower Classification

A beginner-friendly machine learning project that predicts the species of an Iris flower
based on four measurements: sepal length, sepal width, petal length, and petal width.

Built with Python, Scikit-learn, Pandas, Matplotlib, Joblib, and Streamlit.

---

## Project Structure

```
Iris-Flower-Classifier/
├── train.py              # Trains the model, evaluates it, and saves it to disk
├── app.py                # Streamlit web app for predictions and model metrics
├── requirements.txt      # Python package dependencies
├── README.md             # This file
└── .gitignore            # Files excluded from git
```

> `model.pkl` and `confusion_matrix.png` are generated when you run `train.py`.

---

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

---

## Setup Instructions

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd Iris-Flower-Classifier
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on macOS / Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Train the model

```bash
python train.py
```

This will:
- Load the built-in Iris dataset
- Train a Logistic Regression model with feature scaling
- Print accuracy, classification report, and save a confusion matrix image
- Save the trained model to `model.pkl`

### Step 2 — Launch the web app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

---

## How the App Works

1. Use the **sidebar sliders** to enter the four flower measurements.
2. Click the **Predict** button to see the predicted species and prediction probabilities.
3. Expand the **Model Performance** section to view accuracy, classification report, and the confusion matrix.

---

## Dataset

The [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) is a classic dataset
built into Scikit-learn. It contains 150 samples across three species:

| Species | Samples |
|---|---|
| Iris Setosa | 50 |
| Iris Versicolor | 50 |
| Iris Virginica | 50 |

---

## Model

| Component | Detail |
|---|---|
| Preprocessing | `StandardScaler` — normalises feature values |
| Classifier | `LogisticRegression` (max_iter=200, random_state=42) |
| Pipeline | Scikit-learn `Pipeline` combining scaler + classifier |
| Train / Test Split | 80% training, 20% testing (random_state=42) |

---

## Future Experiments

Once you understand the basics, try swapping the model:

```python
# In train.py, replace LogisticRegression with:
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
```

Other ideas:
- Try `KNeighborsClassifier` or `SVC`
- Experiment with different train/test split ratios
- Add cross-validation using `cross_val_score`
- Visualise feature importance (works with RandomForest)

---

## License

This project is open-source and free to use for educational purposes.
