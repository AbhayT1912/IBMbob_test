# Iris Flower Classification — Project Plan

## Top-Level Overview

Build a beginner-friendly, college-level ML mini-project that classifies Iris flower species
(Setosa, Versicolor, Virginica) from four numeric measurements. The project has two parts:

1. **Training pipeline** (`train.py`) — loads the built-in Scikit-learn Iris dataset, trains a
   Logistic Regression classifier inside a Scikit-learn Pipeline (with StandardScaler), evaluates
   it, and saves the complete pipeline to disk using Joblib.
2. **Streamlit app** (`app.py`) — loads the saved pipeline and presents two sections:
   - **Prediction** — user enters four measurements via sidebar sliders and gets a predicted
     species with prediction probabilities.
   - **Model Performance** — displays accuracy, confusion matrix, and classification report.

All code lives inside `Iris-Flower-Classifier/`. No external datasets are needed.

## Approved Design Decisions

- **Model:** `LogisticRegression` wrapped in a `Pipeline` with `StandardScaler` for feature scaling.
  `RandomForestClassifier` is noted in README as a suggested future experiment only.
- **Input UI:** Streamlit sidebar sliders with dataset-appropriate ranges and step sizes.
- **File structure:** `requirements.txt`, `train.py`, `app.py`, `README.md`, `.gitignore`.
  No `utils.py` — shared logic is minimal enough to stay inline.

---

## Sub-Tasks

---

### Sub-Task 1 — Project Scaffolding

**Status:** `[ ] pending`

**Intent**
Create the folder structure, dependency file, and README so the project is immediately
runnable by anyone who clones it.

**Expected Outcomes**
- `requirements.txt` lists all needed packages with no extras.
- `README.md` explains what the project does, how to install dependencies, how to train the
  model, and how to launch the app — in plain language suitable for a beginner.
- The folder layout matches the agreed structure.

**Todo List**
1. Create `requirements.txt` with: `scikit-learn`, `pandas`, `matplotlib`, `joblib`, `streamlit`.
2. Create `.gitignore` ignoring `__pycache__/`, `*.pyc`, `model.pkl`, `confusion_matrix.png`, `.venv/`, `*.egg-info/`.
3. Create `README.md` with:
   - Project description (one paragraph).
   - Prerequisites (Python 3.8+).
   - Installation step: `pip install -r requirements.txt`.
   - Training step: `python train.py`.
   - App launch step: `streamlit run app.py`.
   - Brief description of what each file does.
   - A "Future Experiments" section suggesting swapping in `RandomForestClassifier`.

**Relevant Context**
- Target audience: college students, beginners. Keep instructions simple and literal.
- No version pinning is required — latest stable versions are fine.

---

### Sub-Task 2 — Training Script (`train.py`)

**Status:** `[ ] pending`

**Intent**
Write a standalone script that handles the entire ML pipeline: data loading, splitting,
training, evaluation, and model saving. Running this script produces `model.pkl` and prints
all evaluation metrics to the console.

**Expected Outcomes**
- `model.pkl` is created in the project root after running `python train.py`.
- Console output shows: train/test split sizes, accuracy score, classification report,
  and a saved-model confirmation message.
- A confusion matrix plot is displayed (non-blocking) and optionally saved as `confusion_matrix.png`.

**Todo List**
1. Load the Iris dataset using `sklearn.datasets.load_iris()`.
2. Convert to a `pandas.DataFrame` for clarity and beginner readability; print the first 5 rows.
3. Split into train/test sets (80/20, `random_state=42`).
4. Build a `Pipeline`: step 1 = `StandardScaler()`, step 2 = `LogisticRegression(max_iter=200, random_state=42)`.
5. Fit the pipeline on the training set.
6. Predict on the test set.
7. Print accuracy using `accuracy_score`.
8. Print classification report using `classification_report` with `target_names`.
9. Plot and save the confusion matrix using `ConfusionMatrixDisplay` from Scikit-learn
   and Matplotlib — save as `confusion_matrix.png`.
10. Save the entire pipeline AND the label names list to `model.pkl` using `joblib.dump`
    (save as a dict: `{"model": pipeline, "target_names": target_names}`).
11. Print a confirmation line: `Model saved to model.pkl`.

**Relevant Context**
- The full Pipeline (scaler + classifier) is saved so `app.py` does not need to re-apply scaling.
- Saving `target_names` alongside the model avoids hardcoding species names in `app.py`.
- `random_state=42` ensures reproducible results — important for a demo project.
- `ConfusionMatrixDisplay` is the modern Scikit-learn approach (no manual seaborn heatmap needed).
- `max_iter=200` prevents convergence warnings on the Iris dataset.

---

### Sub-Task 3 — Streamlit App (`app.py`)

**Status:** `[ ] pending`

**Intent**
Build the user-facing Streamlit interface. The app loads `model.pkl` and provides two
clearly separated sections: a prediction form and a model performance panel.

**Expected Outcomes**
- Running `streamlit run app.py` opens a browser page with:
  - A sidebar or main-area form with four numeric sliders (one per measurement).
  - A "Predict" button that shows the predicted species and a bar chart of probabilities.
  - An expandable or tabbed "Model Performance" section showing accuracy, classification
    report (as a table), and the confusion matrix image.
- The app gracefully shows an error message if `model.pkl` is not found (i.e., train first).

**Todo List**
1. Load `model.pkl` with `joblib.load` at startup; wrap in a `try/except` to show a clear
   `st.error` message if the file is missing (instruct user to run `train.py` first).
2. Add a page title and short description using `st.title` and `st.write`.
3. Create a sidebar section "Enter Flower Measurements" with four `st.slider` widgets:
   - Sepal Length (cm): 4.0 – 8.0, step 0.1, default 5.4
   - Sepal Width (cm): 2.0 – 4.5, step 0.1, default 3.4
   - Petal Length (cm): 1.0 – 7.0, step 0.1, default 4.7
   - Petal Width (cm): 0.1 – 2.5, step 0.1, default 1.5
   - Display the four selected values in a summary table below the sliders.
4. Add a "Predict" button in the sidebar; on click:
   - Run `pipeline.predict()` and `pipeline.predict_proba()` on the four values (no manual scaling needed).
   - Display the predicted species name in a `st.success` box.
   - Show a Matplotlib horizontal bar chart of prediction probabilities (one bar per species).
5. Add a "Model Performance" `st.expander` section below the prediction area:
   - Re-load the Iris dataset and run the same 80/20 split (`random_state=42`).
   - Compute and display accuracy as `st.metric`.
   - Display the classification report as a `st.dataframe` (convert via `pandas`).
   - Display the saved `confusion_matrix.png` using `st.image` (show a note if file is missing).

**Relevant Context**
- Slider defaults are the approximate Iris dataset feature means for a realistic starting state.
- The saved Pipeline handles scaling internally — `app.py` passes raw values directly.
- Reproducible split (`random_state=42`) ensures the metrics in the app match `train.py` output.
- Using `st.expander` for Model Performance keeps the UI clean and beginner-friendly.
- `confusion_matrix.png` is generated by `train.py` — the app just displays it as a static image.

---

## File Map

| File | Sub-Task | Purpose |
|---|---|---|
| `requirements.txt` | 1 | Dependency list |
| `.gitignore` | 1 | Excludes generated and env files from git |
| `README.md` | 1 | Setup and usage instructions |
| `train.py` | 2 | Training pipeline, evaluation, model saving |
| `app.py` | 3 | Streamlit prediction + metrics UI |
| `model.pkl` | Generated by 2 | Serialised Pipeline + label names |
| `confusion_matrix.png` | Generated by 2 | Confusion matrix image for the app |

---

## Execution Order

1. Run `pip install -r requirements.txt`
2. Run `python train.py` → produces `model.pkl` and `confusion_matrix.png`
3. Run `streamlit run app.py` → opens the browser UI
