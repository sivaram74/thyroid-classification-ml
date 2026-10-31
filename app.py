
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Thyroid Classification Lab",
    page_icon="🧪",
    layout="wide"
)

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

# ==========================================
# MODEL FILES
# ==========================================

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

# ==========================================
# LOAD METADATA
# ==========================================

@st.cache_resource
def load_metadata():

    with open(
        MODEL_DIR / "model_metadata.json",
        "r"
    ) as file:

        return json.load(file)


@st.cache_resource
def load_model(filename):

    return joblib.load(MODEL_DIR / filename)


metadata = load_metadata()

# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("🧪 Thyroid Classification Lab")

st.markdown(
    "### Machine Learning Classification Dashboard"
)

st.write(
    "Evaluate thyroid classification using "
    "five pre-trained machine learning models."
)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(MODEL_FILES.keys())
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This application uses saved machine "
    "learning models. Uploaded data is "
    "evaluated without retraining."
)

# ==========================================
# CSV UPLOAD
# ==========================================


st.subheader("Upload Test Dataset")

# CSV format guidance
with st.expander("📋 Important: Check the required CSV format before uploading", expanded=True):

    st.info(
        "Please ensure your CSV file follows the same format "
        "as the original test_data.csv."
    )

    st.markdown("""
    **Your CSV file must contain:**
    - All 22 required input features.
    - The target column named `class`.
    - Exact column names and compatible data types.
    - One record per row.
    - CSV format (.csv).
    """)

    # Get expected columns dynamically from metadata
    expected_columns = (
        metadata["feature_columns"] +
        [metadata["target_column"]]
    )

    st.markdown("**Expected column names:**")

    st.code(",".join(expected_columns), language="text")

    st.caption(
        "Tip: You can use the test_data.csv file provided "
        "with this project as a reference for the required format."
    )

uploaded_file = st.file_uploader(
    "Upload your thyroid test CSV file",
    type=["csv"],
    help="Upload a CSV containing all 22 features and the class column."
)


if uploaded_file is None:

    st.info(
        "Please upload test_data.csv to begin evaluation."
    )

    st.stop()

# ==========================================
# READ DATA
# ==========================================

try:

    data = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(f"Unable to read CSV: {error}")

    st.stop()

st.success("Dataset uploaded successfully!")

# ==========================================
# VALIDATE COLUMNS
# ==========================================

feature_columns = metadata["feature_columns"]

target_column = metadata["target_column"]

missing_columns = [
    column
    for column in feature_columns
    if column not in data.columns
]

if missing_columns:

    st.error(
        "Missing required features: "
        + ", ".join(missing_columns)
    )

    st.stop()

if target_column not in data.columns:

    st.error(
        "The uploaded CSV must contain "
        "the target column 'class'."
    )

    st.stop()

if data.empty:

    st.error("The uploaded dataset is empty.")

    st.stop()

# ==========================================
# DATA PREPARATION
# ==========================================

X_test_app = data[feature_columns].copy()

y_test_app = data[target_column].astype(str)

# ==========================================
# LOAD SELECTED MODEL
# ==========================================

model = load_model(
    MODEL_FILES[selected_model]
)

# ==========================================
# PREDICTIONS
# ==========================================

try:

    y_pred = model.predict(X_test_app)

    y_prob = model.predict_proba(X_test_app)

except Exception as error:

    st.error(f"Prediction error: {error}")

    st.stop()

# ==========================================
# EVALUATION METRICS
# ==========================================

accuracy = accuracy_score(
    y_test_app,
    y_pred
)

precision = precision_score(
    y_test_app,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test_app,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test_app,
    y_pred,
    average="weighted",
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test_app,
    y_pred
)

classes = model.named_steps["classifier"].classes_

try:

    auc = roc_auc_score(
        y_test_app,
        y_prob,
        labels=classes,
        multi_class="ovr",
        average="weighted"
    )

except ValueError:

    auc = None

# ==========================================
# DISPLAY METRICS
# ==========================================

st.divider()

st.subheader(
    f"Evaluation Results: {selected_model}"
)

st.caption(
    f"Records evaluated: {len(data)}"
)

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.4f}")

col2.metric(
    "AUC Score",
    f"{auc:.4f}" if auc is not None else "Undefined"
)

col3.metric("Precision", f"{precision:.4f}")

col4, col5, col6 = st.columns(3)

col4.metric("Recall", f"{recall:.4f}")

col5.metric("F1 Score", f"{f1:.4f}")

col6.metric("MCC Score", f"{mcc:.4f}")

# ==========================================
# CONFUSION MATRIX
# ==========================================

st.divider()

st.subheader("Confusion Matrix")

labels = sorted(
    set(y_test_app.unique()) |
    set(pd.Series(y_pred).unique())
)

cm = confusion_matrix(
    y_test_app,
    y_pred,
    labels=labels
)

fig, ax = plt.subplots(figsize=(11, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax
)

ax.set_xlabel("Predicted Class")

ax.set_ylabel("Actual Class")

plt.xticks(rotation=45, ha="right")

plt.yticks(rotation=0)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

st.divider()

st.subheader("Classification Report")

report = classification_report(
    y_test_app,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)

# ==========================================
# PREDICTIONS TABLE
# ==========================================

st.divider()

st.subheader("Prediction Results")

prediction_data = data.copy()

prediction_data["Predicted Class"] = y_pred

st.dataframe(
    prediction_data.head(100),
    use_container_width=True
)

# ==========================================
# DOWNLOAD PREDICTIONS
# ==========================================

csv_data = prediction_data.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Prediction Results",
    data=csv_data,
    file_name="thyroid_predictions.csv",
    mime="text/csv"
)

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Thyroid Classification Lab | "
    "Machine Learning Assignment 2"
)

st.warning(
    "Educational project only. "
    "This application is not intended "
    "for medical diagnosis."
)
