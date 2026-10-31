
# Thyroid Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop machine learning
classification models to predict thyroid-related conditions
using patient-related attributes.

Five classification algorithms are implemented and evaluated
using the same dataset.

The models are compared using Accuracy, AUC, Precision,
Recall, F1 Score and Matthews Correlation Coefficient (MCC).

An interactive Streamlit application is developed to allow
users to upload test data and evaluate the trained models.

## 2. Dataset Description

Dataset Name: Thyroid Dataset

Dataset Source:
https://www.kaggle.com/datasets/sikandaraidev/thyroid-dataset

Original Dataset:
- 9,171 records
- 22 input features
- 1 target variable
- 9 target classes

After removing 102 duplicate records:
- 9,069 records remained.

The dataset contains numerical, categorical and Boolean
features.

Missing numerical values were handled using median
imputation, while categorical missing values were handled
using most-frequent imputation.

Categorical features were converted using One-Hot Encoding.
Numerical features were standardized.

The dataset was divided into:
- 80% Training data
- 20% Testing data

Random state: 42

## 3. GitHub Repository Link

To be updated after repository creation.

## 4. Models Used and Evaluation

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

### Evaluation Metrics

- Accuracy
- AUC Score (One-vs-Rest, weighted)
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- Matthews Correlation Coefficient

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4917 | 0.8938 | 0.8300 | 0.4917 | 0.5578 | 0.4088 |
| Decision Tree | 0.9129 | 0.9303 | 0.9227 | 0.9129 | 0.9170 | 0.8103 |
| KNN | 0.8379 | 0.8940 | 0.8342 | 0.8379 | 0.8221 | 0.5908 |
| Naive Bayes | 0.0728 | 0.6302 | 0.8048 | 0.0728 | 0.0565 | 0.1036 |
| Support Vector Machine | 0.5573 | 0.9383 | 0.8302 | 0.5573 | 0.6162 | 0.4453 |

### Observations on Model Performance

**Logistic Regression**

Achieved 49.17% accuracy and 0.8938 AUC.
It demonstrated reasonable class-ranking performance
but relatively weak classification accuracy.

**Decision Tree**

Achieved 91.29% accuracy, 0.9303 AUC and 0.9170 weighted
F1 score. It provided the strongest overall performance
among the five models.

**K-Nearest Neighbors**

Achieved 83.79% accuracy and 0.8221 weighted F1 score.
It demonstrated good classification performance.

**Gaussian Naive Bayes**

Achieved 7.28% accuracy and 0.0565 weighted F1 score.
It performed poorly on this dataset.

**Support Vector Machine**

Achieved 55.73% accuracy and the highest AUC of 0.9383.
It demonstrated strong class-ranking performance but
relatively low classification accuracy.

### Overall Winner

The Decision Tree Classifier is the overall winner
based on accuracy, weighted F1 score and MCC.

It achieved:
- Accuracy: 91.29%
- AUC: 0.9303
- Precision: 0.9227
- Recall: 0.9129
- F1 Score: 0.9170
- MCC: 0.8103

However, minority-class performance remains a challenge,
as shown by the classification report.

## 5. Streamlit Application

The application is developed using Streamlit.

Features include:

- CSV test dataset upload.
- Selection among five trained ML models.
- Display of six evaluation metrics.
- Confusion matrix visualization.
- Classification report.
- Downloadable prediction results.

The application uses previously trained models and
does not retrain them using uploaded test data.

Live Streamlit Application Link:
To be updated after deployment.

## 6. Project Structure

```text
Thyroid_Classification_Project/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model_comparison_results.csv
|
|-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- svm.joblib
    |-- model_metadata.json
```
## 7. Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 8. Conclusion
Five machine learning classification models were
implemented and evaluated on the thyroid dataset.

The Decision Tree Classifier achieved the best overall
results in this experiment.

The Streamlit application provides an interactive
interface for evaluating the saved classification
models on uploaded test datasets.

This project is intended for educational purposes
and not for medical diagnosis.
