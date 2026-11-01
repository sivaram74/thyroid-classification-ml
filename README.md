# Thyroid Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop machine learning classification models to predict thyroid-related conditions using patient-related attributes.

Five classification algorithms are implemented and evaluated using the same dataset. The models are compared using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

An interactive Streamlit web application is also developed to allow users to upload test data, select a machine learning model and evaluate its performance.

## 2. Dataset Description

**Dataset Name:** Thyroid Dataset

**Dataset Source:** [Kaggle – Thyroid Dataset](https://www.kaggle.com/datasets/sikandaraidev/thyroid-dataset)

**Original Dataset:**
- 9,171 records
- 22 input features
- 1 target variable
- 9 target classes

**After data cleaning:**
- 102 duplicate records removed.
- 9,069 records retained.
- 22 input features and 1 target variable.

The dataset contains numerical, categorical and Boolean features.

**Data preprocessing:**
- Missing numerical values were handled using median imputation.
- Missing categorical values were handled using most-frequent imputation.
- Categorical features were converted using One-Hot Encoding.
- Numerical features were standardized.

**Training and Testing Split:**
- Training data: 80% (7,255 records)
- Testing data: 20% (1,814 records)
- Random state: 42

## 3. GitHub Repository Link

[Thyroid Classification – GitHub Repository](https://github.com/sivaram74/thyroid-classification-ml)

The repository contains the complete source code, trained machine learning models, requirements file, test dataset and project documentation.

## 4. Models Used and Evaluation

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Evaluation Metrics

The following six evaluation metrics were calculated for all five models:

- Accuracy
- AUC Score (One-vs-Rest, weighted)
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4917 | 0.8938 | 0.8300 | 0.4917 | 0.5578 | 0.4088 |
| Decision Tree | 0.9129 | 0.9303 | 0.9227 | 0.9129 | 0.9170 | 0.8103 |
| KNN | 0.8379 | 0.8940 | 0.8342 | 0.8379 | 0.8221 | 0.5908 |
| Naive Bayes | 0.0728 | 0.6302 | 0.8048 | 0.0728 | 0.0565 | 0.1036 |
| Support Vector Machine | 0.5573 | 0.9383 | 0.8302 | 0.5573 | 0.6162 | 0.4453 |

### Observations on Model Performance

**Logistic Regression**

Achieved 49.17% accuracy and 0.8938 AUC. It demonstrated reasonable class-ranking performance but relatively weak classification accuracy.

**Decision Tree**

Achieved 91.29% accuracy, 0.9303 AUC and 0.9170 weighted F1 score. It provided the strongest overall classification performance among the five models.

**K-Nearest Neighbors**

Achieved 83.79% accuracy and 0.8221 weighted F1 score. It demonstrated good classification performance, although it performed below the Decision Tree classifier.

**Gaussian Naive Bayes**

Achieved 7.28% accuracy and 0.0565 weighted F1 score. It performed poorly on this dataset compared with the other classification algorithms.

**Support Vector Machine**

Achieved 55.73% accuracy and the highest AUC of 0.9383. It demonstrated strong class-ranking performance but relatively low classification accuracy.

### Overall Winner

The **Decision Tree Classifier** is the overall winner based on accuracy, weighted F1 score and MCC.

Its evaluation results were:

- Accuracy: 91.29%
- AUC: 0.9303
- Precision: 0.9227
- Recall: 0.9129
- F1 Score: 0.9170
- MCC: 0.8103

The Decision Tree achieved the highest accuracy, weighted F1 score and MCC among all five models.

However, minority-class performance remains a challenge, as indicated by the classification report.

## 5. Streamlit Application

The interactive web application is developed using Streamlit.

**Live Application:** [Thyroid Classification Lab – Streamlit](https://thyroid-classification-ml-2026ak05109.streamlit.app/)

### Application Features

- CSV test dataset upload.
- Selection among five trained machine learning models.
- Display of six evaluation metrics.
- Confusion matrix visualization.
- Classification report.
- Downloadable prediction results.
- CSV format instructions for users.

The application uses previously trained and saved machine learning models. It evaluates uploaded test data without retraining the models.

The uploaded test CSV must contain all 22 input features and the target column named `class`.

## 6. Project Structure

```text
thyroid-classification-ml/
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

Five machine learning classification models were successfully implemented and evaluated using the Thyroid Dataset.

Among the five models, the Decision Tree Classifier achieved the best overall classification performance, with 91.29% accuracy and a weighted F1 score of 0.9170.

An interactive Streamlit application was developed and deployed to demonstrate the trained models. It allows users to upload test datasets, select different classifiers, view evaluation metrics and confusion matrices, and download prediction results.

This project demonstrates an end-to-end machine learning workflow, covering data preprocessing, model development, evaluation, application development and cloud deployment.

**Note:** This project is intended for educational purposes and is not intended for medical diagnosis.
