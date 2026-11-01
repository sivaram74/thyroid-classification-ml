# Thyroid Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop machine learning classification models to predict thyroid-related conditions using patient-related attributes.

Five classification algorithms are implemented and evaluated using the same dataset. The models are compared using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

An interactive Streamlit web application is also developed to allow users to upload test data, select a machine learning model and evaluate its performance.

## 2. Dataset Description

***2.1. Dataset Overview***

For this project, the **Thyroid Dataset** was selected to develop and evaluate multiple machine learning classification models. The dataset was obtained from Kaggle and contains patient-related information that can be used to classify thyroid-related conditions.

The dataset is suitable for this assignment because it contains more than the required minimum of 500 instances and 12 input features. It provides a multi-class classification problem, allowing the performance of different machine learning algorithms to be compared on the same dataset.

****Dataset source:**** [Thyroid Dataset – Kaggle](https://www.kaggle.com/datasets/sikandaraidev/thyroid-dataset)

***2.2.Dataset Size and Structure***

The original dataset contains **9,171 records and 23 columns**.

| Dataset property | Description |
|---|---|
| Dataset name | Thyroid Dataset |
| Data source | Kaggle |
| Original number of records | 9,171 |
| Total number of columns | 23 |
| Number of input features | 22 |
| Target variable | `class` |
| Classification type | Multi-class classification |
| Number of target classes | 9 |

Each row represents one observation, while the columns contain information used by the machine learning models to predict the corresponding target class.

During data preparation, **102 duplicate records were removed**, resulting in a cleaned dataset containing **9,069 records and 23 columns**.

***2.3. Description of Input Features***

The dataset contains 22 input features covering demographic information, thyroid-related measurements, medical indicators, and referral information.

The features can broadly be understood in the following groups:

- ****Demographic information:**** Features such as `age` and `sex` provide basic information about the individual.
- ****Thyroid-related measurements:**** Features such as `TSH`, `T3`, `TT4`, `T4U`, and `FTI` contain laboratory measurements associated with thyroid function.
- ****Medical and treatment indicators:**** Several columns represent recorded conditions, symptoms, or treatment-related indicators. These provide additional information that may help distinguish between the target classes.
- ****Referral information:**** The `referral source` feature records the source from which the observation was referred.

The dataset includes both numerical and categorical information. Some medical indicator columns are represented as Boolean or binary values, while laboratory measurements are numerical.

These different feature types make preprocessing important before applying machine learning algorithms.

***2.4. Target Variable***

The target variable used in this project is ****`class`****.

It represents the category that the machine learning models are expected to predict. Since the target contains ****nine distinct classes****, this project is treated as a multi-class classification problem rather than a binary classification problem.

All five classification algorithms were trained using the same target variable and the same dataset so that their evaluation results could be compared consistently.

***2.5. Data Quality and Missing Values***

The original dataset contained missing values in several columns. Missing observations were particularly present in some demographic and thyroid laboratory measurement fields.

The following columns contained missing values:

- `sex`
- `TSH`
- `T3`
- `TT4`
- `T4U`
- `FTI`

A total of ****5,802 missing cells**** were identified during the initial data inspection.

Instead of deleting every row containing a missing value, an imputation approach was used to retain as much useful information as possible:

- ****Numerical features:**** Missing values were replaced using the median of the corresponding feature.
- ****Categorical features:**** Missing values were replaced using the most frequently occurring category.

Duplicate records were also removed to reduce the possibility of repeated observations affecting model training and evaluation.

***2.6. Data Preprocessing***

Before training the classification models, the dataset was prepared using a preprocessing pipeline.

The following steps were performed:

1. ****Duplicate removal:**** Duplicate rows were identified and removed, reducing the dataset from 9,171 to 9,069 records.
2. ****Feature and target separation:**** The 22 input features were separated from the target column, `class`.
3. ****Train-test split:**** The cleaned dataset was divided into training and testing subsets using an 80:20 ratio.
4. ****Missing-value imputation:**** Median imputation was applied to numerical features, while most-frequent-value imputation was applied to categorical features.
5. ****Categorical encoding:**** One-Hot Encoding was used to convert categorical variables into a numerical representation suitable for machine learning algorithms.
6. ****Feature scaling:**** Standardization was applied to numerical features so that differences in measurement scales would not disproportionately influence algorithms such as K-Nearest Neighbors, Logistic Regression, and Support Vector Machine.

The train-test split was performed with `random_state=42` to make the experiment reproducible.

| Data split | Number of records | Percentage |
|---|---:|---:|
| Training data | 7,255 | 80% |
| Testing data | 1,814 | 20% |
| **Total** | **9,069** | **100%** |

The preprocessing steps were fitted using the training data and then applied to the testing data. This helps prevent information from the test set from influencing the training process.

***2.7. Why This Dataset Was Selected***

The Thyroid Dataset was selected for the following reasons:

- It satisfies the assignment requirements for the minimum number of records and input features.
- It provides a multi-class classification problem, making it useful for comparing different classification algorithms.
- It contains a combination of numerical, categorical, and Boolean features, allowing different preprocessing techniques to be demonstrated.
- It includes missing values, providing an opportunity to implement and demonstrate data-cleaning and imputation techniques.
- It is suitable for evaluating different machine learning algorithms using common classification metrics.

***2.8. Dataset Usage in This Project***

The cleaned and preprocessed dataset was used to train the following five classification models:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine

All five models were trained and evaluated using the same training and testing split. Their performance was compared using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The trained models were saved and integrated into the Streamlit application. The application allows users to upload a test CSV file, select a classification model, and view its evaluation metrics and confusion matrix without retraining the model.

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

| ML Model Name | Observations about model performance |
|---|:---|
| Logistic Regression | Achieved 49.17% accuracy and 0.8938 AUC. It demonstrated reasonable class-ranking performance but relatively weak classification accuracy. |
| Decision Tree | Achieved 91.29% accuracy, 0.9303 AUC and 0.9170 weighted F1 score. It provided the strongest overall classification performance among the five models. |
| KNN | Achieved 83.79% accuracy and 0.8221 weighted F1 score. It demonstrated good classification performance, although it performed below the Decision Tree classifier. |
| Naive Bayes | Achieved 7.28% accuracy and 0.0565 weighted F1 score. It performed poorly on this dataset compared with the other classification algorithms. |
| Support Vector Machine | Achieved 55.73% accuracy and the highest AUC of 0.9383. It demonstrated strong class-ranking performance but relatively low classification accuracy. |
| Overall Winner | The Decision Tree Classifier is the overall winner based on accuracy, weighted F1 score and MCC. It achieved the highest accuracy, weighted F1 score and MCC among all five models. <br> •	Accuracy: 91.29% <br> •	AUC: 0.9303 <br> •	Precision: 0.9227 <br> •	Recall: 0.9129 <br> •	F1 Score: 0.9170 <br> •	MCC: 0.8103 <br> |

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
