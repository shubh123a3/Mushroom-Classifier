

# Mushroom Classifier


https://github.com/user-attachments/assets/bceff8f0-b103-4444-a477-024f58c142da


This project is a binary classification problem where the goal is to classify mushrooms as either edible or poisonous based on their physical characteristics. The dataset used contains 54035 rows and multiple features describing the mushrooms. The project involves data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Steps](#project-steps)
  - [1. Data Loading](#1-data-loading)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
  - [4. Data Preprocessing](#4-data-preprocessing)
  - [5. Model Training](#5-model-training)
  - [6. Model Evaluation](#6-model-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## Overview

This project aims to build a machine learning model to classify mushrooms as edible or poisonous. The dataset includes various features like cap shape, cap color, gill size, etc. By applying data preprocessing techniques and machine learning algorithms, we can predict whether a mushroom is safe to eat or not.

## Dataset

The dataset contains 54035 rows and multiple features describing the physical characteristics of mushrooms. Each row corresponds to a mushroom, and the target variable indicates whether it is edible or poisonous.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/shubh123a3/Mushroom-Classifier.git
    cd Mushroom-Classifier
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook:
    ```bash
    jupyter notebook
    ```

## Project Steps

### 1. Data Loading

The dataset is loaded into a pandas DataFrame. This step involves reading the CSV file and inspecting the first few rows to understand the structure of the data.

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('mushrooms.csv')
df.head()
```

### 2. Data Cleaning

This step involves handling missing values, correcting data types, and removing any duplicate records. Ensuring the data is clean is crucial for building an effective model.

```python
# Check for missing values
df.isnull().sum()

# Drop duplicates if any
df.drop_duplicates(inplace=True)
```

### 3. Exploratory Data Analysis (EDA)

EDA is performed to understand the distribution of the data, relationships between features, and identifying any patterns. Visualizations such as histograms, box plots, and correlation matrices are used.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Visualize the distribution of the target variable
sns.countplot(x='class', data=df)
plt.show()

# Example: Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

### 4. Data Preprocessing

Data preprocessing involves encoding categorical variables, scaling numerical features, and splitting the data into training and testing sets. 

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode categorical features
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Split the data into training and testing sets
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. Model Training

Several machine learning models are trained and evaluated, including Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting. Cross-validation is used to select the best model.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
```

### 6. Model Evaluation

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Confusion matrices and ROC curves are also used to assess model performance.

```python
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Results

The Random Forest model achieved an accuracy of 0.9905616% on the test set. Other metrics such as precision is 99% indicate that the model performs well in distinguishing between edible and poisonous mushrooms.

## Conclusion

The Mushroom Classifier project successfully built a machine learning model to classify mushrooms based on their physical characteristics. The Random Forest model was the best performer among the tested algorithms. Proper data preprocessing and feature engineering played a crucial role in achieving good model performance.

## Future Work

- Experiment with other machine learning algorithms such as XGBoost and SVM.
- Perform hyperparameter tuning to further improve model performance.
- Explore feature importance and reduce dimensionality if necessary.
- Deploy the model using a web application framework like Streamlit or Flask.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- [Scikit-learn](https://scikit-learn.org/) for the machine learning tools.
- [Pandas](https://pandas.pydata.org/) and [Seaborn](https://seaborn.pydata.org/) for data manipulation and visualization.

