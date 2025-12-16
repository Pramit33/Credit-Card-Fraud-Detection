# Credit-Card-Fraud-Detection
A machine learning project that detects fraudulent credit card transactions using supervised learning techniques. This repository contains a Jupyter notebook (Credit_Card_Fraud.ipynb) that walks through data preparation, exploratory analysis, model training, and evaluation.

# Project Overview

Financial institutions face large losses due to fraudulent card transactions. This project provides a reproducible pipeline to:

**1.Explore and preprocess a credit card transactions dataset.**

**2.Train several classification models to separate legitimate transactions from fraud.**

**3.Evaluate models using metrics appropriate for imbalanced data (precision, recall, F1-score, AUC-ROC).**

**4.Compare results and recommend the best-performing approach.**

The notebook is designed for learning, experimentation, and as a template you can extend with your own data or feature engineering.

---

## 📁 Repository Structure

```
Credit-Card-Fraud-Detection/
├─ Credit_Card_Fraud.ipynb   # Main notebook with code, EDA and models
├─ README.md                 # This file
```

---

## 📊 Dataset Overview

The dataset contains credit card transaction records

Features are numerical and anonymized for privacy

Target column:

0 → Legitimate transaction

1 → Fraudulent transaction

Highly imbalanced dataset, requiring special evaluation metrics

## 🔍 Exploratory Data Analysis (EDA)

Checked missing values and data distribution

Analyzed class imbalance between fraud and non-fraud transactions

Visualized transaction patterns using plots

## 🤖 Machine Learning Models Used

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Each model was trained and tested to identify fraudulent transactions effectively.

## 📈 Model Evaluation Metrics

Due to data imbalance, standard accuracy is not sufficient. The following metrics were used:

Precision

Recall

F1-Score

ROC-AUC Score

These metrics help evaluate how well the model detects fraud while minimizing false alarms.

👤 Author

Pramit De
