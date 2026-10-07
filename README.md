# Credit Card Fraud Detection

A machine learning project where I try to spot fraudulent credit card transactions. Fraud is rare (well under 1% of the data), so the interesting part was figuring out how to judge a model fairly when "just guess legitimate every time" already scores 99.8% accuracy.

## What this project does

1. Loads and inspects a dataset of about 285,000 card transactions
2. Checks for missing values and duplicates, and explores the class imbalance
3. Trains four classification models on 80% of the data
4. Tests them on the remaining 20% and compares precision, recall and F1-score
5. Builds a SMOTE-balanced version of the training data for future experiments

## Repository structure

```
Credit-Card-Fraud-Detection/
├── Credit_Card_Fraud.ipynb   # Full code, EDA and model comparison
└── README.md
```

## Dataset

I used the well-known [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (European cardholders, September 2013).

- **284,807** transactions in total
- **492** are fraud, which is only about **0.17%** of everything
- 30 input features: `Time`, `Amount`, and `V1` to `V28` (anonymized, PCA-transformed for privacy)
- Target column `Class`: `0` = legitimate, `1` = fraud
- No missing values, but **1,081 duplicate rows** showed up in the check

## Exploratory analysis

- Confirmed there are no null values in any column
- Plotted a correlation heatmap of all features
- Looked at how transactions are spread over time
- Plotted the fraud vs non-fraud counts, which makes the imbalance obvious

## Models I tried

- Decision Tree
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (linear SVM)

I split the data 80/20 (`random_state=42`), which gave 227,845 rows for training and 56,962 for testing. The test set contains 98 fraud cases, so every fraud the model misses really counts.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree | 0.9991 | 0.70 | 0.79 | 0.74 |
| Logistic Regression | 0.9986 | 0.61 | 0.56 | 0.59 |
| SVM (linear) | 0.9985 | 0.60 | 0.30 | 0.40 |
| KNN | 0.9984 | 1.00 | 0.05 | 0.10 |

**Best model: Decision Tree.** It caught 77 of the 98 fraud cases in the test set (recall of about 79%) and had the best F1-score.

A few things worth noticing:

- **Accuracy is misleading here.** Every model scores around 99.8% or higher, even KNN, which only caught 5 of 98 frauds.
- **KNN looks perfect on precision but fails in practice.** Its precision is 1.0 because it almost never flags anything, so it missed 93 fraud cases out of 98.
- **The SVM and Logistic Regression missed a lot of fraud too**, which is why recall and F1 are the numbers to watch for this problem.

## SMOTE

I also used SMOTE to balance the training data (394 fraud vs 227,451 legitimate rows became 227,451 of each). I haven't trained the models on it yet, so all results above come from the original imbalanced data. That's my next step.

## Tools used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- imbalanced-learn (SMOTE)
- Google Colab

## How to run it

1. Clone the repo:
   ```
   git clone https://github.com/Pramit33/Credit-Card-Fraud-Detection.git
   ```
2. Install the libraries:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
   ```
3. Download `creditcard.csv` from the Kaggle link above and put it in the project folder
4. In the notebook, change the file path in the data-loading cell (it currently points to a Google Drive folder)
5. Run all cells

## What I learned

The biggest lesson was that accuracy can look amazing while the model is basically useless. Looking at precision, recall and the confusion matrix told a very different story. I also saw that a simple Decision Tree beat the other models without any tuning.

## What I'd do next

- Train the models on the SMOTE-balanced data and compare
- Add Random Forest and add ROC-AUC / precision-recall curves
- Scale the features and tune hyperparameters
- Try class weights as another way to handle the imbalance

## Author

**Pramit De**

- LinkedIn: *https://www.linkedin.com/in/pramitde28/*
- Email: *depramit28@gmail.com*
