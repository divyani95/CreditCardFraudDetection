# CreditCardFraudDetection

This project implements a machine learning pipeline to detect fraudulent credit card transactions. It includes data preprocessing, model training, evaluation, and a user-friendly GUI for fraud detection using Python's Tkinter library.

---

## Features

- **Preprocessing**:
  - Handles class imbalance with SMOTE (Synthetic Minority Oversampling Technique).
  - Standardizes transaction amounts using `StandardScaler`.
  - Removes duplicates for cleaner data.

- **Machine Learning Models**:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier

- **GUI Application**:
  - User-friendly interface for making predictions on transactions.
  - Built with Tkinter.

---

## Dataset

The project uses the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains transactions made by European cardholders in September 2013.

- **Features**:
  - `V1, V2, ..., V28`: Principal components from PCA.
  - `Amount`: Transaction amount.
  - `Class`: Target variable (0 = Normal, 1 = Fraudulent).

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/divyani95/CreditCardFraudDetection.git
cd CreditCardFraudDetection

