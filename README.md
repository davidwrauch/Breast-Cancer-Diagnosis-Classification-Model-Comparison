# Breast Cancer Diagnosis Prediction Using Machine Learning

## TLDR

Compared machine learning algorithms for predicting whether breast tumors are benign or malignant using biopsy image features from the Wisconsin Diagnostic Breast Cancer dataset.

### Key Results

- Random Forest: **97.1% accuracy**, **95.2% sensitivity**, **98.1% specificity**, **0.992 AUC**
- Logistic Regression: **94.7% accuracy**, **93.7% sensitivity**, **95.3% specificity**, **0.963 AUC**
- Random Forest outperformed Logistic Regression across all evaluation metrics and achieved near-perfect discrimination between malignant and benign tumors.

This project demonstrates how different classification approaches perform on structured biomedical data and highlights why medical models should be evaluated using sensitivity, specificity, and AUC rather than accuracy alone.

---

## Overview

Machine learning is increasingly used to support medical diagnosis by identifying patterns in clinical and imaging data that may not be obvious through manual review alone.

In this project, I analyze the Wisconsin Diagnostic Breast Cancer dataset, a widely used benchmark dataset containing measurements extracted from digitized images of breast tissue samples. The objective is to predict whether a tumor is benign or malignant based on cellular characteristics observed in biopsy images.

Rather than focusing on a single model, the project compares multiple classification approaches and evaluates their ability to correctly identify malignant tumors while minimizing false alarms.

---

## Dataset

**Source:** Wisconsin Diagnostic Breast Cancer Dataset

- 569 biopsy samples
- 30 numerical predictor variables
- Binary outcome:
  - Benign (B)
  - Malignant (M)

Features describe characteristics of cell nuclei extracted from digitized microscope images, including:

- Radius
- Perimeter
- Texture
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension

Example variables:

- `radius_mean`
- `perimeter_worst`
- `concavity_worst`
- `concave.points_worst`

---

## Modeling Workflow

The analysis follows a standard supervised machine learning workflow:

1. Data preparation and validation
2. Removal of identifier fields
3. Stratified 70/30 train-test split
4. Feature scaling where appropriate
5. Model training
6. Out-of-sample evaluation
7. Comparison of diagnostic performance

### Models Evaluated

- Logistic Regression
- Random Forest

Additional experiments were conducted using XGBoost, K-Nearest Neighbors, and Neural Networks, though the strongest performance was achieved using Random Forest.

---

## Results

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---------|---------|---------|---------|---------|
| Logistic Regression | 94.7% | 93.7% | 95.3% | 0.963 |
| Random Forest | 97.1% | 95.2% | 98.1% | 0.992 |

### Best Performing Model

**Random Forest** achieved the strongest overall performance:

- 97.1% accuracy
- 95.2% sensitivity
- 98.1% specificity
- 0.992 AUC

An AUC of 0.992 indicates that the model almost always ranks malignant tumors above benign tumors regardless of the classification threshold.

---

## Why Sensitivity and Specificity Matter

Medical classification problems require more than accuracy.

### Sensitivity (True Positive Rate)

Measures the ability to correctly identify malignant tumors.

High sensitivity reduces the risk of missing cancer cases.

### Specificity (True Negative Rate)

Measures the ability to correctly identify benign tumors.

High specificity reduces unnecessary follow-up testing and patient anxiety.

### AUC (Area Under the ROC Curve)

Measures how well the model ranks malignant cases above benign cases across all possible classification thresholds.

A higher AUC indicates stronger overall discrimination.

---

## Feature Importance

Random Forest feature importance analysis identified several measurements related to tumor size and boundary irregularity as the strongest predictors of malignancy.

Common high-importance variables included:

- Radius
- Perimeter
- Area
- Concavity
- Concave Points

These findings are consistent with established clinical indicators used in breast cancer diagnosis.

---

## Key Takeaways

- Both Logistic Regression and Random Forest achieved strong diagnostic performance.
- Random Forest outperformed Logistic Regression across all evaluation metrics.
- Ensemble tree-based methods proved highly effective for structured biomedical data.
- Sensitivity, specificity, and AUC provide a more complete evaluation framework than accuracy alone.
- Feature importance analysis helped identify the tumor characteristics most associated with malignancy.

---

## Tools Used

- R
- tidyverse
- caret
- randomForest
- pROC
- xgboost
- keras
- tensorflow

---

## Repository Contents

- `breast_cancer_classification.R` — data preparation, model training, and evaluation
- `README.md` — project documentation

---

## Skills Demonstrated

- Classification Modeling
- Model Evaluation
- Random Forest
- Logistic Regression
- Feature Importance Analysis
- Sensitivity / Specificity Analysis
- ROC Curves and AUC
- Predictive Analytics
- Healthcare Analytics
- R Programming
