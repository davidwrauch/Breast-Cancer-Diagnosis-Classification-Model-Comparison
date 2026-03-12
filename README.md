Breast Cancer Diagnosis Prediction Using Classification Algorithms

TLDR

This project compares machine learning models for predicting whether breast tumors are benign or malignant using biopsy image features. The analysis demonstrates how different classification algorithms perform on the same medical prediction task and highlights the importance of evaluating models using metrics such as sensitivity and specificity, not just accuracy.

Overview

Machine learning is increasingly used to assist medical diagnostics by identifying patterns in clinical or imaging data that help predict disease outcomes.

In this project I analyze the Wisconsin Diagnostic Breast Cancer dataset, a widely used benchmark dataset containing biopsy measurements extracted from digitized images of breast mass tissue samples. The dataset contains 569 samples and 30 numerical features describing characteristics of cell nuclei.

The objective is to train classification models that predict whether a tumor is benign or malignant.

Beyond simply fitting a model, the goal is to compare several algorithms and examine how modeling decisions affect diagnostic performance.

Dataset

Source: Wisconsin Diagnostic Breast Cancer dataset

569 biopsy samples
30 numerical features describing cell nuclei properties
Binary outcome: malignant vs benign

The features represent measurements such as cell radius, perimeter, smoothness, concavity, and symmetry derived from microscopic imaging of tissue samples.

Modeling workflow

The project follows a standard applied machine learning workflow:

data exploration and feature inspection
train/test dataset split
model training using multiple algorithms
model evaluation on unseen data

Performance is evaluated using several metrics, including:

accuracy
sensitivity (true positive rate)
specificity (true negative rate)

In a medical context these metrics have different implications. High sensitivity helps avoid missing malignant cases, while high specificity reduces unnecessary follow-up procedures.

Key takeaway

Even with a relatively small dataset, several machine learning algorithms can achieve strong predictive performance when trained on meaningful biomedical features.

More importantly, comparing models and evaluating error tradeoffs provides better insight than relying on a single algorithm.

Tools used

R
tidyverse
caret
classification algorithms implemented in R

Repository contents

breast cancer classification.R – code used to train and evaluate the models
