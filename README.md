# Essay Vocabulary Scoring
Machine learning pipeline for predicting essay vocabulary scores using TF-IDF, linguistic feature engineering, and LightGBM, with a focus on ordinal evaluation (Quadratic Weighted Kappa).

-------
## Overview

This project builds an end-to-end machine learning pipeline to predict **vocabulary quality of student essays**, based on two human raters (`Vocabulary_1`, `Vocabulary_2`).

The task is modeled as an ordinal regression problem, where:

Numerical accuracy is important  
Ranking quality between essays is critical  

--------

## Problem Definition

Each essay receives two human scores (0–5).

define:

`target_mean` – average of both raters (continuous target)  
`target_bin` – rounded score (ordinal target)  

This allows evaluation from both:
Regression perspective  
Ordinal agreement perspective  

-----

## Dataset Insights

### Score Distribution

Most essays are concentrated around scores **3–4**
Few extreme scores (0 or 5)
# Note: Although the task description defines the score range as 1–5, the dataset includes values from 0–5.  
# All modeling and evaluation were performed using the observed data range.

**Implication:**
Mild class imbalance  
Bias toward mid-range predictions  

------

## Rater Agreement

High agreement overall  
Most disagreements are within **±1 score**

**Implication:**
The dataset contains **label noise**
Performance is inherently bounded by human disagreement  

-------

## Evaluation Metrics

All models are evaluated using **5-fold cross-validation**.

### Primary Metric
**Quadratic Weighted Kappa (QWK)** – measures ordinal agreement  

### Secondary Metrics
- MAE (Mean Absolute Error)  
- RMSE  
- R²  
- Accuracy (rounded predictions)  

### Additional possible Evaluation
- Performance vs each individual rater  

------

## Modeling Strategy

The pipeline is built incrementally across three steps:

-------

# Step 3 — Baseline Model (TF-IDF + Ridge)

## Approach

- Word-level TF-IDF
- Ridge regression

--------

## Initial Cross-Validation Results
- MAE_mean: 0.3911 +- 0.0058 
- RMSE_mean: 0.5005 +- 0.0083  
- R2_mean: 0.3386 +- 0.0088  
- QWK_bin: 0.3426 +- 0.0139  
- ACC_bin: 0.4961 +- 0.0107

----------

## Parameter Tuning

## Best Configuration

- alpha = 0.5  
- ngram_range = (1,2)  
- max_features = 50000  
- min_df = 3  
- sublinear_tf = True  

---

## Final Step 3 Performance (CV)

- MAE ≈ 0.387  
- RMSE ≈ 0.494  
- R² ≈ 0.355  
- QWK ≈ 0.378  
- Accuracy ≈ 0.512

-------

## Insight

TF-IDF already captures a strong lexical signal  
The linear model performs surprisingly well  

-------

# Step 4 — Feature-Enriched Model

## Key Idea

Improve representation, not just the model.

-------

## Features Used

## Text Features
- Word TF-IDF (1–3 grams)
- Character TF-IDF (2–5 grams)

## Handcrafted Features

- Word count  
- Unique words  
- Type-token ratio  
- Hapax ratio  
- Average word length  
- Long word ratio  
- Sentence count  
- Average sentence length  

------

## Feature Statistics (Example)

- Mean word count: 437
- Mean unique words: 151
- Mean type-token ratio: 0.37
- Mean sentence length: ~30 words

---

## Cross-Validation Results

- MAE_mean: 0.3717 +- 0.0035  
- RMSE_mean: 0.4749 +- 0.0061  
- R2_mean: 0.4045 +- 0.0091  
- QWK_bin: 0.4109 +- 0.0080  
- ACC_bin: 0.5253 +- 0.0112

---

## Parameter Tuning
## Best Configuration

alpha = 0.5  
word_ngram = (1,3)  
char_ngram = (2,5)  
min_df = 2  

--------

## Insight

Largest improvement in the entire pipeline  
Character features significantly boost performance  
Feature engineering > model change  

--------

# Step 5 - LightGBM (Non-linear Model)

## Motivation

Capture:
Feature interactions  
Non-linear relationships  

------

## Tuning Strategy

Used targeted parameter search instead of full grid search:

**Reason:**
High-dimensional TF-IDF  
Computational constraints  
Diminishing returns  

## Best Configuration

- n_estimators = 300  
- learning_rate = 0.05  
- num_leaves = 63  
- min_child_samples = 30  
- subsample = 0.9  
- colsample_bytree = 0.9  

-------

## Cross-Validation Results

MAE_mean: 0.3743  
RMSE_mean: 0.4757  
R2_mean: 0.4024  
QWK_bin: 0.4130  
ACC_bin: 0.5234

---

## Insight

- Slight improvement over Step 4  
- Does not significantly outperform Ridge  
- Suggests features already capture most signal  

---

# Final Comparison (Full Data – For Inspection Only)

These results are optimistic (trained on full data)

| Model | MAE | RMSE | R² | QWK | ACC |

| Step 3 | 0.1686 | 0.2190 | 0.8735 | 0.6316 | 0.6681 |
| Step 4 | 0.1722 | 0.2197 | 0.8727 | 0.6547 | 0.6809 |
| Step 5 | 0.0868 | 0.1109 | 0.9675 | 0.6870 | 0.7015 |

-------

#  Key Insights

### 1. Feature Engineering Dominates
Biggest improvement from Step 3 → Step 4

---

### 2. Linear Models Are Strong
Ridge performs very well on sparse features

---

### 3. Character Features Are Critical
Capture subword patterns and improve ranking

---

### 4. Label Noise Limits Performance
Human disagreement sets the performance ceiling

---

### 5. LightGBM Adds Limited Gain
Non-linearity helps slightly, but not dramatically

---
## Visualization – Model Fit

A True vs Predicted scatter plot was generated for the final LightGBM model, with a diagonal line representing perfect predictions.

Predictions generally follow the diagonal, indicating a good overall fit. Performance is strongest in the mid-range scores (3–4), while extreme scores show more variability.

The model exhibits slight compression toward the center (overpredicting low scores and underpredicting high scores), likely due to class imbalance and label noise.


### Results Discussion

Overall, the results can be considered moderate to good. The model achieves reasonable ranking performance (QWK ~0.41 in cross-validation), with most errors being small (often within ±1 score),
which is acceptable given the inherent noise in human scoring.
The main limitation is performance on extreme scores, where predictions tend to be compressed toward the center.
This is likely due to class imbalance and disagreement between raters.

## Possible Improvements

Use ordinal modeling instead of regression + rounding  
Apply threshold optimization to improve QWK  
Incorporate pretrained embeddings (BERT) for richer semantic understanding  
Improve handling of rare scores (weighting or resampling)  

Overall, the results show that the model captures meaningful patterns,
but there is still room for improvement, especially in handling edge cases and better modeling the ordinal nature of the task.

## Reproducibility

- random_state = 42  
- 5-fold cross-validation  
- consistent evaluation pipeline  

---
## Configurations that were not selected as best-performing have been commented out to reduce runtime in subsequent runs.  
## To explore additional configurations and their results, please uncomment the relevant sections in the code.

## How to Run

### 1. Install dependencies
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn

file_path = path_to_your_dataset.xlsx
