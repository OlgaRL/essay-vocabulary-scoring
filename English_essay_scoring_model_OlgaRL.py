# -*- coding: utf-8 -*-
"""
Vocabulary Scoring Modeling Pipeline


Author: Olga Rapp
Date: March 2026

Overview:

This project builds a machine learning pipeline to predict vocabulary scores
for student essays, based on two human raters (Vocabulary_1 and Vocabulary_2).

The task is treated as an ordinal regression problem, where both numerical
accuracy (MAE) and ranking quality (QWK - Quadratic Weighted Kappa) are important.

Pipeline Structure

Step 1 Data Exploration and Preparation:
a)Load dataset and inspect structure
b)Handle missing values (e.g. generate missing text_id values)
c)Analyze score distributions
d)Evaluate agreement between raters
e)Define targets:
 target_mean (average of raters)
 target_bin (rounded score)

Step 2 Evaluation Framework:
a)Define a unified evaluation function including:
   MAE (Mean Absolute Error)
   RMSE
   R²
   QWK (Quadratic Weighted Kappa)
   Accuracy
b)Evaluate both against:
    mean target
    individual raters

Step 3 Baseline Model (TF-IDF + Ridge)
a)Build a simple word-level TF-IDF model
b)Evaluate using cross-validation
c)Perform targeted parameter tuning
d)Establish a strong, interpretable baseline

Step 4 Feature-Enriched Model
a)Extend representation with:
  word-level TF-IDF (1–3 grams)
  character-level TF-IDF (2–5 grams)
  handcrafted linguistic features:
  lexical richness (type-token ratio, hapax)
  word sophistication (word length)
  sentence structure
b)Evaluate with cross-validation
c)Tune parameters in a focused manner
d)Show that feature engineering improves ranking (QWK)

Step 5 – LightGBM Model (Non-linear)
a)Replace Ridge with LightGBM while keeping Step 4 features
b)Capture non-linear relationships and feature interactions
c)Perform targeted parameter tuning (not full grid search) for efficiency
d)Compare performance against previous steps

Model Selection Strategy:
 Primary metric: QWK (ranking quality)
 Secondary metric: MAE (numerical accuracy)
 Accuracy used as supporting metric

Cross-validation results are used for all model comparisons.
Full-data results are used only for inspection and analysis.

Key Insights
Feature engineering contributes more than hyperparameter tuning
inear models perform strongly on sparse TF-IDF features
LightGBM requires careful tuning to outperform Ridge
A balance between model complexity and generalization is critical

Notes
A full grid search was not performed due to computational cost and
diminishing returns; instead, a focused parameter search was used.
The pipeline is designed to be reproducible, interpretable, and efficient.

"""

# %% 0A - Packages and imports

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor

# %% 1A - Load data and basic inspection

pd.set_option("display.max_colwidth", 300)
pd.set_option("display.max_columns", 50)

file_path = input("Enter path to dataset file: ").strip()
df = pd.read_excel(file_path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
display(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isna().sum())
# Create a mask for missing text_id
mask = df["text_id"].isna()

# Generate new IDs only for missing rows
df.loc[mask, "text_id"] = [
    f"missing_id_{i+1}" for i in range(mask.sum())
]

# %% 1B - Human inspection of the data

sample_cols = ["text_id", "Text", "Vocabulary_1", "Vocabulary_2"]
display(df[sample_cols].sample(5, random_state=42))


#%% 1C - Label distributions

v1_counts = df["Vocabulary_1"].value_counts().sort_index()
v2_counts = df["Vocabulary_2"].value_counts().sort_index()
score_values = sorted(set(v1_counts.index).union(set(v2_counts.index)))

v1_plot_data = [v1_counts.get(x, 0) for x in score_values]
v2_plot_data = [v2_counts.get(x, 0) for x in score_values]

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
x = np.arange(len(score_values))

ax.bar(x - bar_width / 2, v1_plot_data, bar_width, label="Vocabulary_1")
ax.bar(x + bar_width / 2, v2_plot_data, bar_width, label="Vocabulary_2")

ax.set_title("Distribution of Vocabulary Scores")
ax.set_xlabel("Score")
ax.set_ylabel("Number of Essays")
ax.set_xticks(x)
ax.set_xticklabels(score_values)
ax.legend()

plt.tight_layout()
plt.show()


#%% 1D - Agreement between raters

df["score_diff"] = df["Vocabulary_1"] - df["Vocabulary_2"]
df["abs_score_diff"] = df["score_diff"].abs()

print("Average absolute difference between raters:",
      df["abs_score_diff"].mean())
print("\nDistribution of absolute differences:")
print(df["abs_score_diff"].value_counts().sort_index())

agreement_table = pd.crosstab(df["Vocabulary_1"], df["Vocabulary_2"])
display(agreement_table)

exact_agreement = (df["Vocabulary_1"] == df["Vocabulary_2"]).mean()
print(f"Exact agreement rate: {exact_agreement:.3f}")

plt.figure(figsize=(6, 5))
sns.heatmap(agreement_table, annot=True, fmt="d", cmap="Blues")
plt.title("Agreement Table (Rater 1 vs Rater 2)")
plt.xlabel("Vocabulary_2")
plt.ylabel("Vocabulary_1")
plt.show()

#%% 1E: Targets

df["target_mean"] = (df["Vocabulary_1"] + df["Vocabulary_2"]) / 2
df["target_bin"] = df["target_mean"].round().astype(int)

y_mean = df["target_mean"]
y_bin = df["target_bin"]
y1 = df["Vocabulary_1"]
y2 = df["Vocabulary_2"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

#%% 2 - Target and evaluation function


def evaluate_model(y_true_mean, y_true_bin, y_pred, y_true_v1=None, y_true_v2=None):
    y_pred = np.asarray(y_pred)
    y_pred_round = np.clip(np.round(y_pred), 0, 5).astype(int)

    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_true_mean_rounded = np.clip(np.round(y_true_mean), 0, 5).astype(int)

    results = {}

    # Main target metrics
    results["MAE_mean"] = mean_absolute_error(y_true_mean, y_pred)
    results["RMSE_mean"] = np.sqrt(mean_squared_error(y_true_mean, y_pred))
    results["R2_mean"] = r2_score(y_true_mean, y_pred)
    results["QWK_bin"] = cohen_kappa_score(
        y_true_bin, y_pred_round, weights="quadratic")
    results["ACC_bin"] = (y_pred_round == y_true_bin).mean()
    results["QWK_meanRounded"] = cohen_kappa_score(
        y_true_mean_rounded, y_pred_round, weights="quadratic"
    )

    # Optional: compare against each rater
    if y_true_v1 is not None:
        y_true_v1 = np.asarray(y_true_v1).astype(int)
        results["MAE_v1"] = mean_absolute_error(y_true_v1, y_pred)
        results["QWK_v1"] = cohen_kappa_score(
            y_true_v1, y_pred_round, weights="quadratic")
        results["ACC_v1"] = (y_pred_round == y_true_v1).mean()

    if y_true_v2 is not None:
        y_true_v2 = np.asarray(y_true_v2).astype(int)
        results["MAE_v2"] = mean_absolute_error(y_true_v2, y_pred)
        results["QWK_v2"] = cohen_kappa_score(
            y_true_v2, y_pred_round, weights="quadratic")
        results["ACC_v2"] = (y_pred_round == y_true_v2).mean()

    return results

#%% 3A - Baseline model: word TF-IDF + Ridge
# Ddefine an initial simple baseline model
# define X,y

X = df["Text"].astype(str)
y = df["target_mean"]

pipeline_step3 = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=30000,
        min_df=3,
        stop_words="english"
    )),
    ("model", Ridge(alpha=1.0))
])

#%% 3B - Cross-validation: Step 3 baseline
# evaluate the initial baseline with CV
step3_cv_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    print(f"\nStep 3 - Fold {fold}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    y_bin_val = y_bin.iloc[val_idx]
    y1_val = y1.iloc[val_idx]
    y2_val = y2.iloc[val_idx]

    pipeline_step3.fit(X_train, y_train)
    y_pred = pipeline_step3.predict(X_val)

    metrics = evaluate_model(
        y_true_mean=y_val,
        y_true_bin=y_bin_val,
        y_pred=y_pred,
        y_true_v1=y1_val,
        y_true_v2=y2_val
    )

    print(metrics)
    step3_cv_results.append(metrics)

step3_cv_df = pd.DataFrame(step3_cv_results)

print("\n Cross-validation results: Step 3 ")
print(f"MAE_mean: {step3_cv_df['MAE_mean'].mean():.4f} ± {
      step3_cv_df['MAE_mean'].std():.4f}")
print(f"RMSE_mean: {step3_cv_df['RMSE_mean'].mean():.4f} ± {
      step3_cv_df['RMSE_mean'].std():.4f}")
print(f"R2_mean: {step3_cv_df['R2_mean'].mean():.4f} ± {
      step3_cv_df['R2_mean'].std():.4f}")
print(f"QWK_bin: {step3_cv_df['QWK_bin'].mean():.4f} ± {
      step3_cv_df['QWK_bin'].std():.4f}")
print(f"ACC_bin: {step3_cv_df['ACC_bin'].mean():.4f} ± {
      step3_cv_df['ACC_bin'].std():.4f}")


#%% 3C - Tune Step 3 baseline parameters
# Tune Step 3 baseline parameters with CV

X = df["Text"].astype(str)
y = df["target_mean"]

param_grid = [

    {"alpha": 0.5, "ngram_range": (1, 2), "max_features": 30000, "min_df": 2, "sublinear_tf": False},
    {"alpha": 1.0, "ngram_range": (1, 2), "max_features": 30000, "min_df": 2, "sublinear_tf": False},
    {"alpha": 2.0, "ngram_range": (1, 2), "max_features": 30000, "min_df": 2, "sublinear_tf": False},

    {"alpha": 0.5, "ngram_range": (1, 2), "max_features": 50000, "min_df": 3, "sublinear_tf": True},
    {"alpha": 1.0, "ngram_range": (1, 2), "max_features": 50000, "min_df": 3, "sublinear_tf": True},
    {"alpha": 2.0, "ngram_range": (1, 2), "max_features": 50000, "min_df": 3, "sublinear_tf": True},

    {"alpha": 0.5, "ngram_range": (1, 3), "max_features": 50000, "min_df": 2, "sublinear_tf": True},
    {"alpha": 1.0, "ngram_range": (1, 3), "max_features": 50000, "min_df": 2, "sublinear_tf": True},
    {"alpha": 2.0, "ngram_range": (1, 3), "max_features": 50000, "min_df": 2, "sublinear_tf": True},
]

step3_tuning_results = []

for i, params in enumerate(param_grid, start=1):
    print(f"\nRunning config {i}/{len(param_grid)}: {params}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=params["ngram_range"],
            max_features=params["max_features"],
            min_df=params["min_df"],
            stop_words="english",
            sublinear_tf=params["sublinear_tf"]
        )),
        ("model", Ridge(alpha=params["alpha"]))
    ])

    fold_results = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        y_bin_val = y_bin.iloc[val_idx]
        y1_val = y1.iloc[val_idx]
        y2_val = y2.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        metrics = evaluate_model(
            y_true_mean=y_val,
            y_true_bin=y_bin_val,
            y_pred=y_pred,
            y_true_v1=y1_val,
            y_true_v2=y2_val
        )

        fold_results.append(metrics)

    fold_df = pd.DataFrame(fold_results)

    step3_tuning_results.append({
        "alpha": params["alpha"],
        "ngram_range": str(params["ngram_range"]),
        "max_features": params["max_features"],
        "min_df": params["min_df"],
        "sublinear_tf": params["sublinear_tf"],
        "MAE_mean": fold_df["MAE_mean"].mean(),
        "RMSE_mean": fold_df["RMSE_mean"].mean(),
        "R2_mean": fold_df["R2_mean"].mean(),
        "QWK_bin": fold_df["QWK_bin"].mean(),
        "ACC_bin": fold_df["ACC_bin"].mean()
    })

step3_tuning_df = pd.DataFrame(step3_tuning_results)
step3_tuning_df = step3_tuning_df.sort_values(
    by=["QWK_bin", "MAE_mean"],
    ascending=[False, True]
).reset_index(drop=True)

display(step3_tuning_df.round(4))

#%% 3D - Final optimized Step 3 baseline
# Build final optimized Step 3 model

best_step3_params = step3_tuning_df.iloc[0].to_dict()
print("Best Step 3 params:")
print(best_step3_params)

pipeline_step3 = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=eval(best_step3_params["ngram_range"]),
        max_features=int(best_step3_params["max_features"]),
        min_df=int(best_step3_params["min_df"]),
        stop_words="english",
        sublinear_tf=bool(best_step3_params["sublinear_tf"])
    )),
    ("model", Ridge(alpha=float(best_step3_params["alpha"])))
])

#%% 3E - Fit final optimized Step 3 model on full data and store predictions
pipeline_step3.fit(X, y)

df["pred_step3"] = pipeline_step3.predict(X)
df["pred_step3_rounded"] = np.clip(np.round(df["pred_step3"]), 0, 5).astype(int)
#%% Summary Step 3 Baseline model: word TF-IDF + Ridge

print("""
Step 3 Summary – Optimized Baseline Model

A baseline TF-IDF + Ridge regression model was first evaluated using cross-validation,
achieving moderate performance (QWK ~0.34).

To improve this baseline, key parameters were tuned, including:
n-gram range
vocabulary size (max_features)
minimum document frequency
sublinear TF scaling
Ridge regularization strength (alpha)

The optimized model achieved slightly improved performance:
MAE decreased from ~0.39 to ~0.387
QWK increased from ~0.34 to ~0.38
Accuracy increased from ~0.50 to ~0.51

These results demonstrate that even simple linear models can benefit from
careful tuning of text representation and regularization.

The optimized baseline provides a stronger reference point for evaluating more advanced
feature engineering and modeling approaches in subsequent steps.

""")
# %% 4A - Feature engineering for improved modeling

def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def lexical_features(text):
    words = tokenize(text)
    n_words = len(words)

    if n_words == 0:
        return {
            "fe_word_count": 0,
            "fe_unique_words": 0,
            "fe_type_token_ratio": 0,
            "fe_hapax_ratio": 0
        }

    unique_words = len(set(words))
    word_counts = {}

    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1

    hapax = sum(1 for _, c in word_counts.items() if c == 1)

    return {
        "fe_word_count": n_words,
        "fe_unique_words": unique_words,
        "fe_type_token_ratio": unique_words / n_words,
        "fe_hapax_ratio": hapax / n_words
    }


def word_sophistication_features(text):
    words = tokenize(text)

    if len(words) == 0:
        return {
            "fe_avg_word_length": 0,
            "fe_long_word_ratio": 0
        }

    lengths = [len(w) for w in words]

    return {
        "fe_avg_word_length": np.mean(lengths),
        "fe_long_word_ratio": sum(l >= 7 for l in lengths) / len(words)
    }


def sentence_features(text):
    sentences = re.split(r"[.!?]+", str(text))
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    if len(sentences) == 0:
        return {
            "fe_sentence_count": 0,
            "fe_avg_sentence_length": 0
        }

    word_counts = [len(tokenize(s)) for s in sentences]

    return {
        "fe_sentence_count": len(sentences),
        "fe_avg_sentence_length": np.mean(word_counts)
    }


def extract_features(text):
    features = {}
    features.update(lexical_features(text))
    features.update(word_sophistication_features(text))
    features.update(sentence_features(text))
    return features


feature_list = df["Text"].astype(str).apply(extract_features)
feature_df = pd.DataFrame(feature_list.tolist(), index=df.index)

print(feature_df.head())
print(feature_df.describe())

# %% 4B - Improved modeling: word TF-IDF + char TF-IDF + handcrafted features
# initial Step 4 model

X_full = df[["Text"]].copy()
X_full = pd.concat([X_full, feature_df], axis=1)

numeric_feature_cols = feature_df.columns.tolist()

preprocessor_step4 = ColumnTransformer(
    transformers=[
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
            max_df=0.9, #reduce dominance of frequent words
            stop_words="english"
        ), "Text"),

        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=50000,
            sublinear_tf=True,
            max_df=0.9,
            min_df=2
        ), "Text"),

        ("num", StandardScaler(with_mean=False), numeric_feature_cols)
    ]
)

# Tune alpha
# for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
#     model = Ridge(alpha=alpha)

# pileline
pipeline_step4 = Pipeline([
    ("features", preprocessor_step4),
    ("model", Ridge(alpha=2))
])

# %% 4C - Cross-validation: improved Step 4 model
# CV for initial Step 4 model

step4_cv_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_full), start=1):
    print(f"\nStep 4 - Fold {fold}")

    X_train = X_full.iloc[train_idx]
    X_val = X_full.iloc[val_idx]

    y_train = y_mean.iloc[train_idx]
    y_val = y_mean.iloc[val_idx]

    y_bin_val = y_bin.iloc[val_idx]
    y1_val = y1.iloc[val_idx]
    y2_val = y2.iloc[val_idx]

    pipeline_step4.fit(X_train, y_train)
    y_pred = pipeline_step4.predict(X_val)

    metrics = evaluate_model(
        y_true_mean=y_val,
        y_true_bin=y_bin_val,
        y_pred=y_pred,
        y_true_v1=y1_val,
        y_true_v2=y2_val
    )

    print(metrics)
    step4_cv_results.append(metrics)

step4_cv_df = pd.DataFrame(step4_cv_results)

print("\n Cross-validation results: Step 4")
print(f"MAE_mean: {step4_cv_df['MAE_mean'].mean():.4f} ± {
      step4_cv_df['MAE_mean'].std():.4f}")
print(f"RMSE_mean: {step4_cv_df['RMSE_mean'].mean():.4f} ± {
      step4_cv_df['RMSE_mean'].std():.4f}")
print(f"R2_mean: {step4_cv_df['R2_mean'].mean():.4f} ± {
      step4_cv_df['R2_mean'].std():.4f}")
print(f"QWK_bin: {step4_cv_df['QWK_bin'].mean():.4f} ± {
      step4_cv_df['QWK_bin'].std():.4f}")
print(f"ACC_bin: {step4_cv_df['ACC_bin'].mean():.4f} ± {
      step4_cv_df['ACC_bin'].std():.4f}")

# %% 4D - Tune Step 4 parameters

param_grid = [
    # {"alpha": 0.5, "word_ngram": (1,2), "char_ngram": (3,5), "min_df": 2},
    # {"alpha": 1.0, "word_ngram": (1,2), "char_ngram": (3,5), "min_df": 2},
    # {"alpha": 2.0, "word_ngram": (1,2), "char_ngram": (3,5), "min_df": 2},
    # {"alpha": 5.0, "word_ngram": (1,2), "char_ngram": (3,5), "min_df": 2},

    {"alpha": 0.5, "word_ngram": (1,3), "char_ngram": (2,5), "min_df": 2}, #this is found best, recommending in additional run to reduce runtime- comment everithing else!!
    #{"alpha": 1.0, "word_ngram": (1,3), "char_ngram": (2,5), "min_df": 2},
    # {"alpha": 2.0, "word_ngram": (1,3), "char_ngram": (2,5), "min_df": 2},
    # {"alpha": 5.0, "word_ngram": (1,3), "char_ngram": (2,5), "min_df": 2},

    # {"alpha": 1.0, "word_ngram": (1,3), "char_ngram": (3,6), "min_df": 3},
    # {"alpha": 2.0, "word_ngram": (1,3), "char_ngram": (3,6), "min_df": 3},
    # {"alpha": 5.0, "word_ngram": (1,3), "char_ngram": (3,6), "min_df": 3},

    # {"alpha": 1.0, "word_ngram": (1,2), "char_ngram": (2,6), "min_df": 2},
    # {"alpha": 2.0, "word_ngram": (1,2), "char_ngram": (2,6), "min_df": 2},
    # {"alpha": 2.0, "word_ngram": (1,3), "char_ngram": (2,6), "min_df": 2},

    # {"alpha": 2.0, "word_ngram": (1,3), "char_ngram": (2,5), "min_df": 1},
    # {"alpha": 2.0, "word_ngram": (1,3), "char_ngram": (2,5), "min_df": 4},
]

step4_tuning_results = []

for i, params in enumerate(param_grid, start=1):
    print(f"\nRunning config {i}/{len(param_grid)}: {params}")

    preprocessor = ColumnTransformer([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=params["word_ngram"],
            max_features=50000,
            min_df=params["min_df"],
            stop_words="english"
        ), "Text"),

        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=params["char_ngram"],
            max_features=50000,
            min_df=params["min_df"]
        ), "Text"),

        ("num", StandardScaler(), numeric_feature_cols)
    ])

    pipeline = Pipeline([
        ("features", preprocessor),
        ("model", Ridge(alpha=params["alpha"]))
    ])

    fold_results = []

    for train_idx, val_idx in kf.split(X_full):
        X_train = X_full.iloc[train_idx]
        X_val = X_full.iloc[val_idx]

        y_train = y_mean.iloc[train_idx]
        y_val = y_mean.iloc[val_idx]

        y_bin_val = y_bin.iloc[val_idx]
        y1_val = y1.iloc[val_idx]
        y2_val = y2.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        metrics = evaluate_model(
            y_true_mean=y_val,
            y_true_bin=y_bin_val,
            y_pred=y_pred,
            y_true_v1=y1_val,
            y_true_v2=y2_val
        )

        fold_results.append(metrics)

    fold_df = pd.DataFrame(fold_results)

    step4_tuning_results.append({
        "alpha": params["alpha"],
        "word_ngram": str(params["word_ngram"]),
        "char_ngram": str(params["char_ngram"]),
        "min_df": params["min_df"],
        "MAE_mean": fold_df["MAE_mean"].mean(),
        "QWK_bin": fold_df["QWK_bin"].mean(),
        "ACC_bin": fold_df["ACC_bin"].mean()
    })

step4_tuning_df = pd.DataFrame(step4_tuning_results)

step4_tuning_df = step4_tuning_df.sort_values(
    by=["QWK_bin", "MAE_mean"],
    ascending=[False, True]
).reset_index(drop=True)

display(step4_tuning_df.round(4))

# %% 4E - Final optimized Step 4 model

best_step4_params = step4_tuning_df.iloc[0].to_dict()
print("Best Step 4 params:")
print(best_step4_params)

preprocessor_step4 = ColumnTransformer([
    ("word_tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=eval(best_step4_params["word_ngram"]),
        max_features=50000,
        min_df=int(best_step4_params["min_df"]),
        stop_words="english"
    ), "Text"),

    ("char_tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=eval(best_step4_params["char_ngram"]),
        max_features=50000,
        min_df=int(best_step4_params["min_df"])
    ), "Text"),

    ("num", StandardScaler(), numeric_feature_cols)
])

pipeline_step4 = Pipeline([
    ("features", preprocessor_step4),
    ("model", Ridge(alpha=float(best_step4_params["alpha"])))
])

#%% 4F fit full data with refined parameters:
# fit final optimized Step 4 on full data   
pipeline_step4.fit(X_full, y_mean)

df["pred_step4"] = pipeline_step4.predict(X_full)
df["pred_step4_rounded"] = np.clip(np.round(df["pred_step4"]), 0, 5).astype(int)

# %% 4G - Compare optimized Step 3 and optimized Step 4

results_step3 = evaluate_model(
    y_true_mean=df["target_mean"],
    y_true_bin=df["target_bin"],
    y_pred=df["pred_step3"],
    y_true_v1=df["Vocabulary_1"],
    y_true_v2=df["Vocabulary_2"]
)

results_step4 = evaluate_model(
    y_true_mean=df["target_mean"],
    y_true_bin=df["target_bin"],
    y_pred=df["pred_step4"],
    y_true_v1=df["Vocabulary_1"],
    y_true_v2=df["Vocabulary_2"]
)

results_table = pd.DataFrame({
    "Step 3": results_step3,
    "Step 4": results_step4
}).T.round(4)

display(results_table)
#%% Summary Step 4 word TF-IDF + char TF-IDF + handcrafted features
print("""
Step 4 Summary – Feature-Enriched Model

The Step 4 model extends the baseline by incorporating word-level TF-IDF,
character-level TF-IDF, and handcrafted linguistic features.

Cross-validation results show a clear improvement over the optimized Step 3 baseline:
QWK increased from ~0.38 to ~0.42
Accuracy improved accordingly
MAE remained similar, with a slight increase

Parameter tuning provided a modest additional improvement, indicating that the
initial feature design already captured most of the available signal.

Overall, these results demonstrate that richer feature representations improve
the model’s ability to rank essays by vocabulary quality, which is critical for
this task.
""")
print(""" Step 3 baseline → works
Step 3 tuning → improves
Step 4 features → improves more
Step 4 tuning → small refinemen""")

print('''
Up to Step 4, improved the TEXT REPRESENTATION:
word TF-IDF
character TF-IDF
handcrafted linguistic features

In Step 5, keep that strong representation from Step 4,
but replace the linear Ridge model with LightGBM,
which can capture non-linear relationships and interactions
between features.

This may help improve both:
MAE (numerical accuracy)
QWK (ranking / ordinal performance) ''')

# %% 5A - Step 5: LightGBM model

lgbm_model = LGBMRegressor(
    n_estimators=300,        
    learning_rate=0.05,      
    num_leaves=31,          
    min_child_samples=20,   
    subsample=0.8,          
    colsample_bytree=0.8,   
    random_state=42,
    n_jobs=-1
)

# Build the full Step 5 pipeline:
# 1. Use the Step 4 feature preprocessor (word TF-IDF + char TF-IDF + numeric features)
# 2. Feed those features into LightGBM
pipeline_step5 = Pipeline([
    ("features", preprocessor_step4),
    ("model", lgbm_model)
])

#%% 5B - Tune Step 5 (LightGBM) parameters with cross-validation
# Selection priority: Highest QWK_bin, Lowest MAE_mean as tie-breaker
# LightGBM Parameter Tuning Strategy

#In this step, I did not perform a wide grid search over all possible hyperparameters.
#Instead, I used a small, targeted set of configurations.
#Instead of a full grid search, I used a small, targeted parameter set to efficiently 
#explore the most impactful model configurations. This approach balances performance, 
#computational cost, and interpretability, and is sufficient to determine whether LightGBM improves over the baseline.


lgbm_param_grid = [
    #current safe baseline
    # {
    #     "n_estimators": 300,
    #     "learning_rate": 0.05,
    #     "num_leaves": 31,
    #     "min_child_samples": 20,
    #     "subsample": 0.8,
    #     "colsample_bytree": 0.8
    # },
    #stronger but still controlled
    # {
    #     "n_estimators": 500,
    #     "learning_rate": 0.03,
    #     "num_leaves": 63,
    #     "min_child_samples": 30,
    #     "subsample": 0.8,
    #     "colsample_bytree": 0.8
    # },
    #moderately stronger with slightly less randomness
    ###this is found best, recommending in additional run to reduce runtime- comment everything else!!
    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.9,
        "colsample_bytree": 0.9
    }
]

step5_tuning_results = []

for i, params in enumerate(lgbm_param_grid, start=1):
    print(f"\nRunning LightGBM config {i}/{len(lgbm_param_grid)}: {params}")

    lgbm_model = LGBMRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("features", preprocessor_step4),
        ("model", lgbm_model)
    ])

    fold_results = []

    for train_idx, val_idx in kf.split(X_full):
        X_train = X_full.iloc[train_idx]
        X_val = X_full.iloc[val_idx]

        y_train = y_mean.iloc[train_idx]
        y_val = y_mean.iloc[val_idx]

        y_bin_val = y_bin.iloc[val_idx]
        y1_val = y1.iloc[val_idx]
        y2_val = y2.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        metrics = evaluate_model(
            y_true_mean=y_val,
            y_true_bin=y_bin_val,
            y_pred=y_pred,
            y_true_v1=y1_val,
            y_true_v2=y2_val
        )

        fold_results.append(metrics)

    fold_df = pd.DataFrame(fold_results)

    step5_tuning_results.append({
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "num_leaves": params["num_leaves"],
        "min_child_samples": params["min_child_samples"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "MAE_mean": fold_df["MAE_mean"].mean(),
        "RMSE_mean": fold_df["RMSE_mean"].mean(),
        "R2_mean": fold_df["R2_mean"].mean(),
        "QWK_bin": fold_df["QWK_bin"].mean(),
        "ACC_bin": fold_df["ACC_bin"].mean()
    })

step5_tuning_df = pd.DataFrame(step5_tuning_results)

# Sort by main metric first (QWK), then by MAE
step5_tuning_df = step5_tuning_df.sort_values(
    by=["QWK_bin", "MAE_mean"],
    ascending=[False, True]
).reset_index(drop=True)

display(step5_tuning_df.round(4))

# %% 5C - Build final optimized Step 5 model

best_step5_params = step5_tuning_df.iloc[0].to_dict()

print("Best Step 5 params:")
print(best_step5_params)

lgbm_model = LGBMRegressor(
    n_estimators=int(best_step5_params["n_estimators"]),
    learning_rate=float(best_step5_params["learning_rate"]),
    num_leaves=int(best_step5_params["num_leaves"]),
    min_child_samples=int(best_step5_params["min_child_samples"]),
    subsample=float(best_step5_params["subsample"]),
    colsample_bytree=float(best_step5_params["colsample_bytree"]),
    random_state=42,
    n_jobs=-1
)

pipeline_step5 = Pipeline([
    ("features", preprocessor_step4),
    ("model", lgbm_model)
])

# %% 5D - Fit final optimized Step 5 model on full data
# This is for predictions/analysis only, not for honest evaluation.

pipeline_step5.fit(X_full, y_mean)

df["pred_step5"] = pipeline_step5.predict(X_full)
df["pred_step5_rounded"] = np.clip(
    np.round(df["pred_step5"]), 0, 5
).astype(int)

# %% 5E - Compare optimized Step 3, Step 4, and Step 5 on full dataset
# these full-data results are for inspection only. 
# final performance should be reported from CV results.

results_step5 = evaluate_model(
    y_true_mean=df["target_mean"],
    y_true_bin=df["target_bin"],
    y_pred=df["pred_step5"],
    y_true_v1=df["Vocabulary_1"],
    y_true_v2=df["Vocabulary_2"]
)

results_table = pd.DataFrame({
    "Step 3 (Optimized TF-IDF + Ridge)": results_step3,
    "Step 4 (Feature-enriched Ridge)": results_step4,
    "Step 5 (Optimized LightGBM)": results_step5
}).T.round(4)

display(results_table)

#%% Plot: True vs Predicted (Step 5)

y_true = df["target_mean"]
y_pred = df["pred_step5"]

plt.figure(figsize=(7,7))

# Add small jitter to reduce overlap
noise = np.random.normal(0, 0.03, size=len(y_true))

plt.scatter(y_true, y_pred + noise, alpha=0.3)

# Diagonal
plt.plot([0, 5], [0, 5])

plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title("True vs Predicted Scores (Step 5 - LightGBM)")

plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()

#%% Summary – Step 5 (LightGBM Model)
print('''
In this step, the linear Ridge model was replaced with LightGBM to capture potential 
non-linear relationships and feature interactions, while keeping the strong feature representation developed in Step 4.
A targeted parameter tuning approach was used instead of a full grid search, 
focusing on a small number of impactful configurations to balance performance and computational efficiency. 

The best-performing configuration achieved:

- QWK ≈ 0.413  
- MAE ≈ 0.374  

This represents a slight improvement over the Step 4 model.

Overall, the results indicate that while LightGBM can capture additional complexity,
the majority of the predictive signal was already extracted through feature engineering.
As a result, the gain from switching to a non-linear model was modest.
      ''')
#%% Summary Total 
print(''' 
Final Model Summary
This project demonstrates a structured approach to solving an ordinal NLP prediction 
task through progressive model improvement.
Key findings:

#A strong baseline (TF-IDF + Ridge) already achieved solid performance, 
highlighting the importance of lexical signals in the data  

#The largest performance improvement came from feature engineering,
particularly the addition of character-level features and linguistic features (Step 4)  

#Switching to a more complex non-linear model (LightGBM) provided only a modest improvement,
indicating that most of the signal was already captured by the feature representation  

Overall performance progression:

- Step 3 → Establish strong baseline  
- Step 4 → Significant improvement through feature engineering  
- Step 5 → Small gain from model complexity  

Final Conclusion

The results show that in this task:

Feature representation is more important than model complexity
Linear models can be highly competitive when combined with strong features  
Non-linear models add value, but only after meaningful feature engineering  

Additionally, performance is inherently limited by:
- Class imbalance (few extreme scores)  
- Label noise (disagreement between human raters)  

The final model achieves strong ranking performance (QWK), 
making it suitable for practical scoring applications, 
while also highlighting clear directions for future improvement (ordinal modeling, threshold optimization).
''')