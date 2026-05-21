"""
Model Training Pipeline — Skin Cancer Classification
======================================================
Trains classifiers on ABCD features + patient metadata
to predict one of 7 HAM10000 dx classes.

Usage:
    python train_model.py --data_csv data/HAM10000_metadata.csv \
                          --image_dir data/images/ \
                          --model_out models/skin_cancer_model.pkl
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Local module
import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features_batch

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

TARGET_COLUMN = "dx"
METADATA_FEATURES = ["age", "sex", "localization", "dx_type"]
RANDOM_STATE = 42

LABEL_MAP = {
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "bcc":   "Basal cell carcinoma",
    "akiec": "Actinic keratosis",
    "bkl":   "Benign keratosis",
    "df":    "Dermatofibroma",
    "vasc":  "Vascular lesion",
}

MALIGNANT = {"mel", "bcc", "akiec"}  # for binary sub-task


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"image_id", "dx"} | set(METADATA_FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {missing}")
    print(f"Loaded {len(df)} records from metadata.")
    return df


def build_image_paths(df: pd.DataFrame, image_dir: str) -> list:
    paths = []
    for img_id in df["image_id"]:
        # HAM10000 stores images as ISIC_{id}.jpg across two folders
        p = os.path.join(image_dir, f"{img_id}.jpg")
        if not os.path.exists(p):
            p = os.path.join(image_dir, f"ISIC_{img_id}.jpg")
        paths.append(p)
    return paths


# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────

def encode_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical metadata columns.
    """
    cat_cols = ["sex", "localization", "dx_type"]
    df = df.copy()

    # Fill NaN in age with median
    df["age"] = df["age"].fillna(df["age"].median())

    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df


def merge_features(df_meta: pd.DataFrame, abcd_records: list) -> pd.DataFrame:
    """
    Merge ABCD features with metadata.
    Drops rows where feature extraction failed.
    """
    abcd_df = pd.DataFrame(abcd_records)

    # Remove error rows
    if "error" in abcd_df.columns:
        good = abcd_df["error"].isna()
        print(f"  ABCD extraction: {good.sum()} succeeded, {(~good).sum()} failed.")
        abcd_df = abcd_df[good].drop(columns=["error"])

    # Extract image_id from path
    abcd_df["image_id"] = abcd_df["image_path"].apply(
        lambda p: os.path.splitext(os.path.basename(p))[0]
    )
    abcd_df.drop(columns=["image_path"], inplace=True)

    merged = df_meta.merge(abcd_df, on="image_id", how="inner")
    print(f"  Merged dataset: {len(merged)} rows.")
    return merged


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return all feature columns (everything except metadata identifiers and target).
    """
    exclude = {"image_id", "lesion_id", TARGET_COLUMN}
    return [c for c in df.columns if c not in exclude]


# ──────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────

def build_models() -> dict:
    return {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "XGBoost (GBM)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=RANDOM_STATE,
            )),
        ]),
        "SVM (RBF)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# ──────────────────────────────────────────────
# Training & evaluation
# ──────────────────────────────────────────────

def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    label_encoder, feature_cols, output_dir
):
    models = build_models()
    results = {}
    best_model, best_f1, best_name = None, 0.0, ""

    print("\n── Training models ──────────────────────────────")
    for name, pipeline in models.items():
        print(f"\n  [{name}]")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
        print(f"  CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"  Test accuracy: {acc:.4f}   Test F1: {f1:.4f}")

        results[name] = {
            "pipeline": pipeline,
            "accuracy": acc,
            "f1_weighted": f1,
            "y_pred": y_pred,
            "cv_mean": cv_scores.mean(),
        }

        if f1 > best_f1:
            best_f1, best_model, best_name = f1, pipeline, name

    print(f"\n  Best model: {best_name} (F1={best_f1:.4f})")

    # Full report for best model
    class_names = label_encoder.classes_
    y_pred_best = results[best_name]["y_pred"]
    print("\n── Classification Report ────────────────────────")
    print(classification_report(
        y_test, y_pred_best,
        target_names=[LABEL_MAP.get(c, c) for c in class_names]
    ))

    # Save confusion matrix
    _plot_confusion_matrix(
        y_test, y_pred_best,
        class_names=[LABEL_MAP.get(c, c) for c in class_names],
        title=f"Confusion Matrix — {best_name}",
        output_path=os.path.join(output_dir, "confusion_matrix.png"),
    )

    # Feature importance (RF only)
    if best_name == "Random Forest":
        _plot_feature_importance(
            best_model.named_steps["clf"].feature_importances_,
            feature_cols,
            output_path=os.path.join(output_dir, "feature_importance.png"),
        )

    return best_model, best_name, results


def _plot_confusion_matrix(y_true, y_pred, class_names, title, output_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def _plot_feature_importance(importances, feature_names, output_path, top_n=20):
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in reversed(idx)],
        [importances[i] for i in reversed(idx)],
        color="#5E4DC8",
    )
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def main(args):
    os.makedirs(args.model_out.replace(os.path.basename(args.model_out), ""), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load metadata
    print("\n[1/5] Loading metadata...")
    df = load_metadata(args.data_csv)

    # 2. Extract ABCD features from images
    print("\n[2/5] Extracting ABCD features from images...")
    if args.cache_features and os.path.exists(args.cache_features):
        print(f"  Loading cached features from {args.cache_features}")
        abcd_df = pd.read_csv(args.cache_features)
        abcd_records = abcd_df.to_dict("records")
    else:
        image_paths = build_image_paths(df, args.image_dir)
        abcd_records = extract_features_batch(image_paths, verbose=True)
        if args.cache_features:
            pd.DataFrame(abcd_records).to_csv(args.cache_features, index=False)
            print(f"  Cached features → {args.cache_features}")

    # 3. Build combined dataset
    print("\n[3/5] Merging features and encoding metadata...")
    df_encoded = encode_metadata(df)
    merged = merge_features(df_encoded, abcd_records)

    feature_cols = get_feature_columns(merged)
    X = merged[feature_cols].values
    y_raw = merged[TARGET_COLUMN].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Class distribution:")
    for cls, cnt in zip(*np.unique(y_raw, return_counts=True)):
        print(f"    {LABEL_MAP.get(cls, cls):30s} {cnt:5d}")

    # 4. Train/test split
    print("\n[4/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # 5. Train, evaluate, save
    print("\n[5/5] Training and evaluating...")
    best_model, best_name, results = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        le, feature_cols, args.output_dir
    )

    # Save model bundle
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "label_map": LABEL_MAP,
    }
    with open(args.model_out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Model saved → {args.model_out}")
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train skin cancer classifier")
    parser.add_argument("--data_csv", required=True, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--image_dir", required=True, help="Directory containing lesion images")
    parser.add_argument("--model_out", default="models/skin_cancer_model.pkl")
    parser.add_argument("--output_dir", default="outputs/", help="Directory for plots")
    parser.add_argument("--cache_features", default=None, help="CSV file to cache/load ABCD features")
    args = parser.parse_args()
    main(args)
