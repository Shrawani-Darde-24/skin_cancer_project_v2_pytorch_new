"""
Improved Training Pipeline — CNN + ABCD Features
==================================================
Combines EfficientNetB0 deep features (1280-dim) with
ABCD clinical features for significantly better accuracy.

Usage:
    python src/train_model_v2.py \
        --data_csv data/HAM10000_metadata.csv \
        --image_dir data/images/ \
        --model_out models/skin_cancer_model_v2.pkl \
        --output_dir outputs/
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF logs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features_batch, extract_all_features
from cnn_features import load_efficientnet, extract_cnn_features_batch

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

TARGET_COLUMN  = "dx"
RANDOM_STATE   = 42
CNN_CACHE_FILE = "data/cnn_features_cache.npy"
ABCD_CACHE_FILE = "data/abcd_features_cache.csv"

LABEL_MAP = {
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "bcc":   "Basal cell carcinoma",
    "akiec": "Actinic keratosis",
    "bkl":   "Benign keratosis",
    "df":    "Dermatofibroma",
    "vasc":  "Vascular lesion",
}


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────

def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} records.")
    return df


def build_image_paths(df: pd.DataFrame, image_dir: str) -> list:
    paths = []
    for img_id in df["image_id"]:
        p = os.path.join(image_dir, f"{img_id}.jpg")
        if not os.path.exists(p):
            p = os.path.join(image_dir, f"ISIC_{img_id}.jpg")
        paths.append(p)
    return paths


def encode_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df = pd.get_dummies(df, columns=["sex", "localization", "dx_type"], drop_first=False)
    return df


# ──────────────────────────────────────────────
# Feature combination
# ──────────────────────────────────────────────

def build_combined_features(
    df: pd.DataFrame,
    image_paths: list,
    cnn_model,
    args
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Returns (X, y, feature_names)
    X combines: CNN features (PCA-reduced) + ABCD features + metadata
    """

    # ── 1. CNN features ──────────────────────────────────────
    print("\n  [CNN] Extracting EfficientNetB0 features...")
    if args.cnn_cache and os.path.exists(args.cnn_cache):
        print(f"  Loading cached CNN features from {args.cnn_cache}")
        cnn_features = np.load(args.cnn_cache)
    else:
        cnn_features = extract_cnn_features_batch(image_paths, cnn_model, batch_size=32)
        if args.cnn_cache:
            np.save(args.cnn_cache, cnn_features)
            print(f"  Saved CNN cache → {args.cnn_cache}")

    print(f"  CNN feature matrix: {cnn_features.shape}")

    # Reduce CNN dimensions with PCA (keep 95% variance, max 256 dims)
    print("  Applying PCA to CNN features...")
    pca = PCA(n_components=min(256, cnn_features.shape[0] - 1), random_state=RANDOM_STATE)
    cnn_reduced = pca.fit_transform(cnn_features)
    explained = pca.explained_variance_ratio_.cumsum()
    n_components = int(np.searchsorted(explained, 0.95)) + 1
    cnn_reduced = cnn_reduced[:, :n_components]
    print(f"  PCA: {cnn_features.shape[1]} → {n_components} dims (95% variance)")

    # ── 2. ABCD features ────────────────────────────────────
    print("\n  [ABCD] Extracting clinical features...")
    if args.abcd_cache and os.path.exists(args.abcd_cache):
        print(f"  Loading cached ABCD features from {args.abcd_cache}")
        abcd_df = pd.read_csv(args.abcd_cache)
        abcd_records = abcd_df.to_dict("records")
    else:
        abcd_records = extract_features_batch(image_paths, verbose=True)
        if args.abcd_cache:
            pd.DataFrame(abcd_records).to_csv(args.abcd_cache, index=False)

    abcd_df = pd.DataFrame(abcd_records)
    if "error" in abcd_df.columns:
        abcd_df["error"] = abcd_df["error"].fillna("")
        abcd_df = abcd_df.copy()

    abcd_numeric_cols = [
        c for c in abcd_df.columns
        if c not in ["image_path", "error"]
        and abcd_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    abcd_matrix = abcd_df[abcd_numeric_cols].fillna(0).values
    print(f"  ABCD feature matrix: {abcd_matrix.shape}")

    # ── 3. Patient metadata ──────────────────────────────────
    df_encoded = encode_metadata(df)
    meta_exclude = {"image_id", "lesion_id", TARGET_COLUMN, "label", "is_malignant"}
    meta_cols = [c for c in df_encoded.columns if c not in meta_exclude
                 and df_encoded[c].dtype in [np.float64, np.float32, np.int64, np.int32, bool]]
    meta_matrix = df_encoded[meta_cols].fillna(0).values.astype(np.float32)
    print(f"  Metadata matrix: {meta_matrix.shape}")

    # ── 4. Combine all features ──────────────────────────────
    X = np.hstack([cnn_reduced, abcd_matrix, meta_matrix])
    feature_names = (
        [f"cnn_pca_{i}" for i in range(cnn_reduced.shape[1])] +
        abcd_numeric_cols +
        meta_cols
    )
    print(f"\n  Combined feature matrix: {X.shape}")

    # Labels
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COLUMN].values)

    return X, y, feature_names, le, pca, n_components, abcd_numeric_cols, meta_cols


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────

def build_models():
    return {
        "Random Forest (CNN+ABCD)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "XGBoost (CNN+ABCD)": Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )),
]),
}


# ──────────────────────────────────────────────
# Training & evaluation
# ──────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test, le, output_dir):
    models = build_models()
    best_model, best_f1, best_name = None, 0.0, ""
    results = {}

    print("\n── Training models ──────────────────────────────")
    for name, pipeline in models.items():
        print(f"\n  [{name}]")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=cv, scoring="f1_weighted", n_jobs=-1)
        print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        print(f"  Test accuracy: {acc:.4f}   Test F1: {f1:.4f}")

        results[name] = {"pipeline": pipeline, "accuracy": acc, "f1": f1, "y_pred": y_pred}
        if f1 > best_f1:
            best_f1, best_model, best_name = f1, pipeline, name

    print(f"\n  Best model: {best_name} (F1={best_f1:.4f})")

    # Classification report
    class_names = le.classes_
    y_pred_best = results[best_name]["y_pred"]
    print("\n── Classification Report ────────────────────────")
    print(classification_report(
        y_test, y_pred_best,
        target_names=[LABEL_MAP.get(c, c) for c in class_names]
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[LABEL_MAP.get(c, c) for c in class_names],
                yticklabels=[LABEL_MAP.get(c, c) for c in class_names], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix_v2.png")
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f"  Saved: {cm_path}")

    return best_model, best_name, results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(args):
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[1/5] Loading metadata...")
    df = load_metadata(args.data_csv)

    print("\n[2/5] Building image paths...")
    image_paths = build_image_paths(df, args.image_dir)
    found = sum(os.path.exists(p) for p in image_paths)
    print(f"  Found {found}/{len(image_paths)} images on disk.")

    print("\n[3/5] Loading EfficientNetB0...")
    cnn_model = load_efficientnet()

    print("\n[4/5] Extracting features (CNN + ABCD + metadata)...")
    X, y, feature_names, le, pca, n_components, abcd_cols, meta_cols = \
        build_combined_features(df, image_paths, cnn_model, args)

    print(f"\n  Class distribution:")
    for cls, cnt in zip(*np.unique(df[TARGET_COLUMN].values, return_counts=True)):
        print(f"    {LABEL_MAP.get(cls, cls):30s} {cnt:5d}")

    print("\n[5/5] Splitting and training...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    best_model, best_name, results = train_and_evaluate(
        X_train, X_test, y_train, y_test, le, args.output_dir
    )

    # Save model bundle
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "label_encoder": le,
        "feature_names": feature_names,
        "pca": pca,
        "n_pca_components": n_components,
        "abcd_cols": abcd_cols,
        "meta_cols": meta_cols,
        "label_map": LABEL_MAP,
        "version": "v2_cnn_abcd",
    }
    with open(args.model_out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Model saved → {args.model_out}")
    print("\nDone! ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN+ABCD skin cancer classifier")
    parser.add_argument("--data_csv",   required=True)
    parser.add_argument("--image_dir",  required=True)
    parser.add_argument("--model_out",  default="models/skin_cancer_model_v2.pkl")
    parser.add_argument("--output_dir", default="outputs/")
    parser.add_argument("--cnn_cache",  default="data/cnn_features_cache.npy",
                        help="Cache file for CNN features (.npy)")
    parser.add_argument("--abcd_cache", default="data/abcd_features_cache.csv",
                        help="Cache file for ABCD features (.csv)")
    args = parser.parse_args()
    main(args)
