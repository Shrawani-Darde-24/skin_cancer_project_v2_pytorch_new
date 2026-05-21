# Skin Cancer Classifier — ABCD Feature Pipeline

Classifies skin lesions into 7 diagnostic categories using clinical ABCD features
extracted from dermoscopy images, combined with patient metadata.

## Dataset
Download HAM10000 from Kaggle:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Place files as:
```
data/
  HAM10000_metadata.csv
  images/
    ISIC_0024306.jpg
    ISIC_0024307.jpg
    ...
```

## Project Structure
```
skin_cancer_project/
├── src/
│   ├── feature_extraction.py   ← ABCD feature extractor (A/B/C/D)
│   ├── train_model.py          ← ML training pipeline (RF, XGBoost, SVM)
│   └── predict.py              ← Inference wrapper
├── notebooks/
│   └── eda_and_training.ipynb  ← EDA, visualizations, class analysis
├── webapp/
│   └── app.py                  ← Flask web app (upload image → get prediction)
├── models/                     ← Saved model bundles (after training)
├── outputs/                    ← Plots, confusion matrix, feature importance
├── data/                       ← Place dataset here
└── requirements.txt
```

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Extract features and train model
```bash
python src/train_model.py \
  --data_csv data/HAM10000_metadata.csv \
  --image_dir data/images/ \
  --model_out models/skin_cancer_model.pkl \
  --output_dir outputs/ \
  --cache_features data/abcd_features_cache.csv
```

The `--cache_features` flag saves extracted ABCD features to CSV so you don't
re-run extraction every time. On subsequent runs, it loads from cache.

### 3. Run the web app
```bash
python webapp/app.py
```
Then open http://localhost:5000 — upload a lesion image, fill in patient info,
and get a 7-class prediction with ABCD scores.

### 4. Predict a single image from code
```python
from src.predict import SkinCancerPredictor

pred = SkinCancerPredictor("models/skin_cancer_model.pkl")
result = pred.predict(
    "my_lesion.jpg",
    age=52,
    sex="female",
    localization="back",
    dx_type="histo"
)
print(result["predicted_label"])     # e.g. "Melanoma"
print(result["risk_level"])          # "HIGH — malignant"
print(result["abcd_features"])       # all extracted ABCD scores
```

### 5. Explore the EDA notebook
```bash
jupyter notebook notebooks/eda_and_training.ipynb
```

---

## ABCD Features Extracted

| Feature | Method | Key metric |
|---|---|---|
| **A — Asymmetry** | Fold mask along 2 axes, compute non-overlap ratio | `asymmetry_score` (0=symmetric, 1=fully asymmetric) |
| **B — Border** | Compactness index (4π·area/perimeter²), contour roughness | `border_irregularity` |
| **C — Color** | HSV channel stats, grayscale entropy, color variance | `color_entropy`, `color_h_std` |
| **D — Diameter** | Ellipse fit + bounding box diagonal | `diameter_max_mm`, `diameter_major_mm` |

---

## 7 Cancer Classes (HAM10000)

| Code | Name | Malignancy |
|---|---|---|
| `mel` | Melanoma | ⚠ Malignant |
| `nv` | Melanocytic nevi | Benign |
| `bcc` | Basal cell carcinoma | ⚠ Malignant |
| `akiec` | Actinic keratosis | ⚠ Malignant |
| `bkl` | Benign keratosis | Benign |
| `df` | Dermatofibroma | Benign |
| `vasc` | Vascular lesion | Benign |

---

## Disclaimer
This tool is for **educational and research purposes only**.
It is not a medical device and must not be used for clinical diagnosis.
Always consult a certified dermatologist.
