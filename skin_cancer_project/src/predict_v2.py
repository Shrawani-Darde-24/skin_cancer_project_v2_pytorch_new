"""
Predictor V2 — CNN + ABCD Skin Cancer Classifier
==================================================
Inference using the combined EfficientNetB0 + ABCD model.
"""

import os
import pickle
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_all_features
from cnn_features import load_efficientnet, extract_single_cnn_feature


class SkinCancerPredictorV2:

    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        self.model          = bundle["model"]
        self.model_name     = bundle["model_name"]
        self.label_encoder  = bundle["label_encoder"]
        self.feature_names  = bundle["feature_names"]
        self.pca            = bundle["pca"]
        self.n_pca          = bundle["n_pca_components"]
        self.abcd_cols      = bundle["abcd_cols"]
        self.meta_cols      = bundle["meta_cols"]
        self.label_map      = bundle.get("label_map", {})

        print("  Loading EfficientNetB0 for inference...")
        self.cnn_model = load_efficientnet()
        print("  Model ready!")

    def predict(self, image_path: str, age=None, sex=None,
                localization=None, dx_type=None) -> dict:

        from tensorflow.keras.applications.efficientnet import preprocess_input
        import cv2

        # 1. CNN features
        cnn_raw = extract_single_cnn_feature(image_path, self.cnn_model)
        cnn_pca = self.pca.transform(cnn_raw.reshape(1, -1))[0, :self.n_pca]

        # 2. ABCD features
        abcd = extract_all_features(image_path)
        abcd.pop("image_path", None)
        abcd.pop("error", None)
        abcd_vec = np.array([float(abcd.get(c, 0.0)) for c in self.abcd_cols])

        # 3. Metadata
        meta = {"age": age if age is not None else 45.0}
        for s in ["male", "female", "unknown"]:
            meta[f"sex_{s}"] = 1.0 if sex == s else 0.0
        locs = ["abdomen","acral","back","chest","ear","face","foot",
                "genital","hand","lower extremity","neck","scalp",
                "trunk","unknown","upper extremity"]
        for loc in locs:
            meta[f"localization_{loc}"] = 1.0 if localization == loc else 0.0
        for dt in ["histo","follow_up","consensus","confocal"]:
            meta[f"dx_type_{dt}"] = 1.0 if dx_type == dt else 0.0
        meta_vec = np.array([float(meta.get(c, 0.0)) for c in self.meta_cols])

        # 4. Combine
        x = np.hstack([cnn_pca, abcd_vec, meta_vec]).reshape(1, -1)

        # 5. Predict
        proba    = self.model.predict_proba(x)[0]
        pred_idx = np.argmax(proba)
        pred_cls = self.label_encoder.inverse_transform([pred_idx])[0]
        pred_lbl = self.label_map.get(pred_cls, pred_cls)
        confidence = float(proba[pred_idx])

        malignant = {"mel", "bcc", "akiec"}
        risk = "HIGH — malignant" if pred_cls in malignant else "LOW — benign"

        all_classes = self.label_encoder.inverse_transform(range(len(proba)))
        class_probs = {
            self.label_map.get(c, c): round(float(p), 4)
            for c, p in zip(all_classes, proba)
        }

        return {
            "predicted_class":   pred_cls,
            "predicted_label":   pred_lbl,
            "confidence":        round(confidence, 4),
            "risk_level":        risk,
            "all_probabilities": class_probs,
            "abcd_features":     {k: v for k, v in abcd.items()},
        }
