"""
Predictor — Skin Cancer Classifier Inference
=============================================
Load a trained model bundle and classify new lesion images.

Usage:
    from predict import SkinCancerPredictor
    pred = SkinCancerPredictor("models/skin_cancer_model.pkl")
    result = pred.predict("my_lesion.jpg", age=45, sex="male", localization="back")
    print(result)
"""

import os
import pickle
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_all_features


class SkinCancerPredictor:
    """
    Wraps a trained model bundle for inference on new images.
    """

    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        self.model = bundle["model"]
        self.model_name = bundle["model_name"]
        self.label_encoder = bundle["label_encoder"]
        self.feature_cols = bundle["feature_cols"]
        self.label_map = bundle.get("label_map", {})

    def predict(
        self,
        image_path: str,
        age: float = None,
        sex: str = None,
        localization: str = None,
        dx_type: str = None,
        pixel_spacing_mm: float = 0.1,
    ) -> dict:
        """
        Classify a single lesion image.

        Args:
            image_path: Path to the lesion image
            age: Patient age (optional)
            sex: 'male' or 'female' (optional)
            localization: Body site e.g. 'back', 'face', 'scalp' (optional)
            dx_type: Diagnosis method e.g. 'histo', 'follow_up' (optional)
            pixel_spacing_mm: Pixel size in mm for diameter calculation

        Returns:
            dict with predicted class, probabilities, ABCD scores, risk level
        """
        # 1. Extract ABCD features
        abcd = extract_all_features(image_path, pixel_spacing_mm)
        abcd.pop("image_path", None)
        abcd.pop("error", None)

        # 2. Build metadata features (one-hot style to match training columns)
        meta = {
            "age": age if age is not None else 45.0,
        }

        # One-hot encode sex
        for s in ["male", "female", "unknown"]:
            meta[f"sex_{s}"] = 1.0 if sex == s else 0.0

        # One-hot encode localization
        locs = [
            "abdomen", "acral", "back", "chest", "ear", "face", "foot",
            "genital", "hand", "lower extremity", "neck", "scalp",
            "trunk", "unknown", "upper extremity"
        ]
        for loc in locs:
            meta[f"localization_{loc}"] = 1.0 if localization == loc else 0.0

        # One-hot encode dx_type
        for dt in ["histo", "follow_up", "consensus", "confocal"]:
            meta[f"dx_type_{dt}"] = 1.0 if dx_type == dt else 0.0

        # 3. Merge into row aligned with training feature columns
        all_feats = {**abcd, **meta}
        row = pd.DataFrame([{col: all_feats.get(col, 0.0) for col in self.feature_cols}])

        # 4. Predict
        proba = self.model.predict_proba(row)[0]
        pred_idx = np.argmax(proba)
        pred_class = self.label_encoder.inverse_transform([pred_idx])[0]
        pred_label = self.label_map.get(pred_class, pred_class)
        confidence = float(proba[pred_idx])

        # 5. Risk level
        malignant = {"mel", "bcc", "akiec"}
        risk = "HIGH — malignant" if pred_class in malignant else "LOW — benign"

        # 6. All class probabilities
        all_classes = self.label_encoder.inverse_transform(range(len(proba)))
        class_probs = {
            self.label_map.get(c, c): round(float(p), 4)
            for c, p in zip(all_classes, proba)
        }

        return {
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "risk_level": risk,
            "all_probabilities": class_probs,
            "abcd_features": {k: v for k, v in abcd.items()},
        }

    def predict_batch(self, image_paths: list, metadata: list = None) -> list:
        """
        Classify multiple images.

        Args:
            image_paths: List of image paths
            metadata: Optional list of dicts with keys age/sex/localization/dx_type

        Returns:
            List of result dicts
        """
        if metadata is None:
            metadata = [{}] * len(image_paths)

        results = []
        for path, meta in zip(image_paths, metadata):
            try:
                result = self.predict(path, **meta)
                result["image_path"] = path
                results.append(result)
            except Exception as e:
                results.append({"image_path": path, "error": str(e)})
        return results


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model.pkl> <image.jpg> [age] [sex] [localization]")
        sys.exit(1)

    predictor = SkinCancerPredictor(sys.argv[1])
    kwargs = {}
    if len(sys.argv) > 3: kwargs["age"] = float(sys.argv[3])
    if len(sys.argv) > 4: kwargs["sex"] = sys.argv[4]
    if len(sys.argv) > 5: kwargs["localization"] = sys.argv[5]

    result = predictor.predict(sys.argv[2], **kwargs)
    print(json.dumps(result, indent=2))
