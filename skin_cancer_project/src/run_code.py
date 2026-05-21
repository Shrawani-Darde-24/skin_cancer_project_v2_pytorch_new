import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from predict_v2 import SkinCancerPredictorV2

# ── Config ──────────────────────────────────────
MODEL_PATH = "models/skin_cancer_model_v2.pkl"
IMAGE_PATH = input("Enter image path: ").strip().strip('"')

# Optional metadata (press Enter to skip)
age          = input("Age (or Enter to skip): ").strip()
sex          = input("Sex [male/female/unknown] (or Enter to skip): ").strip()
localization = input("Localization e.g. back/face/arm (or Enter to skip): ").strip()

age = float(age) if age else None
sex = sex if sex else None
localization = localization if localization else None

# ── Predict ─────────────────────────────────────
print("\nLoading model...")
predictor = SkinCancerPredictorV2(MODEL_PATH)

print("\nRunning prediction...")
result = predictor.predict(IMAGE_PATH, age=age, sex=sex, localization=localization)

# ── Output ──────────────────────────────────────
print("\n" + "="*50)
print(f"  Predicted:  {result['predicted_label']}")
print(f"  Confidence: {result['confidence']*100:.1f}%")
print(f"  Risk Level: {result['risk_level']}")
print("\n  All Probabilities:")
for label, prob in sorted(result['all_probabilities'].items(), key=lambda x: -x[1]):
    print(f"    {label:<30} {prob*100:.1f}%")
print("="*50)