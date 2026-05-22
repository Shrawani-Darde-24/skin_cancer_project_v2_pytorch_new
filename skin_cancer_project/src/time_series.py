"""
Time Series Analysis for Lesion Progression
=============================================
Tracks lesion changes across multiple visits using
ABCD feature deltas and trend analysis.
"""

import json
import os
import numpy as np
from datetime import datetime


HISTORY_FILE = "data/patient_history.json"


def load_history(patient_id: str, history_file: str = HISTORY_FILE) -> list:
    if not os.path.exists(history_file):
        return []
    with open(history_file) as f:
        db = json.load(f)
    return db.get(patient_id, [])


def save_visit(
    patient_id: str,
    features: dict,
    prediction: dict,
    visit_date: str = None,
    history_file: str = HISTORY_FILE
):
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    db = {}
    if os.path.exists(history_file):
        with open(history_file) as f:
            db = json.load(f)

    if patient_id not in db:
        db[patient_id] = []

    visit = {
        "date": visit_date or datetime.now().strftime("%Y-%m-%d"),
        "features": {k: float(v) for k, v in features.items()
                     if isinstance(v, (int, float))},
        "prediction": {
            "class": prediction.get("predicted_class"),
            "label": prediction.get("predicted_label"),
            "confidence": prediction.get("confidence"),
            "risk_level": prediction.get("risk_level"),
        }
    }
    db[patient_id].append(visit)

    with open(history_file, "w") as f:
        json.dump(db, f, indent=2)

    return visit


def analyze_progression(patient_id: str, history_file: str = HISTORY_FILE) -> dict:
    """
    Analyze how key ABCD features have changed over time.
    Returns trends, alerts, and summary statistics.
    """
    history = load_history(patient_id, history_file)

    if len(history) < 2:
        return {
            "patient_id": patient_id,
            "visits": len(history),
            "status": "insufficient_data",
            "message": "At least 2 visits needed for trend analysis.",
            "history": history,
        }

    dates  = [v["date"] for v in history]
    key_features = [
        "asymmetry_score", "border_irregularity",
        "color_entropy", "diameter_max_mm", "lesion_area_ratio"
    ]

    trends = {}
    alerts = []

    for feat in key_features:
        values = [v["features"].get(feat) for v in history]
        values = [x for x in values if x is not None]

        if len(values) < 2:
            continue

        delta      = values[-1] - values[0]
        pct_change = (delta / (values[0] + 1e-8)) * 100
        direction  = "increasing" if delta > 0 else "decreasing" if delta < 0 else "stable"

        # Alert thresholds
        alert = None
        if feat == "asymmetry_score" and delta > 0.05:
            alert = f"⚠ Asymmetry increased by {pct_change:.1f}% — monitor closely"
        elif feat == "border_irregularity" and delta > 0.05:
            alert = f"⚠ Border irregularity increased by {pct_change:.1f}%"
        elif feat == "color_entropy" and delta > 0.3:
            alert = f"⚠ Color variation increased significantly ({pct_change:.1f}%)"
        elif feat == "diameter_max_mm" and delta > 1.0:
            alert = f"⚠ Lesion diameter grew by {delta:.2f}mm — consult dermatologist"

        if alert:
            alerts.append(alert)

        trends[feat] = {
            "values": values,
            "dates": dates[:len(values)],
            "delta": round(delta, 4),
            "pct_change": round(pct_change, 1),
            "direction": direction,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "latest": round(values[-1], 4),
        }

    # Class changes
    classes = [v["prediction"]["class"] for v in history if v.get("prediction")]
    class_changed = len(set(classes)) > 1

    overall_risk = "stable"
    if alerts:
        overall_risk = "concerning"
    if class_changed and classes[-1] in {"mel", "bcc", "akiec"}:
        overall_risk = "high"
        alerts.insert(0, "⚠ Diagnosis changed to a malignant class — urgent review needed")

    return {
        "patient_id": patient_id,
        "visits": len(history),
        "dates": dates,
        "trends": trends,
        "alerts": alerts,
        "class_history": classes,
        "class_changed": class_changed,
        "overall_risk": overall_risk,
        "history": history,
    }
