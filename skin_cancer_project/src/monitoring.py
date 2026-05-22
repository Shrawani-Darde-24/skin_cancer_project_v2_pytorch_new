"""
AI Monitoring — Recurring Skin Abnormality Tracker
====================================================
Automatically detects and flags recurring or worsening
abnormalities across multiple patient visits using:
  - ABCD feature trend analysis
  - Diagnosis consistency scoring
  - Anomaly detection on feature trajectories
  - Recurrence pattern recognition

Usage:
    from monitoring import AbnormalityMonitor
    monitor = AbnormalityMonitor()
    report  = monitor.analyze(patient_id)
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta


HISTORY_FILE = "data/patient_history.json"

MONITORED_FEATURES = [
    "asymmetry_score",
    "border_irregularity",
    "color_entropy",
    "diameter_max_mm",
    "lesion_area_ratio",
    "color_h_std",
    "color_variance",
    "contour_roughness",
]

MALIGNANT = {"mel", "bcc", "akiec"}

# Thresholds — if feature changes by more than this between visits → flag
CHANGE_THRESHOLDS = {
    "asymmetry_score":     0.05,
    "border_irregularity": 0.05,
    "color_entropy":       0.30,
    "diameter_max_mm":     1.00,
    "lesion_area_ratio":   0.03,
    "color_h_std":         5.00,
    "color_variance":      200.0,
    "contour_roughness":   0.10,
}


class AbnormalityMonitor:

    def __init__(self, history_file: str = HISTORY_FILE):
        self.history_file = history_file

    def _load(self, patient_id: str) -> list:
        if not os.path.exists(self.history_file):
            return []
        with open(self.history_file) as f:
            db = json.load(f)
        return db.get(patient_id, [])

    def analyze(self, patient_id: str) -> dict:
        """
        Full AI monitoring analysis for a patient.
        Returns structured report with flags, trends, and recurrence score.
        Called automatically from app.py after every visit is saved.
        """
        history = self._load(patient_id)

        if len(history) == 0:
            return self._empty_report(patient_id, "No visit history found.")
        if len(history) == 1:
            return self._single_visit_report(patient_id, history[0])

        return self._full_analysis(patient_id, history)

    def _full_analysis(self, patient_id: str, history: list) -> dict:
        dates       = [v["date"] for v in history]
        flags       = []
        trends      = {}
        recurrence  = []

        # ── 1. Feature trend analysis ────────────────────────────────────────
        for feat in MONITORED_FEATURES:
            values = []
            for visit in history:
                val = visit.get("features", {}).get(feat)
                if val is not None:
                    values.append(float(val))

            if len(values) < 2:
                continue

            delta        = values[-1] - values[0]
            pct_change   = (delta / (abs(values[0]) + 1e-8)) * 100
            threshold    = CHANGE_THRESHOLDS.get(feat, 0.1)
            consecutive  = self._count_consecutive_increase(values)
            direction    = "increasing" if delta > 0 else "decreasing" if delta < 0 else "stable"

            # Anomaly: sudden spike between last two visits
            recent_delta = values[-1] - values[-2] if len(values) >= 2 else 0
            is_anomaly   = abs(recent_delta) > threshold * 2

            # Flag if worsening
            flag = None
            if feat in ("asymmetry_score", "border_irregularity", "color_entropy",
                        "diameter_max_mm", "lesion_area_ratio") and delta > threshold:
                severity = "critical" if delta > threshold * 3 else "warning"
                flag = {
                    "feature":    feat,
                    "severity":   severity,
                    "message":    f"{feat.replace('_',' ').title()} increased by {pct_change:.1f}% over {len(history)} visits",
                    "delta":      round(delta, 4),
                    "is_anomaly": is_anomaly,
                }
                flags.append(flag)

            trends[feat] = {
                "values":                values,
                "dates":                 dates[:len(values)],
                "delta":                 round(delta, 4),
                "pct_change":            round(pct_change, 1),
                "direction":             direction,
                "consecutive_increases": consecutive,
                "is_anomaly":            is_anomaly,
                "latest":                round(values[-1], 4),
            }

        # ── 2. Diagnosis consistency check ───────────────────────────────────
        classes = [v.get("prediction", {}).get("class") for v in history if v.get("prediction")]
        classes = [c for c in classes if c]

        diagnosis_changed    = len(set(classes)) > 1
        malignant_detections = sum(1 for c in classes if c in MALIGNANT)
        latest_class         = classes[-1] if classes else None
        malignant_now        = latest_class in MALIGNANT

        if diagnosis_changed:
            if malignant_now:
                flags.append({
                    "feature":    "diagnosis",
                    "severity":   "critical",
                    "message":    f"Diagnosis changed to malignant class ({latest_class}) — immediate review required",
                    "delta":      None,
                    "is_anomaly": True,
                })
            else:
                flags.append({
                    "feature":    "diagnosis",
                    "severity":   "warning",
                    "message":    f"Diagnosis has changed across {len(history)} visits — monitor closely",
                    "delta":      None,
                    "is_anomaly": False,
                })

        # ── 3. Recurrence pattern detection ──────────────────────────────────
        if len(classes) >= 3:
            for cls in set(classes):
                indices = [i for i, c in enumerate(classes) if c == cls]
                if len(indices) >= 2:
                    recurrence.append({
                        "class":        cls,
                        "occurrences":  len(indices),
                        "visits":       [dates[i] for i in indices if i < len(dates)],
                        "is_malignant": cls in MALIGNANT,
                    })

        # ── 4. Recurrence score (0–1) ─────────────────────────────────────────
        recurrence_score = self._compute_recurrence_score(
            flags, trends, malignant_detections, len(history)
        )

        # ── 5. AI recommendation ──────────────────────────────────────────────
        recommendation = self._generate_recommendation(
            recurrence_score, flags, malignant_now, len(history)
        )

        return {
            "patient_id":           patient_id,
            "visits":               len(history),
            "dates":                dates,
            "status":               "analyzed",
            "flags":                flags,
            "trends":               trends,
            "recurrence_patterns":  recurrence,
            "recurrence_score":     recurrence_score,
            "diagnosis_changed":    diagnosis_changed,
            "malignant_detections": malignant_detections,
            "latest_class":         latest_class,
            "malignant_now":        malignant_now,
            "recommendation":       recommendation,
            "critical_count":       sum(1 for f in flags if f["severity"] == "critical"),
            "warning_count":        sum(1 for f in flags if f["severity"] == "warning"),
        }

    def _count_consecutive_increase(self, values: list) -> int:
        count = 0
        for i in range(len(values)-1, 0, -1):
            if values[i] > values[i-1]:
                count += 1
            else:
                break
        return count

    def _compute_recurrence_score(self, flags, trends, malignant_count, visit_count) -> float:
        score = 0.0
        critical_flags = sum(1 for f in flags if f["severity"] == "critical")
        warning_flags  = sum(1 for f in flags if f["severity"] == "warning")
        score += critical_flags * 0.25
        score += warning_flags  * 0.10
        score += (malignant_count / max(visit_count, 1)) * 0.30

        # Bonus for consistent multi-visit increases in key features
        for feat in ("asymmetry_score", "border_irregularity", "diameter_max_mm"):
            if feat in trends and trends[feat]["consecutive_increases"] >= 2:
                score += 0.10

        return round(min(score, 1.0), 3)

    def _generate_recommendation(self, score, flags, malignant_now, visit_count) -> dict:
        if malignant_now or score >= 0.75:
            return {
                "urgency":  "URGENT",
                "action":   "Immediate dermatologist consultation required. Do not delay.",
                "interval": "Within 1 week",
                "color":    "danger",
            }
        elif score >= 0.50:
            return {
                "urgency":  "HIGH",
                "action":   "Schedule a dermatology appointment as soon as possible.",
                "interval": "Within 2–4 weeks",
                "color":    "warning",
            }
        elif score >= 0.25:
            return {
                "urgency":  "MODERATE",
                "action":   "Continue monitoring. Book a routine dermatology review.",
                "interval": "Within 1–3 months",
                "color":    "caution",
            }
        else:
            return {
                "urgency":  "LOW",
                "action":   "Continue regular self-examinations and annual checkups.",
                "interval": "6–12 months",
                "color":    "safe",
            }

    def _empty_report(self, patient_id, message):
        return {
            "patient_id":       patient_id,
            "status":           "no_data",
            "message":          message,
            "flags":            [],
            "recurrence_score": 0.0,
            "recommendation":   None,
        }

    def _single_visit_report(self, patient_id, visit):
        cls = visit.get("prediction", {}).get("class", "")
        return {
            "patient_id":       patient_id,
            "status":           "single_visit",
            "visits":           1,
            "flags":            [],
            "recurrence_score": 0.0,
            "malignant_now":    cls in MALIGNANT,
            "latest_class":     cls,
            "message":          "Only one visit recorded — need 2+ visits for trend analysis.",
            "recommendation": {
                "urgency":  "LOW",
                "action":   "Come back for a follow-up to enable trend monitoring.",
                "interval": "3 months",
                "color":    "safe",
            }
        }
