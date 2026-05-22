"""
Heatmap Generator — High-Risk Skin Cancer Regions
===================================================
Generates multi-layer risk heatmaps on lesion images showing:
  - Overall risk intensity map
  - ABCD feature overlay (color-coded per zone)
  - High-risk region bounding boxes with risk score labels

Usage:
    from heatmap import generate_risk_heatmap
    result = generate_risk_heatmap("lesion.jpg", abcd_features, predicted_class)
"""

import cv2
import numpy as np
import base64


# Risk weights per ABCD feature (how much each contributes to heatmap intensity)
FEATURE_WEIGHTS = {
    "asymmetry_score":       0.30,
    "border_irregularity":   0.25,
    "color_entropy":         0.25,
    "lesion_area_ratio":     0.20,
}

MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}

RISK_COLORS = {
    "extreme": (0,   0,   220),   # BGR red
    "high":    (0,   100, 255),   # BGR orange
    "moderate":(0,   200, 255),   # BGR yellow
    "low":     (100, 220, 100),   # BGR green
}


def _encode(img: np.ndarray) -> str:
    """Encode a cv2 image to a base64 data URI for the frontend."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _compute_risk_score(abcd: dict, predicted_class: str) -> float:
    """
    Compute a 0–1 overall risk score from ABCD features + class.
    Malignant class predictions receive a 1.35× boost (capped at 1.0).
    """
    score = 0.0
    for feat, weight in FEATURE_WEIGHTS.items():
        val = abcd.get(feat, 0.0)
        # Normalize each feature to 0–1 range
        if feat == "asymmetry_score":
            norm = min(val / 0.5, 1.0)
        elif feat == "border_irregularity":
            norm = min(val / 0.8, 1.0)
        elif feat == "color_entropy":
            norm = min(val / 5.0, 1.0)
        elif feat == "lesion_area_ratio":
            norm = min(val / 0.4, 1.0)
        else:
            norm = min(float(val), 1.0)
        score += norm * weight

    # Boost score if malignant class predicted
    if predicted_class in MALIGNANT_CLASSES:
        score = min(score * 1.35, 1.0)

    return round(score, 3)


def _risk_level(score: float) -> str:
    if score >= 0.75: return "extreme"
    if score >= 0.50: return "high"
    if score >= 0.25: return "moderate"
    return "low"


def _build_abcd_heatmap(image: np.ndarray, mask: np.ndarray, abcd: dict):
    """
    Build a spatial heatmap by weighting pixel intensities using ABCD scores.
    Each ABCD component contributes a different spatial pattern:
      A (Asymmetry)   → non-overlapping half-flip regions
      B (Border)      → edge pixels from erosion
      C (Color)       → high-saturation pixels within lesion
      D (Diameter)    → distance transform from lesion centre
    """
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    binary = (mask > 0).astype(np.uint8)

    # A — Asymmetry: XOR of top vs flipped-bottom half
    half   = binary.shape[0] // 2
    top    = binary[:half, :]
    bottom = np.flipud(binary[half:half*2, :])
    min_h  = min(top.shape[0], bottom.shape[0])
    asym_mask = np.logical_xor(top[:min_h], bottom[:min_h]).astype(np.float32)
    asym_full = np.zeros_like(heatmap)
    asym_full[:min_h, :] = asym_mask
    heatmap += asym_full * abcd.get("asymmetry_score", 0) * FEATURE_WEIGHTS["asymmetry_score"]

    # B — Border: morphological edge pixels
    kernel  = np.ones((5, 5), np.uint8)
    eroded  = cv2.erode(binary, kernel, iterations=2)
    border  = (binary - eroded).astype(np.float32)
    heatmap += border * abcd.get("border_irregularity", 0) * FEATURE_WEIGHTS["border_irregularity"]

    # C — Color: HSV saturation within lesion
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    sat[mask == 0] = 0
    heatmap += sat * abcd.get("color_entropy", 0) / 5.0 * FEATURE_WEIGHTS["color_entropy"]

    # D — Diameter: distance transform (highlights lesion centre)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = dist / dist.max()
    heatmap += dist * abcd.get("lesion_area_ratio", 0) * FEATURE_WEIGHTS["lesion_area_ratio"]

    # Normalize to 0–255
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    return heatmap_uint8, heatmap


def generate_risk_heatmap(
    image_path: str,
    abcd_features: dict,
    predicted_class: str,
    pixel_spacing_mm: float = 0.1,
) -> dict:
    """
    Generate full risk heatmap analysis for a lesion image.

    Args:
        image_path:       Path to the lesion image file.
        abcd_features:    Dict of ABCD scores from the predictor.
        predicted_class:  Short class code e.g. "mel", "nv".
        pixel_spacing_mm: Physical pixel size (unused here, reserved).

    Returns dict with:
        original      — base64 original image (320×320)
        heatmap_only  — base64 pure JET-colormap heatmap
        overlay       — base64 heatmap blended on original (50/50)
        annotated     — base64 image with bounding boxes + risk labels
        risk_score    — float 0–1 overall risk
        risk_level    — str  extreme / high / moderate / low
        risk_zones    — list of dicts {zone_id, x, y, w, h, score, risk_level}
        zone_count    — int  number of detected zones
    """
    # Load + resize to fixed canvas
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    img = cv2.resize(img, (320, 320))

    # Lesion segmentation via Otsu threshold on green channel
    gray = img[:, :, 1]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Overall risk score
    risk_score = _compute_risk_score(abcd_features, predicted_class)
    level      = _risk_level(risk_score)

    # Build ABCD spatial heatmap
    heatmap_raw, heatmap_float = _build_abcd_heatmap(img, mask, abcd_features)

    # Apply JET colormap and blend with original
    colored_heatmap = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.50, colored_heatmap, 0.50, 0)

    # Detect high-risk zones (regions above 60% of peak intensity)
    threshold   = heatmap_float.max() * 0.60
    binary_risk = (heatmap_float >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_risk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated  = overlay.copy()
    risk_zones = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 100:          # skip tiny noise blobs
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        zone_score  = float(heatmap_float[y:y+h, x:x+w].mean())
        zone_level  = _risk_level(zone_score * 2)   # amplified for local zones
        color       = RISK_COLORS.get(zone_level, RISK_COLORS["moderate"])

        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        label = f"Zone {i+1}: {zone_level.upper()}"
        cv2.putText(annotated, label, (x, max(y-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        risk_zones.append({
            "zone_id":    i + 1,
            "x": x, "y": y, "w": w, "h": h,
            "score":      round(zone_score, 3),
            "risk_level": zone_level,
        })

    # Overall risk badge on annotated image
    badge_color = RISK_COLORS.get(level, RISK_COLORS["low"])
    cv2.rectangle(annotated, (4, 4), (220, 26), (20, 20, 20), -1)
    cv2.putText(annotated, f"Overall risk: {level.upper()} ({risk_score:.2f})",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_color, 1, cv2.LINE_AA)

    return {
        "original":       _encode(img),
        "heatmap_only":   _encode(colored_heatmap),
        "overlay":        _encode(overlay),
        "annotated":      _encode(annotated),
        "risk_score":     risk_score,
        "risk_level":     level,
        "risk_zones":     risk_zones,
        "zone_count":     len(risk_zones),
    }
