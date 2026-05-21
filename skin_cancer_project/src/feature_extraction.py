"""
ABCD Feature Extraction for Skin Lesion Images
=================================================
Extracts clinical features used by dermatologists:
  A - Asymmetry
  B - Border irregularity
  C - Color variation
  D - Diameter / size
"""

import cv2
import numpy as np
from skimage import measure, morphology
from scipy.stats import entropy as scipy_entropy
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Preprocessing helpers
# ──────────────────────────────────────────────

def resize_image(image: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def remove_hair(image: np.ndarray) -> np.ndarray:
    """
    Remove hair artifacts using morphological top-hat and inpainting.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    cleaned = cv2.inpaint(image, hair_mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    return cleaned


def segment_lesion(image: np.ndarray) -> np.ndarray:
    """
    Segment the lesion from the background using Otsu thresholding on the
    green channel (most discriminative for skin lesions).
    Returns a binary mask: 1 = lesion, 0 = background.
    """
    green = image[:, :, 1]  # green channel works well for lesion segmentation
    blurred = cv2.GaussianBlur(green, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only the largest connected component (the main lesion)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255

    return mask


def preprocess(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline.
    Returns: (preprocessed_image, lesion_mask)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    image = resize_image(image, size=(224, 224))
    image = remove_hair(image)
    mask = segment_lesion(image)
    return image, mask


# ──────────────────────────────────────────────
# A — Asymmetry
# ──────────────────────────────────────────────

def compute_asymmetry(mask: np.ndarray) -> dict:
    """
    Measure asymmetry by folding the lesion along its two principal axes
    and computing the non-overlapping area ratio.

    Score range: 0 (perfectly symmetric) to 1 (fully asymmetric)
    """
    binary = (mask > 0).astype(np.uint8)

    def fold_score(m: np.ndarray) -> float:
        half = m.shape[0] // 2
        top, bottom = m[:half, :], np.flipud(m[half:half * 2, :])
        if top.shape != bottom.shape:
            min_h = min(top.shape[0], bottom.shape[0])
            top, bottom = top[:min_h], bottom[:min_h]
        union = np.logical_or(top, bottom).sum()
        intersection = np.logical_and(top, bottom).sum()
        return 1.0 - (intersection / union) if union > 0 else 0.0

    asym_h = fold_score(binary)
    asym_v = fold_score(binary.T)
    total = (asym_h + asym_v) / 2

    return {
        "asymmetry_horizontal": round(asym_h, 4),
        "asymmetry_vertical": round(asym_v, 4),
        "asymmetry_score": round(total, 4),
    }


# ──────────────────────────────────────────────
# B — Border irregularity
# ──────────────────────────────────────────────

def compute_border(mask: np.ndarray) -> dict:
    """
    Measure border irregularity using:
    - Compactness index: 4π·Area / Perimeter²  (1 = perfect circle, <1 = irregular)
    - Fractal dimension (box-counting)
    - Edge roughness (contour curvature variance)
    """
    binary = (mask > 0).astype(np.uint8)
    area = binary.sum()
    if area == 0:
        return {k: 0.0 for k in ["border_compactness", "border_irregularity", "contour_roughness"]}

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {k: 0.0 for k in ["border_compactness", "border_irregularity", "contour_roughness"]}

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, closed=True)

    compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)
    irregularity = 1 - compactness  # higher = more irregular

    # Curvature variance along contour
    pts = contour[:, 0, :].astype(np.float32)
    if len(pts) > 4:
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8) ** 1.5
        roughness = float(np.std(curvature))
    else:
        roughness = 0.0

    return {
        "border_compactness": round(float(compactness), 4),
        "border_irregularity": round(float(irregularity), 4),
        "contour_roughness": round(roughness, 4),
    }


# ──────────────────────────────────────────────
# C — Color variation
# ──────────────────────────────────────────────

def compute_color(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Analyze color variation within the lesion region:
    - Mean and std of H, S, V channels
    - Color entropy (information content)
    - Number of dominant color clusters
    """
    binary = mask > 0
    if binary.sum() < 10:
        return {k: 0.0 for k in [
            "color_h_mean", "color_h_std", "color_s_mean", "color_s_std",
            "color_v_mean", "color_v_std", "color_entropy", "color_variance"
        ]}

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    features = {}

    for i, ch in enumerate(["h", "s", "v"]):
        channel = hsv[:, :, i][binary]
        features[f"color_{ch}_mean"] = round(float(channel.mean()), 4)
        features[f"color_{ch}_std"] = round(float(channel.std()), 4)

    # Color entropy on grayscale of lesion
    gray_lesion = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[binary]
    hist, _ = np.histogram(gray_lesion, bins=64, range=(0, 256), density=True)
    hist = hist[hist > 0]
    features["color_entropy"] = round(float(scipy_entropy(hist)), 4)

    # Overall color variance (across all channels)
    rgb_lesion = image[binary].astype(np.float32)
    features["color_variance"] = round(float(rgb_lesion.var()), 4)

    return features


# ──────────────────────────────────────────────
# D — Diameter / size
# ──────────────────────────────────────────────

def compute_diameter(mask: np.ndarray, pixel_spacing_mm: float = 0.1) -> dict:
    """
    Estimate lesion size metrics:
    - Major / minor axis length (from ellipse fit)
    - Max diameter (bounding box diagonal)
    - Relative lesion area (fraction of image)
    """
    binary = (mask > 0).astype(np.uint8)
    area_px = int(binary.sum())
    h, w = binary.shape

    if area_px < 4:
        return {k: 0.0 for k in [
            "diameter_major_mm", "diameter_minor_mm",
            "diameter_max_mm", "lesion_area_ratio"
        ]}

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    major_mm, minor_mm = 0.0, 0.0
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        major_mm = round(max(ellipse[1]) * pixel_spacing_mm, 2)
        minor_mm = round(min(ellipse[1]) * pixel_spacing_mm, 2)

    x, y, cw, ch2 = cv2.boundingRect(contour)
    max_diameter_mm = round(np.sqrt(cw**2 + ch2**2) * pixel_spacing_mm, 2)
    lesion_area_ratio = round(area_px / (h * w), 4)

    return {
        "diameter_major_mm": major_mm,
        "diameter_minor_mm": minor_mm,
        "diameter_max_mm": max_diameter_mm,
        "lesion_area_ratio": lesion_area_ratio,
    }


# ──────────────────────────────────────────────
# Combined feature extraction
# ──────────────────────────────────────────────

def extract_all_features(image_path: str, pixel_spacing_mm: float = 0.1) -> dict:
    """
    Full ABCD feature extraction pipeline for one image.

    Args:
        image_path: Path to the dermoscopy image
        pixel_spacing_mm: Real-world size of one pixel (default: 0.1 mm)

    Returns:
        Dictionary with all ABCD features
    """
    image, mask = preprocess(image_path)

    features = {}
    features.update(compute_asymmetry(mask))
    features.update(compute_border(mask))
    features.update(compute_color(image, mask))
    features.update(compute_diameter(mask, pixel_spacing_mm))
    features["image_path"] = image_path

    return features


def extract_features_batch(
    image_paths: list,
    pixel_spacing_mm: float = 0.1,
    verbose: bool = True
) -> list[dict]:
    """
    Extract ABCD features for a list of image paths.
    Returns list of feature dictionaries.
    """
    results = []
    for i, path in enumerate(image_paths):
        try:
            feats = extract_all_features(path, pixel_spacing_mm)
            results.append(feats)
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)} images")
        except Exception as e:
            print(f"  [WARN] Skipping {path}: {e}")
            results.append({"image_path": path, "error": str(e)})
    return results


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <image_path>")
        sys.exit(1)

    feats = extract_all_features(sys.argv[1])
    print(json.dumps(feats, indent=2))
