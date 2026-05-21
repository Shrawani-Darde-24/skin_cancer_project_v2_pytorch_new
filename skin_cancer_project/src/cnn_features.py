"""
CNN Feature Extraction using EfficientNetB0 (PyTorch version)
==============================================================
Works on Python 3.14+ and Windows.

Install:
    pip install torch torchvision timm
"""

import os
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")


def load_efficientnet():
    """
    Load EfficientNetB0 pretrained on ImageNet via timm (PyTorch).
    Removes classification head — outputs 1280-dim feature vector.
    """
    import torch
    import timm

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"  EfficientNetB0 loaded (PyTorch) — device: {device.upper()}")
    return model


def preprocess_for_efficientnet(image_path: str) -> np.ndarray:
    """
    Load and preprocess image for EfficientNetB0.
    Returns numpy array (3, 224, 224) normalized for ImageNet.
    """
    import torch

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std

    # HWC → CHW
    img = img.transpose(2, 0, 1)
    return img.astype(np.float32)


def extract_cnn_features_batch(
    image_paths: list,
    model,
    batch_size: int = 32,
    verbose: bool = True
) -> np.ndarray:
    """
    Extract EfficientNet features for a list of image paths.
    Returns np.ndarray of shape (N, 1280)
    """
    import torch

    device = next(model.parameters()).device
    all_features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs  = []

        for path in batch_paths:
            try:
                img = preprocess_for_efficientnet(path)
            except Exception:
                img = np.zeros((3, 224, 224), dtype=np.float32)
            batch_imgs.append(img)

        batch_tensor = torch.tensor(np.stack(batch_imgs)).to(device)

        with torch.no_grad():
            feats = model(batch_tensor).cpu().numpy()

        all_features.append(feats)

        if verbose:
            done = min(i + batch_size, len(image_paths))
            print(f"  CNN features: {done}/{len(image_paths)} images", end="\r")

    if verbose:
        print()

    return np.vstack(all_features)


def extract_single_cnn_feature(image_path: str, model) -> np.ndarray:
    """
    Extract CNN features for a single image.
    Returns np.ndarray of shape (1280,)
    """
    import torch

    device = next(model.parameters()).device
    img = preprocess_for_efficientnet(image_path)
    tensor = torch.tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(tensor).cpu().numpy()[0]

    return feat
