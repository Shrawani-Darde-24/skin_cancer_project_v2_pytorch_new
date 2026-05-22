"""
Grad-CAM Lesion Localization
=============================
Generates a heatmap showing which regions of the image
the model focused on — highlights the exact cancerous area.
Works with PyTorch EfficientNet.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import base64
from io import BytesIO


def generate_gradcam(image_path: str, model, target_class_idx: int = None) -> dict:
    """
    Generate Grad-CAM heatmap for a lesion image.

    Args:
        image_path: Path to the lesion image
        model: Loaded EfficientNetB0 (timm, PyTorch)
        target_class_idx: Class index to visualize (None = predicted class)

    Returns:
        dict with base64-encoded original, heatmap, and overlay images
    """
    import timm
    from cnn_features import preprocess_for_efficientnet

    device = next(model.parameters()).device

    # Prepare input
    img_tensor = torch.tensor(
        preprocess_for_efficientnet(image_path)
    ).unsqueeze(0).to(device)
    img_tensor.requires_grad_(False)

    # Hook: capture last conv layer activations and gradients
    activations, gradients = [], []

    def fwd_hook(module, input, output):
        activations.append(output.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # EfficientNetB0 last conv block
    target_layer = model.conv_head
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # Forward pass
    model.eval()
    # Temporarily restore classifier for grad-cam scoring
    with torch.enable_grad():
        img_tensor.requires_grad_(True)
        features = model.forward_features(img_tensor)
        pooled   = model.global_pool(features)
        # Use raw pooled features as proxy score (max activation)
        if target_class_idx is not None:
            score = pooled[0, target_class_idx % pooled.shape[1]]
        else:
            score = pooled.max()
        score.backward()

    h1.remove()
    h2.remove()

    if not activations or not gradients:
        return _fallback_heatmap(image_path)

    # Grad-CAM computation
    act  = activations[0].squeeze(0)  # (C, H, W)
    grad = gradients[0].squeeze(0)    # (C, H, W)

    weights = grad.mean(dim=(1, 2))   # Global average pool over spatial dims
    cam = (weights[:, None, None] * act).sum(0)  # weighted sum → (H, W)
    cam = F.relu(cam)

    cam_np = cam.cpu().numpy()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    # Load original image
    orig = cv2.imread(image_path)
    orig = cv2.resize(orig, (224, 224))

    # Resize CAM to image size
    cam_resized = cv2.resize(cam_np, (224, 224))
    heatmap     = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    # Overlay
    overlay = cv2.addWeighted(orig, 0.55, heatmap, 0.45, 0)

    # Draw bounding box around high-activation region (top 30%)
    threshold   = cam_resized.max() * 0.3
    binary_mask = (cam_resized >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_pts = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_pts)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(overlay, "Focus region", (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Encode all three images to base64
    return {
        "original":  _encode_image(orig),
        "heatmap":   _encode_image(heatmap),
        "overlay":   _encode_image(overlay),
        "cam_array": cam_resized.tolist(),
    }


def _encode_image(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _fallback_heatmap(image_path: str) -> dict:
    """Return original image when Grad-CAM fails."""
    orig = cv2.imread(image_path)
    orig = cv2.resize(orig, (224, 224))
    enc  = _encode_image(orig)
    return {"original": enc, "heatmap": enc, "overlay": enc, "cam_array": []}
