"""
Parse grasp labels from .pt files into normalized 5-DoF grasp parameters.

The Grasp-Anything-pp dataset stores grasp labels as tensors. This module
converts them into the (x, y, w, h, theta) format our model predicts.

Returns ALL grasps per sample for proper multi-grasp loss and evaluation.
"""

import math
import torch
import numpy as np


def parse_grasp_label(label_tensor: torch.Tensor, image_size: int = 416) -> torch.Tensor:
    """
    Convert a raw grasp label tensor to normalized (x, y, w, h, theta).
    Returns ALL valid grasps for the sample.

    Args:
        label_tensor: raw label from .pt file
        image_size: image dimension for normalization

    Returns:
        [N, 5] tensor: N grasps, each (x, y, w, h, theta)
        where x, y, w, h in [0,1] and theta in radians [-pi/2, pi/2].
    """
    if not isinstance(label_tensor, torch.Tensor):
        label_tensor = torch.tensor(label_tensor, dtype=torch.float32)

    label = label_tensor.float()

    # Squeeze unnecessary batch dims
    while label.dim() > 2 and label.shape[0] == 1:
        label = label.squeeze(0)

    # [N, 6] format: (quality, x, y, w, h, angle_deg) — most common in our dataset
    if label.dim() == 2 and label.shape[-1] == 6:
        grasps = []
        for i in range(label.shape[0]):
            grasps.append(_normalize_params_with_deg_angle(label[i, 1:], image_size))
        return torch.stack(grasps)

    # [1, 6] squeezed to [6]
    if label.dim() == 1 and label.shape[0] == 6:
        return _normalize_params_with_deg_angle(label[1:], image_size).unsqueeze(0)

    # [N, 5]
    if label.dim() == 2 and label.shape[-1] == 5:
        grasps = []
        for i in range(label.shape[0]):
            grasps.append(_normalize_params(label[i], image_size))
        return torch.stack(grasps)

    # [5]
    if label.dim() == 1 and label.shape[0] == 5:
        return _normalize_params(label, image_size).unsqueeze(0)

    # [4, 2] single set of corners
    if label.dim() == 2 and label.shape == (4, 2):
        return _corners_to_params(label, image_size).unsqueeze(0)

    # [N, 4, 2] multiple corner sets
    if label.dim() == 3 and label.shape[1:] == (4, 2):
        grasps = []
        for i in range(label.shape[0]):
            grasps.append(_corners_to_params(label[i], image_size))
        return torch.stack(grasps)

    # Fallback: [N, 4] as (x1, y1, x2, y2)
    if label.dim() == 2 and label.shape[-1] == 4:
        grasps = []
        for i in range(label.shape[0]):
            box = label[i]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            cx = float((x1 + x2) / 2.0 / image_size)
            cy = float((y1 + y2) / 2.0 / image_size)
            w = float(abs(x2 - x1) / image_size)
            h = float(abs(y2 - y1) / image_size)
            grasps.append(torch.tensor([cx, cy, w, h, 0.0], dtype=torch.float32))
        return torch.stack(grasps)

    # Last resort
    flat = label.flatten()
    if flat.shape[0] >= 5:
        return _normalize_params(flat[:5], image_size).unsqueeze(0)

    return torch.zeros(1, 5, dtype=torch.float32)


def _normalize_params_with_deg_angle(params: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    Normalize (x, y, w, h, angle_deg) from pixel space.
    Angle is expected in degrees [0, 180) and converted to radians [-pi/2, pi/2].
    """
    x, y, w, h, angle_deg = params[0], params[1], params[2], params[3], params[4]

    x = x / image_size
    y = y / image_size
    w = w / image_size
    h = h / image_size

    # Convert degrees [0, 180) to radians [-pi/2, pi/2]
    angle_rad = float(angle_deg)
    if angle_rad > 90.0:
        angle_rad = angle_rad - 180.0
    angle_rad = math.radians(angle_rad)

    x = torch.clamp(x, 0.0, 1.0)
    y = torch.clamp(y, 0.0, 1.0)
    w = torch.clamp(w, 0.0, 1.0)
    h = torch.clamp(h, 0.0, 1.0)

    return torch.tensor([float(x), float(y), float(w), float(h), angle_rad], dtype=torch.float32)


def _normalize_params(params: torch.Tensor, image_size: int) -> torch.Tensor:
    """Normalize pixel-space params to [0,1] range (except angle stays in radians)."""
    x, y, w, h, theta = params[0], params[1], params[2], params[3], params[4]

    if x > 1.0 or y > 1.0:
        x = x / image_size
        y = y / image_size
    if w > 1.0 or h > 1.0:
        w = w / image_size
        h = h / image_size

    x = torch.clamp(x, 0.0, 1.0)
    y = torch.clamp(y, 0.0, 1.0)
    w = torch.clamp(w, 0.0, 1.0)
    h = torch.clamp(h, 0.0, 1.0)

    if abs(theta) > math.pi:
        theta = torch.tensor(math.radians(float(theta)), dtype=torch.float32)

    return torch.tensor([float(x), float(y), float(w), float(h), float(theta)], dtype=torch.float32)


def _corners_to_params(corners: torch.Tensor, image_size: int) -> torch.Tensor:
    """Convert 4 corner points [4, 2] to (x, y, w, h, theta) normalized."""
    pts = corners.numpy().astype(np.float32)

    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()

    w = np.sqrt((pts[1, 0] - pts[0, 0]) ** 2 + (pts[1, 1] - pts[0, 1]) ** 2)
    h = np.sqrt((pts[2, 0] - pts[1, 0]) ** 2 + (pts[2, 1] - pts[1, 1]) ** 2)

    theta = math.atan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0])

    cx /= image_size
    cy /= image_size
    w /= image_size
    h /= image_size

    return torch.tensor([cx, cy, w, h, theta], dtype=torch.float32)
