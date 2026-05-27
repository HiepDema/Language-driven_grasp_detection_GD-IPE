"""
Parse grasp labels from .pt files into normalized 5-DoF grasp parameters.

The Grasp-Anything-pp dataset stores grasp labels as tensors. This module
converts them into the (x, y, w, h, theta) format our model predicts.
"""

import math
import torch
import numpy as np


def parse_grasp_label(label_tensor: torch.Tensor, image_size: int = 416) -> torch.Tensor:
    """
    Convert a raw grasp label tensor to normalized (x, y, w, h, theta).

    Handles common label formats in Grasp-Anything-pp:
    - [N, 5]: multiple grasps as (x, y, w, h, theta) already in pixel coords
    - [N, 4, 2]: multiple grasps as 4 corner points
    - [5]: single grasp as (x, y, w, h, theta)
    - [4, 2]: single grasp as 4 corner points

    Args:
        label_tensor: raw label from .pt file
        image_size: image dimension for normalization

    Returns:
        [5] tensor: (x, y, w, h, theta) normalized to [0,1] for x,y,w,h
        and radians for theta. If multiple grasps, returns the first one.
    """
    if not isinstance(label_tensor, torch.Tensor):
        label_tensor = torch.tensor(label_tensor, dtype=torch.float32)

    label = label_tensor.float()

    # Squeeze unnecessary batch dims
    while label.dim() > 2 and label.shape[0] == 1:
        label = label.squeeze(0)

    if label.dim() == 1 and label.shape[0] == 5:
        # Already (x, y, w, h, theta) - just normalize
        return _normalize_params(label, image_size)

    if label.dim() == 1 and label.shape[0] == 6:
        # Format: (quality, x, y, w, h, angle_deg)
        # Skip quality (col 0), take cols 1-5, convert angle from degrees to radians
        return _normalize_params_with_deg_angle(label[1:], image_size)

    if label.dim() == 2 and label.shape[-1] == 6:
        # [N, 6] format: each row is (quality, x, y, w, h, angle_deg)
        # Take the best grasp (first row, highest quality), skip quality column
        return _normalize_params_with_deg_angle(label[0, 1:], image_size)

    if label.dim() == 2 and label.shape[-1] == 5:
        # [N, 5] - take first grasp
        return _normalize_params(label[0], image_size)

    if label.dim() == 2 and label.shape == (4, 2):
        # 4 corner points
        return _corners_to_params(label, image_size)

    if label.dim() == 3 and label.shape[1:] == (4, 2):
        # [N, 4, 2] - take first grasp
        return _corners_to_params(label[0], image_size)

    # Fallback: if shape is [N, 4] interpret as (x1, y1, x2, y2) bbox
    if label.dim() == 2 and label.shape[-1] == 4:
        box = label[0] if label.dim() == 2 else label
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        cx = (x1 + x2) / 2.0 / image_size
        cy = (y1 + y2) / 2.0 / image_size
        w = abs(x2 - x1) / image_size
        h = abs(y2 - y1) / image_size
        return torch.tensor([cx, cy, w, h, 0.0], dtype=torch.float32)

    # Last resort: try to reshape
    flat = label.flatten()
    if flat.shape[0] >= 5:
        return _normalize_params(flat[:5], image_size)

    # Cannot parse - return zeros
    return torch.zeros(5, dtype=torch.float32)


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

    # If values suggest pixel coordinates (> 1), normalize
    if x > 1.0 or y > 1.0:
        x = x / image_size
        y = y / image_size
    if w > 1.0 or h > 1.0:
        w = w / image_size
        h = h / image_size

    # Clamp to valid range
    x = torch.clamp(x, 0.0, 1.0)
    y = torch.clamp(y, 0.0, 1.0)
    w = torch.clamp(w, 0.0, 1.0)
    h = torch.clamp(h, 0.0, 1.0)

    # Ensure theta is in radians [-pi/2, pi/2]
    if abs(theta) > math.pi:
        theta = torch.tensor(math.radians(float(theta)), dtype=torch.float32)

    return torch.tensor([float(x), float(y), float(w), float(h), float(theta)], dtype=torch.float32)


def _corners_to_params(corners: torch.Tensor, image_size: int) -> torch.Tensor:
    """Convert 4 corner points [4, 2] to (x, y, w, h, theta) normalized."""
    pts = corners.numpy().astype(np.float32)

    # Center
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()

    # Width: distance between first two points (top edge)
    w = np.sqrt((pts[1, 0] - pts[0, 0]) ** 2 + (pts[1, 1] - pts[0, 1]) ** 2)
    # Height: distance between second and third points (side edge)
    h = np.sqrt((pts[2, 0] - pts[1, 0]) ** 2 + (pts[2, 1] - pts[1, 1]) ** 2)

    # Angle from top edge
    theta = math.atan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0])

    # Normalize
    cx /= image_size
    cy /= image_size
    w /= image_size
    h /= image_size

    return torch.tensor([cx, cy, w, h, theta], dtype=torch.float32)
