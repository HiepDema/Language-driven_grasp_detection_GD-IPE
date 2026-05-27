"""
Visualization utilities for grasp rectangles.
"""

import math
import numpy as np
import cv2
from PIL import Image


def draw_grasp_rect(
    image: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    theta: float,
    color=(0, 255, 0),
    thickness: int = 2,
    normalized: bool = True,
) -> np.ndarray:
    """
    Draw a rotated grasp rectangle on an image.

    Args:
        image: HxWx3 BGR image
        x, y: center point (normalized [0,1] if normalized=True)
        w, h: width, height (normalized [0,1] if normalized=True)
        theta: rotation angle in radians
        color: BGR color tuple
        thickness: line thickness
        normalized: whether params are normalized to [0,1]

    Returns:
        Image with rectangle drawn
    """
    img = image.copy()
    img_h, img_w = img.shape[:2]

    if normalized:
        x, y = x * img_w, y * img_h
        w, h = w * img_w, h * img_h

    rect = ((x, y), (w, h), math.degrees(theta))
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(img, [box], 0, color, thickness)

    # Draw center point
    cv2.circle(img, (int(x), int(y)), 3, color, -1)

    return img


def visualize_prediction(
    image: np.ndarray,
    pred_params: tuple,
    gt_params: tuple = None,
    instruction: str = None,
    save_path: str = None,
) -> np.ndarray:
    """
    Visualize predicted (and optionally ground truth) grasp on image.

    Args:
        image: HxWx3 BGR image
        pred_params: (x, y, w, h, theta) predicted grasp
        gt_params: (x, y, w, h, theta) ground truth grasp (optional)
        instruction: text instruction to display
        save_path: path to save visualization

    Returns:
        Annotated image
    """
    vis = image.copy()

    # Draw ground truth in green
    if gt_params is not None:
        vis = draw_grasp_rect(vis, *gt_params, color=(0, 255, 0), thickness=2)

    # Draw prediction in red
    vis = draw_grasp_rect(vis, *pred_params, color=(0, 0, 255), thickness=2)

    # Add instruction text
    if instruction:
        text = instruction[:80]
        cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add legend
    cv2.putText(vis, "Pred", (10, vis.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    if gt_params is not None:
        cv2.putText(vis, "GT", (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if save_path:
        cv2.imwrite(save_path, vis)

    return vis
