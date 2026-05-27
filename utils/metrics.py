"""
Evaluation metrics for grasp pose prediction.
"""

import math
import numpy as np
import torch


def compute_angle_diff(pred_angle: float, gt_angle: float) -> float:
    """Compute minimal angular difference in degrees, handling periodicity."""
    diff = abs(pred_angle - gt_angle) % 180.0
    return min(diff, 180.0 - diff)


def grasp_rect_vertices(x, y, w, h, theta):
    """Compute 4 corner vertices of a rotated rectangle."""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = w / 2
    dy = h / 2
    corners = [
        (x + cos_t * dx - sin_t * dy, y + sin_t * dx + cos_t * dy),
        (x - cos_t * dx - sin_t * dy, y - sin_t * dx + cos_t * dy),
        (x - cos_t * dx + sin_t * dy, y - sin_t * dx - cos_t * dy),
        (x + cos_t * dx + sin_t * dy, y + sin_t * dx - cos_t * dy),
    ]
    return corners


def polygon_area(vertices):
    """Shoelace formula for polygon area."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def compute_grasp_iou(pred_params, gt_params, image_size: int = 416) -> float:
    """
    Approximate IoU between predicted and ground truth grasp rectangles.
    Uses axis-aligned bounding box approximation for speed.

    Args:
        pred_params: (x, y, w, h, theta) - normalized [0,1] for x,y,w,h; radians for theta
        gt_params: same format
        image_size: image dimension for denormalization

    Returns:
        IoU value in [0, 1]
    """
    px, py, pw, ph, pt = [float(v) for v in pred_params]
    gx, gy, gw, gh, gt_angle = [float(v) for v in gt_params]

    # Denormalize
    px, py, pw, ph = px * image_size, py * image_size, pw * image_size, ph * image_size
    gx, gy, gw, gh = gx * image_size, gy * image_size, gw * image_size, gh * image_size

    # Approximate with axis-aligned bounding boxes of the rotated rectangles
    def aabb(cx, cy, w, h, theta):
        cos_t = abs(math.cos(theta))
        sin_t = abs(math.sin(theta))
        aabb_w = w * cos_t + h * sin_t
        aabb_h = w * sin_t + h * cos_t
        return (cx - aabb_w / 2, cy - aabb_h / 2, cx + aabb_w / 2, cy + aabb_h / 2)

    pred_box = aabb(px, py, pw, ph, pt)
    gt_box = aabb(gx, gy, gw, gh, gt_angle)

    # Intersection
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = pred_area + gt_area - intersection

    return intersection / max(union, 1e-6)


class GraspMetrics:
    """Accumulates and computes grasp detection metrics over a dataset."""

    def __init__(self, iou_threshold: float = 0.25, angle_threshold: float = 30.0):
        self.iou_threshold = iou_threshold
        self.angle_threshold = angle_threshold
        self.reset()

    def reset(self):
        self.total = 0
        self.iou_correct = 0
        self.angle_correct = 0
        self.both_correct = 0
        self.iou_sum = 0.0
        self.angle_diff_sum = 0.0
        self.xy_error_sum = 0.0
        self.wh_error_sum = 0.0

    def update(self, pred_params: torch.Tensor, gt_params: torch.Tensor):
        """
        Update metrics with a batch.

        Args:
            pred_params: [B, 5] predicted (x, y, w, h, theta)
            gt_params: [B, 5] ground truth (x, y, w, h, theta)
        """
        pred_np = pred_params.detach().cpu().numpy()
        gt_np = gt_params.detach().cpu().numpy()
        batch_size = pred_np.shape[0]

        for i in range(batch_size):
            pred = pred_np[i]
            gt = gt_np[i]

            iou = compute_grasp_iou(pred, gt)
            angle_diff = compute_angle_diff(
                math.degrees(pred[4]), math.degrees(gt[4])
            )

            self.iou_sum += iou
            self.angle_diff_sum += angle_diff
            self.xy_error_sum += np.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)
            self.wh_error_sum += np.sqrt((pred[2] - gt[2]) ** 2 + (pred[3] - gt[3]) ** 2)

            iou_ok = iou >= self.iou_threshold
            angle_ok = angle_diff <= self.angle_threshold

            if iou_ok:
                self.iou_correct += 1
            if angle_ok:
                self.angle_correct += 1
            if iou_ok and angle_ok:
                self.both_correct += 1

            self.total += 1

    def compute(self) -> dict:
        """Compute final metrics."""
        if self.total == 0:
            return {}
        return {
            "accuracy": self.both_correct / self.total,
            "iou_accuracy": self.iou_correct / self.total,
            "angle_accuracy": self.angle_correct / self.total,
            "mean_iou": self.iou_sum / self.total,
            "mean_angle_diff": self.angle_diff_sum / self.total,
            "mean_xy_error": self.xy_error_sum / self.total,
            "mean_wh_error": self.wh_error_sum / self.total,
            "total_samples": self.total,
        }
