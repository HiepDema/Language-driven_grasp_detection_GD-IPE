"""
Loss functions for grasp pose prediction.

Supports multi-grasp GT: for each sample, computes loss against ALL GT grasps
and takes the minimum (best-matching GT grasp).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspLoss(nn.Module):
    """
    Combined loss for grasp parameter prediction with multi-grasp support.

    Model outputs:
        - center: [B, 2] in [0, 1]
        - size: [B, 2] in [0, 1]
        - sin2_cos2: [B, 2] softmax probabilities representing (sin^2(theta/2), cos^2(theta/2))

    GT format: [N, 5] as (x, y, w, h, theta) where theta in radians [0, pi).
    """

    def __init__(
        self,
        center_weight: float = 1.0,
        size_weight: float = 1.0,
        angle_weight: float = 1.0,
        smooth_l1_beta: float = 0.5,
    ):
        super().__init__()
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.angle_weight = angle_weight
        self.smooth_l1_beta = smooth_l1_beta

    def _single_loss(self, pred_center, pred_size, pred_sin2_cos2, target):
        """Compute loss between one prediction and one GT grasp [5]."""
        target_xy = target[:2]
        target_wh = target[2:4]
        target_theta = target[4]

        center_loss = F.smooth_l1_loss(pred_center, target_xy, beta=self.smooth_l1_beta)
        size_loss = F.smooth_l1_loss(pred_size, target_wh, beta=self.smooth_l1_beta)

        # GT theta in [0, pi]
        half = target_theta / 2
        target_sin2 = torch.sin(half) ** 2
        target_cos2 = torch.cos(half) ** 2
        target_vec = torch.stack([target_sin2, target_cos2])
        angle_loss = F.mse_loss(pred_sin2_cos2, target_vec)

        total = (
            self.center_weight * center_loss
            + self.size_weight * size_loss
            + self.angle_weight * angle_loss
        )
        return total, center_loss, size_loss, angle_loss

    def forward(self, pred: dict, targets: list) -> dict:
        """
        Args:
            pred: dict with keys "center" [B,2], "size" [B,2], "sin2_cos2" [B,2]
            targets: list of [N_i, 5] tensors, one per sample in batch.

        Returns:
            dict with "total", "center_loss", "size_loss", "angle_loss"
        """
        batch_size = pred["center"].shape[0]
        device = pred["center"].device
        total_loss = torch.tensor(0.0, device=device)
        total_center = torch.tensor(0.0, device=device)
        total_size = torch.tensor(0.0, device=device)
        total_angle = torch.tensor(0.0, device=device)

        for i in range(batch_size):
            pred_center = pred["center"][i]
            pred_size = pred["size"][i]
            pred_sin2_cos2 = pred["sin2_cos2"][i]
            gt_grasps = targets[i]

            if gt_grasps.shape[0] == 1:
                loss, c_l, s_l, a_l = self._single_loss(
                    pred_center, pred_size, pred_sin2_cos2, gt_grasps[0]
                )
            else:
                losses = []
                for j in range(gt_grasps.shape[0]):
                    l, _, _, _ = self._single_loss(
                        pred_center, pred_size, pred_sin2_cos2, gt_grasps[j]
                    )
                    losses.append(l)
                losses_t = torch.stack(losses)
                best_idx = torch.argmin(losses_t)
                loss, c_l, s_l, a_l = self._single_loss(
                    pred_center, pred_size, pred_sin2_cos2, gt_grasps[best_idx]
                )

            total_loss = total_loss + loss
            total_center = total_center + c_l
            total_size = total_size + s_l
            total_angle = total_angle + a_l

        total_loss = total_loss / batch_size
        total_center = total_center / batch_size
        total_size = total_size / batch_size
        total_angle = total_angle / batch_size

        return {
            "total": total_loss,
            "center_loss": total_center,
            "size_loss": total_size,
            "angle_loss": total_angle,
        }
