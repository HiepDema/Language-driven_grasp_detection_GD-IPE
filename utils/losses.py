"""
Loss functions for grasp pose prediction.

Supports multi-grasp GT: for each sample, computes loss against ALL GT grasps
and takes the minimum (best-matching GT grasp).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspLoss(nn.Module):
    """
    Combined loss for grasp parameter prediction with multi-grasp support.

    For samples with multiple GT grasps, the loss is computed against the
    GT grasp that best matches the prediction (min-over-targets strategy).
    """

    def __init__(
        self,
        xy_weight: float = 1.0,
        wh_weight: float = 1.0,
        angle_weight: float = 1.0,
        smooth_l1_beta: float = 0.5,
    ):
        super().__init__()
        self.xy_weight = xy_weight
        self.wh_weight = wh_weight
        self.angle_weight = angle_weight
        self.smooth_l1_beta = smooth_l1_beta

    def _single_loss(self, pred_xy, pred_wh, pred_sincos, target):
        """Compute loss between one prediction and one GT grasp [5]."""
        target_xy = target[:2]
        target_wh = target[2:4]
        target_angle = target[4:5]

        xy_loss = F.smooth_l1_loss(pred_xy, target_xy, beta=self.smooth_l1_beta)
        wh_loss = F.smooth_l1_loss(pred_wh, target_wh, beta=self.smooth_l1_beta)

        target_sin = torch.sin(target_angle)
        target_cos = torch.cos(target_angle)
        target_sincos = torch.cat([target_sin, target_cos], dim=-1)
        angle_loss = F.mse_loss(pred_sincos, target_sincos)

        total = (
            self.xy_weight * xy_loss
            + self.wh_weight * wh_loss
            + self.angle_weight * angle_loss
        )
        return total, xy_loss, wh_loss, angle_loss

    def forward(self, pred: dict, targets: list) -> dict:
        """
        Args:
            pred: dict from GraspHead with keys "xy", "wh", "sin_cos"
                  shapes: xy [B, 2], wh [B, 2], sin_cos [B, 2]
            targets: list of [N_i, 5] tensors, one per sample in batch.
                     Each has N_i GT grasps (x, y, w, h, theta) in radians.

        Returns:
            dict with "total", "xy_loss", "wh_loss", "angle_loss"
        """
        batch_size = pred["xy"].shape[0]
        total_loss = torch.tensor(0.0, device=pred["xy"].device)
        total_xy = torch.tensor(0.0, device=pred["xy"].device)
        total_wh = torch.tensor(0.0, device=pred["xy"].device)
        total_angle = torch.tensor(0.0, device=pred["xy"].device)

        for i in range(batch_size):
            pred_xy = pred["xy"][i]       # [2]
            pred_wh = pred["wh"][i]       # [2]
            pred_sincos = pred["sin_cos"][i]  # [2]
            gt_grasps = targets[i]        # [N_i, 5]

            if gt_grasps.shape[0] == 1:
                # Single GT — no need to search
                loss, xy_l, wh_l, angle_l = self._single_loss(
                    pred_xy, pred_wh, pred_sincos, gt_grasps[0]
                )
            else:
                # Multiple GT — find the best matching one (min loss)
                losses = []
                for j in range(gt_grasps.shape[0]):
                    l, _, _, _ = self._single_loss(
                        pred_xy, pred_wh, pred_sincos, gt_grasps[j]
                    )
                    losses.append(l)
                losses_t = torch.stack(losses)
                best_idx = torch.argmin(losses_t)
                loss, xy_l, wh_l, angle_l = self._single_loss(
                    pred_xy, pred_wh, pred_sincos, gt_grasps[best_idx]
                )

            total_loss = total_loss + loss
            total_xy = total_xy + xy_l
            total_wh = total_wh + wh_l
            total_angle = total_angle + angle_l

        total_loss = total_loss / batch_size
        total_xy = total_xy / batch_size
        total_wh = total_wh / batch_size
        total_angle = total_angle / batch_size

        return {
            "total": total_loss,
            "xy_loss": total_xy,
            "wh_loss": total_wh,
            "angle_loss": total_angle,
        }
