"""
Loss functions for grasp pose prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspLoss(nn.Module):
    """
    Combined loss for grasp parameter prediction.
    Handles x, y, w, h with Smooth L1 and angle with angular loss.
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

    def forward(self, pred: dict, target: torch.Tensor) -> dict:
        """
        Args:
            pred: dict from GraspHead with keys "xy", "wh", "angle", "sin_cos"
            target: [B, 5] ground truth (x, y, w, h, theta)
                    where x, y, w, h in [0, 1] and theta in radians

        Returns:
            dict with "total", "xy_loss", "wh_loss", "angle_loss"
        """
        target_xy = target[:, :2]
        target_wh = target[:, 2:4]
        target_angle = target[:, 4:5]

        # Smooth L1 for position and size
        xy_loss = F.smooth_l1_loss(pred["xy"], target_xy, beta=self.smooth_l1_beta)
        wh_loss = F.smooth_l1_loss(pred["wh"], target_wh, beta=self.smooth_l1_beta)

        # Angular loss using sin/cos representation
        target_sin = torch.sin(target_angle)
        target_cos = torch.cos(target_angle)
        target_sincos = torch.cat([target_sin, target_cos], dim=-1)
        angle_loss = F.mse_loss(pred["sin_cos"], target_sincos)

        total = (
            self.xy_weight * xy_loss
            + self.wh_weight * wh_loss
            + self.angle_weight * angle_loss
        )

        return {
            "total": total,
            "xy_loss": xy_loss,
            "wh_loss": wh_loss,
            "angle_loss": angle_loss,
        }
