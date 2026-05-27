"""
Grasp prediction head: takes fused image-text features and predicts {x, y, w, h, theta}.
"""

import torch
import torch.nn as nn


class GraspHead(nn.Module):
    """
    MLP head that predicts 5 grasp parameters: (x, y, w, h, theta).
    - x, y: center coordinates normalized to [0, 1]
    - w, h: width and height normalized to [0, 1]
    - theta: rotation angle, output as (sin, cos) pair internally, converted to angle
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Predict x, y, w, h (sigmoid activated) and sin_theta, cos_theta
        self.xy_head = nn.Linear(hidden_dim // 2, 2)
        self.wh_head = nn.Linear(hidden_dim // 2, 2)
        self.angle_head = nn.Linear(hidden_dim // 2, 2)  # sin, cos

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: [B, input_dim] fused multimodal features

        Returns:
            dict with keys: "xy", "wh", "angle", "params"
            - xy: [B, 2] center (x, y) in [0, 1]
            - wh: [B, 2] width, height in [0, 1]
            - angle: [B, 1] theta in radians [-pi/2, pi/2]
            - params: [B, 5] concatenated (x, y, w, h, theta)
        """
        h = self.mlp(features)

        xy = torch.sigmoid(self.xy_head(h))
        wh = torch.sigmoid(self.wh_head(h))

        angle_sincos = self.angle_head(h)
        sin_theta = angle_sincos[:, 0:1]
        cos_theta = angle_sincos[:, 1:2]
        theta = torch.atan2(sin_theta, cos_theta)

        params = torch.cat([xy, wh, theta], dim=-1)

        return {
            "xy": xy,
            "wh": wh,
            "angle": theta,
            "sin_cos": angle_sincos,
            "params": params,
        }
