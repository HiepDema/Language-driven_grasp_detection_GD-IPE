import torch
import torch.nn as nn

from models.cnn import CNNBackbone
from models.vit import ViTBackbone
from models.nlp import TextEncoder


class CrossAttention(nn.Module):
    def __init__(self, d_model, text_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, kdim=text_dim, vdim=text_dim,
            batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, key_padding_mask=None):
        q = query.unsqueeze(1)
        out, _ = self.attn(q, context, context, key_padding_mask=key_padding_mask)
        return self.dropout(out.squeeze(1))


class GraspDetectionModel(nn.Module):
    """
    Language-driven grasp detection model.
    Input: image (B, 3, 416, 416), input_ids (B, seq_len), attention_mask (B, seq_len)
    Output: dict with center (x,y), size (w,h), sin(theta/2)
    """

    def __init__(self, d_model=256, dropout=0.3):
        super().__init__()
        self.cnn = CNNBackbone(d_model=d_model)
        self.vit = ViTBackbone(d_model=d_model, dropout=0.1)
        self.text_encoder = TextEncoder(d_model=d_model)

        text_dim = 768

        self.cross_attn = CrossAttention(d_model, text_dim, dropout=0.1)

        self.center_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid(),
        )

        self.b_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.d_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.angle_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self.size_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
            nn.Sigmoid(),
        )

    def forward(self, image, input_ids, attention_mask):
        A = self.cnn(image)
        B = self.vit(image)
        C, D = self.text_encoder(input_ids, attention_mask)

        padding_mask = (attention_mask == 0)
        E = self.cross_attn(A, C, key_padding_mask=padding_mask)

        center = self.center_head(E)

        F = self.b_mlp(B) + self.d_mlp(D)

        sin_theta_half = self.angle_head(F)

        EF = torch.cat([E, F], dim=-1)
        size = self.size_head(EF)

        return {
            "center": center,
            "size": size,
            "sin_theta_half": sin_theta_half,
        }