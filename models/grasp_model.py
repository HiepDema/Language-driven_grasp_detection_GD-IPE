"""
GraspCLIPModel: CLIP-based multimodal model for language-driven grasp detection.

Architecture:
    - CLIP image encoder -> image features
    - CLIP text encoder -> text features
    - Feature fusion (concatenation + projection)
    - Grasp prediction head -> {x, y, w, h, theta}
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

from models.grasp_head import GraspHead


class GraspCLIPModel(nn.Module):
    """
    Language-driven grasp pose prediction using CLIP as backbone.
    Given an image and a text instruction, predicts a 5-DoF grasp rectangle.
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        grasp_head_hidden: int = 512,
        dropout: float = 0.1,
        freeze_clip: bool = False,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        clip_dim = self.clip.config.projection_dim  # typically 512

        self.fusion = nn.Sequential(
            nn.Linear(clip_dim * 2, clip_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.grasp_head = GraspHead(
            input_dim=clip_dim,
            hidden_dim=grasp_head_hidden,
            dropout=dropout,
        )

        if freeze_clip:
            self.freeze_clip_params()

    def freeze_clip_params(self):
        for param in self.clip.parameters():
            param.requires_grad = False

    def unfreeze_clip_params(self):
        for param in self.clip.parameters():
            param.requires_grad = True

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through CLIP vision encoder."""
        outputs = self.clip.get_image_features(pixel_values=pixel_values)
        return outputs

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text through CLIP text encoder."""
        outputs = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        Args:
            pixel_values: [B, 3, 224, 224] preprocessed images
            input_ids: [B, seq_len] tokenized text
            attention_mask: [B, seq_len] attention mask

        Returns:
            dict from GraspHead with grasp parameters
        """
        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)

        # L2 normalize (as CLIP does)
        image_features = nn.functional.normalize(image_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)

        # Fuse features
        fused = torch.cat([image_features, text_features], dim=-1)
        fused = self.fusion(fused)

        # Predict grasp
        output = self.grasp_head(fused)
        return output
