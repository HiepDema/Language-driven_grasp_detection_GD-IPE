import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class CNNBackbone(nn.Module):
    """
    Lightweight CNN backbone for feature extraction.
    Input: (B, 3, 416, 416)
    Output: (B, d_model)
    """

    def __init__(self, d_model=512, img_size=416):
        super().__init__()
        self.d_model = d_model

        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        x_pos = torch.arange(img_size).float().unsqueeze(1).expand(img_size, img_size) / img_size
        y_pos = x_pos.T
        self.register_buffer('x_position', x_pos.unsqueeze(0).unsqueeze(0))
        self.register_buffer('y_position', y_pos.unsqueeze(0).unsqueeze(0))

        # 416x416 -> 208x208
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
        )

        # 208x208 -> 104x104
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            ResidualBlock(64),
        )

        # 104x104 -> 52x52
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            ResidualBlock(128),
        )

        # 52x52 -> 26x26
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=2),
            ResidualBlock(256),
        )

        # 26x26 -> 13x13
        self.stage5 = nn.Sequential(
            DepthwiseSeparableConv(256, 256, stride=2),
            ResidualBlock(256),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, d_model)

    def forward(self, x):
        x = x + self.a * self.x_position + self.b * self.y_position
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
