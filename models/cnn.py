import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
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


class PositionalEncoding2D(nn.Module):
    def __init__(self, num_channels, img_size):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

        x_pos = torch.arange(img_size).float().unsqueeze(1).expand(img_size, img_size) / img_size
        y_pos = x_pos.T
        self.register_buffer('x_pos', x_pos.unsqueeze(0).unsqueeze(0))
        self.register_buffer('y_pos', y_pos.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        return x + self.a * self.x_pos + self.b * self.y_pos


class CNNBackbone(nn.Module):

    def __init__(self, d_model=512, img_size=416):
        super().__init__()
        self.d_model = d_model

        # Positional Encoding cho 3 stage đầu
        self.pe1 = PositionalEncoding2D(3,  img_size)           # 416×416, 3ch
        self.pe2 = PositionalEncoding2D(6,  img_size)           # 416×416, 6ch
        self.pe3 = PositionalEncoding2D(9,  img_size // 2)      # 208×208, 9ch

        # Stage 1: 3→6, 416×416 (stride=1) 
        self.stage1 = nn.Sequential(
            nn.Conv2d(3,  6,  3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
        )

        # Stage 2: 6→9, 416→208 (stride=2) 
        self.stage2 = nn.Sequential(
            nn.Conv2d(6,  9,  3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(9),
            nn.ReLU(inplace=True),
        )

        # Stage 3: 9→12, 208×208 (stride=1)
        self.stage3 = nn.Sequential(
            nn.Conv2d(9,  12, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
        )

        # Stage 4~7: giảm kích thước, tăng kênh + ResBlock 
        self.stage4 = nn.Sequential(                # 208→104, 12→64
            ConvBlock(12,  64,  stride=2),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        self.stage5 = nn.Sequential(                # 104→52, 64→128
            ConvBlock(64,  128, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.stage6 = nn.Sequential(                # 52→26, 128→256
            ConvBlock(128, 256, stride=2),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.stage7 = nn.Sequential(                # 26→13, 256→512
            ConvBlock(256, 512, stride=2),
            ResidualBlock(512),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        x = self.pe1(x)         
        x = self.stage1(x)      

        x = self.pe2(x)        
        x = self.stage2(x)      

        x = self.pe3(x)         
        x = self.stage3(x)      

        x = self.stage4(x)      
        x = self.stage5(x)     
        x = self.stage6(x)      
        x = self.stage7(x)      

        x = self.global_pool(x) 
        x = x.flatten(1)        
        x = self.fc(x)          
        return x