import torch
from mamba_ssm import Mamba, Mamba2
from torch import nn


class Omnidirectional_3D_Mamba(nn.Module):
    def __init__(self, in_channels, d_state=64, d_conv=4):
        super(Omnidirectional_3D_Mamba, self).__init__()
        expand = int(64 / in_channels)
        print(expand)
        self.mamba = nn.ModuleList([Mamba2(d_model=in_channels, d_state=d_state, d_conv=d_conv, expand=expand, headdim=4) for _ in range(3)])
        self.projection_dim = nn.ModuleList([nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, 1)
        ) for _ in range(3)])
        self.pool = nn.AdaptiveMaxPool1d(256)

    def forward(self, x):
        B, C, D, H, W = x.shape
        elements = D * H * W

        feature_D = x.permute(0, 1, 2, 3, 4).reshape(B, C, elements).transpose(-1, -2)
        feature_H = x.permute(0, 1, 3, 4, 2).reshape(B, C, elements).transpose(-1, -2)
        feature_W = x.permute(0, 1, 4, 2, 3).reshape(B, C, elements).transpose(-1, -2)

        feature_D = self.mamba[0](feature_D)
        feature_H = self.mamba[1](feature_H)
        feature_W = self.mamba[2](feature_W)

        feature_D = feature_D.view(-1, C)
        feature_H = feature_H.view(-1, C)
        feature_W = feature_W.view(-1, C)

        feature_D = self.projection_dim[0](feature_D)
        feature_H = self.projection_dim[1](feature_H)
        feature_W = self.projection_dim[2](feature_W)

        feature_D = feature_D.view(B, elements, 1).transpose(-1, -2)
        feature_H = feature_H.view(B, elements, 1).transpose(-1, -2)
        feature_W = feature_W.view(B, elements, 1).transpose(-1, -2)

        feature_D = self.pool(feature_D)
        feature_H = self.pool(feature_H)
        feature_W = self.pool(feature_W)

        feature = torch.cat([feature_D, feature_H, feature_W], dim=-1)

        return feature


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int = 1,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class Mamba_block(nn.Module):
    def __init__(self):
        super(Mamba_block, self).__init__()
        self.mamba_block = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=1,  # Model dimension d_model 1
            d_state=32,  # SSM state expansion factor 16
            d_conv=4,  # Local convolution width 4
            expand=8,  # Block expansion factor 2
        )
        self.norm = RMSNorm()

    def forward(self, x):
        return self.mamba_block(self.norm(x)) + x


class Radiomic_mamba_encoder(nn.Module):
    def __init__(self, num_features: int = 1781, depth: int = 4):
        """
            feature num -> 1781
            Radiomic_encoder based on Mamba model
        """
        super().__init__()
        self.projection1 = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.LayerNorm(2048)
        )
        self.blocks = nn.ModuleList([Mamba_block() for _ in range(depth)])
        self.projection2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        x = self.projection1(x)
        x = torch.unsqueeze(x, dim=2)
        for block in self.blocks:
            x = block(x)
        x = torch.squeeze(x, dim=2)
        return self.projection2(x)


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    input = torch.randn((2, 1, 32, 256, 256)).to(device)
    model = Omnidirectional_3D_Mamba(in_channels=1).to(device)
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    print(f"num: {num_params}")
    output = model(input)
    print(output.shape)
