import torch
import torch.nn as nn
# from mamba_ssm import Mamba


class Radiomic_encoder(nn.Module):
    def __init__(self, num_features):
        """
            Radiomic_encoder to extract valid radiomic feature
        """
        super().__init__()
        self.fc1 = nn.Linear(num_features, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(True)
        self.projection_head = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)

        feat = x
        pj_feat = self.projection_head(feat)
        return feat, pj_feat


class Radiomic_mamba_encoder(nn.Module):
    def __init__(self, num_features: int = 1775):
        """
            feature num -> 1775
            Radiomic_encoder based on Mamba model
        """
        super().__init__()
        self.projection1 = nn.Sequential(
            nn.Linear(1775, 2048, bias=False),
            nn.BatchNorm1d(2048),
        )
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=1,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.projection2 = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False)
        )

    def forward(self, x):
        x = self.projection1(x)
        x = torch.unsqueeze(x, dim=2)
        x = self.mamba(x)
        x = torch.squeeze(x, dim=2)
        return self.projection2(x)



if __name__ == '__main__':
    pass
    # radio = torch.randn(size=(2, 1782))
    # model = Radiomic_encoder(num_features=1782)
    # output = model(radio)[0]
    # print(output.shape)
    # batch, length, dim = 2, 64, 16
    # x = torch.randn(batch, length, dim).to("cuda")
    # model = Mamba(
    #     # This module uses roughly 3 * expand * d_model^2 parameters
    #     d_model=dim,  # Model dimension d_model
    #     d_state=16,  # SSM state expansion factor
    #     d_conv=4,  # Local convolution width
    #     expand=2,  # Block expansion factor
    # ).to("cuda")
    # y = model(x)
    # assert y.shape == x.shape

