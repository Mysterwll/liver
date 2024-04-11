import torch
import torch.nn as nn

class Radiomic_encoder(nn.Module):
    def __init__(self, num_features):
        """
            Radiomic_encoder to extract valid radiomic feature
        """
        super().__init__()
        self.fc1 = nn.Linear(num_features,1024,bias=False)
        self.fc2 = nn.Linear(1024 ,512 ,bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(True)
        self.projection_head = nn.Sequential(
                            nn.Linear(512, 128, bias = False),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 128, bias = False)
                            ) 
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        feat = x
        pj_feat = self.projection_head(feat)
        return feat, pj_feat

if __name__ == '__main__':
    radio = torch.randn(size=(2,1782))
    model = Radiomic_encoder(num_features=1782)
    output = model(radio)[0]
    print(output.shape)
