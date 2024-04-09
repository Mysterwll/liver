import torch
import torch.nn as nn

class Radiomic_encoder(nn.Module):
    def __init__(self, num_features):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.fc1 = nn.Linear(num_features,1024,bias=False)
        self.fc2 = nn.Linear(1024 ,512 ,bias=False)
        self.fc3 = nn.Linear(512 ,256 ,bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)        

        return x
    
if __name__ == '__main__':
    radio = torch.randn(size=(2,1800))
    model = Radiomic_encoder(num_features=1800)
    output = model(radio)
    print(output.shape)
