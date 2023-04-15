import torch
import torch.nn as nn
import torchvision.transforms.functional

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConvolution, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()   
        )

    def forward(self, x:torch.Tensor):
        y = self.net(x)
        return y
    
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x:torch.Tensor):
        return self.pool(x)
    
class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x:torch.Tensor):
        return self.up(x)
    