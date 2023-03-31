import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils

class LinearRegressor(nn.Module):
    def __init__(self, input_size:int, output_size:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor):
        return self.net(x)
    
class ConvNet(nn.Module):
    def __init__(self, cin:int, cout:int, kernel_size:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, 256, kernel_size=kernel_size),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=cout, kernel_size=kernel_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Init machine runtime env
utils.__init_env__()

I = torch.randn(2, 1, 512, 512)
conv = ConvNet(1, 10, 3)
y = conv(I)
print(y.shape)

net = LinearRegressor(10, 2)
x = np.arange(0, 20, dtype=np.float32)
x = x.reshape(2, -1)
x = torch.from_numpy(x)
x.to(utils.device())
print("Input features", x)
y = net(x)
print("Responses", y)
y = y.permute(1, 0)
print("Responses", y)