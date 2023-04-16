import torch
import torch.nn as nn
import torchvision.transforms.functional as functional

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
    
class DownSampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x:torch.Tensor):
        return self.pool(x)
    
class UpSampleNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x:torch.Tensor):
        return self.up(x)

class CropAndConcat(nn.Module):
    def forward(self, x:torch.Tensor, contract_x:torch.Tensor):
        contract_x=functional.center_crop(contract_x,[x.shape[2], x.shape[3]])
        x = torch.cat([x, contract_x], dim=1)
        return x
    
class UNet(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.down_conv=nn.ModuleList(
            [DoubleConvolution(i,o) for i, o in [(in_channels,64), (64,128), (128, 256), (256, 512)]]
        )
        self.down_sample=nn.ModuleList([DownSampleNet() for _ in range(4)])
        self.middle_conv=DoubleConvolution(512, 1024)

        self.up_sample = nn.ModuleList(
            [UpSampleNet(i,o) for i, o in [(1024,512), (512,256), (256, 128), (128, 64)]]
        )
        self.up_conv=nn.ModuleList(
            [DoubleConvolution(i,o) for i,o in [(1024, 512), (512,256), (256,128), (128, 64)]]
            )
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x:torch.Tensor):
        pass_through=[]
        print(x.shape)
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            print(x.shape)
            pass_through.append(x)
            x = self.down_sample[i](x)
            print(x.shape)
        x = self.middle_conv(x)
        print(x.shape)
        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            print(x.shape)
            x = self.concat[i](x, pass_through.pop())
            print(x.shape)
            x = self.up_conv[i](x)
            print(x.shape)
        x = self.final_conv(x)
        print(x.shape)
        return x

model = UNet(1, 2)
x = torch.rand(1, 1, 1024, 1024)
y = model(x)
