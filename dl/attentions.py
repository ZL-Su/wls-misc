import torch
import torch.nn as nn
import torch.nn.functional as functional

class ChannelAttention(nn.Module):
    '''
    Brief:
      Impl of channel attention mechanism
    Args:
     - `in_planes` Channel planes of input feature map
     - `ratio` Reduction ratio of channels from input to output, i.e. `out_planes = ratio * in_planes`
    '''
    def __init__(self, in_planes:int, ratio:int=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #
        self.max_pool = nn.AdaptiveMaxPool2d(1) # 
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False), # 1x1 conv-based fully connected layer
            nn.ReLU(inplace=True), # Default performed in-place
            nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False) # 1x1 conv-based fully connected layer
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        - Formula: `sigma(MLP(AvgPool(x))+MLP(MaxPool(x)))`
        '''
        avg_branch = self.mlp(self.avg_pool(x))
        max_branch = self.mlp(self.max_pool(x))
        sum_out = avg_branch+max_branch
        return self.sigmoid(sum_out)
    
class SpatialAttention(nn.Module):
    '''
    Brief:
      Impl of spatial attention mechanism
    Args:
     - `kernel_size` Kernel size specified to perform convolution on catenated feature map 
    '''
    def __init__(self, kernel_size:int=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), 'kernel size in `SpatialAttention` must be 3 or 7'
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    
class SpatialTransformerNet(nn.Module):
    r'''
    \brief:
       Impl of the `Spatial Transformer` plug-in module
    \author Zhilong Su, Jun/2/2023
    '''
    def __init__(self, cin:int=1, kernel_size:int=7):
        super(SpatialTransformerNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=cin, out_channels=8, kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.regres = nn.Sequential(
            nn.Linear(in_features=10*3*3, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )

    def forward(self, x:torch.Tensor):
        y = self.conv(x)
        y = y.view(-1, 10*3*3)
        p = self.regres(y)
        p = p.view(-1, 2, 3)

        grid = functional.affine_grid(p, x.size())
        x = functional.grid_sample(x, grid=grid)

        return x
