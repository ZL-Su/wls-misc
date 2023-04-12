import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    '''
    Brief:
      Impl of channel attention mechanism
    Args:
     - `in_planes` Channel planes of input feature map
     - `ratio` Reduction ratio of channels from input to output, i.e. `out_planes = ratio * in_planes`
    '''
    def __init__(self, in_planes:int, ratio:int=16):
        super().__init__(ChannelAttention, self)
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
        super().__init__(SpatialAttention, self)
        assert kernel_size in (3,7), 'kernel size in `SpatialAttention` must be 3 or 7'
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)