import torch.nn as nn

class ChannelAttention(nn.Module):
    '''
    @brief Impl of channel attention mechanism
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
        

