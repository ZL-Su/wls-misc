from torch import nn
import torch, pypose as pp
from pypose.optim import LM
from pypose.optim.strategy import Constant
from pypose.optim.scheduler import StopOnPlateau
import utils

# random so(3) tensor of Lie Algebra
r = pp.randn_so3(2, requires_grad=True)
print(r)

# get the Lie group of the Lie Algebra
R = pp.Exp(r)
print(R)

# rotate a random point
p = R @ torch.randn(3)
print("3D point\n", p)
p.sum().backward()
print("Gradient\n", r.grad)

class InvNet(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        init = pp.randn_SE3(*dim)
        self.pose = pp.Parameter(init)

    def forward(self, p):
        error = (self.pose*p).Log()
        return error.tensor()
    
print("Arch of InvNet\n", InvNet)
    
input = pp.randn_SE3(2, 2, device=utils.device())
invnet = InvNet(2, 2).to(device=utils.device())
strategy = Constant(damping=1e-4)
optimizer = LM(invnet, strategy=strategy)
scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)

#1st option, full optimization
scheduler.optimize(input=input)
#2nd option, step optimization
while scheduler.continual():
    loss = optimizer.step(input)
    scheduler.step(loss)