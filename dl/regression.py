import torch
import torch.nn as nn
import utils
import attentions as attns
import matplotlib.pyplot as plt
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg', 'pdf')

class LinearRegressor(nn.Module):
    '''
    @brief Simple linear regression $y = wx+b$
    '''
    def __init__(self, input_size:int, output_size:int):
        super(LinearRegressor, self).__init__()
        self.net =  nn.Linear(input_size, output_size)
    def forward(self, x:torch.Tensor):
        return self.net(x)

class LogisticRegressor(nn.Module):
    '''
    @brief Regressor $y = tanh(wx+b)$ to solve binary classification problem.
           It should be trained by the `BCELossWithLogits` function with supervision.
    '''
    def __init__(self, input_size:int, output_size:int, hidden_size:list=[10]):
        super(LogisticRegressor,self).__init__()
        net_cat = []
        hidden_size.insert(0, input_size)
        for idx in range(len(hidden_size)-1):
            net_cat += [nn.Linear(hidden_size[idx], hidden_size[idx+1])]
            net_cat += [nn.Tanh()]
        net_cat += [nn.Linear(hidden_size[len(hidden_size)-1], output_size)]
        self.net = nn.Sequential(*net_cat)

    def forward(self, x):
        return self.net(x)

# Init machine runtime env
utils.__init_env__()

model = LogisticRegressor(2,2)
print("Arch of a logistic regressor model", model)
y = model(torch.Tensor([1,2]))

CAM = attns.ChannelAttention(1, 1)
print("Arch of ChannelAttention", CAM)
SAM = attns.SpatialAttention(7)
print("Arch of SpatialAttention", SAM)

# Init model
model = LinearRegressor(input_size=1, output_size=1)
# Def loss function and optimizer
lossf = nn.MSELoss() # mean-squared error
optim = torch.optim.SGD(model.parameters(), lr=0.01)

# Make input data and labels
data_list = []
for v in torch.arange(0, 10):
    data_list.append([v.item()])
data = torch.Tensor(data_list)
label = data*2.0
data = data+torch.randn(data.shape)

# Training...
loss_hist = []
for epoch in range(100):
    # forward pass
    preds = model(data)
    loss = lossf(preds, label)
    print(f'Train loss: {loss:.4f}')
    loss_hist.append(loss.item())
    # backward pass
    optim.zero_grad()
    loss.backward()
    optim.step()
plt.plot(loss_hist)
plt.show()

# Test data
x = torch.Tensor([[6],[7],[8],[9]])
y = torch.Tensor([[12],[14],[16],[18]])
# Test...
with torch.no_grad():
    preds = model(x)
    loss = lossf(preds, y)
    print(f'Test loss: {loss:.4f}')