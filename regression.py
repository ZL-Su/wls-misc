import torch
import torch.nn as nn
import utils
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
    @brief Regressor $y = \sigmoid(wx+b)$ to solve binary classification problem.
           It should be trained by the `BCELoss` function with supervision.
    '''
    def __init__(self, input_size:int, output_size:int):
        super(LogisticRegressor,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Init machine runtime env
utils.__init_env__()

# Init model
model = LinearRegressor(input_size=1, output_size=1)
# Def loss function and optimizer
lossf = nn.MSELoss() # mean-squared error
optim = torch.optim.SGD(model.parameters(), lr=0.01)

# Make input data and labels
data = torch.Tensor([[1],[2],[3],[4],[5]])
label = torch.Tensor([[2],[4],[6],[8],[10]])

# Training...
for epoch in range(100):
    # forward pass
    preds = model(data)
    loss = lossf(preds, label)
    print(f'Train loss: {loss:.4f}')
    # backward pass
    optim.zero_grad()
    loss.backward()
    optim.step()

# Test data
x = torch.Tensor([[6],[7],[8],[9]])
y = torch.Tensor([[12],[14],[16],[18]])
# Test...
with torch.no_grad():
    preds = model(x)
    loss = lossf(preds, y)
    print(f'Test loss: {loss:.4f}')