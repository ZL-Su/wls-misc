import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pylab as plt

N = 200
X = np.random.random((N,1))
base = -np.ones((N,1))
exp = np.random.randint(2, size=(N,1))
sign = base**exp
Y = np.sqrt(X)*sign
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()

model = torch.nn.Sequential(
   nn.Linear(1, 100),
   nn.ReLU(inplace=True),
   nn.Linear(100, 1, bias=False),
)
loss = nn.MSELoss()
solver = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(500):
   y = model(X)
   error = loss(y, Y)
   print(f'Train loss: {error:.4f}')
   solver.zero_grad()
   error.backward()
   solver.step()

x = X.squeeze(dim=-1).detach().numpy()
y = Y.squeeze(dim=-1).detach().numpy()[:,]
plt.plot(x, y, '.', label='Data points', color="lightgray")
y = model(X).squeeze(dim=-1).detach().numpy()
plt.plot(x, y, '.', label='Supervised', color="red")

def loss_dp(y_true, y_pred):
   return loss(y_true, y_pred**2)
model_dp = torch.nn.Sequential(
   nn.Linear(1, 100),
   nn.ReLU(inplace=True),
   nn.Linear(100, 1, bias=False)
)
solver_dp = torch.optim.Adam(model_dp.parameters(), lr=0.001)
for epoch in range(500):
   y = model_dp(X)
   error = loss_dp(X, y)
   print(f'Train loss: {error:.4f}')
   solver_dp.zero_grad()
   error.backward()
   solver_dp.step()

y_dp = model_dp(X).squeeze(dim=-1).detach().numpy()
plt.plot(x, y_dp, '.', label='Supervised DP', color="green")
plt.xlabel('y')
plt.ylabel('x')
plt.title('Standard approach')
plt.legend()
plt.show()