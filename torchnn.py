import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt



class LSTMNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        batch_size = x.size()[1]
        h0 = torch.zeros([1, batch_size, self.hidden_dim])
        c0 = torch.zeros([1, batch_size, self.hidden_dim])
        fx, _ = self.lstm.forward(x, (h0, c0))
        return self.linear.forward(fx[-1])






class MLP(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MLP, self).__init__()
        self.Wh = nn.Linear(D_in, H)
        self.Wz = nn.Linear(H, D_out)

    def forward(self,x):
        Dh = self.Wh(x).clamp(min=0)
        Dz = self.Wz(Dh)

        return Dz



lstm = MLP(50, 30, 50)
crit = nn.MSELoss()
optim = torch.optim.SGD(lstm.parameters(), lr=1e-4)


x = torch.tensor([[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]]).float()
y = torch.tensor([[[10],[9],[8],[7],[6],[5],[4],[3],[2],[1]]]).float()

x = np.arange(0,100, 2)
xx = torch.from_numpy(x)
x = xx.view(1,50).float()
y = np.arange(2, 102, 2)
y = torch.from_numpy(y)
y = y.view(1,50).float()
#print x


h0 = torch.randn(1, 10).long()
for i in range(100):
    out = lstm(x)
    #print out
    #y = torch.empty(25, dtype=torch.long).random_(3)
    #print y
    #print x
    #print y
    #print out
    loss = crit(out, y)
    print('Epoch: {} Loss: {}').format(i, loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()

print x
print out
#print hn

