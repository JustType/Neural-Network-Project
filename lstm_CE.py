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



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hc = None, hn = None):
        batch_size = x.size()[1]
        if hc == None and hn == None:
            hn = torch.zeros([1, batch_size, self.hidden])
            hc = torch.zeros([1, batch_size, self.hidden])
        l, _ = self.lstm(x, (hn, hc))
        out = self.linear(l[0])

        return out





class MLP(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MLP, self).__init__()
        self.Wh = nn.Linear(D_in, H)
        self.Wz = nn.Linear(H, D_out)

    def forward(self,x):
        Dh = self.Wh(x).clamp(min=0)
        Dz = self.Wz(Dh)

        return Dz


def toBinary(lst):
    out = torch.zeros(len(lst),50, dtype=torch.float)
    for i,list in enumerate(lst):
        for item in list:
            out[i][int(item)-1] = 1
    return out


def toInt(lst):
    nums = []
    #val, i = torch.max(lst, 0)
    #print('Value: {}, index: {}'). format(val,i)
    lst = torch.round(lst)
    for batch in lst:
        for i,item in enumerate(batch):
            if item == 1:
                nums.append(str(i + 1))
        print ','.join(nums)
        nums = []
        print '\n'


def getMax(lst):
    lst = lst.view(-1)
    out = []
    for i in range(5):
        ind = torch.argmax(lst,0)
        lst[ind] = -99
    for i in range(len(lst)):
        if lst[i] == -99:
            out.append(i + 1)

    return out



b = [[39,5,22,13,36], [21,9,5,7,1], [16,35,33,24,43], [47,18,30,9,48], [38,7,45,24,2], [13,46,34,14,21], [43,33,42,40,13], [40,50,43,2,22], [25,31,12,4,8], [4,17,21,15,23], [12,2,43,44,32]]

x = toBinary(b)
y = x[1:]
x = x[:-1]
x = x.view(1,len(b) - 1, 50)



lstm = LSTM(50, 100, 50)
crit = nn.MSELoss()
optim = torch.optim.Adam(lstm.parameters(), lr=1e-3)


#x = torch.empty(50, 50 ,dtype=torch.float).random_(2)
#x = x.view(1,50,50)
#y = torch.empty(10, dtype=torch.long).random_(2)
#y = torch.empty(50, 50, dtype=torch.float).random_(2)
#print x
#print y

for i in range(1000):
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
print y
print toInt(out)

z = raw_input()
l = z.split(',')
l = [l]
print l
l = toBinary(l)
print l
l = l.view(1,1,50)
out = lstm(l)
print out
toInt(out)
print '---------'
print getMax(out)
#print hn
