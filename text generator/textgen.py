import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import string



class LSTMNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        batch_size = x.size()[1]
        h0 = torch.zeros([1, batch_size, self.hidden_dim])
        c0 = torch.zeros([1, batch_size, self.hidden_dim])
        fx, _ = self.lstm.forward(x, (h0, c0))
        return self.linear.forward(fx[-1])


class RNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.GRU(hidden_dim, hidden_dim, num_layers=n_layers)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim, n_layers)


    def forward(self, x, hidden):
        #batch_size = x.size()[1]
        #h0 = torch.zeros([1, batch_size, self.hidden_dim])
        #c0 = torch.zeros([1, batch_size, self.hidden_dim])
        embed = self.embedding(x.view(1,-1))
        embed = embed.view(1,1,-1)
        fx, hidden = self.lstm.forward(embed, hidden)
        output = self.linear(fx.view(1,-1))

        return output, hidden


    def init_hidden(self):

        hidden = torch.zeros(self.n_layers, 1, self.hidden_dim)

        return hidden



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


class LSTM2_L(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM2_L, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, output_size)


    def forward(self, x):
        d1, _ = self.lstm1(x)
        d2, _ = self.lstm2(d1)
        return d2[-1]




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
    for i in range(2):
        ind = torch.argmax(lst,0)
        lst[ind] = -99
    for i in range(len(lst)):
        if lst[i] == -99:
            out.append(i)

    return out


def findLetter(text, uni):
    out = torch.zeros(1,len(text),len(uni))
    for i in range(len(text)):
        for n in range(len(uni)):
            if text[i] == uni[n]:
                out[0][i][n] = 1

    return out

def getIndex(y):
    out = torch.zeros(len(y), dtype=torch.long)
    for i in range(len(y)):
        for n in range(len(y[i])):
            if y[i][n] == 1:
                out[i] = n

    return out


def toLetter(index, uni):
    return uni[index]


def str2tensor(string):

    tensor = [ord(c) for c in string]

    tensor = torch.Tensor(tensor).long()

    return tensor


def save_model(model, name):
    torch.save(model,'models/' + name + '.pt')
    torch.save(model.state_dict(),'models/' + name + 'state_dict.pt')


def load_model(model_name):
    model = torch.load('models/' + model_name + '.pt')
    state = torch.load('models/' + model_name + 'state_dict.pt')
    model.load_state_dict(state)

    return model


def randomChoice():
    ascii = string.ascii_letters


    return ascii[random.randint(0, len(ascii) - 1)]


PATH_TO_FILE = '' # Choose file for network to learn

text = open(PATH_TO_FILE, 'r').read()

END_TOKEN = 127



'''
a = torch.zeros(1,1,10)
b = torch.zeros(1,1,10)

c = torch.cat((a,b), 1)
c = torch.cat((c,a), 1)
print a
print c
print c.size()
quit()
'''


lines = text.splitlines()
words = text.split()
sentences = text.split('.')
for sentence in range(len(sentences)):
    sentences[sentence] += '.'




uni = set(text)
print uni
print len(uni)
uni = list(uni)
words = text.split()
#y = torch.zeros(1,1,len(uni))
'''
x = findLetter(text,uni)
y = findLetter(text,uni)
x = x[0][:-1]
y = y[0][1:]
y = getIndex(y)
x = x.view(1, len(text)-1, len(uni))
'''

def generate(model, prime_str='A', predict_len=100, n_seq=1, temperature=0.8):

    hidden = model.init_hidden()
    prime_input = str2tensor(prime_str)
    predicted = prime_str
    seq = 0


    for p in range(len(prime_input) - 1):
        _, hidden = model(prime_input[p], hidden)

    inp = prime_input[-1]

    for s in range(n_seq):
        for p in range(predict_len):
            output, hidden = model(inp, hidden)


            #output_dist = output.data.view(-1).div(temperature).exp()
            #top_i = torch.multinomial(output_dist, 1)[0]
            top_n, top_i = output.topk(2)
            top_i = top_i[0][random.randint(0,1)]

            predicted_char = chr(top_i)
            if top_i == END_TOKEN:
                break
                if seq >= seq_len:
                    return predicted
            else:

                predicted += predicted_char
                inp = str2tensor(predicted_char)

    return predicted





def train(word):

    word += chr(END_TOKEN)
    input = str2tensor(word[:-1])
    target = str2tensor(word[1:])


    #if word == '' or len(word) == 1:
    #    return

    #print word
    loss = 0

    #print word
    #print input
    #model_in = input[0]

    hidden = lstm.init_hidden()

    for i in range(len(input)):
        print input[i]
        quit()
        out, hidden = lstm(input[i], hidden)
        #print out.size()
        #print target[i].view(1)
        #print 'Letter Index: ', i
        loss += crit(out.float(), target[i].view(1))
        #print('Epoch: {} Loss: {}').format(i, loss.item())
        #model_in = out.max(1)[1]


    optim.zero_grad()
    loss.backward()
    optim.step()


    return loss.data[0] / len(input)



modelLoad = False


if modelLoad == True:
    model = load_model('dobsinsky')
else:
    lstm = RNN(128, 150, 128, 3)
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    loss = 0

    for epoch in range(100):
        for n in range(len(sentences)):
            #print 'Sentence: ', n
            #print sentences[n]
            loss = train(sentences[n])

            if n % 99 == 0:
                print('Epoch: {}, Loss: {}').format(epoch,loss)
                print generate(lstm, randomChoice(), 100), '\n'


    print 'Done. Saving model.'

    save_model(lstm, 'dobsinsky')






print generate(model, randomChoice(), 100,10), '\n'













