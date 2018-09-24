
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as opt

import cv2
from time import sleep


file2save = 'cam_model.pth'

input_size = 1
hidden_size = 512
output_size = 6

normalize = False


def save_model(model):
    torch.save(model.state_dict(), file2save)

def load_model():
    return torch.load(file2save)


class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.temp_size = 64

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=4,kernel_size=(7,7),stride=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(5,5),stride=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(5,5),stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,5), stride=1)
        #self.filter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1)

        self.l1 = nn.Linear(self.temp_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, 256)
        self.l3 = nn.Linear(256, output_size)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.5)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,x):
        if normalize:
            x = x / 255

        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool(out)
        #out = self.conv4(out)
        #out = self.relu(out)
        #out = self.pool(out)

        flatten = out.view(out.size(0), -1)

        final = self.l1(flatten)
        final = self.tanh(final)
        final = self.dropout(final)
        final = self.l2(final)
        final = self.tanh(final)
        final = self.dropout(final)
        final = self.l3(final)


        #print flatten.size()
        return final


def preprcess_image(img):
    small = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
    frame = small.transpose(2,0,1)
    t_img = torch.Tensor(frame).unsqueeze(0)
    #t_img = t_img.transpose(1,2)
    return t_img



def predict(x):
    with torch.no_grad():
        return model(x).item()


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    diff = 255 - value
    v[v > diff] = 255
    v[v <= diff] += value

    final = cv2.merge((h,s,v))
    img = cv2.cvtColor(final, cv2.COLOR_HSV2BGR)

    return img


model = CNN(input_size,hidden_size,output_size)


if __name__ == '__main__':



    optim = opt.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    loss = 0
    avg_loss = 0.0

    for epoch in range(300):
        for dir in range(6):
            for img in range(10):
                image = cv2.imread('gallery/' + str(dir) + '/capture' + str(img) + '.jpg')
                #target = torch.zeros(6)
                target = torch.Tensor([dir]).long()
                t_img = preprcess_image(image)
                out = model(t_img)
                loss = crit(out, target)

                model.zero_grad()
                loss.backward()
                optim.step()
                avg_loss += loss.item()
        print 'Epoch: {}, Loss: {}, Average Loss: {}'.format(epoch,loss.item(), avg_loss/60)
        if epoch % 3 == 0: save_model(model)
        avg_loss = 0.0






    quit()
    for im in range(out.size(0)):
        frame = out[im].detach().numpy()
        cv2.imwrite('outputs/' + str(im) + '.jpg', frame)
