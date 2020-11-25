import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
import torch.optim as optim
from Models.UNetEx import UNetEx
import train_model
import test_model
import pickle

file = open('data.pickle', 'rb')
x_data, y_data = np.array(pickle.load(file))
file.close()

x_train = x_data[:700]
x_test = x_data[700:]

y_train = y_data[:700]*10e4
y_test = y_data[700:]*10e4

samples, channels, length, width = y_train.shape

print(y_train.shape)

x_train = torch.FloatTensor(x_train).cuda()
y_train = torch.FloatTensor(y_train).cuda()
x_test = torch.FloatTensor(x_test).cuda()
y_test = torch.FloatTensor(y_test).cuda()

channels_weights = torch.sqrt(torch.mean(y_train.permute(0,2,3,1).reshape((length*width*samples,channels))**2, dim=0)).view(1,-1,1,1).cuda()

train_dataset, test_dataset = TensorDataset(x_train,y_train), TensorDataset(x_test,y_test)  

torch.manual_seed(0)
lr = 0.0001
kernel_size = 5
filters = [8,16,32,32]
bn = False
wn = False
wd = 0.005
batch_size = 32

model = UNetEx(3,3, filters=filters,kernel_size=kernel_size,batch_norm=bn,weight_norm=wn,layers = 5).cuda()
optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd) 
epochs = 1000

best_model = train_model.train(model, epochs, optimizer, train_dataset, test_dataset, channels_weights, batch_size)

test_model.test(best_model,x_test[-3:],y_test[-3:],-1)
