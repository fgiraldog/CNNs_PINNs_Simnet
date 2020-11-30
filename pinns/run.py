import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
import torch.optim as optim
from Models.UNetEx import UNetEx
import train_model_informed
import test_model
import pickle

# importing data to train and test the model
file = open('data.pickle', 'rb')
x_data, y_data = np.array(pickle.load(file))
file.close()

# separating and scaling data for the proper training and testing of the model
x_train = x_data[:700] # [SDF Surfaces, Multi Channel Region, SDF Edges]
x_test = x_data[700:] # [SDF Surfaces, Multi Channel Region, SDF Edges]

y_train = y_data[:700]*10e4 # [u-vel,v-vel,pressure]
y_test = y_data[700:]*10e4 # [u-vel,v-vel,pressure]

samples, channels, length, width = y_train.shape

# converting the data to torch tensors stored in the GPU memory
x_train = torch.FloatTensor(x_train).cuda()
y_train = torch.FloatTensor(y_train).cuda()
x_test = torch.FloatTensor(x_test).cuda()
y_test = torch.FloatTensor(y_test).cuda()

# calculation of the y-data weights for the proper normalization of the loss function
channels_weights = torch.sqrt(torch.mean(y_train.permute(0,2,3,1).reshape((length*width*samples,channels))**2, dim=0)).view(1,-1,1,1).cuda()

# converting the torch tensors to torch datasets
train_dataset, test_dataset = TensorDataset(x_train,y_train), TensorDataset(x_test,y_test)  

# hyperparameters
torch.manual_seed(0)
lr = 0.00001
kernel_size = 9
filters = [8,16,32,32]
bn = False
wn = False
wd = 0.005
batch_size = 64

# model initialization
model = UNetEx(3,3, filters=filters,kernel_size=kernel_size,batch_norm=bn,weight_norm=wn,layers = 5).cuda()
optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd) 
epochs = 1000

# model training
best_model = train_model_informed.train(model, epochs, optimizer, train_dataset, test_dataset, channels_weights, batch_size)

# model testing
test_model.test(best_model,x_test[-3:],y_test[-3:],-1)
