import numpy as np
import torch
import copy
import time

# loss function 
def loss_func(model, tensors,channels_weights):
	
	x_train, y_train = tensors
	out = model(x_train)

	lossu = ((out[:,0,:,:] - y_train[:,0,:,:]) ** 2).reshape((out.shape[0],1,out.shape[2],out.shape[3])) # l2 regularization (u-field)
	lossv = ((out[:,1,:,:] - y_train[:,1,:,:]) ** 2).reshape((out.shape[0],1,out.shape[2],out.shape[3])) # l2 regularization (v-field)
	lossp = torch.abs((out[:,2,:,:] - y_train[:,2,:,:])).reshape((out.shape[0],1,out.shape[2],out.shape[3])) # l1 regularization (pressure-field)
	loss = (lossu + lossv + lossp)/channels_weights # normalization
	
	return torch.sum(loss), out

# training procedure per epoch
def epoch_train(model, optimizer, batch, training, channels_weights):
	
	total_loss = 0

	if training == True:
		model.train() # training
	else:
		model.eval() # validation

	for element in batch:
		optimizer.zero_grad()
		loss, out = loss_func(model, element, channels_weights)
		if training == True:
			loss.backward()
			optimizer.step()

		total_loss += loss.item()

	return total_loss

# training of the model
def train(model, epochs, optimizer, train_dataset, val_dataset, channels_weights, batch_size):

	loss_train = []
	loss_val = []
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	for epoch in range(0, epochs):
		t = time.time()
		print('Epoch: {}/{}'.format(epoch+1,epochs))

		# training
		train_loss = epoch_train(model, optimizer, train_loader, True, channels_weights)
		print('Train Loss = {}'.format(train_loss))

		# validation
		with torch.no_grad():
			val_loss = epoch_train(model, optimizer, val_loader, False, channels_weights)
			print('Validation Loss = {}'.format(val_loss))
			print('\n')

		loss_train.append(train_loss)
		loss_val.append(val_loss)

		# selection
		if epoch == 0:
			val_min = val_loss
		if val_loss < val_min:
			val_min = val_loss
			best_model = copy.deepcopy(model)
			print('Model saved')
			print('\n')

		print(time.time()-t)
		print('\n')


	return best_model

