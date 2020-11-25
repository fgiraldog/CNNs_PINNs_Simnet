import numpy as np
import torch
import copy

def loss_func(model, tensors,channels_weights):
	
	x_train, y_train = tensors
	out = model(x_train)

	lossu = ((out[:,0,:,:] - y_train[:,0,:,:]) ** 2).reshape((out.shape[0],1,out.shape[2],out.shape[3])) 
	lossv = ((out[:,1,:,:] - y_train[:,1,:,:]) ** 2).reshape((out.shape[0],1,out.shape[2],out.shape[3])) 
	lossp = torch.abs((out[:,2,:,:] - y_train[:,2,:,:])).reshape((out.shape[0],1,out.shape[2],out.shape[3])) 
	loss = (lossu + lossv + lossp)/channels_weights
	
	return torch.sum(loss), out

def epoch_train(model, optimizer, batch, training, channels_weights):
	
	total_loss = 0

	if training == True:
		model.train()
	else:
		model.eval()

	for element in batch:
		optimizer.zero_grad()
		loss, out = loss_func(model, element, channels_weights)
		if training == True:
			loss.backward()
			optimizer.step()

		total_loss += loss.item()

	return total_loss

def train(model, epochs, optimizer, train_dataset, val_dataset, channels_weights, batch_size):

	loss_train = []
	loss_val = []
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	for epoch in range(0, epochs):
		print('Epoch: {}/{}'.format(epoch+1,epochs))

		#Training
		train_loss = epoch_train(model, optimizer, train_loader, True, channels_weights)
		print('Train Loss = {}'.format(train_loss))

		# Validation
		with torch.no_grad():
			val_loss = epoch_train(model, optimizer, val_loader, False, channels_weights)
			print('Validation Loss = {}'.format(val_loss))
			print('\n')

		loss_train.append(train_loss)
		loss_val.append(val_loss)

		# Selection
		if epoch == 0:
			val_min = val_loss
		if val_loss < val_min:
			val_min = val_loss
			best_model = copy.deepcopy(model)
			print('Model saved')
			print('\n')


	return best_model

