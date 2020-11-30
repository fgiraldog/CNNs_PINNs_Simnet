import numpy as np
import torch
import copy

# loss function 
def loss_func(model, tensors, channels_weights, batch_size):
	
	nu = 1.588E-5
	x_train, y_train = tensors
	out = model(x_train)

	u = out[:,0,:,:]
	v = out[:,1,:,:]
	p = out[:,2,:,:]

	# calculating gradients for residuals
	for i in range(0,out.shape[0]):
		sol = out[i].cpu().detach().numpy()

		u_xi, u_yi = np.gradient(sol[0], 6./384., 2./128)
		u_xxi, u_xyi = np.gradient(u_xi, 6./384., 2./128)
		u_yxi, u_yyi = np.gradient(u_yi, 6./384., 2./128)

		v_xi, v_yi = np.gradient(sol[1], 6./384., 2./128)
		v_xxi, v_xyi = np.gradient(v_xi, 6./384., 2./128)
		v_yxi, v_yyi = np.gradient(v_yi, 6./384., 2./128)

		p_xi, p_yi = np.gradient(sol[2], 6./364., 2./128)
		
		if i == 0:
			u_x, u_y, u_xx, u_yy = u_xi, u_yi, u_xxi, u_yyi
			v_x, v_y, v_xx, v_yy = v_xi, v_yi, v_xxi, v_yyi
			p_x, p_y= p_xi, p_yi

		else:
			u_x = np.vstack((u_x,u_xi))
			u_y = np.vstack((u_y,u_yi))
			u_xx = np.vstack((u_xx,u_xxi))
			u_yy = np.vstack((u_yy,u_yyi))

			v_x = np.vstack((v_x,v_xi))
			v_y = np.vstack((v_y,v_yi))
			v_xx = np.vstack((v_xx,v_xxi))
			v_yy = np.vstack((v_yy,v_yyi))

			p_x = np.vstack((p_x,p_xi))
			p_y = np.vstack((p_y,p_yi))

	# organizing gradients
	u_x, u_y, u_xx, u_yy = torch.Tensor(u_x).reshape(out.shape[0],128,384).cuda(), torch.Tensor(u_y).reshape(out.shape[0],128,384).cuda(), torch.Tensor(u_xx).reshape(out.shape[0],128,384).cuda(), torch.Tensor(u_yy).reshape(out.shape[0],128,384).cuda()
	v_x, v_y, v_xx, v_yy = torch.Tensor(v_x).reshape(out.shape[0],128,384).cuda(), torch.Tensor(v_y).reshape(out.shape[0],128,384).cuda(), torch.Tensor(v_xx).reshape(out.shape[0],128,384).cuda(), torch.Tensor(v_yy).reshape(out.shape[0],128,384).cuda()
	p_x, p_y = torch.Tensor(p_x).reshape(out.shape[0],128,384).cuda(), torch.Tensor(p_y).reshape(out.shape[0],128,384).cuda()

	#calculation of residuals
	cont = (u_x + v_y)
	Re_x = (u*u_x) + (v*u_y) + p_x - (nu*(u_xx + u_yy))
	Re_y = (u*v_x) + (v*v_y) + p_y - (nu*(v_xx + v_yy))

	# weights of each residual for normalization
	cont_weight = torch.sqrt(torch.mean(cont.reshape((out.shape[0]*128*384,1))**2, dim=0)).cuda()
	Rey_weight = torch.sqrt(torch.mean(Re_x.reshape((out.shape[0]*128*384,1))**2, dim=0)).cuda()
	Rex_weight = torch.sqrt(torch.mean(Re_y.reshape((out.shape[0]*128*384,1))**2, dim=0)).cuda()


	lossu = ((u - y_train[:,0,:,:]) ** 2).reshape((out.shape[0],1,out.shape[2],out.shape[3])) # l2 regularization (u-field)
	lossv = ((v - y_train[:,1,:,:]) ** 2).reshape((out.shape[0],1,out.shape[2],out.shape[3])) # l2 regularization (v-field)
	lossp = torch.abs((p - y_train[:,2,:,:])).reshape((out.shape[0],1,out.shape[2],out.shape[3])) # l1 regularization (pressure-field)
	loss = (lossu + lossv + lossp)/channels_weights # normalization

	lossre = (Re_x ** 2).reshape(out.shape[0],1,out.shape[2],out.shape[3])/Rex_weight + (Re_y ** 2).reshape(out.shape[0],1,out.shape[2],out.shape[3])/Rey_weight # l2 normalized regularization (momentum)

	lossc = (cont ** 2).reshape(out.shape[0],1,out.shape[2],out.shape[3])/cont_weight # l2 normalized regularization (continuity) 

	return torch.sum(loss + lossre + lossc), out


# training procedure per epoch
def epoch_train(model, optimizer, loader, channels_weights, batch_size, training):
	
	total_loss = 0

	if training == True:
		model.train() # training 
	else:
		model.eval() # validation

	for tensors in loader:
		loss, out = loss_func(model, tensors, channels_weights, batch_size)
		if training == True:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		total_loss += loss.item()

	return total_loss

#training of the model
def train(model, epochs, optimizer, train_dataset, val_dataset, channels_weights, batch_size):
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	for epoch in range(0, epochs+1):
		print('Epoch: {}/{}'.format(epoch+1,epochs))

		# training
		train_loss = epoch_train(model, optimizer, train_loader, channels_weights, batch_size, training=True)
		print('Train Loss = {}'.format(train_loss))

		# validation
		with torch.no_grad():
			val_loss = epoch_train(model, optimizer, val_loader, channels_weights, batch_size, training=False)
			print('Validation Loss = {}'.format(val_loss))
			print('\n')


		# selection
		if epoch == 0:
			val_min = val_loss
		if val_loss < val_min:
			counter = 0
			val_min = val_loss
			best_model = copy.deepcopy(model)
			print('Model saved')
			print('\n')
		if val_loss > val_min:
			counter += 1

	return best_model
