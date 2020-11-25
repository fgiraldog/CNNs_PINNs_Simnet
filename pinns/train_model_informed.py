import numpy as np
import torch
import copy

def loss_func(model, tensors, channels_weights, batch_size):
	
	nu = 1.588E-5
	x_train, y_train = tensors
	out = model(x_train)

	u = out[:,0,:,:]
	v = out[:,1,:,:]
	p = out[:,2,:,:]

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

	u_x, u_y, u_xx, u_yy = torch.Tensor(u_x).reshape(out.shape[0],128,384).cuda(), torch.Tensor(u_y).reshape(out.shape[0],128,384).cuda(), torch.Tensor(u_xx).reshape(out.shape[0],128,384).cuda(), torch.Tensor(u_yy).reshape(out.shape[0],128,384).cuda()
	v_x, v_y, v_xx, v_yy = torch.Tensor(v_x).reshape(out.shape[0],128,384).cuda(), torch.Tensor(v_y).reshape(out.shape[0],128,384).cuda(), torch.Tensor(v_xx).reshape(out.shape[0],128,384).cuda(), torch.Tensor(v_yy).reshape(out.shape[0],128,384).cuda()
	p_x, p_y = torch.Tensor(p_x).reshape(out.shape[0],128,384).cuda(), torch.Tensor(p_y).reshape(out.shape[0],128,384).cuda()

	cont = (u_x + v_y)
	Re_x = (u*u_x) + (v*u_y) + p_x - (nu*(u_xx + u_yy))
	Re_y = (u*v_x) + (v*v_y) + p_y - (nu*(v_xx + v_yy))

	cont_weight = torch.sqrt(torch.mean(cont.reshape((out.shape[0]*128*384,1))**2, dim=0)).cuda()
	Rey_weight = torch.sqrt(torch.mean(Re_x.reshape((out.shape[0]*128*384,1))**2, dim=0)).cuda()
	Rex_weight = torch.sqrt(torch.mean(Re_y.reshape((out.shape[0]*128*384,1))**2, dim=0)).cuda()

	lossu = ((u - y_train[:,0,:,:]) ** 2).reshape((out.shape[0],1,out.shape[2],out.shape[3])) 
	lossv = ((v - y_train[:,1,:,:]) ** 2).reshape((out.shape[0],1,out.shape[2],out.shape[3])) 
	lossp = torch.abs((p - y_train[:,2,:,:])).reshape((out.shape[0],1,out.shape[2],out.shape[3])) 
	loss = (lossu + lossv + lossp)/channels_weights

	lossre = (Re_x ** 2).reshape(out.shape[0],1,out.shape[2],out.shape[3])/Rex_weight + (Re_y ** 2).reshape(out.shape[0],1,out.shape[2],out.shape[3])/Rey_weight

	lossc = (cont ** 2).reshape(out.shape[0],1,out.shape[2],out.shape[3])/cont_weight

	return torch.sum(loss + lossre + lossc), out

def epoch_train(model, optimizer, loader, channels_weights, batch_size, training):
	
	total_loss = 0

	if training == True:
		model.train()
	else:
		model.eval()

	for tensors in loader:
		loss, out = loss_func(model, tensors, channels_weights, batch_size)
		if training == True:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		total_loss += loss.item()

	return total_loss

def train(model, epochs, optimizer, train_dataset, val_dataset, channels_weights, batch_size):
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	counter = 0
	for epoch in range(0, epochs+1):
		print('Epoch: {}/{}'.format(epoch+1,epochs))

		#Training
		train_loss = epoch_train(model, optimizer, train_loader, channels_weights, batch_size, training=True)
		print('Train Loss = {}'.format(train_loss))

		# Validation
		with torch.no_grad():
			val_loss = epoch_train(model, optimizer, val_loader, channels_weights, batch_size, training=False)
			print('Validation Loss = {}'.format(val_loss))
			print('\n')


		# Selection
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
		
		if counter == 1000:
			break
	return best_model
