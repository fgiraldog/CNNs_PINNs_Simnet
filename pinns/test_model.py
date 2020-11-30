import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import time
from matplotlib import rc
rc('text', usetex=True)

# function to properly plot the results
def plot_contour(x_data, y_data, reshape):
	not_geometry = reshape != 0
	plt.contourf(x_data, y_data, reshape, levels = 50)
	cb = plt.colorbar(format = '$%.2f$',pad = 0.02)
	cb.ax.tick_params(labelsize=30)

# function to test the results obtained
def test(model,x_test,y_test,number = -1):

	start_time = time.time()
	model.eval()
	out = model(x_test)
	print((time.time() - start_time)/len(x_test)) # time needed to use the model

	random_number = number # selection of the benchmark

	truth = y_test[random_number].cpu().detach().numpy() # ground truth [u-vel,v-vel,pressure]
	pred = out[random_number].cpu().detach().numpy() # predicted results by the model [u-vel,v-vel,pressure]

	# plots of the ground truth vs. predicted results
	xp = np.linspace(-3,3,384)
	yp = np.linspace(-1,1,128)

	size = 30

	plt.figure(figsize = (28,15))
	plt.subplot(3,2,1)
	plot_contour(xp,yp,truth[0])
	plt.xlabel('$x\:(m)$', size = size)
	plt.ylabel('$y\:(m)$', size = size)
	plt.xticks(size = size)
	plt.yticks([-1,-.5,0,.5,1],size = size)
	plt.title(r'$u_{true} \:(\times 10^{5} \: \mathrm{m/s})$', size = size)
	plt.subplot(3,2,2)
	plot_contour(xp,yp,pred[0])
	plt.xlabel('$x\:(m)$', size = size)
	plt.ylabel('$y\:(m)$', size = size)
	plt.xticks(size = size)
	plt.yticks([-1,-.5,0,.5,1],size = size)
	plt.title(r'$ u_{pred}\: (\times 10^{5} \: \mathrm{m/s})$', size = size)

	plt.subplot(3,2,3)
	plot_contour(xp,yp,truth[1])
	plt.xlabel('$x\:(m)$', size = size)
	plt.ylabel('$y\:(m)$', size = size)
	plt.xticks(size = size)
	plt.yticks([-1,-.5,0,.5,1],size = size)
	plt.title(r'$v_{true} \:(\times 10^{5} \: \mathrm{m/s})$', size = size)
	plt.subplot(3,2,4)
	plot_contour(xp,yp,pred[1])
	plt.xlabel('$x\:(m)$', size = size)
	plt.ylabel('$y\:(m)$', size = size)
	plt.xticks(size = size)
	plt.yticks([-1,-.5,0,.5,1],size = size)
	plt.title(r'$ v_{pred}\: (\times 10^{5} \: \mathrm{m/s})$', size = size)

	plt.subplot(3,2,5)
	plot_contour(xp,yp,truth[2])
	plt.xlabel('$x\:(m)$', size = size)
	plt.ylabel('$y\:(m)$', size = size)
	plt.xticks(size = size)
	plt.yticks([-1,-.5,0,.5,1],size = size)
	plt.title(r'$p_{true} \:(\times 10^{5} \: \mathrm{m/s})$', size = size)
	plt.subplot(3,2,6)
	plot_contour(xp,yp,pred[2])
	plt.xlabel('$x\:(m)$', size = size)
	plt.ylabel('$y\:(m)$', size = size)
	plt.xticks(size = size)
	plt.yticks([-1,-.5,0,.5,1],size = size)
	plt.title(r'$ p_{pred}\: (\times 10^{5} \: \mathrm{m/s})$', size = size)
	plt.subplots_adjust(hspace=0.39, wspace=0.11, bottom=0.08, left=0.05, right=0.97, top=0.93)
	plt.tight_layout()
	plt.savefig('figure4.png',dpi=300, bbox_inches='tight')

	# calculation of the RMSE for each field predicted 
	print(np.sqrt(np.sum((truth[0] - pred[0])**2)/(384*128))) # u-velocity
	print(np.sqrt(np.sum((truth[1] - pred[1])**2)/(384*128))) # v-velocity
	print(np.sqrt(np.sum((truth[2] - pred[2])**2)/(384*128))) # pressure

	# saving the results
	with open('result_informed.pickle', 'wb') as f:
		pickle.dump([truth,pred], f)
