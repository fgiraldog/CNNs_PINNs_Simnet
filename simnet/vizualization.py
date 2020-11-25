import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

data_pred = np.load('Val_pred.npz')
data_true = np.load('Val_true.npz')

u_pred = data_pred['u']
v_pred = data_pred['v']
p_pred = data_pred['p']

x_pred = data_pred['x']
y_pred = data_pred['y']

u_true = data_true['u']
v_true = data_true['v']
p_true = data_true['p']

x_true = data_true['x']
y_true = data_true['y']

truth = [u_true*(10**5), v_true*(10**5), p_true*(10**5)]
pred = [u_pred*(10**5), v_pred*(10**5), p_pred*(10**5)]

 
def plot_contour(x,y,attribute):
	x_data = np.unique(x)
	y_data = np.unique(y)

	reshape = np.zeros((128,384))
	for i,a in enumerate(attribute):
		x_ = x[i]
		y_ = y[i]
		index_x = np.where(x_data == x_)[0][0]
		index_y = np.where(y_data == y_)[0][0]

		reshape[index_y,index_x] = a

	not_geometry = reshape != 0.0
	plt.contourf(x_data, y_data, reshape, levels = 50)
	cb = plt.colorbar(format = '$%.2f$',pad = 0.02)
	cb.ax.tick_params(labelsize=25)

xp = data_true['x']
yp = data_true['y']

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
plt.savefig('figure6.png',dpi=300, bbox_inches='tight')

print(np.sqrt(np.sum((truth[0] - pred[0])**2)/(len(pred[0]))))
print(np.sqrt(np.sum((truth[1] - pred[1])**2)/(len(pred[1]))))
print(np.sqrt(np.sum((truth[2] - pred[2])**2)/(len(pred[2]))))
