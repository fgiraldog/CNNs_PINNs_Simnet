from sympy import Symbol, Eq, Abs

import tensorflow as tf

from simnet.solver import Solver
from simnet.dataset import TrainDomain, ValidationDomain, InferenceDomain
from simnet.data import Validation, Inference
from simnet.sympy_utils.geometry_2d import Rectangle, Channel2D, Line
from simnet.csv_utils.csv_rw import csv_to_dict
from simnet.PDES.navier_stokes import NavierStokes
from simnet.controller import SimNetController
from simnet.architecture import FullyConnectedArch
from simnet.optimizer import AdamOptimizerAnnealing

# parameters for domain
inlet_vel = 0.0003
channel_length = (-3, 3)
channel_width = (-1, 1)

# defines channel
channel = Channel2D((channel_length[0], channel_width[0]),(channel_length[1], channel_width[1]))

# defines geometry
geometry = Rectangle((-0.93, -0.3), (0.07, 0.7))
geometry.rotate(0.1, (-0.93, -0.3))
inside = channel - geometry
inlet = Line((channel_length[0], channel_width[0]),(channel_length[0], channel_width[1]), -1)
outlet = Line((channel_length[1], channel_width[0]),(channel_length[1], channel_width[1]), 1)

# defines sympy varaibles to parametize domain curves
x, y = Symbol('x'), Symbol('y')

# boundary conditions and training parameters
class ssff_train(TrainDomain):
	def __init__(self, **config):
		super(ssff_train, self).__init__()

		# inlet
		inlet_bc = inlet.boundary_bc(outvar_sympy={'u': inlet_vel, 'v': 0}, batch_size_per_area=32, quasirandom = True)
		self.add(inlet_bc, name="Inlet")

		# outlet
		outlet_bc = outlet.boundary_bc(outvar_sympy={'p': 0},batch_size_per_area=32, quasirandom = True)
		self.add(outlet_bc, name="Outlet")

		# walls
		channel_wall = channel.boundary_bc(outvar_sympy={'u__y': 0, 'v': 0},batch_size_per_area=64, quasirandom = True)
		self.add(channel_wall, name="ChannelWall")

		# geometry walls
		geo_wall = geometry.boundary_bc(outvar_sympy={'u': 0, 'v': 0}, batch_size_per_area=32, quasirandom = True)
		self.add(geo_wall, name="GeoWall")

		# interior
		interior = inside.interior_bc(outvar_sympy={'continuity': 0, 'momentum_x': 0, 'momentum_y': 0},
		              bounds={x: channel_length, y: channel_width},
		              batch_size_per_area=500, quasirandom = True)
		self.add(interior, name="Interior")

# validation data
mapping = {'x': 'x', 'y': 'y', 'u': 'u', 'v': 'v', 'p': 'p'}
openfoam_var = csv_to_dict('openfoam/benchmark.csv', mapping)
openfoam_invar_numpy = {key: value for key, value in openfoam_var.items() if key in ['x', 'y']}
openfoam_outvar_numpy = {key: value for key, value in openfoam_var.items() if key in ['u', 'v', 'p']}

# validation data definition
class ssff_val(ValidationDomain):
	def __init__(self, **config):
	    super(ssff_val, self).__init__()
	    val = Validation.from_numpy(openfoam_invar_numpy, openfoam_outvar_numpy)
	    self.add(val, name='Val')

# solver definition
class ssff_solver(Solver):
	train_domain = ssff_train
	val_domain = ssff_val
	arch = FullyConnectedArch

	def __init__(self, **config):
		super(ssff_solver, self).__init__(**config)

		self.equations = NavierStokes(nu=1.588e-5, rho=1.205, dim=2, time=False).make_node() # navier-stokes time indpendent
		flow_net = self.arch.make_node(name='flow_net',inputs=['x', 'y'],outputs=['u', 'v', 'p'])
		self.nets = [flow_net]

	# architecture definition
	@classmethod
	def update_defaults(cls, defaults):
		defaults.update({
			'network_dir': './network_checkpoint_ssff',
	        'start_lr': 1e-2,
	        'decay_steps': 4000,
	        'max_steps': 400000,
			'layer_size': 512,
			# 'nr_layers': 17,
			# 'epsilon': 10e-8,
			# 'beta1': 0.9,
			# 'beta2': 0.999
			# 'layer_size': 

		})


if __name__ == '__main__':
	ctr = SimNetController(ssff_solver)
	ctr.run()
