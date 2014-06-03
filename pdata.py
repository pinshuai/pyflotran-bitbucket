""" Class for pflotran data """

import numpy as np
from copy import deepcopy
from copy import copy
import os,time
import platform

from ptool import*

WINDOWS = platform.system()=='Windows'
if WINDOWS: copyStr = 'copy'; delStr = 'del'; slash = '\\'
else: copyStr = 'cp'; delStr = 'rm'; slash = '/'

cards = ['mode','grid','timestepper','material_property','time','newton_solver','output',
		'fluid_property','saturation_function']
headers = ['mode','grid','time stepping','material properties',
		   'time','newton solver','output','fluid properties','saturation functions']
headers = dict(zip(cards,headers))

class pmaterial(object):
	""" Class for material property card

	"""

	def __init__(self,id,name,porosity=0.1,tortuosity=0.1,density=2.5e3,
		     specific_heat=1.e3,cond_dry=0.5,cond_wet=0.5,
		     saturation='sf2',permeability=[1.e-15,1.e-15,1.e-15]):
		self._id = id
		self._name = name
		self._porosity = porosity
		self._tortuosity = tortuosity
		self._density = density
		self._specific_heat = specific_heat
		self._cond_dry = cond_dry
		self._cond_wet = cond_wet
		self._saturation = saturation
		self._permeability = permeability

	def _get_id(self): return self._id
	def _set_id(self,value): self._id = value
	id = property(_get_id, _set_id) #: (**)
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name) #: (**)
	def _get_porosity(self): return self._porosity
	def _set_porosity(self,value): self._porosity = value
	porosity = property(_get_porosity, _set_porosity) #: (**)
	def _get_tortuosity(self): return self._tortuosity
	def _set_tortuosity(self,value): self._tortuosity = value
	tortuosity = property(_get_tortuosity, _set_tortuosity) #: (**)
	def _get_density(self): return self._density
	def _set_density(self,value): self._density = value
	density = property(_get_density, _set_density) #: (**)
	def _get_specific_heat(self): return self._specific_heat
	def _set_specific_heat(self,value): self._specific_heat = value
	specific_heat = property(_get_specific_heat, _set_specific_heat) #: (**)
	def _get_cond_dry(self): return self._cond_dry
	def _set_cond_dry(self,value): self._cond_dry = value
	cond_dry = property(_get_cond_dry, _set_cond_dry) #: (**)
	def _get_cond_wet(self): return self._cond_wet
	def _set_cond_wet(self,value): self._cond_wet = value
	cond_wet = property(_get_cond_wet, _set_cond_wet) #: (**)
	def _get_saturation(self): return self._saturation
	def _set_saturation(self,value): self._saturation = value
	saturation = property(_get_saturation, _set_saturation) #: (**)
	def _get_permeability(self): return self._permeability
	def _set_permeability(self,value): self._permeability = value
	permeability = property(_get_permeability, _set_permeability) #: (**)


class ptime(object):
	""" Class for time property

	"""
	
	def __init__(self,tf=10.,dti=1.e-2,dtf=50.,dtf_lv=[None],dtf_li=[None],dtf_i=0):
		self._tf = tf
		self._dti = dti
		self._dtf = dtf
		self._dtf_lv = dtf_lv
		self._dtf_li = dtf_li
		self._dtf_i = dtf_i		# Index, also being used as a counter
		
	def _get_tf(self): return self._tf
	def _set_tf(self,value): self._tf = value
	tf = property(_get_tf, _set_tf) #: (**)
	def _get_dti(self): return self._dti
	def _set_dti(self,value): self._dti = value
	dti = property(_get_dti, _set_dti) #: (**)
	
	def _get_dtf(self): return self._dtf
	def _set_dtf(self,value): self._dtf = value
	dtf = property(_get_dtf, _set_dtf) #: (**)
	
	# The dtf lists are for multiple max time step entries at specified time intervals
	def _get_dtf_lv(self): return self._dtf_lv
	def _set_dtf_lv(self,value): self._dtf_lv = value
	dtf_lv = property(_get_dtf_lv, _set_dtf_lv) #: (**)
	def _get_dtf_li(self): return self._dtf_li
	def _set_dtf_li(self,value): self._dtf_li = value
	dtf_li = property(_get_dtf_li, _set_dtf_li) #: (**)
	def _get_dtf_i(self): return self._dtf_i
	def _set_dtf_i(self,value): self._dtf_i = value
	dtf_i = property(_get_dtf_i, _set_dtf_i) #: (**)
	
class pgrid(object):		# discretization
	""" Class for grid property

	"""

	def __init__(self,type='structured',lower_bounds=[0.0,0.0,0.0],upper_bounds=[50.0,50.0,50.0],
				 bounds_bool=True,origin=[0.0,0.0,0.0],nxyz=[10,10,10],dxyz=[5,5,5],gravity_bool=False,
				 gravity=[0.0,0.0,-9.8068],filename=''):
		self._type = type
		self._lower_bounds = lower_bounds
		self._upper_bounds = upper_bounds
		self._origin = origin
		self._nxyz = nxyz
		self._dxyz = dxyz
		self._gravity = gravity
		self._filename = filename
		self._bounds_bool = bounds_bool
		self._gravity_bool = gravity_bool

	def _get_type(self): return self._type
	def _set_type(self,value): self._type = value
	type = property(_get_type, _set_type) #: (**)
	def _get_lower_bounds(self): return self._lower_bounds
	def _set_lower_bounds(self,value): self._lower_bounds = value
	lower_bounds = property(_get_lower_bounds, _set_lower_bounds) #: (**)
	def _get_upper_bounds(self): return self._upper_bounds
	def _set_upper_bounds(self,value): self._upper_bounds = value
	upper_bounds = property(_get_upper_bounds, _set_upper_bounds) #: (**)
	def _get_origin(self): return self._origin
	def _set_origin(self,value): self._origin = value
	origin = property(_get_origin, _set_origin) #: (**)
	def _get_nxyz(self): return self._nxyz
	def _set_nxyz(self,value): self._nxyz = value
	nxyz = property(_get_nxyz, _set_nxyz) #: (**)
	def _get_dxyz(self): return self._dxyz
	def _set_dxyz(self,value): self._dxyz = value
	dxyz = property(_get_dxyz, _set_dxyz) #: (**)
	def _get_gravity(self): return self._gravity
	def _set_gravity(self,value): self._gravity = value
	gravity = property(_get_gravity, _set_gravity) #: (**)
	def _get_filename(self): return self._filename
	def _set_filename(self,value): self._filename = value
	filename = property(_get_filename, _set_filename) #: (**)
	def _get_bounds_bool(self): return self._bounds_bool
	def _set_bounds_bool(self,value): self._bounds_bool = value
	bounds_bool = property(_get_bounds_bool, _set_bounds_bool) #: (**)
	def _get_gravity_bool(self): return self._gravity_bool
	def _set_gravity_bool(self,value): self._gravity_bool = value
	gravity_bool = property(_get_gravity_bool, _set_gravity_bool) #: (**)


class pmode(object):
	""" Class for mode

	"""

	def __init__(self,name='richards'):
		self._name = name

	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name)


class ptimestepper(object):
	""" Class for time stepper

	"""

	def __init__(self,ts_acceleration=None,num_steps_after_cut=None,max_steps=None,max_ts_cuts=None,
                          cfl_limiter=None,initialize_to_steady_state=False,run_as_steady_state=False,
                          max_pressure_change=None,max_temperature_change=None,max_concentration_change=None,
                          max_saturation_change=None):
		self._ts_acceleration = ts_acceleration
		self._num_steps_after_cut = num_steps_after_cut 
		self._max_steps = max_steps
		self._max_ts_cuts = max_ts_cuts
		self._cfl_limiter = cfl_limiter
		self._initialize_to_steady_state = initialize_to_steady_state
		self._run_as_steady_state = run_as_steady_state
		self._max_pressure_change = max_pressure_change
		self._max_temperature_change = max_temperature_change
		self._max_concentration_change = max_concentration_change
		self._max_saturation_change = max_saturation_change

	def _get_ts_acceleration(self): return self._ts_acceleration
	def _set_ts_acceleration(self,value): self._ts_acceleration = value
	ts_acceleration = property(_get_ts_acceleration, _set_ts_acceleration)
	def _get_num_steps_after_cut(self): return self._num_steps_after_cut
	def _set_num_steps_after_cut(self,value): self._num_steps_after_cut = value
	num_steps_after_cut = property(_get_num_steps_after_cut, _set_num_steps_after_cut)
	def _get_max_steps(self): return self._max_steps
	def _set_max_steps(self,value): self._max_steps = value
	max_steps = property(_get_max_steps, _set_max_steps)
	def _get_max_ts_cuts(self): return self._max_ts_cuts
	def _set_max_ts_cuts(self,value): self._max_ts_cuts = value
	max_ts_cuts = property(_get_max_ts_cuts, _set_max_ts_cuts)
	def _get_cfl_limiter(self): return self._cfl_limiter
	def _set_cfl_limiter(self,value): self._cfl_limiter = value
	cfl_limiter = property(_get_cfl_limiter, _set_cfl_limiter)
	def _get_initialize_to_steady_state(self): return self._initialize_to_steady_state
	def _set_initialize_to_steady_state(self,value): self._initialize_to_steady_state = value
	initialize_to_steady_state = property(_get_initialize_to_steady_state, _set_initialize_to_steady_state)
	def _get_run_as_steady_state(self): return self._run_as_steady_state
	def _set_run_as_steady_state(self,value): self._run_as_steady_state = value
	run_as_steady_state = property(_get_run_as_steady_state, _set_run_as_steady_state)
	def _get_max_pressure_change(self): return self._max_pressure_change
	def _set_max_pressure_change(self,value): self._max_pressure_change = value
	max_pressure_change = property(_get_max_pressure_change, _set_max_pressure_change)
	def _get_max_temperature_change(self): return self._max_temperature_change
	def _set_max_temperature_change(self,value): self._max_temperature_change = value
	max_temperature_change = property(_get_max_temperature_change, _set_max_temperature_change)
	def _get_max_concentration_change(self): return self._max_concentration_change
	def _set_max_concentration_change(self,value): self._max_concentration_change = value
	max_concentration_change = property(_get_max_concentration_change, _set_max_concentration_change)
	def _get_max_saturation_change(self): return self._max_saturation_change
	def _set_max_saturation_change(self,value): self._max_saturation_change = value
	max_saturation_change = property(_get_max_saturation_change, _set_max_saturation_change)

class pnsolver(object):
	""" Class for newton solver

	"""

	def __init__(self,name,atol=None,rtol=None,stol=None,dtol=None,itol=None,
			max_it=None,max_f=None):
		self._name = name	# Indicates Flow or Tran for Transport
		self._atol = atol
		self._rtol = rtol
		self._stol = stol
		self._dtol = dtol
		self._itol = itol
		self._max_it = max_it
		self._max_f = max_f
		
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name)
	def _get_atol(self): return self._atol
	def _set_atol(self,value): self._atol = value
	atol = property(_get_atol, _set_atol)
	def _get_rtol(self): return self._rtol
	def _set_rtol(self,value): self._rtol = value
	rtol = property(_get_rtol, _set_rtol)
	def _get_stol(self): return self._stol
	def _set_stol(self,value): self._stol = value
	stol = property(_get_stol, _set_stol)
	def _get_dtol(self): return self._dtol
	def _set_dtol(self,value): self._dtol = value
	dtol = property(_get_dtol, _set_dtol)
	def _get_itol(self): return self._itol
	def _set_itol(self,value): self._itol = value
	itol = property(_get_itol, _set_itol)
	def _get_max_it(self): return self._max_it
	def _set_max_it(self,value): self._max_it = value
	max_it = property(_get_max_it, _set_max_it)
	def _get_max_f(self): return self._max_f
	def _set_max_f(self,value): self._max_f = value
	max_f = property(_get_max_f, _set_max_f)
	
class poutput(object):
	"""Class for output options

	"""
	
	def __init__(self,mass_balance=None,print_column_ids=None,periodic_observation_timestep=None,
			format=[],velocities=None):
		self._mass_balance = mass_balance
		self._print_column_ids = print_column_ids
		self._periodic_observation_timestep = periodic_observation_timestep
		self._format = format
		self._velocities = velocities
		
	def _get_mass_balance(self): return self._mass_balance
	def _set_mass_balance(self,value): self._mass_balance = value
	mass_balance = property(_get_mass_balance, _set_mass_balance)	
	def _get_print_column_ids(self): return self._print_column_ids
	def _set_print_column_ids(self,value): self._print_column_ids = value
	print_column_ids = property(_get_print_column_ids, _set_print_column_ids)
	def _get_periodic_observation_timestep(self): return self._periodic_observation_timestep
	def _set_periodic_observation_timestep(self,value): self._periodic_observation_timestep = value
	periodic_observation_timestep = property(_get_periodic_observation_timestep, _set_periodic_observation_timestep)
	def _get_format(self): return self._format
	def _set_format(self,value): self._format = value
	format = property(_get_format, _set_format)
	def _get_velocities(self): return self._velocities
	def _set_velocities(self,value): self._velocities = value
	velocities = property(_get_velocities, _set_velocities)	
	
class pfluid(object):
	"""Class for fluid properties

	"""
	
	def __init__(self,diffusion_coefficient=None):
		self._diffusion_coefficient = diffusion_coefficient
		
	def _get_diffusion_coefficient(self): return self._diffusion_coefficient
	def _set_diffusion_coefficient(self,value): self._diffusion_coefficient = value
	diffusion_coefficient = property(_get_diffusion_coefficient, _set_diffusion_coefficient)
	
class psaturation(object):
	"""Class for saturation functions

	"""
	
	def __init__(self,name,permeability_function_type=None,saturation_function_type=None,
			residual_saturation_liquid=None,residual_saturation_gas=None,
			a_lambda=None,alpha=None,max_capillary_pressure=None,
			betac=None,power=None):
		self._name = name
		self._permeability_function_type = permeability_function_type
		self._saturation_function_type = saturation_function_type
		self._residual_saturation_liquid = residual_saturation_liquid
		self._residual_saturation_gas = residual_saturation_gas
		self._a_lambda = a_lambda
		self._alpha = alpha
		self._max_capillary_pressure = max_capillary_pressure
		self._betac = betac
		self._power = power
		
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name)
	def _get_permeability_function_type(self): return self._permeability_function_type
	def _set_permeability_function_type(self,value): self._permeability_function_type = value
	permeability_function_type = property(_get_permeability_function_type, _set_permeability_function_type)
	def _get_saturation_function_type(self): return self._saturation_function_type
	def _set_saturation_function_type(self,value): self._saturation_function_type = value
	saturation_function_type = property(_get_saturation_function_type, _set_saturation_function_type)	
	def _get_residual_saturation_liquid(self): return self._residual_saturation_liquid
	def _set_residual_saturation_liquid(self,value): self._residual_saturation_liquid = value
	residual_saturation_liquid = property(_get_residual_saturation_liquid, _set_residual_saturation_liquid)
	def _get_residual_saturation_gas(self): return self._residual_saturation_gas
	def _set_residual_saturation_gas(self,value): self._residual_saturation_gas = value
	residual_saturation_gas = property(_get_residual_saturation_gas, _set_residual_saturation_gas)
	def _get_a_lambda(self): return self._a_lambda
	def _set_a_lambda(self,value): self._a_lambda = value
	a_lambda = property(_get_a_lambda, _set_a_lambda)
	def _get_alpha(self): return self._alpha
	def _set_alpha(self,value): self._alpha = value
	alpha = property(_get_alpha, _set_alpha)
	def _get_max_capillary_pressure(self): return self._max_capillary_pressure
	def _set_max_capillary_pressure(self,value): self._max_capillary_pressure = value
	max_capillary_pressure = property(_get_max_capillary_pressure, _set_max_capillary_pressure)
	def _get_betac(self): return self._betac
	def _set_betac(self,value): self._betac = value
	betac = property(_get_betac, _set_betac)
	def _get_power(self): return self._power
	def _set_power(self,value): self._power = value
	power = property(_get_power, _set_power)
	
class pdata(object):
	"""Class for pflotran data file

	"""

	def __init__(self, filename=None):
		from copy import copy
		self._filename = filename
		self._proplist = []	# There are multiple material properties objects
		self._time = ptime()
		self._mode = ''
		self._grid = pgrid()
		self._timestepper = ptimestepper()
		self._nsolver = pnsolver('')
		self._output = poutput()
		self._fluid = pfluid()
		self._saturation = psaturation('')
		
		if filename: self.read(filename) 		# read in file if requested upon initialisation
	
	def __repr__(self): return self.filename 	# print to screen when called
	
	def read(self, filename):
		if not os.path.isfile(filename): print filename + ' not found...'
		self._filename = filename 	# assign filename attribute
		read_fn = dict(zip(cards, 	
				[self._read_mode,
				 self._read_grid,
				 self._read_timestepper,
				 self._read_prop,
				 self._read_time,
				 self._read_nsolver,
				 self._read_output,
				 self._read_fluid,
				 self._read_saturation]
				 ))  # associate each card name with a read function, defined further below
		with open(self._filename,'r') as infile:
			keepReading = True
			while keepReading:
				line = infile.readline()
#				print line.strip()
				if not line: keepReading = False
				if len(line.strip())==0: continue
				card = line.split()[0].lower() 		# make card lower case
				if card in cards: 			# check if a valid cardname
					if card in ['material_property','mode','grid','timestepper',
							'newton_solver','saturation_function']:
						read_fn[card](infile,line)
					else:
						read_fn[card](infile)
	
	def write(self, filename=None):
		"""Write pdata object to pflotran input file.

		:param filename: Name of pflotran input file.
		:type filename: str
		"""
		if filename: self._filename = filename
		outfile = open(self.filename,'w')
		if self.mode: self._write_mode(outfile)
		if self.grid: self._write_grid(outfile)
		if self.timestepper : self._write_timestepper(outfile)
		if self.time: self._write_time(outfile)
		if self.proplist: self._write_prop(outfile)
		if self.nsolver: self._write_nsolver(outfile)
		if self.output: self._write_output(outfile)
		if self.fluid: self._write_fluid(outfile)
		if self.saturation: self._write_saturation(outfile)
		outfile.close()
		
	def _read_mode(self,infile,line):
		mode_name = line.split()[-1]
		new_mode = pmode(mode_name)
		self._mode = new_mode
	
	def _write_mode(self,outfile):
		self._header(outfile,headers['mode'])
		outfile.write('MODE\t')
		outfile.write(self.mode.name+'\n\n')
	
	def _read_grid(self,infile,line):
		g = pgrid()				# assign defaults before reading in values
		ng_type = g.type
		ng_lower_bounds = g.lower_bounds
		ng_upper_bounds = g.upper_bounds
		ng_origin = g.origin
		ng_nxyz = g.nxyz
		ng_gravity = g.gravity
		ng_filename = g.filename
		ng_dxyz = g.dxyz

		keepReading = True
		bounds_key = False
		gravity_key = False
		while keepReading:
			line = infile.readline() 			# get next line
			key = line.strip().split()[0].lower() 		# take first keyword
			if key in ['#']:
				pass
			if key == 'type':
				ng_type = line.split()[-1]
			elif key == 'bounds':
				bounds_key = True
				keepReading2 = True
				while keepReading2:
					line1 = infile.readline()
					ng_lower_bounds[0] = floatD(line1.split()[0])
					ng_lower_bounds[1] = floatD(line1.split()[1])
					ng_lower_bounds[2] = floatD(line1.split()[2])
					line2 = infile.readline()
					ng_upper_bounds[0] = floatD(line2.split()[0])
					ng_upper_bounds[1] = floatD(line2.split()[1])
					ng_upper_bounds[2] = floatD(line2.split()[2])
					line3 = infile.readline()
					if line3.strip().split()[0].lower() in ['/','end']: keepReading2 = False
			elif key == 'origin':
				ng_origin[0] = floatD(line.strip().split()[1])
				ng_origin[1] = floatD(line.strip().split()[2])
				ng_origin[2] = floatD(line.strip().split()[3])
			elif key == 'nxyz':
				ng_nxyz[0] = floatD(line.split()[1])
				ng_nxyz[1] = floatD(line.split()[2])
				ng_nxyz[2] = floatD(line.split()[3])
			elif key == 'gravity':
				gravity_key = True
				ng_gravity[0] = floatD(line.split()[1])
				ng_gravity[1] = floatD(line.split()[2])
				ng_gravity[2] = floatD(line.split()[3])
			elif key == 'filename':
				if ng_type != 'unstructured': print 'Error - filename not need with structure grid'; return
				ng_filename = line.split()[-1]
			elif key == 'dxyz':
				if bounds_key: print 'Error - specify either bounds of dxyz'; return
				keepReading2 = True
				count = 0
				while keepReading2:
					line = infile.readline()
					if line.strip().split()[0].lower() in ['/','end']:
						keepReading2 = False
					else:
						ng_dxyz[count] = floatD(line.strip().split()[0])
						count = count + 1
			elif key in ['/','end']: keepReading = False
		new_grid = pgrid(ng_type,ng_lower_bounds,ng_upper_bounds,bounds_key,ng_origin,ng_nxyz,
						 ng_dxyz,gravity_key,ng_gravity,ng_filename) 		# create an empty grid
		self._grid = new_grid
	
	def _write_grid(self,outfile):
		self._header(outfile,headers['grid'])
		grid = self.grid
		outfile.write('GRID\n')
		outfile.write('\tTYPE\t' + grid.type + '\n')
		if grid.bounds_bool:
			outfile.write('\tBOUNDS\n')
			outfile.write('\t\t')
			for i in range(3):
				outfile.write(str(grid.lower_bounds[i]) + ' ')
			outfile.write('\n')
			outfile.write('\t\t')
			for i in range(3):
				outfile.write(str(grid.upper_bounds[i]) + ' ')
			outfile.write('\n')
			outfile.write('\tEND\n')
		else:
			outfile.write('\tDXYZ\n')
			for i in range(3):
				outfile.write('\t\t' + str(grid.dxyz[i]) + '\n')
			outfile.write('\tEND\n')
		outfile.write('\tORIGIN' + ' ')
		for i in range(3):
			outfile.write(str(grid.origin[i]) + ' ')
		outfile.write('\n')
		outfile.write('\tNXYZ' + ' ')
		for i in range(3):
			outfile.write(str(grid.nxyz[i]) + ' ')
		outfile.write('\n')
		if grid.gravity_bool:
			outfile.write('\tGRAVITY' + ' ')
			for i in range(3):
				outfile.write(str(grid.gravity[i]) + ' ')
			outfile.write('\n')
		if grid.type == 'unstructured':
			outfile.write('\tFILENAME' + grid.filename + '\n')
		outfile.write('END\n\n')
	
	def _read_timestepper(self,infile,line):
		p = ptimestepper()
		np_ts_acceleration = p.ts_acceleration
		np_num_steps_after_cut = p.num_steps_after_cut 
		np_max_steps = p.max_steps
		np_max_ts_cuts = p.max_ts_cuts
		np_cfl_limiter = p.cfl_limiter
		np_initialize_to_steady_state = p.initialize_to_steady_state
		np_run_as_steady_state = p.run_as_steady_state
		np_max_pressure_change = p.max_pressure_change
		np_max_temperature_change = p.max_temperature_change
		np_max_concentration_change = p.max_concentration_change
		np_max_saturation_change = p.max_saturation_change

		keepReading = True

		while keepReading: 			# read through all cards
			line = infile.readline() 			# get next line
			key = line.strip().split()[0].lower() 		# take first keyword
			if key == 'ts_acceleration':
				np_ts_acceleration = int(line.split()[-1])
			elif key == 'num_steps_after_cut':
				np_num_steps_after_cut = int(line.split()[-1])
			elif key == 'max_steps':
				np_max_steps = int(line.split()[-1])
			elif key == 'max_ts_cuts':
				np_max_ts_cuts = int(line.split()[-1])
			elif key == 'cfl_limiter':
				np_cfl_limiter = float(line.split()[-1])
			elif key == 'initialize_to_steady_state':
				np_initialize_to_steady_state = True 
			elif key == 'run_as_steady_state':
				np_run_as_steady_state = True 
			elif key == 'max_pressure_change':
				np_max_pressure_change = float(line.split()[-1])
			elif key == 'max_temperature_change':
				np_max_temperature_change = float(line.split()[-1]) 
			elif key == 'max_concentration_change':
				np_max_concentration_change = float(line.split()[-1])
 			elif key == 'max_saturation_change':
				np_max_saturation_change = float(line.split()[-1])
			elif key in ['/','end']: keepReading = False

		new_timestep = ptimestepper(np_ts_acceleration,np_num_steps_after_cut,np_max_steps,
					    np_max_ts_cuts,np_cfl_limiter,np_initialize_to_steady_state,
					    np_run_as_steady_state,np_max_pressure_change,
					    np_max_temperature_change,np_max_concentration_change,
		                            np_max_saturation_change)

		self._timestepper = new_timestep
	
	def _write_timestepper(self,outfile):
		self._header(outfile,headers['timestepper'])
		outfile.write('TIMESTEPPER\n')
		if self.timestepper.ts_acceleration:
                  outfile.write('\t' + 'TS_ACCELERATION ' + str(self.timestepper.ts_acceleration) + '\n')
		if self.timestepper.num_steps_after_cut:
                  outfile.write('\t' + 'NUM_STEPS_AFTER_CUT ' + 
                                str(self.timestepper.num_steps_after_cut) + '\n')
		if self.timestepper.max_steps:
                  outfile.write('\t' + 'MAX_STEPS ' + str(self.timestepper.max_steps) + '\n')
		if self.timestepper.cfl_limiter:
                  outfile.write('\t' + 'CFL_LIMITER ' + str(self.timestepper.cfl_limiter) + '\n')
		if self.timestepper.initialize_to_steady_state:
                  outfile.write('\t' + 'INITIALIZE_TO_STEADY_STATE ' + '\n')
		if self.timestepper.run_as_steady_state:
                  outfile.write('\t' + 'RUN_AS_STEADY_STATE ' + '\n')
		if self.timestepper.max_pressure_change:
                  outfile.write('\t' + 'MAX_PRESSURE_CHANGE' + str(self.timestepper.max_pressure_change) + '\n')
		if self.timestepper.max_temperature_change:
                  outfile.write('\t' + 'MAX_TEMPERATURE_CHANGE' + 
                                str(self.timestepper.max_temperature_change) + '\n')
		if self.timestepper.max_concentration_change:
                  outfile.write('\t' + 'MAX_CONCENTRATION_CHANGE' +
                                str(self.timestepper.max_concentration_change) + '\n')
		if self.timestepper.max_saturation_change:
                  outfile.write('\t' + 'MAX_SATURATION_CHANGE' + 
                                str(self.timestepper.max_saturation_change) + '\n')
		outfile.write('END\n')

	def _read_prop(self,infile,line):
		np_name = line.split()[-1] 		# property name
		np_id = None
		p = pmaterial(0,'')				# assign defaults before reading in values
		np_porosity=p.porosity
		np_tortuosity=p.tortuosity
		np_density=p.density
		np_specific_heat=p.specific_heat
		np_cond_dry=p.cond_dry
		np_cond_wet=p.cond_wet
		np_saturation=p.saturation
		np_permeability=p.permeability

		keepReading = True

		while keepReading: 			# read through all cards
			line = infile.readline() 			# get next line
			key = line.strip().split()[0].lower() 		# take first keyword
			if key == 'id':
				np_id = int(line.split()[-1])
			elif key == 'porosity':
				np_porosity = floatD(line.split()[-1])
			elif key == 'tortuosity':
				np_tortuosity = floatD(line.split()[-1])
			elif key == 'rock_density':
				np_density = floatD(line.split()[-1])
			elif key == 'specific_heat':
				np_specific_heat = floatD(line.split()[-1])
			elif key == 'thermal_conductivity_dry':
				np_cond_dry = floatD(line.split()[-1])
			elif key == 'thermal_conductivity_wet':
				np_cond_wet = floatD(line.split()[-1])
			elif key == 'saturation_function':
				np_saturation = line.split()[-1]
			elif key == 'permeability':
				keepReading2 = True
				while keepReading2:
					line = infile.readline() 			# get next line
					key = line.split()[0].lower() 		# take first keyword
					if key == 'perm_x':
						np_permeability[0] = floatD(line.split()[-1])
					elif key == 'perm_y':
						np_permeability[1] = floatD(line.split()[-1])
					elif key == 'perm_z':
						np_permeability[2] = floatD(line.split()[-1])
					elif key in ['/','end']: keepReading2 = False
			elif key in ['/','end']: keepReading = False
		new_prop = pmaterial(np_id,np_name,np_porosity,np_tortuosity,np_density,
		                     np_specific_heat,np_cond_dry,np_cond_wet,
				             np_saturation,np_permeability) 		# create an empty material property

		self._proplist.append(new_prop)	
	
	def _write_prop(self,outfile):
		self._header(outfile,headers['material_property'])
		for prop in self.proplist:
			outfile.write('MATERIAL_PROPERTY\t')
			outfile.write(prop.name+'\n')
			outfile.write('\tID\t'+str(prop.id)+'\n')
			outfile.write('\tPOROSITY\t'+str(prop.porosity)+'\n')
			outfile.write('\tTORTUOSITY\t'+str(prop.tortuosity)+'\n')
			outfile.write('\tROCK_DENSITY\t'+str(prop.density)+'\n')
			outfile.write('\tSPECIFIC_HEAT\t'+str(prop.specific_heat)+'\n')
			outfile.write('\tTHERMAL_CONDUCTIVITY_DRY\t'+str(prop.cond_dry)+'\n')
			outfile.write('\tTHERMAL_CONDUCTIVITY_WET\t'+str(prop.cond_wet)+'\n')
			outfile.write('\tSATURATION_FUNCTION\t'+prop.saturation+'\n')
			outfile.write('\tPERMEABILITY\n')
			outfile.write('\t\tPERM_X\t'+strD(prop.permeability[0])+'\n')
			outfile.write('\t\tPERM_Y\t'+strD(prop.permeability[1])+'\n')
			outfile.write('\t\tPERM_Z\t'+strD(prop.permeability[2])+'\n')
			outfile.write('\tEND\n')
			outfile.write('END\n\n')
	
	def _read_time(self,infile):
		p = ptime()
		np_tf = p._tf
		np_dti = p._dti
		np_dtf = p._dtf
		np_dtf_lv = p._dtf_lv
		np_dtf_li = p._dtf_li
		np_dtf_i = p._dtf_i
		
		keepReading = True
		while keepReading:
			line = infile.readline() 			# get next line
			key = line.split()[0].lower() 		# take first keyword
			if key == 'final_time':
				tstring = line.split()[1:]
				if len(tstring) == 2:
					if tstring[-1] == 'y':
						np_tf = floatD(tstring[0])*365.25*24*3600
				else:
					np_tf = floatD(tstring[0])
			elif key == 'initial_timestep_size':
				tstring = line.split()[1:]
				if len(tstring) == 2:
					if tstring[-1] == 'y':
						np_dti = floatD(tstring[0])*365.25*24*3600
				else:
					np_dti = floatD(tstring[0])
			elif key == 'maximum_timestep_size':
				if ('at' not in line):
					tstring = line.split()[1:]
					if len(tstring) == 2:
						if tstring[-1] == 'y':
							np_dtf = floatD(tstring[0])*365.25*24*3600
					else:
						np_dtf = floatD(tstring[0])
						
				elif ('at' in line):
					## Read maximum_timestep_size with AT keyword 
					if (key == 'maximum_timestep_size'):
						
						#Read before AT
						tstring = line.split()[1:]
						if len(tstring) >= 2:
							if tstring[1] == 'y': # Detect the y after 1st #, not the last y on the line
								np_dtf_lv.append(1)
								np_dtf_lv[np_dtf_i] = floatD(tstring[0])*365.25*24*3600
						else:
							np_dtf_lv[np_dtf_i] = floatD(tstring[0])
							
						#Read after AT
						at_i = tstring.index('at') # Find index # in list (Not string)
						tstring = line.split()[at_i+2:] # Use string only after 'at'
						
						if len(tstring) == 2:
							if tstring[-1] == 'y':
								np_dtf_li.append(1)
								np_dtf_li[np_dtf_i] = floatD(tstring[0])*365.25*24*3600
						else:
							np_dtf_li[np_dtf_i] = floatD(tstring[0])
							
						np_dtf_i = np_dtf_i + 1
							
			elif key in ['/','end']: keepReading = False
			
		# Craete new empty time object and assign values read in.	
		new_time = ptime(np_tf,np_dti,np_dtf,np_dtf_lv,np_dtf_li,np_dtf_i)
		self._time = new_time

	def _write_time(self,outfile):
		self._header(outfile,headers['time'])
		time = self.time
		outfile.write('TIME\n')
		outfile.write('\tFINAL_TIME\t')
		
		# write FINAL_TIME statement
		if time.tf>365.25*3600*24*0.1:
			outfile.write(str(time.tf/(365.25*24*3600))+' y\n')
		else:
			outfile.write(str(time.tf)+'\n')
		outfile.write('\tINITIAL_TIMESTEP_SIZE\t')
		
		# write INITIAL_TIMESTEP_SIZE statement
		if time.dti>365.25*3600*24*0.1:
			outfile.write(str(time.dti/(365.25*24*3600))+' y\n')
		else:
			outfile.write(str(time.dti)+'\n')
		
		# write MAXIMUM_TIMESTEP_SIZE statement	
			outfile.write('\tMAXIMUM_TIMESTEP_SIZE\t')
			if time.dtf>365.25*3600*24*0.1:
				outfile.write(str(time.dtf/(365.25*24*3600))+' y\n')
			else:
				outfile.write(str(time.dtf)+'\n')
						
		# write more MAXIMUM_TIMESTEP_SIZE statements if applicable
		for i in range(0, time.dtf_i):
			# write before AT
			outfile.write('\tMAXIMUM_TIMESTEP_SIZE\t')
			if time.dtf_lv[i]>365.25*3600*24*0.1:
				outfile.write(str(time.dtf_lv[i]/(365.25*24*3600))+' y')
			else:
				outfile.write(str(time.dtf_lv[i]))
			# write after AT
			outfile.write(' at ')
			if time.dtf_li[i]>365.25*3600*24*0.1:
				outfile.write(str(time.dtf_li[i]/(365.25*24*3600))+' y\n')
			else:
				outfile.write(str(time.dtf_li[i])+'\n')				
		outfile.write('END\n\n')
		
	def _read_nsolver(self,infile,line):
		np_name = line.split()[-1].lower()	# newton solver type - tran_solver or flow_solver
		p = pnsolver('')		# Assign Defaults
		np_atol = p.atol
		np_rtol = p.rtol
		np_stol = p.stol
		np_dtol = p.dtol
		np_itol = p.itol
		np_max_it = p.max_it
		np_max_f = p.max_f
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first key word
			
			if key == 'atol':
				np_atol = floatD(line.split()[-1])
			if key == 'rtol':
				np_rtol = floatD(line.split()[-1])
			if key == 'stol':
				np_stol = floatD(line.split()[-1])
			if key == 'dtol':
				np_dtol = floatD(line.split()[-1])
			if key == 'itol':
				np_itol = floatD(line.split()[-1])
			if key == 'maxit':
				np_max_it = float(line.split()[-1])
			if key == 'maxf':
				np_max_f = float(line.split()[-1])
			elif key in ['/','end']: keepReading = False
		
		new_nsolver = pnsolver(np_name,np_atol,np_rtol,np_stol,np_dtol,np_itol,
					np_max_it,np_max_f)	# Create an empty newton solver
		self._nsolver = new_nsolver
		
	def _write_nsolver(self,outfile):
		self._header(outfile,headers['newton_solver'])
		nsolver = self.nsolver
		outfile.write('NEWTON_SOLVER\n')
		
		# Write Newton Solver Type - Not certain this is correct.
		if nsolver.name == 'flow':	# default
			outfile.write('\tTRAN,DEFAULT\n')
		elif nsolver.name == 'tran':
			outfile.write('\tTRAN,TRANSPORT\n')
			
		outfile.write('\tATOL\t' + strD(nsolver.atol) + '\n')
		outfile.write('\tRTOL\t' + strD(nsolver.rtol) + '\n')
		outfile.write('\tSTOL\t' + strD(nsolver.stol) + '\n')
		outfile.write('\tDTOL\t' + strD(nsolver.dtol) + '\n')
		outfile.write('\tITOL\t' + strD(nsolver.itol) + '\n')
		outfile.write('\tMAXIT\t' + str(nsolver.max_it) + '\n')
		outfile.write('\tMAXF\t' + str(nsolver.max_f) + '\n')
		outfile.write('END\n\n')
	
	def _read_output(self,infile):
		p = poutput()
		np_mass_balance = p.mass_balance
		np_print_column_ids = p.print_column_ids
		np_periodic_observation_timestep = p.periodic_observation_timestep
		np_format = p.format
		np_velocities = p.velocities
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first key word
			
			if key == 'periodic_observation':
				tstring = line.strip().split()[1].lower()	# Read the 2nd word
				if tstring == 'timestep':
					np_periodic_observation_timestep = int(line.split()[-1])
# Needed later			#elif tstring == 'time':
					#np_periodic_observation_time = float(line.split()[-1])	
			elif key == 'print_column_ids':
				np_print_column_ids = 'PRINT_COLUMN_IDS'
			elif key == 'format':
				tstring = (line.strip().split()[1:])
				tstring = ' '.join(tstring).lower()	# Convert list into a string seperated by a space
				#tstring.lower()
				if tstring == 'tecplot block':
					np_format.append('TECPLOT BLOCK')
				elif tstring == 'tecplot point':
					np_format.append('TECPLOT POINT')
				elif tstring == 'hdf5':
					np_format.append('HDF5')
				elif tstring == 'vtk':
					np_format.append('VTK')
			elif key == 'velocities':
				np_velocities = 'VELOCITIES'
			elif key == 'mass_balance':
				np_mass_balance = 'MASS_BALANCE'
			elif key in ['/','end']: keepReading = False
			
		# Create new empty output object and assign values read in.
		new_output = poutput(np_mass_balance,np_print_column_ids,
					np_periodic_observation_timestep,np_format,
					np_velocities)	# Create an empty output
		self._output = new_output	# Values read in are assigned now'
		
	def _write_output(self,outfile):
		self._header(outfile,headers['output'])
		output = self.output
		
		# Write Output - if used so null/None entries are not written
		outfile.write('OUTPUT\n')
# This is here on purpose - Needed later
		#if output.periodic_observation_time:
			#outfile.write('\tPERIODIC_OBSERVATION TIME\t'+
					#str(output.periodic_observation_time)+'\n')		
		if output.periodic_observation_timestep:
			outfile.write('\tPERIODIC_OBSERVATION TIMESTEP\t'+
					str(output.periodic_observation_timestep)+'\n')
		if output.print_column_ids:
			outfile.write('\t'+output.print_column_ids+'\n')
		if output.format:
			outfile.write('\tFORMAT\n')
			for i in range(0, len(output.format)):
				outfile.write('\t\t' + str(output.format[i]) + '\n')
		if output.velocities:
			outfile.write('\t'+output.velocities+'\n')
		if output.mass_balance:
			outfile.write('\t'+output.mass_balance+'\n')
		outfile.write('END\n\n')
		
	def _read_fluid(self,infile):
		p = pfluid()
		np_diffusion_coefficient = p.diffusion_coefficient
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first
				
			if key == 'diffusion_coefficient':
				np_diffusion_coefficient = floatD(line.split()[-1]) # Read last entry
			elif key in ['/','end']: keepReading = False
		
		# Create new employ fluid properties object and assign read in values to it
		new_fluid = pfluid(np_diffusion_coefficient)
		self._fluid = new_fluid
			
	def _write_fluid(self,outfile):
		self._header(outfile,headers['fluid_property'])
		fluid = self.fluid
		outfile.write('FLUID_PROPERTY\n')
		
		# Write out requested (not null) fluid properties
		if fluid.diffusion_coefficient:
			outfile.write('\tDIFFUSION_COEFFICIENT\t' + 
					strD(fluid.diffusion_coefficient) + '\n') # Read last entry
		outfile.write('END\n\n')
		
	def _read_saturation(self,infile,line):
		np_name = line.split()[-1].lower()	# saturation function name, passed in.
		p = psaturation('')	# assign defaults before reading in values
		np_permeability_function_type = p.permeability_function_type
		np_saturation_function_type = p.saturation_function_type
		np_residual_saturation_liquid = p.residual_saturation_liquid
		np_residual_saturation_gas = p.residual_saturation_gas
		np_a_lambda = p.a_lambda
		np_alpha = p.alpha
		np_max_capillary_pressure = p.max_capillary_pressure
		np_betac = p.betac
		np_power = p.power
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first  key word
			
			if key == 'permeability_function_type':
				np_permeability_function_type = line.split()[-1]
			elif key == 'saturation_function_type':
				np_saturation_function_type = line.split()[-1]
			elif key == 'residual_saturation_liquid':
				np_residual_saturation_liquid = float(line.split()[-1])
			elif key == 'residual_saturation_gas':
				np_residual_saturation_gas = float(line.split()[-1])
			elif key == 'residual_saturation':	# Alternative to check
				tstring = line.strip().split()[1].lower()	# take 2nd key word
				if tstring == 'liquid_phase':
					np_residual_saturation_liquid = float(line.split()[-1])
				elif tstring == 'gas_phase':
					np_residual_saturation_gas = float(line.split()[-1])
			elif key == 'lambda':
				np_a_lambda = floatD(line.split()[-1])
			elif key == 'alpha':
				np_alpha = floatD(line.split()[-1])
			elif key == 'max_capillary_pressure':
				np_max_capillary_pressure = floatD(line.split()[-1])
			elif key == 'betac':
				np_betac = floatD(line.split()[-1])
			elif key == 'power':
				np_power = floatD(line.split()[-1])
			elif key in ['/','end']: keepReading = False
			
		# Create an empty saturation function and assign the values read in
		new_saturation = psaturation(np_name,np_permeability_function_type,
						np_saturation_function_type,
						np_residual_saturation_liquid,
						np_residual_saturation_gas,np_a_lambda,
						np_alpha,np_max_capillary_pressure,
						np_betac,np_power)
		self._saturation = new_saturation
		
	def _write_saturation(self,outfile):
		self._header(outfile,headers['saturation_function'])
		saturation = self.saturation
		
		# Write out saturation properties that exist
		outfile.write('SATURATION_FUNCTION')
		if saturation.name:
			outfile.write('\t' + saturation.name + '\n')
		else:
			outfile.write('n')
		if saturation.permeability_function_type:
			outfile.write('\tPERMEABILITY_FUNCTION_TYPE\t' +
					saturation.permeability_function_type + '\n')
		if saturation.saturation_function_type:
			outfile.write('\tSATURATION_FUNCTION_TYPE\t' + 
					saturation.saturation_function_type + '\n')
		if saturation.residual_saturation_liquid or saturation.residual_saturation_liquid ==0:
			outfile.write('\tRESIDUAL_SATURATION LIQUID_PHASE\t' + 
					str(saturation.residual_saturation_liquid) + '\n')
		if saturation.residual_saturation_gas or saturation.residual_saturation_gas == 0:
			outfile.write('\tRESIDUAL_SATURATION GAS_PHASE\t' +
					str(saturation.residual_saturation_gas) + '\n')
		if saturation.a_lambda:
			outfile.write('\tLAMBDA\t' + strD(saturation.a_lambda) + '\n')
		if saturation.alpha:
			outfile.write('\tALPHA\t' + strD(saturation.alpha) + '\n')
		if saturation.max_capillary_pressure:
			outfile.write('\tMAX_CAPILLARY_PRESSURE\t' + 
					strD(saturation.max_capillary_pressure) + '\n')
		if saturation.betac:
			outfile.write('\tBETAC\t' + strD(saturation.betac) + '\n')
		if saturation.power:
			outfile.write('\tPOWER\t' + strD(saturation.power) + '\n')
		outfile.write('END\n\n')
		
	def _header(self,outfile,header):
		if not header: return
		ws = '# '
		pad = int(np.floor((80 - len(header) - 4)/2))
		for i in range(pad): ws+='='
		ws+=' '+header+' '
		for i in range(pad): ws+='='
		ws+='\n'
		outfile.write(ws)
	
	def _get_grid(self): return self._grid
	grid = property(_get_grid) #: (**)
	def _get_time(self): return self._time
	time = property(_get_time) #: (**)
	def _get_proplist(self): return self._proplist
	proplist = property(_get_proplist) #: (**) list of material properties
	def _get_prop(self): return dict([(p.id,p) for p in self.proplist]+[(p.name,p) for p in self.proplist])
	prop = property(_get_prop) #: (**) dictionary of material properties, indexable by ID or name
	def _get_filename(self): return self._filename
	def _set_filename(self,value): self._filename = value
	filename = property(_get_filename, _set_filename) #: (**)
	def _get_mode(self): return self._mode
	mode = property(_get_mode) #: (**)
	def _get_timestepper(self): return self._timestepper
	timestepper = property(_get_timestepper) #: (**)
	def _get_nsolver(self): return self._nsolver
	nsolver = property(_get_nsolver) #: (**)
	def _get_output(self): return self._output
	output = property(_get_output) #: (**)	
	def _get_fluid(self): return self._fluid
	fluid = property(_get_fluid) #: (**)
	def _get_saturation(self): return self._saturation
	saturation = property(_get_saturation) #: (**)
