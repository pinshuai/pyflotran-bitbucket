""" Class for pflotran data """
print	# Makes console output a little easier to read

import numpy as np
from copy import deepcopy
from copy import copy
import os,time
import platform
#from subprocess import Popen, PIPE
import subprocess

from ptool import*
from pdflt import*

dflt = pdflt()

WINDOWS = platform.system()=='Windows'
if WINDOWS: copyStr = 'copy'; delStr = 'del'; slash = '\\'
else: copyStr = 'cp'; delStr = 'rm'; slash = '/'

# Multiple classes/key words - allowed strings
time_units_allowed = ['s', 'sec','m', 'min', 'h', 'hr', 'd', 'day', 'w', 'week', 'mo', 'month', 'y']
solver_names_allowed = ['transport', 'tran', 'flow'] # newton and linear

# mode - allowed strings
mode_names_allowed = ['richards', 'mphase', 'flash2', 'thc', 'th', 'immis']

# grid - allowed strings
grid_types_allowed = ['structured', 'structured_mimetic', 'unstructured', 'amr']
grid_symmetry_types_allowed = ['cartesian', 'cylindrical', 'spherical'] # cartesian is default in pflotran

# output - allowed strings
output_formats_allowed = ['TECPLOT BLOCK', 'TECPLOT POINT', 'HDF5', 
			  'HDF5 MULTIPLE_FILES', 'MAD', 'VTK']

# saturation_function - allowed strings
saturation_function_types_allowed = ['VAN_GENUCHTEN', 'BROOKS_COREY', 'THOMEER_COREY', 
				     'NMT_EXP', 'PRUESS_1']
permeability_function_types_allowed = ['VAN_GENUCHTEN', 'MUALEM', 'BURDINE', 
				       'NMT_EXP', 'PRUESS_1']

# material_property, region, initial_condition, boundary_condition, source_sink, stratigraphy_couplers - manual does not appear to document all valid entries

# flow_conditions - allowed strings
flow_condition_type_names_allowed = ['PRESSURE', 'RATE', 'FLUX', 'TEMPERATURE', 
				'CONCENTRATION', 'SATURATION', 'ENTHALPY']
pressure_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient', 'conductance', 'seepage']
rate_types_allowed = ['mass_rate', 'volumetric_rate', 'scaled_volumetric_rate']
flux_types_allowed = ['dirichlet', 'neumann, mass_rate', 'hydrostatic, conductance',
		      'zero_gradient', 'production_well', 'seepage', 'volumetric',
		      'volumetric_rate', 'equilibrium']
temperature_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient']
concentration_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient']
saturation_types_allowed = ['dirichlet']
enthalpy_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient']

# transport_condition - allowed strings
transport_condition_types_allowed = ['dirichlet', 'dirichlet_zero_gradient', 'equilibrium', 
				     'neumann', 'mole', 'mole_rate', 'zero_gradient']

cards = ['co2_database','uniform_velocity','mode','checkpoint','restart','chemistry','grid',
		'timestepper','material_property','time','linear_solver','newton_solver',
		'output','fluid_property','saturation_function','region','observation',
		'flow_condition','transport_condition','initial_condition',
		'boundary_condition','source_sink','strata','constraint']
headers = ['co2 database path','uniform velocity','mode','checkpoint','restart','chemistry','grid',
	   'time stepping','material properties','time','linear solver','newton solver','output',
	   'fluid properties','saturation functions','regions','observation','flow conditions',
	   'transport conditions','initial condition','boundary conditions','source sink',
	   'stratigraphy couplers','constraints']
headers = dict(zip(cards,headers))

buildWarnings = []

def _buildWarnings(s):
	global buildWarnings
	buildWarnings.append(s)
	
class puniform_velocity(object):
	""" Class for uniform velocity card for transport observation. Optional
with transport problem when not coupling with a flow mode.
	
	:param value_list: List of variables of uniform_velocity. First 3 variables are vlx, vly, vlz in unit [m/s]. 4th variable specifies unit(dist./time) e.g. [14.4e0, 0.e0, 0.e0, 'm/yr']
	:type value_list: [float,float,float,str]
	"""
	
	def __init__(self, value_list=[0.0,0.0,0.0]):
		self._value_list = value_list
		
	def _get_value_list(self): return self._value_list
	def _set_value_list(self,value): self._value_list = value
	value_list = property(_get_value_list, _set_value_list) #: (**)

class pmaterial(object):
	""" Class for material property card. 
	Syntax to instantiate/create default class object is 'pmaterial()'. 
	Multiple material property objects can be created.
	
	:param id: Unique identifier of material property.
	:type id: int
	:param name: Name of material property. e.g. 'soil1'
	:type name: str
	:param porosity: Porosity of material property.
	:type porosity: float
	:param tortuosity: Tortuosity of material property.
	:type tortuosity: float
	:param density: Rock density of material property in kg/m^3.
	:type density: float
	:param specific_heat: Specific heat of material property in J/kg/K.
	:type specific_heat: float
	:param cond_dry: Thermal dry conductivity of material property in W/m/K.
	:type cond_dry: float
	:param cond_wet: Thermal wet conductivity of material property in W/m/K.
	:type cond_wet: float
	:param saturation: Saturation function of material property. e.g. 'sf2'
	:type saturation: str
	:param permeability: Permeability of material property. Input is a list of 3 floats. Uses diagonal permeability in unit order: k_xx [m^2], k_yy [m^2], k_zz [m^2]. e.g. [1.e-15,1.e-15,1.e-17]
	:type permeability: [float]*3
	"""

	# definitions are put on one line to work better with rst/latex/sphinx.
	def __init__(self, id=None, name='', porosity=None, tortuosity=None, density=None, specific_heat=None, cond_dry=None, cond_wet=None, saturation='', permeability=[]):
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
	""" Class for time property card. 
	Time works by using time values and by specifying its' unit of measurement. 
	Acceptable time units are: (s, m, h, d, mo, y). 
	Syntax to instantiate/create default class object is 'ptime()'. 
	List length (# of variables assigned to a parameter/attribute) of dtf_lv, dtf_lv_unit, dtf_li, dtf_li_unit should all be equal.
	
	:param tf: time final. 1st variable is time value. 2nd variable specifies time unit. e.g. [0.25e0, 'y']
	:type tf: [float, str]
	:param dti: delta (change) time initial a.k.a. initial timestep size. 1st variable is time value. 2nd variable specifies time unit. e.g. [0.25e0, 'y']
	:type dti: [float, str]
	:param dtf: delta (change) time final a.k.a. maximum timestep size. 1st variable is time value. 2nd variable specifies time unit. e.g. [50.e0, 'y']
	:type dtf: [float, str]
	:param dtf_list: delta (change) time final list a.ka. maximum timestep size. The dtf lists are for multiple max time step entries at specified time intervals. Uses key word 'at'. Input is a list that can have multiple lists appended to it. e.g. time.dtf_list.append([1.e2, 's', 5.e3, 's'])
	:type dtf_list: [ [float, str, float, str] ]
	"""
	
	# definitions are put on one line to work better with rst/latex/sphinx.
	def __init__(self, tf=[], dti=[], dtf=[], dtf_list=[]):
		self._tf = tf		# Final Time, 2nd parameter is unit, same for all other 
					# variables except dtf_i
		self._dti = dti		# Initial Timestep Size
		self._dtf = dtf		# Maximum Timestep Size
		self._dtf_list = dtf_list # Maximum Timestep Size using keyword 'at'
					  # Lists (manually) are assigned to lists
		
	def _get_tf(self): return self._tf
	def _set_tf(self,value): self._tf = value
	tf = property(_get_tf, _set_tf) #: (**)
	def _get_dti(self): return self._dti
	def _set_dti(self,value): self._dti = value
	dti = property(_get_dti, _set_dti) #: (**)s
	def _get_dtf(self): return self._dtf
	def _set_dtf(self,value): self._dtf = value
	dtf = property(_get_dtf, _set_dtf) #: (**)
	
	# The dtf lists are for multiple max time step entries at specified time intervals
	def _get_dtf_list(self): return self._dtf_list
	def _set_dtf_list(self,value): self._dtf_list = value
	dtf_list = property(_get_dtf_list, _set_dtf_list) #: (**)
	
class pgrid(object):
	""" Class for grid property card. *Required. 
	Defines the discretization scheme, the type of grid and resolution, and the geometry employed in the simulation.
	
	:param type: Grid type. Valid entries include: 'structured', 'structured_mimetic', 'unstructured', 'amr'. e.g. 'structured'
	:type type: str
	:param lower_bounds: Lower/Minimum 3D boundaries coordinates in order of x_min, y_min, z_min. Input is a list of 3 floats. e.g. [0.e0, 0.e0, 0.e0]
	:type lower_bounds: [float]*3
	:param upper_bounds: Upper/Maximum 3D boundaries coordinates in order of x_max, y_max, z_max. Input is a list of 3 floats. e.g. [321.e0, 1.e0, 51.e0]
	:type lower_bounds: [float]*3
	:param origin: Coordinates of grid origin. Optional. Input is a list of 3 floats. Default: [0.e0, 0.e0, 0.e0]
	:type origin: [float]*3
	:param nxyz: Number of grid cells in x,y,z directions. Only works with type='structured'. Input is a list of 3 floats. e.g. [107, 1, 51]
	:type nxyz: [float]*3
	:param dxyz: Specifies grid spacing of structured cartesian grid in order of dx, dy, dz. Input is a list of 3 floats. e.g. [5, 5, 5]
	:type dxyz: [float]*3
	:param gravity: Specifies gravity vector in unit 'm/s^2'. Input is a list of 3 floats. Default: [0, 0, 9.8068]
	:type gravity: [float]*3
	:param filename: Specify name of file containing grid information. Only works with type='unstructured'.
	:type filename: str
	"""

	# definitions are put on one line to work better with rst/latex/sphinx.
	def __init__(self, type='structured', lower_bounds=[0.0,0.0,0.0], upper_bounds=[50.0,50.0,50.0], origin=[0.0,0.0,0.0],nxyz=[10,10,10], dxyz=[5,5,5], gravity=[0.0,0.0,-9.8068], filename=''):
		self._type = type
		self._lower_bounds = lower_bounds
		self._upper_bounds = upper_bounds
		self._origin = origin
		self._nxyz = nxyz
		self._dxyz = dxyz
		self._gravity = gravity
		self._filename = filename

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


class pmode(object):
	""" Class for mode card. Determines the flow mode. 
	Richards (variably saturated porous media); 
	MPH, MPHASE, FLASH2 (CO_2 + H_2O); 
	THC (Thermal-Hydrologic-Chemical, in progress); 
	IMMIS, THS(Immisible).
	
	:param name: Specify mode name. Options include: 'richards', 'mphase', 'mph','flash2', 'thc', 'immis', 'ims', 'ths'.
	:type name: str
	"""

	def __init__(self, name=''):
		self._name = name

	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name)

class ptimestepper(object):
	""" Class for time stepper card. Controls time stepping.
	
	:param ts_acceleration: Integer indexing time step acceleration ramp (expert users only) [5]. Timestepper acceleration. e.g. 8
	:type ts_acceleration: int
	:param num_steps_after_cut: Number of time steps after a time step cut that the time step size is held constant [5].
	:type num_steps_after_cut: int
	:param max_steps: Maximum time step after which the simulation will be terminated [999999].
	:type max_steps: int
	:param max_ts_cuts: Maximum number of consecutive time step cuts before the simulation is terminated with plot of the current solution printed to a XXX_cut_to_failure.tec file for debugging [16].
	:type max_ts_cuts: int
	:param cfl_limiter: The maximum CFL number allowed for transport. Enables Courant (CFL) number lim-iting on the transport time step.
	:type cfl_limiter: float
	:param initialize_to_steady_state: Boolean flag indicating that the simulation is to be run as steady state (Warning: not robust).
	:type initialize_to_steady_state: bool - True or False
	:param run_as_steady_state: Boolean flag requesting that a steady state solution be computed based on boundary and initial conditions at the beginning of the simulation (Warning: not robust).
	:type run_as_steady_state: bool - True or False
	:param max_pressure_change: Maximum change in pressure for a time step [5.d4 Pa].
	:type max_pressure_change: float
	:param max_temperature_change: Maximum change in temperature for a time step [5 C].
	:type max_temperature_change: float
	:param max_concentration_change: Maximum change in pressure for a time step [1. mol/L].
	:type max_concentration_change: float
	:param max_saturation_change: Maximum change in pressure for a time step [0.5].
	:type max_saturation_change: float
	"""
	
	# definitions are put on one line to work better with rst/latex/sphinx.
	def __init__(self, ts_acceleration=None, num_steps_after_cut=None, max_steps=None, max_ts_cuts=None, cfl_limiter=None, initialize_to_steady_state=False, run_as_steady_state=False, max_pressure_change=None, max_temperature_change=None, max_concentration_change=None, max_saturation_change=None):
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

class plsolver(object):
	""" Class for linear solver card. Multiple linear solver objects can be created.
	
	:param name: Specify linear solver to use. Options include: 'tran', 'transport','flow'.
	:type name: str
	:param solver: Specify solver type: Options include: 'solver', 'krylov_type', 'krylov', 'ksp', 'ksp_type'
	:type solver: str
	"""
	
	def __init__(self, name='', solver=''):
		self._name = name	# TRAN, TRANSPORT / FLOW
		self._solver = solver	# Solver Type
		
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name)
	def _get_solver(self): return self._solver
	def _set_solver(self,value): self._solver = value
	solver = property(_get_solver, _set_solver)

class pnsolver(object):
	""" Class for newton solver card. Multiple newton solver objects can be created.
	
	:param name: Specify newton solver to use: Options include: 'tran', 'transport', 'tran_solver', 'flow_solver'. Default: 'flow_solver'
	:type name: str
	:param atol: 
	:type atol: float
	:param rtol: 
	:type rtol: float
	:param stol: 
	:type stol: float
	:param dtol: 
	:type dtol: float
	:param itol: 
	:type itol: float
	:param max_it: Cuts time step if the number of iterations exceed this value.
	:type max_it: int
	:param max_f: 
	:type max_f: int
	"""

	# definitions are put on one line to work better with rst/latex/sphinx.
	def __init__(self, name='', atol=None, rtol=None, stol=None, dtol=None, itol=None, max_it=None, max_f=None):
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
	"""Class for output options card.
	Acceptable time units (units of measurements) are: 's', 'min', 'h', 'd', 'w', 'mo', 'y'.
	
	:param time_list: List of time values. 1st variable specifies time unit to be used. Remaining variable(s) are floats.
	:type time_list: [str, float*]
	:param print_column_ids: Flag to indicate whether to print column numbers in observation and mass balance output files. Default: False
	:type print_column_ids: bool - True or False
	:param screen_periodic: Print to screen every <integer> time steps.
	:type screen_periodic: int
	:param periodic_time: 1st variable is value, 2nd variable is time unit.
	:type periodic_time: [float, str]
	:param periodic_timestep: 1st variable is value, 2nd variable is time unit.
	:type periodic_timestep: [float, str]
	:param periodic_observation_time: Output the results at observation points and mass balance output at specified output time. 1st variable is value, 2nd variable is time unit.
	:type periodic_observation_time: [float, str]
	:param periodic_observation_timestep: Outputs the results at observation points and mass balance output at specified time steps.
	:type periodic_observation_timestep: int
	:param format_list: Specify file format options to use for specifying the snapshop in time file type.. Input is a list of strings. Multiple formats can be specified. File format input options include: 'TECPLOT BLOCK' - TecPlot block format, 'TECPLOT POINT' - TecPlot point format (requires a single processor), 'HDF5' - produces single HDF5 file named pflotran.h5, 'HDF5 MULTIPLE_FILES' - produces a separate HDF5 file at each output time, 'MAD' - (not supported), 'VTK' - VTK format.
	:type format_list: [str]
	:param permeability: 
	:type permeability: bool - True or False
	:param porosity: 
	:type porosity: bool - True or False
	:param velocities: 
	:type velocities: bool - True or False
	:param mass_balance: Flag to indicate whether to output the mass balance of the system if this card is activated. It includes global mass balance as well as fluxes at all boundaries for water and chemical species specified for output in the CHEMISTRY card. For the MPHASE mode only global mass balances are provided including supercritical CO_2. Output times are controlled by PERIODIC_OBSERVATION TIMESTEP and TIME, and printout times. Default: False
	:type mass_balance: bool - True or False
	"""
	
	# definitions are put on one line to work better with rst/latex/sphinx.
	def __init__(self, time_list=[], print_column_ids=False, screen_periodic=None, periodic_time=[],periodic_timestep=[], periodic_observation_time=[], periodic_observation_timestep=None, format_list=[], permeability=False, porosity=False, velocities=False,  mass_balance=False):
		self._time_list = time_list
		self._print_column_ids = print_column_ids
		self._screen_periodic = screen_periodic	# int
		self._periodic_time = periodic_time	# [float, str]
		self._periodic_timestep = periodic_timestep # [float, str]
		self._periodic_observation_time = periodic_observation_time # [float, str] 
		self._periodic_observation_timestep = periodic_observation_timestep # int
		self._format_list = format_list
		self._permeability = permeability
		self._porosity = porosity
		self._velocities = velocities
		self._mass_balance = mass_balance
		
		
	def _get_time_list(self): return self._time_list
	def _set_time_list(self,value): self._time_list = value
	time_list = property(_get_time_list, _set_time_list)
	def _get_mass_balance(self): return self._mass_balance
	def _set_mass_balance(self,value): self._mass_balance = value
	mass_balance = property(_get_mass_balance, _set_mass_balance)	
	def _get_print_column_ids(self): return self._print_column_ids
	def _set_print_column_ids(self,value): self._print_column_ids = value
	print_column_ids = property(_get_print_column_ids, _set_print_column_ids)
	def _get_screen_periodic(self): return self._screen_periodic
	def _set_screen_periodic(self,value): self._screen_periodic = value
	screen_periodic = property(_get_screen_periodic, _set_screen_periodic)
	def _get_periodic_time(self): return self._periodic_time
	def _set_periodic_time(self,value): self._periodic_time = value
	periodic_time = property(_get_periodic_time, _set_periodic_time)
	def _get_periodic_timestep(self): return self._periodic_timestep
	def _set_periodic_timestep(self,value): self._periodic_timestep = value
	periodic_timestep = property(_get_periodic_timestep, _set_periodic_timestep)
	def _get_periodic_observation_time(self): return self._periodic_observation_time
	def _set_periodic_observation_time(self,value): self._periodic_observation_time = value
	periodic_observation_time = property(_get_periodic_observation_time, _set_periodic_observation_time)
	def _get_periodic_observation_timestep(self): return self._periodic_observation_timestep
	def _set_periodic_observation_timestep(self,value): self._periodic_observation_timestep = value
	periodic_observation_timestep = property(_get_periodic_observation_timestep, _set_periodic_observation_timestep)
	def _get_format_list(self): return self._format_list
	def _set_format_list(self,value): self._format_list = value
	format_list = property(_get_format_list, _set_format_list)
	def _get_permeability(self): return self._permeability
	def _set_permeability(self,value): self._permeability = value
	permeability = property(_get_permeability, _set_permeability)	
	def _get_porosity(self): return self._porosity
	def _set_porosity(self,value): self._porosity = value
	porosity = property(_get_porosity, _set_porosity)
	def _get_velocities(self): return self._velocities
	def _set_velocities(self,value): self._velocities = value
	velocities = property(_get_velocities, _set_velocities)
	
class pfluid(object):
	"""Class for fluid properties card.
	
	:param diffusion_coefficient: Unit of measurement is [m^2/s]. Default: 1.0000000e-09
	:type diffusion_coefficient: float
	"""
	
	def __init__(self, diffusion_coefficient=None):
		self._diffusion_coefficient = diffusion_coefficient
		
	def _get_diffusion_coefficient(self): return self._diffusion_coefficient
	def _set_diffusion_coefficient(self,value): self._diffusion_coefficient = value
	diffusion_coefficient = property(_get_diffusion_coefficient, _set_diffusion_coefficient)
	
class psaturation(object):
	"""Class for saturation functions
	
	:param name: Saturation function name. e.g. 'sf2'
	:type name: str
	:param permeability_function_type: Options include: 'VAN_GENUCHTEN', 'MUALEM', 'BURDINE', 'NMT_EXP', 'PRUESS_1'.
	:type permeability_function_type: str
	:param saturation_function_type: Options include: 'VAN_GENUCHTEN', 'BROOKS_COREY', 'THOMEER_COREY', 'NMT_EXP', 'PRUESS_1'.
	:type saturation_function_type: str
	:param residual_saturation: MODES: RICHARDS, TH, THC
	:type residual_saturation: float
	:param residual_saturation_liquid: MODES: MPHASE
	:type residual_saturation_liquid: float
	:param residual_saturation_gas: MODES: MPHASE
	:type residual_saturation_gas: float
	:param a_lambda: lambda
	:type a_lambda: float
	:param alpha: Pa^-1
	:type alpha: float
	:param max_capillary_pressure: Pa
	:type max_capillary_pressure: float
	:param betac: 
	:type betac: float
	:param power: 
	:type power: float
	"""
	
	# definitions are put on one line to work better with rst/latex/sphinx.
	def __init__(self, name='', permeability_function_type=None, saturation_function_type=None, residual_saturation=None, residual_saturation_liquid=None, residual_saturation_gas=None, a_lambda=None, alpha=None, max_capillary_pressure=None, betac=None, power=None):
		self._name = name
		self._permeability_function_type = permeability_function_type
		self._saturation_function_type = saturation_function_type
		self._residual_saturation = residual_saturation # float
		self._residual_saturation_liquid = residual_saturation_liquid # float
		self._residual_saturation_gas = residual_saturation_gas # float
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
	def _get_residual_saturation(self): return self._residual_saturation
	def _set_residual_saturation(self,value): self._residual_saturation = value
	residual_saturation = property(_get_residual_saturation, _set_residual_saturation)
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
	
class pregion(object):
	"""Class for regions. Multiple region objects can be created.
	The REGION keyword defines a set of grid cells encompassed by a volume or intersected by a plane or point, or a list of grid cell ids. The REGION name can then be used to link this set of grid cells to material properties, strata, boundary and initial conditions, source sinks, observation points, etc. Although a region may be defined though the use of (I, J, K) indices using the BLOCK keyword, the user is encouraged to define regions either through COORDINATES or lists read in from an HDF5 file in order to minimize the dependence of the input file on grid resolution. In the case of the FILE keyword, a list of grid cell ids is read from an HDF5 file where the region_name defines the HDF5 data set. It should be noted that given a region defined by a plane or point shared by two grid cells (e.g. a plane defining the surface between two grid cells), PFLOTRAN will select the upwind cell(s) as the region. Please note there is currently a minor glitch when using the add function for pregion. A false WARNING may print to console stating a region with name 'user_value' already exists when reading another input deck. This can occur as a result of a region name being used in another key words such as INITIAL_CONDITION. In cases where WARNING messages are produced due to this glitch, it will not affect the outcome of the program and can be ignored. The pdata script is incorrectly reading pregion key words more than once and automatically correcting itself which produces the warning message.

	:param name: Region name. Options include: 'all', 'top', 'west', 'east', 'well'.
	:type name: str
	:param coordinates_lower: Lower/minimum 3D coordinates for defining a volumetric, planar, or point region between two points in space in order of x1, y1, z1. e.g. [0.e0, 0.e0, 0.e0]
	:type coordinates_lower: [float]*3
	:param coordinates_upper: Upper/maximum 3D coordinates for defining a volumetric, planar, or point region between two points in space in order of x2, y2, z2. e.g.[321.e0, 1.e0,  51.e0]
	:type coordinates_upper: [float]*3
	:param face: Defines the face of the grid cell to which boundary conditions are connected. Options include: 'west', 'east', 'north', 'south', 'bottom', 'top'. (structured grids only).
	:type face: str
	"""
	
	def __init__(self,name='',coordinates_lower=[0.0,0.0,0.0],coordinates_upper=[0.0,0.0,0.0],
			face=None):
		self._name = name.lower()
		self._coordinates_lower = coordinates_lower	# 3D coordinates
		self._coordinates_upper = coordinates_upper	# 3D coordinates
		self._face = face
		
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value.lower()
	name = property(_get_name, _set_name)
	def _get_coordinates_lower(self): return self._coordinates_lower
	def _set_coordinates_lower(self,value): self._coordinates_lower = value
	coordinates_lower = property(_get_coordinates_lower, _set_coordinates_lower)
	def _get_coordinates_upper(self): return self._coordinates_upper
	def _set_coordinates_upper(self,value): self._coordinates_upper = value
	coordinates_upper = property(_get_coordinates_upper, _set_coordinates_upper)
	def _get_face(self): return self._face
	def _set_face(self,value): self._face = value
	face = property(_get_face, _set_face)
	
class pobservation(object):
	"""Class for observation card.
	The OBSERVATION card specifies a location (REGION) at which flow and transport results (e.g. pressure, saturation, flow velocities, solute concentrations, etc.) will be monitored in the output. The user must specify either a region or boundary condition to which the observation object is linked. The velocity keyword toggles on the printing of velocities at a point in space.
	Currently, only region is supported in PyFLOTRAN.
	
	:param region: Defines the name of the region (usually a point in space) to which the observation point is linked. e.g. 'obs'.
	:type region: str
	"""
	
	def __init__(self,region=None):
		self._region = region
		
	def _get_region(self): return self._region
	def _set_region(self,value): self._region = value
	region = property(_get_region, _set_region) 
	
class pflow(object):
	"""Class for flow conditions card. There can be multiple flow condition objects.
	The FLOW_CONDITION keyword specifies scalar or vector data sets to be associated with a given boundary or initial condition. For instance, to specify a hydrostatic boundary condition, the user would specify a condition with a pressure associated with a point in space (i.e. datum) in space and a gradient, both vector quantities. Note that in the case of a hydrostatic boundary condition, the vertical gradient specified in the input deck must be zero in order to enable the hydrostatic pressure calculation. Otherwise, the specified vertical gradient overrides the hydrostatic pressure. Transient pressures, temperatures, concentrations, datums, gradients, etc. are specified using the FILE filename combination for the name of the data set.
	
	:param name: Options include: 'initial', 'top', 'source'.
	:type name: str
	:param units_list: Not currently supported.
	:type units_list: [str]
	:param iphase: 
	:type iphase: int
	:param sync_timestep_with_update: Flag that indicates whether to use sync_timestep_with_update. Default: False.
	:type sync_timestep_with_update: bool - True or False
	:param varlist: Input is a list of pflow_variable objects. Sub-class of pflow. It is recommended to use dat.add(obj=pflow_variable) for easy appending. Use dat.add(index='pflow_variable.name' or dat.add(index=pflow_variable) to specify pflow object to add pflow_variable to. If no pflow object is specified, pflow_variable will be appended to the last pflow object appended to pdata. E.g. dat.add(variable, 'initial') if variable = pflow_variable and pflow.name='initial'.
	:type varlist: [pflow_variable]

	"""
	
	def __init__(self,name='',units_list=None,
			iphase=None,sync_timestep_with_update=False,
			varlist=[]):
		self._name = name.lower()	# Include initial, top, source
		self._units_list = units_list	# Specify type of units to display such as
						# time,length,rate,pressure,velocity, temperature,
						# concentration, and enthalpy.
						# May be used to determine each variable unit
		self._iphase = iphase			# Holds 1 int
		self._sync_timestep_with_update = sync_timestep_with_update	# Boolean
		self._varlist = varlist
		
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value.lower()
	name = property(_get_name, _set_name)
	def _get_units_list(self): return self._units_list
	def _set_units_list(self,value): self._units_list = value
	units_list = property(_get_units_list, _set_units_list)
	def _get_iphase(self): return self._iphase
	def _set_iphase(self,value): self._iphase = value
	iphase = property(_get_iphase, _set_iphase)
	def _get_sync_timestep_with_update(self): return self._sync_timestep_with_update
	def _set_sync_timestep_with_update(self,value): self._sync_timestep_with_update = value
	sync_timestep_with_update = property(_get_sync_timestep_with_update, _set_sync_timestep_with_update)
	def _get_varlist(self): return self._varlist
	def _set_varlist(self,value): self._varlist = value
	varlist = property(_get_varlist, _set_varlist)
	
	# Code below is an attempt to change the way sub-classes are added.
	# it's not necessary. (Attempting to make it possible to do a flow.add(variable)
	# instead of dat.add(variable). Current way of specifying which flow object to
	# add to is dat.add(variable,flow)
	'''
	# Definitions for sub-class of pflow object
	
	# Adds a new pflow_variable to a pflow object
	def add(self,variable,overwrite=False):
		
		# Establish super-class flow variable reference
		flow = super(self,)
		
		# check if flow_variable already exists
		if isinstance(self.flow_variable,pflow_variable):		
#			if flow_variable.name in super(self)._get_flow_variable(self).keys(): #testing
			if flow_variable.name in self._get_flow_variable(self).keys():
				if not overwrite:
					warning = 'WARNING: A flow_variable with name \''+str(flow_variable.name)+'\' already exists in flow with name \''+str(flow.name)+'\'. Flow_variable will not be defined, use overwrite = True in add() to overwrite the old flow_variable.'
					print warning; print
					_buildWarnings(warning)
					return
				else: # Executes if overwrite = True
					self.delete(self._get_flow_variable(flow)[flow_variable.name],
						    flow)
		
		# Add flow_variable to flow (as a sub-class) if flow_variable does
		# not exist in specified flow object
		if flow_variable not in flow.varlist:
			flow.varlist.append(flow_variable)
			
	def _get_flow_variable(self, flow):
		return dict([flow_variable.name, flow_variable] for flow_variable in flow.varlist if flow_variable.name)
	flow_variable = property(_get_flow_variable)#: (*dict[pflow_variable]*) Dictionary of pflow_variable objects in a specified flow object, indexed by flow_variable name
	'''
	
class pflow_variable(object):
	"""Sub-class of pflow for each kind of variable (Includes type and value). There can be multiple pflow_variable objects appended to a single pflow object.
	
	:param name: Indicates name for keyword TYPE under keyword FLOW_CONDITION. Options include: ['PRESSURE', 'RATE', 'FLUX', 'TEMPERATURE', 'CONCENTRATION', 'SATURATION', 'ENTHALPY'].
	:type name: str
	:param type: Indicates type that is associated with name under keyword TYPE. Options for PRESSURE include: 'dirichlet', 'hydrostatic', 'zero_gradient', 'conductance', 'seepage'. Options for RATE include: 'mass_rate', 'volumetric_rate', 'scaled_volumetric_rate'. Options for FLUX include: 'dirichlet', 'neumann, mass_rate', 'hydrostatic, conductance', 'zero_gradient', 'production_well', 'seepage', 'volumetric', 'volumetric_rate', 'equilibrium'. Options for TEMPERATURE include: 'dirichlet', 'hydrostatic', 'zero_gradient'. Options for CONCENTRATION include: 'dirichlet', 'hydrostatic', 'zero_gradient'. Options for SATURATION include: 'dirichlet'. Options for ENTHALPY include: 'dirichlet', 'hydrostatic', 'zero_gradient'
	:type type: str
	:param valuelist: Provide one or two values associated with a single Non-list alternative, do not use with list alternative. The 2nd float is optional.
	:type valuelist: [float, float]
	:param unit: Currently not supported. Non-list alternative, do not use with list alternative.
	:type unit: str
	:param time_unit_type: List alternative, do not use with non-list alternative attributes/parameters. 
	:type time_unit_type: str
	:param data_unit_type: List alternative, do not use with non-list alternative attributes/parameters.
	:type data_unit_type: str
	:param list: List alternative, do not use with non-list alternative attributes/parameters. Input is a list of pflow_variable_list objects. Sub-class of pflow_variable. The add function currently does not support adding pflow_variable_list to pflow_variable objects. Appending to can be done manually. e.g. variable.list.append(var_list) if var_list=pflow_variable_list.
	:type list: [pflow_variable_list]
	"""
	
	def __init__(self,name='',type=None, valuelist=[], unit='',
		     time_unit_type='', data_unit_type='', list=[]):
		self._name = name.lower()# Pressure,temp., concen.,enthalpy...(String)
		self._type = type	# hydrostatic, zero_gradient, dirichlet ...(String)
		
		# The Following attributes are a stand alone single list w/out lists
		# (e.g. Rate instead of Rate List)
		self._valuelist = valuelist	# Holds 2 floats - 2nd is optional
		self._unit = unit	# Possible to overide Parent class? - sorda?
		
		# Following attributes are used with lists (eg. Rate Lists instead of Rate)
		self._time_unit_type = time_unit_type # e.g. 'y'
		self._data_unit_type = data_unit_type # e.g. 'kg/s'
		self._list = list	# Holds a list of pflow_variable_lists objects
		
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value.lower()
	name = property(_get_name, _set_name)		
	def _get_type(self): return self._type
	def _set_type(self,value): self._type = value
	type = property(_get_type, _set_type)
	def _get_unit(self): return self._unit
	def _set_unit(self,value): self._unit = value
	unit = property(_get_unit, _set_unit)
	def _get_valuelist(self): return self._valuelist
	def _set_valuelist(self,value): self._valuelist = value
	valuelist = property(_get_valuelist, _set_valuelist)
	def _get_time_unit_type(self): return self._time_unit_type
	def _set_time_unit_type(self,value): self._time_unit_type = value
	time_unit_type = property(_get_time_unit_type, _set_time_unit_type)
	def _get_data_unit_type(self): return self._data_unit_type
	def _set_data_unit_type(self,value): self._data_unit_type = value
	data_unit_type = property(_get_data_unit_type, _set_data_unit_type)
	def _get_list(self): return self._list
	def _set_list(self,value): self._list = value
	list = property(_get_list, _set_list)
	
class pflow_variable_list(object):
	"""Sub-class of pflow_variable.
	Used for pflow_variables that are lists instead of standalone. 
	(e.g. Rate List instead of Rate)

	"""
	
	def __init__(self, time_unit_value=None, data_unit_value_list=[]):
		self._time_unit_value = time_unit_value # 1 float
		self._data_unit_value_list = data_unit_value_list # 2 floats? (2nd optional?)
		
	def _get_time_unit_value(self): return self._time_unit_value
	def _set_time_unit_value(self,value): self._time_unit_value = value
	time_unit_value = property(_get_time_unit_value, _set_time_unit_value)
	def _get_data_unit_value_list(self): return self._data_unit_value_list
	def _set_data_unit_value_list(self,value): self._data_unit_value_list = value
	data_unit_value_list = property(_get_data_unit_value_list, _set_data_unit_value_list)

class pinitial_condition(object):
	"""Class for initial condition - a condition coupler

	"""
	
	def __init__(self,flow=None,transport=None,region=None):
		self._flow = flow	# Flow Condition (e.g. initial)
		self._transport = transport
		self._region = region	# Define region (e.g. west, east, well)
		
	def _get_flow(self): return self._flow
	def _set_flow(self,value): self._flow = value
	flow = property(_get_flow, _set_flow)
	def _get_transport(self): return self._transport
	def _set_transport(self,value): self._transport = value
	transport = property(_get_transport, _set_transport)
	def _get_region(self): return self._region
	def _set_region(self,value): self._region = value
	region = property(_get_region, _set_region)
	
class pboundary_condition(object):
	"""Class for boundary conditions - a condition coupler

	"""
	
	def __init__(self,name='',flow=None,transport=None,region=None):
		self._name = name	# Name of boundary condition. (e.g. west, east)
		self._flow = flow	# Flow Condition (e.g. initial)
		self._transport = transport	# Transport Condition (e.g. river_chemistry)
		self._region = region	# Define region (e.g. west, east, well)
		
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name)
	def _get_flow(self): return self._flow
	def _set_flow(self,value): self._flow = value
	flow = property(_get_flow, _set_flow)
	def _get_transport(self): return self._transport
	def _set_transport(self,value): self._transport = value
	transport = property(_get_transport, _set_transport)
	def _get_region(self): return self._region
	def _set_region(self,value): self._region = value
	region = property(_get_region, _set_region)

class psource_sink(object):
	"""Class for source sink - a condition coupler

	"""
	
	def __init__(self,flow=None,region=None):
		self._flow = flow	# Flow Condition (e.g. initial)
		self._region = region	# Define region (e.g. west, east, well)
		
	def _get_flow(self): return self._flow
	def _set_flow(self,value): self._flow = value
	flow = property(_get_flow, _set_flow)
	def _get_region(self): return self._region
	def _set_region(self,value): self._region = value
	region = property(_get_region, _set_region)
	
class pstrata(object):
	"""Class for stratigraphy couplers

	"""
	
	def __init__(self,region=None,material=None):
		self._region = region
		self._material = material
		
	def _get_region(self): return self._region
	def _set_region(self,value): self._region = value
	region = property(_get_region, _set_region)
	def _get_material(self): return self._material
	def _set_material(self,value): self._material = value
	material = property(_get_material, _set_material)
	
class pcheckpoint(object):
	"""Class for checkpoint card. 
	Checkpoint files enable the restart of a simulation at any discrete point in simulation where a checkpoint file has been printed. When the CHECKPOINT card is included in the input deck, checkpoint files are printed every N time steps, where N is the checkpoint frequency, and at the end of the simulation, should the simulation finish or the be shut down properly mid-simulation using the WALL_CLOCK_STOP card. Checkpoint files are named pflotran.chkN, where N is the number of the timestep when the checkpoint file was printed. A file named restart.chk will also be written when PFLOTRAN properly terminates execution. One use of this file is to pick up from where the simulation stopped by increasing the final time. Checkpointing can be used to start from an initial steady-state solution, but note that porosity and permeability are checkpointed as there are scenarios where they can change over time. To override this behavior add: OVERWRITE_RESTART_FLOW_PARAMS to the input file to set porosity/permeability to their read-in values
	
	:param frequency: checkpoint_frequency.
	:type frequency: int
	:param overwrite: Currently not supported. Intended to be used with OVERWRITE_RESTART_FLOW_PARAMS.
	:type overwrite: bool - True or False
	"""
	
	def __init__(self, frequency=None, overwrite=False):
		self._frequency = frequency # int
		self._overwrite = overwrite # Intended for OVERWRITE_RESTART_FLOW_PARAMS, incomplete, uncertain how to write it.
		
	def _get_frequency(self): return self._frequency
	def _set_frequency(self,value): self._frequency = value
	frequency = property(_get_frequency, _set_frequency)
	
class prestart(object):
	"""Class for restart card.
	The RESTART card defines a checkpoint file from which the current simulation should be restarted. If a time is specified after the file name, the initial simulation time is set to that time. Defines the checkpoint filename to be read in to restart a simulation at the specified time.
	
	:param file_name: Specify file path and name for restart.chk file.
	:type file_name: str
	:param time_value: Specify time value.
	:type time_value: float
	:param time_unit: Specify unit of measurement to use for time. Options include: 's', 'sec','m', 'min', 'h', 'hr', 'd', 'day', 'w', 'week', 'mo', 'month', 'y'.
	:type time_unit: str
	"""
	
	def __init__(self, file_name='', time_value=None, time_unit=''):
		self._file_name = file_name	# restart.chk file name
		self._time_value = time_value	# float
		self._time_unit = time_unit	# unit of measurement to use for time - str
		
	def _get_file_name(self): return self._file_name
	def _set_file_name(self,value): self._file_name = value
	file_name = property(_get_file_name, _set_file_name)
	def _get_time_value(self): return self._time_value
	def _set_time_value(self,value): self._time_value = value
	time_value = property(_get_time_value, _set_time_value)
	def _get_time_unit(self): return self._time_unit
	def _set_time_unit(self,value): self._time_unit = value
	time_unit = property(_get_time_unit, _set_time_unit)
	
class pchemistry(object):
	"""Class for chemistry
	m_kinetics_list (RATE_CONSTANT) is not complete
	"""
	
	def __init__(self, pspecies_list=[], sec_species_list=[], gas_species_list=[],
		     minerals_list=[], m_kinetics_list=[], log_formulation=False,
		     database=None, activity_coefficients=None, molal=False, output_list=[] ):
		self.pspecies_list = pspecies_list	# primary_species (eg. 'A(aq') - string
		self._sec_species_list = sec_species_list # Secondary_species (E.g. 'OH-' - string
		self._gas_species_list = gas_species_list # E.g. 'CO2(g)'
		self._minerals_list = minerals_list	# E.g. 'Calcite'
		self._m_kinetics_list = m_kinetics_list	# has pchemistry_m_kinetic assigned to it
		self._log_formulation = log_formulation
		self._database = database		# Database path (String)
		self._activity_coefficients = activity_coefficients
		self._molal = molal		# boolean
		self._output_list = output_list		# incl. molarity/all, species and mineral names - string
		
	def _get_pspecies_list(self): return self._pspecies_list
	def _set_pspecies_list(self,value): self._pspecies_list = value
	pspecies_list = property(_get_pspecies_list, _set_pspecies_list)
	def _get_sec_species_list(self): return self._sec_species_list
	def _set_sec_species_list(self,value): self._sec_species_list = value
	sec_species_list = property(_get_sec_species_list, _set_sec_species_list)
	def _get_gas_species_list(self): return self._gas_species_list
	def _set_gas_species_list(self,value): self._gas_species_list = value
	gas_species_list = property(_get_gas_species_list, _set_gas_species_list)	
	def _get_minerals_list(self): return self._minerals_list
	def _set_minerals_list(self,value): self._minerals_list = value
	minerals_list = property(_get_minerals_list, _set_minerals_list)	
	def _get_m_kinetics_list(self): return self._m_kinetics_list
	def _set_m_kinetics_list(self,value): self._m_kinetics_list = value
	m_kinetics_list = property(_get_m_kinetics_list, _set_m_kinetics_list)
	def _get_log_formulation(self): return self._log_formulation
	def _set_log_formulation(self,value): self._log_formulation = value
	log_formulation = property(_get_log_formulation, _set_log_formulation)
	def _get_database(self): return self._database
	def _set_database(self,value): self._database = value
	database = property(_get_database, _set_database)
	def _get_activity_coefficients(self): return self._activity_coefficients
	def _set_activity_coefficients(self,value): self._activity_coefficients = value
	activity_coefficients = property(_get_activity_coefficients, _set_activity_coefficients)
	def _get_molal(self): return self._molal
	def _set_molal(self,value): self._molal = value
	molal = property(_get_molal, _set_molal)
	def _get_output_list(self): return self._output_list
	def _set_output_list(self,value): self._output_list = value
	output_list = property(_get_output_list, _set_output_list)

class pchemistry_m_kinetic(object):
	"""Sub-class for pchemistry. 
	mineral kinetics are assigned to m_kinetics_list in pchemistry.
	"""
	
	def __init__(self, name=None, rate_constant_list=[]):
		self.name = name
		self.rate_constant_list = rate_constant_list
	
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name)
	def _get_rate_constant_list(self): return self._rate_constant_list
	def _set_rate_constant_list(self,value): self._rate_constant_list = value
	rate_constant_list = property(_get_rate_constant_list, _set_rate_constant_list)
	
class ptransport(object):
	"""Class for transport conditions

	"""
	
	def __init__(self, name='', type=None, constraint_list_value=[],
		     constraint_list_type=[]):
		self._name = name	# e.g. initial, west, east
		self._type = type	# e.g. dirichlet, zero_gradient
		self._constraint_list_value = constraint_list_value
		self._constraint_list_type = constraint_list_type
		
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name)
	def _get_type(self): return self._type
	def _set_type(self,value): self._type = value
	type = property(_get_type, _set_type)
	def _get_constraint_list_value(self): return self._constraint_list_value
	def _set_constraint_list_value(self,value): self._constraint_list_value = value
	constraint_list_value = property(_get_constraint_list_value, _set_constraint_list_value)
	def _get_constraint_list_type(self): return self._constraint_list_type
	def _set_constraint_list_type(self,value): self._constraint_list_type = value
	constraint_list_type = property(_get_constraint_list_type, _set_constraint_list_type)
	
class pconstraint(object):
	"""Class for constraints

	"""
	
	def __init__(self, name='', concentration_list=[], mineral_list=[]):
		self._name = name.lower()
		self._concentration_list = concentration_list # Composed of pconstraint_concentration objects
		self._mineral_list = mineral_list # list of minerals
		
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value.lower()
	name = property(_get_name, _set_name)
	def _get_concentration_list(self): return self._concentration_list
	def _set_concentration_list(self,value): self._concentration_list = value
	concentration_list = property(_get_concentration_list, _set_concentration_list)
	def _get_mineral_list(self): return self._mineral_list
	def _set_mineral_list(self,value): self._mineral_list = value
	mineral_list = property(_get_mineral_list, _set_mineral_list)
		
class pconstraint_concentration(object):
	"""Concentration unit
	Sub-class for constraints

	"""
	
	def __init__(self, pspecies='', value=None, constraint='', element=''):
		self._pspecies = pspecies	# Primary Species Name (H+, O2(aq), etc.)
		self._value = value
		self._constraint = constraint	# (F, T, TOTAL_SORB, SC, etc.)
		self._element = element		# mineral or gas
		
	def _get_pspecies(self): return self._pspecies
	def _set_pspecies(self,value): self._pspecies = value
	pspecies = property(_get_pspecies, _set_pspecies)
	def _get_value(self): return self._value
	def _set_value(self,value): self._value = value
	value = property(_get_value, _set_value)
	def _get_constraint(self): return self._constraint
	def _set_constraint(self,value): self._constraint = value
	constraint = property(_get_constraint, _set_constraint)
	def _get_element(self): return self._element
	def _set_element(self,value): self._element = value
	element = property(_get_element, _set_element)
    
class pconstraint_mineral(object):
	"""Class for mineral in a constraint with vol. fraction and surface area

	"""

	def __init__(self, name='', volume_fraction=None, surface_area=None):
		self._name = name 
		self._volume_fraction = volume_fraction 
		self._surface_area = surface_area

	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name)
	def _get_volume_fraction(self): return self._volume_fraction
	def _set_volume_fraction(self,value): self._volume_fraction = value
	volume_fraction = property(_get_volume_fraction, _set_volume_fraction) 
	def _get_surface_area(self): return self._surface_area
	def _set_surface_area(self,value): self._surface_area = value
	surface_area = property(_get_surface_area, _set_surface_area)       
	
class pdata(object):
	"""Class for pflotran data file. Use 'from pdata import*' to access pdata library
	"""

	# definitions are put on one line to work better with rst/latex/sphinx.
	def __init__(self, filename='', work_dir=''):
		from copy import copy
		# Note that objects need to be instantiated when hard-coded when it's set to
		# None here.
		self._co2_database = ''
		self._uniform_velocity = puniform_velocity()
		self._mode = pmode()
		self._checkpoint = pcheckpoint()
		self._restart = prestart()
		self._chemistry = None
		self._grid = pgrid()
		self._timestepper = None
		self._proplist = []
		self._time = ptime()
		self._lsolverlist = []	# Possible to have 1 or 2 lsolver lists. FLOW/TRAN
		self._nsolverlist = []	# Possible to have 1 or 2 nsolver lists. FLOW/TRAN
		self._output = poutput()
		self._fluid = pfluid()
		self._saturation = None
		self._regionlist = []	# There are multiple regions
		self._observation = None
		self._flowlist = []
		self._transportlist = []
		self._initial_condition = pinitial_condition()
		self._boundary_condition_list = []
		self._source_sink = None
		self._strata_list = []
		self._constraint_list = []
		self._filename = filename
		
		# run object
		self._path = ppath(parent=self)
		self._running = False	# boolean indicating whether a simulation is in progress
		self.work_dir = work_dir
		if self.work_dir:
			try:
				os.makedirs(self.work_dir)
			except:
				pass
		
		# OPTIONS
		temp_path = ppath(); temp_path.filename = filename
		if temp_path.filename:
			if temp_path.absolute_to_file != os.getcwd():
				self.work_dir = temp_path.absolute_to_file
			self._path.filename = filename
			self.read(filename)
		else:
			return

	def run(self,input='', exe=pdflt().pflotran_path):
		'''Run a pflotran simulation for a given input file.
	
		:param input: Name of input file. 
		:type input: str
		:param exe: Path to PFLOTRAN executable.
		:type exe: str
		'''
		
		# set up and check path to executable
		exe_path = ppath()
		exe_path.filename = exe
		
		if not os.path.isfile(exe_path.full_path): # if can't find the executable, halt
			print('ERROR: Default location is' +exe + '. No executable at location '+exe)
			return
		
		# option to write input file to new name
		if input: self._path.filename = input
		
		# ASSEMBLE FILES IN CORRECT DIRECTORIES
		if self.work_dir: wd = self.work_dir + os.sep
		else: wd = os.getcwd() + os.sep
#		print wd # testing?
#		print self._path.filename # testing?
		returnFlag = self.write(wd+self._path.filename) # ALWAYS write input file
#		print returnFlag # testing?
		if returnFlag: 
			print('ERROR: writing files')
			return
		
	
		# RUN SIMULATION
		cwd = os.getcwd()
		if self.work_dir: os.chdir(self.work_dir)
		subprocess.call(exe_path.full_path + ' -pflotranin ' + self._path.filename,shell=True)
		
		# After executing simulation, go back to the parent directory
		if self.work_dir: os.chdir(cwd)
		
	def __repr__(self): return self.filename 	# print to screen when called
	
	def read(self, filename):
		if not os.path.isfile(filename): print filename + ' not found...'
		self._filename = filename 	# assign filename attribute
		read_fn = dict(zip(cards, 	
				[self._read_co2_database,
				 self._read_uniform_velocity,
				 self._read_mode,
				 self._read_checkpoint,
				 self._read_restart,
				 self._read_chemistry,
				 self._read_grid,
				 self._read_timestepper,
				 self._read_prop,
				 self._read_time,
				 self._read_lsolver,
				 self._read_nsolver,
				 self._read_output,
				 self._read_fluid,
				 self._read_saturation,
				 self._read_region,
				 self._read_observation,
				 self._read_flow,
				 self._read_transport,
				 self._read_initial_condition,
				 self._read_boundary_condition,
				 self._read_source_sink,
				 self._read_strata,
				 self._read_constraint]
				 
				 ))  # associate each card name with a read function, defined further below
		with open(self._filename,'r') as infile:
			keepReading = True
			while keepReading:
				line = infile.readline()
				if not line: keepReading = False
				if len(line.strip())==0: continue
				card = line.split()[0].lower() 		# make card lower case
				if card in cards: 			# check if a valid cardname
					if card in ['co2_database','checkpoint','restart','material_property',
					 'mode','grid',
					 'timestepper','linear_solver','newton_solver',
					 'saturation_function','region','flow_condition',
					 'boundary_condition','transport_condition','constraint',
					 'uniform_velocity']:
						
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
		
		# Presumes uniform_velocity.value_list is required
		if self.uniform_velocity.value_list: self._write_uniform_velocity(outfile)
		
		# Presumes mode.name is required
		if self.mode.name: self._write_mode(outfile)
		
		if self.co2_database: self._write_co2_database(outfile)
		
		if self.checkpoint.frequency: self._write_checkpoint(outfile)
		else: print 'info: checkpoint not detected\n'
		
		if self.restart.file_name: self._write_restart(outfile)
		else: print 'info: restart not detected\n'

		if self.chemistry: self._write_chemistry(outfile)
		else: print 'info: chemistry not detected\n'
		
		if self.grid: self._write_grid(outfile)
		else: print 'ERROR: grid is required, it is currently reading as empty\n'
		
		if self.timestepper : self._write_timestepper(outfile)
		else: print 'info: timestepper not detected\n'
		
		if self.time: self._write_time(outfile)
		else: print 'ERROR: time is required, it is currently reading as empty\n'
		
		if self.proplist: self._write_prop(outfile)
		else: print 'ERROR: proplist is required, it is currently reading as empty\n'
		
		if self.lsolverlist: self._write_lsolver(outfile)
		else: print 'info: lsolverlist (linear solver list) not detected\n'
		
		if self.nsolverlist: self._write_nsolver(outfile)
		else: print 'info: nsolverlist (newton solver list) not detected\n'
		
		if self.output: self._write_output(outfile)
		else: print 'ERROR: output is required, it is currently reading as empty\n'
		
		if self.fluid: self._write_fluid(outfile)
		else: print 'ERROR: fluid is required, it is currently reading as empty\n'
		
		if self.saturation: self._write_saturation(outfile)
		else: print 'info: saturation not detected\n'
		
		if self.regionlist: self._write_region(outfile)
		else: print 'ERROR: regionlist is required, it is currently reading as empty\n'
		
		if self.observation: self._write_observation(outfile)
		
		if self.flowlist: self._write_flow(outfile)
		else: print 'info: flowlist not detected\n'
		
		if self.transportlist: self._write_transport(outfile)
		
		if self.initial_condition: self._write_initial_condition(outfile)
		else: print 'ERROR: initial_condition is required, it is currently reading as empty\n'
		
		if self.boundary_condition_list: self._write_boundary_condition(outfile)
		else: print 'ERROR: boundary_condition_list is required, it is currently reading as empty\n'
		
		if self.source_sink: self._write_source_sink(outfile)
		else: print 'info: source_sink not detected\n'

		if self.strata_list: self._write_strata(outfile)
		else: print 'info: (stratigraphy_coupler) strata is required, it is currently reading as empty\n'
		
		if self.constraint_list: self._write_constraint(outfile)
		outfile.close()
        
	def add(self,obj,index='',overwrite=False):	#Adds a new object to the file
		'''Attach an object associated w/ list (e.g. region) to the data file.
		
		:param obj: Object to be added to the data file.
		:type obj: object(eg. pregion)
		:param index: (Optional) Used to find an object that is using a string as an index in a dictionary. Intended for the super class object. (E.g. Index represents flow.name if instance is pflow_variable.) Default if not specified is to use the last super-class object added to pdata.
		:type index: String
		:param overwrite: Flag to overwrite macro if already exists for a particular zone.
		:type overwrite: bool
		'''
		
		# Always make index lower case if it's being used as a string
		if isinstance(index,str): index=index.lower()
		if isinstance(obj,pmaterial): self._add_prop(obj,overwrite)
		if isinstance(obj,pchemistry_m_kinetic): 
			self._add_chemistry_m_kinetic(obj,overwrite)
		if isinstance(obj,plsolver): self._add_lsolver(obj,overwrite)
		if isinstance(obj,pnsolver): self._add_nsolver(obj,overwrite)
		if isinstance(obj,pregion): self._add_region(obj,overwrite)
		if isinstance(obj,pflow): self._add_flow(obj,overwrite)
		if isinstance(obj, pflow_variable): 
			self._add_flow_variable(obj,index,overwrite)
		if isinstance(obj,pboundary_condition):
			self._add_boundary_condition(obj,overwrite)
		if isinstance(obj,pstrata): self._add_strata(obj,overwrite)
		if isinstance(obj,ptransport): self._add_transport(obj,overwrite)
		if isinstance(obj,pconstraint): self._add_constraint(obj,overwrite)
		if isinstance(obj,pconstraint_concentration): 
			self._add_constraint_concentration(obj,index,overwrite)
		
	def delete(self,obj,super_obj=None):	#Deletes an object from the file
		'''Delete an object that is assigned to a list e.g.(region) from the data file.
		
		:param obj: Object to be deleted from the data file. Can be a list of objects.
		:type obj: object(eg. pregion), list
		'''

		if isinstance(obj,pmaterial): self._delete_prop(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):	# obji = object index
				if isinstance(obji,pmaterial): self._delete_prop(obji)
				
		if isinstance(obj,plsolver): self._delete_lsolver(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):
				if isinstance(obji,plsolver): self._delete_lsolver(obji)
		
		if isinstance(obj,pnsolver): self._delete_nsolver(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):
				if isinstance(obji,pnsolver): self._delete_nsolver(obji)
		
		if isinstance(obj,pregion): self._delete_region(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):
				if isinstance(obji,pregion): self._delete_region(obji)
				
		if isinstance(obj,pflow): self._delete_flow(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):
				if isinstance(obji,pflow): self._delete_flow(obji)
		
		if isinstance(obj,pflow_variable): # Flow object needs to be specified
			self._delete_flow_variable(obj, super_obj)
		elif isinstance(obj,list): # Condition not tested
			for obji in copy(obj):
				if isinstance(obji,pflow_variable): self._delete_flow_variable(obji)
				
		if isinstance(obj,pboundary_condition): self._delete_boundary_condition(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):
				if isinstance(obji,pboundary_condition): self._delete_boundary_condition(obji)
				
		if isinstance(obj,pstrata): self._delete_strata(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):
				if isinstance(obji,pstrata): self._delete_strata(obji)
				
		if isinstance(obj,ptransport): self._delete_transport(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):
				if isinstance(obji,ptransport): self._delete_transport(obji)
				
		if isinstance(obj,pconstraint): self._delete_constraint(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):
				if isinstance(obji,pconstraint): self._delete_constraint(obji)
				
		if isinstance(obj,pconstraint_concentration): # Constraint object needs to be specified
			self._delete_constraint_concentration(obj, super_obj)
		elif isinstance(obj,list): # Condition not tested
			for obji in copy(obj):
				if isinstance(obji,pconstraint_concentration): self._delete_constraint_concentration(obji)
		
	def _read_uniform_velocity(self,infile,line):
		np_value_list = []
		tstring = line.split()[1:]	# Convert to list, ignore 1st word
		
		i=0	# index/count
		while i < len(tstring):
			try :
				np_value_list.append(floatD(tstring[i]))
			except ValueError, e:
				np_value_list.append(tstring[i])
			i += 1
		
		new_uniform_velocity = puniform_velocity(np_value_list)
		self._uniform_velocity = new_uniform_velocity
		
	def _write_uniform_velocity(self,outfile):
		self._header(outfile,headers['uniform_velocity'])
		outfile.write('UNIFORM_VELOCITY ')
		for v in self.uniform_velocity.value_list:	# value in value_list
			outfile.write(strD(v) + ' ')
		outfile.write('\n\n')
		
	def _read_mode(self,infile,line):
		mode_name = line.split()[-1]
		new_mode = pmode(mode_name)
		self._mode = new_mode
	
	def _write_mode(self,outfile):
		self._header(outfile,headers['mode'])
		mode = self.mode
		if mode.name.lower() in mode_names_allowed:
			outfile.write('MODE ')
			outfile.write(mode.name.lower()+'\n\n')
		else:
			print 'ERROR: mode.name: \''+ mode.name +'\' is invalid.'
			print '       valid mode.names:', mode_names_allowed, '\n'
			
	def _read_co2_database(self,infile,line):
		self._co2_database = del_extra_slash(line.split()[-1])
	
	def _write_co2_database(self,outfile):
		self._header(outfile,headers['co2_database'])
		outfile.write('CO2_DATABASE ' + self._co2_database + '\n\n')
		
	def _read_grid(self,infile,line):
		grid = pgrid()				# assign defaults before reading in values

		keepReading = True
		bounds_key = False
		gravity_key = False
		while keepReading:
			line = infile.readline() 			# get next line
			key = line.strip().split()[0].lower() 		# take first keyword
			if key in ['#']:
				pass
			if key == 'type':
				grid.type = line.split()[-1]
			elif key == 'bounds':
				keepReading2 = True
				while keepReading2:
					line1 = infile.readline()
					grid.lower_bounds[0] = floatD(line1.split()[0])
					grid.lower_bounds[1] = floatD(line1.split()[1])
					grid.lower_bounds[2] = floatD(line1.split()[2])
					line2 = infile.readline()
					grid.upper_bounds[0] = floatD(line2.split()[0])
					grid.upper_bounds[1] = floatD(line2.split()[1])
					grid.upper_bounds[2] = floatD(line2.split()[2])
					line3 = infile.readline()
					if line3.strip().split()[0].lower() in ['/','end']: keepReading2 = False
			elif key == 'origin':
				grid.origin[0] = floatD(line.strip().split()[1])
				grid.origin[1] = floatD(line.strip().split()[2])
				grid.origin[2] = floatD(line.strip().split()[3])
			elif key == 'nxyz':
				grid.nxyz[0] = int(line.split()[1])
				grid.nxyz[1] = int(line.split()[2])
				grid.nxyz[2] = int(line.split()[3])
			elif key == 'gravity':
				grid.gravity[0] = floatD(line.split()[1])
				grid.gravity[1] = floatD(line.split()[2])
				grid.gravity[2] = floatD(line.split()[3])
			elif key == 'filename':
				if grid.type != 'unstructured': print 'Error - filename not need with structure grid'; return
				grid.filename = line.split()[-1]
			elif key == 'dxyz':
				if bounds_key: print 'Error - specify either bounds of dxyz'; return
				keepReading2 = True
				count = 0
				while keepReading2:
					line = infile.readline()
					if line.strip().split()[0].lower() in ['/','end']:
						keepReading2 = False
					else:
						grid.dxyz[count] = floatD(line.strip().split()[0])
						count = count + 1
			elif key in ['/','end']: keepReading = False
		self._grid = grid
	
	def _write_grid(self,outfile):
		self._header(outfile,headers['grid'])
		grid = self.grid
		outfile.write('GRID\n')
		if grid.type in grid_types_allowed:
			outfile.write('  TYPE ' + grid.type + '\n')
		else:
			print 'ERROR: grid.type: \''+ grid.type +'\' is invalid.'
			print '       valid grid.types:', grid_types_allowed, '\n'
		if grid.lower_bounds:
			outfile.write('  BOUNDS\n')
			outfile.write('    ')
			for i in range(3):
				outfile.write(strD(grid.lower_bounds[i]) + ' ')
			outfile.write('\n    ')
			for i in range(3):
				outfile.write(strD(grid.upper_bounds[i]) + ' ')
			outfile.write('\n  /\n') # / marks end of writing out bounds
		else:	# DXYZ is only written if no bounds are provided
			outfile.write('  DXYZ\n')
			for i in range(3):
				outfile.write('    ' + strD(grid.dxyz[i]) + '\n')
			outfile.write('  /\n')
		outfile.write('  ORIGIN' + ' ')
		for i in range(3):
			outfile.write(strD(grid.origin[i]) + ' ')
		outfile.write('\n')
		outfile.write('  NXYZ' + ' ')
		for i in range(3):
			outfile.write(strD(grid.nxyz[i]) + ' ')
		outfile.write('\n')
		if grid.gravity:
			outfile.write('  GRAVITY' + ' ')
			for i in range(3):
				outfile.write(strD(grid.gravity[i]) + ' ')
			outfile.write('\n')
		if grid.type == 'unstructured':
			outfile.write('  FILENAME' + grid.filename + '\n')
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
				np_cfl_limiter = floatD(line.split()[-1])
			elif key == 'initialize_to_steady_state':
				np_initialize_to_steady_state = True 
			elif key == 'run_as_steady_state':
				np_run_as_steady_state = True 
			elif key == 'max_pressure_change':
				np_max_pressure_change = floatD(line.split()[-1])
			elif key == 'max_temperature_change':
				np_max_temperature_change = floatD(line.split()[-1]) 
			elif key == 'max_concentration_change':
				np_max_concentration_change = floatD(line.split()[-1])
 			elif key == 'max_saturation_change':
				np_max_saturation_change = floatD(line.split()[-1])
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
			outfile.write('  ' + 'TS_ACCELERATION ' + 
                                str(self.timestepper.ts_acceleration) + '\n')
		if self.timestepper.num_steps_after_cut:
			outfile.write('  ' + 'NUM_STEPS_AFTER_CUT ' + 
                                str(self.timestepper.num_steps_after_cut) + '\n')
		if self.timestepper.max_ts_cuts:
			outfile.write('  ' + 'MAX_TS_CUTS ' + str(self.timestepper.max_ts_cuts) + '\n')
		if self.timestepper.max_steps:
			outfile.write('  ' + 'MAX_STEPS ' + str(self.timestepper.max_steps) + '\n')
		if self.timestepper.cfl_limiter:
			outfile.write('  ' + 'CFL_LIMITER ' + strD(self.timestepper.cfl_limiter) + '\n')
		if self.timestepper.initialize_to_steady_state:
			outfile.write('  ' + 'INITIALIZE_TO_STEADY_STATE ' + '\n')
		if self.timestepper.run_as_steady_state:
			outfile.write('  ' + 'RUN_AS_STEADY_STATE ' + '\n')
		if self.timestepper.max_pressure_change:
			outfile.write('  ' + 'MAX_PRESSURE_CHANGE' + 
                                strD(self.timestepper.max_pressure_change) + '\n')
		if self.timestepper.max_temperature_change:
			outfile.write('  ' + 'MAX_TEMPERATURE_CHANGE' +
                                strD(self.timestepper.max_temperature_change) + '\n')
		if self.timestepper.max_concentration_change:
			outfile.write('  ' + 'MAX_CONCENTRATION_CHANGE' +
                                strD(self.timestepper.max_concentration_change) + '\n')
		if self.timestepper.max_saturation_change:
                  outfile.write('  ' + 'MAX_SATURATION_CHANGE' + 
                                strD(self.timestepper.max_saturation_change) + '\n')
		outfile.write('END\n\n')

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
						np_permeability.append(floatD(line.split()[-1]))
					elif key == 'perm_y':
						np_permeability.append(floatD(line.split()[-1]))
					elif key == 'perm_z':
						np_permeability.append(floatD(line.split()[-1]))
					elif key in ['/','end']: keepReading2 = False
			elif key in ['/','end']: keepReading = False
		new_prop = pmaterial(np_id,np_name,np_porosity,np_tortuosity,np_density,
		                     np_specific_heat,np_cond_dry,np_cond_wet,
				             np_saturation,np_permeability) 		# create an empty material property

		self.add(new_prop)
		
	def _add_prop(self,prop=pmaterial(),overwrite=False):	#Adds a prop object.
		# check if prop already exists
		if isinstance(prop,pmaterial):		
			if prop.id in self.prop.keys():
				if not overwrite:
					warning = 'WARNING: A Material Property with id \''+str(prop.id)+'\' already exists. Prop will not be defined, use overwrite = True in add() to overwrite the old prop.'
					print warning; print
					_buildWarnings(warning)
					return
				else: # Executes if overwrite = True
					self.delete(self.prop[prop.id])
					
		if prop not in self._proplist:
			self._proplist.append(prop)
			
	def _delete_prop(self,prop=pmaterial()):
		self._proplist.remove(prop)
	
	def _write_prop(self,outfile):
		self._header(outfile,headers['material_property'])
		for prop in self.proplist:
			if prop.name:
				outfile.write('MATERIAL_PROPERTY ' + prop.name + '\n')
			if prop.id:
				outfile.write('  ID '+str(prop.id)+'\n')
			if prop.porosity:
				outfile.write('  POROSITY '+strD(prop.porosity)+'\n')
			if prop.tortuosity:
				outfile.write('  TORTUOSITY '+strD(prop.tortuosity)+'\n')
			if prop.density:
				outfile.write('  ROCK_DENSITY '+strD(prop.density)+'\n')
			if prop.specific_heat:
				outfile.write('  SPECIFIC_HEAT '+strD(prop.specific_heat)+'\n')
			if prop.cond_dry:
				outfile.write('  THERMAL_CONDUCTIVITY_DRY '+strD(prop.cond_dry)+'\n')
			if prop.cond_wet:
				outfile.write('  THERMAL_CONDUCTIVITY_WET '+strD(prop.cond_wet)+'\n')
			if prop.saturation:
				outfile.write('  SATURATION_FUNCTION '+prop.saturation+'\n')
			if prop.permeability:
				outfile.write('  PERMEABILITY\n')
				outfile.write('    PERM_X '+strD(prop.permeability[0])+'\n')
				outfile.write('    PERM_Y '+strD(prop.permeability[1])+'\n')
				outfile.write('    PERM_Z '+strD(prop.permeability[2])+'\n')
				outfile.write('  /\n')
			outfile.write('END\n\n')
	
	def _read_time(self,infile):
		time = ptime()
		time.dtf_list = []
		
		keepReading = True
		while keepReading:
			line = infile.readline() 		# get next line
			key = line.split()[0].lower() 		# take first keyword
			if key == 'final_time':
				tstring = line.split()[1:]	# temp list of strings, 
								# do not include 1st sub-string
				if len(tstring) == 2:	# Do this if there is a time unit to read
					time.tf.append(floatD(tstring[0]))
					time.tf.append(tstring[-1])
				else:			# No time unit being read in
					time.tf.append(floatD(tstring[0]))
			elif key == 'initial_timestep_size':
				tstring = line.split()[1:]
				if len(tstring) == 2:
					time.dti.append(floatD(tstring[0]))
					time.dti.append(tstring[-1])	
				else:
					time.dti.append(floatD(tstring[0]))
			elif key == 'maximum_timestep_size':
				if ('at' not in line):
					tstring = line.split()[1:]
					if len(tstring) == 2:
						time.dtf.append(floatD(tstring[0]))
						time.dtf.append(tstring[-1])
					else:
						time.dtf.append(floatD(tstring[0]))
				elif ('at' in line):
					## Read maximum_timestep_size with AT keyword 
					if (key == 'maximum_timestep_size'):
						
						# temporary variable
						dtf_more = []
						
						#Read before AT
						tstring = line.split()[1:]
						if len(tstring) >= 2:
							# assign 1st value
							dtf_more.append(floatD(tstring[0]))
							
							# assign 1st unit
							dtf_more.append(tstring[1])
							
						#Read after AT
						at_i = tstring.index('at') # Find index # in list (Not string)
						tstring = line.split()[at_i+2:] # Use string only after 'at'
						
						if len(tstring) == 2:
							
							# assign 2nd value (increment)
							dtf_more.append(floatD(tstring[0]))
							
							# assign 2nd unit (increment)
							dtf_more.append(tstring[1])
							
						time.dtf_list.append(dtf_more)
							
			elif key in ['/','end']: keepReading = False
			
		self._time = time

	def _write_time(self,outfile):
		self._header(outfile,headers['time'])
		time = self.time
		outfile.write('TIME\n')
		
		# write FINAL_TIME statement (tf)
		if time.tf:
			try:
				outfile.write('  FINAL_TIME ' + strD(time.tf[0])) # Write value
				if time.tf[1].lower() in time_units_allowed:
					outfile.write(' ' + time.tf[1].lower() +'\n')# Write time unit
				else:
					print 'ERROR: time.tf[1]: \'' + time.tf[1] + '\' is invalid.'
					print '       valid time.units', time_units_allowed, '\n'
			except:
				print 'ERROR: time.tf (final time) input is invalid. Format should be a list: [number, string]\n'
		
		# write INITIAL_TIMESTEP_SIZE statement (dti)
		if time.dti:
			try:
				outfile.write('  INITIAL_TIMESTEP_SIZE ' + 
                                        strD(time.dti[0]))		# Write value
				if time.dti[1].lower() in time_units_allowed:
					outfile.write(' ' + time.dti[1] +'\n')	# Write time unit
				else:
					print 'ERROR: time.dti[1]: \'' + time.dti[1] + '\' is invalid.'
					print '       valid time.units', time_units_allowed, '\n'
			except:
				print 'ERROR: time.dti (initial timestep size) input is invalid. Format should be a list: [number, string]\n'
		
		# write MAXIMUM_TIMESTEP_SIZE statement	dtf
		if time.dtf:
			try:
				outfile.write('  MAXIMUM_TIMESTEP_SIZE ' + strD(time.dtf[0]))
				if time.dtf[1].lower() in time_units_allowed:
					outfile.write(' ' + time.dtf[1] +'\n')
				else:
					print 'ERROR: time.dtf[1]: \'' + time.dtf[1] + '\' is invalid.'
					print '       valid time.units', time_units_allowed, '\n'
			except:
				print 'ERROR: time.dtf (maximum timestep size) input is invalid. Format should be a list: [number, string]\n'
				
		# Write more MAXIMUM_TIME_STEP_SIZE statements if applicable
		for dtf in time.dtf_list:
			outfile.write('  MAXIMUM_TIMESTEP_SIZE ')
			
			try:
				# Write 1st value before 'at'
				if isinstance(dtf[0], float):
					outfile.write(strD(dtf[0]) + ' ')
				else:
					print 'ERROR: The 1st variable in a dtf_list is not recognized as a float.\n'
					
				# Write 1st time unit before 'at'
				if isinstance(dtf[1], str):
					outfile.write((dtf[1]) + ' ')
				else:
					print 'ERROR: The 2nd variable in a dtf_list is not recognized as a str (string).\n'
					
				outfile.write('at ')
					
				# Write 2nd value after 'at'
				if isinstance(dtf[2], float):
					outfile.write(strD(dtf[2]) + ' ')
				else:
					print 'ERROR: The 3rd variable in a dtf_list is not recognized as a float.\n'
					
				# Write 2nd time unit after 'at'
				if isinstance(dtf[3], str):
					outfile.write((dtf[3]))
				else:
					print 'ERROR: The 4th variable in a dtf_list is not recognized as a str (string).\n'
			except:
				print 'ERROR: time.dtf_list (maximum timestep size with \'at\') is invalid. Format should be a list: [float, str, float, str]\n'
			outfile.write('\n')
		'''
		# Determine dtf_i size, the length of the smallest sized list being used
		# with MAXIMUM_TIMESTEP_SIZE with key word 'at'.
		# Displays a warning if the lists are not all of equal length.
		if time.dtf_i == 0:	# Checks to see if user manually specified length so that 
					# it does not re-assign user input
			# Assign minimum value
			time.dtf_i = min(len(time.dtf_lv), len(time.dtf_li), 
					 len(time.dtf_lv_unit), len(time.dtf_li_unit))
			
			# Display warning if lists are not all of equal length
			# This check may not be needed.
			if not all(i == time.dtf_i for i in (len(time.dtf_lv), len(time.dtf_li),
							     len(time.dtf_lv_unit),
							     len(time.dtf_li_unit))):
				print 'WARNING: The lengths of time.dtf_lv, time.dtf_li, time.dtf_lv, and time.dtf_li are not all of equal length.\n\tSome values assigned will be missing.\n'
				
		# Write more MAXIMUM_TIMESTEP_SIZE statements if applicable
		for i in range(0, time.dtf_i):
			try:
				# write before key word 'AT'
				time.dtf_lv_unit[i] = time.dtf_lv_unit[i].lower()# lower capitalization
				outfile.write('  MAXIMUM_TIMESTEP_SIZE ')
				outfile.write(strD(time.dtf_lv[i]) + ' ') # Write Value 
				if time.dtf_lv_unit[i] in time_units_allowed:
					outfile.write(time.dtf_lv_unit[i])# Write Time Unit
				else:
					print 'ERROR: time.dtf_lv_unit: \'' + time.dtf_lv_unit[i] + '\' is invalid.'
					print '       valid time.units', time_units_allowed, '\n'
				
				# write after key word 'AT'
				time.dtf_li_unit[i] = time.dtf_li_unit[i].lower()# lower capitalization
				outfile.write(' at ')
				outfile.write(strD(time.dtf_li[i]) + ' ') # Write Value
				if time.dtf_li_unit[i] in time_units_allowed:
					outfile.write(time.dtf_li_unit[i]) # Write Time Unit
				else:
					print 'ERROR: time.dtf_li_unit: \'' + time.dtf_li_unit[i] + '\' is invalid.'
					print '       valid time.units', time_units_allowed, '\n'
				outfile.write('\n')
			except:
				print 'ERROR: Invalid input at maximum_time_step_size with key word \'at\'. time.dtf_lv and time.dtf_li should be a list of floats. time_dtf_lv_unit and time_dtf_li_unit should be a list of strings. All lists should be of equal length.\n'
		'''
		outfile.write('END\n\n')
		
		
	def _read_lsolver(self,infile,line):
		lsolver = plsolver()	# temporary object while reading
		lsolver.name =  line.split()[-1].lower() # solver type - tran_solver or flow_solver
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first key word
			
			if key == 'solver':
				lsolver.solver = line.split()[-1] # Assign last word
			elif key in ['/','end']: keepReading = False
		
		self.add(lsolver) # Assign object

	def _add_lsolver(self,lsolver=plsolver(),overwrite=False):	#Adds a Linear Solver object.
		# check if lsolver already exists
		if isinstance(lsolver,plsolver):
			if lsolver.name in self.lsolver.keys():
				if not overwrite:
					warning = 'WARNING: A linear solver with name \''+str(lsolver.name)+'\' already exists. lsolver will not be defined, use overwrite = True in add() to overwrite the old lsolver.'
					print warning; warning
					_buildWarnings(warning)
					return
				else:
					self.delete(self.lsolver[lsolver.name])
					
		if lsolver not in self._lsolverlist:
			self._lsolverlist.append(lsolver)
			
	def _delete_lsolver(self,lsolver=plsolver()):
		self._lsolverlist.remove(lsolver)

	def _write_lsolver(self,outfile):
		self._header(outfile,headers['linear_solver'])
		
		for lsolver in self.lsolverlist:
			if lsolver.name.lower() in solver_names_allowed:
				outfile.write('LINEAR_SOLVER ' + lsolver.name.lower() + '\n')
			else:
				print 'ERROR: lsolver.name: \''+ lsolver.name +'\' is invalid.'
				print '       valid solver.names', solver_names_allowed, '\n'
			if lsolver.solver:
				outfile.write('  SOLVER ' + lsolver.solver.lower() + '\n')
			outfile.write('END\n\n')
		
	def _read_nsolver(self,infile,line):
		
		nsolver = pnsolver('')		# Assign Defaults
		
		nsolver.name = line.split()[-1].lower() # newton solver type - tran_solver or flow_solver
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first key word
			
			if key == 'atol':
				nsolver.atol = floatD(line.split()[-1])
			if key == 'rtol':
				nsolver.rtol = floatD(line.split()[-1])
			if key == 'stol':
				nsolver.stol = floatD(line.split()[-1])
			if key == 'dtol':
				nsolver.dtol = floatD(line.split()[-1])
			if key == 'itol':
				nsolver.itol = floatD(line.split()[-1])
			if key == 'maxit':
				nsolver.max_it = int(line.split()[-1])
			if key == 'maxf':
				nsolver.max_f = int(line.split()[-1])
			elif key in ['/','end']: keepReading = False
		self.add(nsolver)	# Assign
		
	def _add_nsolver(self,nsolver=pnsolver(),overwrite=False):	#Adds a Newton Solver object.
		# check if nsolver already exists
		if isinstance(nsolver,pnsolver):		
			if nsolver.name in self.nsolver.keys():
				if not overwrite:
					warning = 'WARNING: A newton solver with name \''+str(nsolver.name)+'\' already exists. nsolver will not be defined, use overwrite = True in add() to overwrite the old nsolver.'
					print warning; print
					_buildWarnings(warning)
					return
				else:
					self.delete(self.nsolver[nsolver.name])
					
		if nsolver not in self._nsolverlist:
			self._nsolverlist.append(nsolver)
			
	def _delete_nsolver(self,nsolver=pnsolver()):
		self._nsolverlist.remove(nsolver)
		
	def _write_nsolver(self,outfile):
		self._header(outfile,headers['newton_solver'])
		
		for nsolver in self.nsolverlist:
			# Write Newton Solver Type - Not certain this is correct.
			
#			if nsolver.name.lower() == 'flow' or nsolver.name.lower() == 'transport':	# default
#				outfile.write('NEWTON_SOLVER ' + nsolver.name.lower() + '\n')
#			elif nsolver.name.lower() == 'tran':
#				outfile.write('NEWTON_SOLVER ' + nsolver.name.lower() + '\n')
			if nsolver.name.lower() in solver_names_allowed:
				outfile.write('NEWTON_SOLVER ' + nsolver.name.lower() + '\n')
			else:
				print 'ERROR: nsolver.name: \''+ nsolver.name +'\' is invalid.'
				print '       valid solver.names', solver_names_allowed, '\n'
			
			if nsolver.atol:
				outfile.write('  ATOL ' + strD(nsolver.atol) + '\n')
			if nsolver.rtol:
				outfile.write('  RTOL ' + strD(nsolver.rtol) + '\n')
			if nsolver.stol:
				outfile.write('  STOL ' + strD(nsolver.stol) + '\n')
			if nsolver.dtol:
				outfile.write('  DTOL ' + strD(nsolver.dtol) + '\n')
			if nsolver.itol:
				outfile.write('  ITOL ' + strD(nsolver.itol) + '\n')
			if nsolver.max_it:
				outfile.write('  MAXIT ' + str(nsolver.max_it) + '\n')
			if nsolver.max_f:
				outfile.write('  MAXF ' + str(nsolver.max_f) + '\n')
			outfile.write('END\n\n')
	
	def _read_output(self,infile):
		output = poutput()
		output.time_list = []
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first key word
			
			if key == 'times':
				tstring = line.split()[1:] # Turn into list, exempt 1st word
				i=0
				while i < len(tstring):
					try:
						output.time_list.append(floatD(tstring[i]))
					except:
						output.time_list.append(tstring[i])
					i += 1
			elif key == 'screen':
				tstring = line.strip().split()[1].lower() # Read the 2nd word
				if tstring == 'periodic':
					output.screen_periodic = int(line.split()[-1])
			elif key == 'periodic':
				tstring = line.strip().split()[1].lower() # Read the 2nd word
				if tstring == 'time':
					output.periodic_time.append(floatD(line.split()[-2])) # 2nd from last word.
					output.periodic_time.append(line.split()[-1]) # last word
				elif tstring == 'timestep':
					output.periodic_timestep.append(floatD(line.split()[-2])) # 2nd from last word.
					output.periodic_timestep.append(line.split()[-1]) # last word
			elif key == 'periodic_observation':
				tstring = line.strip().split()[1].lower() # Read the 2nd word
				if tstring == 'time':
					output.periodic_observation_time.append(floatD(line.split()[-2])) # 2nd from last word.
					output.periodic_observation_time.append(line.split()[-1]) # last word
				elif tstring == 'timestep':
					output.periodic_observation_timestep = int(line.split()[-1])
			elif key == 'print_column_ids':
				output.print_column_ids = True
			elif key == 'format':
				tstring = (line.strip().split()[1:]) # Do not include 1st sub-string
				tstring = ' '.join(tstring).lower()	# Convert list into a string seperated by a space
				output.format_list.append(tstring)	# assign
			elif key == 'permeability':
				output.permeability = True
			elif key == 'porosity':
				output.porosity = True
			elif key == 'velocities':
				output.velocities = True
			elif key == 'mass_balance':
				output.mass_balance = True
			elif key in ['/','end']: keepReading = False
			
		self._output = output
		
	def _write_output(self,outfile):
		self._header(outfile,headers['output'])
		output = self.output
		
		# Write Output - if used so null/None entries are not written
		outfile.write('OUTPUT\n')
		
		if output.time_list:
			# Check if 1st variable in list a valid time unit
			if output.time_list[0].lower() in time_units_allowed:
				outfile.write('  TIMES ')
				# Write remaining number(s) after time unit is specified
				for value in output.time_list:
						outfile.write(' '+strD(value).lower())
			else:
				print 'ERROR: output.time_list[0]: \''+ output.time_list[0] +'\' is invalid.'
				print '       valid time.units', time_units_allowed, '\n'
			outfile.write('\n')
					
# This is here on purpose - Needed later
		#if output.periodic_observation_time:
			#outfile.write('  PERIODIC_OBSERVATION TIME  '+
					#str(output.periodic_observation_time)+'\n')
		if output.screen_periodic:
			try: # Error checking to ensure screen_periodic is int (integer).
				output.screen_periodic = int(output.screen_periodic)
				outfile.write('  '+'SCREEN PERIODIC '+str(output.screen_periodic)+'\n')
			except(ValueError):
				print 'ERROR: output.screen_periodic: \''+output.screen_periodic+'\' is not int (integer).\n'
		if output.periodic_time:
			try: # Error checking to ensure periodic_time is [float, str].
				output.periodic_time[0] = floatD(output.periodic_time[0])
				if output.periodic_time[1].lower() in time_units_allowed:
					output.periodic_time[1] = str(output.periodic_time[1].lower())
				else:
					output.periodic_time[1] = str(output.periodic_time[1].lower())
					print 'ERROR: time unit in output.periodic_time[1] is invalid. Valid time units are:', time_units_allowed, '\n'
				outfile.write('  '+'PERIODIC TIME ')
				outfile.write(strD(output.periodic_time[0])+' ')
				outfile.write(output.periodic_time[1]+'\n')
			except:
				print 'ERROR: output.periodic_time: \''+str(output.periodic_time)+'\' is not [float, str].\n'
		if output.periodic_timestep:
			try: # Error checking to ensure periodic_timestep is [float, str].
				output.periodic_timestep[0] = floatD(output.periodic_timestep[0])
				if output.periodic_timestep[1].lower() in time_units_allowed:
					output.periodic_timestep[1] = str(output.periodic_timestep[1].lower())
				else:
					output.periodic_timestep[1] = str(output.periodic_timestep[1].lower())
					print 'ERROR: time unit in output.periodic_timestep[1] is invalid. Valid time units are:', time_units_allowed, '\n'
				outfile.write('  '+'PERIODIC TIMESTEP ')
				outfile.write(strD(output.periodic_timestep[0])+' ')
				outfile.write(output.periodic_timestep[1]+'\n')
			except:
				print 'ERROR: output.periodic_timestep: \''+str(output.periodic_timestep)+'\' is not [float, str].\n'
		if output.periodic_observation_time:
			try: 
				# Error checking to ensure periodic_observation_time is [float, str].
				output.periodic_observation_time[0] = floatD(output.periodic_observation_time[0])
				if output.periodic_observation_time[1].lower() in time_units_allowed:
					output.periodic_observation_time[1] = str(output.periodic_observation_time[1].lower())
				else:
					output.periodic_observation_time[1] = str(output.periodic_observation_time[1].lower())
					print 'ERROR: time unit in output.periodic_observation_time[1] is invalid. Valid time units are:', time_units_allowed, '\n'
					
				# Writing out results
				outfile.write('  '+'PERIODIC_OBSERVATION TIME ')
				outfile.write(strD(output.periodic_observation_time[0])+' ')
				outfile.write(output.periodic_observation_time[1]+'\n')
			except:
				print 'ERROR: output.periodic_observation_time: \''+str(output.periodic_observation_time)+'\' is not [float, str].\n'
		if output.periodic_observation_timestep:
			outfile.write('  PERIODIC_OBSERVATION TIMESTEP '+
					str(output.periodic_observation_timestep)+'\n')
		if output.print_column_ids:
			outfile.write('  '+'PRINT_COLUMN_IDS'+'\n')
		for format in output.format_list:
			if format.upper() in output_formats_allowed:
				outfile.write('  FORMAT ')
				outfile.write(format.upper() + '\n')
			else:
				print 'ERROR: output.format: \''+ format +'\' is invalid.'
				print '       valid output.format:', output_formats_allowed, '\n'
		if output.permeability:
			outfile.write('  '+'PERMEABILITY'+'\n')
		if output.porosity:
			outfile.write('  '+'POROSITY'+'\n')
		if output.velocities:
			outfile.write('  '+'VELOCITIES'+'\n')
		if output.mass_balance:
			outfile.write('  '+'MASS_BALANCE'+'\n')
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
			outfile.write('  DIFFUSION_COEFFICIENT ' + 
					strD(fluid.diffusion_coefficient) + '\n') # Read last entry
		outfile.write('END\n\n')
		
	def _read_saturation(self,infile,line):
		
		saturation = psaturation()	# assign defaults before reading in values
		saturation.name = line.split()[-1].lower() # saturation function name, passed in.
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first  key word
			
			if key == 'permeability_function_type':
				saturation.permeability_function_type = line.split()[-1]
			elif key == 'saturation_function_type':
				saturation.saturation_function_type = line.split()[-1]
			elif key == 'residual_saturation_liquid':
				saturation.residual_saturation_liquid = floatD(line.split()[-1])
			elif key == 'residual_saturation_gas':
				saturation.residual_saturation_gas = floatD(line.split()[-1])
			elif key == 'residual_saturation':	# Alternative to check
				tstring = line.strip().split()[1].lower()	# take 2nd key word
				if tstring == 'liquid_phase':
					saturation.residual_saturation_liquid = floatD(line.split()[-1])
				elif tstring == 'gas_phase':
					saturation.residual_saturation_gas = floatD(line.split()[-1])
				else: # if no 2nd word exists
					saturation.residual_saturation = floatD(line.split()[-1])
			elif key == 'lambda':
				saturation.a_lambda = floatD(line.split()[-1])
			elif key == 'alpha':
				saturation.alpha = floatD(line.split()[-1])
			elif key == 'max_capillary_pressure':
				saturation.max_capillary_pressure = floatD(line.split()[-1])
			elif key == 'betac':
				saturation.betac = floatD(line.split()[-1])
			elif key == 'power':
				saturation.power = floatD(line.split()[-1])
			elif key in ['/','end']: keepReading = False
			
		# Create an empty saturation function and assign the values read in
		self._saturation = saturation
		
	def _write_saturation(self,outfile):
		self._header(outfile,headers['saturation_function'])
		saturation = self.saturation
		
		# Write out saturation properties that exist
		outfile.write('SATURATION_FUNCTION')
		if saturation.name:
			outfile.write('  ' + saturation.name + '\n')
		else:
			outfile.write('n')
		if saturation.permeability_function_type:
			if saturation.permeability_function_type in permeability_function_types_allowed:
				outfile.write('  PERMEABILITY_FUNCTION_TYPE ' +
					saturation.permeability_function_type + '\n')
			else:
				print 'ERROR: saturation.permeability_function_type: \'' + saturation.permeability_function_type +'\' is invalid.'
				print '       valid saturation.permeability_function_types', permeability_function_types_allowed, '\n'
		if saturation.saturation_function_type:
			if saturation.saturation_function_type in saturation_function_types_allowed:
				outfile.write('  SATURATION_FUNCTION_TYPE ' + 
						saturation.saturation_function_type + '\n')
			else:
				print 'ERROR: saturation.saturation_function_type: \'' + saturation.saturation_function_type +'\' is invalid.'
				print '       valid saturation.permeability_function_types', saturation_function_types_allowed, '\n'
		if saturation.residual_saturation or saturation.residual_saturation==0:
			outfile.write('  RESIDUAL_SATURATION ' + 
					strD(saturation.residual_saturation) + '\n')
		if saturation.residual_saturation_liquid or saturation.residual_saturation_liquid ==0:
			outfile.write('  RESIDUAL_SATURATION LIQUID_PHASE ' + 
					strD(saturation.residual_saturation_liquid) + '\n')
		if saturation.residual_saturation_gas or saturation.residual_saturation_gas == 0:
			outfile.write('  RESIDUAL_SATURATION GAS_PHASE ' +
					strD(saturation.residual_saturation_gas) + '\n')
		if saturation.a_lambda:
			outfile.write('  LAMBDA ' + strD(saturation.a_lambda) + '\n')
		if saturation.alpha:
			outfile.write('  ALPHA ' + strD(saturation.alpha) + '\n')
		if saturation.max_capillary_pressure:
			outfile.write('  MAX_CAPILLARY_PRESSURE ' + 
					strD(saturation.max_capillary_pressure) + '\n')
		if saturation.betac:
			outfile.write('  BETAC ' + strD(saturation.betac) + '\n')
		if saturation.power:
			outfile.write('  POWER ' + strD(saturation.power) + '\n')
		outfile.write('END\n\n')
		
	def _read_region(self,infile,line):
		
		region = pregion()
		region.coordinates_lower = [None]*3
		region.coordinates_upper = [None]*3
		
		region.name = line.split()[-1].lower()
		
		keepReading = True
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first keyword
			if key == 'coordinates':
				keepReading2 = True
				while keepReading2:
					line1 = infile.readline()
					region.coordinates_lower[0] = floatD(line1.split()[0])
					region.coordinates_lower[1] = floatD(line1.split()[1])
					region.coordinates_lower[2] = floatD(line1.split()[2])
					line2 = infile.readline()
					region.coordinates_upper[0] = floatD(line2.split()[0])
					region.coordinates_upper[1] = floatD(line2.split()[1])
					region.coordinates_upper[2] = floatD(line2.split()[2])
					line3 = infile.readline()
					if line3.strip().split()[0].lower() in ['/','end']: keepReading2 = False	
			elif key == 'face':
				region.face = line.strip().split()[-1].lower()
			elif key in ['/','end']: keepReading = False
			
			self.add(region)

	def _add_region(self,region=pregion(),overwrite=False):		#Adds a Region object.
		# check if region already exists
		if isinstance(region,pregion):
			if region.name in self.region.keys():
				if not overwrite:
					warning = 'WARNING: A region with name \''+str(region.name)+'\' already exists. Region will not be defined, use overwrite = True in add() to overwrite the old region.'
					print warning; print
					_buildWarnings(warning)
					return
				else:
					self.delete(self.region[region.name])
					
		if region not in self._regionlist:
			self._regionlist.append(region)
			
	def _delete_region(self,region=pregion()):
		self._regionlist.remove(region)
		
	def _write_region(self,outfile):
		self._header(outfile,headers['region'])
		
		# Write out all valid region object entries with Region as Key word
		for region in self.regionlist:
			outfile.write('REGION ')
			outfile.write(region.name.lower() + '\n')
			if region.face:
				outfile.write('  FACE ' + region.face.lower() + '\n')
			# no if statement below to ensure 0's are accepted for coordinates
			outfile.write('  COORDINATES\n')
			outfile.write('    ')
			for i in range(3):
				outfile.write(strD(region.coordinates_lower[i]) + ' ')
			outfile.write('\n    ')
			for i in range(3):
				outfile.write(strD(region.coordinates_upper[i]) + ' ')
			outfile.write('\n')
			outfile.write('  END\n')
			outfile.write('END\n\n')
			
	def _read_observation(self,infile):
		p = pobservation()
		np_region = p.region
		 
		keepReading = True
		
		while keepReading:
			line = infile.readline() 			# get next line
			key = line.strip().split()[0].lower() 		# take first keyword
			if key == 'region':
				np_region = line.split()[-1]
			elif key in ['/','end']: keepReading = False
			
		new_observation = pobservation(np_region)
		
		self._observation = new_observation
		
	def _write_observation(self,outfile):
		self._header(outfile,headers['observation'])
		observation = self.observation
		
		outfile.write('OBSERVATION\n')
		if observation.region:
			outfile.write('  REGION '+observation.region.lower()+'\n')
		outfile.write('END\n\n')
				
	def _read_flow(self,infile,line):
		flow = pflow()
		flow.varlist = []
		flow.name = line.split()[-1].lower()	# Flow Condition name passed in.
		
		keepReading = True
		isValid = False # Used so that entries outside flow conditions are ignored
		end_count = 0
		total_end_count = 1
		while keepReading:	# Read through all cards
			
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first keyword
			if key == 'type':
				total_end_count = 2 # Basically ensures that both read ifs for
						    # the varlist will execute 
						    # This # indicates how many time a / or 'end' 
						    # can be read before loop terminates.
				
			elif key == 'rate' or key == 'pressure' or key == 'temperature' or key == 'concentration' or key == 'enthalpy':
				if end_count == 0:
					'''
					Appending and instantiation of new 
					flow_variables occur here. 
					Only two entries are filled, 
					the rest are assigned in the elif 
					code block where end_count == 1
					'''
					var = pflow_variable()
					var.name = key
					var.type = line.strip().split()[-1].lower()
					var.valuelist = []
					var.list = []
					
					isValid = True	# Indicates the entries read here should be written so that entries outside flow conditions are ignored.
					tstring = line.split()[0:] # Convert string into list
					
					flow.varlist.append(var)
								
				elif end_count == 1:			
					count = 0
					
					tstring2name = line.strip().split()[0]	# Assigns the 1st word on a line
					tstring2 = line.split()[1:] 	# Assigns the rest of the line
					# #2 used because this is the 2nd reading of the variables
					
					# Deterine if variable is a list or stand-alone
					if tstring2[0].lower() == 'list': # Executes only if 2nd word on line == 'list'
						
						# for each list in a pflow_variable object, check all
						# pflow_variable objects by name to determine correct assignment
						# before assigning in values from a list
						keepReadingList = True
						while keepReadingList:
							
							line = infile.readline() # get next line
							tstring2 = line.split()[:] # split the whole string/line
							
							var.list = [] # list needs to be reset
							for var in flow.varlist: # var represents a pflow_variable object
							
								if tstring2name.lower() == var.name.lower():
									if line[0] == ':' or line[0] == '#' or line[0] == '/':
										pass	# ignore a commented line
											# line[0] == '/' is a temporary fix
									elif tstring2[0].lower() == 'time_units':
										var.time_unit_type = tstring2[1]
									elif tstring2[0].lower() == 'data_units':
											var.data_unit_type = tstring2[1]
									else:
										tvarlist = pflow_variable_list()
										tvarlist.time_unit_value = floatD(tstring2[0])
										tvarlist.data_unit_value_list = []
										tvarlist.data_unit_value_list.append(floatD(tstring2[1]))
										tvarlist.data_unit_value_list.append(floatD(tstring2[2]))
										var.list.append(tvarlist)
							if line.split()[0] in ['/','end']: keepReadingList = False
					else:
						# for each single variable in a pflow_variable object, check all
						# pflow_variable object by name to determine correct assignment
						for substring in tstring2:	# Checks all values/types on this line
							for var in flow.varlist: # var represents a pflow_variable object
								if tstring2name.lower() == var.name.lower():
									try:
										var.valuelist.append(floatD(substring))
									# If a string (e.g. C for temp.), assign to unit
									except(ValueError):
										var.unit = substring
			elif key == 'iphase':
				flow.iphase = int(line.split()[-1])
			elif key == 'sync_timestep_with_update':
				flow.sync_timestep_with_update = True
					
			# Detect if there is carriage return after '/' or 'end' to end loop
			# Alternative method of count implemented by Satish
			elif key in ['/','end']:
				end_count = end_count + 1
				if end_count == total_end_count:
					keepReading = False
					
		if isValid:
			self.add(flow)
			
	def _add_flow(self,flow=pflow(),overwrite=False):	#Adds a Flow object.
		# check if flow already exists
		if isinstance(flow,pflow):
			if flow.name in self.flow.keys():
				if not overwrite:
					warning = 'WARNING: A flow with name \''+str(flow.name)+'\' already exists. Flow will not be defined, use overwrite = True in add() to overwrite the old flow.'
					print warning; print
					_buildWarnings(warning)
					return
				else: # Executes if overwrite = True
					self.delete(self.flow[flow.name])
					
		if flow not in self._flowlist:
			self._flowlist.append(flow)
			
	def _delete_flow(self,flow=pflow()):
		self._flowlist.remove(flow)
	
	'''
	Automate adding the sub-class flow_variable to a flow object. The flow object 
	can be specified by name. If flow object name is not specified, the function
	will append pflow_variable to the last flow object added to the list.
	Function will provided a warning if a flow_variable.name already exists
	in the flow object it is trying to add it to.
	'''
	def _add_flow_variable(self,flow_variable=pflow_variable(),index='',
			       overwrite=False):	#Adds a flow_variable object.
		
		# check if flow.name was specified
		if index:
			if isinstance(index,str):
				flow = self.flow.get(index) # Assign flow object to existing flow object with string type name/index
				if not flow: # Occurs if index/string is not found in flow object
					print 'WARNING: a flow object with flow.name', index, 'was not found. Current found entries are:', self.flow.keys(), 'pflow_variable was not added.\n'
					return

			elif isinstance(index,pflow):
				flow = index # Assigns if index is the flow object reference
		else: # Set flow to last flow object in list
			flow = self.flowlist[-1]
		
		# check if flow_variable already exists
		if isinstance(flow_variable,pflow_variable):
			if flow_variable.name in self._get_flow_variable(flow).keys():
				if not overwrite:
					warning = 'WARNING: A flow_variable with name \''+str(flow_variable.name)+'\' already exists in flow with name \''+str(flow.name)+'\'. Flow_variable will not be defined, use overwrite = True in add() to overwrite the old flow_variable. Use flow=\'name\' if you want to specify the flow object to add flow_variable to.'
					print warning; print
					_buildWarnings(warning)
					return
				else: # Executes if overwrite = True
					self.delete(self._get_flow_variable(flow)[flow_variable.name],flow)
		
		# Add flow_variable to flow (as a sub-class) if flow_variable does
		# not exist in specified flow object
		if flow_variable not in flow.varlist:
			flow.varlist.append(flow_variable)
			
	def _delete_flow_variable(self,flow_variable=pflow_variable(),flow=pflow()):
		flow.varlist.remove(flow_variable)
		
	def _write_flow(self,outfile):
		self._header(outfile,headers['flow_condition'])
		
		# Function is used to determine which flow_condition type allowed list 
		# to check depending on the flow_condition name specified.
		# Also does the work of writing or error reporting
		def check_condition_type(condition_name, condition_type):
			if condition_name.upper() == 'PRESSURE':
				if condition_type.lower() in pressure_types_allowed: 
					outfile.write(condition_type.lower())
				else:
					print 'ERROR: flow.varlist.type: \'' + condition_type +'\' is invalid.'
					print '       valid flow_condition pressure_types_allowed:', pressure_types_allowed, '\n'	
				return 0 # Break out of function
			elif condition_name.upper() == 'RATE':
				if condition_type.lower() in rate_types_allowed: 
					outfile.write(condition_type.lower())
				else:
					print 'ERROR: flow.varlist.type: \'' + condition_type +'\' is invalid.'
					print '       valid flow_condition rate_types_allowed:', rate_types_allowed, '\n'	
				return 0 # Break out of function
			elif condition_name.upper() == 'FLUX':
				if condition_type.lower() in flux_types_allowed: 
					outfile.write(condition_type.lower())
				else:
					print 'ERROR: flow.varlist.type: \'' + condition_type +'\' is invalid.'
					print '       valid flow_condition flux_types_allowed:', flux_types_allowed, '\n'	
				return 0 # Break out of function
			elif condition_name.upper() == 'TEMPERATURE':
				if condition_type.lower() in temperature_types_allowed: 
					outfile.write(condition_type.lower())
				else:
					print 'ERROR: flow.varlist.type: \'' + condition_type +'\' is invalid.'
					print '       valid flow_condition temperature_types_allowed:', temperature_types_allowed, '\n'	
				return 0 # Break out of function
			elif condition_name.upper() == 'CONCENTRATION':
				if condition_type.lower() in concentration_types_allowed: 
					outfile.write(condition_type.lower())
				else:
					print 'ERROR: flow.varlist.type: \'' + condition_type +'\' is invalid.'
					print '       valid flow_condition concentration_types_allowed:', concentration_types_allowed, '\n'	
				return 0 # Break out of function
			elif condition_name.upper() == 'SATURATION':
				if condition_type.lower() in saturation_types_allowed: 
					outfile.write(condition_type.lower())
				else:
					print 'ERROR: flow.varlist.type: \'' + condition_type +'\' is invalid.'
					print '       valid flow_condition saturation_types_allowed:', saturation_types_allowed, '\n'	
				return 0 # Break out of function
			elif condition_name.upper() == 'ENTHALPY':
				if condition_type.lower() in enthalpy_types_allowed: 
					outfile.write(condition_type.lower())
				else:
					print 'ERROR: flow.varlist.type: \'' + condition_type +'\' is invalid.'
					print '       valid flow_condition enthalpy_types_allowed:', enthalpy_types_allowed, '\n'	
				return 0 # Break out of function
			else:
				pass # Error reporting for flow_condition.name is done elsewhere
				# name should be validated before this function is called.
			
		# Write out all valid flow_conditions objects with FLOW_CONDITION as keyword
		for flow in self.flowlist:
			outfile.write('FLOW_CONDITION  ' + flow.name.lower() + '\n')
			if flow.sync_timestep_with_update:
				outfile.write('  SYNC_TIMESTEP_WITH_UPDATE\n')
			outfile.write('  TYPE\n')
			
			# variable name and type from lists go here
			i = 0
			while i< len(flow.varlist):
				if flow.varlist[i].name.upper() in flow_condition_type_names_allowed:
					outfile.write('    ' + flow.varlist[i].name.upper() + '  ')
				else:
					print 'ERROR: flow.varlist.name: \'' + flow.varlist[i].name +'\' is invalid.'
					print '       valid flow_condition.names:', flow_condition_type_names_allowed, '\n'
				
				# Checks flow.varlist[i].type and performs write or error reporting
				check_condition_type(flow.varlist[i].name, flow.varlist[i].type)
				
				outfile.write('\n')
				i += 1
				
			outfile.write('  END\n')
			if flow.iphase:
				outfile.write('  IPHASE '+str(flow.iphase)+'\n')
				
			# variable name and values from lists along with units go here
			i = 0
			while i< len(flow.varlist):
				# Write if using non-list format (Single line)
				if flow.varlist[i].valuelist:	
					outfile.write('    ' + flow.varlist[i].name.upper())
					j = 0
					while j < len(flow.varlist[i].valuelist):
						try:
							outfile.write(' ' + strD(flow.varlist[i].valuelist[j]))
						except:
							print 'error: writing flow.varlist should only contain floats, not strings'
						j += 1
					# Write out possible unit here
					if flow.varlist[i].unit:
						outfile.write(' ' + flow.varlist[i].unit.upper())
					outfile.write('\n')
				# Write if using list format (multiple lines)
				elif flow.varlist[i].list:	
					outfile.write('    ' + flow.varlist[i].name.upper() + ' LIST' + '\n')
					if flow.varlist[i].time_unit_type:
						outfile.write('      TIME_UNITS ' + flow.varlist[i].time_unit_type + '\n')
					if flow.varlist[i].data_unit_type:
						outfile.write('      DATA_UNITS ' + flow.varlist[i].data_unit_type + '\n')
					for k in flow.varlist[i].list:
						outfile.write('        ' + strD(k.time_unit_value))
						for p in range(len(k.data_unit_value_list)):
							outfile.write('  ' + strD(k.data_unit_value_list[p]))
						outfile.write('\n')
					outfile.write('    /\n')
				i += 1
			outfile.write('  END\n\n')
			
	def _read_initial_condition(self,infile):
		p = pinitial_condition()
		np_flow = p.flow
		np_transport = p.transport
		np_region = p.region
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first  key word
			
			if key == 'flow_condition':
				np_flow = line.split()[-1]
			elif key == 'transport_condition':
				np_transport = line.split()[-1]
			elif key == 'region':
				np_region = line.split()[-1]
			elif key in ['/','end']: keepReading = False
			
		# Create an empty initial condition and assign the values read in
		new_initial_condition = pinitial_condition(np_flow,np_transport,np_region)
		self._initial_condition = new_initial_condition
		
	def _write_initial_condition(self,outfile):
		self._header(outfile,headers['initial_condition'])
		initial_condition = self.initial_condition
		outfile.write('INITIAL_CONDITION\n')
		
		# Write out initial_condition variables
		if initial_condition.flow:
			outfile.write('  FLOW_CONDITION ' + initial_condition.flow.lower() + '\n')
			
		if initial_condition.transport:
			outfile.write('  TRANSPORT_CONDITION  '+initial_condition.transport.lower()+'\n')
			
		if initial_condition.region:
			outfile.write('  REGION ' + initial_condition.region.lower() + '\n')
		else:
			print 'ERROR: initial_condition.region is required\n'
		
		outfile.write('END\n\n')
		
	def _read_boundary_condition(self,infile,line):
		if len(line.split()) > 1: 
			np_name = line.split()[-1].lower()	# Flow Condition name passed in.
		else:
			np_name = None
		p = pboundary_condition('')
		np_flow = p.flow
		np_transport = p.transport
		np_region = p.region
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.split()[0].lower()	# take first key word
			
			if key == 'flow_condition':
				np_flow = line.split()[-1]	# take last word
			elif key == 'transport_condition':
				np_transport = line.split()[-1]
			elif key == 'region':
				np_region = line.split()[-1]
			elif key in ['/','end']: keepReading = False
		
		# Create an empty boundary condition and assign the values read in
		new_boundary_condition = pboundary_condition(np_name,np_flow,np_transport,np_region)
		self.add(new_boundary_condition)
		
	def _add_boundary_condition(self,boundary_condition=pboundary_condition(),overwrite=False):			#Adds a boundary_condition object.
		# check if flow already exists
		if isinstance(boundary_condition,pboundary_condition):
			if boundary_condition.region in self.boundary_condition.keys():
				if not overwrite:
					warning = 'WARNING: A boundary_condition with region \''+str(boundary_condition.region)+'\' already exists. boundary_condition will not be defined, use overwrite = True in add() to overwrite the old boundary_condition.'
					print warning; print
					_buildWarnings(warning)
					return
				else:	
					self.delete(self.boundary_condition[boundary_condition.region])
					
		if boundary_condition not in self._boundary_condition_list:
			self._boundary_condition_list.append(boundary_condition)
			
	def _delete_boundary_condition(self,boundary_condition=pboundary_condition()):
		self._boundary_condition_list.remove(boundary_condition)
		
	def _write_boundary_condition(self,outfile):
		self._header(outfile,headers['boundary_condition'])

		# Write all boundary conditions to file
		try:
			for b in self.boundary_condition_list:	# b = boundary_condition
				if b.name:
					outfile.write('BOUNDARY_CONDITION ' + b.name.lower() + '\n')
				else:
					outfile.write('BOUNDARY_CONDITION\n')
				if b.flow:
					outfile.write('  FLOW_CONDITION ' + b.flow.lower() + '\n')
				if b.transport:
					outfile.write('  TRANSPORT_CONDITION '+b.transport.lower()+'\n')
				if b.region:
					outfile.write('  REGION ' + b.region.lower() + '\n')
				outfile.write('END\n\n')
		except:
			print 'Error: At least one boundary_condition with valid attributes is required\n'
		
	def _read_source_sink(self,infile):
		p = psource_sink()
		np_flow = p.flow
		np_region = p.region
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first key word
			
			if key == 'flow_condition':
				np_flow = line.split()[-1]	# take last word
			elif key == 'region':
				np_region = line.split()[-1]
			elif key in ['/','end']: keepReading = False
			
		# Create an empty source sink and assign the values read in
		new_source_sink = psource_sink(np_flow,np_region)
		self._source_sink = new_source_sink
		
	def _write_source_sink(self,outfile):
		self._header(outfile,headers['source_sink'])
		ss = self.source_sink
		outfile.write('SOURCE_SINK\n')
		
		# Write out initial_condition variables
		if ss.flow:
			outfile.write('  FLOW_CONDITION ' + ss.flow.lower() + '\n')
		else:
			print 'error: source_sink.flow (flow_condition) is required\n'
		if ss.region:
			outfile.write('  REGION ' + ss.region.lower() + '\n')
		else:
			print 'error: source_sink.region is required\n'
		outfile.write('END\n\n')
		
	def _delete_strata(self,strata=pstrata()):
		self._strata_list.remove(strata)
			
	def _read_strata(self,infile):
		strata = pstrata()
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first key word
			
			if key == 'region':
				strata.region = line.split()[-1]	# take last word
			elif key == 'material':
				strata.material = line.split()[-1]	# take last word
			elif key in ['/','end']: keepReading = False
			
		# Create an empty source sink and assign the values read in
		self.add(strata)
		
	def _add_strata(self,strata=pstrata(),overwrite=False):			#Adds a strata object.
		# check if stratigraphy coupler already exists
		if isinstance(strata,pstrata):		
			if strata.region in self.strata.keys():
				if not overwrite:
					warning = 'WARNING: A strata with name \''+str(strata.region)+'\' already exists. strata will not be defined, use overwrite = True in add() to overwrite the old strata.'
					print warning; print
					_buildWarnings(warning)
					return
				else:
					self.delete(self.strata[strata.region])
					
		if strata not in self._strata_list:
			self._strata_list.append(strata)
			
	def _delete_strata(self,strata=pstrata()):
		self._strata_list.remove(strata)
		
	def _write_strata(self,outfile):
		self._header(outfile,headers['strata'])
#		strata = self.strata
		outfile.write('STRATA\n')
		
		# Write out strata condition variables
		for strata in self.strata_list:
			if strata.region:
				outfile.write('  REGION ' + strata.region.lower() + '\n')
			else:
				print 'ERROR: strata.region is required\n'
			if strata.material:
				outfile.write('  MATERIAL ' + strata.material.lower() + '\n')
			else:
				print 'ERROR: strata.material is required\n'
			outfile.write('END\n\n')
	
	def _read_checkpoint(self, infile, line):
		checkpoint = pcheckpoint()
		
		checkpoint.frequency = line.split()[-1].lower() # checkpoint int passed in.
		
		self._checkpoint = checkpoint
		
	def _write_checkpoint(self, outfile):
		self._header(outfile,headers['checkpoint'])
		checkpoint = self.checkpoint
		
		try:
			# error-checking to make sure checkpoint.frequency is int (integer)
			checkpoint.frequency = int(checkpoint.frequency)
			
			# write results
			outfile.write('CHECKPOINT ')
			outfile.write(str(checkpoint.frequency))
			outfile.write('\n')
		except(ValueError):
			print 'ERROR: checkpoint.frequency is not int (integer).\n'
			
			# write results
			outfile.write('CHECKPOINT ')
			outfile.write(str(checkpoint.frequency))
			outfile.write('\n')
			
		outfile.write('\n')
		
	def _read_restart(self, infile, line):
		restart = prestart()
		
		tstring = line.split()[1:] # 1st line for restart passed in
		
		restart.file_name = tstring[0]
		restart.time_value = floatD(tstring[1])
		restart.time_unit = tstring[2]
		
		self._restart = restart
		
	def _write_restart(self, outfile):
		self._header(outfile,headers['restart'])
		restart = self.restart
		
		# write file name
		outfile.write('RESTART ' + str(restart.file_name) + ' ')
		
		# Write time value
		try:
			# error-checking
			restart.time_value = floatD(restart.time_value)
			
			# writing
			outfile.write(strD(restart.time_value) + ' ')
		except:
			print 'ERROR: restart.time_value is not float.'
			
			# writing
			outfile.write(strD(restart.time_value) + ' ')
		
		# Write time unit of measurement
		restart.time_unit = str(restart.time_unit).lower()
		if restart.time_unit in time_units_allowed:
			outfile.write(restart.time_unit)
		else:
			print 'ERROR: restart.time_unit \'',restart.time_unit,'\' is invalid. Valid times units are:',time_units_allowed,'\n'
			outfile.write(restart.time_unit)
			
		outfile.write('\n\n')
		
	def _read_chemistry(self, infile):
		chem = pchemistry()
		
		# lists needs to be reset in python so their not continually appended to.
		chem.pspecies_list = []
		chem.sec_species_list = []
		chem.gas_species_list = []
		chem.minerals_list = []
		chem.m_kinetics_list = []
		chem.output_list = []
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			try:
				key = line.strip().split()[0].lower()	# take first key word
			except(IndexError):
				continue # Read the next line if line is empty.
			if key == 'primary_species':
				while True:
					line = infile.readline()	# get next line
					if line.strip() in ['/','end']: break
					chem.pspecies_list.append(line.strip())
			elif key == 'secondary_species':
				while True:
					line = infile.readline()	# get next line
					if line.strip() in ['/','end']: break
					chem.sec_species_list.append(line.strip())
			elif key == 'gas_species':
				while True:
					line = infile.readline()	# get next line
					if line.strip() in ['/','end']: break
					chem.gas_species_list.append(line.strip())
			elif key == 'minerals':
				while True:
					line = infile.readline()	# get next line
					if line.strip() in ['/','end']: break
					chem.minerals_list.append(line.strip())
			elif key == 'mineral_kinetics':
				while True:
					line = infile.readline()	# get next line
					if line.strip() in ['/','end']: break
					
					mkinetic = pchemistry_m_kinetic() # temporary object
					mkinetic.rate_constant_list = []
					
					# assign kinetic mineral name
					mkinetic.name = line.strip()
					
					# Write mineral attributes here
					while True:
						line = infile.readline()	# get next line
						if line.strip().lower() in ['/','end']: break
						
						# key is a kinetic mineral attribute here
						key = line.strip().split()[0].lower() # take 1st
						
						tstring = line.split()[1:] # Assigns the rest of the line
						
						# assign kinetic mineral attributes
						if key == 'rate_constant':
							for substring in tstring:
								try:
									mkinetic.rate_constant_list.append(floatD(substring))
								except(ValueError):
									mkinetic.rate_constant_list.append(substring)
								
					chem.m_kinetics_list.append(mkinetic) # object assigned
			elif key == 'database':
				chem.database = line.split()[-1]	# take last word
			elif key == 'log_formulation':
				chem.log_formulation = True
			elif key == 'activity_coefficients':
				chem.activity_coefficients = line.split()[-1]
			elif key == 'molal':
				chem.molal = True
			elif key == 'output':
				while True:
					line = infile.readline()	# get next line
					if line.strip() in ['/','end']: break
					chem.output_list.append(line.strip())
			elif key in ['/','end']: keepReading = False
			
		# Create an empty chemistry object and assign the values read in
		self._chemistry = chem
		
	'''
	Automate adding the sub-class m_kinetic to chemistry object.
	Function will provide a warning if a m_kinetic.name already exists
	in the chemistry object it is trying to add it to.
	'''
	def _add_chemistry_m_kinetic(self,m_kinetic=pchemistry_m_kinetic(),
				     overwrite=False): # Adds a mineral_kinetic object
		
		chemistry = self._chemistry
		
		# check if m_kinetic already exists
		if isinstance(m_kinetic,pchemistry_m_kinetic):
			if m_kinetic.name in self._get_m_kinetic().keys():
				if not overwrite:
					warning = 'WARNING: A m_kinetic with name \''+str(m_kinetic.name)+'\' already exists in chemistry. Mineral_Kinetic will not be defined, use overwrite = True in add() to overwrite the old m_kinetic.'
					print warning; print
					_buildWarnings(warning)
					return # exit function
				else: # Executes if overwrite = True
					self.delete(self._get_m_kinetic()[m_kinetic.name], chemistry)
					
		# Add m_kinetic to chemistry (as a sub-class) if that specific 
		# m_kinetic does not exist in chemistry object
		if m_kinetic not in chemistry.m_kinetics_list:
			chemistry.m_kinetics_list.append(m_kinetic)
			
	def _write_chemistry(self,outfile):
		self._header(outfile,headers['chemistry'])
		c = self.chemistry
		outfile.write('CHEMISTRY\n')
		
		# Write out chemistry variables
		if c.pspecies_list:
			outfile.write('  PRIMARY_SPECIES\n')
			for p in c.pspecies_list: # p = primary_specie in primary_species_list
				outfile.write('    ' + p + '\n')
			outfile.write('  /\n')
		if c.sec_species_list:
			outfile.write('  SECONDARY_SPECIES\n')
			for s in c.sec_species_list: # s = secondary_specie
				outfile.write('    ' + s + '\n')
			outfile.write('  /\n')
		if c.gas_species_list:
			outfile.write('  GAS_SPECIES\n')
			for g in c.gas_species_list: # s = gas_specie
				outfile.write('    ' + g + '\n')
			outfile.write('  /\n')
		if c.minerals_list:
			outfile.write('  MINERALS\n')
			for m in c.minerals_list: # m = mineral
				outfile.write('    ' + m + '\n')
			outfile.write('  /\n')
		if c.m_kinetics_list:
			outfile.write('  MINERAL_KINETICS\n')
			for mk in c.m_kinetics_list: # mk = mineral_kinetics
				outfile.write('    ' + mk.name + '\n')
				
				if mk.rate_constant_list:
					outfile.write('      RATE_CONSTANT ')
				for rate in mk.rate_constant_list: 
					try:
						outfile.write(strD(rate) + ' ')
					except(TypeError):
						outfile.write(rate + ' ')
				outfile.write('\n    /\n') # marks end for mineral name
			outfile.write('  /\n') # marks end for mineral_kinetics
		if c.database:
			outfile.write('  DATABASE ' + c.database + '\n')
		if c.log_formulation:
			outfile.write('  LOG_FORMULATION\n')
		if c.activity_coefficients:
			outfile.write('  ACTIVITY_COEFFICIENTS ' + c.activity_coefficients.upper() + '\n')
		if c.molal:
			outfile.write('  MOLAL\n')
		if c.output_list:
			outfile.write('  OUTPUT\n')
			for o in c.output_list:	# o = output in in output_list
				outfile.write('    ' + o + '\n')
			outfile.write('  /\n')
		outfile.write('END\n\n')
		
	def _read_transport(self,infile,line):
		p = ptransport('')
		np_name = line.split()[-1].lower()	# Transport Condition name passed in.
		np_type = p.type
		np_constraint_list_value = []
		np_constraint_list_type = []
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.split()[0].lower()	# take first key word
			
			if key == 'type':
				if len(line.split()) == 2: # Only Assign if 2 words are on the line
					np_type = line.split()[-1]	# take last word
			elif key == 'constraint_list':
				keepReading2 = True
				line = infile.readline()
				while keepReading2:
					try:
						np_constraint_list_value.append(floatD(line.split()[0].lower())) # Read 1st word online
						np_constraint_list_type.append(line.split()[1].lower()) # Read 2nd word on line
					except:
						print 'Error: constraint_list_value and constraint_list_type requires at least one value. Value should = Number and type should = String\n'
					
					line = infile.readline()	# get next line
					key = line.split()[0].lower()	# Used to stop loop when / or end is read
					if key in ['/','end']: keepReading2 = False
			elif key in ['/','end']: keepReading = False
		
		# Create an empty transport condition and assign the values read in
		new_transport = ptransport(np_name,np_type,np_constraint_list_value,
					   np_constraint_list_type)
		self.add(new_transport)
		
	def _add_transport(self,transport=ptransport(),overwrite=False):	#Adds a transport object.
		# check if transport already exists
		if isinstance(transport,ptransport):	
			if transport.name in self.transport.keys():
				if not overwrite:
					warning = 'WARNING: A transport with name \''+str(transport.name)+'\' already exists. transport will not be defined, use overwrite = True in add() to overwrite the old transport.'
					print warning; warning
					_buildWarnings(warning)
					return
				else:
					self.delete(self.transport[transport.name])
					
		if transport not in self._transportlist:
			self._transportlist.append(transport)
			
	def _delete_transport(self,transport=ptransport()):
		self._transportlist.remove(transport)
		
	def _write_transport(self,outfile):
		self._header(outfile,headers['transport_condition'])
		tl = self.transportlist
		for t in tl: # t for transport
			if t.name:
				outfile.write('TRANSPORT_CONDITION '+t.name.lower()+'\n')
			else:
				print 'Error: transport_condition['+str(tl.index(t))+'].name is required.\n'
			if t.type.lower() in transport_condition_types_allowed:
				outfile.write('  TYPE '+t.type.lower()+'\n')
			else:
				print 'ERROR: transport.type: \'' + t.type +'\' is invalid.'
				print '       valid transport_condition.types:', transport_condition_types_allowed, '\n'	
			try :
				outfile.write('  CONSTRAINT_LIST\n')

				clv = t.constraint_list_value
				clt = t.constraint_list_type
	
				i=0 # index for constraint_list_value and constraint_list_type
				for i in range(0, len(clv)):
					if clv[i] != None:
						outfile.write('    '+strD(clv[i]))
					if clt[i] != None:
						outfile.write('  '+str(clt[i]).lower())
					else:
						print 'Error: transport['+str(tl.index(t))+'].constraint_list_type['+str(clt.index(i))+'] is required to have a value when transport.constraint_list_value does.\n'
			except:
				print 'Error: transport.constraint_list_value and transport.constraint_list_type should be in list format, be equal in length, and have at least one value.\n'
			outfile.write('\n  END\n')	# END FOR CONSTRAINT_LIST
			outfile.write('END\n\n')	# END FOR TRANSPORT_CONDITION
			
	def _read_constraint(self, infile, line):
		constraint = pconstraint()
		constraint.name = line.split()[-1].lower()	# constraint name passed in.
		constraint.concentration_list = []
		constraint.mineral_list = []
	
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.split()[0].lower()	# take first key word
			
			if key == 'concentrations':
				
				while True:
					line = infile.readline() # get next line
					tstring = line.split()	# Convert line to temporary list of strings
					
					if line.strip().lower() in ['/','end']: 
						break # Stop loop if line is a / or 'end'
					
					concentrations = pconstraint_concentration()
					
					# Assign concentrations - one line
					try:
						concentrations.pspecies = tstring[0]
						concentrations.value = floatD(tstring[1])
						concentrations.constraint = tstring[2]
						concentrations.element = tstring[3]
					except(IndexError):
						pass # No assigning is done if a value doesn't exist while being read in.
					constraint.concentration_list.append(concentrations)

			elif key == 'minerals':
				while True:
					line = infile.readline()
					tstring = line.split()
					if line.strip().lower() in ['/','end']:
						break
					mineral = pconstraint_mineral()

					try:
						mineral.name = tstring[0]
						mineral.volume_fraction = floatD(tstring[1])
						mineral.surface_area = floatD(tstring[2])

					except(IndexError):
						pass # No assigning is done if a value doesn't exist while being read in.
				
					constraint.mineral_list.append(mineral)

			elif key in ['/','end']: keepReading = False
			
		self.add(constraint)
		
	def _add_constraint(self,constraint=pconstraint(),overwrite=False):	#Adds a constraint object.
		# check if constraint already exists
		if isinstance(constraint,pconstraint):		
			if constraint.name in self.constraint.keys():
				if not overwrite:
					warning = 'WARNING: A constraint with name \''+str(constraint.name)+'\' already exists. constraint will not be defined, use overwrite = True in add() to overwrite the old constraint.'
					print warning; print
					_buildWarnings(warning)
					return
				else:
					self.delete(self.constraint[constraint.name])
					
		if constraint not in self._constraint_list:
			self._constraint_list.append(constraint)
			
	def _delete_constraint(self,constraint=pconstraint()):
		self._constraint_list.remove(constraint)
		
	# Adds a constraint_concentration object
	def _add_constraint_concentration(self,
					  constraint_concentration=pconstraint_concentration(),
					  index='',overwrite=False):
		
		# check if constraint.name was specified
		if index:
			if isinstance(index,str):
				constraint = self.constraint.get(index) # Assign constraint object to existing constraint object with string type name/index
				if not constraint: # Occurs if index/string is not found in constraint object
					print 'WARNING: a constraint object with constraint.name', index, 'was not found. Current found entries are:', self.constraint.keys(), 'pconstraint_concentration was not added.\n'
					return

			elif isinstance(index,pconstraint):
				constraint = index # Assigns if index is the constraint object reference
		else: # Set constraint to last constraint object in list
			constraint = self.constraint_list[-1]
		
		# check if constraint_concentration already exists
		if isinstance(constraint_concentration,pconstraint_concentration):
			if constraint_concentration.pspecies in self._get_constraint_concentration(constraint).keys():
				if not overwrite:
					warning = 'WARNING: A constraint_concentration with pspecies \''+str(constraint_concentration.pspecies)+'\' already exists in constraint with name \''+str(constraint.name)+'\'. constraint_concentration will not be defined, use overwrite = True in add() to overwrite the old constraint_concentration. Use constraint=\'name\' if you want to specify the constraint object to add constraint_concentration to.'
					print warning; print
					_buildWarnings(warning)
					return
				else: # Executes if overwrite = True
					self.delete(self._get_constraint_concentration(constraint)[constraint_concentration.pspecies],
						    constraint)
		
		# Add constraint_concentration to constraint (as a sub-class) if constraint_concentration does not exist in specified constraint object
		if constraint_concentration not in constraint.concentration_list:
			constraint.concentration_list.append(constraint_concentration)
			
	def _delete_constraint_concentration(self,
					     constraint_concentration=pconstraint_concentration(),
					     constraint=pconstraint()):
		constraint.concentration_list.remove(constraint_concentration)
		
	def _write_constraint(self, outfile):
		self._header(outfile,headers['constraint'])
		cl = self.constraint_list
		
		for c in cl:	# c = constraint, cl = constraint_list
			if c.name:
				outfile.write('CONSTRAINT '+c.name.lower()+'\n')
			else:
				print 'Error: constraint_list['+str(cl.index(c))+'].name is required.\n'

			outfile.write('  CONCENTRATIONS\n')
			
			for concn in c.concentration_list: # concn = concentration, c = constraint
				if concn.pspecies:
					outfile.write('    ' + concn.pspecies)
				if concn.value:
					outfile.write('  ' + strD(concn.value))
				if concn.constraint:
					outfile.write('  ' + concn.constraint)
				if concn.element:
					outfile.write('  ' + concn.element)
				outfile.write('\n')

			outfile.write('  /\n') 	# END for concentrations
			if c.mineral_list:
				outfile.write('  MINERALS\n')
				for mineral in c.mineral_list:
					if mineral.name:
						outfile.write('    ' + mineral.name)
					if mineral.volume_fraction:
						outfile.write('  ' + strD(mineral.volume_fraction))
					if mineral.surface_area:
						outfile.write('  ' + strD(mineral.surface_area))
					outfile.write('\n')
				outfile.write('  /\n') 	# END for concentrations
			outfile.write('END\n\n')	# END for constraint
			
	def _header(self,outfile,header):
		if not header: return
		ws = '# '
		pad = int(np.floor((80 - len(header) - 4)/2))
		for i in range(pad): ws+='='
		ws+=' '+header+' '
		for i in range(pad): ws+='='
		ws+='\n'
	
	def _get_co2_database(self): return self._co2_database
	def _set_co2_database(self,value): self._co2_database = value
	co2_database = property(_get_co2_database, _set_co2_database) #: (**)
	
	def _get_uniform_velocity(self): return self._uniform_velocity
	def _set_uniform_velocity(self, object): self._uniform_velocity = object
	uniform_velocity = property(_get_uniform_velocity, _set_uniform_velocity) #: (**)
	
	def _get_mode(self): return self._mode
	def _set_mode(self, object): self._mode = object
	mode = property(_get_mode, _set_mode) #: (**)	
	
	def _get_grid(self): return self._grid
	def _set_grid(self, object): self._grid = object
	grid = property(_get_grid, _set_grid) #: (**)
	
	def _get_time(self): return self._time
	def _set_time(self, object): self._time = object
	time = property(_get_time, _set_time) #: (**)
	
	def _get_proplist(self): return self._proplist
	proplist = property(_get_proplist) #: (**) list of material properties
	def _get_prop(self): 
		return dict([(p.id,p) for p in self.proplist]+[(p.id,p) for p in self.proplist])
	prop = property(_get_prop) #: (**) dictionary of material properties, indexable by ID or name
	
	def _get_filename(self): return self._filename
	def _set_filename(self,value): self._filename = value
	filename = property(_get_filename, _set_filename) #: (**)
	
	def _get_timestepper(self): return self._timestepper
	def _set_timestepper(self, object): self._timestepper = object
	timestepper = property(_get_timestepper, _set_timestepper) #: (**)
	
	def _get_lsolverlist(self): return self._lsolverlist
	def _set_lsolverlist(self, object): self._lsolverlist = object
	lsolverlist = property(_get_lsolverlist, _set_lsolverlist) #: (**)
	def _get_lsolver(self):
		return dict([lsolv.name,lsolv] for lsolv in self.lsolverlist if lsolv.name)
	lsolver = property(_get_lsolver)#: (*dict[pnsolver]*) Dictionary of linear solver objects, indexed by linear solver name.
	
	def _get_nsolverlist(self): return self._nsolverlist
	def _set_nsolverlist(self, object): self._nsolverlist = object
	nsolverlist = property(_get_nsolverlist, _set_nsolverlist) #: (**)
	def _get_nsolver(self):
		return dict([nsolv.name,nsolv] for nsolv in self.nsolverlist if nsolv.name)
	nsolver = property(_get_nsolver)#: (*dict[pnsolver]*) Dictionary of newton solver objects, indexed by newton solver name.
	
	def _get_output(self): return self._output
	def _set_output(self, object): self._output = object
	output = property(_get_output, _set_output) #: (**)
	
	def _get_fluid(self): return self._fluid
	def _set_fluid(self, object): self._fluid = object
	fluid = property(_get_fluid, _set_fluid) #: (**)
	
	def _get_saturation(self): return self._saturation
	def _set_saturation(self, object): self._saturation = object
	saturation = property(_get_saturation, _set_saturation) #: (**)
	
	def _get_regionlist(self): return self._regionlist
	def _set_regionlist(self, object): self._regionlist = object
	regionlist = property(_get_regionlist, _set_regionlist) #: (**)
	def _get_region(self):
		return dict([region.name.lower(),region] for region in self.regionlist if region.name)
	region = property(_get_region)#: (*dict[pregion]*) Dictionary of region objects, indexed by region name.
	
	def _get_observation(self): return self._observation
	def _set_observation(self, object): self._observation = object
	observation = property(_get_observation, _set_observation) #: (**)
	
	def _get_flowlist(self): return self._flowlist
	def _set_flowlist(self, object): self._flowlist = object
	flowlist = property(_get_flowlist, _set_flowlist) #: (**)
	def _get_flow(self):
		return dict([flow.name.lower(),flow] for flow in self.flowlist if flow.name.lower)
	flow = property(_get_flow)#: (*dict[pflow]*) Dictionary of flow objects, indexed by flow name.
	
	def _get_flow_variable(self, flow=pflow()):
		return dict([flow_variable.name.lower(), flow_variable] for flow_variable in flow.varlist if flow_variable.name.lower())
	flow_variable = property(_get_flow_variable)#: (*dict[pflow_variable]*) Dictionary of pflow_variable objects in a specified flow object, indexed by flow_variable name
	
	def _get_initial_condition(self): return self._initial_condition
	def _set_initial_condition(self, object): self._initial_condition = object
	initial_condition = property(_get_initial_condition, _set_initial_condition) #: (**)
	
	def _get_boundary_condition_list(self): return self._boundary_condition_list
	def _set_boundary_condition_list(self, object): self._boundary_condition_list = object
	boundary_condition_list = property(_get_boundary_condition_list, _set_boundary_condition_list) #: (**)
	def _get_boundary_condition(self):
		return dict([boundary_condition.region,boundary_condition] for boundary_condition in self.boundary_condition_list if boundary_condition.region)
	boundary_condition = property(_get_boundary_condition)#: (*dict[pboundary_condition]*) Dictionary of boundary_condition objects, indexed by flow region.
	
	def _get_source_sink(self): return self._source_sink
	def _set_source_sink(self, object): self._source_sink = object
	source_sink = property(_get_source_sink, _set_source_sink) #: (**)
	
	def _get_strata_list(self): return self._strata_list
	def _set_strata_list(self, object): self._strata_list = object
	strata_list = property(_get_strata_list, _set_strata_list) #: (**)
	def _get_strata(self):
		return dict([strata.region,strata] for strata in self.strata_list if strata.region)
	strata = property(_get_strata)#: (*dict[pstrata]*) Dictionary of strata objects, indexed by strata region.
	
	def _get_checkpoint(self): return self._checkpoint
	def _set_checkpoint(self, object): self._checkpoint = object
	checkpoint = property(_get_checkpoint, _set_checkpoint) #: (**)
	
	def _get_restart(self): return self._restart
	def _set_restart(self, object): self._restart = object
	restart = property(_get_restart, _set_restart) #: (**)
	
	def _get_chemistry(self): return self._chemistry
	def _set_chemistry(self, object): self._chemistry = object
	chemistry = property(_get_chemistry, _set_chemistry) #: (**)
	
	def _get_m_kinetic(self):
		chemistry = self._chemistry
		return dict([m_kinetic.name, m_kinetic] for m_kinetic in chemistry.m_kinetics_list if m_kinetic.name)
	m_kinetic = property(_get_m_kinetic)#: (*dict[pchemistry_m_kinetic]*) Dictionary of pfchemistry_m_kinetic objects in chemistry object, indexed by m_kinetic name
	
	def _get_transportlist(self): return self._transportlist
	def _set_transportlist(self, object): self._transportlist = object
	transportlist = property(_get_transportlist, _set_transportlist) #: (**)
	def _get_transport(self):
		return dict([transport.name,transport] for transport in self.transportlist if transport.name)
	transport = property(_get_transport)#: (*dict[ptransport]*) Dictionary of transport objects, indexed by transport name.
	
	def _get_constraint_list(self): return self._constraint_list
	def _set_constraint_list(self, object): self._constraint_list = object
	constraint_list = property(_get_constraint_list, _set_constraint_list) #: (**)
	def _get_constraint(self):
		return dict([constraint.name.lower(),constraint] for constraint in self.constraint_list if constraint.name.lower())
	constraint = property(_get_constraint)#: (*dict[pconstraint]*) Dictionary of constraint objects, indexed by constraint name.
	
	def _get_constraint_concentration(self, constraint=pconstraint()):
		return dict([constraint_concentration.pspecies,constraint_concentration] for constraint_concentration in constraint.concentration_list if constraint_concentration.pspecies)
	constraint_concentration = property(_get_constraint_concentration)#: (*dict[pconstraint_concentration]*) Dictionary of pconstraint_concentration objects in a specified constraint object, indexed by constraint_concentration pspecies
	
