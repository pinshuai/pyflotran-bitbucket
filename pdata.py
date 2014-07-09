""" /lass for pflotran data """
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

pressure_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient', 'conductance', 'seepage']
temperature_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient']
concentration_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient']


cards = ['co2_database','uniform_velocity','mode','chemistry','grid','timestepper',
	 'material_property','time','linear_solver','newton_solver','output','fluid_property',
		'saturation_function','region','observation','flow_condition',
		'transport_condition','initial_condition','boundary_condition','source_sink',
		'strata','constraint']
headers = ['co2 database path','uniform velocity','mode','chemistry','grid',
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
	""" Class for uniform velcity for transport observation.
	Example:
	UNIFORM_VELOCITY 3.84259d-6 0.d0 0.d0 ! 1.38333 cm/h
	"""
	
	def __init__(self,value_list=[]):
		self._value_list = value_list
		
	def _get_value_list(self): return self._value_list
	def _set_value_list(self,value): self._value_list = value
	value_list = property(_get_value_list, _set_value_list) #: (**)

class pmaterial(object):
	""" Class for material property card

	"""

	def __init__(self,id=None,name='',porosity=None,tortuosity=None,density=None,
		     specific_heat=None,cond_dry=None,cond_wet=None,
		     saturation='',permeability=[]):
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
	
	def __init__(self,tf=[],dti=[],dtf=[],dtf_lv=[],
		     dtf_li=[],dtf_i=0,dtf_lv_unit=[], dtf_li_unit=[]):
		self._tf = tf		# Final Time, 2nd parameter is unit, same for all other 
					# variables except dtf_i
		self._dti = dti		# Initial Timestep Size
		self._dtf = dtf		# Maximum Timestep Size
		self._dtf_lv = dtf_lv	# Maximum Timestep Size list - # before 'y' - 
					# goes after 1st entry (dtf) - lv = list value
		self._dtf_li = dtf_li	# Maximum Timestep Size list - # after 'y' - 
					# goes after 1st entry (dtf) - li = list increment
		self._dtf_i = dtf_i	# Index, also being used as a counter
					# 2nd and 3rd paramenter can be a string to specify time 
					# unit for a dtf list
		self._dtf_lv_unit = dtf_lv_unit	# time unit list for lv (s,m,h,d,mo,y)	- Needed for hardcoding and a user interface - not needed for reading test files
		self._dtf_li_unit = dtf_li_unit # time unit list for li (s,m,h,d,mo,y) - Needed for hardcoding and a user interface - not needed for reading test files
		
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
	def _get_dtf_lv_unit(self): return self._dtf_lv_unit
	def _set_dtf_lv_unit(self,value): self._dtf_lv_unit = value
	dtf_lv_unit = property(_get_dtf_lv_unit, _set_dtf_lv_unit) #: (**)
	def _get_dtf_li_unit(self): return self._dtf_li_unit
	def _set_dtf_li_unit(self,value): self._dtf_li_unit = value
	dtf_li_unit = property(_get_dtf_li_unit, _set_dtf_li_unit) #: (**)
	
class pgrid(object):		# discretization
	""" Class for grid property

	"""

	def __init__(self,type='structured',lower_bounds=[0.0,0.0,0.0],upper_bounds=[50.0,50.0,50.0],
				 origin=[0.0,0.0,0.0],nxyz=[10,10,10],dxyz=[5,5,5],gravity=[0.0,0.0,-9.8068],filename=''):
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
	""" Class for mode

	"""

	def __init__(self,name=''):
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

class plsolver(object):
	""" Class for linear solver

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
	""" Class for newton solver

	"""

	def __init__(self,name='',atol=None,rtol=None,stol=None,dtol=None,itol=None,
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
	
	# Remember to add time_list=[] back in, temporarily removed
	def __init__(self, time_list=[], mass_balance=False, print_column_ids=False,
		     periodic_observation_timestep=None, format=[],velocities=False):
		self._time_list = time_list
		self._mass_balance = mass_balance
		self._print_column_ids = print_column_ids
		self._periodic_observation_timestep = periodic_observation_timestep
		self._format = format
		self._velocities = velocities
		
	def _get_time_list(self): return self._time_list
	def _set_time_list(self,value): self._time_list = value
	time_list = property(_get_time_list, _set_time_list)
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
	
	def __init__(self,name='',permeability_function_type=None,saturation_function_type=None,
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
	
class pregion(object):
	"""Class for regions
	
	There are currently unresolved issues with the initializer not working correctly.
	This has caused problems with reading in list attributes from list objects.

	"""
	
	def __init__(self,name='',coordinates_lower=[0.0,0.0,0.0],coordinates_upper=[0.0,0.0,0.0],
			face=None):
		self._name = name
		self._coordinates_lower = coordinates_lower	# 3D coordinates
		self._coordinates_upper = coordinates_upper	# 3D coordinates
		self._face = face
		
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
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
	
	def __init__(self,region=None):
		self._region = region
		
	def _get_region(self): return self._region
	def _set_region(self,value): self._region = value
	region = property(_get_region, _set_region) 
	
class pflow(object):
	"""Class for flow conditions - There can be multiple flow condition objects

	"""
	
	def __init__(self,name='',units_list=None,
			iphase=None,sync_timestep_with_update=False,
			varlist=[]):
		self._name = name		# Include initial, top, source
		self._units_list = units_list	# Specify type of units to display such as
						# time,length,rate,pressure,velocity, temperature,
						# concentration, and enthalpy.
						# May be used to determine each variable unit
		self._iphase = iphase			# Holds 1 int
		self._sync_timestep_with_update = sync_timestep_with_update	# Boolean
		self._varlist = varlist
		
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
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
	
class pflow_variable(object):
	"""Sub-class of pflow for each kind of variable (Includes type and value)

	"""
	
	def __init__(self,name='',type=None, valuelist=[], unit='',
		     time_unit_type='', data_unit_type='', list=[]):
		self._name = name	# Pressure,temp., concen.,enthalpy...(String)
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
	def _set_name(self,value): self._name = value
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
		self._name = name
		self._concentration_list = concentration_list # Composed of pconstraint_concentration objects
		self._mineral_list = mineral_list # list of minerals
		
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
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
    
class pmineral_constraint(object):
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
	"""Class for pflotran data file

	"""

	def __init__(self, filename='', work_dir=''):
		from copy import copy
		# Note that objects need to be instantiated when hard-coded when it's set to
		# None here.
		self._co2_database = ''
		self._uniform_velocity = puniform_velocity()
		self._mode = pmode()
		self._chemistry = None
		self._grid = pgrid()
		self._timestepper = None
		self._proplist = []
		self._time = ptime()
		self._lsolverlist = []	# Possible to have 1 or 2 nsolver lists. FLOW/TRAN
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
		print wd
		print self._path.filename
		returnFlag = self.write(wd+self._path.filename) # ALWAYS write input file
		print returnFlag
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
					if card in ['co2_database','material_property','mode','grid',
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
        
	def add(self,obj,overwrite=False):					#Adds a new object to the file
		'''Attach a region, boundary condition object to the data file.
		
		:param obj: Object to be added to the data file.
		:type obj: pregion
		:param overwrite: Flag to overwrite macro if already exists for a particular zone.
		:type overwrite: bool
		'''
		if isinstance(obj,pregion): self._add_region(obj,overwrite)

	def delete(self,obj): 								#Deletes an object from the file
		'''Delete a region, boundary condition object from the data file.
		
		:param obj: Object to be deleted from the data file. Can be a list of objects.
		:type obj: pregion, list
		'''
		if isinstance(obj,pregion): self._delete_region(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):
				if isinstance(obji,pregion): self._delete_region(obji)
		
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
		if self.mode.name:
			outfile.write('MODE ')
			outfile.write(self.mode.name.lower()+'\n\n')
			
	def _read_co2_database(self,infile,line):
		self._co2_database = line.split()[-1]
	
	def _write_co2_database(self,outfile):
		self._header(outfile,headers['co2_database'])
		outfile.write('CO2_DATABASE ' + self._co2_database + '\n\n')
		
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
				ng_nxyz[0] = int(line.split()[1])
				ng_nxyz[1] = int(line.split()[2])
				ng_nxyz[2] = int(line.split()[3])
			elif key == 'gravity':
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
		new_grid = pgrid(ng_type,ng_lower_bounds,ng_upper_bounds,ng_origin,ng_nxyz,
						 ng_dxyz,ng_gravity,ng_filename) 		# create an empty grid
		self._grid = new_grid
	
	def _write_grid(self,outfile):
		self._header(outfile,headers['grid'])
		grid = self.grid
		outfile.write('GRID\n')
		if grid.type:
			outfile.write('  TYPE ' + grid.type + '\n')
		else:
			print 'error: grid.type is required'
		if grid.lower_bounds or grid.upper_bounds:
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

		self._proplist.append(new_prop)	
	
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
		p = ptime()
		np_tf = p._tf
		np_dti = p._dti
		np_dtf = p._dtf
		np_dtf_lv = [] #p._dtf_lv
		np_dtf_li = [] #p._dtf_li
		np_dtf_i = p._dtf_i
		np_dtf_lv_unit = p._dtf_lv_unit
		np_dtf_li_unit = p._dtf_li_unit
		
		keepReading = True
		while keepReading:
			line = infile.readline() 		# get next line
			key = line.split()[0].lower() 		# take first keyword
			if key == 'final_time':
				tstring = line.split()[1:]	# temp list of strings, 
								# do not include 1st sub-string
				if len(tstring) == 2:	# Do this if there is a time unit to read
					np_tf.append(floatD(tstring[0]))
					np_tf.append(tstring[-1])
				else:			# No time unit being read in
					np_tf.append(floatD(tstring[0]))
			elif key == 'initial_timestep_size':
				tstring = line.split()[1:]
				if len(tstring) == 2:
					np_dti.append(floatD(tstring[0]))
					np_dti.append(tstring[-1])	
				else:
					np_dti.append(floatD(tstring[0]))
			elif key == 'maximum_timestep_size':
				if ('at' not in line):
					tstring = line.split()[1:]
					if len(tstring) == 2:
						np_dtf.append(floatD(tstring[0]))
						np_dtf.append(tstring[-1])
					else:
						np_dtf.append(floatD(tstring[0]))
				elif ('at' in line):
					## Read maximum_timestep_size with AT keyword 
					if (key == 'maximum_timestep_size'):
						
						#Read before AT
						tstring = line.split()[1:]
						if len(tstring) >= 2:
							np_dtf_lv.append(floatD(tstring[0]))
							np_dtf_lv_unit.append(tstring[1])
						else:
							np_dtf_lv[np_dtf_i] = floatD(tstring[0])
							
						#Read after AT
						at_i = tstring.index('at') # Find index # in list (Not string)
						tstring = line.split()[at_i+2:] # Use string only after 'at'
						
						if len(tstring) == 2:
							np_dtf_li.append(floatD(tstring[0]))
							np_dtf_li_unit.append(tstring[1])
						else:
							np_dtf_li[np_dtf_i] = floatD(tstring[0])
							
						np_dtf_i = np_dtf_i + 1
							
			elif key in ['/','end']: keepReading = False
			
		# Create new empty time object and assign values read in.	
		new_time = ptime(np_tf,np_dti,np_dtf,np_dtf_lv,np_dtf_li,np_dtf_i)
		self._time = new_time

	def _write_time(self,outfile):
		self._header(outfile,headers['time'])
		time = self.time
		outfile.write('TIME\n')
		
		# write FINAL_TIME statement (tf)
		if time.tf:
			try:
				outfile.write('  FINAL_TIME ' + strD(time.tf[0])) # Write value
				outfile.write(' ' + time.tf[1] +'\n')		  # Write time unit
			except:
				print 'ERROR: time.tf (final time) input is invalid. Format should be a list: [number, string]\n'
		
		# write INITIAL_TIMESTEP_SIZE statement (dti)
		if time.dti:
			try:
				outfile.write('  INITIAL_TIMESTEP_SIZE ' + 
                                        strD(time.dti[0]))		# Write value
				outfile.write(' ' + time.dti[1] +'\n')	# Write time unit
			except:
				print 'ERROR: time.dti (initial timestep size) input is invalid. Format should be a list: [number, string]\n'
		
		# write MAXIMUM_TIMESTEP_SIZE statement	dtf
		if time.dtf:
			try:
				outfile.write('  MAXIMUM_TIMESTEP_SIZE ' + strD(time.dtf[0]))
				outfile.write(' ' + time.dtf[1] +'\n')
			except:
				print 'ERROR: time.dtf (maximum timestep size) input is invalid. Format should be a list: [number, string]\n'
				
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
				outfile.write(strD(time.dtf_lv[i]) + ' ' + 	# Value 
					      time.dtf_lv_unit[i])		# Time Unit
				# write after key word 'AT'
				time.dtf_li_unit[i] = time.dtf_li_unit[i].lower()# lower capitalization
				outfile.write(' at ')
				outfile.write(strD(time.dtf_li[i]) + ' ' + 
					      time.dtf_li_unit[i] + '\n')
			except:
				print 'ERROR: Invalid input at maximum_time_step_size with key word \'at\'. time.dtf_lv and time.dtf_li should be a list of floats. time_dtf_lv_unit and time_dtf_li_unit should be a list of strings. All lists should be of equal length.\n'
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
		
		self._lsolverlist.append(lsolver)	# Assign object
		
	def _read_nsolver(self,infile,line):
		np_name = line.split()[-1].lower() # newton solver type - tran_solver or flow_solver
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
				np_max_it = int(line.split()[-1])
			if key == 'maxf':
				np_max_f = int(line.split()[-1])
			elif key in ['/','end']: keepReading = False
		
		new_nsolver = pnsolver(np_name,np_atol,np_rtol,np_stol,np_dtol,np_itol,
					np_max_it,np_max_f)	# Create an empty newton solver
		self._nsolverlist.append(new_nsolver)
		
	def _write_lsolver(self,outfile):
		self._header(outfile,headers['linear_solver'])
		
		for lsolver in self.lsolverlist:
			if lsolver.name:
				outfile.write('LINEAR_SOLVER ' + lsolver.name.lower() + '\n')
			else: 
				print 'ERROR: name is required when using linear solver.'
			if lsolver.solver:
				outfile.write('  SOLVER ' + lsolver.solver.lower() + '\n')
			outfile.write('END\n\n')
		
	def _write_nsolver(self,outfile):
		self._header(outfile,headers['newton_solver'])
		
		for nsolver in self.nsolverlist:
			# Write Newton Solver Type - Not certain this is correct.
			
			if nsolver.name.lower() == 'flow' or nsolver.name.lower() == 'transport':	# default
				outfile.write('NEWTON_SOLVER ' + nsolver.name.lower() + '\n')
			elif nsolver.name.lower() == 'tran':
				outfile.write('NEWTON_SOLVER ' + nsolver.name.lower() + '\n')
			else:
				print 'error: nsolver_name (newton solver name) is invalid, unrecognized, or missing.\n'
			
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
		p = poutput()
		np_time_list = []
		np_mass_balance = p.mass_balance
		np_print_column_ids = p.print_column_ids
		np_periodic_observation_timestep = p.periodic_observation_timestep
		np_format = p.format
		np_velocities = p.velocities
		
		keepReading = True
		
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first key word
			
			if key == 'times':
				tstring = line.split()[1:] # Turn into list, exempt 1st word
				i=0
				while i < len(tstring):
					try:
						np_time_list.append(floatD(tstring[i]))
					except:
						np_time_list.append(tstring[i])
					i += 1
			if key == 'periodic_observation':
				tstring = line.strip().split()[1].lower()	# Read the 2nd word
				if tstring == 'timestep':
					np_periodic_observation_timestep = int(line.split()[-1])
			elif key == 'print_column_ids':
				np_print_column_ids = True
			elif key == 'format':
				tstring = (line.strip().split()[1:])
				tstring = ' '.join(tstring).lower()	# Convert list into a string seperated by a space
				if tstring == 'tecplot block':
					np_format.append('TECPLOT BLOCK')
				elif tstring == 'tecplot point':
					np_format.append('TECPLOT POINT')
				elif tstring == 'hdf5':
					np_format.append('HDF5')
				elif tstring == 'vtk':
					np_format.append('VTK')
			elif key == 'velocities':
				np_velocities = True
			elif key == 'mass_balance':
				np_mass_balance = True
			elif key in ['/','end']: keepReading = False
			
		# Create new empty output object and assign values read in.
		new_output = poutput(np_time_list, np_mass_balance,np_print_column_ids,
					np_periodic_observation_timestep,np_format,
					np_velocities)	# Create an empty output
		self._output = new_output	# Values read in are assigned now'
		
	def _write_output(self,outfile):
		self._header(outfile,headers['output'])
		output = self.output
		
		# Write Output - if used so null/None entries are not written
		outfile.write('OUTPUT\n')
# Further improvements can be made here in time_list for verifying 1st element is a time unit
		if output.time_list:
			outfile.write('  TIMES ')
			for i in output.time_list:
				outfile.write(' '+strD(i))
			outfile.write('\n')
					
# This is here on purpose - Needed later
		#if output.periodic_observation_time:
			#outfile.write('  PERIODIC_OBSERVATION TIME  '+
					#str(output.periodic_observation_time)+'\n')		
		if output.periodic_observation_timestep:
			outfile.write('  PERIODIC_OBSERVATION TIMESTEP '+
					str(output.periodic_observation_timestep)+'\n')
		if output.print_column_ids:
			outfile.write('  '+'PRINT_COLUMN_IDS'+'\n')
		if output.format:
			for i in range(0, len(output.format)):
				outfile.write('  FORMAT ')
				outfile.write(str(output.format[i].upper()) + '\n')
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
		np_name = line.split()[-1].lower()	# saturation function name, passed in.
		p = psaturation()	# assign defaults before reading in values
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
				np_residual_saturation_liquid = floatD(line.split()[-1])
			elif key == 'residual_saturation_gas':
				np_residual_saturation_gas = floatD(line.split()[-1])
			elif key == 'residual_saturation':	# Alternative to check
				tstring = line.strip().split()[1].lower()	# take 2nd key word
				if tstring == 'liquid_phase':
					np_residual_saturation_liquid = floatD(line.split()[-1])
				elif tstring == 'gas_phase':
					np_residual_saturation_gas = floatD(line.split()[-1])
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
			outfile.write('  ' + saturation.name + '\n')
		else:
			outfile.write('n')
		if saturation.permeability_function_type:
			outfile.write('  PERMEABILITY_FUNCTION_TYPE ' +
					saturation.permeability_function_type + '\n')
		if saturation.saturation_function_type:
			outfile.write('  SATURATION_FUNCTION_TYPE ' + 
					saturation.saturation_function_type + '\n')
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
#		region_name = line.split()[-1].lower()	# saturation function name, passed in.
		np_name = line.split()[-1].lower()	# saturation function name, passed in.
		
		# Assign defaults before reading in value - default initializer is not working
		# correctly. It can initialize values correctly if specified though in the
		# parameters.
		p = pregion()	# Not being used because it's not working correctly
		np_coordinates_lower = [None]*3
		np_coordinates_upper = [None]*3
		np_face = None
		
		keepReading = True
		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first keyword
			if key == 'coordinates':
				keepReading2 = True
				while keepReading2:
					line1 = infile.readline()
					np_coordinates_lower[0] = floatD(line1.split()[0])
					np_coordinates_lower[1] = floatD(line1.split()[1])
					np_coordinates_lower[2] = floatD(line1.split()[2])
					line2 = infile.readline()
					np_coordinates_upper[0] = floatD(line2.split()[0])
					np_coordinates_upper[1] = floatD(line2.split()[1])
					np_coordinates_upper[2] = floatD(line2.split()[2])
					line3 = infile.readline()
					if line3.strip().split()[0].lower() in ['/','end']: keepReading2 = False	
			elif key == 'face':
				np_face = line.strip().split()[-1].lower()
			elif key in ['/','end']: keepReading = False
			
			new_region = pregion(np_name,np_coordinates_lower,np_coordinates_upper,
						np_face)
			self._regionlist.append(new_region)

	def _add_region(self,region=pregion(),overwrite=False):			#Adds a Region object.
		# check if region already exists
		if isinstance(region,pregion):		
			if region.name in self.region.keys():
				if not overwrite:
					_buildWarnings('WARNING: A region with name \''+str(region.name)+'\' already exists. Region will not be defined, use overwrite = True in add() to overwrite the old region.')
					return
				else:
					self.delete(self.pregion[pregion.name])
		
		#region._parent = self
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
			self._flowlist.append(flow)
		
			
	def _write_flow(self,outfile):
		self._header(outfile,headers['flow_condition'])
		
		# Write out all valid flow_conditions objects with FLOW_CONDITION as keyword
		for flow in self.flowlist:
			outfile.write('FLOW_CONDITION  ' + flow.name.lower() + '\n')
			if flow.sync_timestep_with_update:
				outfile.write('  SYNC_TIMESTEP_WITH_UPDATE\n')
			outfile.write('  TYPE\n')
			
			# variable name and type from lists go here
			i = 0
			while i< len(flow.varlist):
				outfile.write('    ' + flow.varlist[i].name.upper() + '  ' +
							  flow.varlist[i].type.lower() + '\n')
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
						outfile.write('  ' + strD(k.data_unit_value_list[0]))
						outfile.write('  ' + strD(k.data_unit_value_list[1]))
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
		self._boundary_condition_list.append(new_boundary_condition)
		
	def _write_boundary_condition(self,outfile):
		self._header(outfile,headers['boundary_condition'])

		# Write all boundary conditions to file
#		if boundary_condition_list:
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
		self._strata_list.append(strata)
		
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
		self._transportlist.append(new_transport)
		
	def _write_transport(self,outfile):
		self._header(outfile,headers['transport_condition'])
		tl = self.transportlist
		for t in tl: # t for transport
			if t.name:
				outfile.write('TRANSPORT_CONDITION '+t.name.lower()+'\n')
			else:
				print 'Error: transport_condition['+str(tl.index(t))+'].name is required.\n'
			if t.type:
				outfile.write('  TYPE '+t.type.lower()+'\n')
			else:
				print 'Error: transport['+str(tl.index(t))+'].type is required\n'
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
					mineral = pmineral_constraint()

					try:
						mineral.name = tstring[0]
						mineral.volume_fraction = floatD(tstring[1])
						mineral.surface_area = floatD(tstring[2])

					except(IndexError):
						pass # No assigning is done if a value doesn't exist while being read in.
				
					constraint.mineral_list.append(mineral)

			elif key in ['/','end']: keepReading = False
			
		self._constraint_list.append(constraint)
		
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
			print c.name
			if c.mineral_list:
				outfile.write('  MINERALS\n')
				for mineral in c.mineral_list:
					print mineral.name
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
	def _get_prop(self): return dict([(p.id,p) for p in self.proplist]+[(p.name,p) for p in self.proplist])
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
	def _get_nsolverlist(self): return self._nsolverlist
	def _set_nsolverlist(self, object): self._nsolverlist = object
	nsolverlist = property(_get_nsolverlist, _set_nsolverlist) #: (**)
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
	def _get_observation(self): return self._observation
	def _set_observation(self, object): self._observation = object
	observation = property(_get_observation, _set_observation) #: (**)
	def _get_flowlist(self): return self._flowlist
	def _set_flowlist(self, object): self._flowlist = object
	flowlist = property(_get_flowlist, _set_flowlist) #: (**)
	def _get_initial_condition(self): return self._initial_condition
	def _set_initial_condition(self, object): self._initial_condition = object
	initial_condition = property(_get_initial_condition, _set_initial_condition) #: (**)
	def _get_boundary_condition_list(self): return self._boundary_condition_list
	def _set_boundary_condition_list(self, object): self._boundary_condition_list = object
	boundary_condition_list = property(_get_boundary_condition_list, _set_boundary_condition_list) #: (**)
	def _get_source_sink(self): return self._source_sink
	def _set_source_sink(self, object): self._source_sink = object
	source_sink = property(_get_source_sink, _set_source_sink) #: (**)
	def _get_strata_list(self): return self._strata_list
	def _set_strata_list(self, object): self._strata_list = object
	strata_list = property(_get_strata_list, _set_strata_list) #: (**)
	def _get_chemistry(self): return self._chemistry
	def _set_chemistry(self, object): self._chemistry = object
	chemistry = property(_get_chemistry, _set_chemistry) #: (**)
	def _get_transportlist(self): return self._transportlist
	def _set_transportlist(self, object): self._transportlist = object
	transportlist = property(_get_transportlist, _set_transportlist) #: (**)
	def _get_constraint_list(self): return self._constraint_list
	constraint_list = property(_get_constraint_list) #: (**)
	def _get_region(self):
		return dict([rgn.name,rgn] for rgn in self.regionlist if rgn.name)
	region = property(_get_region)#: (*dict[pregion]*) Dictionary of region objects, indexed by region name.
