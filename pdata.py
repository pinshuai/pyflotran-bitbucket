""" Class for pflotran data """

"""
PyFLOTRAN v1.0.0 LA-CC-14-094 

Copyright (c) 2014, Los Alamos National Security, LLC.  
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__author__ = "Satish Karra, Cory Kitay"
__version__ = "1.0.0"
__maintainer__ = "Satish Karra"
__email__ = "satkarra@lanl.gov"

print('=========================================')
print('PyFLOTRAN ' + __version__)	# Makes console output a little easier to read
print('Authors: ' + __author__)
print('Contact: ' + __email__ + ' (' + __maintainer__ + ')') 
print('=========================================')

import numpy as np
from copy import deepcopy
from copy import copy
import os,time,sys
import platform
#from subprocess import Popen, PIPE
import pdb
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import itertools as it
from matplotlib import rc
rc('text',usetex=True)

from ptool import*
from pdflt import*

dflt = pdflt()

WINDOWS = platform.system()=='Windows'
if WINDOWS: copyStr = 'copy'; delStr = 'del'; slash = '\\'
else: copyStr = 'cp'; delStr = 'rm'; slash = '/'

# Multiple classes/key words - allowed strings
time_units_allowed = ['s', 'sec','m', 'min', 'h', 'hr', 'd', 
		      'day', 'w', 'week', 'mo', 'month', 'y']
solver_names_allowed = ['transport', 'tran', 'flow'] # newton and linear
# simulation type - allowed strings
simulation_types_allowed = ['subsurface','surface_subsurface']
# mode - allowed strings
mode_names_allowed = ['richards', 'mphase', 'mph',  'flash2',
		      'th no_freezing', 'th freezing', 'immis']

# grid - allowed strings
grid_types_allowed = ['structured', 'structured_mimetic', 'unstructured', 'amr']
grid_symmetry_types_allowed = ['cartesian', 'cylindrical', 'spherical'] # cartesian is default in pflotran

# output - allowed strings
output_formats_allowed = ['TECPLOT BLOCK', 'TECPLOT POINT', 'HDF5', 
			  'HDF5 MULTIPLE_FILES', 'MAD', 'VTK']

output_variables_allowed = ['liquid_pressure','liquid_saturation','liquid_density','liquid_mobility','liquid_energy','liquid_mole_fractions','gas_pressure','gas_saturation','gas_density','gas_mobility','gas_mole_fractions','air_pressure','capillary_pressure','thermodynamic_state','temperature','residual','porosity','mineral_porosity','permeability',
'mineral_porosity']

# saturation_function - allowed strings
saturation_function_types_allowed = ['VAN_GENUCHTEN', 'BROOKS_COREY', 'THOMEER_COREY', 
				     'NMT_EXP', 'PRUESS_1']
permeability_function_types_allowed = ['VAN_GENUCHTEN', 'MUALEM', 'BURDINE', 
				       'NMT_EXP', 'PRUESS_1']

# characteristic_curves - allowed strings - saturation & permeability functions
characteristic_curves_saturation_function_types_allowed = ['VAN_GENUCHTEN', 'BROOKS_COREY']
characteristic_curves_gas_permeability_function_types_allowed = ['MAULEM_VG_GAS','BURDINE_BC_GAS']
characteristic_curves_liquid_permeability_function_types_allowed = ['MAULEM','BURDINE']

# material_property, region, initial_condition, boundary_condition, 
# source_sink, stratigraphy_couplers - manual does not appear to document 
# all valid entries

# flow_conditions - allowed strings
flow_condition_type_names_allowed = ['PRESSURE', 'RATE', 'FLUX', 'TEMPERATURE', 
				'CONCENTRATION', 'SATURATION', 'ENTHALPY']
pressure_types_allowed = ['dirichlet', 'heterogeneous_dirichlet', 'hydrostatic', 'zero_gradient', 'conductance', 'seepage']
rate_types_allowed = ['mass_rate', 'volumetric_rate', 'scaled_volumetric_rate']
flux_types_allowed = ['dirichlet', 'neumann','mass_rate', 'hydrostatic, conductance',
		      'zero_gradient', 'production_well', 'seepage', 'volumetric',
		      'volumetric_rate', 'equilibrium']
temperature_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient']
concentration_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient']
saturation_types_allowed = ['dirichlet']
enthalpy_types_allowed = ['dirichlet', 'hydrostatic', 'zero_gradient']

# transport_condition - allowed strings
transport_condition_types_allowed = ['dirichlet', 'dirichlet_zero_gradient', 'equilibrium', 
				     'neumann', 'mole', 'mole_rate', 'zero_gradient']

cards = ['co2_database','uniform_velocity','simulation','regression','checkpoint','restart', 'dataset','chemistry','grid',
                'timestepper', 'material_property','time','linear_solver','newton_solver',
                'output','fluid_property','saturation_function', 'characteristic_curves','region','observation',
                'flow_condition','transport_condition','initial_condition',
                'boundary_condition','source_sink','strata','constraint']

headers = ['co2 database path','uniform velocity','simulation','regression','checkpoint','restart', 'dataset', 'chemistry','grid',
	   'time stepping','material properties','time','linear solver','newton solver','output',
	   'fluid properties','saturation functions', 'characteristic curves', 'regions','observation','flow conditions',
	   'transport conditions','initial condition','boundary conditions','source sink',
	   'stratigraphy couplers','constraints']


headers = dict(zip(cards,headers))

buildWarnings = []

def _buildWarnings(s):
	global buildWarnings
	buildWarnings.append(s)
	
class puniform_velocity(object):
	""" Class for specifiying uniform velocity with transport. Optional
	    with transport problem when not coupling with any flow mode. 
	    If not specified, assumes diffusion transport only.
	
	:param value_list: List of variables of uniform_velocity.
	 First 3 variables are vlx, vly, vlz in unit [m/s]. 4th variable specifies unit.
	  e.g., [14.4e0, 0.e0, 0.e0, 'm/yr']
	:type value_list: [float,float,float,str]
	"""
	
	def __init__(self, value_list=[]):
		self._value_list = value_list
		
	def _get_value_list(self): return self._value_list
	def _set_value_list(self,value): self._value_list = value
	value_list = property(_get_value_list, _set_value_list) #: (**)

class pmaterial(object):
	""" Class for defining a material property. 
	Multiple material property objects can be created.
	
	:param id: Unique identifier of material property.
	:type id: int
	:param name: Name of material property. e.g., 'soil1'.
	:type name: str
        :param characteristic_curves: Unique identifier of material characteristic curves
        :type characteristic_curves: str
	:param porosity: Porosity of material.
	:type porosity: float
	:param tortuosity: Tortuosity of material.
	:type tortuosity: float
	:param density: Rock density of material in kg/m^3.
	:type density: float
	:param specific_heat: Specific heat of material in J/kg/K.
	:type specific_heat: float
	:param cond_dry: Thermal dry conductivity of material in W/m/K.
	:type cond_dry: float
	:param cond_wet: Thermal wet conductivity of material in W/m/K.
	:type cond_wet: float
	:param saturation: Saturation function of material property. e.g., 'sf2'
	:type saturation: str
	:param permeability: Permeability of material property. Input is a list of 3 floats. Uses diagonal permeability in unit order: k_xx [m^2], k_yy [m^2], k_zz [m^2]. e.g., [1.e-15,1.e-15,1.e-17].
	:type permeability: [float]*3
	"""

	# definitions are put on one line to work better with rst/latex/sphinx.
	def __init__(self, id=None, name='', characteristic_curves = '', porosity=None, tortuosity=None, density=None,
		     specific_heat=None, cond_dry=None, cond_wet=None, saturation='', permeability=[],permeability_power='',permeability_critical_porosity='',permeability_min_scale_factor=''):
		self._id = id
		self._name = name
		self._characteristic_curves = characteristic_curves
                self._porosity = porosity
		self._tortuosity = tortuosity
		self._density = density
		self._specific_heat = specific_heat
		self._cond_dry = cond_dry
		self._cond_wet = cond_wet
		self._saturation = saturation
		self._permeability = permeability
		self._permeability_power = permeability_power
		self._permeability_critical_porosity = permeability_critical_porosity
		self._permeability_min_scale_factor = permeability_min_scale_factor

	def _get_id(self): return self._id
	def _set_id(self,value): self._id = value
	id = property(_get_id, _set_id) #: (**)
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name) #: (**)
	def _get_porosity(self): return self._porosity
	def _set_porosity(self,value): self._porosity = value
        porosity = property(_get_porosity, _set_porosity) #: (**)
        def _get_characteristic_curves(self): return self._characteristic_curves
	def _set_characteristic_curves(self,value): self._characteristic_curves = value
	characteristic_curves = property(_get_characteristic_curves, _set_characteristic_curves) #: (**)
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
	def _get_permeability_power(self): return self._permeability_power
	def _set_permeability_power(self,value): self._permeability_power = value
	permeability_power = property(_get_permeability_power, _set_permeability_power) #: (**)
	def _get_permeability_critical_porosity(self): return self._permeability_critical_porosity
	def _set_permeability_critical_porosity(self,value): self._permeability_critical_porosity = value
	permeability_critical_porosity = property(_get_permeability_critical_porosity, _set_permeability_critical_porosity) #: (**)
	def _get_permeability_min_scale_factor(self): return self._permeability_min_scale_factor
	def _set_permeability_min_scale_factor(self,value): self._permeability_min_scale_factor = value
	permeability_min_scale_factor = property(_get_permeability_min_scale_factor, _set_permeability_min_scale_factor) #: (**)

class ptime(object):
	""" Class for time. Used to specify final time of simulation, 
	   initial timestep size, maximum timestep size (throughout the 
	   simulation or a particular instant of time). Time values and 
	   units need to be specified. Acceptable time units are: (s, m, h, d, mo, y). 
	
	:param tf: final tim. 1st variable is time value. 2nd variable specifies time unit. e.g., [0.25e0, 'y']
	:type tf: [float, str]
	:param dti: delta (change) time initial a.k.a. initial timestep size. 1st variable is time value. 2nd variable specifies time unit. e.g., [0.25e0, 'y']
	:type dti: [float, str]
	:param dtf: delta (change) time final a.k.a. maximum timestep size. 1st variable is time value. 2nd variable specifies time unit. e.g., [50.e0, 'y']
	:type dtf: [float, str]
	:param dtf_list: delta (change) time starting at a given time instant.  Input is a list that can have multiple lists appended to it. e.g., time.dtf_list.append([1.e2, 's', 5.e3, 's'])
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
	""" Class for defining a grid. Used to define type, resolution and geometry of the gird
	
	:param type: Grid type. Valid entries include: 'structured', 'unstructured'. 
	:type type: str
	:param lower_bounds: Lower/Minimum 3D boundaries coordinates in order of x_min, y_min, z_min. Input is a list of 3 floats. e.g., [0.e0, 0.e0, 0.e0].
	:type lower_bounds: [float]*3
	:param upper_bounds: Upper/Maximum 3D boundaries coordinates in order of x_max, y_max, z_max. Input is a list of 3 floats. e.g., [321.e0, 1.e0, 51.e0].
	:type lower_bounds: [float]*3
	:param origin: Coordinates of grid origin. Optional. Input is a list of 3 floats. Default: [0.e0, 0.e0, 0.e0].
	:type origin: [float]*3
	:param nxyz: Number of grid cells in x,y,z directions. Only works with type='structured'. Input is a list of 3 floats. e.g., [107, 1, 51].
	:type nxyz: [float]*3
	:param dx: Specifies grid spacing of structured cartesian grid in the x-direction. e.g., [0.1, 0.2, 0.3, 0.4, 1, 1, 1, 1]. 
	:type dx: [float]
	:param dy: Specifies grid spacing of structured cartesian grid in the y-direction. 
	:type dy: [float]
	:param dz: Specifies grid spacing of structured cartesian grid in the z-direction
	:type dz: [float]
	:param gravity: Specifies gravity vector in m/s^2. Input is a list of 3 floats. 
	:type gravity: [float]*3
	:param filename: Specify name of file containing grid information. Only works with type='unstructured'.
	:type filename: str
	"""

	# definitions are put on one line to work better with rst/latex/sphinx.
	def __init__(self, type='structured', lower_bounds=[0.0,0.0,0.0], upper_bounds=[1.0,1.0,1.0],
		     origin=[],nxyz=[10,10,10], dx=[],dy=[],dz=[], gravity=[], filename=''):
		self._type = type
		self._lower_bounds = lower_bounds
		self._upper_bounds = upper_bounds
		self._origin = origin
		self._nxyz = nxyz
		self._dx = dx
		self._dy = dy
		self._dz = dz
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
	def _get_dx(self): return self._dx
	def _set_dx(self,value): self._dx = value
	def _get_dy(self): return self._dy
	def _set_dy(self,value): self._dy = value
	def _get_dz(self): return self._dz
	def _set_dz(self,value): self._dz = value
	dx = property(_get_dx, _set_dx) #: (**)
	dy = property(_get_dy, _set_dy) #: (**)
	dz = property(_get_dz, _set_dz) #: (**)
	def _get_gravity(self): return self._gravity
	def _set_gravity(self,value): self._gravity = value
	gravity = property(_get_gravity, _set_gravity) #: (**)
	def _get_filename(self): return self._filename
	def _set_filename(self,value): self._filename = value
	filename = property(_get_filename, _set_filename) #: (**)


class psimulation(object):
	""" Class for specifying simulation type and simulation mode. 

	:param simulation_type: Specify simulation type. Options include: 'surface','subsurface.
	:type simulation_type: str
	:param subsurface_flow: Specify the process model. 
	:type subsurface_flow: str
	:param subsurface_transport: Specify the process model. 
	:type subsurface_transport: str
	:param mode: Specify the mode for the subsurface flow model
	:type mode: str
	:param flowtran_coupling: Specify the type for the flow transport coupling 
	:type mode: str
	"""
	def __init__(self, simulation_type='', subsurface_flow='', subsurface_transport='', mode='',flowtran_coupling=''):
		self._simulation_type = simulation_type
		self._subsurface_flow = subsurface_flow
		self._subsurface_transport = subsurface_transport
		self._flowtran_coupling = flowtran_coupling
		self._mode = mode

	def _get_simulation_type(self): return self._simulation_type
	def _set_simulation_type(self,value): self._simulation_type = value
	simulation_type = property(_get_simulation_type, _set_simulation_type)
	def _get_subsurface_flow(self): return self._subsurface_flow
	def _set_subsurface_flow(self,value): self._subsurface_flow = value
	subsurface_flow = property(_get_subsurface_flow, _set_subsurface_flow)
	def _get_subsurface_transport(self): return self._subsurface_transport
	def _set_subsurface_transport(self,value): self._subsurface_transport = value
	subsurface_transport = property(_get_subsurface_transport, _set_subsurface_transport)
	def _get_mode(self): return self._mode
	def _set_mode(self,value): self._mode = value
	def _get_flowtran_coupling(self): return self._flowtran_coupling
	def _set_flowtran_coupling(self,value): self._flowtran_coupling = value
	flowtran_coupling = property(_get_flowtran_coupling, _set_flowtran_coupling)

class pregression(object):
	""" Class for specifying regression details. 

	:param cells: Specify cells for regression. 
	:type cells: list of int 
	:param cells_per_process: Specify the number cells per process. 
	:type cells_per_process: int 
	"""
	def __init__(self, cells=[], cells_per_process=''): 
		self._cells = cells 
		self._cells_per_process = cells_per_process 

	def _get_cells(self): return self._cells
	def _set_cells(self,value): self._cells = value
	cells = property(_get_cells, _set_cells)
	def _get_cells_per_process(self): return self._cells_per_process
	def _set_cells_per_process(self,value): self._cells_per_process = value
	cells_per_process = property(_get_cells_per_process, _set_cells_per_process)

class ptimestepper(object):
	""" Class for controling time stepping.
        
        :param ts_mode: FLOW or TRAN mode
	:type ts_mode: string	
	:param ts_acceleration: Integer for time step acceleration ramp. 
	:type ts_acceleration: int
	:param num_steps_after_cut: Number of time steps after a time step cut that the time step size is held constant. 
	:type num_steps_after_cut: int
	:param max_steps: Maximum time step after which the simulation will be terminated. 
	:type max_steps: int
	:param max_ts_cuts: Maximum number of consecutive time step cuts before the simulation is terminated.
	:type max_ts_cuts: int
	:param cfl_limiter: CFL number for transport. 
	:type cfl_limiter: float
	:param initialize_to_steady_state: Boolean flag to initialize a simulation to steady state 
	:type initialize_to_steady_state: bool - True or False
	:param run_as_steady_state: Boolean flag to run a simulation to steady state 
	:type run_as_steady_state: bool - True or False
	:param max_pressure_change: Maximum change in pressure for a time step. Default: 5.d4 Pa.
	:type max_pressure_change: float
	:param max_temperature_change: Maximum change in temperature for a time step. Default: 5 C.
	:type max_temperature_change: float
	:param max_concentration_change: Maximum change in pressure for a time step. Default: 1. mol/L.
	:type max_concentration_change: float
	:param max_saturation_change: Maximum change in saturation for a time step. Default: 0.5.
	:type max_saturation_change: float
	"""
	
	# definitions are put on one line to work better with rst/latex/sphinx.
	def __init__(self, ts_mode='flow',ts_acceleration=None, num_steps_after_cut=None, max_steps=None,
		     max_ts_cuts=None, cfl_limiter=None, initialize_to_steady_state=False,
		     run_as_steady_state=False, max_pressure_change=None, max_temperature_change=None,
		     max_concentration_change=None, max_saturation_change=None):
		self._ts_mode = ts_mode
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

	def _get_ts_mode(self): return self._ts_mode
	def _set_ts_mode(self,value): self._ts_mode= value
	ts_mode= property(_get_ts_mode, _set_ts_mode)
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
	""" Class for specifying linear solver. Multiple linear solver objects
	    can be created one for flow and one for transport.
	
	:param name: Specify name of the physics for which the linear solver is
	 being defined. Options include: 'tran', 'transport','flow'.
	:type name: str
	:param solver: Specify solver type: Options include: 'solver', 'krylov_type', 'krylov', 'ksp', 'ksp_type'
	:type solver: str
	:param preconditioner: Specify preconditioner type: Options include: 'ilu' 
	:type solver: str
	"""
	
	def __init__(self, name='', solver='',preconditioner=''):
		self._name = name	# TRAN, TRANSPORT / FLOW
		self._solver = solver	# Solver Type
		self._preconditioner = preconditioner
		
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name)
	def _get_solver(self): return self._solver
	def _set_solver(self,value): self._solver = value
	solver = property(_get_solver, _set_solver)
	def _get_preconditioner(self): return self._preconditioner
	def _set_preconditioner(self,value): self._preconditioner= value
	preconditioner = property(_get_preconditioner, _set_preconditioner)

class pnsolver(object):
	""" Class for newton solver card. Multiple newton solver objects 
	    can be created, one for flow and one for transport.
	
	:param name: Specify newton solver to use: Options include: 'tran',
	 'transport', 'tran_solver', 'flow_solver'. Default: 'flow_solver'
	:type name: str
	:param atol: Absolute tolerance.
	:type atol: float
	:param rtol: Relative tolerance w.r.t previous iteration. 
	:type rtol: float
	:param stol: Relative tolerance of the update w.r.t previous iteration. 
	:type stol: float
	:param dtol: Divergence tolerance.
	:type dtol: float
	:param itol: Tolerance compared to infinity norm. 
	:type itol: float
	:param max_it: Cuts time step if the number of iterations exceed this value.
	:type max_it: int
	:param max_f: Maximum function evaluations (useful with linesearch methods.)
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
	"""Class for dumping simulation output.
	Acceptable time units (units of measurements) are: 's', 'min', 'h', 'd', 'w', 'mo', 'y'.
	
	:param time_list: List of time values. 1st variable specifies time unit to be used. Remaining variable(s) are floats.
	:type time_list: [str, float*]
	:param print_column_ids: Flag to indicate whether to print column numbers in observation
	 and mass balance output files. Default: False
	:type print_column_ids: bool - True or False
	:param screen_output: Turn the screen output on/off.
	:type screen_periodic: bool 
	:param screen_periodic: Print to screen every <integer> time steps.
	:type screen_periodic: int
	:param periodic_time: 1st variable is value, 2nd variable is time unit.
	:type periodic_time: [float, str]
	:param periodic_timestep: 1st variable is value, 2nd variable is time unit.
	:type periodic_timestep: [float, str]
	:param periodic_observation_time: Output the results at observation points and mass balance
	 output at specified output time. 1st variable is value, 2nd variable is time unit.
	:type periodic_observation_time: [float, str]
	:param periodic_observation_timestep: Outputs the results at observation points and mass 
	balance output at specified time steps.
	:type periodic_observation_timestep: int
	:param format_list: Specify the file format for time snapshot of the simulation in
	 time file type. Input is a list of strings. Multiple formats can be specified. 
	 File format options include: 'TECPLOT BLOCK' - TecPlot block format, 'TECPLOT POINT' - 
	 TecPlot point format (requires a single processor), 'HDF5' - produces single HDF5 file
	 and xml for unstructured grids,  'HDF5 MULTIPLE_FILES' - produces a separate HDF5 file
	 at each output time, 'VTK' - VTK format.
	:type format_list: [str]
	:param velocities: Turn velocity output on/off. 
	:type velocities: bool - True or False
	:param velocity_at_center: Turn velocity output on/off. 
	:type velocity_at_center: bool - True or False
	:param mass_balance: Flag to indicate whether to output the mass balance of the system. 
	:type mass_balance: bool - True or False
	:param variables_list: List of variables to be printed in the output file
	:type variables_list: [str]
	"""
	
	# definitions are put on one line to work better with rst/latex/sphinx.
	def __init__(self, time_list=[], print_column_ids=False, screen_periodic=None, 
		     screen_output=True, periodic_time=[],periodic_timestep=[],
		     periodic_observation_time=[], periodic_observation_timestep=None, 
		     format_list=[], permeability=False, porosity=False, velocities=False,velocity_at_center=False,
		      mass_balance=False, variables_list = []):
		self._time_list = time_list
		self._print_column_ids = print_column_ids
		self._screen_output = screen_output # Bool
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
		self._variables_list = variables_list
		self._velocity_at_center = velocity_at_center
	
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
	def _get_screen_output(self): return self._screen_output
	def _set_screen_output(self,value): self._screen_output = value
	screen_output = property(_get_screen_output, _set_screen_output)
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
	def _get_velocities(self): return self._velocities
	def _set_velocities(self,value): self._velocities = value
	velocities = property(_get_velocities, _set_velocities)	
	def _get_velocity_at_center(self): return self._velocity_at_center
	def _set_velocity_at_center(self,value): self._velocity_at_center= value
	velocity_at_center= property(_get_velocity_at_center, _set_velocity_at_center)	
	def _get_variables_list(self): return self._variables_list
	def _set_variables_list(self,value): self._variables_list = value
	variables_list = property(_get_variables_list, _set_variables_list)

class pfluid(object):
	"""Class for specifying fluid properties.
	
	:param diffusion_coefficient: Unit of measurement is [m^2/s]. Default: 1e-09
	:type diffusion_coefficient: float
	"""
	
	def __init__(self, diffusion_coefficient=1.e-9):
		self._diffusion_coefficient = diffusion_coefficient
		
	def _get_diffusion_coefficient(self): return self._diffusion_coefficient
	def _set_diffusion_coefficient(self,value): self._diffusion_coefficient = value
	diffusion_coefficient = property(_get_diffusion_coefficient, _set_diffusion_coefficient)
	
class psaturation(object):
	"""Class for specifying saturation functions.
	
	:param name: Saturation function name. e.g., 'sf2'
	:type name: str
	:param permeability_function_type: Options include: 'VAN_GENUCHTEN', 'MUALEM', 'BURDINE', 'NMT_EXP', 'PRUESS_1'.
	:type permeability_function_type: str
	:param saturation_function_type: Options include: 'VAN_GENUCHTEN', 'BROOKS_COREY',
	 'THOMEER_COREY', 'NMT_EXP', 'PRUESS_1'.
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
	def __init__(self, name='', permeability_function_type=None, saturation_function_type=None,
		     residual_saturation=None, residual_saturation_liquid=None,
		     residual_saturation_gas=None, a_lambda=None, alpha=None, 
		     max_capillary_pressure=None, betac=None, power=None):
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

class ppoint(object):
	""" Class for a point. 

	:param name: point name
	:type name: str
	:param coordinate: Coordinate of the point
	:type coordinate: [float]*3
	"""
	def __init__(self,name='',coordinate=[0.0,0.0,0.0]):
		self._name = name.lower()
		self._coordinate = coordinate
	def _set_name(self,value): self._name = value.lower()
	def _get_name(self): return self._name
	name = property(_get_name, _set_name)
	def _get_coordinate(self): return self._coordinate
	def _set_coordinate(self,value): self._coordinate=value
	coordinate = property(_get_coordinate, _set_coordinate)

class pcharacteristic_curves(object):
	"""Class for specifying characteristic curves. This card is used only in GENERAL mode; the SATURATION_FUNCTION card should be used in RICHARDS mode.
	
	:param name: Characteristic curve name. e.g., 'cc1'
	:param saturation_function_type: Options include: 'VAN_GENUCHTEN', 'BROOKS_COREY'.
	:type saturation_function_type: str
        :param sf_alpha: Pa^-1 
        :type sf_alpha: float
        :param sf_m: Van Genutchen m
        :type sf_m: float
        :param sf_lambda: lambda: Brooks Corey lambda
        :type sf_lambda: float
        :param sf_liquid_residual_saturation: Residual saturation for liquid phase
        :type sf_liquid_residual_saturation: float 
        :param sf_gas_residual_saturation: Residual saturation for gas phase
        :type sf_gas_residual_saturation: float
        :param max_capillary_pressure: Pa 
        :type max_capillary_pressure: float
        :param smooth: Recommended for BROOKS_COREY
        :type smooth: No value, just a flag. Input 1 to turn flag on
	:param power: Placeholder. Currently not used
	:type power: float
	:param default: sets up dummy saturation and permeability functions for saturated single phase flow
	:type default: No value, just a flag. Input 1 to turn flag on
	:param liquid_permeability_function_type: Options include: 'MAULEM', 'BURDINE'.
	:type liquid_permeability_function_type: str
        :param lpf_m: Van Genutchen m
        :type lpf_m: float
        :param lpf_lambda: lambda: Brooks Corey lambda
        :type lpf_lambda: float
        :param lpf_liquid_residual_saturation: Residual saturation for liquid phase
        :type lpf_liquid_residual_saturation: float  
	:param gas_permeability_function_type: Options include: 'MAULEM_VG_GAS', 'BURDINE_BC_GAS'.
	:type gas_permeability_function_type: str
        :param gpf_m: Van Genutchen m
        :type gpf_m: float
        :param gpf_lambda: lambda: Brooks Corey lambda
        :type gpf_lambda: float
        :param gpf_liquid_residual_saturation: Residual saturation for liquid phase
        :type gpf_liquid_residual_saturation: float 
        :param gf_gas_residual_saturation: Residual saturation for gas phase
        :type gf_gas_residual_saturation: float
	"""
	
	# definitions are put on one line to work better with rst/latex/sphinx.
	def __init__(self, name='', saturation_function_type=None, sf_alpha=None, sf_m=None, sf_lambda=None, sf_liquid_residual_saturation=None, sf_gas_residual_saturation=None, max_capillary_pressure=None, smooth='', power=None, default = None,  liquid_permeability_function_type=None, lpf_m=None, lpf_lambda=None, lpf_liquid_residual_saturation=None, gas_permeability_function_type=None, gpf_m=None, gpf_lambda=None, gpf_liquid_residual_saturation=None, gpf_gas_residual_saturation=None):
		self._name = name
		self._saturation_function_type = saturation_function_type
                self._sf_alpha = sf_alpha
                self._sf_m = sf_m
                self._sf_lambda = sf_lambda
		self._sf_liquid_residual_saturation = sf_liquid_residual_saturation # float
		self._sf_gas_residual_saturation = sf_gas_residual_saturation # float
		self._max_capillary_pressure = max_capillary_pressure
		self._smooth = smooth 
		self._power = power
		self._default = default
		self._liquid_permeability_function_type = liquid_permeability_function_type
		self._lpf_m = lpf_m
		self._lpf_lambda = lpf_lambda
		self._lpf_liquid_residual_saturation = lpf_liquid_residual_saturation
		self._gas_permeability_function_type =  gas_permeability_function_type
		self._gpf_m = gpf_m
		self._gpf_lambda = gpf_lambda
		self._gpf_liquid_residual_saturation = gpf_liquid_residual_saturation
		self._gpf_gas_residual_saturation = gpf_gas_residual_saturation	
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name)
	def _get_saturation_function_type(self): return self._saturation_function_type
	def _set_saturation_function_type(self,value): self._saturation_function_type = value
	saturation_function_type = property(_get_saturation_function_type, _set_saturation_function_type)	
	def _get_sf_alpha(self): return self._sf_alpha
	def _set_sf_alpha(self,value): self._sf_alpha = value
	sf_alpha = property(_get_sf_alpha, _set_sf_alpha)
	def _get_sf_m(self): return self._sf_m
	def _set_sf_m(self,value): self._sf_m = value
	sf_m = property(_get_sf_m, _set_sf_m)
	def _get_sf_lambda(self): return self._sf_lambda
	def _set_sf_lambda(self,value): self._sf_lambda = value
	sf_lambda = property(_get_sf_lambda, _set_sf_lambda)
	def _get_sf_liquid_residual_saturation(self): return self._sf_liquid_residual_saturation
	def _set_sf_liquid_residual_saturation(self,value): self._sf_liquid_residual_saturation = value
	sf_liquid_residual_saturation = property(_get_sf_liquid_residual_saturation, _set_sf_liquid_residual_saturation)
	def _get_sf_gas_residual_saturation(self): return self._sf_gas_residual_saturation
	def _set_sf_gas_residual_saturation(self,value): self._sf_gas_residual_saturation = value
	sf_gas_residual_saturation = property(_get_sf_gas_residual_saturation, _set_sf_gas_residual_saturation)
	def _get_max_capillary_pressure(self): return self._max_capillary_pressure
	def _set_max_capillary_pressure(self,value): self._max_capillary_pressure = value
	max_capillary_pressure = property(_get_max_capillary_pressure, _set_max_capillary_pressure)
	def _get_smooth(self): return self._smooth
	def _set_smooth(self,value): self._smooth = value
	smooth = property(_get_smooth, _set_smooth)
	def _get_power(self): return self._power
	def _set_power(self,value): self._power = value
	power = property(_get_power, _set_power)
	def _get_default(self): return self._default
	def _set_default(self,value): self._default = value
	default = property(_get_default, _set_default)
	def _get_liquid_permeability_function_type(self): return self._liquid_permeability_function_type
	def _set_liquid_permeability_function_type(self,value): self._liquid_permeability_function_type = value
	liquid_permeability_function_type = property(_get_liquid_permeability_function_type, _set_liquid_permeability_function_type)	
	def _get_lpf_m(self): return self._lpf_m
	def _set_lpf_m(self,value): self._lpf_m = value
	lpf_m = property(_get_lpf_m, _set_lpf_m)
	def _get_lpf_lambda(self): return self._lpf_lambda
	def _set_lpf_lambda(self,value): self._lpf_lambda = value
	lpf_lambda = property(_get_lpf_lambda, _set_lpf_lambda)
	def _get_lpf_liquid_residual_saturation(self): return self._lpf_liquid_residual_saturation
	def _set_lpf_liquid_residual_saturation(self,value): self._lpf_liquid_residual_saturation = value
	lpf_liquid_residual_saturation = property(_get_lpf_liquid_residual_saturation, _set_lpf_liquid_residual_saturation)
	def _get_gas_permeability_function_type(self): return self._gas_permeability_function_type
	def _set_gas_permeability_function_type(self,value): self._gas_permeability_function_type = value
	gas_permeability_function_type = property(_get_gas_permeability_function_type, _set_gas_permeability_function_type)	
	def _get_gpf_m(self): return self._gpf_m
	def _set_gpf_m(self,value): self._gpf_m = value
	gpf_m = property(_get_gpf_m, _set_gpf_m)
	def _get_gpf_lambda(self): return self._gpf_lambda
	def _set_gpf_lambda(self,value): self._gpf_lambda = value
	gpf_lambda = property(_get_gpf_lambda, _set_gpf_lambda)
	def _get_gpf_liquid_residual_saturation(self): return self._gpf_liquid_residual_saturation
	def _set_gpf_liquid_residual_saturation(self,value): self._gpf_liquid_residual_saturation = value
	gpf_liquid_residual_saturation = property(_get_gpf_liquid_residual_saturation, _set_gpf_liquid_residual_saturation)
	def _get_gpf_gas_residual_saturation(self): return self._gpf_gas_residual_saturation
	def _set_gpf_gas_residual_saturation(self,value): self._gpf_gas_residual_saturation = value
	gpf_gas_residual_saturation = property(_get_gpf_gas_residual_saturation, _set_gpf_gas_residual_saturation)
#endif /* VERSION1 */

class pregion(object):
	"""Class for specifying a PFLOTRAN region. Multiple region objects can be created.

	:param name: Region name. 
	:type name: str
	:param coordinates_lower: Lower/minimum 3D coordinates for defining a volumetric, 
	 planar, or point region between two points in space in order of x1, y1, z1. e.g., [0.e0, 0.e0, 0.e0]
	:type coordinates_lower: [float]*3
	:param coordinates_upper: Upper/maximum 3D coordinates for defining a volumetric, 
	 planar, or point region between two points in space in order of x2, y2, z2. e.g., [321.e0, 1.e0,  51.e0]
	:type coordinates_upper: [float]*3
	:param face: Defines the face of the grid cell to which boundary conditions are connected.
	 Options include: 'west', 'east', 'north', 'south', 'bottom', 'top'. (structured grids only).
	:type face: str
	"""
	
	def __init__(self,name='',coordinates_lower=[0.0,0.0,0.0],coordinates_upper=[0.0,0.0,0.0],
			face=None):
		self._name = name.lower()
		self._coordinates_lower = coordinates_lower	# 3D coordinates
		self._coordinates_upper = coordinates_upper	# 3D coordinates
		self._face = face
		self._point_list = []
		
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
	def _get_point_list(self): return self._point_list
	def _set_point_list(self,value): self._point_list= value
	point_list= property(_get_point_list, _set_point_list)

class pobservation(object):
	"""Class for specifying an observation region. Multiple observation objects may be added.
	Currently, only region is supported in PyFLOTRAN.
	
	:param region: Defines the name of the region to which the observation object is linked.
	:type region: str
	"""
	
	def __init__(self,region=None):
		self._region = region
		
	def _get_region(self): return self._region
	def _set_region(self,value): self._region = value
	region = property(_get_region, _set_region) 
	
class pflow(object):
	"""Class for specifying a PFLOTRAN flow condition. There can be multiple flow condition objects.
	
	:param name: Name of the flow condition.  
	:type name: str
	:param units_list: Not currently supported.
	:type units_list: [str]
	:param iphase: 
	:type iphase: int
	:param sync_timestep_with_update: Flag that indicates whether to use sync_timestep_with_update. Default: False.
	:type sync_timestep_with_update: bool - True or False
	:param data_unit_type: List alternative, do not use with non-list alternative attributes/parameters.
	:type data_unit_type: str
	:param datum: Input is either a list of [d_dx, d_dy, d_dz] OR a 'file_name' 
	 with a list of [d_dx, d_dy, d_dz]. Choose one format type or the other, not both.
	 If both are used, then only the file name will be written to the input deck.
	:type datum: Multiple [float, float, float] or str.
	:param datum_type: file or dataset
	:type datum_type: str
	:param varlist: Input is a list of pflow_variable objects. Sub-class of pflow.
	 It is recommended to use dat.add(obj=pflow_variable) for easy appending. 
	 Use dat.add(index='pflow_variable.name' or dat.add(index=pflow_variable) to 
	 specify pflow object to add pflow_variable to. If no pflow object is specified, 
	 pflow_variable will be appended to the last pflow object appended to pdata.
	 E.g., dat.add(variable, 'initial') if variable = pflow_variable and pflow.name='initial'.
	:type varlist: [pflow_variable]

	"""
	
	def __init__(self,name='',units_list=None,
			iphase=None,sync_timestep_with_update=False,
			datum=[],datum_type='',
			varlist=[]):
		self._name = name.lower()	# Include initial, top, source
		self._units_list = units_list	# Specify type of units to display such as
						# time,length,rate,pressure,velocity, temperature,
						# concentration, and enthalpy.
						# May be used to determine each variable unit
		self._iphase = iphase			# Holds 1 int
		self._sync_timestep_with_update = sync_timestep_with_update	# Boolean
		self._datum = datum	# x, y, z, and a file name. [float,float,float,str]
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
	def _get_datum(self): return self._datum
	def _set_datum(self,value): self._datum = value
	datum = property(_get_datum, _set_datum)
	def _get_datum_type(self): return self._datum_type
	def _set_datum_type(self,value): self._datum_type = value
	datum_type = property(_get_datum_type, _set_datum_type)
	def _get_varlist(self): return self._varlist
	def _set_varlist(self,value): self._varlist = value
	varlist = property(_get_varlist, _set_varlist)
	
	# Code below is an attempt to change the way sub-classes are added.
	# it's not necessary. (Attempting to make it possible to do a flow.add(variable)
	# instead of dat.add(variable). Current way of specifying which flow object to
	# add to is dat.add(variable,flow) (e.g. Rate List instead of Rate)
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
	"""Sub-class of pflow for each kind of variable (includes type and value) such as
 	   pressure, temperature, etc. There can be multiple pflow_variable objects appended to a single pflow object.
	
	:param name: Indicates name of the flow variable. Options include: ['PRESSURE', 'RATE', 'FLUX', 'TEMPERATURE', 'CONCENTRATION', 'SATURATION', 'ENTHALPY'].
	:type name: str
	:param type: Indicates type that is associated with name under keyword TYPE.
	 Options for PRESSURE include: 'dirichlet', 'hydrostatic', 'zero_gradient', 'conductance',
	 'seepage'. Options for RATE include: 'mass_rate', 'volumetric_rate', 'scaled_volumetric_rate'.
	 Options for FLUX include: 'dirichlet', 'neumann, mass_rate', 'hydrostatic, conductance', 
	 'zero_gradient', 'production_well', 'seepage', 'volumetric', 'volumetric_rate', 'equilibrium'. 
  	 Options for TEMPERATURE include: 'dirichlet', 'hydrostatic', 'zero_gradient'. 
	 Options for CONCENTRATION include: 'dirichlet', 'hydrostatic', 'zero_gradient'.
 	 Options for SATURATION include: 'dirichlet'. Options for ENTHALPY include: 'dirichlet', 'hydrostatic', 'zero_gradient'
	:type type: str
	:param valuelist: Provide one or two values associated with a single Non-list alternative, do not use with list alternative. The 2nd float is optional and is needed for multiphase simulations.
	:type valuelist: [float, float]
	:param unit: Non-list alternative, do not use with list alternative. Specify unit of measurement.
	:type unit: str
	:param time_unit_type: List alternative, do not use with non-list alternative attributes/parameters. 
	:type time_unit_type: str
	:param list: List alternative, do not use with non-list alternative attributes/parameters. Input is a list of pflow_variable_list objects. Sub-class of pflow_variable. The add function currently does not support adding pflow_variable_list to pflow_variable objects. Appending to can be done manually. e.g., variable.list.append(var_list) if var_list=pflow_variable_list.
	:type list: [pflow_variable_list]
	"""
	
	def __init__(self,name='',type=None, valuelist=[], unit='',
		     time_unit_type='', data_unit_type='', list=[]):
		self._name = name.lower()# Pressure,temp., concen.,enthalpy...(String)
		self._type = type	# hydrostatic, zero_gradient, dirichlet ...(String)
		
		# The Following attributes are a stand alone single list w/out lists
		# (e.g., Rate instead of Rate List)
		self._valuelist = valuelist	# Holds 2 floats - 2nd is optional
		self._unit = unit	# Possible to overide Parent class? - sorda?
		
		# Following attributes are used with lists (eg. Rate Lists instead of Rate)
		self._time_unit_type = time_unit_type # e.g., 'y'
		self._data_unit_type = data_unit_type # e.g., 'kg/s'
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
	Used for pflow_variables that are lists (as function of time) instead of a single value. Each of these list objects can hold multiple lines (from a Python input file) with each line holding one time_unit_value and a data_unit_value_list that can hold multiple values.
	
	:param time_unit_value: 
	:type time_unit_value: float
	:param data_unit_value_list: 
	:type data_unit_value_list: [float]

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
	"""Class for initial condition - a coupler between regions and initial flow and transport conditions.
	
	:param flow: Specify flow condition name
	:type flow: str
	:param transport: Specify transport condition name
	:type transport: str
	:param region: Specify region to apply the above specified flow and transport conditions as initial conditions.
	:type region: str
	"""
	
	def __init__(self,flow=None,transport=None,region=None):
		self._flow = flow	# Flow Condition (e.g., initial)
		self._transport = transport
		self._region = region	# Define region (e.g., west, east, well)
		
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
	"""Class for boundary conditions - performs coupling between a region and a flow/transport condition which are to be set as boundary conditions to that region.
	Multiple objects can be created.
	
	:param name: Name of boundary condition. (e.g., west, east)
	:type name: str
	:param flow: Defines the name of the flow condition to be linked to this boundary condition.
	:type flow: str
	:param transport: Defines the name of the transport condition to be linked to this boundary condition
	:type transport: str
	:param region: Defines the name of the region to which the conditions are linked
	:type region: str
	"""
	
	def __init__(self,name='',flow='',transport='',region=''):
		self._name = name	# Name of boundary condition. (e.g., west, east)
		self._flow = flow	# Flow Condition (e.g., initial)
		self._transport = transport	# Transport Condition (e.g., river_chemistry)
		self._region = region	# Define region (e.g., west, east, well)
		
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
	"""Class for specifying source sink - this is also a condition coupler that links a region to the source sink condition.
	
	:param flow: Name of the flow condition the source/sink term is applied to.
	:type flow: str
	:param region: Name of the region the source/sink term is applied to.
	:type region: str
	"""
	
	def __init__(self,flow=None,region=None):
		self._flow = flow	# Flow Condition (e.g., initial)
		self._region = region	# Define region (e.g., west, east, well)
		
	def _get_flow(self): return self._flow
	def _set_flow(self,value): self._flow = value
	flow = property(_get_flow, _set_flow)
	def _get_region(self): return self._region
	def _set_region(self,value): self._region = value
	region = property(_get_region, _set_region)
	
class pstrata(object):
	"""Class for specifying stratigraphy coupler. Multiple stratigraphy couplers can be created. Couples material properties with a region. 
	
	:param region: Name of the material property to be associated with a region.
	:type region: str
	:param material: Name of region associated with a material property.
	:type material: str
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
	"""Class for specifying checkpoint options. 
	
	:param frequency: Checkpoint dump frequency.
	:type frequency: int
	:param overwrite: Intended to be used for the PFLOTRAN keyword OVERWRITE_RESTART_FLOW_PARAMS.
	:type overwrite: bool - True or False
	"""
	
	def __init__(self, frequency=None, overwrite=False):
		self._frequency = frequency # int
		self._overwrite = overwrite # Intended for OVERWRITE_RESTART_FLOW_PARAMS, incomplete, uncertain how to write it.

	def _get_frequency(self): return self._frequency
	def _set_frequency(self,value): self._frequency = value
	frequency = property(_get_frequency, _set_frequency)
	def _get_overwrite(self): return self._overwrite
	def _set_overwrite(self,value): self._overwrite = value
	overwrite = property(_get_overwrite, _set_overwrite)
	
class prestart(object):
	"""Class for restarting a simulation.
	
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
	
class pdataset(object):
	"""Class for incorporating data within a model.

	:param dataset_name: Opens the card block with the name of the data set in the string. I name is not given the NAME entry is required.
	:type dataset_name: str
	:param dataset_mapped_name: Adds the MAPPED flag to the DATASET and allows for the dataset to be named.
	:type dataset_name: str
	:param name: Name of the data set if not included with DATASET card. Note: this string overwrites the name specified with DATASET
	:type name: str
	:param file_name: Name of the file containing the data
	:type file_name: str
	:param hdf5_dataset_name: Name of the group within the hdf5 file where the data resides
	:type hdf5_dataset_name: str
	:param map_hdf5_dataset_name: Name of the group within the hdf5 file where the map information for the data resides
	:type map_hdf5_dataset_name: str
	:param max_buffer_size: size of internal buffer for storing transient data
	:type max_buffer_size: float
	"""
	
	def __init__(self, dataset_name='', dataset_mapped_name='',name='', file_name='', hdf5_dataset_name='', map_hdf5_dataset_name='', max_buffer_size =''):
		
		self._dataset_name = dataset_name # name of dataset
		self._dataset_mapped_name = dataset_mapped_name
		self._name = name	# name of dataset (overwrites dataset_name)
		self._file_name = file_name	# name of file containing the data
		self._hdf5_dataset_name = hdf5_dataset_name	# name of hdf5 group
		self._map_hdf5_dataset_name = map_hdf5_dataset_name
		self._max_buffer_size = max_buffer_size

	def _get_dataset_name(self): return self._dataset_name
	def _set_dataset_name(self,value): self._dataset_name = value
	dataset_name = property(_get_dataset_name, _set_dataset_name)
	def _get_dataset_mapped_name(self): return self._dataset_mapped_name
	def _set_dataset_mapped_name(self,value): self._dataset_mapped_name = value
	dataset_mapped_name = property(_get_dataset_mapped_name, _set_dataset_mapped_name)
	def _get_name(self): return self._name
	def _set_name(self,value): self._name = value
	name = property(_get_name, _set_name)
	def _get_file_name(self): return self._file_name
	def _set_file_name(self,value): self._file_name = value
	file_name = property(_get_file_name, _set_file_name)	
	def _get_hdf5_dataset_name(self): return self._hdf5_dataset_name
	def _set_hdf5_dataset_name(self,value): self._hdf5_dataset_name = value
	hdf5_dataset_name = property(_get_hdf5_dataset_name, _set_hdf5_dataset_name)
	def _get_map_hdf5_dataset_name(self): return self._map_hdf5_dataset_name
	def _set_map_hdf5_dataset_name(self,value): self._map_hdf5_dataset_name = value
	map_hdf5_dataset_name = property(_get_map_hdf5_dataset_name, _set_map_hdf5_dataset_name)
	def _get_max_buffer_size(self): return self._max_buffer_size
	def _set_max_buffer_size(self,value): self._max_buffer_size = value
	max_buffer_size = property(_get_max_buffer_size, _set_max_buffer_size)

class pchemistry(object):
	"""Class for specifying chemistry.

 	:param pspecies_list: List of primary species that fully describe the chemical composition of the fluid. The set of primary species must form an independent set of species in terms of which all homogeneous aqueous equilibrium reactions can be expressed.
	:type pspecies_list: [str]
	:param sec_species_list: List of aqueous species in equilibrium with primary species.
	:type sec_species_list: [str]
	:param gas_species_list: List of gas species.
	:type gas_species_list: [str]
	:param minerals_list: List of mineral names.
	:type minerals_list: [str]
	:param m_kinetics_list: List of pchemistry_m_kinetic objects. Holds kinetics information about a specified mineral name. Works with add function so that m_kinetics_list does not need to be remembered. e.g., dat.add(mineral_kinetic).
	:type m_kinetics_list: [pchemistry_m_kinetic]
	:param log_formulation: 
	:type log_formulation: bool - True or False
	:param update_porosity: 
	:type update_porosity: bool - True or False
	:param update_permeability: 
	:type update_permeability: bool - True or False
	:param database: 
	:type database: str
	:param activity_coefficients: Options include: 'LAG', 'NEWTON', 'TIMESTEP', 'NEWTON_ITERATION'.
	:type activity_coefficients: str
	:param molal: 
	:type molal: bool - True or False
	:param output_list: To print secondary aqueous complex concentrations, either add the names of the secondary species of interest or the keyword 'SECONDARY_SPECIES' for all secondary species to the CHEMISTRY OUTPUT card. e.g., output_list = 'SECONDARY_SPECIES' or output_list = ['CO2(aq), 'PH']. By default, if ALL or MINERALS are listed under CHEMISTRY OUTPUT, the volume fractions and rates of kinetic minerals are printed. To print out the saturation indices of minerals listed under the MINERAL keyword, add the name of the mineral to the OUTPUT specification.
	:type output_list: [str]

	"""
	
	def __init__(self, pspecies_list=[], sec_species_list=[], gas_species_list=[],
		     minerals_list=[], m_kinetics_list=[], log_formulation=False,
		     database=None, activity_coefficients=None, molal=False, output_list=[],update_permeability=False,update_porosity=False ):
		self.pspecies_list = pspecies_list	# primary_species (eg. 'A(aq') - string
		self._sec_species_list = sec_species_list # Secondary_species (E.g. 'OH-' - string
		self._gas_species_list = gas_species_list # E.g. 'CO2(g)'
		self._minerals_list = minerals_list	# E.g. 'Calcite'
		self._m_kinetics_list = m_kinetics_list	# has pchemistry_m_kinetic assigned to it
		self._log_formulation = log_formulation
		self._update_permeability = update_permeability
		self._update_porosity = update_porosity
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
	def _get_update_permeability(self): return self._update_permeability
	def _set_update_permeability(self,value): self._update_permeability= value
	update_permeability= property(_get_update_permeability, _set_update_permeability)
	def _get_update_porosity(self): return self._update_porosity
	def _set_update_porosity(self,value): self._update_porosity= value
	update_porosity = property(_get_update_porosity, _set_update_porosity)
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
	"""Sub-class of pchemistry. Mineral kinetics are assigned to m_kinetics_list in pchemistry. The add function can do this automatically. e.g., dat.add(mineral_kinetic).
	
	:param name: Mineral name.
	:type name: str
	:param rate_constant_list: Value, Unit of Measurement. e.g., rate_constant_list=[1.e-6, 'mol/m^2-sec']
	:type rate_constant_list: [float, str] 
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
	"""Class for specifying a transport condition. Multiple transport objects can be created.
	Specifies a transport condition based on various user defined constraints with minerals, gases, pH, charge balance, free ion, and total concentrations.
	
	:param name: Transport condition name.
	:type name: str
	:param type: Options include: 'dirichlet', 'dirichlet_zero_gradient', 'equilibrium', 'neumann', 'mole', 'mole_rate', 'zero_gradient'.
	:type type: str
	:param constraint_list_value: List of constraint values. The position of each value in the list correlates with the position of each type in constraint_list_type.
	:type constraint_list_value: [float]
	:param constraint_list_type: List of constraint types. The position of each value in the list correlates with the position of each value in constraint_list_value. E.g., 'initial_constraint', 'inlet_constraint'.
	:type constraint_list_type: [str]
	"""
	
	def __init__(self, name='', type='', constraint_list_value=[],
		     constraint_list_type=[]):
		self._name = name	# e.g., initial, west, east
		self._type = type	# e.g., dirichlet, zero_gradient
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
	"""Class for specifying a transport constraint.  Multiple constraint objects can be created.

	:param name: Constraint name.
	:type name: str
	:param concentration_list: List of pconstraint_concentration objects.
	:type concentration_list: [pconstraint_concentration]. Works with add function so that concentration_list does not need to be remembered. e.g., dat.add(concentration). Used for key word CONC or CONCENTRATIONS
	:param mineral_list: List of pconstraint_mineral objects. Currently does not work with add function. Used for key word MNRL OR MINERALS.
	:type mineral_list: [pconstraint_mineral]
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
	"""Concentration unit, Sub-class for constraint. There can be multiple pconstraint_concentration objects appended to a single pconstraint object. Works with add function so that concentration_list in pconstraint does not need to be remembered. e.g., dat.add(concentration) instead of dat.constraint.concentration_list.append(concentration).
	
	:param pspecies: Primary species name for concentration.
	:type pspecies: str
	:param value: Concentration value.
	:type value: float
	:param constraint: Constraint name for concentration. Options include: 'F', 'FREE', 'T', 'TOTAL', 'TOTAL_SORB', 'P', 'pH', 'L', 'LOG', 'M', 'MINERAL', 'MNRL', 'G', 'GAS', 'SC', 'CONSTRAINT_SUPERCRIT_CO2
	:type constraint: str
	:param element: Name of mineral or gas.
	:type element: str
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
	"""Class for mineral in a constraint with vol. fraction and surface area. There can be multiple pconstraint_concentration objects appended to a single pconstraint object. Currently does not work with add function. pconstraint_mineral can be manually appended to minerals_list in a pconstraint object. e.g., 'constraint.mineral_list.append(mineral)'.

	:param name: Mineral name.
	:type name: str
	:param volume_fraction: Volume fraction. [--]
	:type volume_fraction: float
	:param surface_area: Surface area. [m^-1]
	:type surface_area: float
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
		self._overwrite_restart_flow_params = False
		self._regression = pregression()
		self._simulation = psimulation()
		self._checkpoint = pcheckpoint()
		self._restart = prestart()
		self._datasetlist = [] 
		self._chemistry = None
		self._grid = pgrid()
		self._timestepper = None
		self._proplist = []
		self._time = ptime()
		self._lsolverlist = []	# Possible to have 1 or 2 lsolver lists. FLOW/TRAN
		self._nsolverlist = []	# Possible to have 1 or 2 nsolver lists. FLOW/TRAN
		self._output = poutput()
		self._fluid = pfluid()
		self._saturationlist = [] 
		self._regionlist = []	# There are multiple regions
	        self._charlist = []
         	self._regionlist = []	# There are multiple regions
		self._observation_list = []
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

	def run(self,input='', input_prefix = '', num_procs=1, exe=pdflt().pflotran_path):
		'''Run a pflotran simulation for a given input file with specified number of processors.
	
		:param input: Name of input file. Uses default -pflotranin flag 
		:type input: str	
		:param input_prefix: Name of input file prefix. Uses the -input_prefix flag.
		:type input_prefix: str
		:param exe: Path to PFLOTRAN executable.
		:type exe: str
		:param num_procs: Number of processors
		:type num_procs: int
		'''
		
		# set up and check path to executable
		exe_path = ppath()
		exe_path.filename = exe
		
		if not os.path.isfile(exe_path.full_path): # if can't find the executable, halt
			print('ERROR: Default location is' +exe + '. No executable at location '+exe)
			return
		
		# option to write input file to new name
		if input: self._path.filename = input
		if input_prefix: self._path.filename = input_prefix
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
		if input and input_prefix:
			print('ERROR: Cannot specify both input and input_prefix')
			return
		if input: 
			subprocess.call('mpirun -np ' + str(num_procs) + ' ' +  exe_path.full_path + ' -pflotranin ' + self._path.filename,shell=True)
		if input_prefix:
			subprocess.call('mpirun -np ' + str(num_procs) + ' ' +  exe_path.full_path + ' -input_prefix ' + self._path.filename,shell=True)


		# After executing simulation, go back to the parent directory
		if self.work_dir: os.chdir(cwd)
		
	def __repr__(self): return self.filename 	# print to screen when called
	
	def plot_data_from_tec(self,direction='X',variable_list=[],tec_filenames=[],legend_list=[],plot_filename='',fontsize=10,xlabel='',ylabel_list=[],xtype='linear',ytype='linear',xrange=(),yrange=(),xfactor=1.0,yfactor=1.0):
	    
		for var,ylabel in zip(variable_list,ylabel_list):
			fig = plt.figure()
			ax = fig.add_subplot(1,1,1)
			ax.set_xlabel(xlabel)
			ax.set_ylabel(ylabel)
			ax.set_xscale(xtype)
			ax.set_yscale(ytype)
			if xrange: ax.set_xlim(xrange)
			if yrange: ax.set_ylim(yrange)
			lns = []
			for file in tec_filenames:
				variable = []
				var_values_dict = {}
				f = open(file,'r')
				time = f.readline().split('"')[1]
				title = f.readline()
				title = title.split(',')
				for i in title:
					variable.append(i.strip('"'))
				data = np.genfromtxt(file,skip_header=3)
				data = data.T.tolist()
				var_values_dict = dict(zip(variable,data))
				for key in var_values_dict.keys():
					if direction.upper() in key:
						xval = [val*xfactor for val in var_values_dict[key]]
					if var in key: 
						dat = [val*yfactor for val in var_values_dict[key]]				
					else:
						print 'Variable ' + var + ' not found in the tec files.'	
				ln, = ax.plot(xval,dat)
				lns.append(ln)
			ax.legend(lns,legend_list,ncol=1,fancybox=True,shadow=False,prop={'size':str(fontsize)},loc='best') 
			if '.pdf' in plot_filename:
				plot_filename = plot_filename.replace(".pdf","")
			if ' ' in var:
				var = var.replace(" ","_")
			print 'Plotting variable [' + var + '] in [' + direction +'] direction'
			fig.savefig(plot_filename + '_' + var + '.pdf')

		return 0
	
	def plot_observation(self, variable_list=[], observation_list=[], observation_filenames=[],plot_filename='',legend_list=[],fontsize=10,xlabel='',ylabel='',xtype='linear',ytype='linear',xrange=(),yrange=(),xfactor=1.0,yfactor=1.0):
		''' Plot time-series data from observation files at a given set of observation points.

		:param variable_list: List of the variables to be plotted
		:type variable_list: [str]
		:param observation_list: List of observation names to be plotted
		:type observation_list: [str]
		:param observation_filenames: List of observation filenames that are to be used for extracting data
		:type observation_filenames: [str]
		:param plot_filename: Name of the file to which the plot is saved
		:type plot_filename: str
		:param legend_list: List of legend
		:type legend_list: [str]
		:param fontsize: size of the legend font
		:type fontsize: float
		:param xlabel: label on the x-axis
		:type xlabel: str
		:param ylabel: label on the y-axis
		:type ylabel: str
		:param xtype: type of plot in the x-direction, e.g., 'log', 'linear', 'symlog'
		:type xtype: str
		:param ytype: type of plot in the y-direction, e.g., 'log', 'linear', 'symlog'
		:type ytype: str
		:param xrange: limits on the x-axis range, e.g., (0,100)
		:type xrange: (float,float)
		:param yrange: limits on the y-axis range, e.g., (0,100)
		:type yrange: (float,float)
		'''
		combined_dict = {}
		for file in observation_filenames:
			variable = []
			var_values_dict = {}
			f = open(file,'r')
			title = f.readline()
			title = title.split(',')
			for i in title:
				variable.append(i.strip('"'))
			data = np.genfromtxt(file,skip_header=1)
			data = data.T.tolist()
			var_values_dict = dict(zip(variable,data))
			combined_dict.update(var_values_dict)	

		for key in combined_dict.keys():
			if 'Time' in key:
				time = combined_dict[key]
		
		combined_var_obs_list = [variable_list,observation_list]
		combined_var_obs_list = list(it.product(*combined_var_obs_list))
		
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_xscale(xtype)
		ax.set_yscale(ytype)
		if xrange: ax.set_xlim(xrange)
		if yrange: ax.set_ylim(yrange)
		lns = []
		for item in combined_var_obs_list:
			for key in combined_dict.keys():
				if item[0] in key and item[1] in key:
					time_new = [t*xfactor for t in time]
					var_new = [v*yfactor for v in combined_dict[key]]
					ln, = ax.plot(time_new,var_new)
					lns.append(ln)
		ax.legend(lns,legend_list,ncol=1,fancybox=True,shadow=False,prop={'size':str(fontsize)},loc='best') 
		fig.savefig(plot_filename)
        
		return 0

	def read(self, filename=''):
		'''Read a given PFLOTRAN input file. This method is useful for reading an existing a PFLOTRAN input deck and all the corresponding PyFLOTRAN objects and data structures are autmatically created.
	
		:param filename: Name of input file. 
		:type filename: str
		'''
		if not os.path.isfile(filename): print filename + ' not found...'
		self._filename = filename 	# assign filename attribute
		read_fn = dict(zip(cards, 	
				[self._read_co2_database,
				 self._read_uniform_velocity,
				 self._read_simulation,
				 self._read_regression,
				 self._read_checkpoint,
				 self._read_restart,
				 self._read_dataset,
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
				 self._read_characteristic_curves,
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
				 
		skip_readline = False	
		line = ''		# Memorizes the most recent line read in.
		
		def get_next_line(skip_readline=skip_readline, line=line):
			"""Used by read function to avoid skipping a line in cases where a particular read function might read an extra line.
			"""
			
			if skip_readline:
				skip_readline = False
				return line
			else:
				line = infile.readline()
				return line
			
		with open(self._filename,'r') as infile:
			keepReading = True
			while keepReading:
				line = get_next_line()
				if not line: keepReading = False
				if len(line.strip())==0: continue
				card = line.split()[0].lower() 		# make card lower case
				if card == 'overwrite_restart_flow_params': self._overwrite_restart_flow_params = True
				if card == 'skip':
					keepReading1 = True
					while keepReading1:
						line1 = get_next_line() 
						if not line1: keepReading1 = False
						if len(line1.strip())==0: continue
						card1 = line1.split()[0].lower()
						if card1 == 'noskip':keepReading1 = False

				if card in cards: 			# check if a valid cardname
					if card in ['co2_database','checkpoint','restart','dataset','material_property',
					 'simulation','regression','grid',
					 'timestepper','linear_solver','newton_solver',
					 'saturation_function','region','flow_condition',
					 'boundary_condition','transport_condition','constraint',
					 'uniform_velocity']:
						
						read_fn[card](infile,line)
					else:
						read_fn[card](infile)
		
#	def _get_skip_readline(self): return self._skip_readline
#	def _set_skip_readline(self, object): self._skip_readline = object
#	skip_readline = property(_get_skip_readline, _set_skip_readline) #: (**)

	# Memorizes the most recent line read in.
#	def _get_line(self): return self._line
#	def _set_line(self, object): self._line = object
#	line = property(_get_line, _set_line) #: (**)
	
	def write(self, filename=''):
		"""Write pdata object to PFLOTRAN input file. Does not execute the input file - only writes a corresponding PFLOTRAN input file.

		:param filename: Name of PFLOTRAN input file.
		:type filename: str
		"""
		if filename: self._filename = filename
		outfile = open(self.filename,'w')
		
		# Presumes simulation.simulation_type is required
		if self.simulation.simulation_type: self._write_simulation(outfile)
		else: 
			print 'ERROR: simulation is required, it is currently reading as empty\n'
			return
	
		
		if self.simulation.subsurface_flow or self.simulation.subsurface_transport:
			self._write_subsurface_simulation_begin(outfile)

		if self.regression.cells or self.regression.cells_per_process: self._write_regression(outfile)
		
		# Presumes uniform_velocity.value_list is required
		if self.uniform_velocity.value_list: self._write_uniform_velocity(outfile)

		if self.co2_database: self._write_co2_database(outfile)
		
		if self._overwrite_restart_flow_params: self._write_overwrite_restart(outfile)
		
		if self.checkpoint.frequency: self._write_checkpoint(outfile)
#		else: print 'info: checkpoint not detected\n'
		
		if self.restart.file_name: self._write_restart(outfile)
#		else: print 'info: restart not detected\n'

		if self.datasetlist: self._write_dataset(outfile)
#		else: print 'info: dataset name not detected\n'

		if self.chemistry: self._write_chemistry(outfile)
#		else: print 'info: chemistry not detected\n'
		
		if self.grid: self._write_grid(outfile)
		else: print 'ERROR: grid is required, it is currently reading as empty\n'
		
		if self.timestepper : self._write_timestepper(outfile)
#		else: print 'info: timestepper not detected\n'
		
		if self.time: self._write_time(outfile)
		else: print 'ERROR: time is required, it is currently reading as empty\n'
		
		if self.proplist: self._write_prop(outfile)
		else: print 'ERROR: proplist is required, it is currently reading as empty\n'
		
		if self.lsolverlist: self._write_lsolver(outfile)
#		else: print 'info: lsolverlist (linear solver list) not detected\n'
		
		if self.nsolverlist: self._write_nsolver(outfile)
#		else: print 'info: nsolverlist (newton solver list) not detected\n'
		
		if self.output: self._write_output(outfile)
		else: print 'ERROR: output is required, it is currently reading as empty\n'
		
		if self.fluid: self._write_fluid(outfile)
		else: print 'ERROR: fluid is required, it is currently reading as empty\n'
		
		if self.saturationlist: self._write_saturation(outfile)
		else: print 'ERROR: saturationlist is required, it is currently reading as empty\n'
				
#		if self.charlist: self._write_characteristic_curves(outfile)
#		else: print 'info: characteristic curves not detected\n'
		
                if self.regionlist: self._write_region(outfile)
		else: print 'ERROR: regionlist is required, it is currently reading as empty\n'
		
		if self.observation_list: self._write_observation(outfile)
#		else: print 'info: observation_list not detect\n'
		
		if self.flowlist: self._write_flow(outfile)
		else: print 'ERROR: flowlist not detected\n'
		
		if self.transportlist: self._write_transport(outfile)
		
		if self.initial_condition: self._write_initial_condition(outfile)
		else: print 'ERROR: initial_condition is required, it is currently reading as empty\n'
		
		if self.boundary_condition_list: self._write_boundary_condition(outfile)
		else: print 'ERROR: boundary_condition_list is required, it is currently reading as empty\n'
		
#		if self.source_sink: self._write_source_sink(outfile)
#		else: print 'info: source_sink not detected\n'

		if self.strata_list: self._write_strata(outfile)
		else: print 'ERROR: (stratigraphy_coupler) strata is required, it is currently reading as empty\n'
		
		if self.constraint_list: self._write_constraint(outfile)
		
		if self.simulation.subsurface_flow or self.simulation.subsurface_transport:
			self._write_subsurface_simulation_end(outfile)

		outfile.close()
        
	def add(self,obj,index='',overwrite=False):	#Adds a new object to the file
		'''Attach an object associated w/ a list (e.g., pregion) that belongs to a pdata object. 
		
		:param obj: Object to be added to the data file.
		:type obj: object(e.g., pregion)
		:param index: (Optional) Used to find an object that is using a string as an index in a dictionary. Intended for the super class object. (E.g. Index represents flow.name if instance is pflow_variable.) Default if not specified is to use the last super-class object added to pdata.
		:type index: String
		:param overwrite: Flag to overwrite an object if it already exists in a pdata object. 
		:type overwrite: bool
		'''
	
		add_checklist = [pmaterial,pdataset,psaturation,pcharacteristic_curves,pchemistry_m_kinetic,plsolver,pnsolver,pregion,pobservation,pflow,pflow_variable,pboundary_condition,pstrata,ptransport,pconstraint,pconstraint_concentration]

	 	# Check if obj first is an object that belongs to add_checklist
		checklist_bool = [isinstance(obj,item) for item in add_checklist]
		if True in checklist_bool:
			pass	
		else:
			print 'pdata.add used incorrectly! Cannot use pdata.add with one of the specificed object.'
			sys.exit()
	
		# Always make index lower case if it is being used as a string
		if isinstance(index,str): index=index.lower()
		if isinstance(obj,pmaterial): self._add_prop(obj,overwrite)
		if isinstance(obj,pdataset): self._add_dataset(obj,overwrite)
		if isinstance(obj,psaturation): self._add_saturation(obj,overwrite)
		if isinstance(obj,pcharacteristic_curves): self._add_characteristic_curves(obj,overwrite)
		if isinstance(obj,pchemistry_m_kinetic): 
			self._add_chemistry_m_kinetic(obj,overwrite)
		if isinstance(obj,plsolver): self._add_lsolver(obj,overwrite)
		if isinstance(obj,pnsolver): self._add_nsolver(obj,overwrite)
		if isinstance(obj,pregion): self._add_region(obj,overwrite)
		if isinstance(obj,pobservation): self._add_observation(obj, overwrite)
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
		'''Delete an object that is assigned to a list of objects belong to a pdata object, e.g., pregion. 
		
		:param obj: Object to be deleted from the data file. Can be a list of objects.
		:type obj: Object (e.g., pregion), list
		'''

		if isinstance(obj,pmaterial): self._delete_prop(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):	# obji = object index
				if isinstance(obji,pmaterial): self._delete_prop(obji)

		if isinstance(obj,pcharacteristic_curves): self._delete_characteristic_curves(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):	# obji = object index
				if isinstance(obji,pcharacteristic_curves): self._delete_characteristic_curves(obji)

		if isinstance(obj,pchemistry_m_kinetic): self._delete_pchemistry_m_kinetic(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):
				if isinstance(obji,pchemistry_m_kinetic): self._delete_pchemistry_m_kinetic(obji)
			
		if isinstance(obj,plsolver): self._delete_lsolver(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):
				if isinstance(obji,plsolver): self._delete_lsolver(obji)
		
		if isinstance(obj,pnsolver): self._delete_nsolver(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):
				if isinstance(obji,pnsolver): self._delete_nsolver(obji)
				
		if isinstance(obj,pobservation): self._delete_observation(obj)
		elif isinstance(obj,list):
			for obji in copy(obj):
				if isinstance(obji,pobservation): self._delete_observation(obji)
		
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

	def _read_simulation(self,infile,line):
		simulation = psimulation()
		keepReading = True
		
		while keepReading: #Read through all cards
                        line = infile.readline()        # get next line
                        key = line.strip().split()[0].lower()   # take first key word
			if key == 'simulation_type':
				simulation.simulation_type = line.split()[-1]
                        elif key == 'subsurface_flow':
				simulation.subsurface_flow = line.split()[-1] 
       				keepReading1 = True
				while keepReading1:
					line = infile.readline()
					key1 = line.strip().split()[0].lower()	
					if key1== 'mode':
						simulation.mode = line.split()[-1].lower()                         
					elif key1 in ['/','end']:keepReading1 = False
					else:
						print('ERROR: mode is missing!')
                	elif key == 'subsurface_transport':
				simulation.subsurface_transport = line.split()[-1] 
				keepReading2 = True
				while keepReading2:
					line1 = infile.readline()
					key1 = line1.strip().split()[0].lower()
					if key1 == 'global_implicit':
						simulation.flowtran_coupling = key1
					elif key1 in ['/','end']:keepReading2 = False
#				else:
#					print('ERROR: flow tran coupling type missing!')
			elif key in ['/','end']: keepReading = False 

		self._simulation = simulation                               
                              
	def _write_simulation(self,outfile):
		self._header(outfile,headers['simulation'])
		simulation = self.simulation
		#Write out simulation header
		outfile.write('SIMULATION' +'\n')
		if simulation.simulation_type.lower() in simulation_types_allowed:
			outfile.write('  SIMULATION_TYPE '+ simulation.simulation_type.upper() + '\n' )
		else:
			print 'ERROR: simulation.simulation_type: \''+ simulation.simulation_type +'\' is invalid.'
			print '       valid simulation.simulation_type:', simulation_types_allowed, '\n'
		if simulation.subsurface_flow and simulation.subsurface_transport:
                        outfile.write('  PROCESS_MODELS' +'\n')		
                        outfile.write('    SUBSURFACE_FLOW '+ simulation.subsurface_flow +'\n')
			if simulation.mode in mode_names_allowed:		
	                        outfile.write('      MODE '+ simulation.mode +'\n')		
                        else:
				print 'ERROR: simulation.mode: \''+ simulation.mode +'\' is invalid.'
				print '       valid simulation.mode:', mode_names_allowed, '\n'
			outfile.write('    / '+'\n')
                        outfile.write('    SUBSURFACE_TRANSPORT '+ simulation.subsurface_transport +'\n')
			if simulation.flowtran_coupling:
				outfile.write('      ' + simulation.flowtran_coupling.upper() + '\n')
			outfile.write('    / '+'\n')
                        outfile.write('  / '+'\n')
                        outfile.write('END'+'\n\n')
		elif simulation.subsurface_flow:
                        outfile.write('  PROCESS_MODELS' +'\n')		
                        outfile.write('    SUBSURFACE_FLOW '+ simulation.subsurface_flow +'\n') 
			if simulation.mode in mode_names_allowed:		
	                        outfile.write('      MODE '+ simulation.mode +'\n')		
                        else:
				print 'ERROR: simulation.mode: \''+ simulation.mode +'\' is invalid.'
				print '       valid simulation.mode:', mode_names_allowed, '\n'
                        outfile.write('    / '+'\n')
                        outfile.write('  / '+'\n')
                        outfile.write('END'+'\n\n')
		elif simulation.subsurface_transport:
                        outfile.write('  PROCESS_MODELS' +'\n')		
                        outfile.write('    SUBSURFACE_TRANSPORT '+ simulation.subsurface_transport +'\n')
                        outfile.write('    / '+'\n')
                        outfile.write('  / '+'\n')
                        outfile.write('END'+'\n\n')

        def _write_subsurface_simulation_begin(self,outfile):
		if self.simulation.subsurface_flow or self.simulation.subsurface_transport:
                        outfile.write('SUBSURFACE\n\n')		

        def _write_subsurface_simulation_end(self,outfile):
		if self.simulation.subsurface_flow or self.simulation.subsurface_transport:
                        outfile.write('END_SUBSURFACE\n\n')		

	def _read_co2_database(self,infile,line):
		self._co2_database = del_extra_slash(line.split()[-1])
		
	def _write_overwrite_restart(self,outfile):
		outfile.write('OVERWRITE_RESTART_FLOW_PARAMS' + '\n\n')

	def _write_co2_database(self,outfile):
		self._header(outfile,headers['co2_database'])
		outfile.write('CO2_DATABASE ' + self._co2_database + '\n\n')

	def _read_regression(self,infile,line):
		regression = pregression()
		keepReading = True
		while keepReading: #Read through all cards
                        line = infile.readline()        # get next line
                        key = line.strip().split()[0].lower()   # take first key word

			if key == 'cells':
				keepReading2 = True
				while keepReading2:
					cell_list = []
					for i in range(100):
						line1 = infile.readline()
						if line1.strip().split()[0].lower() in ['/','end']:
							keepReading2 = False
							break
						cell_list.append(int(line1))
				regression.cells = cell_list 
                        elif key == 'cells_per_process':
				regression.cells_per_process= line.split()[-1] 
			elif key in ['/','end']: keepReading = False

		self._regression = regression 

	def _write_regression(self,outfile):
		self._header(outfile,headers['regression'])
		regression = self.regression
		outfile.write('REGRESSION' +'\n')
		if regression.cells:
			outfile.write('  CELLS' + '\n' )
			for cell in regression.cells:
				outfile.write('    ' + str(cell) + '\n')
			outfile.write('  /' + '\n' )
		if regression.cells_per_process:
			outfile.write('  CELLS_PER_PROCESS' + ' ' + str(regression.cells_per_process) + '\n' )
		outfile.write('END'+'\n\n')
	

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
			for j in range(len(grid.dx)):
				outfile.write('    ' + strD(grid.dx[j]))
				if (j%5 == 4):
					outfile.write('   ' + '\\' + '\n')
			outfile.write('\n')
			for j in range(len(grid.dy)):
				outfile.write('    ' + strD(grid.dy[j]))
				if (j%5 == 4):
					outfile.write('   ' + '\\' + '\n')
			outfile.write('\n')
			for j in range(len(grid.dz)):
				outfile.write('    ' + strD(grid.dz[j]))
				if (j%5 == 4):
					outfile.write('   ' + '\\' + '\n')
			outfile.write('\n')
			outfile.write('  END\n')
		if grid.origin:
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
                np_ts_mode = p.ts_mode
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
			if key == 'ts_mode':
				np_ts_mode= str(line.split()[-1])
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

		new_timestep = ptimestepper(np_ts_mode,np_ts_acceleration,np_num_steps_after_cut,np_max_steps,
					    np_max_ts_cuts,np_cfl_limiter,np_initialize_to_steady_state,
					    np_run_as_steady_state,np_max_pressure_change,
					    np_max_temperature_change,np_max_concentration_change,
		                            np_max_saturation_change)

		self._timestepper = new_timestep
	
	def _write_timestepper(self,outfile):
		self._header(outfile,headers['timestepper'])
		outfile.write('TIMESTEPPER ' + self.timestepper.ts_mode + '\n')
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
                np_characteristic_curves = p.characteristic_curves 
		np_tortuosity=p.tortuosity
		np_density=p.density
		np_specific_heat=p.specific_heat
		np_cond_dry=p.cond_dry
		np_cond_wet=p.cond_wet
		np_saturation=p.saturation
		np_permeability=[]
		np_permeability_critical_porosity=p.permeability_critical_porosity
		np_permeability_power=p.permeability_power
		np_permeablity_min_scale_factor=p.permeability_min_scale_factor

		keepReading = True

		while keepReading: 			# read through all cards
			line = infile.readline() 			# get next line
			key = line.strip().split()[0].lower() 		# take first keyword
			if key == 'id':
				np_id = int(line.split()[-1])
                        elif key == 'characteristic_curves':
                                np_characteristic_curves = line.split()[-1]
			elif key == 'porosity':
				if line.split()[1].lower() == 'dataset':
					np_porosity = line.split()[-1]
				else:	
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
			elif key == 'permeability_power':
				np_permeability_power = line.split()[-1]
			elif key == 'permeability_critical_porosity':
				np_permeability_critical_porosity = line.split()[-1]
			elif key == 'permeability_min_scale_factor':
				np_permeability_min_scale_factor = line.split()[-1]
			elif key == 'permeability':
				keepReading2 = True
				while keepReading2:
					line = infile.readline() 			# get next line
					key = line.split()[0].lower() 		# take first keyword
					if key == 'perm_iso':
						np_permeability.append(floatD(line.split()[-1]))
					elif key == 'perm_x':
						np_permeability.append(floatD(line.split()[-1]))
					elif key == 'perm_y':
						np_permeability.append(floatD(line.split()[-1]))
					elif key == 'perm_z':
						np_permeability.append(floatD(line.split()[-1]))
					elif key in ['/','end']: keepReading2 = False
			elif key in ['/','end']: keepReading = False
		new_prop = pmaterial(np_id,np_name,np_characteristic_curves,np_porosity,np_tortuosity,np_density,
		                     np_specific_heat,np_cond_dry,np_cond_wet,
				     np_saturation,np_permeability,np_permeability_power,np_permeability_critical_porosity,np_permeability_min_scale_factor) 		# create an empty material property

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
                        if prop.characteristic_curves:
                                outfile.write('  CHARACTERISTIC_CURVES '+prop.characteristic_curves+'\n')
			if prop.porosity:
				if type(prop.porosity) is str:
					outfile.write('  POROSITY DATASET '+ prop.porosity +'\n')
				else:
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
			if prop.permeability_power:
				outfile.write('  PERMEABILITY_POWER '+prop.permeability_power+'\n')
			if prop.permeability_critical_porosity:
				outfile.write('  PERMEABILITY_CRITICAL_POROSITY '+prop.permeability_critical_porosity+'\n')
			if prop.permeability_min_scale_factor:
				outfile.write('  PERMEABILITY_MIN_SCALE_FACTOR '+prop.permeability_min_scale_factor+'\n')

			if prop.permeability:
				outfile.write('  PERMEABILITY\n')
				if len(prop.permeability) == 1:
					outfile.write('    PERM_ISO '+strD(prop.permeability[0])+'\n')
				else:	
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
			if key == 'preconditioner':
				lsolver.preconditioner = line.split()[-1]
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
				outfile.write('  SOLVER ' + lsolver.solver.upper() + '\n')
			if lsolver.preconditioner:
				outfile.write('  PRECONDITIONER ' + lsolver.preconditioner.upper() + '\n')
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
		output.format_list = []
		output.variables_list = []
	
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
			elif key == 'velocities':
				output.velocities = True
			elif key == 'velocity_at_center':
				output.velocity_at_center = True
			elif key == 'mass_balance':
				output.mass_balance = True
			elif key == 'variables':	
				keepReading1 = True
				while keepReading1:
					line1 = infile.readline()
					key1 = line1.strip().split()[0].lower()
					if key1 in output_variables_allowed:
						output.variables_list.append(key1)
					elif key1 in ['/','end']: keepReading1 = False
					else:
						print('ERROR: variable ' + str(key1) + ' cannot be an output variable.')
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
		if not output.screen_output:
			try: # Error checking to ensure screen_output is Bool.
				output.screen_output = bool(output.screen_output)
				outfile.write('  '+'SCREEN OFF'+'\n')
			except(ValueError):
				print 'ERROR: output.screen_output: \''+output.screen_output+'\' is not bool.\n'

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
		if output.velocities:
			outfile.write('  '+'VELOCITIES'+'\n')
		if output.velocity_at_center:
			outfile.write('  '+'VELOCITY_AT_CENTER'+'\n')
		if output.mass_balance:
			outfile.write('  '+'MASS_BALANCE'+'\n')
		if output.variables_list:
			outfile.write('  VARIABLES \n')
			for variable in output.variables_list:
				if variable.lower() in output_variables_allowed:
					outfile.write('    ' + variable.upper() + '\n')
				else:
					print 'ERROR: output.variable: \''+ variable +'\' is invalid.'
					print '       valid output.variable:', output_variables_allowed, '\n'
			outfile.write('  /\n')
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
		self.add(saturation)

	def _add_saturation(self,sat=psaturation(),overwrite=False):	#Adds a saturation object.
		# check if saturation already exists
		if isinstance(sat,psaturation):		
			if sat.name in self.saturation.keys():
				if not overwrite:
					warning = 'WARNING: A saturation function with name \''+str(sat.name)+'\' already exists. Use overwrite = True in add() to overwrite the old saturation function.'
					print warning; print
					_buildWarnings(warning)
					return
				else: # Executes if overwrite = True
					self.delete(self.sat[saturation.name])
					
		if sat not in self._saturationlist:
			self._saturationlist.append(sat)
			
	def _delete_saturation(self,sat=psaturation()):
		self._saturationlist.remove(sat)
		
	def _write_saturation(self,outfile):
		self._header(outfile,headers['saturation_function'])
		for sat in self._saturationlist:	
			# Write out saturation properties that exist
			outfile.write('SATURATION_FUNCTION')
			if sat.name:
				outfile.write('  ' + sat.name + '\n')
			else:
				outfile.write('\n')
			if sat.permeability_function_type:
				if sat.permeability_function_type in permeability_function_types_allowed:
					outfile.write('  PERMEABILITY_FUNCTION_TYPE ' +
						sat.permeability_function_type + '\n')
				else:		
					print('ERROR: saturation.saturation_function_type: \'' + sat.saturation_function_type +'\' is invalid.')
					print('       valid saturation.permeability_function_types', saturation_function_types_allowed, '\n')
			if sat.saturation_function_type:
				if sat.saturation_function_type in saturation_function_types_allowed:
					outfile.write('  SATURATION_FUNCTION_TYPE ' +
							sat.saturation_function_type + '\n')
			if sat.residual_saturation or sat.residual_saturation==0:
				outfile.write('  RESIDUAL_SATURATION ' + 
						strD(sat.residual_saturation) + '\n')
			if sat.residual_saturation_liquid or sat.residual_saturation_liquid ==0:
				outfile.write('  RESIDUAL_SATURATION LIQUID_PHASE ' + 
						strD(sat.residual_saturation_liquid) + '\n')
			if sat.residual_saturation_gas or sat.residual_saturation_gas == 0:
				outfile.write('  RESIDUAL_SATURATION GAS_PHASE ' +
						strD(sat.residual_saturation_gas) + '\n')
			if sat.a_lambda:
				outfile.write('  LAMBDA ' + strD(sat.a_lambda) + '\n')
			if sat.alpha:
				outfile.write('  ALPHA ' + strD(sat.alpha) + '\n')
			if sat.max_capillary_pressure:
				outfile.write('  MAX_CAPILLARY_PRESSURE ' + 
						strD(sat.max_capillary_pressure) + '\n')
			if sat.betac:
				outfile.write('  BETAC ' + strD(sat.betac) + '\n')
			if sat.power:
				outfile.write('  POWER ' + strD(sat.power) + '\n')
			outfile.write('END\n\n')
		
	
        def _read_characteristic_curves(self,infile,line):
		
		characteristic_curves = pcharacteristic_curves()	# assign defaults before reading in values 
        	characteristic_curves.name = line.split()[-1].lower() # Characteristic curve name, passed in.
		
		keepReading = True

		while keepReading:	# Read through all cards
			line = infile.readline()	# get next line
			key = line.strip().split()[0].lower()	# take first  key word
			
			if key == 'saturation_function_type':
				characteristic_curves.saturation_function_type = line.split()[-1]
			elif key == 'sf_alpha':
				characteristic_curves.sf_alpha = floatD(line.split()[-1])
			elif key == 'sf_m':
				characteristic_curves.sf_m = floatD(line.split()[-1])
			elif key == 'sf_lambda':
				characteristic_curves.sf_lambda = floatD(line.split()[-1])
			elif key == 'sf_liquid_residual_saturation':
				characteristic_curves.sf_liquid_residual_saturation = floatD(line.split()[-1])
			elif key == 'sf_gas_residual_saturation':
				characteristic_curves.sf_gas_residual_saturation = floatD(line.split()[-1])
			elif key == 'max_capillary_pressure':
				characteristic_curves.max_capillary_pressure = floatD(line.split()[-1])
			elif key == 'smooth':
				characteristic_curves.smooth = floatD(line.split()[-1])  
			elif key == 'power':
				characteristic_curves.power = floatD(line.split()[-1])
			elif key == 'default':
				characteristic_curves.default = floatD(line.split()[-1])
			elif key == 'liquid_permeability_function_type':
				characteristic_curves.liquid_permeability_function_type = line.split()[-1]
			elif key == 'lpf_m':
				characteristic_curves.lpf_m = floatD(line.split()[-1])
			elif key == 'lpf_lambda':
				characteristic_curves.lpf_lambda = floatD(line.split()[-1])
			elif key == 'lpf_liquid_residual_saturation':
				characteristic_curves.lpf_liquid_residual_saturation = floatD(line.split()[-1])
			elif key == 'gas_permeability_function_type':
				characteristic_curves.gas_permeability_function_type = line.split()[-1]
			elif key == 'gpf_m':
				characteristic_curves.gpf_m = floatD(line.split()[-1])
			elif key == 'gpf_lambda':
				characteristic_curves.gpf_lambda = floatD(line.split()[-1])
			elif key == 'gpf_liquid_residual_saturation':
				characteristic_curves.gpf_liquid_residual_saturation = floatD(line.split()[-1])
			elif key == 'gpf_gas_residual_saturation':
				characteristic_curves.gpf_gas_residual_saturation = floatD(line.split()[-1])
             		elif key in ['/','end']: keepReading = False

		new_cc = pcharacteristic_curves(characteristic_curves.name, characteristic_curves.saturation_function_type,characteristic_curves.sf_alpha, characteristic_curves.sf_m, characteristic_curves.sf_lambda, characteristic_curves.sf_liquid_residual_saturation, characteristic_curves.sf_gas_residual_saturation, characteristic_curves.max_capillary_pressure, characteristic_curves.smooth, characteristic_curves.power, characteristic_curves.default, characteristic_curves.liquid_permeability_function_type, characteristic_curves.lpf_m, characteristic_curves.lpf_lambda, characteristic_curves.lpf_liquid_residual_saturation, characteristic_curves.gas_permeability_function_type, characteristic_curves.gpf_m, characteristic_curves.gpf_lambda, characteristic_curves.gpf_liquid_residual_saturation, characteristic_curves.gpf_gas_residual_saturation)     
      		
	        self.add(new_cc)

        def _add_characteristic_curves(self,char=pcharacteristic_curves(),overwrite=False):   #Adds a char object.
                # check if char already exists
                if isinstance(char,pcharacteristic_curves):
                        if char.name in self.char.keys():
                                if not overwrite:
                                        warning = 'WARNING: A Characteristic Curve with name \''+str(char.name)+'\' already exists. Characteristic curve will not be defined, use overwrite = True in add() to overwrite the old characteristic curve.'
                                        print warning; print
                                        _buildWarnings(warning)
                                        return
                                else: # Executes if overwrite = True
                                        self.delete(self.char[char.name])

                if char not in self._charlist:
                        self._charlist.append(char)

        def _delete_char(self,char=pcharacteristic_curves()):
                self._charlist.remove(char)
		
	def _write_characteristic_curves(self,outfile): 
		
		self._header(outfile,headers['characteristic_curves'])
		characteristic_curves = pcharacteristic_curves()
		for char in self.charlist:		
			# Write out characteristic curve properties that exist
			if char.name:
				outfile.write('CHARACTERISTIC_CURVES ' + char.name + '\n')		
			if char.saturation_function_type:
				if char.saturation_function_type in characteristic_curves_saturation_function_types_allowed:
					outfile.write('  SATURATION_FUNCTION ' +
						char.saturation_function_type + '\n')
				else:
					print 'ERROR: char.saturation_function_type: \'' + char.saturation_function_type +'\' is invalid.'
					print '       valid  char.saturation_function_types', characteristic_curves_saturation_function_types_allowed, '\n'	
				if char.sf_alpha:
					outfile.write('   ALPHA ' + strD(char.sf_alpha) + '\n')
				if char.sf_m:
					outfile.write('   M ' + strD(char.sf_m) + '\n')
				if char.sf_lambda:
					outfile.write('   LAMBDA ' + strD(char.sf_lambda) + '\n')
				if char.sf_liquid_residual_saturation or char.sf_liquid_residual_saturation==0:
					outfile.write('   LIQUID_RESIDUAL_SATURATION ' + 
							strD(char.sf_liquid_residual_saturation) + '\n')
				if char.sf_gas_residual_saturation or char.sf_gas_residual_saturation==0:
					outfile.write('   GAS_RESIDUAL_SATURATION ' + 
							strD(char.sf_gas_residual_saturation) + '\n')
				if char.max_capillary_pressure:
					outfile.write('   MAX_CAPILLARY_PRESSURE ' + 
							strD(char.max_capillary_pressure) + '\n')
				if char.smooth:
					outfile.write('   SMOOTH ' + '\n') # This just prints the SMOOTH flag
				outfile.write('  / '+'\n')

			if char.power:
				outfile.write('  POWER ' + strD(char.power) + '\n')
			if char.default:
				outfile.write('  DEFAULT ' + '\n') # This just prints the DEFAULT flag
			if char.liquid_permeability_function_type:
				if char.liquid_permeability_function_type in characteristic_curves_liquid_permeability_function_types_allowed:
					outfile.write('  PERMEABILITY_FUNCTION ' +
						char.liquid_permeability_function_type + '\n')
					outfile.write('   PHASE LIQUID' + '\n')
				else:
					print 'ERROR: char.liquid_permeability_function_type: \'' + char.liquid_permeability_function_type +'\' is invalid.'
					print '       valid  char.liquid_permeability_function_types', characteristic_curves_liquid_permeability_function_types_allowed, '\n'	
				if char.lpf_m:
					outfile.write('   M ' + strD(char.lpf_m) + '\n')
				if char.lpf_lambda:
					outfile.write('   LAMBDA ' + strD(char.lpf_lambda) + '\n')
				if char.lpf_liquid_residual_saturation or char.lpf_liquid_residual_saturation==0:
					outfile.write('   LIQUID_RESIDUAL_SATURATION ' + 
							strD(char.lpf_liquid_residual_saturation) + '\n')
				outfile.write('  / ' + '\n')

			if char.gas_permeability_function_type:
				if char.gas_permeability_function_type in characteristic_curves_gas_permeability_function_types_allowed:	
					outfile.write('  PERMEABILITY_FUNCTION ' +
						char.gas_permeability_function_type + '\n')
					outfile.write('   PHASE GAS' + '\n')
				else:

					print 'ERROR: char.gas_permeability_function_type: \'' + char.gas_permeability_function_type +'\' is invalid.'
					print '       valid  char.gas_permeability_function_types', characteristic_curves_gas_permeability_function_types_allowed, '\n'	
				if char.gpf_m:
					outfile.write('   M ' + strD(char.gpf_m) + '\n')
				if char.gpf_lambda:
					outfile.write('   LAMBDA ' + strD(char.gpf_lambda) + '\n')
				if char.gpf_liquid_residual_saturation or char.gpf_liquid_residual_saturation==0:
					outfile.write('   LIQUID_RESIDUAL_SATURATION ' + 
							strD(char.gpf_liquid_residual_saturation) + '\n')	
				if char.gpf_gas_residual_saturation or char.gpf_gas_residual_saturation==0:
					outfile.write('   GAS_RESIDUAL_SATURATION ' + 
							strD(char.gpf_gas_residual_saturation) + '\n')
				outfile.write('  / ' + '\n')

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
			elif key == 'coordinate':
					line1 = line.split()[1::]
					point = ppoint()
					point.name = region.name
					point.coordinate[0] = floatD(line1[0])
					point.coordinate[1] = floatD(line1[1])
					point.coordinate[2] = floatD(line1[2])
					region.point_list.append(point)
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
			if region.point_list:
				for point in region.point_list:
					outfile.write('  COORDINATE ')
					for i in range(3):
						outfile.write(strD(point.coordinate[i]) + ' ')
					outfile.write('\n')
			else:
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
		observation = pobservation()
		
		keepReading = True
		
		while keepReading:
			line = infile.readline() 			# get next line
			key = line.strip().split()[0].lower() 		# take first keyword
			if key == 'region':
				observation.region = line.split()[-1]
			elif key in ['/','end']: keepReading = False
			
		self._observation_list.append(observation)
		
	def _add_observation(self,observation=pobservation(),overwrite=False):	#Adds a Observation object.
		# check if observation already exists
		if isinstance(observation,pobservation):
			if observation.region in self.observation.keys():
				if not overwrite:
					warning = 'WARNING: A observation with region \''+str(observation.region)+'\' already exists. Observation will not be defined, use overwrite = True in add() to overwrite the old observation.'
					print warning; print
					_buildWarnings(warning)
					return
				else:
					self.delete(self.observation[observation.region])
					
		if observation not in self._observation_list:
			self._observation_list.append(observation)
			
	def _delete_observation(self,observation=pobservation()):
		self._observation_list.remove(observation)
		
	def _write_observation(self,outfile):
		self._header(outfile,headers['observation'])
		
		for observation in self.observation_list:
			outfile.write('OBSERVATION\n')
			if observation.region:
				outfile.write('  REGION '+observation.region.lower()+'\n')
			outfile.write('END\n\n')
				
	def _read_flow(self,infile,line):
		flow = pflow()
		flow.datum = []
		flow.varlist = []
		flow.datum_type = ''
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
				
			elif key == 'rate' or key == 'pressure' or key == 'temperature' or key == 'concentration' or key == 'enthalpy' or key == 'flux':
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
							for var in flow.varlist: # var represents a pflow_variable object
								if tstring2name.lower() == var.name.lower():
									if line[0] == ':' or line[0] == '#' or line[0] == '/':
										pass	# ignore a commented line
											# line[0] == '/' is a temporary fix
									elif tstring2[0].lower() == 'time_units':
										var.time_unit_type = tstring2[1]
									elif tstring2[0].lower() == 'data_units':
											var.data_unit_type = tstring2[1]
									elif line.split()[0] in ['/','end']: keepReadingList = False
									else:
										tvarlist = pflow_variable_list()
										tvarlist.time_unit_value = floatD(tstring2[0])
										tvarlist.data_unit_value_list = []
										tvarlist.data_unit_value_list.append(floatD(tstring2[1]))
										if len(tstring2) > 2:tvarlist.data_unit_value_list.append(floatD(tstring2[2]))
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
									# If a string (e.g., C for temp.), assign to unit
									except(ValueError):
										var.unit = substring
			elif key == 'iphase':
				flow.iphase = int(line.split()[-1])
			elif key == 'sync_timestep_with_update':
				flow.sync_timestep_with_update = True
			elif key == 'datum':
				# Assign file_name with list of d_dx, d_dy, d_dz values.
				if line.strip().split()[1].upper() == 'FILE':
					flow.datum_type = 'file'
					flow.datum = line.split()[1]
				if line.strip().split()[1].upper() == '':
					flow.datum_type = 'DATASET'
					flow.datum = line.split()[1]
				# Assign d_dx, d_dy, d_dz values
				else:
					temp_list = []
					temp_list.append(floatD(line.split()[1]))
					temp_list.append(floatD(line.split()[2]))
					temp_list.append(floatD(line.split()[3]))
					flow.datum.append(temp_list)
					
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
			elif condition_name.upper() == 'FLUX':
				if condition_type.lower() in flux_types_allowed: 
					outfile.write(condition_type.lower())
				else:
					print 'ERROR: flow.varlist.type: \'' + condition_type +'\' is invalid.'
					print '       valid flow_condition flux_types_allowed:', flux_types_allowed, '\n'	
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
				
			if flow.datum:	# error-checking not yet added
				
				outfile.write('  DATUM')
				
				if isinstance(flow.datum, str):
					if flow.datum_type == 'file':
						outfile.write(' FILE ')
					if flow.datum_type == 'dataset':
						outfile.write(' DATASET ')
					outfile.write(flow.datum)
				else: # Applies if datum is a list of [d_dx, d_dy, d_dz]
					# write out d_dx, d_dy, d_dz
					for line in flow.datum:
						outfile.write(' ')
						outfile.write(strD(line[0])+' ')
						outfile.write(strD(line[1])+' ')
						outfile.write(strD(line[2]))
				outfile.write('\n')
				
			outfile.write('  TYPE\n') # Following code is paired w/ this statement.
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

			for i in range(len(flow.varlist)):
				# Write if using non-list format (Single line)
				if flow.varlist[i].valuelist:
					outfile.write('    ' + flow.varlist[i].name.upper())	
					if isinstance(flow.varlist[i].valuelist[0], str):
				        	outfile.write(' DATASET '+ flow.varlist[i].valuelist[0])
					else:
				        	j = 0	
						while j < len(flow.varlist[i].valuelist):
							outfile.write(' ' + strD(flow.varlist[i].valuelist[j]))
					#	try:
					#		outfile.write(' ' + strD(flow.varlist[i].valuelist[j]))
					#	except:
					#		outfile.write(' DATASET ' + (flow.varlist[i].valuelist[j]))
						
							j += 1
					# Write out possible unit here
					if flow.varlist[i].unit:
						outfile.write(' ' + flow.varlist[i].unit.lower())
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
			outfile.write('END\n\n')
		
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
		
		# Write out strata condition variables
		for strata in self.strata_list:
	
			outfile.write('STRATA\n')
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
		if len(tstring) > 1:
			restart.time_value = floatD(tstring[1])
		elif len(tstring) > 2:
			restart.time_unit = tstring[2]
		
		self._restart = restart
		
	def _write_restart(self, outfile):
		self._header(outfile,headers['restart'])
		restart = self.restart
		
		# write file name
		outfile.write('RESTART ' + str(restart.file_name) + ' ')
		
		# Write time value
		if restart.time_value:
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
		if restart.time_unit:
			restart.time_unit = str(restart.time_unit).lower()
			if restart.time_unit in time_units_allowed:
				outfile.write(restart.time_unit)
			else:
				print 'ERROR: restart.time_unit \'',restart.time_unit,'\' is invalid. Valid times units are:',time_units_allowed,'\n'
				outfile.write(restart.time_unit)
			
		outfile.write('\n\n')

	def _read_dataset(self, infile, line):
		dataset = pdataset()	 
                keepReading = True
		dataset.dataset_name = line.split()[-1]
                while keepReading:      # Read through all cards
                        line = infile.readline()        # get next line
                	key = line.strip().split()[0].lower()   # take first  key word
                        if key == 'dataset_mapped_name':
                                dataset.dataset_mapped_name = line.split()[-1]
                        elif key == 'name':
                                dataset.name= line.split()[-1]
                        elif key == 'filename':
                                dataset.file_name = line.split()[-1]
                        elif key == 'hdf5_dataset_name':
                                dataset.hdf5_dataset_name = line.split()[-1]
                        elif key == 'map_hdf5_dataset_name':
                                dataset.map_hdf5_dataset_name = line.split()[-1]
			elif key == 'max_buffer_size':
				dataset.max_buffer_size = floatD(line.split()[-1])
                        elif key in ['/','end']: keepReading = False
			
		self.add(dataset)
 

	def _write_dataset(self, outfile):
		self._header(outfile,headers['dataset'])
		for dataset in self._datasetlist:
			# Write out dataset variables
			if dataset.dataset_name:
				outfile.write('DATASET ' + dataset.dataset_name + '\n')
			if dataset.dataset_mapped_name:
				outfile.write('DATASET MAPPED ' + dataset.dataset_mapped_name + '\n')
			if dataset.dataset_name and dataset.dataset_mapped_name:
				print 'ERROR: Cannot use both DATASET and DATASET MAPPED'
			if dataset.name:
				outfile.write('  NAME '+dataset.name+'\n')
			if dataset.file_name:
				outfile.write('  FILENAME ' + dataset.file_name + '\n')
			if dataset.hdf5_dataset_name:
				outfile.write('  HDF5_DATASET_NAME ' + dataset.hdf5_dataset_name + '\n')
			if dataset.map_hdf5_dataset_name:
				outfile.write('  MAP_HDF5_DATASET_NAME ' + dataset.map_hdf5_dataset_name + '\n')
			if dataset.max_buffer_size:
				outfile.write('  MAX_BUFFER_SIZE ' + strD(dataset.max_buffer_size) + '\n')
			outfile.write('END\n\n')
      
	def _add_dataset(self,dat=pdataset(),overwrite=False):	#Adds a dataset object.
		# check if dataset already exists
		if isinstance(dat,pdataset):		
			if dat.name in self.dataset.keys():
				if not overwrite:
					warning = 'WARNING: A dataset with name \''+str(dat.name)+'\' already exists. Use overwrite = True in add() to overwrite the old dataset.'
					print warning; print
					_buildWarnings(warning)
					return
				else: # Executes if overwrite = True
					self.delete(self.dat[dataset.name])
					
		if dat not in self._datasetlist:
			self._datasetlist.append(dat)
			
	def _delete_dataset(self,dat=pdataset()):
		self._datasetlist.remove(dat)
 
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
			elif key == 'skip':
				keepReading1 = True
				while keepReading1:
					line1 = infile.readline()
					if line1.strip().split()[0].lower() == 'noskip': keepReading1 = False 
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
			elif key == 'update_porosity':
				chem.update_porosity = True
			elif key == 'update_permeability':
				chem.update_permeability = True
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
					self.delete(self._get_m_kinetic()[m_kinetic.name])
					
		# Add m_kinetic to chemistry (as a sub-class) if that specific 
		# m_kinetic does not exist in chemistry object
		if m_kinetic not in chemistry.m_kinetics_list:
			chemistry.m_kinetics_list.append(m_kinetic)

	def _delete_pchemistry_m_kinetic(self,m_kinetic=pchemistry_m_kinetic()):
		self._chemistry._m_kinetics_list.remove(m_kinetic)	

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
		if c.update_permeability:
			outfile.write('  UPDATE_PERMEABILITY\n')
		if c.update_porosity:
			outfile.write('  UPDATE_POROSITY\n')
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
#						print np_constraint_list_value,line.split()[0].lower()
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
						if tstring[1].lower() == 'dataset':
							mineral.volume_fraction = tstring[2]
							if tstring[3].lower() == 'dataset':
								mineral.surface_area = tstring[4]
							else:
								mineral.surface_area = floatD(tstring[3])
						else:
							mineral.volume_fraction = floatD(tstring[1])
							if tstring[2].lower() == 'dataset':
								mineral.surface_area = tstring[3]
							else:
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
					if type(mineral.volume_fraction) is str:
						outfile.write('  ' + 'DATASET ' + mineral.volume_fraction)
					else:
						outfile.write('  ' + strD(mineral.volume_fraction))
					if type(mineral.surface_area) is str:
						outfile.write('  ' + 'DATASET ' + mineral.surface_area)
					else:
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
	
	def _get_regression(self): return self._regression
	def _set_regression(self, object): self._regression = object
	regression = property(_get_regression, _set_regression) #: (**)	

	def _get_simulation(self): return self._simulation
	def _set_simulation(self, object): self._simulation = object
	simulation = property(_get_simulation, _set_simulation) #: (**)	
	
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

	def _get_datasetlist(self): return self._datasetlist
	datasetlist = property(_get_datasetlist) #: (**) list of datasets 
	def _get_dataset(self): 
		return dict([(p.dataset_name,p) for p in self.datasetlist])
	dataset = property(_get_dataset) #: (**) dictionary of datasets, indexable by ID or name
	

	def _get_saturationlist(self): return self._saturationlist
	saturationlist = property(_get_saturationlist) #: (**) list of saturation functions 
	def _get_saturation(self): 
		return dict([(p.name,p) for p in self.saturationlist])
	saturation = property(_get_saturation) #: (**) dictionary of saturation properties, indexable by ID or name
	
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
	
	def _get_charlist(self): return self._charlist
	charlist = property(_get_charlist)
	def _get_char(self): 
		return dict([(characteristic_curves.name.lower(),characteristic_curves) for characteristic_curves in self.charlist]+[(characteristic_curves.name.lower(),characteristic_curves) for characteristic_curves in self.charlist])
	char = property(_get_char) #: (**) dictionary of material properties, indexable by ID or name

	def _get_regionlist(self): return self._regionlist
	def _set_regionlist(self, object): self._regionlist = object
	regionlist = property(_get_regionlist, _set_regionlist) #: (**)
	def _get_region(self):
		return dict([region.name.lower(),region] for region in self.regionlist if region.name)
	region = property(_get_region)#: (*dict[pregion]*) Dictionary of region objects, indexed by region name.
	
	def _get_observation_list(self): return self._observation_list
	def _set_observation_list(self, object): self._observation_list = object
	observation_list = property(_get_observation_list, _set_observation_list) #: (**)
	def _get_observation(self):
		return dict([observation.region.lower(),observation] for observation in self.observation_list if observation.region)
	observation = property(_get_observation)#: (*dict[pobservation]*) Dictionary of observation objects, indexed by observation region.
	
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
