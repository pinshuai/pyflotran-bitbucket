"""
This script is not intended to be executed by itself.
It is a template that demonstrates what PyFLOTRAN is currently capable of.
"""

import sys
sys.path.append('../../.')
from pdata import*
import os
try:
  pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
  print('PFLOTRAN_DIR must point to PFLOTRAN installation directory and be defined in system environment variables.')
  sys.exit(1)
sys.path.append(pflotran_dir + '/src/python')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import pflotran as pft

print('******************************************')
print('Using PyFLOTRAN')
print('******************************************')
###############################################################

# Read input file
dat = pdata('input_deck_name.in')

# initialize without reading in test data
#--------------------------------------------------------------
dat = pdata('')
#--------------------------------------------------------------

# set uniform_velocity
#--------------------------------------------------------------
dat.uniform_velocity.value_list = [] # assign uniform velocity object
#--------------------------------------------------------------

# set mode
#--------------------------------------------------------------
mode = pmode()  # new mode object
mode.name = ''
dat.mode = mode # assign mode object
#--------------------------------------------------------------

# set chemistry
#--------------------------------------------------------------
chemistry = pchemistry() # new chemistry object
chemistry.pspecies_list = []
chemistry.sec_species_list = []
chemistry.gas_species_list = []
chemistry.minerals_list = [] # mineral_kinetic objects are appended to this list
mineral_kinetic = pchemistry_m_kinetic() # new mineral kinetic object, sub-class of chemistry
mineral_kinetic.name = None
mineral_kinetic.rate_constant_list = []
chemistry.m_kinetics_list.append(mineral_kinetic) # append mineral kinetic object to chemistry
chemistry.log_formulation = False
chemistry.database = pflotran_dir + 'database/hanford.dat'
chemistry.activity_coefficients = None
chemistry.molal = False
chemistry.output_list = []
dat.chemistry = chemistry # assign chemistry object
#--------------------------------------------------------------

# set co2 database
#--------------------------------------------------------------
dat.co2_database = del_extra_slash(pflotran_dir + '/database/co2data0.dat')
#--------------------------------------------------------------

# set grid
#--------------------------------------------------------------
grid = pgrid()  # new grid object
grid.type = ''
grid.lower_bounds = []
grid.upper_bounds = []
grid.orign = []
grid.nxyz = []
grid.dxyz = []
grid.gravity = []
grid.filename = ''
dat.grid = grid # assign grid object
#--------------------------------------------------------------

# set time
#--------------------------------------------------------------
time = ptime()  # new time object
time.tf = []
time.dti = []
time.dtf = []
time.dtf_lv = []
time.dtf_lv_unit = [] 
time.dtf_li = []
time.dtf_li_unit = []
dat.time = time # assign time object
#--------------------------------------------------------------


# set time stepping
#--------------------------------------------------------------
time_stepper = ptimestepper() # new time stepper object
time_stepper.ts_acceleration = None
time_stepper.num_steps_after_cut = None
time_stepper.max_steps = None
time_stepper.max_ts_cuts = None
time_stepper.cfl_limiter = None
time_stepper.initialize_to_steady_state = None
time_stepper.run_as_steady_state = None
time_stepper.max_pressure_change = None
time_stepper.max_temperature_change = None
time_stepper.max_concentration_change = None
time_stepper.max_saturation_change = None
dat.timestepper = time_stepper # assign time stepper object
#--------------------------------------------------------------

# set material properties aka prop_list
#--------------------------------------------------------------
material = pmaterial('','') # new material properties object
material.id = None
material.name = ''
material.porosity = None
material.tortuosity = None
material.density = None
material.specific_heat = None
material.cond_dry = None
material.cond_wet = None
material.saturation = ''
material.permeability = []
dat.add(material)          # append material properties object
#--------------------------------------------------------------

# set linear solvers
#--------------------------------------------------------------
linear_solver = plsolver() # new linear solver object
linear_solver.name = ''
linear_solver.solver = ''
dat.add(linear_solver)     # append linear solver object
#--------------------------------------------------------------

# set newton solvers
#--------------------------------------------------------------
newton_solver = pnsolver('') # new newton solver object
newton_solver.name = ''
newton_solver.atol = None
newton_solver.rtol = None
newton_solver.stol = None
newton_solver.dtol = None
newton_solver.itol = None
newton_solver.max_it = None
newton_solver.max_f = None
dat.add(newton_solver)       # append newton solver object
#--------------------------------------------------------------

# set output
#--------------------------------------------------------------
output = poutput()  # new output object
output.time_list = []
output.mass_balance = False
output.print_column_ids = False
output.periodic_observation_timestep = None
output.format_list = []
output.velocities = False
dat.output = output # assign output object
#--------------------------------------------------------------

# set fluid properties
#--------------------------------------------------------------
fluid = pfluid()  # new fluid object
fluid.diffusion_coefficient = None
dat.fluid = fluid # assign fluid object
#--------------------------------------------------------------

# set saturation functions
#--------------------------------------------------------------
saturation = psaturation()  # new saturation object
saturation.name = ''
saturation.permeability_function_type = ''
saturation.saturation_function_type = ''
saturation.residual_saturation_liquid = None
saturation.residual_saturation_gas = None
saturation.a_lambda = None
saturation.alpha = None
saturation.max_capillary_pressure = None
saturation.betac = None
saturation.power = None
dat.saturation = saturation # assign saturation object
#--------------------------------------------------------------

# set observation
#--------------------------------------------------------------
observation = pobservation()  # new observation object
observation.region = ''
dat.observation = observation # append observation object
#--------------------------------------------------------------

# set regions
#--------------------------------------------------------------
region = pregion() # new region object
region.name = ''
region.coordinates_lower = []
region.coordinates_upper = []
region.face = None
dat.add(region)    # append region object
#--------------------------------------------------------------

# set transport conditions
#--------------------------------------------------------------
transport_condition = ptransport() # new transport object
transport_condition.name = ''
transport_condition.type = ''
transport_condition.constraint_list_value = []
transport_condition.constraint_list_type = []
dat.add(transport_condition)       # append transport object
#--------------------------------------------------------------

# set flow conditions
#--------------------------------------------------------------
# initial flow condition
flow = pflow() # new flow object
flow.name = ''
flow.units_list = None
flow.iphase = None
flow.sync_timestep_with_update = False
flow.varlist = [] # variable objects are appended to this list
dat.add(flow)	  # append flow object
variable = pflow_variable('') # new variable object, sub-class of flow object
variable.name = ''
variable.type = ''
variable.valuelist = []
dat.add(variable) 	      # append variable object to flow object
variable = pflow_variable('') # new flow var object, sub-class of flow object, list format
variable.name = ''
variable.type = ''
variable.time_unit_type = ''
variable.data_unit_type = ''
variable.list = []            # flow variable list objects are appended to this list
dat.add(variable)             # append variable object to flow object
var_list = pflow_variable_list()   # new flow variable list object, sub-class of variable object
var_list.time_unit_value = None
var_list.data_unit_value_list = []
variable.list.append(var_list)     # append flow variable list object to variable object

#--------------------------------------------------------------

# set initial condition
#--------------------------------------------------------------
initial_condition = pinitial_condition()  # new initial condition object
initial_condition.flow = ''
initial_condition.transport = ''
initial_condition.region = ''
dat.initial_condition = initial_condition # assign initial condition object
#--------------------------------------------------------------

# set boundary conditions
#--------------------------------------------------------------
boundary_condition = pboundary_condition() # new boundary condition object
boundary_condition.name = ''
boundary_condition.flow = ''
boundary_condition.transport = None
boundary_condition.region = ''
dat.add(boundary_condition)                # append boundary condition object
#--------------------------------------------------------------

# set source sink
#--------------------------------------------------------------
source_sink = psource_sink()  # new source sink object
source_sink.flow = ''
source_sink.region = ''
dat.source_sink = source_sink # assign source sink object
#--------------------------------------------------------------

# set stratigraphy couplers
#--------------------------------------------------------------
stratigraphy_coupler = pstrata() # new stratigraphy coupler object
stratigraphy_coupler.region = '' 
stratigraphy_coupler.material = ''
dat.add(stratigraphy_coupler)    # append stratigraphy coupler object
#--------------------------------------------------------------

# set constraints
#--------------------------------------------------------------
constraint = pconstraint()         # new constraint object
constraint.name = ''
constraint.concentration_list = [] # concentration objects are appended to this list
constraint.mineral_list = []       # mineral objects are appended to this list
dat.add(constraint)	           # append constraint object
concentration = pconstraint_concentration()	    # new concentration object
concentration.pspecies = ''
concentration.value = None
concentration.constraint = ''
concentraiton.element = ''
constraint.concentration_list.append(concentration) # append concentration to constraint object
mineral = pconstraint_mineral()			    # new mineral object
mineral.name = ''
mineral.volume_fraction = None
mineral.surface_area = None
constraint.mineral_list.append(mineral)	            # append mineral object to constraint object
#--------------------------------------------------------------

# Test write - Use to run PyFlOTRAN only. (Comment out when running PFLOTRAN)
#dat.write('input_deck_name')

print('******************************************')
print('Writing PFLOTRAN input file and executing it.')
print('******************************************')

# Execute PyFLOTRAN/PFLOTRAN
pflotran_exe = del_extra_slash(pflotran_dir + '/src/pflotran/pflotran')
# Write to file and execute that input file
dat.run(input='input_deck_name.in',exe=pflotran_exe)