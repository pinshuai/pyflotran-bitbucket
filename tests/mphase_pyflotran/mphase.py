import sys
import os

try:
    pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
    print('PFLOTRAN_DIR must point to PFLOTRAN installation directory and be defined in system environment variables.')
    sys.exit(1)
sys.path.append(pflotran_dir + '/src/python')

try:
    pyflotran_dir = os.environ['PYFLOTRAN_DIR']
except KeyError:
    print(
        'PYFLOTRAN_DIR must point to PYFLOTRAN installation directory and be defined in system environment variables.')
    sys.exit(1)
sys.path.append(pyflotran_dir)
from pdata import *

test_dir = '/tests/mphase_pyflotran'

###############################################################

# initialize without reading in test data
# --------------------------------------------------------------
dat = pdata('', work_dir=pyflotran_dir + test_dir)
# --------------------------------------------------------------
# set simulation
# --------------------------------------------------------------
simulation = psimulation()
simulation.simulation_type = 'subsurface'
simulation.subsurface_flow = 'flow'
simulation.mode = 'mphase'
dat.simulation = simulation


# --------------------------------------------------------------

# set co2 database
# --------------------------------------------------------------
dat.co2_database = del_extra_slash(pflotran_dir + '/database/co2data0.dat')
# --------------------------------------------------------------

# set checkpoint # testing
# --------------------------------------------------------------
# dat.checkpoint.frequency = 1000 # testing
# --------------------------------------------------------------

# set restart - Has not been tested with PFLOTRAN - testing
# --------------------------------------------------------------
# restart = prestart() # testing
# restart.file_name = 'restart.chk' # testing
# restart.time_value = 5.e0 # testing
# restart.time_unit = 'y' # testing
# dat.restart = restart # testing
# --------------------------------------------------------------

# set grid
# --------------------------------------------------------------
grid = pgrid()
grid.type = 'structured'
grid.lower_bounds = [0.e0, 0.e0, 0.e0]
grid.upper_bounds = [321.e0, 1.e0, 51.e0]
grid.nxyz = [107, 1, 51]
grid.gravity = [0.0, 0.0, -9.8068]  # Should not write
grid.filename = ''
dat.grid = grid
# --------------------------------------------------------------

# set time stepping
# --------------------------------------------------------------
ts = ptimestepper()
ts.ts_acceleration = 8
dat.timestepper = ts
# --------------------------------------------------------------

# set material properties aka prop_list
# --------------------------------------------------------------
material = pmaterial('', '')
material.id = 1
material.name = 'soil1'
material.porosity = 0.15e0
material.tortuosity = 1e-1
material.density = 2.65E3  # ROCK_DENSITY
material.specific_heat = 1E3
material.cond_dry = 0.5  # THERMAL_CONDUCTIVITY_DRY
material.cond_wet = 0.5  # THERMAL_CONDUCTIVITY_WET
material.saturation = 'sf2'
material.permeability = [1.e-15, 1.e-15, 1.e-17]
dat.add(material)
# --------------------------------------------------------------

# set time
# --------------------------------------------------------------
time = ptime()
time.tf = [0.25e0, 'y']  # FINAL_TIME
time.dti = [1.e-6, 'y']  # INITIAL_TIMESTEP_SIZE
time.dtf = [50.e0, 'y']  # MAXIMUM_TIMESTEP_SIZE
time.dtf_list.append([200.e0, 'y', 50., 'y'])
time.dtf_list.append([500.e0, 'y', 20000., 'y'])
time.dtf_list.append([1000.e0, 'y', 50000., 'y'])
time.dtf_list.append([5000.e0, 'y', 100000., 'y'])
dat.time = time
# --------------------------------------------------------------

# set newton solvers
# --------------------------------------------------------------
newton_solver = pnsolver('')
newton_solver.name = 'FLOW'
newton_solver.atol = 1e-12
newton_solver.rtol = 1e-12
newton_solver.stol = 1e-30
newton_solver.dtol = 1e15
newton_solver.itol = 1e-8
newton_solver.max_it = 25
newton_solver.max_f = 100
dat.add(newton_solver)

# --------------------------------------------------------------

# set output
# --------------------------------------------------------------
output = poutput()
output.print_column_ids = True
# output.screen_periodic = 2 # testing
# output.periodic_time = [4.e1, 'y'] # testing
# output.periodic_timestep = [4.e1, 'y'] # testing - Does not work in PFLOTRAN
# output.periodic_observation_time = [3.2e1, 'y']# testing
# output.permeability = True # testing
# output.porosity = True # testing
output.mass_balance = True
output.periodic_observation_timestep = 1
output.format_list.append('TECPLOT POINT')
output.format_list.append('VTK')
output.format_list.append('HDF5')
dat.output = output
# --------------------------------------------------------------

# set fluid properties
# --------------------------------------------------------------
fluid = pfluid()
fluid.diffusion_coefficient = 1.e-9
dat.fluid = fluid
# --------------------------------------------------------------

# set saturation functions
# --------------------------------------------------------------
saturation = psaturation('', '')
saturation.name = 'sf2'
saturation.permeability_function_type = 'NMT_EXP'
saturation.saturation_function_type = 'NMT_EXP'
# saturation.residual_saturation = 0.03 # float - testing - does not work with pflotran with this deck.
saturation.residual_saturation_liquid = 0.1
saturation.residual_saturation_gas = 0.0
saturation.a_lambda = 0.762
saturation.alpha = 7.5e-4
saturation.max_capillary_pressure = 1.e6
saturation.betac = 2.e0
saturation.power = 7.e0
dat.add(saturation)
# --------------------------------------------------------------

# set regions
# --------------------------------------------------------------
region = pregion()
region.name = 'all'
region.coordinates_lower = [0.e0, 0.e0, 0.e0]
region.coordinates_upper = [321.e0, 1.e0, 51.e0]
dat.add(region)

region = pregion()
region.name = 'top'
region.face = 'top'
region.coordinates_lower = [0.e0, 0.e0, 51.e0]
region.coordinates_upper = [321.e0, 1.e0, 51.e0]
dat.add(region)

region = pregion()
region.name = 'west'
region.face = 'WEST'
region.coordinates_lower = [0.e0, 0.e0, 0.e0]
region.coordinates_upper = [0.e0, 1.e0, 51.e0]
dat.add(region)

region = pregion()
region.name = 'EAST'
region.face = 'east'
region.coordinates_lower = [321.e0, 0.e0, 0.e0]
region.coordinates_upper = [321.e0, 1.e0, 51.e0]
dat.add(region)

region = pregion()
region.name = 'well'
region.coordinates_lower = [160.e0, 1.e0, 20.e0]
region.coordinates_upper = [160.e0, 1.e0, 20.e0]
dat.add(region)
# --------------------------------------------------------------

# set flow conditions
# --------------------------------------------------------------
# initial flow condition
flow = pflow('')
flow.name = 'initial'
flow.units_list = None
flow.iphase = 1
flow.sync_timestep_with_update = False
flow.datum.append([3., 5., 2.])  # testing - not tested with PFLOTRAN
# flow.datum.append([2., 1., 6.])	 # testing - not tested with PFLOTRAN
# flow.datum = 'file_name' 		 # testing - not tested with PFLOTRAN
# flow.varlist = [] 	# Assigning for this done below
dat.add(flow)
# adding flow_variable to inital flow_condition
variable = pflow_variable('')  # new flow var object
variable.name = 'pressure'
variable.type = 'hydrostatic'
variable.valuelist = [2e7, 2e7]
dat.add(variable)
# adding flow_variable to inital flow_condition
variable = pflow_variable('')  # new flow var object
variable.name = 'TEMPERATURE'
variable.type = 'zero_gradient'
variable.valuelist = [50.0]
# variable.unit = 'C'
dat.add(variable)
# adding flow_variable to inital flow_condition
variable = pflow_variable('')  # new flow var object
variable.name = 'CONCENTRATION'
variable.type = 'zero_gradient'
variable.valuelist = [1e-6]
# variable.unit = 'm'
dat.add(variable, flow)  # alternative
# adding flow_variable to inital flow_condition
variable = pflow_variable('')  # new flow var object
variable.name = 'ENTHALPY'
variable.type = 'dirichlet'
variable.valuelist = [0.e0, 0.e0]  # alternative, specify flow object by passing in direct reference
dat.add(variable, index='initial')  # alternative, specify flow object by its' name

# top flow condition
flow = pflow()
flow.name = 'top'
flow.iphase = 1
flow.datum = []
flow.varlist = []  # Assigning for this done below
dat.add(flow)
# adding flow_variable to top flow_condition
variable = pflow_variable()  # new flow variable object
variable.name = 'pressure'
variable.type = 'dirichlet'
variable.valuelist = [3e7, 2e7]
dat.add(variable)
# adding flow_variable to top flow_condition
variable = pflow_variable()  # new flow variable object
variable.name = 'temperature'
variable.type = 'zero_gradient'
variable.valuelist = [60.0]
dat.add(variable)  # assigning for flow var done here
# adding flow_variable to top flow_condition
variable = pflow_variable()  # new flow variable object
variable.name = 'concentration'
variable.type = 'zero_gradient'
variable.valuelist = [1e-6]
dat.add(variable)  # assigning for flow var done here
# adding flow_variable to top flow_condition
variable = pflow_variable()  # new flow variable object
variable.name = 'enthalpy'
variable.type = 'dirichlet'
variable.valuelist = [0.e0, 0.e0]
dat.add(variable)  # assigning for flow var done here


# source flow condition
flow = pflow('')
flow.name = 'source'
flow.units_list = None
flow.iphase = None
flow.sync_timestep_with_update = True
flow.datum = []
flow.varlist = []  # Assigning for this done below
dat.add(flow)  # Assigning for flow condition done here
# adding flow_variable to source flow_condition
variable = pflow_variable('')  # new flow var object
variable.name = 'rate'
variable.type = 'mass_rate'
variable.time_unit_type = 'y'
variable.data_unit_type = 'kg/s'
tlist = pflow_variable_list()  # sub-class of pflow_variable, assigned to list attribute
tlist.time_unit_value = 0.  # tlist = temporary list
tlist.data_unit_value_list = [0., 1.e-4]
variable.list.append(tlist)
tlist = pflow_variable_list()  # sub-class of pflow_variable, assigned to list attribute
tlist.time_unit_value = 10.
tlist.data_unit_value_list = [0., 0.]
variable.list.append(tlist)
dat.add(variable)  # assigning for flow var done here
# adding flow_variable to source flow_condition
variable = pflow_variable('')  # new flow var object
variable.name = 'pressure'
variable.type = 'dirichlet'
variable.valuelist = [4e7, 2e7]
variable.list = []
variable.unit = None
flow.varlist.append(variable)  # assigning for flow var done here
variable = pflow_variable('')  # new flow var object
variable.name = 'TEMPERATURE'
variable.type = 'dirichlet'
variable.valuelist = [70.0]
variable.list = []
variable.unit = None
dat.add(variable)  # assigning for flow var done here
# adding flow_variable to source flow_condition
variable = pflow_variable('')  # new flow var object
variable.name = 'CONCENTRATION'
variable.type = 'dirichlet'
variable.valuelist = [0.e0]
variable.list = []
variable.unit = None
dat.add(variable)  # assigning for flow var done here
# adding flow_variable to source flow_condition
variable = pflow_variable('')  # new flow var object
variable.name = 'ENTHALPY'
variable.type = 'dirichlet'
variable.valuelist = [0.e0, 0.e0]
variable.list = []
variable.unit = None
dat.add(variable)  # assigning for flow var done here

# --------------------------------------------------------------

# set initial condition
# --------------------------------------------------------------
ic = pinitial_condition()
ic.name = 'initial'
ic.flow = 'INITIAL'
ic.region = 'all'
dat.add(ic)
# --------------------------------------------------------------

# set boundary conditions
# --------------------------------------------------------------
boundary_condition = pboundary_condition()
boundary_condition.name = 'WEST'
boundary_condition.flow = 'INITIAL'
boundary_condition.transport = None
boundary_condition.region = 'west'
dat.add(boundary_condition)

boundary_condition = pboundary_condition()
boundary_condition.name = 'east'
boundary_condition.flow = 'INITIAL'
boundary_condition.transport = None
boundary_condition.region = 'east'
dat.add(boundary_condition)
# --------------------------------------------------------------

# set source sink
# --------------------------------------------------------------
ss = psource_sink()
ss.name = 'injection_well'
ss.flow = 'source'
ss.region = 'WELL'
dat.add(ss)
# --------------------------------------------------------------

# set stratigraphy couplers
# --------------------------------------------------------------
stratigraphy_coupler = pstrata()
stratigraphy_coupler.region = 'ALL'
stratigraphy_coupler.material = 'SOIL1'
dat.add(stratigraphy_coupler)
# --------------------------------------------------------------

# Test write
# dat.write('mphase.in')

pflotran_exe = del_extra_slash(pflotran_dir + '/src/pflotran/pflotran')
# Write to file and execute that input file
dat.run(input='mphase.in', exe=pflotran_exe)
