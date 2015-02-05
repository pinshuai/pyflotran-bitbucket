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


print '\nTEST EXECUTED\n'	# console header

###############################################################

# initialize without reading in test data
#--------------------------------------------------------------
dat = pdata('')
#--------------------------------------------------------------

# set uniform_velocity
#--------------------------------------------------------------
# Comments are alternative way of doing the same thing
#uniform_velocity = puniform_velocity()
#uniform_velocity.value_list = [14.4e0, 0.e0, 0.e0, 'm/yr']
#dat.uniform_velocity = uniform_velocity
dat.uniform_velocity.value_list = [14.4e0, 0.e0, 0.e0, 'm/yr']
#--------------------------------------------------------------

# set chemistry
#--------------------------------------------------------------
c = pchemistry()
c.pspecies_list = ['A(aq)']
c.molal = True
c.output_list = ['All','Free_ion']
dat.chemistry = c
#--------------------------------------------------------------

# set grid
#--------------------------------------------------------------
g = pgrid()
g.type = 'structured'
g.lower_bounds = [0.e0, 0.e0, 0.e0]
g.upper_bounds = [0.04e0, 1.e0, 1.e0]
g.bounds_bool = True
g.orign = [0.e0, 0.e0, 0.e0]
g.nxyz = [100, 1, 1]
dat.grid = g
#--------------------------------------------------------------

# set time stepping
#--------------------------------------------------------------
ts = ptimestepper()
ts.ts_acceleration = 25
ts.max_ts_cuts = 10
ts.max_steps = 10000
dat.timestepper = ts
#--------------------------------------------------------------

# set newton solvers
#--------------------------------------------------------------
newton_solver = pnsolver()
newton_solver.name = 'TRANSPORT'
newton_solver.atol = 1e-15
newton_solver.rtol = 1e-10
newton_solver.stol = 1e-30
newton_solver.dtol = None
newton_solver.itol = 1e-8
newton_solver.max_it = 100
newton_solver.max_f = 100
dat.add(newton_solver)
#--------------------------------------------------------------

# set fluid properties
#--------------------------------------------------------------
f = pfluid()
f.diffusion_coefficient = 1.e-9
dat.fluid = f
#--------------------------------------------------------------

# set material properties aka prop_list
#--------------------------------------------------------------
material_property = pmaterial()	# For assigning defaults
material_property.name = 'soil1'
material_property.id = 1
material_property.porosity = 1.e0
material_property.tortuosity = 1.e0
material_property.density = 2.8e3		# Rock Density
material_property.specific_heat = 1e3
material_property.cond_dry 		# Defaults are assigned
material_property.cond_wet = 0.5		# THERMAL_CONDUCTIVITY_WET
material_property.saturation = 'default'
material_property.permeability = [1.e-15,1.e-15,1.e-15]
dat.add(material_property)
#--------------------------------------------------------------

# set time
#--------------------------------------------------------------
time = ptime()
time.tf = [1.e4, 's']		# FINAL_TIME
time.dti = [1.e-6, 's']		# INITIAL_TIMESTEP_SIZE
time.dtf = [10.e0,'s']		# MAXIMUM_TIMESTEP_SIZE
time.dtf_list.append([1.e2, 's', 5.e3, 's'])
time.dtf_list.append([1.e3, 's', 5.e4, 's'])
dat.time = time
#--------------------------------------------------------------

# set output
#--------------------------------------------------------------
output = poutput()
output.time_list = ['s', 26042.0, 39063.0, 52083.0, 1000000.0]
output.periodic_observation_timestep = 1
output.print_column_ids = True
output.format_list.append('TECPLOT POINT')
dat.output = output
#--------------------------------------------------------------

# set saturation functions
#--------------------------------------------------------------
saturation = psaturation()
saturation.name = 'default'
saturation.saturation_function_type = 'VAN_GENUCHTEN'
saturation.residual_saturation_liquid = 0.1
saturation.residual_saturation_gas = 0.0
saturation.a_lambda = 0.762e0
saturation.alpha = 7.5e-4
saturation.max_capillary_pressure = 1.e6
dat.saturation = saturation
#--------------------------------------------------------------

# set regions
#--------------------------------------------------------------
region = pregion()
region.name = 'all'
region.face = None
region.coordinates_lower = [0.e0, 0.e0, 0.e0]
region.coordinates_upper = [0.04e0, 1.e0,  1.e0]
dat.add(region)

region = pregion()
region.name = 'west'
region.face = 'WEST'
region.coordinates_lower = [0.e0, 0.e0, 0.e0]
region.coordinates_upper = [0.e0, 1.e0,  1.e0]
dat.add(region)

region = pregion()
region.name = 'east'
region.face = 'EAST'
region.coordinates_lower = [0.04e0, 0.e0, 0.e0]
region.coordinates_upper = [0.04e0, 1.e0, 1.e0]
dat.add(region)

region = pregion()
region.name = 'obs'
region.face = None
region.coordinates_lower = [0.04e0, 0.e0, 0.e0]
region.coordinates_upper = [0.04e0, 1.e0, 1.e0]
dat.add(region)
#--------------------------------------------------------------

# set observations
#--------------------------------------------------------------
observation = pobservation()
observation.region = 'obs'
dat.add(observation)
#--------------------------------------------------------------

# set transport conditions
#--------------------------------------------------------------
transport_condition = ptransport()
transport_condition.name = 'initial'
transport_condition.type = 'dirichlet'
transport_condition.constraint_list_value = [0.e0]
transport_condition.constraint_list_type = ['initial']
dat.add(transport_condition)

transport_condition = ptransport()
transport_condition.name = 'WEST'
transport_condition.type = 'dirichlet'
transport_condition.constraint_list_value = [0.e0]
transport_condition.constraint_list_type = ['WEST']
dat.add(transport_condition)

transport_condition = ptransport()
transport_condition.name = 'east'
transport_condition.type = 'ZERO_gradient'
transport_condition.constraint_list_value = [0.e0]
transport_condition.constraint_list_type = ['east']
dat.add(transport_condition)
#--------------------------------------------------------------

# set initial condition
#--------------------------------------------------------------
initial_condition = pinitial_condition()
initial_condition.flow = 'INITIAL'
initial_condition.transport = 'initial'
initial_condition.region = 'all'
dat.initial_condition = initial_condition
#--------------------------------------------------------------

# set boundary conditions
#--------------------------------------------------------------
boundary_condition = pboundary_condition()
boundary_condition.name = ''
boundary_condition.flow = 'west'
boundary_condition.transport = 'west'
boundary_condition.region = 'WEST'
dat.add(boundary_condition)

boundary_condition = pboundary_condition()
boundary_condition.name = ''
boundary_condition.flow = 'east'
boundary_condition.transport = 'EAST'
boundary_condition.region = 'east'
dat.add(boundary_condition)
#--------------------------------------------------------------

# set stratigraphy couplers
#--------------------------------------------------------------
stratigraphy_coupler = pstrata()
stratigraphy_coupler.region = 'ALL' 
stratigraphy_coupler.material = 'SOIL1'
dat.add(stratigraphy_coupler)
#--------------------------------------------------------------

# set constraints
#--------------------------------------------------------------
# initial condition - 1st condition
constraint = pconstraint()
constraint.name = 'initial'
constraint.concentration_list = [] # Assigning for this done below
dat.add(constraint)	# assign constraint
concentration = pconstraint_concentration()	# new concentration object
concentration.pspecies = 'A(aq)'
concentration.value = 0.1
concentration.constraint = 'T'
#constraint.concentration_list.append(concentration)	# assign concentration
dat.add(concentration)		# assign_concentration

# west condition - 2nd condition
constraint = pconstraint()
constraint.name = 'WEST'
constraint.concentration_list = [] # Assigning for this done below
dat.add(constraint)	# assign constraint
concentration = pconstraint_concentration()	# new concentration object
concentration.pspecies = 'A(aq)'
concentration.value = 1.e-8
concentration.constraint = 'T'
dat.add(concentration,'west')		# assign_concentration - alternative

# east condition - 3rd condition
constraint = pconstraint()
constraint.name = 'east'
constraint.concentration_list = [] # Assigning for this done below
dat.add(constraint)
concentration = pconstraint_concentration()	# new concentration object
concentration.pspecies = 'A(aq)'
concentration.value = 1.E-02
concentration.constraint = 'T'
dat.add(concentration,constraint)		# assign_concentration - alternative
#--------------------------------------------------------------

###############################################################

# Test Write
#dat.write('tracer_1D.in')

# Write to File
dat.run(input='tracer_1D.in',exe=pflotran_dir + '/src/pflotran/pflotran')
