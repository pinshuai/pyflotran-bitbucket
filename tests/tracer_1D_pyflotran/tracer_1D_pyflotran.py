
from pdata import*

print '\nTEST EXECUTED\n'	# console header

###############################################################

# initialize without reading in test data
#--------------------------------------------------------------
dat = pdata('')
#--------------------------------------------------------------

# set uniform_velocity
#--------------------------------------------------------------
dat.uniform_velocity.value_list = [14.4e0, 0.e0, 0.e0, 'm/yr']
#--------------------------------------------------------------

# set chemistry
#--------------------------------------------------------------
dat.chemistry = pchemistry()

dat.chemistry.pspecies = 'A(aq)'
dat.chemistry.molal = True
dat.chemistry.output = 'ALL'
#--------------------------------------------------------------

# set grid
#--------------------------------------------------------------
dat.grid.type = 'structured'
dat.grid.lower_bounds = [0.e0, 0.e0, 0.e0]
dat.grid.upper_bounds = [0.04e0, 1.e0, 1.e0]
dat.grid.bounds_bool = True
dat.grid.orign = [0.e0, 0.e0, 0.e0]
dat.grid.nxyz = [100, 1, 1]
#dat.grid.dxyz = [5, 5, 5]	# Should not write
#dat.grid.gravity_bool = False
#dat.grid.gravity =  [0.0, 0.0, -9.8068]	# Should not write
#dat.grid.filename =  ''
#--------------------------------------------------------------

# set time stepping
#--------------------------------------------------------------
dat.timestepper.ts_acceleration = 25
dat.timestepper.num_steps_after_cut = None
dat.timestepper.max_ts_cuts = 10
dat.timestepper.max_steps = 10000
dat.timestepper.cfl_limiter = None

dat.timestepper.initialize_to_steady_state = None
dat.timestepper.run_as_steady_state = None
dat.timestepper.max_pressure_change = None
dat.timestepper.max_temperature_change = None
dat.timestepper.max_concentration_change = None
dat.timestepper.max_saturation_change = None
#--------------------------------------------------------------

# set newton solvers
#--------------------------------------------------------------
name = 'TRANSPORT'
atol = 1e-15
rtol = 1e-10
stol = 1e-30
dtol = None
itol = 1e-8
max_it = 100
max_f = 100

nsolver = pnsolver(name, atol, rtol, stol, dtol, itol, max_it, max_f)
dat.nsolverlist.append(nsolver)
#--------------------------------------------------------------

# set fluid properties
#--------------------------------------------------------------
dat.fluid.diffusion_coefficient = 1.e-9
#--------------------------------------------------------------

# set material properities aka prop_list
#--------------------------------------------------------------
p = pmaterial(0, '')	# For assigning defaults

name = 'soil1'
id = 1
porosity = 1.e0
tortuosity = 1.e0
density = 2.8e3		# Rock Density
specific_heat = 1e3
cond_dry = p.cond_dry	# Defaults are assigned
cond_wet = 0.5		# THERMAL_CONDUCTIVITY_WET
saturation = 'default'
permeability = [1.e-15,1.e-15,1.e-15]
	
material = pmaterial(id, name, porosity, tortuosity, density, specific_heat,
		     cond_dry, cond_wet, saturation, permeability)
dat.proplist.append(material)
#--------------------------------------------------------------

# set time
#--------------------------------------------------------------
dat.time.tf = [1.e6, 's']	# FINAL_TIME
dat.time.dti = [1.e-6, 's']	# INITIAL_TIMESTEP_SIZE
dat.time.dtf = [10.e0,'s']	# MAXIMUM_TIMESTEP_SIZE
dat.time.dtf_lv = [1.e2, 1.e3]	#MAXIMUM_TIMESTEP_SIZE before 'at'
dat.time.dtf_lv_unit = ['s', 's']		# time unit
dat.time.dtf_li = [5.e3, 5.e4]	#MAXIMUM_TIMESTEP_SIZE after 'at'
dat.time.dtf_li_unit = ['s', 's']		# time unit
dat.time.dtf_i = 2	# Needs to be the length of dtf_lv and dtf_li
#--------------------------------------------------------------

# set output
#--------------------------------------------------------------
dat.output.time_list = ['s', 26042.0, 39063.0, 52083.0, 1000000.0]
dat.output.periodic_observation_timestep = 1
dat.output.print_column_ids = True
dat.output.format.append('TECPLOT POINT')
#--------------------------------------------------------------

# set saturation functions
#--------------------------------------------------------------
dat.saturation.name = 'default'
#dat.saturation.permeability_function_type = 'NMT_EXP'
dat.saturation.saturation_function_type = 'VAN_GENUCHTEN'
dat.saturation.residual_saturation_liquid = 0.1
dat.saturation.residual_saturation_gas = 0.0
dat.saturation.a_lambda = 0.762e0
dat.saturation.alpha = 7.5e-4
dat.saturation.max_capillary_pressure = 1.e6
#dat.saturation.betac = 2.e0
#dat.saturation.power = 7.e0
#--------------------------------------------------------------

# set regions
#--------------------------------------------------------------
name = 'all'
face = None
coordinates_lower = [0.e0, 0.e0, 0.e0]
coordinates_upper = [0.04e0, 1.e0,  1.e0]

region = pregion(name, coordinates_lower, coordinates_upper, face)
dat.regionlist.append(region)

name = 'west'
face = 'WEST'
coordinates_lower = [0.e0, 0.e0, 0.e0]
coordinates_upper = [0.e0, 1.e0,  1.e0]

region = pregion(name, coordinates_lower, coordinates_upper, face)
dat.regionlist.append(region)

name = 'east'
face = 'EAST'
coordinates_lower = [0.04e0, 0.e0, 0.e0]
coordinates_upper = [0.04e0, 1.e0, 1.e0]

region = pregion(name, coordinates_lower, coordinates_upper, face)
dat.regionlist.append(region)

name = 'obs'
face = None
coordinates_lower = [0.04e0, 0.e0, 0.e0]
coordinates_upper = [0.04e0, 1.e0, 1.e0]

region = pregion(name, coordinates_lower, coordinates_upper, face)
dat.regionlist.append(region)
#--------------------------------------------------------------

# set observation
#--------------------------------------------------------------
dat.observation = pobservation()

dat.observation.region = 'obs'
#--------------------------------------------------------------

# set transport conditions
#--------------------------------------------------------------
name = 'initial'
type = 'dirichlet'
constraint_list_value = [0.e0]
constraint_list_type = ['initial']

t = ptransport(name,type,constraint_list_value,constraint_list_type)
dat.transportlist.append(t)

name = 'WEST'
type = 'dirichlet'
constraint_list_value = [0.e0]
constraint_list_type = ['WEST']

t = ptransport(name,type,constraint_list_value,constraint_list_type)
dat.transportlist.append(t)

name = 'east'
type = 'ZERO_gradient'
constraint_list_value = [0.e0]
constraint_list_type = ['east']

t = ptransport(name,type,constraint_list_value,constraint_list_type)
dat.transportlist.append(t)
#--------------------------------------------------------------

# set initial condition
#--------------------------------------------------------------
dat.initial_condition.flow = 'INITIAL'
dat.initial_condition.transport = 'initial'
dat.initial_condition.region = 'all'
#--------------------------------------------------------------

# set boundary conditions
#--------------------------------------------------------------
name = None
flow = 'west'
transport = 'west'
region = 'WEST'

b = pboundary_condition(name, flow, transport, region)
dat.boundary_condition_list.append(b)

name = None
flow = 'east'
transport = 'EAST'
region = 'east'

b = pboundary_condition(name, flow, transport, region)
dat.boundary_condition_list.append(b)
#--------------------------------------------------------------

# set stratigraphy couplers
#--------------------------------------------------------------
dat.strata.region = 'ALL' 
dat.strata.material = 'SOIL1'

#--------------------------------------------------------------

# set constraints
#--------------------------------------------------------------
# initial condition - 1st condition
name = 'initial'
concentration_list = [] # Assigning for this done below

pspecies = 'A(aq)'
value = 0.1
constraint = 'T'
concentration = pconstraint_concentration(pspecies, value, constraint)
concentration_list.append(concentration)

constraint = pconstraint(name, concentration_list)
dat.constraint_list.append(constraint)	# Assigning done here

# west condition - 2nd condition
name = 'WEST'
concentration_list = [] # Assigning for this done below

# concentration
pspecies = 'A(aq)'
value = 1.e-8
constraint = 'T'
concentration = pconstraint_concentration(pspecies, value, constraint)
concentration_list.append(concentration)

constraint = pconstraint(name, concentration_list)
dat.constraint_list.append(constraint)	# Assigning done here

# east condition - 3rd condition
name = 'east'
concentration_list = [] # Assigning for this done below

# concentration
pspecies = 'A(aq)'
value = 1.E-02
constraint = 'T'
concentration = pconstraint_concentration(pspecies, value, constraint)
concentration_list.append(concentration)

constraint = pconstraint(name, concentration_list)
dat.constraint_list.append(constraint)	# Assigning done here
#--------------------------------------------------------------

###############################################################

# Print to console the data attributes

if dat.uniform_velocity:
	print 'uniform_velocity:', dat.uniform_velocity
	print 'value_list:', dat.uniform_velocity.value_list

if dat.chemistry:
	print 'chemistry:', dat.chemistry
	print '(primary species) pspecies:', dat.chemistry.pspecies
	print 'molal:', dat.chemistry.molal
	print 'output:', dat.chemistry.output
	print
 
print 'grid:', dat.grid
print 'type:', dat.grid.type
print 'bounds_bool:', dat.grid.bounds_bool
print 'lower_bounds:', dat.grid.lower_bounds
print 'upper_bounds:', dat.grid.upper_bounds
print 'origin:', dat.grid.origin
print 'nxyz:', dat.grid.nxyz
print 'dxyz:', dat.grid.dxyz
print 'gravity_bool:', dat.grid.gravity_bool
print 'gravity:', dat.grid.gravity
print 'filename:', dat.grid.filename
print
 
print 'timestepper:', dat.timestepper
print 'ts_acceleration:', dat.timestepper.ts_acceleration
print 'num_steps_after_cut:', dat.timestepper.num_steps_after_cut
print 'max_steps', dat.timestepper.max_steps
print 'max_ts_cuts', dat.timestepper.max_ts_cuts
print 'cfl_limiter', dat.timestepper.cfl_limiter
print 'initialize_to_steady_state', dat.timestepper.initialize_to_steady_state
print 'run_as_steady_state', dat.timestepper.run_as_steady_state
print 'max_pressure_change', dat.timestepper.max_pressure_change
print 'max_temperature_change', dat.timestepper.max_temperature_change
print 'max_concentration_change', dat.timestepper.max_concentration_change
print 'max_saturation_change', dat.timestepper.max_saturation_change
print
 
print 'time:', dat.time
print '(final_time) tf:', dat.time.tf
print '(initial_timestep_size) dti:', dat.time.dti
print '(maximum_timestep_size) dtf:', dat.time.dtf
print '(maximum_time_step_list_value) dtf_lv:', dat.time.dtf_lv
print '(maximum_time_step_list_increment) dtf_li:', dat.time.dtf_li
print '(maximum_time_step_size_list_value_unit) dtf_lv_unit:', dat.time.dtf_lv_unit
print '(maximum_time_step_size_list_increment_unit) dtf_li_unit:', dat.time.dtf_li_unit
print

print 'proplist:'
for prop in dat.proplist:
	print '(property) prop:', dat.proplist.index(prop),prop
	print 'name:',prop.name
	print 'id:',prop.id
	print 'porosity:',prop.porosity
	print 'tortuosity:',prop.tortuosity
	print 'density:',prop.density
	print 'specific_heat:',prop.specific_heat
	print 'cond_dry:',prop.cond_dry
	print 'cond_wet:',prop.cond_wet
	print 'permeability:',prop.permeability
	print
print	# May double print an empty line - Done just in case a list is empty



print '(newton_solver) nsolverlist:'
for nsolver in dat.nsolverlist:
	print '(newton_solver) nsolver:', dat.nsolverlist.index(nsolver), nsolver
	print 'name:', nsolver.name
	print 'atol:', nsolver.atol
	print 'rtol:', nsolver.rtol
	print 'stol:', nsolver.stol
	print 'dtol:', nsolver.dtol
	print 'itol:', nsolver.itol
	print 'max_it:', nsolver.max_it
	print 'max_f:', nsolver.max_f
	print
print
	
print 'output:', dat.output
print 'times:', dat.output.time_list
print 'periodic_observation_timestep:', dat.output.periodic_observation_timestep
print 'print_column_ids:', dat.output.print_column_ids
print 'format:', dat.output.format
print 'velocities:', dat.output.velocities
print 'mass_balance:', dat.output.mass_balance
print

print 'fluid:', dat.fluid
print 'diffusion_coefficient:', dat.fluid.diffusion_coefficient
print

print 'saturation:', dat.saturation
print 'name:', dat.saturation.name
print 'permeability_function_type:', dat.saturation.permeability_function_type
print 'saturation_function_type:', dat.saturation.saturation_function_type
print 'residual_saturation_liquid:', dat.saturation.residual_saturation_liquid
print 'residual_saturation_gas:', dat.saturation.residual_saturation_gas
print 'lambda:', dat.saturation.a_lambda
print 'alpha:', dat.saturation.alpha
print 'max_capillary_pressure:', dat.saturation.max_capillary_pressure
print 'betac:', dat.saturation.betac
print 'power:', dat.saturation.power
print

print 'regionlist:'
for region in dat.regionlist:
	print 'region:', dat.regionlist.index(region), region
	print 'name:', region.name
	print 'face:', region.face
	print 'coordinates_lower:', region.coordinates_lower
	print 'coordinates_upper:', region.coordinates_upper
	print
print

if dat.observation:
	print 'observation:', dat.observation
	print 'region:', dat.observation.region
	print

print 'flowlist:'
for flow in dat.flowlist:
	print 'flow:', dat.flowlist.index(flow), flow
	print 'name:', flow.name
	print 'units_list:', flow.units_list
	print 'iphase:', flow.iphase
	print 'sync_timestep_with_update:', flow.sync_timestep_with_update
	print 'varlist:'
	for variable in flow.varlist:
		print '\tname:', variable.name
		print '\ttype:', variable.type
		print '\tvaluelist:', variable.valuelist
		print '\tlist:', variable.list
		print '\tunit:', variable.unit
		print
print
		
print 'initial_condition:', dat.initial_condition
print 'flow:', dat.initial_condition.flow
print 'transport:', dat.initial_condition.transport
print 'region:', dat.initial_condition.region
print

print'(transport conditions) transportlist:'
for t in dat.transportlist:
	print 'transport:', dat.transportlist.index(t), t
	print 'name:', t.name
	print 'type:', t.type
	print 'constraint_list_value:', t.constraint_list_value
	print 'constraint_list_type:', t.constraint_list_type
	print
print

print 'boundary_condition_list:'
for bcon in dat.boundary_condition_list:
	print 'boundary_condition:', dat.boundary_condition_list.index(bcon), bcon
	print 'name:', bcon.name
	print 'flow:', bcon.flow
	print 'region:', bcon.region
	print
print

if dat.source_sink:
	print 'source_sink:', dat.source_sink
	print 'flow:', dat.source_sink.flow
	print 'region:', dat.source_sink.region
	print

print '(stratigraphy couplers) strata:', dat.strata
print 'region:', dat.strata.region
print 'material:', dat.strata.material
print

if dat.constraint_list:
	print 'constraint_list:'
	for constraint in dat.constraint_list:
		print 'constraint:', dat.constraint_list.index(constraint),constraint
		print 'name:', constraint.name
		print 'concentration_list:'
		for concentration in constraint.concentration_list:
			print '\t(primary species) pspecies:', concentration.pspecies
			print '\tvalue:', concentration.value
			print '\tconstraint:', concentration.constraint
			print
	print	

# Write to File
dat.write('tracer_1D.in')
