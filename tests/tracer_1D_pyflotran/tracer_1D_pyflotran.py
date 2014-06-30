import sys
sys.path.append('../../.')
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
c = pchemistry()
c.pspecies = 'A(aq)'
c.molal = True
c.output = 'ALL'
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
ns = pnsolver()
ns.name = 'TRANSPORT'
ns.atol = 1e-15
ns.rtol = 1e-10
ns.stol = 1e-30
ns.dtol = None
ns.itol = 1e-8
ns.max_it = 100
ns.max_f = 100
dat.nsolverlist.append(ns)
#--------------------------------------------------------------

# set fluid properties
#--------------------------------------------------------------
f = pfluid()
f.diffusion_coefficient = 1.e-9
dat.fluid = f
#--------------------------------------------------------------

# set material properties aka prop_list
#--------------------------------------------------------------
mp = pmaterial()	# For assigning defaults
mp.name = 'soil1'
mp.id = 1
mp.porosity = 1.e0
mp.tortuosity = 1.e0
mp.density = 2.8e3		# Rock Density
mp.specific_heat = 1e3
mp.cond_dry 		# Defaults are assigned
mp.cond_wet = 0.5		# THERMAL_CONDUCTIVITY_WET
mp.saturation = 'default'
mp.permeability = [1.e-15,1.e-15,1.e-15]
dat.proplist.append(mp)
#--------------------------------------------------------------

# set time
#--------------------------------------------------------------
t = ptime()
t.tf = [1.e4, 's']		# FINAL_TIME
t.dti = [1.e-6, 's']		# INITIAL_TIMESTEP_SIZE
t.dtf = [10.e0,'s']		# MAXIMUM_TIMESTEP_SIZE
t.dtf_lv = [1.e2, 1.e3]		#MAXIMUM_TIMESTEP_SIZE before 'at'
t.dtf_lv_unit = ['s', 's']		# time unit
t.dtf_li = [5.e3, 5.e4]		#MAXIMUM_TIMESTEP_SIZE after 'at'
t.dtf_li_unit = ['s', 's']		# time unit
t.dtf_i = 2			# Needs to be the length of dtf_lv and dtf_li
dat.time = t
#--------------------------------------------------------------

# set output
#--------------------------------------------------------------
o = poutput()
o.time_list = ['s', 26042.0, 39063.0, 52083.0, 1000000.0]
o.periodic_observation_timestep = 1
o.print_column_ids = True
o.format.append('TECPLOT POINT')
dat.output = o
#--------------------------------------------------------------

# set saturation functions
#--------------------------------------------------------------
s = psaturation()
s.name = 'default'
s.saturation_function_type = 'VAN_GENUCHTEN'
s.residual_saturation_liquid = 0.1
s.residual_saturation_gas = 0.0
s.a_lambda = 0.762e0
s.alpha = 7.5e-4
s.max_capillary_pressure = 1.e6
dat.saturation = s
#--------------------------------------------------------------

# set regions
#--------------------------------------------------------------
r = pregion()
r.name = 'all'
r.face = None
r.coordinates_lower = [0.e0, 0.e0, 0.e0]
r.coordinates_upper = [0.04e0, 1.e0,  1.e0]
dat.regionlist.append(r)

r = pregion()
r.name = 'west'
r.face = 'WEST'
r.coordinates_lower = [0.e0, 0.e0, 0.e0]
r.coordinates_upper = [0.e0, 1.e0,  1.e0]
dat.regionlist.append(r)

r = pregion()
r.name = 'east'
r.face = 'EAST'
r.coordinates_lower = [0.04e0, 0.e0, 0.e0]
r.coordinates_upper = [0.04e0, 1.e0, 1.e0]
dat.regionlist.append(r)

r = pregion()
r.name = 'obs'
r.face = None
r.coordinates_lower = [0.04e0, 0.e0, 0.e0]
r.coordinates_upper = [0.04e0, 1.e0, 1.e0]
dat.regionlist.append(r)
#--------------------------------------------------------------

# set observation
#--------------------------------------------------------------
o = pobservation()
o.region = 'obs'
dat.observation = o
#--------------------------------------------------------------

# set transport conditions
#--------------------------------------------------------------
tc = ptransport()
tc.name = 'initial'
tc.type = 'dirichlet'
tc.constraint_list_value = [0.e0]
tc.constraint_list_type = ['initial']
dat.transportlist.append(tc)

tc = ptransport()
tc.name = 'WEST'
tc.type = 'dirichlet'
tc.constraint_list_value = [0.e0]
tc.constraint_list_type = ['WEST']
dat.transportlist.append(tc)

tc = ptransport()
tc.name = 'east'
tc.type = 'ZERO_gradient'
tc.constraint_list_value = [0.e0]
tc.constraint_list_type = ['east']
dat.transportlist.append(tc)
#--------------------------------------------------------------

# set initial condition
#--------------------------------------------------------------
ic = pinitial_condition()
ic.flow = 'INITIAL'
ic.transport = 'initial'
ic.region = 'all'
dat.initial_condition = ic
#--------------------------------------------------------------

# set boundary conditions
#--------------------------------------------------------------
bc = pboundary_condition()
bc.name = ''
bc.flow = 'west'
bc.transport = 'west'
bc.region = 'WEST'
dat.boundary_condition_list.append(bc)

bc = pboundary_condition()
bc.name = ''
bc.flow = 'east'
bc.transport = 'EAST'
bc.region = 'east'
dat.boundary_condition_list.append(bc)
#--------------------------------------------------------------

# set stratigraphy couplers
#--------------------------------------------------------------
sc = pstrata()
sc.region = 'ALL' 
sc.material = 'SOIL1'
dat.strata = sc
#--------------------------------------------------------------

# set constraints
#--------------------------------------------------------------
# initial condition - 1st condition
constraint = pconstraint()
constraint.name = 'initial'
constraint.concentration_list = [] # Assigning for this done below
concentration = pconstraint_concentration()	# new concentration object
concentration.pspecies = 'A(aq)'
concentration.value = 0.1
concentration.constraint = 'T'
constraint.concentration_list.append(concentration)	# assign concentration
dat.constraint_list.append(constraint)	# assign constraint

# west condition - 2nd condition
constraint = pconstraint()
constraint.name = 'WEST'
constraint.concentration_list = [] # Assigning for this done below
concentration = pconstraint_concentration()	# new concentration object
concentration.pspecies = 'A(aq)'
concentration.value = 1.e-8
concentration.constraint = 'T'
constraint.concentration_list.append(concentration)	# assign concentration
dat.constraint_list.append(constraint)	# assign constraint

# east condition - 3rd condition
constraint = pconstraint()
constraint.name = 'east'
constraint.concentration_list = [] # Assigning for this done below
concentration = pconstraint_concentration()	# new concentration object
concentration.pspecies = 'A(aq)'
concentration.value = 1.E-02
concentration.constraint = 'T'
constraint.concentration_list.append(concentration)	# assign concentration
dat.constraint_list.append(constraint)	# assign constraint
#--------------------------------------------------------------

###############################################################

# Print to console the data attributes

print '\n\nEXECUTING\n\n'

print 'co2_database:', dat.co2_database
print

if dat.uniform_velocity.value_list:
	print 'uniform_velocity:', dat.uniform_velocity
	print 'value_list:', dat.uniform_velocity.value_list
	print
	
print 'mode:', dat.mode
print 'name:', dat.mode.name
print

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

if dat.output:
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
	
###############################################################

# Write to File
dat.run(input='tracer_1D.in',exe='/home/satkarra/src/pflotran-dev-PM-RHEL-6.5-nodebug/src/pflotran/pflotran')
