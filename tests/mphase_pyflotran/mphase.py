import sys
sys.path.append('../.')

from pdata import*


'''
Floats can be inputted directly as a float using 'e' or 'E' - E.g. - 0.e0
or can be inserted via FORTRAN way as a string using 'd' or 'D' E.g. - '0.d0'

Some classes (when set to None at the initializer of pdata) need to be instantiated here.
'''

###############################################################

# initialize without reading in test data
#--------------------------------------------------------------
dat = pdata('')
#--------------------------------------------------------------

# set mode
#--------------------------------------------------------------
m = pmode()
m.name = 'mphase'
dat.mode = m
#--------------------------------------------------------------

# set co2 database
#--------------------------------------------------------------
dat.co2_database = '/home/satkarra/src/pflotran-dev-PM-RHEL-6.5-nodebug/database/co2data0.dat'
#--------------------------------------------------------------

# set grid
#--------------------------------------------------------------
g = pgrid()
g.type = 'structured'
g.lower_bounds = [0.e0, 0.e0, 0.e0]
g.upper_bounds = [321.e0, 1.e0, 51.e0]
g.bounds_bool = True
g.orign = [0.e0, 0.e0, 0.e0]
g.nxyz = [107, 1, 51]
g.dxyz = [5, 5, 5]	# Should not write
g.gravity_bool = False
g.gravity =  [0.0, 0.0, -9.8068]	# Should not write
g.filename =  ''
dat.grid = g
#--------------------------------------------------------------

# set time stepping
#--------------------------------------------------------------
ts = ptimestepper()
ts.ts_acceleration = 8
dat.timestepper = ts
#--------------------------------------------------------------

# set material properities aka prop_list
#--------------------------------------------------------------
m = pmaterial('','')
m.id = 1
m.name = 'soil1'
m.porosity = 0.15e0
m.tortuosity = 1e-1
m.density = 2.65E3	# ROCK_DENSITY
m.specific_heat = 1E3
m.cond_dry = 0.5		# THERMAL_CONDUCTIVITY_DRY
m.cond_wet = 0.5		# THERMAL_CONDUCTIVITY_WET
m.saturation = 'sf2'
m.permeability = [1.e-15,1.e-15,1.e-17]
dat.proplist.append(m)
#--------------------------------------------------------------

# set time
#--------------------------------------------------------------
t = ptime()
t.tf = [0.25e0, 'y']	# FINAL_TIME
t.dti = [1.e-6, 'y']	# INITIAL_TIMESTEP_SIZE
t.dtf = [50.e0,'y']	# MAXIMUM_TIMESTEP_SIZE
t.dtf_lv = [200.e0, 500.e0, 1000.e0, 5000.e0]	# MAXIMUM_TIMESTEP_SIZE before 'at'
t.dtf_lv_unit = ['y','y','y','y']	# MAXIMUM_TIMESTEP_SIZE before 'at' time unit (eg. 'y')
t.dtf_li = [50., 20000., 50000., 100000.]	# MAXIMUM_TIMESTEP_SIZE after 'at'
t.dtf_li_unit = ['y','y','y','y']	# MAXIMUM_TIMESTEP_SIZE after 'at' time unit (eg. 'y')
t.dtf_i = 4		# Needs to be the length of dtf_lv or dtf_li
dat.time = t
#--------------------------------------------------------------

# set newton solvers
#--------------------------------------------------------------
ns = pnsolver('')
ns.name = 'FLOW'
ns.atol = 1e-12
ns.rtol = 1e-12
ns.stol = 1e-30
ns.dtol = 1e15
ns.itol = 1e-8
ns.max_it = 25
ns.max_f = 100
dat.nsolverlist.append(ns)

ns = pnsolver('')
ns.name = 'tran'
ns.atol = 1e-12
ns.rtol = 1e-12
ns.stol = 1e-30
ns.dtol = 1e15
ns.itol = 1e-8
ns.max_it = 25
ns.max_f = 100
dat.nsolverlist.append(ns)
#--------------------------------------------------------------

# set output
#--------------------------------------------------------------
o = poutput()
o.mass_balance = True
o.print_column_ids = True
o.periodic_observation_timestep = 1
o.format.append('TECPLOT POINT')
o.format.append('HDF5')
o.velocities = False
dat.output = o
#--------------------------------------------------------------

# set fluid properties
#--------------------------------------------------------------
f = pfluid()
f.diffusion_coefficient = 1.e-9
dat.fluid = f
#--------------------------------------------------------------

# set saturation functions
#--------------------------------------------------------------
s = psaturation('','')
s.name = 'sf2'
s.permeability_function_type = 'NMT_EXP'
s.saturation_function_type = 'NMT_EXP'
s.residual_saturation_liquid = 0.1
s.residual_saturation_gas = 0.0
s.a_lambda = 0.762
s.alpha = 7.5e-4
s.max_capillary_pressure = 1.e6
s.betac = 2.e0
s.power = 7.e0
dat.saturation = s
#--------------------------------------------------------------

# set regions
#--------------------------------------------------------------
r = pregion()
r.name = 'all'
r.face = None
r.coordinates_lower = [0.e0, 0.e0, 0.e0]
r.coordinates_upper = [321.e0, 1.e0,  51.e0]
dat.regionlist.append(r)

r = pregion()
r.name = 'top'
r.face = 'top'
r.coordinates_lower = [0.e0, 0.e0, 51.e0]
r.coordinates_upper = [321.e0, 1.e0,  51.e0]
dat.regionlist.append(r)

r = pregion()
r.name = 'west'
r.face = 'WEST'
r.coordinates_lower = [0.e0, 0.e0, 0.e0]
r.coordinates_upper = [0.e0, 1.e0,  51.e0]
dat.regionlist.append(r)

r = pregion()
r.name = 'EAST'
r.face = 'east'
r.coordinates_lower = [321.e0, 0.e0, 0.e0]
r.coordinates_upper = [321.e0, 1.e0,  51.e0]
dat.regionlist.append(r)

r = pregion()
r.name = 'well'
r.face = None
r.coordinates_lower = [160.e0, 1.e0, 20.e0]
r.coordinates_upper = [160.e0, 1.e0, 20.e0]
dat.regionlist.append(r)
#--------------------------------------------------------------

# set flow conditions
#--------------------------------------------------------------
# initial flow condition
flow = pflow('')
flow.name = 'initial'
flow.units_list = None
flow.iphase = 1
flow.sync_timestep_with_update = False
flow.varlist = [] # Assigning for this done below
var = pflow_variable('')	# new flow var object
var.name = 'pressure'
var.type = 'hydrostatic'
var.valuelist = [2e7, 2e7]
var.list = []
var.unit = None
flow.varlist.append(var)	# assigning for flow var done here
var = pflow_variable('') # new flow var object
var.name = 'TEMPERATURE'
var.type = 'zero_gradient'
var.valuelist = [50.0]
var.list = []
var.unit = 'C'
flow.varlist.append(var)	# assigning for flow var done here
var = pflow_variable('') # new flow var object
var.name = 'CONCENTRATION'
var.type = 'zero_gradient'
var.valuelist = [1e-6]
var.list = []
var.unit = 'm'
flow.varlist.append(var)	# assigning for flow var done here
var = pflow_variable('') # new flow var object
var.name = 'ENTHALPY'
var.type = 'dirichlet'
var.valuelist = [0.e0, 0.e0]
var.list = []
var.unit = None
flow.varlist.append(var)	# assigning for flow var done here
dat.flowlist.append(flow)	# Assigning for flow condition done here

flow = pflow('')
# top flow condition goes here
flow.name = 'top'
flow.units_list = None
flow.iphase = 1
flow.sync_timestep_with_update = False
flow.varlist = [] # Assigning for this done below
var = pflow_variable('')	# new flow var object
var.name = 'pressure'
var.type = 'dirichlet'
var.valuelist = [3e7, 2e7]
var.list = []
var.unit = None
flow.varlist.append(var)	# assigning for flow var done here
var = pflow_variable('') # new flow var object
var.name = 'TEMPERATURE'
var.type = 'zero_gradient'
var.valuelist = [60.0]
var.list = []
var.unit = None
flow.varlist.append(var)	# assigning for flow var done here
var = pflow_variable('') # new flow var object
var.name = 'CONCENTRATION'
var.type = 'zero_gradient'
var.valuelist = [1e-6]
var.list = []
var.unit = None
flow.varlist.append(var)	# assigning for flow var done here
var = pflow_variable('') # new flow var object
var.name = 'ENTHALPY'
var.type = 'dirichlet'
var.valuelist = [0.e0, 0.e0]
var.list = []
var.unit = None
flow.varlist.append(var)	# assigning for flow var done here
dat.flowlist.append(flow)	# Assigning for flow condition done here

# source flow condition
flow = pflow('')
flow.name = 'source'
flow.units_list = None
flow.iphase = None
flow.sync_timestep_with_update = True
flow.varlist = [] # Assigning for this done below
var = pflow_variable('') # new flow var object
var.name = 'rate'
var.type = 'mass_rate'
var.valuelist = [0., 1.e-4]
var.list = []
var.unit = None
flow.varlist.append(var)	# assigning for flow var done here
var = pflow_variable('') # new flow var object
var.name = 'pressure'
var.type = 'dirichlet'
var.valuelist = [4e7, 2e7]
var.list = []
var.unit = None
flow.varlist.append(var)	# assigning for flow var done here
var = pflow_variable('') # new flow var object
var.name = 'TEMPERATURE'
var.type = 'dirichlet'
var.valuelist = [70.0]
var.list = []
var.unit = None
flow.varlist.append(var)	# assigning for flow var done here
var = pflow_variable('') # new flow var object
var.name = 'CONCENTRATION'
var.type = 'dirichlet'
var.valuelist = [0.e0]
var.list = []
var.unit = None
flow.varlist.append(var)	# assigning for flow var done here
var = pflow_variable('') # new flow var object
var.name = 'ENTHALPY'
var.type = 'dirichlet'
var.valuelist = [0.e0, 0.e0]
var.list = []
var.unit = None
flow.varlist.append(var)	# assigning for flow var done here
dat.flowlist.append(flow)	# Assigning for flow condition done here
#--------------------------------------------------------------

# set initial condition
#--------------------------------------------------------------
ic = pinitial_condition()
ic.flow = 'INITIAL'
ic.region = 'all'
dat.initial_condition = ic
#--------------------------------------------------------------

# set boundary conditions
#--------------------------------------------------------------
bc = pboundary_condition()
bc.name = 'WEST'
bc.flow = 'INITIAL'
bc.transport = None
bc.region = 'west'
dat.boundary_condition_list.append(bc)

bc = pboundary_condition()
bc.name = 'east'
bc.flow = 'INITIAL'
bc.transport = None
bc.region = 'east'
dat.boundary_condition_list.append(bc)
#--------------------------------------------------------------

# set source sink
#--------------------------------------------------------------
ss = psource_sink()
ss.flow = 'source'
ss.region = 'WELL'
dat.source_sink = ss
#--------------------------------------------------------------

# set stratigraphy couplers
#--------------------------------------------------------------
sc = pstrata()
sc.region = 'ALL' 
sc.material = 'SOIL1'
dat.strata = sc
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
dat.write('mphase.in')
