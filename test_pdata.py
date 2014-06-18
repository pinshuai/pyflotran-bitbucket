
from pdata import*

print '\nTEST EXECUTED\n'	# console header

###############################################################

# Read Test Data
dat = pdata('pflotran.in')
#dat = pdata('tracer_1D_SC.in')
###############################################################
'''
# initialize without reading in test data
#--------------------------------------------------------------
dat = pdata('')
#--------------------------------------------------------------

# set mode
#--------------------------------------------------------------
dat.mode.name = 'MPHASE'
#--------------------------------------------------------------

# set chemistry
#--------------------------------------------------------------
dat.chemistry.pspecies = 'A(aq)'
dat.chemistry.molal = True
dat.chemistry.output = 'ALL'
#--------------------------------------------------------------

# set grid
#--------------------------------------------------------------
dat.grid.type = 'structured'
dat.grid.lower_bounds = [0.e0, 0.e0, 0.e0]
dat.grid.upper_bounds = [321.e0, 1.e0, 51.e0]
dat.grid.bounds_bool = True
dat.grid.orign = [0.e0, 0.e0, 0.e0]
dat.grid.nxyz = [107, 1, 51]
dat.grid.dxyz = [5, 5, 5]	# Should not write
dat.grid.gravity_bool = False
dat.grid.gravity =  [0.0, 0.0, -9.8068]	# Should not write?
dat.grid.filename =  ''
#--------------------------------------------------------------

# set time stepping
#--------------------------------------------------------------
dat.timestepper.ts_acceleration = 8
dat.timestepper.num_steps_after_cut = None
dat.timestepper.max_steps = None
dat.timestepper.max_ts_cuts = None
dat.timestepper.cfl_limiter = None

dat.timestepper.initialize_to_steady_state = None
dat.timestepper.run_as_steady_state = None
dat.timestepper.max_pressure_change = None
dat.timestepper.max_temperature_change = None
dat.timestepper.max_concentration_change = None
dat.timestepper.max_saturation_change = None
#--------------------------------------------------------------

# set material properities aka prop_list
#--------------------------------------------------------------
id = 1
name = 'soil1'
porosity = 0.15e0
tortuosity = 1e-1
density = 2.65E3
specific_heat = 1E3
cond_dry = 0.5
cond_wet = 0.5
saturation = 'sf2'
permeability = [1.e-15,1.e-15,1.e-17]
	
material = pmaterial(id, name, porosity, tortuosity, density, specific_heat,
		     cond_dry, cond_wet, saturation, permeability)
dat.proplist.append(material)
#--------------------------------------------------------------

# set time
#--------------------------------------------------------------
dat.time.tf = [0.25e0, 'y']
dat.time.dti = [1.e-6, 'y']
dat.time.dtf = [50.e0,'y']
dat.time.dtf_lv = [200.e0, 500.e0, 1000.e0, 5000.e0]
dat.time.dtf_lv_unit = ['y', 'y', 'y', 'y']
dat.time.dtf_li = [15., 50., 20000., 50000., 100000.]
dat.time.dtf_li_unit = ['y', 'y', 'y', 'y']
dat.time.dtf_i = 4	# Needs to be the length of dtf_lv or dtf_li
#--------------------------------------------------------------

# set newton solvers
#--------------------------------------------------------------
name = 'FLOW'
atol = 1e-12
rtol = 1e-12
stol = 1e-30
dtol = 1e15
itol = 1e-8
max_it = 25
max_f = 100

nsolver = pnsolver(name, atol, rtol, stol, dtol, itol, max_it, max_f)
dat.nsolverlist.append(nsolver)

name = 'tran'
atol = 1e-12
rtol = 1e-12
stol = 1e-30
dtol = 1e15
itol = 1e-8
max_it = 25
max_f = 100

nsolver = pnsolver(name, atol, rtol, stol, dtol, itol, max_it, max_f)
dat.nsolverlist.append(nsolver)
#--------------------------------------------------------------

# set output
#--------------------------------------------------------------
dat.output.mass_balance = True
dat.output.print_column_ids = True
dat.output.periodic_observation_timestep = 1
dat.output.format.append('TECPLOT POINT')
dat.output.format.append('HDF5')
dat.output.velocities = True
#--------------------------------------------------------------

# set fluid properties
#--------------------------------------------------------------
dat.fluid.diffusion_coefficient = 1.e-9
#--------------------------------------------------------------

# set saturation functions
#--------------------------------------------------------------
dat.saturation.name = 'sf2'
dat.saturation.permeability_function_type = 'NMT_EXP'
dat.saturation.saturation_function_type = 'NMT_EXP'
dat.saturation.residual_saturation_liquid = 0.1
dat.saturation.residual_saturation_gas = 0.0
dat.saturation.a_lambda = 762e0
dat.saturation.alpha = 7.5e-4
dat.saturation.max_capillary_pressure = 1.e6
dat.saturation.betac = 2.e0
dat.saturation.power = 7.e0
#--------------------------------------------------------------

# set regions
#--------------------------------------------------------------
name = 'all'
face = None
coordinates_lower = [0.e0, 0.e0, 0.e0]
coordinates_upper = [321.e0, 1.e0,  51.e0]

region = pregion(name, coordinates_lower, coordinates_upper, face)
dat.regionlist.append(region)

name = 'top'
face = 'top'
coordinates_lower = [0.e0, 0.e0, 51.e0]
coordinates_upper = [321.e0, 1.e0,  51.e0]

region = pregion(name, coordinates_lower, coordinates_upper, face)
dat.regionlist.append(region)

name = 'west'
face = 'WEST'
coordinates_lower = [0.e0, 0.e0, 0.e0]
coordinates_upper = [0.e0, 1.e0,  51.e0]

region = pregion(name, coordinates_lower, coordinates_upper, face)
dat.regionlist.append(region)

name = 'EAST'
face = 'east'
coordinates_lower = [321.e0, 0.e0, 0.e0]
coordinates_upper = [321.e0, 1.e0,  51.e0]


region = pregion(name, coordinates_lower, coordinates_upper, face)
dat.regionlist.append(region)

name = 'well'
face = None
coordinates_lower = [160.e0, 1.e0, 20.e0]
coordinates_upper = [160.e0, 1.e0, 20.e0]

region = pregion(name, coordinates_lower, coordinates_upper, face)
dat.regionlist.append(region)
#--------------------------------------------------------------

# set flow conditions
#--------------------------------------------------------------
# initial flow condition
name = 'initial'
units_list = None
iphase = 1
sync_timestep_with_update = False
varlist = [] # Assigning for this done below

vname = 'pressure'
vtype = 'hydrostatic'
vvaluelist = [2e7, 2e7]
vlist = []
vunit = None
var = pflow_variable(vname, vtype, vvaluelist, vlist, vunit)
varlist.append(var)

vname = 'TEMPERATURE'
vtype = 'zero_gradient'
vvaluelist = [50]
vlist = []
vunit = 'C'
var = pflow_variable(vname, vtype, vvaluelist, vlist, vunit)
varlist.append(var)

vname = 'CONCENTRATION'
vtype = 'zero_gradient'
vvaluelist = [1e-6]
vlist = []
vunit = 'm'
var = pflow_variable(vname, vtype, vvaluelist, vlist, vunit)
varlist.append(var)

vname = 'ENTHALPY'
vtype = 'dirichlet'
vvaluelist = [0.e0, 0.e0]
vlist = []
vunit = None
var = pflow_variable(vname, vtype, vvaluelist, vlist, vunit)
varlist.append(var)

flow = pflow(name, units_list, iphase, sync_timestep_with_update, varlist)
dat.flowlist.append(flow)	# Assigning done here

# top flow condition
name = 'top'
units_list = None
iphase = 1
sync_timestep_with_update = False
varlist = [] # Assigning for this done below

vname = 'pressure'
vtype = 'dirichlet'
vvaluelist = [3e7, 2e7]
vlist = []
vunit = None
var = pflow_variable(vname, vtype, vvaluelist, vlist, vunit)
varlist.append(var)

vname = 'TEMPERATURE'
vtype = 'zero_gradient'
vvaluelist = [60]
vlist = []
vunit = None
var = pflow_variable(vname, vtype, vvaluelist, vlist, vunit)
varlist.append(var)

vname = 'CONCENTRATION'
vtype = 'zero_gradient'
vvaluelist = [1e-6]
vlist = []
vunit = None
var = pflow_variable(vname, vtype, vvaluelist, vlist, vunit)
varlist.append(var)

vname = 'ENTHALPY'
vtype = 'dirichlet'
vvaluelist = [0.e0, 0.e0]
vlist = []
vunit = None
var = pflow_variable(vname, vtype, vvaluelist, vlist, vunit)
varlist.append(var)

flow = pflow(name, units_list, iphase, sync_timestep_with_update, varlist)
dat.flowlist.append(flow)	# Assigning done here

# source flow condition
name = 'source'
units_list = None
iphase = None
sync_timestep_with_update = True
varlist = [] # Assigning for this done below

vname = 'pressure'
vtype = 'dirichlet'
vvaluelist = [4e7, 2e7]
vlist = []
vunit = None
var = pflow_variable(vname, vtype, vvaluelist, vlist, vunit)
varlist.append(var)

vname = 'TEMPERATURE'
vtype = 'dirichlet'
vvaluelist = [70]
vlist = []
vunit = None
var = pflow_variable(vname, vtype, vvaluelist, vlist, vunit)
varlist.append(var)

vname = 'CONCENTRATION'
vtype = 'dirichlet'
vvaluelist = [0.e0]
vlist = []
vunit = None
var = pflow_variable(vname, vtype, vvaluelist, vlist, vunit)
varlist.append(var)

vname = 'ENTHALPY'
vtype = 'dirichlet'
vvaluelist = [0.e0, 0.e0]
vlist = []
vunit = None
var = pflow_variable(vname, vtype, vvaluelist, vlist, vunit)
varlist.append(var)

flow = pflow(name, units_list, iphase, sync_timestep_with_update, varlist)
dat.flowlist.append(flow)	# Assigning done here
#--------------------------------------------------------------

# set initial condition
#--------------------------------------------------------------
dat.initial_condition.flow = 'INITIAL'
dat.initial_condition.region = 'all'
#--------------------------------------------------------------

# set transport conditions
#--------------------------------------------------------------
name = 'initial'
type = 'dirichlet'
constraint_list_value = [0.e0]
constraint_list_type = ['initial']

t = ptransport(name,type,constraint_list_value,constraint_list_type)
dat.transportlist.append(t)
#--------------------------------------------------------------

# set boundary conditions
#--------------------------------------------------------------
name = 'WEST'
flow = 'INITIAL'
region = 'ALL'

b = pboundary_condition(name, flow, region)
dat.boundary_condition_list.append(b)
#--------------------------------------------------------------

# set source sink
#--------------------------------------------------------------
dat.source_sink.flow = 'source'
dat.source_sink.region = 'WELL'
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



constraint = pconstraint(name, concentration_list)
dat.constraint_list.append(constraint)	# Assigning done here

# west condition - 2nd condition
name = 'west'
concentration_list = [] # Assigning for this done below

# concentration
pspecies = 'A(aq)'
value = 1.e-8
constraint = 'T'
concentration = pconstraint_concentration(pspecies, value, constraint)
concentration_list.append(concentration)

constraint = pconstraint(name, concentration_list)
dat.constraint_list.append(constraint)	# Assigning done here

# east condition - 2nd condition
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
'''
# Print to console the data attributes

if dat.chemistry:
	print 'chemistry:', dat.chemistry
	print '(primary species) pspecies:', dat.chemistry.pspecies
	print 'molal:', dat.chemistry.molal
	print 'output:', dat.chemistry.output
	print
 
print 'grid:', dat.grid
print 'nxyz:', dat.grid.nxyz
print 'origin:', dat.grid.origin
print 'dxyz:', dat.grid.dxyz
print 'upper_bounds:', dat.grid.upper_bounds
print 'gravity:', dat.grid.gravity
print
 
print 'timestepper:', dat.timestepper
print 'ts_acceleration:', dat.timestepper.ts_acceleration
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
print '',
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

#print 'filename',dat.filename
#
#print	
#print 'mode', dat.mode
#print 'name', dat.mode.name

###############################################################

# Write to File
dat.write('pflotran2.in')
