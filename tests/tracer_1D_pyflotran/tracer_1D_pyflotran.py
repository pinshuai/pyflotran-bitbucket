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
c.pspecies_list = ['A(aq)']
c.molal = True
c.output_list = ['All']
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
dat.time = t
#--------------------------------------------------------------

# set output
#--------------------------------------------------------------
o = poutput()
o.time_list = ['s', 26042.0, 39063.0, 52083.0, 1000000.0]
o.periodic_observation_timestep = 1
o.print_column_ids = True
o.format_list.append('TECPLOT POINT')
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
dat.strata_list.append(sc)
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

# Test Write
dat.write('tracer_1D.in')

# Write to File
#dat.run(input='tracer_1D.in',exe='/home/satkarra/src/pflotran-dev-PM-RHEL-6.5-nodebug/src/pflotran/pflotran')