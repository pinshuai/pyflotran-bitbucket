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
import pflotran as pft

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
dat.co2_database = pflotran_dir + '/database/co2data0.dat'
#--------------------------------------------------------------

# set grid
#--------------------------------------------------------------
g = pgrid()
g.type = 'structured'
g.lower_bounds = [0.e0, 0.e0, 0.e0]
g.upper_bounds = [321.e0, 1.e0, 51.e0]
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
t.dtf = [50.e0, 'y']	# MAXIMUM_TIMESTEP_SIZE
t.dtf_lv = [200.e0, 500.e0, 1000.e0, 5000.e0]	# MAXIMUM_TIMESTEP_SIZE before 'at'
t.dtf_lv_unit = ['y','y','y','y']	# MAXIMUM_TIMESTEP_SIZE before 'at' time unit (eg. 'y')
t.dtf_li = [50., 20000., 50000., 100000.]	# MAXIMUM_TIMESTEP_SIZE after 'at'
t.dtf_li_unit = ['y','y','y','y']	# MAXIMUM_TIMESTEP_SIZE after 'at' time unit (eg. 'y')
#t.dtf_i = 4		# Needs to be the length of dtf_lv or dtf_li
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
var.time_unit_type = 'y'
var.data_unit_type = 'kg/s'
tlist = pflow_variable_list()	# sub-class of pflow_variable, assigned to list attribute
tlist.time_unit_value = 0.	# tlist = temporary list
tlist.data_unit_value_list = [0., 1.e-4]
var.list.append(tlist)
tlist = pflow_variable_list()	# sub-class of pflow_variable, assigned to list attribute
tlist.time_unit_value = 10.
tlist.data_unit_value_list = [0., 0.]
var.list.append(tlist)
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
dat.strata_list.append(sc)

#--------------------------------------------------------------

# Test write
#dat.write('mphase.in')

pflotran_exe = pflotran_dir + '/src/pflotran/pflotran'
# Write to file and execute that input file
dat.run(input='mphase.in',exe=pflotran_exe)

