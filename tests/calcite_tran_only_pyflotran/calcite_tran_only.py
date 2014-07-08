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
dat.uniform_velocity.value_list = [1.e0, 0.e0, 0.e0, 'm/yr']
#--------------------------------------------------------------

# set chemistry
#--------------------------------------------------------------
chemistry = pchemistry()
chemistry.pspecies_list = ['H+', 'HCO3-', 'Ca++']
chemistry.sec_species_list = ['OH-', 'CO3--', 'CO2(aq)', 'CaCO3(aq)', 'CaHCO3+', 'CaOH+']
chemistry.gas_species_list = ['CO2(g)']
chemistry.minerals_list = ['Calcite']
mineral_kinetic = pchemistry_m_kinetic()	# new mineral kinetic object
mineral_kinetic.name = 'Calcite'
mineral_kinetic.rate_constant_list = [1.e-6, 'mol/m^2-sec']
chemistry.m_kinetics_list.append(mineral_kinetic)	# assigning for mineral kinetic object done here
chemistry.database = '/home/satkarra/src/pflotran-dev-PM-RHEL-6.5-nodebug/database/hanford.dat'
chemistry.log_formulation = True
chemistry.activity_coefficients = 'TIMESTEP'
chemistry.output_list = ['PH','all']
dat.chemistry = chemistry	# assign chemistry object
#--------------------------------------------------------------

# set grid
#--------------------------------------------------------------
grid = pgrid()
grid.type = 'structured'
grid.lower_bounds = [0.0, 0.0, 0.0]
grid.upper_bounds = [100.0, 1.0, 1.0]
grid.bounds_bool = True
grid.orign = [0.0, 0.0, 0.0 ]
grid.nxyz = [100, 1, 1]
dat.grid = grid
#--------------------------------------------------------------

# set time
#--------------------------------------------------------------
time = ptime()
time.tf = [25.0, 'y']	# FINAL_TIME
time.dti = [1.0, 'h']	# INITIAL_TIMESTEP_SIZE
time.dtf = [0.25 ,'y']	# MAXIMUM_TIMESTEP_SIZE
dat.time = time
#--------------------------------------------------------------

# set material properties aka prop_list
#--------------------------------------------------------------
material_propery = pmaterial()
material_propery.name = 'soil1'
material_propery.id = 1
material_propery.porosity = 0.25
material_propery.tortuosity = 1.0
dat.proplist.append(material_propery)
#--------------------------------------------------------------

# set linear solvers
#--------------------------------------------------------------
ls = plsolver()
ls.name = 'TRANSPORT'
ls.solver = 'DirECT'
dat.lsolverlist.append(ls)
#--------------------------------------------------------------

# set output
#--------------------------------------------------------------
o = poutput()
o.time_list = ['y', 5.0, 10.0, 15.0, 20.0]
o.format.append('tecplot point')
dat.output = o
#--------------------------------------------------------------

# set fluid properties
#--------------------------------------------------------------
f = pfluid()
f.diffusion_coefficient = 1.0000000e-09
dat.fluid = f
#--------------------------------------------------------------

# set regions
#--------------------------------------------------------------
r = pregion()
r.name = 'ALL'
r.face = None
r.coordinates_lower = [0.0, 0.0, 0.0]
r.coordinates_upper = [100.0, 1.0, 1.0]
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
r.coordinates_lower = [100.0, 0.0, 0.0]
r.coordinates_upper = [100.0, 1.0, 1.0]
dat.regionlist.append(r)
#--------------------------------------------------------------

# set transport conditions
#--------------------------------------------------------------
tc = ptransport()
tc.name = 'background_CONC'
tc.type = 'zero_gradient'
tc.constraint_list_value = [0.e0]
tc.constraint_list_type = ['initial_CONSTRAINT']
dat.transportlist.append(tc)

tc = ptransport()
tc.name = 'inlet_conc'
tc.type = 'dirichlet_zero_gradient'
tc.constraint_list_value = [0.e0]
tc.constraint_list_type = ['inlet_constraint']
dat.transportlist.append(tc)
#--------------------------------------------------------------

# set initial condition
#--------------------------------------------------------------
ic = pinitial_condition()
ic.transport = 'background_CONC'
ic.region = 'ALL'
dat.initial_condition = ic
#--------------------------------------------------------------

# set boundary conditions
#--------------------------------------------------------------
bc = pboundary_condition()
bc.name = 'OUTLEt'
bc.transport = 'background_CONC'
bc.region = 'EAST'
dat.boundary_condition_list.append(bc)

bc = pboundary_condition()
bc.name = 'inlet'
bc.transport = 'inlet_conc'
bc.region = 'west'
dat.boundary_condition_list.append(bc)
#--------------------------------------------------------------

# set stratigraphy couplers
#--------------------------------------------------------------
sc = pstrata()
sc.region = 'all' 
sc.material = 'soil1'
dat.strata = sc
#--------------------------------------------------------------

# set constraints
#--------------------------------------------------------------
# initial condition - 1st condition
constraint = pconstraint()
constraint.name = 'initial_CONSTRAINT'
constraint.concentration_list = [] # Assigning for this done below
concentration = pconstraint_concentration()		# new concentration object
concentration.pspecies = 'H+'
concentration.value = 1.e-8
concentration.constraint = 'F'
constraint.concentration_list.append(concentration)	# assign concentration
concentration = pconstraint_concentration()		# new concentration object
concentration.pspecies = 'HCO3-'
concentration.value = 1.e-3
concentration.constraint = 'G'
concentration.element = 'CO2(g)'
constraint.concentration_list.append(concentration)	# assign concentration
concentration = pconstraint_concentration()		# new concentration object
concentration.pspecies = 'Ca++'
concentration.value = 5.e-4
concentration.constraint = 'M'
concentration.element = 'Calcite'
constraint.concentration_list.append(concentration)	# assign concentration
dat.constraint_list.append(constraint)	# assign constraint

# west condition - 2nd condition
constraint = pconstraint()
constraint.name = 'inlet_constraint'
constraint.concentration_list = [] # Assigning for this done below
concentration = pconstraint_concentration()		# new concentration object
concentration.pspecies = 'H+'
concentration.value = 5.
concentration.constraint = 'P'
constraint.concentration_list.append(concentration)	# assign concentration
concentration = pconstraint_concentration()		# new concentration object
concentration.pspecies = 'HCO3-'
concentration.value = 1.e-3
concentration.constraint = 'T'
constraint.concentration_list.append(concentration)	# assign concentration
concentration = pconstraint_concentration()		# new concentration object
concentration.pspecies = 'Ca++'
concentration.value = 1.e-6
concentration.constraint = 'Z'
constraint.concentration_list.append(concentration)	# assign concentration
dat.constraint_list.append(constraint)	# assign constraint
#--------------------------------------------------------------

# Testing write
# dat.write('calcite_tran_only.in')

# Write to file and execute that input file
dat.run(input='calcite_tran_only.in',exe='/home/satkarra/src/pflotran-dev-PM-RHEL-6.5-nodebug/src/pflotran/pflotran')
