#--------------------------------------------------
# Example to run 1D calcite problem with PyFLOTRAN.
# Cory Kitay and Satish Karra
# July 8, 2014
#--------------------------------------------------

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

print '\nTEST EXECUTED\n'	# console header

print('******************************************')
print('Using PyFLOTRAN')
print('******************************************')
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
chemistry.database = pflotran_dir + 'database/hanford.dat'
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
dat.add(material_propery)
#--------------------------------------------------------------

# set linear solvers
#--------------------------------------------------------------
linear_solver = plsolver()
linear_solver.name = 'TRANSPORT'
linear_solver.solver = 'DirECT'
dat.add(linear_solver)
#--------------------------------------------------------------

# set output
#--------------------------------------------------------------
output = poutput()
output.time_list = ['y', 5.0, 10.0, 15.0, 20.0]
output.format_list.append('tecplot point')
dat.output = output
#--------------------------------------------------------------

# set fluid properties
#--------------------------------------------------------------
fluid = pfluid()
fluid.diffusion_coefficient = 1.0000000e-09
dat.fluid = fluid
#--------------------------------------------------------------

# set regions
#--------------------------------------------------------------
region = pregion()
region.name = 'ALL'
region.face = None
region.coordinates_lower = [0.0, 0.0, 0.0]
region.coordinates_upper = [100.0, 1.0, 1.0]
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
region.coordinates_lower = [100.0, 0.0, 0.0]
region.coordinates_upper = [100.0, 1.0, 1.0]
dat.add(region)
#--------------------------------------------------------------

# set transport conditions
#--------------------------------------------------------------
transport_condition = ptransport()
transport_condition.name = 'background_CONC'
transport_condition.type = 'zero_gradient'
transport_condition.constraint_list_value = [0.e0]
transport_condition.constraint_list_type = ['initial_CONSTRAINT']
dat.add(transport_condition)

transport_condition = ptransport()
transport_condition.name = 'inlet_conc'
transport_condition.type = 'dirichlet_zero_gradient'
transport_condition.constraint_list_value = [0.e0]
transport_condition.constraint_list_type = ['inlet_constraint']
dat.add(transport_condition)
#--------------------------------------------------------------

# set initial condition
#--------------------------------------------------------------
initial_condition = pinitial_condition()
initial_condition.transport = 'background_CONC'
initial_condition.region = 'ALL'
dat.initial_condition = initial_condition
#--------------------------------------------------------------

# set boundary conditions
#--------------------------------------------------------------
boundary_condition = pboundary_condition()
boundary_condition.name = 'OUTLEt'
boundary_condition.transport = 'background_CONC'
boundary_condition.region = 'EAST'
dat.add(boundary_condition)

boundary_condition = pboundary_condition()
boundary_condition.name = 'inlet'
boundary_condition.transport = 'inlet_conc'
boundary_condition.region = 'west'
dat.add(boundary_condition)
#--------------------------------------------------------------

# set stratigraphy couplers
#--------------------------------------------------------------
stratigraphy_coupler = pstrata()
stratigraphy_coupler.region = 'all' 
stratigraphy_coupler.material = 'soil1'
dat.add(stratigraphy_coupler)
#--------------------------------------------------------------

# set constraints
#--------------------------------------------------------------
# initial condition - 1st condition
constraint = pconstraint()
constraint.name = 'initial_CONSTRAINT'
constraint.concentration_list = [] # Assigning for this done below
constraint.mineral_list = []
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
mineral = pconstraint_mineral()				# new mineral object
mineral.name = 'Calcite'
mineral.volume_fraction = 1.e-5
mineral.surface_area = 1.e0
constraint.mineral_list.append(mineral)			# assign mineral object
dat.add(constraint)	# assign constraint

# west condition - 2nd condition
constraint = pconstraint()
constraint.name = 'inlet_constraint'
constraint.concentration_list = [] # Assigning for this done below
constraint.mineral_list = []
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
dat.add(constraint)	# assign constraint
#--------------------------------------------------------------

# Testing write
#dat.write('calcite_tran_only.in')

print('******************************************')
print('Writing PFLOTRAN input file and executing it.')
print('******************************************')

executable = del_extra_slash(pflotran_dir + '/src/pflotran/pflotran')
# Write to file and execute that input file
dat.write('calcite_tran_only.in')


dat.run(input='calcite_tran_only.in',exe=executable)


#------------------------------
#       Plotting the pH
#------------------------------


print('******************************************')
print('Plotting in matplotlib')
print('******************************************')

path = []
path.append('.')

files = pft.get_tec_filenames('calcite_tran_only',range(1,6))
filenames = pft.get_full_paths(path,files)

f = plt.figure(figsize=(6,6))
plt.subplot(1,1,1)
f.suptitle("1D Calcite",fontsize=16)
plt.xlabel('X [m]')
plt.ylabel('pH')

#plt.xlim(0.,1.)
plt.ylim(4.8,8.2)
#plt.grid(True)

for ifile in range(len(filenames)):
  data = pft.Dataset(filenames[ifile],1,4)
  plt.plot(data.get_array('x'),data.get_array('y'),label=data.title)

#'best'         : 0, (only implemented for axis legends)
#'upper right'  : 1,
#'upper left'   : 2,
#'lower left'   : 3,
#'lower right'  : 4,
#'right'        : 5,
#'center left'  : 6,
#'center right' : 7,
#'lower center' : 8,
#'upper center' : 9,
#'center'       : 10,
plt.legend(loc=4,title='Time [y]')
# xx-small, x-small, small, medium, large, x-large, xx-large, 12, 14
plt.setp(plt.gca().get_legend().get_texts(),fontsize='small')
#plt.setp(plt.gca().get_legend().get_texts(),linespacing=0.)
plt.setp(plt.gca().get_legend().get_frame().set_fill(False))
plt.setp(plt.gca().get_legend().draw_frame(False))
#plt.gca().yaxis.get_major_formatter().set_powerlimits((-1,1))

f.subplots_adjust(hspace=0.2,wspace=0.2,
                  bottom=.12,top=.9,
                  left=.12,right=.9)

#plt.show()
plt.savefig('pH_tran_only.pdf')