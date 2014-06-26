import sys
from pdata import*

sys.path.append('../.')
dat = pdata('mphase.in')

###############################################################

# Print to console the data attributes

print '\n\nEXECUTING\n\n'

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

dat.write('mphase2.in')
