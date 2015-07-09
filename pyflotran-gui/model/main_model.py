from pyflotran.pdata import *
from object_space import *


names = {'Zero Gradient': 'zero_gradient', 'Hydrostatic': 'hydrostatic', 'Dirichlet': 'dirichlet',
         'Mass Rate': 'mass_rate', 'Dirichlet Zero Gradient': 'dirichlet_zero_gradient'}


def str_to_bool(value):
    return value.lower() in ('True', 'true', 'yes', 'Yes')


class PlotObservationObject(object):
    def __init__(self, variable_list, observation_list, observation_filenames, plot_filename, legend_list,
                 legend_font_size, x_label, x_type, x_range_1, x_range_2, y_label, y_type, y_range_1,
                 y_range_2):
        """
        Plot Observation Object
        :param variable_list: str
        :param observation_list: str
        :param observation_filenames: list
        :param plot_filename: str
        :param legend_list: str
        :param legend_font_size: float
        :param x_label: str
        :param x_type: str
        :param x_range_1: float
        :param x_range_2: float
        :param y_label: str
        :param y_type: str
        :param y_range_1: float
        :param y_range_2: float
        :return:
        """
        self.variable_list = variable_list
        self.observation_list = observation_list
        self.observation_filenames = observation_filenames
        self.plot_filename = plot_filename
        self.legend_list = legend_list
        self.legend_font_size = legend_font_size
        self.x_label = x_label
        self.x_type = x_type
        self.x_range_1 = x_range_1
        self.x_range_2 = x_range_2
        self.y_label = y_label
        self.y_type = y_type
        self.y_range_1 = y_range_1
        self.y_range_2 = y_range_2

        self.__add_to_dict()

    def __add_to_dict(self):
        PLOT_OBSERVATION_OBJECT[0] = self

    @property
    def __unicode__(self):
        return str(id(self))


def apply_simulation_settings(s_flow, s_trans, s_geo, mode):
    simulation = psimulation()

    simulation.simulation_type = 'subsurface'
    if s_flow is True:
        simulation.subsurface_flow = 'flow'
    if s_trans is True:
        simulation.subsurface_transport = 'transport'
    if s_geo is True:
        pass

    simulation.mode = mode

    SIMULATION_OBJECT[0] = simulation
    ALL_OBJECTS[str(id(simulation))] = simulation

    return str(id(simulation))


def apply_grid_settings(grid_type, lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, origin_x, origin_y, origin_z,
                        g_spacing_x, g_spacing_y, g_spacing_z, gravity_x, gravity_y, gravity_z,
                        g_cell_x, g_cell_y, g_cell_z, file_path):
    grid = pgrid()
    if grid_type == 'structured':
        grid.type = 'structured'
        grid.lower_bounds = [lb_x, lb_y, lb_z]
        grid.upper_bounds = [ub_x, ub_y, ub_z]
        grid.origin = [origin_x, origin_y, origin_z] if (origin_x or origin_y or origin_z) != 0 else []
        grid.nxyz = [g_cell_x, g_cell_y, g_cell_z]
        grid.dx = g_spacing_x
        grid.dy = g_spacing_y
        grid.dz = g_spacing_z
        grid.gravity = [gravity_x, gravity_y, gravity_z] if (gravity_x or gravity_y or gravity_z) != 0 else []
    else:
        grid_type.type = 'unstructured'
        grid.filename = file_path
        grid.gravity = [gravity_x, gravity_y, gravity_z]

    GRID_OBJECT[0] = grid
    ALL_OBJECTS[str(id(grid))] = grid

    return str(id(grid))


def apply_material_properties(material_id, name, char_curve, porosity, tortuosity, density, specific_heat,
                              condition_dry, condition_wet, saturation, perm_x, perm_y, perm_z):
    material = pmaterial()
    material.id = material_id
    material.name = name
    material.characteristic_curves = char_curve
    material.porosity = porosity
    material.tortuosity = tortuosity
    material.density = density
    material.specific_heat = specific_heat
    material.cond_dry = condition_dry
    material.cond_wet = condition_wet
    material.saturation = saturation
    material.permeability = [perm_x, perm_y, perm_z] if (perm_x or perm_y or perm_z) != 0 else []

    MATERIAL_PROP_OBJECT[0] = material
    ALL_OBJECTS[str(id(material))] = material

    return str(id(material))


def apply_output_settings(time_value_list, time_value_list_unit, print_column_id_bool, screen_output,
                          screen_periodic, periodic_time, periodic_time_unit, periodic_timestep,
                          periodic_time_step_unit, periodic_observation_time, periodic_observation_time_unit,
                          periodic_obs_ts, tecplot_block, tecplot_point, hdf5, hdf5_multi, vtk, permeability, porosity,
                          velocities, mass_balance, variables_list):
    output = poutput

    if time_value_list:
        # strip time_value_list of white spaces and separate into a list
        time_value_list = time_value_list.replace(' ', '').split(',')
        time_list = [time_value_list_unit]
        for time_value in time_value_list:
            time_list.append(time_value)

        output.time_list = time_list

    output.print_column_ids = str_to_bool(print_column_id_bool)
    output.screen_output = str_to_bool(screen_output)
    output.screen_periodic = screen_periodic
    output.periodic_time = [periodic_time, periodic_time_unit] if periodic_time != 0 else []
    output.periodic_timestep = [periodic_timestep, periodic_time_step_unit] if periodic_timestep != 0 else []
    output.periodic_observation_time = [periodic_observation_time, periodic_observation_time_unit] if \
        periodic_observation_time != 0 else []
    output.periodic_observation_timestep = periodic_obs_ts

    format_list = []
    if tecplot_block:
        format_list.append('TECPLOT BLOCK')
    if tecplot_point:
        format_list.append('TECPLOT POINT')
    if hdf5:
        format_list.append('HDF5')
    if hdf5_multi:
        format_list.append('HDF5 MULTIPLE_FILES')
    if vtk:
        format_list.append('VTK')
    if len(format_list) > 0:
        output.format_list = format_list
    output._permeability = str_to_bool(permeability)
    output._porosity = str_to_bool(porosity)
    output.velocities = str_to_bool(velocities)
    output.mass_balance = str_to_bool(mass_balance)
    if len(variables_list) > 0:
        output.variables_list = variables_list.replace(' ', '').split(',')

    OUTPUT_OBJECT[0] = output
    ALL_OBJECTS[str(id(output))] = output

    return str(id(output))


def apply_observation_setting(region):
    observation = pobservation(region=region)

    OBSERVATION_OBJECT[0] = observation
    ALL_OBJECTS[str(id(observation))] = observation

    return str(id(observation))


def apply_strata_settings(region, material):
    strata = pstrata(region=region,
                     material=material)

    STRATA_OBJECT[0] = strata
    ALL_OBJECTS[str(id(strata))] = strata

    return str(id(strata))


def apply_flow_variable(name, var_type, value_list_1, value_list_2, unit, time_unit_type, data_unit_type,
                        flow_variable_list_objects, flow_variable_list):

    def float_each(a_list):
        new_list = []
        for element in a_list:
            new_list.append(float(element))
        return new_list

    f_var_list = []
    for item in flow_variable_list:
        temp_var_list = pflow_variable_list(item[0], float_each(item[1].replace(' ', '').split(',')))
        f_var_list.append(temp_var_list)

    flow_variable = pflow_variable(name=name,
                                   type=names[var_type])
    if len(flow_variable_list) > 0:
        flow_variable.list = f_var_list
        flow_variable.time_unit_type = time_unit_type
        flow_variable.data_unit_type = data_unit_type
    else:
        if value_list_2 != 0:
            flow_variable.valuelist = [value_list_1, value_list_2]
        else:
            flow_variable.valuelist = [value_list_1]
            flow_variable.unit = unit.upper()

    obj_id = str(id(flow_variable))
    FLOW_VARIABLES_OBJECTS[obj_id] = flow_variable
    ALL_OBJECTS[obj_id] = flow_variable

    return obj_id


def apply_flow_settings(name, unit_list, iphase, sync_ts_update, data_unit, dx, dy, dz, datum_filepath, variable_list):
    new_variable_list = []
    for flow_id in variable_list:
        new_variable_list.append(FLOW_VARIABLES_OBJECTS[flow_id])

    flow = pflow(name=name,
                 units_list=unit_list,
                 iphase=iphase if iphase != 0 else None,
                 sync_timestep_with_update=str_to_bool(sync_ts_update),
                 varlist=new_variable_list
                 )

    flow.datum_type = ''
    flow.datum = []
    if data_unit == 'list':
        if (dx or dy or dz) != 0:
            flow.datum_type = 'dataset'
            flow.datum.append([dx, dy, dz])
    elif data_unit == 'filename':
        flow.datum_type = 'file'
        flow.datum = datum_filepath

    obj_id = str(id(flow))
    FLOW_AND_VARIABLES_OBJECTS[obj_id] = flow
    ALL_OBJECTS[obj_id] = flow

    return obj_id


def apply_transport_condition(name, constraint_type, condition_list):
    # build constraint_type, constraint_list_value, constraint_list_type
    constraint_list_value = []
    constraint_list_type = []
    for condition in condition_list:
        temp_condition_list = condition.replace(' ', '').split(',')
        constraint_list_value.append(float(temp_condition_list[0]))
        constraint_list_type.append(temp_condition_list[1])
    transport = ptransport(name=name,
                           type=names[constraint_type],
                           constraint_list_value=constraint_list_value,
                           constraint_list_type=constraint_list_type)

    obj_id = str(id(transport))
    TRANSPORT_OBJECTS[obj_id] = transport
    ALL_OBJECTS[obj_id] = transport

    return obj_id


def apply_constraint_condition_mineral(name, volume_fraction, surface_area):
    cc_mineral = pconstraint_mineral(name=name,
                                     volume_fraction=volume_fraction,
                                     surface_area=surface_area)

    obj_id = str(id(cc_mineral))
    CC_MINERAL[obj_id] = cc_mineral
    ALL_OBJECTS[obj_id] = cc_mineral

    return obj_id


def apply_constraint_condition_concentration(species, value, constraint, element):
    cc_concentration = pconstraint_concentration(pspecies=species,
                                                 value=value,
                                                 constraint=constraint,
                                                 element=element)

    obj_id = str(id(cc_concentration))
    CC_CONCENTRATION[obj_id] = cc_concentration
    ALL_OBJECTS[obj_id] = cc_concentration

    return obj_id


def apply_constraint_condition(name, concentration_list, mineral_list):
    concentrations = []
    minerals = []
    for cc1 in concentration_list:
        concentrations.append(cc1.replace(' ', '').split('id:')[1])
    for cc2 in mineral_list:
        minerals.append(cc2.replace(' ', '').split('id:')[1])

    cc = pconstraint(name=name,
                     concentration_list=[CC_CONCENTRATION[key] for key in concentrations],
                     mineral_list=[CC_MINERAL[key] for key in minerals])

    obj_id = str(id(cc))

    CONSTRAINT_CONDITION_OBJECTS[obj_id] = cc
    ALL_OBJECTS[obj_id] = cc

    return obj_id


def apply_initial_condition(name, flow, transport, region):
    init_condition = pinitial_condition(flow=flow,
                                        transport=transport,
                                        region=region,
                                        name=name)

    obj_id = str(id(init_condition))

    INITIAL_CONDITION_OBJECT[0] = init_condition
    ALL_OBJECTS[obj_id] = init_condition

    return obj_id


def apply_boundary_condition(name, flow, transport, region):
    boundary_condition = pboundary_condition(name=name,
                                             flow=flow if flow != 'None' else None,
                                             transport=transport if transport != 'None' else None,
                                             region=region)

    obj_id = str(id(boundary_condition))

    BOUNDARY_CONDITION_OBJECTS[obj_id] = boundary_condition
    ALL_OBJECTS[obj_id] = boundary_condition

    return obj_id


def apply_source_sink_condition(name, flow, transport, region):
    ss = psource_sink(name=name,
                      flow=flow,
                      transport=transport,
                      region=region)

    obj_id = str(id(ss))

    SOURCE_SINK_OBJECT[0] = ss
    ALL_OBJECTS[obj_id] = ss

    return obj_id


def apply_newton_solver(name, abs_tolerance, rel_tolerance, rel_tolerance_update, div_tolerance, tol_compared_inf_norm,
                        max_iter, max_func_eval):
    newton_s = pnsolver(name=name,
                        atol=float(abs_tolerance),
                        rtol=float(rel_tolerance),
                        stol=float(rel_tolerance_update),
                        dtol=float(div_tolerance) if div_tolerance != '' else None,
                        itol=float(tol_compared_inf_norm),
                        max_it=max_iter,
                        max_f=max_func_eval)

    obj_id = str(id(newton_s))

    NEWTON_SOLVER_OBJECTS[obj_id] = newton_s
    ALL_OBJECTS[obj_id] = newton_s

    return obj_id


def apply_linear_solver_settings(name, solver, preconditioner):
    l_solver = plsolver(name=name,
                        solver=solver,
                        preconditioner=preconditioner if preconditioner != 'None' else '')

    obj_id = str(id(l_solver))

    LINEAR_SOLVER_OBJECTS[obj_id] = l_solver
    ALL_OBJECTS[obj_id] = l_solver

    return obj_id


def apply_timestepper_settings(ts_mode, ts_acceleration, num_steps_after_cut, max_steps, max_ts_cuts,
                               cfl_limiter, init_steady_state, run_as_steady_state, max_pressure_change,
                               max_temp_change, max_concentration_change, max_saturation_change):
    ts = ptimestepper(ts_mode=ts_mode.lower(),
                      ts_acceleration=ts_acceleration,
                      num_steps_after_cut=num_steps_after_cut,
                      max_steps=max_steps,
                      max_ts_cuts=max_ts_cuts,
                      cfl_limiter=cfl_limiter,
                      initialize_to_steady_state=str_to_bool(init_steady_state),
                      run_as_steady_state=str_to_bool(run_as_steady_state),
                      max_pressure_change=max_pressure_change,
                      max_temperature_change=max_temp_change,
                      max_concentration_change=max_concentration_change,
                      max_saturation_change=max_saturation_change)

    obj_id = str(id(ts))

    TIMESTEPPER_OBJECT[0] = ts
    ALL_OBJECTS[obj_id] = ts

    return obj_id


def apply_time_settings(final_t_val, final_t_unit, delta_t_init_value, delta_t_init_unit, delta_t_inst_value,
                        delta_t_inst_unit, list_items, run_as_steady_state):
    dtf_list = []
    for time_block in list_items:
        dti_dtf = time_block.split(',')
        dti = dti_dtf[0].split(' ')
        dtf = dti_dtf[1].split(' ')
        del dtf[0]  # remove blank entry
        dtf_list.append([float(dti[0]), str(dti[1]), float(dtf[0]), str(dtf[1])])
    time_inst = ptime(tf=[final_t_val, final_t_unit],
                      dti=[delta_t_init_value, delta_t_init_unit],
                      dtf=[delta_t_inst_value, delta_t_inst_unit],
                      dtf_list=dtf_list,
                      steady_state=str_to_bool(run_as_steady_state))

    obj_id = str(id(time_inst))

    TIME_OBJECT[0] = time_inst
    ALL_OBJECTS[obj_id] = time_inst

    return obj_id


def apply_checkpoint_settings(frequency, overwrite):
    checkpoint_inst = pcheckpoint(frequency=frequency,
                                  overwrite=str_to_bool(overwrite))

    obj_id = str(id(checkpoint_inst))

    CHEMISTRY_OBJECT[0] = checkpoint_inst
    ALL_OBJECTS[obj_id] = checkpoint_inst

    return obj_id


def apply_restart_settings(filename, time_value, time_value_unit):
    restart_inst = prestart(file_name=filename,
                            time_value=time_value,
                            time_unit=time_value_unit)

    obj_id = str(id(restart_inst))

    RESTART_OBJECT[0] = restart_inst
    ALL_OBJECTS[obj_id] = restart_inst

    return obj_id


def apply_uniform_velocity_settings(vlx, vly, vlz, unit):
    uni_vel_inst = puniform_velocity([vlx, vly, vlz, unit])

    obj_id = str(id(uni_vel_inst))

    UNIFORM_VELOCITY_OBJECT[0] = uni_vel_inst
    ALL_OBJECTS[obj_id] = uni_vel_inst

    return obj_id


def apply_nonuniform_velocity(filepath):
    nonuniform_vel = pnonuniform_velocity(filename=filepath)

    obj_id = str(id(nonuniform_vel))

    NONUNIFORM_VELOCITY_OBJECT[0] = nonuniform_vel
    ALL_OBJECTS[obj_id] = nonuniform_vel

    return obj_id


def apply_saturation_function(name, permeability_func_type, sat_func_type, residual_sat, residual_sat_value,
                              residual_sat_liq, residual_sat_liq_value,  residual_sat_gas, residual_sat_gas_value,
                              a_lambda, alpha, max_cap_pressure, beta, power):
    sat_function = psaturation(name=name,
                               permeability_function_type=permeability_func_type if permeability_func_type != 'None'
                               else None,
                               saturation_function_type=sat_func_type if sat_func_type != 'None' else None,
                               residual_saturation=residual_sat_value if residual_sat_value != 0.0 else None,
                               residual_saturation_liquid=residual_sat_liq_value,
                               residual_saturation_gas=residual_sat_gas_value,
                               a_lambda=a_lambda,
                               alpha=float(alpha),
                               max_capillary_pressure=max_cap_pressure,
                               betac=beta,
                               power=power)

    obj_id = str(id(sat_function))

    SATURATION_FUNCTION_OBJECT[0] = sat_function
    ALL_OBJECTS[obj_id] = sat_function

    return obj_id


def apply_fluid_properties(diffusion_value):
    fluid_property = pfluid(diffusion_coefficient=float(diffusion_value))

    obj_id = str(id(fluid_property))

    FLUID_PROPERTIES_OBJECT[0] = fluid_property
    ALL_OBJECTS[obj_id] = fluid_property

    return obj_id


def apply_chemistry_settings(species_list, aq_species_list, gas_species_list, mineral_list,
                             log_formulation, database, activity_co, molal, output_list, m_kinetic_list):
    mineral_kinetic_list = []
    for m_kinetic in m_kinetic_list:
        temp = m_kinetic.split(',')  # 0: name 1: rate value 2: rate unit
        m_kinetic_obj = pchemistry_m_kinetic(name=temp[0],
                                             rate_constant_list=[temp[1], temp[2]])
        mineral_kinetic_list.append(m_kinetic_obj)

    chem_inst = pchemistry(pspecies_list=species_list.replace(' ', '').split(',') if species_list != '' else [],
                           sec_species_list=aq_species_list.replace(' ', '').split(',') if aq_species_list != '' else [],
                           gas_species_list=gas_species_list.replace(' ', '').split(',') if gas_species_list != '' else [],
                           minerals_list=mineral_list.replace(' ', '').split(',') if mineral_list != '' else [],
                           m_kinetics_list=mineral_kinetic_list if len(mineral_kinetic_list) > 0 else [],
                           log_formulation=str_to_bool(log_formulation),
                           database=database if database != '' else None,
                           activity_coefficients=activity_co if activity_co != 'None' else None,
                           molal=str_to_bool(molal),
                           output_list=output_list.replace(' ', '').split(','))

    obj_id = str(id(chem_inst))

    CHEMISTRY_OBJECT[0] = chem_inst
    ALL_OBJECTS[obj_id] = chem_inst

    return obj_id


def apply_regression_settings(cells, cells_per_process):
    reg = pregression(cells=cells.replace(' ', '').split(','),
                      cells_per_process=cells_per_process)

    obj_id = str(id(reg))

    REGRESSION_OBJECT[0] = reg
    ALL_OBJECTS[obj_id] = reg

    return obj_id


def apply_dataset_settings(dataset_name, dataset_mapped_name, name, filename, hdf5_dataset_name,
                           map_hdf5_dataset_name, max_buffer_value):
    ds_inst = pdataset(dataset_name=dataset_name,
                       dataset_mapped_name=dataset_mapped_name,
                       name=name,
                       file_name=filename,
                       hdf5_dataset_name=hdf5_dataset_name,
                       map_hdf5_dataset_name=map_hdf5_dataset_name,
                       max_buffer_size=max_buffer_value)

    obj_id = str(id(ds_inst))

    DATASET_OBJECT[0] = ds_inst
    ALL_OBJECTS[obj_id] = ds_inst

    return obj_id


def apply_characteristic_curve_setting(name, sat_func_type, sf_alpha, sf_m, sf_lambda, sf_liq_residual_sat,
                                       sf_gas_residual_sat, max_cap_pressure, smooth, power, default,
                                       liq_perm_func_type, lpf_m, lpf_lambda, lpf_liq_residual_sat, gas_perm_func_type,
                                       gpf_m, gpf_lambda, gpf_liq_residual_saturation, gpf_gas_residual_saturation):
    cc_inst = pcharacteristic_curves(name=name,
                                     saturation_function_type=sat_func_type,
                                     sf_alpha=sf_alpha,
                                     sf_m=sf_m,
                                     sf_lambda=sf_lambda,
                                     sf_liquid_residual_saturation=sf_liq_residual_sat,
                                     sf_gas_residual_saturation=sf_gas_residual_sat,
                                     max_capillary_pressure=max_cap_pressure,
                                     smooth=smooth,
                                     power=power,
                                     default=default,
                                     liquid_permeability_function_type=liq_perm_func_type,
                                     lpf_m=lpf_m,
                                     lpf_lambda=lpf_lambda,
                                     lpf_liquid_residual_saturation=lpf_liq_residual_sat,
                                     gas_permeability_function_type=gas_perm_func_type,
                                     gpf_m=gpf_m,
                                     gpf_lambda=gpf_lambda,
                                     gpf_liquid_residual_saturation=gpf_liq_residual_saturation,
                                     gpf_gas_residual_saturation=gpf_gas_residual_saturation)

    obj_id = str(id(cc_inst))

    CHARACTERISTIC_CURVE_OBJECT[0] = cc_inst
    ALL_OBJECTS[obj_id] = cc_inst

    return obj_id


def apply_region_settings(name, lower_x, lower_y, lower_z, upper_x, upper_y, upper_z, face):
    region = pregion(name=name,
                     coordinates_lower=[lower_x, lower_y, lower_z],
                     coordinates_upper=[upper_x, upper_y, upper_z],
                     face=face if face != 'None' else '')

    obj_id = str(id(region))

    REGION_OBJECTS[obj_id] = region
    ALL_OBJECTS[obj_id] = region

    return obj_id


def plot_observation_settings(variable_list, observation_list, observation_filenames, plot_filename, legend_list,
                              legend_font_size, x_label, x_type, x_range_1, x_range_2, y_label, y_type, y_range_1,
                              y_range_2):
    obs_filenames = observation_filenames.split('\n')
    po_inst = PlotObservationObject(variable_list, observation_list, obs_filenames, plot_filename, legend_list,
                                    legend_font_size, x_label, x_type, x_range_1, x_range_2, y_label, y_type, y_range_1,
                                    y_range_2)
    return po_inst.__unicode__


def generate_input_file(main_view_inst):
    dat = pdata()

    if isinstance(SIMULATION_OBJECT[0], psimulation):
        dat.simulation = SIMULATION_OBJECT[0]

    if isinstance(GRID_OBJECT[0], pgrid):
        dat.grid = GRID_OBJECT[0]

    if isinstance(MATERIAL_PROP_OBJECT[0], pmaterial):
        dat.add(MATERIAL_PROP_OBJECT[0])

    if isinstance(OUTPUT_OBJECT[0], poutput):
        dat.output = OUTPUT_OBJECT[0]

    if isinstance(OBSERVATION_OBJECT[0], pobservation):
        dat.add(obj=OBSERVATION_OBJECT[0], overwrite=True)

    if isinstance(STRATA_OBJECT[0], pstrata):
        dat.add(obj=STRATA_OBJECT[0])

    for flow in FLOW_AND_VARIABLES_OBJECTS.values():
        if isinstance(flow, pflow):
            dat.add(obj=flow)
    for transport in TRANSPORT_OBJECTS.values():
        if isinstance(transport, ptransport):
            dat.add(obj=transport)

    for cc in CONSTRAINT_CONDITION_OBJECTS.values():
        if isinstance(cc, pconstraint):
            dat.add(obj=cc)

    if isinstance(INITIAL_CONDITION_OBJECT[0], pinitial_condition):
        dat.add(obj=INITIAL_CONDITION_OBJECT[0])

    for boundary_condition in BOUNDARY_CONDITION_OBJECTS.values():
        if isinstance(boundary_condition, pboundary_condition):
            dat.add(obj=boundary_condition)

    if isinstance(SOURCE_SINK_OBJECT[0], psource_sink):
        dat.add(obj=SOURCE_SINK_OBJECT[0])

    for n_solver in NEWTON_SOLVER_OBJECTS.values():
        if isinstance(n_solver, pnsolver):
            dat.add(obj=n_solver)

    for l_solver in LINEAR_SOLVER_OBJECTS.values():
        if isinstance(l_solver, plsolver):
            dat.add(obj=l_solver)

    if isinstance(TIMESTEPPER_OBJECT[0], ptimestepper):
        dat.timestepper = TIMESTEPPER_OBJECT[0]

    if isinstance(TIME_OBJECT[0], ptime):
        dat.time = TIME_OBJECT[0]

    if isinstance(CHECKPOINT_OBJECT[0], pcheckpoint):
        dat.checkpoint = CHECKPOINT_OBJECT[0]

    if isinstance(RESTART_OBJECT[0], prestart):
        dat.restart = RESTART_OBJECT[0]

    if isinstance(UNIFORM_VELOCITY_OBJECT[0], puniform_velocity):
        dat.uniform_velocity = UNIFORM_VELOCITY_OBJECT[0]

    if isinstance(NONUNIFORM_VELOCITY_OBJECT[0], pnonuniform_velocity):
        dat.nonuniform_velocity = NONUNIFORM_VELOCITY_OBJECT[0]

    if isinstance(SATURATION_FUNCTION_OBJECT[0], psaturation):
        dat.add(SATURATION_FUNCTION_OBJECT[0])

    if isinstance(FLUID_PROPERTIES_OBJECT[0], pfluid):
        dat.fluid = FLUID_PROPERTIES_OBJECT[0]

    if isinstance(CHEMISTRY_OBJECT[0], pchemistry):
        dat.chemistry = CHEMISTRY_OBJECT[0]

    if isinstance(REGRESSION_OBJECT[0], pregression):
        dat.regression = REGRESSION_OBJECT[0]

    if isinstance(DATASET_OBJECT[0], pdataset):
        dat.add(obj=DATASET_OBJECT[0])

    if isinstance(CHARACTERISTIC_CURVE_OBJECT[0], pcharacteristic_curves):
        dat.add(obj=CHARACTERISTIC_CURVE_OBJECT[0])

    for region in REGION_OBJECTS.values():
        if isinstance(region, pregion):
            dat.add(obj=region)
    try:
        dat.write('file.in')
    except PyFLOTRAN_ERROR, e:
        main_view_inst.log_append_message(e)

