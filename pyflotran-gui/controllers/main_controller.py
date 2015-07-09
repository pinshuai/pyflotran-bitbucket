from model import main_model, flow_models, object_space

main_view = None

# Generic function which will check input fields and return False if it is blank. Required args: str
def validate_text(*args):
    for arg in args:
        if arg == '':
            return False
    return True


# Generic function to check if a checkbox is checked. Required args: bool
def validate_check_boxes(*args):
    for arg in args:
        if arg:
            return True
    return False


def open_file(file_location):
    return validate_text(file_location)


def get_flow_variable_types(current_index):
    return flow_models.get_flow_variable_types(current_index)


def generate_input_file():
    if main_model.generate_input_file(main_view):
        return True
    return False


def apply_simulation_settings(s_flow, s_trans, s_geo, mode):
    if validate_check_boxes(s_flow, s_trans, s_geo):
        return main_model.apply_simulation_settings(s_flow, s_trans, s_geo, mode)


def apply_material_properties(material_id, name, char_curve, porosity, tortuosity, density, specific_heat,
                              condition_dry, condition_wet, saturation, perm_x, perm_y, perm_z):
    if validate_text(name):
        return main_model.apply_material_properties(material_id, name, char_curve, porosity, tortuosity, density,
                                                    specific_heat, condition_dry, condition_wet, saturation, perm_x,
                                                    perm_y, perm_z)


def apply_grid_settings(grid_type, lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, origin_x, origin_y, origin_z, g_spacing_x,
                        g_spacing_y, g_spacing_z, gravity_x, gravity_y, gravity_z, g_cell_x, g_cell_y, g_cell_z,
                        file_path):
    return main_model.apply_grid_settings(grid_type, lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, origin_x, origin_y,
                                          origin_z, g_spacing_x, g_spacing_y, g_spacing_z, gravity_x, gravity_y,
                                          gravity_z, g_cell_x, g_cell_y, g_cell_z, file_path)


def apply_time_settings(final_t_val, final_t_unit, delta_t_init_value, delta_t_init_unit, delta_t_inst_value,
                        delta_t_inst_unit, list_items, run_as_steady_state):
    if validate_text(final_t_val, final_t_unit, delta_t_init_value, delta_t_init_unit, delta_t_inst_value,
                     delta_t_inst_unit):

        return main_model.apply_time_settings(final_t_val, final_t_unit, delta_t_init_value, delta_t_init_unit,
                                              delta_t_inst_value, delta_t_inst_unit, list_items,
                                              run_as_steady_state)
    else:
        return None


def apply_timestepper_settings(ts_mode, ts_acceleration, num_steps_after_cut, max_steps, max_ts_cuts, cfl_limiter,
                               init_steady_state, run_as_steady_state, max_pressure_change, max_temp_change,
                               max_concentration_change, max_saturation_change):
    if validate_text(ts_acceleration):
        return main_model.apply_timestepper_settings(ts_mode, ts_acceleration, num_steps_after_cut, max_steps,
                                                     max_ts_cuts, cfl_limiter, init_steady_state, run_as_steady_state,
                                                     max_pressure_change, max_temp_change, max_concentration_change,
                                                     max_saturation_change)
    else:
        return None


def apply_newton_solver_settings(name, abs_tolerance, rel_tolerance, rel_tolerance_update, div_tolerance,
                                 tol_compared_inf_norm, max_iter, max_func_eval):
    if validate_text(abs_tolerance, rel_tolerance, rel_tolerance_update, tol_compared_inf_norm,
                     max_iter, max_func_eval):
        return main_model.apply_newton_solver(name, abs_tolerance, rel_tolerance, rel_tolerance_update, div_tolerance,
                                              tol_compared_inf_norm, max_iter, max_func_eval)
    else:
        return None


def apply_linear_solver_settings(name, solver, preconditioner):
    return main_model.apply_linear_solver_settings(name, solver, preconditioner)


def apply_output_settings(time_value_list, time_value_list_unit, print_column_id_bool, screen_output,
                          screen_periodic, periodic_time, periodic_time_unit, periodic_timestep,
                          periodic_time_step_unit, periodic_observation_time, periodic_observation_time_unit,
                          periodic_obs_timestep, tecplot_block, tecplot_point, hdf5, hdf5_multi, vtk, permeability,
                          porosity, velocities, mass_balance, variables_list):
    if validate_check_boxes(tecplot_block, tecplot_point, hdf5, hdf5_multi, vtk):
        return main_model.apply_output_settings(time_value_list, time_value_list_unit, print_column_id_bool,
                                                screen_output, screen_periodic, periodic_time, periodic_time_unit,
                                                periodic_timestep, periodic_time_step_unit, periodic_observation_time,
                                                periodic_observation_time_unit, periodic_obs_timestep, tecplot_block,
                                                tecplot_point, hdf5, hdf5_multi, vtk, permeability, porosity,
                                                velocities, mass_balance, variables_list)
    else:
        return None


def apply_saturation_func_settings(name, permeability_func_type, sat_func_type, residual_sat, residual_sat_value,
                                   residual_sat_liq, residual_sat_liq_value, residual_sat_gas, residual_sat_gas_value,
                                   a_lambda, alpha, max_cap_pressure, beta, power):
    if validate_text(name):
        return main_model.apply_saturation_function(name, permeability_func_type, sat_func_type, residual_sat,
                                                    residual_sat_value, residual_sat_liq, residual_sat_liq_value,
                                                    residual_sat_gas, residual_sat_gas_value, a_lambda, alpha,
                                                    max_cap_pressure, beta, power)
    else:
        return None


def apply_flow_variable_adder_settings(name, var_type, value_list_1, value_list_2, unit, time_unit_type, data_unit_type,
                                       flow_variable_list_objects, flow_variable_list):
    flow_list = []
    for value_pair in flow_variable_list:
        value_1, value_2 = value_pair.replace(' ', '').split(',')
        flow_list.append((value_1, value_2))

    if validate_text(value_list_1):
        return main_model.apply_flow_variable(name, var_type, value_list_1, value_list_2, unit, time_unit_type,
                                              data_unit_type, flow_variable_list_objects, flow_list)
    else:
        return None


def apply_flow_settings(name, unit_list, iphase, sync_ts_update, data_unit, dx, dy, dz, datum_filepath, variable_list):
    if validate_text(name):
        return main_model.apply_flow_settings(name, unit_list, iphase, sync_ts_update, data_unit, dx, dy, dz,
                                              datum_filepath, variable_list)
    else:
        return None


def apply_initial_condition(name, flow, transport, region):
    if validate_text(name, region):
        return main_model.apply_initial_condition(name, flow, transport, region)
    else:
        return None


def apply_boundary_condition(name, flow, transport, region):
    if validate_text(name, region):
        return main_model.apply_boundary_condition(name, flow, transport, region)
    else:
        return None


def apply_source_sink_condition(name, flow, transport, region):
    if validate_text(name, region):
        return main_model.apply_source_sink_condition(name, flow, transport, region)
    else:
        return None


def apply_stratigraphy_settings(name, material):
    if validate_text(name, material):
        return main_model.apply_strata_settings(name, material)
    else:
        return None


def apply_checkpoint_settings(frequency, overwrite):
    if validate_text(frequency):
        return main_model.apply_checkpoint_settings(frequency, overwrite)
    else:
        return None


def apply_restart_settings(filename, time_value, time_value_unit):
    if validate_text(filename):
        return main_model.apply_restart_settings(filename, time_value, time_value_unit)
    else:
        return None


def apply_uniform_velocity_settings(vlx, vly, vlz, unit):
    if validate_text(unit):
        return main_model.apply_uniform_velocity_settings(vlx, vly, vlz, unit)
    else:
        return None


def nonuniform_velocity(filepath):
    return main_model.apply_nonuniform_velocity(filepath)


def apply_fluid_properties(diffusion_value):
    return main_model.apply_fluid_properties(diffusion_value)


def apply_observation_settings(region):
    if validate_text(region):
        return main_model.apply_observation_setting(region)
    else:
        return None


def apply_chemistry_settings(species_list, aq_species_list, gas_species_list, mineral_list,
                             log_formulation, database, activity_co, molal, output_list, m_kinetic_list):
    if validate_text(species_list):
        return main_model.apply_chemistry_settings(species_list, aq_species_list, gas_species_list, mineral_list,
                                                   log_formulation, database, activity_co, molal, output_list,
                                                   m_kinetic_list)
    else:
        return None


def apply_transport_condition(name, constraint_type, condition_list):
    if validate_text(name):
        if len(condition_list) > 0:
            return main_model.apply_transport_condition(name, constraint_type, condition_list)
    else:
        return None


def apply_constraint_condition(name, concentration_list, mineral_list):
    if validate_text(name):
        return main_model.apply_constraint_condition(name, concentration_list, mineral_list)
    else:
        return None


def apply_constraint_condition_mineral(name, volume_fraction, surface_area):
    if validate_text(name):
        return main_model.apply_constraint_condition_mineral(name, volume_fraction, surface_area)
    else:
        return None


def apply_constraint_condition_concentration(species, value, constraint, element):
    if validate_text(species):
        return main_model.apply_constraint_condition_concentration(species, value, constraint, element)
    else:
        return None


def apply_regression_settings(cells, cells_per_process):
    if validate_text(cells):
        return main_model.apply_regression_settings(cells, cells_per_process)
    else:
        return None


def apply_dataset_settings(dataset_name, dataset_mapped_name, name, filename, hdf5_dataset_name, map_hdf5_dataset_name,
                           max_buffer_value):
    if validate_text(dataset_name, dataset_mapped_name, name, filename, hdf5_dataset_name, map_hdf5_dataset_name):
        return main_model.apply_dataset_settings(dataset_name, dataset_mapped_name, name, filename, hdf5_dataset_name,
                                                 map_hdf5_dataset_name, max_buffer_value)
    else:
        return None


def apply_region_settings(name, lower_x, lower_y, lower_z, upper_x, upper_y, upper_z, face):
    if validate_text(name):
        return main_model.apply_region_settings(name, lower_x, lower_y, lower_z, upper_x, upper_y, upper_z, face)
    else:
        return None


def plot_observation_settings(variable_list, observation_list, observation_filenames, plot_filename, legend_list,
                              legend_font_size, x_label, x_type, x_range_1, x_range_2, y_label, y_type, y_range_1,
                              y_range_2):
    if validate_text(variable_list, observation_list, observation_filenames, plot_filename, legend_list, x_label,
                     x_type, y_label, y_type):
        return main_model.plot_observation_settings(variable_list, observation_list, observation_filenames,
                                                    plot_filename, legend_list, legend_font_size, x_label, x_type,
                                                    x_range_1, x_range_2, y_label, y_type, y_range_1, y_range_2)
    else:
        return None


def apply_characteristic_curve_setting(name, sat_func_type, sf_alpha, sf_m, sf_lambda, sf_liq_residual_sat,
                                       sf_gas_residual_sat, max_cap_pressure, smooth, power, default,
                                       liq_perm_func_type, lpf_m, lpf_lambda, lpf_liq_residual_sat, gas_perm_func_type,
                                       gpf_m, gpf_lambda, gpf_liq_residual_saturation, gpf_gas_residual_saturation):
    if validate_text(name):
        return main_model.apply_characteristic_curve_setting(name, sat_func_type, sf_alpha, sf_m, sf_lambda,
                                                             sf_liq_residual_sat, sf_gas_residual_sat, max_cap_pressure,
                                                             smooth, power, default, liq_perm_func_type, lpf_m,
                                                             lpf_lambda, lpf_liq_residual_sat, gas_perm_func_type,
                                                             gpf_m, gpf_lambda, gpf_liq_residual_saturation,
                                                             gpf_gas_residual_saturation)
    else:
        return None


def id_search(object_id):
    """
    Searches for an instance of a class using the object id generated by id()
    :param object_id: str
    :return: Attribute object or None
    """
    ret_object = object_space.ALL_OBJECTS.get(object_id, None)
    return ret_object


def delete_object(key_id):
    """
    Deletes the object from its dictionary and ALL_OBJECTS dictionary
    :param key_id: str - dictionary key
    :return:
    """

    del object_space.ALL_OBJECTS[key_id]
    main_view.log_append_message('Object with ID:' + key_id + ' deleted.')


def remove_all_attributes():
    object_space.ALL_OBJECTS.clear()
    # TODO: delete objects from each dictionary
