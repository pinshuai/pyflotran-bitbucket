from PySide import QtGui, QtCore
from gen import simulation_ui, material_ui, grid_ui, time_ui, output_ui, timestepper_ui, linear_solver_ui
from gen import newton_solver_ui, saturation_function_ui, flow_ui, flow_variable_adder_ui, initial_condition_ui
from gen import boundary_condition_ui, source_sink_ui, stratigraphy_coupler_ui, checkpoint_ui, restart_ui
from gen import uniform_velocity, fluid_properties_ui, observation_ui, chemistry_ui, transport_condition
from gen import constraint_condition_ui, constraint_condition_concentration_ui, constraint_condition_mineral_ui
from gen import regression_ui, dataset_ui, region_ui, plot_observation_ui, generic_warning_ui, point_ui
from gen import characteristic_curves_ui

from model import object_space

"""
This file contains all the component views that complement the main view (e.g., Grid settings window).
:param controller: This is the reference to controller which is the middleman between the views and PyPFLOTRAN.

The views inherit from their respect QtGUI components (QDialog, QWidget, etc.).
"""

field_missing = 'Some fields are missing.'


class GenericWarningView(QtGui.QDialog):
    def __init__(self, warning_message=''):
        super(GenericWarningView, self).__init__()
        self.ui = generic_warning_ui.Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.label.setText(warning_message)


class SimulationView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(SimulationView, self).__init__()
        self.ui = simulation_ui.Ui_Dialog()
        self.build_ui()
        self.warning = GenericWarningView("A checkbox must be ticked.")

    def build_ui(self):
        self.ui.setupUi(self)

        self.ui.mode_comboBox.hide()
        self.ui.mode_label.hide()

        self.ui.subsurface_flow_checkBox.stateChanged.connect(self.__toggle_flow_combobox)

    def accept(self):
        self.apply_sim_settings(self.ui.subsurface_flow_checkBox, self.ui.subsurface_trans_checkBox,
                                self.ui.geomechanics_checkBox, self.ui.mode_comboBox.currentText())

    def reject(self):
        self.hide()

    @QtCore.Slot()
    def __toggle_flow_combobox(self, state):
        if state == 2:
            self.ui.mode_comboBox.show()
            self.ui.mode_label.show()
        else:
            self.ui.mode_label.hide()
            self.ui.mode_comboBox.hide()

    def apply_sim_settings(self, s_flow_checkbox, s_trans_checkbox, geo_checkbox, mode):
        """
        Calls the main controller to apply simulation settings.

        :rtype : null
        :type s_flow_checkbox: QtGui.QCheckBox
        :type s_trans_checkbox: QtGui.QCheckBox
        :type geo_checkbox: QtGui.QCheckBox
        """
        s_flow = False
        s_trans = False
        s_geo = False
        if s_flow_checkbox.isChecked():
            s_flow = True
        if s_trans_checkbox.isChecked():
            s_trans = True
        if geo_checkbox.isChecked():
            s_geo = True
        # Pass each checkbox value onto the controller.
        sim_inst = self.controller.apply_simulation_settings(s_flow, s_trans, s_geo, mode)
        if sim_inst is not None:
            item = QtGui.QListWidgetItem()
            item.setText(sim_inst)
            self.controller.main_view.main_ui.material_list.addItem(item)
            self.controller.main_view.log_append_message('Simulation object, ID: ' + sim_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def auto_fill(self, id_text):
        sim_inst = object_space.SIMULATION_OBJECT[0]
        if sim_inst.subsurface_flow != '':
            self.ui.subsurface_flow_checkBox.setChecked(True)
        if sim_inst.subsurface_transport != '':
            self.ui.subsurface_trans_checkBox.setChecked(True)

        index = self.ui.mode_comboBox.findText(sim_inst.mode)
        if index >= 0:
            self.ui.mode_comboBox.setCurrentIndex(index)


class MaterialView(QtGui.QWidget):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(MaterialView, self).__init__()
        self.ui = material_ui.Ui_Form()
        self.build_ui()
        self.warning = GenericWarningView(field_missing)
        # setAttribute forces the widget to close when QMainWindow does
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def build_ui(self):
        self.ui.setupUi(self)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(self.hide)
        # Consume the data from the text fields and pass it to the controller
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).clicked.connect(lambda: self.apply_material_properties(
            self.ui.id_spinBox.value(),
            self.ui.name_lineEdit.text(),
            self.ui.char_curves_lineEdit.text(),
            self.ui.porosity_dSpinBox.value(),
            self.ui.tortuosity_dSpinBox.value(),
            self.ui.rock_density_dSpinBox.value(),
            self.ui.specific_heat_dSpinBox.value(),
            self.ui.conductivity_dry_dSpinBox.value(),
            self.ui.conductivity_wet_dSpinBox.value(),
            self.ui.saturation_lineEdit.text(),
            self.ui.permeability_x_dSpinBox.value(),
            self.ui.permeability_y_dSpinBox.value(),
            self.ui.permeability_z_dSpinBox.value(),
        ))

    @QtCore.Slot()
    def apply_material_properties(self, material_id, name, char_curve, porosity, tortuosity, density, specific_heat,
                                  condition_dry, condition_wet, saturation, perm_x, perm_y, perm_z):
        material_inst = self.controller.apply_material_properties(material_id, name, char_curve, porosity, tortuosity,
                                                                  density, specific_heat, condition_dry, condition_wet,
                                                                  saturation, perm_x, perm_y, perm_z)
        if material_inst is not None:
            self.controller.main_view.log_append_message('Material Properties object, ID: ' + material_inst
                                                         + ', created.')
            self.hide()
        else:
            self.warning.exec_()


class GridView(QtGui.QWidget):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        self.structured = False
        self.unstructured = False
        self._grid_type = 'structured'
        super(GridView, self).__init__()
        self.ui = grid_ui.Ui_Form()
        self.build_ui()
        self.warning = GenericWarningView(field_missing)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def build_ui(self):
        self.ui.setupUi(self)
        # set default view to structured
        self.ui.structured_button.click()

        # Browse button functionality
        self.ui.browse_button.clicked.connect(lambda: self._browse_button_function(self.ui.filename_lineEdit))
        QtCore.QObject.connect(self.ui.structured_button, QtCore.SIGNAL("toggled(bool)"), self._grid_structured)
        QtCore.QObject.connect(self.ui.unstructured_button, QtCore.SIGNAL("toggled(bool)"), self._grid_unstructured)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(self.hide)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).clicked.connect(lambda: self.apply_grid_settings(
            self.ui.lower_bounds_x_dSpinBox.value(),
            self.ui.lower_bounds_y_dSpinBox.value(),
            self.ui.lower_bounds_z_dSpinBox.value(),
            self.ui.upper_bounds_x_dSpinBox.value(),
            self.ui.upper_bounds_y_dSpinBox.value(),
            self.ui.upper_bounds_z_dSpinBox.value(),
            self.ui.origin_x_dSpinBox.value(),
            self.ui.origin_y_dSpinBox.value(),
            self.ui.origin_z_dSpinBox.value(),
            self.ui.grid_spacing_dx_dSpinBox.value(),
            self.ui.grid_spacing_dy_dSpinBox.value(),
            self.ui.grid_spacing_dz_dSpinBox.value(),
            self.ui.gravity_vector_x_dSpinBox.value(),
            self.ui.gravity_vector_y_dSpinBox.value(),
            self.ui.gravity_vector_z_dSpinBox.value(),
            self.ui.grid_cells_x_dSpinBox.value(),
            self.ui.grid_cells_y_dSpinBox.value(),
            self.ui.grid_cells_z_dSpinBox.value(),
            self.ui.filename_lineEdit.text()
        ))

    def _grid_structured(self):
        self.structured = True
        self.unstructured = False
        self._grid_type = 'structured'

    def _grid_unstructured(self):
        self.unstructured = True
        self.structured = False
        self._grid_type = 'unstructured'

    @QtCore.Slot()
    def _browse_button_function(self, line_edit):
        grid_settings_file, _ = QtGui.QFileDialog.getOpenFileName(self, "Select grid setting file")
        if len(grid_settings_file) > 0:
            line_edit.setText(grid_settings_file)

    @QtCore.Slot()
    def apply_grid_settings(self, lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, origin_x, origin_y, origin_z,
                            g_spacing_x, g_spacing_y, g_spacing_z, gravity_x, gravity_y, gravity_z,
                            g_cell_x, g_cell_y, g_cell_z, file_path):
        grid_inst = self.controller.apply_grid_settings(self._grid_type, lb_x, lb_y, lb_z, ub_x, ub_y, ub_z, origin_x,
                                                        origin_y, origin_z, g_spacing_x, g_spacing_y, g_spacing_z,
                                                        gravity_x, gravity_y, gravity_z, g_cell_x, g_cell_y, g_cell_z,
                                                        file_path)
        if grid_inst is not None:
            self.controller.main_view.log_append_message('Grid object, ID: ' + grid_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()


class TimeView(QtGui.QWidget):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(TimeView, self).__init__()
        self.ui = time_ui.Ui_Form()
        self._build_ui()
        self.warning_select = GenericWarningView("A time value must be selected first.")
        self.warning_missing = GenericWarningView(field_missing)
        self.warning_equal = GenericWarningView('Initial and final time are equal.')
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.false_radioButton.click()

        self.ui.add_delta_time_row_button.clicked.connect(self._add_delta_button)
        self.ui.remove_time_value_unit_push_button.clicked.connect(self._delete_delta_button)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).clicked.connect(self._apply_time_settings)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(self.close)

    @QtCore.Slot()
    def _add_delta_button(self):
        dti = self.ui.delta_t_given_inst_time_value_dSpinBox.text()
        dti_unit = self.ui.delta_t_given_inst_unit_comboBox.currentText()
        dtf = self.ui.second_dt_value_dSpinBox.text()
        dtf_unit = self.ui.second_dt_unit_comboBox.currentText()
        if (dti == dtf) and (dti_unit == dtf_unit):
            self.warning_equal.exec_()
        else:
            self.ui.listWidget.addItem(dti + ' ' + dti_unit + ', ' + dtf + ' ' + dtf_unit)

    @QtCore.Slot()
    def _delete_delta_button(self):
        selected_items = self.ui.listWidget.selectedIndexes()
        if len(selected_items) > 0:
            for item in selected_items:
                self.ui.listWidget.takeItem(item.row())
        else:
            self.warning_select.exec_()

    @QtCore.Slot()
    def _apply_time_settings(self):
        run_as_steady_state = 'False'
        if self.ui.true_radioButton.isChecked():
            run_as_steady_state = 'True'

        time = self.controller.apply_time_settings(self.ui.final_time_dSpinBox.text(),
                                                   self.ui.final_time_unit_comboBox.currentText(),
                                                   self.ui.delta_time_init_dSpinBox.text(),
                                                   self.ui.delta_time_init_unit_comboBox.currentText(),
                                                   self.ui.delta_time_final_dSpinBox.text(),
                                                   self.ui.delta_time_final_unit_comboBox.currentText(),
                                                   [self.ui.listWidget.item(i).text() for i in
                                                    xrange(self.ui.listWidget.count())],
                                                   run_as_steady_state)
        if time is not None:
            self.controller.main_view.log_append_message('Time object, ID: ' + time + ', created.')
            self.hide()
        else:
            self.warning_missing.exec_()


class TimestepperView(QtGui.QWidget):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(TimestepperView, self).__init__()
        self.ui = timestepper_ui.Ui_Form()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def _build_ui(self):
        self.ui.setupUi(self)

        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(self.hide)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).clicked.connect(self._apply_timestepper_settings)

    @QtCore.Slot()
    def _apply_timestepper_settings(self):
        ts_inst = self.controller.apply_timestepper_settings(self.ui.ts_mode_comboBox.currentText(),
                                                             self.ui.ts_acceleration_spinBox.value(),
                                                             self.ui.num_steps_after_cut_spinBox.value(),
                                                             self.ui.max_steps_spinBox.value(),
                                                             self.ui.max_ts_cuts_spinBox.value(),
                                                             self.ui.cfl_limiter_dSpinBox.value(),
                                                             self.ui.init_steady_state_comboBox.currentText(),
                                                             self.ui.run_as_steady_state_comboBox.currentText(),
                                                             self.ui.max_pressure_change_dSpinBox.value(),
                                                             self.ui.max_temp_change_dSpinBox.value(),
                                                             self.ui.max_concentration_change_dSpinBox.value(),
                                                             self.ui.max_saturation_change_dSpinBox.value())
        if ts_inst is not None:
            self.controller.main_view.log_append_message('Timestepper object, ID: ' + ts_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()


class LinearSolverView(QtGui.QWidget):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(LinearSolverView, self).__init__()
        self.ui = linear_solver_ui.Ui_Form()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.buttonBox.button(self.ui.buttonBox.Ok).clicked.connect(self._apply_linear_solver_settings)
        self.ui.buttonBox.button(self.ui.buttonBox.Cancel).clicked.connect(self.hide)

    @QtCore.Slot()
    def _apply_linear_solver_settings(self):
        ls_obj = self.controller.apply_linear_solver_settings(self.ui.name_comboBox.currentText(),
                                                              self.ui.solver_comboBox.currentText(),
                                                              self.ui.preconditioner_comboBox.currentText())
        if ls_obj is not None:
            self.controller.main_view.log_append_message('Linear Solver object, ID: ' + ls_obj + ', created.')
            self.hide()
        else:
            self.warning.exec_()


class NewtonSolverView(QtGui.QWidget):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(NewtonSolverView, self).__init__()
        self.ui = newton_solver_ui.Ui_Form()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(self.hide)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).clicked.connect(self._apply_newton_solver_settings)

    @QtCore.Slot()
    def _apply_newton_solver_settings(self):
        newton_s = self.controller.apply_newton_solver_settings(self.ui.name_comboBox.currentText(),
                                                                self.ui.abs_tolerance_lineEdit.text(),
                                                                self.ui.rel_tolerance_lineEdit.text(),
                                                                self.ui.rel_tolerance_update_lineEdit.text(),
                                                                self.ui.divergence_tol_lineEdit.text(),
                                                                self.ui.tol_compared_to_inf_norm_lineEdit.text(),
                                                                self.ui.max_iterations_dSpinBox.value(),
                                                                self.ui.max_function_eval_dSpinBox.value())
        if newton_s is not None:
            self.controller.main_view.log_append_message('Newton Solver object, ID: ' + newton_s + ', created.')
            self.hide()
        else:
            self.warning.exec_()


class OutputView(QtGui.QWidget):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(OutputView, self).__init__()
        self.ui = output_ui.Ui_Form()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(self.hide)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).clicked.connect(self._apply_output_settings)

    @QtCore.Slot()
    def _apply_output_settings(self):
        output_inst = self.controller.apply_output_settings(self.ui.time_values_list_lineEdit.text(),
                                                            self.ui.time_value_list_unit_comboBox.currentText(),
                                                            self.ui.print_column_ids_comboBox.currentText(),
                                                            self.ui.screen_output_comboBox.currentText(),
                                                            self.ui.screen_periodic_spinBox.value(),
                                                            self.ui.periodic_time_dSpinBox.value(),
                                                            self.ui.periodic_time_comboBox.currentText(),
                                                            self.ui.periodic_timestep_dSpinBox.value(),
                                                            self.ui.periodic_timestep_comboBox.currentText(),
                                                            self.ui.periodic_observation_time_dSpinBox.value(),
                                                            self.ui.periodic_observation_time_comboBox.currentText(),
                                                            self.ui.periodic_obs_timestep_spinBox.value(),
                                                            self.ui.tecplot_block_checkBox.isChecked(),
                                                            self.ui.tecplot_point_checkBox.isChecked(),
                                                            self.ui.hdf5_checkBox.isChecked(),
                                                            self.ui.hdf5_multiple_files_checkBox.isChecked(),
                                                            self.ui.vtk_checkBox.isChecked(),
                                                            self.ui.permeability_combBox.currentText(),
                                                            self.ui.porosity_comboBox.currentText(),
                                                            self.ui.velocities_comboBox.currentText(),
                                                            self.ui.mass_balance_comboBox.currentText(),
                                                            self.ui.variable_list_lineEdit.text())
        if output_inst is not None:
            self.controller.main_view.log_append_message('Output object, ID: ' + output_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()


class SaturationFunctionView(QtGui.QWidget):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(SaturationFunctionView, self).__init__()
        self.ui = saturation_function_ui.Ui_Form()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(self.hide)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).clicked.connect(self._apply_saturation_func_settings)

    @QtCore.Slot()
    def _apply_saturation_func_settings(self):
        sat_func = \
            self.controller.apply_saturation_func_settings(self.ui.name_lineEdit.text(),
                                                           self.ui.permeability_function_type_comboBox.currentText(),
                                                           self.ui.saturation_function_type_comboBox.currentText(),
                                                           self.ui.residual_saturation_comboBox.currentText(),
                                                           self.ui.residual_saturation_dSpinBox.value(),
                                                           self.ui.residual_saturation_liq_comboBox.currentText(),
                                                           self.ui.residual_saturation_liq_dSpinBox.value(),
                                                           self.ui.residual_saturation_gas_comboBox.currentText(),
                                                           self.ui.residual_saturation_gas_dSpinBox.value(),
                                                           self.ui.lambda_dSpinBox.value(),
                                                           self.ui.alpha_lineEdit.text(),
                                                           self.ui.max_cap_pressure_dSpinBox.value(),
                                                           self.ui.beta_dSpinBox.value(),
                                                           self.ui.power_dSpinBox.value())
        if sat_func is not None:
            self.controller.main_view.log_append_message('Saturation Function object, ID: ' + sat_func + ', created.')
            self.hide()
        else:
            self.warning.exec_()


class FlowView(QtGui.QWidget):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(FlowView, self).__init__()
        self.ui = flow_ui.Ui_Form()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)
        # flow_variable_id_list is a list of object IDs of flow variables.
        self.flow_variable_list = []

    def _build_ui(self):
        self.ui.setupUi(self)

        # disable unit list label/lineEdit
        self.ui.unit_list_lineEdit.setDisabled(True)
        self.ui.unit_list_label.setDisabled(True)

        self.ui.browse_button.clicked.connect(self._browse_button_function)
        self.ui.list_radioButton.clicked.connect(self._change_datum_label)
        self.ui.file_radioButton.clicked.connect(self._change_datum_label_to_original)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(self.hide)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).clicked.connect(self._apply_flow_settings)
        self.ui.add_pushButton.clicked.connect(self._open_flow_variable_adder_view)
        self.ui.remove_pushButton.clicked.connect(self._remove_button_function)

        self.ui.list_radioButton.click()

    def closeEvent(self, *args, **kwargs):
        if self.flow_var_adder is not None:
            self.flow_var_adder.close()
        self.closeEvent()

    def _init_list_widget(self, list_of_flow_variables):
        for each_list in list_of_flow_variables:
            self.ui.variable_list_listWidget.addItem(each_list)

    @QtCore.Slot()
    def _open_flow_variable_adder_view(self):
        self.flow_var_adder = FlowVariableAdderView(self.model, self.controller, self.ui.variable_list_listWidget,
                                                    self.flow_variable_list)
        self.flow_var_adder.show()

    @QtCore.Slot()
    def _remove_button_function(self):
        selected_rows = self.ui.variable_list_listWidget.selectedIndexes()
        if len(selected_rows) > 0:
            for row in selected_rows:
                self.controller.main_view.log_append_message(
                    'Flow variable object, ID: ' + self.flow_variable_list[row.row()] + ', destroyed.')
                del self.flow_variable_list[row.row()]
                self.ui.variable_list_listWidget.takeItem(row.row())

    @QtCore.Slot()
    def _change_datum_label(self):
        self.ui.datum_label.setText('Datum: dx, dy, dz')

    @QtCore.Slot()
    def _change_datum_label_to_original(self):
        self.ui.datum_label.setText('Datum')

    @QtCore.Slot()
    def _apply_flow_settings(self):
        data_unit_type = ''
        if self.ui.list_radioButton.isChecked():
            data_unit_type = 'list'
        elif self.ui.file_radioButton.isChecked():
            data_unit_type = 'filename'
        flow_inst = self.controller.apply_flow_settings(self.ui.name_lineEdit.text(),
                                                        self.ui.unit_list_lineEdit.text(),
                                                        self.ui.iphase_spinBox.value(),
                                                        self.ui.sync_ts_with_update_comboBox.currentText(),
                                                        data_unit_type,
                                                        self.ui.datum_dx_dSpinBox.value(),
                                                        self.ui.datum_dy_dSpinBox.value(),
                                                        self.ui.datum_dz_dSpinBox.value(),
                                                        self.ui.datum_lineEdit.text(),
                                                        self.flow_variable_list)
        if flow_inst is not None:
            self.controller.main_view.log_append_message('Flow object, ID: ' + flow_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    @QtCore.Slot()
    def _browse_button_function(self):
        file_location = QtGui.QFileDialog.getOpenFileName(self)
        self.ui.datum_lineEdit.setText(file_location[0])


class FlowVariableAdderView(QtGui.QWidget):
    def __init__(self, model, controller, flow_list_widget, flow_variable_list):
        self.controller = controller
        self.model = model
        self.flow_listWidget = flow_list_widget
        self.flow_variable_id_list = flow_variable_list
        super(FlowVariableAdderView, self).__init__()
        self.ui = flow_variable_adder_ui.Ui_Form()
        self._build_ui()
        self.warning_row = GenericWarningView("Select a row first.")
        self.warning = GenericWarningView(field_missing)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def __unicode__(self):
        return 'variable= ' + self.ui.name_comboBox.currentText()

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.name_comboBox.currentIndexChanged.connect(self._update_type_combo_box)
        self.ui.push_down_tool_button.clicked.connect(self._push_down_tool_button_function)
        self.ui.remove_push_button.clicked.connect(self._remove_push_button_function)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(self.hide)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).clicked.connect(self._apply_flow_variable_adder_settings)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Reset).clicked.connect(self._reset_view)

    @QtCore.Slot()
    def _update_type_combo_box(self):
        list_of_flow_types = self.controller.get_flow_variable_types(self.ui.name_comboBox.currentIndex())
        self.ui.type_comboBox.clear()
        for flow_item_type in list_of_flow_types:
            self.ui.type_comboBox.addItem(flow_item_type)

    @QtCore.Slot()
    def _push_down_tool_button_function(self):
        new_row = self.ui.time_unit_value_dSpinBox.text() + ' | ' + self.ui.data_unit_value_list_lineEdit.text()
        self.ui.listWidget.addItem(new_row)

    @QtCore.Slot()
    def _remove_push_button_function(self):
        selected_items = self.ui.listWidget.selectedIndexes()
        if len(selected_items) > 0:
            for item in selected_items:
                self.ui.listWidget.takeItem(item.row())
        else:
            self.warning_row.exec_()

    @QtCore.Slot()
    def _reset_view(self):
        line_edits = self.findChildren(QtGui.QLineEdit)
        for line_edit in line_edits:
            line_edit.setText('')
        self.ui.flow_variable_list_objects_textEdit.clear()
        self.ui.listWidget.clear()
        self.ui.name_comboBox.setCurrentIndex(0)

    @QtCore.Slot()
    def _apply_flow_variable_adder_settings(self):
        flow_var = self.controller.apply_flow_variable_adder_settings(self.ui.name_comboBox.currentText(),
                                                                      self.ui.type_comboBox.currentText(),
                                                                      self.ui.value_list_1_dSpinBox.value(),
                                                                      self.ui.value_list_2_dSpinBox.value(),
                                                                      self.ui.unit_lineEdit.text(),
                                                                      self.ui.time_unit_type_lineEdit.text(),
                                                                      self.ui.data_unit_type_lineEdit.text(),
                                                                      self.ui.flow_variable_list_objects_textEdit.toPlainText(),
                                                                      [self.ui.listWidget.item(i).text() for i in
                                                                       xrange(self.ui.listWidget.count())])
        if flow_var is not None:
            self.flow_listWidget.addItem(self.__unicode__())
            self.flow_variable_id_list.append(flow_var)
            self.controller.main_view.log_append_message(
                'Flow variable object, ID: ' + flow_var + ', created.')
            self.close()
        else:
            self.warning.exec_()


class InitialConditionView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(InitialConditionView, self).__init__()
        self.ui = initial_condition_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)

    def accept(self):
        init_cond_inst = self.controller.apply_initial_condition(self.ui.name_lineEdit.text(),
                                                                 self.ui.flow_lineEdit.text(),
                                                                 self.ui.transport_lineEdit.text(),
                                                                 self.ui.region_lineEdit.text())
        if init_cond_inst is not None:
            self.controller.main_view.log_append_message(
                'Initial condition object, ID: ' + init_cond_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class BoundaryConditionView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(BoundaryConditionView, self).__init__()
        self.ui = boundary_condition_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)

    def accept(self):
        bound_cond = self.controller.apply_boundary_condition(self.ui.nameLineEdit.text(),
                                                              self.ui.flowLineEdit.text(),
                                                              self.ui.transportLineEdit.text(),
                                                              self.ui.regionLineEdit.text())
        if bound_cond is not None:
            self.controller.main_view.log_append_message('Boundary Condition object, ID: ' + bound_cond + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class SourceSinkView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(SourceSinkView, self).__init__()
        self.ui = source_sink_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)

    def accept(self):
        ss_inst = self.controller.apply_source_sink_condition(self.ui.name_lineEdit.text(),
                                                              self.ui.flow_lineEdit.text(),
                                                              self.ui.transport_lineEdit.text(),
                                                              self.ui.region_lineEdit.text())
        if ss_inst is not None:
            self.controller.main_view.log_append_message('Source Sink object, ID: ' + ss_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class StratigraphyCouplerView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(StratigraphyCouplerView, self).__init__()
        self.ui = stratigraphy_coupler_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)

    def accept(self):
        strata = self.controller.apply_stratigraphy_settings(self.ui.region_lineEdit.text(),
                                                             self.ui.material_lineEdit.text())
        if strata is not None:
            self.controller.main_view.log_append_message(
                'Stratigraphy coupler object, ID: ' + strata + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class CheckpointView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(CheckpointView, self).__init__()
        self.ui = checkpoint_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)

    def accept(self):
        checkpoint_inst = self.controller.apply_checkpoint_settings(self.ui.frequency_spinBox.value(),
                                                                    self.ui.overwriteComboBox.currentText())
        if checkpoint_inst is not None:
            self.controller.main_view.log_append_message('Checkpoint object, ID: ' + checkpoint_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class RestartView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(RestartView, self).__init__()
        self.ui = restart_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.browse_tool_button.clicked.connect(self._browse_file_location_button_function)

    @QtCore.Slot()
    def _browse_file_location_button_function(self):
        file_location = QtGui.QFileDialog.getSaveFileName(self, self.tr('Save file'), 'restart', self.tr('Chk (*.chk)'))
        self.ui.filename_line_edit.setText(file_location[0])

    def accept(self):
        restart_inst = self.controller.apply_restart_settings(self.ui.filename_line_edit.text(),
                                                              self.ui.time_value_dSpinBox.value(),
                                                              self.ui.time_value_unit_combo_box.currentText())
        if restart_inst is not None:
            self.controller.main_view.log_append_message('Restart object, ID: ' + restart_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class UniformVelocityView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(UniformVelocityView, self).__init__()
        self.ui = uniform_velocity.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)

    def accept(self):
        uniform_vel_inst = self.controller.apply_uniform_velocity_settings(self.ui.vlx_dSpinBox.value(),
                                                                           self.ui.vly_dSpinBox.value(),
                                                                           self.ui.vlz_dSpinBox.value(),
                                                                           self.ui.unit_lineEdit.text())
        if uniform_vel_inst is not None:
            self.controller.main_view.log_append_message(
                'Uniform Velocity object, ID: ' + uniform_vel_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class FluidPropertiesView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(FluidPropertiesView, self).__init__()
        self.ui = fluid_properties_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)

    def accept(self):
        fluid_property = self.controller.apply_fluid_properties(self.ui.diff_coeff_lineEdit.text())
        if fluid_property is not None:
            self.controller.main_view.log_append_message('Fluid Property object, ID: ' + fluid_property + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class ObservationView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(ObservationView, self).__init__()
        self.ui = observation_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)

    def accept(self):
        obs_inst = self.controller.apply_observation_settings(self.ui.region_line_edit.text())
        if obs_inst is not None:
            self.controller.main_view.log_append_message(
                'Observation object, ID: ' + obs_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class ChemistryView(QtGui.QWidget):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(ChemistryView, self).__init__()
        self.ui = chemistry_ui.Ui_Form()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(self.hide)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).clicked.connect(self._apply_chemistry_settings)

        self.ui.m_kinetic_add_pushButton.clicked.connect(self._add_m_kinetic)
        self.ui.m_kinetic_delete_pushButton.clicked.connect(self._del_m_kinetic)

    @QtCore.Slot()
    def _add_m_kinetic(self):
        name = self.ui.mineral_kinetic_name_lineEdit.text()
        rate = self.ui.rate_const_list_dSpinBox.text()
        unit = self.ui.rate_const_list_unit_lineEdit.text()
        if (name == '') or (unit == ''):
            self.warning.exec_()
        else:
            self.ui.m_kinetic_listWidget.addItem(name + ',' + rate + ',' + unit)

    @QtCore.Slot()
    def _del_m_kinetic(self):
        selected_rows = self.ui.m_kinetic_listWidget.selectedIndexes()
        for item in selected_rows:
            self.ui.m_kinetic_listWidget.takeItem(item.row())

    @QtCore.Slot()
    def _apply_chemistry_settings(self):
        chem_inst = self.controller.apply_chemistry_settings(
            self.ui.species_list_textEdit.toPlainText(),
            self.ui.aq_species_in_eq_with_primary_species_textEdit.toPlainText(),
            self.ui.gas_species_textEdit.toPlainText(),
            self.ui.mineral_list_textEdit.toPlainText(),
            self.ui.log_formulation_comboBox.currentText(),
            self.ui.database_lineEdit.text(),
            self.ui.activity_coefficients_comboBox.currentText(),
            self.ui.molal_comboBox.currentText(),
            self.ui.output_list_lineEdit.text(),
            [self.ui.m_kinetic_listWidget.item(i).text() for i in xrange(self.ui.m_kinetic_listWidget.count())])

        if chem_inst is not None:
            self.controller.main_view.log_append_message('Chemistry object, ID: ' + chem_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()


class TransportConditionView(QtGui.QWidget):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(TransportConditionView, self).__init__()
        self.ui = transport_condition.Ui_Form()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)
        self.warning_row = GenericWarningView("A row must be selected.")
        self.warning_type = GenericWarningView('Type cannot be None.')
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.push_down_tool_button.clicked.connect(self._push_down_tool_button_function)
        self.ui.remove_push_button.clicked.connect(self._remove_selected_row)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(self.hide)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).clicked.connect(self._apply_transport_conditions)

    @QtCore.Slot()
    def _push_down_tool_button_function(self):
        if self.ui.type_comboBox.currentText() == 'None':
            self.warning_type.exec_()
        else:
            row = self.ui.constraint_list_value_dSpinBox.text() + ", " + self.ui.constraint_list_type_lineEdit.text()
            if self.ui.constraint_list_type_lineEdit.text() == '':
                self.warning.exec_()
            else:
                self.ui.listWidget.addItem(row)

    @QtCore.Slot()
    def _remove_selected_row(self):
        selected_rows = self.ui.listWidget.selectedIndexes()
        if len(selected_rows) > 0:
            for item in selected_rows:
                self.ui.listWidget.takeItem(item.row())
        else:
            self.warning_row.exec_()

    @QtCore.Slot()
    def _apply_transport_conditions(self):
        tran_inst = self.controller.apply_transport_condition(self.ui.name_lineEdit.text(),
                                                              self.ui.type_comboBox.currentText(),
                                                              [self.ui.listWidget.item(i).text() for i in
                                                               xrange(self.ui.listWidget.count())])
        if tran_inst is not None:
            self.controller.main_view.log_append_message(
                'Transport object, ID: ' + tran_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()


class ConstraintConditionView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(ConstraintConditionView, self).__init__()
        self.ui = constraint_condition_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.add_concentration_pushButton.clicked.connect(self._open_concentration_view)
        self.ui.add_mineral_pushButton.clicked.connect(self._open_mineral_view)
        self.ui.remove_concentration_pushButton.clicked.connect(self._remove_concentration_item)
        self.ui.remove_mineral_pushButton.clicked.connect(self.remove_mineral_item)

    def accept(self):
        constr_cond_int = self.controller.apply_constraint_condition(self.ui.name_lineEdit.text(),
                                                                     [self.ui.concentration_listWidget.item(i).text()
                                                                      for i in
                                                                      xrange(self.ui.concentration_listWidget.count())],
                                                                     [self.ui.mineral_listWidget.item(i).text() for i in
                                                                      xrange(self.ui.mineral_listWidget.count())])
        if constr_cond_int is not None:
            self.controller.main_view.log_append_message(
                'Constraint Condition object, ID: ' + constr_cond_int + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()

    @QtCore.Slot()
    def _open_concentration_view(self):
        self.concentration_view = ConstraintConditionConcentrationView(self.model, self.controller, self)
        self.concentration_view.show()

    @QtCore.Slot()
    def _open_mineral_view(self):
        self.mineral_view = ConstraintConditionMineralView(self.model, self.controller, self)
        self.mineral_view.show()

    @QtCore.Slot()
    def _remove_concentration_item(self):
        selected_rows = self.ui.concentration_listWidget.selectedIndexes()
        if len(selected_rows) > 0:
            for row in selected_rows:
                self.ui.concentration_listWidget.takeItem(row.row())

    @QtCore.Slot()
    def remove_mineral_item(self):
        selected_rows = self.ui.mineral_listWidget.selectedIndexes()
        if len(selected_rows) > 0:
            for row in selected_rows:
                self.ui.mineral_listWidget.takeItem(row.row())


class ConstraintConditionMineralView(QtGui.QDialog):
    def __init__(self, model, controller, main_constraint_view):
        self.model = model
        self.controller = controller
        self.main_constraint_view = main_constraint_view
        super(ConstraintConditionMineralView, self).__init__()
        self.ui = constraint_condition_mineral_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)
        self.id = ''

    def __unicode__(self):
        return 'name: ' + self.ui.name_lineEdit.text() + ', id: ' + self.id

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Reset).clicked.connect(self._reset_fields)

    def accept(self):
        cc_mineral_inst = self.controller.apply_constraint_condition_mineral(self.ui.name_lineEdit.text(),
                                                                             self.ui.volume_fraction_dSpinBox.value(),
                                                                             self.ui.surface_area_dSpinBox.value())
        if cc_mineral_inst is not None:
            self.id = cc_mineral_inst
            self.main_constraint_view.ui.mineral_listWidget.addItem(self.__unicode__())
            self.controller.main_view.log_append_message(
                'Constraint Condition: Mineral object, ID: ' + cc_mineral_inst + ', created.')
            self.close()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()

    def _reset_fields(self):
        line_edits = self.findChildren(QtGui.QLineEdit)
        for line_edit in line_edits:
            line_edit.setText('')


class ConstraintConditionConcentrationView(QtGui.QDialog):
    def __init__(self, model, controller, main_constraint_view):
        self.model = model
        self.controller = controller
        self.main_constraint_view = main_constraint_view
        super(ConstraintConditionConcentrationView, self).__init__()
        self.ui = constraint_condition_concentration_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)
        self.id = ''

    def __unicode__(self):
        return 'species: ' + self.ui.species_lineEdit.text() + ', id: ' + self.id

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Reset).clicked.connect(self._reset_fields)

    def accept(self):
        cc_concentration = self.controller.apply_constraint_condition_concentration(
            self.ui.species_lineEdit.text(),
            self.ui.value_dSpinBox.value(),
            self.ui.constraint_comboBox.currentText(),
            self.ui.element_lineEdit.text())
        if cc_concentration is not None:
            self.id = cc_concentration
            self.main_constraint_view.ui.concentration_listWidget.addItem(self.__unicode__())
            self.controller.main_view.log_append_message(
                'Constraint Condition: Concentration object, ID: ' + cc_concentration + ', created.')
            self.close()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()

    @QtCore.Slot()
    def _reset_fields(self):
        line_edits = self.findChildren(QtGui.QLineEdit)
        if len(line_edits) > 0:
            for line_edit in line_edits:
                line_edit.setText('')
        self.ui.constraint_comboBox.setCurrentIndex(0)


class RegressionView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(RegressionView, self).__init__()
        self.ui = regression_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)

    def accept(self):
        reg_inst = self.controller.apply_regression_settings(self.ui.cells_line_edit.text(),
                                                             self.ui.cells_per_process_d_spin_box.value())
        if reg_inst is not None:
            self.controller.main_view.log_append_message('Regression object, ID: ' + reg_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class DatasetView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(DatasetView, self).__init__()
        self.ui = dataset_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.browse_button.clicked.connect(self._browse_button_function)

    @QtCore.Slot()
    def _browse_button_function(self):
        file_location = QtGui.QFileDialog.getOpenFileName(self)
        self.ui.file_name_line_edit.setText(file_location[0])

    def accept(self):
        ds_inst = self.controller.apply_dataset_settings(self.ui.dataset_name_line_edit.text(),
                                                         self.ui.dataset_mapped_name.text(),
                                                         self.ui.name_line_edit.text(),
                                                         self.ui.file_name_line_edit.text(),
                                                         self.ui.hdf5_dataset_name.text(),
                                                         self.ui.map_hdf5_dataset_name.text(),
                                                         self.ui.max_buffer_size_spin_box.value())
        if ds_inst is not None:
            self.controller.main_view.log_append_message('Dataset object, ID: ' + ds_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class RegionView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(RegionView, self).__init__()
        self.ui = region_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)

    def accept(self):
        region_inst = self.controller.apply_region_settings(self.ui.name_line_edit.text(),
                                                            self.ui.lower_coordinates_x_dSpinBox.value(),
                                                            self.ui.lower_coordinates_y_dSpinBox.value(),
                                                            self.ui.lower_coordinates_z_dSpinBox.value(),
                                                            self.ui.upper_coordinates_x_dSpinBox.value(),
                                                            self.ui.upper_coordinates_y_dSpinBox.value(),
                                                            self.ui.upper_coordinates_z_dSpinBox.value(),
                                                            self.ui.face_combo_box.currentText())
        if region_inst is not None:
            self.controller.main_view.log_append_message('Region object, ID: ' + region_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class PlotObservationView(QtGui.QDialog):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller
        super(PlotObservationView, self).__init__()
        self.ui = plot_observation_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)
        self.ui.observation_filenames_toolButton.clicked.connect(self._obs_filenames_function)
        self.ui.plot_filename_toolButton.clicked.connect(self._plot_filenames_function)

    @QtCore.Slot()
    def _obs_filenames_function(self):
        filenames = QtGui.QFileDialog.getOpenFileNames(self)
        for filename in filenames[0]:
            self.ui.observation_filenames_textEdit.append(filename)

    @QtCore.Slot()
    def _plot_filenames_function(self):
        filename = QtGui.QFileDialog.getOpenFileName(self)
        self.ui.plot_filename_lineEdit.setText(filename[0])

    def accept(self):
        plot_obs_int = self.controller.plot_observation_settings(self.ui.variable_list_line_edit.text(),
                                                                 self.ui.observation_list_line_edit.text(),
                                                                 self.ui.observation_filenames_textEdit.toPlainText(),
                                                                 self.ui.plot_filename_lineEdit.text(),
                                                                 self.ui.legend_list_line_edit.text(),
                                                                 self.ui.legend_font_size_d_spin_box.value(),
                                                                 self.ui.x_label_line_edit.text(),
                                                                 self.ui.x_type_line_edit.text(),
                                                                 self.ui.x_range_1_d_spin_box.value(),
                                                                 self.ui.x_range_2_d_spin_box.value(),
                                                                 self.ui.y_label_line_edit.text(),
                                                                 self.ui.y_type_line_edit.text(),
                                                                 self.ui.y_range_1_d_spin_box.value(),
                                                                 self.ui.y_range_2_d_spin_box.value())
        if plot_obs_int is not None:
            self.controller.main_view.log_append_message('Plot Observation object, ID: ' + plot_obs_int + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class PointView(QtGui.QDialog):
    def __init__(self, mode, controller):
        self.controller = controller
        super(PointView, self).__init__()
        self.ui = point_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)

    def accept(self):
        pt_inst = self.controller.apply_point(self.ui.name_lineEdit.text(), self.ui.coordinate_x_dSpinBox.value(),
                                              self.ui.coordinate_y_dSpinBox.value(),
                                              self.ui.coordinate_z_dSpinBox.value())
        if pt_inst is not None:
            self.controller.main_view.log_append_message('Point object, ID: ' + pt_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()


class CharacteristicCurvesView(QtGui.QDialog):
    def __init__(self, mode, controller):
        self.controller = controller
        super(CharacteristicCurvesView, self).__init__()
        self.ui = characteristic_curves_ui.Ui_Dialog()
        self._build_ui()
        self.warning = GenericWarningView(field_missing)

    def _build_ui(self):
        self.ui.setupUi(self)
        # set power spinbox and label to disabled
        self.ui.power_dSpinBox.setDisabled(True)
        self.ui.power_label.setDisabled(True)

    def accept(self):
        cc_inst = self.controller.apply_characteristic_curve_setting(
            self.ui.name_lineEdit.text(),
            self.ui.saturation_func_type_comboBox.currentText(),
            self.ui.sf_alpha_dSpinBox.value(),
            self.ui.sf_m_dSpinBox.value(),
            self.ui.sf_lambda_dSpinBox.value(),
            self.ui.sf_liquid_residual_saturation_dSpinBox.value(),
            self.ui.sf_gas_residual_saturation_dSpinBox.value(),
            self.ui.max_cap_pressure_dSpinBox.value(),
            self.ui.smooth_comboBox.currentText(),
            self.ui.power_dSpinBox.value(),
            self.ui.default_comboBox.currentText(),
            self.ui.liquid_perm_function_type_comboBox.currentText(),
            self.ui.lpf_m_dSpinBox.value(),
            self.ui.lpf_lambda_dSpinBox.value(),
            self.ui.lpf_liquid_residual_sat_dSpinBox.value(),
            self.ui.gas_perm_function_type_comboBox.currentText(),
            self.ui.gpf_m_dSpinBox.value(),
            self.ui.gpf_lambda_doubleSpinBox.value(),
            self.ui.gpf_liquid_residual_sat_dSpinBox.value(),
            self.ui.gpf_gas_residual_sat_dSpinBox.value())

        if cc_inst is not None:
            self.controller.main_view.log_append_message('Characteristic Curve object, ID: ' + cc_inst + ', created.')
            self.hide()
        else:
            self.warning.exec_()

    def reject(self):
        self.hide()
