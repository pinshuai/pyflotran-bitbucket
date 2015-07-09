from PySide import QtGui
from PySide.QtTest import QTest
from PySide.QtCore import Qt

import unittest
import os

from views import main_view, other_views
from model import main_model, object_space
from controllers import main_controller

from pyflotran import pdata

_app = QtGui.QApplication([])
gui = main_view.MainView(main_model, main_controller)
p_objects = object_space

class TestCALCITE(unittest.TestCase):
    def setUp(self):
        self.gui = gui
        QTest.qWaitForWindowShown(self.gui)
        self.p_objects = p_objects

    def test_open_input_views(self):
        # Set simulation settings
        self.gui.main_ui.actionSimulation.trigger()
        self.assertIsInstance(self.gui.sim_view, other_views.SimulationView)
        self.gui.sim_view.ui.subsurface_trans_checkBox.setChecked(True)
        self.gui.sim_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        # test to make sure psimulation object is made
        self.assertIsInstance(self.p_objects.SIMULATION_OBJECT[0], pdata.psimulation)

        # Set uniform velocity
        self.gui.main_ui.action_uniform_velocity.trigger()
        self.assertIsInstance(self.gui.uniform_velocity_view, other_views.UniformVelocityView)
        self.gui.uniform_velocity_view.ui.vlx_dSpinBox.setValue(14.4)
        self.gui.uniform_velocity_view.ui.unit_lineEdit.setText('m/yr')
        self.gui.uniform_velocity_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertIsInstance(self.p_objects.UNIFORM_VELOCITY_OBJECT[0], pdata.puniform_velocity)

        # Set chemistry
        self.gui.main_ui.action_add_chemistry.trigger()
        self.assertIsInstance(self.gui.chemistry_view, other_views.ChemistryView)
        self.gui.chemistry_view.ui.species_list_textEdit.setText('A(aq)')
        self.gui.chemistry_view.ui.molal_comboBox.setCurrentIndex(1)  # True
        self.gui.chemistry_view.ui.output_list_lineEdit.setText('All, Free_ion')
        self.gui.chemistry_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set grid settings
        # TODO: add bounds_bool
        self.gui.main_ui.action_grid.trigger()
        self.assertIsInstance(self.gui.grid_view, other_views.GridView)
        self.gui.grid_view.ui.upper_bounds_x_dSpinBox.setValue(0.04)
        self.gui.grid_view.ui.upper_bounds_y_dSpinBox.setValue(1)
        self.gui.grid_view.ui.upper_bounds_z_dSpinBox.setValue(1)
        self.gui.grid_view.ui.grid_cells_x_dSpinBox.setValue(100)
        self.gui.grid_view.ui.grid_cells_y_dSpinBox.setValue(1)
        self.gui.grid_view.ui.grid_cells_z_dSpinBox.setValue(1)
        self.gui.grid_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        # assert if pgrid exists
        self.assertIsInstance(self.p_objects.GRID_OBJECT[0], pdata.pgrid)

        # Set timestepper settings
        self.gui.main_ui.action_add_timestepper.trigger()
        self.assertIsInstance(self.gui.timestepper_view, other_views.TimestepperView)
        self.gui.timestepper_view.ui.ts_acceleration_spinBox.setValue(25)
        self.gui.timestepper_view.ui.max_ts_cuts_spinBox.setValue(10)
        self.gui.timestepper_view.ui.max_steps_spinBox.setValue(10000)
        self.gui.timestepper_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertIsInstance(self.p_objects.TIMESTEPPER_OBJECT[0], pdata.ptimestepper)

        # Set newton solver
        self.gui.main_ui.action_add_newton.trigger()
        self.assertIsInstance(self.gui.newton_solver_view, other_views.NewtonSolverView)
        flow_index = self.gui.newton_solver_view.ui.name_comboBox.findText('TRANSPORT')
        self.gui.newton_solver_view.ui.name_comboBox.setCurrentIndex(flow_index)
        self.gui.newton_solver_view.ui.abs_tolerance_lineEdit.setText('1e-15')
        self.gui.newton_solver_view.ui.rel_tolerance_lineEdit.setText('1e-10')
        self.gui.newton_solver_view.ui.rel_tolerance_update_lineEdit.setText('1e-30')
        self.gui.newton_solver_view.ui.tol_compared_to_inf_norm_lineEdit.setText('1e-8')
        self.gui.newton_solver_view.ui.max_iterations_dSpinBox.setValue(100)
        self.gui.newton_solver_view.ui.max_function_eval_dSpinBox.setValue(100)
        self.gui.newton_solver_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set fluid properties
        self.gui.main_ui.action_add_fluid_properties.trigger()
        self.assertIsInstance(self.gui.fluid_properties, other_views.FluidPropertiesView)
        self.gui.fluid_properties.ui.diff_coeff_lineEdit.setText('1.e-9')
        self.gui.fluid_properties.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertIsInstance(object_space.FLUID_PROPERTIES_OBJECT[0], pdata.pfluid)

        # Set material properties settings
        self.gui.main_ui.actionMaterial_Properties.trigger()
        self.assertIsInstance(self.gui.material_view, other_views.MaterialView)
        self.gui.material_view.ui.id_spinBox.setValue(1)
        self.gui.material_view.ui.name_lineEdit.setText('soil1')
        self.gui.material_view.ui.porosity_dSpinBox.setValue(1)
        self.gui.material_view.ui.tortuosity_dSpinBox.setValue(1)
        self.gui.material_view.ui.rock_density_dSpinBox.setValue(2.8e3)
        self.gui.material_view.ui.specific_heat_dSpinBox.setValue(1e3)
        self.gui.material_view.ui.conductivity_wet_dSpinBox.setValue(0.5)
        self.gui.material_view.ui.permeability_x_dSpinBox.setValue(1e-15)
        self.gui.material_view.ui.permeability_y_dSpinBox.setValue(1e-15)
        self.gui.material_view.ui.permeability_z_dSpinBox.setValue(1e-15)
        self.gui.material_view.ui.saturation_lineEdit.setText('default')
        self.gui.material_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertIsInstance(self.p_objects.MATERIAL_PROP_OBJECT[0], pdata.pmaterial)

        # set time settings
        self.gui.main_ui.action_add_time.trigger()
        self.assertIsInstance(self.gui.time_view, other_views.TimeView)
        self.gui.time_view.ui.final_time_dSpinBox.setValue(1e4)
        y_index = self.gui.time_view.ui.final_time_unit_comboBox.findText('s')
        self.gui.time_view.ui.final_time_unit_comboBox.setCurrentIndex(y_index)
        self.gui.time_view.ui.delta_time_init_dSpinBox.setValue(1e-6)
        y_index = self.gui.time_view.ui.delta_time_init_unit_comboBox.findText('s')
        self.gui.time_view.ui.delta_time_init_unit_comboBox.setCurrentIndex(y_index)
        self.gui.time_view.ui.delta_time_final_dSpinBox.setValue(10)
        y_index = self.gui.time_view.ui.delta_time_final_unit_comboBox.findText('s')
        self.gui.time_view.ui.delta_time_final_unit_comboBox.setCurrentIndex(y_index)
        self.gui.time_view.ui.delta_t_given_inst_time_value_dSpinBox.setValue(1e2)
        self.gui.time_view.ui.second_dt_value_dSpinBox.setValue(5e3)
        self.gui.time_view.ui.add_delta_time_row_button.click()
        self.gui.time_view.ui.delta_t_given_inst_time_value_dSpinBox.setValue(1e3)
        self.gui.time_view.ui.second_dt_value_dSpinBox.setValue(5e4)
        self.gui.time_view.ui.add_delta_time_row_button.click()
        self.gui.time_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set output settings
        self.gui.main_ui.action_add_output.trigger()
        self.assertIsInstance(self.gui.output_view, other_views.OutputView)
        self.gui.output_view.ui.time_values_list_lineEdit.setText('26042.0, 39063.0, 52083.0, 1000000.0')
        self.gui.output_view.ui.print_column_ids_comboBox.setCurrentIndex(1)
        self.gui.output_view.ui.periodic_obs_timestep_spinBox.setValue(1)
        self.gui.output_view.ui.tecplot_point_checkBox.setChecked(True)
        self.gui.output_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set saturation functions
        self.gui.main_ui.action_add_saturation_function.trigger()
        self.assertIsInstance(self.gui.saturation_function_view, other_views.SaturationFunctionView)
        self.gui.saturation_function_view.ui.name_lineEdit.setText('default')
        self.gui.saturation_function_view.ui.residual_saturation_liq_dSpinBox.setValue(0.1)
        self.gui.saturation_function_view.ui.saturation_function_type_comboBox.setCurrentIndex(1)
        self.gui.saturation_function_view.ui.lambda_dSpinBox.setValue(0.762)
        self.gui.saturation_function_view.ui.alpha_lineEdit.setText('7.5e-4')
        self.gui.saturation_function_view.ui.max_cap_pressure_dSpinBox.setValue(1e6)
        self.gui.saturation_function_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set regions
        self.gui.main_ui.action_add_region.trigger()
        self.assertIsInstance(self.gui.region_view, other_views.RegionView)
        self.gui.region_view.ui.name_line_edit.setText('all')
        self.gui.region_view.ui.upper_coordinates_x_dSpinBox.setValue(0.04)
        self.gui.region_view.ui.upper_coordinates_y_dSpinBox.setValue(1)
        self.gui.region_view.ui.upper_coordinates_z_dSpinBox.setValue(1)
        self.gui.region_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_region.trigger()
        self.assertIsInstance(self.gui.region_view, other_views.RegionView)
        self.gui.region_view.ui.name_line_edit.setText('west')
        face_index = self.gui.region_view.ui.face_combo_box.findText('west')
        self.gui.region_view.ui.face_combo_box.setCurrentIndex(face_index)
        self.gui.region_view.ui.upper_coordinates_y_dSpinBox.setValue(1)
        self.gui.region_view.ui.upper_coordinates_z_dSpinBox.setValue(1)
        self.gui.region_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_region.trigger()
        self.assertIsInstance(self.gui.region_view, other_views.RegionView)
        self.gui.region_view.ui.name_line_edit.setText('east')
        face_index = self.gui.region_view.ui.face_combo_box.findText('east')
        self.gui.region_view.ui.face_combo_box.setCurrentIndex(face_index)
        self.gui.region_view.ui.lower_coordinates_x_dSpinBox.setValue(0.04)
        self.gui.region_view.ui.upper_coordinates_x_dSpinBox.setValue(0.04)
        self.gui.region_view.ui.upper_coordinates_y_dSpinBox.setValue(1)
        self.gui.region_view.ui.upper_coordinates_z_dSpinBox.setValue(1)
        self.gui.region_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_region.trigger()
        self.assertIsInstance(self.gui.region_view, other_views.RegionView)
        self.gui.region_view.ui.name_line_edit.setText('obs')
        self.gui.region_view.ui.lower_coordinates_x_dSpinBox.setValue(0.04)
        self.gui.region_view.ui.upper_coordinates_x_dSpinBox.setValue(0.04)
        self.gui.region_view.ui.upper_coordinates_y_dSpinBox.setValue(1)
        self.gui.region_view.ui.upper_coordinates_z_dSpinBox.setValue(1)
        self.gui.region_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_observation.trigger()
        self.assertIsInstance(self.gui.observation_view, other_views.ObservationView)
        self.gui.observation_view.ui.region_line_edit.setText('obs')
        self.gui.observation_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # set transport conditions
        self.gui.main_ui.action_add_transport.trigger()
        self.assertIsInstance(self.gui.transport_condition_view, other_views.TransportConditionView)
        self.gui.transport_condition_view.ui.name_lineEdit.setText('initial')
        self.gui.transport_condition_view.ui.type_comboBox.setCurrentIndex(1)  # 'dirichlet'
        self.gui.transport_condition_view.ui.constraint_list_type_lineEdit.setText('initial')
        self.gui.transport_condition_view.ui.push_down_tool_button.click()
        self.gui.transport_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_transport.trigger()
        self.assertIsInstance(self.gui.transport_condition_view, other_views.TransportConditionView)
        self.gui.transport_condition_view.ui.name_lineEdit.setText('WEST')
        self.gui.transport_condition_view.ui.type_comboBox.setCurrentIndex(1)  # ' dirichlet'
        self.gui.transport_condition_view.ui.constraint_list_type_lineEdit.setText('west    ')
        self.gui.transport_condition_view.ui.push_down_tool_button.click()
        self.gui.transport_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_transport.trigger()
        self.assertIsInstance(self.gui.transport_condition_view, other_views.TransportConditionView)
        self.gui.transport_condition_view.ui.name_lineEdit.setText('east')
        self.gui.transport_condition_view.ui.type_comboBox.setCurrentIndex(7)  # 'zero dirichlet'
        self.gui.transport_condition_view.ui.constraint_list_type_lineEdit.setText('east')
        self.gui.transport_condition_view.ui.push_down_tool_button.click()
        self.gui.transport_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # set Initial Condition
        self.gui.main_ui.action_add_initial_condition.trigger()
        self.assertIsInstance(self.gui.init_condition_view, other_views.InitialConditionView)
        self.gui.init_condition_view.ui.name_lineEdit.setText('initial')
        self.gui.init_condition_view.ui.flow_lineEdit.setText('initial')
        self.gui.init_condition_view.ui.transport_lineEdit.setText('initial')
        self.gui.init_condition_view.ui.region_lineEdit.setText('ALL')
        self.gui.init_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set boundary conditions
        self.gui.main_ui.action_add_boundary_condition.trigger()
        self.assertIsInstance(self.gui.boundary_condition_view, other_views.BoundaryConditionView)
        self.gui.boundary_condition_view.ui.nameLineEdit.setText('west')
        self.gui.boundary_condition_view.ui.flowLineEdit.setText('west')
        self.gui.boundary_condition_view.ui.transportLineEdit.setText('west')
        self.gui.boundary_condition_view.ui.regionLineEdit.setText('west')
        self.gui.boundary_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_boundary_condition.trigger()
        self.assertIsInstance(self.gui.boundary_condition_view, other_views.BoundaryConditionView)
        self.gui.boundary_condition_view.ui.nameLineEdit.setText('east')
        self.gui.boundary_condition_view.ui.flowLineEdit.setText('east')
        self.gui.boundary_condition_view.ui.transportLineEdit.setText('EAST')
        self.gui.boundary_condition_view.ui.regionLineEdit.setText('east')
        self.gui.boundary_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set Strata
        self.gui.main_ui.action_add_strata.trigger()
        self.assertIsInstance(self.gui.strata_view, other_views.StratigraphyCouplerView)
        self.gui.strata_view.ui.region_lineEdit.setText('all')
        self.gui.strata_view.ui.material_lineEdit.setText('soil1')
        self.gui.strata_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set constraints - 1st
        self.gui.main_ui.action_add_constraints.trigger()
        self.assertIsInstance(self.gui.constraints_condition_view, other_views.ConstraintConditionView)
        self.gui.constraints_condition_view.ui.name_lineEdit.setText('initial')
        # add concentrations
        self.gui.constraints_condition_view.ui.add_concentration_pushButton.click()
        self.assertIsInstance(
            self.gui.constraints_condition_view.concentration_view, other_views.ConstraintConditionConcentrationView)
        self.gui.constraints_condition_view.concentration_view.ui.species_lineEdit.setText('A(aq)')
        self.gui.constraints_condition_view.concentration_view.ui.value_dSpinBox.setValue(0.1)
        self.gui.constraints_condition_view.concentration_view.ui.constraint_comboBox.setCurrentIndex(2)  # total
        self.gui.constraints_condition_view.concentration_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.constraints_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # set constraint - 2nd
        self.gui.main_ui.action_add_constraints.trigger()
        self.assertIsInstance(self.gui.constraints_condition_view, other_views.ConstraintConditionView)
        self.gui.constraints_condition_view.ui.name_lineEdit.setText('WEST')
        # add concentrations
        self.gui.constraints_condition_view.ui.add_concentration_pushButton.click()
        self.assertIsInstance(
            self.gui.constraints_condition_view.concentration_view, other_views.ConstraintConditionConcentrationView)
        self.gui.constraints_condition_view.concentration_view.ui.species_lineEdit.setText('A(aq)')
        self.gui.constraints_condition_view.concentration_view.ui.value_dSpinBox.setValue(1.e-8)
        self.gui.constraints_condition_view.concentration_view.ui.constraint_comboBox.setCurrentIndex(2)  # total
        self.gui.constraints_condition_view.concentration_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.constraints_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # set constraint - 3rd
        self.gui.main_ui.action_add_constraints.trigger()
        self.assertIsInstance(self.gui.constraints_condition_view, other_views.ConstraintConditionView)
        self.gui.constraints_condition_view.ui.name_lineEdit.setText('east')
        # add concentrations
        self.gui.constraints_condition_view.ui.add_concentration_pushButton.click()
        self.assertIsInstance(
            self.gui.constraints_condition_view.concentration_view, other_views.ConstraintConditionConcentrationView)
        self.gui.constraints_condition_view.concentration_view.ui.species_lineEdit.setText('A(aq)')
        self.gui.constraints_condition_view.concentration_view.ui.value_dSpinBox.setValue(1.E-02)
        self.gui.constraints_condition_view.concentration_view.ui.constraint_comboBox.setCurrentIndex(2)  # total
        self.gui.constraints_condition_view.concentration_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.constraints_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_generate_input_file.trigger()
        self.assertTrue(os.path.exists('file.in'))

    def test_compare_gold(self):
        gold = ''
        test = ''
        with open('tracer_1D.gold', 'r') as f:
            gold = f.read()
        with open('file.in', 'r') as f:
            test = f.read()

        self.assertEqual(gold, test)

    def tearDown(self):
        self.gui.close()



if __name__ == '__main__':
    unittest.main()
