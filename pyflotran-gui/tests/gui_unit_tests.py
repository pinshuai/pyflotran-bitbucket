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


class TestMainWindow(unittest.TestCase):
    """
    Test the MainWindow, and the component GUIs functionality
    """

    def setUp(self):
        self.gui = main_view.MainView(main_model, main_controller)

    def test_ipython_widget(self):
        QTest.mouseClick(self.gui.main_ui.py_shell_tab, Qt.LeftButton)

    def test_open_simulate(self):
        with self.assertRaises(AttributeError):
            self.assertIsNone(self.gui.sim_view)
        self.gui.main_ui.actionSimulation.trigger()
        self.gui.sim_view.hide()
        self.assertIsNotNone(self.gui.sim_view)

    def test_open_grid_view(self):
        with self.assertRaises(AttributeError):
            self.assertIsNone(self.gui.grid_view)
        self.gui.main_ui.action_grid.trigger()
        self.gui.grid_view.hide()
        self.assertIsNotNone(self.gui.grid_view)

    def test_open_output_view(self):
        with self.assertRaises(AttributeError):
            self.assertIsNone(self.gui.output_view)
        self.gui.main_ui.action_add_output.trigger()
        self.gui.output_view.hide()
        self.assertIsNotNone(self.gui.output_view)

    def tearDown(self):
        self.gui.close()


class TestGenericWarningView(unittest.TestCase):
    """
    Test to make sure dialog shows text given at instantiation
    """

    def setUp(self):
        self.gui = main_view.MainView(main_model, main_controller)
        self.blank_message = other_views.GenericWarningView()
        self.rand_message = other_views.GenericWarningView("A message!")

    def test_instantiated_message(self):
        self.assertEqual(self.blank_message.ui.label.text(), '')
        self.assertEqual(self.rand_message.ui.label.text(), 'A message!')

    def tearDown(self):
        self.blank_message.close()
        self.rand_message.close()


class TestSimulationView(unittest.TestCase):
    def setUp(self):
        self.ui = other_views.SimulationView(main_model, main_controller)

    def test_check_boxes(self):
        """
        Check on of the checkboxes, run accept() and test to see if ui is hidden (controller returned True)
        :return:
        """
        self.ui.ui.subsurface_flow_checkBox.setChecked(True)
        self.ui.accept()
        self.assertTrue(self.ui.isHidden())

    def tearDown(self):
        self.ui.close()


class TestMaterialView(unittest.TestCase):
    def setUp(self):
        self.ui = other_views.MaterialView(main_model, main_controller)

    def test_with_all_fields_filled(self):
        """
        Fills in all lineEdits and clicks OK button. Then checks to see if it is hidden (controller returns True)
        :return:
        """
        self.assertIsNotNone(self.ui)
        line_edits = self.ui.findChildren(QtGui.QLineEdit)
        for line_edit in line_edits:
            line_edit.setText('test')
        self.ui.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertTrue(self.ui.isHidden())

    def tearDown(self):
        self.ui.close()


class TestGridView(unittest.TestCase):
    """
    Fills in all lineEdits and clicks OK button. Then checks to see if it is hidden (controller returns True)
    """

    def setUp(self):
        self.ui = other_views.GridView(main_model, main_controller)

    def test_with_all_fields_filled(self):
        self.assertIsNotNone(self.ui)
        self.ui.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertTrue(self.ui.isHidden())

    def tearDown(self):
        self.ui.close()


gui = main_view.MainView(main_model, main_controller)
p_objects = object_space

class TestMPHASE(unittest.TestCase):
    def setUp(self):
        self.gui = gui
        QTest.qWaitForWindowShown(self.gui)
        self.p_objects = p_objects

    def test_open_input_views(self):
        # Set simulation settings
        self.gui.main_ui.actionSimulation.trigger()
        self.assertIsInstance(self.gui.sim_view, other_views.SimulationView)
        self.gui.sim_view.ui.subsurface_flow_checkBox.setChecked(True)
        mphase_index = self.gui.sim_view.ui.mode_comboBox.findText('mphase')
        self.gui.sim_view.ui.mode_comboBox.setCurrentIndex(mphase_index)
        self.gui.sim_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        # test to make sure psimulation object is made
        self.assertIsInstance(self.p_objects.SIMULATION_OBJECT[0], pdata.psimulation)

        # Set grid settings
        self.gui.main_ui.action_grid.trigger()
        self.assertIsInstance(self.gui.grid_view, other_views.GridView)
        self.gui.grid_view.ui.upper_bounds_x_dSpinBox.setValue(321)
        self.gui.grid_view.ui.upper_bounds_y_dSpinBox.setValue(1)
        self.gui.grid_view.ui.upper_bounds_z_dSpinBox.setValue(51)
        self.gui.grid_view.ui.grid_cells_x_dSpinBox.setValue(107)
        self.gui.grid_view.ui.grid_cells_y_dSpinBox.setValue(1)
        self.gui.grid_view.ui.grid_cells_z_dSpinBox.setValue(51)
        self.gui.grid_view.ui.gravity_vector_z_dSpinBox.setValue(-9.8068)
        self.gui.grid_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        # assert if pgrid exists
        self.assertIsInstance(self.p_objects.GRID_OBJECT[0], pdata.pgrid)

        # Set timestepper settings
        self.gui.main_ui.action_add_timestepper.trigger()
        self.assertIsInstance(self.gui.timestepper_view, other_views.TimestepperView)
        self.ui = self.gui.timestepper_view.ui
        self.ui.ts_acceleration_spinBox.setValue(8)
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertIsInstance(self.p_objects.TIMESTEPPER_OBJECT[0], pdata.ptimestepper)

        # Set material properties settings
        self.gui.main_ui.actionMaterial_Properties.trigger()
        self.assertIsInstance(self.gui.material_view, other_views.MaterialView)
        self.gui.material_view.ui.id_spinBox.setValue(1)
        self.gui.material_view.ui.name_lineEdit.setText('soil1')
        self.gui.material_view.ui.porosity_dSpinBox.setValue(0.15)
        self.gui.material_view.ui.tortuosity_dSpinBox.setValue(0.1)
        self.gui.material_view.ui.rock_density_dSpinBox.setValue(2650)
        self.gui.material_view.ui.specific_heat_dSpinBox.setValue(1000)
        self.gui.material_view.ui.conductivity_dry_dSpinBox.setValue(0.5)
        self.gui.material_view.ui.conductivity_wet_dSpinBox.setValue(0.5)
        self.gui.material_view.ui.saturation_lineEdit.setText('sf2')
        self.gui.material_view.ui.permeability_x_dSpinBox.setValue(1e-15)
        self.gui.material_view.ui.permeability_y_dSpinBox.setValue(1e-15)
        self.gui.material_view.ui.permeability_z_dSpinBox.setValue(1e-17)
        self.gui.material_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertIsInstance(self.p_objects.MATERIAL_PROP_OBJECT[0], pdata.pmaterial)

        # Set time settings
        self.gui.main_ui.action_add_time.trigger()
        self.assertIsInstance(self.gui.time_view, other_views.TimeView)
        self.gui.time_view.ui.final_time_dSpinBox.setValue(0.25)
        y_index = self.gui.time_view.ui.final_time_unit_comboBox.findText('y')
        self.gui.time_view.ui.final_time_unit_comboBox.setCurrentIndex(y_index)
        self.gui.time_view.ui.delta_time_init_dSpinBox.setValue(1e-6)
        y_index = self.gui.time_view.ui.delta_time_init_unit_comboBox.findText('y')
        self.gui.time_view.ui.delta_time_init_unit_comboBox.setCurrentIndex(y_index)
        self.gui.time_view.ui.delta_time_final_dSpinBox.setValue(50)
        y_index = self.gui.time_view.ui.delta_time_final_unit_comboBox.findText('y')
        self.gui.time_view.ui.delta_time_final_unit_comboBox.setCurrentIndex(y_index)

        self.gui.time_view.ui.delta_t_given_inst_time_value_dSpinBox.setValue(200)
        y_index = self.gui.time_view.ui.delta_t_given_inst_unit_comboBox.findText('y')
        self.gui.time_view.ui.delta_t_given_inst_unit_comboBox.setCurrentIndex(y_index)
        self.gui.time_view.ui.second_dt_value_dSpinBox.setValue(50)
        s_y_index = self.gui.time_view.ui.second_dt_unit_comboBox.findText('y')
        self.gui.time_view.ui.second_dt_unit_comboBox.setCurrentIndex(s_y_index)
        self.gui.time_view.ui.add_delta_time_row_button.click()

        self.gui.time_view.ui.delta_t_given_inst_time_value_dSpinBox.setValue(500)
        self.gui.time_view.ui.second_dt_value_dSpinBox.setValue(20000)
        self.gui.time_view.ui.delta_t_given_inst_unit_comboBox.setCurrentIndex(y_index)
        self.gui.time_view.ui.second_dt_unit_comboBox.setCurrentIndex(s_y_index)
        self.gui.time_view.ui.add_delta_time_row_button.click()

        self.gui.time_view.ui.delta_t_given_inst_time_value_dSpinBox.setValue(1000)
        self.gui.time_view.ui.second_dt_value_dSpinBox.setValue(50000)
        self.gui.time_view.ui.delta_t_given_inst_unit_comboBox.setCurrentIndex(y_index)
        self.gui.time_view.ui.second_dt_unit_comboBox.setCurrentIndex(s_y_index)
        self.gui.time_view.ui.add_delta_time_row_button.click()

        self.gui.time_view.ui.delta_t_given_inst_time_value_dSpinBox.setValue(5000)
        self.gui.time_view.ui.second_dt_value_dSpinBox.setValue(100000)
        self.gui.time_view.ui.delta_t_given_inst_unit_comboBox.setCurrentIndex(y_index)
        self.gui.time_view.ui.second_dt_unit_comboBox.setCurrentIndex(s_y_index)
        self.gui.time_view.ui.add_delta_time_row_button.click()

        self.gui.time_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertIsInstance(object_space.TIME_OBJECT[0], pdata.ptime)

        # Set newton solver
        self.gui.main_ui.action_add_newton.trigger()
        self.assertIsInstance(self.gui.newton_solver_view, other_views.NewtonSolverView)
        flow_index = self.gui.newton_solver_view.ui.name_comboBox.findText('FLOW')
        self.gui.newton_solver_view.ui.name_comboBox.setCurrentIndex(flow_index)
        self.gui.newton_solver_view.ui.abs_tolerance_lineEdit.setText('1e-12')
        self.gui.newton_solver_view.ui.rel_tolerance_lineEdit.setText('1e-12')
        self.gui.newton_solver_view.ui.rel_tolerance_update_lineEdit.setText('1e-30')
        self.gui.newton_solver_view.ui.divergence_tol_lineEdit.setText('1e15')
        self.gui.newton_solver_view.ui.tol_compared_to_inf_norm_lineEdit.setText('1e-8')
        self.gui.newton_solver_view.ui.max_iterations_dSpinBox.setValue(25)
        self.gui.newton_solver_view.ui.max_function_eval_dSpinBox.setValue(100)
        self.gui.newton_solver_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set output settings
        self.gui.main_ui.action_add_output.trigger()
        self.assertIsInstance(self.gui.output_view, other_views.OutputView)
        pci_index = self.gui.output_view.ui.print_column_ids_comboBox.findText('True')
        self.gui.output_view.ui.print_column_ids_comboBox.setCurrentIndex(pci_index)
        out_index = self.gui.output_view.ui.mass_balance_comboBox.findText('True')
        self.gui.output_view.ui.screen_output_comboBox.setCurrentIndex(out_index)
        self.gui.output_view.ui.periodic_obs_timestep_spinBox.setValue(1)
        self.gui.output_view.ui.tecplot_point_checkBox.setChecked(True)
        self.gui.output_view.ui.hdf5_checkBox.setChecked(True)
        mass_index = self.gui.output_view.ui.mass_balance_comboBox.findText('True')
        self.gui.output_view.ui.mass_balance_comboBox.setCurrentIndex(mass_index)
        self.gui.output_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set fluid properties
        self.gui.main_ui.action_add_fluid_properties.trigger()
        self.assertIsInstance(self.gui.fluid_properties, other_views.FluidPropertiesView)
        self.gui.fluid_properties.ui.diff_coeff_lineEdit.setText('1.e-9')
        self.gui.fluid_properties.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertIsInstance(object_space.FLUID_PROPERTIES_OBJECT[0], pdata.pfluid)

        # Set saturation functions
        self.gui.main_ui.action_add_saturation_function.trigger()
        self.assertIsInstance(self.gui.saturation_function_view, other_views.SaturationFunctionView)
        self.gui.saturation_function_view.ui.name_lineEdit.setText('sf2')
        pm_index = self.gui.saturation_function_view.ui.permeability_function_type_comboBox.findText('NMT_EXP')
        self.gui.saturation_function_view.ui.permeability_function_type_comboBox.setCurrentIndex(pm_index)
        sat_index = self.gui.saturation_function_view.ui.saturation_function_type_comboBox.findText('NMT_EXP')
        self.gui.saturation_function_view.ui.saturation_function_type_comboBox.setCurrentIndex(sat_index)
        self.gui.saturation_function_view.ui.residual_saturation_liq_dSpinBox.setValue(0.1)
        self.gui.saturation_function_view.ui.lambda_dSpinBox.setValue(0.762)
        self.gui.saturation_function_view.ui.alpha_lineEdit.setText('7.5e-4')
        self.gui.saturation_function_view.ui.max_cap_pressure_dSpinBox.setValue(1e6)
        self.gui.saturation_function_view.ui.beta_dSpinBox.setValue(2)
        self.gui.saturation_function_view.ui.power_dSpinBox.setValue(7)
        self.gui.saturation_function_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.assertIsInstance(object_space.SATURATION_FUNCTION_OBJECT[0], pdata.psaturation)

        # Set regions
        self.gui.main_ui.action_add_region.trigger()
        self.assertIsInstance(self.gui.region_view, other_views.RegionView)
        self.gui.region_view.ui.name_line_edit.setText('all')
        self.gui.region_view.ui.lower_coordinates_x_dSpinBox.setValue(0)
        self.gui.region_view.ui.lower_coordinates_y_dSpinBox.setValue(0)
        self.gui.region_view.ui.lower_coordinates_z_dSpinBox.setValue(0)
        self.gui.region_view.ui.upper_coordinates_x_dSpinBox.setValue(321)
        self.gui.region_view.ui.upper_coordinates_y_dSpinBox.setValue(1)
        self.gui.region_view.ui.upper_coordinates_z_dSpinBox.setValue(51)
        self.gui.region_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_region.trigger()
        self.assertIsInstance(self.gui.region_view, other_views.RegionView)
        self.gui.region_view.ui.name_line_edit.setText('top')
        face_index = self.gui.region_view.ui.face_combo_box.findText('top')
        self.gui.region_view.ui.face_combo_box.setCurrentIndex(face_index)
        self.gui.region_view.ui.lower_coordinates_x_dSpinBox.setValue(0)
        self.gui.region_view.ui.lower_coordinates_y_dSpinBox.setValue(0)
        self.gui.region_view.ui.lower_coordinates_z_dSpinBox.setValue(51)
        self.gui.region_view.ui.upper_coordinates_x_dSpinBox.setValue(321)
        self.gui.region_view.ui.upper_coordinates_y_dSpinBox.setValue(1)
        self.gui.region_view.ui.upper_coordinates_z_dSpinBox.setValue(51)
        self.gui.region_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_region.trigger()
        self.assertIsInstance(self.gui.region_view, other_views.RegionView)
        self.gui.region_view.ui.name_line_edit.setText('west')
        face_index = self.gui.region_view.ui.face_combo_box.findText('west')
        self.gui.region_view.ui.face_combo_box.setCurrentIndex(face_index)
        self.gui.region_view.ui.lower_coordinates_x_dSpinBox.setValue(0)
        self.gui.region_view.ui.lower_coordinates_y_dSpinBox.setValue(0)
        self.gui.region_view.ui.lower_coordinates_z_dSpinBox.setValue(0)
        self.gui.region_view.ui.upper_coordinates_x_dSpinBox.setValue(0)
        self.gui.region_view.ui.upper_coordinates_y_dSpinBox.setValue(1)
        self.gui.region_view.ui.upper_coordinates_z_dSpinBox.setValue(51)
        self.gui.region_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_region.trigger()
        self.assertIsInstance(self.gui.region_view, other_views.RegionView)
        self.gui.region_view.ui.name_line_edit.setText('EAST')
        face_index = self.gui.region_view.ui.face_combo_box.findText('east')
        self.gui.region_view.ui.face_combo_box.setCurrentIndex(face_index)
        self.gui.region_view.ui.lower_coordinates_x_dSpinBox.setValue(321)
        self.gui.region_view.ui.lower_coordinates_y_dSpinBox.setValue(0)
        self.gui.region_view.ui.lower_coordinates_z_dSpinBox.setValue(0)
        self.gui.region_view.ui.upper_coordinates_x_dSpinBox.setValue(321)
        self.gui.region_view.ui.upper_coordinates_y_dSpinBox.setValue(1)
        self.gui.region_view.ui.upper_coordinates_z_dSpinBox.setValue(51)
        self.gui.region_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_region.trigger()
        self.assertIsInstance(self.gui.region_view, other_views.RegionView)
        self.gui.region_view.ui.name_line_edit.setText('well')
        self.gui.region_view.ui.lower_coordinates_x_dSpinBox.setValue(160)
        self.gui.region_view.ui.lower_coordinates_y_dSpinBox.setValue(1)
        self.gui.region_view.ui.lower_coordinates_z_dSpinBox.setValue(20)
        self.gui.region_view.ui.upper_coordinates_x_dSpinBox.setValue(160)
        self.gui.region_view.ui.upper_coordinates_y_dSpinBox.setValue(1)
        self.gui.region_view.ui.upper_coordinates_z_dSpinBox.setValue(20)
        self.gui.region_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set Flow and flow variables
        # First flow:
        self.gui.main_ui.action_add_flow.trigger()
        self.assertIsInstance(self.gui.flow_view, other_views.FlowView)
        self.gui.flow_view.ui.name_lineEdit.setText('initial')
        self.gui.flow_view.ui.iphase_spinBox.setValue(1)
        self.gui.flow_view.ui.datum_dx_dSpinBox.setValue(3)
        self.gui.flow_view.ui.datum_dy_dSpinBox.setValue(5)
        self.gui.flow_view.ui.datum_dz_dSpinBox.setValue(2)

        self.gui.flow_view.ui.add_pushButton.click()
        self.assertIsInstance(self.gui.flow_view.flow_var_adder, other_views.FlowVariableAdderView)
        name_index = self.gui.flow_view.flow_var_adder.ui.name_comboBox.findText('PRESSURE')
        self.gui.flow_view.flow_var_adder.ui.name_comboBox.setCurrentIndex(name_index)
        type_index = self.gui.flow_view.flow_var_adder.ui.type_comboBox.findText('Hydrostatic')
        self.gui.flow_view.flow_var_adder.ui.type_comboBox.setCurrentIndex(type_index)
        self.gui.flow_view.flow_var_adder.ui.value_list_1_dSpinBox.setValue(2e7)
        self.gui.flow_view.flow_var_adder.ui.value_list_2_dSpinBox.setValue(2e7)
        self.gui.flow_view.flow_var_adder.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.flow_view.ui.add_pushButton.click()
        self.assertIsInstance(self.gui.flow_view.flow_var_adder, other_views.FlowVariableAdderView)
        name_index = self.gui.flow_view.flow_var_adder.ui.name_comboBox.findText('TEMPERATURE')
        self.gui.flow_view.flow_var_adder.ui.name_comboBox.setCurrentIndex(name_index)
        type_index = self.gui.flow_view.flow_var_adder.ui.type_comboBox.findText('Zero Gradient')
        self.gui.flow_view.flow_var_adder.ui.type_comboBox.setCurrentIndex(type_index)
        self.gui.flow_view.flow_var_adder.ui.value_list_1_dSpinBox.setValue(50)
        self.gui.flow_view.flow_var_adder.ui.unit_lineEdit.setText('C')
        self.gui.flow_view.flow_var_adder.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.flow_view.ui.add_pushButton.click()
        self.assertIsInstance(self.gui.flow_view.flow_var_adder, other_views.FlowVariableAdderView)
        name_index = self.gui.flow_view.flow_var_adder.ui.name_comboBox.findText('CONCENTRATION')
        self.gui.flow_view.flow_var_adder.ui.name_comboBox.setCurrentIndex(name_index)
        type_index = self.gui.flow_view.flow_var_adder.ui.type_comboBox.findText('Zero Gradient')
        self.gui.flow_view.flow_var_adder.ui.type_comboBox.setCurrentIndex(type_index)
        self.gui.flow_view.flow_var_adder.ui.value_list_1_dSpinBox.setValue(1e-6)
        self.gui.flow_view.flow_var_adder.ui.unit_lineEdit.setText('M')
        self.gui.flow_view.flow_var_adder.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.flow_view.ui.add_pushButton.click()
        self.assertIsInstance(self.gui.flow_view.flow_var_adder, other_views.FlowVariableAdderView)
        name_index = self.gui.flow_view.flow_var_adder.ui.name_comboBox.findText('ENTHALPY')
        self.gui.flow_view.flow_var_adder.ui.name_comboBox.setCurrentIndex(name_index)
        type_index = self.gui.flow_view.flow_var_adder.ui.type_comboBox.findText('Dirichlet')
        self.gui.flow_view.flow_var_adder.ui.type_comboBox.setCurrentIndex(type_index)
        self.gui.flow_view.flow_var_adder.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.flow_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Second flow
        self.gui.main_ui.action_add_flow.trigger()
        self.assertIsInstance(self.gui.flow_view, other_views.FlowView)
        self.gui.flow_view.ui.name_lineEdit.setText('top')
        self.gui.flow_view.ui.iphase_spinBox.setValue(1)

        self.gui.flow_view.ui.add_pushButton.click()
        self.assertIsInstance(self.gui.flow_view.flow_var_adder, other_views.FlowVariableAdderView)
        name_index = self.gui.flow_view.flow_var_adder.ui.name_comboBox.findText('PRESSURE')
        self.gui.flow_view.flow_var_adder.ui.name_comboBox.setCurrentIndex(name_index)
        type_index = self.gui.flow_view.flow_var_adder.ui.type_comboBox.findText('Dirichlet')
        self.gui.flow_view.flow_var_adder.ui.type_comboBox.setCurrentIndex(type_index)
        self.gui.flow_view.flow_var_adder.ui.value_list_1_dSpinBox.setValue(3e7)
        self.gui.flow_view.flow_var_adder.ui.value_list_2_dSpinBox.setValue(2e7)
        self.gui.flow_view.flow_var_adder.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.flow_view.ui.add_pushButton.click()
        self.assertIsInstance(self.gui.flow_view.flow_var_adder, other_views.FlowVariableAdderView)
        name_index = self.gui.flow_view.flow_var_adder.ui.name_comboBox.findText('TEMPERATURE')
        self.gui.flow_view.flow_var_adder.ui.name_comboBox.setCurrentIndex(name_index)
        type_index = self.gui.flow_view.flow_var_adder.ui.type_comboBox.findText('Zero Gradient')
        self.gui.flow_view.flow_var_adder.ui.type_comboBox.setCurrentIndex(type_index)
        self.gui.flow_view.flow_var_adder.ui.value_list_1_dSpinBox.setValue(60)
        self.gui.flow_view.flow_var_adder.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.flow_view.ui.add_pushButton.click()
        self.assertIsInstance(self.gui.flow_view.flow_var_adder, other_views.FlowVariableAdderView)
        name_index = self.gui.flow_view.flow_var_adder.ui.name_comboBox.findText('CONCENTRATION')
        self.gui.flow_view.flow_var_adder.ui.name_comboBox.setCurrentIndex(name_index)
        type_index = self.gui.flow_view.flow_var_adder.ui.type_comboBox.findText('Zero Gradient')
        self.gui.flow_view.flow_var_adder.ui.type_comboBox.setCurrentIndex(type_index)
        self.gui.flow_view.flow_var_adder.ui.value_list_1_dSpinBox.setValue(1e-6)
        self.gui.flow_view.flow_var_adder.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.flow_view.ui.add_pushButton.click()
        self.assertIsInstance(self.gui.flow_view.flow_var_adder, other_views.FlowVariableAdderView)
        name_index = self.gui.flow_view.flow_var_adder.ui.name_comboBox.findText('ENTHALPY')
        self.gui.flow_view.flow_var_adder.ui.name_comboBox.setCurrentIndex(name_index)
        type_index = self.gui.flow_view.flow_var_adder.ui.type_comboBox.findText('Dirichlet')
        self.gui.flow_view.flow_var_adder.ui.type_comboBox.setCurrentIndex(type_index)
        self.gui.flow_view.flow_var_adder.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.flow_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Third flow
        self.gui.main_ui.action_add_flow.trigger()
        self.assertIsInstance(self.gui.flow_view, other_views.FlowView)
        self.gui.flow_view.ui.name_lineEdit.setText('source')
        self.gui.flow_view.ui.iphase_spinBox.setValue(0)
        self.gui.flow_view.ui.sync_ts_with_update_comboBox.setCurrentIndex(1)

        self.gui.flow_view.ui.add_pushButton.click()
        self.assertIsInstance(self.gui.flow_view.flow_var_adder, other_views.FlowVariableAdderView)
        name_index = self.gui.flow_view.flow_var_adder.ui.name_comboBox.findText('RATE')
        self.gui.flow_view.flow_var_adder.ui.name_comboBox.setCurrentIndex(name_index)
        type_index = self.gui.flow_view.flow_var_adder.ui.type_comboBox.findText('Mass Rate')
        self.gui.flow_view.flow_var_adder.ui.type_comboBox.setCurrentIndex(type_index)
        self.gui.flow_view.flow_var_adder.ui.time_unit_type_lineEdit.setText('y')
        self.gui.flow_view.flow_var_adder.ui.data_unit_type_lineEdit.setText('kg/s')
        self.gui.flow_view.flow_var_adder.ui.time_unit_value_dSpinBox.setValue(0)
        self.gui.flow_view.flow_var_adder.ui.data_unit_value_list_lineEdit.setText('0, 1.e-4')
        self.gui.flow_view.flow_var_adder.ui.push_down_tool_button.click()
        self.gui.flow_view.flow_var_adder.ui.time_unit_value_dSpinBox.setValue(10)
        self.gui.flow_view.flow_var_adder.ui.data_unit_value_list_lineEdit.setText('0, 0')
        self.gui.flow_view.flow_var_adder.ui.push_down_tool_button.click()
        self.gui.flow_view.flow_var_adder.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.flow_view.ui.add_pushButton.click()
        self.assertIsInstance(self.gui.flow_view.flow_var_adder, other_views.FlowVariableAdderView)
        name_index = self.gui.flow_view.flow_var_adder.ui.name_comboBox.findText('PRESSURE')
        self.gui.flow_view.flow_var_adder.ui.name_comboBox.setCurrentIndex(name_index)
        type_index = self.gui.flow_view.flow_var_adder.ui.type_comboBox.findText('Dirichlet')
        self.gui.flow_view.flow_var_adder.ui.type_comboBox.setCurrentIndex(type_index)
        self.gui.flow_view.flow_var_adder.ui.value_list_1_dSpinBox.setValue(40000000)
        self.gui.flow_view.flow_var_adder.ui.value_list_2_dSpinBox.setValue(20000000)
        self.gui.flow_view.flow_var_adder.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.flow_view.ui.add_pushButton.click()
        self.assertIsInstance(self.gui.flow_view.flow_var_adder, other_views.FlowVariableAdderView)
        name_index = self.gui.flow_view.flow_var_adder.ui.name_comboBox.findText('TEMPERATURE')
        self.gui.flow_view.flow_var_adder.ui.name_comboBox.setCurrentIndex(name_index)
        type_index = self.gui.flow_view.flow_var_adder.ui.type_comboBox.findText('Dirichlet')
        self.gui.flow_view.flow_var_adder.ui.type_comboBox.setCurrentIndex(type_index)
        self.gui.flow_view.flow_var_adder.ui.value_list_1_dSpinBox.setValue(70)
        self.gui.flow_view.flow_var_adder.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.flow_view.ui.add_pushButton.click()
        self.assertIsInstance(self.gui.flow_view.flow_var_adder, other_views.FlowVariableAdderView)
        name_index = self.gui.flow_view.flow_var_adder.ui.name_comboBox.findText('CONCENTRATION')
        self.gui.flow_view.flow_var_adder.ui.name_comboBox.setCurrentIndex(name_index)
        type_index = self.gui.flow_view.flow_var_adder.ui.type_comboBox.findText('Dirichlet')
        self.gui.flow_view.flow_var_adder.ui.type_comboBox.setCurrentIndex(type_index)
        self.gui.flow_view.flow_var_adder.ui.value_list_1_dSpinBox.setValue(0)
        self.gui.flow_view.flow_var_adder.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.flow_view.ui.add_pushButton.click()
        self.assertIsInstance(self.gui.flow_view.flow_var_adder, other_views.FlowVariableAdderView)
        name_index = self.gui.flow_view.flow_var_adder.ui.name_comboBox.findText('ENTHALPY')
        self.gui.flow_view.flow_var_adder.ui.name_comboBox.setCurrentIndex(name_index)
        type_index = self.gui.flow_view.flow_var_adder.ui.type_comboBox.findText('Dirichlet')
        self.gui.flow_view.flow_var_adder.ui.type_comboBox.setCurrentIndex(type_index)
        self.gui.flow_view.flow_var_adder.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.flow_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # set Initial Condition
        self.gui.main_ui.action_add_initial_condition.trigger()
        self.assertIsInstance(self.gui.init_condition_view, other_views.InitialConditionView)
        self.gui.init_condition_view.ui.name_lineEdit.setText('initial')
        self.gui.init_condition_view.ui.flow_lineEdit.setText('initial')
        self.gui.init_condition_view.ui.region_lineEdit.setText('all')
        self.gui.init_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set boundary conditions
        self.gui.main_ui.action_add_boundary_condition.trigger()
        self.assertIsInstance(self.gui.boundary_condition_view, other_views.BoundaryConditionView)
        self.gui.boundary_condition_view.ui.nameLineEdit.setText('WEST')
        self.gui.boundary_condition_view.ui.flowLineEdit.setText('INITIAL')
        self.gui.boundary_condition_view.ui.transportLineEdit.setText('None')
        self.gui.boundary_condition_view.ui.regionLineEdit.setText('west')
        self.gui.boundary_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_boundary_condition.trigger()
        self.assertIsInstance(self.gui.boundary_condition_view, other_views.BoundaryConditionView)
        self.gui.boundary_condition_view.ui.nameLineEdit.setText('east')
        self.gui.boundary_condition_view.ui.flowLineEdit.setText('INITIAL')
        self.gui.boundary_condition_view.ui.transportLineEdit.setText('None')
        self.gui.boundary_condition_view.ui.regionLineEdit.setText('east')
        self.gui.boundary_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set source sink condition
        self.gui.main_ui.action_add_source_sink_condition.trigger()
        self.assertIsInstance(self.gui.source_sink_view, other_views.SourceSinkView)
        self.gui.source_sink_view.ui.name_lineEdit.setText(' ')
        self.gui.source_sink_view.ui.flow_lineEdit.setText('source')
        self.gui.source_sink_view.ui.region_lineEdit.setText('WELL')
        self.gui.source_sink_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set Strata
        self.gui.main_ui.action_add_strata.trigger()
        self.assertIsInstance(self.gui.strata_view, other_views.StratigraphyCouplerView)
        self.gui.strata_view.ui.region_lineEdit.setText('ALL')
        self.gui.strata_view.ui.material_lineEdit.setText('SOIL1')
        self.gui.strata_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_generate_input_file.trigger()
        self.assertTrue(os.path.exists('file.in'))

    def test_compare_gold(self):
        gold = ''
        test = ''
        with open('mphase.gold', 'r') as f:
            gold = f.read()
        with open('file.in', 'r') as f:
            test = f.read()
        # self.assertEqual(gold, test)

    def tearDown(self):
        self.gui.close()


if __name__ == '__main__':
    unittest.main()
