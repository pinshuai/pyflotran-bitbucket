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
        self.gui.uniform_velocity_view.ui.vlx_dSpinBox.setValue(1)
        self.gui.uniform_velocity_view.ui.unit_lineEdit.setText('m/yr')
        self.gui.uniform_velocity_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertIsInstance(self.p_objects.UNIFORM_VELOCITY_OBJECT[0], pdata.puniform_velocity)

        # Set chemistry
        self.gui.main_ui.action_add_chemistry.trigger()
        self.assertIsInstance(self.gui.chemistry_view, other_views.ChemistryView)
        self.gui.chemistry_view.ui.species_list_textEdit.setText('H+, HCO3-, Ca++')
        self.gui.chemistry_view.ui.aq_species_in_eq_with_primary_species_textEdit.setText(
            'OH-, CO3--, CO2(aq), CaCO3(aq), CaHCO3+, CaOH+')
        self.gui.chemistry_view.ui.gas_species_textEdit.setText('CO2(g)')
        self.gui.chemistry_view.ui.mineral_list_textEdit.setText('Calcite')
        self.gui.chemistry_view.ui.database_lineEdit.setText('/database/hanford.dat')
        self.gui.chemistry_view.ui.log_formulation_comboBox.setCurrentIndex(1) #  Sets it to 'True'
        self.gui.chemistry_view.ui.activity_coefficients_comboBox.setCurrentIndex(3)
        self.gui.chemistry_view.ui.output_list_lineEdit.setText('PH, all, FREE_ION')
        # make mineral kinetic object
        self.gui.chemistry_view.ui.mineral_kinetic_name_lineEdit.setText('Calcite')
        self.gui.chemistry_view.ui.rate_const_list_dSpinBox.setValue(1e-6)
        self.gui.chemistry_view.ui.rate_const_list_unit_lineEdit.setText('mol/m^2-sec')
        self.gui.chemistry_view.ui.m_kinetic_add_pushButton.click()
        self.gui.chemistry_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertIsInstance(object_space.CHEMISTRY_OBJECT[0], pdata.pchemistry)

        # Set grid settings
        self.gui.main_ui.action_grid.trigger()
        self.assertIsInstance(self.gui.grid_view, other_views.GridView)
        self.gui.grid_view.ui.upper_bounds_x_dSpinBox.setValue(100)
        self.gui.grid_view.ui.upper_bounds_y_dSpinBox.setValue(1)
        self.gui.grid_view.ui.upper_bounds_z_dSpinBox.setValue(1)
        self.gui.grid_view.ui.grid_cells_x_dSpinBox.setValue(100)
        self.gui.grid_view.ui.grid_cells_y_dSpinBox.setValue(1)
        self.gui.grid_view.ui.grid_cells_z_dSpinBox.setValue(1)
        self.gui.grid_view.ui.gravity_vector_z_dSpinBox.setValue(-9.8068)
        self.gui.grid_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        # assert if pgrid exists
        self.assertIsInstance(self.p_objects.GRID_OBJECT[0], pdata.pgrid)

        # Set time settings
        self.gui.main_ui.action_add_time.trigger()
        self.assertIsInstance(self.gui.time_view, other_views.TimeView)
        self.gui.time_view.ui.final_time_dSpinBox.setValue(25)
        y_index = self.gui.time_view.ui.final_time_unit_comboBox.findText('y')
        self.gui.time_view.ui.final_time_unit_comboBox.setCurrentIndex(y_index)
        self.gui.time_view.ui.delta_time_init_dSpinBox.setValue(1)
        y_index = self.gui.time_view.ui.delta_time_init_unit_comboBox.findText('h')
        self.gui.time_view.ui.delta_time_init_unit_comboBox.setCurrentIndex(y_index)
        self.gui.time_view.ui.delta_time_final_dSpinBox.setValue(0.25)
        y_index = self.gui.time_view.ui.delta_time_final_unit_comboBox.findText('y')
        self.gui.time_view.ui.delta_time_final_unit_comboBox.setCurrentIndex(y_index)
        self.gui.time_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set material properties settings
        self.gui.main_ui.actionMaterial_Properties.trigger()
        self.assertIsInstance(self.gui.material_view, other_views.MaterialView)
        self.gui.material_view.ui.id_spinBox.setValue(1)
        self.gui.material_view.ui.name_lineEdit.setText('soil1')
        self.gui.material_view.ui.porosity_dSpinBox.setValue(0.25)
        self.gui.material_view.ui.tortuosity_dSpinBox.setValue(1)
        self.gui.material_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertIsInstance(self.p_objects.MATERIAL_PROP_OBJECT[0], pdata.pmaterial)

        # Set newton solver
        self.gui.main_ui.action_add_linear.trigger()
        self.assertIsInstance(self.gui.linear_solver_view, other_views.LinearSolverView)
        self.gui.linear_solver_view.ui.solver_comboBox.setCurrentIndex(3) # set to DirECT
        self.gui.linear_solver_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set output settings
        self.gui.main_ui.action_add_output.trigger()
        self.assertIsInstance(self.gui.output_view, other_views.OutputView)
        self.gui.output_view.ui.time_value_list_unit_comboBox.setCurrentIndex(6) # sets to 'y'
        self.gui.output_view.ui.time_values_list_lineEdit.setText('5, 10, 15, 20')
        self.gui.output_view.ui.tecplot_point_checkBox.setChecked(True)
        self.gui.output_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set fluid properties
        self.gui.main_ui.action_add_fluid_properties.trigger()
        self.assertIsInstance(self.gui.fluid_properties, other_views.FluidPropertiesView)
        self.gui.fluid_properties.ui.diff_coeff_lineEdit.setText('1.e-9')
        self.gui.fluid_properties.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()
        self.assertIsInstance(object_space.FLUID_PROPERTIES_OBJECT[0], pdata.pfluid)

        # Set regions
        self.gui.main_ui.action_add_region.trigger()
        self.assertIsInstance(self.gui.region_view, other_views.RegionView)
        self.gui.region_view.ui.name_line_edit.setText('ALL')
        self.gui.region_view.ui.upper_coordinates_x_dSpinBox.setValue(100)
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
        self.gui.region_view.ui.lower_coordinates_x_dSpinBox.setValue(100)
        self.gui.region_view.ui.upper_coordinates_x_dSpinBox.setValue(100)
        self.gui.region_view.ui.upper_coordinates_y_dSpinBox.setValue(1)
        self.gui.region_view.ui.upper_coordinates_z_dSpinBox.setValue(1)
        self.gui.region_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # set transport conditions
        self.gui.main_ui.action_add_transport.trigger()
        self.assertIsInstance(self.gui.transport_condition_view, other_views.TransportConditionView)
        self.gui.transport_condition_view.ui.name_lineEdit.setText('background_CONC')
        self.gui.transport_condition_view.ui.type_comboBox.setCurrentIndex(7)  # 'zero gradient'
        self.gui.transport_condition_view.ui.constraint_list_type_lineEdit.setText('initial_CONSTRAINT')
        self.gui.transport_condition_view.ui.push_down_tool_button.click()
        self.gui.transport_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_transport.trigger()
        self.assertIsInstance(self.gui.transport_condition_view, other_views.TransportConditionView)
        self.gui.transport_condition_view.ui.name_lineEdit.setText('inlet_conc')
        self.gui.transport_condition_view.ui.type_comboBox.setCurrentIndex(2)  # 'dirichlet_zero_gradient'
        self.gui.transport_condition_view.ui.constraint_list_type_lineEdit.setText('inlet_constraint')
        self.gui.transport_condition_view.ui.push_down_tool_button.click()
        self.gui.transport_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # set Initial Condition
        self.gui.main_ui.action_add_initial_condition.trigger()
        self.assertIsInstance(self.gui.init_condition_view, other_views.InitialConditionView)
        self.gui.init_condition_view.ui.name_lineEdit.setText('initial')
        self.gui.init_condition_view.ui.transport_lineEdit.setText('background_CONC')
        self.gui.init_condition_view.ui.region_lineEdit.setText('ALL')
        self.gui.init_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # Set boundary conditions
        self.gui.main_ui.action_add_boundary_condition.trigger()
        self.assertIsInstance(self.gui.boundary_condition_view, other_views.BoundaryConditionView)
        self.gui.boundary_condition_view.ui.nameLineEdit.setText('OUTLEt')
        self.gui.boundary_condition_view.ui.transportLineEdit.setText('background_CONC')
        self.gui.boundary_condition_view.ui.regionLineEdit.setText('EAST')
        self.gui.boundary_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_add_boundary_condition.trigger()
        self.assertIsInstance(self.gui.boundary_condition_view, other_views.BoundaryConditionView)
        self.gui.boundary_condition_view.ui.nameLineEdit.setText('inlet')
        self.gui.boundary_condition_view.ui.transportLineEdit.setText('inlet_conc')
        self.gui.boundary_condition_view.ui.regionLineEdit.setText('west')
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
        self.gui.constraints_condition_view.ui.name_lineEdit.setText('initial_CONSTRAINT')
        # add concentrations
        self.gui.constraints_condition_view.ui.add_concentration_pushButton.click()
        self.assertIsInstance(
            self.gui.constraints_condition_view.concentration_view, other_views.ConstraintConditionConcentrationView)
        self.gui.constraints_condition_view.concentration_view.ui.species_lineEdit.setText('H+')
        self.gui.constraints_condition_view.concentration_view.ui.value_dSpinBox.setValue(1e-8)
        self.gui.constraints_condition_view.concentration_view.ui.constraint_comboBox.setCurrentIndex(1) # free
        self.gui.constraints_condition_view.concentration_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.constraints_condition_view.ui.add_concentration_pushButton.click()
        self.assertIsInstance(
            self.gui.constraints_condition_view.concentration_view, other_views.ConstraintConditionConcentrationView)
        self.gui.constraints_condition_view.concentration_view.ui.species_lineEdit.setText('HCO3-')
        self.gui.constraints_condition_view.concentration_view.ui.value_dSpinBox.setValue(1e-3)
        self.gui.constraints_condition_view.concentration_view.ui.constraint_comboBox.setCurrentIndex(7)  # gas
        self.gui.constraints_condition_view.concentration_view.ui.element_lineEdit.setText('CO2(g)')
        self.gui.constraints_condition_view.concentration_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.constraints_condition_view.ui.add_concentration_pushButton.click()
        self.assertIsInstance(
            self.gui.constraints_condition_view.concentration_view, other_views.ConstraintConditionConcentrationView)
        self.gui.constraints_condition_view.concentration_view.ui.species_lineEdit.setText('Ca++')
        self.gui.constraints_condition_view.concentration_view.ui.value_dSpinBox.setValue(5e-4)
        self.gui.constraints_condition_view.concentration_view.ui.constraint_comboBox.setCurrentIndex(6)  # mineral
        self.gui.constraints_condition_view.concentration_view.ui.element_lineEdit.setText('Calcite')
        self.gui.constraints_condition_view.concentration_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # set mineral
        self.gui.constraints_condition_view.ui.add_mineral_pushButton.click()
        self.assertIsInstance(
            self.gui.constraints_condition_view.mineral_view, other_views.ConstraintConditionMineralView)
        self.gui.constraints_condition_view.mineral_view.ui.name_lineEdit.setText('Calcite')
        self.gui.constraints_condition_view.mineral_view.ui.volume_fraction_dSpinBox.setValue(1e-5)
        self.gui.constraints_condition_view.mineral_view.ui.surface_area_dSpinBox.setValue(1)
        self.gui.constraints_condition_view.mineral_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.constraints_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        # set constraint - 2nd
        self.gui.main_ui.action_add_constraints.trigger()
        self.assertIsInstance(self.gui.constraints_condition_view, other_views.ConstraintConditionView)
        self.gui.constraints_condition_view.ui.name_lineEdit.setText('inlet_constraint')
        # add concentrations
        self.gui.constraints_condition_view.ui.add_concentration_pushButton.click()
        self.assertIsInstance(
            self.gui.constraints_condition_view.concentration_view, other_views.ConstraintConditionConcentrationView)
        self.gui.constraints_condition_view.concentration_view.ui.species_lineEdit.setText('H+')
        self.gui.constraints_condition_view.concentration_view.ui.value_dSpinBox.setValue(5)
        self.gui.constraints_condition_view.concentration_view.ui.constraint_comboBox.setCurrentIndex(4)  # pH
        self.gui.constraints_condition_view.concentration_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.constraints_condition_view.ui.add_concentration_pushButton.click()
        self.assertIsInstance(
            self.gui.constraints_condition_view.concentration_view, other_views.ConstraintConditionConcentrationView)
        self.gui.constraints_condition_view.concentration_view.ui.species_lineEdit.setText('HCO3-')
        self.gui.constraints_condition_view.concentration_view.ui.value_dSpinBox.setValue(1e-3)
        self.gui.constraints_condition_view.concentration_view.ui.constraint_comboBox.setCurrentIndex(2)  # total
        self.gui.constraints_condition_view.concentration_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.constraints_condition_view.ui.add_concentration_pushButton.click()
        self.assertIsInstance(
            self.gui.constraints_condition_view.concentration_view, other_views.ConstraintConditionConcentrationView)
        self.gui.constraints_condition_view.concentration_view.ui.species_lineEdit.setText('Ca++')
        self.gui.constraints_condition_view.concentration_view.ui.value_dSpinBox.setValue(1e-6)
        self.gui.constraints_condition_view.concentration_view.ui.constraint_comboBox.setCurrentIndex(10)  # unknown what is Z?
        self.gui.constraints_condition_view.concentration_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.constraints_condition_view.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).click()

        self.gui.main_ui.action_generate_input_file.trigger()
        self.assertTrue(os.path.exists('file.in'))

    def test_compare_gold(self):
        gold = ''
        test = ''
        with open('calcite_tran_only.gold', 'r') as f:
            gold = f.read()
        with open('file.in', 'r') as f:
            test = f.read()

        self.assertEqual(gold, test)

    def tearDown(self):
        self.gui.close()



if __name__ == '__main__':
    unittest.main()
