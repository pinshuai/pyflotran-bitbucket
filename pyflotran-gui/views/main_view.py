from gen.main_ui import Ui_MainWindow
import other_views
from model import main_model
from PySide import QtGui, QtCore
import time
from pyflotran import pdata

"""
Using .exec_() on forms will cause the program flow to stop. This causes issues when testing because
the code stops working until the window stops halting the flow.
"""


class ProgressBar(QtGui.QProgressBar):
    """
    A generic class to quickly make QProgressBars.
    """
    def __init__(self, min_value, max_value):
        super(ProgressBar, self).__init__()
        self.setMinimum(min_value)
        self.setMaximum(max_value)

    def take_step(self, step_value):
        self.setValue(step_value)


class MainView(QtGui.QMainWindow):
    """
    The MainWindow view which links all the menu items to functions that open the associated view (form/widget)
    """
    def __init__(self, model, controller):
        super(MainView, self).__init__()
        self.model = model
        self.controller = controller
        self.default_window_state = QtCore.QSettings('LANL', 'PyFLOTRAN/DEFAULT')
        self._build_ui()
        self.__ipy_inst = False
        self.save_default_state()
        self.restore_window_settings()

        # Set the tabWidgets to index=0 (placeholder tab) to allow the other tabs to work as if they were buttons.
        self.main_ui.sidebar_tabWidget.setCurrentIndex(0)
        self.main_ui.bottom_bar_tabWidget.setCurrentIndex(0)
        # Hide the ipython widget for aesthetics purposes since the console does not load until it is needed.
        self.main_ui.ipython_dockWidget.hide()

        self.controller.main_view = self

    def closeEvent(self, event):
        """
        Overloads the closeEvent() to save the geometry and state of the window.
        :param event:
        :return:
        """
        settings = QtCore.QSettings("LANL", "PyFLOTRAN")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        QtGui.QMainWindow.closeEvent(self, event)

    def restore_window_settings(self):
        settings = QtCore.QSettings("LANL", "PyFLOTRAN")
        self.restoreGeometry(settings.value("geometry"))
        self.restoreState(settings.value("windowState"))

    # NOTE: This only needs to be done once, (saved in registry)
    def save_default_state(self):
        self.default_window_state.setValue('geometry', self.saveGeometry())
        self.default_window_state.setValue('windowState', self.saveState())

    def restore_default_state(self):
        self.restoreGeometry(self.default_window_state.value('geometry'))
        self.restoreState(self.default_window_state.value('windowState'))
        self.main_ui.ipython_dockWidget.hide()

    def _build_ui(self):
        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)

        # Adding functionality for the menu buttons
        self.main_ui.actionOpen.triggered.connect(self._open_file)
        self.main_ui.action_add_remove_current_attributes.triggered.connect(self._remove_all_attributes)
        self.main_ui.actionSimulation.triggered.connect(self._open_simulation_view)
        self.main_ui.action_grid.triggered.connect(self._open_grid_view)
        self.main_ui.action_add_output.triggered.connect(self._open_output_view)
        self.main_ui.action_add_observation.triggered.connect(self._open_observation_view)
        self.main_ui.action_add_strata.triggered.connect(self._open_strata_view)
        self.main_ui.actionMaterial_Properties.triggered.connect(self._open_material_view)
        self.main_ui.action_add_initial_condition.triggered.connect(self._open_initial_condition_view)
        self.main_ui.action_add_boundary_condition.triggered.connect(self._open_boundary_condition_view)
        self.main_ui.action_add_transport.triggered.connect(self._open_transport_condition_view)
        self.main_ui.action_add_source_sink_condition.triggered.connect(self._open_source_sink_view)
        self.main_ui.action_add_newton.triggered.connect(self._open_newton_solver_view)
        self.main_ui.action_add_linear.triggered.connect(self._open_linear_solver_view)
        self.main_ui.action_add_time.triggered.connect(self._open_time_view)
        self.main_ui.action_add_timestepper.triggered.connect(self._open_timestepper_view)
        self.main_ui.actionCheckpoint.triggered.connect(self._open_checkpoint_view)
        self.main_ui.actionRestart.triggered.connect(self._open_restart_view)
        self.main_ui.action_add_flow.triggered.connect(self._open_flow_view)
        self.main_ui.action_add_constraints.triggered.connect(self._open_constraints_condition_view)
        self.main_ui.action_add_saturation_function.triggered.connect(self._open_saturation_function_view)
        self.main_ui.action_uniform_velocity.triggered.connect(self._open_uniform_velocity_view)
        self.main_ui.action_nonuniform_velocity.triggered.connect(self._open_nonuniform_velocity_view)
        self.main_ui.action_add_fluid_properties.triggered.connect(self._open_fluid_properties)
        self.main_ui.action_add_chemistry.triggered.connect(self._open_chemistry_view)
        self.main_ui.actionRegression.triggered.connect(self._open_regression_view)
        self.main_ui.actionDataset.triggered.connect(self._open_dataset_view)
        self.main_ui.actionPlot_Observation.triggered.connect(self._open_plot_observation_view)
        self.main_ui.action_add_region.triggered.connect(self._open_region_view)
        self.main_ui.action_characteristic_curves.triggered.connect(self._open_char_curves)
        self.main_ui.action_restore_default_layout.triggered.connect(self.restore_default_state)
        self.main_ui.action_dump_log.triggered.connect(self._dump_log)
        self.main_ui.action_generate_input_file.triggered.connect(self._generate_input_file)
        self.main_ui.id_search_pushButton.clicked.connect(self._id_search_function)

        self.main_ui.sidebar_tabWidget.currentChanged.connect(self.side_bar_function)
        self.main_ui.bottom_bar_tabWidget.currentChanged.connect(self.bottom_bar_function)
        self.main_ui.id_spinBox.valueChanged.connect(self._lock_id_search_buttons)

    def log_append_message(self, text_message):
        time_stamp = time.strftime("%Y-%m-%d %H:%M")
        self.main_ui.log_textEdit.append(time_stamp + ': ' + str(text_message))

    @QtCore.Slot()
    def side_bar_function(self, index):
        if index == 1:
            self._open_ipython_shell()
        self.main_ui.sidebar_tabWidget.setCurrentIndex(0)

    @QtCore.Slot()
    def bottom_bar_function(self, index):
        if index == 1:
            self._open_close_log_window()
        if index == 2:
            self._open_id_search()
        self.main_ui.bottom_bar_tabWidget.setCurrentIndex(0)

    @QtCore.Slot()
    def _open_file(self):
        self.file_browser = QtGui.QFileDialog()
        self.file_location = self.file_browser.getOpenFileName(self, self.tr('Open Input File'), '',
                                                               self.tr('Input File (*.in)'))
        self.main_ui.statusbar.showMessage('Mounting input file...')
        self.log_append_message('Attempting to mount input file...')
        if len(self.file_location[0]) > 0:
            if self.controller.open_file(self.file_location[0]):
                self.log_append_message('Current mounted input file: ' + self.file_location[0])
                self.main_ui.statusbar.clearMessage()
            else:
                self.main_ui.statusbar.showMessage('Error mounting input file!', 4000)
                self.log_append_message('Error mounting input file!')

    def ipy_console(self):
        """
        Source: http://stackoverflow.com/questions/26666583/embedding-ipython-qtconsole-and-passing-objects
        :param parent: QtGui.QWidget
        :param shell_button: QtGui.QPushButton
        :return: dict
        """
        from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
        from IPython.qt.inprocess import QtInProcessKernelManager

        parent = self.main_ui.ipython_widget

        if parent.layout() is None:
            layout = QtGui.QVBoxLayout(parent)
        else:
            layout = parent.layout()
        widget = RichIPythonWidget(parent=parent)
        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel = kernel_manager.kernel
        kernel.gui = 'qt4'

        kernel_client = kernel_manager.client()
        kernel_client.start_channels()
        kernel_client.namespace = parent

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()

        layout.addWidget(widget)
        widget.kernel_manager = kernel_manager
        widget.kernel_client = kernel_client
        widget.exit_requested.connect(stop)
        ipython_widget = widget
        parent.setVisible(True)
        ipython_widget.show()

        kernel.shell.push({'widget': widget, 'kernel': kernel, 'parent': parent})
        return {'widget': widget, 'kernel': kernel}

    @QtCore.Slot()
    def _lock_id_search_buttons(self):
        self.main_ui.id_view_pushButton.setDisabled(True)
        self.main_ui.id_delete_pushButton.setDisabled(True)

    def open_instance(self, id_text, result_inst, delete=False):
        if delete:
            self.controller.delete_object(id_text)
            self._lock_id_search_buttons()
            self.main_ui.id_spinBox.clear()
            self.log_append_message('Object deleted: ' + id_text)
        else:
            self.sim_view.auto_fill(id_text)
            self.sim_view.show()
            self.sim_view.setDisabled(True)

    @QtCore.Slot()
    def _id_search_function(self):
        id_text = self.main_ui.id_spinBox.text()
        result = self.controller.id_search(id_text)

        if result is not None:
            self.main_ui.id_view_pushButton.setDisabled(False)
            self.main_ui.id_delete_pushButton.setDisabled(False)
            self.main_ui.id_view_pushButton.clicked.connect(lambda: self.open_instance(id_text, result))
            self.main_ui.id_delete_pushButton.clicked.connect(lambda: self.open_instance(id_text, result, True))
        else:
            self.main_ui.id_view_pushButton.setDisabled(True)
            self.main_ui.id_delete_pushButton.setDisabled(True)
            self.log_append_message(id_text + ' not found.')

    @QtCore.Slot()
    def _open_ipython_shell(self):
        if not self.__ipy_inst:
            try:
                # Error-handling to check if the IPython module exists
                self.ipy_console()
            except ImportError:
                self.log_append_message('IPython module missing!')
                return 0

            self.log_append_message('IPython console instance created.')
            self.__ipy_inst = True

        if self.main_ui.ipython_dockWidget.isHidden():
            self.main_ui.ipython_dockWidget.show()
        else:
            self.main_ui.ipython_dockWidget.hide()

    @QtCore.Slot()
    def _open_id_search(self):
        if self.main_ui.id_search_dockWidget.isHidden():
            self.main_ui.id_search_dockWidget.show()
        else:
            self.main_ui.id_search_dockWidget.hide()

    def _open_close_log_window(self):
        if self.main_ui.log_dockWidget.isHidden():
            self.main_ui.log_dockWidget.show()
        else:
            self.main_ui.log_dockWidget.hide()

    @QtCore.Slot()
    def _dump_log(self):
        full_filename_location = 'logs/' + time.strftime("%Y-%m-%d-%H-%M") + '-log.txt'
        f = open(full_filename_location, 'w')
        f.write(self.main_ui.log_textEdit.toPlainText())
        f.close()
        self.prompt = other_views.GenericWarningView('File saved to logs/')
        self.prompt.show()

    @QtCore.Slot()
    def _generate_input_file(self):
        self.controller.generate_input_file()

    @QtCore.Slot()
    def _remove_all_attributes(self):
        self.controller.remove_all_attributes()
        self.log_append_message('All attributes deleted.')

    @QtCore.Slot()
    def _open_simulation_view(self):
        self.sim_view = other_views.SimulationView(main_model, self.controller)
        self.sim_view.show()

    @QtCore.Slot()
    def _open_grid_view(self):
        self.grid_view = other_views.GridView(main_model, self.controller)
        self.grid_view.show()

    @QtCore.Slot()
    def _open_output_view(self):
        self.output_view = other_views.OutputView(main_model, self.controller)
        self.output_view.show()

    @QtCore.Slot()
    def _open_observation_view(self):
        self.observation_view = other_views.ObservationView(main_model, self.controller)
        self.observation_view.show()

    @QtCore.Slot()
    def _open_strata_view(self):
        self.strata_view = other_views.StratigraphyCouplerView(main_model, self.controller)
        self.strata_view.show()

    @QtCore.Slot()
    def _open_material_view(self):
        self.material_view = other_views.MaterialView(main_model, self.controller)
        self.material_view.show()

    @QtCore.Slot()
    def _open_initial_condition_view(self):
        self.init_condition_view = other_views.InitialConditionView(main_model, self.controller)
        self.init_condition_view.show()

    @QtCore.Slot()
    def _open_boundary_condition_view(self):
        self.boundary_condition_view = other_views.BoundaryConditionView(main_model, self.controller)
        self.boundary_condition_view.show()

    @QtCore.Slot()
    def _open_transport_condition_view(self):
        self.transport_condition_view = other_views.TransportConditionView(main_model, self.controller)
        self.transport_condition_view.show()

    @QtCore.Slot()
    def _open_source_sink_view(self):
        self.source_sink_view = other_views.SourceSinkView(main_model, self.controller)
        self.source_sink_view.show()

    @QtCore.Slot()
    def _open_newton_solver_view(self):
        self.newton_solver_view = other_views.NewtonSolverView(main_model, self.controller)
        self.newton_solver_view.show()

    @QtCore.Slot()
    def _open_linear_solver_view(self):
        self.linear_solver_view = other_views.LinearSolverView(main_model, self.controller)
        self.linear_solver_view.show()

    @QtCore.Slot()
    def _open_time_view(self):
        self.time_view = other_views.TimeView(main_model, self.controller)
        self.time_view.show()

    @QtCore.Slot()
    def _open_timestepper_view(self):
        self.timestepper_view = other_views.TimestepperView(main_model, self.controller)
        self.timestepper_view.show()

    @QtCore.Slot()
    def _open_checkpoint_view(self):
        self.checkpoint_view = other_views.CheckpointView(main_model, self.controller)
        self.checkpoint_view.show()

    @QtCore.Slot()
    def _open_restart_view(self):
        self.restart_view = other_views.RestartView(main_model, self.controller)
        self.restart_view.show()

    @QtCore.Slot()
    def _open_flow_view(self):
        self.flow_view = other_views.FlowView(main_model, self.controller)
        self.flow_view.show()

    @QtCore.Slot()
    def _open_constraints_condition_view(self):
        self.constraints_condition_view = other_views.ConstraintConditionView(main_model, self.controller)
        self.constraints_condition_view.show()

    @QtCore.Slot()
    def _open_saturation_function_view(self):
        self.saturation_function_view = other_views.SaturationFunctionView(main_model, self.controller)
        self.saturation_function_view.show()

    @QtCore.Slot()
    def _open_uniform_velocity_view(self):
        self.uniform_velocity_view = other_views.UniformVelocityView(main_model, self.controller)
        self.uniform_velocity_view.show()

    @QtCore.Slot()
    def _open_nonuniform_velocity_view(self):
        filename_path = QtGui.QFileDialog.getOpenFileName(self, self.tr('Select file with nonuniform velocity',
                                                                        ''))
        self.controller.nonuniform_velocity(filename_path)

    @QtCore.Slot()
    def _open_fluid_properties(self):
        self.fluid_properties = other_views.FluidPropertiesView(main_model, self.controller)
        self.fluid_properties.show()

    @QtCore.Slot()
    def _open_chemistry_view(self):
        self.chemistry_view = other_views.ChemistryView(main_model, self.controller)
        self.chemistry_view.show()

    @QtCore.Slot()
    def _open_regression_view(self):
        self.regression_view = other_views.RegressionView(main_model, self.controller)
        self.regression_view.show()

    @QtCore.Slot()
    def _open_dataset_view(self):
        self.dataset_view = other_views.DatasetView(main_model, self.controller)
        self.dataset_view.show()

    @QtCore.Slot()
    def _open_plot_observation_view(self):
        self.plot_observation_view = other_views.PlotObservationView(main_model, self.controller)
        self.plot_observation_view.show()

    @QtCore.Slot()
    def _open_region_view(self):
        self.region_view = other_views.RegionView(main_model, self.controller)
        self.region_view.show()

    @QtCore.Slot()
    def _open_char_curves(self):
        self.char_view = other_views.CharacteristicCurvesView(main_model, self.controller)
        self.char_view.show()
