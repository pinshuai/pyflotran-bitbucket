# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'timestepper_ui.ui'
#
# Created: Thu Jul 09 13:43:34 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(308, 366)
        Form.setMinimumSize(QtCore.QSize(0, 366))
        Form.setMaximumSize(QtCore.QSize(16777215, 366))
        self.formLayout = QtGui.QFormLayout(Form)
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName("formLayout")
        self.tSModeLabel = QtGui.QLabel(Form)
        self.tSModeLabel.setObjectName("tSModeLabel")
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.tSModeLabel)
        self.ts_mode_comboBox = QtGui.QComboBox(Form)
        self.ts_mode_comboBox.setObjectName("ts_mode_comboBox")
        self.ts_mode_comboBox.addItem("")
        self.ts_mode_comboBox.addItem("")
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.ts_mode_comboBox)
        self.label = QtGui.QLabel(Form)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label)
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_2)
        self.label_3 = QtGui.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.label_3)
        self.label_4 = QtGui.QLabel(Form)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(5, QtGui.QFormLayout.LabelRole, self.label_4)
        self.label_5 = QtGui.QLabel(Form)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(6, QtGui.QFormLayout.LabelRole, self.label_5)
        self.init_steady_state_comboBox = QtGui.QComboBox(Form)
        self.init_steady_state_comboBox.setObjectName("init_steady_state_comboBox")
        self.init_steady_state_comboBox.addItem("")
        self.init_steady_state_comboBox.addItem("")
        self.formLayout.setWidget(6, QtGui.QFormLayout.FieldRole, self.init_steady_state_comboBox)
        self.label_6 = QtGui.QLabel(Form)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(7, QtGui.QFormLayout.LabelRole, self.label_6)
        self.run_as_steady_state_comboBox = QtGui.QComboBox(Form)
        self.run_as_steady_state_comboBox.setObjectName("run_as_steady_state_comboBox")
        self.run_as_steady_state_comboBox.addItem("")
        self.run_as_steady_state_comboBox.addItem("")
        self.formLayout.setWidget(7, QtGui.QFormLayout.FieldRole, self.run_as_steady_state_comboBox)
        self.label_7 = QtGui.QLabel(Form)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(8, QtGui.QFormLayout.LabelRole, self.label_7)
        self.label_8 = QtGui.QLabel(Form)
        self.label_8.setWordWrap(True)
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(9, QtGui.QFormLayout.LabelRole, self.label_8)
        self.label_9 = QtGui.QLabel(Form)
        self.label_9.setWordWrap(True)
        self.label_9.setObjectName("label_9")
        self.formLayout.setWidget(10, QtGui.QFormLayout.LabelRole, self.label_9)
        self.label_10 = QtGui.QLabel(Form)
        self.label_10.setObjectName("label_10")
        self.formLayout.setWidget(11, QtGui.QFormLayout.LabelRole, self.label_10)
        self.buttonBox = QtGui.QDialogButtonBox(Form)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.formLayout.setWidget(12, QtGui.QFormLayout.FieldRole, self.buttonBox)
        self.ts_acceleration_spinBox = QtGui.QSpinBox(Form)
        self.ts_acceleration_spinBox.setMaximum(999999)
        self.ts_acceleration_spinBox.setObjectName("ts_acceleration_spinBox")
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.ts_acceleration_spinBox)
        self.num_steps_after_cut_spinBox = QtGui.QSpinBox(Form)
        self.num_steps_after_cut_spinBox.setMaximum(999999)
        self.num_steps_after_cut_spinBox.setObjectName("num_steps_after_cut_spinBox")
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.num_steps_after_cut_spinBox)
        self.max_ts_cuts_spinBox = QtGui.QSpinBox(Form)
        self.max_ts_cuts_spinBox.setMaximum(999999)
        self.max_ts_cuts_spinBox.setObjectName("max_ts_cuts_spinBox")
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.max_ts_cuts_spinBox)
        self.cfl_limiter_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.cfl_limiter_dSpinBox.setMaximum(999999.0)
        self.cfl_limiter_dSpinBox.setObjectName("cfl_limiter_dSpinBox")
        self.formLayout.setWidget(5, QtGui.QFormLayout.FieldRole, self.cfl_limiter_dSpinBox)
        self.max_pressure_change_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.max_pressure_change_dSpinBox.setMaximum(999999.0)
        self.max_pressure_change_dSpinBox.setObjectName("max_pressure_change_dSpinBox")
        self.formLayout.setWidget(8, QtGui.QFormLayout.FieldRole, self.max_pressure_change_dSpinBox)
        self.max_temp_change_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.max_temp_change_dSpinBox.setMaximum(999999.0)
        self.max_temp_change_dSpinBox.setObjectName("max_temp_change_dSpinBox")
        self.formLayout.setWidget(9, QtGui.QFormLayout.FieldRole, self.max_temp_change_dSpinBox)
        self.max_concentration_change_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.max_concentration_change_dSpinBox.setMaximum(999999.0)
        self.max_concentration_change_dSpinBox.setObjectName("max_concentration_change_dSpinBox")
        self.formLayout.setWidget(10, QtGui.QFormLayout.FieldRole, self.max_concentration_change_dSpinBox)
        self.max_saturation_change_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.max_saturation_change_dSpinBox.setMaximum(999999.0)
        self.max_saturation_change_dSpinBox.setObjectName("max_saturation_change_dSpinBox")
        self.formLayout.setWidget(11, QtGui.QFormLayout.FieldRole, self.max_saturation_change_dSpinBox)
        self.label_11 = QtGui.QLabel(Form)
        self.label_11.setObjectName("label_11")
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_11)
        self.max_steps_spinBox = QtGui.QSpinBox(Form)
        self.max_steps_spinBox.setMinimum(-999999999)
        self.max_steps_spinBox.setMaximum(999999999)
        self.max_steps_spinBox.setProperty("value", 0)
        self.max_steps_spinBox.setObjectName("max_steps_spinBox")
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.max_steps_spinBox)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Timestepper", None, QtGui.QApplication.UnicodeUTF8))
        self.tSModeLabel.setText(QtGui.QApplication.translate("Form", "TS Mode", None, QtGui.QApplication.UnicodeUTF8))
        self.ts_mode_comboBox.setItemText(0, QtGui.QApplication.translate("Form", "Flow", None, QtGui.QApplication.UnicodeUTF8))
        self.ts_mode_comboBox.setItemText(1, QtGui.QApplication.translate("Form", "Tran", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Form", "TS Acceleration", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Form", "Nunber of Steps After Cut", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Form", "Max TS Cuts", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Form", "CFL Limiter", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("Form", "Initialize to Steady State", None, QtGui.QApplication.UnicodeUTF8))
        self.init_steady_state_comboBox.setItemText(0, QtGui.QApplication.translate("Form", "False", None, QtGui.QApplication.UnicodeUTF8))
        self.init_steady_state_comboBox.setItemText(1, QtGui.QApplication.translate("Form", "True", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("Form", "Run as Steady State", None, QtGui.QApplication.UnicodeUTF8))
        self.run_as_steady_state_comboBox.setItemText(0, QtGui.QApplication.translate("Form", "False", None, QtGui.QApplication.UnicodeUTF8))
        self.run_as_steady_state_comboBox.setItemText(1, QtGui.QApplication.translate("Form", "True", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("Form", "Max Pressure Change [Pa]", None, QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("Form", "Max Temperature Change [C]", None, QtGui.QApplication.UnicodeUTF8))
        self.label_9.setText(QtGui.QApplication.translate("Form", "Max Concentration Change [mol/L]", None, QtGui.QApplication.UnicodeUTF8))
        self.label_10.setText(QtGui.QApplication.translate("Form", "Max Saturation Change", None, QtGui.QApplication.UnicodeUTF8))
        self.label_11.setText(QtGui.QApplication.translate("Form", "Max Steps", None, QtGui.QApplication.UnicodeUTF8))
