# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'material_ui.ui'
#
# Created: Mon Jul 06 13:56:48 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(291, 460)
        Form.setMaximumSize(QtCore.QSize(16777215, 460))
        self.gridLayout_2 = QtGui.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.buttonBox = QtGui.QDialogButtonBox(Form)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_2.addWidget(self.buttonBox, 11, 1, 1, 2)
        self.groupBox = QtGui.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtGui.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.k_zzLabel = QtGui.QLabel(self.groupBox)
        self.k_zzLabel.setObjectName("k_zzLabel")
        self.gridLayout.addWidget(self.k_zzLabel, 2, 0, 1, 1)
        self.k_yyLabel = QtGui.QLabel(self.groupBox)
        self.k_yyLabel.setObjectName("k_yyLabel")
        self.gridLayout.addWidget(self.k_yyLabel, 1, 0, 1, 1)
        self.k_xxLabel = QtGui.QLabel(self.groupBox)
        self.k_xxLabel.setObjectName("k_xxLabel")
        self.gridLayout.addWidget(self.k_xxLabel, 0, 0, 1, 1)
        self.permeability_z_dSpinBox = QtGui.QDoubleSpinBox(self.groupBox)
        self.permeability_z_dSpinBox.setDecimals(20)
        self.permeability_z_dSpinBox.setMinimum(-999999999.0)
        self.permeability_z_dSpinBox.setMaximum(999999999.0)
        self.permeability_z_dSpinBox.setObjectName("permeability_z_dSpinBox")
        self.gridLayout.addWidget(self.permeability_z_dSpinBox, 2, 1, 1, 1)
        self.permeability_x_dSpinBox = QtGui.QDoubleSpinBox(self.groupBox)
        self.permeability_x_dSpinBox.setDecimals(20)
        self.permeability_x_dSpinBox.setMinimum(-999999999.0)
        self.permeability_x_dSpinBox.setMaximum(999999999.0)
        self.permeability_x_dSpinBox.setObjectName("permeability_x_dSpinBox")
        self.gridLayout.addWidget(self.permeability_x_dSpinBox, 0, 1, 1, 1)
        self.permeability_y_dSpinBox = QtGui.QDoubleSpinBox(self.groupBox)
        self.permeability_y_dSpinBox.setDecimals(20)
        self.permeability_y_dSpinBox.setMinimum(-999999999.0)
        self.permeability_y_dSpinBox.setMaximum(999999999.0)
        self.permeability_y_dSpinBox.setObjectName("permeability_y_dSpinBox")
        self.gridLayout.addWidget(self.permeability_y_dSpinBox, 1, 1, 1, 1)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 3, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox, 10, 0, 1, 3)
        self.iDLabel = QtGui.QLabel(Form)
        self.iDLabel.setObjectName("iDLabel")
        self.gridLayout_2.addWidget(self.iDLabel, 0, 0, 1, 1)
        self.nameLabel = QtGui.QLabel(Form)
        self.nameLabel.setObjectName("nameLabel")
        self.gridLayout_2.addWidget(self.nameLabel, 1, 0, 1, 1)
        self.name_lineEdit = QtGui.QLineEdit(Form)
        self.name_lineEdit.setObjectName("name_lineEdit")
        self.gridLayout_2.addWidget(self.name_lineEdit, 1, 2, 1, 1)
        self.characteristicCurvesLabel = QtGui.QLabel(Form)
        self.characteristicCurvesLabel.setObjectName("characteristicCurvesLabel")
        self.gridLayout_2.addWidget(self.characteristicCurvesLabel, 2, 0, 1, 2)
        self.char_curves_lineEdit = QtGui.QLineEdit(Form)
        self.char_curves_lineEdit.setObjectName("char_curves_lineEdit")
        self.gridLayout_2.addWidget(self.char_curves_lineEdit, 2, 2, 1, 1)
        self.porosityLabel = QtGui.QLabel(Form)
        self.porosityLabel.setObjectName("porosityLabel")
        self.gridLayout_2.addWidget(self.porosityLabel, 3, 0, 1, 1)
        self.tortuosityLabel = QtGui.QLabel(Form)
        self.tortuosityLabel.setObjectName("tortuosityLabel")
        self.gridLayout_2.addWidget(self.tortuosityLabel, 4, 0, 1, 1)
        self.densityLabel = QtGui.QLabel(Form)
        self.densityLabel.setWordWrap(True)
        self.densityLabel.setObjectName("densityLabel")
        self.gridLayout_2.addWidget(self.densityLabel, 5, 0, 1, 1)
        self.specificHeatLabel = QtGui.QLabel(Form)
        self.specificHeatLabel.setWordWrap(True)
        self.specificHeatLabel.setObjectName("specificHeatLabel")
        self.gridLayout_2.addWidget(self.specificHeatLabel, 6, 0, 1, 1)
        self.conditionDryLabel = QtGui.QLabel(Form)
        self.conditionDryLabel.setWordWrap(True)
        self.conditionDryLabel.setObjectName("conditionDryLabel")
        self.gridLayout_2.addWidget(self.conditionDryLabel, 7, 0, 1, 2)
        self.conductivityWetLabel = QtGui.QLabel(Form)
        self.conductivityWetLabel.setWordWrap(True)
        self.conductivityWetLabel.setObjectName("conductivityWetLabel")
        self.gridLayout_2.addWidget(self.conductivityWetLabel, 8, 0, 1, 2)
        self.saturationLabel = QtGui.QLabel(Form)
        self.saturationLabel.setObjectName("saturationLabel")
        self.gridLayout_2.addWidget(self.saturationLabel, 9, 0, 1, 1)
        self.saturation_lineEdit = QtGui.QLineEdit(Form)
        self.saturation_lineEdit.setObjectName("saturation_lineEdit")
        self.gridLayout_2.addWidget(self.saturation_lineEdit, 9, 2, 1, 1)
        self.id_spinBox = QtGui.QSpinBox(Form)
        self.id_spinBox.setMaximum(99999)
        self.id_spinBox.setObjectName("id_spinBox")
        self.gridLayout_2.addWidget(self.id_spinBox, 0, 2, 1, 1)
        self.porosity_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.porosity_dSpinBox.setMaximum(99999.0)
        self.porosity_dSpinBox.setObjectName("porosity_dSpinBox")
        self.gridLayout_2.addWidget(self.porosity_dSpinBox, 3, 2, 1, 1)
        self.rock_density_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.rock_density_dSpinBox.setMaximum(99999.0)
        self.rock_density_dSpinBox.setObjectName("rock_density_dSpinBox")
        self.gridLayout_2.addWidget(self.rock_density_dSpinBox, 5, 2, 1, 1)
        self.specific_heat_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.specific_heat_dSpinBox.setMaximum(99999.0)
        self.specific_heat_dSpinBox.setObjectName("specific_heat_dSpinBox")
        self.gridLayout_2.addWidget(self.specific_heat_dSpinBox, 6, 2, 1, 1)
        self.conductivity_dry_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.conductivity_dry_dSpinBox.setMaximum(99999.0)
        self.conductivity_dry_dSpinBox.setObjectName("conductivity_dry_dSpinBox")
        self.gridLayout_2.addWidget(self.conductivity_dry_dSpinBox, 7, 2, 1, 1)
        self.conductivity_wet_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.conductivity_wet_dSpinBox.setMaximum(99999.0)
        self.conductivity_wet_dSpinBox.setObjectName("conductivity_wet_dSpinBox")
        self.gridLayout_2.addWidget(self.conductivity_wet_dSpinBox, 8, 2, 1, 1)
        self.tortuosity_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.tortuosity_dSpinBox.setMaximum(99999.0)
        self.tortuosity_dSpinBox.setObjectName("tortuosity_dSpinBox")
        self.gridLayout_2.addWidget(self.tortuosity_dSpinBox, 4, 2, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Material Properties", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("Form", "Permeability", None, QtGui.QApplication.UnicodeUTF8))
        self.k_zzLabel.setText(QtGui.QApplication.translate("Form", "zz [m^2]", None, QtGui.QApplication.UnicodeUTF8))
        self.k_yyLabel.setText(QtGui.QApplication.translate("Form", "yy [m^2]", None, QtGui.QApplication.UnicodeUTF8))
        self.k_xxLabel.setText(QtGui.QApplication.translate("Form", "xx [m^2]", None, QtGui.QApplication.UnicodeUTF8))
        self.iDLabel.setText(QtGui.QApplication.translate("Form", "ID", None, QtGui.QApplication.UnicodeUTF8))
        self.nameLabel.setText(QtGui.QApplication.translate("Form", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.name_lineEdit.setPlaceholderText(QtGui.QApplication.translate("Form", "e.g. \'soil1\'", None, QtGui.QApplication.UnicodeUTF8))
        self.characteristicCurvesLabel.setText(QtGui.QApplication.translate("Form", "Characteristic Curves", None, QtGui.QApplication.UnicodeUTF8))
        self.porosityLabel.setText(QtGui.QApplication.translate("Form", "Porosity", None, QtGui.QApplication.UnicodeUTF8))
        self.tortuosityLabel.setText(QtGui.QApplication.translate("Form", "Tortuosity", None, QtGui.QApplication.UnicodeUTF8))
        self.densityLabel.setText(QtGui.QApplication.translate("Form", "Rock Density [kg/m^3]", None, QtGui.QApplication.UnicodeUTF8))
        self.specificHeatLabel.setText(QtGui.QApplication.translate("Form", "Specific Heat [W/m/K]", None, QtGui.QApplication.UnicodeUTF8))
        self.conditionDryLabel.setText(QtGui.QApplication.translate("Form", "Conductivity Dry [W/m/K]", None, QtGui.QApplication.UnicodeUTF8))
        self.conductivityWetLabel.setText(QtGui.QApplication.translate("Form", "Conductivity Wet [W/m/K", None, QtGui.QApplication.UnicodeUTF8))
        self.saturationLabel.setText(QtGui.QApplication.translate("Form", "Saturation", None, QtGui.QApplication.UnicodeUTF8))
        self.saturation_lineEdit.setPlaceholderText(QtGui.QApplication.translate("Form", "e.g. \'sf2\'", None, QtGui.QApplication.UnicodeUTF8))
