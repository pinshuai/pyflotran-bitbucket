# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'constraint_condition_mineral_ui.ui'
#
# Created: Wed Jul 08 10:23:00 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(330, 160)
        Dialog.setMinimumSize(QtCore.QSize(330, 160))
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_5 = QtGui.QLabel(Dialog)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 1)
        self.volume_fraction_dSpinBox = QtGui.QDoubleSpinBox(Dialog)
        self.volume_fraction_dSpinBox.setDecimals(20)
        self.volume_fraction_dSpinBox.setMinimum(-999999999.0)
        self.volume_fraction_dSpinBox.setMaximum(999999999.0)
        self.volume_fraction_dSpinBox.setObjectName("volume_fraction_dSpinBox")
        self.gridLayout.addWidget(self.volume_fraction_dSpinBox, 1, 1, 1, 1)
        self.surface_area_dSpinBox = QtGui.QDoubleSpinBox(Dialog)
        self.surface_area_dSpinBox.setDecimals(20)
        self.surface_area_dSpinBox.setMinimum(-999999999.0)
        self.surface_area_dSpinBox.setMaximum(999999999.0)
        self.surface_area_dSpinBox.setObjectName("surface_area_dSpinBox")
        self.gridLayout.addWidget(self.surface_area_dSpinBox, 2, 1, 1, 1)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 2, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok|QtGui.QDialogButtonBox.Reset)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 3, 0, 1, 3)
        self.name_lineEdit = QtGui.QLineEdit(Dialog)
        self.name_lineEdit.setObjectName("name_lineEdit")
        self.gridLayout.addWidget(self.name_lineEdit, 0, 1, 1, 2)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Constraint Condition: Mineral", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Dialog", "Volume Fraction", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("Dialog", "Surface Area", None, QtGui.QApplication.UnicodeUTF8))

