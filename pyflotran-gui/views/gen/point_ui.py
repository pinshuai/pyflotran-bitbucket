# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'point_ui.ui'
#
# Created: Mon Jun 22 11:32:42 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(303, 151)
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.coordinate_y_dSpinBox = QtGui.QDoubleSpinBox(Dialog)
        self.coordinate_y_dSpinBox.setDecimals(4)
        self.coordinate_y_dSpinBox.setMaximum(999999.0)
        self.coordinate_y_dSpinBox.setObjectName("coordinate_y_dSpinBox")
        self.gridLayout.addWidget(self.coordinate_y_dSpinBox, 2, 1, 1, 1)
        self.coordinate_x_dSpinBox = QtGui.QDoubleSpinBox(Dialog)
        self.coordinate_x_dSpinBox.setDecimals(4)
        self.coordinate_x_dSpinBox.setMaximum(999999.0)
        self.coordinate_x_dSpinBox.setObjectName("coordinate_x_dSpinBox")
        self.gridLayout.addWidget(self.coordinate_x_dSpinBox, 1, 1, 1, 1)
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label = QtGui.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.coordinate_z_dSpinBox = QtGui.QDoubleSpinBox(Dialog)
        self.coordinate_z_dSpinBox.setDecimals(4)
        self.coordinate_z_dSpinBox.setMaximum(999999.0)
        self.coordinate_z_dSpinBox.setObjectName("coordinate_z_dSpinBox")
        self.gridLayout.addWidget(self.coordinate_z_dSpinBox, 3, 1, 1, 1)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 2, 1, 1)
        self.name_lineEdit = QtGui.QLineEdit(Dialog)
        self.name_lineEdit.setObjectName("name_lineEdit")
        self.gridLayout.addWidget(self.name_lineEdit, 0, 1, 1, 2)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 4, 1, 1, 2)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Point", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Coordinate", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "Name", None, QtGui.QApplication.UnicodeUTF8))

