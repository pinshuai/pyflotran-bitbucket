# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'linear_solver_ui.ui'
#
# Created: Wed Jul 08 10:14:03 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(263, 119)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtGui.QLabel(Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.name_comboBox = QtGui.QComboBox(Form)
        self.name_comboBox.setObjectName("name_comboBox")
        self.name_comboBox.addItem("")
        self.name_comboBox.addItem("")
        self.gridLayout.addWidget(self.name_comboBox, 0, 1, 1, 1)
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.solver_comboBox = QtGui.QComboBox(Form)
        self.solver_comboBox.setObjectName("solver_comboBox")
        self.solver_comboBox.addItem("")
        self.solver_comboBox.addItem("")
        self.solver_comboBox.addItem("")
        self.solver_comboBox.addItem("")
        self.gridLayout.addWidget(self.solver_comboBox, 1, 1, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(Form)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 3, 1, 1, 1)
        self.label_3 = QtGui.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.preconditioner_comboBox = QtGui.QComboBox(Form)
        self.preconditioner_comboBox.setObjectName("preconditioner_comboBox")
        self.preconditioner_comboBox.addItem("")
        self.preconditioner_comboBox.addItem("")
        self.gridLayout.addWidget(self.preconditioner_comboBox, 2, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        Form.setTabOrder(self.name_comboBox, self.solver_comboBox)
        Form.setTabOrder(self.solver_comboBox, self.preconditioner_comboBox)
        Form.setTabOrder(self.preconditioner_comboBox, self.buttonBox)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Linear Solver", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Form", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.name_comboBox.setItemText(0, QtGui.QApplication.translate("Form", "Transport", None, QtGui.QApplication.UnicodeUTF8))
        self.name_comboBox.setItemText(1, QtGui.QApplication.translate("Form", "Flow", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Form", "Solver", None, QtGui.QApplication.UnicodeUTF8))
        self.solver_comboBox.setItemText(0, QtGui.QApplication.translate("Form", "Solver", None, QtGui.QApplication.UnicodeUTF8))
        self.solver_comboBox.setItemText(1, QtGui.QApplication.translate("Form", "Krylov Type", None, QtGui.QApplication.UnicodeUTF8))
        self.solver_comboBox.setItemText(2, QtGui.QApplication.translate("Form", "Ksp Type", None, QtGui.QApplication.UnicodeUTF8))
        self.solver_comboBox.setItemText(3, QtGui.QApplication.translate("Form", "DirECT", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Form", "Preconditioner", None, QtGui.QApplication.UnicodeUTF8))
        self.preconditioner_comboBox.setItemText(0, QtGui.QApplication.translate("Form", "None", None, QtGui.QApplication.UnicodeUTF8))
        self.preconditioner_comboBox.setItemText(1, QtGui.QApplication.translate("Form", "ilu", None, QtGui.QApplication.UnicodeUTF8))

