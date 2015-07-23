# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'newton_solver_ui.ui'
#
# Created: Wed Jul 08 09:21:31 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(296, 316)
        Form.setMaximumSize(QtCore.QSize(16777215, 16777212))
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_7 = QtGui.QLabel(Form)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 5, 0, 1, 1)
        self.label_8 = QtGui.QLabel(Form)
        self.label_8.setWordWrap(True)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 6, 0, 1, 1)
        self.name_comboBox = QtGui.QComboBox(Form)
        self.name_comboBox.setObjectName("name_comboBox")
        self.name_comboBox.addItem("")
        self.name_comboBox.addItem("")
        self.gridLayout.addWidget(self.name_comboBox, 0, 1, 1, 1)
        self.groupBox = QtGui.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtGui.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_3 = QtGui.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 1)
        self.label_4 = QtGui.QLabel(self.groupBox)
        self.label_4.setWordWrap(True)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 1, 0, 1, 1)
        self.rel_tolerance_lineEdit = QtGui.QLineEdit(self.groupBox)
        self.rel_tolerance_lineEdit.setObjectName("rel_tolerance_lineEdit")
        self.gridLayout_2.addWidget(self.rel_tolerance_lineEdit, 0, 1, 1, 1)
        self.rel_tolerance_update_lineEdit = QtGui.QLineEdit(self.groupBox)
        self.rel_tolerance_update_lineEdit.setObjectName("rel_tolerance_update_lineEdit")
        self.gridLayout_2.addWidget(self.rel_tolerance_update_lineEdit, 1, 1, 1, 1)
        self.gridLayout.addWidget(self.groupBox, 2, 0, 1, 2)
        self.buttonBox = QtGui.QDialogButtonBox(Form)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 7, 1, 1, 1)
        self.label = QtGui.QLabel(Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_5 = QtGui.QLabel(Form)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.label_6 = QtGui.QLabel(Form)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)
        self.max_function_eval_dSpinBox = QtGui.QSpinBox(Form)
        self.max_function_eval_dSpinBox.setMaximum(999999)
        self.max_function_eval_dSpinBox.setObjectName("max_function_eval_dSpinBox")
        self.gridLayout.addWidget(self.max_function_eval_dSpinBox, 6, 1, 1, 1)
        self.max_iterations_dSpinBox = QtGui.QSpinBox(Form)
        self.max_iterations_dSpinBox.setMaximum(999999)
        self.max_iterations_dSpinBox.setObjectName("max_iterations_dSpinBox")
        self.gridLayout.addWidget(self.max_iterations_dSpinBox, 5, 1, 1, 1)
        self.abs_tolerance_lineEdit = QtGui.QLineEdit(Form)
        self.abs_tolerance_lineEdit.setObjectName("abs_tolerance_lineEdit")
        self.gridLayout.addWidget(self.abs_tolerance_lineEdit, 1, 1, 1, 1)
        self.divergence_tol_lineEdit = QtGui.QLineEdit(Form)
        self.divergence_tol_lineEdit.setObjectName("divergence_tol_lineEdit")
        self.gridLayout.addWidget(self.divergence_tol_lineEdit, 3, 1, 1, 1)
        self.tol_compared_to_inf_norm_lineEdit = QtGui.QLineEdit(Form)
        self.tol_compared_to_inf_norm_lineEdit.setObjectName("tol_compared_to_inf_norm_lineEdit")
        self.gridLayout.addWidget(self.tol_compared_to_inf_norm_lineEdit, 4, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        Form.setTabOrder(self.name_comboBox, self.max_iterations_dSpinBox)
        Form.setTabOrder(self.max_iterations_dSpinBox, self.max_function_eval_dSpinBox)
        Form.setTabOrder(self.max_function_eval_dSpinBox, self.buttonBox)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Newton Solver", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Form", "Absolute Tolerance", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setToolTip(QtGui.QApplication.translate("Form", "<html><head/><body><p>Cuts time step if the number of iterations exceed this value.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("Form", "Max Iterations", None, QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("Form", "Max Function Evalutions", None, QtGui.QApplication.UnicodeUTF8))
        self.name_comboBox.setItemText(0, QtGui.QApplication.translate("Form", "FLOW", None, QtGui.QApplication.UnicodeUTF8))
        self.name_comboBox.setItemText(1, QtGui.QApplication.translate("Form", "TRANSPORT", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("Form", "With Respect to Previous Iteration", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Form", "Relative Tolerance", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Form", "Relative Tolerance of the Update ", None, QtGui.QApplication.UnicodeUTF8))
        self.rel_tolerance_lineEdit.setPlaceholderText(QtGui.QApplication.translate("Form", "Float", None, QtGui.QApplication.UnicodeUTF8))
        self.rel_tolerance_update_lineEdit.setPlaceholderText(QtGui.QApplication.translate("Form", "Float", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Form", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("Form", "Divergence Tolerance", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("Form", "Tolerance Compared to Infinity Norm", None, QtGui.QApplication.UnicodeUTF8))
        self.abs_tolerance_lineEdit.setPlaceholderText(QtGui.QApplication.translate("Form", "Float", None, QtGui.QApplication.UnicodeUTF8))
        self.divergence_tol_lineEdit.setPlaceholderText(QtGui.QApplication.translate("Form", "Float", None, QtGui.QApplication.UnicodeUTF8))
        self.tol_compared_to_inf_norm_lineEdit.setPlaceholderText(QtGui.QApplication.translate("Form", "Float", None, QtGui.QApplication.UnicodeUTF8))
