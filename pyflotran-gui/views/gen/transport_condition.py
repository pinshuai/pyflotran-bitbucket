# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'transport_condition.ui'
#
# Created: Thu Jul 02 09:57:34 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(647, 306)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtGui.QDialogButtonBox(Form)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 4, 5, 1, 1)
        self.label_4 = QtGui.QLabel(Form)
        self.label_4.setWordWrap(True)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 4, 1, 1)
        self.push_down_tool_button = QtGui.QToolButton(Form)
        self.push_down_tool_button.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.push_down_tool_button.setArrowType(QtCore.Qt.DownArrow)
        self.push_down_tool_button.setObjectName("push_down_tool_button")
        self.gridLayout.addWidget(self.push_down_tool_button, 1, 6, 1, 1)
        self.label_3 = QtGui.QLabel(Form)
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 2, 1, 1)
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.name_lineEdit = QtGui.QLineEdit(Form)
        self.name_lineEdit.setObjectName("name_lineEdit")
        self.gridLayout.addWidget(self.name_lineEdit, 0, 1, 1, 2)
        self.type_comboBox = QtGui.QComboBox(Form)
        self.type_comboBox.setObjectName("type_comboBox")
        self.type_comboBox.addItem("")
        self.type_comboBox.addItem("")
        self.type_comboBox.addItem("")
        self.type_comboBox.addItem("")
        self.type_comboBox.addItem("")
        self.type_comboBox.addItem("")
        self.type_comboBox.addItem("")
        self.type_comboBox.addItem("")
        self.gridLayout.addWidget(self.type_comboBox, 1, 1, 1, 1)
        self.label = QtGui.QLabel(Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.constraint_list_type_lineEdit = QtGui.QLineEdit(Form)
        self.constraint_list_type_lineEdit.setObjectName("constraint_list_type_lineEdit")
        self.gridLayout.addWidget(self.constraint_list_type_lineEdit, 1, 5, 1, 1)
        self.remove_push_button = QtGui.QPushButton(Form)
        self.remove_push_button.setObjectName("remove_push_button")
        self.gridLayout.addWidget(self.remove_push_button, 4, 1, 1, 1)
        self.constraint_list_value_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.constraint_list_value_dSpinBox.setDecimals(4)
        self.constraint_list_value_dSpinBox.setMaximum(9999999.0)
        self.constraint_list_value_dSpinBox.setObjectName("constraint_list_value_dSpinBox")
        self.gridLayout.addWidget(self.constraint_list_value_dSpinBox, 1, 3, 1, 1)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 3, 1, 1, 5)
        self.listWidget = QtGui.QListWidget(Form)
        self.listWidget.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        self.listWidget.setObjectName("listWidget")
        self.gridLayout.addWidget(self.listWidget, 2, 3, 1, 3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Transport Condition", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Form", "Constraint List Type", None, QtGui.QApplication.UnicodeUTF8))
        self.push_down_tool_button.setText(QtGui.QApplication.translate("Form", "+", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Form", "Constraint List Value", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Form", "Type", None, QtGui.QApplication.UnicodeUTF8))
        self.type_comboBox.setItemText(0, QtGui.QApplication.translate("Form", "None", None, QtGui.QApplication.UnicodeUTF8))
        self.type_comboBox.setItemText(1, QtGui.QApplication.translate("Form", "Dirichlet", None, QtGui.QApplication.UnicodeUTF8))
        self.type_comboBox.setItemText(2, QtGui.QApplication.translate("Form", "Dirichlet Zero Gradient", None, QtGui.QApplication.UnicodeUTF8))
        self.type_comboBox.setItemText(3, QtGui.QApplication.translate("Form", "Equilibrium", None, QtGui.QApplication.UnicodeUTF8))
        self.type_comboBox.setItemText(4, QtGui.QApplication.translate("Form", "Neumann", None, QtGui.QApplication.UnicodeUTF8))
        self.type_comboBox.setItemText(5, QtGui.QApplication.translate("Form", "Mole", None, QtGui.QApplication.UnicodeUTF8))
        self.type_comboBox.setItemText(6, QtGui.QApplication.translate("Form", "Mole Rate", None, QtGui.QApplication.UnicodeUTF8))
        self.type_comboBox.setItemText(7, QtGui.QApplication.translate("Form", "Zero Gradient", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Form", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.remove_push_button.setText(QtGui.QApplication.translate("Form", "Remove", None, QtGui.QApplication.UnicodeUTF8))

