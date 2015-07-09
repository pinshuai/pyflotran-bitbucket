# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'constraint_condition_concentration_ui.ui'
#
# Created: Wed Jul 08 14:56:41 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(330, 220)
        Dialog.setMinimumSize(QtCore.QSize(330, 220))
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label_5 = QtGui.QLabel(Dialog)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 2, 2, 1, 1)
        self.constraint_comboBox = QtGui.QComboBox(Dialog)
        self.constraint_comboBox.setObjectName("constraint_comboBox")
        self.constraint_comboBox.addItem("")
        self.constraint_comboBox.addItem("")
        self.constraint_comboBox.addItem("")
        self.constraint_comboBox.addItem("")
        self.constraint_comboBox.addItem("")
        self.constraint_comboBox.addItem("")
        self.constraint_comboBox.addItem("")
        self.constraint_comboBox.addItem("")
        self.constraint_comboBox.addItem("")
        self.constraint_comboBox.addItem("")
        self.constraint_comboBox.addItem("")
        self.gridLayout.addWidget(self.constraint_comboBox, 2, 1, 1, 1)
        self.label_4 = QtGui.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.species_lineEdit = QtGui.QLineEdit(Dialog)
        self.species_lineEdit.setObjectName("species_lineEdit")
        self.gridLayout.addWidget(self.species_lineEdit, 0, 1, 1, 2)
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.element_lineEdit = QtGui.QLineEdit(Dialog)
        self.element_lineEdit.setObjectName("element_lineEdit")
        self.gridLayout.addWidget(self.element_lineEdit, 3, 1, 1, 2)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok|QtGui.QDialogButtonBox.Reset)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 4, 0, 1, 3)
        self.value_dSpinBox = QtGui.QDoubleSpinBox(Dialog)
        self.value_dSpinBox.setDecimals(20)
        self.value_dSpinBox.setMinimum(-999999999.0)
        self.value_dSpinBox.setMaximum(999999999.0)
        self.value_dSpinBox.setObjectName("value_dSpinBox")
        self.gridLayout.addWidget(self.value_dSpinBox, 1, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Constraint Condition: Concentration", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("Dialog", "Element", None, QtGui.QApplication.UnicodeUTF8))
        self.constraint_comboBox.setItemText(0, QtGui.QApplication.translate("Dialog", "None", None, QtGui.QApplication.UnicodeUTF8))
        self.constraint_comboBox.setItemText(1, QtGui.QApplication.translate("Dialog", "FREE", None, QtGui.QApplication.UnicodeUTF8))
        self.constraint_comboBox.setItemText(2, QtGui.QApplication.translate("Dialog", "TOTAL", None, QtGui.QApplication.UnicodeUTF8))
        self.constraint_comboBox.setItemText(3, QtGui.QApplication.translate("Dialog", "TOTAL_SORB", None, QtGui.QApplication.UnicodeUTF8))
        self.constraint_comboBox.setItemText(4, QtGui.QApplication.translate("Dialog", "pH", None, QtGui.QApplication.UnicodeUTF8))
        self.constraint_comboBox.setItemText(5, QtGui.QApplication.translate("Dialog", "LOG", None, QtGui.QApplication.UnicodeUTF8))
        self.constraint_comboBox.setItemText(6, QtGui.QApplication.translate("Dialog", "MINERAL", None, QtGui.QApplication.UnicodeUTF8))
        self.constraint_comboBox.setItemText(7, QtGui.QApplication.translate("Dialog", "GAS", None, QtGui.QApplication.UnicodeUTF8))
        self.constraint_comboBox.setItemText(8, QtGui.QApplication.translate("Dialog", "SC", None, QtGui.QApplication.UnicodeUTF8))
        self.constraint_comboBox.setItemText(9, QtGui.QApplication.translate("Dialog", "CONSTRAINT SUPERCRIT CO2", None, QtGui.QApplication.UnicodeUTF8))
        self.constraint_comboBox.setItemText(10, QtGui.QApplication.translate("Dialog", "Z", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Dialog", "Constraint", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Species", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Dialog", "Value", None, QtGui.QApplication.UnicodeUTF8))

