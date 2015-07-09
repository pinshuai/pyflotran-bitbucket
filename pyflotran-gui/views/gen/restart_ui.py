# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'restart_ui.ui'
#
# Created: Thu Jul 02 12:22:07 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(385, 147)
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.browse_tool_button = QtGui.QToolButton(Dialog)
        self.browse_tool_button.setObjectName("browse_tool_button")
        self.gridLayout.addWidget(self.browse_tool_button, 0, 2, 1, 1)
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_4 = QtGui.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)
        self.time_value_unit_combo_box = QtGui.QComboBox(Dialog)
        self.time_value_unit_combo_box.setBaseSize(QtCore.QSize(20, 10))
        self.time_value_unit_combo_box.setObjectName("time_value_unit_combo_box")
        self.time_value_unit_combo_box.addItem("")
        self.time_value_unit_combo_box.addItem("")
        self.time_value_unit_combo_box.addItem("")
        self.time_value_unit_combo_box.addItem("")
        self.time_value_unit_combo_box.addItem("")
        self.time_value_unit_combo_box.addItem("")
        self.time_value_unit_combo_box.addItem("")
        self.gridLayout.addWidget(self.time_value_unit_combo_box, 2, 1, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 3, 1, 1, 1)
        self.filename_line_edit = QtGui.QLineEdit(Dialog)
        self.filename_line_edit.setObjectName("filename_line_edit")
        self.gridLayout.addWidget(self.filename_line_edit, 0, 1, 1, 1)
        self.time_value_dSpinBox = QtGui.QDoubleSpinBox(Dialog)
        self.time_value_dSpinBox.setObjectName("time_value_dSpinBox")
        self.gridLayout.addWidget(self.time_value_dSpinBox, 1, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Restart", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Filename", None, QtGui.QApplication.UnicodeUTF8))
        self.browse_tool_button.setText(QtGui.QApplication.translate("Dialog", "...", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Dialog", "Time Value", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Dialog", "Time Unit", None, QtGui.QApplication.UnicodeUTF8))
        self.time_value_unit_combo_box.setItemText(0, QtGui.QApplication.translate("Dialog", "sec", None, QtGui.QApplication.UnicodeUTF8))
        self.time_value_unit_combo_box.setItemText(1, QtGui.QApplication.translate("Dialog", "min", None, QtGui.QApplication.UnicodeUTF8))
        self.time_value_unit_combo_box.setItemText(2, QtGui.QApplication.translate("Dialog", "hr", None, QtGui.QApplication.UnicodeUTF8))
        self.time_value_unit_combo_box.setItemText(3, QtGui.QApplication.translate("Dialog", "day", None, QtGui.QApplication.UnicodeUTF8))
        self.time_value_unit_combo_box.setItemText(4, QtGui.QApplication.translate("Dialog", "week", None, QtGui.QApplication.UnicodeUTF8))
        self.time_value_unit_combo_box.setItemText(5, QtGui.QApplication.translate("Dialog", "month", None, QtGui.QApplication.UnicodeUTF8))
        self.time_value_unit_combo_box.setItemText(6, QtGui.QApplication.translate("Dialog", "y", None, QtGui.QApplication.UnicodeUTF8))

