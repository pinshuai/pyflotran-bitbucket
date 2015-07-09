# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'simulation_ui.ui'
#
# Created: Thu Jul 02 08:21:48 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(316, 160)
        self.gridLayout_2 = QtGui.QGridLayout(Dialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.geomechanics_checkBox = QtGui.QCheckBox(Dialog)
        self.geomechanics_checkBox.setObjectName("geomechanics_checkBox")
        self.gridLayout_2.addWidget(self.geomechanics_checkBox, 3, 0, 1, 1)
        self.mode_label = QtGui.QLabel(Dialog)
        self.mode_label.setObjectName("mode_label")
        self.gridLayout_2.addWidget(self.mode_label, 4, 0, 1, 1)
        self.subsurface_flow_checkBox = QtGui.QCheckBox(Dialog)
        self.subsurface_flow_checkBox.setObjectName("subsurface_flow_checkBox")
        self.gridLayout_2.addWidget(self.subsurface_flow_checkBox, 2, 0, 1, 1)
        self.subsurface_trans_checkBox = QtGui.QCheckBox(Dialog)
        self.subsurface_trans_checkBox.setObjectName("subsurface_trans_checkBox")
        self.gridLayout_2.addWidget(self.subsurface_trans_checkBox, 1, 0, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_2.addWidget(self.buttonBox, 6, 1, 1, 1)
        self.mode_comboBox = QtGui.QComboBox(Dialog)
        self.mode_comboBox.setObjectName("mode_comboBox")
        self.mode_comboBox.addItem("")
        self.mode_comboBox.addItem("")
        self.mode_comboBox.addItem("")
        self.mode_comboBox.addItem("")
        self.mode_comboBox.addItem("")
        self.mode_comboBox.addItem("")
        self.mode_comboBox.addItem("")
        self.gridLayout_2.addWidget(self.mode_comboBox, 5, 0, 1, 2)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Simulation Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.geomechanics_checkBox.setText(QtGui.QApplication.translate("Dialog", "Geomechanics", None, QtGui.QApplication.UnicodeUTF8))
        self.mode_label.setText(QtGui.QApplication.translate("Dialog", "Mode", None, QtGui.QApplication.UnicodeUTF8))
        self.subsurface_flow_checkBox.setText(QtGui.QApplication.translate("Dialog", "Subsurface Flow", None, QtGui.QApplication.UnicodeUTF8))
        self.subsurface_trans_checkBox.setText(QtGui.QApplication.translate("Dialog", "Subsurface Transport", None, QtGui.QApplication.UnicodeUTF8))
        self.mode_comboBox.setItemText(0, QtGui.QApplication.translate("Dialog", "richards", None, QtGui.QApplication.UnicodeUTF8))
        self.mode_comboBox.setItemText(1, QtGui.QApplication.translate("Dialog", "mphase", None, QtGui.QApplication.UnicodeUTF8))
        self.mode_comboBox.setItemText(2, QtGui.QApplication.translate("Dialog", "mph", None, QtGui.QApplication.UnicodeUTF8))
        self.mode_comboBox.setItemText(3, QtGui.QApplication.translate("Dialog", "flash2", None, QtGui.QApplication.UnicodeUTF8))
        self.mode_comboBox.setItemText(4, QtGui.QApplication.translate("Dialog", "th no_freezing", None, QtGui.QApplication.UnicodeUTF8))
        self.mode_comboBox.setItemText(5, QtGui.QApplication.translate("Dialog", "th freezing", None, QtGui.QApplication.UnicodeUTF8))
        self.mode_comboBox.setItemText(6, QtGui.QApplication.translate("Dialog", "immis", None, QtGui.QApplication.UnicodeUTF8))

