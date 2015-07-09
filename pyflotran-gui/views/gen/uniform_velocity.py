# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uniform_velocity.ui'
#
# Created: Thu Jun 25 14:28:26 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(245, 182)
        Dialog.setMaximumSize(QtCore.QSize(16777215, 182))
        self.formLayout = QtGui.QFormLayout(Dialog)
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName("formLayout")
        self.groupBox = QtGui.QGroupBox(Dialog)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtGui.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.unit_lineEdit = QtGui.QLineEdit(self.groupBox)
        self.unit_lineEdit.setObjectName("unit_lineEdit")
        self.gridLayout.addWidget(self.unit_lineEdit, 3, 1, 1, 1)
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtGui.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_3 = QtGui.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_4 = QtGui.QLabel(self.groupBox)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.vlx_dSpinBox = QtGui.QDoubleSpinBox(self.groupBox)
        self.vlx_dSpinBox.setDecimals(4)
        self.vlx_dSpinBox.setMaximum(999999.0)
        self.vlx_dSpinBox.setObjectName("vlx_dSpinBox")
        self.gridLayout.addWidget(self.vlx_dSpinBox, 0, 1, 1, 1)
        self.vlz_dSpinBox = QtGui.QDoubleSpinBox(self.groupBox)
        self.vlz_dSpinBox.setDecimals(4)
        self.vlz_dSpinBox.setMaximum(999999.0)
        self.vlz_dSpinBox.setObjectName("vlz_dSpinBox")
        self.gridLayout.addWidget(self.vlz_dSpinBox, 2, 1, 1, 1)
        self.vly_dSpinBox = QtGui.QDoubleSpinBox(self.groupBox)
        self.vly_dSpinBox.setDecimals(4)
        self.vly_dSpinBox.setMaximum(999999.0)
        self.vly_dSpinBox.setObjectName("vly_dSpinBox")
        self.gridLayout.addWidget(self.vly_dSpinBox, 1, 1, 1, 1)
        self.formLayout.setWidget(0, QtGui.QFormLayout.SpanningRole, self.groupBox)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.buttonBox)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Uniform Velocity", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("Dialog", "Value List", None, QtGui.QApplication.UnicodeUTF8))
        self.unit_lineEdit.setPlaceholderText(QtGui.QApplication.translate("Dialog", "e.g. m/yr", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "vlx [m/s]", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "vly [m/s]", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Dialog", "vlz [m/s]", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Dialog", "Unit", None, QtGui.QApplication.UnicodeUTF8))

