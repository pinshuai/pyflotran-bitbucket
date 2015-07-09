# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'stratigraphy_coupler_ui.ui'
#
# Created: Mon Jun 29 08:47:35 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(342, 95)
        Dialog.setMaximumSize(QtCore.QSize(16777215, 95))
        self.formLayout = QtGui.QFormLayout(Dialog)
        self.formLayout.setObjectName("formLayout")
        self.nameLabel_3 = QtGui.QLabel(Dialog)
        self.nameLabel_3.setObjectName("nameLabel_3")
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.nameLabel_3)
        self.region_lineEdit = QtGui.QLineEdit(Dialog)
        self.region_lineEdit.setObjectName("region_lineEdit")
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.region_lineEdit)
        self.regionLabel_3 = QtGui.QLabel(Dialog)
        self.regionLabel_3.setObjectName("regionLabel_3")
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.regionLabel_3)
        self.material_lineEdit = QtGui.QLineEdit(Dialog)
        self.material_lineEdit.setObjectName("material_lineEdit")
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.material_lineEdit)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.buttonBox)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Stratigraphy Coupler", None, QtGui.QApplication.UnicodeUTF8))
        self.nameLabel_3.setText(QtGui.QApplication.translate("Dialog", "Region", None, QtGui.QApplication.UnicodeUTF8))
        self.regionLabel_3.setText(QtGui.QApplication.translate("Dialog", "Material", None, QtGui.QApplication.UnicodeUTF8))

