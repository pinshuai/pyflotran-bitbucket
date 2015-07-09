# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'boundary_condition_ui.ui'
#
# Created: Mon Jun 29 14:39:53 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(320, 150)
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.nameLabel = QtGui.QLabel(Dialog)
        self.nameLabel.setObjectName("nameLabel")
        self.gridLayout.addWidget(self.nameLabel, 0, 0, 1, 1)
        self.transportLineEdit = QtGui.QLineEdit(Dialog)
        self.transportLineEdit.setObjectName("transportLineEdit")
        self.gridLayout.addWidget(self.transportLineEdit, 2, 1, 1, 1)
        self.nameLineEdit = QtGui.QLineEdit(Dialog)
        self.nameLineEdit.setObjectName("nameLineEdit")
        self.gridLayout.addWidget(self.nameLineEdit, 0, 1, 1, 1)
        self.transportLabel = QtGui.QLabel(Dialog)
        self.transportLabel.setObjectName("transportLabel")
        self.gridLayout.addWidget(self.transportLabel, 2, 0, 1, 1)
        self.flowLabel = QtGui.QLabel(Dialog)
        self.flowLabel.setObjectName("flowLabel")
        self.gridLayout.addWidget(self.flowLabel, 1, 0, 1, 1)
        self.flowLineEdit = QtGui.QLineEdit(Dialog)
        self.flowLineEdit.setObjectName("flowLineEdit")
        self.gridLayout.addWidget(self.flowLineEdit, 1, 1, 1, 1)
        self.regionLabel = QtGui.QLabel(Dialog)
        self.regionLabel.setObjectName("regionLabel")
        self.gridLayout.addWidget(self.regionLabel, 3, 0, 1, 1)
        self.regionLineEdit = QtGui.QLineEdit(Dialog)
        self.regionLineEdit.setObjectName("regionLineEdit")
        self.gridLayout.addWidget(self.regionLineEdit, 3, 1, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 4, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.nameLineEdit, self.flowLineEdit)
        Dialog.setTabOrder(self.flowLineEdit, self.transportLineEdit)
        Dialog.setTabOrder(self.transportLineEdit, self.regionLineEdit)
        Dialog.setTabOrder(self.regionLineEdit, self.buttonBox)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Boundary Condition", None, QtGui.QApplication.UnicodeUTF8))
        self.nameLabel.setText(QtGui.QApplication.translate("Dialog", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.transportLabel.setText(QtGui.QApplication.translate("Dialog", "Transport", None, QtGui.QApplication.UnicodeUTF8))
        self.flowLabel.setText(QtGui.QApplication.translate("Dialog", "Flow", None, QtGui.QApplication.UnicodeUTF8))
        self.regionLabel.setText(QtGui.QApplication.translate("Dialog", "Region", None, QtGui.QApplication.UnicodeUTF8))

