# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'initial_condition_ui.ui'
#
# Created: Mon Jun 29 14:38:50 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(320, 166)
        Dialog.setMinimumSize(QtCore.QSize(320, 166))
        Dialog.setMaximumSize(QtCore.QSize(16777215, 166))
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.transport_lineEdit = QtGui.QLineEdit(Dialog)
        self.transport_lineEdit.setObjectName("transport_lineEdit")
        self.gridLayout.addWidget(self.transport_lineEdit, 2, 1, 1, 1)
        self.flow_lineEdit = QtGui.QLineEdit(Dialog)
        self.flow_lineEdit.setObjectName("flow_lineEdit")
        self.gridLayout.addWidget(self.flow_lineEdit, 1, 1, 1, 1)
        self.transportLabel = QtGui.QLabel(Dialog)
        self.transportLabel.setObjectName("transportLabel")
        self.gridLayout.addWidget(self.transportLabel, 2, 0, 1, 1)
        self.regionLabel = QtGui.QLabel(Dialog)
        self.regionLabel.setObjectName("regionLabel")
        self.gridLayout.addWidget(self.regionLabel, 3, 0, 1, 1)
        self.flowLabel = QtGui.QLabel(Dialog)
        self.flowLabel.setObjectName("flowLabel")
        self.gridLayout.addWidget(self.flowLabel, 1, 0, 1, 1)
        self.region_lineEdit = QtGui.QLineEdit(Dialog)
        self.region_lineEdit.setObjectName("region_lineEdit")
        self.gridLayout.addWidget(self.region_lineEdit, 3, 1, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setMaximumSize(QtCore.QSize(16777215, 25))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 4, 1, 1, 1)
        self.label = QtGui.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.name_lineEdit = QtGui.QLineEdit(Dialog)
        self.name_lineEdit.setObjectName("name_lineEdit")
        self.gridLayout.addWidget(self.name_lineEdit, 0, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.name_lineEdit, self.flow_lineEdit)
        Dialog.setTabOrder(self.flow_lineEdit, self.transport_lineEdit)
        Dialog.setTabOrder(self.transport_lineEdit, self.region_lineEdit)
        Dialog.setTabOrder(self.region_lineEdit, self.buttonBox)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Initial Condition", None, QtGui.QApplication.UnicodeUTF8))
        self.transportLabel.setText(QtGui.QApplication.translate("Dialog", "Transport", None, QtGui.QApplication.UnicodeUTF8))
        self.regionLabel.setText(QtGui.QApplication.translate("Dialog", "Region", None, QtGui.QApplication.UnicodeUTF8))
        self.flowLabel.setText(QtGui.QApplication.translate("Dialog", "Flow", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "Name", None, QtGui.QApplication.UnicodeUTF8))

