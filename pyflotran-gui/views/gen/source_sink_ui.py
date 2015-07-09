# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'source_sink_ui.ui'
#
# Created: Mon Jun 29 14:40:28 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(320, 148)
        Dialog.setMinimumSize(QtCore.QSize(320, 100))
        Dialog.setMaximumSize(QtCore.QSize(16777215, 150))
        self.formLayout = QtGui.QFormLayout(Dialog)
        self.formLayout.setObjectName("formLayout")
        self.nameLabel_2 = QtGui.QLabel(Dialog)
        self.nameLabel_2.setObjectName("nameLabel_2")
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.nameLabel_2)
        self.name_lineEdit = QtGui.QLineEdit(Dialog)
        self.name_lineEdit.setObjectName("name_lineEdit")
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.name_lineEdit)
        self.regionLabel_2 = QtGui.QLabel(Dialog)
        self.regionLabel_2.setObjectName("regionLabel_2")
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.regionLabel_2)
        self.region_lineEdit = QtGui.QLineEdit(Dialog)
        self.region_lineEdit.setObjectName("region_lineEdit")
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.region_lineEdit)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.buttonBox)
        self.label = QtGui.QLabel(Dialog)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label)
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_2)
        self.flow_lineEdit = QtGui.QLineEdit(Dialog)
        self.flow_lineEdit.setObjectName("flow_lineEdit")
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.flow_lineEdit)
        self.transport_lineEdit = QtGui.QLineEdit(Dialog)
        self.transport_lineEdit.setObjectName("transport_lineEdit")
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.transport_lineEdit)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.name_lineEdit, self.flow_lineEdit)
        Dialog.setTabOrder(self.flow_lineEdit, self.transport_lineEdit)
        Dialog.setTabOrder(self.transport_lineEdit, self.region_lineEdit)
        Dialog.setTabOrder(self.region_lineEdit, self.buttonBox)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Source Sink", None, QtGui.QApplication.UnicodeUTF8))
        self.nameLabel_2.setText(QtGui.QApplication.translate("Dialog", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.regionLabel_2.setText(QtGui.QApplication.translate("Dialog", "Region", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "Flow", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Transport", None, QtGui.QApplication.UnicodeUTF8))

