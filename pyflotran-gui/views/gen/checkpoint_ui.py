# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'checkpoint_ui.ui'
#
# Created: Mon Jun 22 14:23:37 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(320, 100)
        Dialog.setMinimumSize(QtCore.QSize(320, 100))
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.frequencyLabel = QtGui.QLabel(Dialog)
        self.frequencyLabel.setObjectName("frequencyLabel")
        self.gridLayout.addWidget(self.frequencyLabel, 0, 0, 1, 1)
        self.overwriteComboBox = QtGui.QComboBox(Dialog)
        self.overwriteComboBox.setObjectName("overwriteComboBox")
        self.overwriteComboBox.addItem("")
        self.overwriteComboBox.addItem("")
        self.gridLayout.addWidget(self.overwriteComboBox, 1, 1, 1, 1)
        self.overwriteLabel = QtGui.QLabel(Dialog)
        self.overwriteLabel.setObjectName("overwriteLabel")
        self.gridLayout.addWidget(self.overwriteLabel, 1, 0, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 2, 1, 1, 1)
        self.frequency_spinBox = QtGui.QSpinBox(Dialog)
        self.frequency_spinBox.setObjectName("frequency_spinBox")
        self.gridLayout.addWidget(self.frequency_spinBox, 0, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Checkpoint", None, QtGui.QApplication.UnicodeUTF8))
        self.frequencyLabel.setText(QtGui.QApplication.translate("Dialog", "Frequency", None, QtGui.QApplication.UnicodeUTF8))
        self.overwriteComboBox.setItemText(0, QtGui.QApplication.translate("Dialog", "False", None, QtGui.QApplication.UnicodeUTF8))
        self.overwriteComboBox.setItemText(1, QtGui.QApplication.translate("Dialog", "True", None, QtGui.QApplication.UnicodeUTF8))
        self.overwriteLabel.setText(QtGui.QApplication.translate("Dialog", "Overwrite", None, QtGui.QApplication.UnicodeUTF8))

