# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'constraint_condition_ui.ui'
#
# Created: Mon Jun 22 15:18:58 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(489, 324)
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtGui.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.name_lineEdit = QtGui.QLineEdit(Dialog)
        self.name_lineEdit.setObjectName("name_lineEdit")
        self.gridLayout.addWidget(self.name_lineEdit, 0, 1, 1, 1)
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.add_concentration_pushButton = QtGui.QPushButton(Dialog)
        self.add_concentration_pushButton.setObjectName("add_concentration_pushButton")
        self.gridLayout.addWidget(self.add_concentration_pushButton, 1, 2, 1, 1)
        self.remove_concentration_pushButton = QtGui.QPushButton(Dialog)
        self.remove_concentration_pushButton.setObjectName("remove_concentration_pushButton")
        self.gridLayout.addWidget(self.remove_concentration_pushButton, 2, 2, 1, 1)
        self.concentration_listWidget = QtGui.QListWidget(Dialog)
        self.concentration_listWidget.setAlternatingRowColors(True)
        self.concentration_listWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.concentration_listWidget.setObjectName("concentration_listWidget")
        self.gridLayout.addWidget(self.concentration_listWidget, 1, 1, 2, 1)
        self.mineral_listWidget = QtGui.QListWidget(Dialog)
        self.mineral_listWidget.setAlternatingRowColors(True)
        self.mineral_listWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.mineral_listWidget.setObjectName("mineral_listWidget")
        self.gridLayout.addWidget(self.mineral_listWidget, 3, 1, 2, 1)
        self.remove_mineral_pushButton = QtGui.QPushButton(Dialog)
        self.remove_mineral_pushButton.setObjectName("remove_mineral_pushButton")
        self.gridLayout.addWidget(self.remove_mineral_pushButton, 4, 2, 1, 1)
        self.add_mineral_pushButton = QtGui.QPushButton(Dialog)
        self.add_mineral_pushButton.setObjectName("add_mineral_pushButton")
        self.gridLayout.addWidget(self.add_mineral_pushButton, 3, 2, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 5, 1, 1, 2)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Constraint Condition", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Concentration List", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Dialog", "Mineral List", None, QtGui.QApplication.UnicodeUTF8))
        self.add_concentration_pushButton.setText(QtGui.QApplication.translate("Dialog", "Add", None, QtGui.QApplication.UnicodeUTF8))
        self.remove_concentration_pushButton.setText(QtGui.QApplication.translate("Dialog", "Remove", None, QtGui.QApplication.UnicodeUTF8))
        self.remove_mineral_pushButton.setText(QtGui.QApplication.translate("Dialog", "Remove", None, QtGui.QApplication.UnicodeUTF8))
        self.add_mineral_pushButton.setText(QtGui.QApplication.translate("Dialog", "Add", None, QtGui.QApplication.UnicodeUTF8))

