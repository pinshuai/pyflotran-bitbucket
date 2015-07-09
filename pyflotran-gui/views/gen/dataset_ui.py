# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dataset_ui.ui'
#
# Created: Mon Jun 22 14:27:39 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 251)
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtGui.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.dataset_name_line_edit = QtGui.QLineEdit(Dialog)
        self.dataset_name_line_edit.setObjectName("dataset_name_line_edit")
        self.gridLayout.addWidget(self.dataset_name_line_edit, 0, 1, 1, 1)
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.dataset_mapped_name = QtGui.QLineEdit(Dialog)
        self.dataset_mapped_name.setObjectName("dataset_mapped_name")
        self.gridLayout.addWidget(self.dataset_mapped_name, 1, 1, 1, 1)
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.name_line_edit = QtGui.QLineEdit(Dialog)
        self.name_line_edit.setObjectName("name_line_edit")
        self.gridLayout.addWidget(self.name_line_edit, 2, 1, 1, 1)
        self.label_4 = QtGui.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.file_name_line_edit = QtGui.QLineEdit(Dialog)
        self.file_name_line_edit.setObjectName("file_name_line_edit")
        self.gridLayout.addWidget(self.file_name_line_edit, 3, 1, 1, 1)
        self.browse_button = QtGui.QToolButton(Dialog)
        self.browse_button.setObjectName("browse_button")
        self.gridLayout.addWidget(self.browse_button, 3, 2, 1, 1)
        self.label_5 = QtGui.QLabel(Dialog)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.hdf5_dataset_name = QtGui.QLineEdit(Dialog)
        self.hdf5_dataset_name.setObjectName("hdf5_dataset_name")
        self.gridLayout.addWidget(self.hdf5_dataset_name, 4, 1, 1, 1)
        self.label_6 = QtGui.QLabel(Dialog)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 5, 0, 1, 1)
        self.map_hdf5_dataset_name = QtGui.QLineEdit(Dialog)
        self.map_hdf5_dataset_name.setObjectName("map_hdf5_dataset_name")
        self.gridLayout.addWidget(self.map_hdf5_dataset_name, 5, 1, 1, 1)
        self.label_8 = QtGui.QLabel(Dialog)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 6, 0, 1, 1)
        self.max_buffer_size_spin_box = QtGui.QDoubleSpinBox(Dialog)
        self.max_buffer_size_spin_box.setObjectName("max_buffer_size_spin_box")
        self.gridLayout.addWidget(self.max_buffer_size_spin_box, 6, 1, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 8, 1, 1, 2)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 7, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Dataset", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "Dataset Name", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Dataset Mapped Name", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Dialog", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Dialog", "Filename", None, QtGui.QApplication.UnicodeUTF8))
        self.browse_button.setText(QtGui.QApplication.translate("Dialog", "...", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("Dialog", "HDF5 Dataset Name", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("Dialog", "Map HDF5 Dataset Name", None, QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("Dialog", "Max Buffer Size", None, QtGui.QApplication.UnicodeUTF8))

