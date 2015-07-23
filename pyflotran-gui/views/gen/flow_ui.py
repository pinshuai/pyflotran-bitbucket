# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'flow_ui.ui'
#
# Created: Fri Jun 26 09:38:21 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(499, 470)
        Form.setMinimumSize(QtCore.QSize(438, 389))
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 9, 2, 1, 1)
        self.variable_list_listWidget = QtGui.QListWidget(Form)
        self.variable_list_listWidget.setAlternatingRowColors(True)
        self.variable_list_listWidget.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        self.variable_list_listWidget.setProperty("isWrapping", True)
        self.variable_list_listWidget.setWordWrap(True)
        self.variable_list_listWidget.setObjectName("variable_list_listWidget")
        self.gridLayout.addWidget(self.variable_list_listWidget, 6, 1, 4, 1)
        self.label = QtGui.QLabel(Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.name_lineEdit = QtGui.QLineEdit(Form)
        self.name_lineEdit.setObjectName("name_lineEdit")
        self.gridLayout.addWidget(self.name_lineEdit, 0, 1, 1, 2)
        self.unit_list_label = QtGui.QLabel(Form)
        self.unit_list_label.setObjectName("unit_list_label")
        self.gridLayout.addWidget(self.unit_list_label, 1, 0, 1, 1)
        self.unit_list_lineEdit = QtGui.QLineEdit(Form)
        self.unit_list_lineEdit.setReadOnly(True)
        self.unit_list_lineEdit.setObjectName("unit_list_lineEdit")
        self.gridLayout.addWidget(self.unit_list_lineEdit, 1, 1, 1, 1)
        self.label_3 = QtGui.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_4 = QtGui.QLabel(Form)
        self.label_4.setWordWrap(True)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.sync_ts_with_update_comboBox = QtGui.QComboBox(Form)
        self.sync_ts_with_update_comboBox.setObjectName("sync_ts_with_update_comboBox")
        self.sync_ts_with_update_comboBox.addItem("")
        self.sync_ts_with_update_comboBox.addItem("")
        self.gridLayout.addWidget(self.sync_ts_with_update_comboBox, 3, 1, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 3, 2, 1, 1)
        self.label_8 = QtGui.QLabel(Form)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 6, 0, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(Form)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 10, 2, 1, 1)
        self.add_pushButton = QtGui.QPushButton(Form)
        self.add_pushButton.setObjectName("add_pushButton")
        self.gridLayout.addWidget(self.add_pushButton, 6, 2, 1, 1)
        self.iphase_spinBox = QtGui.QSpinBox(Form)
        self.iphase_spinBox.setMaximum(99999999)
        self.iphase_spinBox.setObjectName("iphase_spinBox")
        self.gridLayout.addWidget(self.iphase_spinBox, 2, 1, 1, 1)
        self.remove_pushButton = QtGui.QPushButton(Form)
        self.remove_pushButton.setObjectName("remove_pushButton")
        self.gridLayout.addWidget(self.remove_pushButton, 7, 2, 1, 1)
        self.datum_label = QtGui.QLabel(Form)
        self.datum_label.setObjectName("datum_label")
        self.gridLayout.addWidget(self.datum_label, 5, 0, 1, 1)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.datum_dz_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.datum_dz_dSpinBox.setObjectName("datum_dz_dSpinBox")
        self.gridLayout_2.addWidget(self.datum_dz_dSpinBox, 4, 3, 1, 1)
        self.file_radioButton = QtGui.QRadioButton(Form)
        self.file_radioButton.setObjectName("file_radioButton")
        self.gridLayout_2.addWidget(self.file_radioButton, 0, 3, 1, 1)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem2, 0, 4, 1, 1)
        self.datum_dy_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.datum_dy_dSpinBox.setObjectName("datum_dy_dSpinBox")
        self.gridLayout_2.addWidget(self.datum_dy_dSpinBox, 4, 2, 1, 1)
        self.list_radioButton = QtGui.QRadioButton(Form)
        self.list_radioButton.setObjectName("list_radioButton")
        self.gridLayout_2.addWidget(self.list_radioButton, 0, 2, 1, 1)
        self.browse_button = QtGui.QToolButton(Form)
        self.browse_button.setObjectName("browse_button")
        self.gridLayout_2.addWidget(self.browse_button, 1, 5, 1, 1)
        self.label_7 = QtGui.QLabel(Form)
        self.label_7.setWordWrap(False)
        self.label_7.setMargin(10)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 0, 0, 1, 1)
        self.datum_dx_dSpinBox = QtGui.QDoubleSpinBox(Form)
        self.datum_dx_dSpinBox.setObjectName("datum_dx_dSpinBox")
        self.gridLayout_2.addWidget(self.datum_dx_dSpinBox, 4, 1, 1, 1)
        self.datum_lineEdit = QtGui.QLineEdit(Form)
        self.datum_lineEdit.setObjectName("datum_lineEdit")
        self.gridLayout_2.addWidget(self.datum_lineEdit, 1, 1, 1, 4)
        self.gridLayout.addLayout(self.gridLayout_2, 4, 1, 2, 2)

        self.retranslateUi(Form)
        QtCore.QObject.connect(self.list_radioButton, QtCore.SIGNAL("clicked()"), self.browse_button.hide)
        QtCore.QObject.connect(self.file_radioButton, QtCore.SIGNAL("clicked()"), self.browse_button.show)
        QtCore.QObject.connect(self.list_radioButton, QtCore.SIGNAL("clicked()"), self.datum_lineEdit.hide)
        QtCore.QObject.connect(self.file_radioButton, QtCore.SIGNAL("clicked()"), self.datum_dz_dSpinBox.hide)
        QtCore.QObject.connect(self.file_radioButton, QtCore.SIGNAL("clicked()"), self.datum_dy_dSpinBox.hide)
        QtCore.QObject.connect(self.file_radioButton, QtCore.SIGNAL("clicked(bool)"), self.datum_dx_dSpinBox.hide)
        QtCore.QObject.connect(self.list_radioButton, QtCore.SIGNAL("clicked()"), self.datum_dx_dSpinBox.show)
        QtCore.QObject.connect(self.list_radioButton, QtCore.SIGNAL("clicked()"), self.datum_dy_dSpinBox.show)
        QtCore.QObject.connect(self.list_radioButton, QtCore.SIGNAL("clicked(bool)"), self.datum_dz_dSpinBox.show)
        QtCore.QObject.connect(self.file_radioButton, QtCore.SIGNAL("clicked()"), self.datum_lineEdit.show)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Flow", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Form", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.unit_list_label.setToolTip(QtGui.QApplication.translate("Form", "<html><head/><body><p>Not currently supported.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.unit_list_label.setText(QtGui.QApplication.translate("Form", "Unit List", None, QtGui.QApplication.UnicodeUTF8))
        self.unit_list_lineEdit.setToolTip(QtGui.QApplication.translate("Form", "<html><head/><body><p>Not currently supported.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.unit_list_lineEdit.setText(QtGui.QApplication.translate("Form", "None", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Form", "IPhase", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Form", "Sync Timestep with Update", None, QtGui.QApplication.UnicodeUTF8))
        self.sync_ts_with_update_comboBox.setItemText(0, QtGui.QApplication.translate("Form", "False", None, QtGui.QApplication.UnicodeUTF8))
        self.sync_ts_with_update_comboBox.setItemText(1, QtGui.QApplication.translate("Form", "True", None, QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("Form", "Variable List", None, QtGui.QApplication.UnicodeUTF8))
        self.add_pushButton.setText(QtGui.QApplication.translate("Form", "Add", None, QtGui.QApplication.UnicodeUTF8))
        self.remove_pushButton.setText(QtGui.QApplication.translate("Form", "Remove", None, QtGui.QApplication.UnicodeUTF8))
        self.datum_label.setText(QtGui.QApplication.translate("Form", "Datum", None, QtGui.QApplication.UnicodeUTF8))
        self.file_radioButton.setText(QtGui.QApplication.translate("Form", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.list_radioButton.setText(QtGui.QApplication.translate("Form", "List", None, QtGui.QApplication.UnicodeUTF8))
        self.browse_button.setText(QtGui.QApplication.translate("Form", "...", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("Form", "Data Unit Type", None, QtGui.QApplication.UnicodeUTF8))
        self.datum_lineEdit.setPlaceholderText(QtGui.QApplication.translate("Form", "Filepath", None, QtGui.QApplication.UnicodeUTF8))
