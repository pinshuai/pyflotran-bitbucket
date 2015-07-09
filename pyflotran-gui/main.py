from PySide import QtGui
from views.main_view import MainView
from controllers import main_controller
from model import main_model
import sys

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ui = MainView(main_model, main_controller)
    ui.show()
    app.exec_()
