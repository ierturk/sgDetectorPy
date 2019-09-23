import sys

from PySide2.QtWidgets import QApplication
from PySide2 import QtCore
from mainwindow import MainWindow


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    main_window = MainWindow('mainwindow.ui')
    main_window.captureThreadRight.start()
    main_window.captureThreadLeft.start()

    main_window.captureThreadRight.start()
    main_window.captureThreadLeft.start()

    sys.exit(app.exec_())
