"""
===============
Embedding in Qt
===============

Simple Qt application embedding Matplotlib canvases.  This program will work
equally well using Qt4 and Qt5.  Either version of Qt can be selected (for
example) by setting the ``MPLBACKEND`` environment variable to "Qt4Agg" or
"Qt5Agg", or by first importing the desired version of PyQt.
"""

import sys
import time

import numpy as np
import seaborn as sns

import subprocess

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--hiddenlayers', dest='hiddenLayers', metavar='N', type=int, nargs='+',
                    help='amount of neurons in hidden layers')
parser.add_argument('--rays', dest='rays', type=int, 
                    help='amount of ladar ray', default = 5)
parser.add_argument("-s", "--steps", type=int)
parser.add_argument("-f", "--filename", type=str)
parser.add_argument("--seed", type=int, default = 23)

args = parser.parse_args()
print(args.steps, args.seed, args.filename, args.rays, args.hiddenLayers)

import zmq
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas)
    from PyQt5.QtCore import QSocketNotifier
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas)
from matplotlib.figure import Figure

hiddenLayersList = args.hiddenLayers if args.hiddenLayers else [30, 20]
hiddenLayers = list(map(str, hiddenLayersList))
print(args.hiddenLayers)
print(hiddenLayers)

def startRunCar():
    popenargs = [sys.executable, 'run_car.py', 
                                        '--seed', str(args.seed), 
                                        '--rays', str(args.rays),
                                        '--hiddenlayers']
    popenargs.extend(hiddenLayers)                                        
    subprocess.Popen(popenargs)

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # zmq
        self._zmq_context = zmq.Context()
        self._zmq_sock = self._zmq_context.socket(zmq.SUB)
        self._zmq_sock.connect("tcp://localhost:5555")
        self._zmq_sock.setsockopt(zmq.SUBSCRIBE, b'weights')
        self._zmq_sock.setsockopt(zmq.SUBSCRIBE, b'biases')        
        #self._zmq_sock.setsockopt(zmq.SUBSCRIBE, b"bm_chat")        
        self.read_noti = QSocketNotifier(self._zmq_sock.getsockopt(zmq.FD),
                                             QSocketNotifier.Read,
                                             self)
        self.read_noti.activated.connect(self.on_read_msg)

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        vlayout = QtWidgets.QVBoxLayout(self._main)
        vlayout.setObjectName("verticalLayout")
        
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setObjectName("horizontalLayoutWithHeatMaps")

        self.heat_map_canvas = []
        #self.addToolBar(NavigationToolbar(static_canvas, self))

        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.layout.addWidget(dynamic_canvas)
        #self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(dynamic_canvas, self))

        layout2 = QtWidgets.QHBoxLayout()
        layout2.setObjectName("horizontalLayoutWithErrorGraph")

        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout2.addWidget(static_canvas)

        vlayout.addLayout(self.layout)
        vlayout.addLayout(layout2)
        
        # seaborn !!!
        # sns.swarmplot(x="species", y="petal_length", data=iris, ax=self._seaborn_heatmap_ax)
        # seaborn !!!

        self._static_ax = static_canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        self._static_ax.plot(t, np.tan(t), ".")

        self._dynamic_ax = dynamic_canvas.figure.subplots()
        self._timer = dynamic_canvas.new_timer(
            100, [(self._update_canvas, (), {})])
        self._timer.start()

        QtCore.QTimer.singleShot(1000, self.OnLoad)

    def OnLoad(self):
        startRunCar()

    def _update_canvas(self):
        self._dynamic_ax.clear()
        t = np.linspace(0, 10, 101)
        # Shift the sinusoid as a function of time.
        self._dynamic_ax.plot(t, np.sin(t + time.time()))
        self._dynamic_ax.figure.canvas.draw()

    def update_heatmaps(self, weights, biases):
        if (len(self.heat_map_canvas) == 0):
            self._seaborn_heatmap_ax = []
            for w in weights:
                self.heat_map_canvas.append(FigureCanvas(Figure(figsize=(5, 3))))
                self.layout.addWidget(self.heat_map_canvas[-1])
            
                self._seaborn_heatmap_ax.append(self.heat_map_canvas[-1].figure.subplots())
                #uniform_data = np.random.rand(10, 12)
                sns.heatmap(w, ax=self._seaborn_heatmap_ax[-1])


    def on_read_msg(self):
        self.read_noti.setEnabled(False)

        if self._zmq_sock.getsockopt(zmq.EVENTS) & zmq.POLLIN:
            while self._zmq_sock.getsockopt(zmq.EVENTS) & zmq.POLLIN:
                topic = self._zmq_sock.recv_string()
                data = self._zmq_sock.recv_pyobj()                
                print(topic)
                weights, biases = data
                self.update_heatmaps(weights, biases)
        elif self._zmq_sock.getsockopt(zmq.EVENTS) & zmq.POLLOUT:
            print("[Socket] zmq.POLLOUT")
        elif self._zmq_sock.getsockopt(zmq.EVENTS) & zmq.POLLERR:
            print("[Socket] zmq.POLLERR")

        self.read_noti.setEnabled(True)



if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()

# 
# todo: добавить кнопку по которой будет происзодить обновление heatmap

