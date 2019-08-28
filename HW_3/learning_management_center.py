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

import pickle

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--hiddenlayers', dest='hiddenLayers', metavar='N', type=int, nargs='+',
                    help='amount of neurons in hidden layers')
parser.add_argument('--rays', dest='rays', type=int, 
                    help='amount of ladar ray', default = 21)
parser.add_argument("-s", "--steps", type=int)
parser.add_argument("-f", "--filename", type=str)
parser.add_argument("--seed", type=int, default = 13)

args = parser.parse_args()
print(args.steps, args.seed, args.filename, args.rays, args.hiddenLayers)

import zmq
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas)
    from PyQt5.QtCore import QSocketNotifier
    from PyQt5.QtWidgets import QDoubleSpinBox, QLabel, QPushButton
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas)
from matplotlib.figure import Figure

hiddenLayersList = args.hiddenLayers if args.hiddenLayers else [60, 60, 60] #[65, 45, 25, 10] #[35, 5]
hiddenLayers = list(map(str, hiddenLayersList))
print(args.hiddenLayers)
print(hiddenLayers)

def startRunCar():
    pass
    popenargs = [sys.executable, 'run_car.py', 
                                        '-f', 'network_config_agent_0_layers_25_25_60_60_60_1.txt',
                                        '--seed', str(args.seed), 
                                        '--rays', str(args.rays)
                                        ]
    if (len(hiddenLayersList ) > 0):
        popenargs.extend(['--hiddenlayers'])
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

        self._zmq_sock2 = self._zmq_context.socket(zmq.SUB)
        self._zmq_sock2.connect("tcp://localhost:5557")
        self._zmq_sock2.setsockopt(zmq.SUBSCRIBE, b'progress')
        self.read_noti2 = QSocketNotifier(self._zmq_sock2.getsockopt(zmq.FD),
                                             QSocketNotifier.Read,
                                             self)
        self.read_noti2.activated.connect(self.on_read_msg2)

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        vlayout = QtWidgets.QVBoxLayout(self._main)
        vlayout.setObjectName("verticalLayout")
        
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setObjectName("horizontalLayoutWithHeatMaps")

        self.heat_map_canvas = []
        #self.addToolBar(NavigationToolbar(static_canvas, self))
        
        #self.layout.addWidget(cost_canvas)
        #self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(cost_canvas, self))

        layout2 = QtWidgets.QHBoxLayout()
        layout2.setObjectName("horizontalLayoutWithErrorGraphAndControls")

        layout3 = QtWidgets.QHBoxLayout()
        layout3.setObjectName("horizontalLayoutWithErrorGraph")

        #static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        cost_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        eval_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout3.addWidget(cost_canvas)        
        layout3.addWidget(eval_canvas)
    
        self.learninRate = QDoubleSpinBox()
        self.learninRate.setValue(0.05)
        self.learninRate.setSingleStep(0.005)
        self.learninRate.setDecimals(3)
        
        self.l1 = QDoubleSpinBox()
        self.l1.setValue(0.05)
        self.l1.setSingleStep(0.005)
        self.l1.setDecimals(3)

        self.l2 = QDoubleSpinBox()
        self.l2.setValue(0.05)        
        self.l2.setSingleStep(0.005)
        self.l2.setDecimals(3)

        self.sendButton = QPushButton()
        self.sendButton.setText("Send")
        self.sendButton.clicked.connect(self.sendHiperParameters)

        layout4 = QtWidgets.QVBoxLayout()
        layout4.setObjectName("vertiacalWithControls")
        layout4.addWidget(QLabel("Learning Rate"))
        layout4.addWidget(self.learninRate)        
        layout4.addWidget(QLabel("L1"))
        layout4.addWidget(self.l1)        
        layout4.addWidget(QLabel("L2"))
        layout4.addWidget(self.l2)        
        layout4.addWidget(self.sendButton)

        layout2.addLayout(layout3)
        layout2.addLayout(layout4)

        vlayout.addLayout(self.layout)
        vlayout.addLayout(layout2)
        
        # seaborn !!!
        # sns.swarmplot(x="species", y="petal_length", data=iris, ax=self._seaborn_heatmap_ax)
        # seaborn !!!

        # self._static_ax = static_canvas.figure.subplots()
        # t = np.linspace(0, 10, 501)
        # self._static_ax.plot(t, np.tan(t), ".")

        self._cost_ax = cost_canvas.figure.subplots()
        self._eval_ax = eval_canvas.figure.subplots()
        # self._timer = cost_canvas.new_timer(
        #     100, [(self._update_canvas, (), {})])
        # self._timer.start()
        self.cost = np.array([])
        self.eval = np.array([])

        QtCore.QTimer.singleShot(1000, self.OnLoad)

    def sendHiperParameters(self):
        print(self.learninRate.value(), self.l1.value(), self.l2.value())

    def OnLoad(self):
        startRunCar()

    def updateProgress(self, cost, eval):
        self.cost = np.append(self.cost, [cost])
        self.eval = np.append(self.eval, [eval])

        self._cost_ax.clear()
        t = np.linspace(0, 10, 101)
        # Shift the sinusoid as a function of time.
        self._cost_ax.plot(self.cost)
        self._cost_ax.figure.canvas.draw()

        self._eval_ax.clear()
        t = np.linspace(0, 10, 101)
        # Shift the sinusoid as a function of time.
        self._eval_ax.plot(self.eval)
        self._eval_ax.figure.canvas.draw()


    def update_heatmaps(self, weights, biases):
        if (len(self.heat_map_canvas) == 0):
            self._seaborn_heatmap_ax = []
            for i in range(len(weights)+1):
                figsize = (5, 3) if i < len(weights) else (5, 1)
                self.heat_map_canvas.append(FigureCanvas(Figure(figsize=figsize)))
                self.layout.addWidget(self.heat_map_canvas[-1])            
                self._seaborn_heatmap_ax.append(self.heat_map_canvas[-1].figure.subplots())

        print('update weights')

        weights = np.abs(weights)
        vmin = 0
        vmax = max([np.amax(x) for x in weights])
        
        for (w, i) in  zip(weights, range(len(weights))):        
            a, b = w.shape
            self._seaborn_heatmap_ax[i].clear()            

            if (i < len(weights) - 1):
                sns.heatmap(w, ax=self._seaborn_heatmap_ax[i], vmin = vmin, vmax = vmax, cbar = False)
            else:
                # Обновление еще и colormap
                self._seaborn_heatmap_ax[-1].clear()
                sns.heatmap(w, ax=self._seaborn_heatmap_ax[i], cbar_ax=self._seaborn_heatmap_ax[-1], vmin = vmin, vmax = vmax, cbar = True)
                self._seaborn_heatmap_ax[-1].figure.canvas.draw()

            self._seaborn_heatmap_ax[i].figure.canvas.draw()


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

    def on_read_msg2(self):
        self.read_noti2.setEnabled(False)

        if self._zmq_sock2.getsockopt(zmq.EVENTS) & zmq.POLLIN:
            while self._zmq_sock2.getsockopt(zmq.EVENTS) & zmq.POLLIN:
                topic = self._zmq_sock2.recv_string()
                data = self._zmq_sock2.recv_pyobj()                
                print(topic)
                cost, eval = data
                self.updateProgress(cost, eval)
        elif self._zmq_sock2.getsockopt(zmq.EVENTS) & zmq.POLLOUT:
            print("[Socket] zmq.POLLOUT")
        elif self._zmq_sock2.getsockopt(zmq.EVENTS) & zmq.POLLERR:
            print("[Socket] zmq.POLLERR")

        self.read_noti2.setEnabled(True)


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()

# 
# todo: добавить кнопку по которой будет происзодить обновление heatmap

