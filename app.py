import sys
import json

import numpy as np

from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPolygonF
from PyQt5.QtGui import QPen, QBrush, QColor

from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QGraphicsPolygonItem, QGraphicsLineItem, QGraphicsRectItem, QGraphicsEllipseItem
from PyQt5.QtWidgets import QApplication


class GraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super(GraphicsScene, self).__init__(parent=parent)

        with open("canvas.json", 'r') as f:
            json_content = json.load(f)

        for obstacle in json_content["shapes"]:
            polygon = QGraphicsPolygonItem()
            polygonF = QPolygonF()
            for px, py in obstacle["points"]:
                polygonF.append(QPointF(px, py))
            polygon.setPolygon(polygonF)
            polygon.setPen(QPen(QColor(0, 0, 255)))
            polygon.setBrush(QBrush(QColor(0, 0, 0, 128)))
            self.addItem(polygon)

        x_interval = np.linspace(0.0, 1024.0, 40)
        y_interval = np.linspace(0.0, 1024.0, 40)

        # draw horizontal line
        for y_coor in y_interval:
            lineItem = QGraphicsLineItem()
            lineItem.setPen(QPen(QColor(0, 0, 0), 2))
            lineItem.setLine(x_interval[0], y_coor, x_interval[-1], y_coor)
            self.addItem(lineItem)

        # draw vertical line
        for x_coor in x_interval:
            lineItem = QGraphicsLineItem()
            lineItem.setPen(QPen(QColor(0, 0, 0), 2))
            lineItem.setLine(x_coor, y_interval[0], x_coor, y_interval[-1])
            self.addItem(lineItem)

        x_dot_coor = (x_interval[1:] + x_interval[:-1]) / 2.0
        y_dot_coor = (y_interval[1:] + y_interval[:-1]) / 2.0
        self.dots = np.meshgrid(x_dot_coor, y_dot_coor)
        self.num_x_coor = len(x_dot_coor)
        self.num_y_coor = len(y_dot_coor)

        # for grid collision detection
        self.blockItem = QGraphicsRectItem()
        self.blockWidth = x_interval[1] - x_interval[0]
        self.blockHeight = y_interval[1] - y_interval[0]
        self.blockItem.setRect(0, 0, self.blockWidth, self.blockHeight)
        self.addItem(self.blockItem)

    def checkCollision(self, x, y):
        self.blockItem.setX(x)
        self.blockItem.setY(y)
        if len(self.blockItem.collidingItems()) > 4:
            return False
        else:
            return True

    def runGridCollision(self, counter):
        i = counter // self.num_x_coor
        j = counter % self.num_x_coor

        if i >= self.num_y_coor:
            self.removeItem(self.blockItem)
            return True

        radius = 3
        diameter = 6
        x = self.dots[0][i, j]
        y = self.dots[1][i, j]
        isValid = self.checkCollision(x - self.blockWidth / 2.0, y - self.blockHeight / 2.0)
        dotItem = QGraphicsEllipseItem()
        if isValid:
            dotItem.setPen(QPen(QColor(0, 255, 0)))
            dotItem.setBrush(QBrush(QColor(0, 255, 0, 255)))
        else:
            dotItem.setPen(QPen(QColor(255, 0, 0)))
            dotItem.setBrush(QBrush(QColor(255, 0, 0, 255)))
        dotItem.setPos(QPointF(0, 0))
        dotItem.setRect(x - radius, y - radius, diameter, diameter)
        self.addItem(dotItem)

        return False


class CentralWidget(QWidget):
    def __init__(self):
        super(CentralWidget, self).__init__()
        self.counter = 0

        self.setLayout(QHBoxLayout())

        self.scene = GraphicsScene()
        self.view = QGraphicsView(self)
        self.view.setScene(self.scene)

        self.layout().addWidget(self.view)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.checkGridCollision)
        self.timer.setInterval(0.0)

    def checkGridCollision(self):
        isFinished = self.scene.runGridCollision(self.counter)
        if isFinished:
            self.counter = 0
            self.timer.stop()
        else:
            self.counter += 1

    def startDetection(self):
        self.timer.start()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.view = CentralWidget()
        self.setCentralWidget(self.view)
        self.view.startDetection()
        self.statusBar().showMessage("Ready!")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainwindow = MainWindow()
    mainwindow.setWindowTitle("Simple Planner Animation")
    mainwindow.show()
    app.exec_()
