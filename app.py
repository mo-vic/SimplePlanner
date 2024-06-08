import sys
import json
from queue import Queue

import numpy as np

from PyQt5.QtCore import Qt
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

        self.start = (0, 0)
        self.goal = (0, 0)

        self.initFinished = False

        self.path = []

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
        self.grid = np.zeros((self.num_x_coor, self.num_y_coor), dtype=np.int)

        # for grid collision detection
        self.blockItem = QGraphicsRectItem()
        self.blockWidth = x_interval[1] - x_interval[0]
        self.blockHeight = y_interval[1] - y_interval[0]
        self.blockItem.setRect(0, 0, self.blockWidth, self.blockHeight)
        self.addItem(self.blockItem)

        radius = 3
        diameter = 6
        self.startItem = QGraphicsEllipseItem()
        self.startItem.setPen(QPen(QColor(0, 255, 0)))
        self.startItem.setBrush(QBrush(QColor(0, 255, 0, 255)))
        self.startItem.setPos(QPointF(0, 0))
        self.startItem.setRect(self.dots[0][0, 0] - radius, self.dots[1][0, 0] - radius, diameter, diameter)

        self.goalItem = QGraphicsEllipseItem()
        self.goalItem.setPen(QPen(QColor(255, 0, 0)))
        self.goalItem.setBrush(QBrush(QColor(255, 0, 0, 255)))
        self.goalItem.setPos(QPointF(0, 0))
        self.goalItem.setRect(self.dots[0][0, 0] - radius, self.dots[1][0, 0] - radius, diameter, diameter)

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
            self.addItem(self.startItem)
            self.addItem(self.goalItem)
            self.initFinished = True

            return True

        x = self.dots[0][i, j]
        y = self.dots[1][i, j]
        isValid = self.checkCollision(x - self.blockWidth / 2.0, y - self.blockHeight / 2.0)

        if isValid:
            self.grid[i, j] = 0
        else:
            self.grid[i, j] = 1

        return False

    def mouseDoubleClickEvent(self, event):
        radius = 3
        diameter = 6
        x = event.scenePos().x()
        y = event.scenePos().y()

        i = int(x // self.blockWidth)
        j = int(y // self.blockHeight)
        print(event.scenePos().x(), event.scenePos().y())
        if event.button() == Qt.LeftButton:
            self.start = (i, j)
            self.startItem.setRect(self.dots[0][j, i] - radius, self.dots[1][j, i] - radius, diameter, diameter)

        elif event.button() == Qt.RightButton:
            self.goal = (i, j)
            self.goalItem.setRect(self.dots[0][j, i] - radius, self.dots[1][j, i] - radius, diameter, diameter)

        print(self.start, self.goal)

        if self.initFinished:
            print(self.runBFS())

    def runBFS(self):
        for lineItem in self.path:
            self.removeItem(lineItem)

        self.path.clear()

        que = Queue()
        que.put(self.start)

        transition = dict()
        visited = np.zeros_like(self.grid)
        visited[self.start[0], self.start[1]] = 1
        while not que.empty():
            node = que.get()

            if node == self.goal:
                traceNode = self.goal
                while traceNode in transition:
                    preNode = transition[traceNode]
                    a, b = traceNode
                    i, j = preNode
                    lineItem = QGraphicsLineItem()
                    lineItem.setPen(QPen(QColor(0, 0, 255), 3))
                    lineItem.setLine(self.dots[0][b, a], self.dots[1][b, a], self.dots[0][j, i], self.dots[1][j, i])
                    self.path.append(lineItem)
                    self.addItem(lineItem)
                    traceNode = (i, j)

                return True

            a, b = node

            neighbors = [(a, b - 1), (a, b + 1), (a - 1, b), (a + 1, b),
                         (a - 1, b - 1), (a - 1, b + 1), (a + 1, b - 1),  (a + 1, b + 1)]

            for i, j in neighbors:
                if 0 <= i < self.num_x_coor and 0 <= j < self.num_y_coor:
                    if visited[i, j] == 1 or self.grid[j][i] == 1:
                        continue
                    else:
                        que.put((i, j))
                        visited[i, j] = 1
                        transition[(i, j)] = (a, b)

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
