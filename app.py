import sys
import json
import argparse
from collections import deque
from queue import Queue, PriorityQueue

import numpy as np
from matplotlib import cm
from scipy.spatial import KDTree

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
from PyQt5.QtWidgets import QGraphicsPolygonItem, QGraphicsLineItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem
from PyQt5.QtWidgets import QApplication


class GraphicsScene(QGraphicsScene):
    def __init__(self, parent=None, args=None):
        super(GraphicsScene, self).__init__(parent=parent)

        self.algorithm = args.algorithm
        self.radius = args.radius
        self.diameter = self.radius * 2.0

        self.start = (0, 0)
        self.goal = (0, 0)

        self.loadEnv()

        if args.algorithm in ["BFS", "DFS", "Greedy", "AStar"]:
            self.graphSearchAlgoInit(args)
        elif args.algorithm == "PRM":
            self.PRMAlgoInit(args)
        else:
            raise NotImplementedError

    def loadEnv(self):
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

    def graphSearchAlgoInit(self, args):
        self.path = []
        self.textItems = []
        self.colormapItems = []

        self.initFinished = False

        self.num_horizontal_grid = args.num_horizontal_grid
        self.num_vertical_grid = args.num_vertical_grid

        x_interval = np.linspace(0.0, 1024.0, self.num_horizontal_grid)
        y_interval = np.linspace(0.0, 1024.0, self.num_vertical_grid)

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
        self.grid = np.zeros((self.num_y_coor, self.num_x_coor), dtype=np.int)

        # for grid collision detection
        self.blockItem = QGraphicsRectItem()
        self.blockWidth = x_interval[1] - x_interval[0]
        self.blockHeight = y_interval[1] - y_interval[0]
        self.blockItem.setRect(0, 0, self.blockWidth, self.blockHeight)
        self.addItem(self.blockItem)

        self.startItem = QGraphicsEllipseItem()
        self.startItem.setPen(QPen(QColor(0, 255, 0)))
        self.startItem.setBrush(QBrush(QColor(0, 255, 0, 255)))
        self.startItem.setPos(QPointF(0, 0))
        self.startItem.setRect(self.dots[0][0, 0] - self.radius, self.dots[1][0, 0] - self.radius, self.diameter, self.diameter)

        self.goalItem = QGraphicsEllipseItem()
        self.goalItem.setPen(QPen(QColor(255, 0, 0)))
        self.goalItem.setBrush(QBrush(QColor(255, 0, 0, 255)))
        self.goalItem.setPos(QPointF(0, 0))
        self.goalItem.setRect(self.dots[0][0, 0] - self.radius, self.dots[1][0, 0] - self.radius, self.diameter, self.diameter)

    def samplingBasedAlgoCommonInit(self):
        self.workspace_width = 1024
        self.workspace_height = 1024

        self.path = []

        # draw horizontal border line
        for y_coor in [0, self.workspace_height]:
            borderLineItem = QGraphicsLineItem()
            borderLineItem.setPen(QPen(QColor(0, 0, 0), 2))
            borderLineItem.setLine(0, y_coor, self.workspace_width, y_coor)
            self.addItem(borderLineItem)

        # draw vertical border line
        for x_coor in [0, self.workspace_width]:
            borderLineItem = QGraphicsLineItem()
            borderLineItem.setPen(QPen(QColor(0, 0, 0), 2))
            borderLineItem.setLine(x_coor, 0, x_coor, self.workspace_height)
            self.addItem(borderLineItem)

        self.startItem = QGraphicsRectItem()
        self.startItem.setPen(QPen(QColor(0, 255, 0)))
        self.startItem.setBrush(QBrush(QColor(0, 255, 0, 255)))
        self.startItem.setPos(QPointF(0, 0))
        self.startItem.setRect(self.start[0] - self.radius, self.start[1] - self.radius, self.diameter, self.diameter)

        self.goalItem = QGraphicsRectItem()
        self.goalItem.setPen(QPen(QColor(255, 0, 0)))
        self.goalItem.setBrush(QBrush(QColor(255, 0, 0, 255)))
        self.goalItem.setPos(QPointF(0, 0))
        self.goalItem.setRect(self.goal[0] - self.radius, self.goal[1] - self.radius, self.diameter, self.diameter)

    def PRMAlgoInit(self, args):
        self.num_node = args.num_node  # number of nodes to put in the roadmap
        self.num_nearest = args.num_nearest  # number of closest neighbors to examine for each configuration

        self.initFinished = False
        self.sampleFinished = False
        self.constructionFinished = False

        self.sampled_items = []
        self.sampled_nodes = []
        self.edges = set()  # for checking if an edge already exists while constructing the graph
        self.graph = dict()

        # distance and indices return by KDTree querying
        self.dist = None
        self.indices = None

        self.kdTree = None

        self.samplingBasedAlgoCommonInit()

        self.startToRoadMapItem = QGraphicsLineItem()
        self.startToRoadMapItem.setPen(QPen(QColor(0, 255, 0), 3, Qt.DashLine))
        self.startToRoadMapItem.setLine(0, 0, 0, 0)
        self.startToRoadMapItem.setVisible(False)
        self.addItem(self.startToRoadMapItem)

        self.goalToRoadMapItem = QGraphicsLineItem()
        self.goalToRoadMapItem.setPen(QPen(QColor(255, 0, 0), 3, Qt.DashLine))
        self.goalToRoadMapItem.setLine(0, 0, 0, 0)
        self.goalToRoadMapItem.setVisible(False)
        self.addItem(self.goalToRoadMapItem)

        self.startSuccess = False
        self.goalSuccess = False

    def checkBlockCollision(self, x, y):
        self.blockItem.setX(x)
        self.blockItem.setY(y)
        if len(self.blockItem.collidingItems()) > 4:
            return False
        else:
            return True

    def runGridCollisionCheck(self, counter):
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
        isValid = self.checkBlockCollision(x - self.blockWidth / 2.0, y - self.blockHeight / 2.0)

        if isValid:
            self.grid[i, j] = 0
        else:
            self.grid[i, j] = 1

        return False

    def checkSampleCollision(self, vertexItem):
        self.addItem(vertexItem)
        if len(vertexItem.collidingItems()) > 0:
            return False
        else:
            return True

    def runSampling(self, counter):
        if counter == self.num_node:
            self.sampleFinished = True

            self.kdTree = KDTree(np.array(self.sampled_nodes))
            self.dist, self.indices = self.kdTree.query(np.array(self.sampled_nodes), k=self.num_nearest + 1)

            return True

        i = np.random.uniform(0, self.workspace_width)
        j = np.random.uniform(0, self.workspace_height)

        vertexItem = QGraphicsEllipseItem()
        vertexItem.setPen(QPen(QColor(0, 0, 0)))
        vertexItem.setBrush(QBrush(QColor(0, 0, 0, 255)))
        vertexItem.setPos(QPointF(0, 0))
        vertexItem.setRect(i - self.radius, j - self.radius, self.diameter, self.diameter)

        isValid = self.checkSampleCollision(vertexItem)

        if isValid:
            self.sampled_items.append(vertexItem)
            self.sampled_nodes.append((i, j))
            return False
        else:
            self.removeItem(vertexItem)
            return self.runSampling(counter)

    def checkEdgeCollision(self, edgeItem):
        self.addItem(edgeItem)

        collidingItems = edgeItem.collidingItems()

        for collidingItem in collidingItems:
            if isinstance(collidingItem, QGraphicsPolygonItem):
                return False

        return True

    def connectEdge(self, counter):
        if counter >= len(self.sampled_nodes):
            self.constructionFinished = True

            self.addItem(self.startItem)
            self.addItem(self.goalItem)

            return True

        for i in range(1, self.num_nearest + 1):
            q1 = self.sampled_nodes[counter]
            idx = self.indices[counter][i]
            distance = self.dist[counter][i]

            if idx == self.num_node:
                continue

            q2 = self.sampled_nodes[idx]

            edgeItem = QGraphicsLineItem()
            edgeItem.setPen(QPen(QColor(0, 0, 0), 1))
            edgeItem.setLine(q1[0], q1[1], q2[0], q2[1])

            if (q2, q1) in self.edges:
                continue
            else:
                edge = (q1, q2)
                self.edges.add(edge)

            isValid = self.checkEdgeCollision(edgeItem)

            if not isValid:
                self.removeItem(edgeItem)
            else:
                if not q1 in self.graph:
                    self.graph[q1] = []
                if not q2 in self.graph:
                    self.graph[q2] = []
                self.graph[q1].append((q2, distance))
                self.graph[q2].append((q1, distance))

        return False

    def mouseDoubleClickEvent(self, event):
        x = event.scenePos().x()
        y = event.scenePos().y()

        if self.algorithm in ["BFS", "DFS", "Greedy", "AStar"]:
            i = int(y // self.blockHeight)
            j = int(x // self.blockWidth)
            print(event.scenePos().x(), event.scenePos().y())

            if not 0 <= i < self.num_y_coor:
                return

            if not 0 <= j < self.num_x_coor:
                return

            if event.button() == Qt.LeftButton:
                self.start = (i, j)
                self.startItem.setRect(self.dots[0][i, j] - self.radius, self.dots[1][i, j] - self.radius, self.diameter, self.diameter)

            elif event.button() == Qt.RightButton:
                self.goal = (i, j)
                self.goalItem.setRect(self.dots[0][i, j] - self.radius, self.dots[1][i, j] - self.radius, self.diameter, self.diameter)

            print(self.start, self.goal)

            if self.initFinished:
                if self.algorithm == "BFS":
                    print(self.runBFS())
                elif self.algorithm == "DFS":
                    print(self.runDFS())
                elif self.algorithm == "Greedy":
                    print(self.runGreedy())
                elif self.algorithm == "AStar":
                    print(self.runAStarOnGrid())
                else:
                    raise NotImplementedError
        elif self.algorithm == "PRM":
            i, j = x, y
            print(event.scenePos().x(), event.scenePos().y())

            if event.button() == Qt.LeftButton:
                self.start = (i, j)
                self.startItem.setRect(i - self.radius, j - self.radius, self.diameter, self.diameter)

                if self.constructionFinished:
                    # connect q_init to roadmap
                    self.startSuccess = False
                    dist, indices = self.kdTree.query(self.start, self.num_nearest)
                    for i in range(self.num_nearest):
                        q1 = self.start
                        idx = indices[i]
                        q2 = self.sampled_nodes[idx]

                        edgeItem = QGraphicsLineItem()
                        edgeItem.setPen(QPen(QColor(0, 0, 255), 2))
                        edgeItem.setLine(q1[0], q1[1], q2[0], q2[1])

                        isValid = self.checkEdgeCollision(edgeItem)

                        self.removeItem(edgeItem)

                        if isValid:
                            self.startToRoadMapItem.setLine(q1[0], q1[1], q2[0], q2[1])
                            self.startToRoadMapItem.setVisible(True)
                            self.startSuccess = True
                            self.start = q2
                            break

                    if not self.startSuccess:
                        self.startToRoadMapItem.setVisible(False)

            elif event.button() == Qt.RightButton:
                self.goal = (i, j)
                self.goalItem.setRect(i - self.radius, j - self.radius, self.diameter, self.diameter)

                if self.constructionFinished:
                    # connect q_goal to roadmap
                    self.goalSuccess = False
                    dist, indices = self.kdTree.query(self.goal, self.num_nearest)
                    for i in range(self.num_nearest):
                        q1 = self.goal
                        idx = indices[i]
                        q2 = self.sampled_nodes[idx]

                        edgeItem = QGraphicsLineItem()
                        edgeItem.setPen(QPen(QColor(0, 0, 255), 2))
                        edgeItem.setLine(q1[0], q1[1], q2[0], q2[1])

                        isValid = self.checkEdgeCollision(edgeItem)

                        self.removeItem(edgeItem)

                        if isValid:
                            self.goalToRoadMapItem.setLine(q1[0], q1[1], q2[0], q2[1])
                            self.goalToRoadMapItem.setVisible(True)
                            self.goalSuccess = True
                            self.goal = q2
                            break

                    if not self.goalSuccess:
                        self.goalToRoadMapItem.setVisible(False)

            print(self.start, self.goal)

            if self.startSuccess and self.goalSuccess:
                print(self.runAStarOnGraph())
        else:
            raise NotImplementedError

    def drawPathOnGrid(self, transition):
        traceNode = self.goal
        while traceNode in transition:
            preNode = transition[traceNode]
            a, b = traceNode
            i, j = preNode
            lineItem = QGraphicsLineItem()
            lineItem.setPen(QPen(QColor(0, 0, 255), 3))
            lineItem.setLine(self.dots[0][a, b], self.dots[1][a, b], self.dots[0][i, j], self.dots[1][i, j])
            self.path.append(lineItem)
            self.addItem(lineItem)
            traceNode = (i, j)

    def runBFS(self):
        for lineItem in self.path:
            self.removeItem(lineItem)

        for textItem in self.textItems:
            self.removeItem(textItem)

        for colormapItem in self.colormapItems:
            self.removeItem(colormapItem)

        self.path.clear()
        self.textItems.clear()
        self.colormapItems.clear()

        que = Queue()
        que.put(self.start)

        transition = dict()
        visited = np.zeros_like(self.grid)
        visited[self.start[0], self.start[1]] = 1

        numCurrentLayer = 1
        numNextLayer = 0
        layerCounter = 0

        while not que.empty():
            node = que.get()

            numCurrentLayer -= 1
            if node == self.goal:
                self.drawPathOnGrid(transition)

                maxValue = 0
                for colormapItem in self.colormapItems:
                    if colormapItem.data(0) > maxValue:
                        maxValue = colormapItem.data(0)

                for colormapItem in self.colormapItems:
                    colormapItem.setPen(QPen(QColor(0, 0, 0), 2))
                    brushColor = list(cm.jet(int(colormapItem.data(0) / maxValue * 255)))
                    brushColor[0] *= 255
                    brushColor[1] *= 255
                    brushColor[2] *= 255
                    brushColor[3] = 200
                    colormapItem.setZValue(-1)
                    colormapItem.setBrush(QBrush(QColor(*brushColor)))
                    self.addItem(colormapItem)

                return True

            a, b = node

            if layerCounter > 0:
                textItem = QGraphicsTextItem()
                textItem.setPlainText(str(layerCounter))
                textItem.setPos(self.dots[0][a, b] - self.blockWidth / 2, self.dots[1][a, b] - self.blockHeight / 2)
                self.addItem(textItem)
                self.textItems.append(textItem)

                colormapItem = QGraphicsRectItem()
                colormapItem.setRect(self.dots[0][a, b] - self.blockWidth / 2, self.dots[1][a, b] - self.blockHeight / 2
                                     , self.blockWidth, self.blockHeight)
                colormapItem.setData(0, layerCounter)
                self.colormapItems.append(colormapItem)

            neighbors = [(a, b - 1), (a, b + 1), (a - 1, b), (a + 1, b),
                         (a - 1, b - 1), (a - 1, b + 1), (a + 1, b - 1),  (a + 1, b + 1)]

            for i, j in neighbors:
                if 0 <= i < self.num_y_coor and 0 <= j < self.num_x_coor:
                    if visited[i, j] == 1 or self.grid[i][j] == 1:
                        continue
                    else:
                        que.put((i, j))
                        visited[i, j] = 1
                        transition[(i, j)] = (a, b)
                        numNextLayer += 1

            if numCurrentLayer == 0:
                layerCounter += 1
                numCurrentLayer, numNextLayer = numNextLayer, numCurrentLayer

        return False

    def runDFS(self):
        for lineItem in self.path:
            self.removeItem(lineItem)

        for textItem in self.textItems:
            self.removeItem(textItem)

        for colormapItem in self.colormapItems:
            self.removeItem(colormapItem)

        self.path.clear()
        self.textItems.clear()
        self.colormapItems.clear()

        que = deque()
        que.append(self.start)

        transition = dict()
        visited = np.zeros_like(self.grid)
        visited[self.start[0], self.start[1]] = 1

        visitedOrder = 0
        while len(que) > 0:
            node = que.pop()

            if node == self.goal:
                self.drawPathOnGrid(transition)

                maxValue = 0
                for colormapItem in self.colormapItems:
                    if colormapItem.data(0) > maxValue:
                        maxValue = colormapItem.data(0)

                for colormapItem in self.colormapItems:
                    colormapItem.setPen(QPen(QColor(0, 0, 0), 2))
                    brushColor = list(cm.jet(int(colormapItem.data(0) / maxValue * 255)))
                    brushColor[0] *= 255
                    brushColor[1] *= 255
                    brushColor[2] *= 255
                    brushColor[3] = 200
                    colormapItem.setZValue(-1)
                    colormapItem.setBrush(QBrush(QColor(*brushColor)))
                    self.addItem(colormapItem)

                return True

            a, b = node

            if visitedOrder > 0:
                textItem = QGraphicsTextItem()
                textItem.setPlainText(str(visitedOrder))
                textItem.setPos(self.dots[0][a, b] - self.blockWidth / 2, self.dots[1][a, b] - self.blockHeight / 2)
                self.addItem(textItem)
                self.textItems.append(textItem)

                colormapItem = QGraphicsRectItem()
                colormapItem.setRect(self.dots[0][a, b] - self.blockWidth / 2, self.dots[1][a, b] - self.blockHeight / 2
                                     , self.blockWidth, self.blockHeight)
                colormapItem.setData(0, visitedOrder)
                self.colormapItems.append(colormapItem)

            neighbors = [(a - 1, b - 1), (a - 1, b), (a - 1, b + 1),
                         (a, b - 1), (a, b + 1),
                         (a + 1, b - 1), (a + 1, b), (a + 1, b + 1)]

            visitedOrder += 1

            for i, j in neighbors:
                if 0 <= i < self.num_y_coor and 0 <= j < self.num_x_coor:
                    if visited[i, j] == 1 or self.grid[i][j] == 1:
                        continue
                    else:
                        que.append((i, j))
                        visited[i, j] = 1
                        transition[(i, j)] = (a, b)

        return False

    def runGreedy(self):
        for lineItem in self.path:
            self.removeItem(lineItem)

        for textItem in self.textItems:
            self.removeItem(textItem)

        for colormapItem in self.colormapItems:
            self.removeItem(colormapItem)

        self.path.clear()
        self.textItems.clear()
        self.colormapItems.clear()

        que = PriorityQueue()
        que.put((0, self.start))

        transition = dict()
        visited = np.zeros_like(self.grid)
        visited[self.start[0], self.start[1]] = 1

        visitedOrder = 0
        while not que.empty():
            _, node = que.get()

            if node == self.goal:
                self.drawPathOnGrid(transition)

                maxValue = 0
                for colormapItem in self.colormapItems:
                    if colormapItem.data(0) > maxValue:
                        maxValue = colormapItem.data(0)

                for colormapItem in self.colormapItems:
                    colormapItem.setPen(QPen(QColor(0, 0, 0), 2))
                    brushColor = list(cm.jet(int(colormapItem.data(0) / maxValue * 255)))
                    brushColor[0] *= 255
                    brushColor[1] *= 255
                    brushColor[2] *= 255
                    brushColor[3] = 200
                    colormapItem.setZValue(-1)
                    colormapItem.setBrush(QBrush(QColor(*brushColor)))
                    self.addItem(colormapItem)

                return True

            a, b = node

            if visitedOrder > 0:
                textItem = QGraphicsTextItem()
                textItem.setPlainText(str(visitedOrder))
                textItem.setPos(self.dots[0][a, b] - self.blockWidth / 2, self.dots[1][a, b] - self.blockHeight / 2)
                self.addItem(textItem)
                self.textItems.append(textItem)

                colormapItem = QGraphicsRectItem()
                colormapItem.setRect(self.dots[0][a, b] - self.blockWidth / 2, self.dots[1][a, b] - self.blockHeight / 2
                                     , self.blockWidth, self.blockHeight)
                colormapItem.setData(0, visitedOrder)
                self.colormapItems.append(colormapItem)

            neighbors = [(a - 1, b - 1), (a - 1, b), (a - 1, b + 1),
                         (a, b - 1), (a, b + 1),
                         (a + 1, b - 1), (a + 1, b), (a + 1, b + 1)]

            visitedOrder += 1

            for i, j in neighbors:
                if 0 <= i < self.num_y_coor and 0 <= j < self.num_x_coor:
                    if visited[i, j] == 1 or self.grid[i][j] == 1:
                        continue
                    else:
                        min_axis_value = min(abs(self.goal[0] - i), abs(self.goal[1] - j))
                        residual = max(abs(self.goal[0] - i) - min_axis_value, abs(self.goal[1] - j) - min_axis_value)
                        priority = min_axis_value * 1.4 + residual
                        que.put((priority, (i, j)))
                        visited[i, j] = 1
                        transition[(i, j)] = (a, b)

        return False

    def runAStarOnGrid(self):
        for lineItem in self.path:
            self.removeItem(lineItem)

        for textItem in self.textItems:
            self.removeItem(textItem)

        for colormapItem in self.colormapItems:
            self.removeItem(colormapItem)

        self.path.clear()
        self.textItems.clear()
        self.colormapItems.clear()

        que = PriorityQueue()
        que.put((0, self.start))

        transition = dict()
        g = dict()  # total length of a back-pointer path
        g[self.start] = 0  # length from start point to itself is set to 0
        visited = np.zeros_like(self.grid)
        visited[self.start[0], self.start[1]] = 1

        visitedOrder = 0
        while not que.empty():
            _, node = que.get()

            # if node been visited, remove it from open set
            if visited[node[0], node[1]] > 1:
                continue

            if node == self.goal:  # if no more nodes with priority higher than this node, the optimal path was found
                self.drawPathOnGrid(transition)

                maxValue = 0
                for colormapItem in self.colormapItems:
                    if colormapItem.data(0) > maxValue:
                        maxValue = colormapItem.data(0)

                for colormapItem in self.colormapItems:
                    colormapItem.setPen(QPen(QColor(0, 0, 0), 2))
                    brushColor = list(cm.jet(int(colormapItem.data(0) / maxValue * 255)))
                    brushColor[0] *= 255
                    brushColor[1] *= 255
                    brushColor[2] *= 255
                    brushColor[3] = 200
                    colormapItem.setZValue(-1)
                    colormapItem.setBrush(QBrush(QColor(*brushColor)))
                    self.addItem(colormapItem)

                return True

            a, b = node

            # move to closed set
            visited[a, b] += 1

            if visitedOrder > 0:
                textItem = QGraphicsTextItem()
                textItem.setPlainText(str(visitedOrder))
                textItem.setPos(self.dots[0][a, b] - self.blockWidth / 2, self.dots[1][a, b] - self.blockHeight / 2)
                self.addItem(textItem)
                self.textItems.append(textItem)

                colormapItem = QGraphicsRectItem()
                colormapItem.setRect(self.dots[0][a, b] - self.blockWidth / 2, self.dots[1][a, b] - self.blockHeight / 2
                                     , self.blockWidth, self.blockHeight)
                colormapItem.setData(0, visitedOrder)
                self.colormapItems.append(colormapItem)

            # to distinguish between diagonal and axial neighbors
            neighbors = [(a - 1, b - 1, True), (a - 1, b, False), (a - 1, b + 1, True),
                         (a, b - 1, False), (a, b + 1, False),
                         (a + 1, b - 1, True), (a + 1, b, False), (a + 1, b + 1, True)]

            visitedOrder += 1

            for i, j, diag in neighbors:
                if 0 <= i < self.num_y_coor and 0 <= j < self.num_x_coor:
                    # skip collision grid
                    if self.grid[i][j] == 1:
                        continue
                    # if in the open set
                    elif visited[i, j] == 1:
                        tmp = g[(a, b)] + (1.4 if diag else 1)
                        # check and update g
                        if tmp < g[(i, j)]:
                            g[(i, j)] = tmp
                            transition[(i, j)] = (a, b)

                            # recompute the priority, add to open set
                            # it is okay if the node was added multiple times
                            # cause the newly added one has high priority
                            min_axis_value = min(abs(self.goal[0] - i), abs(self.goal[1] - j))
                            residual = max(abs(self.goal[0] - i) - min_axis_value, abs(self.goal[1] - j) - min_axis_value)
                            h = min_axis_value * 1.4 + residual
                            priority = h + g[(i, j)]

                            que.put((priority, (i, j)))
                    # if not in the open set
                    elif visited[i, j] == 0:
                        g[(i, j)] = g[(a, b)] + (1.4 if diag else 1)
                        transition[(i, j)] = (a, b)

                        # compute h, and priority, add to open set
                        min_axis_value = min(abs(self.goal[0] - i), abs(self.goal[1] - j))
                        residual = max(abs(self.goal[0] - i) - min_axis_value, abs(self.goal[1] - j) - min_axis_value)
                        h = min_axis_value * 1.4 + residual
                        priority = h + g[(i, j)]

                        que.put((priority, (i, j)))
                        visited[i, j] = 1

        return False

    def drawPathOnGraph(self, transition):
        traceNode = self.goal
        while traceNode in transition:
            preNode = transition[traceNode]
            a, b = traceNode
            i, j = preNode
            lineItem = QGraphicsLineItem()
            lineItem.setPen(QPen(QColor(0, 0, 255), 3))
            lineItem.setLine(a, b, i, j)
            self.path.append(lineItem)
            self.addItem(lineItem)
            traceNode = (i, j)

    def runAStarOnGraph(self):
        for lineItem in self.path:
            self.removeItem(lineItem)

        self.path.clear()

        que = PriorityQueue()
        que.put((0, self.start))

        transition = dict()
        g = dict()  # total length of a back-pointer path
        g[self.start] = 0  # length from start point to itself is set to 0
        visited = dict()
        for key in self.graph.keys():
            visited[key] = 0
        visited[self.start] = 1

        while not que.empty():
            _, node = que.get()

            # if node been visited, remove it from open set
            if visited[node] > 1:
                continue

            if node == self.goal:  # if no more nodes with priority higher than this node, the optimal path was found
                print("dist is ", g[node])
                self.drawPathOnGraph(transition)
                return True

            a, b = node

            # move to closed set
            visited[node] += 1

            # to distinguish between diagonal and axial neighbors
            neighbors = self.graph[node]

            for next_node, distance in neighbors:
                # if in the open set
                if visited[next_node] == 1:
                    tmp = g[node] + distance
                    # check and update g
                    if tmp < g[next_node]:
                        g[next_node] = tmp
                        transition[next_node] = node

                        # recompute the priority, add to open set
                        # it is okay if the node was added multiple times
                        # cause the newly added one has high priority
                        h = np.linalg.norm(np.array(self.goal) - np.array(next_node))
                        priority = h + g[next_node]

                        que.put((priority, next_node))
                # if not in the open set
                elif visited[next_node] == 0:
                    g[next_node] = g[node] + distance
                    transition[next_node] = node

                    # compute h, and priority, add to open set
                    h = np.linalg.norm(np.array(self.goal) - np.array(next_node))
                    priority = h + g[next_node]

                    que.put((priority, next_node))
                    visited[next_node] = 1

        return False


class CentralWidget(QWidget):
    def __init__(self, args):
        super(CentralWidget, self).__init__()
        self.counter = 0

        self.setLayout(QHBoxLayout())

        self.scene = GraphicsScene(args=args)
        self.view = QGraphicsView(self)
        self.view.setScene(self.scene)

        self.layout().addWidget(self.view)

        self.timer = QTimer(self)
        if args.algorithm in ["BFS", "DFS", "Greedy", "AStar"]:
            self.timer.timeout.connect(self.checkGridCollision)
        elif args.algorithm == "PRM":
            self.timer.timeout.connect(self.runNodeSampling)
        else:
            raise NotImplementedError
        self.timer.setInterval(0.0)

    def checkGridCollision(self):
        isFinished = self.scene.runGridCollisionCheck(self.counter)
        if isFinished:
            self.counter = 0
            self.timer.stop()
        else:
            self.counter += 1

    def startDetection(self):
        self.timer.start()

    def runNodeSampling(self):
        isSamplingFinished = self.scene.runSampling(self.counter)
        if isSamplingFinished:
            self.counter = 0
            self.timer.stop()
            self.timer.timeout.disconnect(self.runNodeSampling)
            self.timer.timeout.connect(self.runEdgeConnecting)
            self.timer.start()
        else:
            self.counter += 1

    def runEdgeConnecting(self):
        isConstructionFinished = self.scene.connectEdge(self.counter)
        if isConstructionFinished:
            self.counter = 0
            self.timer.stop()
            self.timer.timeout.disconnect(self.runEdgeConnecting)
        else:
            self.counter += 1

    def startGraphBuilding(self):
        self.timer.start()


class MainWindow(QMainWindow):
    def __init__(self, args):
        super(MainWindow, self).__init__()
        self.view = CentralWidget(args=args)
        self.setCentralWidget(self.view)
        if args.algorithm in ["BFS", "DFS", "Greedy", "AStar"]:
            self.view.startDetection()
        elif args.algorithm == "PRM":
            self.view.startGraphBuilding()
        else:
            raise NotImplementedError
        self.statusBar().showMessage("Ready!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple Planner Animation.")

    parser.add_argument("--algorithm", type=str, default="BFS", choices=["BFS", "DFS", "Greedy", "AStar", "PRM"], help="Planning algorithm.")

    # Common
    parser.add_argument("--radius", type=float, default=6, help="Radius for the start and goal dot.")

    # Graph Search Based Methods (BFS, DFS, Greedy, AStar)
    parser.add_argument("--num_horizontal_grid", type=int, default=40, help="Number of grid for each row.")
    parser.add_argument("--num_vertical_grid", type=int, default=40, help="Number of grid for each column.")

    # Sampling Based Methods (PRM)
    parser.add_argument("--num_node", type=int, default=1000, help="Number of nodes to put in the roadmap.")
    parser.add_argument("--num_nearest", type=int, default=4, help="Number of closest neighbors to examine for each configuration.")

    args = parser.parse_args()

    print(args)

    app = QApplication(sys.argv)

    mainwindow = MainWindow(args=args)
    mainwindow.setWindowTitle("Simple Planner Animation")
    mainwindow.show()
    app.exec_()
