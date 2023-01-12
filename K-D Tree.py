from typing import List
from collections import namedtuple
import time


class Point(namedtuple("Point", "x y")):
    def __repr__(self) -> str:
        return f'Point{tuple(self)!r}'


class Rectangle(namedtuple("Rectangle", "lower upper")):
    def __repr__(self) -> str:
        return f'Rectangle{tuple(self)!r}'

    def is_contains(self, p: Point) -> bool:
        return self.lower.x <= p.x <= self.upper.x and self.lower.y <= p.y <= self.upper.y


class Node(namedtuple("Node", "location left right")):
    """
    location: Point
    left: Node
    right: Node
    """
    '''
    Create the KdNode class for K-d Tree
    '''

    class Node:
        def __init__(self, location: Point, left=None, right=None):
            self.location = location
            self.left = left
            self.right = right

    def __repr__(self):
        return f'Node(location={self.location}, left={self.left}, right={self.right})'



class KDTree:
    """k-d tree"""

    def __init__(self):
        self._root = None
        self._n = 0
        
    def insert(self, p: List[Point]):
        for point in p:
             self._root = self._insert(point, self._root, 0)
             self._n += 1

    def _insert(self, point, node, depth):
        if node is None:
            return Node(point, None, None)
        axis = depth % 2
        if point[axis] < node.location[axis]:
            node = node._replace(left=self._insert(point, node.left, depth+1))
        else:
            node = node._replace(right=self._insert(point, node.right, depth+1))
        return node


    def range(self, rectangle: Rectangle) -> List[Point]:
        """range query"""
        results = []
        self._range(self._root, rectangle, results)
        return results

    def _range(self, node, rectangle, results):
        if node is None:
            return
        if rectangle.is_contains(node.location):
            results.append(node.location)
        self._range(node.left, rectangle, results)
        self._range(node.right, rectangle, results)
        
    
    def nearest_neighbor(self, point: Point) -> Point:
        closest = None
        depth = 0
        return self._nearest_neighbor(point, self._root, closest, depth)

    def _nearest_neighbor(self, point, node, closest, depth):
        if node is None:
            return closest
        if closest is None or self.distance(point, node.location) < self.distance(point, closest):
            closest = node.location
        axis = depth % 2
        if point[axis] < node.location[axis]:
            closest = self._nearest_neighbor(point, node.left, closest, depth+1)
        else:
            closest = self._nearest_neighbor(point, node.right, closest, depth+1)
        return closest


    def distance(self, p1, p2):
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

def range_test():
    points = [Point(7, 2), Point(5, 4), Point(9, 6), Point(4, 7), Point(8, 1), Point(2, 3)]
    kd = KDTree()
    kd.insert(points)
    result = kd.range(Rectangle(Point(0, 0), Point(6, 6)))
    assert sorted(result) == sorted([Point(2, 3), Point(5, 4)])


def performance_test():
    points = [Point(x, y) for x in range(1000) for y in range(1000)]

    lower = Point(500, 500)
    upper = Point(504, 504)
    rectangle = Rectangle(lower, upper)
    #  naive method
    start = int(round(time.time() * 1000))
    result1 = [p for p in points if rectangle.is_contains(p)]
    end = int(round(time.time() * 1000))
    print(f'Naive method: {end - start}ms')

    kd = KDTree()
    kd.insert(points)
    # k-d tree
    start = int(round(time.time() * 1000))
    result2 = kd.range(rectangle)
    end = int(round(time.time() * 1000))
    print(f'K-D tree: {end - start}ms')

    assert sorted(result1) == sorted(result2)


if __name__ == '__main__':
    range_test()
    performance_test()
    
import matplotlib.pyplot as plt

points = [Point(x, y) for x in range(1000) for y in range(1000)]
x_axis = []
y_axis_kd = []
y_axis_naive = []
for n in range(1, 11):
    lower = Point(500, 500)
    upper = Point(504, 504)
    rectangle = Rectangle(lower, upper)

    # k-d tree method
    kd = KDTree()
    kd.insert(points[:n*1000])
    start = int(round(time.time() * 1000))
    result2 = kd.range(rectangle)
    end = int(round(time.time() * 1000))
    y_axis_kd.append(end - start)
    x_axis.append(n*1000)
    
    # naive method
    start = int(round(time.time() * 1000))
    result1 = [p for p in points[:n*1000] if rectangle.is_contains(p)]
    end = int(round(time.time() * 1000))
    y_axis_naive.append(end - start)

# create line plot
plt.plot(x_axis, y_axis_kd, label='k-d tree method')
plt.plot(x_axis, y_axis_naive, label='naive method')
plt.xlabel('Number of points in tree')
plt.ylabel('Time taken (ms)')
plt.legend()
plt.show()

