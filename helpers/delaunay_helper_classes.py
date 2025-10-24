import numpy as np

# Source: https://www.gorillasun.de/blog/bowyer-watson-algorithm-for-delaunay-triangulation/

class Vertex:
    def __init__(self, x : float, y : float, identity : int = 0) -> None:
        self.x = x
        self.y = y
        self.identity = identity
        self.number_of_connections = 1
    
    def equals(self, vertex : "Vertex") -> bool:
        return self.x == vertex.x and self.y == vertex.y
        
class Edge:
    def __init__(self, vertex_0 : Vertex, vertex_1 : Vertex) -> None:
        self.vertex_0 = vertex_0
        self.vertex_1 = vertex_1
        
    def compare(self, vertex_0 : Vertex, vertex_1 : Vertex) -> bool:
        return (
            (self.vertex_0.equals(vertex_0) and self.vertex_1.equals(vertex_1)) or 
            (self.vertex_0.equals(vertex_1) and self.vertex_1.equals(vertex_0))
        ) 

class Circumference_Circle:
    def __init__(self) -> None:
        self.center_x = 0
        self.center_y = 0
        self.radius = 0
        
    def calculate_circumcenter(self, vertex_0 : Vertex, vertex_1 : Vertex, vertex_2 : Vertex) -> None:
        denominator =  (2 * (
            (vertex_0.x * (vertex_1.y - vertex_2.y)) + 
            (vertex_1.x * (vertex_2.y - vertex_0.y)) +
            (vertex_2.x * (vertex_0.y - vertex_1.y))
        ))
        
        if denominator != 0:
            self.center_x = (
                (vertex_0.x**2 + vertex_0.y**2) * (vertex_1.y - vertex_2.y) + 
                (vertex_1.x**2 + vertex_1.y**2) * (vertex_2.y - vertex_0.y) +
                (vertex_2.x**2 + vertex_2.y**2) * (vertex_0.y - vertex_1.y)
            ) / denominator

            self.center_y = (
                (vertex_0.x**2 + vertex_0.y**2) * (vertex_2.x - vertex_1.x) + 
                (vertex_1.x**2 + vertex_1.y**2) * (vertex_0.x - vertex_2.x) +
                (vertex_2.x**2 + vertex_2.y**2) * (vertex_1.x - vertex_0.x)
            ) / denominator

            self.radius = np.sqrt((vertex_0.x - self.center_x)**2 + (vertex_0.y - self.center_y)**2)


class Triangle:
    def __init__(self, vertex_0 : Vertex, vertex_1 : Vertex, vertex_2 : Vertex) -> None:
        self.vertex_0 = vertex_0
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        self.circumference_circle = Circumference_Circle()
        self.circumference_circle.calculate_circumcenter(vertex_0, vertex_1, vertex_2)
    
    def in_circles_circumference(self, vertex : Vertex) -> bool:
        dx = self.circumference_circle.center_x - vertex.x
        dy = self.circumference_circle.center_y - vertex.y
        
        return np.sqrt(dx**2 + dy**2) <= self.circumference_circle.radius