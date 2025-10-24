import numpy as np
from itertools import chain
from helpers.delaunay_helper_classes import Vertex, Edge, Triangle

class Delaunay:
    def __init__(self, points : list[list[float]]):
        self.vertices = [Vertex(point[0], point[1], identity) for identity, point in enumerate(points)]
        self.weight = 0.5
        
    def compute_super_triangle(self) -> Triangle:
        min_x, min_y = np.inf, np.inf
        max_x, max_y = -np.inf, -np.inf
        
        for vertex in self.vertices:
            min_x = min(min_x, vertex.x)
            min_y = min(min_y, vertex.y)
            max_x = max(max_x, vertex.x)
            max_y = max(max_y, vertex.y)
            
        dx = (max_x - min_x) * 10
        dy = (max_y - min_y) * 10
        
        vertex_0 = Vertex(min_x - dx, min_y - dy * 3)
        vertex_1 = Vertex(min_x - dx, max_y + dy)
        vertex_2 = Vertex(max_x + dx * 3, max_y + dy)
        
        return Triangle(vertex_0, vertex_1, vertex_2)

    def add_vertex(self, vertex : Vertex, triangles : list[Triangle]):
        edges : list[Edge] = []
        new_triangles = []

        for triangle in triangles:
            if triangle.in_circles_circumference(vertex):
                edges.append(Edge(triangle.vertex_0, triangle.vertex_1))
                edges.append(Edge(triangle.vertex_1, triangle.vertex_2))
                edges.append(Edge(triangle.vertex_2, triangle.vertex_0))
            else:
                new_triangles.append(triangle)

        # Remove duplicate cases
        def get_unique_edges(edges : list[Edge]):
            unique_edges = []

            for i in range(len(edges)):
                is_unique = True
                for j in range(len(edges)):
                    if i != j and edges[i].compare(edges[j].vertex_0, edges[j].vertex_1):
                        is_unique = False
                        break
                    
                if is_unique:
                    unique_edges.append(edges[i])

            return unique_edges

        edges = get_unique_edges(edges)
        # Create new triangles from the unique edges and new vertex
        for edge in edges:
            new_triangles.append(Triangle(edge.vertex_0, edge.vertex_1, vertex))

        return new_triangles

    def triangulate(self) -> list[list[float]]:
        super_triangle : Triangle = self.compute_super_triangle()
        triangles : list[Triangle] = [super_triangle]
        mid_points = []

        def are_connected(identity_list : list[int]) -> bool:
            has_odd = any(identity % 2 != 0 for identity in identity_list)
            has_even = any(identity % 2 == 0 for identity in identity_list)
            
            return has_odd == has_even
        
        def are_different_cones(identity_0 : int, identity_1 : int):
            return (identity_0 % 2 == 0 and identity_1 % 2 != 0) or (identity_0 % 2 != 0 and identity_1 % 2 == 0)
        
        def are_almost_connected(identity_list : list[int]):
            tolerance = 2
            return all(abs(identity_list[i] - identity_list[1]) <= tolerance for i in [0, 2])
        
        for vertex in self.vertices:
            triangles = self.add_vertex(vertex, triangles)

        for triangle in triangles:
            # Remove triangles that share edges with super triangle
            has_super_vertex = (
                triangle.vertex_0.equals(super_triangle.vertex_0) or triangle.vertex_0.equals(super_triangle.vertex_1) or triangle.vertex_0.equals(super_triangle.vertex_2) or
                triangle.vertex_1.equals(super_triangle.vertex_0) or triangle.vertex_1.equals(super_triangle.vertex_1) or triangle.vertex_1.equals(super_triangle.vertex_2) or
                triangle.vertex_2.equals(super_triangle.vertex_0) or triangle.vertex_2.equals(super_triangle.vertex_1) or triangle.vertex_2.equals(super_triangle.vertex_2)
            )
            # Removes triangles if they aren't connected
            identity_list = [triangle.vertex_0.identity, triangle.vertex_1.identity, triangle.vertex_2.identity]

            if not has_super_vertex and are_connected(identity_list) and are_almost_connected(identity_list):
                triangle.vertex_0.number_of_connections += 1
                triangle.vertex_1.number_of_connections += 1
                triangle.vertex_2.number_of_connections += 1
                
                vertices = [triangle.vertex_0, triangle.vertex_1, triangle.vertex_2]
                # Find consecutive pair
                for i in range(len(vertices) - 1):
                    if are_different_cones(vertices[i+1].identity, vertices[i].identity):
                        
                        mid_points.append([
                            vertices[i].x * self.weight + vertices[i+1].x * (1 - self.weight),
                            vertices[i].y * self.weight + vertices[i+1].y * (1 - self.weight)
                        ])

        return np.array(mid_points)

class DelaunayPathPlanner:
    def __init__(self, start_point : list[list[float]]):
        self.start_point = start_point
        self.start_points = [start_point]
        
    def reset(self, start_point : list[list[float]]):
        self.start_point = start_point
        self.start_points = [start_point]
    
    def find_path(self, blue_cones : list[list[float]], yellow_cones : list[list[float]], n_steps : int):
        # Store previous length
        previous_length = len(self.start_points)
        
        # Handle empty cone arrays
        if len(blue_cones) == 0 or len(yellow_cones) == 0:
            # Keep previous path if no cones available
            return
        
        all_points = []
        
        max_pairs = min(len(blue_cones), len(yellow_cones))
        for i in range(max_pairs):
            all_points.append(blue_cones[i])
            all_points.append(yellow_cones[i])
        
        # Add remaining cones
        if len(blue_cones) > max_pairs:
            all_points.extend(blue_cones[max_pairs:])
        if len(yellow_cones) > max_pairs:
            all_points.extend(yellow_cones[max_pairs:])
        
        if len(all_points) < 2:
            # Keep previous path if not enough points
            return
        
        # Perform Delaunay triangulation
        delaunay = Delaunay(all_points)
        midpoints = delaunay.triangulate()
        
        # Limit to n_steps
        path_points = midpoints[:n_steps]
        
        # Build new path
        if len(path_points) > 0:
            new_path = [self.start_point] + [np.array(p) for p in path_points]
            
            # Only update if new path is at least as long as previous
            if len(new_path) >= previous_length:
                self.start_points = new_path