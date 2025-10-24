import numpy as np
from config import ConeClasses

from helpers.delaunay import DelaunayPathPlanner

class PathPlanner():
    def __init__(self, opt={"n_steps": 20}):
        self.n_steps = opt["n_steps"]
        self.planner = DelaunayPathPlanner(np.array([0.0, 0.0]))
        
    def find_path(self, cones):
        self.planner.reset(np.array([0.0, 0.0]))
        if cones is None:
            blue_cones = np.zeros((0, 3))
            yellow_cones = np.zeros((0, 3))
        else:
            yellow_cones = cones[cones[:, 2] == ConeClasses.YELLOW, :]
            blue_cones = cones[cones[:, 2] == ConeClasses.BLUE, :]
        try:
            self.planner.find_path(blue_cones[:, :2], yellow_cones[:, :2], n_steps=self.n_steps)
            path = np.vstack(self.planner.start_points)
        except:
            path = np.array([[0., 0.]])

        path_sorted_idxs = path[:, 0].argsort()
        path = path[path_sorted_idxs]

        return path
