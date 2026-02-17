import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class MPC:
    def __init__(self, car_params):
        self.velocity = 4
        self.prediction_horizon = 12
        self.dt = 0.1
        self.wheel_base = 1.0
        
        self.initial_state = (0.0, 0.0, 0.0)
        self.max_steering_angle = math.radians(car_params["max_steering_angle"])
        self.min_steering_angle = math.radians(car_params["min_steering_angle"])
        
        self.previous_steering = 0.0
        self.w_y = 9.0
        self.w_heading = 4.0
        self.w_delta = 0.9
        self.w_delta_rate = 0.0
        
    def kinematic_bicycle_step(self, state_vector : tuple[float, float, float], steering : float) -> tuple[float, float, float]:
        x, y, heading = state_vector

        x_next = x + self.velocity * math.cos(heading) * self.dt
        y_next = y + self.velocity * math.sin(heading) * self.dt
        heading_next = heading + (self.velocity / self.wheel_base) * math.tan(steering) * self.dt

        return (x_next, y_next, heading_next)
            
    def sample_path(self, path : list[list[float]], arc_length_array : list[float], heading_array : list[float]) -> list[tuple[float, float, float]]:
        ref_points = []

        for i in range(self.prediction_horizon):
            s_i = i * self.velocity * self.dt

            index = np.searchsorted(arc_length_array, s_i) - 1
            index = np.clip(index, 0, len(arc_length_array) - 2)

            ds = arc_length_array[index + 1] - arc_length_array[index]
            alpha = 0.0
            if ds > 1e-6:
                alpha = (s_i - arc_length_array[index]) / ds
            alpha = np.clip(alpha, 0.0, 1.0)

            x = (1 - alpha) * path[index, 0] + alpha * path[index + 1, 0]
            y = (1 - alpha) * path[index, 1] + alpha * path[index + 1, 1]
            heading = (1 - alpha) * heading_array[index] + alpha * heading_array[index + 1]

            ref_points.append((x, y, heading))

        return ref_points

    def mpc_cost(self, steering_sequence, ref_points):  
        x, y, heading = self.initial_state
        cost = 0.0

        delta_prev = self.previous_steering

        for k, delta in enumerate(steering_sequence):
            x, y, heading = self.kinematic_bicycle_step((x, y, heading), delta)

            x_ref, y_ref, heading_ref = ref_points[k]

            # Lateral error
            e_y = (x - x_ref) * math.sin(heading_ref) - (y - y_ref) * math.cos(heading_ref)

            # Heading error (wrapped to [-pi, pi])
            e_heading = math.atan2(
                math.sin(heading - heading_ref),
                math.cos(heading - heading_ref)
            )

            cost += self.w_y * e_y**2
            cost += self.w_heading * e_heading**2
            cost += self.w_delta * delta**2
            cost += self.w_delta_rate * (delta - delta_prev)**2

            delta_prev = delta

        return cost
    
    def step(self, path):
        n = len(path)
        
        # Arc length
        current_arc_length = 0.0
        arc_length_array = [0.0]
        for i in range(1, n):
            current_arc_length += math.sqrt(
                (path[i][0] - path[i-1][0])**2 +
                (path[i][1] - path[i-1][1])**2 
            )
            arc_length_array.append(current_arc_length)
            
        # Heading at each control horizon step
        heading_array = [0.0]
        for i in range(n - 1):
            current_heading = math.atan2(
                path[i + 1][1] - path[i][1], 
                path[i + 1][0] - path[i][0]
            )
            heading_array.append(current_heading)

        ref_points = np.array(self.sample_path(path, arc_length_array, heading_array))

        bounds = [
            (self.min_steering_angle, self.max_steering_angle)
            for _ in range(self.prediction_horizon)
        ]
        
        result = minimize(
            fun=self.mpc_cost,
            x0=np.full(self.prediction_horizon, self.previous_steering),
            args=(ref_points),
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": 50,
                "ftol": 1e-3,
                "disp": False
            }
        )

        optimal_delta = result.x[0]
        self.previous_steering = optimal_delta
        return math.degrees(optimal_delta)
        
if __name__ == "__main__":
    mpc = MPC()
    path = np.array([
        [0.00000000, 0.00000000],
        [5.26147721, 7.94353014],
        [6.64948270, 4.64445239],
        [9.16534879, 7.83110239],
        [1.04572921, 1.20643590],
        [1.18277579, 2.07438066],
        [1.34289330, 3.36222113],
        [1.45915406, 4.50748745],
        [1.54534335, 5.64801825],
        [1.64165415, 6.59269463],
        [1.71568966, 7.57258864],
        [1.84013552, 8.72122442]
    ])
    mpc.step(path)