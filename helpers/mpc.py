from scipy.optimize import minimize
from config import car_params
import numpy as np
import math

def mpc_cost(delta, x0, A, B, y_ref, psi_ref, Qy, Qpsi, R):
    """
    delta: control sequence (N,)
    """
    x = x0.copy()
    cost = 0.0

    for k in range(len(delta)):
        # propagate state
        x = A @ x + B.flatten() * delta[k]

        e_y = x[0] - y_ref[k]
        e_psi = x[1] - psi_ref[k]

        cost += (
            Qy   * e_y**2 +
            Qpsi * e_psi**2 +
            R    * delta[k]**2
        )

    return cost

def extract_reference(path, N):
    """
    path: Nx2 in vehicle frame
    returns y_ref, psi_ref (length N)
    """
    y_ref = path[:N, 1]

    psi_ref = []
    for k in range(N):
        if k < len(path) - 1:
            dx = path[k+1, 0] - path[k, 0]
            dy = path[k+1, 1] - path[k, 1]
        else:
            dx = path[k, 0] - path[k-1, 0]
            dy = path[k, 1] - path[k-1, 1]

        psi_ref.append(np.arctan2(dy, dx))

    return np.array(y_ref), np.array(psi_ref)

def build_model(v, L, dt):
    A = np.array([
        [1.0, v * dt],
        [0.0, 1.0]
    ])

    B = np.array([
        [0.0],
        [v * dt / L]
    ])

    return A, B

def lateral_mpc_step(path, v, L, dt):
    # Horizon
    N = min(10, len(path) - 1)

    # Extract reference
    y_ref, psi_ref = extract_reference(path, N)

    # Initial state (vehicle frame!)
    x0 = np.array([0.0, 0.0])   # [e_y, e_psi]

    # Model
    A, B = build_model(v, L, dt)

    # Weights
    Qy   = 8.0 # Centering (higher = more effort)
    Qpsi = 3.0 # Direction (higher = more effort)
    R    = 2 # Steering effort (higher = lazier)

    # Initial guess
    delta0 = np.zeros(N)

    # Steering bounds (rad)
    min_steering_angle = math.radians(car_params["min_steering_angle"])
    max_steering_angle = math.radians(car_params["max_steering_angle"])
    bounds = [(min_steering_angle, max_steering_angle)] * N

    # Solve QP
    res = minimize(
        mpc_cost,
        delta0,
        args=(x0, A, B, y_ref, psi_ref, Qy, Qpsi, R),
        bounds=bounds,
        method="SLSQP"
    )

    # Apply only first control
    return res.x[0]

def mpc_steering(path, velocity):
    print(len(path))
    delta = lateral_mpc_step(path, velocity, car_params["wheel_base"], 0.1) 
    #print(math.degrees(delta))
    return math.degrees(delta)