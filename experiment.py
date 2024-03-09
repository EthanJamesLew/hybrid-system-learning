import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from hybrid_sindy import hybrid_sindy

# Parameters
kappa = 10.0  # Spring constant-like parameter
num_trajectories = 50
time_end = 5
num_time_points = 100

# System of differential equations
def model(t, z):
    y, v = z
    if y <= 1:
        dvdt = 1 - kappa * (y - 1)
    else:
        dvdt = -1
    dydt = v
    return [dydt, dvdt]

def model_np(y):
    y, v = y[:, 0], y[:, 1]
    dy = 1 - kappa * (y - 1)
    dy[y <= 1] = -1
    return np.hstack([np.atleast_2d(v).T, np.atleast_2d(dy).T])

# Initialize an array to store the trajectories
trajectories = []
times = []

# Generate multiple trajectories with different initial conditions
for _ in range(num_trajectories):
    y0 = [np.random.uniform(0, 2), np.random.uniform(-3, 3)]  # Random initial conditions
    t_eval = np.linspace(0, time_end, num_time_points) 
    sol = solve_ivp(model, [0, time_end], y0, t_eval=t_eval, max_step=0.01)
    trajectories.append(
        np.hstack(
            (np.atleast_2d(sol.y[0]).T, 
             np.atleast_2d(sol.y[1]).T,) 
        ))
    times.append(t_eval)

hybrid_sindy(trajectories, times)