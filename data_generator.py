import numpy as np
import pandas as pd
import tensorflow as tf
import json

from pyDOE import lhs
from get_function import get_value
from sklearn.model_selection import train_test_split

def get_normal_data():
    nr, nt = (50, 50)
    r_max, N_b, N_u = get_value()
    b_size = (50 - N_b)/100
    u_size = (50 - N_u)/100
    zeniths = np.linspace(0, r_max, nr)
    azimuths = np.radians(np.linspace(0, 180, nt))
    data_u = np.array(pd.read_csv('prediction.csv', header=None))
    exact_u = np.array(data_u).reshape((nr, nt))

    rv, tv = np.meshgrid(zeniths, azimuths)
    # Preparing the x and y together as an input for predictions in one single array, as x_star
    x_star = np.hstack((rv.flatten()[:, None], tv.flatten()[:, None], data_u))

    # Getting the lowest boundary conditions (r=0)
    xx1 = np.hstack((rv[:, 0:1], tv[:, 0:1], exact_u[:, 0:1]))
    uu1 = exact_u[:, 0:1]

    # Getting the end condition (Phi=3.14)
    xx2 = np.hstack((rv[0:1, :].T, tv[-1:, :].T, exact_u[-1:, :].T))
    uu2 = exact_u[-1:, :].T

    # Getting the initial condition (Phi=0)
    xx3 = np.hstack((rv[0:1, :].T, tv[0:1, :].T, exact_u[0:1, :].T))
    uu3 = exact_u[0:1, :].T

    # Getting the end boundary conditions (r=10)
    xx4 = np.hstack((rv[:, -1:], tv[:, -1:], exact_u[:, -1:]))
    uu4 = exact_u[:, -1:]
    # Stacking all boundary condition the exact value in single variable
    x_b0_train = np.vstack([xx1, xx2, xx3])
    x_b_train = np.vstack([xx4])
    u_b0_train = np.vstack([uu1, uu2, uu3])
    u_b_train = np.vstack([uu4])
    num_cycles = np.random.randint(0, 4)  
    mask1 = []
    location = []
    radius = []
    for _ in range(num_cycles):
        # Random center and radius for the cycle
        c1 = np.random.uniform(1, 2)
        if _==0:
            c2 = np.random.uniform(0, np.pi/2)
        elif _==1:
            c2 = np.random.uniform(np.pi/2, np.pi)
        else:
            c2 = np.random.uniform(np.pi/6, np.pi/3)
        cycle_center = np.array([c1, c2]) 
        cycle_radius = np.random.uniform(0.5, 1)
        location.append(cycle_center.tolist())
        radius.append(cycle_radius)

        # Create a circular mask for the cycle
        mask = (rv - cycle_center[0])**2 + (tv - cycle_center[1])**2 <= cycle_radius**2
        mask1.append(mask.tolist())

        # Apply the mask to the array (reduce the values by 1)
        exact_u[mask] -=  3

    x_b_train, _, y_b_train, _ = train_test_split(x_b_train, u_b_train.reshape(-1, 1), test_size=b_size)
    x_b0_train, _, y_b0_train, _ = train_test_split(x_b0_train, u_b0_train.reshape(-1, 1), test_size=b_size)
    x_u_train, _, y_u_train, _ = train_test_split(x_star, exact_u.reshape(-1, 1), test_size=u_size)
    
    get_data = {
    "x_star": x_star.tolist(),
    "exact_u": exact_u.tolist(),
    "x_u_train": x_u_train.tolist(),
    "y_u_train": y_u_train.tolist(),
    "x_b0_train": x_b0_train.tolist(),
    "y_b0_train": y_b0_train.tolist(),
    'x_b_train': x_b_train.tolist(),
    'y_b_train': y_b_train.tolist(),
    'mask1': mask1,
    'num_circle': num_cycles,
    'location': location,
    'radius': radius
    }
    file_path = "get_data.json"
    with open(file_path, "w") as json_file:
        json.dump(get_data, json_file, indent=4)
    
