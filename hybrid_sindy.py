"""Hybrid-SINDy

    Mangan, N. M., Askham, T., Brunton, S. L., Kutz, J. N., & Proctor, J. L. (2019). 
    Model selection for hybrid dynamical systems via sparse regression. 
    Proceedings of the Royal Society A, 475(2223), 20180534.
"""
from typing import List

import tqdm

import numpy as np
import pysindy as sp

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator

from scipy.integrate import solve_ivp


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


def augment_derivatives(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    """augment state trajectories with an estimate of their gradient
    
        Y = [X dX]
    """
    fd = sp.FiniteDifference()
    Xd = fd._differentiate(X, t)
    return np.hstack((X, Xd)) 


def hybrid_sindy(
        X: List[np.ndarray],
        t: List[np.ndarray], 
        test_size=0.33,
        n_neighbors=30,
        n_neighbors_validation=5,
        aic_rejection = 3.0,
    ):

    # assert that the arrays are parallel
    assert len(X) == len(t)
    assert np.all([len(Xi) == len(ti) for Xi, ti in zip(X, t)])

    # all state dimensions should be the same
    assert len(set([Xi.shape[1] for Xi in X])) == 1

    # augment with gradient estimates
    _Ys = []
    for ti, Xi in zip(t, X):
        _Ys.append(augment_derivatives(Xi, ti))
        d = Xi.shape[1]
    Y = np.vstack(_Ys)

    # split into test and validation dataset
    Y_T, Y_V = train_test_split(Y, test_size=test_size, random_state=42)

    # get knn clusters for training
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(Y_T)
    _, indices = nbrs.kneighbors(Y_T)
    centroids = np.array([np.sum(Y_T[idx], axis=0) for idx in indices])

    # get knn for validation
    nbrs_v = NearestNeighbors(n_neighbors=n_neighbors_validation, algorithm='ball_tree').fit(Y_V)
    _, indices_v = nbrs_v.kneighbors(centroids)

    # collect models
    models = []
    for idx, idx_v, centroid in tqdm.tqdm(zip(indices, indices_v, centroids), total=len(indices)):
        y = Y_T[idx]
        y_v = Y_V[idx_v]
        
        # learn SINDy model
        try:
            sindy_model = sp.SINDy()
            sindy_model.fit(y[:, :d], x_dot = y[:, d:])
            sindy_model.print()
        except Exception as exc:
            print(f"Failure fitting SINDy...")
            continue

        # get expected loss
        losses = []
        for y0 in y_v:
            x0 = y0[:d]
            # TODO: check change point...
            time_end = 0.4
            t_eval = np.linspace(0, time_end, 10)
            try:
                y_pred = sindy_model.simulate(x0, t_eval)
                sol_true = solve_ivp(model, [0, time_end], x0, t_eval=t_eval, max_step=0.1)
                losses.append((1/len(t_eval))*np.sum((sol_true.y.T - y_pred)**2))
            except Exception as exc:
                print(f"Failure validating SINDy...")
                pass

        if len(losses) == 0:
            continue

        expected_loss = np.mean(losses)

        # compute AIC
        K = len(y_v)
        k = len(sindy_model.feature_library.transform(y0[:d]))
        aic =  K*np.log(expected_loss/ K) + 2*k

        # only consider models with suitable aic value
        if aic < aic_rejection:
            models.append(sindy_model)

