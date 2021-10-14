from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def centering(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=1, keepdims=True)
    xc = x - mean
    return xc, mean


def whitening(x: np.ndarray) -> np.ndarray:
    # Compute covariance matrix of data
    cov_mtx = np.cov(x)

    # Compute eigenvectors and eigenvalues
    eig_vals, eig_vecs = np.linalg.eigh(cov_mtx)
    d = np.diag(1.0 / np.sqrt(eig_vals))  # construct diagonal matrix of eigenvalues

    # Compute whitening matrix
    white_mtx = eig_vecs @ d @ eig_vecs.T

    # White data
    xw = white_mtx @ x

    return xw, white_mtx


def _gram_schmidt_decorrelation(w_i_new: np.ndarray, w: np.ndarray, i: int) -> np.ndarray:
    w_i_new -= w_i_new @ w[:i].T @ w[:i]
    return w_i_new


def _symmetric_decorrelation(w: np.ndarray) -> np.ndarray:
    # Compute eigenvectors and eigenvalues
    eig_vals, eig_vecs = np.linalg.eigh(np.dot(w, w.T))
    d = np.diag(1.0 / np.sqrt(eig_vals))  # construct diagonal matrix of eigenvalues

    # Compute new weight matrix
    w = eig_vecs @ d @ eig_vecs.T @ w

    return w


def _ica_def(xw, g, threshold, max_iter):
    n_units, n_samples = xw.shape

    # Initialize weights randomly
    w = np.random.randn(n_units, n_units)

    # Iterate over units
    for i in range(n_units):
        # Initialize i-th neuron
        w_i = w[i, :].copy()
        w_i /= np.linalg.norm(w_i)

        for _ in range(max_iter):
            # (n_units,) @ (n_units, n_samples) -> (n_samples,)
            ws = w_i @ xw
            g_ws, g_ws_prime = g(ws)
            # E[(n_samples,) * (n_units, n_samples)] -> E[(n_units, n_samples)] -> (n_units,)
            a = xw @ g_ws.T / n_samples
            # E[(n_samples,)] * (n_units,) -> (,) * (n_units,) -> (n_units,)
            b = g_ws_prime.mean() * w_i

            # Compute new weight
            w_i_new = a - b
            # Decorrelate weight
            w_i_new = _gram_schmidt_decorrelation(w_i_new, w, i)
            # Normalize
            w_i_new /= np.linalg.norm(w_i_new)

            # Compute distance
            distance = np.abs(np.dot(w_i, w_i_new) - 1)

            # Update weight
            w_i = w_i_new

            if distance < threshold:
                break

        w[i, :] = w_i
    
    return w


def _ica_par(xw, g, threshold, max_iter):
    n_units, n_samples = xw.shape

    # Initialize weights randomly and decorrelate
    w = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # np.random.randn(n_units, n_units)
    w = _symmetric_decorrelation(w)
    
    for _ in range(max_iter):
        # (n_units, n_units) @ (n_units, n_samples) -> (n_units, n_samples)
        ws = w @ xw
        g_ws, g_ws_prime = g(ws)
        # E[(n_units, 1, n_samples) * (n_units, n_samples)] -> E[(n_units, n_units, n_samples)] -> (n_units, n_units)
        a = (xw * g_ws[:, np.newaxis]).mean(axis=-1)
        
        # E[(n_units, n_samples)] * (n_units, n_units) -> (n_units, 1) * (n_units, n_units) -> (n_units, n_units)
        b = g_ws_prime.mean(axis=-1)[:, np.newaxis] * w
        
        # Compute new weight
        w_new = _symmetric_decorrelation(
            a - b  # (n_units, n_units)
        )  # decorrelate

        # Compute distance
        distance = max(abs(abs(np.diag(np.dot(w_new, w.T))) - 1))
        w = w_new

        if distance < threshold:
            break
    
    return w


def _logcosh(x: np.ndarray):
    alpha = 1.0
    # Compute G
    gx = np.tanh(alpha * x)
    # Compute G'
    gx_prime = alpha * (1 - gx**2)

    return gx, gx_prime


def _exp(x: np.ndarray):
    exp = np.exp(-x**2 / 2)
    # Compute G
    gx = x * exp
    # Compute G'
    gx_prime = (1 - x**2) * exp

    return gx, gx_prime


def _cube(x: np.ndarray):
    # Compute G
    gx = x**3
    # Compute G'
    gx_prime = (3 * x**2)

    return gx, gx_prime


def fast_ica(
    x: np.ndarray,
    whiten: bool = True,
    strategy: str = "parallel",
    g_func: str = "logcosh",
    threshold: float = 1e-4,
    max_iter: int = 5000
):
    # Center and whiten, if required
    if whiten:
        xw, x_mean = centering(x)
        xw, white_mtx = whitening(xw)
    else:
        xw = x.copy()

    # Non-quadratic function G
    g_dict = {
        "logcosh": _logcosh,
        "exp": _exp,
        "cube": _cube
    }

    # Strategy dictionary
    func_dict = {
        "deflation": _ica_def,
        "parallel": _ica_par
    }
    kwargs = {
        "g": g_dict[g_func],
        "threshold": threshold,
        "max_iter": max_iter,
    }

    w = func_dict[strategy](xw, **kwargs)
    
    s = w @ xw
    
    # if whiten:
        # De-whiten mixing matrix
        # a_tilde = np.linalg.inv(w)
        # a = np.linalg.inv(white_mtx) @ a_tilde
        # Add mean
        # s_mean = np.linalg.inv(a) @ x_mean
        # s += s_mean
    
    return s
