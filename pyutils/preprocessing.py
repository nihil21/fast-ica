from __future__ import annotations

import numpy as np


def center_signal(x: np.ndarray) -> np.ndarray:
    """Center signal.

    Parameters
    ----------
    x: np.ndarray
        Raw signal with shape (n_channels, n_samples).

    Returns
    -------
    x_center: np.ndarray
        Centered signal with shape (n_channels, n_samples).
    """

    mean = np.mean(x, axis=1, keepdims=True)
    xc = x - mean

    return xc


def whiten_signal(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Whiten signal using ZCA algorithm.

    Parameters
    ----------
    x: np.ndarray
        Raw signal with shape (n_channels, n_samples).

    Returns
    -------
    x_white: np.ndarray
        Whitened signal with shape (n_channels, n_samples).
    white_mtx: np.ndarray
        Whitening matrix.
    """

    # Compute SVD of correlation matrix
    cov_mtx = np.cov(x)
    u, s, vh = np.linalg.svd(cov_mtx)
    # Compute whitening matrix
    eps = 1e-10
    d = np.diag(1.0 / (np.sqrt(s) + eps))
    white_mtx = u @ d @ vh
    xw = white_mtx @ x

    return xw, white_mtx
