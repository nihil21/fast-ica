from __future__ import annotations

import logging
from functools import partial

import numpy as np
from scipy.signal import find_peaks

from preprocessing import center_signal, whiten_signal


def _logcosh(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LogCosh function.

    Parameters
    ----------
    x: np.ndarray
        Input data.

    Returns
    -------
    gx: np.ndarray
        LogCosh output.
    gx_prime: np.ndarray
        LogCosh first derivative.
    gx_sec: np.ndarray
        LogCosh second derivative.
    """

    alpha = 1.0
    # Compute G
    gx = 1 / alpha * np.log(np.cosh(alpha * x))
    # Compute G'
    gx_prime = np.tanh(alpha * x)
    # Compute G''
    gx_sec = alpha * (1 - gx_prime**2)

    return gx, gx_prime, gx_sec


def _exp(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exp function.

    Parameters
    ----------
    x: np.ndarray
        Input data.

    Returns
    -------
    gx: np.ndarray
        Exp output.
    gx_prime: np.ndarray
        Exp derivative.
    """

    gx = -np.exp(-(x**2) / 2)
    # Compute G

    # Compute G'
    gx_prime = -x * gx
    # Compute G''
    gx_sec = (x**2 - 1) * gx

    return gx, gx_prime, gx_sec


def _skew(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Skewness function.

    Parameters
    ----------
    x: np.ndarray
        Input data.

    Returns
    -------
    gx: np.ndarray
        Cubic output.
    gx_prime: np.ndarray
        Cubic derivative.
    """

    # Compute G
    gx = x**3 / 3
    # Compute G'
    gx_prime = x**2
    # Compute G''
    gx_sec = 2 * x

    return gx, gx_prime, gx_sec


def _symmetric_decorrelation(w: np.ndarray) -> np.ndarray:
    # Compute eigenvectors and eigenvalues
    eig_vals, eig_vecs = np.linalg.eigh(np.dot(w, w.T))
    # Construct diagonal matrix of eigenvalues
    eps = 1e-10
    d = np.diag(1.0 / (np.sqrt(eig_vals) + eps))

    # Compute new weight matrix
    w = eig_vecs @ d @ eig_vecs.T @ w

    return w


def _symmetric_decorrelation_approx(w: np.ndarray, conv_th: float, max_iter: int) -> np.ndarray:
    w /= np.linalg.norm(w)
    iter_idx = 0
    while iter_idx < max_iter:
        w = 3 / 2 * w - 1 / 2 * (w @ w.T @ w)
        wwt = w @ w.T
        eye = np.eye(*wwt.shape)
        distance = np.linalg.norm(eye - wwt)
        if distance < conv_th:
            logging.info("Symmetric de-correlation approximation converged "
                        f"after {iter_idx} iterations with distance = {distance:.3e}")
            break
        iter_idx += 1

    return w


class FastICA:
    """Perform blind source separation of signals via FastICA.

    Parameters
    ----------
    n_comp: int
        N. of components to extract.
    fs: float
        Sampling frequency of the signal.
    strategy: str, default="deflation"
        Strategy for de-correlation.
    g_func: str, default="logcosh"
        Contrast function for FastICA.
    conv_th: float, default=1e-4
        Threshold for convergence.
    max_iter: int, default=100
        Maximum n. of iterations.
    min_spike_distance: float, default=10
        Minimum distance between two spikes.
    seed: Optional[int], default=None
        Seed for the internal PRNG.

    Attributes
    ----------
    _is_calibrated: bool
        Whether the instance is calibrated or not.
    _n_comp: int
        N. of components to extract.
    _strategy: str
        Strategy for de-correlation.
    _g: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Contrast function for FastICA.
    _conv_th: float, default=1e-4
        Threshold for convergence.
    _max_iter: int, default=100
        Maximum n. of iterations.
    _prng: np.Generator, default=None
        Actual PRNG.
    _white_mtx: np.ndarray | None
        Whitening matrix.
    _sep_mtx: np.ndarray | None
        Separation matrix.
    """

    def __init__(
        self,
        n_comp: int,
        fs: float,
        strategy: str = "deflation",
        g_func: str = "logcosh",
        conv_th: float = 1e-4,
        max_iter: int = 100,
        seed: int | None = None,
    ):
        # Dictionary for contrast functions
        g_dict = {"logcosh": _logcosh, "exp": _exp, "skew": _skew}

        # Parameter check
        assert n_comp > 0, "The n. of components must be positive."
        assert strategy in (
            "deflation",
            "parallel"
        ), f"Strategy can be either \"deflation\" or \"parallel\": the provided one was {strategy}."
        assert g_func in [
            "logcosh",
            "exp",
            "skew",
        ], f'Contrast function can be either "logcosh", "exp" or "skew": the provided one was {g_func}.'
        assert conv_th > 0, "Convergence threshold must be positive."
        assert max_iter > 0, "The maximum n. of iterations must be positive."

        # External parameters
        self._n_comp = n_comp
        self._fs = fs
        self._strategy = strategy
        self._g = g_dict[g_func]
        self._conv_th = conv_th
        self._max_iter = max_iter
        self._prng = np.random.default_rng(seed)

        # Internal parameters to keep track of
        self._is_calibrated = False  # state of the object (calibrated/not calibrated)
        self._white_mtx = None  # whitening matrix
        self._sep_mtx = None  # separation matrix
    
    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self, x: np.ndarray, approx: bool = True) -> np.ndarray:
        """Calibrate instance on the given signal and return the extracted sources.

        Parameters
        ----------
        x: np.ndarray
            Raw signal with shape (n_channels, n_samples).
        approx: bool, default=True
            Whether to use the approximate symmetric de-correlation (relevant only for parallel strategy).

        Returns
        -------
        sources: np.ndarray
            Source signal with shape (n_components, n_samples).
        """
        assert not self._is_calibrated, "Instance already calibrated."

        # 1. Preprocessing: centering and whitening
        xw = self._preprocessing_train(x)

        # 2. Decomposition: FastICA
        self._sep_mtx = np.zeros(shape=(xw.shape[0], 0))  # initialize separation matrix
        if self._strategy == "deflation":
            self._decomposition_def(xw)
        else:
            self._decomposition_par(xw, approx)

        # Set instance to trained
        self._is_calibrated = True

        return self._sep_mtx.T @ xw

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Decompose given data using pre-computed parameters.

        Parameters
        ----------
        x: np.ndarray
            Raw signal with shape (n_channels, n_samples).

        Returns
        -------
        sources: np.ndarray
            Source signal with shape (n_components, n_samples).
        """
        assert self._is_calibrated, "The instance must be calibrated first."

        # 1. Preprocessing: centering and whitening (with pre-computed matrix)
        xw = self._preprocessing_inference(x)

        return self._sep_mtx.T @ xw

    def compute_negentropy(self, sources: np.ndarray) -> np.ndarray:
        """Given the set of sources, compute their neg-entropy.

        Parameters
        ----------
        sources: np.ndarray
            Source signals with shape (n_components, n_samples).

        Returns
        -------
        negentropy: np.ndarray
            Neg-entropy value for each source.
        """
        g_sources, _, _ = self._g(sources)
        g_std, _, _ = self._g(self._prng.standard_normal(size=sources.shape))
        return np.square(np.mean(g_sources, axis=1) - np.mean(g_std, axis=1))

    def reset(self) -> None:
        """Reset the internal state of the instance."""
        self._is_calibrated = False
        self._white_mtx = None
        self._sep_mtx = None

    def _preprocessing_train(self, x: np.ndarray) -> np.ndarray:
        """Preprocess raw signal (train mode).

        Parameters
        ----------
        x: np.ndarray
            Raw signal with shape (n_channels, n_samples).

        Returns
        -------
        xw: np.ndarray
            Preprocessed signal with shape (n_channels, n_samples).
        """
        # 1. Centering
        xc = center_signal(x)
        # 2. Whitening
        xw, self._white_mtx = whiten_signal(xc)

        return xw

    def _preprocessing_inference(self, x: np.ndarray) -> np.ndarray:
        """Preprocess raw signal (inference mode).

        Parameters
        ----------
        x: np.ndarray
            Raw signal with shape (n_channels, n_samples).

        Returns
        -------
        xw: np.ndarray
            Preprocessed signal with shape (n_channels, n_samples).
        """
        # 1. Centering
        xc = center_signal(x)
        # 2. Whitening
        xw = self._white_mtx @ xc

        return xw

    def _decomposition_def(self, xw: np.ndarray) -> None:
        """Perform decomposition (deflation strategy).

        Parameters
        ----------
        xw: np.ndarray
            Pre-whitened signal with shape (n_channels, n_samples).
        """
        # Initialize separation vector indices
        wi_init_idx = self._init_indices(xw)

        logging.info(f"Deflation strategy")
        for i in range(self._n_comp):
            logging.info(f"----- SOURCE {i + 1} -----")

            # Initialize separation vector
            wi_init = None
            fraction_peaks = 0.75
            if i < fraction_peaks * self._n_comp:
                wi_idx = self._prng.choice(wi_init_idx, size=1)[0]
                logging.info(
                    f"Initialization done using index {wi_idx} with "
                    f"value {np.square(xw[:, wi_idx]).sum(axis=0):.3e}."
                )
                wi_init = xw[:, wi_idx]

            # Run FastICA for the i-th unit
            wi, converged = self._fast_ica_def(xw, wi_init)
            if not converged:
                logging.info("FastICA didn't converge, reinitializing...")
                continue

            # Add wi to separation matrix
            self._sep_mtx = np.concatenate(
                [self._sep_mtx, wi.reshape(-1, 1)], axis=1
            )
    

    def _decomposition_par(self, xw: np.ndarray, approx: bool = True) -> None:
        """Perform decomposition (parallel strategy).

        Parameters
        ----------
        xw: np.ndarray
            Pre-whitened signal with shape (n_channels, n_samples).
        approx: bool, default=True
            Whether to use the approximate symmetric de-correlation.
        """
        # Initialize separation vector indices
        wi_init_idx = self._init_indices(xw)

        logging.info(f"Parallel strategy")

        # Initialize separation vectors
        w_init = np.zeros(shape=(self._n_comp, xw.shape[0]))
        fraction_peaks = 0.75
        for i in range(self._n_comp):
            if i < fraction_peaks * self._n_comp:
                wi_idx = self._prng.choice(wi_init_idx, size=1)[0]
                w_init[i] = xw[:, wi_idx]
            else:
                w_init[i] = self._prng.standard_normal(size=(xw.shape[0], self._n_comp))
                
        # Run FastICA for every unit
        w, converged = self._fast_ica_par(xw, w_init, approx)
        if not converged:
            logging.info("FastICA didn't converge.")

        # Set w as separation matrix
        self._sep_mtx = w.T

    
    def _init_indices(self, xw: np.ndarray) -> np.ndarray:
        """Get initial estimation for separation vectors.

        Parameters
        ----------
        xw: np.ndarray
            Pre-whitened signal with shape (n_channels, n_samples).

        Returns
        -------
        wi_init_idx: np.ndarray
            Indices of the whitened data to use as initial estimation of separation vectors.
        """
        x_sq = np.square(xw).sum(axis=0)
        peaks, _ = find_peaks(x_sq)
        peak_heights = x_sq[peaks]
        sorted_peaks_idx = np.argsort(peak_heights)[::-1]
        # Find peaks in the whitened data to use as initialization points for the fixed-point algorithm
        max_wi_indices = peaks[sorted_peaks_idx]

        # Initialize according to a random peak in the top 25%
        top_max_wi_indices = len(max_wi_indices) // 4
        if top_max_wi_indices < 4 * self._n_comp:
            top_max_wi_indices = 4 * self._n_comp
        return max_wi_indices[:top_max_wi_indices]

    def _fast_ica_def(
        self, xw: np.ndarray, wi_init: np.ndarray | None
    ) -> tuple[np.ndarray, bool]:
        """FastICA one-unit implementation (deflation).

        Parameters
        ----------
        xw: np.ndarray
            Pre-whitened signal with shape (n_channels, n_samples).
        wi_init: np.ndarray
            Initial separation vector.

        Returns
        -------
        wi: np.ndarray
            Separation vector with shape (n_channels,).
        converged: bool
            Whether FastICA converged or not.
        """
        n_channels = xw.shape[0]

        # Initialize separation vector
        if wi_init is not None:
            wi = wi_init
        else:
            wi = self._prng.standard_normal(size=(n_channels,), dtype=float)
        wi /= np.linalg.norm(wi)

        # Iterate until convergence or max_iter are reached
        iter_idx = 0
        converged = False
        while iter_idx < self._max_iter:
            # Compute new weight
            wi_new = self._fast_ica_iter(wi, xw)
            # De-correlate and normalize
            wi_new -= np.dot(self._sep_mtx @ self._sep_mtx.T, wi_new)
            wi_new /= np.linalg.norm(wi_new)

            # Compute distance
            distance = 1 - abs(np.dot(wi_new, wi).item())
            # logging.info(f"FastICA iteration {iter_idx}: {distance:.3e}.")
            # Update separation vector
            wi = wi_new
            # Update iteration count
            iter_idx += 1

            # Check convergence
            if distance < self._conv_th:
                converged = True
                logging.info(
                    f"FastICA converged after {iter_idx} iterations, the distance is: {distance:.3e}."
                )
                break

        return wi, converged
    
    def _fast_ica_par(
        self, xw: np.ndarray, w: np.ndarray, approx: bool = True
    ) -> tuple[np.ndarray, bool]:
        """FastICA one-unit implementation (parallel).

        Parameters
        ----------
        xw: np.ndarray
            Pre-whitened signal with shape (n_channels, n_samples).
        w: np.ndarray
            Initial separation matrix.
        approx: bool, default=True
            Whether to use the approximated symmetric deflation strategy or the precise one (more expensive).

        Returns
        -------
        w: np.ndarray
            Separation matrix with shape (n_channels, n_components).
        converged: bool
            Whether FastICA converged or not.
        """
        decorr_fun = partial(
            _symmetric_decorrelation_approx,
            conv_th=self._conv_th,
            max_iter=self._max_iter
        ) if approx else _symmetric_decorrelation

        # De-correlate
        w = decorr_fun(w)

        # Iterate until convergence or max_iter are reached
        iter_idx = 0
        converged = False
        while iter_idx < self._max_iter:
            # Estimate sources in parallel
            w_new = np.stack([self._fast_ica_iter(w[i], xw) for i in range(self._n_comp)])
            # De-correlate
            w_new = decorr_fun(w_new)

            # Compute distance
            distance = max(abs(abs(np.diag(np.dot(w_new, w.T))) - 1))
            # logging.info(f"FastICA iteration {iter_idx}: {distance:.3e}.")
            # Update separation matrix
            w = w_new
            # Update iteration count
            iter_idx += 1

            # Check convergence
            if distance < self._conv_th:
                converged = True
                logging.info(
                    f"FastICA converged after {iter_idx} iterations, the distance is: {distance:.3e}."
                )
                break

        return w, converged
    
    def _fast_ica_iter(self, wi: np.ndarray, xw: np.ndarray) -> np.ndarray:
        # (n_channels,) @ (n_channels, n_samples) -> (n_samples,)
        _, g_ws_prime, g_ws_sec = self._g(np.dot(wi, xw))
        # (n_channels, n_samples) * (n_samples,) -> (n_channels, 1)
        t1 = (xw * g_ws_prime).mean(axis=-1)
        # E[(n_samples,)] * (n_channels, 1) -> (n_channels, 1)
        t2 = g_ws_sec.mean() * wi
        # Compute new separation vector
        wi_new = t1 - t2

        return wi_new
