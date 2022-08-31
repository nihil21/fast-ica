from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def sine_wave(time: np.ndarray, amp: float, freq: float, phase: float) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * time + phase)


def square_wave(time: np.ndarray, amp: float, freq: float, phase: float) -> np.ndarray:
    return amp * signal.square(2 * np.pi * freq * time + phase)


def sawtooth_wave(time: np.ndarray, amp: float, freq: float, phase: float) -> np.ndarray:
    return amp * signal.sawtooth(2 * np.pi * freq * time + phase)


def plot_signal(
    s: np.ndarray,
    fs: float = 1,
    title: str | None = None,
    fig_size: tuple[int, int] | None = None
) -> None:
    """Plot a signal with multiple channels.

    Parameters
    ----------
    s: np.ndarray
        Signal with shape (n_channels, n_samples).
    fs: float, default=1
        Sampling frequency of the signal.
    title: str | None, default=None
        Title of the whole plot.
    labels: list[tuple[int, int, int]] | None, default=None
        List containing, for each action block, the label of the action together with the first and the last samples.
    fig_size: tuple[int, int] | None, default=None
        Height and width of the plot.
    """
    n_channels, n_samples = s.shape
    x = np.arange(n_samples) / fs

    # Create figure with subplots and shared x-axis
    n_cols = 1
    n_rows = n_channels
    _, ax = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex="all")

    for i in range(n_channels):
        ax[i].plot(x, s[i])
        ax[i].set_ylabel("Voltage [mV]")
    plt.xlabel("Time [s]")

    if title is not None:
        plt.suptitle(title, fontsize="xx-large")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
