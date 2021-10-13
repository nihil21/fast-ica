#!/usr/bin/python3
import struct
import sys
from typing import Dict

from matplotlib import pyplot as plt
import numpy as np


def read_binary(filename: str, fp: Dict) -> np.ndarray:
    with open(filename, "rb") as f:
        # Read first integers
        (n_dim,) = struct.unpack("@i", f.read(4))
        if n_dim == 1:
            # Read first integers
            (l,) = struct.unpack("@i", f.read(4))
            # Read subsequent l double
            tensor = struct.unpack(f"@{l}{fp['type']}", f.read(l * fp["byte"]))
            tensor = np.array(tensor).reshape(l,)
        elif n_dim == 2:
            # Read first two integers
            (h, w) = struct.unpack("@2i", f.read(8))
            # Read subsequent (h * w) double
            tensor = struct.unpack(f"@{h * w}{fp['type']}", f.read(h * w * fp["byte"]))
            tensor = np.array(tensor).reshape(h, w)
        else:
            raise RuntimeError("File format not supported")

    return tensor


def plot_signals(s, x, sr) -> None:
    assert s.shape == x.shape, "Signals and observations should have the same shape"
    assert x.shape == sr.shape, "Observations and reconstructed signals should have the same shape"
    n_comp = s.shape[0]

    plt.figure(figsize=(15, 20))
    for i in range(n_comp):
        plt.subplot(3 * n_comp, 1, i + 1)
        plt.title(f"Observation {i + 1}")
        plt.plot(x[i])
        plt.grid()
    for i in range(n_comp):
        plt.subplot(3 * n_comp, 1, n_comp + i + 1)
        plt.title(f"True signal {i + 1}")
        plt.plot(s[i])
        plt.grid()
    for i in range(n_comp):
        plt.subplot(3 * n_comp, 1, 2 * n_comp + i + 1)
        plt.title(f"Reconstructed signal {i + 1}")
        plt.plot(sr[i])
        plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    fp = {}
    if len(sys.argv) == 1 or (sys.argv == 2 and sys.argv[1] == "float"):
        fp["type"] = "f"
        fp["byte"] = 4
    elif len(sys.argv) == 2 and sys.argv[1] == "double":
        fp["type"] = "d"
        fp["byte"] = 8
    else:
        sys.exit("Usage: plot_signals [float|double] (default: float)")

    s_path, x_path, sr_path = "../S.bin", "../X.bin", "../S_res.bin"
    s = read_binary(s_path, fp)
    x = read_binary(x_path, fp)
    sr = read_binary(sr_path, fp)
    plot_signals(s, x, sr)


if __name__ == "__main__":
    main()
