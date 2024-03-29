#!/usr/bin/python3
import struct
import sys
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt


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


def plot_signals(s, x, s_) -> None:
    models = [s, x, s_]
    names = [
        "Source signals",
        "Signal mixture",
        "Signals recovered using ICA",
    ]
    _, ax = plt.subplots(nrows=len(models), ncols=1, sharex="all", figsize=(15, 10))

    for i, (model, name) in enumerate(zip(models, names)):
        ax[i].set_title(name)
        ax[i].set_ylabel("Amplitude [a.u.]")
        for sig in model:
            ax[i].plot(sig)
        ax[i].grid()
    plt.xlabel("Time [s]")

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

    s = read_binary("../S.bin", fp)
    x = read_binary("../X.bin", fp)
    s_ = read_binary("../S_.bin", fp)
    plot_signals(s, x, s_)


if __name__ == "__main__":
    main()
