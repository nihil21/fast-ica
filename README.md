# fast-ica
C implementation of FastICA algorithm: the program generates three signals (a sine wave, a square wave and a sawtooth wave), mixes them and tries to reconstruct the original signals using FastICA.

### Build:
```
mkdir build  && cd build
cmake ..
make
```

### Usage:
```
./fast_ica [STRATEGY [G_FUNCTION [N_SAMPLES [SAMPLING_WINDOW_SIZE [THRESHOLD [ MAX_ITER [VERBOSE]]]]]]
```
where:
- `strategy` can be `0` (*Deflation*, default) or `1` (*Parallel*);
- `g_function` can be `0` (*LogCosh*, default), `1` (*Exp*) or `2` (*Cube*);
- `n_samples` must be a non-negative integer defining the number of samples in each time signal;
- `sampling_window_size` must be a non-negative float defining the range in which the samples are taken;
- `threshold` must be a non-negative float defining the convergence threshold for FastICA;
- `threshold` must be a non-negative integer defining the maximum number of iteration of FastICA.

The program writes the original signals, the mixtures and the reconstructed signals in three binary files (`S.bin`, `X.bin` and `S_res.bin`, respectively).
Such binary files can be read and plotted using the `pyutils/plot_signals.py` Python scripts, which accepts an argument describing the precision of the data type, either `float` (default) for 32 bit or `double` for 64 bit.

Moreover, in the `pyutils` folder there's a Jupyter notebook and a `fast_ica.py` modules which implement the same algorithms in Python.