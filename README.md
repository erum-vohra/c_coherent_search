# Coherent Search (in C!)

This program implements coherent search for pulsars in C, based on the Python implemetation found at https://github.com/scottransom/coherent_search/.
The equations this project is based on can be found in [Ransom, Eikenberry, and Middleditch (2002)]{https://arxiv.org/pdf/astro-ph/0204349}. Several filetypes in this project are types used in [PRESTO]{https://github.com/scottransom/presto}, a pulsar analysis toolkit.

The goal of this repository is to quickly do complex Fourier interpolation and use that to perform coherent harmonic summing pulsation searches of FFT to look for pulsar candidates.

To use this repository, you will need to have FFTW and tqdm installed, which you can do with:
```
conda install fftw
conda install tqdm
```

Next, the most computationally-expensive (in theory) functions are written in C, so you need to build the C files. Within the `src` directory:
```
gcc -O2 -march=native -ffast-math -shared -fPIC -o fourierinterp.so fourierinterp.c -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lfftw3 -lm
```
This is the command that worked for me, but your milage may vary and you may need different flags.

You can run `c_coherent_search` through the command line in the following format.
```
coherent_search.py [-o OUTPUT_FILE_NAME] [flags] [FFTFILE]
```
At minimum, you need a PRESTO fftfile to be searched. All of the flag options are in the wiki and example fftfiles with simulated data are in the examples. Generally, a fftfile must be barycentered and have interference and rednoise removed before using this program.