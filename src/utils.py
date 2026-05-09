import os
import numpy as np
import src.fourierinterp_ctypes as fi
from pathlib import Path
from typing import Union

class simpleinf:
    "A simple PRESTO .inf file reader (only key params)"

    def __init__(self, inf: Union[str, os.PathLike]) -> None:
        """Initialize a PRESTO .inf file class instance.

        Parameters
        ----------
        inf : file or str or Path
            The PRESTO .inf file to open
        """
        self.inf: os.PathLike = inf if isinstance(inf, os.PathLike) else Path(inf)
        try:
            with open(self.inf, "r") as file:
                for line in file:
                    if line.startswith(" Object being observed"):
                        self.object = line.split("=")[-1].strip()
                        continue
                    if line.startswith(" Epoch"):
                        self.epoch = float(line.split("=")[-1].strip())
                        continue
                    if line.startswith(" Number of bins"):
                        self.N = int(line.split("=")[-1].strip())
                        continue
                    if line.startswith(" Width of each time series bin"):
                        self.dt = float(line.split("=")[-1].strip())
                        continue
                    if line.startswith(" Dispersion measure"):
                        self.DM = float(line.split("=")[-1].strip())
                        continue
        except FileNotFoundError:
            print(f"Error: The .inf file '{self.inf}' was not found.")


class fftfile:
    "A PRESTO FFT file (i.e. with suffix '.fft') and associated metadata"

    def __init__(self, ff: Union[str, os.PathLike]) -> None:
        """Initialize a PRESTO fftfile class instance.

        Parameters
        ----------
        ff : file or str or Path
            The PRESTO .fft file to open
        """
        self.ff: os.PathLike = ff if isinstance(ff, os.PathLike) else Path(ff)
        self.amps = np.memmap(self.ff, dtype=np.complex64)
        self.inf = simpleinf(f"{str(self.ff)[:-4]}.inf")
        self.N: int = self.inf.N
        self.T: float = self.N * self.inf.dt
        self.dereddened = True if "_red.fft" in str(self.ff) else False
        self.detrended = True if self.dereddened else False
        self.DC, self.Nyquist = self.amps[0].real, self.amps[0].imag
        self.df: float = 1.0 / self.T

    @property
    def freqs(self) -> np.ndarray:
        """The frequencies (in Hz) for the FFT amplitudes."""
        self._freqs = np.linspace(0.0, self.N // 2 * self.df, self.N // 2)
        return self._freqs


class FourierInterpolator:
    "Class to perform running Fourier interpolation through a PRESTO FFT file"

    def __init__(
        self,
        ft: fftfile,
        lobin: int,
        numbetween: int,
        m: int,
        fftlen: int,
        coeffs=None,
    ) -> None:
        """Build a Fourier interpolator that will walk through an FFT

        Parameters
        ----------
        ft : utils.fftfile
            A PRESTO FFT file object to interpolate through.
        lobin : int
            The integer FFT bin number for the lowest return value.
        numbetween : int
            The number of interpolated points between each FFT bin.
        m : int
            Number of interpolation coefficients (even).
        fftlen : int
            Length of the FFT to use for correlation (must be >= numbetween * m).
        coeffs : _type_, optional
            Precomputed Fourier interpolation coefficients for numbetween and m, by default None
        """
        self.ft = ft
        self.lobin = lobin
        self.numbetween = numbetween
        self.m = m
        self.fftlen = fftlen
        # This is the number of full FFT bins we will interpolate each time
        self.numbins = (fftlen // numbetween) - m - 1
        self.nextbin = lobin + self.numbins
        if coeffs is None:
            self.coeffs = fi.get_finterp_FFT_coeffs(numbetween, m, fftlen)
        else:
            self.coeffs = coeffs
        self.ftamps = self.get_ftamps(lobin)

    def get_ftamps(self, lobin: int) -> np.ndarray:
        """Get the Fourier-interpolated FFT amplitudes starting at lobin

        Parameters
        ----------
        lobin : int
            The integer FFT bin number for the lowest return value.

        Returns
        -------
        np.ndarray
            The Fourier-interpolated FFT amplitudes starting at lobin.
        """
        self._rs = (
            np.arange(self.numbins * self.numbetween) / self.numbetween + self.lobin
        )
        if lobin + self.numbins + self.m // 2 >= self.ft.N // 2:
            return np.zeros_like(self._rs, dtype=np.complex128)
        else:
            return fi.finterp_FFT(
                lobin,
                self.numbins,
                self.numbetween,
                self.ft.amps,
                self.m,
                coeffs=self.coeffs,
            )

    @property
    def rs(self) -> np.ndarray:
        """The real-valued Fourier frequencies (bins) for the current interpolation"""
        if not hasattr(self, "_rs"):
            self._rs = (
                np.arange(self.numbins * self.numbetween) / self.numbetween + self.lobin
            )
        return self._rs

    def interpolated_ftamps(self, rs: np.ndarray) -> np.ndarray:
        """Return the linear-interpolated Fourier amplitudes at the Fourier frequencies rs

        Parameters
        ----------
        rs : np.ndarray
            Fourier frequencies at which to interpolate the current FFT amplitudes.

        Returns
        -------
        np.ndarray
            The linear-interpolated FFT amplitudes at the given Fourier frequencies rs.
        """
        if rs.max() > self.rs[-1]:
            self.lobin = int(np.floor(rs.min()))
            self.ftamps = self.get_ftamps(self.lobin)
            self.nextbin = self.lobin + self.numbins
        return np.interp(rs, self.rs, self.ftamps)