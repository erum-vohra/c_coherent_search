import ctypes
import os
import numpy as np
from numpy import ctypeslib as npct

# load shared library
_HERE = os.path.dirname(os.path.abspath(__file__))
_lib  = ctypes.CDLL(os.path.join(_HERE, "fourierinterp.so"))

class _cplexd(ctypes.Structure):
    ''' double complex in C (two doubles: real, imag) '''
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

    def to_python(self):
        return complex(self.real, self.imag)

_c64_1d   = npct.ndpointer(dtype=np.complex64,  flags="C_CONTIGUOUS")
_c128_1d  = npct.ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS")
_f64_1d   = npct.ndpointer(dtype=np.float64,    flags="C_CONTIGUOUS")

_lib.next_pow_of_2.restype  = ctypes.c_int
_lib.next_pow_of_2.argtypes = [ctypes.c_int]

_lib.get_finterp_coeffs.restype  = None
_lib.get_finterp_coeffs.argtypes = [
    ctypes.c_double,
    ctypes.c_int,  
    _c128_1d,      
]

_lib.get_nearby_fourier_bins.restype  = None
_lib.get_nearby_fourier_bins.argtypes = [
    ctypes.c_double,  
    _c64_1d,           
    ctypes.c_int64,         
    ctypes.c_int,          
    _c128_1d,               
]

_lib.fourier_interp.restype  = _cplexd
_lib.fourier_interp.argtypes = [
    ctypes.c_double, 
    _c64_1d,        
    ctypes.c_int64,
    ctypes.c_int,     
]

_lib.get_finterp_multi_coeffs.restype  = None
_lib.get_finterp_multi_coeffs.argtypes = [
    _f64_1d,      
    ctypes.c_int,     
    ctypes.c_int,     
    _c128_1d,       
]

_lib.finterp_multi.restype  = None
_lib.finterp_multi.argtypes = [
    _f64_1d,           
    ctypes.c_int,
    _c64_1d,            
    ctypes.c_int64,      
    ctypes.c_int,       
    ctypes.c_void_p,     
    _c128_1d,              
]

_lib.get_finterp_FFT_coeffs.restype  = None
_lib.get_finterp_FFT_coeffs.argtypes = [
    ctypes.c_int,      
    ctypes.c_int,  
    ctypes.c_int,     
    _c128_1d,          
]

_lib.finterp_FFT.restype  = None
_lib.finterp_FFT.argtypes = [
    ctypes.c_int,           
    ctypes.c_int,           
    ctypes.c_int,           
    _c64_1d,                
    ctypes.c_int64,         
    ctypes.c_int,     
    ctypes.c_void_p,
    _c128_1d,
]

def next_pow_of_2(n: int) -> int:
    return _lib.next_pow_of_2(int(n))


def get_finterp_coeffs(dr: float, m: int) -> np.ndarray:
    assert m % 2 == 0
    coeffs = np.empty(m, dtype=np.complex128)
    _lib.get_finterp_coeffs(float(dr), int(m), coeffs)
    return coeffs


def get_nearby_fourier_bins(r: float, ft: np.ndarray, m: int) -> np.ndarray:
    ft = np.ascontiguousarray(ft, dtype=np.complex64)
    out = np.empty(m, dtype=np.complex128)
    _lib.get_nearby_fourier_bins(float(r), ft, len(ft), int(m), out)
    return out


def fourier_interp(r: float, ft: np.ndarray, m: int) -> complex:
    ft = np.ascontiguousarray(ft, dtype=np.complex64)
    result = _lib.fourier_interp(float(r), ft, len(ft), int(m))
    return result.to_python()


def get_finterp_multi_coeffs(rs: np.ndarray, m: int) -> np.ndarray:
    rs = np.ascontiguousarray(rs, dtype=np.float64)
    coeffs = np.empty((len(rs), m), dtype=np.complex128)
    _lib.get_finterp_multi_coeffs(rs, len(rs), int(m), coeffs)
    return coeffs


def finterp_multi(rs: np.ndarray, ft: np.ndarray, m: int,
                  coeffs: np.ndarray | None = None) -> np.ndarray:
    rs = np.ascontiguousarray(rs, dtype=np.float64)
    ft = np.ascontiguousarray(ft, dtype=np.complex64)
    out = np.empty(len(rs), dtype=np.complex128)

    if coeffs is not None:
        coeffs = np.ascontiguousarray(coeffs, dtype=np.complex128)
        coeffs_ptr = coeffs.ctypes.data_as(ctypes.c_void_p)
    else:
        coeffs_ptr = ctypes.c_void_p(None) 

    _lib.finterp_multi(rs, len(rs), ft, len(ft), int(m), coeffs_ptr, out)
    return out


def get_finterp_FFT_coeffs(numbetween: int, m: int, fftlen: int) -> np.ndarray:
    assert m % 2 == 0
    assert fftlen >= numbetween * m
    coeffs = np.empty(fftlen, dtype=np.complex128)
    _lib.get_finterp_FFT_coeffs(int(numbetween), int(m), int(fftlen), coeffs)
    return coeffs


def finterp_FFT(lobin: int, numbins: int, numbetween: int,
                ft: np.ndarray, m: int,
                coeffs: np.ndarray | None = None) -> np.ndarray:
    ft = np.ascontiguousarray(ft, dtype=np.complex64)
    out = np.empty(numbins * numbetween, dtype=np.complex128)

    if coeffs is not None:
        coeffs = np.ascontiguousarray(coeffs, dtype=np.complex128)
        coeffs_ptr = coeffs.ctypes.data_as(ctypes.c_void_p)
    else:
        coeffs_ptr = ctypes.c_void_p(None) 

    _lib.finterp_FFT(int(lobin), int(numbins), int(numbetween),
                     ft, len(ft), int(m), coeffs_ptr, out)
    return out