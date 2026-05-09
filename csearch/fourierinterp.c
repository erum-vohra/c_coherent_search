#include "fourierinterp.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>

/* helper function that calculates sinc(x) */
static inline double sinc(double x) {
    if (x == 0.0) return 1.0;
    double px = M_PI * x;
    return sin(px) / px;
}

/* return the smallest power of 2 greater than or equal to n */
int next_pow_of_2(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* compute m Fourier interpolation coeffs for Fourier frequency offset dr */
/* parameters: dr (Fourier frequency offset in bins), m (number of coeffs to compute) */
/* return: coeffs (array of Fourier interpolation coeffs) */
void get_finterp_coeffs(double dr, int m, cplex *coeffs) {
    assert(m % 2 == 0);
    assert(dr >= 0.0 && dr < 1.0);

    for (int ii = 0; ii < m; ii++) {
        double offset = dr - (-(m / 2) + 1 + ii);
        coeffs[ii] = sinc(offset) * cexp(offset * M_PI * I);
    }
}

/* get the Fourier bins around a real-valued Fourier frequency r */
/* parameters: r (real-valued Fourier frequency), ft (Fourier transform array), ft_len (length of ft), m (number of bins to return) */
/* return: out (array of complex Fourier amplitudes) */
void get_nearby_fourier_bins(double r, const cplex_f *ft, int64_t ft_len, int m, cplex *out) {
    assert (m % 2 == 0);
    int r_int = (int)floor(r + 1e-15) + 1;
    int lo = r_int - (m / 2);

    for (int ii = 0; ii < m; ii++) {
        int idx = lo + ii;
        if (idx >= 0 && idx < (int)ft_len)
            out[ii] = (cplex)ft[idx];
        else
            out[ii] = 0.0 + 0.0 * I;
    }
}

/* perform Fourier interpolation at real-valued Fourier frequency r */
/* parameters: r (real-valued Fourier frequency), ft (Fourier transform array), ft_len (length of ft), m (number of interpolation coeffs) */
/* return: interpolated Fourier amplitude at frequency r */
cplex fourier_interp(double r, const cplex_f *ft, int64_t ft_len, int m) {
    assert(r >= 0.0);
    assert(m % 2 == 0);

    cplex *coeffs = malloc(m * sizeof(cplex));
    cplex *bins = malloc(m * sizeof(cplex));
    cplex result = 0.0 + 0.0 * I;

    get_finterp_coeffs(fmod(r, 1.0), m, coeffs);
    get_nearby_fourier_bins(r, ft, ft_len, m, bins);

    for (int ii = 0; ii < m; ii++)
        result += conj(coeffs[ii]) * bins[ii];

    free(coeffs);
    free(bins);

    return result;
}

/* compute Fourier interpolation coeffs for multiple real-valued Fourier frequencies */
/* parameters: rs (real valued Fourier frequencies to interpolate), n_rs (length of rs), m (number of interpolation coeffs) */
/* return: coeffs (array of Fourier interpolation coefficents) */
void get_finterp_multi_coeffs(const double *rs, int n_rs, int m, cplex *coeffs) {
    assert(m % 2 == 0);

    for (int ii = 0; ii < n_rs; ii++) {
        double frac = fmod(rs[ii], 1.0);
        for (int jj = 0; jj < m; jj++) {
            double offset = frac - (-(m / 2) + 1 + jj);
            coeffs[ii * m + jj] = sinc(offset) * cexp(offset * M_PI * I);
        }
    }
}

/* perform Fourier interpolation at multiple real-valued Fourier frequencies */
/* parameters: rs (real valued Fourier frequencies to interpolate), n_rs (length of rs), ft (Fourier transform array), ft_len (length of ft), m (number of interpolation coeffs), e_coeffs (precomputed Fourier interpolation coeffs for rs and m) */
/* return: out (interpolated Fourier amplitudes at frequencies rs) */
void finterp_multi(const double *rs, int n_rs, const cplex_f *ft, int64_t ft_len, int m, const cplex *e_coeffs, cplex *out) {
    cplex *bins = malloc(m * sizeof(cplex));
    cplex *coeffs = NULL;
    int free_coeffs = 0;

    if (e_coeffs != NULL) {
        coeffs = (cplex*)(e_coeffs);
    } else {
        coeffs = malloc((size_t)n_rs * m * sizeof(cplex));
        free_coeffs = 1;
        get_finterp_multi_coeffs(rs, n_rs, m, coeffs);
    }

    for (int ii = 0; ii < n_rs; ii++) {
        get_nearby_fourier_bins(rs[ii], ft, ft_len, m, bins);
        cplex result = 0.0 + 0.0 * I;
        const cplex *row = coeffs + ii * m;
        for (int jj = 0; jj < m; jj++)
            result += conj(row[jj]) * bins[jj];
        out[ii] = result;
    }

    free(bins);
    if (free_coeffs) free(coeffs);
}

/* compute Fourier interpolation coeffs for FFT correlation method */
/* parameters: numbetween (number of interpolated points between each FFT bin),  m (number of interpolation coeffs), fftlen (length of FFT to use) */
/* return: coeffs (FFT's Fourier interpolation coefficents) */
void get_finterp_FFT_coeffs(int numbetween, int m, int fftlen, cplex *coeffs) {
    assert(m % 2 == 0);
    assert(fftlen >= numbetween * m);
    assert(fftlen == next_pow_of_2(fftlen));

    int h_len = numbetween * (m / 2);
    fftw_complex *tmp = fftw_alloc_complex(fftlen);
    memset(tmp, 0, fftlen * sizeof(fftw_complex));

    for (int ii = 0; ii < h_len; ii++) {
        double offset = ii * (1.0 / numbetween);
        double re = sinc(offset) * cos(-M_PI * offset);
        double im = sinc(offset) * sin(-M_PI * offset);
        ((double*)tmp)[2 * ii] = re;
        ((double*)tmp)[(2 * ii) + 1] = im;
    }

    for (int ii = 0; ii < h_len; ii++) {
        double offset = -((h_len - 1 - ii) * (1.0 / numbetween) + (1.0 / numbetween));
        double re = sinc(offset) * cos(-M_PI * offset);
        double im = sinc(offset) * sin(-M_PI * offset);     
        int idx = fftlen - h_len + ii;
        ((double*)tmp)[2 * idx] = re;
        ((double*)tmp)[(2 * idx) + 1] = im;       
    }

    fftw_plan plan = fftw_plan_dft_1d(fftlen, tmp, tmp, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    for (int ii = 0; ii < fftlen; ii++)
        coeffs[ii] = conj(((double*)tmp)[2 * ii] + I * ((double*)tmp)[(2 * ii) + 1]);
    
    fftw_free(tmp);
}

/* perform Fourier interpolation for many frequencies using FFT correlation */
/* parameters: lobin (integer FFT bin num for the lowest return value), numbins (num of returned FFT bins), numbetween (number of interpolated points between each bin), ft (Fourier transform array), ft_len (length of ft), m (number of interpolation coeffs), e_coeffs (precomputed Fourier interpolation coeffs) */
/* return: interpolated Fourier amplitudes at freqs */
void finterp_FFT(int lobin, int numbins, int numbetween, const cplex_f *ft, int64_t ft_len, int m, const cplex *e_coeffs, cplex *out) {
    cplex *coeffs = NULL;
    int free_coeffs = 0;
    int fftlen = next_pow_of_2((numbins + m) * numbetween);
    
    if (e_coeffs != NULL) {
        coeffs = (cplex*)e_coeffs;
    } else {
        coeffs = malloc(fftlen * sizeof(cplex));
        free_coeffs = 1;
        get_finterp_FFT_coeffs(numbetween, m, fftlen, coeffs);
    }

    fftw_complex *arr = fftw_alloc_complex(fftlen);
    memset(arr, 0, fftlen * sizeof(fftw_complex));
    
    int tot = numbins + m;
    double *arr_d = (double*)arr;

    for (int ii = 0; ii < tot; ii++) {
        int s = lobin - (m / 2) + ii;
        double re = 0.0, im = 0.0;
        if (s >= 0 && s < (int)ft_len) {
            re = crealf(ft[s]);
            im = cimagf(ft[s]);
        }
        int idx = ii * numbetween;
        arr_d[2 * idx] = re;
        arr_d[2 * idx + 1] = im;
    }

    fftw_plan forward = fftw_plan_dft_1d(fftlen, arr, arr, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(forward);
    fftw_destroy_plan(forward);

    for (int ii = 0; ii < fftlen; ii++) {
        double ar = arr_d[2*ii] , ai = arr_d[2 * ii + 1];
        double br = creal(coeffs[ii]), bi = cimag(coeffs[ii]);
        arr_d[2 * ii]   = ar * br - ai * bi;
        arr_d[2 * ii + 1] = ar * bi + ai * br;        
    }

    fftw_plan inverse = fftw_plan_dft_1d(fftlen, arr, arr, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(inverse);
    fftw_destroy_plan(inverse);

    int ct = numbins * numbetween;
    for (int ii = 0; ii < ct; ii++) {
        int s = ((m / 2) * numbetween) + ii;
        out[ii] = (arr_d[2 * s] + I * arr_d[2 * s + 1]) * (1.0 / fftlen);
    }

    fftw_free(arr);
    if (free_coeffs) free(coeffs);
}
