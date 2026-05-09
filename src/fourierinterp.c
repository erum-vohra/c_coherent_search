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
void get_finterp_multi_coeffs(const double *rs, int n_rs, int m, cplex *coeffs) {
    assert(m % 2 == 0);

    for (int ii = 0; ii < m; ii++) {
        double frac = fmod(rs[ii], 1.0);
        for (int jj = 0; jj < m; jj++) {
            double offset = frac - (-(m / 2) + 1 + jj);
            coeffs[ii * m + jj] = sinc(offset) * cexp(offset * M_PI * I);
        }
    }
}

/* perform Fourier interpolation at multiple real-valued Fourier frequencies */
void finterp_multi(const double *rs, int n_rs, const cplex_f *ft, int64_t ft_len, int m, const cplex *coeffs, cpelx *out) {
    
}

/* compute Fourier interpolation coeffs for FFT correlation method */
void get_finterp_FFT_coeffs(int numbetween, int m, int fftlen, cplex *coeffs) {
    
}

/* perform Fourier interpolation for many frequencies using FFT correlation */
void finterp_FFT(int lobin, int numbins, int numbetween, const cplex *ft, int64_t ft_len, int m, const cplex *coeffs, cplex *out) {
    
}
