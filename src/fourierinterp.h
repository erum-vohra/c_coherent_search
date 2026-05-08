#include <stdint.h>
#include <complex.h>

/* types aliases (to make things a little easier to read) */
typedef double complex cplex;
typedef float complex cplex_f;

/* return the smallest power of 2 greater than or equal to n */
int next_pow_of_2(int n);

/* compute m Fourier interpolation coeffs for Fourier frequency offset dr */
void get_finterp_coeffs(double dr, int m, cplex *coeffs);

/* get the Fourier bins around a real-valued Fourier frequency r */
void get_nearby_fourier_bins(double r, const cplex_f *ft, int64_t ft_len, int m, cplex *out);

/* perform Fourier interpolation at real-valued Fourier frequency r */
cplex fourier_interp(double r, const cplex_f *ft, int64_t ft_len, int m);

/* compute Fourier interpolation coeffs for multiple real-valued Fourier frequencies */
void get_finterp_multi_coeffs(const double *rs, int n_rs, int m, cplex *coeffs);

/* perform Fourier interpolation at multiple real-valued Fourier frequencies */
void finterp_multi(const double *rs, int n_rs, const cplex_f *ft, int64_t ft_len, int m, const cplex *coeffs, cpelx *out);

/* compute Fourier interpolation coeffs for FFT correlation method */
void get_finterp_FFT_coeffs(int numbetween, int m, int fftlen, cplex *coeffs);

/* perform Fourier interpolation for many frequencies using FFT correlation */
void finterp_FFT(int lobin, int numbins, int numbetween, const cplex *ft, int64_t ft_len, int m, const cplex *coeffs, cplex *out);
