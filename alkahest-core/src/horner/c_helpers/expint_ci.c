#ifndef ALKAHEST_EXPINT_CI_DEFINED
#define ALKAHEST_EXPINT_CI_DEFINED
/* Cosine integral Ci(x) = -int_x^inf cos t / t dt for x > 0; Ci(0) = -INFINITY,
 * Ci(INFINITY) = 0, NAN for x < 0 (complex).  Absolute error <= 3e-16 for
 * x <= 1e6 (relative accuracy is lost only at the zeros, x = 0.6165..., ...). */
static double alkahest_ci(double x) {
    double ci, si;
    if (isnan(x) || x < 0.0) return NAN;
    if (x == 0.0) return -INFINITY;
    alkahest_cisi_pos(x, &ci, &si);
    return ci;
}
#endif
