#ifndef ALKAHEST_EXPINT_CHI_DEFINED
#define ALKAHEST_EXPINT_CHI_DEFINED
/* Hyperbolic cosine integral Chi(x) = gamma + log x + int_0^x (cosh t - 1)/t dt
 * for x > 0; Chi(0) = -INFINITY, NAN for x < 0 (complex).  Power series for
 * x <= 10, (Ei(x) - E1(x)) / 2 beyond.  Relative error <= 3e-15 except near the
 * zero at x = 0.5238..., where the error is absolute (~3e-16). */
static double alkahest_chi(double x) {
    int k;
    if (isnan(x) || x < 0.0) return NAN;
    if (x == 0.0) return -INFINITY;
    if (x <= 10.0) {
        double t = x * x, term = 1.0, sum = 0.0;
        for (k = 1; k < 200; ++k) {
            term *= t / ((2.0 * k - 1.0) * (2.0 * k));
            sum += term / (2.0 * k);
            if (term < 1e-18 * (sum + 1.0)) break;
        }
        return 0.5772156649015329 + log(x) + sum;
    }
    return 0.5 * (alkahest_ei(x) - alkahest_e1_pos(x));
}
#endif
