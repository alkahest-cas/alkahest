#ifndef ALKAHEST_EXPINT_EI_DEFINED
#define ALKAHEST_EXPINT_EI_DEFINED
/* Exponential integral Ei(x) = PV int_{-inf}^x exp(t)/t dt for real x.
 * Ei(0) = -INFINITY, Ei(+INFINITY) = +INFINITY, Ei(-INFINITY) = 0.
 * x < 0: -E1(-x).  0 < x < 45: power series (all terms positive).  x >= 45:
 * asymptotic series.  Relative error <= 3e-15 except within 1e-2 of the zero
 * at x = 0.3725..., where the error is absolute (~3e-16). */
static double alkahest_ei(double x) {
    int k;
    if (isnan(x)) return NAN;
    if (x == 0.0) return -INFINITY;
    if (x < 0.0) return x == -INFINITY ? 0.0 : -alkahest_e1_pos(-x);
    if (x < 45.0) {
        double term = 1.0, sum = 0.0;
        for (k = 1; k < 200; ++k) {
            term *= x / k;
            sum += term / k;
            if (term / k < 1e-17 * sum) break;
        }
        return 0.5772156649015329 + log(x) + sum;
    } else {
        double term = 1.0, sum = 1.0;
        if (x > 710.0) return INFINITY;
        for (k = 1; k < 100; ++k) {
            double next = term * k / x;
            if (next > term) break;
            term = next;
            sum += term;
            if (term < 1e-17 * sum) break;
        }
        return exp(x) / x * sum;
    }
}
#endif
