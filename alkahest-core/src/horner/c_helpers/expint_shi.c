#ifndef ALKAHEST_EXPINT_SHI_DEFINED
#define ALKAHEST_EXPINT_SHI_DEFINED
/* Hyperbolic sine integral Shi(x) = int_0^x sinh t / t dt, odd.  Power series
 * (all terms positive) for |x| <= 10, (Ei(x) + E1(x)) / 2 beyond; overflows to
 * +-INFINITY like exp.  Relative error <= 3e-15. */
static double alkahest_shi(double x) {
    double ax = fabs(x), r;
    int k;
    if (isnan(x)) return NAN;
    if (x == 0.0) return x;
    if (ax <= 10.0) {
        double t = ax * ax, term = 1.0, sum = 1.0;
        for (k = 1; k < 200; ++k) {
            term *= t / ((2.0 * k) * (2.0 * k + 1.0));
            sum += term / (2.0 * k + 1.0);
            if (term < 1e-17 * sum) break;
        }
        r = ax * sum;
    } else {
        r = 0.5 * (alkahest_ei(ax) + alkahest_e1_pos(ax));
    }
    return x < 0.0 ? -r : r;
}
#endif
