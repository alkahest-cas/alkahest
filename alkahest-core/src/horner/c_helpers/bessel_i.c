#ifndef ALKAHEST_BESSEL_I_DEFINED
#define ALKAHEST_BESSEL_I_DEFINED
/* Modified Bessel function of the first kind I_n(x) for INTEGER order
 * |n| <= 10000 (NAN otherwise) and any real x; I_n(+-INFINITY) = +-INFINITY by
 * parity, and +INFINITY once the true value exceeds DBL_MAX (|x| > ~713).
 * |x| <= 2: power series; otherwise Miller's backward recurrence normalised by
 * exp(x) = I_0 + 2 sum I_k.  Relative error <= 3e-15 for orders 0..50. */
static double alkahest_bessel_i(double order, double x) {
    int n, neg = 0, m, k;
    double r;
    if (isnan(order) || isnan(x) || order != floor(order) || fabs(order) > 10000.0)
        return NAN;
    n = (int)order;
    if (n < 0) n = -n;
    if (x < 0.0) { x = -x; neg = n & 1; }
    if (isinf(x) || x > 714.0) return neg ? -INFINITY : INFINITY;
    if (x == 0.0) return n == 0 ? 1.0 : 0.0;
    if (x <= 2.0) {
        double h = 0.5 * x, t = 1.0, q = h * h, sum;
        for (k = 1; k <= n; ++k) t *= h / k;
        sum = t;
        for (k = 1; k < 60; ++k) {
            t *= q / ((double)k * (double)(k + n));
            sum += t;
            if (t <= 1e-18 * sum) break;
        }
        r = sum;
    } else {
        double hi = 0.0, lo = 1.0, norm = 0.0, rn = 0.0, tox = 2.0 / x, e;
        m = n > (int)sqrt(100.0 * x) ? n : (int)sqrt(100.0 * x);
        m += 40;
        for (k = m; k >= 0; --k) {
            if (k == n) rn = lo;
            norm += (k == 0) ? lo : 2.0 * lo;
            if (k > 0) {
                double nx = hi + (double)k * tox * lo;
                hi = lo;
                lo = nx;
                if (lo > 1e250) {
                    lo *= 1e-250; hi *= 1e-250; norm *= 1e-250; rn *= 1e-250;
                }
            }
        }
        e = exp(0.5 * x);
        r = e * ((rn / norm) * e);
    }
    return neg ? -r : r;
}
#endif
