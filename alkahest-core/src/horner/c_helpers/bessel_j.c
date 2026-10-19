#ifndef ALKAHEST_BESSEL_J_DEFINED
#define ALKAHEST_BESSEL_J_DEFINED
/* Bessel function of the first kind J_n(x) for INTEGER order |n| <= 10000
 * (NAN for a non-integer or larger order) and any real x (J_n(+-INFINITY) = 0).
 * |x| <= 2: power series; 2 < |x| < 25 or n > |x|: Miller's backward
 * recurrence; |x| >= 25: Hankel's asymptotic expansion for J_0, J_1 and the
 * (stable, n <= |x|) forward recurrence.  Error <= 2e-15 absolute for orders
 * 0..50 and |x| <= 1e5, and <= 1e-14 relative to max(|J_n(x)|, sqrt(2/(pi|x|)))
 * hence relative everywhere except within ~1e-3 of a zero of J_n. */
static double alkahest_bessel_j(double order, double x) {
    int n, neg = 0, k;
    double r;
    if (isnan(order) || isnan(x) || order != floor(order) || fabs(order) > 10000.0)
        return NAN;
    n = (int)order;
    if (n < 0) { n = -n; neg = n & 1; }
    if (isinf(x)) return 0.0;
    if (x < 0.0) { x = -x; if (n & 1) neg ^= 1; }
    if (x == 0.0) return n == 0 ? 1.0 : 0.0;
    if (x <= 2.0) {
        r = alkahest_bessel_jseries(n, x);
    } else if (x >= 25.0 && (double)n <= x) {
        double j0, j1, y0, y1, a, b;
        alkahest_bessel_asym01(x, &j0, &j1, &y0, &y1);
        if (n == 0) r = j0;
        else {
            a = j0; b = j1;
            for (k = 1; k < n; ++k) {
                double c = (2.0 * k / x) * b - a;
                a = b; b = c;
            }
            r = b;
        }
    } else {
        double m[5];
        alkahest_bessel_miller(n, x, m);
        r = m[0];
    }
    return neg ? -r : r;
}
#endif
