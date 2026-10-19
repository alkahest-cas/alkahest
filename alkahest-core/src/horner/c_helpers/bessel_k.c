#ifndef ALKAHEST_BESSEL_K_DEFINED
#define ALKAHEST_BESSEL_K_DEFINED
/* Modified Bessel function of the second kind K_n(x) for INTEGER order
 * |n| <= 10000 (NAN otherwise) and x > 0: K_n(0) = +INFINITY, NAN for x < 0
 * (complex), K_n(INFINITY) = 0.  K_0 and K_1 from the integral
 * K_v(x) = int_0^inf exp(-x cosh t) cosh(v t) dt by the trapezoidal rule (the
 * integrand is analytic in a strip and positive, so the rule converges
 * geometrically with no cancellation; step 0.7/sqrt(x) capped at 0.2), higher
 * orders by the stable forward recurrence.  Relative error <= 3e-15 for orders
 * 0..50 and 1e-6 <= x <= 700. */
static double alkahest_bessel_k(double order, double x) {
    int n, k;
    double h, s0, s1, sh, a, b, e;
    if (isnan(order) || isnan(x) || order != floor(order) || fabs(order) > 10000.0)
        return NAN;
    n = (int)order;
    if (n < 0) n = -n;
    if (x < 0.0) return NAN;
    if (x == 0.0) return INFINITY;
    if (isinf(x)) return 0.0;
    h = 0.7 / sqrt(x);
    if (h > 0.2) h = 0.2;
    s0 = 0.5; s1 = 0.5;
    for (k = 1; k < 100000; ++k) {
        double t = h * k, f;
        sh = sinh(0.5 * t);
        f = x * 2.0 * sh * sh; /* x (cosh t - 1) */
        if (f > 60.0) break;
        f = exp(-f);
        s0 += f;
        s1 += f * cosh(t);
    }
    e = exp(-x);
    a = e * h * s0;
    b = e * h * s1;
    if (n == 0) return a;
    for (k = 1; k < n; ++k) {
        double c = a + (2.0 * k / x) * b;
        a = b; b = c;
        if (isinf(b)) return b;
    }
    return b;
}
#endif
