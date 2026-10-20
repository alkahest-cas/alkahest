#ifndef ALKAHEST_BESSEL_Y_DEFINED
#define ALKAHEST_BESSEL_Y_DEFINED
/* Bessel function of the second kind Y_n(x) for INTEGER order |n| <= 10000
 * (NAN otherwise) and x > 0: Y_n(0) = -INFINITY, NAN for x < 0 (complex),
 * Y_n(INFINITY) = 0.  Y_0, Y_1: ascending series for x <= 2 (DLMF 10.8),
 * Neumann series on the Miller-recurrence J_k for 2 < x < 25, Hankel's
 * asymptotic expansion for x >= 25; higher orders by the stable forward
 * recurrence (-INFINITY once it overflows).  Error <= 3e-15 absolute and
 * <= 1e-14 relative to max(|Y_n(x)|, sqrt(2/(pi x))) for orders 0..50 and
 * x <= 1e5. */
static double alkahest_bessel_y(double order, double x) {
    const double g = 0.5772156649015329, tp = 0.6366197723675814;
    int n, neg = 0, k;
    double y0 = 0.0, y1 = 0.0, r;
    if (isnan(order) || isnan(x) || order != floor(order) || fabs(order) > 10000.0)
        return NAN;
    n = (int)order;
    if (n < 0) { n = -n; neg = n & 1; }
    if (x < 0.0) return NAN;
    if (x == 0.0) return -INFINITY;
    if (isinf(x)) return 0.0;
    if (x <= 2.0) {
        double h = 0.5 * x, q = h * h, lg = log(h), t0 = 1.0, t1 = 1.0, hk = 0.0;
        double s0 = 0.0, s1 = 0.0, j0 = alkahest_bessel_jseries(0, x),
               j1 = alkahest_bessel_jseries(1, x);
        /* t0 = (x^2/4)^k / (k!)^2 ; t1 = (x^2/4)^k / (k! (k+1)!) */
        for (k = 0; k < 60; ++k) {
            double hk1 = hk + 1.0 / (k + 1.0); /* H_(k+1) */
            double psi_sum = -2.0 * g + hk + hk1;
            double sg = (k & 1) ? -1.0 : 1.0;
            if (k >= 1) s0 -= sg * hk * t0; /* sum (-1)^(k+1) H_k t0 */
            s1 += sg * psi_sum * t1;
            if (k >= 1 && fabs(t0) < 1e-20 && fabs(t1) < 1e-20) break;
            t0 *= q / ((double)(k + 1) * (double)(k + 1));
            t1 *= q / ((double)(k + 1) * (double)(k + 2));
            hk = hk1;
        }
        y0 = tp * ((lg + g) * j0 + s0);
        /* -2/(pi x) = -tp / x */
        y1 = -tp / x + tp * lg * j1 - tp * 0.5 * h * s1;
    } else if (x >= 25.0) {
        double j0 = 0.0, j1 = 0.0;
        alkahest_bessel_asym01(x, &j0, &j1, &y0, &y1);
    } else {
        double m[5], lg = log(0.5 * x) + g;
        alkahest_bessel_miller(1, x, m);
        y0 = tp * (lg * m[1] + 2.0 * m[3]);
        y1 = tp * (lg * m[2] - m[1] / x - m[4]);
    }
    if (n == 0) r = y0;
    else {
        double a = y0, b = y1;
        for (k = 1; k < n; ++k) {
            double c = (2.0 * k / x) * b - a;
            a = b; b = c;
            if (isinf(b)) return neg ? INFINITY : -INFINITY;
        }
        r = b;
    }
    return neg ? -r : r;
}
#endif
