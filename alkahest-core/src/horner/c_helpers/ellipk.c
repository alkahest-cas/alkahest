#ifndef ALKAHEST_ELLIPK_DEFINED
#define ALKAHEST_ELLIPK_DEFINED
/* Complete elliptic integral of the first kind K(m), parameter m = k^2
 * (Mathematica's EllipticK[m]).  Arithmetic-geometric mean; INFINITY at
 * m = 1, NAN for m > 1.  Relative error <= 4e-16 for m in [-1e6, 1 - 1e-12]. */
static double alkahest_ellipk(double m) {
    double a = 1.0, b;
    int i;
    if (isnan(m) || m > 1.0) return NAN;
    if (m == 1.0) return INFINITY;
    if (m == -INFINITY) return 0.0;
    b = sqrt(1.0 - m);
    for (i = 0; i < 60 && fabs(a - b) > 1.1e-16 * a; ++i) {
        double an = 0.5 * (a + b);
        b = sqrt(a * b);
        a = an;
    }
    return 1.5707963267948966 / (0.5 * (a + b));
}
#endif
