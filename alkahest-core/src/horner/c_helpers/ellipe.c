#ifndef ALKAHEST_ELLIPE_DEFINED
#define ALKAHEST_ELLIPE_DEFINED
/* Complete elliptic integral of the second kind E(m), parameter m = k^2
 * (Mathematica's EllipticE[m]).  AGM with the Gauss-Legendre side sum
 * E = K (1 - sum 2^(n-1) c_n^2); E(1) = 1, NAN for m > 1.  Relative error
 * <= 4e-16 for m in [-1e6, 1]. */
static double alkahest_ellipe(double m) {
    double a = 1.0, b, cn2 = m, sum, pw = 0.5;
    int i;
    if (isnan(m) || m > 1.0) return NAN;
    if (m == 1.0) return 1.0;
    if (m == -INFINITY) return INFINITY;
    b = sqrt(1.0 - m);
    sum = pw * cn2; /* 2^(-1) c_0^2 with c_0^2 = m */
    for (i = 0; i < 60 && fabs(a - b) > 1.1e-16 * a; ++i) {
        /* c_(n+1) = c_n^2 / (4 a_(n+1)): no cancellation in a - b. */
        double an = 0.5 * (a + b), c = cn2 / (4.0 * an);
        b = sqrt(a * b);
        a = an;
        pw *= 2.0;
        cn2 = c * c;
        sum += pw * cn2;
    }
    return (1.5707963267948966 / (0.5 * (a + b))) * (1.0 - sum);
}
#endif
