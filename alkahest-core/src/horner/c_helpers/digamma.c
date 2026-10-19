#ifndef ALKAHEST_DIGAMMA_DEFINED
#define ALKAHEST_DIGAMMA_DEFINED
/* Digamma psi(x) = Gamma'(x)/Gamma(x) for real x; NAN at x = 0, -1, -2, ...
 * and at -INFINITY.  Reflection for x < 0, upward recurrence to x >= 10, then
 * the asymptotic series.  Relative error <= 2e-15 away from the zeros of psi
 * (x = 1.4616..., -0.5040..., ...), where the error is absolute (~2e-16). */
static double alkahest_digamma(double x) {
    double acc = 0.0, r, z;
    if (isnan(x) || x == -INFINITY) return NAN;
    if (isinf(x)) return x;
    if (x <= 0.0 && x == floor(x)) return NAN;
    if (x < 0.0) {
        /* psi(x) = psi(1 - x) - pi / tan(pi x); tan has period 1. */
        double f = x - floor(x + 0.5); /* in [-0.5, 0.5] */
        acc = -3.141592653589793 / tan(3.141592653589793 * f);
        x = 1.0 - x;
    }
    while (x < 10.0) {
        acc -= 1.0 / x;
        x += 1.0;
    }
    if (x > 1e8) return acc + log(x) - 0.5 / x;
    r = 1.0 / x;
    z = r * r;
    return acc + log(x) - 0.5 * r
        - z * (1.0 / 12.0 + z * (-1.0 / 120.0 + z * (1.0 / 252.0
        + z * (-1.0 / 240.0 + z * (1.0 / 132.0 + z * (-691.0 / 32760.0
        + z * (1.0 / 12.0)))))));
}
#endif
