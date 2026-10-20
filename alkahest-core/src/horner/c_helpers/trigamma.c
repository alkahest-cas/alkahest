#ifndef ALKAHEST_TRIGAMMA_DEFINED
#define ALKAHEST_TRIGAMMA_DEFINED
/* Trigamma psi'(x) for real x; NAN at x = 0, -1, -2, ... and at -INFINITY.
 * Reflection for x < 0, upward recurrence to x >= 10, then the asymptotic
 * series.  Relative error <= 2e-15 (psi' > 0 on x > 0, so no zero crossing). */
static double alkahest_trigamma(double x) {
    double acc = 0.0, r, z;
    if (isnan(x) || x == -INFINITY) return NAN;
    if (isinf(x)) return 0.0;
    if (x <= 0.0 && x == floor(x)) return NAN;
    if (x < 0.0) {
        /* psi'(x) = -psi'(1 - x) + pi^2 / sin(pi x)^2 */
        double f = x - floor(x + 0.5);
        double s = sin(3.141592653589793 * f);
        return 9.869604401089358 / (s * s) - alkahest_trigamma(1.0 - x);
    }
    while (x < 10.0) {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    if (x > 1e8) return acc + 1.0 / x + 0.5 / (x * x);
    r = 1.0 / x;
    z = r * r;
    return acc + r + 0.5 * z
        + r * z * (1.0 / 6.0 + z * (-1.0 / 30.0 + z * (1.0 / 42.0
        + z * (-1.0 / 30.0 + z * (5.0 / 66.0 + z * (-691.0 / 2730.0
        + z * (7.0 / 6.0)))))));
}
#endif
