#ifndef ALKAHEST_EXPINT_SI_DEFINED
#define ALKAHEST_EXPINT_SI_DEFINED
/* Sine integral Si(x) = int_0^x sin t / t dt, odd; Si(+-INFINITY) = +-pi/2.
 * Absolute error <= 3e-16 for |x| <= 1e6. */
static double alkahest_si(double x) {
    double ci, si;
    if (isnan(x)) return NAN;
    if (x == 0.0) return x;
    alkahest_cisi_pos(fabs(x), &ci, &si);
    return x < 0.0 ? -si : si;
}
#endif
