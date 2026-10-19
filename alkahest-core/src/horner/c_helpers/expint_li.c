#ifndef ALKAHEST_EXPINT_LI_DEFINED
#define ALKAHEST_EXPINT_LI_DEFINED
/* Logarithmic integral li(x) = PV int_0^x dt/log(t) = Ei(log x), x >= 0.
 * li(0) = 0, li(1) = -INFINITY, NAN for x < 0.  Accuracy as alkahest_ei with
 * the argument log(x) (absolute near the zero at x = 1.4513...). */
static double alkahest_li(double x) {
    if (isnan(x) || x < 0.0) return NAN;
    if (x == 0.0) return 0.0;
    if (x == 1.0) return -INFINITY;
    return alkahest_ei(log(x));
}
#endif
