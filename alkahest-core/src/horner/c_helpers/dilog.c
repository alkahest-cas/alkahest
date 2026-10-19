#ifndef ALKAHEST_DILOG_DEFINED
#define ALKAHEST_DILOG_DEFINED
/* Dilogarithm Li2(x) = -integral_0^x log(1 - t)/t dt on x <= 1 (real branch);
 * NAN for x > 1 (the principal value is complex there) and for NAN input.
 * Bernoulli series in u = -log(1 - x) on [-1, 1/2], reflected onto it with
 * Li2(x) + Li2(1 - x) = pi^2/6 - log(x) log(1 - x) and
 * Li2(x) + Li2(1/x) = -pi^2/6 - log(-x)^2 / 2.  Relative error <= 1e-15 on
 * x in [-1e6, 1] (the one absolute-error point is the zero at x = 0). */
static double alkahest_dilog_core(double x) {
    static const double c[10] = {
        2.7777777777777776e-02, -2.777777777777778e-04,
        4.72411186696901e-06, -9.185773074661964e-08,
        1.8978869988971e-09, -4.0647616451442256e-11,
        8.921691020456452e-13, -1.9939295860721074e-14,
        4.518980029619918e-16, -1.0356517612181247e-17};
    double u = -log1p(-x), t = u * u, p = 0.0;
    int i;
    for (i = 9; i >= 0; --i) p = p * t + c[i];
    return u - 0.25 * t + u * t * p;
}
static double alkahest_dilog(double x) {
    const double pi2_6 = 1.6449340668482264;
    if (isnan(x) || x > 1.0) return NAN;
    if (x == 1.0) return pi2_6;
    if (x == -INFINITY) return -INFINITY;
    if (x < -1.0) {
        double l = log(-x);
        return -alkahest_dilog_core(1.0 / x) - pi2_6 - 0.5 * l * l;
    }
    if (x > 0.5) return pi2_6 - log(x) * log1p(-x) - alkahest_dilog_core(1.0 - x);
    return alkahest_dilog_core(x);
}
#endif
