#ifndef ALKAHEST_BESSEL_ASYM_DEFINED
#define ALKAHEST_BESSEL_ASYM_DEFINED
/* J0, J1, Y0, Y1 at x >= 25 from Hankel's asymptotic expansion (DLMF 10.17);
 * the series is summed to its smallest term, ~exp(-2x) < 1e-21 here.  The phase
 * x - (nu/2 + 1/4) pi is expanded with cos/sin of x itself, so no rounding of
 * x - pi/4 enters. */
static void alkahest_bessel_asym01(double x, double *j0, double *j1, double *y0,
                                   double *y1) {
    const double r2 = 0.7071067811865476;
    double amp = sqrt(0.6366197723675814 / x);
    double cx = cos(x), sx = sin(x);
    int nu;
    for (nu = 0; nu <= 1; ++nu) {
        double mu = 4.0 * nu * nu, term = 1.0, p = 1.0, qq = 0.0, prev = 1.0;
        double cc, ss, j, y;
        int k;
        for (k = 1; k < 80; ++k) {
            double odd = (double)(2 * k - 1);
            term *= (mu - odd * odd) / (8.0 * k * x);
            if (fabs(term) > prev) break;
            prev = fabs(term);
            if (k & 1) qq += ((k & 3) == 1) ? term : -term;
            else p += ((k & 3) == 0) ? term : -term;
            if (fabs(term) < 1e-18) break;
        }
        if (nu == 0) { cc = (cx + sx) * r2; ss = (sx - cx) * r2; }
        else { cc = (sx - cx) * r2; ss = -(sx + cx) * r2; }
        j = amp * (p * cc - qq * ss);
        y = amp * (p * ss + qq * cc);
        if (nu == 0) { *j0 = j; *y0 = y; } else { *j1 = j; *y1 = y; }
    }
}
#endif
