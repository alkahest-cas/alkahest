#ifndef ALKAHEST_BESSEL_JSERIES_DEFINED
#define ALKAHEST_BESSEL_JSERIES_DEFINED
/* J_n(x) = sum_k (-1)^k (x/2)^(2k+n) / (k! (k+n)!) for integer n >= 0 and
 * |x| <= 2 (internal; a convergent, nearly cancellation-free series there). */
static double alkahest_bessel_jseries(int n, double x) {
    double h = 0.5 * x, t = 1.0, q = -h * h, sum;
    int i, k;
    for (i = 1; i <= n; ++i) t *= h / i;
    sum = t;
    for (k = 1; k < 60; ++k) {
        t *= q / ((double)k * (double)(k + n));
        sum += t;
        if (fabs(t) <= 1e-18 * fabs(sum)) break;
    }
    return sum;
}
#endif
