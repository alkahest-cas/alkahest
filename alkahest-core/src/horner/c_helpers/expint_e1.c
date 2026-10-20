#ifndef ALKAHEST_EXPINT_E1_DEFINED
#define ALKAHEST_EXPINT_E1_DEFINED
/* E1(x) = int_x^inf exp(-t)/t dt for x > 0 (internal; callers handle x <= 0).
 * Series for x <= 1, modified-Lentz continued fraction beyond.  Relative error
 * <= 2e-15. */
static double alkahest_e1_pos(double x) {
    int k;
    if (x <= 1.0) {
        double term = 1.0, sum = 0.0;
        for (k = 1; k < 80; ++k) {
            term *= -x / k;
            sum -= term / k; /* -(-x)^k / (k k!) */
            if (fabs(term) < 1e-20) break;
        }
        return -0.5772156649015329 - log(x) + sum;
    } else {
        /* x + 1 - 1^2/(x + 3 - 2^2/(x + 5 - ...)), evaluated bottom-up from a
         * depth chosen for convergence ~ exp(-4 sqrt(N x)) < 1e-18. */
        int depth = (int)(150.0 / x) + 20;
        double t;
        if (depth > 400) depth = 400;
        t = x + 2.0 * depth + 1.0;
        for (k = depth; k >= 1; --k) t = x + 2.0 * k - 1.0 - (double)k * (double)k / t;
        return exp(-x) / t;
    }
}
#endif
