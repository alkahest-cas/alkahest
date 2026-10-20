#ifndef ALKAHEST_EXPINT_CISI_DEFINED
#define ALKAHEST_EXPINT_CISI_DEFINED
/* Si(x) = int_0^x sin t / t dt and Ci(x) = -int_x^inf cos t / t dt for x > 0
 * (internal).  Power series for x <= 2, complex continued fraction beyond.
 * Absolute error <= 3e-16 (Si), <= 3e-16 (Ci) for x <= 1e6. */
static void alkahest_cisi_pos(double x, double *ci_out, double *si_out) {
    int k;
    if (isinf(x)) { *ci_out = 0.0; *si_out = 1.5707963267948966; return; }
    if (x <= 2.0) {
        double t = x * x, term = 1.0, ssum = 0.0, csum = 0.0;
        /* Si = x sum (-1)^k t^k / ((2k+1)(2k+1)!),  Ci = gamma + log x + sum_{k>=1}
         * (-1)^k t^k / (2k (2k)!). */
        double ts = 1.0; /* (-t)^k / (2k+1)!  */
        double tc = 1.0; /* (-t)^k / (2k)!    */
        (void)term;
        ssum = 1.0;
        for (k = 1; k < 60; ++k) {
            tc *= -t / ((2.0 * k - 1.0) * (2.0 * k));
            ts *= -t / ((2.0 * k) * (2.0 * k + 1.0));
            csum += tc / (2.0 * k);
            ssum += ts / (2.0 * k + 1.0);
            if (fabs(tc) < 1e-20 && fabs(ts) < 1e-20) break;
        }
        *si_out = x * ssum;
        *ci_out = 0.5772156649015329 + log(x) + csum;
    } else {
        double br = 1.0, bi = x, cr = 1e300, ci = 0.0, den = br * br + bi * bi;
        double dr = br / den, di = -bi / den, hr = dr, hi = di;
        double cx = cos(x), sx = sin(x), tr, ti;
        for (k = 2; k < 4000; ++k) {
            double a = -(double)(k - 1) * (double)(k - 1), m, nr, ni, delr, deli, qr, qi;
            br += 2.0;
            nr = a * dr + br; ni = a * di + bi;
            m = nr * nr + ni * ni;
            dr = nr / m; di = -ni / m;
            m = cr * cr + ci * ci;
            qr = a * cr / m; qi = -a * ci / m;
            cr = br + qr; ci = bi + qi;
            delr = cr * dr - ci * di; deli = cr * di + ci * dr;
            nr = hr * delr - hi * deli; ni = hr * deli + hi * delr;
            hr = nr; hi = ni;
            if (fabs(delr - 1.0) + fabs(deli) < 1e-16) break;
        }
        /* h *= (cos x - i sin x) */
        tr = cx * hr + sx * hi; ti = cx * hi - sx * hr;
        *ci_out = -tr;
        *si_out = 1.5707963267948966 + ti;
    }
}
#endif
