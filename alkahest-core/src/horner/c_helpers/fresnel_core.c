#ifndef ALKAHEST_FRESNEL_CORE_DEFINED
#define ALKAHEST_FRESNEL_CORE_DEFINED
/* Fresnel integrals S(x) = int_0^x sin(pi t^2/2) dt, C(x) = int_0^x cos(pi t^2/2) dt
 * (the pi/2 normalisation, as Alkahest's fresnels / fresnelc).  Power series for
 * |x| <= 1.5, a complex continued fraction (modified Lentz) for |x| > 1.5.  The
 * phase pi x^2 / 2 is reduced modulo 2 pi from the exactly-split product x*x
 * (fma), so large |x| keep full accuracy.  Absolute error <= 3e-16 for
 * |x| <= 1e6; S, C -> +-1/2 as |x| -> infinity (returned for |x| > 1e150). */
static void alkahest_fresnel_sc(double x, double *s_out, double *c_out) {
    double ax = fabs(x), s, c;
    if (isnan(x)) { *s_out = NAN; *c_out = NAN; return; }
    if (ax > 1e150) { s = 0.5; c = 0.5; }
    else if (ax <= 1.5) {
        double t = 1.5707963267948966 * ax * ax, t2 = t * t;
        double term_c = 1.0, term_s = 1.0, sc = 0.0, ss = 0.0;
        int k;
        for (k = 0; k < 60; ++k) {
            double dc = term_c / (4.0 * k + 1.0), ds = term_s / (4.0 * k + 3.0);
            sc += dc;
            ss += ds;
            if (fabs(dc) < 1e-18 && fabs(ds) < 1e-18) break;
            /* ratio of consecutive terms: -t^2 / ((2k+1)(2k+2)) resp. ((2k+2)(2k+3)) */
            term_c *= -t2 / ((2.0 * k + 1.0) * (2.0 * k + 2.0));
            term_s *= -t2 / ((2.0 * k + 2.0) * (2.0 * k + 3.0));
        }
        c = ax * sc;
        s = ax * t * ss;
    } else {
        /* S + iC-like combination: h = 1/(1 - i pi x^2 + 4 + ...) cf (NR 6.9). */
        double pix2 = 3.141592653589793 * ax * ax;
        double br = 1.0, bi = -pix2;
        double cr = 1e300, ci = 0.0;
        double den = br * br + bi * bi;
        double dr = br / den, di = -bi / den;
        double hr = dr, hi = di;
        double x2 = ax * ax, e = fma(ax, ax, -x2);
        double ph, cp, sp, tr, ti, wr, wi;
        int n = -1, k;
        for (k = 2; k <= 2000; ++k) {
            double a, dr2, di2, qr, qi, delr, deli, nr, ni, m;
            n += 2;
            a = -(double)n * (double)(n + 1);
            br += 4.0;
            /* d = 1 / (a d + b) */
            nr = a * dr + br; ni = a * di + bi;
            m = nr * nr + ni * ni;
            dr2 = nr / m; di2 = -ni / m;
            dr = dr2; di = di2;
            /* c = b + a / c */
            m = cr * cr + ci * ci;
            qr = a * cr / m; qi = -a * ci / m;
            cr = br + qr; ci = bi + qi;
            /* del = c d; h *= del */
            delr = cr * dr - ci * di; deli = cr * di + ci * dr;
            nr = hr * delr - hi * deli; ni = hr * deli + hi * delr;
            hr = nr; hi = ni;
            if (fabs(delr - 1.0) + fabs(deli) < 1e-16) break;
        }
        /* h *= (ax - i ax) */
        tr = ax * hr + ax * hi; ti = ax * hi - ax * hr;
        /* phase = pi x^2 / 2 reduced mod 2 pi: x^2 = x2 + e, x2 mod 4 exact. */
        ph = 1.5707963267948966 * (fmod(x2, 4.0) + e);
        cp = cos(ph); sp = sin(ph);
        /* w = (1 - (cos + i sin) * t) */
        wr = 1.0 - (cp * tr - sp * ti);
        wi = -(cp * ti + sp * tr);
        c = 0.5 * (wr - wi);
        s = 0.5 * (wr + wi);
    }
    if (x < 0.0) { c = -c; s = -s; }
    *s_out = s; *c_out = c;
}
#endif
