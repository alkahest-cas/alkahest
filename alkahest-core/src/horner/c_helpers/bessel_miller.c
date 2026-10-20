#ifndef ALKAHEST_BESSEL_MILLER_DEFINED
#define ALKAHEST_BESSEL_MILLER_DEFINED
/* Miller's backward recurrence for J_k(x), x > 0, normalised by
 * 1 = J_0 + 2 sum J_2k.  r[0] = J_n, r[1] = J_0, r[2] = J_1 and the Neumann
 * sums r[3] = sum_{k>=1} (-1)^(k+1) J_2k / k,
 * r[4] = sum_{k>=1} (-1)^(k+1) (J_(2k-1) - J_(2k+1)) / k (internal). */
static void alkahest_bessel_miller(int n, double x, double r[5]) {
    int m = n > (int)x ? n : (int)x, k;
    double hi = 0.0, lo = 1.0, tox = 2.0 / x, norm = 0.0;
    double jn = 0.0, j0 = 0.0, j1 = 0.0, s0 = 0.0, s1 = 0.0;
    m += 40 + (int)sqrt(100.0 * (double)m);
    for (k = m; k >= 0; --k) {
        /* lo = J_k (unnormalised), hi = J_(k+1). */
        double sg;
        if (k == n) jn = lo;
        if (k == 0) j0 = lo;
        if (k == 1) j1 = lo;
        if (k >= 2 && (k & 1) == 0) {
            sg = ((k / 2) & 1) ? 1.0 : -1.0;
            norm += 2.0 * lo;
            s0 += sg * lo / (k / 2);
        }
        if (k & 1) {
            int kk = (k + 1) / 2; /* k = 2 kk - 1 */
            sg = (kk & 1) ? 1.0 : -1.0;
            s1 += sg * lo / kk;
            if (k >= 3) {
                kk = (k - 1) / 2; /* k = 2 kk + 1 */
                sg = (kk & 1) ? 1.0 : -1.0;
                s1 -= sg * lo / kk;
            }
        }
        if (k > 0) {
            double nx = (double)k * tox * lo - hi;
            hi = lo;
            lo = nx;
            if (fabs(lo) > 1e250) {
                lo *= 1e-250; hi *= 1e-250; norm *= 1e-250; jn *= 1e-250;
                j0 *= 1e-250; j1 *= 1e-250; s0 *= 1e-250; s1 *= 1e-250;
            }
        }
    }
    norm += j0;
    r[0] = jn / norm; r[1] = j0 / norm; r[2] = j1 / norm;
    r[3] = s0 / norm; r[4] = s1 / norm;
}
#endif
