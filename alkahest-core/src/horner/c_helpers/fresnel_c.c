#ifndef ALKAHEST_FRESNEL_C_DEFINED
#define ALKAHEST_FRESNEL_C_DEFINED
/* Fresnel C(x) = int_0^x cos(pi t^2/2) dt; see alkahest_fresnel_sc. */
static double alkahest_fresnel_c(double x) {
    double s, c;
    alkahest_fresnel_sc(x, &s, &c);
    return c;
}
#endif
