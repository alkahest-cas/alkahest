#ifndef ALKAHEST_FRESNEL_S_DEFINED
#define ALKAHEST_FRESNEL_S_DEFINED
/* Fresnel S(x) = int_0^x sin(pi t^2/2) dt; see alkahest_fresnel_sc. */
static double alkahest_fresnel_s(double x) {
    double s, c;
    alkahest_fresnel_sc(x, &s, &c);
    return s;
}
#endif
