/* alkahest complex fallback: a point whose real arithmetic gives NaN is
 * re-evaluated in complex arithmetic with a running rounding-error bound; the
 * real part is returned when the imaginary part is zero within that bound,
 * NaN when the value is not real (or not decidable in double precision). */
#ifndef ALKAHEST_COMPLEX_FALLBACK
#define ALKAHEST_COMPLEX_FALLBACK
#if defined(__GNUC__) || defined(__clang__)
#define AK_FN static __attribute__((unused))
#else
#define AK_FN static
#endif
typedef struct { double re, im; } ak_c;
typedef struct { ak_c z; double err; int real; } ak_cv;
#define AK_U 2.220446049250313e-16
#define AK_CV0 {{0.0, 0.0}, 0.0, 0}
#define AK_ACCEPT 16.0
#define AK_GUARD 1.4901161193847656e-08
AK_FN ak_c ak_cn(double re, double im) { ak_c z; z.re = re; z.im = im; return z; }
AK_FN ak_c ak_cfail(void) { return ak_cn(NAN, NAN); }
AK_FN ak_c ak_cadd(ak_c a, ak_c b) { return ak_cn(a.re + b.re, a.im + b.im); }
AK_FN ak_c ak_cmul(ak_c a, ak_c b) { return ak_cn(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
AK_FN double ak_cabs(ak_c z) { return hypot(z.re, z.im); }
AK_FN ak_c ak_cpowi(ak_c z, long long n) {
    if (n == 0) return ak_cn(1.0, 0.0);
    if (n < 0) {
        if (z.re == 0.0 && z.im == 0.0) return ak_cfail();
        ak_c p = ak_cpowi(z, -n);
        double d = p.re * p.re + p.im * p.im;
        if (d == 0.0 || !isfinite(d)) return ak_cfail();
        return ak_cn(p.re / d, -p.im / d);
    }
    ak_c acc = ak_cn(1.0, 0.0), base = z;
    long long e = n;
    while (e != 0) {
        if (e & 1) acc = ak_cmul(acc, base);
        e >>= 1;
        if (e != 0) base = ak_cmul(base, base);
    }
    return acc;
}
AK_FN ak_c ak_cexp(ak_c z) { double s = exp(z.re); return ak_cn(s * cos(z.im), s * sin(z.im)); }
AK_FN ak_c ak_cln(ak_c z) {
    if (z.re == 0.0 && z.im == 0.0) return ak_cfail();
    double r = sqrt(z.re * z.re + z.im * z.im);
    return ak_cn(log(r), atan2(z.im, z.re));
}
AK_FN ak_c ak_cpowc(ak_c z, ak_c w) {
    if (z.re == 0.0 && z.im == 0.0) {
        if (w.im == 0.0 && w.re > 0.0) return ak_cn(0.0, 0.0);
        return ak_cfail();
    }
    return ak_cexp(ak_cmul(w, ak_cln(z)));
}
AK_FN ak_c ak_csqrt(ak_c z) {
    if (z.re == 0.0 && z.im == 0.0) return ak_cn(0.0, 0.0);
    return ak_cpowc(z, ak_cn(0.5, 0.0));
}
AK_FN ak_c ak_csin(ak_c z) { return ak_cn(sin(z.re) * cosh(z.im), cos(z.re) * sinh(z.im)); }
AK_FN ak_c ak_ccos(ak_c z) { return ak_cn(cos(z.re) * cosh(z.im), -sin(z.re) * sinh(z.im)); }
AK_FN ak_c ak_csinh(ak_c z) { return ak_cn(sinh(z.re) * cos(z.im), cosh(z.re) * sin(z.im)); }
AK_FN ak_c ak_ccosh(ak_c z) { return ak_cn(cosh(z.re) * cos(z.im), sinh(z.re) * sin(z.im)); }
AK_FN ak_c ak_casin_root(ak_c z) {
    return ak_csqrt(ak_cadd(ak_cn(1.0, 0.0), ak_cmul(ak_cn(-1.0, 0.0), ak_cmul(z, z))));
}
AK_FN ak_c ak_cacos(ak_c z) {
    ak_c i = ak_cn(0.0, 1.0);
    ak_c ln = ak_cln(ak_cadd(ak_cmul(i, z), ak_casin_root(z)));
    return ak_cadd(ak_cn(1.5707963267948966, 0.0), ak_cmul(i, ln));
}
AK_FN ak_c ak_casin(ak_c z) {
    if (z.im == 0.0 && fabs(z.re) <= 1.0) return ak_cn(asin(z.re), 0.0);
    ak_c i = ak_cn(0.0, 1.0);
    ak_c ln = ak_cln(ak_cadd(ak_cmul(i, z), ak_casin_root(z)));
    return ak_cmul(ak_cn(0.0, -1.0), ln);
}
AK_FN ak_c ak_catan(ak_c z) {
    if (z.im == 0.0) return ak_cn(atan(z.re), 0.0);
    ak_c one = ak_cn(1.0, 0.0);
    ak_c iz = ak_cmul(ak_cn(0.0, 1.0), z);
    ak_c a = ak_cln(ak_cadd(one, ak_cn(-iz.re, -iz.im)));
    ak_c b = ak_cln(ak_cadd(one, iz));
    return ak_cmul(ak_cn(0.0, 0.5), ak_cadd(a, ak_cn(-b.re, -b.im)));
}
AK_FN ak_c ak_ctan(ak_c z) {
    if (z.im == 0.0) return ak_cn(tan(z.re), 0.0);
    ak_c s = ak_csin(z), c = ak_ccos(z);
    double d = c.re * c.re + c.im * c.im;
    if (d == 0.0) return ak_cfail();
    return ak_cn((s.re * c.re + s.im * c.im) / d, (s.im * c.re - s.re * c.im) / d);
}
AK_FN ak_cv ak_mk(ak_c z, double err, int real) {
    ak_cv v;
    v.z = real ? ak_cn(z.re, 0.0) : z;
    v.err = err;
    v.real = real;
    return v;
}
AK_FN double ak_vabs(ak_cv a) { return ak_cabs(a.z); }
AK_FN int ak_ok(ak_cv a) { return isfinite(a.z.re) && isfinite(a.z.im) && !isnan(a.err); }
AK_FN ak_cv ak_v_mul2(ak_cv a, ak_cv b) {
    ak_c z = ak_cmul(a.z, b.z);
    double ma = ak_vabs(a), mb = ak_vabs(b);
    double err = ma * b.err + mb * a.err + a.err * b.err + 2.0 * AK_U * ma * mb;
    return ak_mk(z, err, a.real && b.real);
}
AK_FN ak_cv ak_v_powint(ak_cv a, long long n) {
    ak_c z = ak_cpowi(a.z, n);
    double m = ak_vabs(a);
    unsigned long long un = n < 0 ? (unsigned long long)(-(n + 1)) + 1ULL : (unsigned long long)n;
    double k = (double)un;
    double bits = 0.0;
    for (unsigned long long t = un; t != 0; t >>= 1) bits += 1.0;
    double rounding = AK_U * 2.0 * (bits + 1.0) * ak_cabs(z);
    double prop;
    if (n == 0 || a.err == 0.0) prop = 0.0;
    else if (n > 0) prop = m == 0.0 ? pow(a.err, k) : pow(m, k) * expm1(k * log1p(a.err / m));
    else if (a.err >= m) prop = INFINITY;
    else prop = pow(m, -k) * expm1(-k * log1p(-a.err / m));
    return ak_mk(z, prop + rounding, a.real);
}
AK_FN int ak_straddles(ak_cv a) { return !a.real && fabs(a.z.im) <= a.err && a.z.re < a.err; }
AK_FN ak_cv ak_v_log(ak_cv a) {
    ak_c w = ak_cln(a.z);
    double m = ak_vabs(a);
    if (a.err >= m || ak_straddles(a) || (a.real && fabs(a.z.re) <= a.err)) return ak_mk(w, INFINITY, 0);
    double err = a.err / (m - a.err) + AK_U * (1.0 + ak_cabs(w));
    return ak_mk(w, err, a.real && a.z.re > 0.0);
}
AK_FN ak_cv ak_v_exp(ak_cv a) {
    ak_c w = ak_cexp(a.z);
    double mw = ak_cabs(w);
    return ak_mk(w, mw * expm1(a.err) + AK_U * mw * (2.0 + ak_vabs(a)), a.real);
}
AK_FN ak_cv ak_v_pow(ak_cv b, ak_cv e) {
    if (e.real && e.err == 0.0 && e.z.re == trunc(e.z.re) && fabs(e.z.re) < 9.0e15)
        return ak_v_powint(b, (long long)e.z.re);
    ak_c z = ak_cpowc(b.z, e.z);
    if (b.z.re == 0.0 && b.z.im == 0.0) {
        double err;
        if (b.err == 0.0) err = 0.0;
        else if (e.real && e.z.re > 0.0) err = pow(b.err, e.z.re) * (1.0 + e.err);
        else err = INFINITY;
        return ak_mk(z, err, e.real);
    }
    ak_cv x = ak_v_exp(ak_v_mul2(e, ak_v_log(b)));
    int real = b.real && e.real && b.z.re > 0.0 && b.err < b.z.re;
    return ak_mk(z, x.err, real);
}
AK_FN ak_cv ak_v_trig(ak_cv a, ak_c w) {
    double ch = cosh(fabs(a.z.im) + a.err);
    return ak_mk(w, a.err * ch + AK_U * ch * (2.0 + ak_vabs(a)), a.real);
}
AK_FN ak_cv ak_v_sin(ak_cv a) { return ak_v_trig(a, ak_csin(a.z)); }
AK_FN ak_cv ak_v_cos(ak_cv a) { return ak_v_trig(a, ak_ccos(a.z)); }
AK_FN ak_cv ak_v_hyp(ak_cv a, ak_c w) {
    double ch = cosh(fabs(a.z.re) + a.err);
    return ak_mk(w, a.err * ch + AK_U * ch * (2.0 + ak_vabs(a)), a.real);
}
AK_FN ak_cv ak_v_sinh(ak_cv a) { return ak_v_hyp(a, ak_csinh(a.z)); }
AK_FN ak_cv ak_v_cosh(ak_cv a) { return ak_v_hyp(a, ak_ccosh(a.z)); }
AK_FN ak_cv ak_v_tan(ak_cv a) {
    ak_c w = ak_ctan(a.z);
    double cw = ak_cabs(w);
    double d = 1.0 + cw * cw;
    double err = 4.0 * a.err * d >= 1.0 ? INFINITY : 2.0 * a.err * d + AK_U * d * (2.0 + ak_vabs(a));
    return ak_mk(w, err, a.real);
}
AK_FN ak_cv ak_v_abs(ak_cv a) { double m = ak_vabs(a); return ak_mk(ak_cn(m, 0.0), a.err + AK_U * m, 1); }
AK_FN ak_cv ak_v_re(ak_cv a) { return ak_mk(ak_cn(a.z.re, 0.0), a.err, 1); }
AK_FN ak_cv ak_v_im(ak_cv a) { return ak_mk(ak_cn(a.z.im, 0.0), a.err, 1); }
AK_FN ak_cv ak_v_conj(ak_cv a) { return ak_mk(ak_cn(a.z.re, -a.z.im), a.err, a.real); }
AK_FN ak_cv ak_v_sqrt(ak_cv a) {
    ak_c w = ak_csqrt(a.z);
    double m = ak_vabs(a), mw = ak_cabs(w);
    if (a.err == 0.0) return ak_mk(w, AK_U * mw, a.real && a.z.re >= 0.0);
    if (2.0 * a.err >= m) return ak_mk(w, sqrt(m + a.err) + mw, 0);
    if (ak_straddles(a)) return ak_mk(w, INFINITY, 0);
    return ak_mk(w, a.err / (2.0 * sqrt(m - a.err)) + AK_U * mw, a.real && a.z.re > a.err);
}
AK_FN ak_cv ak_v_arcsin_like(ak_cv a, ak_c w) {
    double m = ak_vabs(a), mw = ak_cabs(w);
    if (a.real) {
        double x = fabs(a.z.re);
        if (x + a.err < 1.0)
            return ak_mk(w, a.err / sqrt(1.0 - (x + a.err) * (x + a.err)) + AK_U * (1.0 + mw), 1);
        if (x - a.err > 1.0)
            return ak_mk(w, a.err / sqrt((x - a.err) * (x - a.err) - 1.0) + AK_U * (1.0 + mw), 0);
        return ak_mk(w, 4.0 * sqrt(a.err * (1.0 + x)) + AK_U * (1.0 + mw), 0);
    }
    int on_cut = fabs(a.z.im) <= a.err && fabs(a.z.re) + a.err > 1.0;
    ak_c omz2 = ak_cadd(ak_cn(1.0, 0.0), ak_cmul(ak_cn(-1.0, 0.0), ak_cmul(a.z, a.z)));
    double dmin = ak_cabs(omz2) - a.err * (2.0 * m + a.err);
    if (on_cut || dmin <= 0.0) return ak_mk(w, INFINITY, 0);
    return ak_mk(w, a.err / sqrt(dmin) + AK_U * (1.0 + mw), 0);
}
AK_FN ak_cv ak_v_acos(ak_cv a) { return ak_v_arcsin_like(a, ak_cacos(a.z)); }
AK_FN ak_cv ak_v_asin(ak_cv a) { return ak_v_arcsin_like(a, ak_casin(a.z)); }
AK_FN ak_cv ak_v_atan(ak_cv a) {
    ak_c w = ak_catan(a.z);
    double m = ak_vabs(a), mw = ak_cabs(w);
    if (a.real) {
        double lo = fabs(a.z.re) - a.err;
        if (lo < 0.0) lo = 0.0;
        return ak_mk(w, a.err / (1.0 + lo * lo) + AK_U * (1.0 + mw), 1);
    }
    int on_cut = fabs(a.z.re) <= a.err && fabs(a.z.im) + a.err > 1.0;
    ak_c opz2 = ak_cadd(ak_cn(1.0, 0.0), ak_cmul(a.z, a.z));
    double dmin = ak_cabs(opz2) - a.err * (2.0 * m + a.err);
    if (on_cut || dmin <= 0.0) return ak_mk(w, INFINITY, 0);
    return ak_mk(w, a.err / dmin + AK_U * (1.0 + mw), 0);
}
AK_FN double ak_decide(ak_cv out, double scale) {
    if (out.real) return out.z.re;
    if (!(isfinite(out.err) && out.err <= AK_GUARD * scale)) return NAN;
    return fabs(out.z.im) <= AK_ACCEPT * out.err ? out.z.re : NAN;
}
enum {
    AK_OP_CONST, AK_OP_INPUT, AK_OP_ADD, AK_OP_MUL, AK_OP_POWINT, AK_OP_POW,
    AK_OP_SIN, AK_OP_COS, AK_OP_TAN, AK_OP_EXP, AK_OP_SINH, AK_OP_COSH, AK_OP_ABS,
    AK_OP_LOG, AK_OP_SQRT, AK_OP_ACOS, AK_OP_ASIN, AK_OP_ATAN, AK_OP_RE, AK_OP_IM, AK_OP_CONJ
};
typedef struct { int code, first, count; long long n; } ak_op;
/* Evaluate operations 0..=root (children precede parents); an operation is
 * usable when its children are and its value is finite, and carries the
 * largest magnitude in its subtree. The decision reads the root's own. */
AK_FN double ak_run(const ak_op *ops, const int *args, const double *consts, const double *in,
                    int root, ak_cv *v, int *ok, double *sc) {
    for (int i = 0; i <= root; i++) {
        const ak_op o = ops[i];
        const int *c = args + o.first;
        int good = 1;
        double s = 0.0;
        for (int j = 0; j < o.count; j++) {
            good = good && ok[c[j]];
            if (ok[c[j]]) s = fmax(s, sc[c[j]]);
        }
        if (!good) { ok[i] = 0; sc[i] = 0.0; continue; }
        ak_cv r;
        switch (o.code) {
        case AK_OP_CONST: {
            const double *q = consts + 4 * o.first;
            r = ak_mk(ak_cn(q[0], q[1]), q[2], q[3] != 0.0);
            break;
        }
        case AK_OP_INPUT: r = ak_mk(ak_cn(in[o.first], 0.0), 0.0, 1); break;
        case AK_OP_ADD: {
            ak_c z = ak_cn(0.0, 0.0);
            double err = 0.0, mag = 0.0;
            int real = 1;
            for (int j = 0; j < o.count; j++) {
                z = ak_cadd(z, v[c[j]].z);
                err += v[c[j]].err;
                mag += ak_vabs(v[c[j]]);
                real = real && v[c[j]].real;
            }
            r = ak_mk(z, err + AK_U * (double)o.count * mag, real);
            break;
        }
        case AK_OP_MUL:
            r = ak_mk(ak_cn(1.0, 0.0), 0.0, 1);
            for (int j = 0; j < o.count; j++) r = ak_v_mul2(r, v[c[j]]);
            break;
        case AK_OP_POWINT: r = ak_v_powint(v[c[0]], o.n); break;
        case AK_OP_POW: r = ak_v_pow(v[c[0]], v[c[1]]); break;
        case AK_OP_SIN: r = ak_v_sin(v[c[0]]); break;
        case AK_OP_COS: r = ak_v_cos(v[c[0]]); break;
        case AK_OP_TAN: r = ak_v_tan(v[c[0]]); break;
        case AK_OP_EXP: r = ak_v_exp(v[c[0]]); break;
        case AK_OP_SINH: r = ak_v_sinh(v[c[0]]); break;
        case AK_OP_COSH: r = ak_v_cosh(v[c[0]]); break;
        case AK_OP_ABS: r = ak_v_abs(v[c[0]]); break;
        case AK_OP_LOG: r = ak_v_log(v[c[0]]); break;
        case AK_OP_SQRT: r = ak_v_sqrt(v[c[0]]); break;
        case AK_OP_ACOS: r = ak_v_acos(v[c[0]]); break;
        case AK_OP_ASIN: r = ak_v_asin(v[c[0]]); break;
        case AK_OP_ATAN: r = ak_v_atan(v[c[0]]); break;
        case AK_OP_RE: r = ak_v_re(v[c[0]]); break;
        case AK_OP_IM: r = ak_v_im(v[c[0]]); break;
        default: r = ak_v_conj(v[c[0]]); break;
        }
        v[i] = r;
        ok[i] = ak_ok(r);
        sc[i] = ok[i] ? fmax(ak_vabs(r), s) : 0.0;
    }
    return ok[root] ? ak_decide(v[root], sc[root]) : NAN;
}
#endif /* ALKAHEST_COMPLEX_FALLBACK */
