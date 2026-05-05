/// LaTeX and Unicode pretty-printing for symbolic expressions.
///
/// Mirrors the logic of `python/alkahest/_pretty.py` but dispatches on the
/// typed `ExprData` enum instead of string-tagged Python lists, eliminating
/// the `if t == "symbol":` / `if t == "add":` pattern-matching in Python.
use crate::kernel::expr::{ExprData, PredicateKind};
use crate::kernel::{ExprId, ExprPool};

const PREC_ADD: i32 = 10;
const PREC_MUL: i32 = 20;
const PREC_POW: i32 = 30;
const PREC_ATOM: i32 = 100;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Render `id` as a LaTeX string.
pub fn render_latex(id: ExprId, pool: &ExprPool) -> String {
    latex_r(id, pool).0
}

/// Render `id` as a Unicode pretty-printed string.
pub fn render_unicode(id: ExprId, pool: &ExprPool) -> String {
    unicode_r(id, pool).0
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn greek_latex(name: &str) -> Option<&'static str> {
    match name {
        "alpha" => Some(r"\alpha"),
        "beta" => Some(r"\beta"),
        "gamma" => Some(r"\gamma"),
        "delta" => Some(r"\delta"),
        "epsilon" => Some(r"\epsilon"),
        "zeta" => Some(r"\zeta"),
        "eta" => Some(r"\eta"),
        "theta" => Some(r"\theta"),
        "iota" => Some(r"\iota"),
        "kappa" => Some(r"\kappa"),
        "lamda" | "lambda" => Some(r"\lambda"),
        "mu" => Some(r"\mu"),
        "nu" => Some(r"\nu"),
        "xi" => Some(r"\xi"),
        "pi" => Some(r"\pi"),
        "rho" => Some(r"\rho"),
        "sigma" => Some(r"\sigma"),
        "tau" => Some(r"\tau"),
        "upsilon" => Some(r"\upsilon"),
        "phi" => Some(r"\phi"),
        "chi" => Some(r"\chi"),
        "psi" => Some(r"\psi"),
        "omega" => Some(r"\omega"),
        "Alpha" => Some(r"\Alpha"),
        "Beta" => Some(r"\Beta"),
        "Gamma" => Some(r"\Gamma"),
        "Delta" => Some(r"\Delta"),
        "Epsilon" => Some(r"\Epsilon"),
        "Zeta" => Some(r"\Zeta"),
        "Eta" => Some(r"\Eta"),
        "Theta" => Some(r"\Theta"),
        "Iota" => Some(r"\Iota"),
        "Kappa" => Some(r"\Kappa"),
        "Lambda" => Some(r"\Lambda"),
        "Mu" => Some(r"\Mu"),
        "Nu" => Some(r"\Nu"),
        "Xi" => Some(r"\Xi"),
        "Pi" => Some(r"\Pi"),
        "Rho" => Some(r"\Rho"),
        "Sigma" => Some(r"\Sigma"),
        "Tau" => Some(r"\Tau"),
        "Upsilon" => Some(r"\Upsilon"),
        "Phi" => Some(r"\Phi"),
        "Chi" => Some(r"\Chi"),
        "Psi" => Some(r"\Psi"),
        "Omega" => Some(r"\Omega"),
        "inf" | "oo" => Some(r"\infty"),
        _ => None,
    }
}

fn greek_unicode(name: &str) -> Option<&'static str> {
    match name {
        "alpha" => Some("α"),
        "beta" => Some("β"),
        "gamma" => Some("γ"),
        "delta" => Some("δ"),
        "epsilon" => Some("ε"),
        "zeta" => Some("ζ"),
        "eta" => Some("η"),
        "theta" => Some("θ"),
        "iota" => Some("ι"),
        "kappa" => Some("κ"),
        "lamda" | "lambda" => Some("λ"),
        "mu" => Some("μ"),
        "nu" => Some("ν"),
        "xi" => Some("ξ"),
        "pi" => Some("π"),
        "rho" => Some("ρ"),
        "sigma" => Some("σ"),
        "tau" => Some("τ"),
        "upsilon" => Some("υ"),
        "phi" => Some("φ"),
        "chi" => Some("χ"),
        "psi" => Some("ψ"),
        "omega" => Some("ω"),
        "Alpha" => Some("Α"),
        "Beta" => Some("Β"),
        "Gamma" => Some("Γ"),
        "Delta" => Some("Δ"),
        "Epsilon" => Some("Ε"),
        "Zeta" => Some("Ζ"),
        "Eta" => Some("Η"),
        "Theta" => Some("Θ"),
        "Iota" => Some("Ι"),
        "Kappa" => Some("Κ"),
        "Lambda" => Some("Λ"),
        "Mu" => Some("Μ"),
        "Nu" => Some("Ν"),
        "Xi" => Some("Ξ"),
        "Pi" => Some("Π"),
        "Rho" => Some("Ρ"),
        "Sigma" => Some("Σ"),
        "Tau" => Some("Τ"),
        "Upsilon" => Some("Υ"),
        "Phi" => Some("Φ"),
        "Chi" => Some("Χ"),
        "Psi" => Some("Ψ"),
        "Omega" => Some("Ω"),
        "inf" | "oo" => Some("∞"),
        _ => None,
    }
}

fn to_superscript(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        let sup = match ch {
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            '9' => '⁹',
            '+' => '⁺',
            '-' => '⁻',
            _ => return None,
        };
        out.push(sup);
    }
    Some(out)
}

fn unicode_frac(num: i64, den: i64) -> String {
    match (num, den) {
        (1, 2) => "½".into(),
        (1, 3) => "⅓".into(),
        (2, 3) => "⅔".into(),
        (1, 4) => "¼".into(),
        (3, 4) => "¾".into(),
        (1, 5) => "⅕".into(),
        (2, 5) => "⅖".into(),
        (3, 5) => "⅗".into(),
        (4, 5) => "⅘".into(),
        (1, 6) => "⅙".into(),
        (5, 6) => "⅚".into(),
        (1, 7) => "⅐".into(),
        (1, 8) => "⅛".into(),
        (3, 8) => "⅜".into(),
        (5, 8) => "⅝".into(),
        (7, 8) => "⅞".into(),
        (1, 9) => "⅑".into(),
        (1, 10) => "⅒".into(),
        _ => format!("{num}/{den}"),
    }
}

// ---------------------------------------------------------------------------
// LaTeX renderer
// ---------------------------------------------------------------------------

fn latex_frac(num: &str, den: &str) -> String {
    format!(r"\frac{{{num}}}{{{den}}}")
}

fn latex_symbol(name: &str) -> String {
    if let Some(g) = greek_latex(name) {
        return g.to_string();
    }
    if let Some((base, sub)) = name.split_once('_') {
        return format!("{{{base}}}_{{{sub}}}");
    }
    name.to_string()
}

fn latex_wrap(id: ExprId, pool: &ExprPool, req_prec: i32) -> String {
    let (s, prec) = latex_r(id, pool);
    if prec < req_prec {
        format!(r"\left({s}\right)")
    } else {
        s
    }
}

/// Returns `(sign, abs_latex)` for a term that might be negated.
fn latex_signed(id: ExprId, pool: &ExprPool) -> (i32, String) {
    pool.with(id, |data| match data {
        ExprData::Integer(n) => {
            let v = n.0.to_i64().unwrap_or(0);
            if v < 0 {
                (-1, (-v).to_string())
            } else {
                (1, v.to_string())
            }
        }
        ExprData::Rational(r) => {
            let num = r.0.numer();
            let den = r.0.denom();
            let s = latex_frac(num.to_string().trim_start_matches('-'), &den.to_string());
            if *num < 0 {
                (-1, s)
            } else {
                (1, s)
            }
        }
        ExprData::Mul(args) => latex_signed_mul(args, pool),
        _ => {
            let (s, _) = latex_r(id, pool);
            (1, s)
        }
    })
}

fn latex_signed_mul(args: &[ExprId], pool: &ExprPool) -> (i32, String) {
    let mut numer_i = 1i64;
    let mut denom_i = 1i64;
    let mut others: Vec<ExprId> = Vec::new();

    for &child in args {
        pool.with(child, |data| match data {
            ExprData::Integer(n) => {
                numer_i *= n.0.to_i64().unwrap_or(1);
            }
            ExprData::Rational(r) => {
                numer_i *= r.0.numer().to_i64().unwrap_or(1);
                denom_i *= r.0.denom().to_i64().unwrap_or(1);
            }
            _ => others.push(child),
        });
    }

    let sign = if numer_i < 0 {
        numer_i = -numer_i;
        -1i32
    } else {
        1i32
    };
    let sign = if denom_i < 0 {
        denom_i = -denom_i;
        -sign
    } else {
        sign
    };

    let mut num_parts: Vec<String> = Vec::new();
    let mut den_parts: Vec<String> = Vec::new();

    for &child in &others {
        let pushed = pool.with(child, |data| {
            if let ExprData::Pow { base, exp } = data {
                if let ExprData::Integer(n) = pool.get(*exp) {
                    let v = n.0.to_i64().unwrap_or(0);
                    if v < 0 {
                        let exp_abs = -v;
                        let base_tex = latex_wrap(*base, pool, PREC_POW + 1);
                        if exp_abs == 1 {
                            den_parts.push(base_tex);
                        } else {
                            den_parts.push(format!("{{{base_tex}}}^{{{exp_abs}}}"));
                        }
                        return true;
                    }
                }
            }
            false
        });
        if !pushed {
            num_parts.push(latex_wrap(child, pool, PREC_MUL));
        }
    }

    if numer_i != 1 || denom_i != 1 {
        let coeff = if denom_i != 1 {
            latex_frac(&numer_i.to_string(), &denom_i.to_string())
        } else {
            numer_i.to_string()
        };
        num_parts.insert(0, coeff);
    }

    if num_parts.is_empty() && den_parts.is_empty() {
        return (sign, "1".into());
    }

    if !den_parts.is_empty() {
        let num_str = if num_parts.is_empty() {
            "1".into()
        } else {
            num_parts.join(" ")
        };
        let den_str = den_parts.join(" ");
        return (sign, latex_frac(&num_str, &den_str));
    }

    (sign, num_parts.join(" "))
}

fn latex_add(args: &[ExprId], pool: &ExprPool) -> String {
    let mut parts: Vec<String> = Vec::new();
    for &child in args {
        let (sign, tex) = latex_signed(child, pool);
        if parts.is_empty() {
            if sign < 0 {
                parts.push(format!("-{tex}"));
            } else {
                parts.push(tex);
            }
        } else if sign < 0 {
            parts.push(format!(" - {tex}"));
        } else {
            parts.push(format!(" + {tex}"));
        }
    }
    parts.concat()
}

fn latex_pow(base: ExprId, exp: ExprId, pool: &ExprPool) -> String {
    // x^(1/n) → nth-root
    if let ExprData::Rational(r) = pool.get(exp) {
        let num = r.0.numer().to_i64().unwrap_or(0);
        let den = r.0.denom().to_i64().unwrap_or(1);
        if num == 1 && den >= 2 {
            let base_tex = latex_wrap(base, pool, PREC_POW + 1);
            return if den == 2 {
                format!(r"\sqrt{{{base_tex}}}")
            } else {
                format!(r"\sqrt[{den}]{{{base_tex}}}")
            };
        }
    }
    // x^(-1) → 1/x
    if let ExprData::Integer(n) = pool.get(exp) {
        if n.0.to_i64() == Some(-1) {
            let base_tex = latex_wrap(base, pool, PREC_POW + 1);
            return latex_frac("1", &base_tex);
        }
    }
    let base_tex = latex_wrap(base, pool, PREC_POW);
    let (exp_tex, _) = latex_r(exp, pool);
    let exp_braced = if exp_tex.len() == 1 {
        exp_tex
    } else {
        format!("{{{exp_tex}}}")
    };
    format!("{base_tex}^{exp_braced}")
}

fn latex_func(name: &str, args: &[ExprId], pool: &ExprPool) -> String {
    match name {
        "abs" => {
            let (inner, _) = latex_r(args[0], pool);
            format!(r"\left|{inner}\right|")
        }
        "floor" => {
            let (inner, _) = latex_r(args[0], pool);
            format!(r"\lfloor {inner} \rfloor")
        }
        "ceil" => {
            let (inner, _) = latex_r(args[0], pool);
            format!(r"\lceil {inner} \rceil")
        }
        "sqrt" => {
            let (inner, _) = latex_r(args[0], pool);
            format!(r"\sqrt{{{inner}}}")
        }
        "exp" => {
            let (inner, _) = latex_r(args[0], pool);
            let exp_braced = if inner.len() == 1 {
                inner
            } else {
                format!("{{{inner}}}")
            };
            format!(r"e^{exp_braced}")
        }
        _ => {
            let fn_latex = latex_func_name(name);
            let rendered: Vec<String> = args.iter().map(|&a| latex_r(a, pool).0).collect();
            format!(r"{fn_latex}\!\left({}\right)", rendered.join(", "))
        }
    }
}

fn latex_func_name(name: &str) -> String {
    match name {
        "sin" => r"\sin".into(),
        "cos" => r"\cos".into(),
        "tan" => r"\tan".into(),
        "sinh" => r"\sinh".into(),
        "cosh" => r"\cosh".into(),
        "tanh" => r"\tanh".into(),
        "asin" => r"\arcsin".into(),
        "acos" => r"\arccos".into(),
        "atan" => r"\arctan".into(),
        "log" => r"\ln".into(),
        "sign" => r"\operatorname{sign}".into(),
        "round" => r"\operatorname{round}".into(),
        "erf" => r"\operatorname{erf}".into(),
        "erfc" => r"\operatorname{erfc}".into(),
        "gamma" => r"\Gamma".into(),
        other => format!(r"\operatorname{{{other}}}"),
    }
}

fn latex_predicate(kind: &PredicateKind, args: &[ExprId], pool: &ExprPool) -> String {
    match kind {
        PredicateKind::True => r"\top".into(),
        PredicateKind::False => r"\bot".into(),
        PredicateKind::Not => {
            let (inner, _) = latex_r(args[0], pool);
            format!(r"\lnot {inner}")
        }
        _ => {
            let op = match kind {
                PredicateKind::Lt => "<",
                PredicateKind::Le => r"\le",
                PredicateKind::Gt => ">",
                PredicateKind::Ge => r"\ge",
                PredicateKind::Eq => "=",
                PredicateKind::Ne => r"\ne",
                PredicateKind::And => r"\land",
                PredicateKind::Or => r"\lor",
                _ => unreachable!(),
            };
            let rendered: Vec<String> = args.iter().map(|&a| latex_r(a, pool).0).collect();
            rendered.join(&format!(" {op} "))
        }
    }
}

fn latex_piecewise(branches: &[(ExprId, ExprId)], default: ExprId, pool: &ExprPool) -> String {
    let mut rows: Vec<String> = Vec::new();
    for &(cond, val) in branches {
        let (val_tex, _) = latex_r(val, pool);
        let (cond_tex, _) = latex_r(cond, pool);
        rows.push(format!(r"{val_tex} & \text{{if }} {cond_tex}"));
    }
    let (def_tex, _) = latex_r(default, pool);
    rows.push(format!(r"{def_tex} & \text{{otherwise}}"));
    format!(r"\begin{{cases}} {} \end{{cases}}", rows.join(r" \\ "))
}

fn latex_r(id: ExprId, pool: &ExprPool) -> (String, i32) {
    pool.with(id, |data| match data {
        ExprData::Symbol { name, .. } => (latex_symbol(name), PREC_ATOM),
        ExprData::Integer(n) => (n.0.to_string(), PREC_ATOM),
        ExprData::Rational(r) => {
            let num = r.0.numer();
            let den = r.0.denom();
            let s = latex_frac(num.to_string().trim_start_matches('-'), &den.to_string());
            if *num < 0 {
                (format!("-{s}"), PREC_ATOM)
            } else {
                (s, PREC_ATOM)
            }
        }
        ExprData::Float(f) => (f.inner.to_string(), PREC_ATOM),
        ExprData::Add(args) => (latex_add(args, pool), PREC_ADD),
        ExprData::Mul(args) => {
            let (sign, tex) = latex_signed_mul(args, pool);
            let s = if sign < 0 { format!("-{tex}") } else { tex };
            (s, PREC_MUL)
        }
        ExprData::Pow { base, exp } => (latex_pow(*base, *exp, pool), PREC_POW),
        ExprData::Func { name, args } => (latex_func(name, args, pool), PREC_ATOM),
        ExprData::Piecewise { branches, default } => {
            (latex_piecewise(branches, *default, pool), PREC_ATOM)
        }
        ExprData::Predicate { kind, args } => (latex_predicate(kind, args, pool), PREC_ADD),
        ExprData::Forall { var, body } => {
            let (v, _) = latex_r(*var, pool);
            let (b, _) = latex_r(*body, pool);
            (format!(r"\forall {v} \, . \, {b}"), PREC_ATOM)
        }
        ExprData::Exists { var, body } => {
            let (v, _) = latex_r(*var, pool);
            let (b, _) = latex_r(*body, pool);
            (format!(r"\exists {v} \, . \, {b}"), PREC_ATOM)
        }
        ExprData::BigO(arg) => {
            let (a, _) = latex_r(*arg, pool);
            (format!(r"\mathcal{{O}}\!\left({a}\right)"), PREC_ATOM)
        }
    })
}

// ---------------------------------------------------------------------------
// Unicode renderer
// ---------------------------------------------------------------------------

fn unicode_symbol(name: &str) -> String {
    if let Some(g) = greek_unicode(name) {
        return g.to_string();
    }
    name.to_string()
}

fn unicode_wrap(id: ExprId, pool: &ExprPool, req_prec: i32) -> String {
    let (s, prec) = unicode_r(id, pool);
    if prec < req_prec {
        format!("({s})")
    } else {
        s
    }
}

fn unicode_signed(id: ExprId, pool: &ExprPool) -> (i32, String) {
    pool.with(id, |data| match data {
        ExprData::Integer(n) => {
            let v = n.0.to_i64().unwrap_or(0);
            if v < 0 {
                (-1, (-v).to_string())
            } else {
                (1, v.to_string())
            }
        }
        ExprData::Rational(r) => {
            let num = r.0.numer().to_i64().unwrap_or(0);
            let den = r.0.denom().to_i64().unwrap_or(1);
            if num < 0 {
                (-1, unicode_frac(-num, den))
            } else {
                (1, unicode_frac(num, den))
            }
        }
        ExprData::Mul(args) => unicode_signed_mul(args, pool),
        _ => {
            let (s, _) = unicode_r(id, pool);
            (1, s)
        }
    })
}

fn unicode_signed_mul(args: &[ExprId], pool: &ExprPool) -> (i32, String) {
    let mut numer_i = 1i64;
    let mut denom_i = 1i64;
    let mut others: Vec<ExprId> = Vec::new();

    for &child in args {
        pool.with(child, |data| match data {
            ExprData::Integer(n) => {
                numer_i *= n.0.to_i64().unwrap_or(1);
            }
            ExprData::Rational(r) => {
                numer_i *= r.0.numer().to_i64().unwrap_or(1);
                denom_i *= r.0.denom().to_i64().unwrap_or(1);
            }
            _ => others.push(child),
        });
    }

    let sign = if numer_i < 0 {
        numer_i = -numer_i;
        -1i32
    } else {
        1i32
    };
    let sign = if denom_i < 0 {
        denom_i = -denom_i;
        -sign
    } else {
        sign
    };

    let mut num_parts: Vec<String> = Vec::new();
    let mut den_parts: Vec<String> = Vec::new();

    for &child in &others {
        let pushed = pool.with(child, |data| {
            if let ExprData::Pow { base, exp } = data {
                if let ExprData::Integer(n) = pool.get(*exp) {
                    let v = n.0.to_i64().unwrap_or(0);
                    if v < 0 {
                        let exp_abs = -v;
                        let base_tex = unicode_wrap(*base, pool, PREC_POW + 1);
                        let sup = to_superscript(&exp_abs.to_string());
                        den_parts.push(match sup {
                            Some(s) => format!("{base_tex}{s}"),
                            None => format!("{base_tex}^{exp_abs}"),
                        });
                        return true;
                    }
                }
            }
            false
        });
        if !pushed {
            num_parts.push(unicode_wrap(child, pool, PREC_MUL));
        }
    }

    if numer_i != 1 || denom_i != 1 {
        let coeff = if denom_i != 1 {
            unicode_frac(numer_i, denom_i)
        } else {
            numer_i.to_string()
        };
        num_parts.insert(0, coeff);
    }

    if num_parts.is_empty() && den_parts.is_empty() {
        return (sign, "1".into());
    }

    if !den_parts.is_empty() {
        let num_str = if num_parts.is_empty() {
            "1".into()
        } else {
            num_parts.join("·")
        };
        let den_str = den_parts.join("·");
        let s = if den_parts.len() > 1 {
            format!("({num_str})/({den_str})")
        } else {
            format!("{num_str}/{den_str}")
        };
        return (sign, s);
    }

    (sign, num_parts.join("·"))
}

fn unicode_add(args: &[ExprId], pool: &ExprPool) -> String {
    let mut parts: Vec<String> = Vec::new();
    for &child in args {
        let (sign, tex) = unicode_signed(child, pool);
        if parts.is_empty() {
            if sign < 0 {
                parts.push(format!("-{tex}"));
            } else {
                parts.push(tex);
            }
        } else if sign < 0 {
            parts.push(format!(" - {tex}"));
        } else {
            parts.push(format!(" + {tex}"));
        }
    }
    parts.concat()
}

fn unicode_pow(base: ExprId, exp: ExprId, pool: &ExprPool) -> String {
    if let ExprData::Rational(r) = pool.get(exp) {
        let num = r.0.numer().to_i64().unwrap_or(0);
        let den = r.0.denom().to_i64().unwrap_or(1);
        if num == 1 {
            let base_tex = unicode_wrap(base, pool, PREC_POW + 1);
            return match den {
                2 => format!("√{base_tex}"),
                3 => format!("∛{base_tex}"),
                4 => format!("∜{base_tex}"),
                _ => {
                    let sup = to_superscript(&den.to_string())
                        .map(|s| format!("{}√{base_tex}", s))
                        .unwrap_or_else(|| format!("{base_tex}^(1/{den})"));
                    sup
                }
            };
        }
    }
    let base_tex = unicode_wrap(base, pool, PREC_POW);
    if let ExprData::Integer(n) = pool.get(exp) {
        if let Some(v) = n.0.to_i64() {
            if let Some(sup) = to_superscript(&v.to_string()) {
                return format!("{base_tex}{sup}");
            }
        }
    }
    let (exp_tex, _) = unicode_r(exp, pool);
    format!("{base_tex}^({exp_tex})")
}

fn unicode_func(name: &str, args: &[ExprId], pool: &ExprPool) -> String {
    match name {
        "sqrt" => {
            let inner = unicode_wrap(args[0], pool, PREC_POW + 1);
            format!("√{inner}")
        }
        "abs" => {
            let (inner, _) = unicode_r(args[0], pool);
            format!("|{inner}|")
        }
        "floor" => {
            let (inner, _) = unicode_r(args[0], pool);
            format!("⌊{inner}⌋")
        }
        "ceil" => {
            let (inner, _) = unicode_r(args[0], pool);
            format!("⌈{inner}⌉")
        }
        "exp" => {
            let (inner, _) = unicode_r(args[0], pool);
            if let Some(sup) = to_superscript(&inner) {
                format!("e{sup}")
            } else {
                format!("e^({inner})")
            }
        }
        _ => {
            let fn_name = match name {
                "gamma" => "Γ",
                "asin" => "arcsin",
                "acos" => "arccos",
                "atan" => "arctan",
                "log" => "ln",
                other => other,
            };
            let rendered: Vec<String> = args.iter().map(|&a| unicode_r(a, pool).0).collect();
            format!("{fn_name}({})", rendered.join(", "))
        }
    }
}

fn unicode_predicate(kind: &PredicateKind, args: &[ExprId], pool: &ExprPool) -> String {
    match kind {
        PredicateKind::True => "⊤".into(),
        PredicateKind::False => "⊥".into(),
        PredicateKind::Not => {
            let (inner, _) = unicode_r(args[0], pool);
            format!("¬{inner}")
        }
        _ => {
            let op = match kind {
                PredicateKind::Lt => "<",
                PredicateKind::Le => "≤",
                PredicateKind::Gt => ">",
                PredicateKind::Ge => "≥",
                PredicateKind::Eq => "=",
                PredicateKind::Ne => "≠",
                PredicateKind::And => "∧",
                PredicateKind::Or => "∨",
                _ => unreachable!(),
            };
            let rendered: Vec<String> = args.iter().map(|&a| unicode_r(a, pool).0).collect();
            rendered.join(&format!(" {op} "))
        }
    }
}

fn unicode_piecewise(branches: &[(ExprId, ExprId)], default: ExprId, pool: &ExprPool) -> String {
    let mut rows: Vec<String> = Vec::new();
    for &(cond, val) in branches {
        let (val_tex, _) = unicode_r(val, pool);
        let (cond_tex, _) = unicode_r(cond, pool);
        rows.push(format!("{val_tex}  if {cond_tex}"));
    }
    let (def_tex, _) = unicode_r(default, pool);
    rows.push(format!("{def_tex}  otherwise"));
    format!("{{ {}", rows.join("\n  "))
}

fn unicode_r(id: ExprId, pool: &ExprPool) -> (String, i32) {
    pool.with(id, |data| match data {
        ExprData::Symbol { name, .. } => (unicode_symbol(name), PREC_ATOM),
        ExprData::Integer(n) => (n.0.to_string(), PREC_ATOM),
        ExprData::Rational(r) => {
            let num = r.0.numer().to_i64().unwrap_or(0);
            let den = r.0.denom().to_i64().unwrap_or(1);
            let s = unicode_frac(num.abs(), den);
            if num < 0 {
                (format!("-{s}"), PREC_ATOM)
            } else {
                (s, PREC_ATOM)
            }
        }
        ExprData::Float(f) => (f.inner.to_string(), PREC_ATOM),
        ExprData::Add(args) => (unicode_add(args, pool), PREC_ADD),
        ExprData::Mul(args) => {
            let (sign, tex) = unicode_signed_mul(args, pool);
            let s = if sign < 0 { format!("-{tex}") } else { tex };
            (s, PREC_MUL)
        }
        ExprData::Pow { base, exp } => (unicode_pow(*base, *exp, pool), PREC_POW),
        ExprData::Func { name, args } => (unicode_func(name, args, pool), PREC_ATOM),
        ExprData::Piecewise { branches, default } => {
            (unicode_piecewise(branches, *default, pool), PREC_ATOM)
        }
        ExprData::Predicate { kind, args } => (unicode_predicate(kind, args, pool), PREC_ADD),
        ExprData::Forall { var, body } => {
            let (v, _) = unicode_r(*var, pool);
            let (b, _) = unicode_r(*body, pool);
            (format!("∀{v}.{b}"), PREC_ATOM)
        }
        ExprData::Exists { var, body } => {
            let (v, _) = unicode_r(*var, pool);
            let (b, _) = unicode_r(*body, pool);
            (format!("∃{v}.{b}"), PREC_ATOM)
        }
        ExprData::BigO(arg) => {
            let (a, _) = unicode_r(*arg, pool);
            (format!("O({a})"), PREC_ATOM)
        }
    })
}
