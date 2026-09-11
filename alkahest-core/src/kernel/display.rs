/// LaTeX and Unicode pretty-printing for symbolic expressions.
///
/// Mirrors the logic of `python/alkahest/_pretty.py` but dispatches on the
/// typed `ExprData` enum instead of string-tagged Python lists, eliminating
/// the `if t == "symbol":` / `if t == "add":` pattern-matching in Python.
use crate::kernel::expr::{ExprData, PredicateKind};
use crate::kernel::{ExprId, ExprPool};

/// Logical connectives, all *below* [`PREC_ADD`] so an arithmetic operand of a
/// comparison is never parenthesised.  The relative order is the standard one
/// (`∀/∃` scope widest, then `∨`, then `∧`, then a comparison, then `¬`): a
/// printer that renders them all at one level emits `a ∨ b ∧ c` for
/// `(a ∨ b) ∧ c`, which re-reads as `a ∨ (b ∧ c)` — a different proposition.
const PREC_QUANT: i32 = 2;
const PREC_OR: i32 = 4;
const PREC_AND: i32 = 6;
const PREC_CMP: i32 = 8;
const PREC_NOT: i32 = 9;
const PREC_ADD: i32 = 10;
const PREC_MUL: i32 = 20;
/// Unary minus: binds tighter than `*` but looser than `^`, matching `BP_UNARY`
/// in `parse.rs` (and Python, and sympy).  A literal that renders with a leading
/// `-` therefore has to be parenthesised as a power base — `(-1)^n`, never
/// `-1^n`, which would re-read as `-(1^n)`.
const PREC_NEG: i32 = 25;
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

/// Precedence of a rendered numeric literal: [`PREC_NEG`] when it carries a
/// leading `-`, [`PREC_ATOM`] otherwise.
fn literal_prec(rendered: &str) -> i32 {
    if rendered.starts_with('-') {
        PREC_NEG
    } else {
        PREC_ATOM
    }
}

/// Precedence of a rendered float.
///
/// `rug::Float` prints `0.25` as `2.5e-1`, which is not an atom: the `-1` is
/// exposed, so `√0.25` printed as `√2.5e-1` reads as `(√2.5e) − 1` and
/// `0.25²` as `2.5e-1²`.  Treating it as a quotient gets it parenthesised
/// wherever a quotient would be.
fn float_prec(rendered: &str) -> i32 {
    if rendered.contains(['e', 'E']) {
        PREC_MUL.min(literal_prec(rendered))
    } else {
        literal_prec(rendered)
    }
}

/// Collapse maximal runs of the *identical* factor into `(factor, count)`.
///
/// `Mul([x, x])` is `x²`; printing it as `x x` (LaTeX) sets two juxtaposed
/// variables, which is how a power stops being visible in the output.  Only
/// **consecutive** equal ids are merged, so the rewrite never reorders a
/// product: `ExprPool::mul` sorts its arguments *only* when every factor is
/// multiplicatively commutative, and this has to stay correct for the
/// non-commutative generators too.
fn collapse_runs(args: &[ExprId]) -> Vec<(ExprId, i64)> {
    let mut out: Vec<(ExprId, i64)> = Vec::with_capacity(args.len());
    for &id in args {
        match out.last_mut() {
            Some((prev, n)) if *prev == id => *n += 1,
            _ => out.push((id, 1)),
        }
    }
    out
}

/// `Some(numerator)` when a rational is really an integer (`4/2` interns as
/// `Rational(2, 1)`, a node distinct from `Integer(2)`).  Rendering it as
/// `\frac{2}{1}` or `2/1` is not wrong, but `x^(1/1)` reached the Unicode
/// `ⁿ√` branch and printed `¹√x`.
fn rational_as_integer(r: &crate::kernel::expr::BigRat) -> Option<String> {
    if *r.0.denom() == 1 {
        Some(r.0.numer().to_string())
    } else {
        None
    }
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

/// `base^exp`, braced unless the exponent is a single character.
fn latex_sup(base_tex: &str, exp_tex: &str) -> String {
    if exp_tex.chars().count() == 1 {
        format!("{base_tex}^{exp_tex}")
    } else {
        format!("{base_tex}^{{{exp_tex}}}")
    }
}

/// A float literal, with scientific notation typeset rather than copied.
///
/// `rug::Float` prints `0.25` as `2.5e-1`.  Emitted verbatim into math mode
/// that reads as *e minus 1* — a subtraction against a variable called `e` —
/// so the exponent has to become an actual power of ten.  The result is a
/// product, hence [`PREC_MUL`] rather than an atom's precedence.
fn latex_float(s: &str) -> (String, i32) {
    match s.split_once(['e', 'E']) {
        Some((mantissa, exponent)) if !exponent.is_empty() => {
            let exponent = exponent.strip_prefix('+').unwrap_or(exponent);
            (
                format!(r"{mantissa} \times 10^{{{exponent}}}"),
                PREC_MUL.min(literal_prec(mantissa)),
            )
        }
        _ => (s.to_string(), literal_prec(s)),
    }
}

/// Render one factor of a product.
///
/// [`latex_wrap`] at [`PREC_MUL`] is not enough on its own: a factor that
/// renders with a leading `-` (a negative `Float`, a directly interned nested
/// `Mul`) is juxtaposed against its left neighbour as `x -1.5`, which sets —
/// and re-reads — as the *subtraction* `x − 1.5`.
fn latex_factor(id: ExprId, pool: &ExprPool) -> String {
    let (s, prec) = latex_r(id, pool);
    if prec < PREC_MUL || s.starts_with('-') {
        format!(r"\left({s}\right)")
    } else {
        s
    }
}

/// Join the factors of a product.
///
/// LaTeX multiplication is juxtaposition, so a factor that *begins with a
/// digit* fuses with whatever precedes it: `2 \cdot 3^n` written as `2 3^n`
/// sets as `23^n`, and `1.5 \cdot 2.5` as `1.52.5`.  An explicit `\cdot` is
/// the standard separator in exactly that position.
fn latex_join_factors(parts: &[String]) -> String {
    let mut out = String::new();
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            if part.starts_with(|c: char| c.is_ascii_digit() || c == '.') {
                out.push_str(r" \cdot ");
            } else {
                out.push(' ');
            }
        }
        out.push_str(part);
    }
    out
}

fn latex_symbol(name: &str) -> String {
    // Exact Greek name: "alpha" -> "\alpha"
    if let Some(g) = greek_latex(name) {
        return g.to_string();
    }
    // Explicit underscore subscript: "u_0" -> "{u}_{0}", "alpha_1" -> "{\alpha}_{1}"
    if let Some((base, sub)) = name.split_once('_') {
        let base_tex = greek_latex(base)
            .map(str::to_string)
            .unwrap_or_else(|| base.to_string());
        return format!("{{{base_tex}}}_{{{sub}}}");
    }
    // Implicit subscript: trailing digits on a pure-letter base.
    // "u0" -> "{u}_{0}", "x12" -> "{x}_{12}", "alpha1" -> "{\alpha}_{1}"
    if let Some(i) = name.find(|c: char| c.is_ascii_digit()).filter(|&i| {
        i > 0
            && name[..i].chars().all(|c| c.is_ascii_alphabetic())
            && name[i..].chars().all(|c| c.is_ascii_digit())
    }) {
        let base = &name[..i];
        let sub = &name[i..];
        let base_tex = greek_latex(base)
            .map(str::to_string)
            .unwrap_or_else(|| base.to_string());
        return format!("{{{base_tex}}}_{{{sub}}}");
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
            let s = match rational_as_integer(r) {
                Some(n) => n.trim_start_matches('-').to_string(),
                None => latex_frac(num.to_string().trim_start_matches('-'), &den.to_string()),
            };
            if *num < 0 {
                (-1, s)
            } else {
                (1, s)
            }
        }
        ExprData::Mul(args) => latex_signed_mul(args, pool),
        _ => {
            let (s, _) = latex_r(id, pool);
            // A negative float is a subtraction in a sum, like every other
            // negative literal: `x + -1.5` becomes `x - 1.5`.
            match s.strip_prefix('-') {
                Some(rest) if matches!(data, ExprData::Float(_)) => (-1, rest.to_string()),
                _ => (1, s),
            }
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

    for (child, mult) in collapse_runs(&others) {
        let pushed = pool.with(child, |data| {
            if let ExprData::Pow { base, exp } = data {
                if let ExprData::Integer(n) = pool.get(*exp) {
                    let v = n.0.to_i64().unwrap_or(0);
                    if v < 0 {
                        let exp_abs = (-v).saturating_mul(mult);
                        if exp_abs == 1 {
                            // `\frac{}{}` groups its denominator already, so the
                            // factor only has to be safe against its siblings.
                            den_parts.push(latex_factor(*base, pool));
                        } else {
                            let base_tex = latex_wrap(*base, pool, PREC_POW + 1);
                            den_parts.push(latex_sup(&base_tex, &exp_abs.to_string()));
                        }
                        return true;
                    }
                }
            }
            false
        });
        if !pushed {
            if mult > 1 {
                let base_tex = latex_wrap(child, pool, PREC_POW + 1);
                num_parts.push(latex_sup(&base_tex, &mult.to_string()));
            } else {
                num_parts.push(latex_factor(child, pool));
            }
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
            latex_join_factors(&num_parts)
        };
        let den_str = latex_join_factors(&den_parts);
        return (sign, latex_frac(&num_str, &den_str));
    }

    (sign, latex_join_factors(&num_parts))
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
            // The radical is delimited by its own braces, so the radicand needs
            // no parentheses of its own.
            let (base_tex, _) = latex_r(base, pool);
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
            let base_tex = latex_factor(base, pool);
            return latex_frac("1", &base_tex);
        }
    }
    let base_tex = latex_wrap(base, pool, PREC_POW + 1);
    let (exp_tex, _) = latex_r(exp, pool);
    latex_sup(&base_tex, &exp_tex)
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
            latex_sup("e", &inner)
        }
        // Elliptic integrals (parameter convention m = k²).  We mirror the
        // classical typeset notation: K(m), E(m), F(φ|m), E(φ|m), Π(n;φ|m).
        "EllipticK" => {
            let (m, _) = latex_r(args[0], pool);
            format!(r"K\!\left({m}\right)")
        }
        "EllipticE" if args.len() == 1 => {
            let (m, _) = latex_r(args[0], pool);
            format!(r"E\!\left({m}\right)")
        }
        "EllipticE" => {
            let (phi, _) = latex_r(args[0], pool);
            let (m, _) = latex_r(args[1], pool);
            format!(r"E\!\left({phi}\,\middle|\,{m}\right)")
        }
        "EllipticF" => {
            let (phi, _) = latex_r(args[0], pool);
            let (m, _) = latex_r(args[1], pool);
            format!(r"F\!\left({phi}\,\middle|\,{m}\right)")
        }
        "EllipticPi" => {
            let (n, _) = latex_r(args[0], pool);
            let (phi, _) = latex_r(args[1], pool);
            let (m, _) = latex_r(args[2], pool);
            format!(r"\Pi\!\left({n};{phi}\,\middle|\,{m}\right)")
        }
        _ => {
            let fn_latex = latex_func_name(name);
            let rendered: Vec<String> = args.iter().map(|&a| latex_r(a, pool).0).collect();
            format!(r"{fn_latex}\!\left({}\right)", rendered.join(", "))
        }
    }
}

/// Precedence of a rendered function application.
///
/// A call is an atom — `\sin\!\left(x\right)` carries its own delimiters — with
/// one exception per renderer: a form that ends in an *open* superscript or
/// starts with a prefix operator does not, and appending `^2` to it silently
/// re-scopes the exponent.  In LaTeX that is `exp`, printed `e^{u}`: written as
/// a power base it produces `e^{u}^2`, which is not only ambiguous but the hard
/// TeX error "Double superscript".  In Unicode it is `exp` *and* `sqrt`, since
/// `√u` has no closing delimiter and `√u²` reads as `√(u²)`.
fn latex_func_prec(name: &str) -> i32 {
    match name {
        "exp" => PREC_POW,
        _ => PREC_ATOM,
    }
}

fn unicode_func_prec(name: &str) -> i32 {
    match name {
        "exp" | "sqrt" => PREC_POW,
        _ => PREC_ATOM,
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
        "asinh" => r"\operatorname{arsinh}".into(),
        "acosh" => r"\operatorname{arcosh}".into(),
        "atanh" => r"\operatorname{artanh}".into(),
        "log" => r"\ln".into(),
        "sign" => r"\operatorname{sign}".into(),
        "round" => r"\operatorname{round}".into(),
        "erf" => r"\operatorname{erf}".into(),
        "erfc" => r"\operatorname{erfc}".into(),
        "gamma" => r"\Gamma".into(),
        "digamma" => r"\psi".into(),
        "lambert_w" => r"W".into(),
        "bessel_j0" => r"J_0".into(),
        "bessel_j1" => r"J_1".into(),
        // Exponential-integral family. `\operatorname` (not `\mathrm`)
        // because these are function names, and upright either way; the
        // fallback arm below would produce the same text, but pinning them
        // here keeps them from drifting if the fallback changes.
        "Ei" => r"\operatorname{Ei}".into(),
        "li" => r"\operatorname{li}".into(),
        "Si" => r"\operatorname{Si}".into(),
        "Ci" => r"\operatorname{Ci}".into(),
        "Shi" => r"\operatorname{Shi}".into(),
        "Chi" => r"\operatorname{Chi}".into(),
        // 3.10.0.
        // 3.10.0.  `S`/`C` are the DLMF §7.2(iii) names for the *normalised*
        // (π/2) Fresnel integrals, `ψ₁` is trigamma, and `Li₂` is the
        // principal-branch dilogarithm.
        "fresnels" => r"S".into(),
        "fresnelc" => r"C".into(),
        "trigamma" => r"\psi_1".into(),
        "dilog" => r"\operatorname{Li}_2".into(),
        // `\operatorname{}` typesets its argument in math mode, where a bare
        // `_` is a subscript: `\operatorname{my_func}` renders as `my_func`
        // with a subscripted `f`, and a second underscore is the hard error
        // "Double subscript" that stops a document compiling.
        other => format!(r"\operatorname{{{}}}", other.replace('_', r"\_")),
    }
}

/// Precedence of a rendered predicate, and the precedence its operands have to
/// clear.  `¬` and the quantifiers take the *whole* proposition to their right,
/// so their operand requirement is one above everything a connective produces.
fn predicate_prec(kind: &PredicateKind) -> i32 {
    match kind {
        PredicateKind::True | PredicateKind::False => PREC_ATOM,
        PredicateKind::Not => PREC_NOT,
        PredicateKind::And => PREC_AND,
        PredicateKind::Or => PREC_OR,
        _ => PREC_CMP,
    }
}

fn latex_predicate(kind: &PredicateKind, args: &[ExprId], pool: &ExprPool) -> String {
    match kind {
        PredicateKind::True => r"\top".into(),
        PredicateKind::False => r"\bot".into(),
        PredicateKind::Not => {
            let inner = latex_wrap(args[0], pool, PREC_NOT + 1);
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
            // `∧`/`∨` are associative, so a same-kind operand needs no
            // parentheses; anything that binds *looser* does.
            let req = predicate_prec(kind);
            let rendered: Vec<String> = args.iter().map(|&a| latex_wrap(a, pool, req)).collect();
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
        ExprData::Integer(n) => (n.0.to_string(), literal_prec(&n.0.to_string())),
        ExprData::Rational(r) => {
            let num = r.0.numer();
            let den = r.0.denom();
            if let Some(n) = rational_as_integer(r) {
                let prec = literal_prec(&n);
                return (n, prec);
            }
            let s = latex_frac(num.to_string().trim_start_matches('-'), &den.to_string());
            // A fraction is a quotient, not an atom: it needs parentheses under
            // `^` just like any other product/quotient does.
            if *num < 0 {
                (format!("-{s}"), PREC_MUL)
            } else {
                (s, PREC_MUL)
            }
        }
        ExprData::Float(f) => latex_float(&f.inner.to_string()),
        ExprData::Add(args) => (latex_add(args, pool), PREC_ADD),
        ExprData::Mul(args) => {
            let (sign, tex) = latex_signed_mul(args, pool);
            let s = if sign < 0 { format!("-{tex}") } else { tex };
            (s, PREC_MUL)
        }
        ExprData::Pow { base, exp } => (latex_pow(*base, *exp, pool), PREC_POW),
        ExprData::Func { name, args } => (latex_func(name, args, pool), latex_func_prec(name)),
        ExprData::Piecewise { branches, default } => {
            (latex_piecewise(branches, *default, pool), PREC_ATOM)
        }
        ExprData::Predicate { kind, args } => {
            (latex_predicate(kind, args, pool), predicate_prec(kind))
        }
        ExprData::Forall { var, body } => {
            let (v, _) = latex_r(*var, pool);
            let (b, _) = latex_r(*body, pool);
            (format!(r"\forall {v} \, . \, {b}"), PREC_QUANT)
        }
        ExprData::Exists { var, body } => {
            let (v, _) = latex_r(*var, pool);
            let (b, _) = latex_r(*body, pool);
            (format!(r"\exists {v} \, . \, {b}"), PREC_QUANT)
        }
        ExprData::BigO(arg) => {
            let (a, _) = latex_r(*arg, pool);
            (format!(r"\mathcal{{O}}\!\left({a}\right)"), PREC_ATOM)
        }
        ExprData::RootSum { poly, var, body } => {
            let (p, _) = latex_r(*poly, pool);
            let (v, _) = latex_r(*var, pool);
            let (b, _) = latex_r(*body, pool);
            (format!(r"\sum_{{{v} \, : \, {p} = 0}} {b}"), PREC_ATOM)
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

/// `base^exp` in Unicode superscripts where the exponent allows it.
fn unicode_sup(base_tex: &str, exp: &str) -> String {
    match to_superscript(exp) {
        Some(s) => format!("{base_tex}{s}"),
        None => format!("{base_tex}^({exp})"),
    }
}

/// One factor of a product.  See [`latex_factor`]: a factor rendering with a
/// leading `-` reads as a subtraction when it follows the `·`-free path, and
/// `x·-1.5` is at best unpleasant.
fn unicode_factor(id: ExprId, pool: &ExprPool) -> String {
    let (s, prec) = unicode_r(id, pool);
    if prec < PREC_MUL || s.starts_with('-') {
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
            let render = |n: i64| match rational_as_integer(r) {
                Some(_) => n.to_string(),
                None => unicode_frac(n, den),
            };
            if num < 0 {
                (-1, render(-num))
            } else {
                (1, render(num))
            }
        }
        ExprData::Mul(args) => unicode_signed_mul(args, pool),
        _ => {
            let (s, _) = unicode_r(id, pool);
            match s.strip_prefix('-') {
                Some(rest) if matches!(data, ExprData::Float(_)) => (-1, rest.to_string()),
                _ => (1, s),
            }
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

    for (child, mult) in collapse_runs(&others) {
        let pushed = pool.with(child, |data| {
            if let ExprData::Pow { base, exp } = data {
                if let ExprData::Integer(n) = pool.get(*exp) {
                    let v = n.0.to_i64().unwrap_or(0);
                    if v < 0 {
                        let exp_abs = (-v).saturating_mul(mult);
                        let base_tex = unicode_wrap(*base, pool, PREC_POW + 1);
                        // `x^-1` is `1/x`, not `1/x¹`: the exponent has already
                        // been spent by moving the factor into the denominator.
                        den_parts.push(if exp_abs == 1 {
                            base_tex
                        } else {
                            unicode_sup(&base_tex, &exp_abs.to_string())
                        });
                        return true;
                    }
                }
            }
            false
        });
        if !pushed {
            if mult > 1 {
                let base_tex = unicode_wrap(child, pool, PREC_POW + 1);
                num_parts.push(unicode_sup(&base_tex, &mult.to_string()));
            } else {
                num_parts.push(unicode_factor(child, pool));
            }
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
        // `den >= 2` guards the radical branch: `x^(1/1)` is `x`, and the
        // fallback below printed it as the nonsense `¹√x`.
        if num == 1 && den >= 2 {
            let base_tex = unicode_wrap(base, pool, PREC_POW + 1);
            return match den {
                2 => format!("√{base_tex}"),
                3 => format!("∛{base_tex}"),
                4 => format!("∜{base_tex}"),
                _ => to_superscript(&den.to_string())
                    .map(|s| format!("{s}√{base_tex}"))
                    .unwrap_or_else(|| format!("{base_tex}^(1/{den})")),
            };
        }
    }
    let base_tex = unicode_wrap(base, pool, PREC_POW + 1);
    if let Some(v) = integral_exponent(exp, pool) {
        if v == 1 {
            return base_tex;
        }
        if let Some(sup) = to_superscript(&v.to_string()) {
            return format!("{base_tex}{sup}");
        }
    }
    let (exp_tex, _) = unicode_r(exp, pool);
    format!("{base_tex}^({exp_tex})")
}

/// The exponent as an `i64` when it is one — an `Integer`, or a `Rational`
/// that reduced to one (`Rational(2, 1)` is a distinct node from `Integer(2)`).
fn integral_exponent(exp: ExprId, pool: &ExprPool) -> Option<i64> {
    match pool.get(exp) {
        ExprData::Integer(n) => n.0.to_i64(),
        ExprData::Rational(r) if *r.0.denom() == 1 => r.0.numer().to_i64(),
        _ => None,
    }
}

fn unicode_func(name: &str, args: &[ExprId], pool: &ExprPool) -> String {
    match name {
        "sqrt" => {
            let inner = unicode_wrap(args[0], pool, PREC_POW + 1);
            format!("√{inner}")
        }
        "abs" => {
            let (inner, _) = unicode_r(args[0], pool);
            // `|` is its own mirror, so a nested absolute value gives `||2||`,
            // which no reader can pair up.  Parenthesising the inner one makes
            // the grouping explicit: `|(|2|)|`.
            if inner.contains('|') {
                format!("|({inner})|")
            } else {
                format!("|{inner}|")
            }
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
                "digamma" => "ψ",
                "trigamma" => "ψ₁",
                "bessel_j0" => "J₀",
                "bessel_j1" => "J₁",
                "lambert_w" => "W",
                // Exponential-integral family: the ASCII spelling *is* the
                // conventional Unicode one (DLMF renders `Ei`, `li`, `Si`,
                // `Ci`, `Shi`, `Chi` upright), so these are pinned identities
                // rather than translations. `Chi` in particular must not
                // become `\u{3a7}` — it is a function, not the Greek letter.
                "Ei" => "Ei",
                "li" => "li",
                "Si" => "Si",
                "Ci" => "Ci",
                "Shi" => "Shi",
                "Chi" => "Chi",
                // DLMF §7.2(iii) names for the normalised (π/2) Fresnel
                // integrals, and the principal-branch dilogarithm.
                "fresnels" => "S",
                "fresnelc" => "C",
                "dilog" => "Li₂",
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
            let inner = unicode_wrap(args[0], pool, PREC_NOT + 1);
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
            let req = predicate_prec(kind);
            let rendered: Vec<String> = args.iter().map(|&a| unicode_wrap(a, pool, req)).collect();
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
        ExprData::Integer(n) => (n.0.to_string(), literal_prec(&n.0.to_string())),
        ExprData::Rational(r) => {
            let num = r.0.numer().to_i64().unwrap_or(0);
            let den = r.0.denom().to_i64().unwrap_or(1);
            let s = match rational_as_integer(r) {
                Some(_) => num.abs().to_string(),
                None => unicode_frac(num.abs(), den),
            };
            // `unicode_frac` returns a single vulgar-fraction glyph (`½`) for a
            // handful of values and a `num/den` quotient otherwise; only the
            // former is atomic under `^`.
            let prec = if s.contains('/') { PREC_MUL } else { PREC_ATOM };
            if num < 0 {
                (format!("-{s}"), prec.min(PREC_NEG))
            } else {
                (s, prec)
            }
        }
        ExprData::Float(f) => {
            let s = f.inner.to_string();
            let prec = float_prec(&s);
            (s, prec)
        }
        ExprData::Add(args) => (unicode_add(args, pool), PREC_ADD),
        ExprData::Mul(args) => {
            let (sign, tex) = unicode_signed_mul(args, pool);
            let s = if sign < 0 { format!("-{tex}") } else { tex };
            (s, PREC_MUL)
        }
        ExprData::Pow { base, exp } => (unicode_pow(*base, *exp, pool), PREC_POW),
        ExprData::Func { name, args } => (unicode_func(name, args, pool), unicode_func_prec(name)),
        ExprData::Piecewise { branches, default } => {
            (unicode_piecewise(branches, *default, pool), PREC_ATOM)
        }
        ExprData::Predicate { kind, args } => {
            (unicode_predicate(kind, args, pool), predicate_prec(kind))
        }
        ExprData::Forall { var, body } => {
            let (v, _) = unicode_r(*var, pool);
            let (b, _) = unicode_r(*body, pool);
            (format!("∀{v}.{b}"), PREC_QUANT)
        }
        ExprData::Exists { var, body } => {
            let (v, _) = unicode_r(*var, pool);
            let (b, _) = unicode_r(*body, pool);
            (format!("∃{v}.{b}"), PREC_QUANT)
        }
        ExprData::BigO(arg) => {
            let (a, _) = unicode_r(*arg, pool);
            (format!("O({a})"), PREC_ATOM)
        }
        ExprData::RootSum { poly, var, body } => {
            let (p, _) = unicode_r(*poly, pool);
            let (v, _) = unicode_r(*var, pool);
            let (b, _) = unicode_r(*body, pool);
            (format!("∑_{{{v}:{p}=0}} {b}"), PREC_ATOM)
        }
    })
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    /// A negative power base must be parenthesised in every exported form:
    /// `-1^n` means `-(1^n)` in LaTeX and in every parser that reads these
    /// strings back, so `(-1)^n` is the only round-trippable rendering.
    #[test]
    fn negative_pow_base_is_parenthesised() {
        let p = ExprPool::new();
        let n = p.symbol("n", Domain::Real);
        let pow_m1 = p.pow(p.integer(-1_i32), n);
        assert_eq!(render_latex(pow_m1, &p), r"\left(-1\right)^n");
        assert_eq!(render_unicode(pow_m1, &p), "(-1)^(n)");

        let pow_m2 = p.pow(p.integer(-2_i32), n);
        assert_eq!(render_latex(pow_m2, &p), r"\left(-2\right)^n");
        assert_eq!(render_unicode(pow_m2, &p), "(-2)^(n)");

        let pow_mhalf = p.pow(p.rational(-1, 2), n);
        assert_eq!(render_latex(pow_mhalf, &p), r"\left(-\frac{1}{2}\right)^n");
        assert_eq!(render_unicode(pow_mhalf, &p), "(-½)^(n)");
    }

    /// A fractional base is a quotient, not an atom, so it needs parentheses
    /// too — except where the Unicode renderer has a single glyph for it.
    #[test]
    fn fractional_pow_base_is_parenthesised() {
        let p = ExprPool::new();
        let n = p.symbol("n", Domain::Real);
        let pow_half = p.pow(p.rational(1, 2), n);
        assert_eq!(render_latex(pow_half, &p), r"\left(\frac{1}{2}\right)^n");
        assert_eq!(render_unicode(pow_half, &p), "½^(n)");

        let pow_3_7 = p.pow(p.rational(3, 7), n);
        assert_eq!(render_latex(pow_3_7, &p), r"\left(\frac{3}{7}\right)^n");
        assert_eq!(render_unicode(pow_3_7, &p), "(3/7)^(n)");
    }

    /// The negative base survives being embedded in a product — this is the
    /// `b(n) = -16 * (-2)^n` inhomogeneity shape that surfaced the bug.
    #[test]
    fn negative_pow_base_inside_product() {
        let p = ExprPool::new();
        let n = p.symbol("n", Domain::Real);
        let prod = p.mul(vec![p.integer(-16_i32), p.pow(p.integer(-2_i32), n)]);
        assert_eq!(render_latex(prod, &p), r"-16 \left(-2\right)^n");
        assert_eq!(render_unicode(prod, &p), "-16·(-2)^(n)");
    }

    /// Non-negative atoms stay bare — the fix must not add noise everywhere.
    #[test]
    fn positive_pow_base_is_bare() {
        let p = ExprPool::new();
        let n = p.symbol("n", Domain::Real);
        let x = p.symbol("x", Domain::Real);
        let pow_2 = p.pow(p.integer(2_i32), n);
        assert_eq!(render_latex(pow_2, &p), "2^n");
        assert_eq!(render_unicode(pow_2, &p), "2^(n)");
        let pow_x = p.pow(x, p.integer(2_i32));
        assert_eq!(render_latex(pow_x, &p), "x^2");
        assert_eq!(render_unicode(pow_x, &p), "x²");
    }

    /// A negative coefficient in a product is still printed bare (`-2 x`):
    /// unary minus binds tighter than `*`, so parentheses are unnecessary.
    #[test]
    fn negative_coefficient_in_product_stays_bare() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let prod = p.mul(vec![p.integer(-2_i32), x]);
        assert_eq!(render_latex(prod, &p), "-2 x");
        assert_eq!(render_unicode(prod, &p), "-2·x");
    }

    /// `Mul([x, x])` is `x²`.  Printed as `x x` it sets as `xx`, because LaTeX
    /// math mode discards white space — the power stops being visible.
    #[test]
    fn a_repeated_factor_prints_as_a_power() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        assert_eq!(render_latex(p.mul(vec![x, x]), &p), "x^2");
        assert_eq!(render_unicode(p.mul(vec![x, x]), &p), "x²");
        assert_eq!(render_latex(p.mul(vec![x, x, x]), &p), "x^3");
        assert_eq!(render_latex(p.mul(vec![x, x, y]), &p), "x^2 y");
        // Distinct factors stay a juxtaposed product.
        assert_eq!(render_latex(p.mul(vec![x, y]), &p), "x y");
        // A repeated function application too.
        let s = p.func("sin", vec![x]);
        assert_eq!(
            render_latex(p.mul(vec![s, s]), &p),
            r"\sin\!\left(x\right)^2"
        );
    }

    /// Only *consecutive* equal factors are merged, so a non-commutative
    /// product — which `ExprPool::mul` does not sort — is never reordered.
    #[test]
    fn run_collapsing_does_not_reorder_a_product() {
        assert_eq!(
            collapse_runs(&[ExprId(1), ExprId(2), ExprId(1)]),
            vec![(ExprId(1), 1), (ExprId(2), 1), (ExprId(1), 1)]
        );
        assert_eq!(
            collapse_runs(&[ExprId(1), ExprId(1), ExprId(2)]),
            vec![(ExprId(1), 2), (ExprId(2), 1)]
        );
    }

    /// A factor that begins with a digit fuses with the one before it: `2 3^n`
    /// sets as `23^n`.  `\cdot` is the standard separator in that position.
    #[test]
    fn a_numeral_before_a_numeral_gets_a_cdot() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let n = p.symbol("n", Domain::Real);
        let two_three_n = p.mul(vec![p.integer(2_i32), p.pow(p.integer(3_i32), n)]);
        assert_eq!(render_latex(two_three_n, &p), r"2 \cdot 3^n");
        // A coefficient before a *symbol* is unambiguous and stays bare.
        assert_eq!(render_latex(p.mul(vec![p.integer(2_i32), x]), &p), "2 x");
    }

    /// A factor whose rendering starts with `-` is read as a subtraction:
    /// `x -1.5` is `x − 1.5`, a different expression.
    #[test]
    fn a_negative_float_factor_is_parenthesised() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let prod = p.mul(vec![x, p.float(-1.5_f64, 53)]);
        assert!(
            render_latex(prod, &p).contains(r"\left(-1.5"),
            "{}",
            render_latex(prod, &p)
        );
        assert!(render_unicode(prod, &p).contains("(-1.5"));
    }

    /// `rug::Float` prints `0.25` as `2.5e-1`; in math mode that reads as
    /// `2.5·e − 1`.  LaTeX gets a real power of ten.
    #[test]
    fn a_float_in_exponent_notation_is_typeset_as_a_power_of_ten() {
        let p = ExprPool::new();
        let quarter = p.float(0.25_f64, 53);
        assert_eq!(
            render_latex(quarter, &p),
            r"2.5000000000000000 \times 10^{-1}"
        );
        // It is a product, so it is parenthesised where a product would be.
        let n = p.symbol("n", Domain::Real);
        assert_eq!(
            render_latex(p.pow(quarter, n), &p),
            r"\left(2.5000000000000000 \times 10^{-1}\right)^n"
        );
        // Unicode keeps the machine-readable spelling, but it is not an atom:
        // `√2.5e-1` would read as `(√2.5e) − 1`.
        assert_eq!(
            render_unicode(p.pow(quarter, n), &p),
            "(2.5000000000000000e-1)^(n)"
        );
    }

    /// `x^-1` is `1/x`, not `1/x¹`: moving the factor into the denominator is
    /// what spends the exponent.
    #[test]
    fn a_reciprocal_factor_carries_no_exponent_of_one() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let one = p.integer(1_i32);
        let denom = p.add(vec![one, x]);
        let recip = p.mul(vec![one, p.pow(denom, p.integer(-1_i32))]);
        assert_eq!(render_unicode(recip, &p), "1/(x + 1)");
        assert_eq!(render_latex(recip, &p), r"\frac{1}{\left(x + 1\right)}");
        // A genuine exponent survives.
        let squared = p.mul(vec![one, p.pow(denom, p.integer(-2_i32))]);
        assert_eq!(render_unicode(squared, &p), "1/(x + 1)²");
    }

    /// `∧` binds tighter than `∨`, so `(a ∨ b) ∧ c` needs the group and
    /// `a ∨ (b ∧ c)` does not.  Printed at one level, the first re-reads as the
    /// second — a different proposition.
    #[test]
    fn connective_precedence_is_parenthesised() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let a = p.predicate(PredicateKind::Lt, vec![x, p.integer(0_i32)]);
        let b = p.predicate(PredicateKind::Gt, vec![x, p.integer(1_i32)]);
        let c = p.predicate(PredicateKind::Eq, vec![x, p.integer(2_i32)]);
        let or_in_and = p.predicate(
            PredicateKind::And,
            vec![p.predicate(PredicateKind::Or, vec![a, b]), c],
        );
        assert_eq!(
            render_latex(or_in_and, &p),
            r"\left(x < 0 \lor x > 1\right) \land x = 2"
        );
        assert_eq!(render_unicode(or_in_and, &p), "(x < 0 ∨ x > 1) ∧ x = 2");

        let and_in_or = p.predicate(
            PredicateKind::Or,
            vec![a, p.predicate(PredicateKind::And, vec![b, c])],
        );
        assert_eq!(render_latex(and_in_or, &p), r"x < 0 \lor x > 1 \land x = 2");

        // `¬` takes one operand, not the rest of the line.
        let not_and = p.predicate(
            PredicateKind::Not,
            vec![p.predicate(PredicateKind::And, vec![a, b])],
        );
        assert_eq!(
            render_latex(not_and, &p),
            r"\lnot \left(x < 0 \land x > 1\right)"
        );
        assert_eq!(render_unicode(not_and, &p), "¬(x < 0 ∧ x > 1)");
    }

    /// A quantifier scopes over everything to its right, so it has to be
    /// grouped when it is an operand of a connective.
    #[test]
    fn a_quantifier_inside_a_connective_is_parenthesised() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let a = p.predicate(PredicateKind::Lt, vec![x, p.integer(0_i32)]);
        let b = p.predicate(PredicateKind::Gt, vec![x, p.integer(1_i32)]);
        let quantified = p.forall(x, a);
        let conj = p.predicate(PredicateKind::And, vec![quantified, b]);
        assert_eq!(
            render_latex(conj, &p),
            r"\left(\forall x \, . \, x < 0\right) \land x > 1"
        );
        assert_eq!(render_unicode(conj, &p), "(∀x.x < 0) ∧ x > 1");
    }

    /// `exp` prints as `e^{u}` and `sqrt` as `√u`; neither closes, so as a
    /// power base both have to be grouped.  `e^{x}^2` is also the hard TeX
    /// error "Double superscript".
    #[test]
    fn an_open_ended_function_rendering_is_grouped_as_a_power_base() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let two = p.integer(2_i32);
        let e = p.func("exp", vec![x]);
        assert_eq!(render_latex(p.pow(e, two), &p), r"\left(e^x\right)^2");
        assert_eq!(render_unicode(p.pow(e, two), &p), "(e^(x))²");
        let r = p.func("sqrt", vec![x]);
        assert_eq!(render_unicode(p.pow(r, two), &p), "(√x)²");
        // A delimited rendering still needs no group.
        let s = p.func("sin", vec![x]);
        assert_eq!(render_latex(p.pow(s, two), &p), r"\sin\!\left(x\right)^2");
    }

    /// A bare `_` inside `\operatorname{}` is a subscript, and two of them are
    /// the error "Double subscript".  A symbol's subscript is left alone.
    #[test]
    fn an_unregistered_head_escapes_its_underscores() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        assert_eq!(
            render_latex(p.func("my_func", vec![x]), &p),
            r"\operatorname{my\_func}\!\left(x\right)"
        );
        assert_eq!(render_latex(p.symbol("u_0", Domain::Real), &p), "{u}_{0}");
    }

    /// `4/2` interns as `Rational(2, 1)`, a node distinct from `Integer(2)`.
    /// It is an integer on the page, and `x^(1/1)` reached the Unicode `ⁿ√`
    /// branch and printed the nonsense `¹√x`.
    #[test]
    fn a_rational_with_denominator_one_prints_as_an_integer() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let two_over_one = p.rational(4, 2);
        assert_eq!(render_latex(two_over_one, &p), "2");
        assert_eq!(render_unicode(two_over_one, &p), "2");
        assert_eq!(render_latex(p.pow(x, p.rational(1, 1)), &p), "x^1");
        assert_eq!(render_unicode(p.pow(x, p.rational(1, 1)), &p), "x");
    }

    /// `|` is its own mirror, so `||x||` has no unique reading.
    #[test]
    fn a_nested_absolute_value_is_grouped() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let inner = p.func("abs", vec![x]);
        assert_eq!(render_unicode(inner, &p), "|x|");
        assert_eq!(render_unicode(p.func("abs", vec![inner]), &p), "|(|x|)|");
    }
}
