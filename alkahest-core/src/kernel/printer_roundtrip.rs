//! The round-trip property for the printers, over a generated corpus.
//!
//! A printer that emits something which re-parses to a *different* expression
//! is a silent wrong answer with extra steps: the user copies the output, feeds
//! it back, and gets different mathematics with no error raised anywhere.  So
//! round-trip fidelity is tested as a correctness property rather than as a
//! collection of golden strings.
//!
//! The property is checked in two ways, and a case passes if either holds:
//!
//! * **structural**, after a normalisation that only removes a difference the
//!   parser cannot avoid — the parser builds binary `Add`/`Mul` spines
//!   (`parse("a*b*c")` is `(a*b)*c`) while [`ExprPool::mul`] builds a flat
//!   n-ary node, and the two are documented to be unequal (CONTRIBUTING.md,
//!   "The parser exists twice").  [`renormalise`] rebuilds both trees bottom-up
//!   through the pool constructors, which flatten and canonically sort exactly
//!   those two node kinds and change nothing else;
//! * **semantic**, by evaluating both at sample points.  This is what catches a
//!   precedence or juxtaposition error, which renormalisation cannot repair.
//!
//! A case where neither side evaluates anywhere (`log` of a negative argument
//! at every sample point, an overflowed power) is counted as *undecided* and
//! reported; the test fails if the undecided share grows large enough to make
//! the corpus vacuous.

#![cfg(test)]

use std::collections::HashMap;

use crate::eval::eval_f64;
use crate::kernel::expr::ExprData;
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::parse::parse;

// ---------------------------------------------------------------------------
// Corpus generation
// ---------------------------------------------------------------------------

/// xorshift64*, so the corpus is identical on every machine and a failure is
/// reproducible from the seed printed in the assertion message.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
}

const SYMBOL_NAMES: [&str; 3] = ["a", "b", "c"];
const FUNCS: [&str; 8] = ["sin", "cos", "exp", "log", "sqrt", "abs", "atan", "erf"];

fn atom(rng: &mut Rng, pool: &ExprPool, syms: &[ExprId]) -> ExprId {
    match rng.below(10) {
        0..=3 => syms[rng.below(syms.len() as u64) as usize],
        4..=6 => pool.integer(rng.pick(&[-7_i64, -2, -1, 0, 1, 2, 3, 10])),
        7..=8 => {
            let (n, d) = rng.pick(&[(1_i64, 2_i64), (-1, 2), (3, 7), (-5, 3), (4, 2), (2, 1)]);
            pool.rational(n, d)
        }
        _ => pool.float(rng.pick(&[-1.5_f64, 0.25, 3.5, 2.0]), 53),
    }
}

fn build(rng: &mut Rng, pool: &ExprPool, syms: &[ExprId], depth: u32) -> ExprId {
    if depth == 0 {
        return atom(rng, pool, syms);
    }
    let lhs = build(rng, pool, syms, depth - 1);
    let rhs = build(rng, pool, syms, depth - 1);
    match rng.below(14) {
        0 => pool.add(vec![lhs, rhs]),
        1 => pool.add(vec![lhs, rhs, atom(rng, pool, syms)]),
        // Subtraction, as `diff`/`simplify` build it.
        2 => pool.add(vec![lhs, pool.mul(vec![pool.integer(-1_i64), rhs])]),
        3 => pool.mul(vec![lhs, rhs]),
        4 => pool.mul(vec![lhs, rhs, atom(rng, pool, syms)]),
        // Division, and the repeated factor that has to print as a power.
        5 => pool.mul(vec![lhs, pool.pow(rhs, pool.integer(-1_i64))]),
        6 => pool.mul(vec![lhs, lhs]),
        7 => pool.mul(vec![pool.integer(-1_i64), lhs]),
        8 => pool.pow(lhs, pool.integer(rng.pick(&[-3_i64, -2, -1, 2, 3]))),
        9 => pool.pow(lhs, pool.rational(1, rng.pick(&[2_i64, 3, 5]))),
        10 => pool.pow(lhs, rhs),
        11 => pool.pow(pool.pow(lhs, rhs), atom(rng, pool, syms)),
        12 => pool.func(rng.pick(&FUNCS), vec![lhs]),
        _ => pool.func("atan2", vec![lhs, rhs]),
    }
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Rebuild `id` bottom-up through the pool constructors.
///
/// The only nodes this changes are `Add` and `Mul`, which `ExprPool::add` /
/// `ExprPool::mul` splice flat and sort when every factor is commutative.  That
/// is precisely the documented difference between a parsed spine and a
/// builder-constructed n-ary node; nothing else is normalised, so a precedence
/// or sign error survives it.
fn renormalise(id: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(id) {
        ExprData::Add(args) => pool.add(
            args.iter()
                .map(|&a| renormalise(a, pool))
                .collect::<Vec<_>>(),
        ),
        ExprData::Mul(args) => {
            let parts: Vec<ExprId> = args.iter().map(|&a| renormalise(a, pool)).collect();
            fold_exact_coefficient(parts, pool)
        }
        ExprData::Pow { base, exp } => pool.pow(renormalise(base, pool), renormalise(exp, pool)),
        ExprData::Func { name, args } => pool.func(
            name,
            args.iter()
                .map(|&a| renormalise(a, pool))
                .collect::<Vec<_>>(),
        ),
        _ => id,
    }
}

/// Multiply out the exact numeric factors of a product into one atom.
///
/// The second artefact the parser cannot avoid: there is no rational *literal*
/// token, so `1/2` necessarily comes back as `1 · 2^-1` rather than as the
/// `Rational(1, 2)` atom that was printed.  Folding exact integer/rational
/// factors — and `q^-1` for an exact `q` — is the inverse of that and touches
/// nothing else; a wrong coefficient still compares unequal, because both sides
/// are folded the same way.
fn fold_exact_coefficient(parts: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    let mut coeff = rug::Rational::from(1);
    let mut rest: Vec<ExprId> = Vec::new();
    for part in parts {
        match pool.get(part) {
            ExprData::Integer(n) => coeff *= rug::Rational::from(n.0.clone()),
            ExprData::Rational(r) => coeff *= r.0.clone(),
            ExprData::Pow { base, exp } => {
                let reciprocal = matches!(pool.get(exp), ExprData::Integer(e) if e.0 == -1);
                let value = match pool.get(base) {
                    ExprData::Integer(n) if n.0 != 0 => Some(rug::Rational::from(n.0.clone())),
                    ExprData::Rational(r) if *r.0.numer() != 0 => Some(r.0.clone()),
                    _ => None,
                };
                match (reciprocal, value) {
                    (true, Some(v)) => coeff /= v,
                    _ => rest.push(part),
                }
            }
            _ => rest.push(part),
        }
    }
    let (num, den) = (coeff.numer().to_i64(), coeff.denom().to_i64());
    let atom = match (num, den) {
        (Some(n), Some(1)) => pool.integer(n),
        (Some(n), Some(d)) => pool.rational(n, d),
        // Out of `i64` range: leave the product alone rather than guess.
        _ => return pool.mul(rest),
    };
    if rest.is_empty() {
        return atom;
    }
    if coeff == 1 {
        return pool.mul(rest);
    }
    rest.push(atom);
    pool.mul(rest)
}

/// Sample points for the semantic comparison.  All strictly positive: the
/// corpus contains `log`, `sqrt` and fractional powers, whose real branch is
/// undefined on the negatives, and an undefined value decides nothing.
const SAMPLES: [[f64; 3]; 5] = [
    [0.37, 1.21, 2.03],
    [1.7, 0.61, 0.44],
    [2.9, 3.3, 1.05],
    [0.11, 2.4, 0.83],
    [1.0, 1.0, 1.0],
];

/// A sample point only decides a case when the value is inside a window where
/// an `f64` comparison still carries information.  A power tower reaches
/// `1e84` from inputs of order one, and there a last-bit difference is
/// multiplied by every exponent above it.  Such a point is reported as
/// undecided, never as a mismatch.
///
/// The comparison is deliberately **real only**.  Evaluating on the complex
/// principal branch decides more cases, but it also makes two forms that are
/// exactly equal on paper — `(-2)^-3` against `((-2)^3)^-1` — disagree by
/// orders of magnitude, because their `f64` imaginary parts land on opposite
/// sides of zero, `arg` flips between `+pi` and `-pi`, and a non-integer
/// exponent above them turns that into a different branch.  A test that
/// reports a branch cut as a printer defect is worse than a test that reports
/// fewer cases.
fn decides(v: f64) -> bool {
    let m = v.abs();
    m == 0.0 || (1e-10..=1e10).contains(&m)
}

/// Relative agreement demanded of the two evaluations.  Loose enough to
/// survive a power tower, whose relative error is multiplied by every exponent
/// above it; a printer that loses a precedence or a sign is out by orders of
/// magnitude, not by parts in a million.
const TOL: f64 = 1e-6;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Verdict {
    Structural,
    Numeric,
    Undecided,
    Mismatch,
}

fn compare(original: ExprId, reparsed: ExprId, pool: &ExprPool, syms: &[ExprId]) -> Verdict {
    if original == reparsed || renormalise(original, pool) == renormalise(reparsed, pool) {
        return Verdict::Structural;
    }
    let mut agreed = 0;
    for point in SAMPLES.iter() {
        let bindings: HashMap<ExprId, f64> =
            syms.iter().copied().zip(point.iter().copied()).collect();
        let want = match eval_f64(original, pool, &bindings) {
            Ok(v) if decides(v) => v,
            _ => continue,
        };
        // The printed form has to be usable wherever the original was.
        let got = match eval_f64(reparsed, pool, &bindings) {
            Err(_) => return Verdict::Mismatch,
            Ok(v) => v,
        };
        let scale = want.abs().max(got.abs()).max(1.0);
        if (want - got).abs() > TOL * scale {
            if std::env::var("PRINTER_DEBUG").is_ok() {
                eprintln!("DIFF want={want:?} got={got:?}");
                eprintln!("  original: {}", pool.display(original));
                eprintln!("  reparsed: {}", pool.display(reparsed));
            }
            return Verdict::Mismatch;
        }
        agreed += 1;
    }
    if agreed == 0 {
        Verdict::Undecided
    } else {
        Verdict::Numeric
    }
}

// ---------------------------------------------------------------------------
// Reading the typeset forms back
// ---------------------------------------------------------------------------

/// Boundary between two source tokens in an expanded LaTeX string.
const UNIT: &str = "\u{1}";

/// Function names a juxtaposed `(` binds to as an *application* rather than as
/// a product.  Everything else followed by `(` is multiplication.
const CALLABLE: [&str; 13] = [
    "sin", "cos", "tan", "log", "exp", "sqrt", "abs", "atan", "asin", "acos", "erf", "atan2",
    "gamma",
];

fn take_group(s: &[char], i: &mut usize) -> Result<String, String> {
    if s.get(*i) != Some(&'{') {
        return Err(format!("expected '{{' at offset {i}"));
    }
    let start = *i + 1;
    let mut depth = 1usize;
    let mut j = start;
    while j < s.len() && depth > 0 {
        match s[j] {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
        j += 1;
    }
    if depth != 0 {
        return Err("unbalanced braces".into());
    }
    *i = j;
    Ok(s[start..j - 1].iter().collect())
}

/// Expand the emitted LaTeX into the parser's ASCII grammar, one *unit* per
/// source token, joined by [`UNIT`].
///
/// **Whitespace is dropped**, because that is what LaTeX math mode does with
/// it: `2 3^n` sets exactly as `23^n`.  Modelling the space as a multiplication
/// would assume away the very defect this is looking for; keeping the unit
/// boundary is what lets [`insert_juxtaposition`] decide, for each adjacent
/// pair, whether a reader sees a product or one fused token.
fn expand_latex(src: &str) -> Result<String, String> {
    let s: Vec<char> = src.chars().collect();
    let mut units: Vec<String> = Vec::new();
    macro_rules! emit {
        ($e:expr) => {
            units.push(String::from($e))
        };
    }
    let mut i = 0usize;
    while i < s.len() {
        let c = s[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '\\' {
            let mut j = i + 1;
            while j < s.len() && s[j].is_ascii_alphabetic() {
                j += 1;
            }
            let (name, mut k) = if j == i + 1 {
                (s[i + 1..i + 2].iter().collect::<String>(), i + 2)
            } else {
                (s[i + 1..j].iter().collect::<String>(), j)
            };
            match name.as_str() {
                "!" | "," | ";" | " " | "quad" => i = k,
                "cdot" | "times" => {
                    emit!("*");
                    i = k;
                }
                "left" | "right" => {
                    let d = *s.get(k).ok_or_else(|| format!("dangling \\{name}"))?;
                    match (name.as_str(), d) {
                        ("left", '(') => emit!("("),
                        ("left", '|') => emit!("abs("),
                        ("right", ')') | ("right", '|') => emit!(")"),
                        _ => return Err(format!("unhandled \\{name}{d}")),
                    }
                    i = k + 1;
                }
                "frac" => {
                    let num = take_group(&s, &mut k)?;
                    let den = take_group(&s, &mut k)?;
                    emit!(format!(
                        "(({})/({}))",
                        expand_latex(&num)?,
                        expand_latex(&den)?
                    ));
                    i = k;
                }
                "sqrt" => {
                    if s.get(k) == Some(&'[') {
                        let close = s[k..]
                            .iter()
                            .position(|&ch| ch == ']')
                            .ok_or("unterminated \\sqrt[")?
                            + k;
                        let index: String = s[k + 1..close].iter().collect();
                        let mut kk = close + 1;
                        let radicand = take_group(&s, &mut kk)?;
                        emit!(format!(
                            "(({}))^(1/({}))",
                            expand_latex(&radicand)?,
                            expand_latex(&index)?
                        ));
                        i = kk;
                    } else {
                        let radicand = take_group(&s, &mut k)?;
                        emit!(format!("sqrt({})", expand_latex(&radicand)?));
                        i = k;
                    }
                }
                "operatorname" => {
                    let body = take_group(&s, &mut k)?;
                    emit!(body.replace("\\_", "_"));
                    i = k;
                }
                other => {
                    let ascii = match other {
                        "sin" => "sin",
                        "cos" => "cos",
                        "tan" => "tan",
                        "sinh" => "sinh",
                        "cosh" => "cosh",
                        "tanh" => "tanh",
                        "arcsin" => "asin",
                        "arccos" => "acos",
                        "arctan" => "atan",
                        "ln" => "log",
                        _ => return Err(format!("unhandled macro \\{other}")),
                    };
                    emit!(ascii);
                    i = k;
                }
            }
            continue;
        }
        // `exp(u)` is emitted as `e^{u}`; no corpus symbol is called `e`.
        if c == 'e' && s.get(i + 1) == Some(&'^') {
            let mut k = i + 2;
            let arg = if s.get(k) == Some(&'{') {
                take_group(&s, &mut k)?
            } else {
                let ch = *s.get(k).ok_or("dangling e^")?;
                k += 1;
                ch.to_string()
            };
            emit!(format!("exp({})", expand_latex(&arg)?));
            i = k;
            continue;
        }
        if c == '^' {
            // `^` scopes to exactly one token — `x^c y` is `(x^c)·y`, not
            // `x^(cy)` — so a brace-less exponent is one character.
            i += 1;
            let mut k = i;
            let exponent = if s.get(i) == Some(&'{') {
                take_group(&s, &mut k)?
            } else {
                let ch = *s.get(i).ok_or("dangling ^")?;
                k = i + 1;
                ch.to_string()
            };
            // The exponent belongs to the same unit as the `^` that carries it.
            emit!(format!("^({})", expand_latex(&exponent)?));
            i = k;
            continue;
        }
        if c == '{' {
            let mut k = i;
            let g = take_group(&s, &mut k)?;
            emit!(format!("({})", expand_latex(&g)?));
            i = k;
            continue;
        }
        if c == '}' {
            return Err("stray '}'".into());
        }
        // A numeral is one token: `23` is twenty-three, whereas `2 3` is two
        // tokens that the page fuses into twenty-three.  Only the second is a
        // defect, so the two must not look alike here.
        if c.is_ascii_digit() {
            let start = i;
            while i < s.len() && (s[i].is_ascii_digit() || s[i] == '.') {
                i += 1;
            }
            emit!(s[start..i].iter().collect::<String>());
            continue;
        }
        emit!(c.to_string());
        i += 1;
    }
    units.retain(|u| !u.is_empty());
    Ok(units.join(UNIT))
}

/// Resolve each token boundary the way the page resolves it.
///
/// LaTeX multiplication is juxtaposition, so at a boundary a reader either sees
/// a product or sees the two tokens *fuse into one*:
///
/// * digit next to digit (`2 3` → `23`) and letter followed by digit
///   (`x 2^n` → `x2^n`) fuse — the emitter has to break these with `\cdot`;
/// * everything else reads as a product (`2 x`, `(a+b)c`, `x \sin(y)`), except
///   a function name followed by `(`, which is an application.
fn insert_juxtaposition(src: &str) -> String {
    #[derive(PartialEq, Clone, Copy)]
    enum Class {
        Digit,
        Letter,
        Open,
        Close,
        Other,
    }
    fn class(c: char) -> Class {
        if c.is_ascii_digit() || c == '.' {
            Class::Digit
        } else if c.is_ascii_alphabetic() || c == '_' {
            Class::Letter
        } else if c == '(' {
            Class::Open
        } else if c == ')' {
            Class::Close
        } else {
            Class::Other
        }
    }

    let mut out = String::new();
    for unit in src.split(UNIT) {
        let Some(first) = unit.chars().next() else {
            continue;
        };
        if let Some(last) = out.chars().last() {
            let prev = class(last);
            let here = class(first);
            let juxtaposed = matches!(prev, Class::Digit | Class::Letter | Class::Close)
                && matches!(here, Class::Digit | Class::Letter | Class::Open);
            // A fused pair is deliberately left fused: it is what the reader
            // gets, and the round trip is supposed to notice.
            let fuses = matches!(
                (prev, here),
                (Class::Digit, Class::Digit) | (Class::Letter, Class::Digit)
            );
            let ident: String = out
                .chars()
                .rev()
                .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let is_call = here == Class::Open && CALLABLE.contains(&ident.as_str());
            if juxtaposed && !fuses && !is_call {
                out.push('*');
            }
        }
        out.push_str(unit);
    }
    out
}

fn read_latex(src: &str) -> Result<String, String> {
    Ok(insert_juxtaposition(&expand_latex(src)?))
}

/// Superscript digits back to an ASCII exponent.
fn from_superscript(c: char) -> Option<char> {
    Some(match c {
        '⁰' => '0',
        '¹' => '1',
        '²' => '2',
        '³' => '3',
        '⁴' => '4',
        '⁵' => '5',
        '⁶' => '6',
        '⁷' => '7',
        '⁸' => '8',
        '⁹' => '9',
        '⁺' => '+',
        '⁻' => '-',
        _ => return None,
    })
}

fn vulgar_fraction(c: char) -> Option<&'static str> {
    Some(match c {
        '½' => "(1/2)",
        '⅓' => "(1/3)",
        '⅔' => "(2/3)",
        '¼' => "(1/4)",
        '¾' => "(3/4)",
        '⅕' => "(1/5)",
        '⅖' => "(2/5)",
        '⅗' => "(3/5)",
        '⅘' => "(4/5)",
        '⅙' => "(1/6)",
        '⅚' => "(5/6)",
        '⅐' => "(1/7)",
        '⅛' => "(1/8)",
        '⅜' => "(3/8)",
        '⅝' => "(5/8)",
        '⅞' => "(7/8)",
        '⅑' => "(1/9)",
        '⅒' => "(1/10)",
        _ => return None,
    })
}

/// Read the Unicode pretty-printed form back.  Unlike LaTeX this form writes
/// every product with an explicit `·`, so there is no juxtaposition to recover;
/// what has to be undone is the superscripts, the radicals, the bars and the
/// vulgar fractions.
fn read_unicode(src: &str) -> Result<String, String> {
    let s: Vec<char> = src.chars().collect();
    let mut out = String::new();
    let mut i = 0usize;
    while i < s.len() {
        let c = s[i];
        // A superscript run is an exponent — unless a radical follows, in which
        // case it is the root *index* (`⁵√x`).
        if from_superscript(c).is_some() {
            let mut j = i;
            let mut digits = String::new();
            while j < s.len() {
                match from_superscript(s[j]) {
                    Some(d) => {
                        digits.push(d);
                        j += 1;
                    }
                    None => break,
                }
            }
            if s.get(j) == Some(&'√') {
                let mut k = j + 1;
                let operand = read_unicode_atom(&s, &mut k)?;
                out.push_str(&format!("({operand})^(1/{digits})"));
                i = k;
            } else {
                out.push_str(&format!("^({digits})"));
                i = j;
            }
            continue;
        }
        if let Some(f) = vulgar_fraction(c) {
            out.push_str(f);
            i += 1;
            continue;
        }
        match c {
            '·' => {
                out.push('*');
                i += 1;
            }
            '√' | '∛' | '∜' => {
                let index = match c {
                    '√' => 2,
                    '∛' => 3,
                    _ => 4,
                };
                let mut k = i + 1;
                let operand = read_unicode_atom(&s, &mut k)?;
                out.push_str(&format!("({operand})^(1/{index})"));
                i = k;
            }
            '|' => {
                let (inner, next) = read_bar_group(&s, i)?;
                out.push_str(&format!("abs({})", read_unicode(&inner)?));
                i = next;
            }
            'e' if s.get(i + 1) == Some(&'^') && s.get(i + 2) == Some(&'(') => {
                let close = matching_paren(&s, i + 2)?;
                let inner: String = s[i + 3..close].iter().collect();
                out.push_str(&format!("exp({})", read_unicode(&inner)?));
                i = close + 1;
            }
            'e' if s.get(i + 1).copied().and_then(from_superscript).is_some() => {
                let mut j = i + 1;
                let mut digits = String::new();
                while let Some(d) = s.get(j).copied().and_then(from_superscript) {
                    digits.push(d);
                    j += 1;
                }
                out.push_str(&format!("exp({digits})"));
                i = j;
            }
            _ if c.is_ascii_alphabetic() => {
                let start = i;
                let mut j = i;
                while j < s.len() && (s[j].is_ascii_alphanumeric() || s[j] == '_') {
                    j += 1;
                }
                let ident: String = s[start..j].iter().collect();
                out.push_str(map_unicode_ident(&ident));
                i = j;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    Ok(out)
}

/// The Unicode printer's spelling of a function back to the node name the
/// parser knows (`arctan` is printed, `atan` is the primitive).
fn map_unicode_ident(ident: &str) -> &str {
    match ident {
        "arcsin" => "asin",
        "arccos" => "acos",
        "arctan" => "atan",
        "ln" => "log",
        other => other,
    }
}

/// The operand a radical applies to: a parenthesised group, an absolute value,
/// a vulgar fraction, or the atom that follows it.
fn read_unicode_atom(s: &[char], i: &mut usize) -> Result<String, String> {
    if let Some(f) = s.get(*i).copied().and_then(vulgar_fraction) {
        *i += 1;
        return Ok(f.to_owned());
    }
    if s.get(*i) == Some(&'|') {
        let (inner, next) = read_bar_group(s, *i)?;
        *i = next;
        return Ok(format!("abs({})", read_unicode(&inner)?));
    }
    if s.get(*i) == Some(&'(') {
        let close = matching_paren(s, *i)?;
        let inner: String = s[*i + 1..close].iter().collect();
        *i = close + 1;
        return read_unicode(&inner);
    }
    let start = *i;
    let mut j = *i;
    while j < s.len() && (s[j].is_ascii_alphanumeric() || s[j] == '.' || s[j] == '_') {
        j += 1;
    }
    if j == start {
        return Err(format!("nothing for the radical to apply to at {start}"));
    }
    // An identifier followed by `(` is a call, and the call is the operand.
    if s.get(j) == Some(&'(') {
        let close = matching_paren(s, j)?;
        let name: String = s[start..j].iter().collect();
        let args: String = s[j + 1..close].iter().collect();
        *i = close + 1;
        return Ok(format!(
            "{}({})",
            map_unicode_ident(&name),
            read_unicode(&args)?
        ));
    }
    *i = j;
    Ok(s[start..j].iter().collect())
}

/// The body of a `|…|` starting at `open`, and the index just past it.
///
/// `|` is its own mirror, so pairing by "scan to the next bar" is wrong the
/// moment one absolute value contains another.  The printer parenthesises the
/// inner one (`|(|2|)|`) precisely so that this stays decidable, and that is
/// the case handled first.
fn read_bar_group(s: &[char], open: usize) -> Result<(String, usize), String> {
    if s.get(open + 1) == Some(&'(') {
        let close = matching_paren(s, open + 1)?;
        // Only when the group *is* the whole body — `|(x)·y|` is an ordinary
        // absolute value that happens to start with a parenthesis.
        if s.get(close + 1) == Some(&'|') {
            return Ok((s[open + 2..close].iter().collect(), close + 2));
        }
    }
    let close = s[open + 1..]
        .iter()
        .position(|&ch| ch == '|')
        .ok_or("unpaired |")?
        + open
        + 1;
    let inner: String = s[open + 1..close].iter().collect();
    if inner.contains('|') {
        return Err(format!("ambiguous nested bars in {inner:?}"));
    }
    Ok((inner, close + 1))
}

fn matching_paren(s: &[char], open: usize) -> Result<usize, String> {
    let mut depth = 0usize;
    for (j, &ch) in s.iter().enumerate().skip(open) {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(j);
                }
            }
            _ => {}
        }
    }
    Err("unbalanced parentheses".into())
}

struct Harness {
    pool: ExprPool,
    syms: Vec<ExprId>,
    names: HashMap<String, ExprId>,
}

impl Harness {
    fn new() -> Self {
        let pool = ExprPool::new();
        let syms: Vec<ExprId> = SYMBOL_NAMES
            .iter()
            .map(|n| pool.symbol(*n, Domain::Real))
            .collect();
        let names = SYMBOL_NAMES
            .iter()
            .map(|n| ((*n).to_owned(), pool.symbol(*n, Domain::Real)))
            .collect();
        Self { pool, syms, names }
    }

    /// `(printed, verdict)` for one expression.
    fn check(&self, id: ExprId) -> (String, Verdict) {
        let printed = self.pool.display(id).to_string();
        let mut names = self.names.clone();
        match parse(&printed, &self.pool, &mut names) {
            // A rendering the parser rejects is a *loud* failure, not a silent
            // one, but it still means the output is not usable as input.
            Err(e) => panic!("printed form does not parse: {printed:?} — {e:?}"),
            Ok(back) => (printed, compare(id, back, &self.pool, &self.syms)),
        }
    }

    /// The same property for a typeset form, read back through `reader`.
    fn check_typeset(
        &self,
        id: ExprId,
        render: fn(ExprId, &ExprPool) -> String,
        reader: fn(&str) -> Result<String, String>,
    ) -> (String, Verdict) {
        let printed = render(id, &self.pool);
        let ascii = match reader(&printed) {
            Ok(a) => a,
            Err(e) => panic!("cannot read back {printed:?}: {e}"),
        };
        let mut names = self.names.clone();
        if std::env::var("PRINTER_DEBUG").is_ok() {
            eprintln!("DBG {printed}  ==>  {ascii}");
        }
        match parse(&ascii, &self.pool, &mut names) {
            Err(e) => panic!("{printed:?} reads as {ascii:?}, which does not parse: {e:?}"),
            Ok(back) => (
                format!("{printed}   ->   {ascii}"),
                compare(id, back, &self.pool, &self.syms),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// The property
// ---------------------------------------------------------------------------

#[test]
fn generated_corpus_round_trips_through_the_parser() {
    let h = Harness::new();
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    let mut counts = [0usize; 4];
    let mut failures: Vec<String> = Vec::new();
    let mut undecided: Vec<String> = Vec::new();

    for _ in 0..4000 {
        let depth = 1 + rng.below(4) as u32;
        let id = build(&mut rng, &h.pool, &h.syms, depth);
        let (printed, verdict) = h.check(id);
        match verdict {
            Verdict::Structural => counts[0] += 1,
            Verdict::Numeric => counts[1] += 1,
            Verdict::Undecided => {
                counts[2] += 1;
                if undecided.len() < 12 {
                    undecided.push(printed.clone());
                }
            }
            Verdict::Mismatch => {
                counts[3] += 1;
                if failures.len() < 10 {
                    failures.push(printed);
                }
            }
        }
    }

    println!(
        "round-trip corpus: {} structural, {} numeric, {} undecided, {} mismatched",
        counts[0], counts[1], counts[2], counts[3]
    );
    assert!(
        failures.is_empty(),
        "printed forms that re-read as a different expression \
         (structural {}, numeric {}, undecided {}, mismatched {}):\n{}",
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        failures.join("\n")
    );
    // Guard against the corpus quietly becoming vacuous.
    let decided = counts[0] + counts[1];
    assert!(
        decided > 3000,
        "only {decided} of 4000 cases were decided; the corpus has gone vacuous\n{}",
        undecided.join("\n")
    );
}

/// The shapes where a printer classically loses the expression: non-associative
/// operators nested on the right, stacked powers, unary minus, negative and
/// fractional exponents, juxtaposed numeric factors, repeated factors.
fn precedence_shapes(h: &Harness) -> Vec<ExprId> {
    let p = &h.pool;
    let (a, b, c) = (h.syms[0], h.syms[1], h.syms[2]);
    let neg = |x: ExprId| p.mul(vec![p.integer(-1_i64), x]);
    let sub = |x: ExprId, y: ExprId| p.add(vec![x, neg(y)]);
    let div = |x: ExprId, y: ExprId| p.mul(vec![x, p.pow(y, p.integer(-1_i64))]);

    vec![
        // a - (b - c) ≠ (a - b) - c
        sub(a, sub(b, c)),
        sub(sub(a, b), c),
        // a / (b / c) ≠ (a / b) / c
        div(a, div(b, c)),
        div(div(a, b), c),
        // a^(b^c) ≠ (a^b)^c
        p.pow(a, p.pow(b, c)),
        p.pow(p.pow(a, b), c),
        // unary minus against `^` and `*`
        neg(p.pow(a, p.integer(2_i64))),
        p.pow(neg(a), p.integer(2_i64)),
        neg(p.mul(vec![a, b])),
        p.mul(vec![neg(a), b]),
        // negative literal bases: `-1^n` re-reads as `-(1^n)`
        p.pow(p.integer(-1_i64), a),
        p.pow(p.rational(-1, 2), a),
        p.pow(p.float(-1.5, 53), a),
        p.mul(vec![p.integer(-16_i64), p.pow(p.integer(-2_i64), a)]),
        // negative and fractional exponents
        p.pow(a, p.integer(-2_i64)),
        p.pow(a, p.rational(-1, 2)),
        p.pow(a, p.rational(1, 3)),
        p.pow(p.rational(3, 7), a),
        // a numeric factor next to a factor that begins with a digit
        p.mul(vec![p.integer(2_i64), p.pow(p.integer(3_i64), a)]),
        p.mul(vec![p.float(1.5, 53), p.float(2.5, 53)]),
        p.mul(vec![a, p.float(-1.5, 53)]),
        // repeated factors — the power that must not be lost
        p.mul(vec![a, a]),
        p.mul(vec![a, a, a, b]),
        p.mul(vec![p.func("sin", vec![a]), p.func("sin", vec![a])]),
        div(a, p.mul(vec![b, c])),
        div(
            p.func("sin", vec![a]),
            p.add(vec![p.integer(1_i64), p.mul(vec![a, a])]),
        ),
        // multi-argument application
        p.func("atan2", vec![a, b]),
        p.func("atan2", vec![sub(a, b), div(b, c)]),
    ]
}

#[test]
fn associativity_and_precedence_shapes_round_trip() {
    let h = Harness::new();
    let mut bad: Vec<String> = Vec::new();
    for id in precedence_shapes(&h) {
        let (printed, verdict) = h.check(id);
        if verdict == Verdict::Mismatch || verdict == Verdict::Undecided {
            bad.push(format!("{verdict:?}: {printed}"));
        }
    }
    assert!(bad.is_empty(), "round-trip failures:\n{}", bad.join("\n"));
}

// ---------------------------------------------------------------------------
// The same property for the typeset forms
// ---------------------------------------------------------------------------

fn typeset_report(
    h: &Harness,
    ids: impl IntoIterator<Item = ExprId>,
    render: fn(ExprId, &ExprPool) -> String,
    reader: fn(&str) -> Result<String, String>,
) -> ([usize; 4], Vec<String>) {
    let mut counts = [0usize; 4];
    let mut failures = Vec::new();
    for id in ids {
        let (printed, verdict) = h.check_typeset(id, render, reader);
        match verdict {
            Verdict::Structural => counts[0] += 1,
            Verdict::Numeric => counts[1] += 1,
            Verdict::Undecided => counts[2] += 1,
            Verdict::Mismatch => {
                counts[3] += 1;
                if failures.len() < 10 {
                    failures.push(printed);
                }
            }
        }
    }
    (counts, failures)
}

fn generated_ids(n: usize, h: &Harness) -> Vec<ExprId> {
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    (0..n)
        .map(|_| {
            let depth = 1 + rng.below(4) as u32;
            build(&mut rng, &h.pool, &h.syms, depth)
        })
        .collect()
}

/// LaTeX, read back the way the page reads.
///
/// This is the check that `x x` cannot pass: math mode discards the space, so
/// the emitted product of two `x`s has to carry a visible exponent or an
/// explicit `\cdot`, and a numeric factor followed by a digit has to carry the
/// `\cdot` or it fuses into one number.
#[test]
fn latex_round_trips_as_it_is_typeset() {
    let h = Harness::new();
    let mut ids = precedence_shapes(&h);
    ids.extend(generated_ids(2000, &h));
    let (counts, failures) = typeset_report(&h, ids, crate::kernel::render_latex, read_latex);
    println!(
        "latex round-trip: {} structural, {} numeric, {} undecided, {} mismatched",
        counts[0], counts[1], counts[2], counts[3]
    );
    assert!(
        failures.is_empty(),
        "LaTeX that re-reads as a different expression:\n{}",
        failures.join("\n")
    );
    assert!(
        counts[0] + counts[1] > counts.iter().sum::<usize>() / 2,
        "the LaTeX corpus went vacuous"
    );
}

#[test]
fn unicode_round_trips_as_it_is_printed() {
    let h = Harness::new();
    let mut ids = precedence_shapes(&h);
    ids.extend(generated_ids(2000, &h));
    let (counts, failures) = typeset_report(&h, ids, crate::kernel::render_unicode, read_unicode);
    println!(
        "unicode round-trip: {} structural, {} numeric, {} undecided, {} mismatched",
        counts[0], counts[1], counts[2], counts[3]
    );
    assert!(
        failures.is_empty(),
        "Unicode that re-reads as a different expression:\n{}",
        failures.join("\n")
    );
    assert!(
        counts[0] + counts[1] > counts.iter().sum::<usize>() / 2,
        "the Unicode corpus went vacuous"
    );
}

/// Structural validity of the emitted LaTeX, independently of what it means:
/// unbalanced braces or an unmatched `\left` stop a document compiling even
/// when the mathematics is right.
#[test]
fn emitted_latex_is_structurally_valid() {
    let h = Harness::new();
    let mut ids = precedence_shapes(&h);
    ids.extend(generated_ids(2000, &h));
    let mut bad: Vec<String> = Vec::new();
    for id in ids {
        let tex = crate::kernel::render_latex(id, &h.pool);
        let mut braces = 0i32;
        let mut ok = true;
        let chars: Vec<char> = tex.chars().collect();
        for (i, &c) in chars.iter().enumerate() {
            let escaped = i > 0 && chars[i - 1] == '\\';
            match c {
                '{' if !escaped => braces += 1,
                '}' if !escaped => braces -= 1,
                _ => {}
            }
            if braces < 0 {
                ok = false;
            }
        }
        let lefts = tex.matches(r"\left").count();
        let rights = tex.matches(r"\right").count();
        // A bare `_` in math mode is a subscript; `\operatorname{a_b}` renders
        // wrong and two of them are the hard "Double subscript" error.
        let bare_underscore = tex
            .char_indices()
            .any(|(i, c)| c == '_' && !tex[..i].ends_with('\\') && !tex[..i].ends_with('}'));
        if !ok || braces != 0 || lefts != rights || bare_underscore || double_superscript(&tex) {
            bad.push(tex);
            if bad.len() > 5 {
                break;
            }
        }
    }
    assert!(bad.is_empty(), "invalid LaTeX emitted:\n{}", bad.join("\n"));
}

/// `x^a^b` — ambiguous to a reader and the TeX error "Double superscript".
fn double_superscript(tex: &str) -> bool {
    let s: Vec<char> = tex.chars().collect();
    let mut i = 0usize;
    while i < s.len() {
        if s[i] != '^' {
            i += 1;
            continue;
        }
        let mut k = i + 1;
        if s.get(k) == Some(&'{') {
            if take_group(&s, &mut k).is_err() {
                return false;
            }
        } else {
            k += 1;
        }
        while s.get(k) == Some(&' ') {
            k += 1;
        }
        if s.get(k) == Some(&'^') {
            return true;
        }
        i = k;
    }
    false
}

#[test]
fn a_double_superscript_is_detected() {
    assert!(double_superscript("e^{x}^2"));
    assert!(double_superscript("x^a^b"));
    assert!(!double_superscript("x^{a^b}"));
    assert!(!double_superscript(r"\left(e^{x}\right)^2"));
}

/// The readers themselves have to be able to fail, or the three tests above
/// pass by construction.
#[test]
fn the_typeset_readers_are_not_vacuous() {
    // A digit next to a digit fuses, exactly as it does on the page.
    assert_eq!(read_latex("2 3").unwrap(), "23");
    assert_eq!(read_latex("x 2").unwrap(), "x2");
    // Two variables read as a product; a variable next to a number does not.
    assert_eq!(read_latex("x x").unwrap(), "x*x");
    assert_eq!(read_latex(r"2 \cdot 3^n").unwrap(), "2*3^(n)");
    assert_eq!(read_latex("2 x").unwrap(), "2*x");
    assert_eq!(read_latex(r"\frac{1}{2} x").unwrap(), "((1)/(2))*x");
    assert_eq!(read_latex(r"x^{2}").unwrap(), "x^(2)");
    // `^` binds one token: `x^c y` is a product, not `x^(cy)`.
    assert_eq!(read_latex(r"x^c y").unwrap(), "x^(c)*y");
    assert_eq!(read_latex(r"2.5 \times 10^{-1}").unwrap(), "2.5*10^(-1)");
    assert_eq!(read_latex(r"\sin\!\left(x\right)").unwrap(), "sin(x)");
    assert_eq!(read_unicode("∛|b|").unwrap(), "(abs(b))^(1/3)");
    // The Unicode reader undoes superscripts and radicals.
    assert_eq!(read_unicode("x²").unwrap(), "x^(2)");
    assert_eq!(read_unicode("x·y").unwrap(), "x*y");
    assert_eq!(read_unicode("(x + 1)¹").unwrap(), "(x + 1)^(1)");
    assert_eq!(read_unicode("√x").unwrap(), "(x)^(1/2)");
    assert_eq!(read_unicode("½·x").unwrap(), "(1/2)*x");
}
