//! SMT-LIB 2 emission for [`Formula`] — the *export* half of the SMT/SAT bridge (P2-3).
//!
//! The unit of exchange with an external solver is SMT-LIB 2 **text**, exactly as
//! the unit of exchange with Lean is `.lean` text: alkahest emits a standard
//! artifact and an independently maintained checker consumes it.  No solver is
//! vendored into this crate and none ever should be — see
//! `docs/mdbook/src/smt.md`.
//!
//! The emitter lives here, next to [`Formula`], because it must be *exhaustive*
//! over [`Formula`] and [`PredicateKind`] and `rustc`'s match-exhaustiveness
//! check is the enforcement mechanism.  **There is deliberately no `_ =>` arm
//! anywhere in this file**: a node added to `ExprData`, `Formula`, or
//! `PredicateKind` later must fail to compile here rather than silently emit
//! wrong SMT-LIB.  `tests/test_smt.py` pins that property from the outside too.
//!
//! # What is emitted
//!
//! A complete, runnable script:
//!
//! ```smt2
//! ; alkahest SMT-LIB 2 export
//! (set-logic QF_NRA)
//! (set-option :produce-models true)
//! (declare-fun x () Real)
//! (assert (> (* x x) 2))
//! (check-sat)
//! (get-model)
//! ```
//!
//! # What is refused
//!
//! Everything whose SMT-LIB rendering would not mean *exactly* the same thing as
//! the alkahest expression: float literals (write `pool.rational(1, 10)`, not
//! `0.1`), complex-domain symbols, non-integer exponents, unregistered function
//! heads, `BigO`, `RootSum`, and symbol names that cannot be quoted.  Every
//! refusal is [`SmtLibError`] with the stable code `E-SMT-002`.

use super::{formula_from_expr, Formula, LogicError};
use crate::kernel::expr::PredicateKind;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Largest integer exponent expanded into repeated multiplication.
///
/// SMT-LIB 2 has no standard `^`, so `x**k` is emitted as `(* x x … x)`.  That
/// is portable across z3/cvc5/yices, but it is also an enumeration, so it is
/// bounded: a loop must never be able to ask for a megabyte of `x`.
pub const MAX_POW_EXPANSION: i64 = 128;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A formula (or a sub-term of one) is outside the SMT-LIB fragment alkahest
/// will export.
///
/// Every variant carries the stable code `E-SMT-002`.  The other `E-SMT-*`
/// codes (`001` no solver, `003` inexact model value, `004` model failed
/// back-substitution) belong to the Python driver in `alkahest/smt.py` and are
/// deliberately *not* registered here — nothing in Rust raises them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtLibError {
    /// An expression node has no exact SMT-LIB rendering.
    UnsupportedNode(String),
    /// A symbol's [`Domain`] has no SMT-LIB sort (currently only `Complex`).
    UnsupportedDomain(String),
    /// A function head is not in the exactly-translatable allow-list.
    UnsupportedFunction(String),
    /// A power is not an integer, or expanding it would exceed [`MAX_POW_EXPANSION`].
    UnsupportedExponent(String),
    /// The requested logic name is unknown, or too weak for this formula.
    UnsupportedLogic(String),
    /// One name denotes two different sorts, or is both bound and free.
    SymbolConflict(String),
    /// The expression is not a predicate/quantified formula at all.
    NotAFormula(String),
}

impl fmt::Display for SmtLibError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtLibError::UnsupportedNode(s) => write!(f, "to_smtlib: unsupported expression: {s}"),
            SmtLibError::UnsupportedDomain(s) => write!(f, "to_smtlib: unsupported domain: {s}"),
            SmtLibError::UnsupportedFunction(s) => {
                write!(f, "to_smtlib: unsupported function: {s}")
            }
            SmtLibError::UnsupportedExponent(s) => write!(f, "to_smtlib: unsupported power: {s}"),
            SmtLibError::UnsupportedLogic(s) => write!(f, "to_smtlib: unusable logic: {s}"),
            SmtLibError::SymbolConflict(s) => write!(f, "to_smtlib: symbol conflict: {s}"),
            SmtLibError::NotAFormula(s) => write!(f, "to_smtlib: not a formula: {s}"),
        }
    }
}

impl std::error::Error for SmtLibError {}

impl crate::errors::AlkahestError for SmtLibError {
    fn code(&self) -> &'static str {
        "E-SMT-002"
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            SmtLibError::UnsupportedNode(_) => {
                "the SMT-LIB fragment is polynomial (in)equalities over Int/Real symbols, \
                 boolean structure, quantifiers and Piecewise; rewrite the expression into \
                 that fragment, and use pool.rational(p, q) rather than a float literal"
            }
            SmtLibError::UnsupportedDomain(_) => {
                "declare the symbol over Real, Integer, Positive, NonNegative or NonZero; \
                 complex-domain symbols have no SMT-LIB arithmetic sort"
            }
            SmtLibError::UnsupportedFunction(_) => {
                "only `abs` has an exact SMT-LIB rendering today; substitute or approximate \
                 transcendental heads before exporting, or keep the query in-tree \
                 (prove_nonneg / decide)"
            }
            SmtLibError::UnsupportedExponent(_) => {
                "use a non-negative integer exponent no larger than \
                 alkahest_core::logic::smtlib::MAX_POW_EXPANSION; SMT-LIB 2 has no portable \
                 `^`, so powers are expanded into products and the expansion is bounded"
            }
            SmtLibError::UnsupportedLogic(_) => {
                "pass logic=\"auto\" (the default) and let the emitter pick, or name a logic \
                 at least as strong as the formula needs"
            }
            SmtLibError::SymbolConflict(_) => {
                "rename one of the symbols; SMT-LIB has one namespace, so two alkahest \
                 symbols that share a name but not a domain cannot both be declared"
            }
            SmtLibError::NotAFormula(_) => {
                "pass a predicate or quantified Expr; build one with pool.gt/le/… or \
                 alkahest.And/Or/Not"
            }
        })
    }
}

impl From<LogicError> for SmtLibError {
    fn from(e: LogicError) -> Self {
        SmtLibError::NotAFormula(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Sorts, requirements, logics
// ---------------------------------------------------------------------------

/// The two SMT-LIB arithmetic sorts alkahest emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Sort {
    Int,
    Real,
}

impl Sort {
    /// The SMT-LIB sort name.
    pub fn name(self) -> &'static str {
        match self {
            Sort::Int => "Int",
            Sort::Real => "Real",
        }
    }
}

/// What a formula needs from a solver — the input to logic selection, and the
/// thing `alkahest.smt.supported` plans against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Requirements {
    /// At least one `Int`-sorted symbol occurs.
    pub ints: bool,
    /// At least one `Real`-sorted symbol, rational literal, or division occurs.
    pub reals: bool,
    /// At least one product of two non-constant terms (or a variable divisor).
    pub nonlinear: bool,
    /// At least one `Forall`/`Exists` binder occurs.
    pub quantifiers: bool,
}

/// The arithmetic fragment of an SMT-LIB logic name.
///
/// # `LIRA` / `NIRA` are solver-facing names, not catalog names
///
/// The official SMT-LIB logic catalog (2.7) has no `QF_LIRA`, `QF_NIRA`,
/// `LIRA`, or `NIRA`: for mixed `Int`/`Real` it stops at `AUFLIRA` /
/// `AUFNIRA`.  Alkahest emits the `LIRA`/`NIRA` family anyway, deliberately:
///
/// * they are what the solvers this bridge drives actually use for the mixed
///   fragment.  z3 accepts them silently, where an unknown name draws
///   `ignoring unsupported logic`; Yices documents `QF_LIRA` / `QF_NIRA` among
///   the names it recognises beyond the official set; SMT-COMP runs `QF_LIRA`
///   and `QF_NIRA` divisions over SMT-LIB benchmarks.
/// * the catalog alternatives are strictly worse for what alkahest emits.
///   `AUFLIRA`/`AUFNIRA` are *quantified* logics that additionally carry arrays
///   and free function symbols, so naming one for a quantifier-free mixed
///   formula throws away the `QF_` hint that decides which solver core runs and
///   claims a much larger fragment than is being used.
///
/// Mixed `Int`/`Real` is the capability this bridge exists to add, so the
/// contract is stated rather than hedged: **the inferred logic name for a mixed
/// formula is solver-facing, not catalog-standard.**  A consumer that will only
/// accept catalog names can ask for `AUFLIRA` / `AUFNIRA` explicitly (both are
/// in [`SUPPORTED_LOGICS`] and are sound supersets of everything emitted here),
/// or for `ALL`.  `tests/test_smt.py` pins that the installed solver accepts
/// the names actually emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arith {
    Lia,
    Nia,
    Lra,
    Nra,
    Lira,
    Nira,
}

impl Arith {
    fn quantified(self) -> &'static str {
        match self {
            Arith::Lia => "LIA",
            Arith::Nia => "NIA",
            Arith::Lra => "LRA",
            Arith::Nra => "NRA",
            Arith::Lira => "LIRA",
            Arith::Nira => "NIRA",
        }
    }

    fn quantifier_free(self) -> &'static str {
        match self {
            Arith::Lia => "QF_LIA",
            Arith::Nia => "QF_NIA",
            Arith::Lra => "QF_LRA",
            Arith::Nra => "QF_NRA",
            Arith::Lira => "QF_LIRA",
            Arith::Nira => "QF_NIRA",
        }
    }
}

impl Requirements {
    /// The weakest standard SMT-LIB logic that can express this formula.
    pub fn logic(&self) -> &'static str {
        let arith = match (self.ints, self.reals, self.nonlinear) {
            (true, true, false) => Arith::Lira,
            (true, true, true) => Arith::Nira,
            (true, false, false) => Arith::Lia,
            (true, false, true) => Arith::Nia,
            // No symbols at all still needs a well-formed logic; reals are the
            // most permissive choice and cost nothing.
            (false, false, false) | (false, true, false) => Arith::Lra,
            (false, false, true) | (false, true, true) => Arith::Nra,
        };
        if self.quantifiers {
            arith.quantified()
        } else {
            arith.quantifier_free()
        }
    }
}

/// Every logic name [`to_smtlib`] accepts explicitly, in the order a planner
/// should read them: quantifier-free first, then quantified, then the two
/// SMT-LIB catalog logics for mixed `Int`/`Real` (`AUFLIRA` / `AUFNIRA` — see
/// the note on the internal `Arith` type for why the non-catalog `QF_LIRA` and
/// `QF_NIRA` are still preferred for quantifier-free formulas), then the
/// catch-all.
pub const SUPPORTED_LOGICS: &[&str] = &[
    "QF_LIA", "QF_NIA", "QF_LRA", "QF_NRA", "QF_LIRA", "QF_NIRA", "LIA", "NIA", "LRA", "NRA",
    "LIRA", "NIRA", "AUFLIRA", "AUFNIRA", "ALL",
];

/// What a logic name permits, on the four axes [`Requirements`] tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LogicCaps {
    quantifiers: bool,
    nonlinear: bool,
    ints: bool,
    reals: bool,
}

const fn caps(quantifiers: bool, nonlinear: bool, ints: bool, reals: bool) -> LogicCaps {
    LogicCaps {
        quantifiers,
        nonlinear,
        ints,
        reals,
    }
}

/// What every name in [`SUPPORTED_LOGICS`] permits, in the same order.
///
/// A table rather than a letter heuristic on the name: `AUFLIRA` is a *linear*
/// logic that does not start with `L`, so `body.starts_with('L')` would quietly
/// wave a nonlinear formula through it.
const LOGIC_CAPS: &[LogicCaps] = &[
    //   quant  nonlin ints   reals
    caps(false, false, true, false), // QF_LIA
    caps(false, true, true, false),  // QF_NIA
    caps(false, false, false, true), // QF_LRA
    caps(false, true, false, true),  // QF_NRA
    caps(false, false, true, true),  // QF_LIRA
    caps(false, true, true, true),   // QF_NIRA
    caps(true, false, true, false),  // LIA
    caps(true, true, true, false),   // NIA
    caps(true, false, false, true),  // LRA
    caps(true, true, false, true),   // NRA
    caps(true, false, true, true),   // LIRA
    caps(true, true, true, true),    // NIRA
    // The catalog names for mixed Int/Real.  Both are quantified and also carry
    // arrays and free function symbols, which alkahest never emits — a superset
    // is sound, and these are the names a consumer that only knows the official
    // SMT-LIB catalog will accept.
    caps(true, false, true, true), // AUFLIRA
    caps(true, true, true, true),  // AUFNIRA
    caps(true, true, true, true),  // ALL
];

/// The fragment `name` denotes, or `None` if [`to_smtlib`] does not accept it.
fn logic_caps(name: &str) -> Option<LogicCaps> {
    let index = SUPPORTED_LOGICS.iter().position(|&n| n == name)?;
    LOGIC_CAPS.get(index).copied()
}

fn check_logic(requested: &str, req: &Requirements) -> Result<(), SmtLibError> {
    let Some(caps) = logic_caps(requested) else {
        return Err(SmtLibError::UnsupportedLogic(format!(
            "{requested:?} is not one of {SUPPORTED_LOGICS:?}"
        )));
    };
    if req.quantifiers && !caps.quantifiers {
        return Err(SmtLibError::UnsupportedLogic(format!(
            "formula has quantifiers but {requested:?} is quantifier-free"
        )));
    }
    if req.nonlinear && !caps.nonlinear {
        return Err(SmtLibError::UnsupportedLogic(format!(
            "formula is nonlinear but {requested:?} is a linear logic"
        )));
    }
    if req.ints && !caps.ints {
        return Err(SmtLibError::UnsupportedLogic(format!(
            "formula has Int-sorted symbols but {requested:?} has no Int sort"
        )));
    }
    if req.reals && !caps.reals {
        return Err(SmtLibError::UnsupportedLogic(format!(
            "formula needs Real arithmetic but {requested:?} has no Real sort"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Emission options for [`to_smtlib`] / [`formula_to_smtlib`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmtLibOptions<'a> {
    /// Logic name, or `None` to infer the weakest one that fits.
    pub logic: Option<&'a str>,
    /// Append `(check-sat)`.
    pub check_sat: bool,
    /// Set `:produce-models` and append `(get-model)`.
    pub get_model: bool,
}

impl Default for SmtLibOptions<'_> {
    fn default() -> Self {
        SmtLibOptions {
            logic: None,
            check_sat: true,
            get_model: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Symbol names
// ---------------------------------------------------------------------------

/// SMT-LIB 2 reserved words and core/theory symbols that must be `|quoted|`
/// when they occur as an alkahest symbol name.
const RESERVED: &[&str] = &[
    "!",
    "_",
    "as",
    "let",
    "exists",
    "forall",
    "match",
    "par",
    "assert",
    "check-sat",
    "declare-const",
    "declare-fun",
    "declare-sort",
    "define-fun",
    "define-sort",
    "exit",
    "get-model",
    "get-value",
    "push",
    "pop",
    "set-info",
    "set-logic",
    "set-option",
    "Bool",
    "Int",
    "Real",
    "true",
    "false",
    "not",
    "and",
    "or",
    "xor",
    "ite",
    "distinct",
    "abs",
    "div",
    "mod",
    "to_real",
    "to_int",
    "is_int",
    "divisible",
];

/// Characters SMT-LIB 2 allows in a *simple* symbol beyond `[a-zA-Z0-9]`.
const SIMPLE_EXTRA: &str = "~!@$%^&*_-+=<>.?/";

fn smt_symbol(name: &str) -> Result<String, SmtLibError> {
    if name.contains('|') || name.contains('\\') {
        return Err(SmtLibError::SymbolConflict(format!(
            "symbol name {name:?} contains '|' or '\\', which SMT-LIB 2 cannot quote"
        )));
    }
    let simple = !name.is_empty()
        && !name.starts_with(|c: char| c.is_ascii_digit())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || SIMPLE_EXTRA.contains(c))
        && !RESERVED.contains(&name);
    if simple {
        Ok(name.to_string())
    } else {
        Ok(format!("|{name}|"))
    }
}

fn sort_of_domain(name: &str, domain: Domain) -> Result<Sort, SmtLibError> {
    match domain {
        Domain::Integer => Ok(Sort::Int),
        Domain::Real | Domain::Positive | Domain::NonNegative | Domain::NonZero => Ok(Sort::Real),
        Domain::Complex => Err(SmtLibError::UnsupportedDomain(format!(
            "symbol {name:?} is Complex; SMT-LIB arithmetic is ordered and real"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Symbol collection (pre-pass)
// ---------------------------------------------------------------------------

/// Every symbol reachable from `id`, keyed by name.
///
/// Runs before emission so the emitter knows whether an `Int` sort is in play,
/// which decides whether `to_real` coercions are legal (they exist only in the
/// `Reals_Ints` theory; under plain `Reals`, numerals already *are* `Real`).
fn collect_symbols_expr(
    id: ExprId,
    pool: &ExprPool,
    out: &mut BTreeMap<String, (Sort, Domain)>,
) -> Result<(), SmtLibError> {
    match pool.get(id) {
        ExprData::Symbol { name, domain, .. } => {
            let sort = sort_of_domain(&name, domain)?;
            if let Some(&(prev_sort, prev_domain)) = out.get(&name) {
                if prev_sort != sort || prev_domain != domain {
                    return Err(SmtLibError::SymbolConflict(format!(
                        "symbol {name:?} occurs with domains {prev_domain} and {domain}"
                    )));
                }
            }
            out.insert(name, (sort, domain));
            Ok(())
        }
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => Ok(()),
        ExprData::Add(args) | ExprData::Mul(args) => {
            for a in args {
                collect_symbols_expr(a, pool, out)?;
            }
            Ok(())
        }
        ExprData::Pow { base, exp } => {
            collect_symbols_expr(base, pool, out)?;
            collect_symbols_expr(exp, pool, out)
        }
        ExprData::Func { name: _, args } => {
            for a in args {
                collect_symbols_expr(a, pool, out)?;
            }
            Ok(())
        }
        ExprData::Piecewise { branches, default } => {
            for (cond, value) in branches {
                collect_symbols_expr(cond, pool, out)?;
                collect_symbols_expr(value, pool, out)?;
            }
            collect_symbols_expr(default, pool, out)
        }
        ExprData::Predicate { kind: _, args } => {
            for a in args {
                collect_symbols_expr(a, pool, out)?;
            }
            Ok(())
        }
        ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
            collect_symbols_expr(var, pool, out)?;
            collect_symbols_expr(body, pool, out)
        }
        ExprData::BigO(inner) => collect_symbols_expr(inner, pool, out),
        ExprData::RootSum { poly, var, body } => {
            collect_symbols_expr(poly, pool, out)?;
            collect_symbols_expr(var, pool, out)?;
            collect_symbols_expr(body, pool, out)
        }
    }
}

fn collect_symbols_formula(
    f: &Formula,
    pool: &ExprPool,
    out: &mut BTreeMap<String, (Sort, Domain)>,
) -> Result<(), SmtLibError> {
    match f {
        Formula::True | Formula::False => Ok(()),
        Formula::Atom { kind: _, args } => {
            for &a in args {
                collect_symbols_expr(a, pool, out)?;
            }
            Ok(())
        }
        Formula::And(a, b) | Formula::Or(a, b) => {
            collect_symbols_formula(a, pool, out)?;
            collect_symbols_formula(b, pool, out)
        }
        Formula::Not(a) => collect_symbols_formula(a, pool, out),
        Formula::Forall { var, body } | Formula::Exists { var, body } => {
            collect_symbols_expr(*var, pool, out)?;
            collect_symbols_formula(body, pool, out)
        }
    }
}

/// Does `id` mention any symbol?  Drives the linear/nonlinear split.
fn has_symbol(id: ExprId, pool: &ExprPool) -> bool {
    let mut found = BTreeMap::new();
    // A collection failure (e.g. a Complex symbol) still means "has symbols";
    // the emitter reports the real error a moment later.
    match collect_symbols_expr(id, pool, &mut found) {
        Ok(()) => !found.is_empty(),
        Err(_) => true,
    }
}

// ---------------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------------

struct Emitter<'a> {
    pool: &'a ExprPool,
    /// Is the script in a `Reals_Ints` logic, where `to_real` exists and an
    /// `Int` numeral in a `Real` position needs it?  See [`formula_export`].
    use_to_real: bool,
    req: Requirements,
    bound: Vec<String>,
    bound_ever: BTreeSet<String>,
}

impl<'a> Emitter<'a> {
    fn coerce(&self, text: String, from: Sort, to: Sort) -> String {
        match (from, to) {
            // `to_real` lives in the `Reals_Ints` theory.  Under plain `Reals`
            // (QF_LRA / QF_NRA) numerals already *are* `Real` and emitting
            // `to_real` would make the script unparseable; under `Reals_Ints`
            // they are `Int` and omitting it would rely on mixed-sort sugar.
            (Sort::Int, Sort::Real) => {
                if self.use_to_real {
                    format!("(to_real {text})")
                } else {
                    text
                }
            }
            (Sort::Int, Sort::Int) | (Sort::Real, Sort::Real) | (Sort::Real, Sort::Int) => text,
        }
    }

    fn zero(&self, sort: Sort) -> String {
        self.coerce("0".to_string(), Sort::Int, sort)
    }

    fn one(&self, sort: Sort) -> String {
        self.coerce("1".to_string(), Sort::Int, sort)
    }

    // -- terms ------------------------------------------------------------

    /// Emit `id` as an SMT-LIB term, returning `(text, sort)`.
    fn term(&mut self, id: ExprId) -> Result<(String, Sort), SmtLibError> {
        match self.pool.get(id) {
            ExprData::Symbol { name, domain, .. } => {
                let sort = sort_of_domain(&name, domain)?;
                match sort {
                    Sort::Int => self.req.ints = true,
                    Sort::Real => self.req.reals = true,
                }
                Ok((smt_symbol(&name)?, sort))
            }
            ExprData::Integer(n) => Ok((fmt_int(&n.0), Sort::Int)),
            ExprData::Rational(r) => {
                self.req.reals = true;
                // `/` is `Real Real -> Real`; under `Reals_Ints` the numerals are
                // `Int` and need coercing, under plain `Reals` they already are
                // `Real` and coercing would not parse.
                let numer = self.coerce(fmt_int(r.0.numer()), Sort::Int, Sort::Real);
                let denom = self.coerce(fmt_int(r.0.denom()), Sort::Int, Sort::Real);
                Ok((format!("(/ {numer} {denom})"), Sort::Real))
            }
            ExprData::Float(f) => Err(SmtLibError::UnsupportedNode(format!(
                "float literal {}; a float is not an exact rational question, so exporting \
                 it would change what is being asked",
                f.inner
            ))),
            ExprData::Add(args) => self.nary("+", &args, "0", false),
            ExprData::Mul(args) => self.nary("*", &args, "1", true),
            ExprData::Pow { base, exp } => self.pow(base, exp),
            ExprData::Func { name, args } => self.func(&name, &args),
            ExprData::Piecewise { branches, default } => self.piecewise(&branches, default),
            ExprData::Predicate { .. } => Err(SmtLibError::UnsupportedNode(
                "a Bool-valued predicate used where an arithmetic term is required; SMT-LIB 2 \
                 does not coerce Bool to Int/Real"
                    .to_string(),
            )),
            ExprData::Forall { .. } | ExprData::Exists { .. } => Err(SmtLibError::UnsupportedNode(
                "a quantified formula used where an arithmetic term is required".to_string(),
            )),
            ExprData::BigO(_) => Err(SmtLibError::UnsupportedNode(
                "O(...) is an asymptotic order bound, not a value an SMT solver can reason about"
                    .to_string(),
            )),
            ExprData::RootSum { .. } => Err(SmtLibError::UnsupportedNode(
                "RootSum binds an algebraic-number placeholder; SMT-LIB 2 has no such term"
                    .to_string(),
            )),
        }
    }

    fn nary(
        &mut self,
        op: &str,
        args: &[ExprId],
        identity: &str,
        track_nonlinearity: bool,
    ) -> Result<(String, Sort), SmtLibError> {
        if args.is_empty() {
            // Empty sum is 0, empty product is 1 — matching the kernel.
            return Ok((identity.to_string(), Sort::Int));
        }
        if track_nonlinearity {
            let non_constant = args.iter().filter(|&&a| has_symbol(a, self.pool)).count();
            if non_constant >= 2 {
                self.req.nonlinear = true;
            }
        }
        let mut parts = Vec::with_capacity(args.len());
        let mut sort = Sort::Int;
        for &a in args {
            let (text, s) = self.term(a)?;
            if s == Sort::Real {
                sort = Sort::Real;
            }
            parts.push((text, s));
        }
        if parts.len() == 1 {
            let (text, s) = parts.pop().expect("length checked");
            return Ok((text, s));
        }
        let joined = parts
            .into_iter()
            .map(|(text, s)| self.coerce(text, s, sort))
            .collect::<Vec<_>>()
            .join(" ");
        Ok((format!("({op} {joined})"), sort))
    }

    fn pow(&mut self, base: ExprId, exp: ExprId) -> Result<(String, Sort), SmtLibError> {
        let k = match self.pool.get(exp) {
            ExprData::Integer(n) => n.0.to_i64().ok_or_else(|| {
                SmtLibError::UnsupportedExponent(format!("exponent {} does not fit in i64", n.0))
            })?,
            other => {
                return Err(SmtLibError::UnsupportedExponent(format!(
                    "exponent must be an integer literal, got {}",
                    describe(&other)
                )))
            }
        };
        if k.unsigned_abs() > MAX_POW_EXPANSION as u64 {
            return Err(SmtLibError::UnsupportedExponent(format!(
                "|{k}| exceeds MAX_POW_EXPANSION ({MAX_POW_EXPANSION}); SMT-LIB 2 has no \
                 portable `^`, so powers are expanded into products"
            )));
        }
        let base_has_symbol = has_symbol(base, self.pool);
        let (base_text, base_sort) = self.term(base)?;
        if k == 0 {
            // The kernel defines `0^0 = 1`, so the *value* is `1` whatever the
            // base is — but the base still has to be visited.  Every refusal
            // this emitter makes (float literals, transcendental heads, complex
            // domains) is raised by `term`, and `ExprPool::pow` interns
            // `sin(x)^0` verbatim, so returning early would emit a script for a
            // formula alkahest cannot translate.  Visiting also records the
            // base's `Requirements`, which keeps logic selection consistent
            // with the symbols `collect_symbols_expr` goes on to declare.
            return Ok(("1".to_string(), Sort::Int));
        }
        let n = k.unsigned_abs() as usize;
        if n >= 2 && base_has_symbol {
            self.req.nonlinear = true;
        }
        let repeated = if n == 1 {
            base_text
        } else {
            let joined = vec![base_text.as_str(); n].join(" ");
            format!("(* {joined})")
        };
        if k > 0 {
            return Ok((repeated, base_sort));
        }
        // Negative exponent: real division, and dividing by a variable is not
        // linear arithmetic even when the numerator is.
        self.req.reals = true;
        if base_has_symbol {
            self.req.nonlinear = true;
        }
        let numer = self.one(Sort::Real);
        let denom = self.coerce(repeated, base_sort, Sort::Real);
        Ok((format!("(/ {numer} {denom})"), Sort::Real))
    }

    fn func(&mut self, name: &str, args: &[ExprId]) -> Result<(String, Sort), SmtLibError> {
        // Exactly-translatable heads only.  `abs` is the whole list: anything
        // transcendental has no SMT-LIB 2 rendering that means the same thing,
        // and approximating it here would be the silent-error shape this
        // subsystem exists to prevent.
        if name == "abs" && args.len() == 1 {
            let (text, sort) = self.term(args[0])?;
            return Ok(match sort {
                // SMT-LIB `abs` is Int-only; the Real case is an exact `ite`.
                Sort::Int => (format!("(abs {text})"), Sort::Int),
                Sort::Real => {
                    let zero = self.zero(Sort::Real);
                    (
                        format!("(ite (>= {text} {zero}) {text} (- {text}))"),
                        Sort::Real,
                    )
                }
            });
        }
        Err(SmtLibError::UnsupportedFunction(format!(
            "{name}/{}",
            args.len()
        )))
    }

    fn piecewise(
        &mut self,
        branches: &[(ExprId, ExprId)],
        default: ExprId,
    ) -> Result<(String, Sort), SmtLibError> {
        let mut conds = Vec::with_capacity(branches.len());
        let mut values = Vec::with_capacity(branches.len());
        let mut sort = Sort::Int;
        for &(cond, value) in branches {
            let sub = formula_from_expr(cond, self.pool)?;
            conds.push(self.formula(&sub)?);
            let (text, s) = self.term(value)?;
            if s == Sort::Real {
                sort = Sort::Real;
            }
            values.push((text, s));
        }
        let (default_text, default_sort) = self.term(default)?;
        if default_sort == Sort::Real {
            sort = Sort::Real;
        }
        let mut out = self.coerce(default_text, default_sort, sort);
        for (cond, (text, s)) in conds.into_iter().zip(values).rev() {
            let value = self.coerce(text, s, sort);
            out = format!("(ite {cond} {value} {out})");
        }
        Ok((out, sort))
    }

    // -- formulas ---------------------------------------------------------

    /// Emit a comparison between two terms, coercing to a common sort.
    fn compare(&mut self, op: &str, args: &[ExprId]) -> Result<String, SmtLibError> {
        if args.len() != 2 {
            return Err(SmtLibError::NotAFormula(format!(
                "relation {op:?} needs exactly 2 operands, got {}",
                args.len()
            )));
        }
        let (lhs, lhs_sort) = self.term(args[0])?;
        let (rhs, rhs_sort) = self.term(args[1])?;
        let sort = if lhs_sort == Sort::Real || rhs_sort == Sort::Real {
            Sort::Real
        } else {
            Sort::Int
        };
        let lhs = self.coerce(lhs, lhs_sort, sort);
        let rhs = self.coerce(rhs, rhs_sort, sort);
        Ok(format!("({op} {lhs} {rhs})"))
    }

    /// Emit an `ExprId` that must itself be a formula.
    fn sub_formula(&mut self, id: ExprId) -> Result<String, SmtLibError> {
        let f = formula_from_expr(id, self.pool)?;
        self.formula(&f)
    }

    fn formula(&mut self, f: &Formula) -> Result<String, SmtLibError> {
        match f {
            Formula::True => Ok("true".to_string()),
            Formula::False => Ok("false".to_string()),
            Formula::And(a, b) => {
                let a = self.formula(a)?;
                let b = self.formula(b)?;
                Ok(format!("(and {a} {b})"))
            }
            Formula::Or(a, b) => {
                let a = self.formula(a)?;
                let b = self.formula(b)?;
                Ok(format!("(or {a} {b})"))
            }
            Formula::Not(a) => {
                let a = self.formula(a)?;
                Ok(format!("(not {a})"))
            }
            Formula::Forall { var, body } => self.quantifier("forall", *var, body),
            Formula::Exists { var, body } => self.quantifier("exists", *var, body),
            // `formula_from_expr` lifts And/Or/Not/True/False out of `Atom`, but
            // `Formula` is public and can be built by hand, so every
            // `PredicateKind` is handled here rather than assumed away.
            Formula::Atom { kind, args } => match kind {
                PredicateKind::Lt => self.compare("<", args),
                PredicateKind::Le => self.compare("<=", args),
                PredicateKind::Gt => self.compare(">", args),
                PredicateKind::Ge => self.compare(">=", args),
                PredicateKind::Eq => self.compare("=", args),
                PredicateKind::Ne => Ok(format!("(not {})", self.compare("=", args)?)),
                PredicateKind::And => {
                    if args.is_empty() {
                        return Ok("true".to_string());
                    }
                    let parts = args
                        .iter()
                        .map(|&a| self.sub_formula(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(if parts.len() == 1 {
                        parts.into_iter().next().expect("length checked")
                    } else {
                        format!("(and {})", parts.join(" "))
                    })
                }
                PredicateKind::Or => {
                    if args.is_empty() {
                        return Ok("false".to_string());
                    }
                    let parts = args
                        .iter()
                        .map(|&a| self.sub_formula(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(if parts.len() == 1 {
                        parts.into_iter().next().expect("length checked")
                    } else {
                        format!("(or {})", parts.join(" "))
                    })
                }
                PredicateKind::Not => {
                    if args.len() != 1 {
                        return Err(SmtLibError::NotAFormula(format!(
                            "Not needs exactly 1 operand, got {}",
                            args.len()
                        )));
                    }
                    Ok(format!("(not {})", self.sub_formula(args[0])?))
                }
                PredicateKind::True => Ok("true".to_string()),
                PredicateKind::False => Ok("false".to_string()),
            },
        }
    }

    fn quantifier(
        &mut self,
        binder: &str,
        var: ExprId,
        body: &Formula,
    ) -> Result<String, SmtLibError> {
        let (name, domain) = match self.pool.get(var) {
            ExprData::Symbol { name, domain, .. } => (name, domain),
            other => {
                return Err(SmtLibError::NotAFormula(format!(
                    "{binder} must bind a symbol, got {}",
                    describe(&other)
                )))
            }
        };
        let sort = sort_of_domain(&name, domain)?;
        match sort {
            Sort::Int => self.req.ints = true,
            Sort::Real => self.req.reals = true,
        }
        self.req.quantifiers = true;
        let smt_name = smt_symbol(&name)?;
        self.bound.push(name.clone());
        self.bound_ever.insert(name);
        let body_text = self.formula(body);
        self.bound.pop();
        let body_text = body_text?;
        // A refined domain travels with the binder, not with a declaration:
        // `∀ x:Positive . P` is `∀ x:Real . x > 0 ⇒ P`, and `∃` takes `∧`.
        // Getting this backwards is a soundness bug, so the two are written out.
        let scoped = match self.domain_guard(&smt_name, domain, sort) {
            None => body_text,
            Some(guard) => {
                if binder == "forall" {
                    format!("(=> {guard} {body_text})")
                } else {
                    format!("(and {guard} {body_text})")
                }
            }
        };
        Ok(format!(
            "({binder} (({smt_name} {})) {scoped})",
            sort.name()
        ))
    }

    /// The side condition implied by a refined [`Domain`].
    ///
    /// Dropping these would silently widen the question: `Positive` means
    /// `x > 0` and an SMT solver has no other way to know that.
    fn domain_guard(&self, smt_name: &str, domain: Domain, sort: Sort) -> Option<String> {
        let zero = self.zero(sort);
        match domain {
            Domain::Positive => Some(format!("(> {smt_name} {zero})")),
            Domain::NonNegative => Some(format!("(>= {smt_name} {zero})")),
            Domain::NonZero => Some(format!("(not (= {smt_name} {zero}))")),
            Domain::Real | Domain::Integer | Domain::Complex => None,
        }
    }
}

fn fmt_int(n: &rug::Integer) -> String {
    let s = n.to_string();
    match s.strip_prefix('-') {
        Some(rest) => format!("(- {rest})"),
        None => s,
    }
}

fn describe(data: &ExprData) -> &'static str {
    match data {
        ExprData::Symbol { .. } => "a symbol",
        ExprData::Integer(_) => "an integer",
        ExprData::Rational(_) => "a rational",
        ExprData::Float(_) => "a float",
        ExprData::Add(_) => "a sum",
        ExprData::Mul(_) => "a product",
        ExprData::Pow { .. } => "a power",
        ExprData::Func { .. } => "a function application",
        ExprData::Piecewise { .. } => "a piecewise expression",
        ExprData::Predicate { .. } => "a predicate",
        ExprData::Forall { .. } => "a universally quantified formula",
        ExprData::Exists { .. } => "an existentially quantified formula",
        ExprData::BigO(_) => "a big-O remainder",
        ExprData::RootSum { .. } => "a root sum",
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// A finished export: the script plus everything a planner wants to know about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmtLibExport {
    /// The complete SMT-LIB 2 script.
    pub text: String,
    /// The logic named in `(set-logic …)`.
    pub logic: String,
    /// What the formula actually needs, independent of the logic that was named.
    pub requirements: Requirements,
    /// Declared (free) symbols, in declaration order, as `(name, sort)`.
    pub symbols: Vec<(String, Sort)>,
}

/// Export a predicate/quantified `ExprId` as an SMT-LIB 2 script.
pub fn to_smtlib(
    expr: ExprId,
    pool: &ExprPool,
    opts: &SmtLibOptions<'_>,
) -> Result<String, SmtLibError> {
    Ok(export(expr, pool, opts)?.text)
}

/// [`to_smtlib`] plus the logic, requirements, and declared symbols.
pub fn export(
    expr: ExprId,
    pool: &ExprPool,
    opts: &SmtLibOptions<'_>,
) -> Result<SmtLibExport, SmtLibError> {
    let f = formula_from_expr(expr, pool)?;
    formula_export(&f, pool, opts)
}

/// Export an already-lifted [`Formula`].
pub fn formula_to_smtlib(
    f: &Formula,
    pool: &ExprPool,
    opts: &SmtLibOptions<'_>,
) -> Result<String, SmtLibError> {
    Ok(formula_export(f, pool, opts)?.text)
}

/// Export an already-lifted [`Formula`], with metadata.
pub fn formula_export(
    f: &Formula,
    pool: &ExprPool,
    opts: &SmtLibOptions<'_>,
) -> Result<SmtLibExport, SmtLibError> {
    let mut symbols: BTreeMap<String, (Sort, Domain)> = BTreeMap::new();
    collect_symbols_formula(f, pool, &mut symbols)?;
    // `to_real` exists only inside a `Reals_Ints` logic, and there are two ways
    // to be in one: an `Int`-sorted symbol forces it, or the caller named a
    // logic that carries both sorts.  Emitting `to_real` outside such a logic
    // would not parse; *omitting* it inside one would lean on the mixed-sort
    // sugar that only some logics define, so both cases turn it on.
    let use_to_real = symbols.values().any(|&(sort, _)| sort == Sort::Int)
        || opts
            .logic
            .and_then(logic_caps)
            .is_some_and(|c| c.ints && c.reals);

    let mut emitter = Emitter {
        pool,
        use_to_real,
        req: Requirements::default(),
        bound: Vec::new(),
        bound_ever: BTreeSet::new(),
    };
    let body = emitter.formula(f)?;

    // A name that is both quantifier-bound and free would be declared *and*
    // shadowed; SMT-LIB would accept it and mean something different from the
    // alkahest expression, so refuse instead.
    let free: Vec<(String, (Sort, Domain))> = symbols
        .iter()
        .filter(|(name, _)| !emitter.bound_ever.contains(*name))
        .map(|(name, spec)| (name.clone(), *spec))
        .collect();
    if let Some(name) = symbols
        .keys()
        .find(|n| emitter.bound_ever.contains(*n) && mentions_free(f, pool, n))
    {
        return Err(SmtLibError::SymbolConflict(format!(
            "symbol {name:?} occurs both quantifier-bound and free"
        )));
    }

    let requirements = emitter.req;
    let logic = match opts.logic {
        Some(name) => {
            check_logic(name, &requirements)?;
            name.to_string()
        }
        None => requirements.logic().to_string(),
    };

    let mut out = String::new();
    out.push_str("; alkahest SMT-LIB 2 export\n");
    out.push_str(&format!("(set-logic {logic})\n"));
    if opts.get_model {
        out.push_str("(set-option :produce-models true)\n");
    }
    for (name, (sort, _)) in &free {
        let smt_name = smt_symbol(name)?;
        out.push_str(&format!("(declare-fun {smt_name} () {})\n", sort.name()));
    }
    for (name, (sort, domain)) in &free {
        let smt_name = smt_symbol(name)?;
        if let Some(guard) = emitter.domain_guard(&smt_name, *domain, *sort) {
            out.push_str(&format!("(assert {guard})\n"));
        }
    }
    out.push_str(&format!("(assert {body})\n"));
    if opts.check_sat {
        out.push_str("(check-sat)\n");
    }
    if opts.get_model {
        out.push_str("(get-model)\n");
    }

    Ok(SmtLibExport {
        text: out,
        logic,
        requirements,
        symbols: free
            .into_iter()
            .map(|(name, (sort, _))| (name, sort))
            .collect(),
    })
}

/// Would [`to_smtlib`] succeed on this expression?  The plan-ahead predicate.
pub fn supported(expr: ExprId, pool: &ExprPool) -> bool {
    to_smtlib(expr, pool, &SmtLibOptions::default()).is_ok()
}

/// Does `name` occur outside every binder that binds it?
fn mentions_free(f: &Formula, pool: &ExprPool, name: &str) -> bool {
    match f {
        Formula::True | Formula::False => false,
        Formula::Atom { kind: _, args } => args.iter().any(|&a| expr_mentions(a, pool, name)),
        Formula::And(a, b) | Formula::Or(a, b) => {
            mentions_free(a, pool, name) || mentions_free(b, pool, name)
        }
        Formula::Not(a) => mentions_free(a, pool, name),
        Formula::Forall { var, body } | Formula::Exists { var, body } => {
            if expr_mentions(*var, pool, name) {
                false
            } else {
                mentions_free(body, pool, name)
            }
        }
    }
}

fn expr_mentions(id: ExprId, pool: &ExprPool, name: &str) -> bool {
    let mut found = BTreeMap::new();
    let _ = collect_symbols_expr(id, pool, &mut found);
    found.contains_key(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn opts() -> SmtLibOptions<'static> {
        SmtLibOptions::default()
    }

    #[test]
    fn linear_real_formula() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_and(vec![
            p.pred_gt(x, p.integer(0_i32)),
            p.pred_lt(x, p.integer(3_i32)),
        ]);
        let e = export(f, &p, &opts()).unwrap();
        assert_eq!(e.logic, "QF_LRA");
        assert!(e.text.contains("(declare-fun x () Real)"), "{}", e.text);
        assert!(
            e.text.contains("(assert (and (> x 0) (< x 3)))"),
            "{}",
            e.text
        );
        assert!(e.text.ends_with("(check-sat)\n(get-model)\n"), "{}", e.text);
    }

    #[test]
    fn nonlinear_bumps_the_logic() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_gt(p.mul(vec![x, x]), p.integer(2_i32));
        let e = export(f, &p, &opts()).unwrap();
        assert_eq!(e.logic, "QF_NRA");
        assert!(e.requirements.nonlinear);
    }

    #[test]
    fn powers_expand_into_products() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_ge(p.pow(x, p.integer(3_i32)), p.integer(8_i32));
        let text = to_smtlib(f, &p, &opts()).unwrap();
        assert!(text.contains("(>= (* x x x) 8)"), "{text}");
    }

    #[test]
    fn zero_exponent_is_one_but_still_checks_the_base() {
        // `0^0 = 1` in the kernel, so the value is not in question — but the
        // base is still translated, and an untranslatable base is a refusal
        // rather than a silently emitted `1`.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);

        let ok = p.pred_eq(p.pow(x, p.integer(0_i32)), p.integer(1_i32));
        let text = to_smtlib(ok, &p, &opts()).unwrap();
        assert!(text.contains("(= 1 1)"), "{text}");

        let zero_zero = p.pred_eq(p.pow(p.integer(0_i32), p.integer(0_i32)), p.integer(1_i32));
        assert!(to_smtlib(zero_zero, &p, &opts()).is_ok());

        let sin_x = p.func("sin", vec![x]);
        let bad = p.pred_eq(p.pow(sin_x, p.integer(0_i32)), p.integer(1_i32));
        let err = to_smtlib(bad, &p, &opts()).unwrap_err();
        assert!(
            matches!(err, SmtLibError::UnsupportedFunction(_)),
            "{err:?}"
        );

        let float_base = p.pred_eq(p.pow(p.float(0.1, 53), p.integer(0_i32)), p.integer(1_i32));
        let err = to_smtlib(float_base, &p, &opts()).unwrap_err();
        assert!(matches!(err, SmtLibError::UnsupportedNode(_)), "{err:?}");
    }

    #[test]
    fn huge_exponent_is_refused_not_expanded() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_ge(p.pow(x, p.integer(100_000_i32)), p.integer(1_i32));
        let err = to_smtlib(f, &p, &opts()).unwrap_err();
        assert!(
            matches!(err, SmtLibError::UnsupportedExponent(_)),
            "{err:?}"
        );
    }

    #[test]
    fn mixed_int_real_uses_to_real() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let n = p.symbol("n", Domain::Integer);
        let f = p.pred_gt(x, n);
        let e = export(f, &p, &opts()).unwrap();
        assert_eq!(e.logic, "QF_LIRA");
        assert!(e.text.contains("(> x (to_real n))"), "{}", e.text);
        assert!(e.text.contains("(declare-fun n () Int)"), "{}", e.text);
    }

    #[test]
    fn pure_real_does_not_coerce_numerals() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_gt(x, p.integer(2_i32));
        let text = to_smtlib(f, &p, &opts()).unwrap();
        assert!(!text.contains("to_real"), "{text}");
    }

    #[test]
    fn refined_domains_emit_side_conditions() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Positive);
        let f = p.pred_lt(x, p.integer(1_i32));
        let text = to_smtlib(f, &p, &opts()).unwrap();
        assert!(text.contains("(assert (> x 0))"), "{text}");
    }

    #[test]
    fn rationals_are_exact() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_eq(x, p.rational(-1, 3));
        let text = to_smtlib(f, &p, &opts()).unwrap();
        assert!(text.contains("(/ (- 1) 3)"), "{text}");
    }

    #[test]
    fn rational_literals_coerce_under_reals_ints() {
        // `/` is Real→Real; under QF_LIRA the numerals are Int and must be lifted.
        let p = ExprPool::new();
        let n = p.symbol("n", Domain::Integer);
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_and(vec![
            p.pred_gt(x, p.rational(1, 4)),
            p.pred_gt(n, p.integer(0_i32)),
        ]);
        let e = export(f, &p, &opts()).unwrap();
        assert_eq!(e.logic, "QF_LIRA");
        assert!(e.text.contains("(/ (to_real 1) (to_real 4))"), "{}", e.text);
    }

    #[test]
    fn floats_are_refused() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_gt(x, p.float(0.1, 53));
        let err = to_smtlib(f, &p, &opts()).unwrap_err();
        assert!(matches!(err, SmtLibError::UnsupportedNode(_)), "{err:?}");
    }

    #[test]
    fn complex_symbols_are_refused() {
        let p = ExprPool::new();
        let z = p.symbol("z", Domain::Complex);
        let f = p.pred_gt(z, p.integer(0_i32));
        let err = to_smtlib(f, &p, &opts()).unwrap_err();
        assert!(matches!(err, SmtLibError::UnsupportedDomain(_)), "{err:?}");
    }

    #[test]
    fn transcendental_heads_are_refused() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_gt(p.func("sin", vec![x]), p.integer(0_i32));
        let err = to_smtlib(f, &p, &opts()).unwrap_err();
        assert!(
            matches!(err, SmtLibError::UnsupportedFunction(_)),
            "{err:?}"
        );
    }

    #[test]
    fn quantifiers_drop_the_qf_prefix() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let inner = p.pred_gt(p.add(vec![x, y]), p.integer(0_i32));
        let f = p.forall(x, inner);
        let e = export(f, &p, &opts()).unwrap();
        assert_eq!(e.logic, "LRA");
        assert!(e.text.contains("(forall ((x Real))"), "{}", e.text);
        assert!(e.text.contains("(declare-fun y () Real)"), "{}", e.text);
        assert!(!e.text.contains("(declare-fun x () Real)"), "{}", e.text);
    }

    #[test]
    fn bound_and_free_same_name_is_refused() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let inner = p.pred_gt(x, p.integer(0_i32));
        let f = p.pred_and(vec![p.forall(x, inner), inner]);
        let err = to_smtlib(f, &p, &opts()).unwrap_err();
        assert!(matches!(err, SmtLibError::SymbolConflict(_)), "{err:?}");
    }

    #[test]
    fn explicit_logic_must_be_strong_enough() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_gt(p.mul(vec![x, x]), p.integer(2_i32));
        let mut o = opts();
        o.logic = Some("QF_LRA");
        let err = formula_to_smtlib(&formula_from_expr(f, &p).unwrap(), &p, &o).unwrap_err();
        assert!(matches!(err, SmtLibError::UnsupportedLogic(_)), "{err:?}");
        o.logic = Some("QF_NRA");
        assert!(to_smtlib(f, &p, &o).is_ok());
    }

    #[test]
    fn every_supported_logic_has_a_capability_entry() {
        assert_eq!(SUPPORTED_LOGICS.len(), LOGIC_CAPS.len());
        for name in SUPPORTED_LOGICS {
            assert!(logic_caps(name).is_some(), "{name} has no LogicCaps entry");
        }
        assert_eq!(logic_caps("QF_BV"), None);
    }

    #[test]
    fn catalog_mixed_logics_are_accepted_and_still_checked() {
        // `AUFLIRA`/`AUFNIRA` are the *official* mixed Int/Real names, offered
        // for a consumer that only accepts the catalog.  `AUFLIRA` is linear
        // despite not starting with `L`, so a nonlinear formula must still be
        // refused under it.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let n = p.symbol("n", Domain::Integer);
        let linear = p.pred_gt(x, n);
        let mut o = opts();
        o.logic = Some("AUFLIRA");
        let text = to_smtlib(linear, &p, &o).unwrap();
        assert!(text.contains("(set-logic AUFLIRA)"), "{text}");
        assert!(text.contains("(> x (to_real n))"), "{text}");

        let nonlinear = p.pred_gt(p.mul(vec![x, n]), p.integer(0_i32));
        let err = to_smtlib(nonlinear, &p, &o).unwrap_err();
        assert!(matches!(err, SmtLibError::UnsupportedLogic(_)), "{err:?}");
        o.logic = Some("AUFNIRA");
        assert!(to_smtlib(nonlinear, &p, &o).is_ok());
    }

    #[test]
    fn a_reals_ints_logic_never_relies_on_mixed_sort_sugar() {
        // Pure-real formula, but the caller named a logic that carries both
        // sorts: the numeral is `Int` there and needs lifting.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_gt(x, p.integer(0_i32));
        let mut o = opts();
        o.logic = Some("ALL");
        let text = to_smtlib(f, &p, &o).unwrap();
        assert!(text.contains("(> x (to_real 0))"), "{text}");
        // ... and never emits it where the theory has no such function.
        o.logic = Some("QF_LRA");
        assert!(!to_smtlib(f, &p, &o).unwrap().contains("to_real"));
    }

    #[test]
    fn unknown_logic_name_is_refused() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_gt(x, p.integer(0_i32));
        let mut o = opts();
        o.logic = Some("QF_BV");
        assert!(matches!(
            to_smtlib(f, &p, &o).unwrap_err(),
            SmtLibError::UnsupportedLogic(_)
        ));
    }

    #[test]
    fn reserved_names_are_quoted() {
        let p = ExprPool::new();
        let x = p.symbol("ite", Domain::Real);
        let f = p.pred_gt(x, p.integer(0_i32));
        let text = to_smtlib(f, &p, &opts()).unwrap();
        assert!(text.contains("(declare-fun |ite| () Real)"), "{text}");
    }

    #[test]
    fn ne_becomes_not_eq() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_ne(x, p.integer(0_i32));
        let text = to_smtlib(f, &p, &opts()).unwrap();
        assert!(text.contains("(not (= x 0))"), "{text}");
    }

    #[test]
    fn every_predicate_kind_emits() {
        // If a `PredicateKind` is added, this fails to compile in `Emitter::formula`
        // first; this test keeps the *behaviour* pinned too.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let z = p.integer(0_i32);
        let leaf = p.pred_gt(x, z);
        let cases = [
            p.pred_lt(x, z),
            p.pred_le(x, z),
            p.pred_gt(x, z),
            p.pred_ge(x, z),
            p.pred_eq(x, z),
            p.pred_ne(x, z),
            p.pred_and(vec![leaf, leaf]),
            p.pred_or(vec![leaf, leaf]),
            p.pred_not(leaf),
            p.pred_true(),
            p.pred_false(),
        ];
        for (i, &case) in cases.iter().enumerate() {
            assert!(to_smtlib(case, &p, &opts()).is_ok(), "case {i} failed");
        }
    }

    #[test]
    fn every_formula_variant_emits() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let leaf = Formula::Atom {
            kind: PredicateKind::Gt,
            args: vec![x, p.integer(0_i32)],
        };
        let variants = [
            Formula::True,
            Formula::False,
            leaf.clone(),
            Formula::and(leaf.clone(), leaf.clone()),
            Formula::or(leaf.clone(), leaf.clone()),
            Formula::not(leaf.clone()),
            Formula::Forall {
                var: x,
                body: Box::new(leaf.clone()),
            },
            Formula::Exists {
                var: x,
                body: Box::new(leaf.clone()),
            },
        ];
        for (i, v) in variants.iter().enumerate() {
            assert!(
                formula_to_smtlib(v, &p, &opts()).is_ok(),
                "variant {i} failed"
            );
        }
    }

    #[test]
    fn hand_built_atom_with_boolean_kind_emits() {
        // `formula_from_expr` never produces these, but `Formula` is public.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let leaf = p.pred_gt(x, p.integer(0_i32));
        for kind in [PredicateKind::And, PredicateKind::Or] {
            let f = Formula::Atom {
                kind,
                args: vec![leaf, leaf],
            };
            let text = formula_to_smtlib(&f, &p, &opts()).unwrap();
            assert!(text.contains("(> x 0) (> x 0)"), "{text}");
        }
        let f = Formula::Atom {
            kind: PredicateKind::Not,
            args: vec![leaf],
        };
        assert!(formula_to_smtlib(&f, &p, &opts())
            .unwrap()
            .contains("(not (> x 0))"));
        for kind in [PredicateKind::True, PredicateKind::False] {
            let f = Formula::Atom { kind, args: vec![] };
            assert!(formula_to_smtlib(&f, &p, &opts()).is_ok());
        }
    }

    #[test]
    fn supported_matches_to_smtlib() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        assert!(supported(p.pred_gt(x, p.integer(0_i32)), &p));
        assert!(!supported(
            p.pred_gt(p.func("sin", vec![x]), p.integer(0_i32)),
            &p
        ));
        assert!(!supported(x, &p));
    }

    #[test]
    fn piecewise_becomes_ite() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let cond = p.pred_gt(x, p.integer(0_i32));
        let pw = p.piecewise(vec![(cond, p.integer(1_i32))], p.integer(-1_i32));
        let f = p.pred_eq(pw, p.integer(1_i32));
        let text = to_smtlib(f, &p, &opts()).unwrap();
        assert!(text.contains("(ite (> x 0) 1 (- 1))"), "{text}");
    }

    #[test]
    fn negative_power_is_real_division() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_gt(p.pow(x, p.integer(-1_i32)), p.integer(0_i32));
        let e = export(f, &p, &opts()).unwrap();
        assert!(e.text.contains("(/ 1 x)"), "{}", e.text);
        assert!(e.requirements.nonlinear);
    }
}
