//! V2-21 — Pratt recursive-descent expression parser (Rust port).
//!
//! Mirrors `python/alkahest/_parse.py` exactly: same grammar, same function
//! names, same precedence levels.  The Python layer can delegate to this once
//! the PyO3 binding is wired up.
//!
//! # Grammar (informal)
//!
//! ```text
//! expr     ::= term (('+' | '-') term)*
//! term     ::= factor (('*' | '/') factor)*
//! factor   ::= unary ('^' | '**') factor   -- right-assoc
//! unary    ::= '-' unary | primary
//! primary  ::= NUMBER | IDENT | IDENT '(' args ')' | '(' expr ')'
//! args     ::= expr (',' expr)*
//! ```
//!
//! Binding powers (Pratt):
//! - `+` / `-` infix: 10
//! - `*` / `/` infix: 20
//! - `^` / `**` infix: 30 (right-associative: right-bp = 29)
//! - unary `-` / `+`: 25
//!
//! # Example
//!
//! ```
//! use alkahest_cas::{ExprPool, parse};
//! use alkahest_cas::kernel::Domain;
//! use std::collections::HashMap;
//!
//! let pool = ExprPool::new();
//! let x = pool.symbol("x", Domain::Real);
//! let mut syms = HashMap::from([("x".to_owned(), x)]);
//! let e = parse("x^2 + 2*x + 1", &pool, &mut syms).unwrap();
//! ```

use std::collections::HashMap;

use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// A lexical or syntactic error produced by [`parse`].
///
/// Every `ParseError` carries a stable diagnostic code (`E-PARSE-NNN`) and an
/// optional byte-offset span into the source string.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Option<(usize, usize)>,
    code_idx: u8, // 1 = E-PARSE-001, 2 = E-PARSE-002, 3 = E-PARSE-003, 4 = E-PARSE-004
}

impl ParseError {
    fn lex(msg: impl Into<String>, span: (usize, usize)) -> Self {
        ParseError {
            message: msg.into(),
            span: Some(span),
            code_idx: 1,
        }
    }

    fn syntax(msg: impl Into<String>, span: (usize, usize)) -> Self {
        ParseError {
            message: msg.into(),
            span: Some(span),
            code_idx: 2,
        }
    }

    fn unknown_func(msg: impl Into<String>, span: (usize, usize)) -> Self {
        ParseError {
            message: msg.into(),
            span: Some(span),
            code_idx: 3,
        }
    }

    /// Input nested more deeply than the recursive-descent parser's stack
    /// budget allows — see [`MAX_PARSE_DEPTH`].
    fn too_deep(msg: impl Into<String>, span: (usize, usize)) -> Self {
        ParseError {
            message: msg.into(),
            span: Some(span),
            code_idx: 4,
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code(), self.message)?;
        if let Some((s, e)) = self.span {
            write!(f, " (bytes {s}–{e})")?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseError {}

impl AlkahestError for ParseError {
    fn code(&self) -> &'static str {
        match self.code_idx {
            1 => "E-PARSE-001",
            2 => "E-PARSE-002",
            4 => "E-PARSE-004",
            _ => "E-PARSE-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self.code_idx {
            1 => Some("only ASCII arithmetic expressions are supported"),
            2 => Some("check parentheses and operator placement"),
            4 => Some("flatten the expression — deeply nested parentheses, prefix signs or function calls exceed the parser's recursion budget"),
            _ => Some("use a known function: sin, cos, tan, sec, csc, cot, sinh, cosh, tanh, sech, csch, coth, asin, acos, atan, asinh, acosh, atanh, atan2, exp, log, sqrt, cbrt, abs, sign, floor, ceil, round, erf, erfc, gamma, lambert_w, digamma, trigamma, Ei, li, Si, Ci, Shi, Chi, fresnels, fresnelc, dilog"),
        }
    }

    fn span(&self) -> Option<(usize, usize)> {
        self.span
    }
}

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(String),   // integer or float literal
    Ident(String), // identifier / function name
    Plus,
    Minus,
    Star,
    Slash,
    Caret,    // ^
    StarStar, // **
    LParen,
    RParen,
    Comma,
    Eof,
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    offset: usize, // byte offset in source
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

fn tokenize(src: &str) -> Result<Vec<Token>, ParseError> {
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut pos = 0;
    let mut tokens = Vec::new();

    while pos < n {
        let b = bytes[pos];

        // Whitespace
        if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' {
            pos += 1;
            continue;
        }

        // Number: digits optionally followed by '.digits' and/or 'e[+-]digits'
        if b.is_ascii_digit() || (b == b'.' && pos + 1 < n && bytes[pos + 1].is_ascii_digit()) {
            let start = pos;
            while pos < n && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            if pos < n && bytes[pos] == b'.' {
                pos += 1;
                while pos < n && bytes[pos].is_ascii_digit() {
                    pos += 1;
                }
            }
            if pos < n && (bytes[pos] == b'e' || bytes[pos] == b'E') {
                // An exponent marker must be followed by at least one digit.
                // Consuming `e`/`E` and an optional sign unconditionally made
                // `"1e"` lex as `Num("1e")`, and `"1e".parse::<f64>()` is an
                // `Err` — `nud` unwrapped it and the parser *panicked* on a
                // two-character input.  Reject it here, with a span, so the
                // caller gets the same structured `E-PARSE-001` every other
                // piece of malformed text gets.
                //
                // This is deliberately an error and not a backtrack: `2e` does
                // not become `2 * e`.  Implicit multiplication is a separate
                // grammar decision, and silently reinterpreting a typo'd
                // exponent as a product with Euler's number is exactly the kind
                // of guess this parser should not make.
                let exp_marker = pos;
                let mut scan = pos + 1;
                if scan < n && (bytes[scan] == b'+' || bytes[scan] == b'-') {
                    scan += 1;
                }
                if scan < n && bytes[scan].is_ascii_digit() {
                    while scan < n && bytes[scan].is_ascii_digit() {
                        scan += 1;
                    }
                    pos = scan;
                } else {
                    return Err(ParseError::lex(
                        format!(
                            "malformed number literal {:?}: exponent marker {:?} needs at \
                             least one digit after it",
                            &src[start..scan],
                            &src[exp_marker..scan],
                        ),
                        (start, scan),
                    ));
                }
            }
            tokens.push(Token {
                tok: Tok::Num(src[start..pos].to_owned()),
                offset: start,
            });
            continue;
        }

        // Identifier
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = pos;
            while pos < n && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
                pos += 1;
            }
            tokens.push(Token {
                tok: Tok::Ident(src[start..pos].to_owned()),
                offset: start,
            });
            continue;
        }

        // `**` must come before `*`
        if b == b'*' && pos + 1 < n && bytes[pos + 1] == b'*' {
            tokens.push(Token {
                tok: Tok::StarStar,
                offset: pos,
            });
            pos += 2;
            continue;
        }

        let tok = match b {
            b'+' => Tok::Plus,
            b'-' => Tok::Minus,
            b'*' => Tok::Star,
            b'/' => Tok::Slash,
            b'^' => Tok::Caret,
            b'(' => Tok::LParen,
            b')' => Tok::RParen,
            b',' => Tok::Comma,
            _ => {
                return Err(ParseError::lex(
                    format!("unexpected character {:?}", b as char),
                    (pos, pos + 1),
                ))
            }
        };
        tokens.push(Token { tok, offset: pos });
        pos += 1;
    }

    tokens.push(Token {
        tok: Tok::Eof,
        offset: n,
    });
    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Binding powers
// ---------------------------------------------------------------------------

const BP_ADD: u8 = 10;
const BP_MUL: u8 = 20;
const BP_POW: u8 = 30;
const BP_UNARY: u8 = 25;

fn infix_bp(tok: &Tok) -> u8 {
    match tok {
        Tok::Plus | Tok::Minus => BP_ADD,
        Tok::Star | Tok::Slash => BP_MUL,
        Tok::Caret | Tok::StarStar => BP_POW,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Unary minus on a literal
// ---------------------------------------------------------------------------

/// If `id` is an exact numeric literal, return the interned literal for its
/// negation; otherwise `None`.
///
/// Prefix `-` is otherwise built as `(-1) · operand`, which for a literal
/// operand leaves an unevaluated product in the pool: `x^(-1)` used to intern
/// as `x^(1 · -1)` while `1/x` interned as `x^(-1)`.  The two are the same
/// function, but every structural detector that reads an exponent by matching
/// `ExprData::Integer` saw only the second one, so the *spelling* of an
/// integrand decided its route through the integrator.  Folding here means the
/// divergence never enters the pool.
///
/// Scope is deliberately just `Integer` and `Rational`:
///
/// * `Float` is left alone — negating it is exact, but `-0.0` and `(-1)·0.0`
///   are then two different literals rather than two spellings of one, and no
///   detector keys on a float exponent.
/// * No arithmetic is evaluated. `-(2+3)` and `x^(2-3)` keep their trees;
///   constant folding is the simplifier's job, not the parser's.
///
/// The `(-1) · literal` shape is still reachable through the pool builder API
/// (`pool.mul(vec![pool.integer(-1), pool.integer(1)])`, `Expr.__neg__`), so
/// the detectors keep their own normalising view of an integer exponent — see
/// [`crate::integrate::risch::tower::literal_integer`].  This is the first of
/// two layers, not a replacement for the second.
fn negate_literal(id: ExprId, pool: &ExprPool) -> Option<ExprId> {
    match pool.get(id) {
        ExprData::Integer(n) => Some(pool.integer(-n.0)),
        ExprData::Rational(r) => {
            let (num, den) = r.0.into_numer_denom();
            Some(pool.rational(-num, den))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Known function names
// ---------------------------------------------------------------------------

const KNOWN_FUNCS: &[&str] = &[
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    "atan2",
    "exp",
    "log",
    "sqrt",
    "abs",
    "sign",
    "floor",
    "ceil",
    "round",
    "erf",
    "erfc",
    "gamma",
    "lambert_w",
    "digamma",
    "bessel_j0",
    "bessel_j1",
    "EllipticK",
    "EllipticE",
    "EllipticF",
    "EllipticPi",
    // The non-elementary output basis (3.10.0).  Without these the integrator
    // can emit an antiderivative that neither parser can read back, so
    // `parse(str(integrate(f)))` is not a round trip — which is how a printed
    // result stops being usable input.  Every one is a registered primitive
    // with a derivative rule and an `f64` kernel; see
    // `alkahest_cas::primitive::{expint, fresnel, polylog}`.
    "Ei",
    "li",
    "Si",
    "Ci",
    "Shi",
    "Chi",
    "fresnels",
    "fresnelc",
    "dilog",
    "trigamma",
    // Reciprocal trig / hyperbolic functions.  These are *desugared* in
    // `parse_funcall` to their elementary reciprocal definitions (e.g.
    // `sec(x) → cos(x)^(-1)`); no `sec`/`csc`/… node ever enters the pool.
    "sec",
    "csc",
    "cot",
    "sech",
    "csch",
    "coth",
    // Also desugared: `cbrt(u) → u^(1/3)`.  It is not a registered primitive
    // and is not being made one — the power node already differentiates,
    // evaluates, simplifies and integrates, so a primitive would only add a
    // second spelling of the same object.  The one thing the desugar does not
    // reproduce is `libm::cbrt`'s real branch on negatives: `cbrt(-8)` is
    // `(-8)^(1/3)`, which the numeric interpreter reports as no-value rather
    // than as `-2`.  That is a refusal, not a wrong answer, and it is the
    // principal-branch convention the rest of the pool already uses for
    // fractional powers.
    "cbrt",
];

fn is_known_func(name: &str) -> bool {
    KNOWN_FUNCS.contains(&name)
}

/// If `name` is a reciprocal trig/hyperbolic function, return the elementary
/// primitive it is the reciprocal of (`sec → cos`, `csc → sin`, `cot → tan`,
/// and the hyperbolic analogues).  These are desugared to `base(x)^(-1)` at
/// parse time so every downstream stage (diff, eval, integrate, simplify)
/// operates purely on the existing `cos`/`sin`/`tan`/`cosh`/`sinh`/`tanh`
/// primitives.
fn reciprocal_base(name: &str) -> Option<&'static str> {
    match name {
        "sec" => Some("cos"),
        "csc" => Some("sin"),
        "cot" => Some("tan"),
        "sech" => Some("cosh"),
        "csch" => Some("sinh"),
        "coth" => Some("tanh"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Deepest grammatical nesting [`parse`] will accept.
///
/// The parser is recursive descent, so `"((((…x…))))"` or `"sin(sin(sin(…)))"`
/// costs native stack frames per level and overflows — a `SIGSEGV`, not an
/// error — long before it runs out of input.  This cap is the parser's
/// counterpart to [`crate::kernel::depth::MAX_EXPR_DEPTH`]; it has to be
/// counted separately because the overflow happens *before* any node is
/// interned, so there is no cached node depth to consult yet.
///
/// Deliberately equal to `MAX_EXPR_DEPTH`: text that parses should be text
/// whose result can then be simplified and printed.
const MAX_PARSE_DEPTH: u32 = crate::kernel::depth::MAX_EXPR_DEPTH;

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    pool: &'a ExprPool,
    symbols: &'a mut HashMap<String, ExprId>,
    /// Grammatical nesting depth of the production currently being parsed.
    depth: u32,
}

impl<'a> Parser<'a> {
    fn new(
        tokens: Vec<Token>,
        pool: &'a ExprPool,
        symbols: &'a mut HashMap<String, ExprId>,
    ) -> Self {
        Parser {
            tokens,
            pos: 0,
            pool,
            symbols,
            depth: 0,
        }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        if tok.tok != Tok::Eof {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, expected: &Tok) -> Result<Token, ParseError> {
        let tok = self.advance();
        if &tok.tok == expected {
            Ok(tok)
        } else {
            let label = format!("{expected:?}");
            if tok.tok == Tok::Eof {
                Err(ParseError::syntax(
                    format!("expected {label} but reached end of input"),
                    (tok.offset, tok.offset),
                ))
            } else {
                Err(ParseError::syntax(
                    format!("expected {label}"),
                    (tok.offset, tok.offset + 1),
                ))
            }
        }
    }

    fn parse_expr(&mut self, rbp: u8) -> Result<ExprId, ParseError> {
        // Every nested production — a parenthesis, a prefix minus, a function
        // argument — re-enters here, so this is the one place that has to count
        // to keep the recursion off the end of the stack.
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            let offset = self.peek().offset;
            self.depth -= 1;
            return Err(ParseError::too_deep(
                format!("expression nesting exceeds the limit of {MAX_PARSE_DEPTH}"),
                (offset, offset + 1),
            ));
        }
        let result = self.parse_expr_inner(rbp);
        self.depth -= 1;
        result
    }

    fn parse_expr_inner(&mut self, rbp: u8) -> Result<ExprId, ParseError> {
        let tok = self.advance();
        let mut left = self.nud(tok)?;
        loop {
            let lbp = infix_bp(&self.peek().tok);
            if lbp <= rbp {
                break;
            }
            let op = self.advance();
            left = self.led(op, left)?;
        }
        Ok(left)
    }

    /// Null denotation — prefix position / atom.
    fn nud(&mut self, tok: Token) -> Result<ExprId, ParseError> {
        let pool = self.pool;
        match &tok.tok {
            Tok::Num(s) => {
                let s = s.clone();
                if s.contains('.') || s.to_ascii_lowercase().contains('e') {
                    // The lexer only emits shapes `f64::from_str` accepts, but
                    // an `unwrap` here once turned a lexer gap into a process
                    // abort.  Keep the failure structured no matter what the
                    // lexer hands over.
                    float_literal(pool, &s).ok_or_else(|| {
                        ParseError::lex(
                            format!("malformed or out-of-range number literal: {s}"),
                            (tok.offset, tok.offset + s.len()),
                        )
                    })
                } else {
                    // Arbitrary precision, like every integer the pool
                    // holds: parsing into `i64` refused `10^20` written out,
                    // so a printed expression with a large coefficient did
                    // not read back.
                    let n: rug::Integer = s.parse().map_err(|_| {
                        ParseError::lex(
                            format!("malformed integer literal: {s}"),
                            (tok.offset, tok.offset + s.len()),
                        )
                    })?;
                    Ok(pool.integer(n))
                }
            }

            Tok::Ident(name) => {
                let name = name.clone();
                if self.peek().tok == Tok::LParen {
                    self.parse_funcall(&name, tok.offset)
                } else {
                    // Look up in caller-supplied map, or intern a new Real symbol.
                    let id = if let Some(&id) = self.symbols.get(&name) {
                        id
                    } else {
                        let id = pool.symbol(name.clone(), Domain::Real);
                        self.symbols.insert(name, id);
                        id
                    };
                    Ok(id)
                }
            }

            Tok::Minus => {
                let operand = self.parse_expr(BP_UNARY)?;
                // -3  →  the literal -3;  -x  →  (-1) * x
                if let Some(folded) = negate_literal(operand, self.pool) {
                    return Ok(folded);
                }
                let neg1 = self.pool.integer(-1i64);
                Ok(self.pool.mul(vec![neg1, operand]))
            }

            Tok::Plus => self.parse_expr(BP_UNARY),

            Tok::LParen => {
                if self.peek().tok == Tok::RParen {
                    return Err(ParseError::syntax(
                        "empty parentheses",
                        (tok.offset, tok.offset + 1),
                    ));
                }
                let inner = self.parse_expr(0)?;
                self.expect(&Tok::RParen)?;
                Ok(inner)
            }

            other => Err(ParseError::syntax(
                format!("unexpected token {other:?}"),
                (tok.offset, tok.offset + 1),
            )),
        }
    }

    /// Left denotation — infix position.
    fn led(&mut self, op: Token, left: ExprId) -> Result<ExprId, ParseError> {
        let pool = self.pool;
        match op.tok {
            Tok::Plus => {
                let right = self.parse_expr(BP_ADD)?;
                Ok(pool.add(vec![left, right]))
            }
            Tok::Minus => {
                let right = self.parse_expr(BP_ADD)?;
                // left - right  →  left + (-1)*right
                let neg1 = pool.integer(-1i64);
                let neg_right = pool.mul(vec![neg1, right]);
                Ok(pool.add(vec![left, neg_right]))
            }
            Tok::Star => {
                let right = self.parse_expr(BP_MUL)?;
                Ok(pool.mul(vec![left, right]))
            }
            Tok::Slash => {
                let right = self.parse_expr(BP_MUL)?;
                // left / right  →  left * right^(-1)
                let neg1 = pool.integer(-1i64);
                let inv = pool.pow(right, neg1);
                Ok(pool.mul(vec![left, inv]))
            }
            Tok::Caret | Tok::StarStar => {
                // Right-associative: right-bp = BP_POW - 1
                let right = self.parse_expr(BP_POW - 1)?;
                Ok(pool.pow(left, right))
            }
            other => Err(ParseError::syntax(
                format!("unexpected token {other:?} in infix position"),
                (op.offset, op.offset + 1),
            )),
        }
    }

    fn parse_funcall(&mut self, name: &str, offset: usize) -> Result<ExprId, ParseError> {
        if !is_known_func(name) {
            return Err(ParseError::unknown_func(
                format!("unknown function '{name}'"),
                (offset, offset + name.len()),
            ));
        }
        self.advance(); // consume "("
        let mut args = Vec::new();
        if self.peek().tok != Tok::RParen {
            args.push(self.parse_expr(0)?);
            while self.peek().tok == Tok::Comma {
                self.advance(); // consume ","
                args.push(self.parse_expr(0)?);
            }
        }
        self.expect(&Tok::RParen)?;

        // Desugar reciprocal trig/hyperbolic calls to `base(x)^(-1)` so no
        // `sec`/`csc`/… node ever reaches the pool.  Only the single-argument
        // form is meaningful; any other arity is a syntax error, mirroring how
        // the other unary functions reject extra arguments downstream.
        if let Some(base) = reciprocal_base(name) {
            if args.len() != 1 {
                return Err(ParseError::syntax(
                    format!("{name} takes exactly 1 argument, got {}", args.len()),
                    (offset, offset + name.len()),
                ));
            }
            let inner = self.pool.func(base, args);
            let neg1 = self.pool.integer(-1_i64);
            return Ok(self.pool.pow(inner, neg1));
        }

        // Desugar `cbrt(u)` to `u^(1/3)`.  See the note in `KNOWN_FUNCS`.
        if name == "cbrt" {
            if args.len() != 1 {
                return Err(ParseError::syntax(
                    format!("cbrt takes exactly 1 argument, got {}", args.len()),
                    (offset, offset + name.len()),
                ));
            }
            let third = self.pool.rational(1_i64, 3_i64);
            return Ok(self.pool.pow(args[0], third));
        }

        // A built-in name at the wrong arity (`sin()`, `sin(x, x)`,
        // `EllipticPi(x)`) is not an expression: every consumer indexes a
        // built-in's arguments by position.  Refuse it here, as the
        // reciprocal and `cbrt` desugars above already do for theirs.
        self.pool
            .try_func(name, args)
            .map_err(|e| ParseError::syntax(e.to_string(), (offset, offset + name.len())))
    }
}

/// Precision, in bits, at which a decimal float literal is read back.
///
/// `rug` renders a `prec`-bit float with `1 + ⌈prec · log₁₀ 2⌉` significant
/// digits — 17 for the default 53 — and [`crate::kernel::BigFloat`]'s
/// `Display` prints exactly that.  Reading every literal as an `f64`, as the
/// parser used to, cut a printed 200-bit float back to 53 bits.  So:
///
/// * up to 17 significant digits: 53 bits, the `f64` it always was;
/// * more: the *largest* precision whose rendering has that many digits.
///   That is never less than the precision that printed it, so no digit
///   written is thrown away, and re-printing the result gives the same text.
///
/// The precision itself is not in the text, so a float printed at, say, 200
/// bits reads back at 202 (the three precisions 200–202 all print 62 digits).
pub fn float_literal_prec(text: &str) -> u32 {
    let mantissa = text.split(['e', 'E']).next().unwrap_or("");
    let digits = mantissa
        .chars()
        .filter(char::is_ascii_digit)
        .skip_while(|&c| c == '0')
        .count();
    if digits <= 17 {
        return 53;
    }
    // Largest p with 1 + ceil(p·log10 2) == digits, i.e. p·log10 2 ≤ digits − 1.
    let p = ((digits - 1) as f64 / std::f64::consts::LOG10_2).floor();
    if p >= f64::from(rug::float::prec_max()) {
        rug::float::prec_max()
    } else {
        p as u32
    }
}

/// Intern the decimal float literal `text` (`1.5`, `2.5e-3`, `.5`) at
/// [`float_literal_prec`] bits, or `None` if it is not one.
///
/// A literal past the `f64` exponent range (`1e999999`, `1e-999999`) is read
/// at 53 bits through MPFR, whose exponent reaches about ±3·10⁸ decimal
/// digits, instead of rounding to `inf` or `0`: `parse("1e999999")` used to
/// be infinity, silently.  A literal past even that range is `None` — the
/// callers report it as an out-of-range literal — never `±inf` and never a
/// zero standing in for a nonzero value.
pub fn float_literal(pool: &ExprPool, text: &str) -> Option<ExprId> {
    let prec = float_literal_prec(text);
    let mantissa_nonzero = text
        .split(['e', 'E'])
        .next()
        .is_some_and(|m| m.bytes().any(|b| (b'1'..=b'9').contains(&b)));
    if prec == 53 {
        let v = text.parse::<f64>().ok()?;
        if v.is_finite() && (v != 0.0 || !mantissa_nonzero) {
            return Some(pool.float(v, 53));
        }
    }
    let parsed = rug::Float::parse(text).ok()?;
    let inner = rug::Float::with_val(prec, parsed);
    if !inner.is_finite() || (inner.is_zero() && mantissa_nonzero) {
        return None;
    }
    Some(pool.intern(ExprData::Float(crate::kernel::BigFloat { inner, prec })))
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse a mathematical expression string into an [`ExprId`].
///
/// Uses a Pratt (top-down operator precedence) recursive-descent parser.
/// The grammar supports integer/float literals, identifiers, arithmetic
/// operators (`+`, `-`, `*`, `/`, `^`, `**`), unary `-`/`+`, parentheses,
/// and a fixed set of mathematical functions:
/// `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `atan`,
/// `asinh`, `acosh`, `atanh`, `atan2`, `exp`, `log`, `sqrt`, `abs`, `sign`,
/// `floor`, `ceil`, `round`, `erf`, `erfc`, `gamma`, and the non-elementary
/// output basis `Ei`, `li`, `Si`, `Ci`, `Shi`, `Chi`, `fresnels`, `fresnelc`,
/// `dilog`, `trigamma` — the last group so that an antiderivative the
/// integrator prints can be read back in.
///
/// The reciprocal trig/hyperbolic functions `sec`, `csc`, `cot`, `sech`,
/// `csch`, and `coth` are also accepted; they are desugared at parse time to
/// their elementary reciprocal definitions (`sec(x) → cos(x)^(-1)`,
/// `csc(x) → sin(x)^(-1)`, `cot(x) → tan(x)^(-1)`, and the hyperbolic
/// analogues), so no dedicated node for them exists in the pool.  `cbrt(u)` is
/// desugared the same way, to `u^(1/3)`.
///
/// `symbols` maps identifier names to pre-existing [`ExprId`]s.  Identifiers
/// not in the map are interned as new `Domain::Real` symbols and added to the
/// map so they are reused within the same call.
///
/// # Errors
///
/// Returns [`ParseError`] (`E-PARSE-001` lexical, `E-PARSE-002` syntactic,
/// `E-PARSE-003` unknown function) on failure, with a byte-offset span.
///
/// # Example
///
/// ```
/// use alkahest_cas::{ExprPool, parse};
/// use alkahest_cas::kernel::Domain;
/// use std::collections::HashMap;
///
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// let mut syms = HashMap::from([("x".to_owned(), x)]);
/// let e = parse("sin(x)^2 + cos(x)^2", &pool, &mut syms).unwrap();
/// ```
pub fn parse(
    src: &str,
    pool: &ExprPool,
    symbols: &mut HashMap<String, ExprId>,
) -> Result<ExprId, ParseError> {
    let tokens = tokenize(src)?;
    let first = &tokens[0];
    if first.tok == Tok::Eof {
        return Err(ParseError::syntax("empty expression", (0, 0)));
    }
    let mut parser = Parser::new(tokens, pool, symbols);
    let expr = parser.parse_expr(0)?;
    let tail = parser.peek();
    if tail.tok != Tok::Eof {
        let off = tail.offset;
        return Err(ParseError::syntax(
            format!("unexpected token {:?}", tail.tok),
            (off, off + 1),
        ));
    }
    Ok(expr)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pool_and_x() -> (ExprPool, ExprId, HashMap<String, ExprId>) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let syms = HashMap::from([("x".to_owned(), x)]);
        (pool, x, syms)
    }

    #[test]
    fn integer_literal() {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        let e = parse("42", &pool, &mut syms).unwrap();
        assert_eq!(e, pool.integer(42i64));
    }

    #[test]
    fn float_literal() {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        parse("3.14", &pool, &mut syms).unwrap();
    }

    #[test]
    fn identifier_symbol() {
        let (pool, x, mut syms) = pool_and_x();
        let e = parse("x", &pool, &mut syms).unwrap();
        assert_eq!(e, x);
    }

    #[test]
    fn addition() {
        let (pool, x, mut syms) = pool_and_x();
        let e = parse("x + 1", &pool, &mut syms).unwrap();
        let expected = pool.add(vec![x, pool.integer(1i64)]);
        assert_eq!(e, expected);
    }

    #[test]
    fn unary_minus() {
        let (pool, x, mut syms) = pool_and_x();
        let e = parse("-x", &pool, &mut syms).unwrap();
        let neg1 = pool.integer(-1i64);
        let expected = pool.mul(vec![neg1, x]);
        assert_eq!(e, expected);
    }

    // -----------------------------------------------------------------------
    // Unary minus on a literal folds; unary minus on anything else does not
    // -----------------------------------------------------------------------

    #[test]
    fn unary_minus_on_a_literal_folds() {
        let (pool, _x, mut syms) = pool_and_x();
        assert_eq!(
            parse("-3", &pool, &mut syms).unwrap(),
            pool.integer(-3i64),
            "-3 must intern as the literal -3, not as 1 · -3"
        );
        assert_eq!(
            parse("-(-3)", &pool, &mut syms).unwrap(),
            pool.integer(3i64),
            "the fold has to compose with itself"
        );
        assert_eq!(
            parse("-0", &pool, &mut syms).unwrap(),
            pool.integer(0i64),
            "there is only one integer zero"
        );
    }

    /// The bug this fold exists to kill: `^(-n)` used to intern its exponent as
    /// the unevaluated `Mul[1, -n]`, which every detector that reads an exponent
    /// by matching `ExprData::Integer` saw as a non-literal and bailed on — while
    /// the `/` spelling of the very same function handed it a bare `Integer(-n)`.
    #[test]
    fn a_negative_exponent_is_a_literal_however_it_is_spelled() {
        let (pool, _x, mut syms) = pool_and_x();

        // The exponent node itself: this is what the detectors read.
        for (src, want) in [
            ("x^(-1)", -1_i64),
            ("x^-1", -1),
            ("x^(-2)", -2),
            ("(x^2+1)^(-1)", -1),
            ("(x*log(x))^(-1)", -1),
        ] {
            let e = parse(src, &pool, &mut syms).unwrap();
            let ExprData::Pow { exp, .. } = pool.get(e) else {
                panic!("`{src}` should parse to a Pow, got {}", pool.display(e));
            };
            assert_eq!(
                exp,
                pool.integer(want),
                "`{src}` has exponent {}, not the literal {want}",
                pool.display(exp)
            );
        }

        // …and so `a · b^(-1)` and `a/b` are now literally one node.
        for (a, b) in [
            ("2*x^(-1)", "2/x"),
            ("log(x)*(x^2+1)^(-1)", "log(x)/(x^2+1)"),
            ("sin(x)*(x*log(x))^(-1)", "sin(x)/(x*log(x))"),
        ] {
            let ea = parse(a, &pool, &mut syms).unwrap();
            let eb = parse(b, &pool, &mut syms).unwrap();
            assert_eq!(
                ea,
                eb,
                "`{a}` and `{b}` must hash-cons to one node, got {} vs {}",
                pool.display(ea),
                pool.display(eb)
            );
        }

        // What this does *not* claim.  `/` keeps its left operand, so a bare
        // `1/x` is `1 · x^(-1)` and carries a redundant unit factor that
        // `x^(-1)` does not; and `1/x^2` is `(x^2)^(-1)`, not `x^(-2)`.  Both
        // are pre-existing spelling differences above the exponent, and folding
        // them away is the simplifier's job, not the parser's.  Pinned so this
        // test is not read as claiming more than it does.
        assert_ne!(
            parse("1/x", &pool, &mut syms).unwrap(),
            parse("x^(-1)", &pool, &mut syms).unwrap()
        );
        assert_ne!(
            parse("1/x^2", &pool, &mut syms).unwrap(),
            parse("x^(-2)", &pool, &mut syms).unwrap()
        );
    }

    /// Nothing but a bare `Integer`/`Rational` operand folds.  Over-folding
    /// would be a silent precedence change (`-2^2` is `-(2^2) = -4`, never
    /// `(-2)^2 = 4`) and would strip the `(-1) ·` prefix that `simplify` and
    /// the display layer key on for symbolic negation.
    #[test]
    fn unary_minus_on_a_non_literal_is_left_alone() {
        let (pool, x, mut syms) = pool_and_x();
        let neg1 = pool.integer(-1i64);

        // Symbol.
        assert_eq!(
            parse("-x", &pool, &mut syms).unwrap(),
            pool.mul(vec![neg1, x])
        );
        // Sum — no constant folding: `-(2+3)` keeps its tree.
        let two_plus_three = pool.add(vec![pool.integer(2i64), pool.integer(3i64)]);
        assert_eq!(
            parse("-(2+3)", &pool, &mut syms).unwrap(),
            pool.mul(vec![neg1, two_plus_three])
        );
        // Function application.
        assert_eq!(
            parse("-sin(x)", &pool, &mut syms).unwrap(),
            pool.mul(vec![neg1, pool.func("sin", vec![x])])
        );
        // Double negation of a symbol stays two products, not `x`.
        let neg_x = pool.mul(vec![neg1, x]);
        assert_eq!(
            parse("-(-x)", &pool, &mut syms).unwrap(),
            pool.mul(vec![neg1, neg_x])
        );
        // `^` binds tighter than prefix `-`, so the operand is a `Pow`, never a
        // literal.  These two are the precedence regression guards.
        assert_eq!(
            parse("-x^2", &pool, &mut syms).unwrap(),
            pool.mul(vec![neg1, pool.pow(x, pool.integer(2i64))])
        );
        assert_eq!(
            parse("-2^2", &pool, &mut syms).unwrap(),
            pool.mul(vec![neg1, pool.pow(pool.integer(2i64), pool.integer(2i64))]),
            "-2^2 is -(2^2) = -4; folding it to (-2)^2 = 4 would change the value"
        );
        // A float literal is deliberately out of scope — see `negate_literal`.
        assert_eq!(
            parse("-3.5", &pool, &mut syms).unwrap(),
            pool.mul(vec![neg1, pool.float(3.5, 53)])
        );
    }

    /// `-1/2` is `(-1)/2`, not `-(1/2)`: prefix `-` binds tighter than `/`, so
    /// the fold sees the literal `1` and the division is applied afterwards.
    /// The exponent is still var-free and negative, which is all the detectors
    /// downstream ask of it.
    #[test]
    fn a_negative_rational_exponent_keeps_its_value() {
        let (pool, x, mut syms) = pool_and_x();
        let e = parse("x^(-1/2)", &pool, &mut syms).unwrap();
        let expected_exp = pool.mul(vec![
            pool.integer(-1i64),
            pool.pow(pool.integer(2i64), pool.integer(-1i64)),
        ]);
        assert_eq!(e, pool.pow(x, expected_exp));
        // `1/sqrt(x)` is *not* the same tree: `sqrt` is a `Func`, not a `Pow`.
        // Pinned so nobody reads the test above as claiming more than it does.
        assert_ne!(e, parse("1/sqrt(x)", &pool, &mut syms).unwrap());
    }

    /// The `Rational` arm of [`negate_literal`] is unreachable from the lexer
    /// today (it only emits `Integer` and `Float`), so exercise it directly
    /// rather than leaving it as untested defensive code.
    #[test]
    fn negate_literal_handles_both_exact_kinds() {
        let pool = ExprPool::new();
        assert_eq!(
            negate_literal(pool.integer(7i64), &pool),
            Some(pool.integer(-7i64))
        );
        assert_eq!(
            negate_literal(pool.rational(2i64, 3i64), &pool),
            Some(pool.rational(-2i64, 3i64))
        );
        assert_eq!(
            negate_literal(pool.rational(-2i64, 3i64), &pool),
            Some(pool.rational(2i64, 3i64))
        );
        assert_eq!(negate_literal(pool.float(1.5, 53), &pool), None);
        assert_eq!(negate_literal(pool.symbol("y", Domain::Real), &pool), None);
    }

    /// Every shape the fold touches has to survive `display` → `parse` → the
    /// same node.  A representation change that the printer cannot spell back
    /// is a round-trip bug, not a simplification.
    #[test]
    fn negatives_round_trip_through_display() {
        let (pool, _x, mut syms) = pool_and_x();
        for src in [
            "-x",
            "-3",
            "x - 3",
            "2 - -3",
            "-(-x)",
            "-(-3)",
            "x^-1",
            "x^(-1)",
            "-x^2",
            "1/-x",
            "-2/3",
            "-3.5",
            "1/x",
            "(x^2+1)^(-1)",
        ] {
            let e = parse(src, &pool, &mut syms).unwrap();
            let shown = pool.display(e).to_string();
            let reparsed = parse(&shown, &pool, &mut syms).unwrap_or_else(|err| {
                panic!("`{src}` displayed as `{shown}`, which fails to reparse: {err}")
            });
            assert_eq!(
                e,
                reparsed,
                "`{src}` displayed as `{shown}`, which reparsed to `{}`",
                pool.display(reparsed)
            );
        }
    }

    #[test]
    fn power_right_assoc() {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        // 2^3^2 should parse as 2^(3^2), not (2^3)^2
        let e = parse("2^3^2", &pool, &mut syms).unwrap();
        let two = pool.integer(2i64);
        let three = pool.integer(3i64);
        let inner = pool.pow(three, two); // 3^2 (two is hash-consed: same id as literal 2)
        let expected = pool.pow(two, inner); // 2^(3^2)
        assert_eq!(e, expected);
    }

    #[test]
    fn function_call() {
        let (pool, x, mut syms) = pool_and_x();
        let e = parse("sin(x)", &pool, &mut syms).unwrap();
        let expected = pool.func("sin", vec![x]);
        assert_eq!(e, expected);
    }

    /// Refuse `src` and return the code, from a thread with room to reach the
    /// cap.
    ///
    /// [`MAX_PARSE_DEPTH`] is sized for the shipped **release** build on the
    /// usual 8 MiB stack.  A `cargo test` worker gets 2 MiB and debug frames
    /// are several times larger, so a debug run overflows before the cap is
    /// reached — the test would then abort the whole runner, which is exactly
    /// the outcome this feature exists to prevent.  64 MiB covers both.
    fn parse_code_on_big_stack(src: String) -> &'static str {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let pool = ExprPool::new();
                let mut syms = HashMap::new();
                parse(&src, &pool, &mut syms)
                    .err()
                    .map(|e| e.code())
                    .unwrap_or("OK")
            })
            .expect("spawn")
            .join()
            .expect("deep parse must return, not overflow the stack")
    }

    /// Recursive descent costs native stack frames per nesting level, so
    /// `"((((…x…))))"` used to overflow the stack — a `SIGSEGV` that kills the
    /// process, with no error for the caller to catch.  Just past the limit is
    /// used deliberately: a regression must fail this test, not crash the test
    /// runner.
    #[test]
    fn deeply_nested_parentheses_are_refused_not_fatal() {
        let n = (MAX_PARSE_DEPTH + 8) as usize;
        let src = format!("{}x{}", "(".repeat(n), ")".repeat(n));
        assert_eq!(parse_code_on_big_stack(src), "E-PARSE-004");
    }

    /// Prefix operators and function calls re-enter the same production, so
    /// they must be counted too.
    #[test]
    fn deeply_nested_prefix_and_calls_are_refused() {
        let n = (MAX_PARSE_DEPTH + 8) as usize;
        assert_eq!(
            parse_code_on_big_stack(format!("{}x", "-".repeat(n))),
            "E-PARSE-004"
        );
        assert_eq!(
            parse_code_on_big_stack(format!("{}x{}", "sin(".repeat(n), ")".repeat(n))),
            "E-PARSE-004"
        );
    }

    /// One level under the cap must still parse, so the limit is a real
    /// boundary and not merely "everything deep fails".
    #[test]
    fn just_under_the_parse_cap_still_parses() {
        let n = (MAX_PARSE_DEPTH - 2) as usize;
        assert_eq!(
            parse_code_on_big_stack(format!("{}x{}", "(".repeat(n), ")".repeat(n))),
            "OK"
        );
    }

    /// A long *flat* sum is not nesting and must still parse: the cap counts
    /// depth, not length.
    #[test]
    fn a_long_flat_sum_is_not_nesting() {
        let (pool, _x, mut syms) = pool_and_x();
        let src = vec!["x"; 20_000].join("+");
        parse(&src, &pool, &mut syms).expect("a flat sum has depth 1 per term");
    }

    #[test]
    fn atan2_two_args() {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        parse("atan2(1, 2)", &pool, &mut syms).unwrap();
    }

    #[test]
    fn unknown_function_error() {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        let err = parse("foo(x)", &pool, &mut syms).unwrap_err();
        assert_eq!(err.code(), "E-PARSE-003");
    }

    #[test]
    fn lex_error() {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        let err = parse("x # y", &pool, &mut syms).unwrap_err();
        assert_eq!(err.code(), "E-PARSE-001");
    }

    #[test]
    fn empty_expression_error() {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        let err = parse("", &pool, &mut syms).unwrap_err();
        assert_eq!(err.code(), "E-PARSE-002");
    }

    #[test]
    fn auto_intern_new_symbol() {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        parse("y + 1", &pool, &mut syms).unwrap();
        assert!(syms.contains_key("y"));
    }

    // -----------------------------------------------------------------------
    // Reciprocal trig / hyperbolic desugaring
    // -----------------------------------------------------------------------

    /// Each reciprocal function desugars to `base(x)^(-1)`; no `sec`/`csc`/…
    /// node is ever produced.
    #[test]
    fn reciprocal_trig_desugar_structure() {
        let cases = [
            ("sec(x)", "cos"),
            ("csc(x)", "sin"),
            ("cot(x)", "tan"),
            ("sech(x)", "cosh"),
            ("csch(x)", "sinh"),
            ("coth(x)", "tanh"),
        ];
        for (src, base) in cases {
            let (pool, x, mut syms) = pool_and_x();
            let e = parse(src, &pool, &mut syms).unwrap();
            let neg1 = pool.integer(-1i64);
            let expected = pool.pow(pool.func(base, vec![x]), neg1);
            assert_eq!(e, expected, "{src} should desugar to {base}(x)^(-1)");
        }
    }

    /// The desugared argument is threaded through, not just a bare symbol.
    #[test]
    fn reciprocal_trig_desugar_with_expression_arg() {
        let (pool, x, mut syms) = pool_and_x();
        let e = parse("sec(2*x)", &pool, &mut syms).unwrap();
        let two_x = pool.mul(vec![pool.integer(2i64), x]);
        let neg1 = pool.integer(-1i64);
        let expected = pool.pow(pool.func("cos", vec![two_x]), neg1);
        assert_eq!(e, expected);
    }

    /// Differentiating a reciprocal function succeeds (routes through the
    /// existing `cos`/`sin`/… diff rules via the `^(-1)` desugar).
    #[test]
    fn reciprocal_trig_diff_closes() {
        let (pool, x, mut syms) = pool_and_x();
        let e = parse("sec(x)", &pool, &mut syms).unwrap();
        let d = crate::diff::diff(e, x, &pool);
        assert!(d.is_ok(), "d/dx sec(x) should differentiate");
    }

    /// `∫ sec(x)² dx` closes (== tan(x)) and routes through the reciprocal-square
    /// trig rule: `sec(x)^2` parses to `(cos(x)^(-1))^2`, which `simplify`
    /// canonicalizes to `cos(x)^(-2)` — the exact shape the integrator's
    /// `∫ 1/cos² = tan` rule matches.  Like every integrand, it must be in
    /// canonical (simplified) form; the integrator's internal soundness gate then
    /// guarantees `d/dx(result) == sec(x)²`.
    #[test]
    fn reciprocal_trig_integrate_sec_squared() {
        let (pool, x, mut syms) = pool_and_x();
        let e =
            crate::simplify::simplify(parse("sec(x)^2", &pool, &mut syms).unwrap(), &pool).value;
        let r = crate::integrate::integrate(e, x, &pool);
        assert!(r.is_ok(), "∫ sec(x)² dx should close (== tan(x))");
    }

    /// `∫ csc(x)² dx` closes (== −cot(x)); `csc(x)^2` simplifies to `sin(x)^(-2)`.
    #[test]
    fn reciprocal_trig_integrate_csc_squared() {
        let (pool, x, mut syms) = pool_and_x();
        let e =
            crate::simplify::simplify(parse("csc(x)^2", &pool, &mut syms).unwrap(), &pool).value;
        let r = crate::integrate::integrate(e, x, &pool);
        assert!(r.is_ok(), "∫ csc(x)² dx should close");
    }

    /// A reciprocal function called with the wrong arity is a syntax error.
    #[test]
    fn reciprocal_trig_wrong_arity_errors() {
        let (pool, _x, mut syms) = pool_and_x();
        let err = parse("sec(x, x)", &pool, &mut syms).unwrap_err();
        assert_eq!(err.code(), "E-PARSE-002");
    }

    /// Regression: the base trig/hyperbolic functions and `atan2` still parse
    /// to plain `Func` nodes (unaffected by the desugar).
    #[test]
    fn base_trig_functions_unchanged() {
        for src in [
            "sin(x)", "cos(x)", "tan(x)", "sinh(x)", "cosh(x)", "tanh(x)",
        ] {
            let (pool, _x, mut syms) = pool_and_x();
            parse(src, &pool, &mut syms).unwrap();
        }
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        parse("atan2(1, 2)", &pool, &mut syms).unwrap();
    }

    // -----------------------------------------------------------------------
    // Associativity.  The grammar is left-associative, so `a*b*c` is parsed as
    // `(a*b)*c`; `ExprPool::mul`/`add` splice that back into one flat node, so
    // the parsed form and the n-ary builder form are the same expression.
    // -----------------------------------------------------------------------

    fn pool_xyz() -> (ExprPool, [ExprId; 3], HashMap<String, ExprId>) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let syms = HashMap::from([
            ("x".to_owned(), x),
            ("y".to_owned(), y),
            ("z".to_owned(), z),
        ]);
        (pool, [x, y, z], syms)
    }

    #[test]
    fn parsed_product_chain_is_the_flat_mul() {
        let (pool, [x, y, z], mut syms) = pool_xyz();
        let flat = pool.mul(vec![x, y, z]);
        for src in ["x*y*z", "(x*y)*z", "x*(y*z)"] {
            assert_eq!(
                parse(src, &pool, &mut syms).unwrap(),
                flat,
                "{src} must parse to the flat 3-factor Mul"
            );
        }
    }

    #[test]
    fn parsed_sum_chain_is_the_flat_add() {
        let (pool, [x, y, z], mut syms) = pool_xyz();
        let flat = pool.add(vec![x, y, z]);
        for src in ["x+y+z", "(x+y)+z", "x+(y+z)"] {
            assert_eq!(
                parse(src, &pool, &mut syms).unwrap(),
                flat,
                "{src} must parse to the flat 3-term Add"
            );
        }
    }

    /// A longer chain, and one mixing both operators, to show the splice is not
    /// a one-level special case and does not cross operator boundaries.
    #[test]
    fn deeper_parsed_chains_flatten() {
        let (pool, [x, y, z], mut syms) = pool_xyz();
        let e = parse("x*y*z*x*y", &pool, &mut syms).unwrap();
        assert_eq!(e, pool.mul(vec![x, y, z, x, y]));
        assert_eq!(pool.depth(e), 2);

        let mixed = parse("x*y + z + x*y*z", &pool, &mut syms).unwrap();
        assert_eq!(
            mixed,
            pool.add(vec![pool.mul(vec![x, y]), z, pool.mul(vec![x, y, z])])
        );
    }

    /// `parse → display → parse` is a fixpoint on the flattened form.
    #[test]
    fn display_round_trips_through_the_parser() {
        for src in [
            "x*y*z",
            "x+y+z",
            "x*y + z",
            "(x + y)*z",
            "x*y*z + x*y + z",
            "2*x*y*z",
            "x^2*y*z",
        ] {
            let (pool, _xyz, mut syms) = pool_xyz();
            let once = parse(src, &pool, &mut syms).unwrap();
            let rendered = pool.display(once).to_string();
            let twice = parse(&rendered, &pool, &mut syms).unwrap();
            assert_eq!(once, twice, "round trip changed {src} via {rendered}");
            assert_eq!(pool.display(twice).to_string(), rendered);
        }
    }

    /// Every name the integrator can emit must parse back to the same node.
    ///
    /// The two parsers are separate hand-maintained implementations
    /// (`CONTRIBUTING.md` § "The parser exists twice"), so nothing enforces
    /// this beyond a test on each side.  Without it, `parse(str(integrate(f)))`
    /// stops being a round trip the moment the integrator learns a new name —
    /// which is exactly what a printed result being usable input depends on.
    #[test]
    fn the_special_function_output_basis_round_trips() {
        for name in crate::integrate::SPECIAL_BASIS {
            // Each name at its smallest arity: `EllipticE` has a one-argument
            // complete form, `EllipticF` only the two-argument incomplete one
            // (`EllipticF(x)` is refused, `E-PARSE-002`).
            let n = crate::kernel::known_func_arity(name).map_or(1, |(lo, _)| lo);
            let src = format!("{name}({})", vec!["x"; n].join(", "));
            let (pool, _xyz, mut syms) = pool_xyz();
            let parsed =
                parse(&src, &pool, &mut syms).unwrap_or_else(|e| panic!("{src} must parse: {e}"));
            match pool.get(parsed) {
                crate::kernel::ExprData::Func {
                    name: ref got,
                    ref args,
                } => {
                    assert_eq!(got, name, "{src} parsed to the wrong node");
                    assert_eq!(args.len(), n, "{src} parsed with the wrong arity");
                }
                other => panic!("{src} parsed to {other:?}, not a Func node"),
            }
            let rendered = pool.display(parsed).to_string();
            assert_eq!(rendered, src, "{name} does not print as it parses");
            assert_eq!(
                parse(&rendered, &pool, &mut syms).unwrap(),
                parsed,
                "{name} does not round trip"
            );
        }
    }

    // -----------------------------------------------------------------------
    // A truncated exponent is an error, not a panic
    // -----------------------------------------------------------------------

    /// `"1e"` used to *panic*.
    ///
    /// The lexer consumed `e`/`E` and an optional sign unconditionally once it
    /// had seen a digit, so `"1e"` became `Tok::Num("1e")`; `nud` then called
    /// `"1e".parse::<f64>().unwrap()` on an `Err`.  A parser that aborts the
    /// process on a two-character string is unusable on text the caller did not
    /// write, which is the only kind of text a parser is for.
    #[test]
    fn a_truncated_exponent_is_a_lex_error_not_a_panic() {
        for src in [
            "1e", "1E", "1e+", "1e-", "1E+", "1E-", "2.5e", "2.5E-", ".5e", "1.e", "0e", "1e ",
            "1e*2", "1e+x", "x^2e",
        ] {
            let pool = ExprPool::new();
            let mut syms = HashMap::new();
            let err = match parse(src, &pool, &mut syms) {
                Ok(_) => panic!("{src:?} must not parse"),
                Err(e) => e,
            };
            assert_eq!(err.code(), "E-PARSE-001", "{src:?} got the wrong code");
            let (lo, hi) = err.span().unwrap_or_else(|| panic!("{src:?} has no span"));
            assert!(
                lo < hi && hi <= src.len(),
                "{src:?} span {lo}..{hi} is bogus"
            );
            assert!(
                src[lo..hi].to_ascii_lowercase().contains('e'),
                "{src:?} span {lo}..{hi} does not cover the exponent marker"
            );
        }
    }

    /// The neighbouring shapes that *are* well formed keep working — the fix
    /// must not turn a legal literal into an error, and must not backtrack
    /// `2e` into `2 * e` either.
    #[test]
    fn well_formed_exponents_and_trailing_dots_still_parse() {
        for (src, expect) in [
            ("1e5", 1e5),
            ("1E5", 1e5),
            ("1e+5", 1e5),
            ("1e-5", 1e-5),
            ("2.5e3", 2.5e3),
            (".5e3", 0.5e3),
            ("1.", 1.0),
            ("1.e5", 1e5),
            ("0.", 0.0),
            (".5", 0.5),
        ] {
            let pool = ExprPool::new();
            let mut syms = HashMap::new();
            let id = parse(src, &pool, &mut syms)
                .unwrap_or_else(|e| panic!("{src:?} must parse, got {e}"));
            match pool.get(id) {
                ExprData::Float(f) => assert_eq!(
                    f.inner.to_f64(),
                    expect,
                    "{src:?} parsed to the wrong value"
                ),
                other => panic!("{src:?} parsed to {other:?}, not a float"),
            }
        }
    }

    /// Every prefix of a well-formed numeric literal either parses or returns a
    /// `ParseError` — no input reaches `unwrap` on a `ParseFloatError`.
    #[test]
    fn no_numeric_prefix_panics() {
        for full in ["123", "1.5", "1.5e-7", "0.5E+12", ".25e3", "9.", "1e999"] {
            for end in 1..=full.len() {
                let src = &full[..end];
                let pool = ExprPool::new();
                let mut syms = HashMap::new();
                // The assertion is that this call returns at all.
                let _ = parse(src, &pool, &mut syms);
            }
        }
    }

    /// A literal past the `f64` exponent range is its value at 53 bits, not
    /// `inf` or `0`; one past MPFR's range too is refused, not `inf`.
    #[test]
    fn float_literal_past_f64_range_is_not_inf_or_zero() {
        let pool = ExprPool::new();
        for (lit, sign) in [("1e999999", 1), ("1e-999999", 1), ("0.5e-400", 1)] {
            let id = super::float_literal(&pool, lit).expect(lit);
            match pool.get(id) {
                ExprData::Float(f) => {
                    assert!(f.inner.is_finite(), "{lit} read as {}", f.inner);
                    assert!(!f.inner.is_zero(), "{lit} read as zero");
                    assert_eq!(f.inner.cmp0(), Some(sign.cmp(&0)));
                }
                other => panic!("{lit} parsed to {other:?}"),
            }
        }
        // Zero is still zero, and ordinary literals are the plain f64 node.
        assert_eq!(
            super::float_literal(&pool, "0e999999"),
            Some(pool.float(0.0, 53))
        );
        assert_eq!(
            super::float_literal(&pool, "1e300"),
            Some(pool.float(1e300, 53))
        );
        for lit in ["1e99999999999999", "1e-99999999999999"] {
            assert!(super::float_literal(&pool, lit).is_none(), "{lit}");
            let mut syms = HashMap::new();
            let err = parse(lit, &pool, &mut syms).unwrap_err();
            assert_eq!(err.code(), "E-PARSE-001", "{lit}: {err}");
        }
    }

    /// An integer literal past `i64` is an ordinary integer.  It used to be a
    /// `ParseError`, so the printed form of `10^20·x` did not read back.
    #[test]
    fn integer_literals_past_i64_parse_exactly() {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        for lit in [
            "9223372036854775808",
            "100000000000000000000",
            "51090942171709440000",
        ] {
            let id = parse(lit, &pool, &mut syms).expect("parses");
            let want: rug::Integer = lit.parse().unwrap();
            assert_eq!(id, pool.integer(want), "{lit}");
        }
        let id = parse("-9223372036854775808", &pool, &mut syms).expect("parses");
        let x = pool.display(id).to_string();
        assert!(x.contains("9223372036854775808"), "{x}");
    }
}

/// Audit C1 (parser half) and A8 (float round trip).
#[cfg(test)]
mod canonical_input_tests {
    use super::*;

    fn p(src: &str, pool: &ExprPool) -> Result<ExprId, ParseError> {
        parse(src, pool, &mut HashMap::new())
    }

    #[test]
    fn a_builtin_at_the_wrong_arity_is_a_parse_error() {
        let pool = ExprPool::new();
        for src in [
            "sin()",
            "sin(x, x)",
            "sqrt()",
            "exp()",
            "gamma(x, y, z)",
            "lambert_w(x, 1, 2)",
            "EllipticPi(x)",
            "EllipticPi(1,2,3,4)",
            "EllipticE(1,2,3)",
            "EllipticF(x)",
            "atan2(x)",
            "atan2()",
            "log(x, y, z)",
            "Si()",
            "dilog(x, y)",
        ] {
            let err = p(src, &pool).expect_err(src);
            assert_eq!(err.code(), "E-PARSE-002", "{src}: {err}");
            assert!(err.to_string().contains("takes"), "{src}: {err}");
        }
        // The right arities still parse.
        for src in [
            "sin(x)",
            "atan2(x, y)",
            "EllipticE(x)",
            "EllipticE(x, y)",
            "EllipticF(x, y)",
            "EllipticPi(x, y, z)",
        ] {
            p(src, &pool).unwrap_or_else(|e| panic!("{src}: {e}"));
        }
    }

    #[test]
    fn float_literal_precision_follows_the_digits() {
        assert_eq!(float_literal_prec("1.5"), 53);
        assert_eq!(float_literal_prec("1.0000000000000000"), 53);
        assert_eq!(float_literal_prec("0.00000000000000000000001"), 53);
        assert_eq!(float_literal_prec("1.2345678901234567e300"), 53);
        assert!(float_literal_prec("1.23456789012345678") > 53);
    }

    /// `str` of a float parses back to a float of at least the precision it
    /// was printed at, holding the same value to that precision, and prints
    /// the same text again.
    #[test]
    fn printed_floats_read_back_without_losing_digits() {
        let pool = ExprPool::new();
        for prec in [2u32, 24, 53, 64, 100, 113, 128, 200, 256, 1000] {
            for v in [0.1_f64, 1.0 / 3.0, 2.0, 1e-300, 6.02214076e23] {
                // A value that really uses `prec` bits: v + 2^-(prec-1)·v.
                let mut exact = rug::Float::with_val(prec, v);
                let bump = rug::Float::with_val(prec, &exact >> (prec - 1));
                exact += bump;
                let e = pool.intern(ExprData::Float(crate::kernel::BigFloat {
                    inner: exact.clone(),
                    prec,
                }));
                let s = pool.display(e).to_string();
                let back = p(&s, &pool).unwrap_or_else(|err| panic!("{s}: {err}"));
                let ExprData::Float(f) = pool.get(back) else {
                    panic!("{s} (prec {prec}) re-parsed as {:?}", pool.get(back));
                };
                if prec <= 53 {
                    // The f64 path: exact only for precisions an f64 holds.
                    assert_eq!(f.prec, 53);
                } else {
                    assert!(f.prec >= prec, "{s}: read at {} < {prec}", f.prec);
                    let err = rug::Float::with_val(f.prec + 8, &f.inner - &exact).abs();
                    let ulp = rug::Float::with_val(prec, &exact >> (prec - 1)).abs();
                    assert!(err <= ulp, "{s}: value moved by {err} at prec {prec}");
                    // Once read at the wider precision, the value is stable:
                    // printing and reading again changes nothing.
                    let s2 = pool.display(back).to_string();
                    assert_eq!(p(&s2, &pool).unwrap(), back, "{s2} is not a fixpoint");
                }
            }
        }
        // 53 bits round-trips structurally.
        let e = pool.float(0.1, 53);
        let s = pool.display(e).to_string();
        assert_eq!(p(&s, &pool).unwrap(), e);
        let z = pool.float(0.0, 53);
        assert_eq!(p(&pool.display(z).to_string(), &pool).unwrap(), z);
        // `super::float_literal` — the test fn above shadows it here.
        assert!(super::float_literal(&pool, "abc").is_none());
    }
}
