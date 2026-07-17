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
use crate::kernel::{Domain, ExprId, ExprPool};

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
    code_idx: u8, // 1 = E-PARSE-001, 2 = E-PARSE-002, 3 = E-PARSE-003
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
            _ => "E-PARSE-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self.code_idx {
            1 => Some("only ASCII arithmetic expressions are supported"),
            2 => Some("check parentheses and operator placement"),
            _ => Some("use a known function: sin, cos, tan, sec, csc, cot, sinh, cosh, tanh, sech, csch, coth, asin, acos, atan, asinh, acosh, atanh, atan2, exp, log, sqrt, abs, sign, floor, ceil, round, erf, erfc, gamma, lambert_w"),
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
                pos += 1;
                if pos < n && (bytes[pos] == b'+' || bytes[pos] == b'-') {
                    pos += 1;
                }
                while pos < n && bytes[pos].is_ascii_digit() {
                    pos += 1;
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
    // Reciprocal trig / hyperbolic functions.  These are *desugared* in
    // `parse_funcall` to their elementary reciprocal definitions (e.g.
    // `sec(x) → cos(x)^(-1)`); no `sec`/`csc`/… node ever enters the pool.
    "sec",
    "csc",
    "cot",
    "sech",
    "csch",
    "coth",
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

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    pool: &'a ExprPool,
    symbols: &'a mut HashMap<String, ExprId>,
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
                    Ok(pool.float(s.parse::<f64>().unwrap(), 53))
                } else {
                    let n: i64 = s.parse().map_err(|_| {
                        ParseError::lex(
                            format!("integer literal out of range: {s}"),
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
                // -x  →  (-1) * x
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

        Ok(self.pool.func(name, args))
    }
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
/// `floor`, `ceil`, `round`, `erf`, `erfc`, `gamma`.
///
/// The reciprocal trig/hyperbolic functions `sec`, `csc`, `cot`, `sech`,
/// `csch`, and `coth` are also accepted; they are desugared at parse time to
/// their elementary reciprocal definitions (`sec(x) → cos(x)^(-1)`,
/// `csc(x) → sin(x)^(-1)`, `cot(x) → tan(x)^(-1)`, and the hyperbolic
/// analogues), so no dedicated node for them exists in the pool.
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
}
