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
}
