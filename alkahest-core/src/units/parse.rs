//! Unit-string parser.
//!
//! ```text
//! product  := factor ( ('*' | '·' | '/' | <whitespace>) factor )*
//! factor   := primary ( ('^' | '**') exponent | superscripts )?
//! primary  := NAME | NUMBER | '(' product ')'
//! exponent := ['+'|'-'] NUMBER | '(' ['+'|'-'] INT ['/' INT] ')'
//! ```
//!
//! `/` divides by the single factor that follows (`a/b*c = a·c/b`), so
//! `"mg*h/L"`, `"m/s^2"`, `"kg m s^-2"`, `"kg*m^2/(s^2*A)"` all read as
//! written. A name immediately followed by a digit (`m2`) is refused rather
//! than guessed at.

use super::table::{lookup, try_dec};
use super::{Dimension, Scale, Unit, UnitError};
use rug::Rational;

fn err(input: &str, message: impl Into<String>) -> UnitError {
    UnitError::Parse {
        input: input.to_string(),
        message: message.into(),
    }
}

/// The message for an unknown unit name; logarithmic scales get a pointer to
/// the explicit conversion functions, since they are not units one can
/// multiply (`10 dB + 10 dB` is not `20 dB`).
fn unknown_msg(name: &str) -> String {
    if matches!(
        name,
        "dB" | "dBm" | "dBW" | "dBi" | "dBV" | "dBu" | "Np" | "neper" | "bel"
    ) {
        format!(
            "`{name}` is a logarithmic scale, not a unit with a scale factor, so it takes no part in unit arithmetic; convert it explicitly with alkahest.units.db_to_ratio / np_to_ratio (and ratio_to_db / ratio_to_np)"
        )
    } else {
        format!("unknown unit `{name}`")
    }
}

fn is_name_char(c: char) -> bool {
    c.is_alphabetic() || c == '_' || c == '°' || c == '%'
}

fn superscript_digit(c: char) -> Option<u32> {
    Some(match c {
        '⁰' => 0,
        '¹' => 1,
        '²' => 2,
        '³' => 3,
        '⁴' => 4,
        '⁵' => 5,
        '⁶' => 6,
        '⁷' => 7,
        '⁸' => 8,
        '⁹' => 9,
        _ => return None,
    })
}

struct Parser<'a> {
    input: &'a str,
    chars: Vec<char>,
    pos: usize,
}

/// A numeric literal used as a unit factor (`1000*m`): dimensionless, scale =
/// its exact value, named by its spelling.
fn number_unit(text: &str, value: Rational) -> Unit {
    if value == 1 {
        return Unit::dimensionless();
    }
    Unit::atom(
        text,
        Scale {
            ratio: value,
            pi_power: Rational::new(),
        },
        Dimension::dimensionless(),
        None,
    )
}

/// Resolve one token — a unit name or a positive number literal.
pub(super) fn parse_atom(token: &str) -> Result<Unit, UnitError> {
    if token.starts_with(|c: char| c.is_ascii_digit()) {
        let v = parse_decimal(token).ok_or_else(|| err(token, "malformed number"))?;
        return Ok(number_unit(token, v));
    }
    lookup(token).ok_or_else(|| err(token, unknown_msg(token)))
}

fn parse_decimal(s: &str) -> Option<Rational> {
    try_dec(s)
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.pos += 1;
        }
    }

    fn at_factor_start(&self) -> bool {
        self.peek()
            .is_some_and(|c| is_name_char(c) || c.is_ascii_digit() || c == '(')
    }

    fn product(&mut self) -> Result<Unit, UnitError> {
        self.skip_ws();
        let mut acc = self.factor()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some('*') | Some('·') | Some('⋅') => {
                    self.pos += 1;
                    if self.peek() == Some('*') {
                        return Err(err(self.input, "`**` must follow a unit (exponent)"));
                    }
                    self.skip_ws();
                    let f = self.factor()?;
                    acc = acc.mul(&f)?;
                }
                Some('/') => {
                    self.pos += 1;
                    self.skip_ws();
                    let f = self.factor()?;
                    acc = acc.div(&f)?;
                }
                _ if self.at_factor_start() => {
                    let f = self.factor()?;
                    acc = acc.mul(&f)?;
                }
                _ => return Ok(acc),
            }
        }
    }

    fn factor(&mut self) -> Result<Unit, UnitError> {
        let base = self.primary()?;
        // exponent
        let save = self.pos;
        self.skip_ws();
        if self.peek() == Some('^') {
            self.pos += 1;
        } else if self.peek() == Some('*') && self.chars.get(self.pos + 1) == Some(&'*') {
            self.pos += 2;
        } else if self
            .peek()
            .is_some_and(|c| superscript_digit(c).is_some() || c == '⁻')
        {
            let neg = self.peek() == Some('⁻');
            if neg {
                self.pos += 1;
            }
            let mut v: u32 = 0;
            let mut any = false;
            while let Some(d) = self.peek().and_then(superscript_digit) {
                v = v.saturating_mul(10).saturating_add(d);
                any = true;
                self.pos += 1;
            }
            if !any {
                return Err(err(
                    self.input,
                    "`⁻` must be followed by a superscript digit",
                ));
            }
            let q = Rational::from(if neg { -(v as i64) } else { v as i64 });
            return base.pow(&q);
        } else {
            self.pos = save;
            return Ok(base);
        }
        self.skip_ws();
        let q = self.exponent()?;
        if base.is_affine() {
            return Err(UnitError::AffineUnit {
                unit: base.name(),
                context: "a power".to_string(),
            });
        }
        base.pow(&q)
    }

    fn exponent(&mut self) -> Result<Rational, UnitError> {
        let paren = self.peek() == Some('(');
        if paren {
            self.pos += 1;
            self.skip_ws();
        }
        let mut neg = false;
        if matches!(self.peek(), Some('-') | Some('+') | Some('−')) {
            neg = self.peek() != Some('+');
            self.pos += 1;
        }
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_digit() || c == '.') {
            self.pos += 1;
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        let mut q =
            parse_decimal(&text).ok_or_else(|| err(self.input, "expected a number after `^`"))?;
        if paren {
            self.skip_ws();
            if self.peek() == Some('/') {
                self.pos += 1;
                self.skip_ws();
                let s2 = self.pos;
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.pos += 1;
                }
                let t2: String = self.chars[s2..self.pos].iter().collect();
                let d = parse_decimal(&t2)
                    .filter(|d| *d != 0)
                    .ok_or_else(|| err(self.input, "expected a non-zero denominator"))?;
                q /= d;
            }
            self.skip_ws();
            if self.peek() != Some(')') {
                return Err(err(self.input, "unclosed `(` in exponent"));
            }
            self.pos += 1;
        } else if self.peek() == Some('/')
            && self
                .chars
                .get(self.pos + 1)
                .is_some_and(|c| c.is_ascii_digit())
        {
            return Err(err(
                self.input,
                "a fractional exponent must be parenthesised, e.g. m^(1/2)",
            ));
        }
        Ok(if neg { -q } else { q })
    }

    fn primary(&mut self) -> Result<Unit, UnitError> {
        match self.peek() {
            Some('(') => {
                self.pos += 1;
                let u = self.product()?;
                self.skip_ws();
                if self.peek() != Some(')') {
                    return Err(err(self.input, "unclosed `(`"));
                }
                self.pos += 1;
                Ok(u)
            }
            Some(c) if c.is_ascii_digit() => {
                let start = self.pos;
                while self.peek().is_some_and(|c| c.is_ascii_digit() || c == '.') {
                    self.pos += 1;
                }
                if matches!(self.peek(), Some('e') | Some('E')) {
                    let next = self.chars.get(self.pos + 1).copied();
                    let next2 = self.chars.get(self.pos + 2).copied();
                    let is_exp = next.is_some_and(|c| c.is_ascii_digit())
                        || (matches!(next, Some('-') | Some('+'))
                            && next2.is_some_and(|c| c.is_ascii_digit()));
                    if is_exp {
                        self.pos += 2;
                        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                            self.pos += 1;
                        }
                    }
                }
                let text: String = self.chars[start..self.pos].iter().collect();
                if self.peek().is_some_and(is_name_char) {
                    return Err(err(
                        self.input,
                        format!("`{text}` runs into a unit name; separate them with `*`"),
                    ));
                }
                let v = parse_decimal(&text)
                    .ok_or_else(|| err(self.input, format!("malformed number `{text}`")))?;
                if v == 0 {
                    return Err(err(self.input, "a unit factor cannot be 0"));
                }
                Ok(number_unit(&text, v))
            }
            Some(c) if is_name_char(c) => {
                let start = self.pos;
                while self.peek().is_some_and(is_name_char) {
                    self.pos += 1;
                }
                let name: String = self.chars[start..self.pos].iter().collect();
                if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    return Err(err(
                        self.input,
                        format!("`{name}` is followed by a digit; write an exponent as `{name}^2`"),
                    ));
                }
                lookup(&name).ok_or_else(|| err(self.input, unknown_msg(&name)))
            }
            Some(c) => Err(err(self.input, format!("unexpected `{c}`"))),
            None => Err(err(self.input, "expected a unit")),
        }
    }
}

/// Parse a unit string such as `"mg*h/L"`, `"m/s^2"`, `"kg m s^-2"`,
/// `"m^(1/2)"`, `"µg/mL"`, `"°C"` or `"1"`.
///
/// Refuses malformed input and unknown names with `E-UNIT-002`, and an affine
/// temperature unit inside a compound unit with `E-UNIT-003`.
pub fn parse_unit(s: &str) -> Result<Unit, UnitError> {
    let mut p = Parser {
        input: s,
        chars: s.chars().collect(),
        pos: 0,
    };
    p.skip_ws();
    if p.peek().is_none() {
        return Err(err(s, "empty unit string (use \"1\" for a pure number)"));
    }
    let u = p.product()?;
    p.skip_ws();
    if let Some(c) = p.peek() {
        return Err(err(s, format!("unexpected `{c}`")));
    }
    Ok(u)
}
