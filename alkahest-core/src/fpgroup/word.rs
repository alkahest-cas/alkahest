//! Free groups and their elements: [`Word`] as a freely reduced sequence of
//! signed generator indices, and [`FreeGroup`] as the named alphabet those
//! letters index into.
//!
//! # Letters are signed and 1-based
//!
//! A letter is a non-zero `i32`: `+k` is the `k`-th generator (`k - 1` in
//! 0-based generator numbering) and `-k` its inverse. Zero is not a letter —
//! the identity is the *empty* word. The 1-based encoding is the one GAP and
//! the literature use for `AssocWord`, and it is the only encoding in which
//! negating a letter inverts it.
//!
//! # Every `Word` is freely reduced
//!
//! There is no constructor that returns an unreduced word: `x x⁻¹` cancels on
//! the way in, and [`Word::times`], [`Word::inverse`] and [`Word::pow`]
//! preserve the invariant. Two words are therefore `==` exactly when they are
//! equal in the free group, which is what makes `Word` usable as a `HashMap`
//! key and what makes the word problem in a *free* group decidable here. It
//! says nothing about equality in a quotient `⟨X | R⟩`, which is undecidable in
//! general and is what Todd–Coxeter enumeration approximates.

use super::error::FpGroupError;
use std::fmt;

/// Largest number of generators a presentation may have.
///
/// The coset table is `2 · rank` machine words wide per coset, and
/// Reidemeister–Schreier multiplies the generator count by the index, so a
/// four-figure rank is a memory decision rather than a mathematical one.
pub const MAX_FREE_RANK: usize = 1024;

/// An element of a free group: a freely reduced sequence of signed, 1-based
/// generator indices.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Word {
    letters: Vec<i32>,
}

/// Free reduction: cancel adjacent inverse pairs, left to right.
///
/// One pass with a stack suffices, and is idempotent — a proptest asserts that.
fn free_reduce(letters: &[i32]) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::with_capacity(letters.len());
    for &l in letters {
        if out.last().is_some_and(|&last| last == -l) {
            out.pop();
        } else {
            out.push(l);
        }
    }
    out
}

impl Word {
    /// The identity: the empty word.
    pub fn identity() -> Word {
        Word {
            letters: Vec::new(),
        }
    }

    /// A word from raw letters, freely reduced.
    ///
    /// Letters are checked for being non-zero but **not** against any rank; use
    /// [`FreeGroup::word`] when the rank is known.
    pub fn from_letters(letters: &[i32]) -> Result<Word, FpGroupError> {
        for &l in letters {
            if l == 0 {
                return Err(FpGroupError::InvalidGenerator { letter: 0, rank: 0 });
            }
        }
        Ok(Word {
            letters: free_reduce(letters),
        })
    }

    /// The `index`-th generator (0-based), as a one-letter word.
    pub fn generator(index: usize) -> Result<Word, FpGroupError> {
        let letter = i32::try_from(index + 1).map_err(|_| FpGroupError::InvalidPresentation {
            detail: format!("generator index {index} does not fit a 32-bit letter"),
        })?;
        Ok(Word {
            letters: vec![letter],
        })
    }

    /// The commutator `[a, b] = a b a⁻¹ b⁻¹`, freely reduced.
    pub fn commutator(a: &Word, b: &Word) -> Word {
        a.times(b).times(&a.inverse()).times(&b.inverse())
    }

    /// The letters, freely reduced.
    pub fn letters(&self) -> &[i32] {
        &self.letters
    }

    /// Length of the reduced word.
    pub fn len(&self) -> usize {
        self.letters.len()
    }

    /// Is this the identity?
    pub fn is_empty(&self) -> bool {
        self.letters.is_empty()
    }

    /// The formal inverse: reverse the letters and negate each.
    pub fn inverse(&self) -> Word {
        Word {
            letters: self.letters.iter().rev().map(|l| -l).collect(),
        }
    }

    /// Product `self · other`, freely reduced.
    pub fn times(&self, other: &Word) -> Word {
        let mut letters = self.letters.clone();
        for &l in &other.letters {
            if letters.last().is_some_and(|&last| last == -l) {
                letters.pop();
            } else {
                letters.push(l);
            }
        }
        Word { letters }
    }

    /// `self^exponent`, for any sign of exponent.
    ///
    /// Linear in the length of the result, not quadratic: repeated `times`
    /// would copy the accumulator on every step, and the parser admits
    /// exponents up to five figures.
    pub fn pow(&self, exponent: i32) -> Word {
        if exponent == 0 || self.is_empty() {
            return Word::identity();
        }
        let base = if exponent < 0 {
            self.inverse()
        } else {
            self.clone()
        };
        let repeats = exponent.unsigned_abs() as usize;
        let mut letters: Vec<i32> = Vec::with_capacity(base.len().saturating_mul(repeats));
        for _ in 0..repeats {
            for &l in base.letters() {
                if letters.last().is_some_and(|&last| last == -l) {
                    letters.pop();
                } else {
                    letters.push(l);
                }
            }
        }
        Word { letters }
    }

    /// The largest 0-based generator index occurring, or `None` for the
    /// identity. Used to check a word against a presentation's rank.
    pub fn max_generator(&self) -> Option<usize> {
        self.letters
            .iter()
            .map(|l| l.unsigned_abs() as usize - 1)
            .max()
    }

    /// Exponent sum of each generator — the image of this word in the
    /// abelianisation of the free group, `ℤ^rank`.
    ///
    /// This is exactly one row of the relation matrix whose Smith normal form
    /// gives `G/[G, G]`.
    pub fn exponent_sums(&self, rank: usize) -> Result<Vec<i64>, FpGroupError> {
        let mut sums = vec![0i64; rank];
        for &l in &self.letters {
            let g = l.unsigned_abs() as usize - 1;
            if g >= rank {
                return Err(FpGroupError::InvalidGenerator { letter: l, rank });
            }
            let delta = if l > 0 { 1 } else { -1 };
            sums[g] = sums[g]
                .checked_add(delta)
                .ok_or_else(|| FpGroupError::Internal {
                    detail: "exponent sum overflowed i64".into(),
                })?;
        }
        Ok(sums)
    }

    /// Render using the supplied generator names, e.g. `a^2*b^-1`.
    ///
    /// Names shorter than the word's alphabet fall back to `x<k>`.
    pub fn to_string_with(&self, names: &[String]) -> String {
        if self.letters.is_empty() {
            return "1".to_string();
        }
        let name_of = |g: usize| -> String {
            names
                .get(g)
                .cloned()
                .unwrap_or_else(|| format!("x{}", g + 1))
        };
        let mut parts: Vec<String> = Vec::new();
        let mut i = 0;
        while i < self.letters.len() {
            let l = self.letters[i];
            let mut run = 1;
            while i + run < self.letters.len() && self.letters[i + run] == l {
                run += 1;
            }
            let g = l.unsigned_abs() as usize - 1;
            let exp = if l > 0 { run as i64 } else { -(run as i64) };
            if exp == 1 {
                parts.push(name_of(g));
            } else {
                parts.push(format!("{}^{}", name_of(g), exp));
            }
            i += run;
        }
        parts.join("*")
    }
}

impl fmt::Display for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_with(&[]))
    }
}

/// A free group of finite rank, with names for its generators.
///
/// The names exist for input and output only: words are stored as signed
/// indices, and two free groups of the same rank with different names generate
/// the same words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreeGroup {
    names: Vec<String>,
}

impl FreeGroup {
    /// A free group of rank `rank`, with generators named `x1 … x<rank>`.
    pub fn new(rank: usize) -> Result<FreeGroup, FpGroupError> {
        if rank > MAX_FREE_RANK {
            return Err(FpGroupError::InvalidPresentation {
                detail: format!("rank {rank} exceeds MAX_FREE_RANK ({MAX_FREE_RANK})"),
            });
        }
        Ok(FreeGroup {
            names: (1..=rank).map(|i| format!("x{i}")).collect(),
        })
    }

    /// A free group with the given generator names.
    ///
    /// Names must be non-empty, pairwise distinct, and must not contain any of
    /// the characters the word syntax reserves (`*`, `^`, `(`, `)`, whitespace,
    /// digits at the start, `-`, `+`).
    pub fn with_names<S: AsRef<str>>(names: &[S]) -> Result<FreeGroup, FpGroupError> {
        if names.len() > MAX_FREE_RANK {
            return Err(FpGroupError::InvalidPresentation {
                detail: format!(
                    "rank {} exceeds MAX_FREE_RANK ({MAX_FREE_RANK})",
                    names.len()
                ),
            });
        }
        let mut out: Vec<String> = Vec::with_capacity(names.len());
        for (i, n) in names.iter().enumerate() {
            let n = n.as_ref();
            if n.is_empty() {
                return Err(FpGroupError::InvalidPresentation {
                    detail: format!("generator {i} has an empty name"),
                });
            }
            let bad = |c: char| {
                c.is_whitespace() || matches!(c, '*' | '^' | '(' | ')' | '-' | '+' | ',' | '.')
            };
            if let Some(c) = n.chars().find(|&c| bad(c)) {
                return Err(FpGroupError::InvalidPresentation {
                    detail: format!("generator name {n:?} contains the reserved character {c:?}"),
                });
            }
            if n.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                return Err(FpGroupError::InvalidPresentation {
                    detail: format!("generator name {n:?} starts with a digit"),
                });
            }
            if out.iter().any(|m| m == n) {
                return Err(FpGroupError::InvalidPresentation {
                    detail: format!("generator name {n:?} is used twice"),
                });
            }
            out.push(n.to_string());
        }
        Ok(FreeGroup { names: out })
    }

    /// The number of generators.
    pub fn rank(&self) -> usize {
        self.names.len()
    }

    /// The generator names, in order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The `index`-th generator (0-based) as a word.
    pub fn generator(&self, index: usize) -> Result<Word, FpGroupError> {
        if index >= self.rank() {
            return Err(FpGroupError::InvalidGenerator {
                letter: index as i32 + 1,
                rank: self.rank(),
            });
        }
        Word::generator(index)
    }

    /// A word from signed 1-based letters, checked against the rank and freely
    /// reduced.
    pub fn word(&self, letters: &[i32]) -> Result<Word, FpGroupError> {
        let rank = self.rank();
        for &l in letters {
            if l == 0 || l.unsigned_abs() as usize > rank {
                return Err(FpGroupError::InvalidGenerator { letter: l, rank });
            }
        }
        Ok(Word {
            letters: free_reduce(letters),
        })
    }

    /// Render a word with this group's names.
    pub fn format(&self, w: &Word) -> String {
        w.to_string_with(&self.names)
    }

    /// Parse a word from text.
    ///
    /// The syntax is a product of powers: generator names and parenthesised
    /// sub-words, each optionally raised to an integer power, joined by `*` (or
    /// by juxtaposition). `1` is the identity. Examples, for generators
    /// `a, b`:
    ///
    /// ```text
    ///     a^2              (ab)^5            a*b^-1*a^-1*b
    ///     abAB  ← NOT accepted: case does not mean inversion here
    /// ```
    ///
    /// Names are matched longest-first, so an alphabet containing both `a` and
    /// `ab` is unambiguous.
    pub fn parse(&self, input: &str) -> Result<Word, FpGroupError> {
        let bytes = input.as_bytes();
        let mut pos = 0usize;
        let w = self.parse_word(bytes, &mut pos, 0)?;
        skip_ws(bytes, &mut pos);
        if pos != bytes.len() {
            return Err(FpGroupError::WordSyntax {
                position: pos,
                detail: format!(
                    "unexpected {:?}",
                    input[pos..].chars().next().unwrap_or(' ')
                ),
            });
        }
        Ok(w)
    }

    fn parse_word(
        &self,
        bytes: &[u8],
        pos: &mut usize,
        depth: usize,
    ) -> Result<Word, FpGroupError> {
        if depth > 64 {
            return Err(FpGroupError::WordSyntax {
                position: *pos,
                detail: "parentheses nested more than 64 deep".into(),
            });
        }
        let mut acc = Word::identity();
        loop {
            skip_ws(bytes, pos);
            if *pos >= bytes.len() || bytes[*pos] == b')' {
                return Ok(acc);
            }
            if bytes[*pos] == b'*' {
                *pos += 1;
                skip_ws(bytes, pos);
                if *pos >= bytes.len() || bytes[*pos] == b')' || bytes[*pos] == b'*' {
                    return Err(FpGroupError::WordSyntax {
                        position: *pos,
                        detail: "`*` must be followed by a generator or a sub-word".into(),
                    });
                }
                continue;
            }
            let factor = self.parse_factor(bytes, pos, depth)?;
            acc = acc.times(&factor);
        }
    }

    fn parse_factor(
        &self,
        bytes: &[u8],
        pos: &mut usize,
        depth: usize,
    ) -> Result<Word, FpGroupError> {
        skip_ws(bytes, pos);
        let base = if *pos < bytes.len() && bytes[*pos] == b'(' {
            *pos += 1;
            let inner = self.parse_word(bytes, pos, depth + 1)?;
            skip_ws(bytes, pos);
            if *pos >= bytes.len() || bytes[*pos] != b')' {
                return Err(FpGroupError::WordSyntax {
                    position: *pos,
                    detail: "unclosed `(`".into(),
                });
            }
            *pos += 1;
            inner
        } else if *pos < bytes.len() && bytes[*pos] == b'1' {
            *pos += 1;
            Word::identity()
        } else {
            // Longest matching generator name.
            let rest = &bytes[*pos..];
            let mut best: Option<usize> = None;
            for (i, n) in self.names.iter().enumerate() {
                let nb = n.as_bytes();
                let longer_than_best = match best {
                    None => true,
                    Some(b) => self.names[b].len() < nb.len(),
                };
                if rest.len() >= nb.len() && &rest[..nb.len()] == nb && longer_than_best {
                    best = Some(i);
                }
            }
            match best {
                Some(i) => {
                    *pos += self.names[i].len();
                    Word::generator(i)?
                }
                None => {
                    return Err(FpGroupError::WordSyntax {
                        position: *pos,
                        detail: "expected a generator name, `1`, or `(`".into(),
                    })
                }
            }
        };
        skip_ws(bytes, pos);
        if *pos < bytes.len() && bytes[*pos] == b'^' {
            *pos += 1;
            let e = parse_exponent(bytes, pos)?;
            return Ok(base.pow(e));
        }
        Ok(base)
    }
}

fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

/// Upper bound on a literal exponent. `a^1000000` is a million letters; the
/// cap is there so that a typo cannot ask for a gigabyte of `Vec<i32>`.
const MAX_LITERAL_EXPONENT: i64 = 100_000;

fn parse_exponent(bytes: &[u8], pos: &mut usize) -> Result<i32, FpGroupError> {
    skip_ws(bytes, pos);
    let start = *pos;
    let mut sign = 1i64;
    if *pos < bytes.len() && (bytes[*pos] == b'-' || bytes[*pos] == b'+') {
        if bytes[*pos] == b'-' {
            sign = -1;
        }
        *pos += 1;
    }
    let digits_start = *pos;
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if *pos == digits_start {
        return Err(FpGroupError::WordSyntax {
            position: start,
            detail: "`^` must be followed by an integer exponent".into(),
        });
    }
    let text =
        std::str::from_utf8(&bytes[digits_start..*pos]).map_err(|_| FpGroupError::WordSyntax {
            position: start,
            detail: "exponent is not valid UTF-8".into(),
        })?;
    let magnitude: i64 = text.parse().map_err(|_| FpGroupError::WordSyntax {
        position: start,
        detail: format!("exponent {text} does not fit"),
    })?;
    if magnitude > MAX_LITERAL_EXPONENT {
        return Err(FpGroupError::WordSyntax {
            position: start,
            detail: format!("exponent {magnitude} is above the literal cap {MAX_LITERAL_EXPONENT}"),
        });
    }
    Ok((sign * magnitude) as i32)
}
