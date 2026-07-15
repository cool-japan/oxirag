//! A recursive-descent parser for this module's regex dialect, producing a
//! [`RegexAst`].
//!
//! The crate has no `regex` dependency and is not getting one; the dialect below
//! is the whole of what this module understands, and it is parsed by hand — the
//! same approach `program_of_thought` takes to its arithmetic grammar.
//!
//! # Grammar
//!
//! ```text
//! pattern     := alternation EOF
//! alternation := concat ('|' concat)*
//! concat      := repetition*
//! repetition  := atom quantifier?
//! quantifier  := '*' | '+' | '?' | '{' bounds '}'
//! bounds      := digits | digits ',' | digits ',' digits
//! atom        := '(' '?:'? alternation ')'
//!              | '[' '^'? class_item+ ']'
//!              | '.'
//!              | escape
//!              | literal
//! class_item  := class_atom ('-' class_atom)?
//! class_atom  := escape | any byte except ']'
//! escape      := '\' ('d' | 'D' | 'w' | 'W' | 's' | 'S'
//!                    | 'n' | 'r' | 't' | 'f' | 'v' | '0'
//!                    | 'x' hex hex
//!                    | any ASCII punctuation byte)
//! literal     := any byte except  \ . * + ? ( ) [ ] { } | ^ $
//! ```
//!
//! # Deliberate divergences from PCRE
//!
//! * **Patterns are implicitly anchored.** The automaton must consume the entire
//!   output, so `^` and `$` would be redundant everywhere and wrong somewhere.
//!   They are *rejected* rather than silently ignored; write `\^` and `\$` for the
//!   literal characters.
//! * **`.` is a byte, not a character**, and means "any byte except `0x0a`". This
//!   is a byte-level engine: see the [module documentation](crate::constrained_decoding)
//!   for why, and use an explicit class if you mean something narrower.
//! * **No laziness, no backreferences, no look-around, no capture.** A recogniser
//!   that only ever asks "is this whole string in the language?" has no use for any
//!   of them, and three of the four are not regular. `(?:...)` is accepted and is
//!   exactly `(...)`, since grouping never captures here.
//! * **Stacked quantifiers are rejected.** `a**` is an error ("nothing to repeat"),
//!   not a synonym for `a*`. Write `(a*)*` if that is really what you want.
//! * **`{` must begin a quantifier.** It is not silently demoted to a literal brace
//!   when the bounds fail to parse; write `\{`.

use crate::constrained_decoding::types::{
    ByteRange, CharClass, ConstrainedDecodingError, ConstrainedDecodingResult, MAX_REPEAT,
    RegexAst, RegexRepeat,
};

/// The default limit on group nesting.
///
/// Parsing and compilation are both recursive, so an unbounded nesting depth is
/// an unbounded stack depth. The limit turns a stack overflow — which is a crash,
/// not an error — into [`ConstrainedDecodingError::NestingTooDeep`].
pub const MAX_REGEX_DEPTH: u32 = 64;

/// Parse `pattern` into a [`RegexAst`].
///
/// # Errors
///
/// Returns [`ConstrainedDecodingError::RegexSyntax`] for a malformed pattern,
/// [`ConstrainedDecodingError::InvalidRepeat`] for a `{m,n}` whose bounds are
/// reversed or larger than [`MAX_REPEAT`], and
/// [`ConstrainedDecodingError::NestingTooDeep`] when groups nest more than
/// [`MAX_REGEX_DEPTH`] deep.
pub fn parse_regex(pattern: &str) -> ConstrainedDecodingResult<RegexAst> {
    let mut parser = RegexParser {
        input: pattern.as_bytes(),
        pos: 0,
        depth: 0,
    };
    let ast = parser.parse_alternation()?;
    if parser.pos != parser.input.len() {
        return Err(parser.error(format!(
            "unexpected `{}`",
            char::from(parser.input[parser.pos])
        )));
    }
    Ok(ast)
}

/// What a backslash escape denotes: either one specific byte (which may be a range
/// endpoint inside a class) or a whole set of them (which may not).
enum EscapeValue {
    Byte(u8),
    Class(CharClass),
}

struct RegexParser<'a> {
    input: &'a [u8],
    pos: usize,
    depth: u32,
}

impl RegexParser<'_> {
    fn error(&self, reason: impl Into<String>) -> ConstrainedDecodingError {
        ConstrainedDecodingError::RegexSyntax {
            position: self.pos,
            reason: reason.into(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let byte = self.peek();
        if byte.is_some() {
            self.pos += 1;
        }
        byte
    }

    fn eat(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// `alternation := concat ('|' concat)*`
    fn parse_alternation(&mut self) -> ConstrainedDecodingResult<RegexAst> {
        let mut branches = vec![self.parse_concat()?];
        while self.eat(b'|') {
            branches.push(self.parse_concat()?);
        }
        if branches.len() == 1 {
            // `Vec::pop` cannot fail here; the `while` never removes anything.
            return branches
                .pop()
                .ok_or_else(|| self.error("internal: empty alternation"));
        }
        Ok(RegexAst::Alternate(branches))
    }

    /// `concat := repetition*`
    ///
    /// Terminates at `|`, `)` or end of input, so an *empty* concatenation is
    /// legal and means the empty string: `(a|)` matches `a` and `""`.
    fn parse_concat(&mut self) -> ConstrainedDecodingResult<RegexAst> {
        let mut items: Vec<RegexAst> = Vec::new();
        while let Some(byte) = self.peek() {
            if byte == b'|' || byte == b')' {
                break;
            }
            items.push(self.parse_repetition()?);
        }
        match items.len() {
            0 => Ok(RegexAst::Empty),
            1 => items
                .pop()
                .ok_or_else(|| self.error("internal: empty concatenation")),
            _ => Ok(RegexAst::Concat(items)),
        }
    }

    /// `repetition := atom quantifier?`
    fn parse_repetition(&mut self) -> ConstrainedDecodingResult<RegexAst> {
        let atom = self.parse_atom()?;
        let Some(repeat) = self.parse_quantifier()? else {
            return Ok(atom);
        };
        repeat.validate()?;
        Ok(RegexAst::repeat(atom, repeat))
    }

    /// `quantifier := '*' | '+' | '?' | '{' bounds '}'`, or nothing.
    fn parse_quantifier(&mut self) -> ConstrainedDecodingResult<Option<RegexRepeat>> {
        match self.peek() {
            Some(b'*') => {
                self.pos += 1;
                Ok(Some(RegexRepeat::star()))
            }
            Some(b'+') => {
                self.pos += 1;
                Ok(Some(RegexRepeat::plus()))
            }
            Some(b'?') => {
                self.pos += 1;
                Ok(Some(RegexRepeat::optional()))
            }
            Some(b'{') => {
                self.pos += 1;
                self.parse_bounds().map(Some)
            }
            _ => Ok(None),
        }
    }

    /// `bounds := digits | digits ',' | digits ',' digits`, having consumed `{`.
    fn parse_bounds(&mut self) -> ConstrainedDecodingResult<RegexRepeat> {
        let min = self.parse_count()?;
        let repeat = if self.eat(b',') {
            if self.peek() == Some(b'}') {
                RegexRepeat::at_least(min)
            } else {
                RegexRepeat::between(min, self.parse_count()?)
            }
        } else {
            RegexRepeat::exactly(min)
        };
        if !self.eat(b'}') {
            return Err(self.error("expected `}` to close a `{m,n}` repetition"));
        }
        Ok(repeat)
    }

    /// A decimal count inside `{...}`.
    fn parse_count(&mut self) -> ConstrainedDecodingResult<u32> {
        let start = self.pos;
        let mut value: u32 = 0;
        while let Some(byte) = self.peek() {
            if !byte.is_ascii_digit() {
                break;
            }
            self.pos += 1;
            value = value
                .checked_mul(10)
                .and_then(|scaled| scaled.checked_add(u32::from(byte - b'0')))
                .filter(|candidate| *candidate <= MAX_REPEAT)
                .ok_or(ConstrainedDecodingError::InvalidRepeat {
                    min: MAX_REPEAT,
                    max: None,
                    reason: "repetition count exceeds the repetition limit",
                })?;
        }
        if self.pos == start {
            return Err(self.error("expected a decimal count inside `{...}`"));
        }
        Ok(value)
    }

    /// `atom := group | class | '.' | escape | literal`
    fn parse_atom(&mut self) -> ConstrainedDecodingResult<RegexAst> {
        let Some(byte) = self.peek() else {
            return Err(self.error("unexpected end of pattern"));
        };
        match byte {
            b'(' => self.parse_group(),
            b'[' => {
                self.pos += 1;
                Ok(RegexAst::Class(self.parse_class()?))
            }
            b'.' => {
                self.pos += 1;
                // Any byte but a newline. Bytes, not characters: a multi-byte
                // codepoint is several `.`, which is the honest thing for an engine
                // whose transitions are bytes.
                Ok(RegexAst::Class(
                    CharClass::any_byte().difference(&CharClass::single(b'\n')),
                ))
            }
            b'\\' => {
                self.pos += 1;
                match self.parse_escape()? {
                    EscapeValue::Byte(value) => Ok(RegexAst::literal_byte(value)),
                    EscapeValue::Class(class) => Ok(RegexAst::Class(class)),
                }
            }
            b'*' | b'+' | b'?' => {
                Err(self.error(format!("nothing to repeat before `{}`", char::from(byte))))
            }
            b'{' => {
                Err(self
                    .error("nothing to repeat before `{`; write `\\{` for a literal opening brace"))
            }
            b'}' | b']' => Err(self.error(format!(
                "unmatched `{}`; write `\\{}` for the literal character",
                char::from(byte),
                char::from(byte)
            ))),
            b'^' | b'$' => Err(self.error(format!(
                "`{}` is not supported: patterns are implicitly anchored, so an anchor \
                 is redundant; write `\\{}` for the literal character",
                char::from(byte),
                char::from(byte)
            ))),
            other => {
                self.pos += 1;
                Ok(RegexAst::literal_byte(other))
            }
        }
    }

    /// `'(' '?:'? alternation ')'`
    fn parse_group(&mut self) -> ConstrainedDecodingResult<RegexAst> {
        self.pos += 1; // `(`
        self.depth += 1;
        if self.depth > MAX_REGEX_DEPTH {
            return Err(ConstrainedDecodingError::NestingTooDeep {
                limit: MAX_REGEX_DEPTH,
            });
        }
        // `(?:` is accepted and means nothing: grouping never captures here, so
        // the non-capturing form is already the only form there is.
        if self.peek() == Some(b'?') {
            if self.input.get(self.pos + 1) == Some(&b':') {
                self.pos += 2;
            } else {
                return Err(self.error(
                    "`(?` introduces an extension group; only the non-capturing `(?:` is supported",
                ));
            }
        }
        let inner = self.parse_alternation()?;
        if !self.eat(b')') {
            return Err(self.error("expected `)` to close a group"));
        }
        self.depth -= 1;
        Ok(inner)
    }

    /// A bracketed class, having consumed the opening `[`.
    fn parse_class(&mut self) -> ConstrainedDecodingResult<CharClass> {
        let negated = self.eat(b'^');
        let mut class = CharClass::empty();
        let mut saw_item = false;

        loop {
            let Some(byte) = self.peek() else {
                return Err(self.error("unterminated character class: expected `]`"));
            };
            // A `]` in the first position is the literal character, per POSIX:
            // `[]]` is the class containing `]`.
            if byte == b']' && saw_item {
                self.pos += 1;
                break;
            }
            class = class.union(&self.parse_class_item()?);
            saw_item = true;
        }

        Ok(if negated { class.negate() } else { class })
    }

    /// `class_item := class_atom ('-' class_atom)?`
    fn parse_class_item(&mut self) -> ConstrainedDecodingResult<CharClass> {
        let low = self.parse_class_atom()?;
        // `-` is a literal when it is the last character of the class (`[a-]`) or
        // when the left-hand side was a multi-byte escape (`[\d-]`).
        let is_range = self.peek() == Some(b'-')
            && self
                .input
                .get(self.pos + 1)
                .is_some_and(|next| *next != b']');
        match (low, is_range) {
            (EscapeValue::Class(class), false) => Ok(class),
            (EscapeValue::Byte(value), false) => Ok(CharClass::single(value)),
            (EscapeValue::Class(_), true) => Err(self.error(
                "a multi-byte escape such as `\\d` cannot be an endpoint of a `[a-z]` range",
            )),
            (EscapeValue::Byte(start), true) => {
                self.pos += 1; // `-`
                let EscapeValue::Byte(end) = self.parse_class_atom()? else {
                    return Err(self.error(
                        "a multi-byte escape such as `\\d` cannot be an endpoint of a `[a-z]` range",
                    ));
                };
                if start > end {
                    return Err(self.error(format!(
                        "character range `{}-{}` is reversed",
                        char::from(start),
                        char::from(end)
                    )));
                }
                CharClass::range(start, end)
            }
        }
    }

    /// `class_atom := escape | any byte except ']'`
    fn parse_class_atom(&mut self) -> ConstrainedDecodingResult<EscapeValue> {
        match self.advance() {
            Some(b'\\') => self.parse_escape(),
            Some(byte) => Ok(EscapeValue::Byte(byte)),
            None => Err(self.error("unterminated character class: expected `]`")),
        }
    }

    /// An escape sequence, having consumed the `\`.
    fn parse_escape(&mut self) -> ConstrainedDecodingResult<EscapeValue> {
        let Some(byte) = self.advance() else {
            return Err(self.error("pattern ends with a trailing `\\`"));
        };
        let value = match byte {
            b'd' => EscapeValue::Class(digit_class()),
            b'D' => EscapeValue::Class(digit_class().negate()),
            b'w' => EscapeValue::Class(word_class()),
            b'W' => EscapeValue::Class(word_class().negate()),
            b's' => EscapeValue::Class(space_class()),
            b'S' => EscapeValue::Class(space_class().negate()),
            b'n' => EscapeValue::Byte(b'\n'),
            b'r' => EscapeValue::Byte(b'\r'),
            b't' => EscapeValue::Byte(b'\t'),
            b'f' => EscapeValue::Byte(0x0c),
            b'v' => EscapeValue::Byte(0x0b),
            b'0' => EscapeValue::Byte(0x00),
            b'x' => EscapeValue::Byte(self.parse_hex_byte()?),
            other if other.is_ascii_punctuation() => EscapeValue::Byte(other),
            other => {
                return Err(self.error(format!("unknown escape `\\{}`", char::from(other))));
            }
        };
        Ok(value)
    }

    /// `hex hex`, having consumed `\x`.
    fn parse_hex_byte(&mut self) -> ConstrainedDecodingResult<u8> {
        let high = self.parse_hex_digit()?;
        let low = self.parse_hex_digit()?;
        Ok((high << 4) | low)
    }

    fn parse_hex_digit(&mut self) -> ConstrainedDecodingResult<u8> {
        match self.advance() {
            Some(byte) => char::from(byte)
                .to_digit(16)
                .map(
                    #[allow(clippy::cast_possible_truncation)] // A hex digit is `0..=15`.
                    |digit| digit as u8,
                )
                .ok_or_else(|| ConstrainedDecodingError::RegexSyntax {
                    position: self.pos - 1,
                    reason: format!(
                        "`\\x` needs two hexadecimal digits; found `{}`",
                        char::from(byte)
                    ),
                }),
            None => Err(self.error("`\\x` needs two hexadecimal digits")),
        }
    }
}

/// `\d` — the ASCII decimal digits.
#[must_use]
pub fn digit_class() -> CharClass {
    CharClass::from_ranges(vec![ByteRange {
        start: b'0',
        end: b'9',
    }])
}

/// `\w` — the ASCII word bytes: letters, digits and `_`.
#[must_use]
pub fn word_class() -> CharClass {
    CharClass::from_ranges(vec![
        ByteRange {
            start: b'0',
            end: b'9',
        },
        ByteRange {
            start: b'A',
            end: b'Z',
        },
        ByteRange {
            start: b'_',
            end: b'_',
        },
        ByteRange {
            start: b'a',
            end: b'z',
        },
    ])
}

/// `\s` — the ASCII whitespace bytes: `\t \n \v \f \r` and space.
#[must_use]
pub fn space_class() -> CharClass {
    CharClass::from_ranges(vec![
        ByteRange {
            start: 0x09,
            end: 0x0d,
        },
        ByteRange {
            start: b' ',
            end: b' ',
        },
    ])
}
