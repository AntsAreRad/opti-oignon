//! Onion canonical JSON (OCJ), wire v1: the twin of `opti_oignon/allium/wire.py`.
//!
//! Every function here must accept, refuse and emit exactly as the Python
//! reference does, byte for byte, refusal codes and details included: the
//! first defect met, scanning left to right after a byte pre-pass, decides
//! the refusal.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

pub const MAX_INT: i64 = (1_i64 << 53) - 1;
pub const MAX_DEPTH: usize = 16;

/// The closed set of refusal codes, in the order the engine declares them.
pub const REFUSALS: [&str; 11] = [
    "non_canonical",
    "float",
    "non_ascii",
    "limit",
    "unknown_op",
    "bad_request",
    "unknown_law",
    "chain",
    "bad_fact",
    "budget",
    "engine_panic",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Str(String),
    Arr(Vec<Value>),
    /// Keys strictly increasing in byte order.
    Obj(Vec<(String, Value)>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused {
    pub code: &'static str,
    pub detail: String,
}

pub fn refused(code: &'static str, detail: String) -> Refused {
    Refused { code, detail }
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn keys(&self) -> Vec<&str> {
        match self {
            Value::Obj(members) => members.iter().map(|(k, _)| k.as_str()).collect(),
            _ => Vec::new(),
        }
    }
}

/// An object from members in any order: sorted by key bytes, as emission requires.
pub fn obj(mut members: Vec<(String, Value)>) -> Value {
    members.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    Value::Obj(members)
}

pub fn s(text: &str) -> Value {
    Value::Str(String::from(text))
}

fn is_digit(byte: i32) -> bool {
    (0x30..=0x39).contains(&byte)
}

fn is_lenient_ws(byte: u8) -> bool {
    byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D
}

struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
    lenient: bool,
    max_depth: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> i32 {
        match self.data.get(self.pos) {
            Some(byte) => i32::from(*byte),
            None => -1,
        }
    }

    fn at(&self, offset: usize) -> i32 {
        match self.pos.checked_add(offset).and_then(|index| self.data.get(index)) {
            Some(byte) => i32::from(*byte),
            None => -1,
        }
    }

    fn advance(&mut self, by: usize) {
        self.pos = self.pos.saturating_add(by);
    }

    fn gap(&mut self) -> Result<(), Refused> {
        while let Some(byte) = self.data.get(self.pos) {
            if !is_lenient_ws(*byte) {
                return Ok(());
            }
            if !self.lenient {
                return Err(refused("non_canonical", format!("whitespace at {}", self.pos)));
            }
            self.advance(1);
        }
        Ok(())
    }

    fn expect(&mut self, byte: u8) -> Result<(), Refused> {
        if self.peek() != i32::from(byte) {
            return Err(refused(
                "non_canonical",
                format!("expected {} at {}", byte as char, self.pos),
            ));
        }
        self.advance(1);
        Ok(())
    }

    fn starts_with(&self, word: &[u8]) -> bool {
        match self.pos.checked_add(word.len()) {
            Some(end) => self.data.get(self.pos..end) == Some(word),
            None => false,
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, Refused> {
        self.gap()?;
        let byte = self.peek();
        if byte == 0x7B {
            return self.object(depth.saturating_add(1));
        }
        if byte == 0x5B {
            return self.array(depth.saturating_add(1));
        }
        if byte == 0x22 {
            return self.string().map(Value::Str);
        }
        if byte == 0x2D || is_digit(byte) {
            return self.number();
        }
        if byte == 0x4E || byte == 0x49 {
            return Err(refused("float", format!("constant at {}", self.pos)));
        }
        let words: [(&[u8], Value); 3] = [
            (b"true", Value::Bool(true)),
            (b"false", Value::Bool(false)),
            (b"null", Value::Null),
        ];
        for (word, val) in words {
            if self.starts_with(word) {
                self.advance(word.len());
                return Ok(val);
            }
        }
        if byte == -1 {
            return Err(refused("non_canonical", format!("end of input at {}", self.pos)));
        }
        Err(refused("non_canonical", format!("unexpected byte at {}", self.pos)))
    }

    fn object(&mut self, depth: usize) -> Result<Value, Refused> {
        if depth > self.max_depth {
            return Err(refused("limit", format!("depth at {}", self.pos)));
        }
        self.advance(1);
        let mut members: Vec<(String, Value)> = Vec::new();
        self.gap()?;
        if self.peek() == 0x7D {
            self.advance(1);
            return Ok(Value::Obj(members));
        }
        loop {
            self.gap()?;
            if self.peek() != 0x22 {
                return Err(refused("non_canonical", format!("expected key at {}", self.pos)));
            }
            let at = self.pos;
            let key = self.string()?;
            if let Some((previous, _)) = members.last() {
                if key.as_bytes() <= previous.as_bytes() {
                    return Err(refused("non_canonical", format!("key order at {}", at)));
                }
            }
            self.gap()?;
            self.expect(0x3A)?;
            let value = self.value(depth)?;
            members.push((key, value));
            self.gap()?;
            let byte = self.peek();
            if byte == 0x2C {
                self.advance(1);
                continue;
            }
            if byte == 0x7D {
                self.advance(1);
                return Ok(Value::Obj(members));
            }
            return Err(refused("non_canonical", format!("expected , or }} at {}", self.pos)));
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, Refused> {
        if depth > self.max_depth {
            return Err(refused("limit", format!("depth at {}", self.pos)));
        }
        self.advance(1);
        let mut items: Vec<Value> = Vec::new();
        self.gap()?;
        if self.peek() == 0x5D {
            self.advance(1);
            return Ok(Value::Arr(items));
        }
        loop {
            items.push(self.value(depth)?);
            self.gap()?;
            let byte = self.peek();
            if byte == 0x2C {
                self.advance(1);
                continue;
            }
            if byte == 0x5D {
                self.advance(1);
                return Ok(Value::Arr(items));
            }
            return Err(refused("non_canonical", format!("expected , or ] at {}", self.pos)));
        }
    }

    fn string(&mut self) -> Result<String, Refused> {
        self.advance(1);
        let mut out = String::new();
        loop {
            let byte = self.peek();
            if byte == -1 {
                return Err(refused("non_canonical", format!("unterminated string at {}", self.pos)));
            }
            if byte == 0x22 {
                self.advance(1);
                return Ok(out);
            }
            if byte == 0x5C {
                let follow = self.at(1);
                if follow == 0x22 || follow == 0x5C {
                    out.push(if follow == 0x22 { '"' } else { '\\' });
                    self.advance(2);
                    continue;
                }
                if follow == 0x75 {
                    return Err(refused("non_ascii", format!("unicode escape at {}", self.pos)));
                }
                return Err(refused("non_canonical", format!("escape at {}", self.pos)));
            }
            if byte == 0x09 || byte == 0x0A || byte == 0x0D {
                return Err(refused("non_ascii", format!("byte at {}", self.pos)));
            }
            let ch = u8::try_from(byte).map(char::from).unwrap_or('?');
            out.push(ch);
            self.advance(1);
        }
    }

    fn number(&mut self) -> Result<Value, Refused> {
        let start = self.pos;
        let negative = self.peek() == 0x2D;
        if negative {
            self.advance(1);
            if self.peek() == 0x49 {
                return Err(refused("float", format!("constant at {}", start)));
            }
        }
        if !is_digit(self.peek()) {
            return Err(refused("non_canonical", format!("number at {}", start)));
        }
        if self.peek() == 0x30 {
            self.advance(1);
            if is_digit(self.peek()) {
                return Err(refused("non_canonical", format!("leading zero at {}", start)));
            }
        } else {
            while is_digit(self.peek()) {
                self.advance(1);
            }
        }
        let next = self.peek();
        if next == 0x2E || next == 0x65 || next == 0x45 {
            return Err(refused("float", format!("number at {}", start)));
        }
        let text = self.data.get(start..self.pos).unwrap_or(&[]);
        if text == b"-0" {
            return Err(refused("non_canonical", format!("negative zero at {}", start)));
        }
        let digits = if negative { text.get(1..).unwrap_or(&[]) } else { text };
        if digits.len() > 16 {
            return Err(refused("limit", format!("integer range at {}", start)));
        }
        let mut magnitude: i64 = 0;
        for digit in digits {
            magnitude = magnitude
                .saturating_mul(10)
                .saturating_add(i64::from(digit.saturating_sub(0x30)));
        }
        if magnitude > MAX_INT {
            return Err(refused("limit", format!("integer range at {}", start)));
        }
        Ok(Value::Int(if negative { magnitude.saturating_neg() } else { magnitude }))
    }
}

/// Decode OCJ bytes, refusing anything outside the domain; `lenient` reads an indented file.
pub fn parse(data: &[u8], lenient: bool) -> Result<Value, Refused> {
    for (index, byte) in data.iter().enumerate() {
        if (0x20..=0x7E).contains(byte) {
            continue;
        }
        if lenient && is_lenient_ws(*byte) {
            continue;
        }
        return Err(refused("non_ascii", format!("byte at {}", index)));
    }
    if data.is_empty() {
        return Err(refused("non_canonical", String::from("empty input")));
    }
    let mut parser = Parser { data, pos: 0, lenient, max_depth: MAX_DEPTH };
    let value = parser.value(0)?;
    parser.gap()?;
    if parser.pos != data.len() {
        return Err(refused("non_canonical", format!("trailing bytes at {}", parser.pos)));
    }
    Ok(value)
}

fn emit_into(value: &Value, depth: usize, out: &mut Vec<u8>) -> Result<(), Refused> {
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Int(number) => {
            if *number > MAX_INT || *number < MAX_INT.saturating_neg() {
                return Err(refused("limit", String::from("integer range")));
            }
            out.extend_from_slice(format!("{}", number).as_bytes());
        }
        Value::Str(text) => emit_string(text, out)?,
        Value::Arr(items) => {
            if depth.saturating_add(1) > MAX_DEPTH {
                return Err(refused("limit", String::from("depth")));
            }
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                emit_into(item, depth.saturating_add(1), out)?;
            }
            out.push(b']');
        }
        Value::Obj(members) => {
            if depth.saturating_add(1) > MAX_DEPTH {
                return Err(refused("limit", String::from("depth")));
            }
            out.push(b'{');
            for (index, (key, item)) in members.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                emit_string(key, out)?;
                out.push(b':');
                emit_into(item, depth.saturating_add(1), out)?;
            }
            out.push(b'}');
        }
    }
    Ok(())
}

fn emit_string(text: &str, out: &mut Vec<u8>) -> Result<(), Refused> {
    out.push(b'"');
    for ch in text.chars() {
        let code = u32::from(ch);
        if code == 0x22 {
            out.extend_from_slice(b"\\\"");
        } else if code == 0x5C {
            out.extend_from_slice(b"\\\\");
        } else if (0x20..=0x7E).contains(&code) {
            out.push(u8::try_from(code).unwrap_or(b'?'));
        } else {
            return Err(refused("non_ascii", String::from("character outside printable ASCII")));
        }
    }
    out.push(b'"');
    Ok(())
}

/// Encode a value as OCJ bytes.
pub fn emit(value: &Value) -> Result<Vec<u8>, Refused> {
    let mut out = Vec::new();
    emit_into(value, 0, &mut out)?;
    Ok(out)
}

const HEX: &[u8; 16] = b"0123456789abcdef";

pub fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0F)]));
    }
    out
}

pub fn is_hex(text: &str, length: usize) -> bool {
    text.len() == length && text.bytes().all(|b| HEX.contains(&b))
}

pub fn from_hex(text: &str) -> Option<Vec<u8>> {
    if text.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(text.len() / 2);
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let high = HEX.iter().position(|b| Some(b) == bytes.get(index))?;
        let low = HEX.iter().position(|b| Some(b) == bytes.get(index.saturating_add(1)))?;
        out.push(u8::try_from(high.saturating_mul(16).saturating_add(low)).ok()?);
        index = index.saturating_add(2);
    }
    Some(out)
}

/// Bulk arrays: "<type>:<lowercase hex>", little-endian two's complement.
fn bulk_type(kind: &str) -> Option<(usize, bool)> {
    match kind {
        "u8" => Some((1, false)),
        "i8" => Some((1, true)),
        "u16" => Some((2, false)),
        "i16" => Some((2, true)),
        "u32" => Some((4, false)),
        "i32" => Some((4, true)),
        "u64" => Some((8, false)),
        _ => None,
    }
}

pub fn pack_bulk(kind: &str, values: &[Value]) -> Result<String, Refused> {
    let (width, signed) = bulk_type(kind)
        .ok_or_else(|| refused("bad_request", String::from("unknown bulk type")))?;
    let bits = u32::try_from(width.saturating_mul(8)).unwrap_or(64);
    let top = bits.saturating_sub(1);
    let low: i128 = if signed { (1_i128 << top).saturating_neg() } else { 0 };
    let high: i128 = if signed { (1_i128 << top).saturating_sub(1) } else { (1_i128 << bits).saturating_sub(1) };
    let mut out = String::from(kind);
    out.push(':');
    for value in values {
        let number = match value {
            Value::Int(n) if i128::from(*n) >= low && i128::from(*n) <= high => i128::from(*n),
            _ => return Err(refused("bad_request", String::from("bulk value out of range"))),
        };
        let mut raw = number.rem_euclid(1_i128 << bits);
        for _ in 0..width {
            let byte = u8::try_from(raw & 0xFF).unwrap_or(0);
            out.push(char::from(HEX[usize::from(byte >> 4)]));
            out.push(char::from(HEX[usize::from(byte & 0x0F)]));
            raw >>= 8;
        }
    }
    Ok(out)
}

/// Unpack a bulk string into (type, values); a value past the wire's integers is refused on emission.
pub fn unpack_bulk(value: &Value) -> Result<(String, Vec<i128>), Refused> {
    let text = match value {
        Value::Str(text) if text.contains(':') => text,
        _ => return Err(refused("bad_request", String::from("not a bulk string"))),
    };
    let (kind, digits) = text.split_once(':').unwrap_or(("", ""));
    let (width, signed) = bulk_type(kind)
        .ok_or_else(|| refused("bad_request", String::from("unknown bulk type")))?;
    if digits.len().checked_rem(width.saturating_mul(2)).unwrap_or(1) != 0 {
        return Err(refused("bad_request", String::from("bulk length")));
    }
    if !digits.bytes().all(|b| HEX.contains(&b)) {
        return Err(refused("bad_request", String::from("bulk hex")));
    }
    let bytes = from_hex(digits).unwrap_or_default();
    let bits = u32::try_from(width.saturating_mul(8)).unwrap_or(64);
    let mut values = Vec::new();
    for chunk in bytes.chunks(width) {
        let mut raw: i128 = 0;
        for byte in chunk.iter().rev() {
            raw = (raw << 8) | i128::from(*byte);
        }
        if signed && (raw >> bits.saturating_sub(1)) & 1 == 1 {
            raw = raw.saturating_sub(1_i128 << bits);
        }
        values.push(raw);
    }
    Ok((String::from(kind), values))
}
