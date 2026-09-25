//! The genome, schema 1: the twin of `opti_oignon/allium/ref/organs/genome.py`.
//!
//! A genome is diploid: `pairs` pairs of chromosomes, each chromosome a list
//! of fixed 16-byte records. The order of records on a chromosome is data,
//! since the promoter of a gene is the run of CIS records just before it, so
//! nothing here ever sorts a chromosome.
//!
//! Record layout (every multi-byte field big-endian): kind u8 at 0, flags u8
//! at 1, locus u16 at 2, stage mask u16 at 4, then a 10-byte body laid out
//! per kind. Genome framing: `"ALG" 0x01`, schema u16 = 1, ploidy u8 = 2,
//! pairs u8, then for each chromosome (pair-major, homolog-minor) a u16
//! record count and the records.
//!
//! The kind layout below is compiled into both engines; the boxes, the loci
//! and the species classes are law data, and a law whose kind table
//! disagrees with this layout is refused. The decoder is total: it refuses
//! by name, and the first defect it meets, scanning left to right, decides.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

use crate::ocj::{self, refused, Refused, Value};
use crate::rng;

pub const MAGIC: [u8; 4] = *b"ALG\x01";
pub const SCHEMA: i64 = 1;
pub const PLOIDY: usize = 2;
pub const STAGES: i64 = 11;
pub const RECORD: usize = 16;
pub const BODY: usize = 10;
pub const HEADER: usize = 8;
pub const LOCUS_LIMIT: i64 = 0x8000;
pub const STAGE_MASK_MAX: i64 = 0x7FF;

pub const DOMAIN_FOUNDER: &str = "genome.founder";
pub const DOMAIN_CORNER: &str = "genome.corner";
pub const DOMAINS: [&str; 2] = [DOMAIN_CORNER, DOMAIN_FOUNDER];
pub const CORNER_MAX: i64 = 0xFFFF_FFFF;

pub const DOM_MAX: i64 = 0;
pub const ADD: i64 = 1;
pub const LOAD_REC: i64 = 2;
pub const FLAG_ESSENTIAL: i64 = 0x04;
pub const FLAG_MODE: i64 = 0x03;
pub const FLAG_UNUSED: i64 = 0xE0;

pub const CLASSES: [&str; 5] = ["input", "token", "conserved", "open", "free"];
pub const SPECIES: usize = 64;
pub const SPECIES_NONE: i64 = 255;
pub const NAME_MAX: usize = 48;

/// Structural: never jittered or averaged; fixed on ADD and LOAD_REC loci.
pub const S: u8 = 1;
/// Names a species the gene writes (or consumes): fixed on every locus.
pub const W: u8 = 2;
/// Reads a species: never a token species.
pub const R: u8 = 4;
/// A species reference: legal only when at most 63 or exactly 255.
pub const P: u8 = 8;

pub const KIND_CIS: u8 = 2;
pub const KIND_LOAD: u8 = 15;

// ---------------------------------------------------------------------------
// The kind layout
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub name: &'static str,
    pub byte: usize,
    pub shift: u32,
    pub bits: u32,
    pub signed: bool,
    pub marks: u8,
    pub scale: i64,
}

impl Field {
    /// The integer range this field's position can hold; `bits` is 1 to 16.
    pub const fn type_range(&self) -> (i64, i64) {
        if self.signed {
            let top = 1_i64 << self.bits.wrapping_sub(1);
            (top.wrapping_neg(), top.wrapping_sub(1))
        } else {
            (0, self.mask())
        }
    }

    /// The low `bits` bits set.
    pub const fn mask(&self) -> i64 {
        (1_i64 << self.bits).wrapping_sub(1)
    }
}

const fn field(name: &'static str, byte: usize, bits: u32, signed: bool, marks: u8, shift: u32, scale: i64) -> Field {
    Field { name, byte, shift, bits, signed, marks, scale }
}

const fn u8f(name: &'static str, byte: usize, marks: u8) -> Field {
    field(name, byte, 8, false, marks, 0, 1)
}

const fn i8f(name: &'static str, byte: usize, scale: i64) -> Field {
    field(name, byte, 8, true, 0, 0, scale)
}

const fn u16f(name: &'static str, byte: usize, scale: i64) -> Field {
    field(name, byte, 16, false, 0, 0, scale)
}

const fn i16f(name: &'static str, byte: usize, scale: i64) -> Field {
    field(name, byte, 16, true, 0, 0, scale)
}

const fn bitsf(name: &'static str, byte: usize, shift: u32, bits: u32, marks: u8) -> Field {
    field(name, byte, bits, false, marks, shift, 1)
}

/// The key DOM_MAX compares first.
#[derive(Clone, Copy, Debug)]
pub enum Strength {
    Field(usize),
    Abs(usize),
    SumAbs(&'static [usize]),
    Hash,
}

#[derive(Debug)]
pub struct Kind {
    pub code: u8,
    pub name: &'static str,
    pub fields: &'static [Field],
    pub modes: &'static [i64],
    pub strength: Strength,
    pub reserved: bool,
}

pub static KINDS: [Kind; 16] = [
    Kind {
        code: 1,
        name: "tf",
        fields: &[u8f("out", 0, S | W), u16f("prod", 1, 1), u16f("deg", 3, 1), i16f("bias", 5, 16), u16f("rate", 7, 1)],
        modes: &[DOM_MAX, ADD],
        strength: Strength::Field(1),
        reserved: false,
    },
    Kind {
        code: 2,
        name: "cis",
        fields: &[u8f("src", 0, S | R), i16f("w", 1, 16), u16f("K", 3, 8), u8f("n", 5, 0), u8f("mode", 6, S)],
        modes: &[DOM_MAX, ADD],
        strength: Strength::Abs(1),
        reserved: false,
    },
    Kind {
        code: 3,
        name: "enz",
        fields: &[
            u8f("s1", 0, S | W | R | P),
            u8f("s2", 1, S | W | R | P),
            u8f("p1", 2, S | W | P),
            u8f("p2", 3, S | W | P),
            u16f("kcat", 4, 1),
            u16f("Km", 6, 8),
            u16f("yield", 8, 1),
        ],
        modes: &[DOM_MAX, ADD],
        strength: Strength::Field(4),
        reserved: false,
    },
    Kind {
        code: 4,
        name: "rec",
        fields: &[u8f("channel", 0, S), u8f("species", 1, S | W), i16f("gain", 2, 16), u16f("threshold", 4, 1)],
        modes: &[DOM_MAX, ADD],
        strength: Strength::Abs(2),
        reserved: false,
    },
    Kind {
        code: 5,
        name: "morph",
        fields: &[
            u8f("pred", 0, S),
            u8f("guard", 1, S | R),
            u16f("threshold", 2, 8),
            u8f("succ", 4, S),
            u16f("rate", 5, 1),
            i8f("angle", 7, 8),
        ],
        modes: &[DOM_MAX, ADD],
        strength: Strength::Field(4),
        reserved: false,
    },
    Kind {
        code: 6,
        name: "imm",
        fields: &[u8f("class", 0, S), u16f("strength", 1, 1), u16f("cost", 3, 1)],
        modes: &[DOM_MAX],
        strength: Strength::Field(1),
        reserved: false,
    },
    Kind {
        code: 7,
        name: "pig",
        fields: &[u8f("class", 0, S), u8f("allele", 1, 0)],
        modes: &[DOM_MAX],
        strength: Strength::Field(1),
        reserved: false,
    },
    Kind {
        code: 8,
        name: "plast",
        fields: &[
            u8f("class", 0, S),
            i8f("A", 1, 1),
            i8f("Aneg", 2, 1),
            i8f("B", 3, 1),
            i8f("C", 4, 1),
            i8f("D", 5, 1),
            bitsf("LR", 6, 4, 4, 0),
            bitsf("TAU_X", 6, 0, 4, 0),
            bitsf("TAU_Y", 7, 4, 4, 0),
            bitsf("TAU_E", 7, 0, 4, 0),
            bitsf("M_SEL", 8, 5, 3, S),
            bitsf("M_SIGN", 8, 4, 1, S),
        ],
        modes: &[DOM_MAX, ADD],
        strength: Strength::Abs(1),
        reserved: true,
    },
    Kind {
        code: 9,
        name: "temp",
        fields: &[
            u8f("channel", 0, S),
            i16f("weight", 1, 1),
            i16f("base", 3, 1),
            u8f("habituation", 5, 0),
            u16f("curiosity", 6, 1),
        ],
        modes: &[DOM_MAX, ADD],
        strength: Strength::Abs(1),
        reserved: true,
    },
    Kind {
        code: 10,
        name: "lex",
        fields: &[
            u8f("offset", 0, S),
            u8f("d0", 1, 0),
            u8f("d1", 2, 0),
            u8f("d2", 3, 0),
            u8f("d3", 4, 0),
            u8f("d4", 5, 0),
            u8f("d5", 6, 0),
            u8f("d6", 7, 0),
            u8f("d7", 8, 0),
            u8f("d8", 9, 0),
        ],
        modes: &[DOM_MAX, ADD],
        strength: Strength::Hash,
        reserved: true,
    },
    Kind {
        code: 11,
        name: "shape",
        fields: &[u8f("trait", 0, S), u16f("value", 1, 1), u16f("window", 3, 1)],
        modes: &[DOM_MAX, ADD],
        strength: Strength::Field(1),
        reserved: true,
    },
    Kind {
        code: 12,
        name: "vern",
        fields: &[
            u8f("D_enter", 0, 0),
            u8f("v_cold", 1, 0),
            u8f("v_dorm", 2, 0),
            u16f("VU_req", 3, 1),
            u8f("stalk_days", 5, 0),
            u8f("offsets", 6, 0),
            u8f("rest_days", 7, 0),
            u16f("a_min", 8, 1),
        ],
        modes: &[ADD],
        strength: Strength::Field(3),
        reserved: true,
    },
    Kind {
        code: 13,
        name: "prc",
        fields: &[
            u8f("block", 0, S),
            i16f("shift0", 1, 1),
            i16f("shift1", 3, 1),
            i16f("shift2", 5, 1),
            i16f("shift3", 7, 1),
        ],
        modes: &[ADD],
        strength: Strength::SumAbs(&[1, 2, 3, 4]),
        reserved: true,
    },
    Kind {
        code: 14,
        name: "te",
        fields: &[u16f("activity", 0, 1), u8f("target_bias", 2, 0)],
        modes: &[DOM_MAX, ADD],
        strength: Strength::Field(0),
        reserved: false,
    },
    Kind {
        code: 15,
        name: "load",
        fields: &[u8f("target", 0, S | W), u16f("effect", 1, 1)],
        modes: &[LOAD_REC],
        strength: Strength::Field(1),
        reserved: false,
    },
    Kind {
        code: 16,
        name: "brain",
        fields: &[u8f("param", 0, S), i16f("value", 1, 1)],
        modes: &[DOM_MAX, ADD],
        strength: Strength::Abs(1),
        reserved: true,
    },
];

pub fn kind_by_code(code: u8) -> Option<&'static Kind> {
    KINDS.iter().find(|kind| kind.code == code)
}

pub fn kind_by_name(name: &str) -> Option<&'static Kind> {
    KINDS.iter().find(|kind| kind.name == name)
}

/// The body bits a kind assigns; every other bit is a pad and must be zero.
pub fn used_bits(kind: &Kind) -> [u8; BODY] {
    let mut used = [0_u8; BODY];
    for field in kind.fields {
        match field.bits {
            16 => {
                if let Some(slot) = used.get_mut(field.byte) {
                    *slot = 0xFF;
                }
                if let Some(slot) = field.byte.checked_add(1).and_then(|next| used.get_mut(next)) {
                    *slot = 0xFF;
                }
            }
            8 => {
                if let Some(slot) = used.get_mut(field.byte) {
                    *slot = 0xFF;
                }
            }
            _ => {
                let bits = u8::try_from(field.mask() << field.shift).unwrap_or(0xFF);
                if let Some(slot) = used.get_mut(field.byte) {
                    *slot |= bits;
                }
            }
        }
    }
    used
}

/// A field's value in a body: big-endian, sign-extended when the field is signed.
pub fn read_field(body: &[u8; BODY], field: &Field) -> Option<i64> {
    let first = i64::from(*body.get(field.byte)?);
    let value = match field.bits {
        16 => (first << 8) | i64::from(*body.get(field.byte.checked_add(1)?)?),
        8 => first,
        _ => return Some((first >> field.shift) & field.mask()),
    };
    let (_, high) = field.type_range();
    if field.signed && value > high {
        return value.checked_sub(1_i64 << field.bits);
    }
    Some(value)
}

fn write_field(body: &mut [u8; BODY], field: &Field, value: i64) -> Option<()> {
    let raw = value & field.mask();
    match field.bits {
        16 => {
            *body.get_mut(field.byte)? = u8::try_from(raw >> 8).ok()?;
            *body.get_mut(field.byte.checked_add(1)?)? = u8::try_from(raw & 0xFF).ok()?;
        }
        8 => {
            *body.get_mut(field.byte)? = u8::try_from(raw).ok()?;
        }
        _ => {
            let bits = u8::try_from(raw << field.shift).ok()?;
            let slot = body.get_mut(field.byte)?;
            *slot |= bits;
        }
    }
    Some(())
}

pub fn species_legal(value: i64) -> bool {
    value <= 63 || value == SPECIES_NONE
}

// ---------------------------------------------------------------------------
// Reading parsed law and pool values
// ---------------------------------------------------------------------------

pub fn int_of(value: Option<&Value>) -> Option<i64> {
    match value {
        Some(Value::Int(number)) => Some(*number),
        _ => None,
    }
}

pub fn str_of(value: Option<&Value>) -> Option<&str> {
    match value {
        Some(Value::Str(text)) => Some(text.as_str()),
        _ => None,
    }
}

pub fn arr_of(value: Option<&Value>) -> Option<&Vec<Value>> {
    match value {
        Some(Value::Arr(items)) => Some(items),
        _ => None,
    }
}

/// True when `value` is an object whose keys are exactly `names`.
pub fn keys_are(value: Option<&Value>, names: &[&str]) -> bool {
    match value {
        Some(Value::Obj(members)) => members.len() == names.len() && names.iter().all(|name| members.iter().any(|(key, _)| key == name)),
        _ => false,
    }
}

/// Lowercase ASCII letters, digits and underscores, 1 to 48 characters.
pub fn is_name(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Str(text)) => {
            (1..=NAME_MAX).contains(&text.len())
                && text.bytes().all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        }
        _ => false,
    }
}

/// Equality as the reference's `==` reads it: `true == 1` and `false == 0`.
fn py_eq(a: &Value, b: &Value) -> bool {
    fn number(value: &Value) -> Option<i64> {
        match value {
            Value::Int(n) => Some(*n),
            Value::Bool(true) => Some(1),
            Value::Bool(false) => Some(0),
            _ => None,
        }
    }
    match (a, b) {
        (Value::Arr(x), Value::Arr(y)) => x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| py_eq(p, q)),
        (Value::Obj(x), Value::Obj(y)) => {
            x.len() == y.len() && x.iter().all(|(key, p)| y.iter().any(|(other, q)| key == other && py_eq(p, q)))
        }
        _ => match (number(a), number(b)) {
            (Some(x), Some(y)) => x == y,
            _ => a == b,
        },
    }
}

fn shown(value: Option<&Value>) -> String {
    match value {
        Some(Value::Int(n)) => format!("{}", n),
        Some(other) => ocj::emit(other).map(|bytes| bytes.iter().map(|b| char::from(*b)).collect()).unwrap_or_default(),
        None => String::from("None"),
    }
}

// ---------------------------------------------------------------------------
// The law view: what the codec needs from a validated law
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Locus {
    pub id: i64,
    pub kind: u8,
    pub flags: i64,
    pub stages: i64,
    pub boxes: Vec<(i64, i64)>,
}

impl Locus {
    pub fn mode(&self) -> i64 {
        self.flags & FLAG_MODE
    }

    pub fn essential(&self) -> bool {
        self.flags & FLAG_ESSENTIAL != 0
    }
}

#[derive(Clone, Debug)]
pub struct View {
    pub pairs: usize,
    pub max_records: usize,
    pub max_promoter: usize,
    pub max_bytes: usize,
    pub loci: BTreeMap<i64, Locus>,
    pub order: Vec<i64>,
    pub chromosomes: Vec<Vec<i64>>,
    pub classes: Vec<&'static str>,
    pub deg: Vec<i64>,
    pub token: Vec<bool>,
}

impl View {
    pub fn is_token(&self, value: i64) -> bool {
        usize::try_from(value).ok().and_then(|code| self.token.get(code)).copied().unwrap_or(false)
    }
}

/// The codec's view of a law already validated by `validate_law`; `None` if it cannot be read.
pub fn view(law: &Value) -> Option<View> {
    let genome = law.get("genome")?;
    let size = |key: &str| int_of(genome.get(key)).and_then(|n| usize::try_from(n).ok());
    let mut kind_boxes: BTreeMap<i64, Vec<(i64, i64)>> = BTreeMap::new();
    for entry in arr_of(genome.get("kinds"))? {
        let mut boxes = Vec::new();
        for spec in arr_of(entry.get("fields"))? {
            boxes.push((int_of(spec.get("lo"))?, int_of(spec.get("hi"))?));
        }
        kind_boxes.insert(int_of(entry.get("code"))?, boxes);
    }
    let mut loci = BTreeMap::new();
    let mut order = Vec::new();
    for entry in arr_of(genome.get("loci"))? {
        let kind = kind_by_name(str_of(entry.get("kind"))?)?;
        let mut boxes = kind_boxes.get(&i64::from(kind.code))?.clone();
        for over in arr_of(entry.get("box"))? {
            let name = str_of(over.get("field"))?;
            let index = kind.fields.iter().position(|f| f.name == name)?;
            *boxes.get_mut(index)? = (int_of(over.get("lo"))?, int_of(over.get("hi"))?);
        }
        let id = int_of(entry.get("id"))?;
        let flags = int_of(entry.get("flags"))?;
        let stages = int_of(entry.get("stages"))?;
        loci.insert(id, Locus { id, kind: kind.code, flags, stages, boxes });
        order.push(id);
    }
    let mut chromosomes = Vec::new();
    for ids in arr_of(genome.get("chromosomes"))? {
        let mut list = Vec::new();
        for ident in arr_of(Some(ids))? {
            list.push(int_of(Some(ident))?);
        }
        chromosomes.push(list);
    }
    let mut classes = Vec::new();
    let mut deg = Vec::new();
    let mut token = Vec::new();
    for entry in arr_of(genome.get("species"))? {
        let class = str_of(entry.get("class"))?;
        classes.push(*CLASSES.iter().find(|known| **known == class)?);
        deg.push(int_of(entry.get("deg"))?);
        token.push(class == "token");
    }
    Some(View {
        pairs: size("pairs")?,
        max_records: size("max_records")?,
        max_promoter: size("max_promoter")?,
        max_bytes: size("max_bytes")?,
        loci,
        order,
        chromosomes,
        classes,
        deg,
        token,
    })
}

// ---------------------------------------------------------------------------
// Law validation
// ---------------------------------------------------------------------------

const GENOME_KEYS: [&str; 10] = [
    "chromosomes",
    "kinds",
    "loci",
    "max_bytes",
    "max_promoter",
    "max_records",
    "pairs",
    "schema",
    "species",
    "stages",
];

fn box_ok(field: &Field, low: Option<i64>, high: Option<i64>) -> bool {
    let (lo_t, hi_t) = field.type_range();
    match (low, high) {
        (Some(low), Some(high)) => lo_t <= low && low <= high && high <= hi_t,
        _ => false,
    }
}

fn validate_kinds(kinds: Option<&Value>, out: &mut Vec<String>) -> bool {
    let entries = match arr_of(kinds) {
        Some(entries) if entries.len() == KINDS.len() => entries,
        _ => {
            out.push(String::from("kinds: exactly 16 entries"));
            return false;
        }
    };
    let mut good = true;
    for (entry, kind) in entries.iter().zip(KINDS.iter()) {
        let code = i64::from(kind.code);
        let name = kind.name;
        if !keys_are(Some(entry), &["code", "fields", "name"])
            || int_of(entry.get("code")) != Some(code)
            || str_of(entry.get("name")) != Some(name)
        {
            out.push(format!("kinds: entry {} is not the engine's {}", code, name));
            good = false;
            continue;
        }
        let boxes = match arr_of(entry.get("fields")) {
            Some(boxes) if boxes.len() == kind.fields.len() => boxes,
            _ => {
                out.push(format!("kinds: {} fields differ from the engine layout", name));
                good = false;
                continue;
            }
        };
        for (spec, field) in boxes.iter().zip(kind.fields.iter()) {
            if !keys_are(Some(spec), &["hi", "lo", "name"]) || str_of(spec.get("name")) != Some(field.name) {
                out.push(format!("kinds: {} fields differ from the engine layout", name));
                good = false;
                break;
            }
            let (low, high) = (int_of(spec.get("lo")), int_of(spec.get("hi")));
            let (Some(lo), Some(hi)) = (low, high) else {
                out.push(format!("kinds: {}.{} box outside its type range", name, field.name));
                good = false;
                continue;
            };
            if !box_ok(field, low, high) {
                out.push(format!("kinds: {}.{} box outside its type range", name, field.name));
                good = false;
                continue;
            }
            let fname = field.name;
            if ((name == "cis" && fname == "K") || (name == "enz" && fname == "Km")) && lo < 512 {
                out.push(format!("kinds: {}.{} lo below 512", name, fname));
                good = false;
            }
            if name == "cis" && fname == "n" && (lo < 1 || hi > 4) {
                out.push(String::from("kinds: cis.n outside [1, 4]"));
                good = false;
            }
            if name == "plast" && fname.starts_with("TAU_") && (lo < 1 || hi > 15) {
                out.push(format!("kinds: plast.{} outside [1, 15]", fname));
                good = false;
            }
            if field.marks & P != 0 && !(species_legal(lo) && species_legal(hi)) {
                out.push(format!("kinds: {}.{} species endpoints", name, fname));
                good = false;
            }
        }
    }
    good
}

fn validate_species(species: Option<&Value>, out: &mut Vec<String>) -> bool {
    let entries = match arr_of(species) {
        Some(entries) if entries.len() == SPECIES => entries,
        _ => {
            out.push(String::from("species: exactly 64 entries"));
            return false;
        }
    };
    let mut good = true;
    for (code, entry) in entries.iter().enumerate() {
        let wanted = i64::try_from(code).ok();
        if !keys_are(Some(entry), &["class", "code", "deg", "name"]) || int_of(entry.get("code")).is_none() || int_of(entry.get("code")) != wanted {
            out.push(format!("species: entry {}", code));
            good = false;
            continue;
        }
        if !is_name(entry.get("name")) {
            out.push(format!("species: {} name", code));
            good = false;
        }
        let class = match str_of(entry.get("class")) {
            Some(class) if CLASSES.contains(&class) => class,
            _ => {
                out.push(format!("species: {} class", code));
                good = false;
                continue;
            }
        };
        match int_of(entry.get("deg")) {
            Some(deg) if (0..=65535).contains(&deg) => {
                if (class == "input" || class == "conserved") && deg != 0 {
                    out.push(format!("species: {} deg of an {} species", code, class));
                    good = false;
                }
            }
            _ => {
                out.push(format!("species: {} deg", code));
                good = false;
            }
        }
    }
    good
}

fn class_of(species: &[Value], code: i64) -> Option<&str> {
    let index = usize::try_from(code).ok()?;
    str_of(species.get(index)?.get("class"))
}

fn written_class_ok(kind_name: &str, value: i64, species: &[Value]) -> bool {
    if value == SPECIES_NONE && kind_name == "enz" {
        return true;
    }
    if value > 63 {
        return false;
    }
    let Some(class) = class_of(species, value) else {
        return false;
    };
    match kind_name {
        "tf" | "load" => class == "open" || class == "free",
        "rec" => class == "input" || class == "token" || class == "open" || class == "free",
        "enz" => class != "token",
        _ => false,
    }
}

/// Overrides, then the S, W and R rules, for one locus.
fn validate_locus_fields(entry: &Value, kind: &Kind, kind_boxes: &[(i64, i64)], species: &[Value], out: &mut Vec<String>) {
    let name = str_of(entry.get("name")).unwrap_or("");
    let mut boxes: Vec<(i64, i64)> = kind_boxes.to_vec();
    let Some(overrides) = arr_of(entry.get("box")) else {
        out.push(format!("loci: {} box", name));
        return;
    };
    let mut last: Option<usize> = None;
    for over in overrides {
        let index = match str_of(over.get("field")) {
            Some(fname) if keys_are(Some(over), &["field", "hi", "lo"]) => kind.fields.iter().position(|f| f.name == fname),
            _ => None,
        };
        let Some(index) = index else {
            out.push(format!("loci: {} names a field its kind does not have", name));
            return;
        };
        if last.is_some_and(|previous| index <= previous) {
            out.push(format!("loci: {} overrides out of kind field order", name));
            return;
        }
        last = Some(index);
        let fname = str_of(over.get("field")).unwrap_or("");
        let inside = match (int_of(over.get("lo")), int_of(over.get("hi")), kind_boxes.get(index)) {
            (Some(low), Some(high), Some((k_lo, k_hi))) => *k_lo <= low && low <= high && high <= *k_hi,
            _ => false,
        };
        if !inside {
            out.push(format!("loci: {}.{} outside the kind box", name, fname));
            return;
        }
        if let (Some(slot), Some(low), Some(high)) = (boxes.get_mut(index), int_of(over.get("lo")), int_of(over.get("hi"))) {
            *slot = (low, high);
        }
    }
    let mode = int_of(entry.get("flags")).unwrap_or(0) & FLAG_MODE;
    for (field, (low, high)) in kind.fields.iter().zip(boxes.iter()) {
        let (low, high) = (*low, *high);
        let marks = field.marks;
        let fixed = low == high;
        let fname = field.name;
        if marks & S != 0 && (mode == ADD || mode == LOAD_REC) && !fixed {
            out.push(format!("loci: {}.{} is structural and must be fixed on this mode", name, fname));
        }
        if marks & W != 0 {
            if !fixed {
                out.push(format!("loci: {}.{} names a written species and must be fixed", name, fname));
            } else if !written_class_ok(kind.name, low, species) {
                out.push(format!("loci: {}.{} writes a species of a forbidden class", name, fname));
            }
        }
        if marks & R != 0 && fixed && low <= 63 && class_of(species, low) == Some("token") {
            out.push(format!("loci: {}.{} reads a token species", name, fname));
        }
        if marks & P != 0 && !(species_legal(low) && species_legal(high)) {
            out.push(format!("loci: {}.{} species endpoints", name, fname));
        }
    }
}

/// Every defect of a law's genome section, named; empty when it is sound.
pub fn validate_law(law: &Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let genome = match law.get("genome") {
        Some(genome) if keys_are(Some(genome), &GENOME_KEYS) => genome,
        _ => return vec![String::from("genome: missing, or its keys are not exactly the schema's")],
    };
    for key in ["max_bytes", "max_promoter", "max_records", "pairs", "schema", "stages"] {
        if int_of(genome.get(key)).is_none() {
            out.push(format!("genome: {} is not an integer", key));
        }
    }
    if !out.is_empty() {
        return out;
    }
    let number = |key: &str| int_of(genome.get(key)).unwrap_or(0);
    let (pairs, max_records, max_promoter) = (number("pairs"), number("max_records"), number("max_promoter"));
    if number("schema") != SCHEMA {
        out.push(String::from("genome: schema"));
    }
    if number("stages") != STAGES {
        out.push(String::from("genome: stages"));
    }
    if !(1..=8).contains(&pairs) {
        out.push(String::from("genome: pairs outside 1..8"));
    }
    if !(1..=64).contains(&max_records) {
        out.push(String::from("genome: max_records outside 1..64"));
    }
    if !(1..=16).contains(&max_promoter) {
        out.push(String::from("genome: max_promoter outside 1..16"));
    }
    let kinds_ok = validate_kinds(genome.get("kinds"), &mut out);
    let species_ok = validate_species(genome.get("species"), &mut out);
    if !out.is_empty() || !kinds_ok || !species_ok {
        return out;
    }
    let mut kind_boxes: BTreeMap<i64, Vec<(i64, i64)>> = BTreeMap::new();
    for entry in arr_of(genome.get("kinds")).map(|v| v.as_slice()).unwrap_or(&[]) {
        let mut boxes = Vec::new();
        for spec in arr_of(entry.get("fields")).map(|v| v.as_slice()).unwrap_or(&[]) {
            boxes.push((int_of(spec.get("lo")).unwrap_or(0), int_of(spec.get("hi")).unwrap_or(0)));
        }
        kind_boxes.insert(int_of(entry.get("code")).unwrap_or(0), boxes);
    }
    let species: &[Value] = arr_of(genome.get("species")).map(|v| v.as_slice()).unwrap_or(&[]);
    let loci = match arr_of(genome.get("loci")) {
        Some(loci) if !loci.is_empty() => loci,
        _ => return vec![String::from("loci: a non-empty list")],
    };
    let mut names: Vec<&str> = Vec::new();
    let mut previous: i64 = -1;
    let mut kinds_of: BTreeMap<i64, u8> = BTreeMap::new();
    for entry in loci {
        if !keys_are(Some(entry), &["box", "flags", "id", "kind", "name", "stages"]) {
            out.push(String::from("loci: an entry's keys are not exactly the schema's"));
            continue;
        }
        let (Some(ident), Some(flags), Some(stages)) = (int_of(entry.get("id")), int_of(entry.get("flags")), int_of(entry.get("stages"))) else {
            out.push(String::from("loci: id, flags and stages are integers"));
            continue;
        };
        let label = match str_of(entry.get("name")) {
            Some(label) if is_name(entry.get("name")) => label,
            _ => {
                out.push(format!("loci: {} name", ident));
                continue;
            }
        };
        if names.contains(&label) {
            out.push(format!("loci: {} named twice", label));
        }
        names.push(label);
        if ident <= previous || ident >= LOCUS_LIMIT || ident < 0 || (ident >> 8) >= pairs {
            out.push(format!("loci: {} id out of order or out of range", label));
        }
        previous = previous.max(ident);
        let Some(kind) = str_of(entry.get("kind")).and_then(kind_by_name) else {
            out.push(format!("loci: {} kind", label));
            continue;
        };
        kinds_of.insert(ident, kind.code);
        if !(0..=255).contains(&flags) || flags & FLAG_UNUSED != 0 || !kind.modes.contains(&(flags & FLAG_MODE)) {
            out.push(format!("loci: {} flags", label));
            continue;
        }
        if !(1..=STAGE_MASK_MAX).contains(&stages) {
            out.push(format!("loci: {} stages", label));
        }
        let boxes = kind_boxes.get(&i64::from(kind.code)).map(|v| v.as_slice()).unwrap_or(&[]);
        validate_locus_fields(entry, kind, boxes, species, &mut out);
    }
    if !out.is_empty() {
        return out;
    }
    let chromosomes = match arr_of(genome.get("chromosomes")) {
        Some(lists) if i64::try_from(lists.len()).ok() == Some(pairs) => lists,
        _ => return vec![String::from("chromosomes: one list per pair")],
    };
    let mut seen: BTreeMap<i64, ()> = BTreeMap::new();
    for (pair, ids) in chromosomes.iter().enumerate() {
        let ids = match arr_of(Some(ids)) {
            Some(ids) if i64::try_from(ids.len()).is_ok_and(|n| n <= max_records) => ids,
            _ => {
                out.push(format!("chromosomes: pair {} list", pair));
                continue;
            }
        };
        let pair_id = i64::try_from(pair).unwrap_or(-1);
        let mut run: i64 = 0;
        for item in ids {
            let listed = match int_of(Some(item)) {
                Some(ident) if kinds_of.contains_key(&ident) && (ident >> 8) == pair_id && !seen.contains_key(&ident) => Some(ident),
                _ => None,
            };
            let Some(ident) = listed else {
                out.push(format!("chromosomes: pair {} lists {} wrongly", pair, shown(Some(item))));
                continue;
            };
            seen.insert(ident, ());
            if kinds_of.get(&ident) == Some(&KIND_CIS) {
                run = run.saturating_add(1);
                if run > max_promoter {
                    out.push(format!("chromosomes: pair {} template promoter too long", pair));
                }
            } else {
                run = 0;
            }
        }
    }
    if seen.len() != kinds_of.len() {
        out.push(String::from("chromosomes: every locus exactly once"));
    }
    let expected = i64::try_from(loci.len())
        .ok()
        .and_then(|n| n.checked_mul(32))
        .and_then(|records| pairs.checked_mul(4).and_then(|counts| counts.checked_add(records)))
        .and_then(|body| body.checked_add(8));
    if expected != Some(number("max_bytes")) {
        out.push(String::from("genome: max_bytes"));
    }
    out
}

// ---------------------------------------------------------------------------
// Pool validation
// ---------------------------------------------------------------------------

/// One founder allele, as the derivation reads it.
#[derive(Clone, Debug)]
pub struct Allele {
    pub freq: i64,
    pub body: Vec<i64>,
    pub window: Vec<i64>,
}

fn ints(value: Option<&Value>) -> Option<Vec<i64>> {
    arr_of(value)?.iter().map(|item| int_of(Some(item))).collect()
}

fn allele_defect(allele: &Value, locus: &Locus, lawview: &View) -> Option<String> {
    if !keys_are(Some(allele), &["body", "freq", "name", "window"]) {
        return Some(String::from("allele keys"));
    }
    if !int_of(allele.get("freq")).is_some_and(|freq| (1..=1000).contains(&freq)) {
        return Some(String::from("freq"));
    }
    if !is_name(allele.get("name")) {
        return Some(String::from("allele name"));
    }
    let Some(kind) = kind_by_code(locus.kind) else {
        return Some(String::from("body or window length"));
    };
    let (body, window) = match (arr_of(allele.get("body")), arr_of(allele.get("window"))) {
        (Some(body), Some(window)) if body.len() == kind.fields.len() && window.len() == body.len() => (body, window),
        _ => return Some(String::from("body or window length")),
    };
    for (index, field) in kind.fields.iter().enumerate() {
        let Some((low, high)) = locus.boxes.get(index).copied() else {
            return Some(format!("body {} outside its box", field.name));
        };
        let value = match int_of(body.get(index)) {
            Some(value) if low <= value && value <= high => value,
            _ => return Some(format!("body {} outside its box", field.name)),
        };
        let marks = field.marks;
        if marks & P != 0 && !species_legal(value) {
            return Some(format!("body {} species", field.name));
        }
        if marks & R != 0 && value <= 63 && lawview.is_token(value) {
            return Some(format!("body {} reads a token species", field.name));
        }
        let span = high.checked_sub(low);
        let win = match (int_of(window.get(index)), span) {
            (Some(win), Some(span)) if 0 <= win && win <= span => win,
            _ => return Some(format!("window {}", field.name)),
        };
        if win != 0 && (marks & (S | W) != 0 || low == high) {
            return Some(format!("window {} on a field that may not vary", field.name));
        }
    }
    None
}

/// Every defect of a founder pool against its validated law; empty when sound.
pub fn validate_pool(law: &Value, lawview: &View, pool: &Value) -> Vec<String> {
    if !keys_are(Some(pool), &["loci", "name", "schema", "version"]) {
        return vec![String::from("pool: keys are not exactly the schema's")];
    }
    let mut out: Vec<String> = Vec::new();
    if int_of(pool.get("schema")) != Some(1) {
        out.push(String::from("pool: schema"));
    }
    if int_of(pool.get("version")).is_none() {
        out.push(String::from("pool: version"));
    }
    let pinned = match law.get("founders") {
        Some(pin @ Value::Obj(_)) => {
            let wanted = pin.get("name").unwrap_or(&Value::Null);
            py_eq(pool.get("name").unwrap_or(&Value::Null), wanted)
        }
        _ => false,
    };
    if !pinned {
        out.push(String::from("pool: name is not the one the law pins"));
    }
    let loci = match arr_of(pool.get("loci")) {
        Some(loci) if loci.len() == lawview.order.len() => loci,
        _ => {
            out.push(String::from("pool: exactly the law's loci"));
            return out;
        }
    };
    for (entry, ident) in loci.iter().zip(lawview.order.iter()) {
        if !keys_are(Some(entry), &["alleles", "locus"]) || int_of(entry.get("locus")) != Some(*ident) {
            out.push(format!("pool: locus {} missing or out of order", ident));
            continue;
        }
        let Some(locus) = lawview.loci.get(ident) else {
            out.push(format!("pool: locus {} missing or out of order", ident));
            continue;
        };
        let alleles = match arr_of(entry.get("alleles")) {
            Some(alleles) if (2..=6).contains(&alleles.len()) => alleles,
            _ => {
                out.push(format!("pool: locus {} holds 2 to 6 alleles", ident));
                continue;
            }
        };
        let mut names: Vec<&str> = Vec::new();
        let mut shapes: Vec<(Vec<i64>, Vec<i64>)> = Vec::new();
        for allele in alleles {
            if let Some(defect) = allele_defect(allele, locus, lawview) {
                out.push(format!("pool: locus {} {}", ident, defect));
                continue;
            }
            let name = str_of(allele.get("name")).unwrap_or("");
            if names.contains(&name) {
                out.push(format!("pool: locus {} allele named twice", ident));
            }
            names.push(name);
            let shape = (ints(allele.get("body")).unwrap_or_default(), ints(allele.get("window")).unwrap_or_default());
            if shapes.contains(&shape) {
                out.push(format!("pool: locus {} two identical alleles", ident));
            }
            shapes.push(shape);
        }
    }
    out
}

/// The pool's alleles by locus id; `None` if a validated pool cannot be read.
pub fn pool_alleles(pool: &Value) -> Option<BTreeMap<i64, Vec<Allele>>> {
    let mut out = BTreeMap::new();
    for entry in arr_of(pool.get("loci"))? {
        let mut alleles = Vec::new();
        for allele in arr_of(entry.get("alleles"))? {
            alleles.push(Allele {
                freq: int_of(allele.get("freq"))?,
                body: ints(allele.get("body"))?,
                window: ints(allele.get("window"))?,
            });
        }
        out.insert(int_of(entry.get("locus"))?, alleles);
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// The codec
// ---------------------------------------------------------------------------

/// One decoded record: its value and its raw body bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub kind: u8,
    pub flags: i64,
    pub locus: i64,
    pub stages: i64,
    pub fields: Vec<i64>,
    pub body: [u8; BODY],
}

fn bad(detail: String) -> Refused {
    refused("bad_request", detail)
}

fn bad_s(detail: &str) -> Refused {
    refused("bad_request", String::from(detail))
}

fn limit(detail: String) -> Refused {
    refused("limit", detail)
}

pub fn engine_panic(detail: &str) -> Refused {
    refused("engine_panic", String::from(detail))
}

/// A check the reference has no refusal for failed: an engine defect, named.
fn decode_panic() -> Refused {
    engine_panic("genome decode")
}

/// Decode and check a genome: its chromosomes of records, or a named refusal.
pub fn decode(data: &[u8], lawview: &View) -> Result<Vec<Vec<Record>>, Refused> {
    if data.len() > lawview.max_bytes {
        return Err(refused("limit", String::from("genome size")));
    }
    let Some((head, mut rest)) = data.split_at_checked(HEADER) else {
        return Err(bad_s("genome header"));
    };
    let [m0, m1, m2, m3, s0, s1, ploidy, pairs] = head else {
        return Err(bad_s("genome header"));
    };
    if [*m0, *m1, *m2, *m3] != MAGIC {
        return Err(bad_s("genome magic"));
    }
    if ((i64::from(*s0) << 8) | i64::from(*s1)) != SCHEMA {
        return Err(bad_s("genome schema"));
    }
    if usize::from(*ploidy) != PLOIDY {
        return Err(bad_s("genome ploidy"));
    }
    if usize::from(*pairs) != lawview.pairs {
        return Err(bad_s("genome pairs"));
    }
    let total = lawview.pairs.checked_mul(PLOIDY).ok_or_else(decode_panic)?;
    let mut chromosomes: Vec<Vec<Record>> = Vec::with_capacity(total);
    let mut presence: Vec<Vec<i64>> = Vec::with_capacity(total);
    for i in 0..total {
        let pair = i.checked_div(PLOIDY).and_then(|p| i64::try_from(p).ok()).ok_or_else(decode_panic)?;
        let Some((count_bytes, after)) = rest.split_at_checked(2) else {
            return Err(bad_s("genome truncated"));
        };
        let [c0, c1] = count_bytes else {
            return Err(bad_s("genome truncated"));
        };
        let count = (usize::from(*c0) << 8) | usize::from(*c1);
        if count > lawview.max_records {
            return Err(limit(format!("genome records c{}", i)));
        }
        let need = count.checked_mul(RECORD).ok_or_else(decode_panic)?;
        let Some((block, after)) = after.split_at_checked(need) else {
            return Err(bad_s("genome truncated"));
        };
        rest = after;
        let mut seen: Vec<i64> = Vec::with_capacity(count);
        let mut run: usize = 0;
        let mut records: Vec<Record> = Vec::with_capacity(count);
        for (j, raw) in block.chunks_exact(RECORD).enumerate() {
            let [kind_code, flags, l0, l1, g0, g1, body @ ..] = raw else {
                return Err(bad_s("genome truncated"));
            };
            let Ok(body) = <[u8; BODY]>::try_from(body) else {
                return Err(bad_s("genome truncated"));
            };
            let kind_code = *kind_code;
            if !(1..=16).contains(&kind_code) {
                return Err(bad(format!("genome kind c{} r{}", i, j)));
            }
            let ident = (i64::from(*l0) << 8) | i64::from(*l1);
            let Some(locus) = lawview.loci.get(&ident) else {
                return Err(bad(format!("genome locus c{} r{}", i, j)));
            };
            if kind_code != locus.kind {
                return Err(bad(format!("genome locus kind c{} r{}", i, j)));
            }
            if (ident >> 8) != pair {
                return Err(bad(format!("genome locus pair c{} r{}", i, j)));
            }
            let flags = i64::from(*flags);
            if flags & FLAG_UNUSED != 0 || flags != locus.flags {
                return Err(bad(format!("genome flags c{} r{}", i, j)));
            }
            let stages = (i64::from(*g0) << 8) | i64::from(*g1);
            if stages != locus.stages {
                return Err(bad(format!("genome stage c{} r{}", i, j)));
            }
            if seen.contains(&ident) {
                return Err(bad(format!("genome duplicate c{} r{}", i, j)));
            }
            seen.push(ident);
            let kind = kind_by_code(kind_code).ok_or_else(decode_panic)?;
            let used = used_bits(kind);
            if body.iter().zip(used.iter()).any(|(byte, mask)| byte & !mask != 0) {
                return Err(bad(format!("genome pad c{} r{}", i, j)));
            }
            if kind.fields.len() != locus.boxes.len() {
                return Err(decode_panic());
            }
            let mut values = Vec::with_capacity(kind.fields.len());
            for (field, (low, high)) in kind.fields.iter().zip(locus.boxes.iter()) {
                let value = read_field(&body, field).ok_or_else(decode_panic)?;
                let refusal = || bad(format!("genome box c{} r{} {}", i, j, field.name));
                if value < *low || value > *high {
                    return Err(refusal());
                }
                if field.marks & P != 0 && !species_legal(value) {
                    return Err(refusal());
                }
                if field.marks & R != 0 && value <= 63 && lawview.is_token(value) {
                    return Err(refusal());
                }
                values.push(value);
            }
            if kind_code == KIND_CIS {
                run = run.checked_add(1).ok_or_else(decode_panic)?;
                if run > lawview.max_promoter {
                    return Err(limit(format!("genome promoter c{} r{}", i, j)));
                }
            } else {
                run = 0;
            }
            records.push(Record { kind: kind_code, flags, locus: ident, stages, fields: values, body });
        }
        chromosomes.push(records);
        presence.push(seen);
    }
    if !rest.is_empty() {
        return Err(bad_s("genome trailing"));
    }
    for ident in &lawview.order {
        let locus = lawview.loci.get(ident).ok_or_else(decode_panic)?;
        if !locus.essential() {
            continue;
        }
        let home = usize::try_from(ident >> 8).ok().and_then(|pair| pair.checked_mul(PLOIDY)).ok_or_else(decode_panic)?;
        for h in 0..PLOIDY {
            let chromosome = home.checked_add(h).and_then(|c| presence.get(c)).ok_or_else(decode_panic)?;
            if !chromosome.contains(ident) {
                return Err(bad(format!("genome essential l{} h{}", ident, h)));
            }
        }
    }
    Ok(chromosomes)
}

/// One record's 16 bytes; every field checked against its type range first.
pub fn encode_record(kind_code: u8, flags: i64, locus: i64, stages: i64, values: &[i64], at: &str) -> Result<[u8; RECORD], Refused> {
    let value_refusal = || bad(format!("genome value {}", at));
    let kind = match kind_by_code(kind_code) {
        Some(kind) if values.len() == kind.fields.len() => kind,
        _ => return Err(value_refusal()),
    };
    for (part, cap) in [(flags, 0xFF), (locus, 0xFFFF), (stages, 0xFFFF)] {
        if !(0..=cap).contains(&part) {
            return Err(value_refusal());
        }
    }
    let mut body = [0_u8; BODY];
    for (field, value) in kind.fields.iter().zip(values.iter()) {
        let (low, high) = field.type_range();
        let box_refusal = || bad(format!("genome box {} {}", at, field.name));
        if *value < low || *value > high {
            return Err(box_refusal());
        }
        write_field(&mut body, field, *value).ok_or_else(box_refusal)?;
    }
    let byte = |value: i64| u8::try_from(value).map_err(|_| value_refusal());
    let [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9] = body;
    Ok([
        kind_code,
        byte(flags)?,
        byte(locus >> 8)?,
        byte(locus & 0xFF)?,
        byte(stages >> 8)?,
        byte(stages & 0xFF)?,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
        b8,
        b9,
    ])
}

pub fn sha256_hex(data: &[u8]) -> String {
    ocj::hex(&Sha256::digest(data))
}

/// True when `text` is an even-length lowercase hexadecimal string.
pub fn is_genome_hex(text: &str) -> bool {
    text.len() & 1 == 0 && text.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// `(len + 63) // 64`: the blocks of 64 bytes a genome occupies, rounded up.
pub fn blocks(len: usize) -> Option<u64> {
    len.checked_add(63).and_then(|n| n.checked_div(64)).and_then(|n| u64::try_from(n).ok())
}

// ---------------------------------------------------------------------------
// Founders and corners
// ---------------------------------------------------------------------------

/// A chassis stream that counts every word it draws, rejections included.
pub struct CountingStream {
    stream: rng::Stream,
    pub draws: u64,
}

impl CountingStream {
    pub fn new(seed: &[u8; 32], domain: &str, index: u64) -> CountingStream {
        CountingStream { stream: rng::Stream::new(seed, domain, index), draws: 0 }
    }

    /// The counting stream seeded by a key already derived, its counter at zero.
    pub fn from_key(k: &[u8; 32]) -> CountingStream {
        CountingStream { stream: rng::Stream::from_key(k), draws: 0 }
    }

    pub fn next_u64(&mut self) -> Option<u64> {
        self.draws = self.draws.checked_add(1)?;
        Some(self.stream.next_u64())
    }

    /// A uniform integer in [0, n) by the chassis rejection rule; `None` for n = 0.
    pub fn below(&mut self, n: u64) -> Option<u64> {
        if n == 0 {
            return None;
        }
        let threshold = rng::below_threshold(n);
        loop {
            let word = self.next_u64()?;
            if word >= threshold {
                return word.checked_rem(n);
            }
        }
    }
}

fn frame(lawview: &View, parts: &[(usize, Vec<u8>)]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&u16::try_from(SCHEMA).ok()?.to_be_bytes());
    out.push(u8::try_from(PLOIDY).ok()?);
    out.push(u8::try_from(lawview.pairs).ok()?);
    for (count, records) in parts {
        out.extend_from_slice(&u16::try_from(*count).ok()?.to_be_bytes());
        out.extend_from_slice(records);
    }
    Some(out)
}

/// A founder genome from a 32-byte seed: its bytes, the chosen allele per record, and the work.
///
/// One stream per (locus, homolog), indexed by `locus*2 + h`: the allele
/// first, then one draw per windowed field in kind order. Adding, removing
/// or reordering loci moves no other locus's draws.
pub fn found(seed: &[u8; 32], lawview: &View, alleles_by_locus: &BTreeMap<i64, Vec<Allele>>) -> Result<(Vec<u8>, Vec<i64>, u64), Refused> {
    let guard = || engine_panic("genome found");
    let mut draws: u64 = 0;
    let mut chosen_all: Vec<i64> = Vec::new();
    let mut parts: Vec<(usize, Vec<u8>)> = Vec::new();
    for pair in 0..lawview.pairs {
        let ids = lawview.chromosomes.get(pair).ok_or_else(guard)?;
        for h in 0..PLOIDY {
            let mut records: Vec<u8> = Vec::new();
            for ident in ids {
                let locus = lawview.loci.get(ident).ok_or_else(guard)?;
                let index = u64::try_from(*ident)
                    .ok()
                    .and_then(|id| id.checked_mul(2))
                    .and_then(|id| id.checked_add(u64::try_from(h).ok()?))
                    .ok_or_else(guard)?;
                let mut stream = CountingStream::new(seed, DOMAIN_FOUNDER, index);
                let alleles = alleles_by_locus.get(ident).ok_or_else(guard)?;
                let mut total: i64 = 0;
                for allele in alleles {
                    total = total.checked_add(allele.freq).ok_or_else(guard)?;
                }
                if total < 1 {
                    return Err(guard());
                }
                let bound = u64::try_from(total).map_err(|_| guard())?;
                let pick = stream.below(bound).and_then(|p| i64::try_from(p).ok()).ok_or_else(guard)?;
                let mut chosen: usize = 0;
                let mut cumulative: i64 = 0;
                for (index, allele) in alleles.iter().enumerate() {
                    cumulative = cumulative.checked_add(allele.freq).ok_or_else(guard)?;
                    if cumulative > pick {
                        chosen = index;
                        break;
                    }
                }
                let allele = alleles.get(chosen).ok_or_else(guard)?;
                let mut values: Vec<i64> = Vec::with_capacity(allele.body.len());
                for (index, base) in allele.body.iter().enumerate() {
                    let win = *allele.window.get(index).ok_or_else(guard)?;
                    let mut value = *base;
                    if win > 0 {
                        let span = win.checked_mul(2).and_then(|n| n.checked_add(1)).ok_or_else(guard)?;
                        let span = u64::try_from(span).map_err(|_| guard())?;
                        let draw = stream.below(span).and_then(|d| i64::try_from(d).ok()).ok_or_else(guard)?;
                        let delta = draw.checked_sub(win).ok_or_else(guard)?;
                        let (low, high) = *locus.boxes.get(index).ok_or_else(guard)?;
                        value = base.checked_add(delta).ok_or_else(guard)?.max(low).min(high);
                    }
                    values.push(value);
                }
                records.extend_from_slice(&encode_record(locus.kind, locus.flags, *ident, locus.stages, &values, "c0 r0")?);
                draws = draws.checked_add(stream.draws).ok_or_else(guard)?;
                chosen_all.push(i64::try_from(chosen).map_err(|_| guard())?);
            }
            parts.push((ids.len(), records));
        }
    }
    let data = frame(lawview, &parts).ok_or_else(guard)?;
    let work = blocks(data.len()).and_then(|b| b.checked_add(draws)).ok_or_else(guard)?;
    Ok((data, chosen_all, work))
}

/// A genome at a corner of the box: all lo (0), all hi (1), or keyed (k >= 2).
pub fn corner(k: i64, lawview: &View) -> Result<(Vec<u8>, u64), Refused> {
    let guard = || engine_panic("genome corner");
    if !(0..=CORNER_MAX).contains(&k) {
        return Err(bad_s("corner"));
    }
    let mut stream = if k >= 2 {
        Some(CountingStream::new(&[0_u8; 32], DOMAIN_CORNER, u64::try_from(k).map_err(|_| guard())?))
    } else {
        None
    };
    let mut parts: Vec<(usize, Vec<u8>)> = Vec::new();
    for pair in 0..lawview.pairs {
        let ids = lawview.chromosomes.get(pair).ok_or_else(guard)?;
        for _h in 0..PLOIDY {
            let mut records: Vec<u8> = Vec::new();
            for ident in ids {
                let locus = lawview.loci.get(ident).ok_or_else(guard)?;
                let mut values: Vec<i64> = Vec::with_capacity(locus.boxes.len());
                for (low, high) in &locus.boxes {
                    let value = match (k, stream.as_mut()) {
                        (0, _) => *low,
                        (1, _) => *high,
                        (_, Some(stream)) => {
                            if stream.next_u64().ok_or_else(guard)? >> 63 == 1 {
                                *high
                            } else {
                                *low
                            }
                        }
                        (_, None) => return Err(guard()),
                    };
                    values.push(value);
                }
                records.extend_from_slice(&encode_record(locus.kind, locus.flags, *ident, locus.stages, &values, "c0 r0")?);
            }
            parts.push((ids.len(), records));
        }
    }
    let data = frame(lawview, &parts).ok_or_else(guard)?;
    let words = stream.map(|s| s.draws).unwrap_or(0);
    let work = blocks(data.len()).and_then(|b| b.checked_add(words)).ok_or_else(guard)?;
    Ok((data, work))
}

// ---------------------------------------------------------------------------
// Strength, for dominance
// ---------------------------------------------------------------------------

/// The first four bytes of the SHA-256 of a body, read as a big-endian u32.
pub fn hash32(body: &[u8]) -> Option<u32> {
    let digest = Sha256::digest(body);
    let [a, b, c, d, ..] = digest.as_slice() else {
        return None;
    };
    Some(u32::from_be_bytes([*a, *b, *c, *d]))
}

/// The key DOM_MAX compares first, in i64 from the decoded values.
pub fn strength(kind_code: u8, values: &[i64], body: &[u8; BODY]) -> Option<i64> {
    match kind_by_code(kind_code)?.strength {
        Strength::Field(index) => values.get(index).copied(),
        Strength::Abs(index) => values.get(index)?.checked_abs(),
        Strength::SumAbs(indices) => {
            let mut total: i64 = 0;
            for index in indices {
                total = total.checked_add(values.get(*index)?.checked_abs()?)?;
            }
            Some(total)
        }
        Strength::Hash => hash32(body).map(i64::from),
    }
}

// ---------------------------------------------------------------------------
// The seeded fuzz: mutated fixture genomes through the byte protocol
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laws;
    use crate::rng::Stream;

    const FUZZ_CASES: usize = 10_000;

    fn fixture_view() -> Option<View> {
        let law = laws::parse_file(laws::law("fixture")?).ok()?;
        if !validate_law(&law).is_empty() {
            return None;
        }
        view(&law)
    }

    fn founder(seed: &[u8; 32]) -> Option<Vec<u8>> {
        let request = format!("{{\"law\":\"fixture\",\"op\":\"genome_found\",\"seed\":\"{}\",\"v\":1}}", ocj::hex(seed));
        let answer = crate::call(request.as_bytes());
        let value = ocj::parse(&answer, false).ok()?;
        ocj::from_hex(str_of(value.get("genome"))?)
    }

    /// Where each chromosome's records start and how many it declares.
    fn spans(genome: &[u8]) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        let mut pos = HEADER;
        while let Some([c0, c1]) = pos.checked_add(2).and_then(|end| genome.get(pos..end)).map(|pair| match pair {
            [a, b] => [*a, *b],
            _ => [0, 0],
        }) {
            let count = (usize::from(c0) << 8) | usize::from(c1);
            let start = pos.saturating_add(2);
            out.push((pos, start, count));
            pos = start.saturating_add(count.saturating_mul(RECORD));
            if pos >= genome.len() {
                break;
            }
        }
        out
    }

    fn pick(stream: &mut Stream, n: usize) -> usize {
        let bound = u64::try_from(n.max(1)).unwrap_or(1);
        usize::try_from(stream.below(bound)).unwrap_or(0)
    }

    fn record_at(genome: &[u8], spans: &[(usize, usize, usize)], stream: &mut Stream) -> Option<(usize, usize, usize)> {
        let (count_at, start, count) = *spans.get(pick(stream, spans.len()))?;
        if count == 0 {
            return None;
        }
        let j = pick(stream, count);
        let at = start.checked_add(j.checked_mul(RECORD)?)?;
        if at.checked_add(RECORD)? > genome.len() {
            return None;
        }
        Some((count_at, at, count))
    }

    fn set_count(genome: &mut [u8], count_at: usize, count: usize) {
        let [hi, lo] = u16::try_from(count).unwrap_or(0).to_be_bytes();
        if let Some(slot) = genome.get_mut(count_at) {
            *slot = hi;
        }
        if let Some(slot) = count_at.checked_add(1).and_then(|at| genome.get_mut(at)) {
            *slot = lo;
        }
    }

    fn mutate(base: &[u8], stream: &mut Stream, lawview: &View) -> Vec<u8> {
        let mut genome = base.to_vec();
        let layout = spans(&genome);
        let len = genome.len();
        match stream.below(13) {
            0 => {
                let at = pick(stream, len);
                let flip = u8::try_from(stream.below(255).saturating_add(1)).unwrap_or(1);
                if let Some(byte) = genome.get_mut(at) {
                    *byte ^= flip;
                }
            }
            1 => genome.truncate(pick(stream, len)),
            2 => {
                for _ in 0..=pick(stream, 24) {
                    genome.push(u8::try_from(stream.below(256)).unwrap_or(0));
                }
            }
            3 => {
                if let (Some((_, a, _)), Some((_, b, _))) = (record_at(&genome, &layout, stream), record_at(&genome, &layout, stream)) {
                    for offset in 0..RECORD {
                        if let (Some(x), Some(y)) = (a.checked_add(offset), b.checked_add(offset)) {
                            genome.swap(x, y);
                        }
                    }
                }
            }
            4 => {
                if let Some((count_at, at, count)) = record_at(&genome, &layout, stream) {
                    genome.drain(at..at.saturating_add(RECORD));
                    set_count(&mut genome, count_at, count.saturating_sub(1));
                }
            }
            5 => {
                if let Some((count_at, at, count)) = record_at(&genome, &layout, stream) {
                    let copy: Vec<u8> = genome.get(at..at.saturating_add(RECORD)).map(|r| r.to_vec()).unwrap_or_default();
                    let target = layout.first().map(|(_, start, _)| *start).unwrap_or(HEADER);
                    let insert_at = if stream.below(2) == 0 { at } else { target.min(genome.len()) };
                    for (offset, byte) in copy.iter().enumerate() {
                        genome.insert(insert_at.saturating_add(offset).min(genome.len()), *byte);
                    }
                    if insert_at <= count_at {
                        set_count(&mut genome, count_at.saturating_add(RECORD), count.saturating_add(1));
                    } else {
                        set_count(&mut genome, count_at, count.saturating_add(1));
                    }
                }
            }
            6 => {
                if let Some((count_at, _, _)) = layout.get(pick(stream, layout.len())).copied() {
                    set_count(&mut genome, count_at, pick(stream, 80));
                }
            }
            7 => {
                genome = (0..pick(stream, 96)).map(|_| u8::try_from(stream.below(256)).unwrap_or(0)).collect();
            }
            8 => {
                let at = pick(stream, HEADER);
                if let Some(byte) = genome.get_mut(at) {
                    *byte = u8::try_from(stream.below(256)).unwrap_or(0);
                }
            }
            9 => {
                if let Some((_, at, _)) = record_at(&genome, &layout, stream) {
                    let bit = u8::try_from(stream.below(8)).unwrap_or(0);
                    if let Some(byte) = at.checked_add(6).and_then(|b| b.checked_add(pick(stream, BODY))).and_then(|b| genome.get_mut(b)) {
                        *byte |= 1 << bit;
                    }
                }
            }
            10 => {
                if let Some((_, at, _)) = record_at(&genome, &layout, stream) {
                    let ident = genome
                        .get(at.saturating_add(2)..at.saturating_add(4))
                        .map(|pair| match pair {
                            [a, b] => (i64::from(*a) << 8) | i64::from(*b),
                            _ => -1,
                        })
                        .unwrap_or(-1);
                    if let (Some(locus), Some(kind)) = (lawview.loci.get(&ident), genome.get(at).and_then(|code| kind_by_code(*code))) {
                        let index = pick(stream, kind.fields.len());
                        if let (Some(field), Some((low, high))) = (kind.fields.get(index), locus.boxes.get(index)) {
                            let value = match stream.below(4) {
                                0 => low.saturating_sub(1),
                                1 => high.saturating_add(1),
                                2 => *low,
                                _ => *high,
                            };
                            let (t_lo, t_hi) = field.type_range();
                            let mut body = [0_u8; BODY];
                            if let Some(bytes) = genome.get(at.saturating_add(6)..at.saturating_add(RECORD)) {
                                body.copy_from_slice(bytes);
                            }
                            if (t_lo..=t_hi).contains(&value) {
                                if field.bits < 8 {
                                    if let Some(slot) = body.get_mut(field.byte) {
                                        *slot &= !u8::try_from(field.mask() << field.shift).unwrap_or(0);
                                    }
                                }
                                if write_field(&mut body, field, value).is_some() {
                                    for (offset, byte) in body.iter().enumerate() {
                                        if let Some(slot) = at.saturating_add(6).checked_add(offset).and_then(|b| genome.get_mut(b)) {
                                            *slot = *byte;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            11 => {
                if let Some((_, at, _)) = record_at(&genome, &layout, stream) {
                    if genome.get(at) == Some(&KIND_CIS) {
                        let token = [1_u8, 33, 34, 35];
                        if let (Some(slot), Some(species)) = (genome.get_mut(at.saturating_add(6)), token.get(pick(stream, token.len()))) {
                            *slot = *species;
                        }
                    }
                }
            }
            _ => {
                if let Some((_, at, _)) = record_at(&genome, &layout, stream) {
                    let record: Vec<u8> = genome.get(at..at.saturating_add(RECORD)).map(|r| r.to_vec()).unwrap_or_default();
                    genome.drain(at..at.saturating_add(RECORD));
                    let to = pick(stream, genome.len().saturating_add(1)).min(genome.len());
                    for (offset, byte) in record.iter().enumerate() {
                        genome.insert(to.saturating_add(offset).min(genome.len()), *byte);
                    }
                }
            }
        }
        genome
    }

    fn answer_of(op: &str, genome: &[u8]) -> Option<Value> {
        let request = format!("{{\"genome\":\"{}\",\"law\":\"fixture\",\"op\":\"{}\",\"v\":1}}", ocj::hex(genome), op);
        let answer = crate::call(request.as_bytes());
        ocj::parse(&answer, false).ok()
    }

    #[test]
    fn ten_thousand_mutated_fixture_genomes_are_each_answered_in_canonical_json() {
        let lawview = fixture_view();
        assert!(lawview.is_some(), "the fixture law validates");
        let Some(lawview) = lawview else { return };
        let mut seeds = Stream::new(&[0_u8; 32], "test.genome.fuzz.seeds", 0);
        let mut bases: Vec<Vec<u8>> = Vec::new();
        for _ in 0..8 {
            let mut seed = [0_u8; 32];
            for chunk in seed.chunks_mut(8) {
                chunk.copy_from_slice(&seeds.next_u64().to_be_bytes());
            }
            if let Some(genome) = founder(&seed) {
                bases.push(genome);
            }
        }
        assert_eq!(bases.len(), 8, "every seed yields a founder");
        let mut stream = Stream::new(&[0_u8; 32], "test.genome.fuzz", 0);
        let (mut accepted, mut refused_count) = (0_usize, 0_usize);
        for case in 0..FUZZ_CASES {
            let Some(base) = bases.get(pick(&mut stream, bases.len())) else { continue };
            let mut genome = mutate(base, &mut stream, &lawview);
            if stream.below(4) == 0 {
                genome = mutate(&genome, &mut stream, &lawview);
            }
            for op in ["genome_decode", "genome_compile"] {
                let value = answer_of(op, &genome);
                assert!(value.is_some(), "case {} {} answered outside OCJ: {}", case, op, ocj::hex(&genome));
                let Some(value) = value else { return };
                assert!(matches!(value, Value::Obj(_)), "case {} {} answered a non-object", case, op);
                if value.get("refused").is_some() {
                    let code = str_of(value.get("refused")).unwrap_or("");
                    assert!(code == "bad_request" || code == "limit", "case {} {} refused {}", case, op, code);
                    refused_count = refused_count.saturating_add(1);
                } else {
                    accepted = accepted.saturating_add(1);
                }
            }
        }
        assert!(accepted > 0 && refused_count > 0, "the fuzz reaches both answers: {} accepted, {} refused", accepted, refused_count);
    }

    #[test]
    fn the_worked_example_record_encodes_as_the_specification_says() {
        let record = encode_record(2, 0x0C, 1, 0x07FF, &[16, 4096, 8192, 4, 1], "c0 r0");
        let expected = [0x02_u8, 0x0c, 0x00, 0x01, 0x07, 0xff, 0x10, 0x10, 0x00, 0x20, 0x00, 0x04, 0x01, 0x00, 0x00, 0x00];
        assert_eq!(record, Ok(expected));
    }
}
