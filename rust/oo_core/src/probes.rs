//! Recall probes over text, in the reference's exact terms, for the text
//! whose every code point this core classes exactly as Python does.
//!
//! The reference is `opti_oignon/memory/probes.py`: Python regular
//! expressions over Unicode classes. The `regex` crate has no lookaround,
//! so each expression is reproduced here as a hand-written scanner that
//! follows the reference's own matching order, backtracking included. The
//! classes the expressions depend on -- `\s`, `\w`, `\d`, `str.lower()` and
//! the negation words matched with case ignored -- are reproduced for the
//! accepted set only: ASCII, Latin-1 Supplement, Latin Extended-A without
//! U+0130 (it lowercases to an ASCII "i" and a combining dot), General
//! Punctuation and the euro sign. A text carrying any other code point is
//! refused with `None`, and the caller runs the reference.
//!
//! The expressions travel with every call and are compared with the ones
//! reproduced here: a pattern changed on the Python side is refused, never
//! answered by a scanner that no longer matches it. The word lists and the
//! coverage threshold travel as data, so they cannot drift at all.

use std::collections::HashSet;

use pyo3::prelude::*;

/// The six module expressions with their flags, then the two bounds of a
/// date or number answer: exactly what the scanners below reproduce.
const PATTERNS: [(&str, u32); 8] = [
    (r"(?<=[.!?])\s+", 32),
    (r"\b\d{4}-\d{2}-\d{2}\b", 32),
    (r"(?<![\w-])\d+(?:[.,]\d+)?(?![\w-])", 32),
    (r"[a-z0-9]+", 32),
    (r"\b[A-Z][a-zA-Z]+\b", 32),
    (r"\b(not|never|no|cannot)\b|n't\b", 34),
    (r"(?<![\w-])", 0),
    (r"(?![\w-])", 0),
];

const NEGATION_WORDS: [&str; 4] = ["not", "never", "no", "cannot"];

fn implemented(patterns: &[(String, u32)]) -> bool {
    patterns.len() == PATTERNS.len()
        && patterns
            .iter()
            .zip(PATTERNS.iter())
            .all(|((pattern, flags), (own, own_flags))| pattern == own && flags == own_flags)
}

fn accepted(c: char) -> bool {
    let u = c as u32;
    (u <= 0x17f && u != 0x130) || (0x2000..=0x206f).contains(&u) || u == 0x20ac
}

fn all_accepted(text: &[char]) -> bool {
    text.iter().all(|&c| accepted(c))
}

/// Python's `\s` on the accepted set, which is also what `str.strip()` removes.
fn is_space(c: char) -> bool {
    matches!(
        c,
        '\t'..='\r' | '\u{1c}'..=' ' | '\u{85}' | '\u{a0}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}'
    )
}

/// Python's `\w` on the accepted set: letters, digits, numerics, underscore.
fn is_word(c: char) -> bool {
    matches!(
        c,
        '0'..='9'
            | 'A'..='Z'
            | 'a'..='z'
            | '_'
            | '\u{aa}'
            | '\u{b2}'
            | '\u{b3}'
            | '\u{b5}'
            | '\u{b9}'
            | '\u{ba}'
            | '\u{bc}'..='\u{be}'
            | '\u{c0}'..='\u{d6}'
            | '\u{d8}'..='\u{f6}'
            | '\u{f8}'..='\u{17f}'
    )
}

/// Python's `\d` on the accepted set: the ASCII digits and nothing else.
fn is_digit(c: char) -> bool {
    c.is_ascii_digit()
}

/// `\b` at `i`: a word character on one side only. An empty text has none.
fn boundary(s: &[char], i: usize) -> bool {
    if s.is_empty() {
        return false;
    }
    let before = i > 0 && is_word(s[i - 1]);
    let after = i < s.len() && is_word(s[i]);
    before != after
}

/// `(?<![\w-])` at `i`.
fn free_before(s: &[char], i: usize) -> bool {
    i == 0 || !(is_word(s[i - 1]) || s[i - 1] == '-')
}

/// `(?![\w-])` at `i`.
fn free_after(s: &[char], i: usize) -> bool {
    i >= s.len() || !(is_word(s[i]) || s[i] == '-')
}

fn strip(s: &[char]) -> &[char] {
    let mut start = 0;
    let mut end = s.len();
    while start < end && is_space(s[start]) {
        start += 1;
    }
    while end > start && is_space(s[end - 1]) {
        end -= 1;
    }
    &s[start..end]
}

/// The reference's `_sentences`: split on a whitespace run that follows `.`,
/// `!` or `?`, strip each piece, keep the pieces that are not empty.
fn sentences(text: &[char]) -> Vec<&[char]> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < text.len() {
        if i > 0 && is_space(text[i]) && matches!(text[i - 1], '.' | '!' | '?') {
            let mut end = i;
            while end < text.len() && is_space(text[end]) {
                end += 1;
            }
            out.push(strip(&text[start..i]));
            start = end;
            i = end;
        } else {
            i += 1;
        }
    }
    out.push(strip(&text[start..]));
    out.retain(|piece| !piece.is_empty());
    out
}

/// `[a-z0-9]+` over `text.lower()`.
fn tokens(text: &[char]) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for &c in text {
        for lowered in c.to_lowercase() {
            if lowered.is_ascii_lowercase() || lowered.is_ascii_digit() {
                current.push(lowered);
            } else if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// `\b\d{4}-\d{2}-\d{2}\b`, left to right, without overlap.
fn dates(s: &[char]) -> Vec<(usize, usize)> {
    let digit = |i: usize| is_digit(s[i]);
    let mut out = Vec::new();
    let mut i = 0;
    while i + 10 <= s.len() {
        let found = boundary(s, i)
            && (0..4).all(|k| digit(i + k))
            && s[i + 4] == '-'
            && digit(i + 5)
            && digit(i + 6)
            && s[i + 7] == '-'
            && digit(i + 8)
            && digit(i + 9)
            && boundary(s, i + 10);
        if found {
            out.push((i, i + 10));
            i += 10;
        } else {
            i += 1;
        }
    }
    out
}

/// The sentence with each date replaced by one space, as `_DATE.sub(" ", ...)`.
fn without_dates(s: &[char], found: &[(usize, usize)]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut last = 0;
    for &(start, end) in found {
        out.extend_from_slice(&s[last..start]);
        out.push(' ');
        last = end;
    }
    out.extend_from_slice(&s[last..]);
    out
}

/// `(?<![\w-])\d+(?:[.,]\d+)?(?![\w-])`, in the reference's backtracking
/// order: the whole run of digits with its fraction, else the whole run
/// alone. A shorter run is always followed by a digit, a word character,
/// so no shorter candidate can pass the lookahead.
fn numbers(s: &[char]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        if is_digit(s[i]) && free_before(s, i) {
            let mut run = i;
            while run < s.len() && is_digit(s[run]) {
                run += 1;
            }
            let mut end = None;
            if run + 1 < s.len() && matches!(s[run], '.' | ',') && is_digit(s[run + 1]) {
                let mut fraction = run + 1;
                while fraction < s.len() && is_digit(s[fraction]) {
                    fraction += 1;
                }
                if free_after(s, fraction) {
                    end = Some(fraction);
                }
            }
            if end.is_none() && free_after(s, run) {
                end = Some(run);
            }
            if let Some(end) = end {
                out.push(s[i..end].iter().collect());
                i = end;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// `\b[A-Z][a-zA-Z]+\b`: only the whole run of letters can end on a boundary.
fn capitalised(s: &[char]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        if s[i].is_ascii_uppercase() && boundary(s, i) {
            let mut end = i + 1;
            while end < s.len() && s[end].is_ascii_alphabetic() {
                end += 1;
            }
            if end > i + 1 && boundary(s, end) {
                out.push(s[i..end].iter().collect());
                i = end;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// `s[i..]` starts with the ASCII `word`, case ignored. On the accepted set
/// no code point but the ASCII letters matches these letters so.
fn starts_with_ignoring_case(s: &[char], i: usize, word: &str) -> bool {
    i + word.len() <= s.len()
        && word
            .chars()
            .enumerate()
            .all(|(k, letter)| s[i + k].is_ascii() && s[i + k].to_ascii_lowercase() == letter)
}

/// The end of a `\b(not|never|no|cannot)\b|n't\b` match at `i`, alternatives
/// tried in the reference's order.
fn negation_at(s: &[char], i: usize) -> Option<usize> {
    if boundary(s, i) {
        for word in NEGATION_WORDS {
            let end = i + word.len();
            if starts_with_ignoring_case(s, i, word) && boundary(s, end) {
                return Some(end);
            }
        }
    }
    if starts_with_ignoring_case(s, i, "n't") && boundary(s, i + 3) {
        return Some(i + 3);
    }
    None
}

/// How many negation matches `findall` returns on the sentence.
fn negations(s: &[char]) -> u64 {
    let mut count = 0;
    let mut i = 0;
    while i < s.len() {
        match negation_at(s, i) {
            Some(end) => {
                count += 1;
                i = end;
            }
            None => i += 1,
        }
    }
    count
}

fn contains(haystack: &[char], needle: &[char]) -> bool {
    needle.is_empty() || haystack.windows(needle.len()).any(|window| window == needle)
}

/// A marker is a substring of `sentence.lower()`.
fn is_decision(sentence: &[char], markers: &[Vec<char>]) -> bool {
    let lowered: Vec<char> = sentence.iter().flat_map(|c| c.to_lowercase()).collect();
    markers.iter().any(|marker| contains(&lowered, marker))
}

/// The distinct tokens of the sentence that are not stopwords.
fn decision_key(sentence: &[char], stopwords: &HashSet<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    tokens(sentence)
        .into_iter()
        .filter(|token| !stopwords.contains(token) && seen.insert(token.clone()))
        .collect()
}

/// `(?<![\w-])` + the literal + `(?![\w-])`, anywhere in the text.
fn bounded_literal(text: &[char], literal: &[char]) -> bool {
    if literal.len() > text.len() {
        return false;
    }
    (0..=text.len() - literal.len()).any(|i| {
        text[i..i + literal.len()] == *literal && free_before(text, i) && free_after(text, i + literal.len())
    })
}

/// A probe as drawn: `(turn index, kind, answer, decision key, negations)`.
type Drawn = (usize, String, String, Vec<String>, u64);

/// The probes the reference would draw from these turn texts, in its order,
/// or `None` when a pattern or a code point is not one this core reproduces.
#[pyfunction]
pub fn probe_generate(
    texts: Vec<String>,
    patterns: Vec<(String, u32)>,
    not_entities: Vec<String>,
    stopwords: Vec<String>,
    markers: Vec<String>,
) -> Option<Vec<Drawn>> {
    if !implemented(&patterns) {
        return None;
    }
    let texts: Vec<Vec<char>> = texts.iter().map(|text| text.chars().collect()).collect();
    if !texts.iter().all(|text| all_accepted(text)) {
        return None;
    }
    let not_entities: HashSet<String> = not_entities.into_iter().collect();
    let stopwords: HashSet<String> = stopwords.into_iter().collect();
    let markers: Vec<Vec<char>> = markers.iter().map(|marker| marker.chars().collect()).collect();
    let mut out: Vec<Drawn> = Vec::new();
    for (index, text) in texts.iter().enumerate() {
        for sentence in sentences(text) {
            let mut seen: HashSet<(&str, String)> = HashSet::new();
            let found = dates(sentence);
            for &(start, end) in &found {
                let date: String = sentence[start..end].iter().collect();
                if seen.insert(("date", date.clone())) {
                    out.push((index, "date".into(), date, Vec::new(), 0));
                }
            }
            for number in numbers(&without_dates(sentence, &found)) {
                if seen.insert(("number", number.clone())) {
                    out.push((index, "number".into(), number, Vec::new(), 0));
                }
            }
            for name in capitalised(sentence) {
                if not_entities.contains(&name.to_ascii_lowercase()) || !seen.insert(("entity", name.clone())) {
                    continue;
                }
                out.push((index, "entity".into(), name, Vec::new(), 0));
            }
            if is_decision(sentence, &markers) {
                let key = decision_key(sentence, &stopwords);
                if !key.is_empty() {
                    out.push((index, "decision".into(), sentence.iter().collect(), key, negations(sentence)));
                }
            }
        }
    }
    Some(out)
}

/// A probe as scored: `(kind, answer, decision key, negations)`.
type Scored = (String, String, Vec<String>, i64);

/// The indices of the probes the text does not answer, in order, or `None`
/// when a pattern, a code point or a probe is not one this core takes.
#[pyfunction]
pub fn probe_score(probes: Vec<Scored>, text: String, patterns: Vec<(String, u32)>, coverage: f64) -> Option<Vec<usize>> {
    if !implemented(&patterns) {
        return None;
    }
    let text: Vec<char> = text.chars().collect();
    if !all_accepted(&text) {
        return None;
    }
    let pieces = sentences(&text);
    let piece_words: Vec<HashSet<String>> = pieces.iter().map(|piece| tokens(piece).into_iter().collect()).collect();
    let piece_negations: Vec<u64> = pieces.iter().map(|piece| negations(piece)).collect();
    let words: HashSet<String> = tokens(&text).into_iter().collect();
    let mut failing = Vec::new();
    for (index, (kind, answer, key, wanted)) in probes.iter().enumerate() {
        let answered = match kind.as_str() {
            "decision" => {
                if key.is_empty() {
                    return None;
                }
                piece_words.iter().zip(&piece_negations).any(|(present, &found)| {
                    let shared = key.iter().filter(|word| present.contains(*word)).count();
                    shared as f64 / key.len() as f64 >= coverage && found as i64 == *wanted
                })
            }
            "date" | "number" => bounded_literal(&text, &answer.chars().collect::<Vec<char>>()),
            _ => {
                let answer: Vec<char> = answer.chars().collect();
                if !all_accepted(&answer) {
                    return None;
                }
                let lowered: String = answer.iter().flat_map(|c| c.to_lowercase()).collect();
                words.contains(&lowered)
            }
        };
        if !answered {
            failing.push(index);
        }
    }
    Some(failing)
}

/// Every accepted code point with its classes as this core holds them:
/// `(code point, \s, \w, \d, lowercase)`. The contracts hold each row
/// against Python's own answer.
#[pyfunction]
pub fn probe_text_classes() -> Vec<(u32, bool, bool, bool, String)> {
    (0u32..=0x10ffff)
        .filter_map(char::from_u32)
        .filter(|&c| accepted(c))
        .map(|c| (c as u32, is_space(c), is_word(c), is_digit(c), c.to_lowercase().collect()))
        .collect()
}
