#!/usr/bin/env python3
"""Recall probes: grounded questions a compressed memory must still answer.

A summary may replace verbatim text only once it has been shown to answer for
it. The instrument is a set of probes drawn from the source span -- the
entities it names, the numbers and dates it states, the decisions it records
-- each carrying the turn it came from, and a scorer that says whether a
candidate text answers them. Deterministic, standard library only, and free
of any model: the librarian may later add richer probes on the host, but the
gate itself has to be provable here.

Two rules of the silent-zero family are load-bearing. A generator that finds
nothing on a rich span is a defect: the contracts hold a rich fixture
against it, and the gate holds every span to it, reading the span's facts
again and counting those no probe asks for. A scorer with no probes to score
reports an unknown rate, never 0.0: an absence of questions is not a total
failure.

Only the onion's own modules import this one -- the gate to score a
candidate and count its span's facts, the selection to read a query's
words, the librarian to read a turn's origin, to give the summariser its
code as markers and to find a block by its key -- and nothing else on the
chat path does: the executor, the agent, the routes and the package facade
reach no part of it, and a contract on the tree says so.

A turn is drawn from piece by piece. Each turn declares its origin -- who
wrote its words -- and, when it mixes sources, segments that bound its
parts; each part is a piece with its own base, a turn without segments is
one piece of its own origin, and a character no segment covers is never
drawn from. A declaration outside the grammar is read as legacy, the least
trusted origin. Every probe carries the origin of its piece and the role of
its turn. Only typed text decides: a decision probe is drawn from a typed
piece and from nothing else -- not from the assistant's words, a document,
a refined question or a turn of unknown origin, which yield their dates,
numbers, names and code only.

A decision need not say that it is one. Besides a marker -- "decided", "on
va" -- a typed sentence decides when an act opens it, an instruction, or
when a subject of the act's language stands before the act, within the
reach that language declares. Subjects, reach and acts come from the
decision lexicon of ``onion.yaml``, each act in a class and a language;
without a lexicon only a marker decides. A decision is asked for by its
key: its content words, the class of each of its acts, and its dates and
numbers in their canonical form. A summary keeps it with any word of the
same classes, in either language, and any writing of its dates and
numbers; it loses it with another act, a date moved or a number changed,
however many of its words it keeps.

A text is read as blocks before it is read as sentences: prose paragraphs,
list items without their bullet or number, and fenced code. A fenced block
is an artifact, not sentences: its key is a digest of its body, its one
probe is answered by its marker or by the same block verbatim, and nothing
inside it is probed. Inline code yields no date, number or entity and sets
no polarity; a decision keeps its words.

A date or a quantity is asked for in a canonical form, so that a summary
may write it otherwise and still keep it. A date written in French or in
English, its month named or in figures, reads to a day, a month of a year
or a day of a month: 2026-10-05, 2026-10, --10-05; a relative date is never
read. A number reads to its value as written -- a decimal comma is a point,
the separators of thousands go, the written precision stays -- and a unit
after it to its class in a closed table, where Go is GB and nothing is ever
converted. A writing that reads two ways, 05/10/2026 or 1,500, is never
guessed at: its probe asks for the same writing. The candidate is read the
same way, and the words of a date or a quantity are never names.

A name is read by the span it stands in. Every word at the head of a
sentence or an item takes a capital, so there a capital names nothing unless
the span capitalises the same word off a head; a day of the week is a
relative date, never a name. Names that follow one another a space apart
are one entity, and a part of it that the span goes on to use alone is its
alias: it draws nothing of its own and answers for the whole. The span is
the only source of aliases; a part used before the full name, or never
alone, is none.

The scanning has a second implementation in the native core, asked for at
the call and never at import, and only when it declares the version of the
generator this module is: a twin of another version is never asked. It
answers only for text whose every code point it classes exactly as Python
does, and only for the expressions below as they are written: the
expressions, the tables, the word lists and the coverage threshold travel
with each call, and anything the core does not take is scanned here. This
module stays the reference the core is held to. Every draw and every score
is counted, served by the core or sent to the reference with its reason
(``native_share``): a twin that never answers is a reading, never a silence.
"""

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field

checkpoint_before_apply = True

# What the generator draws, as a number: raised whenever it draws otherwise,
# so a figure names the generator that drew it and a native twin answers only
# for the generator it reproduces. 1 read sentences; 2 reads blocks first;
# 3 reads dates and quantities to a canonical form; 4 reads names by their
# span and decisions from typed text only; 5 reads a decision without a
# marker by its lexicon, and keys it by its acts, dates and numbers; 6 matches
# a decision's markers folded, as its acts are; 7 reads an accented marker
# only where it opens a word, and a negation written as a word in inline code
# as the negation it is.
GENERATOR_VERSION = 7

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# The blanks a date or a quantity may hold: a space, a no-break space and a
# narrow no-break space. The minus sign stands beside the hyphen.
_BLANK = " " + chr(0xA0) + chr(0x202F)
_MINUS = chr(0x2212)
# A date as it is written: a day of the month, a month by name or in
# figures, a year of four figures. What an expression finds is a candidate:
# its bounds, its month word and its day are checked before it is read.
_DAY = r"(?P<d>3[01]|[12][0-9]|0?[1-9])(?![0-9])(?P<suffix>er|st|nd|rd|th)?"
_MONTH_WORD = r"(?P<m>[^\W\d_]+)(?P<dot>\.)?"
_YEAR = r"(?P<y>[1-9][0-9]{3})(?![0-9])"
_SP = "[" + _BLANK + "]+"
_DATE_FORMS = (
    ("dmy", re.compile(_DAY + _SP + "(?:(?P<of>of)" + _SP + ")?" + _MONTH_WORD + "(?P<comma>,)?" + _SP + _YEAR)),
    ("mdy", re.compile(_MONTH_WORD + _SP + _DAY + "(?P<comma>,)?" + _SP + _YEAR)),
    ("dm", re.compile(_DAY + _SP + "(?:(?P<of>of)" + _SP + ")?" + _MONTH_WORD)),
    ("md", re.compile(_MONTH_WORD + _SP + _DAY)),
    ("my", re.compile(_MONTH_WORD + "(?P<comma>,)?" + _SP + _YEAR)),
    ("iso", re.compile(_YEAR + "-(?P<m>[0-9]{2})-(?P<d>[0-9]{2})(?![0-9])")),
    ("iso_month", re.compile(_YEAR + "-(?P<m>[0-9]{2})(?![0-9])")),
    ("ymd", re.compile(_YEAR + "/(?P<m>[0-9]{1,2})/(?P<d>[0-9]{1,2})(?![0-9])")),
    ("numeric", re.compile("(?P<a>[0-9]{1,2})(?P<sep>[/.-])(?P<b>[0-9]{1,2})(?P=sep)" + _YEAR)),
)
# A number as written: figures, and between them the separators the two
# languages use -- a point, a comma, and a blank before a group of three --
# with a sign when one stands before it on its own.
_NUMBER = re.compile(
    "(?P<sign>[-" + _MINUS + "])?"
    "(?P<run>(?:[1-9][0-9]{0,2}(?:[" + _BLANK + "][0-9]{3}(?![0-9]))+|[0-9]+)(?:[.,][0-9]+)*)"
)
# The words a unit may be written in, one or two, after a number.
_UNIT_WORD = re.compile(r"([^\W\d_]+)(?: ([^\W\d_]+))?")
# A word is a run of letters and digits, accents included, so that a French
# word stays whole; on ASCII text it is exactly ``[a-z0-9]+``.
_WORD = re.compile(r"[^\W_]+")
# A name is a run of two letters or more between word boundaries whose first
# letter is a capital -- any capital, so that a name carrying an accent, at
# its head or inside it, is a name. The expression reads the run; the capital
# is checked on the match. On ASCII text this draws exactly what
# ``\b[A-Z][a-zA-Z]+\b`` drew.
_CAPITALISED = re.compile(r"\b[^\W\d_]{2,}\b")
# The first letter or figure of a sentence: where its head word starts.
_HEAD = re.compile(r"[^\W_]")
# English and French. Both halves of "ne ... pas" count, so that "ne ... plus"
# and the familiar form without "ne" are negations too; a summary that writes
# the other form fails its decision probe, and the verbatim stays -- the safe
# side. An apostrophe is the ASCII one or the typographic one.
_NEGATION = re.compile(
    r"\b(not|never|no|cannot|ne|pas|jamais|rien|aucun|aucune)\b|n['\u2019]t\b|\bn['\u2019](?=\w)",
    re.IGNORECASE,
)
# The bounds of a date or number answer, named so the native core can be
# handed the exact expression it has to reproduce.
_ANSWER_BEFORE = r"(?<![\w-])"
_ANSWER_AFTER = r"(?![\w-])"
# A fence opens on three backticks or three tildes or more, indented three
# spaces at most, and closes on a fence of the same character at least as
# long; a backtick fence whose info string holds a backtick is no fence.
_FENCE_OPEN = re.compile(r" {0,3}(`{3,}|~{3,})([^\n]*)")
_FENCE_CLOSE = re.compile(r" {0,3}(`{3,}|~{3,})[ \t]*")
# A list item: a bullet (hyphen, star, plus or the bullet sign) or a number of
# nine digits at most with a period or a parenthesis, then a blank.
_ITEM = re.compile(r"[ \t]*(?:[-*+" + chr(0x2022) + r"]|\d{1,9}[.)])[ \t]+(?=\S)")
# A word: letters, joined by apostrophes ("won't", "n'est"). A negation
# written so inside inline code counts; one fused into a flag does not.
_WORDLIKE = re.compile(r"[^\W\d_]+(?:['" + chr(0x2019) + r"][^\W\d_]+)*")
# Inline code: a run of backticks closed by a run of the same length.
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.DOTALL)
# A code block's key: this many hexadecimal digits of the SHA-256 of its body.
CODE_KEY_LENGTH = 12

_QUESTIONS = {
    "date": "Which date is stated in {}?",
    "number": "Which number is stated in {}?",
    "entity": "Who or what is named in {}?",
    "decision": "What was decided in {}?",
    "code": "Which code block is quoted in {}?",
}

# Markers that make a sentence a recorded decision.
_DECISION_MARKERS = (
    "decided", "agreed", "will ", "must", "chose", "plan to", "shall",
    "d\u00e9cid", "d\u00e9cision", "convenu", "choisi", "opt\u00e9", "on va ", "nous allons ", "je vais ",
    "il faut", "doit", "devons", "devez", "doivent", "pr\u00e9vu de",
)

# The words a decision is said with come from the ``decisions`` section of
# ``onion.yaml``, language by language: subjects, the most words a subject
# may stand before an act, and the acts of each class, as lemmas the rules
# below inflect or as forms listed as written. Every word is lower-case
# ASCII, its accents folded.
_LEXICON_LANGUAGES = ("fr", "en")
_LEXICON_KEYS = ("subjects", "reach", "acts", "as_written")
_LEXICON_NAME = re.compile(r"[a-z]+")
_LEXICON_MEMBER = re.compile(r"[a-z]+(?: [a-z]+)*")
# The endings of a French verb after its stem, first group (-er) and second
# (-ir): infinitive, present, participles, present participle, imperfect,
# future, conditional and, for -ir, the subjunctive where it differs from the
# present. The past historic is left out. A stem in -g takes an e before a
# and o: mangeons, mangeait.
_FR_ER = (
    "er", "e", "es", "ons", "ez", "ent", "ee", "ees", "ant", "ais", "ait", "ions", "iez", "aient",
    "erai", "eras", "era", "erons", "erez", "eront", "erais", "erait", "erions", "eriez", "eraient",
)
_FR_IR = (
    "ir", "is", "it", "issons", "issez", "issent", "i", "ie", "ies", "issant", "issais", "issait", "issions",
    "issiez", "issaient", "isse", "isses", "irai", "iras", "ira", "irons", "irez", "iront", "irais", "irait",
    "irions", "iriez", "iraient",
)
# An English verb whose last consonant follows one vowel, itself after a
# consonant or nothing, doubles it before -ing and -ed; w, x and y never
# double. A syllable is a run of vowels.
_EN_VOWELS = "aeiou"
_EN_CLOSED = re.compile(r"(?:[a-z]*[^aeiou])?[aeiou][bcdfghjklmnpqrstvz]")
_EN_SYLLABLE = re.compile(r"[aeiou]+")
# A lexicon is named on every figure drawn with it by the first hexadecimal
# digits of the SHA-256 of its canonical form.
LEXICON_FINGERPRINT_LENGTH = 12

# Capitalised words that are not names of anything.
_NOT_ENTITIES = frozenset({
    "the", "a", "an", "we", "i", "you", "he", "she", "it", "they", "this",
    "that", "these", "those", "our", "your", "their", "my", "in", "on", "at",
    "and", "but", "or", "if", "when", "then", "there", "here", "yes", "no",
    "le", "la", "les", "un", "une", "des", "du", "nous", "vous", "il", "elle",
    "ils", "elles", "je", "tu", "ce", "cet", "cette", "ces", "mais", "et", "ou",
    "donc", "alors", "si", "quand", "puis", "ensuite", "oui", "non",
    "o\u00f9", "l\u00e0", "\u00e7a", "d\u00e9j\u00e0", "apr\u00e8s", "tr\u00e8s", "\u00e9t\u00e9",
})

# The days of the week, English and French, folded: a relative date, never a
# name, wherever the capital stands.
_WEEKDAYS = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
})

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "we", "i",
    "you", "he", "she", "it", "they", "this", "that", "our", "your", "their",
    "not", "never", "no", "cannot", "do", "does", "did", "don", "doesn", "t",
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "en", "au",
    "aux", "pour", "par", "sur", "dans", "avec", "sans", "nous", "vous", "il",
    "elle", "ils", "elles", "je", "tu", "ce", "cet", "cette", "ces", "que", "qui",
    "est", "sont", "ont", "avons", "avez", "ai", "ne", "pas", "jamais", "rien",
    "aucun", "aucune", "l", "d", "n", "s", "c", "j", "qu",
    "\u00e0", "o\u00f9", "l\u00e0", "\u00e7a", "d\u00e9j\u00e0", "apr\u00e8s", "tr\u00e8s", "\u00e9t\u00e9",
})

# Month names and abbreviations folded to lower-case ASCII, the month each
# names, and how it has to be written: "fr" a French name, in lower case or
# with a capital; "en" an English name, with its capital; "fr." a French
# abbreviation, which takes its period; "en." an English one, with its
# capital, its period or not. A word written otherwise names no month.
_MONTHS = (
    ("janvier", 1, "fr"), ("fevrier", 2, "fr"), ("mars", 3, "fr"), ("avril", 4, "fr"), ("mai", 5, "fr"),
    ("juin", 6, "fr"), ("juillet", 7, "fr"), ("aout", 8, "fr"), ("septembre", 9, "fr"), ("octobre", 10, "fr"),
    ("novembre", 11, "fr"), ("decembre", 12, "fr"),
    ("january", 1, "en"), ("february", 2, "en"), ("march", 3, "en"), ("april", 4, "en"), ("may", 5, "en"),
    ("june", 6, "en"), ("july", 7, "en"), ("august", 8, "en"), ("september", 9, "en"), ("october", 10, "en"),
    ("november", 11, "en"), ("december", 12, "en"),
    ("janv", 1, "fr."), ("fevr", 2, "fr."), ("fev", 2, "fr."), ("avr", 4, "fr."), ("juil", 7, "fr."),
    ("sept", 9, "fr."), ("oct", 10, "fr."), ("nov", 11, "fr."), ("dec", 12, "fr."),
    ("jan", 1, "en."), ("feb", 2, "en."), ("mar", 3, "en."), ("apr", 4, "en."), ("jun", 6, "en."),
    ("jul", 7, "en."), ("aug", 8, "en."), ("sep", 9, "en."), ("sept", 9, "en."), ("oct", 10, "en."),
    ("nov", 11, "en."), ("dec", 12, "en."),
)

# Units after a number and the class each reads as. Two quantities are the
# same only with the same value and the same class, and no class is ever
# converted into another, not even exactly: 90 min is not 1.5 h here. A
# writing joins a class only when it has one meaning after a number; m (a
# metre or a million), s (a second or a decade), g, $ (some dollar) and the
# bits written with a lower-case b (often meant as bytes) are classes of
# their own. Symbols are matched as written, case included: Gb is not GB.
# Each row is a class, then its writings.
_UNIT_SYMBOLS = (
    ("kB", "kB", "ko"), ("KB", "KB", "Ko"), ("MB", "MB", "Mo"), ("GB", "GB", "Go"), ("TB", "TB", "To"),
    ("KiB", "KiB", "Kio"), ("MiB", "MiB", "Mio"), ("GiB", "GiB", "Gio"), ("TiB", "TiB", "Tio"),
    ("kbit", "kbit"), ("Mbit", "Mbit"), ("Gbit", "Gbit"), ("kb", "kb"), ("Mb", "Mb"), ("Gb", "Gb"),
    ("MB/s", "MB/s", "Mo/s"), ("GB/s", "GB/s", "Go/s"), ("Mbit/s", "Mbit/s", "Mbps"), ("Gbit/s", "Gbit/s", "Gbps"),
    ("Hz", "Hz"), ("kHz", "kHz"), ("MHz", "MHz"), ("GHz", "GHz"),
    ("W", "W"), ("kW", "kW"), ("Wh", "Wh"), ("kWh", "kWh"),
    ("ms", "ms"), ("s", "s"), ("second", "sec"), ("min", "min", "mn"), ("h", "h"),
    ("mm", "mm"), ("cm", "cm"), ("m", "m"), ("km", "km"), ("km/h", "km/h"),
    ("m2", "m" + chr(0xB2), "m2"), ("km2", "km" + chr(0xB2), "km2"),
    ("mL", "mL", "ml"), ("L", "L", "l"),
    ("mg", "mg"), ("g", "g"), ("kg", "kg"),
    ("%", "%"), ("EUR", chr(0x20AC), "EUR"), ("USD", "USD"), ("$", "$"),
    ("degC", chr(0xB0) + "C"), ("degF", chr(0xB0) + "F"),
)
# The same, written as words: in any case, with or without their accents,
# folded here to lower-case ASCII.
_UNIT_WORDS = (
    ("byte", "octet", "octets", "byte", "bytes"),
    ("kB", "kilooctet", "kilooctets", "kilobyte", "kilobytes"),
    ("MB", "megaoctet", "megaoctets", "megabyte", "megabytes"),
    ("GB", "gigaoctet", "gigaoctets", "gigabyte", "gigabytes"),
    ("TB", "teraoctet", "teraoctets", "terabyte", "terabytes"),
    ("KiB", "kibioctet", "kibioctets", "kibibyte", "kibibytes"),
    ("MiB", "mebioctet", "mebioctets", "mebibyte", "mebibytes"),
    ("GiB", "gibioctet", "gibioctets", "gibibyte", "gibibytes"),
    ("TiB", "tebioctet", "tebioctets", "tebibyte", "tebibytes"),
    ("ms", "milliseconde", "millisecondes", "millisecond", "milliseconds"),
    ("second", "seconde", "secondes", "second", "seconds"),
    ("min", "minute", "minutes"),
    ("h", "heure", "heures", "hour", "hours"),
    ("day", "jour", "jours", "day", "days"),
    ("week", "semaine", "semaines", "week", "weeks"),
    ("month", "mois", "month", "months"),
    ("year", "ans", "annee", "annees", "year", "years"),
    ("mm", "millimetre", "millimetres", "millimeter", "millimeters"),
    ("cm", "centimetre", "centimetres", "centimeter", "centimeters"),
    ("metre", "metre", "metres", "meter", "meters"),
    ("km", "kilometre", "kilometres", "kilometer", "kilometers"),
    ("m2", "metre carre", "metres carres", "square metre", "square metres", "square meter", "square meters"),
    ("mL", "millilitre", "millilitres", "milliliter", "milliliters"),
    ("L", "litre", "litres", "liter", "liters"),
    ("mg", "milligramme", "milligrammes", "milligram", "milligrams"),
    ("gram", "gramme", "grammes", "gram", "grams"),
    ("kg", "kilogramme", "kilogrammes", "kilogram", "kilograms", "kilo", "kilos"),
    ("W", "watt", "watts"), ("kW", "kilowatt", "kilowatts"),
    ("Hz", "hertz"), ("kHz", "kilohertz"), ("MHz", "megahertz"), ("GHz", "gigahertz"),
    ("%", "pour cent", "pourcent", "percent", "per cent"),
    ("EUR", "euro", "euros"),
    ("$", "dollar", "dollars"),
    ("degC", "degre celsius", "degres celsius", "degree celsius", "degrees celsius"),
    ("degF", "degree fahrenheit", "degrees fahrenheit"),
)
# A currency written before its number.
_UNIT_PREFIXES = (("$", "$"), ("EUR", chr(0x20AC)))
_SYMBOL_CLASSES = tuple(sorted(
    ((written, unit) for unit, *writings in _UNIT_SYMBOLS for written in writings), key=lambda pair: -len(pair[0])
))
_WORD_CLASSES = {written: unit for unit, *writings in _UNIT_WORDS for written in writings}

# The share of a decision's content words a sentence must carry to count as
# the same decision. Below it, the decision is absent, not merely reworded.
DECISION_COVERAGE = 0.7
# The elements of a decision's key a sentence must carry whatever its
# coverage: each act's class, each date and each number. Another act, or a
# date moved by a day, is another decision, not the same one reworded.
_REQUIRED = ("act:", "date:", "number:")

# The turn-origin grammar of conversation.py, as one text: this module is
# loaded alone where it is tested, and a contract holds the copies to one
# text. Here it reads, it never refuses: a declaration outside it is legacy.
_ORIGIN_BASES = ("typed", "refined", "document", "assistant", "legacy")
_ORIGIN_FLAGS = ("tool", "web")
_ORIGIN_ROLES = {
    "user": ("typed", "refined", "document", "legacy"),
    "assistant": ("assistant", "legacy"),
}


def _origin_defect(role, origin, segments, length):
    """Why a turn's origin lies outside the grammar, or None when it lies inside."""
    if not isinstance(origin, str) or not origin:
        return "an origin is a non-empty string"
    base, *flags = origin.split("+")
    if base not in _ORIGIN_BASES:
        return f"origin base {base[:24]!r} is not in the grammar"
    for flag in flags:
        if flag not in _ORIGIN_FLAGS:
            return f"origin flag {flag[:24]!r} is not in the grammar"
    if flags != sorted(set(flags)):
        return "origin flags are written once each, in order"
    if flags and base != "assistant":
        return f"a {base} origin carries no flag"
    allowed = _ORIGIN_ROLES.get(role, ("legacy",)) if isinstance(role, str) else ("legacy",)
    if base not in allowed:
        return f"role {str(role)[:24]!r} cannot carry {base}"
    if not isinstance(segments, (list, tuple)):
        return "segments are a list"
    if segments and base == "legacy":
        return "a legacy turn has no segments"
    end = 0
    for segment in segments:
        if not isinstance(segment, (list, tuple)) or len(segment) != 3:
            return "a segment is [start, end, base]"
        start, stop, label = segment
        if type(start) is not int or type(stop) is not int:
            return "segment bounds are integers"
        if not end <= start < stop <= length:
            return f"segment [{start}, {stop}] overlaps, is empty or leaves the content"
        if label == "legacy" or label not in allowed:
            return f"role {str(role)[:24]!r} carries no {str(label)[:24]} segment"
        end = stop
    return None


def read_origin(turn):
    """A turn's origin and segments as the grammar admits them, with the defect that set them aside.

    Returns ``(origin, segments, defect)``. A turn that declares nothing is
    legacy: it predates origins. A declaration outside the grammar -- a base
    or flag it does not know, a role that cannot carry it, a segment out of
    bounds -- is legacy with no segment, and ``defect`` names the rule it
    broke; the declaration is never guessed at.
    """
    text = str(turn.get("text", "") or "")
    origin = turn.get("origin", "legacy")
    segments = turn.get("segments", [])
    defect = _origin_defect(turn.get("role"), origin, segments, len(text))
    if defect is not None:
        return "legacy", [], defect
    return origin, [[start, stop, label] for start, stop, label in segments], None


def _tokens(text):
    return _WORD.findall(text.lower())


def code_key(body):
    """The key of a code block: the first hexadecimal digits of the SHA-256 of its body."""
    return hashlib.sha256(body.encode("utf-8", "surrogatepass")).hexdigest()[:CODE_KEY_LENGTH]


def code_marker(body):
    """The marker that stands for a code block where its body is not given."""
    return f"[code:{code_key(body)}]"


@dataclass(frozen=True)
class Block:
    """One part of a text: prose, a list item without its marker, or a fenced code block.

    ``start`` and ``end`` bound the part in the text: a code block's bounds
    hold its fences, an item's begin after its marker. ``info`` is a code
    block's info string, its language when it names one.
    """

    kind: str
    text: str
    start: int
    end: int
    info: str = ""

    @property
    def key(self):
        return code_key(self.text) if self.kind == "code" else ""


def segment(text):
    """The blocks of a text, in order.

    A fence opens a code block that runs to the next fence of its character
    at least as long, or to the end of the text. A bullet or a number opens
    an item that the following lines continue; any other line opens a
    paragraph or continues the open one. A blank line, a fence or the next
    item closes it. Lines are split on a line feed only; a carriage return
    before it stays the line's own.
    """
    lines, pos = [], 0
    for line in text.split("\n"):
        lines.append((pos, line))
        pos += len(line) + 1
    blocks, unit, i = [], None, 0
    while i < len(lines):
        start, line = lines[i]
        fence = _FENCE_OPEN.fullmatch(line.rstrip("\r"))
        if fence and not (fence.group(1)[0] == "`" and "`" in fence.group(2)):
            if unit is not None:
                blocks.append(Block(*unit))
                unit = None
            mark, body, j = fence.group(1), [], i + 1
            while j < len(lines):
                closing = _FENCE_CLOSE.fullmatch(lines[j][1].rstrip("\r"))
                if closing and closing.group(1)[0] == mark[0] and len(closing.group(1)) >= len(mark):
                    break
                body.append(lines[j][1])
                j += 1
            end = lines[j][0] + len(lines[j][1]) if j < len(lines) else len(text)
            blocks.append(Block("code", "\n".join(body), start, end, fence.group(2).strip()))
            i = j + 1
            continue
        item = _ITEM.match(line)
        if not line.strip():
            if unit is not None:
                blocks.append(Block(*unit))
                unit = None
        elif item:
            if unit is not None:
                blocks.append(Block(*unit))
            unit = ("item", line[item.end():], start + item.end(), start + len(line))
        elif unit is None:
            unit = ("prose", line, start, start + len(line))
        else:
            unit = (unit[0], unit[1] + "\n" + line, unit[2], start + len(line))
        i += 1
    if unit is not None:
        blocks.append(Block(*unit))
    return blocks


def code_blocks(text):
    """The fenced code blocks of a text, in order."""
    return [b for b in segment(text) if b.kind == "code"]


def mask_code(text):
    """The text with each fenced code block, fences included, replaced by its marker; nothing else moves."""
    out, cursor = [], 0
    for block in code_blocks(text):
        out.append(text[cursor:block.start])
        out.append(code_marker(block.text))
        cursor = block.end
    out.append(text[cursor:])
    return "".join(out)


def _plain(sentence):
    """A sentence without its inline code: what dates, numbers, entities and polarity are read from.

    The negations inline code holds as words of their own stay, as those
    words ("not", "will not", "won't", "pas"): a negation in backticks
    inverts as a plain one does. Any other inline code reads as no word, and
    so does a negation fused into a flag, such as ``--no-cache``.
    """
    def read(code):
        kept = [word for word in _words_of(code.group(2)) if _NEGATION.search(word)]
        return f" {' '.join(kept)} " if kept else " "

    return _INLINE_CODE.sub(read, sentence)


def _words_of(text):
    """The words of a text, each letters joined by apostrophes, trimmed of the punctuation around it.

    A token of anything else -- a flag such as ``--no-cache``, a path, a
    number -- is no word: a negation fused into it is none.
    """
    return [word for word in (token.strip(".,;:!?()\"'") for token in text.split()) if _WORDLIKE.fullmatch(word)]


def _units(text):
    """The sentences of a text's prose and items, block by block; a code block holds none."""
    return [s for b in segment(text) if b.kind != "code" for s in _sentences(b.text)]


@dataclass(frozen=True)
class Probe:
    """One grounded question, with the turn that grounds it."""

    kind: str
    question: str
    answer: str
    turn_id: str
    key: frozenset = field(default_factory=frozenset)
    negated: bool = False
    # How many negation tokens the decision sentence carries. A boolean was
    # not enough: "will use Docker because the venue has no runtime" still
    # reads as negated, and the inversion of "will not use Docker" passed.
    negations: int = 0
    # Who wrote the words the probe was drawn from: the base of its piece,
    # or the turn's whole origin when the turn has no segments.
    origin: str = "legacy"
    # The role of the turn the probe was drawn from.
    role: str = ""
    # True when the answer is the canonical form of a date or a quantity,
    # answered by any writing that reads to it; False when it is a writing,
    # answered only by the same writing.
    canonical: bool = False
    # The lexicon the probe was drawn with, named on every figure it is scored
    # into; None stands for the empty lexicon. Not part of what the probe
    # asks, so never compared.
    lexicon: object = field(default=None, compare=False, repr=False)


@dataclass
class ProbeResult:
    """The outcome of scoring a candidate text against a probe set."""

    passed: int
    failed: int
    failures: list

    @property
    def rate(self):
        """Fraction answered, or None when there was nothing to answer.

        None and 0.0 are different answers: 0.0 means every question was
        asked and none was answered; None means no question existed.
        """
        total = self.passed + self.failed
        return None if total == 0 else self.passed / total


def _negations(sentence):
    """The negations a sentence says, word by word, in prose as in inline code: one fused into a flag is none."""
    return sum(len(_NEGATION.findall(word)) for word in _words_of(sentence))


def _sentences(text):
    """The sentences of a text: split after a full stop, a question or an exclamation mark, never inside a date."""
    dates = read_dates(text)
    pieces, cursor = [], 0
    for split in _SENTENCE_SPLIT.finditer(text):
        if any(date.start < split.start() < date.end for date in dates):
            continue
        pieces.append(text[cursor:split.start()])
        cursor = split.end()
    pieces.append(text[cursor:])
    return [s.strip() for s in pieces if s.strip()]


# The act index of each lexicon met, keyed by the lexicon itself: equal
# lexicons share one entry, so a gate that reloads its configuration adds none.
_ACT_INDEXES = {}


def _act_index(lexicon):
    """A lexicon's acts by their first word, each ``(words, languages, class)``, the longest first, built once."""
    index = _ACT_INDEXES.get(lexicon)
    if index is None:
        grouped = {}
        for language, words, cls in lexicon.forms:
            grouped.setdefault(words, (cls, set()))[1].add(language)
        index = {}
        for words, (cls, languages) in sorted(grouped.items(), key=lambda item: (-len(item[0]), item[0])):
            index.setdefault(words[0], []).append((words, frozenset(languages), cls))
        _ACT_INDEXES[lexicon] = index
    return index


def _acts(folded, lexicon):
    """The acts said in a run of folded words, each ``(start, stop, languages, class)``.

    At each word the longest form that starts there is read, left to
    right, and the words it spans are read for nothing else.
    """
    index = _act_index(lexicon)
    found, at = [], 0
    while at < len(folded):
        for words, languages, cls in index.get(folded[at], ()):
            if tuple(folded[at:at + len(words)]) == words:
                found.append((at, at + len(words), languages, cls))
                at += len(words)
                break
        else:
            at += 1
    return found


def _decides(plain, folded, acts, lexicon):
    """True when a typed sentence records a decision.

    A marker decides, matched with its accents folded as the acts are, in
    either Unicode form: a French marker typed without its accents is the
    same marker. An accented one decides where it opens a word, with a
    subject or none -- "Il a ete decide", "C'est decide" -- and never inside
    one: "helicopter" and "undecidable" decide nothing. Folded, "decid" is
    also the English "decide": an English word that opens with an accented
    marker is read as a decision, the safe side, where a span stays
    verbatim rather than lose a decision. So does an act that opens the
    sentence, an instruction, and an act that a subject of its own language
    stands before, within the reach that language declares: "On garde
    Docker" decides, "Click on Select" does not -- its subject is French, its
    act English. With no act in the lexicon, only a marker decides.
    """
    # Folded, its blanks one space each: a line break or a no-break space
    # inside a marker of two words is a space.
    folded_plain = " ".join(_fold(plain).split())
    if any(marker in folded_plain for marker in _PLAIN_MARKERS):
        return True
    # The words of the folded sentence, an elided word read whole ("d'" is
    # "de"): a decomposed accent splits no word there.
    words = [_ELIDED.get(word, word) for word in _tokens(folded_plain)]
    if any(words[index].startswith(first) and words[index + 1:index + 1 + len(rest)] == rest
           for index in range(len(words)) for first, *rest in _FOLDED_MARKERS):
        return True
    reach = dict(lexicon.reach)
    for start, _stop, languages, _cls in acts:
        if start == 0:
            return True
        for language in languages:
            before = folded[max(0, start - reach.get(language, 0)):start]
            if any((language, word) in lexicon.subjects for word in before):
                return True
    return False


def _decision_key(sentence, rest, dates, quantities, folded, acts, lexicon):
    """What a decision is asked for by: its content words, each act's class, its dates and numbers as read.

    The words are the sentence's, inline code included, without function
    words, without the lexicon's subjects, without the words of its acts and
    without those of the dates and quantities it states. Each act stands as
    ``act:<class>``, each date as ``date:<answer>`` and each quantity as
    ``number:<answer>``: another word of the class, in either language,
    answers for the act, and any writing that reads to the same answer for
    the date or the number.
    """
    said = {folded[i] for start, stop, _languages, _cls in acts for i in range(start, stop)}
    said.update(word for _language, word in lexicon.subjects)
    words = set(_tokens(rest)) | {w for code in _INLINE_CODE.finditer(sentence) for w in _tokens(code.group(2))}
    key = {w for w in words if w not in _STOPWORDS and _fold(w) not in said}
    key.update("act:" + cls for _start, _stop, _languages, cls in acts)
    key.update("date:" + typed.answer for typed in dates)
    key.update("number:" + typed.answer for typed in quantities)
    return frozenset(key)


class LexiconError(ValueError):
    """The decision lexicon cannot be built as written."""


@dataclass(frozen=True)
class Lexicon:
    """The words a decision is said with, built from ``onion.yaml``.

    ``subjects`` holds ``(language, word)`` pairs; ``forms`` the acts, each
    ``(language, words, class)`` with ``words`` a tuple, in sorted order;
    ``fingerprint`` names the lexicon on every figure drawn with it;
    ``reach`` holds ``(language, words)`` pairs, in sorted order: the most
    words a subject of that language may stand before an act.
    """

    subjects: frozenset = frozenset()
    forms: tuple = ()
    fingerprint: str = ""
    reach: tuple = ()


def _make_lexicon(subjects, forms, reach=()):
    subjects, forms, reach = frozenset(subjects), tuple(sorted(forms)), tuple(sorted(reach))
    lines = sorted(
        [f"subject {language} {word}" for language, word in subjects]
        + [f"act {language} {cls} {' '.join(words)}" for language, words, cls in forms]
        + [f"reach {language} {words}" for language, words in reach]
    )
    digest = hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()
    return Lexicon(subjects, forms, digest[:LEXICON_FINGERPRINT_LENGTH], reach)


# The lexicon of a gate that holds none: no subject and no act, so that only
# a marker makes a decision.
EMPTY_LEXICON = _make_lexicon((), ())


def _inflect_fr(word):
    """The forms of a French verb in -er or -ir, folded; None when neither rule applies."""
    if len(word) > 2 and word.endswith("er"):
        stem = word[:-2]
        soft = "e" if stem.endswith("g") else ""
        return tuple(stem + (soft if ending[0] in "ao" else "") + ending for ending in _FR_ER)
    if len(word) > 2 and word.endswith("ir"):
        return tuple(word[:-2] + ending for ending in _FR_IR)
    return None


def _inflect_en(word):
    """The forms of an English verb by its spelling: itself, -s or -es, -ing and -ed.

    A final e is dropped before -ing unless doubled; a consonant and y make
    -ies and -ied. A closing consonant is doubled in a word of one syllable;
    a longer word keeps both spellings, its stress being beyond the spelling.
    """
    before = word[-2] if len(word) > 1 else "a"
    if word.endswith(("s", "x", "z", "ch", "sh")) or (word.endswith("o") and before not in _EN_VOWELS):
        third = word + "es"
    elif word.endswith("y") and before not in _EN_VOWELS:
        third = word[:-1] + "ies"
    else:
        third = word + "s"
    if word.endswith("y") and before not in _EN_VOWELS:
        ing, past = (word + "ing",), (word[:-1] + "ied",)
    elif word.endswith("e"):
        ing, past = (word + "ing" if word.endswith("ee") else word[:-1] + "ing",), (word + "d",)
    elif _EN_CLOSED.fullmatch(word):
        doubled = word + word[-1]
        if len(_EN_SYLLABLE.findall(word)) == 1:
            ing, past = (doubled + "ing",), (doubled + "ed",)
        else:
            ing, past = (doubled + "ing", word + "ing"), (doubled + "ed", word + "ed")
    else:
        ing, past = (word + "ing",), (word + "ed",)
    return (word, third, *ing, *past)


def _member_forms(language, key, member, subjects, at):
    """The forms one member of a class stands for, or a refusal that names it."""
    if type(member) is not str or not _LEXICON_MEMBER.fullmatch(member):
        raise LexiconError(f"{at}: the member {member!r} is not lower-case ASCII words a space apart")
    words = member.split(" ")
    carried = [word for word in words if word in subjects]
    if carried:
        raise LexiconError(f"{at}: the member {member!r} carries the subject {carried[0]!r}")
    if _NEGATION.search(member):
        raise LexiconError(f"{at}: the member {member!r} carries a negation")
    if key == "as_written":
        return (tuple(words),)
    inflected = (_inflect_fr if language == "fr" else _inflect_en)(words[0])
    if inflected is None:
        raise LexiconError(f"{at}: no rule inflects {member!r}; list its forms under as_written")
    return tuple((form, *words[1:]) for form in inflected)


def build_lexicon(section):
    """The lexicon of the ``decisions`` section of ``onion.yaml``, refused by name when malformed.

    Each language holds ``subjects``; ``reach``, the most words a subject
    may stand before an act, one or more; ``acts`` -- classes of lemmas its
    rule inflects, a member of several words by its first -- and, if it
    needs them, ``as_written`` -- classes of forms taken as they stand. A
    form belongs to one class only, whatever its language.
    """
    if not isinstance(section, dict):
        raise LexiconError("decisions: the section is not a mapping of languages")
    if not section:
        raise LexiconError("decisions: the section names no language")
    subjects, forms, owner, reaches = set(), set(), {}, []
    for language, entry in section.items():
        if language not in _LEXICON_LANGUAGES:
            known = ", ".join(_LEXICON_LANGUAGES)
            raise LexiconError(f"decisions: {language!r} is not a language the rules inflect ({known})")
        where = f"decisions.{language}"
        if not isinstance(entry, dict):
            raise LexiconError(f"{where}: not a mapping of subjects, acts and as_written")
        unknown = [key for key in entry if key not in _LEXICON_KEYS]
        if unknown:
            raise LexiconError(f"{where}: unknown key {unknown[0]!r}")
        said = entry.get("subjects")
        if not isinstance(said, list) or not said:
            raise LexiconError(f"{where}: subjects is not a non-empty list of words")
        for word in said:
            if type(word) is not str or not _LEXICON_NAME.fullmatch(word):
                raise LexiconError(f"{where}: the subject {word!r} is not one lower-case ASCII word")
        own = frozenset(said)
        subjects.update((language, word) for word in own)
        if not isinstance(entry.get("acts"), dict):
            raise LexiconError(f"{where}: acts is not a mapping of classes")
        for key in ("acts", "as_written"):
            classes = entry.get(key, {})
            if not isinstance(classes, dict):
                raise LexiconError(f"{where}.{key}: not a mapping of classes")
            for cls, members in classes.items():
                if type(cls) is not str or not _LEXICON_NAME.fullmatch(cls):
                    raise LexiconError(f"{where}.{key}: the class {cls!r} is not one lower-case ASCII word")
                at = f"{where}.{key}.{cls}"
                if not isinstance(members, list):
                    raise LexiconError(f"{at}: not a list of members")
                if not members:
                    raise LexiconError(f"{at}: the class has no member")
                for member in members:
                    for words in _member_forms(language, key, member, own, at):
                        held = owner.setdefault(words, cls)
                        if held != cls:
                            first, second = sorted((held, cls))
                            raise LexiconError(
                                f"decisions: the form {' '.join(words)!r} is in two classes, {first} and {second}"
                            )
                        forms.add((language, words, cls))
        reach = entry.get("reach")
        if type(reach) is not int or reach < 1:
            raise LexiconError(f"{where}: reach is not a whole number of words, one or more")
        reaches.append((language, reach))
    return _make_lexicon(subjects, forms, reaches)


@dataclass(frozen=True)
class Typed:
    """A date or a quantity read in a text: its bounds, the answer it is asked by, and whether that answer is canonical.

    A canonical answer is the same for every writing of the same date or
    quantity; one that is not is the writing itself, which reads two ways.
    """

    start: int
    end: int
    answer: str
    canonical: bool


def _fold(word):
    """A word in lower case with its accents taken off: how a month or a unit word is looked up."""
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFD", word) if not unicodedata.combining(c)).lower()


def _days_in(year, month):
    """The days of a month in the Gregorian calendar; a February of no stated year may hold a 29th."""
    if month != 2:
        return 30 if month in (4, 6, 9, 11) else 31
    leap = year is None or (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    return 29 if leap else 28


def _free_before(text, i, also=""):
    """True when nothing is glued before position ``i``: no letter, figure or underscore, nor any of ``also``."""
    return i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "_" + also)


def _free_after(text, i, also=""):
    """True when nothing is glued at position ``i``: no letter, figure or underscore, nor any of ``also``."""
    return i >= len(text) or not (text[i].isalnum() or text[i] in "_" + also)


def _month(word, dotted):
    """The month a word names as written: (month, whether English, whether a period after it is its own), or None."""
    folded = _fold(word)
    lower = word == word.lower()
    capital = word[:1].isupper() and word[1:] == word[1:].lower()
    rules = [
        (month, rule) for name, month, rule in _MONTHS
        if name == folded and (
            (rule == "fr" and (lower or capital))
            or (rule == "en" and capital)
            or (rule == "fr." and dotted and (lower or capital))
            or (rule == "en." and capital)
        )
    ]
    if not rules:
        return None
    return rules[0][0], any(rule.startswith("en") for _m, rule in rules), any(rule.endswith(".") for _m, rule in rules)


def _suffix_fits(day, suffix):
    """True when an ordinal suffix is the one its day takes: 1er, 1st, 2nd, 3rd, 4th, 11th, 21st."""
    if not suffix:
        return True
    if suffix == "er":
        return day == 1
    return suffix == ("th" if day in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th"))


def _date_found(form, match, text):
    """A match of one date form, read: (start, end, answer, canonical, possible), or None when it is no date.

    A day the calendar does not hold is found but not possible: it reads as
    no date, and nothing inside it reads as a coarser one.
    """
    groups = match.groupdict()
    start, end = match.start(), match.end()
    named = form in ("dmy", "mdy", "dm", "md", "my")
    if not _free_before(text, start, "" if form in ("mdy", "md", "my") else ".,/-" + _MINUS):
        return None
    year = int(groups["y"]) if groups.get("y") else None
    day = int(groups["d"]) if groups.get("d") else None
    if named:
        read = _month(groups["m"], groups["dot"] is not None)
        if read is None:
            return None
        month, english, dotted = read
        if groups["dot"] is not None and not dotted:
            if form != "dm":
                return None
            end = match.end("m")  # the period after a full name is the sentence's
        if (groups.get("of") or groups.get("comma")) and not english:
            return None
        if not _suffix_fits(day, groups.get("suffix")):
            return None
    elif form == "numeric":
        first, second = int(groups["a"]), int(groups["b"])
        if not first or not second or (first > 12 and second > 12):
            return None
        if first <= 12 and second <= 12 and first != second:
            if not _date_ends(text, end):
                return None
            return start, end, text[start:end], False, True  # day and month either way: kept as written
        day, month = (first, second) if first > 12 or first == second else (second, first)
    else:
        month = int(groups["m"])
    if not _date_ends(text, end):
        return None
    if not 1 <= month <= 12 or (day is not None and not 1 <= day <= _days_in(year, month)):
        return start, end, "", False, False
    if day is None:
        return start, end, f"{year:04d}-{month:02d}", True, True
    if year is None:
        return start, end, f"--{month:02d}-{day:02d}", True, True
    return start, end, f"{year:04d}-{month:02d}-{day:02d}", True, True


def _date_ends(text, i):
    """True when a date may end at ``i``: nothing glued after it, and no figure after a point or a comma."""
    if not _free_after(text, i, "/-" + _MINUS):
        return False
    return not (text[i:i + 1] in (".", ",") and text[i + 1:i + 2].isdigit() and text[i + 1:i + 2].isascii())


def read_dates(text):
    """The dates written in a text, in order: each read once, none guessed.

    Every form is tried. The longest reading is kept; an impossible day
    blocks every reading inside it; two readings of one length that
    disagree block each other, and the text there is read as no date.
    """
    found = []
    for form, expression in _DATE_FORMS:
        for match in expression.finditer(text):
            read = _date_found(form, match, text)
            if read is not None:
                found.append(read)
    found.sort(key=lambda read: (read[0] - read[1], read[0]))
    kept, blocked = [], []

    def clear(read):
        return not any(read[0] < other[1] and other[0] < read[1] for other in kept + blocked)

    for read in found:
        if not clear(read):
            continue
        rivals = [
            other for other in found
            if other is not read and other[1] - other[0] == read[1] - read[0]
            and other[0] < read[1] and read[0] < other[1] and other[2:] != read[2:] and clear(other)
        ]
        (blocked if rivals or not read[4] else kept).append(read)
    return [Typed(start, end, answer, canonical) for start, end, answer, canonical, _possible in sorted(kept)]


def _numeral(run):
    """The canonical writing of a run of figures, or None when it reads two ways.

    A blank or a repeated comma separates thousands, a lone point or comma
    the decimals, which stay as written: 3.10 is not 3.1. One separator
    before three figures (1,500 or 1.500) reads two ways, and so do repeated
    points (1.000.000, an address, a version).
    """
    if any(c in _BLANK for c in run):
        shape = re.fullmatch("([0-9" + _BLANK + "]+)((?:[.,][0-9]+)?)", run)
        if shape is None:
            return None
        head, tail = shape.groups()
        integer, fraction = "".join(c for c in head if c not in _BLANK), tail[1:]
    else:
        separators = [c for c in run if c in ".,"]
        groups = re.split("[.,]", run)
        if not separators:
            integer, fraction = run, ""
        elif len(separators) == 1:
            integer, fraction = groups
            if len(fraction) == 3 and len(integer) <= 3 and integer[0] != "0":
                return None
        else:
            thousands, last = separators[0], separators[-1]
            if any(s != thousands for s in separators[:-1]) or len(groups[0]) > 3 or groups[0][0] == "0":
                return None
            if last == thousands and thousands == ".":
                return None
            body, fraction = (groups, "") if last == thousands else (groups[:-1], groups[-1])
            if any(len(group) != 3 for group in body[1:]):
                return None
            integer = "".join(body)
    return str(int(integer)) + ("." + fraction if fraction else "")


def _unit_ends(text, i, written):
    """True when a unit written as ``written`` may end at ``i``: a word or a symbol is not cut out of a longer one."""
    if not written[-1].isalnum():
        return True
    return _free_after(text, i, "'-" + chr(0x2019))


def _unit_after(text, i):
    """The class of the unit written at ``i``, after at most one blank, and where it ends; None when there is none."""
    j = i + 1 if text[i:i + 1] and text[i] in _BLANK else i
    for written, unit in _SYMBOL_CLASSES:
        if text.startswith(written, j) and _unit_ends(text, j + len(written), written):
            return unit, j + len(written)
    words = _UNIT_WORD.match(text, j)
    if words is None:
        return None
    if words.group(2) is not None:
        unit = _WORD_CLASSES.get(_fold(words.group(1) + " " + words.group(2)))
        if unit is not None and _unit_ends(text, words.end(), words.group(2)):
            return unit, words.end()
    unit = _WORD_CLASSES.get(_fold(words.group(1)))
    if unit is not None and _unit_ends(text, words.end(1), words.group(1)):
        return unit, words.end(1)
    return None


def _unit_before(text, i):
    """The class of a currency written before position ``i``, at most one blank away, and where it starts; or None."""
    j = i - 1 if i and text[i - 1] in _BLANK else i
    for unit, written in _UNIT_PREFIXES:
        k = j - len(written)
        if k >= 0 and text.startswith(written, k) and _free_before(text, k):
            return unit, k
    return None


def read_quantities(text):
    """The numbers written in a text with their units, in order: each read to its value and class, or kept as written.

    A number glued to a word that is no unit (v2, 8k, x86) is no number. A
    hyphen is a sign only after a blank or a parenthesis, so 3-4 reads as
    no number at all, as it always did.
    """
    found = []
    for match in _NUMBER.finditer(text):
        start, end = match.start(), match.end()
        sign = match.group("sign")
        if sign is not None and not (start == 0 or text[start - 1].isspace() or text[start - 1] == "("):
            continue
        if sign is None and not _free_before(text, start, ".,-" + _MINUS):
            continue
        before = None if sign is not None else _unit_before(text, start)
        after = None if before is not None else _unit_after(text, end)
        if before is None and after is None and not _free_after(text, end, "-" + _MINUS):
            continue
        first = before[1] if before is not None else start
        last = after[1] if after is not None else end
        value = _numeral(match.group("run"))
        if value is None:
            found.append(Typed(first, last, text[first:last], False))
            continue
        if sign is not None and value.strip("0.") != "":
            value = "-" + value
        unit = before or after
        found.append(Typed(first, last, f"{value} {unit[0]}" if unit is not None else value, True))
    return found


def _blank_out(text, found):
    """The text with each span read replaced by a blank, so that nothing is read twice."""
    pieces, cursor = [], 0
    for typed in found:
        pieces.append(text[cursor:typed.start])
        pieces.append(" ")
        cursor = typed.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _typed_answers(text):
    """The canonical dates and quantities a candidate gives, read as a source is: unit by unit, inline code aside."""
    dates, quantities = set(), set()
    for sentence in _units(text):
        plain = _plain(sentence)
        found = read_dates(plain)
        dates.update(t.answer for t in found if t.canonical)
        quantities.update(t.answer for t in read_quantities(_blank_out(plain, found)) if t.canonical)
    return frozenset(dates), frozenset(quantities)


def _decision_units(text, reading):
    """A candidate's sentences as a decision reads them, kept in ``reading``: ``(elements, folded, negations)``.

    The elements are a sentence's words, inline code included, with each
    date as ``date:<answer>`` and each quantity as ``number:<answer>``; the
    folded words are where its acts are found, by each probe's lexicon.
    """
    units = reading.get("decision")
    if units is None:
        units = []
        for sentence in _units(text):
            plain, dates, quantities = _read_sentence(sentence)[:3]
            elements = set(_tokens(sentence))
            elements.update("date:" + typed.answer for typed in dates)
            elements.update("number:" + typed.answer for typed in quantities)
            units.append((elements, _tokens(_fold(plain)), _negations(plain)))
        reading["decision"] = units
    return units


def _native():
    """The native core, or None: asked at the call, never at import."""
    try:
        from opti_oignon.native import load
    except Exception:  # noqa: BLE001 - absence is the reference path
        return None
    return load()


def _native_lock():
    """The lock of the counts below; threading is asked for here, never at module scope."""
    import threading

    return threading.Lock()


# What the native core served, call by call: each draw and each score is
# counted once, served or sent to the reference with the reason it was, so
# that a twin that never answers reads as such and never as silence. The
# counts are the process's; persisting them is the status's to do.
_NATIVE_CALLS = ("draw", "score")
_NATIVE_COUNTS = Counter()
_NATIVE_LOCK = _native_lock()


def _native_counted(call, outcome):
    with _NATIVE_LOCK:
        _NATIVE_COUNTS[(call, outcome)] += 1


def native_share():
    """What the native core served since the process began: per call, the calls it served and those the reference served, by reason."""
    with _NATIVE_LOCK:
        counts = sorted(_NATIVE_COUNTS.items())
    return {call: {outcome: n for (kind, outcome), n in counts if kind == call} for call in _NATIVE_CALLS}


def _native_patterns():
    """The expressions the native core must reproduce, as this module holds them now."""
    compiled = (
        _SENTENCE_SPLIT, *(expression for _form, expression in _DATE_FORMS), _NUMBER, _UNIT_WORD,
        _WORD, _CAPITALISED, _NEGATION, _HEAD,
    )
    return tuple((p.pattern, p.flags) for p in compiled) + ((_ANSWER_BEFORE, 0), (_ANSWER_AFTER, 0))


def _native_tables():
    """The tables the native core must read dates, quantities and names by, as this module holds them now."""
    return _MONTHS, _UNIT_SYMBOLS, _UNIT_WORDS, _UNIT_PREFIXES, tuple(sorted(_WEEKDAYS))


def _native_lexicon(lexicon):
    """A decision lexicon as the native core takes it: its subjects, its forms and its reach, each sorted."""
    return tuple(sorted(lexicon.subjects)), lexicon.forms, lexicon.reach


def _same_generator(core):
    """True when the native core declares, as an integer, the version of the generator this module is."""
    version = getattr(core, "probe_generator_version", None)
    return type(version) is int and version == GENERATOR_VERSION


def _native_draw(pieces, lexicon):
    """The probes the native core draws from ``(turn_id, role, origin, text)`` pieces, or None; counted either way."""
    core = _native()
    if core is None:
        _native_counted("draw", "no core")
        return None
    if not _same_generator(core):
        _native_counted("draw", "another generator")
        return None  # a twin of another generator is never asked
    draw = getattr(core, "probe_generate", None)
    if draw is None:
        _native_counted("draw", "not offered")
        return None
    try:
        drawn = draw(
            [text for _turn_id, _role, _origin, text in pieces],
            _native_patterns(),
            _native_tables(),
            sorted(_NOT_ENTITIES),
            sorted(_STOPWORDS),
            list(_DECISION_MARKERS),
            _native_lexicon(lexicon),
        )
    except (TypeError, ValueError):
        _native_counted("draw", "refused by the core")
        return None  # a text the core cannot take: the reference draws
    if drawn is None:
        _native_counted("draw", "declined")
        return None
    if not isinstance(drawn, (list, tuple)):
        _native_counted("draw", "row refused")
        return None  # a draw that is no list of rows: the reference draws
    probes = []
    for row in drawn:
        try:
            index, kind, answer, key, negations, canonical = row
        except (TypeError, ValueError):
            _native_counted("draw", "row refused")
            return None  # a row of another shape: the reference draws
        if type(index) is not int or not 0 <= index < len(pieces) or type(kind) is not str or kind not in _QUESTIONS:
            _native_counted("draw", "row refused")
            return None  # a row that names no piece, or no kind: the reference draws
        if (type(answer) is not str or not isinstance(key, (list, tuple))
                or not all(type(element) is str for element in key)
                or type(negations) is not int or negations < 0 or type(canonical) is not bool
                or (kind == "decision" and not key)):
            _native_counted("draw", "row refused")
            return None  # a row whose fields are not of their kind, or a keyless decision: the reference draws
        turn_id, role, origin, _text = pieces[index]
        if kind == "decision" and origin != "typed":
            continue  # only typed text decides, whatever the twin drew
        probes.append(Probe(
            kind,
            _QUESTIONS[kind].format(turn_id),
            answer,
            turn_id,
            key=frozenset(key),
            negated=negations > 0,
            negations=negations,
            origin=origin,
            role=role,
            canonical=canonical is True,
            lexicon=lexicon,
        ))
    _native_counted("draw", "served")
    return probes


def _native_failures(probes, text):
    """Indices of the probes the native core finds unanswered, or None.

    Only exact shapes are handed over: a probe the reference would treat
    differently from the core -- a key that is not a frozenset of strings,
    an empty key, an answer that is not a string -- is left to the
    reference, which raises what it raises. The core is handed the lexicon
    the decisions were drawn with, and decisions drawn with two lexicons
    are left to the reference.
    """
    if type(text) is not str or type(probes) not in (list, tuple) or type(DECISION_COVERAGE) is not float:
        _native_counted("score", "not handed")
        return None
    core = _native()
    if core is None:
        _native_counted("score", "no core")
        return None
    if not _same_generator(core):
        _native_counted("score", "another generator")
        return None  # a twin of another generator is never asked
    scorer = getattr(core, "probe_score", None)
    if scorer is None:
        _native_counted("score", "not offered")
        return None
    reason, rows, lexicon = _native_rows(probes)
    if reason is not None:
        _native_counted("score", reason)
        return None  # a probe the core is never handed, or decisions of two lexicons: the reference scores
    try:
        failing = scorer(rows, text, _native_patterns(), _native_tables(), DECISION_COVERAGE, lexicon)
    except (AttributeError, TypeError, ValueError, OverflowError):
        _native_counted("score", "refused by the core")
        return None  # a probe the core cannot take: the reference scores
    _native_counted("score", "declined" if failing is None else "served")
    return failing


def _native_rows(probes):
    """The probes as the native core takes them: ``(None, rows, lexicon)``, or the reason it is never handed them."""
    try:
        rows, lexicons = [], set()
        for probe in probes:
            kind, answer, canonical, key = probe.kind, probe.answer, probe.canonical, probe.key
            if type(kind) is not str or type(answer) is not str or type(canonical) is not bool:
                return "probe not handed", None, None
            if type(key) is not frozenset or any(type(word) is not str for word in key):
                return "probe not handed", None, None
            if kind != "decision":
                rows.append((kind, answer, sorted(key), 0, canonical))
                continue
            if not key:
                return "probe not handed", None, None
            rows.append((kind, answer, sorted(key), probe.negations, canonical))
            lexicons.add(EMPTY_LEXICON if probe.lexicon is None else probe.lexicon)
        if len(lexicons) > 1:
            return "two lexicons", None, None
        return None, rows, _native_lexicon(lexicons.pop() if lexicons else EMPTY_LEXICON)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return "probe not handed", None, None


def _read_sentence(sentence):
    """A sentence read before anything is drawn from it: ``(plain, dates, quantities, rest, head, names)``.

    ``rest`` is the plain sentence with its dates and quantities blanked
    out; ``head`` the offset of its first word, or -1 when that word opens a
    date; ``names`` its capitalised words that are no function word and no
    day of the week, each with its offset in ``rest`` -- where the head
    stands at the same offset, nothing before it having been blanked.
    """
    plain = _plain(sentence)
    dates = read_dates(plain)
    rest = _blank_out(plain, dates)
    quantities = read_quantities(rest)
    rest = _blank_out(rest, quantities)
    first = _HEAD.search(plain)
    head = -1 if first is None or any(d.start <= first.start() < d.end for d in dates) else first.start()
    names = [
        (m.start(), m.group()) for m in _CAPITALISED.finditer(rest)
        if m.group()[0].isupper() and m.group().lower() not in _NOT_ENTITIES and _fold(m.group()) not in _WEEKDAYS
    ]
    return plain, dates, quantities, rest, head, names


def _names(readings):
    """The names each reading draws, in order, as ``(name, aliases)`` pairs; None for a code block.

    A head is kept only when the span capitalises the same word off a head.
    Kept names a single space apart make one name. A name of one word that
    is part of a longer name met in an earlier sentence is that name's alias:
    it draws nothing, and the longer name answers to it.
    """
    evidence = {name for r in readings if r is not None for offset, name in r[5] if offset != r[4]}
    runs = []
    for reading in readings:
        if reading is None:
            runs.append(None)
            continue
        rest, head, found = reading[3:]
        joined, end = [], None
        for offset, name in found:
            if offset == head and name not in evidence:
                continue
            if end is not None and rest[end:offset] == " ":
                joined[-1] = joined[-1] + (name,)
            else:
                joined.append((name,))
            end = offset + len(name)
        runs.append(joined)
    first = {}
    for position, joined in enumerate(runs):
        for run in joined or ():
            if len(run) > 1:
                first.setdefault(run, position)
    aliases = {run: set() for run in first}
    kept = []
    for position, joined in enumerate(runs):
        if joined is None:
            kept.append(None)
            continue
        own = []
        for run in joined:
            owners = [full for full, at in first.items() if len(run) == 1 and at < position and run[0] in full]
            for full in owners:
                aliases[full].add(run[0].lower())
            if not owners:
                own.append(run)
        kept.append(own)
    return [
        None if own is None else [(" ".join(run), frozenset(aliases.get(run, ()))) for run in own] for own in kept
    ]


def _pieces(span):
    """The pieces of a span's turns, each ``(turn_id, role, origin, text)``, in order.

    A turn with segments is its segments, each of its own base, and a
    character no segment covers is in none; a turn without segments is one
    piece of its own origin.
    """
    pieces = []
    for turn in span:
        turn_id = str(turn.get("turn_id", ""))
        role = str(turn.get("role", "") or "")
        text = str(turn.get("text", "") or "")
        origin, segments, _defect = read_origin(turn)
        if segments:
            pieces.extend((turn_id, role, label, text[start:stop]) for start, stop, label in segments)
        else:
            pieces.append((turn_id, role, origin, text))
    return pieces


def turn_pieces(turn):
    """The texts of a turn's pieces, as the probes read them: its segments, or the whole turn."""
    return [text for _turn_id, _role, _origin, text in _pieces([turn])]


def mask_turn(turn):
    """A turn's text with each fenced block replaced by its marker, read piece by piece as the probes read it.

    A fence ends with its piece, as the probes read it; the words between
    pieces, which belong to no one, stay as they are.
    """
    text = str(turn.get("text", "") or "")
    _origin, segments, _defect = read_origin(turn)
    if not segments:
        return mask_code(text)
    out, cursor = [], 0
    for start, stop, _label in segments:
        out.append(text[cursor:start])
        out.append(mask_code(text[start:stop]))
        cursor = stop
    out.append(text[cursor:])
    return "".join(out)


def sentences(text):
    """The sentences of a text, as the probes split it."""
    return _sentences(text)


@dataclass(frozen=True)
class Unit:
    """One place a probe is answered in: a sentence of prose, or a fenced block as its marker.

    ``start`` and ``stop`` bound it in its turn's text: a sentence's own
    characters, a code block with its fences. ``text`` is what a peel or an
    anchor shows of it: the sentence as written, or the block's marker,
    never its code.
    """

    turn_id: str
    origin: str
    start: int
    stop: int
    text: str


def units(span):
    """The units of a span's turns, in order: each sentence of prose and each fenced block, with its place.

    Read piece by piece and block by block as the probes read them, so a
    unit is where a probe drawn from the span can be answered. A block
    whose sentences the reader rebuilt from its lines, so that the turn does
    not hold them as written, is one unit, the whole block.
    """
    found = []
    for turn in span:
        turn_id = str(turn.get("turn_id", ""))
        text = str(turn.get("text", "") or "")
        origin, segments, _defect = read_origin(turn)
        parts = [(start, stop, label) for start, stop, label in segments] if segments else [(0, len(text), origin)]
        for offset, end_of_piece, label in parts:
            for block in segment(text[offset:end_of_piece]):
                start, end = offset + block.start, offset + block.end
                if block.kind == "code":
                    found.append(Unit(turn_id, label, start, end, code_marker(block.text)))
                    continue
                placed, cursor = [], start
                for sentence in _sentences(block.text):
                    at = text.find(sentence, cursor, end)
                    if at < 0:
                        placed = None
                        break
                    placed.append(Unit(turn_id, label, at, at + len(sentence), sentence))
                    cursor = at + len(sentence)
                if placed is None:
                    whole = text[start:end].strip()
                    lead = start + len(text[start:end]) - len(text[start:end].lstrip())
                    placed = [Unit(turn_id, label, lead, lead + len(whole), whole)] if whole else []
                found.extend(placed)
    return found


def _read_pieces(pieces):
    """Each piece read block by block: ``(index, unit, reading)``, a code block's body with no reading, in order."""
    read = []
    for index, (_turn_id, _role, _origin, text) in enumerate(pieces):
        for block in segment(text):
            if block.kind == "code":
                read.append((index, block.text, None))
            else:
                read.extend((index, sentence, _read_sentence(sentence)) for sentence in _sentences(block.text))
    return read


def generate_probes(span, lexicon=None):
    """Probes drawn from a span of turns, each a mapping with turn_id and text, with a decision lexicon.

    Every probe carries the turn it was drawn from, so a failing probe names
    where the lost information lived, the origin of the piece it was drawn
    from, and the lexicon it was drawn with, the empty one when none is
    given. Order is deterministic: turn order, then piece order, then
    block order, then sentence order, then kind. A fenced block yields one
    code probe per piece, whatever its repeats. Names are read by the whole
    span before any is drawn: a head needs a capital elsewhere, an alias a
    full name before it. The native core draws the same probes when it
    answers, handed the pieces; the loop below is what it is held to.
    """
    lexicon = EMPTY_LEXICON if lexicon is None else lexicon
    pieces = _pieces(span)
    drawn = _native_draw(pieces, lexicon)
    if drawn is not None:
        return drawn
    read = _read_pieces(pieces)
    named = _names([reading for _index, _unit, reading in read])
    probes, markers = [], set()
    for (index, unit, reading), names in zip(read, named):
        turn_id, role, origin, _text = pieces[index]
        labels = {"origin": origin, "role": role, "lexicon": lexicon}
        if reading is None:
            marker = code_marker(unit)
            if (index, marker) not in markers:
                markers.add((index, marker))
                probes.append(Probe("code", _QUESTIONS["code"].format(turn_id), marker, turn_id, **labels))
            continue
        sentence = unit
        plain, dates, quantities = reading[:3]
        seen = set()
        for kind, found in (("date", dates), ("number", quantities)):
            for typed in found:
                if (kind, typed.answer, typed.canonical) in seen:
                    continue
                seen.add((kind, typed.answer, typed.canonical))
                probes.append(Probe(
                    kind, _QUESTIONS[kind].format(turn_id), typed.answer, turn_id,
                    canonical=typed.canonical, **labels,
                ))
        for name, aliases in names:
            if ("entity", name) in seen:
                continue
            seen.add(("entity", name))
            probes.append(Probe("entity", _QUESTIONS["entity"].format(turn_id), name, turn_id, key=aliases, **labels))
        if origin != "typed":
            continue
        folded = _tokens(_fold(plain))
        acts = _acts(folded, lexicon)
        if _decides(plain, folded, acts, lexicon):
            key = _decision_key(sentence, reading[3], dates, quantities, folded, acts, lexicon)
            if key:
                probes.append(Probe(
                    "decision",
                    _QUESTIONS["decision"].format(turn_id),
                    sentence,
                    turn_id,
                    key=key,
                    negated=_negations(plain) > 0,
                    negations=_negations(plain),
                    **labels,
                ))
    return probes


def answers(probe, text, reading=None):
    """True when the candidate text answers the probe.

    ``reading`` keeps what the candidate reads to when one text is scored
    against several probes: filled at the first typed probe, then reused.

    A date or a number drawn in its canonical form is answered by any
    writing of the candidate that reads to that form, the candidate read as
    a source is; one drawn as written is answered by the same writing. An
    entity is answered by its words, in any order, or by one of its aliases.

    A decision is answered by one sentence that carries enough of its key
    -- its content words, the class of each of its acts, its dates and
    numbers as read -- and every act, date and number of the key, with the
    same polarity. Acts are found by the lexicon the probe was drawn with,
    in either language: another word of the class answers, another act does
    not, however many of the words it keeps. The same words with the
    opposite polarity is the decision inverted, and that is a failure, not a
    match. Polarity is the count of negation tokens, not their presence: one
    negation dropped from a sentence that carried two is an inversion too.
    The candidate is read as the source is, block by block: a list item is
    a sentence of its own, and a negation inside inline code counts where it
    is a word of its own ("`will not`") and nowhere it is fused into a flag
    ("`--no-cache`"). A code block is answered by its marker or by the same
    block verbatim.
    """
    if probe.kind == "code":
        return probe.answer in text or any(code_marker(b.text) == probe.answer for b in code_blocks(text))
    if probe.kind == "decision":
        if reading is None:
            reading = {}
        lexicon = EMPTY_LEXICON if probe.lexicon is None else probe.lexicon
        units = _decision_units(text, reading)
        acts = reading.get(("acts", lexicon))
        if acts is None:
            acts = reading[("acts", lexicon)] = [
                {"act:" + cls for _start, _stop, _languages, cls in _acts(folded, lexicon)}
                for _elements, folded, _negations in units
            ]
        required = {element for element in probe.key if element.startswith(_REQUIRED)}
        for (elements, _folded, negations), said in zip(units, acts):
            carried = elements | said
            if len(probe.key & carried) / len(probe.key) >= DECISION_COVERAGE and required <= carried:
                if negations == probe.negations:
                    return True
        return False
    if probe.kind in ("date", "number"):
        if probe.canonical:
            if reading is None:
                reading = {}
            if "date" not in reading:
                reading["date"], reading["number"] = _typed_answers(text)
            return probe.answer in reading[probe.kind]
        pattern = _ANSWER_BEFORE + re.escape(probe.answer) + _ANSWER_AFTER
        return re.search(pattern, text) is not None
    words = set(_tokens(text))
    parts = _tokens(probe.answer)
    return (bool(parts) and all(part in words for part in parts)) or any(alias in words for alias in probe.key)


def copies(sentence, text, fewest):
    """True when ``sentence`` repeats ``text``: most content words of either one are words of the other.

    Read both ways at the share a decision is answered at, so a sentence
    that copies part of a longer one is a copy, and so is one that copies a
    short one whole; and through at least ``fewest`` shared content words
    (the queue's ``copy_shared_words``), so one word in common -- a name,
    "logs" -- need be no copy. A summary's sentence that copies words from
    outside the conversation is those words kept verbatim, whatever the
    model meant by it.
    """
    said = {_fold(word) for word in _tokens(sentence)} - _FOLDED_STOPWORDS
    held = {_fold(word) for word in _tokens(text)} - _FOLDED_STOPWORDS
    shared = len(said & held)
    return shared >= fewest and shared > 0 and (
        shared / len(said) >= DECISION_COVERAGE or shared / len(held) >= DECISION_COVERAGE
    )


def score(probes, text):
    """Score a candidate text against a probe set."""
    failing = _native_failures(probes, text)
    if failing is None:
        reading = {}
        failures = [p for p in probes if not answers(p, text, reading)]
    else:
        failures = [probes[i] for i in failing]
    return ProbeResult(passed=len(probes) - len(failures), failed=len(failures), failures=failures)


# The second face of the gate reads a summary for what it says, and its span
# for what it holds. The span is read as leniently as the reader allows, the
# summary as strictly: a claim is held only where the span holds it.
_MARKER = re.compile(r"\[code:([0-9a-f]{%d})\]" % CODE_KEY_LENGTH)


@dataclass(frozen=True)
class Holdings:
    """What a span holds that a summary may say.

    ``words`` are the folded words of every turn, inline code and fenced
    code included, its role among them; ``dates`` and ``numbers`` the
    canonical answers its turns read to, block by block, the figures of a
    date among the numbers; ``texts`` the turns' texts, where a writing
    kept as written is looked for; ``keys`` the code keys of its fenced
    blocks and of the markers it writes; ``inline`` the inline code its
    turns write, each span as written. Every origin counts: who said a claim
    matters to a decision only.
    """

    words: frozenset
    dates: frozenset
    numbers: frozenset
    texts: tuple
    keys: frozenset
    inline: frozenset = frozenset()


def holdings(span):
    """What a span of turns holds, each a mapping with its text and its role."""
    words, dates, numbers, texts, keys, inline = set(), set(), set(), [], set(), set()
    for turn in span:
        text = str(turn.get("text", "") or "")
        texts.append(text)
        words.update(_tokens(_fold(text + " " + str(turn.get("role", "") or ""))))
        keys.update(_MARKER.findall(text))
        for block in (block for piece in turn_pieces(turn) for block in segment(piece)):
            if block.kind == "code":
                keys.add(block.key)
            else:
                inline.update(code.group(2) for code in _INLINE_CODE.finditer(block.text))
            found = read_dates(block.text)
            dates.update(t.answer for t in found if t.canonical)
            numbers.update(t.answer for t in read_quantities(block.text) if t.canonical)
            numbers.update(t.answer for t in read_quantities(_blank_out(block.text, found)) if t.canonical)
        read_dates_, read_numbers = _typed_answers(text)
        dates.update(read_dates_)
        numbers.update(read_numbers)
    return Holdings(frozenset(words), frozenset(dates), frozenset(numbers), tuple(texts), frozenset(keys),
                    frozenset(inline))


def _written_in(answer, texts):
    pattern = _ANSWER_BEFORE + re.escape(answer) + _ANSWER_AFTER
    return any(re.search(pattern, text) for text in texts)


def _date_held(typed, held):
    """A date is held as read, or as the month of a day or the day without its year that the span gives."""
    answer = typed.answer
    if not typed.canonical:
        return _written_in(answer, held.texts)
    if answer in held.dates:
        return True
    if answer.startswith("--"):
        return any(len(date) == 10 and date[4:] == answer[1:] for date in held.dates)
    if len(answer) == 7:
        return any(date.startswith(answer + "-") for date in held.dates)
    return False


def _number_held(typed, held):
    """A number is held as read, or without the unit the span gives it; another unit is not the same number."""
    answer = typed.answer
    if not typed.canonical:
        return _written_in(answer, held.texts)
    if answer in held.numbers:
        return True
    return " " not in answer and any(number.split(" ", 1)[0] == answer for number in held.numbers)


def unsupported_claims(held, text):
    """The claims of a summary its span does not hold, each ``(kind, answer, "")``.

    The summary is read as a source is: names by the whole text, a head of
    a sentence only where the text capitalises the word elsewhere; dates
    and numbers sentence by sentence, inline code aside; code by its
    fenced blocks and its markers, a marker read for nothing else, and its
    inline code, held only by the same inline code of the span: a figure, a
    name or a word in backticks is never read past. A name is held when the
    span holds each of its words. Names come first, then dates, numbers and
    code, each in reading order, each once.
    """
    readings = [_read_sentence(sentence) for sentence in _units(_MARKER.sub(" ", text))]
    claims = {"entity": [], "date": [], "number": [], "code": []}

    def add(kind, answer):
        if answer not in claims[kind]:
            claims[kind].append(answer)

    for reading, names in zip(readings, _names(readings)):
        for name, _aliases in names:
            if not all(_fold(word) in held.words for word in _tokens(name)):
                add("entity", name)
        for typed in reading[1]:
            if not _date_held(typed, held):
                add("date", typed.answer)
        for typed in reading[2]:
            if not _number_held(typed, held):
                add("number", typed.answer)
    for block in segment(text):
        for key in [block.key] if block.kind == "code" else _MARKER.findall(block.text):
            if key not in held.keys:
                add("code", key)
        if block.kind != "code":
            for code in _INLINE_CODE.finditer(block.text):
                if code.group(2) not in held.inline and not _MARKER.fullmatch(code.group(2)):
                    add("code", code.group(2))
    return [(kind, answer, "") for kind in claims for answer in claims[kind]]


def unbacked_decisions(probes, text, lexicon=None, reporters=frozenset()):
    """The sentences of a summary that decide with no typed decision of its span behind them.

    Each is ``(kind, sentence, turn_id)``. A sentence decides as a typed one
    does, read with the lexicon the probes were drawn with. One whose
    subject -- its first word that is no function word -- is a reporter
    tells what another speaker said, and is passed over. Every other
    deciding sentence is held by a decision probe it answers, with the same
    polarity. One that answers a probe but for its polarity is an
    ``inversion`` of that probe's turn; one that answers none is a
    ``decision``, with no turn. Each is named once, in reading order.
    """
    from dataclasses import replace

    lexicon = EMPTY_LEXICON if lexicon is None else lexicon
    decisions = [p for p in probes if p.kind == "decision"]
    found = []
    for sentence in _units(text):
        plain = _plain(sentence)
        folded = _tokens(_fold(plain))
        if not _decides(plain, folded, _acts(folded, lexicon), lexicon):
            continue
        if next((word for word in folded if word not in _STOPWORDS), None) in reporters:
            continue
        if any(answers(p, sentence) for p in decisions):
            continue
        said = _negations(plain)
        inverted = next((p for p in decisions if answers(replace(p, negations=said), sentence)), None)
        entry = ("inversion", sentence, inverted.turn_id) if inverted is not None else ("decision", sentence, "")
        if entry not in found:
            found.append(entry)
    return found


# The bounds of the gate read a summary against its span word by word. A
# word is compared by its key: folded, its final s taken off, then its first
# letters -- truncation, the plainest stemming that serves both languages.
# A key held by the span holds every word of that key, "transport" as well
# as "transfer": the bound is coarse on purpose, the claims being the second
# face's to judge.
NOVELTY_KEY_LENGTH = 5


def _novelty_key(word):
    return (word[:-1] if word.endswith("s") else word)[:NOVELTY_KEY_LENGTH]


_FOLDED_STOPWORDS = frozenset(_fold(word) for word in _STOPWORDS)
# The decision markers as a sentence is matched against them, folded. One with
# no accent is matched anywhere, as every marker was in version 6; an accented
# one, as its words, where it opens a word: folded, "opte" lies inside
# "helicopter" and "decid" inside "undecidable".
_PLAIN_MARKERS = tuple(marker for marker in _DECISION_MARKERS if _fold(marker) == marker)
_FOLDED_MARKERS = tuple(tuple(_fold(marker).split()) for marker in _DECISION_MARKERS if _fold(marker) != marker)
# The French words a vowel elides, as a folded token reads them: "prevu
# d'utiliser" holds the marker "prevu de".
_ELIDED = {"c": "ce", "d": "de", "j": "je", "l": "le", "m": "me", "n": "ne", "qu": "que", "s": "se", "t": "te"}
# The function words of both languages, folded: what a text is told with,
# passed over wherever its words are weighed.
FUNCTION_WORDS = _FOLDED_STOPWORDS
# A decision marker can be a stem ("decid"): its words are matched by key.
_MARKER_KEYS = frozenset(_novelty_key(word) for marker in _DECISION_MARKERS for word in _tokens(_fold(marker)))


def novel_words(held, text, lexicon=None, reporters=frozenset()):
    """The content words of a summary, and those its span holds no word of the same key for: ``(content, new)``.

    The summary is read without its code markers and its fenced code, which
    the second face judges. A word is folded. Function words, words with a
    figure, the words of the lexicon's subjects and forms, words whose key a
    decision marker's word has, and the reporters say how a summary tells,
    not what: they are no content words. Every occurrence counts, in reading
    order.
    """
    lexicon = EMPTY_LEXICON if lexicon is None else lexicon
    told = {word for _language, word in lexicon.subjects}
    told.update(word for _language, words, _cls in lexicon.forms for word in words)
    told.update(reporters)
    prose = " ".join(block.text for block in segment(_MARKER.sub(" ", text)) if block.kind != "code")
    content = tuple(
        word for word in (_fold(token) for token in _tokens(prose))
        if word not in _FOLDED_STOPWORDS and not any(c.isdigit() for c in word)
        and word not in told and _novelty_key(word) not in _MARKER_KEYS
    )
    keys = {_novelty_key(word) for word in held.words}
    return content, tuple(word for word in content if _novelty_key(word) not in keys)


def word_count(text):
    """The words of a text as the reader splits them: runs of letters and figures, in any script."""
    return len(_tokens(text))


def folded_words(text):
    """The words of a text as the reader splits them, in lower case with their accents taken off, in order."""
    return _tokens(_fold(text))


# What share of the facts a span holds its probes ask for. A probe set is
# drawn by whoever hands it in -- the reference loop, the native core, a
# caller -- and a rate counts only the probes a summary answers: a set that
# asks for less makes every rate rise. The facts are read here again, as the
# second face reads a summary and never by the native core: the readers are
# the generator's, the facts are collected anew, so a loop, a core or a
# caller that draws less is counted, not believed.


@dataclass(frozen=True)
class Fact:
    """One fact of a span that a probe must ask for, at the first turn that holds it.

    ``what`` is a name, a date or a number as the reader gives it, a code
    block's marker or a deciding sentence; ``mark`` is what a probe asking
    for the fact carries, as ``probe_mark`` reads it.
    """

    kind: str
    what: str
    turn_id: str
    mark: tuple


def probe_mark(probe):
    """What a probe asks for: a name or a block's marker, a date or a number with its form, a decision by its key and polarity."""
    if probe.kind == "decision":
        return ("decision", probe.key, probe.negated)
    if probe.kind in ("date", "number"):
        return (probe.kind, probe.answer, probe.canonical)
    return (probe.kind, probe.answer)


def held_facts(span, lexicon=None):
    """The facts of a span a probe must ask for, each once, in reading order.

    The span is read piece by piece, as its origins bound it, and never by
    the native core: names by the whole span, a head of a sentence only
    where the span capitalises its word elsewhere, an alias never; dates and
    numbers as read or as written, sentence by sentence, inline code aside;
    each fenced block by its marker, and nothing inside it; a decision from
    a typed piece only, by its key and its polarity, read with ``lexicon``.
    Each fact stands at the first turn that holds it.
    """
    lexicon = EMPTY_LEXICON if lexicon is None else lexicon
    pieces = _pieces(span)
    read = _read_pieces(pieces)
    found = {}

    def hold(kind, what, turn_id, mark):
        if mark not in found:
            found[mark] = Fact(kind, what, turn_id, mark)

    for (index, unit, reading), names in zip(read, _names([reading for _index, _unit, reading in read])):
        turn_id, _role, origin, _text = pieces[index]
        if reading is None:
            hold("code", code_marker(unit), turn_id, ("code", code_marker(unit)))
            continue
        plain, dates, quantities, rest = reading[:4]
        for kind, typed_found in (("date", dates), ("number", quantities)):
            for typed in typed_found:
                hold(kind, typed.answer, turn_id, (kind, typed.answer, typed.canonical))
        for name, _aliases in names:
            hold("entity", name, turn_id, ("entity", name))
        if origin == "typed":
            folded = _tokens(_fold(plain))
            acts = _acts(folded, lexicon)
            key = _decides(plain, folded, acts, lexicon) and _decision_key(
                unit, rest, dates, quantities, folded, acts, lexicon)
            if key:
                hold("decision", unit, turn_id, ("decision", key, _negations(plain) > 0))
    return tuple(found.values())


@dataclass(frozen=True)
class ProbeCoverage:
    """What share of the facts a span holds a set of probes asks for.

    ``facts`` counts the span's facts, ``asked`` those a probe of the set
    asks for, and ``unasked`` names the others, each ``(kind, what,
    turn_id)``, in reading order. A probe for a fact the span does not hold
    asks for nothing here. ``share`` is None for a span that holds no fact.
    """

    facts: int
    asked: int
    unasked: tuple

    @property
    def share(self):
        return None if self.facts == 0 else self.asked / self.facts


def probe_coverage(span, probes, lexicon=None):
    """The share of the facts of ``span`` that ``probes`` ask for, the facts read again with ``lexicon``."""
    held = held_facts(span, lexicon)
    marks = {probe_mark(probe) for probe in probes}
    unasked = tuple((fact.kind, fact.what, fact.turn_id) for fact in held if fact.mark not in marks)
    return ProbeCoverage(len(held), len(held) - len(unasked), unasked)


# What the probes recall of the facts a reader keeps, read on a labelled set.
# A labelled set is a JSON file of spans: each holds its turns, as the
# generator reads them, and the facts a reader marked to keep, each by its
# class at the turn that first writes it. A fact is recalled when the
# generator draws a probe of its class at its turn with its answer: a date or
# a number by its canonical form, a name by its writing, a code block by the
# marker of its body, a decision by its polarity and the acts, dates and
# numbers it was marked with. The other words of a decision are the
# generator's to key it by, and are never compared. Every figure is a count:
# a set drawn from real conversations is read for its aggregates, and a
# refusal names the span, the turn and the fact where a defect lies, never
# what their text says.
RECALL_CLASSES = ("entity", "date", "number", "code", "decision")
LABELLED_FORMAT = "labelled-recall-set"
LABELLED_VERSION = 1
LABELLED_TAGS = ("known-miss",)
# A span or turn id is named in a refusal, so it is held to the shape of an
# identifier: it can never carry the words of a conversation.
_LABEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_LABEL_ACT = re.compile(r"[a-z]+")
_LABEL_TYPED = ("act:", "date:", "number:")
_LABEL_TURN_KEYS = ("origin", "role", "text", "turn_id")
_LABEL_FACT_KEYS = {
    "entity": ("answer",), "date": ("answer",), "number": ("answer",), "code": ("answer",),
    "decision": ("acts", "dates", "negated", "numbers"),
}
_LABEL_OPTIONAL = ("note", "tags")


class LabelledSetError(ValueError):
    """A labelled set the instrument refuses, named by where its defect lies."""


@dataclass(frozen=True)
class ProbeRecall:
    """What the probes recall of the facts a reader keeps, class by class, and what it was read with.

    ``classes`` holds ``(class, recalled, size)`` in the order of
    ``RECALL_CLASSES``: of ``size`` facts a reader marked, ``recalled`` had
    a probe drawn for them. ``generator`` and ``lexicon`` are the version of
    the generator and the fingerprint of the lexicon the probes were drawn
    with; ``source`` names the set.
    """

    source: str
    generator: int
    lexicon: str
    classes: tuple

    def entry(self):
        """The figure as a report states it: counts, and their ratio to four places, None over no fact."""
        return {
            "source": self.source, "generator": self.generator, "lexicon": self.lexicon,
            "classes": {name: {"size": size, "recalled": recalled,
                               "recall": round(recalled / size, 4) if size else None}
                        for name, recalled, size in self.classes},
        }


@dataclass(frozen=True)
class RecallReading:
    """What a labelled set read: its probe recall, and each fact missed as ``(span, turn, class, index)``."""

    probe_recall: ProbeRecall
    missed: tuple


def _label_strings(value, nonempty=False):
    return isinstance(value, list) and all(isinstance(v, str) and v for v in value) and (bool(value) or not nonempty)


def _label_id(value):
    return isinstance(value, str) and _LABEL_ID.fullmatch(value) is not None


def _label_mark(fact):
    """What a probe must carry to recall ``fact``."""
    kind, turn_id = fact["class"], fact["turn_id"]
    if kind == "decision":
        typed = ([f"act:{a}" for a in fact["acts"]] + [f"date:{d}" for d in fact["dates"]]
                 + [f"number:{n}" for n in fact["numbers"]])
        return kind, turn_id, fact["negated"], frozenset(typed)
    if kind == "code":
        return kind, turn_id, code_marker(fact["writing"])
    return kind, turn_id, fact["answer"]


def _label_drawn(probe):
    """What ``probe`` carries, in the terms of ``_label_mark``."""
    if probe.kind == "decision":
        typed = frozenset(member for member in probe.key if member.startswith(_LABEL_TYPED))
        return probe.kind, probe.turn_id, probe.negated, typed
    return probe.kind, probe.turn_id, probe.answer


def _label_turn_defect(turn):
    if not isinstance(turn, dict) or not set(_LABEL_TURN_KEYS) <= set(map(str, turn)):
        return "a turn holds turn_id, role, origin and text"
    if set(map(str, turn)) - set(_LABEL_TURN_KEYS) - {"segments"}:
        return "a turn holds turn_id, role, origin, text and its segments, and no other field"
    if not _label_id(turn["turn_id"]):
        return "its turn_id is not an identifier (letters, figures, '.', '_' and '-')"
    if not all(isinstance(turn[k], str) for k in ("role", "origin", "text")):
        return "its role, origin and text are strings"
    defect = read_origin(turn)[2]
    return None if defect is None else f"its origin lies outside the grammar of origins: {defect}"


def _label_fact_defect(fact, texts, order):
    if not isinstance(fact, dict):
        return "a fact is an object"
    kind = fact.get("class")
    if kind not in RECALL_CLASSES:
        return f"its class is none of {', '.join(RECALL_CLASSES)}"
    required = {"class", "turn_id", "writing", *_LABEL_FACT_KEYS[kind]}
    held = set(map(str, fact))
    if required - held:
        return f"a fact of class {kind} holds {', '.join(sorted(required))}"
    if held - required - set(_LABEL_OPTIONAL):
        return f"it holds a field a fact of class {kind} does not take"
    turn_id, writing = fact["turn_id"], fact["writing"]
    if not isinstance(turn_id, str) or turn_id not in texts:
        return "its turn is not a turn of its span"
    if not isinstance(writing, str) or not writing:
        return "its writing is not a non-empty string"
    if writing not in texts[turn_id]:
        return "its writing is not in its turn"
    if any(writing in texts[earlier] for earlier in order[: order.index(turn_id)]):
        return "its writing is first written in an earlier turn: a fact is marked where it is first written"
    if "tags" in fact and not (_label_strings(fact["tags"]) and set(fact["tags"]) <= set(LABELLED_TAGS)):
        return f"a tag outside {', '.join(LABELLED_TAGS)}"
    if "note" in fact and not isinstance(fact["note"], str):
        return "its note is not a string"
    if kind != "decision":
        if not isinstance(fact["answer"], str) or not fact["answer"]:
            return "its answer is not a non-empty string"
        if kind == "code" and fact["answer"] != code_key(writing):
            return "its answer is not the code key of its writing, the body of the block"
        return None
    if not isinstance(fact["negated"], bool):
        return "negated is not true or false"
    if not (_label_strings(fact["acts"], nonempty=True) and all(_LABEL_ACT.fullmatch(a) for a in fact["acts"])):
        return "its acts are not one class at least, each a lower-case word"
    if not (_label_strings(fact["dates"]) and _label_strings(fact["numbers"])):
        return "its dates and numbers are lists of answers"
    return None


def labelled_spans(raw):
    """The spans of a labelled set, refused by name when the set is malformed."""
    if not isinstance(raw, dict) or raw.get("format") != LABELLED_FORMAT:
        raise LabelledSetError(f"a labelled set declares the format {LABELLED_FORMAT}")
    if sorted(map(str, raw)) != ["format", "spans", "version"]:
        raise LabelledSetError("a labelled set holds its format, its version and its spans, and no other field")
    if type(raw["version"]) is not int or raw["version"] != LABELLED_VERSION:
        raise LabelledSetError(f"this instrument reads a labelled set of version {LABELLED_VERSION}")
    if not isinstance(raw["spans"], list) or not raw["spans"]:
        raise LabelledSetError("the labelled set holds no span")
    seen = set()
    for position, span in enumerate(raw["spans"]):
        if not isinstance(span, dict) or sorted(map(str, span)) != ["facts", "language", "span_id", "turns"]:
            raise LabelledSetError(f"span {position}: a span holds span_id, language, turns and facts")
        span_id = span["span_id"]
        if not _label_id(span_id):
            raise LabelledSetError(f"span {position}: its span_id is not an identifier")
        if span_id in seen:
            raise LabelledSetError(f"span {span_id!r}: its span_id is given twice")
        seen.add(span_id)
        if span["language"] not in _LEXICON_LANGUAGES:
            raise LabelledSetError(f"span {span_id!r}: its language is none of {', '.join(_LEXICON_LANGUAGES)}")
        turns, facts = span["turns"], span["facts"]
        if not isinstance(turns, list) or not turns or not isinstance(facts, list):
            raise LabelledSetError(f"span {span_id!r}: its turns are a list of one turn at least, its facts a list")
        order = []
        for index, turn in enumerate(turns):
            defect = _label_turn_defect(turn)
            if defect is not None:
                raise LabelledSetError(f"span {span_id!r} turn {index}: {defect}")
            if turn["turn_id"] in order:
                raise LabelledSetError(f"span {span_id!r} turn {index}: its turn_id is given twice")
            order.append(turn["turn_id"])
        texts = {turn["turn_id"]: turn["text"] for turn in turns}
        marks = set()
        for index, fact in enumerate(facts):
            defect = _label_fact_defect(fact, texts, order)
            at = f" at turn {fact['turn_id']!r}" if isinstance(fact, dict) and _label_id(fact.get("turn_id")) else ""
            if defect is None and _label_mark(fact) in marks:
                defect = "it is marked twice: another fact of its span has its class, its turn and its answer"
            if defect is not None:
                raise LabelledSetError(f"span {span_id!r} fact {index}{at}: {defect}")
            marks.add(_label_mark(fact))
    return tuple(raw["spans"])


def _label_unique(pairs):
    keys = [key for key, _value in pairs]
    if len(keys) != len(set(keys)):
        raise LabelledSetError("a key is given twice in one object of the labelled set")
    return dict(pairs)


def load_labelled(path):
    """The spans of the labelled set at ``path``, refused by name when it is not one."""
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.loads(handle.read(), object_pairs_hook=_label_unique)
    except json.JSONDecodeError as exc:
        raise LabelledSetError(f"the labelled set is not JSON: line {exc.lineno} column {exc.colno}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise LabelledSetError(f"the labelled set cannot be read: {type(exc).__name__}") from None
    return labelled_spans(raw)


def measure_recall(spans, lexicon=None, *, source="fixture", generate=None):
    """The probe recall of the generator on labelled ``spans``, class by class, and each fact it missed.

    The probes are drawn with ``lexicon``, the empty one when none is
    given, by ``generate``, which stands for ``generate_probes``. A fact
    is counted once, however many probes answer it.
    """
    generate = generate_probes if generate is None else generate
    lexicon = EMPTY_LEXICON if lexicon is None else lexicon
    size, recalled, missed = Counter(), Counter(), []
    for span in spans:
        drawn = {_label_drawn(probe) for probe in generate([dict(turn) for turn in span["turns"]], lexicon)}
        for index, fact in enumerate(span["facts"]):
            size[fact["class"]] += 1
            if _label_mark(fact) in drawn:
                recalled[fact["class"]] += 1
            else:
                missed.append((span["span_id"], fact["turn_id"], fact["class"], index))
    classes = tuple((name, recalled[name], size[name]) for name in RECALL_CLASSES)
    return RecallReading(ProbeRecall(source, GENERATOR_VERSION, lexicon.fingerprint, classes), tuple(missed))
