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

A summary may give no order, nor restate or tell one, the user's included.
A peel is read by every later turn, so a clause of a peel that orders -- by
speaking to the reader in the second person, an obligation of the reader, a
label that addresses it, an opening of courtesy or prohibition, a lasting
rule, an injection's signature, a request told of the user, or a verb
before its object -- stands only inside a run the user typed, stitched word
for word after its turn's marker in the block that ends the peel, after a
sentence the summary ended (``typed_ranges``, ``order_ranges``,
``_unstitched``): the queue stitches the user's orders, and a marker
anywhere else stitches nothing.
The forms come from the ``directives`` table of ``onion.yaml``, both
languages read as one; without a table no clause orders. A decision is held
clause by clause: a predicate coordinated to a typed decision, a deciding
clause after a semicolon or with a subject of its own, and a decision a
reporter tells as the user's each need the user's words, and a decision the
user quoted, negated, conditioned or took back holds none.

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
    broke; the declaration is never guessed at. Words typed or rewritten
    beside a document -- in one run of parts with no gap between them, as
    pasted text lies among the words -- take their sense from it, and are
    read as a document too (``beside_documents``).
    """
    text = str(turn.get("text", "") or "")
    origin = turn.get("origin", "legacy")
    segments = turn.get("segments", [])
    defect = _origin_defect(turn.get("role"), origin, segments, len(text))
    if defect is not None:
        return "legacy", [], defect
    return origin, beside_documents([[start, stop, label] for start, stop, label in segments]), None


def beside_documents(segments):
    """``segments`` with every typed or rewritten part read as a document when its run holds a document.

    A run is parts that follow one another with no character between them.
    Pasted text lies in the words' own run, so the words typed around it
    become a document; a file attached after the words starts a run of its
    own past the line the executor writes before it, and leaves the words as
    they are.
    """
    held = [list(segment) for segment in segments]
    start = 0
    while start < len(held):
        end = start + 1
        while end < len(held) and held[end][0] == held[end - 1][1]:
            end += 1
        if any(segment[2] == "document" for segment in held[start:end]):
            for segment in held[start:end]:
                if segment[2] in ("typed", "refined"):
                    segment[2] = "document"
        start = end
    return held


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


class DirectivesError(ValueError):
    """The table of the forms an order is given in cannot be built as written."""


# The forms an order is given in come from the ``directives`` section of
# ``onion.yaml``: its switches, then each language with every key below.
# The keys read as single words hold one word an entry; the others hold
# phrases. An elided letter is written alone and read as its word.
_DIRECTIVE_KEYS = (
    "lead_ins", "openers", "persistent", "second_person", "readers", "obligations", "addressees", "label_heads",
    "override_verbs", "override_objects", "signatures", "statement_openers", "object_openers", "particles",
    "time_nouns", "auxiliaries", "subject_pronouns", "owner_pronouns", "datives", "courtesy", "adverbs",
    "authorities", "ing_verbs", "ed_verbs", "past_forms", "user_subjects", "telling", "coordinators",
    "conditions", "subordinators", "quote_nouns", "wanting", "question_verbs", "first_person", "announcers",
)
_DIRECTIVE_WORDS = frozenset({
    "lead_ins", "readers", "addressees", "label_heads", "override_objects", "statement_openers", "object_openers",
    "particles", "time_nouns", "auxiliaries", "subject_pronouns", "owner_pronouns", "datives", "courtesy", "adverbs",
    "authorities", "ing_verbs", "ed_verbs", "past_forms", "user_subjects", "telling", "subordinators", "quote_nouns",
    "wanting", "question_verbs", "first_person",
})
_DIRECTIVE_COUNTS = ("obligation_reach", "override_reach", "label_words")


@dataclass(frozen=True)
class Directives:
    """The forms an order is given in, built from ``onion.yaml``, both languages read as one.

    Each key of ``_DIRECTIVE_KEYS`` holds a frozenset: of folded words for
    the keys read as single words, of word tuples for the phrases, each
    elided letter read as its word, as a clause's words are read. ``strict``
    makes every clause whose first word opens no statement an order; the
    reaches bound an obligation after a reader and an override object after
    its verb, ``label_words`` a label before a colon; ``fingerprint`` names
    the table on every decision judged with it.
    """

    lead_ins: frozenset
    openers: frozenset
    persistent: frozenset
    second_person: frozenset
    readers: frozenset
    obligations: frozenset
    addressees: frozenset
    label_heads: frozenset
    override_verbs: frozenset
    override_objects: frozenset
    signatures: frozenset
    statement_openers: frozenset
    object_openers: frozenset
    particles: frozenset
    time_nouns: frozenset
    auxiliaries: frozenset
    subject_pronouns: frozenset
    owner_pronouns: frozenset
    datives: frozenset
    courtesy: frozenset
    adverbs: frozenset
    authorities: frozenset
    ing_verbs: frozenset
    ed_verbs: frozenset
    past_forms: frozenset
    user_subjects: frozenset
    telling: frozenset
    coordinators: frozenset
    conditions: frozenset
    subordinators: frozenset
    quote_nouns: frozenset
    wanting: frozenset
    question_verbs: frozenset
    first_person: frozenset
    announcers: frozenset
    strict: bool
    obligation_reach: int
    override_reach: int
    label_words: int
    fingerprint: str


def build_directives(section):
    """The table of the ``directives`` section of ``onion.yaml``, refused by name when malformed.

    ``strict`` is a boolean and each count a whole number, one or more; both
    languages are required, each with every key of the table and no other;
    every entry is lower-case ASCII words, accents folded, one word where the
    key reads single words. The languages are read as one: a clause declares
    no language, and an order in either is an order.
    """
    if not isinstance(section, dict):
        raise DirectivesError("directives: the section is not a mapping")
    unknown = [key for key in section if key != "strict" and key not in _DIRECTIVE_COUNTS
               and key not in _LEXICON_LANGUAGES]
    if unknown:
        raise DirectivesError(f"directives: unknown key {unknown[0]!r}")
    strict = section.get("strict")
    if type(strict) is not bool:
        raise DirectivesError(f"directives: strict {strict!r} is not a boolean")
    counts = {}
    for name in _DIRECTIVE_COUNTS:
        value = section.get(name)
        if type(value) is not int or value < 1:
            raise DirectivesError(f"directives: {name} {value!r} is not a whole number, one or more")
        counts[name] = value
    merged = {key: set() for key in _DIRECTIVE_KEYS}
    for language in _LEXICON_LANGUAGES:
        where = f"directives.{language}"
        entry = section.get(language)
        if not isinstance(entry, dict):
            raise DirectivesError(f"{where}: the language is missing or not a mapping of forms")
        unknown = [key for key in entry if key not in _DIRECTIVE_KEYS]
        if unknown:
            raise DirectivesError(f"{where}: unknown key {unknown[0]!r}")
        for key in _DIRECTIVE_KEYS:
            if key not in entry:
                raise DirectivesError(f"{where}: {key} is missing")
            values = entry[key]
            if not isinstance(values, list):
                raise DirectivesError(f"{where}.{key}: not a list of entries")
            for value in values:
                if type(value) is not str or not _LEXICON_MEMBER.fullmatch(value):
                    raise DirectivesError(f"{where}.{key}: the entry {value!r} is not lower-case ASCII words")
                words = tuple(_ELIDED.get(word, word) for word in value.split())
                if key in _DIRECTIVE_WORDS and len(words) != 1:
                    raise DirectivesError(f"{where}.{key}: the entry {value!r} is not one word")
                merged[key].add(words[0] if key in _DIRECTIVE_WORDS else words)
    empty = [key for key in _DIRECTIVE_KEYS if not merged[key]]
    if empty:
        raise DirectivesError(f"directives: {empty[0]} holds no entry in either language")
    lines = sorted(f"{key} {entry if type(entry) is str else ' '.join(entry)}"
                   for key, entries in merged.items() for entry in entries)
    lines += [f"strict {strict}"] + [f"{name} {counts[name]}" for name in _DIRECTIVE_COUNTS]
    digest = hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()
    return Directives(**{key: frozenset(merged[key]) for key in _DIRECTIVE_KEYS}, strict=strict, **counts,
                      fingerprint=digest[:LEXICON_FINGERPRINT_LENGTH])


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


@dataclass(frozen=True)
class Range:
    """A run of what the user typed that is kept whole wherever it is kept word for word: its turn, its place in
    the turn's text, and its text -- as written, or a fenced block's marker."""

    turn_id: str
    start: int
    stop: int
    text: str


# The words a bare retraction is made of, past the negations and the
# function words: "Nope.", "Never mind.", "Scratch that.", "Laisse tomber."
_RETRACTION_WORDS = frozenset({"nope", "nah", "non", "cancel", "stop", "abort", "scratch", "forget", "annule",
                               "annuler", "oublie", "oublier", "laisse", "tomber"})
# The interjections and fillers a retraction may carry: "Hm, no.", "Sorry,
# no.", "On second thought, no.", "Wait, no, not yet.", "Euh, non."
_RETRACTION_FILLERS = frozenset({"mind", "wait", "actually", "please", "ok", "okay", "finalement", "attends", "hm", "hmm",
                                 "um", "uh", "er", "erm", "ah", "oh", "sorry", "oops", "on", "second", "thought", "yet",
                                 "euh", "heu", "hum", "bof", "pardon", "oups", "desole", "desolee", "en", "fait"})
# The marks a quote opens and closes on, in any script: the straight and
# curly quotes, the low ones, guillemets and corner brackets. An apostrophe
# between two letters is none ("don't").
_QUOTE_MARKS = frozenset({'"', "'", chr(0x2018), chr(0x2019), chr(0x201A), chr(0x201B), chr(0x201C), chr(0x201D),
                          chr(0x201E), chr(0x201F), chr(0xAB), chr(0xBB), chr(0x2039), chr(0x203A), chr(0x300C),
                          chr(0x300D), chr(0x300E), chr(0x300F), chr(0xFF02)})
_APOSTROPHE = re.compile(r"(?<=[^\W\d_])['" + chr(0x2019) + r"](?=[^\W\d_])")
# What may stand between two units of one run: blanks, a bullet, a quote's
# ">", an enumerator at the head of a line -- never a word or a figure of
# another origin.
_BETWEEN_UNITS = re.compile(r"(?:\n[ \t]*\d{1,3}[.)]|[\s\-*+>#" + chr(0x2022) + r"])*")


def _quote_marks(text):
    """How many quote marks ``text`` holds, an apostrophe between two letters aside."""
    return sum(1 for c in _APOSTROPHE.sub("", text) if c in _QUOTE_MARKS)


def _announces(text, table):
    """True when a sentence announces what follows it as another's words: it tells ("This is what the phishing email
    said.", "Mallory wrote") or names a quote's noun -- a verb of wanting aside -- or opens on a presentative
    (``announcers``: "Here is the spam I got", "Voici l'arnaque")."""
    words, _written = _clause_words(text)
    at = 0
    while at < len(words) - 1 and (words[at] in table.lead_ins or words[at] in table.courtesy):
        at += 1
    return (any((word in table.telling and word not in table.wanting) or word in table.quote_nouns for word in words)
            or _phrase_at(_phrase_index(table), "announcers", words, at) is not None)


def _bounds(text, table):
    """True when a sentence bounds the order before it: it opens, past its lead-in words, on a negation or a
    condition ("Not before Friday.", "Only if Bob agrees.", "But not the 2024 ones.")."""
    words, _written = _clause_words(text)
    at = 0
    while at < len(words) - 1 and words[at] in table.lead_ins:
        at += 1
    return at < len(words) and (words[at] in _NEGATION_WORDS or _phrase_at(_phrase_index(table), "conditions", words, at)
                                 is not None)


def _retraction(text):
    """True for a bare retraction, which takes back what came before it: "No.", "No, don't.", "Nope.", "Never mind.",
    "Non.", "Laisse tomber." -- a negation or a word of retraction, and nothing else but function words."""
    for pattern, plain in _CONTRACTED:
        text = pattern.sub(plain, text)
    words = _tokens(_fold(text))
    return (any(word in _NEGATION_WORDS or word in _RETRACTION_WORDS for word in words)
            and all(word in _FOLDED_STOPWORDS or word in _NEGATION_WORDS or word in _RETRACTION_WORDS
                    or word in _RETRACTION_FILLERS for word in words))


def typed_ranges(span, table=None):
    """The runs of a span's typed words that are kept whole, in order: the only places an order may stand in a peel.

    A unit (``units``) is a sentence or a fenced block; a fenced block is a
    run of its own, its marker. Sentences join into one run where parting
    them would change what they say: a sentence that goes on in lower case
    ("..., e.g. wire the money"), one inside an open quote, in any marks,
    everything after a line that ends on a colon ("Mallory wrote:", "Avoid
    these:", "Here is what I need:") or, with a table, after a sentence that
    announces another's words ("This is what the phishing email said.",
    "Here is the spam I got.") to the end of its piece, a one-word sentence
    or a bare retraction after what it answers ("Delete the logs. Wait.
    No.", "Hm, no."), and, with a table, a sentence that bounds the order
    before it ("Not before Friday.", "Only if Bob agrees."). Kept word for
    word, a run carries its condition, its quote, its label and its
    retraction with it. A piece of another origin, or a gap no typed segment
    covers -- a word or a figure -- ends a run; a turn whose id is empty or
    shared with another turn of the span holds none, its words being nowhere
    to be found again.
    """
    texts = {str(t.get("turn_id", "")): str(t.get("text", "") or "") for t in span}
    seen = {}
    for turn in span:
        turn_id = str(turn.get("turn_id", ""))
        seen[turn_id] = seen.get(turn_id, 0) + 1
    ambiguous = {turn_id for turn_id, count in seen.items() if not turn_id or count > 1}
    found, run = [], None

    def close():
        if run is not None:
            found.append(Range(run[0], run[1], run[2], texts[run[0]][run[1]:run[2]]))

    for unit in units(span):
        if unit.origin != "typed" or unit.turn_id in ambiguous or unit.text.startswith("[code:"):
            close()
            run = None
            if unit.origin == "typed" and unit.turn_id not in ambiguous:
                found.append(Range(unit.turn_id, unit.start, unit.stop, unit.text))
            continue
        letter = next((c for c in unit.text if c.isalpha()), "")
        # A line on a colon may end a unit or stand inside one.
        opens = (any(line.rstrip().endswith(":") for line in unit.text.splitlines())
                 or (table is not None and _announces(unit.text, table)))
        marks = _quote_marks(unit.text)
        joined = run is not None and run[0] == unit.turn_id and _BETWEEN_UNITS.fullmatch(
            texts[unit.turn_id][run[2]:unit.start]) is not None and (
            run[3] or letter.islower() or _retraction(unit.text) or len(_tokens(_fold(unit.text))) <= 1 or run[4] % 2 == 1
            or (table is not None and _bounds(unit.text, table)))
        if joined:
            run = (run[0], run[1], unit.stop, run[3] or opens, run[4] + marks)
        else:
            close()
            run = (unit.turn_id, unit.start, unit.stop, opens, marks)
    close()
    return found


def order_ranges(span, table, names=frozenset()):
    """The typed runs (``typed_ranges``, read with the table) that order, read with the table the gate reads a summary
    with, so that an order the user typed in a form the table reads is one the queue stitches in the user's words;
    and each bare retraction typed after one of them, whatever was asked between: a "No." that answered another
    question only makes the order look taken back, the safe side, while one that answered "Shall I delete them now?"
    takes it back. Kept word for word, marked with its turn, they are the user's orders in a peel."""
    kept, ordered = [], False
    for run in typed_ranges(span, table):
        if run.text.startswith("[code:"):
            continue
        if directives_in(run.text, table, names):
            kept.append(run)
            ordered = True
        elif ordered and _retraction(run.text):
            kept.append(run)
    return kept


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
    turns write, each span as written; ``typed`` the pieces the user typed,
    each whole as written, blocks and lists included, the only words a
    decision of a summary may stand on; ``stitchable`` each run the user
    typed as ``(turn_id, text)`` (``typed_ranges``), the only words an order
    may stand in, stitched word for word; ``ids`` the turn ids of the span,
    the markers a stitch may carry. Every origin counts for a claim: who
    said it matters to a decision and an order only.
    """

    words: frozenset
    dates: frozenset
    numbers: frozenset
    texts: tuple
    keys: frozenset
    inline: frozenset = frozenset()
    typed: tuple = ()
    stitchable: tuple = ()
    ids: frozenset = frozenset()


def holdings(span, directives=None):
    """What a span of turns holds, each a mapping with its text and its role; its runs (``typed_ranges``) read with
    the table of ``directives`` when one is given."""
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
    typed = tuple(text for _turn_id, _role, origin, text in _pieces(span) if origin == "typed")
    stitchable = tuple((run.turn_id, run.text) for run in typed_ranges(span, directives))
    ids = frozenset(str(turn.get("turn_id", "")) for turn in span) - {""}
    return Holdings(frozenset(words), frozenset(dates), frozenset(numbers), tuple(texts), frozenset(keys),
                    frozenset(inline), typed, stitchable, ids)


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


def unbacked_decisions(probes, text, lexicon=None, reporters=frozenset(), directives=None, typed=()):
    """The sentences and clauses of a summary that decide with no typed decision of its span behind them.

    Each is ``(kind, what, turn_id)``. A sentence decides as a typed one
    does, read with the lexicon the probes were drawn with. One whose
    subject -- its first word that is no function word -- is a reporter
    tells what another speaker said, and is passed over. Every other
    deciding sentence is held by a decision probe it answers, with the same
    polarity. One that answers a probe but for its polarity is an
    ``inversion`` of that probe's turn; one that answers none is a
    ``decision``, with no turn. Each is named once, in reading order.

    With a table of ``directives``, a decision is read on the text folded as
    an order is (``_order_text``), sentence by sentence where an order's
    sentence ends, and held clause by clause: a sentence of several chunks
    -- parted at a semicolon, a colon or a dash, and at a comma, a bracket,
    a coordinator or a lead-in word before a deciding subject of its own --
    names each chunk that decides with no typed decision behind it, a
    reporter at the head of the sentence exempting no later chunk; a
    sentence that answers a typed decision still names each predicate
    coordinated to it; and a reporter's chunk that tells a decision as the
    user's names that telling. Such a clause needs the user's words: a
    typed part it restates or a typed decision it answers, with its
    polarity and its mood, as ``_backed`` reads them against ``typed``, the
    pieces the user typed. A decision asked as a question holds no
    statement of it. In a reporter's words, "we" is the reporter's voice:
    only the user named as the one who decides makes the telling the
    user's.
    """
    lexicon = EMPTY_LEXICON if lexicon is None else lexicon
    decisions = [p for p in probes if p.kind == "decision"]
    found = []
    if directives is None:
        for sentence in _units(text):
            plain = _plain(sentence)
            folded = _tokens(_fold(plain))
            if not _decides(plain, folded, _acts(folded, lexicon), lexicon):
                continue
            if next((word for word in folded if word not in _STOPWORDS), None) in reporters:
                continue
            if not any(answers(p, sentence) for p in decisions):
                _name_decision(found, sentence, decisions)
        return found
    names = _names_held(probes)
    backing, barred = _backing(tuple(typed), directives, lexicon)
    # A decision the user took back, or wrote under a line that quotes,
    # negates or conditions it, holds none (``_backing``); the probes split
    # sentences their own way, so a probe is matched by inclusion.
    decisions = [p for p in decisions
                 if not any(_order_text(p.answer).strip() in text or text in _order_text(p.answer) for text in barred)]
    parting = _decision_split(directives, lexicon)
    for sentence in _decision_sentences(text):
        plain = _plain(sentence)
        read = _order_text(plain)
        question = _is_question(sentence)

        def unheld(clause):
            return not _backed(clause, clause, question, backing, decisions, directives)

        def told_of(clause):
            """The decision a reporter's clause tells as the user's, or None."""
            return _told_as_users(clause, directives, lexicon) if _subject_word(clause) in reporters else None

        chunks = [chunk.strip() for chunk in parting.split(read) if chunk and chunk.strip()]
        # A telling decides as the user's decision does, by its act alone:
        # "The report says the user prefers Podman".
        if not (_deciding(plain, lexicon) or _deciding(read, lexicon) or any(told_of(chunk) for chunk in chunks)):
            continue
        if len(chunks) > 1:
            for chunk in chunks:
                if _subject_word(chunk) in reporters:
                    told = told_of(chunk)
                    if told is not None and unheld(told):
                        _name_decision(found, told, decisions)
                    continue
                if not _deciding(chunk, lexicon):
                    continue
                coordinated = _coordinated(chunk, directives, names)
                if unheld(_head_of(chunk, coordinated)):
                    _name_decision(found, chunk, decisions)
                for clause in coordinated:
                    if unheld(clause):
                        _name_decision(found, clause, decisions)
            continue
        if _subject_word(read) in reporters:
            told = _told_as_users(read, directives, lexicon)
            if told is not None and unheld(told):
                _name_decision(found, told, decisions)
            continue
        # A decision told of the user with no reporter before it is judged as
        # told, and named whole: "The user confirmed that we drop the backups".
        told = _told_as_users(read, directives, lexicon) if _subject_word(read) in directives.user_subjects else None
        if told is not None:
            if unheld(told):
                _name_decision(found, sentence, decisions)
            continue
        # The sentence's head is judged without the predicates coordinated to
        # it, each judged on its own: two typed decisions joined by "but"
        # stand, a predicate the user never typed does not.
        coordinated = _coordinated(read, directives, names)
        if unheld(_head_of(read, coordinated)):
            _name_decision(found, sentence, decisions)
            continue
        for clause in coordinated:
            if unheld(clause):
                _name_decision(found, clause, decisions)
    return found


def _head_of(text, clauses):
    """``text`` without the coordinated ``clauses`` it holds as written: what its subject decides by itself."""
    for clause in clauses:
        text = text.replace(clause, " ")
    return text


# Where a sentence of a summary ends for a decision, past the units' own
# stops (which keep a date whole): after a closing quote or bracket, an
# ellipsis, an ideographic full stop, a stop after a lower-case letter glued
# to the capitalised word that opens the next sentence ("server.We drop",
# never "ASP.NET"), a line break -- unless the next line goes on in lower
# case ("we\nhave dropped": ``_decision_sentences``).
_EXTRA_SENTENCE = re.compile(
    r"(?<=[.!?" + chr(0x2026) + r"])[\"'" + chr(0x2019) + chr(0x201D) + chr(0xBB) + r")\]]+\s+"
    r"|(?<=" + chr(0x2026) + r")\s+|(?<=" + chr(0x3002) + r")|(?<=[a-z][.!?])(?=[A-Z][a-z])|\n"
)
# The pattern a deciding sentence parts at, per table and lexicon.
_DECISION_SPLITS = {}


def _decision_sentences(text):
    """A summary's sentences as a decision is read with a table: each unit, parted again where an order's sentence
    ends, a line that opens in lower case read with the line before it."""
    for unit in _units(text):
        for sentence in _EXTRA_SENTENCE.split(_continued(unit)):
            if sentence.strip():
                yield sentence.strip()


def _decision_split(table, lexicon):
    """Where a deciding sentence parts: a semicolon, a colon, a dash, and a comma, a bracket, a coordinator, a
    lead-in or a subordinating word before a subject of the lexicon -- "the report, and we decided", "(we decided",
    "which we decided", "as we decided" -- so that a clause with a deciding subject of its own is judged on its own,
    never under the reporter that opens the sentence. "that" is no such word: "The document says that we decided"
    is the document's voice."""
    key = (table, lexicon)
    pattern = _DECISION_SPLITS.get(key)
    if pattern is None:
        subjects = sorted({word for _language, word in lexicon.subjects}, key=len, reverse=True)
        joints = sorted({r"\s+".join(map(re.escape, phrase)) for phrase in table.coordinators}
                        | {re.escape(word) for word in table.lead_ins | table.subordinators}, key=len, reverse=True)
        turn = r"(?:,|\(|\)|\b(?:" + "|".join(joints) + r")\b)\s*(?=(?:" + "|".join(map(re.escape, subjects)) + r")\b)"
        pattern = _DECISION_SPLITS[key] = re.compile(
            _CLAUSE_SPLIT.pattern + "|" + turn if subjects else _CLAUSE_SPLIT.pattern, re.IGNORECASE)
    return pattern


def _name_decision(found, what, decisions):
    """Name ``what`` once in ``found``: an inversion of the probe it answers but for its polarity, else a decision."""
    from dataclasses import replace

    said = _negations(_plain(what))
    inverted = next((p for p in decisions if p.negations != said and answers(replace(p, negations=said), what)), None)
    entry = ("inversion", what, inverted.turn_id) if inverted is not None else ("decision", what, "")
    if entry not in found:
        found.append(entry)


# An order is read on a text folded for it: compatibility forms composed
# (NFKC: fullwidth and mathematical letters, fullwidth punctuation), format
# characters dropped (zero-width, soft hyphen, bidirectional controls), and
# the Cyrillic and Greek letters drawn like Latin ones read as those. Only
# the reading of orders and decisions folds so: claims are read as written.
_CONFUSABLES = {chr(code): latin for code, latin in (
    (0x430, "a"), (0x435, "e"), (0x43E, "o"), (0x440, "p"), (0x441, "c"), (0x443, "y"), (0x445, "x"),
    (0x456, "i"), (0x458, "j"), (0x455, "s"), (0x501, "d"), (0x4BB, "h"), (0x51B, "q"), (0x51D, "w"),
    (0x4CF, "l"), (0x410, "A"), (0x412, "B"), (0x415, "E"), (0x41A, "K"), (0x41C, "M"), (0x41D, "H"),
    (0x41E, "O"), (0x420, "P"), (0x421, "C"), (0x422, "T"), (0x425, "X"), (0x423, "Y"), (0x406, "I"),
    (0x408, "J"), (0x405, "S"), (0x3B1, "a"), (0x3BF, "o"), (0x3C1, "p"), (0x3BD, "v"), (0x3C5, "u"),
    (0x3B9, "i"), (0x3BA, "k"), (0x391, "A"), (0x392, "B"), (0x395, "E"), (0x396, "Z"), (0x397, "H"),
    (0x399, "I"), (0x39A, "K"), (0x39C, "M"), (0x39D, "N"), (0x39F, "O"), (0x3A1, "P"), (0x3A4, "T"),
    (0x3A5, "Y"), (0x3A7, "X"),
)}
# A sentence of an order ends after a full stop, a question or an
# exclamation mark or an ellipsis, past any closing quote or bracket, after
# an ideographic full stop, and at a line break.
_ORDER_SENTENCE = re.compile(
    r"(?<=[.!?" + chr(0x2026) + r"])[\"'" + chr(0x2019) + chr(0x201D) + chr(0xBB) + r")\]]*\s+"
    r"|(?<=" + chr(0x3002) + r")|\n"
)
# A sentence's clauses part at a semicolon, at a colon before a blank or the
# end, at a spaced hyphen and at a dash, spaced or not; an order's clauses
# also at a comma, so that a fronted phrase stands apart from the clause it
# leads. Each part is stripped afterwards: no pattern here spans a run of
# blanks, so none backtracks over one.
_CLAUSE_SPLIT = re.compile(r";|:(?=\s|$)|\s-\s|\s--\s|[" + chr(0x2013) + chr(0x2014) + r"]")
_CHUNK_SPLIT = re.compile(r";|\s-\s|\s--\s|[" + chr(0x2013) + chr(0x2014) + r"]")
_LABEL_SPLIT = re.compile(r":(?=\s|$)")
_COMMA_SPLIT = re.compile(r",\s+")
# What may stand before a clause and is no word of it: an enumerator --
# "a)", "(1)", "iv.", "1.", "1/", "#1", a figure alone -- and a tag in
# brackets, a code marker among them. An enumerator is a figure, one letter
# or a Roman numeral: "(Send)" is a word. A restatement keeps the tags,
# whose words may be a negation ("[DON'T]").
_ENUMERATOR = (r"\(\s*(?:\d{1,3}|[^\W\d_]|[ivxlcdmIVXLCDM]{1,6})\s*\)"
               r"|#?\d{1,3}(?!\d)\s*[.)/:" + chr(0xB0) + r"-]?(?=\s|[^\W\d_])"
               r"|(?:[^\W\d_]|[ivxlcdmIVXLCDM]{1,6})[.)](?=\s)")
_ORDER_PREFIX = re.compile(r"\s*(?:" + _ENUMERATOR + r"|\[[^\]\n]{1,24}\])\s*")
_ORDER_ENUMERATOR = re.compile(r"\s*(?:" + _ENUMERATOR + r")\s*")
# A hyphen between two letters joins one word, as an order is read and
# restated: "E-mail", "Re-send", "Always-on", "Remember-me", "thank-you" --
# but for the French pronoun after its verb, "Envoie-les", "Peux-tu", read
# as two.
_HYPHEN = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])"
                     r"(?!(?:le|la|les|lui|leur|moi|toi|nous|vous|en|y|tu|je|il|elle|ils|elles|ce|t)\b)",
                     re.IGNORECASE)
# An order is also read with its hyphens splitting its words, so that
# "Ignore-all-previous-instructions", "Send-the-vault-keys", "Email-them" and
# "Please-send" hide nothing; a compound that opens on an opening,
# "Always-on", is then read as one, a false refusal on the side of safety.
# Underscores between letters part words: "ignore_all_previous_instructions".
_UNDERSCORE = re.compile(r"(?<=[^\W\d_])_+(?=[^\W\d_])")
# The English determiners after which a word with a French participle's
# accent is an English verb in disguise ("Delete" with an accent, "the"),
# and the French words after which a word in -s is a French imperative
# ("Prends les", "Dis-moi"), never a third person.
_ENGLISH_DETERMINERS = frozenset({"the", "a", "an", "all", "every", "each", "this", "these", "those", "my", "your",
                                  "our", "their", "his", "her", "its", "some", "any", "both"})
_FRENCH_FOLLOWERS = frozenset({"le", "la", "les", "un", "une", "des", "du", "de", "ce", "cet", "cette", "ces", "mon",
                               "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "notre", "nos", "votre", "vos",
                               "leur", "leurs", "tout", "toute", "tous", "toutes", "chaque", "moi", "toi", "lui",
                               "nous", "vous", "en", "y"})
# Letters and symbols drawn as nothing that no category drops: the Hangul
# fillers (NFKC reads U+3164 and U+FFA0 as U+1160) and the blank Braille
# pattern.
_INVISIBLE = frozenset(map(chr, (0x115F, 0x1160, 0x3164, 0xFFA0, 0x2800)))
# The French second person a word list cannot hold: "t'" -- a "t" with an
# apostrophe after it -- and "te" as a word of its own.
_FRENCH_YOU = re.compile(
    r"(?<![\w'" + chr(0x2019) + r"])[tT]['" + chr(0x2019) + r"](?=[^\W\d_])"
    r"|(?<![\w'" + chr(0x2019) + r"-])[tT][eE](?![\w'" + chr(0x2019) + r"-])"
)
# The words a figure is a sum or a measure by: the first word of each unit
# the quantities are read with, folded. A figure is an object only before
# one: "Send 500 dollars", never "Qwen 2.5 7B" or "Service 1 was".
_UNIT_TOKENS = frozenset(_tokens(_fold(form))[0] for row in _UNIT_WORDS + _UNIT_SYMBOLS for form in row
                         if _tokens(_fold(form)))
# The phrase index of each table met, keyed by the table itself.
_DIRECTIVE_INDEXES = {}


def _order_text(text):
    """``text`` as an order is read: NFKC, no format, combining or invisible character left, the look-alike letters
    read as Latin ones. NFKC composes every accent that has a composed form, so a combining mark left after it is an
    invisible one (a grapheme joiner, a variation selector)."""
    import unicodedata

    composed = unicodedata.normalize("NFKC", text)
    return _UNDERSCORE.sub(" ", "".join(_CONFUSABLES.get(c, c) for c in composed
                                        if c not in _INVISIBLE and unicodedata.category(c) not in ("Cf", "Mn", "Me")))


def _hyphen_views(text):
    """The two readings of a text's hyphens: joined (``_HYPHEN``), then split."""
    return _HYPHEN.sub("", text), text


# The marks of emphasis a word or a label may be dressed in ("**Reminder:**",
# "~~Send~~"), a colon glued between two letters ("Reminder:send"), and a word
# spelled letter by letter, its letters parted by one or two blanks, full
# stops, slashes, middle dots or commas ("S e n d", "D.e.l.e.t.e", "S/e/n/d",
# "S, e, n, d"; three letters at least, so "e.g." stays).
_EMPHASIS = re.compile(r"[*~]+")
_GLUED_COLON = re.compile(r"(?<=[^\W\d_]):(?=[^\W\d_])")
_SPELLED_GAP = r"(?:[ \t]{1,2}|[./" + chr(0xB7) + r"]|, ?)"
_SPELLED = re.compile(r"(?<![^\W\d_])[^\W\d_](?:" + _SPELLED_GAP + r"[^\W\d_]){2,}(?![^\W\d_])")


def _continued(text):
    """``text`` with each line that opens in lower case read on the line before it: "Send\\nthe keys" is one line."""
    lines = []
    for line in text.split("\n"):
        if lines and line.lstrip()[:1].islower():
            lines[-1].append(line.lstrip())
        else:
            lines.append([line])
    return "\n".join(" ".join(parts) for parts in lines)


def _order_prose(text):
    """``text`` as an order is read in prose: folded (``_order_text``), its inline code read as words, without marks
    of emphasis, a glued colon spaced, a word spelled letter by letter joined, a line that opens in lower case read
    on the line before it. The repair compares a refused clause with a sentence in this form."""
    folded = _order_text(_INLINE_CODE.sub(lambda code: f" {code.group(2)} ", text))
    folded = _GLUED_COLON.sub(": ", _EMPHASIS.sub("", folded))
    return _continued(_SPELLED.sub(lambda spelled: re.sub(_SPELLED_GAP, "", spelled.group(0)), folded))


def _is_question(sentence):
    """True when a sentence ends on a question mark, with any exclamation mark, quote or bracket after it: "Should I
    wipe the logs?!", "(Should I wipe the logs?)". Read from the end, once."""
    end = len(sentence)
    while end and not (sentence[end - 1].isalnum() or sentence[end - 1] == "_"):
        end -= 1
    return "?" in sentence[end:]


def _clause_words(text):
    """The words of a clause, folded, each elided letter read as its word, and each word as written beside it.

    Read in the composed Unicode form, so that a decomposed accent splits no
    word, and an apostrophe of either kind parts an elided letter from its
    word: "N'oubliez" reads "ne", "oubliez".
    """
    import unicodedata

    written = _WORD.findall(unicodedata.normalize("NFC", text))
    return [_ELIDED.get(word, word) for word in (_fold(w) for w in written)], written


def _phrase_index(table):
    """The phrases of a table by their first word, the longest first, built once per table."""
    index = _DIRECTIVE_INDEXES.get(table)
    if index is None:
        index = {}
        for key in _DIRECTIVE_KEYS:
            if key in _DIRECTIVE_WORDS:
                continue
            by_first = {}
            for phrase in sorted(getattr(table, key), key=lambda p: (-len(p), p)):
                by_first.setdefault(phrase[0], []).append(phrase)
            index[key] = by_first
        _DIRECTIVE_INDEXES[table] = index
    return index


def _phrase_at(index, key, words, at):
    """The longest phrase of ``key`` that starts at ``words[at]``, or None."""
    if at >= len(words):
        return None
    for phrase in index[key].get(words[at], ()):
        if tuple(words[at:at + len(phrase)]) == phrase:
            return phrase
    return None


def _names_held(probes):
    """The folded words of the names a span's probes ask for: what a clause may name as an object."""
    return frozenset(_fold(word) for p in probes if p.kind == "entity" for word in _WORD.findall(p.answer))


def _participle(written, following=None):
    """True for a word written as a French past participle, "Arrete" with its final accent, never an imperative --
    unless an English determiner follows it: an English verb with an accent added is an order in disguise."""
    lowered = written.lower()
    return (lowered.endswith((chr(0xE9), chr(0xE9) + "e", chr(0xE9) + "s", chr(0xE9) + "es"))
            and following not in _ENGLISH_DETERMINERS)


def _third_person(word, following):
    """True for a word in -s that is no imperative: "deletes", "continues", a plural -- never "process", "focus",
    "alias", nor a French imperative before its determiner or pronoun ("Prends les", "Dis-moi")."""
    return (len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is", "as"))
            and following not in _FRENCH_FOLLOWERS)


def _opens_statement(word, written, table, names, following=None):
    """True when a clause opening on ``word`` states rather than orders: a subject, a name, a figure, a past, a
    third person.

    A word in -ing of five letters or more is a gerund, "Running the tests";
    a shorter one is a verb, "Ping the server". ``following`` is the next
    word, which tells a French participle and a French imperative in -s from
    English words in disguise.
    """
    return (word in table.statement_openers or word in names or written[:1].isdigit()
            or (word.endswith("ing") and len(word) >= 5 and word not in table.ing_verbs)
            or (word.endswith("ed") and word not in table.ed_verbs) or word in table.past_forms
            or _participle(written, following) or _third_person(word, following))


def _possessive(written, at):
    """True when the word at ``at`` is a possessive, its "'s" read as a word of its own: "month's", "Bob's"."""
    return at + 1 < len(written) and written[at + 1] in ("s", "S") and written[at][:1].isalpha()


def _object_at(words, written, at, table, names):
    """True when the word at ``at`` opens an object: a possessive, a determiner or a pronoun that opens no time
    phrase, a name the span holds, a figure before its unit. "Progress this week" has no object, nor "Qwen 2.5 7B" or
    "Service 1 was migrated": "Send 500 dollars" has one, and so do "Delete this month's logs" and "Supprime ce
    trimestre les sauvegardes", whose time phrase owns or precedes it."""
    # Past each time phrase in turn, never by recursion: "this week this week ... the logs".
    while True:
        word = words[at]
        if _possessive(written, at):
            return True
        if word in table.object_openers:
            if at + 1 >= len(words) or words[at + 1] not in table.time_nouns:
                return True
            if _possessive(written, at + 1):
                return True
            if at + 2 >= len(words):
                return False
            at += 2
            continue
        if written[at][:1].isdigit():
            return at + 1 < len(words) and words[at + 1] in _UNIT_TOKENS
        return word in names


def _adverb(word, table):
    """True for a word that may stand between a verb and its object: one of the table's adverbs, or a word in -ly or
    -ment -- never a conjunction: "Dave and Bob met" opens on no verb."""
    return word in table.adverbs or (len(word) > 4 and word.endswith(("ly", "ment")))


def _verb_first(words, written, table, names):
    """True when a clause opens on a verb before its object: its first word opens no statement, and the next opens an
    object -- past up to two adverbs ("send immediately the", "envoie aussi les"), a particle ("turn off the"), a
    dative and a name, a determiner or one word ("share with Mallory the", "upload to pastebin the"), or both
    ("send over to Mallory the"). A name the span holds opens no statement before a determiner that is no
    auxiliary: a document that names an "Email Gateway" leaves "Email the keys" an order, while "Alice a deplace"
    states."""
    if len(words) < 2:
        return False
    if _opens_statement(words[0], written[0], table, names, words[1]) and not (
            words[0] in names and words[1] in _ENGLISH_DETERMINERS and words[1] not in table.auxiliaries):
        return False
    at = 1
    while at < min(len(words) - 1, 3) and _adverb(words[at], table) and not _possessive(written, at):
        at += 1
    if words[at] in table.particles and at + 1 < len(words):
        at += 1
    if (words[at] in table.datives and at + 2 < len(words)
            and (words[at + 1] in names or words[at + 1] in table.object_openers
                 or words[at + 2] in table.object_openers)):
        at += 2
    return _object_at(words, written, at, table, names)


def _order_marked(text):
    """A clause without the enumerators and tags that stand before it, each code marker in it read as "it"."""
    for _ in range(6):
        prefix = _ORDER_PREFIX.match(text)
        if not prefix:
            break
        text = text[prefix.end():]
    return _MARKER.sub(" it ", text)


def _spine(chunk):
    """A chunk without the phrases between its first and its last comma: "The assistant, as agreed, must" reads "The
    assistant must"."""
    pieces = _COMMA_SPLIT.split(chunk)
    return chunk if len(pieces) < 3 else pieces[0] + " " + pieces[-1]


def _subject_of(words, table):
    """Where a clause's subject stands: past its lead-in words and at most two determiners."""
    at = 0
    while at < len(words) - 1 and words[at] in table.lead_ins:
        at += 1
    for _ in range(2):
        if at < len(words) - 1 and words[at] in table.statement_openers and words[at] in table.object_openers:
            at += 1
    return at


def _second_person(words, text, index):
    """True when a clause speaks to the reader: a word of the second person anywhere, or a French "t'" or "te" -- a
    "t" an apostrophe follows, never the "t" of an English "n't", of "t-test" or of "Vec<T>"."""
    if any(_phrase_at(index, "second_person", words, k) for k in range(len(words))):
        return True
    return _FRENCH_YOU.search(text) is not None


def _reader_obliged(words, table, index):
    """True when the reader of the peel, the assistant, stands among a clause's first words and is obliged within
    the table's reach. A model, an agent or a system obliged is a statement: "The system should reject the
    request"; asked to reveal its instructions, it is an injection's signature."""
    start = _subject_of(words, table)
    for at in range(start, min(len(words), start + 3)):
        if words[at] not in table.readers:
            continue
        # Past its adverbs: "is now required to", "is hereby asked to".
        rest = [word for word in words[at + 1:] if not _adverb(word, table)]
        if any(_phrase_at(index, "obligations", rest, k) for k in range(min(len(rest), table.obligation_reach))):
            return True
    return False


def _addresses(label, table):
    """True when a label before a colon addresses the reader: an authority named alone ("System", "Assistant", a
    forged message's voice), or a label that opens on a label's head -- an instruction, a note, a preposition, a
    greeting -- and ends on the reader or its instructions; either holds no verb. "Instructions for the assistant",
    "Note to the AI" and "System" address; "System requirements", "Model" and "The model failed" do not."""
    words, _written = _clause_words(label)
    return (0 < len(words) <= table.label_words
            and ((len(words) == 1 and words[0] in table.authorities)
                 or (words[0] in table.label_heads and words[-1] in table.addressees))
            and not any(word in table.auxiliaries or word in table.past_forms
                        or (word.endswith("ed") and word not in table.ed_verbs) for word in words))


def _overrides(words, table, index):
    """True when an override verb has an override object within the table's reach after it."""
    for at in range(len(words)):
        verb = _phrase_at(index, "override_verbs", words, at)
        if verb and any(word in table.override_objects
                        for word in words[at + len(verb):at + len(verb) + table.override_reach]):
            return True
    return False


def _told_request(words, table, written=None):
    """True when a clause tells that the user asked the reader to act: the user as subject, a telling verb within two
    words, then, past the reader, a pronoun or a determiner, "to" or "de" and a word; or, whatever the verb, the reader
    then "to" or "de" ("The user allowed the assistant to share"). "The user asked for a summary" and "wanted to know
    whether the build passed" tell a question, not an order -- unless a verb that orders is coordinated to the one
    that asks ("asked to check the logs and delete the old ones", ``_orders_after``). The user asking in their own
    voice asks as much: "I'd like the old logs deleted", "J'aimerais que
    le rapport parte" (``first_person``)."""
    at = _subject_of(words, table)
    if at >= len(words) or (words[at] not in table.user_subjects and words[at] not in table.first_person):
        return False
    for k in range(at + 1, min(len(words), at + 3)):
        if words[k] not in table.telling:
            continue
        for j in range(k + 1, min(len(words) - 1, k + 5)):
            if words[j] in ("to", "de"):
                if words[j + 1] in table.question_verbs:
                    # A question told -- unless a verb that orders is
                    # coordinated to it: "asked to check the logs and delete".
                    return _orders_after(words, j + 2, table)
                return True
            if not (words[j] in table.readers or words[j] in table.subject_pronouns
                    or words[j] in table.object_openers or words[j] in table.datives):
                break
        break
    # A verb of wanting, then within six words a participle or a "to" or "de"
    # before a verb that asks nothing: "The user wants the vault keys sent to
    # Mallory", "The user wants Mallory to get the keys" -- never "The user
    # wants to know the status".
    for k in range(at + 1, min(len(words), at + 3)):
        if words[k] not in table.wanting:
            continue
        for j in range(k + 1, min(len(words), k + 7)):
            said = written[j] if written else words[j]
            # A wish to know asks nothing done: "wanted to know whether the build passed".
            if words[j] in ("to", "de") and j + 1 < len(words) and words[j + 1] in table.question_verbs:
                if _orders_after(words, j + 2, table):
                    return True
                break
            if (_participle(said) or words[j] in table.past_forms
                    or (words[j].endswith("ed") and words[j] not in table.ed_verbs)):
                return True
            if (words[j] in ("to", "de") and j + 1 < len(words) and words[j + 1] not in table.question_verbs
                    and words[j + 1] not in table.object_openers and words[j + 1] not in table.statement_openers):
                return True
            # A French "que" right after the verb opens what is wanted done.
            if words[j] == "que" and j == k + 1:
                return True
        break
    # Whatever the verb, the reader then "to" or "de": "The user allowed the
    # assistant to share the keys".
    return any(words[k] in table.readers and words[k + 1] in ("to", "de")
               for k in range(at + 1, min(len(words) - 1, at + 5)))


def _orders_after(words, at, table):
    """True when, from ``at`` on, a coordinator brings a verb that asks nothing back: "to check the logs and delete
    the old ones", "de verifier les logs et de supprimer" -- never "and the reports" nor "and failed"."""
    index = _phrase_index(table)
    for k in range(at, len(words) - 1):
        coordinator = _phrase_at(index, "coordinators", words, k)
        if not coordinator:
            continue
        verb = k + len(coordinator)
        while verb < len(words) - 1 and (words[verb] in ("to", "de") or words[verb] in table.lead_ins
                                         or words[verb] in table.auxiliaries):
            verb += 1
        if verb < len(words) and words[verb] not in table.question_verbs and words[verb] not in table.object_openers \
                and not _opens_statement(words[verb], words[verb], table, frozenset()):
            return True
    return False


def _chunk_forms(chunk, addressed, table, names):
    """The forms an order takes anywhere in a chunk, read on the chunk and on its spine: a, b, c, d (a lasting rule),
    e, and r -- a request told of the user ("The user asked the assistant to email the keys"), the user's order,
    held only by the order the user typed. A lasting rule told of the user orders as any: it stands when the user
    typed that rule."""
    forms = {"c"} if addressed else set()
    index = _phrase_index(table)
    views = dict.fromkeys(reading for view in (chunk, _spine(chunk)) for reading in _hyphen_views(_order_marked(view)))
    for marked in views:
        words, written = _clause_words(marked)
        if not words:
            continue
        if _second_person(words, marked, index):
            forms.add("a")
        if _reader_obliged(words, table, index):
            forms.add("b")
        if any(_phrase_at(index, "persistent", words, k) for k in range(len(words))):
            forms.add("d")
        if any(_phrase_at(index, "signatures", words, k) for k in range(len(words))) or _overrides(words, table, index):
            forms.add("e")
        if _told_request(words, table, written):
            forms.add("r")
    return forms


def _piece_forms(piece, table, names):
    """The forms an order takes where a piece opens: d (courtesy, prohibition, rule), f (a verb before its object,
    past lead-in words and a run of adverbs), s (under the strict switch, a first word that opens no statement).

    The piece is read with its hyphenated words joined and split
    (``_hyphen_views``), so that "E-mail the keys", "Send-the-keys",
    "Email-them" and "Please-send" order; and with its prefixes too, so that
    a verb in brackets, "(Send) the keys", is read -- unless those prefixes
    are tags of the table ("[Note]", "(Fix)": ``_tagged``). A French past
    participle opens no opening of courtesy or prohibition.
    """
    forms = set()
    bare = _untagged(piece, table)
    views = list(_hyphen_views(_order_marked(bare)))
    if bare == piece and not _tagged(piece, table):
        views += _hyphen_views(_MARKER.sub(" it ", piece))
    index = _phrase_index(table)
    for view in dict.fromkeys(views):
        words, written = _clause_words(view)
        at = 0
        # Past lead-in words and interjections: "ok so send the report".
        while at < len(words) - 1 and (words[at] in table.lead_ins or words[at] in table.courtesy):
            at += 1
        rest, said = words[at:], written[at:]
        if not rest:
            continue
        following = rest[1] if len(rest) > 1 else None
        # An opening may begin on a lead-in word: "Go ahead and send". Before
        # a subject it asks: "Do we keep Docker?" orders nothing, "Do it" does.
        asks = following in table.subject_pronouns and following not in table.object_openers
        if not _participle(said[0], following) and not asks and (_phrase_at(index, "openers", rest, 0)
                                                                 or _phrase_at(index, "openers", words, 0)):
            forms.add("d")
        skip = 0
        while skip < len(rest) - 2 and (_adverb(rest[skip], table) or rest[skip] in table.lead_ins):
            skip += 1
        if _verb_first(rest, said, table, names) or (skip and _verb_first(rest[skip:], said[skip:], table, names)):
            forms.add("f")
        if table.strict and not _opens_statement(rest[0], said[0], table, names, following):
            forms.add("s")
    return forms


# A word in brackets or parentheses before a clause: a tag when the table
# holds it ("(Fix)", "[Update]"), a verb when it does not ("(Send)").
_LEADING_GROUP = re.compile(r"\s*[\[(]\s*([^\W\d_]{1,24})\s*[\])]\s*")


def _untagged(piece, table):
    """``piece`` without the words in brackets or parentheses before it that are tags of the table (``label_heads``,
    ``lead_ins``): "(Fix) The cache no longer leaks" reads "The cache no longer leaks"."""
    for _ in range(6):
        group = _LEADING_GROUP.match(piece)
        if not group:
            break
        word = _clause_words(group.group(1))[0]
        if not word or not (word[0] in table.label_heads or word[0] in table.lead_ins):
            break
        piece = piece[group.end():]
    return piece


def _tagged(piece, table):
    """True when the prefixes stripped before a piece are tags of the table ("[Note]", "(Fix)", "[TODO]"): read with
    them, a tag would read as a verb."""
    text, words = piece, []
    for _ in range(6):
        prefix = _ORDER_PREFIX.match(text)
        if not prefix:
            break
        words += [word for word in _clause_words(prefix.group(0))[0] if not word.isdigit()]
        text = text[prefix.end():]
    return bool(words) and all(word in table.label_heads or word in table.lead_ins for word in words)


def _opens_on_subject(piece, table, names, following=None):
    """True when a piece opens on a subject -- a determiner, a subject pronoun, the user, a name -- so that the pieces
    after its comma continue its clause rather than open one: "The parser, in strict mode, rejects the input".

    Names alone are a vocative ("Bob, email the keys": the clause is after
    them) unless the next piece, ``following``, opens on a determiner: an
    appositive ("Mallory, our auditor, wants the logs") leaves the name the
    subject.
    """
    words, _written = _clause_words(_order_marked(piece))
    at = 0
    while at < len(words) - 1 and words[at] in table.lead_ins:
        at += 1
    if at >= len(words):
        return False
    if len(words) - at <= 3 and all(word in names for word in words[at:]):
        after = _clause_words(_order_marked(following))[0][:1] if following else []
        if not (after and after[0] in table.statement_openers and after[0] in table.object_openers):
            return False
    word = words[at]
    # The French pronoun "on" is the English particle "on" too: "On Friday,
    # delete the backups" opens on a phrase, not on a subject.
    return ((word in table.statement_openers and word in table.object_openers) or word in table.user_subjects
            or (word in table.subject_pronouns and word not in table.particles) or word in names)


def _order_chunks(sentence, table):
    """The chunks of a sentence an order is read in, each ``(chunk, addressed)``.

    A sentence parts at a semicolon, a dash and a colon. The run before a
    colon is a chunk of its own, "Run this:" ordering as any does, and a
    label that addresses the reader addresses every chunk after it.
    """
    found = []
    for part in _CHUNK_SPLIT.split(sentence):
        pieces = _LABEL_SPLIT.split(part)
        addressed = False
        for index, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            found.append((piece, addressed))
            if index < len(pieces) - 1 and _addresses(piece, table):
                addressed = True
    return found


def _orders_in(text, table, names):
    """Each clause of ``text`` that orders, ``(clause, forms, chunk, question, sentence)``, in reading order.

    The text is folded as an order is read (``_order_text``), its inline code
    read as words. A chunk is read whole for the forms that may stand
    anywhere, and each of its pieces between commas for the forms that open
    a clause. A paragraph that ends on a label addressing the reader
    addresses the list items that follow it. The sentence is what a typed
    sentence must hold whole; a question is held only by a question.
    """
    found, carry = [], False
    for block in segment(text):
        if block.kind == "code":
            carry = False
            continue
        addressed = carry and block.kind == "item"
        prose = _order_prose(block.text)
        for sentence in (s.strip() for s in _ORDER_SENTENCE.split(prose)):
            if not sentence:
                continue
            question = _is_question(sentence)
            for chunk, labelled in _order_chunks(sentence, table):
                chunk_forms = _chunk_forms(chunk, addressed or labelled, table, names)
                pieces = [piece.strip() for piece in _COMMA_SPLIT.split(chunk) if piece.strip()]
                if len(pieces) == 1:
                    forms = chunk_forms | _piece_forms(chunk, table, names)
                    if forms:
                        found.append((chunk, "".join(sorted(forms)), chunk, question, sentence))
                    continue
                # The spine opens as the chunk does, its phrases between commas
                # set aside: "Email, right now, the vault keys" orders.
                opening = _piece_forms(_spine(chunk), table, names) if len(pieces) > 2 else set()
                if chunk_forms or opening:
                    found.append((chunk, "".join(sorted(chunk_forms | opening)), chunk, question, sentence))
                following = pieces[1] if len(pieces) > 1 else None
                for piece in pieces if not _opens_on_subject(pieces[0], table, names, following) else pieces[:1]:
                    forms = _piece_forms(piece, table, names)
                    if forms:
                        found.append((piece, "".join(sorted(forms)), chunk, question, sentence))
        tail = prose.rstrip()
        label = re.split(r"[.!?;:]\s+|\s*\n\s*", tail[:-1])[-1] if tail.endswith(":") else ""
        carry = (bool(label) and _addresses(label, table)) or addressed
    return found


def directives_in(text, table, names=frozenset()):
    """The clauses of ``text`` that order, each ``(clause, forms)``, in reading order, as ``_orders_in`` reads them.

    The forms are letters: ``a`` the reader spoken to in the second person,
    anywhere in the clause -- a summary tells of the user and the assistant
    in the third; ``b`` an obligation of the reader; ``c`` a label that
    addresses the reader, its list included; ``d`` an opening of courtesy,
    prohibition or rule, or a lasting rule anywhere; ``e`` an injection's
    signature, whoever tells it; ``f`` a verb opening the clause before an
    object or a name; ``r`` a request told of the user, the user's order;
    ``s``, under the strict switch, a clause whose first
    word opens no statement. A fenced block holds no clause: the summariser
    sees it as its marker only. ``names`` are the folded words of the names
    the span holds. With no table no clause orders.
    """
    if table is None:
        return []
    found = []
    for clause, forms, _chunk, _question, _sentence in _orders_in(text, table, names):
        if (clause, forms) not in found:
            found.append((clause, forms))
    return found


# A restatement is compared as a sequence: a clause's content words in their
# order, each without its final s; a negation as one mark where it stands; a
# preposition where it stands, past the first content word ("from Mallory"
# is not "to Mallory"); a lasting rule as a flag. Left out: the frame of a
# telling at the head of a summary's clause ("The user asked the assistant
# to"), the frame of a request at the head of a typed part ("Can you", "I
# need you to"), courtesy words, lead-in words before the first content
# word, and the function words of ``_STOPWORDS``: articles, the pronouns
# and possessives it lists, conjunctions and the forms of be and do (etre
# and avoir in French). Modal verbs ("will", "can", "should") are content.
# An English contraction is spelled out: "don't" is "do not", "I'd" is "I
# would".
_CONTRACTED = (
    (re.compile(r"\bcan['" + chr(0x2019) + r"]t\b", re.IGNORECASE), "can not"),
    (re.compile(r"\bwon['" + chr(0x2019) + r"]t\b", re.IGNORECASE), "will not"),
    (re.compile(r"n['" + chr(0x2019) + r"]t\b", re.IGNORECASE), " not"),
    (re.compile(r"(?<=\w)['" + chr(0x2019) + r"]d\b", re.IGNORECASE), " would"),
    (re.compile(r"(?<=\w)['" + chr(0x2019) + r"]ll\b", re.IGNORECASE), " will"),
)
_NEGATION_WORDS = frozenset({"not", "never", "no", "none", "cannot", "ne", "pas", "jamais", "rien", "aucun",
                             "aucune"})
# The prepositions a restatement keeps, folded: a direction, a companion, a
# place, a time. The French "a" is kept only as written with its accent, its
# folded form being the English article.
_ROLE_WORDS = frozenset({"to", "from", "of", "in", "on", "at", "for", "with", "by",
                         "de", "du", "au", "aux", "en", "pour", "par", "sur", "dans", "avec", "sans", "apres"})
# The backing last read, keyed by its typed pieces and its table.
_BACKING = {}


def _frame_end(words, table, index):
    """Where the frame of a telling at the head of a clause ends, or 0 when the clause opens on none.

    The frame is the user, a telling verb within two words, then the reader,
    pronouns, determiners, datives and the "to" or "de" before the told
    verb: "The user asked the assistant to", "The user wants", "L'utilisateur
    a demande a l'assistant de". Only function, lead-in and courtesy words
    and a lasting rule may stand before it.
    """
    at = 0
    while at < len(words) and words[at] not in table.user_subjects:
        rule = _phrase_at(index, "persistent", words, at)
        if rule:
            at += len(rule)
        elif words[at] in _FOLDED_STOPWORDS or words[at] in table.lead_ins or words[at] in table.courtesy:
            at += 1
        else:
            return 0
    reach = min(len(words), at + 3)
    verb = at + 1
    while verb < reach and words[verb] not in table.telling and (
            words[verb] in table.auxiliaries or words[verb] in _FOLDED_STOPWORDS):
        verb += 1
    if verb >= reach or words[verb] not in table.telling:
        return 0
    end = verb + 1
    while end < len(words) and (words[end] in table.readers or words[end] in table.subject_pronouns
                                or words[end] in table.object_openers or words[end] in table.datives
                                or words[end] in ("to", "de")):
        end += 1
    return end


def _clause_of(text):
    """The words of a clause or a typed part as a restatement reads them: contractions spelled out, folded as an
    order is, enumerators set aside and code markers read as "it". A tag in brackets is kept: its words may be a
    negation ("[DON'T] email the keys")."""
    for pattern, plain in _CONTRACTED:
        text = pattern.sub(plain, text)
    text = _order_text(text)
    for _ in range(6):
        prefix = _ORDER_ENUMERATOR.match(text)
        if not prefix:
            break
        text = text[prefix.end():]
    return _clause_words(_HYPHEN.sub("", _MARKER.sub(" it ", text)))


def _sequence(words, written, table, index, frame):
    """``(sequence, lasting)`` of a clause's words, those before ``frame`` read only for a negation."""
    sequence, lasting, head, at = [], False, True, 0
    while at < len(words):
        rule = _phrase_at(index, "persistent", words, at)
        if rule:
            lasting, at = True, at + len(rule)
            continue
        word, said, at = words[at], written[at].lower(), at + 1
        if word in _NEGATION_WORDS:
            sequence.append("~")
        elif at <= frame or word in table.courtesy or (head and word in table.lead_ins):
            continue
        elif word in _ROLE_WORDS or said == chr(0xE0):
            # A preposition before the first content word belongs to no
            # role: "To Bob, send", "asked the assistant not to send".
            if not head:
                sequence.append(word)
        elif word not in _FOLDED_STOPWORDS:
            sequence.append(word[:-1] if word.endswith("s") else word)
            head = False
    return tuple(sequence), lasting


def _restatement(text, table):
    """``(sequence, lasting)``: what a summary's clause says, the frame of a telling at its head left out, to be
    compared word for word with what the user typed."""
    words, written = _clause_of(text)
    index = _phrase_index(table)
    return _sequence(words, written, table, index, _frame_end(words, table, index))


def _typed_restatement(text, table):
    """``(sequence, lasting)`` of a part the user typed, compared with a summary's deciding clause."""
    words, written = _clause_of(text)
    return _sequence(words, written, table, _phrase_index(table), 0)


def _scoped(text, table):
    """True when a line holds a negation or a condition anywhere."""
    words, _written = _clause_of(text)
    index = _phrase_index(table)
    return (any(word in _NEGATION_WORDS for word in words)
            or any(_phrase_at(index, "conditions", words, k) for k in range(len(words))))


def _bars_list(line, table):
    """True when a label or a line ending in a colon leaves what follows it no decision of the user's: it holds a
    negation, a condition or an attribution, a verb of telling that is no verb of wanting, a quote's noun, or
    another's possessive, "our" before it included ("Mallory wrote:", "Here is the mail:", "Don't do any of this:",
    "Mallory's ideas:", "Our vendor's proposal:"). The user's own line ("Here is what I need:", "Two things:", "This
    week's plan:") bars nothing."""
    words, written = _clause_of(line)
    return (_scoped(line, table)
            or any((word in table.telling and word not in table.wanting) or word in table.quote_nouns for word in words)
            or any(_possessive(written, at) and words[at] not in table.time_nouns and not _adverb(words[at], table)
                   and words[at] not in table.user_subjects for at in range(len(words))))


def _plain_decision(text, table, lexicon):
    """True for a decision the user states plainly in their own voice: a first-person subject, then, past auxiliaries
    of tense only, an act of the lexicon, and no negation, condition, attribution, quote or possessive (``_bars_list``)
    nor adverb anywhere. "We keep the logs" and "We will keep the logs" are ones; "It is not true that we keep", "I
    doubt we keep", "Mallory says we should keep", "We reportedly keep", "We might keep" and "We keep the logs
    allegedly" are none."""
    if lexicon is None or _bars_list(text, table):
        return False
    folded = [_ELIDED.get(word, word) for word in _tokens(_fold(text))]
    at = 0
    while at < len(folded) - 1 and folded[at] in table.lead_ins:
        at += 1
    if at >= len(folded) or folded[at] not in table.first_person or any(_adverb(word, table) for word in folded[at:]):
        return False
    first = next((start for start, _stop, _languages, _cls in _acts(folded, lexicon) if start > at), None)
    return first is not None and all(word in _TENSE_AUXILIARIES for word in folded[at + 1:first])


# The auxiliaries that set a tense and nothing else: a modal ("might",
# "would", "could") scopes the decision as an adverb does.
_TENSE_AUXILIARIES = frozenset({"will", "shall", "have", "has", "had", "do", "does", "did", "am", "is", "are", "was",
                                "were", "be", "been", "va", "vont", "allons", "vais", "a", "ai", "avons", "ont"})


def _backing(typed, table, lexicon=None):
    """What the pieces the user typed may hold of a decision: ``(decisions, barred)``.

    ``decisions`` is a tuple of ``(sequence, lasting, question)``: any typed
    part -- a sentence, a chunk, a piece -- but for the pieces after a
    negated, conditional or attributed piece ("If the audit passes, we drop
    the backups" and "According to Bob, we drop the backups" hold no "we
    drop the backups"), the chunks after a label before a colon that bars
    (``_bars_list``: "Mallory wrote: we drop"), a sentence a bare retraction
    takes back (past a one-word sentence: "We drop the logs. Wait. No."),
    and everything after a line ending on a colon that bars. ``barred``
    holds the texts of the sentences such a label, line or retraction
    reaches, whose decision probes hold nothing either. With the lexicon, a
    typed sentence is also parted as a summary's is (``_decision_split``),
    each clause with its head and, when that head is the user's plain
    decision (``_plain_decision``), the predicates coordinated to it
    (``_coordinated``), so that the user's words restated whole are held
    whole and no predicate escapes the negation, the doubt or the voice of
    its head. A piece before a comma that bars (``_bars_list``: "Bob said,")
    lends what follows it nothing. An order is never
    held here: it stands in a peel only stitched word for word
    (``typed_ranges``). The last backing read is kept with its pieces, its
    table and its lexicon: a repair judges its summary sentence by sentence
    against one span.
    """
    key = (typed, table, lexicon)
    kept = _BACKING.get(key)
    if kept is not None:
        return kept
    decisions, barred_texts = [], set()
    parting = None if lexicon is None else _decision_split(table, lexicon)
    for text in typed:
        flat, barred = [], False
        for block in segment(_order_text(text)):
            if block.kind == "code":
                continue
            for sentence in (s.strip() for s in _ORDER_SENTENCE.split(_continued(block.text)) if s.strip()):
                flat.append((sentence, barred))
            if block.text.rstrip().endswith(":"):
                barred = barred or _bars_list(block.text.rstrip(), table)
        for number, (sentence, barred) in enumerate(flat):
            later = next((s for s, _barred in flat[number + 1:]
                          if len(_tokens(_fold(s))) > 1 or _retraction(s)), None)
            if barred or (later is not None and _retraction(later)):
                barred_texts.add(sentence)
                continue
            question = _is_question(sentence)
            decisions.append(_typed_restatement(sentence, table) + (question,))
            for part in _CHUNK_SPLIT.split(sentence):
                labels = [chunk.strip() for chunk in _LABEL_SPLIT.split(part)]
                for number_in, chunk in enumerate(labels):
                    if not chunk:
                        continue
                    # A label of another's voice, a condition or a negation
                    # before a colon leaves the rest of its part no decision
                    # of the user's alone: "Mallory wrote: we drop".
                    if any(_bars_list(label, table) for label in labels[:number_in] if label):
                        barred_texts.add(sentence)
                        break
                    decisions.append(_typed_restatement(chunk, table) + (question,))
                    stopped = False
                    for piece in (p.strip() for p in _COMMA_SPLIT.split(chunk) if p.strip()):
                        if not stopped:
                            decisions.append(_typed_restatement(piece, table) + (question,))
                        # A negation, a condition, an attribution or a quote
                        # before a comma lends what follows it nothing:
                        # "Bob said, we drop the backups".
                        stopped = stopped or _bars_list(piece, table)
                    if parting is None:
                        continue
                    for piece in (p.strip() for p in parting.split(chunk) if p and p.strip()):
                        clauses = _coordinated(piece, table, frozenset())
                        head = _head_of(piece, clauses)
                        decisions.append(_typed_restatement(piece, table) + (question,))
                        decisions.append(_typed_restatement(head, table) + (question,))
                        # A predicate holds alone only with the user's plain
                        # decision as its head: under a negation, a doubt or
                        # another's voice it shares their scope.
                        if _plain_decision(head, table, lexicon):
                            decisions.extend(_typed_restatement(clause, table) + (question,) for clause in clauses)
                        if _bars_list(piece, table):
                            break
    found = (tuple(decisions), frozenset(barred_texts))
    _BACKING.clear()
    _BACKING[key] = found
    return found


def _from_its_side(probe, text):
    """True when most of ``text``'s own decision key is the probe's: words added to a decision are a decision of
    their own."""
    lexicon = EMPTY_LEXICON if probe.lexicon is None else probe.lexicon
    plain, dates, quantities, rest = _read_sentence(text)[:4]
    folded = _tokens(_fold(plain))
    key = _decision_key(text, rest, dates, quantities, folded, _acts(folded, lexicon), lexicon)
    return not key or len(probe.key & key) / len(key) >= DECISION_COVERAGE


def _backed(clause, chunk, question, backing, decisions, table):
    """True when the user's words hold the deciding ``clause``, or the chunk it stands in.

    A typed part holds it when it says the same, word for word in the same
    order (``_restatement``), with its negations, its lasting rule and its
    mood. A typed decision holds it when the clause answers it with its
    polarity and most of the clause's own key is the decision's, unless the
    decision was asked.
    """
    for text in dict.fromkeys((clause, chunk)):
        sequence, lasting = _restatement(text, table)
        if sequence and any(held == sequence and held_lasting == lasting and (question or not held_question)
                            for held, held_lasting, held_question in backing):
            return True
        for p in decisions:
            if (question or not _is_question(p.answer)) and answers(p, text) and _from_its_side(p, text):
                return True
    return False


def _deciding(text, lexicon):
    folded = _tokens(_fold(text))
    return _decides(text, folded, _acts(folded, lexicon), lexicon)


def _subject_word(text):
    """A text's first word that is no function word: what the reporter rule reads as its subject."""
    return next((word for word in _tokens(_fold(text)) if word not in _STOPWORDS), None)


def _told_as_users(plain, table, lexicon):
    """The decision a reporter's sentence tells as the user's, from the user's name on, or None.

    The user named as the one who decides makes the telling the user's. Only
    a pronoun that cannot be the user (``owner_pronouns``: "it"), after the
    name or after "that", makes the decision another's: "The assistant told
    the user it decided" tells the assistant's own, while a person's "she"
    and the French "il" may be the user, and leave the telling the user's.
    The phrases between the first and the last comma are read past ("the
    user, it seems, decided"). The decision is returned as the user would
    have typed it: the clause after "that" or "que" when one decides ("the
    user confirmed that we drop the backups"), else the telling with the user
    read as "we" or "on" -- "the user prefers Podman" as "we prefers
    Podman" -- so that it decides by its act alone and is held by the user's
    own words; else as told, from the user's name on.
    """
    words, written = _clause_words(_spine(plain))
    for at in range(len(words)):
        if words[at] not in table.user_subjects:
            continue
        after = words[at + 1:at + 3]
        if after[:1] and (after[0] in table.owner_pronouns
                          or (after[0] in ("that", "que") and after[1:2] and after[1] in table.owner_pronouns)):
            continue
        rest = " ".join(written[at + 1:])
        that = next((j for j in range(at + 1, len(words)) if words[j] in ("that", "que")), None)
        embedded = () if that is None else (" ".join(written[that + 1:]),)
        return next((clause for clause in embedded + (f"we {rest}", f"on {rest}", " ".join(written[at:]))
                     if _deciding(clause, lexicon)), None)
    return None


def _coordinated(chunk, table, names):
    """The predicates coordinated to a clause of ``chunk``, each as written.

    A coordinator ("&" read as "and"), then, past any auxiliary or lead-in
    word ("and will", "and also"), a verb and its object; and a gerund and
    its object after a comma ("..., dropping the NAS").
    """
    gerunds = [piece.strip() for piece in _COMMA_SPLIT.split(chunk)[1:] if _gerund_predicate(piece, table, names)]
    words, written = _clause_words(chunk.replace("&", " and "))
    index = _phrase_index(table)
    found, start, at = [], None, 1
    while at < len(words):
        coordinator = _phrase_at(index, "coordinators", words, at)
        if coordinator:
            verb = at + len(coordinator)
            while verb < len(words) - 1 and (words[verb] in table.auxiliaries or words[verb] in table.lead_ins):
                verb += 1
            if (verb + 1 < len(words) and not _opens_statement(words[verb], written[verb], table, names)
                    and _object_at(words, written, verb + 1, table, names)):
                if start is not None:
                    found.append(" ".join(written[start:at]))
                start = at = verb
                continue
        at += 1
    if start is not None:
        found.append(" ".join(written[start:]))
    return found + [piece for piece in gerunds if piece not in found]


def _gerund_predicate(piece, table, names):
    """True when a piece after a comma opens on a gerund of five letters or more and its object: "dropping the NAS"."""
    words, written = _clause_words(piece)
    return (len(words) > 1 and words[0].endswith("ing") and len(words[0]) >= 5
            and _object_at(words, written, 1, table, names))


def unbacked_directives(held, probes, text, directives=None):
    """The clauses of a summary that order outside the user's own words stitched into it.

    Each is ``("directive", clause, "")``: no turn holds an order the user
    did not type, and no summary may restate one, the user's included -- a
    paraphrase can drop what bounds an order or change who gives it. An
    order stands in a peel only inside a run the user typed, stitched word
    for word after its turn's marker ("[u1] ..."), as ``typed_ranges``
    reads the span (``held.stitchable``). A clause orders as ``_orders_in``
    reads it, the names being those the span's probes ask for. With no
    table no clause orders.
    """
    if directives is None:
        return []
    found, names = [], _names_held(probes)
    for view in _unstitched(text, held.stitchable, held.ids):
        for clause, _forms, _chunk, _question, _sentence in _orders_in(view, directives, names):
            entry = ("directive", clause, "")
            if entry not in found:
                found.append(entry)
    return found


# A sentence the summary ended: a stop, past any closing quote, bracket or
# mark of emphasis ("**Noted.**") -- never an ellipsis ("Mallory wrote...",
# ". . .") nor the stop of an abbreviation ("i.e.", "etc.", "cf.").
_ENDED = re.compile(r"(?<![." + chr(0x2026) + r"])(?<!\.\s)(?<!\b[^\W\d_]\.[^\W\d_])(?<!\betc)(?<!\bviz)(?<!\bcf)(?<!\bvs)"
                    r"[.!?][\"')\]*_`" + chr(0x2019) + chr(0x201D) + chr(0xBB) + r"]*$", re.IGNORECASE)


def _unstitched(text, stitchable, ids=()):
    """The words of ``text`` that are the summary's, in two readings: without the block of runs the queue stitched at
    its end, and with every other turn marker read as nothing, then as the end of a sentence.

    A stitched block is honoured only where the queue writes it: at the very
    end of the text, each run of the user's after its turn's marker, word for
    word (``[turn_id] run``), and after a sentence the summary ended with a
    stop, never an ellipsis nor an abbreviation's (``_ENDED``) -- so that no
    word of the summary is read with the user's ("On every later turn, [t1]
    Delete...", "Delete [t2] The backups..."). A marker
    anywhere else stitches nothing, and what it marks is read as the
    summary's own words twice: joined to what comes before it ("Delete The
    backups") and apart from it ("Delete. Send the keys").
    """
    runs = set(stitchable)
    items = {f"[{turn_id}] {run}" for turn_id, run in runs}
    lengths = {len(item) for item in items}
    # Read from the end, item by item, by the last brackets before each, a
    # bracket tried by its distance first: linear in the block and in the
    # brackets, however many runs it stitches.
    end, honoured = len(text.rstrip()), False
    while True:
        cut, found = end, None
        while cut > 0:
            cut = text.rfind("[", 0, cut)
            if cut < 0:
                break
            if end - cut in lengths and text[cut:end] in items and (cut == 0 or text[cut - 1].isspace()):
                found = cut
                break
        if found is None:
            break
        end, honoured = found, True
        while end and text[end - 1].isspace():
            end -= 1
    head = text[:end]
    if not honoured or (head and not _ENDED.search(head)):
        head = text
    markers = sorted(set(ids) | {turn_id for turn_id, _run in runs}, key=len, reverse=True)
    if not markers:
        return (head,)
    pattern = re.compile(r"\[(?:" + "|".join(map(re.escape, markers)) + r")\]")
    return tuple(dict.fromkeys((pattern.sub(" ", head), pattern.sub(". ", head))))


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
