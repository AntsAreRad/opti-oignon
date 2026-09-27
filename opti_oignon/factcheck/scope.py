"""What is checked: a claim, its scope, its wrapper, whose it is, and the lexicons that decide it.

A claim is one sentence. It is checked only if it is a standalone declarative
proposition once its wrapper is stripped; otherwise it is out of scope with a
closed reason, decided on the claim alone and in this order:

1. what the markdown pass marked: code, a heading, a table row, an image, markup
   the subset does not read;
2. no word ("empty"), longer than the limit ("too_long"), a question;
3. a wrapper is stripped, never used to hide a claim: an opinion opener (and
   the "that" after an English one) is removed and the claim is "hedged"
   (evidence for its content supports it); an advice opener is removed, and an
   imperative opening kept, and the claim is "advice", checked only if what
   remains carries a digit or a capitalised word past its first word, else it
   is an "instruction"; a wrapper with no word after it is "empty";
4. a subject that refers to something named elsewhere: a pronoun, a
   demonstrative standing alone, a possessive ("its", "their") or an
   anaphoric opener ("another", "both", "one of", "such"), and, in a claim
   about the world, a first- or second-person subject or possessive (it names
   whoever wrote it): "not_standalone";
5. a claim about the world opening on a definite or demonstrative determiner,
   with no capitalised word past the first (a month or a weekday is not a
   name) and no word mixing letters and digits: "subject_unresolved" (its
   referent is somewhere else).

A claim is the owner's own ("own") when its subject is the first person with a
decision marker, or, in assistant text, the second person with one; anything
else is about the world. An own claim is compared with the owner's own sources
after a closed rewrite that knows who wrote each side: in the owner's text
the first person is the owner, in the assistant's text the second person is.
The subject at the opening of the sentence, the auxiliary right after it and
the possessives of that person become one owner token; nothing else moves, so
"You decided ..." (the assistant) and "We decided ..." (the owner) are one
claim, while "we pay you" and "you pay us" stay two, and the assistant's own
"I" is never the owner.

Every lexicon is closed, versioned, stored in ASCII without accents, and matched
on accent-folded lowercase text; accent folding serves these lexicons only,
never the passage check.
"""

import functools
import re
import unicodedata
from dataclasses import dataclass, field

from . import markup, passage

checkpoint_before_apply = True

SCOPE_VERSION = 1

OPINION_OPENERS = ("i think", "i believe", "i feel that", "in my opinion", "je pense que",
                   "je crois que", "a mon avis", "selon moi", "il me semble que")
ADVICE_OPENERS = ("you should", "you must", "you need to", "make sure", "be sure to", "vous devriez",
                  "tu devrais", "il vaut mieux", "pensez a", "pense a")
IMPERATIVE_VERBS = ("take", "use", "avoid", "add", "run", "install", "set", "keep", "do not", "don't",
                    "prenez", "utilisez", "evitez", "ajoutez", "lancez", "installez", "n'oubliez pas")
DISCOURSE_MARKERS = ("however", "but", "and", "also", "so", "then", "thus", "moreover", "furthermore",
                     "in addition", "mais", "et", "donc", "aussi", "ensuite", "puis", "cependant",
                     "pourtant", "de plus", "alors")
PRONOUN_SUBJECTS = ("it", "he", "she", "they", "this", "that", "these", "those", "il", "elle", "ils",
                    "elles", "ce", "cela", "ca", "celui-ci", "celle-ci", "c'")
# A demonstrative is a pronoun subject only when a verb of this list, or nothing,
# follows it; followed by anything else it is a determiner (step 5).
DEMONSTRATIVES = ("this", "that", "these", "those", "ce")
AFTER_PRONOUN = ("is", "was", "are", "were", "has", "had", "have", "will", "would", "can", "could", "may",
                 "might", "must", "shall", "should", "does", "did", "do", "means", "meant", "shows",
                 "showed", "seems", "seemed", "makes", "made", "remains", "becomes", "became", "est",
                 "etait", "sera", "serait", "sont", "etaient", "seront", "fut", "a", "avait", "aura", "ne",
                 "n'", "semble", "montre", "signifie", "reste", "devient")
ANAPHORIC_OPENERS = ("its", "his", "her", "their", "another", "other", "others", "both", "such", "one of",
                     "each of", "either of", "neither of", "the former", "the latter", "son", "sa", "ses",
                     "leur", "leurs", "un autre", "une autre", "d'autres", "l'autre", "les autres",
                     "les deux", "un tel", "une telle", "de tels", "de telles", "ce dernier",
                     "cette derniere")
# In a claim about the world, a subject or possessive of the first or second
# person names whoever wrote the sentence.
PERSONAL_OPENERS = ("i", "we", "you", "je", "j'", "nous", "tu", "vous", "my", "our", "your", "mon", "ma",
                    "mes", "ton", "ta", "tes", "notre", "nos", "votre", "vos")
MONTHS_AND_DAYS = ("january", "february", "march", "april", "may", "june", "july", "august", "september",
                   "october", "november", "december", "monday", "tuesday", "wednesday", "thursday",
                   "friday", "saturday", "sunday", "janvier", "fevrier", "mars", "avril", "mai", "juin",
                   "juillet", "aout", "septembre", "octobre", "novembre", "decembre", "lundi", "mardi",
                   "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
DETERMINERS = ("the", "this", "that", "these", "those", "le", "la", "les", "l'", "ce", "cet", "cette",
               "ces")
FIRST_PERSON = ("i", "we", "je", "j'", "nous", "on")
SECOND_PERSON = ("you", "tu", "vous")
# French "on" is a subject only in a claim not marked English, and only before
# one of these; the English preposition ("On 12 May ...") never is.
ON_VERBS = ("a", "avait", "aura", "est", "etait", "sera", "va", "ne", "n'", "y", "en", "decide", "choisit",
            "opte", "retient", "convient", "tranche", "peut", "doit", "fait", "garde")
PAST_DECISION_VERBS = ("decided", "chose", "chosen", "agreed", "settled on", "opted for", "went with",
                       "picked", "resolved to", "decide", "choisi", "convenu", "opte", "retenu",
                       "tranche")
# Folded to ASCII, the French past forms equal English words ("decide"): they
# count as past only when written with their accent, or in a French claim.
FRENCH_ONLY_PAST = ("decide", "opte", "tranche")
DECISION_MARKERS = PAST_DECISION_VERBS + ("decides", "choose", "chooses", "agree", "agrees", "pick",
                                          "picks", "decidons", "decidez", "choisissons", "choisissez",
                                          "optons", "retenons", "convenons")
TIME_SENSITIVE = ("current", "currently", "now", "today", "latest", "newest", "most recent", "as of",
                  "at present", "incumbent", "still", "actuel", "actuelle", "actuellement",
                  "maintenant", "aujourd'hui", "dernier", "derniere", "en ce moment", "a ce jour")
QUALIFYING_MARKERS = (
    "false", "untrue", "incorrect", "wrong", "myth", "myths", "misconception", "misconceptions",
    "debunked", "rumour", "rumours", "rumor", "rumors", "hoax", "hypothesis", "hypotheses",
    "hypothetical", "unverified", "todo", "not true", "to verify", "to check", "open question",
    "faux", "fausse", "fausses", "errone", "erronee", "errones", "erronees", "inexact", "inexacte",
    "mythe", "mythes", "rumeur", "rumeurs", "canular", "hypothese", "hypotheses", "hypothetique",
    "brouillon", "idee recue", "idees recues", "non verifie", "non verifiee", "a verifier",
    "a confirmer", "question ouverte",
    "wrongly", "falsely", "falsehood", "falsehoods", "fake", "debunk", "legend", "legends", "refuted",
    "disproven", "disputed", "contested", "not sure", "unsure", "doubtful", "allegedly", "supposedly",
    "reportedly", "pas vrai", "plus vrai", "pas du tout", "n'est plus", "douteux", "douteuse",
    "pretendument", "conteste", "contestee", "dementi",
)
# A neighbouring sentence that denies what stands beside it.
DENIALS = ("no", "non", "not so", "not at all", "certainly not", "pas du tout", "absolument pas")
# Read on the headings and lead-ins that introduce a sentence: a frame the
# sentence itself does not carry. Each class gives its closed reason; the
# first three block only when the claim lacks a marker of the same class.
STRENGTH_MARKERS = (
    ("conditional", ("if", "unless", "provided that", "assuming", "suppose", "supposing", "in case", "si",
                     "sauf si", "a condition que", "a moins que", "en cas de")),
    ("evidence_hedged", ("may", "might", "could", "would", "will", "possibly", "probably", "likely",
                         "perhaps", "possible", "potential", "preliminary", "expected", "estimated",
                         "projected", "forecast", "forecasts", "predicted", "scenario", "scenarios",
                         "peut-etre", "pourrait", "pourraient", "devrait", "devraient", "probablement",
                         "possiblement", "potentiellement", "prevu", "prevue", "prevus", "prevision",
                         "previsions")),
    ("population_narrower", ("in mice", "in mouse", "in rats", "in animals", "animal model", "in vitro",
                             "in cell culture", "preclinical", "chez la souris", "chez les souris",
                             "chez le rat", "chez les rats", "chez l'animal", "modele animal",
                             "preclinique")),
    ("context_qualified", ("not", "never", "no longer", "n'est pas", "n'est plus", "ne sont pas",
                           "ne sont plus", "jamais")),
)
ATTRIBUTION_MARKERS = ("according to", "said", "says", "claims", "claimed", "selon", "d'apres",
                       "a declare", "affirme")
OWNER_CONTRACTIONS = (("i've", "i have"), ("we've", "we have"), ("you've", "you have"), ("i'd", "i had"),
                      ("we'd", "we had"))
OWNER_SUBJECTS = ("i", "we", "you", "je", "nous", "on", "tu", "vous")
OWNER_AUXILIARIES = ("ai", "as", "a", "avons", "avez", "suis", "es", "est", "sommes", "etes")
OWNER_POSSESSIVES = (("<owner-poss>", ("my", "our", "your")),
                     ("<owner-poss-sg>", ("mon", "ma", "ton", "ta", "notre", "votre")),
                     ("<owner-poss-pl>", ("mes", "tes", "nos", "vos")))
# Who the owner is, by who wrote the text: the first person in his own text,
# the second person in the assistant's. Subjects at the opening only.
OWNER_PERSONS = {
    "first": {"subjects": ("i", "we", "je", "j'", "nous", "on"),
              "possessives": ("my", "our", "mon", "ma", "notre", "mes", "nos")},
    "second": {"subjects": ("you", "tu", "vous"),
               "possessives": ("your", "ton", "ta", "votre", "tes", "vos")},
}
# A year or a month: a claim that carries its own date (a later reader reads it).
DATE_WORDS = MONTHS_AND_DAYS[:12] + MONTHS_AND_DAYS[19:31]

# The reasons the markdown pass can give, in the order they are read.
MARKUP_REASONS = ("code", "heading", "table_row", "image", "markup_unparsed")
_CLOSERS = "\"'" + chr(0xBB) + chr(0x201D) + chr(0x2019) + ")]}"


def rules():
    """Every lexicon of this module, for the rules digest."""
    return {
        "version": SCOPE_VERSION,
        "opinion_openers": list(OPINION_OPENERS), "advice_openers": list(ADVICE_OPENERS),
        "imperative_verbs": list(IMPERATIVE_VERBS), "discourse_markers": list(DISCOURSE_MARKERS),
        "pronoun_subjects": list(PRONOUN_SUBJECTS), "determiners": list(DETERMINERS),
        "first_person": list(FIRST_PERSON), "second_person": list(SECOND_PERSON),
        "past_decision_verbs": list(PAST_DECISION_VERBS), "decision_markers": list(DECISION_MARKERS),
        "time_sensitive": list(TIME_SENSITIVE), "qualifying_markers": list(QUALIFYING_MARKERS),
        "attribution_markers": list(ATTRIBUTION_MARKERS), "denials": list(DENIALS),
        "strength_markers": [[code, list(words)] for code, words in STRENGTH_MARKERS],
        "may_lowercase": "the modal may counts only written in lowercase",
        "opinion_that": "an English opinion opener takes the that after it",
        "demonstratives": list(DEMONSTRATIVES), "after_pronoun": list(AFTER_PRONOUN),
        "anaphoric_openers": list(ANAPHORIC_OPENERS), "personal_openers": list(PERSONAL_OPENERS),
        "months_and_days": list(MONTHS_AND_DAYS), "on_verbs": list(ON_VERBS),
        "french_only_past": list(FRENCH_ONLY_PAST), "date_words": list(DATE_WORDS),
        "owner": {"contractions": [list(pair) for pair in OWNER_CONTRACTIONS],
                  "subjects": list(OWNER_SUBJECTS), "auxiliaries": list(OWNER_AUXILIARIES),
                  "possessives": [[token, list(words)] for token, words in OWNER_POSSESSIVES],
                  "persons": {name: {k: list(v) for k, v in spec.items()}
                              for name, spec in OWNER_PERSONS.items()}},
    }


@dataclass(frozen=True)
class Claim:
    """One claim as handed in. ``marks`` None means its text is markdown still to be read."""

    text: str
    lang: str = "und"
    kind: str = "auto"
    origin: dict = field(default_factory=dict)
    marks: object = None
    start: object = None
    end: object = None

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError("a claim's text is a string")
        if self.lang not in ("en", "fr", "und"):
            raise ValueError(f"unknown claim language {self.lang!r}")
        if self.kind not in ("world", "own", "auto"):
            raise ValueError(f"unknown claim kind {self.kind!r}")
        if self.marks is not None:
            object.__setattr__(self, "marks", tuple(self.marks))
        object.__setattr__(self, "origin", dict(self.origin or {}))


def as_claim(claim):
    if isinstance(claim, Claim):
        return claim
    if isinstance(claim, str):
        return Claim(text=claim)
    raise TypeError(f"a claim is a string or a Claim, not {type(claim).__name__}")


def _ascii(char):
    return "".join(c for c in unicodedata.normalize("NFD", char) if not unicodedata.combining(c))


_PIECES = {}


def _pieces(char):
    """What one character becomes for lexicon matching (remembered: the answer never changes)."""
    pieces = _PIECES.get(char)
    if pieces is None:
        if char.isspace():
            pieces = " "
        else:
            pieces = "".join(_ascii(c) for c in passage.fold_text(char)).lower()
        _PIECES[char] = pieces
    return pieces


def key(text):
    """Lowercase, accent-free, folded text for lexicon matching, and each character's position."""
    out = []
    positions = []
    for index, char in enumerate(text):
        pieces = _pieces(char)
        for piece in pieces:
            out.append(piece)
            positions.append(index)
    return "".join(out), positions


_COMPILED = {}


def _phrases(phrases):
    """One pattern matching any phrase of ``phrases`` as whole words."""
    pattern = _COMPILED.get(phrases)
    if pattern is None:
        parts = []
        for phrase in sorted(phrases, key=len, reverse=True):
            tail = "" if phrase.endswith("'") else r"(?![^\W_])"
            parts.append(r"(?<![^\W_])" + re.escape(phrase).replace(r"\ ", r"\s+") + tail)
        pattern = re.compile("(?:" + "|".join(parts) + ")")
        _COMPILED[phrases] = pattern
    return pattern


def _opening(phrases):
    """A pattern for ``phrases`` at the start, after an optional discourse marker."""
    marker = _phrases(DISCOURSE_MARKERS).pattern
    return re.compile(r"^\s*(?:" + marker + r"[\s,]+)?(?:" + _phrases(phrases).pattern + ")")


def find_markers(text, phrases):
    """Every phrase of ``phrases`` found in ``text``, in the order of ``phrases``."""
    if not text:
        return []
    return list(_markers(text, tuple(phrases)))


@functools.lru_cache(maxsize=4096)
def _markers(text, phrases):
    folded = key(text)[0]
    found = {m.group(0) for m in _phrases(phrases).finditer(folded)}
    normal = {re.sub(r"\s+", " ", f) for f in found}
    return tuple(p for p in phrases if p in normal)


def qualifiers(text):
    return find_markers(text, QUALIFYING_MARKERS)


def attributions(text):
    return find_markers(text, ATTRIBUTION_MARKERS)


_LOWER_MAY = re.compile(r"(?<![^\W_])may(?![^\W_])")


def strength(text):
    """(reason, marker) for each class of strength marker ``text`` holds, in the order of the classes."""
    return list(_strength(text)) if text else []


@functools.lru_cache(maxsize=4096)
def _strength(text):
    found = []
    for code, words in STRENGTH_MARKERS:
        markers = [m for m in find_markers(text, words) if m != "may" or _LOWER_MAY.search(text)]
        if markers:
            found.append((code, markers[0]))
    return tuple(found)


@functools.lru_cache(maxsize=4096)
def denial(text):
    """The denial a sentence is made of ("No.", "Not so.", or their French forms), or None."""
    if not text:
        return None
    folded = key(text)[0]
    match = re.match(r"^\s*(" + _phrases(DENIALS).pattern + r")\s*(?:[,.!;]|$)", folded)
    return re.sub(r"\s+", " ", match.group(1)) if match else None


def _strip_final(text):
    text = text.rstrip()
    while text and text[-1] in "\"'":
        text = text[:-1].rstrip()
    if text and text[-1] in ".!;":
        text = text[:-1].rstrip()
    while text and text[-1] in "\"'":
        text = text[:-1].rstrip()
    return text


_CONTRACTION = re.compile(r"(?i)(?<![^\W_])(i|we|you)'(ve|d)(?![^\W_])")
_OWNER_TOKEN = re.compile(r"(?i)(?<![^\W_])[jn]'(?=[^\W_])|[^\W_]+")


def _fold_word(word):
    return "".join(_ascii(c) for c in word).lower()


def _on_is_subject(next_word, lang):
    """French "on" as a subject: never in an English claim, and only before a verb of the list."""
    return lang != "en" and next_word in ON_VERBS


def normalise_owner(text, *, person="first", lang="und"):
    """The closed owner rewrite, for text written in ``person``: the opening subject, its auxiliary, possessives."""
    spec = OWNER_PERSONS[person]
    text = _CONTRACTION.sub(lambda m: f"{m.group(1)} {'have' if m.group(2).lower() == 've' else 'had'}", text)
    words = list(_OWNER_TOKEN.finditer(text))
    folded = [_fold_word(m.group(0)) for m in words]
    opening = 0
    for marker in sorted(DISCOURSE_MARKERS, key=lambda m: -len(m.split())):
        parts = marker.split()
        if folded[:len(parts)] == parts:
            opening = len(parts)
            break
    possessive = {word: token for token, group in OWNER_POSSESSIVES for word in group
                  if word in spec["possessives"]}
    replace = {}
    if opening < len(words):
        subject = folded[opening]
        following = folded[opening + 1] if opening + 1 < len(words) else ""
        if subject in spec["subjects"] and (subject != "on" or _on_is_subject(following, lang)):
            replace[opening] = "<owner> " if subject == "j'" else "<owner>"
            if opening + 1 < len(words):
                separator = text[words[opening].end():words[opening + 1].start()]
                if not separator.strip() and following in OWNER_AUXILIARIES:
                    replace[opening + 1] = "<aux>"
    for index, word in enumerate(folded):
        if index not in replace and word in possessive:
            replace[index] = possessive[word]
    out = []
    last = 0
    for index, match in enumerate(words):
        out.append(text[last:match.start()])
        out.append(replace.get(index, match.group(0)))
        last = match.end()
    out.append(text[last:])
    return "".join(out)


def compare_form(text, *, person=None, lang="und"):
    """The form two sentences are compared in: fold v1, final punctuation aside, first letter lowered.

    ``person`` names who the owner is in ``text`` ("first" in the owner's own
    text, "second" in the assistant's) when the owner's pronouns are rewritten.
    """
    folded = passage.fold_text(unicodedata.normalize("NFC", text)).strip(" ")
    folded = _strip_final(folded)
    if folded and folded[0].isalpha():
        folded = folded[0].lower() + folded[1:]
    if person:
        folded = normalise_owner(folded, person=person, lang=lang)
    return folded


_WORD = re.compile(r"[^\W_]")
_CAPITALISED = re.compile(r"(?<![^\W_])[^\W\d_][^\W_]*")
_MIXED = re.compile(r"(?<![^\W_])(?=[^\W_]*[^\W\d_])(?=[^\W_]*\d)[^\W_]+")


def _named_past_first(text):
    """A capitalised word past the first word (a month or a weekday is not a name), or letters mixed with digits."""
    words = list(_CAPITALISED.finditer(text))
    first_start = min((m.start() for m in re.finditer(r"[^\W_]+", text)), default=None)
    if any(m.group(0)[0].isupper() and m.start() != first_start and _fold_word(m.group(0)) not in MONTHS_AND_DAYS
           for m in words):
        return True
    return bool(_MIXED.search(text))


@functools.lru_cache(maxsize=1)
def _past_decision_pattern():
    subject = r"(?:(?:i|we|you|je|nous|on|tu|vous)(?:'ve|'d)?\s+|j')"
    auxiliary = r"(?:(?:" + "|".join(OWNER_AUXILIARIES + ("have", "had")) + r")\s+)?"
    verbs = _phrases(PAST_DECISION_VERBS).pattern
    marker = _phrases(DISCOURSE_MARKERS).pattern
    return re.compile(r"^\s*(?:" + marker + r"[\s,]+)?" + subject + auxiliary + "(?P<verb>" + verbs + ")")


def _person_of_subject(rest, lang):
    """Which person the opening subject is ("first", "second"), or None; ``rest`` is lexicon text."""
    match = re.match(r"^\s*(?:(?:" + _phrases(DISCOURSE_MARKERS).pattern + r")[\s,]+)?(j'|n'|[^\W_]+)"
                     r"(?:\s*(n'|[^\W_]+))?", rest)
    if not match:
        return None
    word, following = match.group(1), match.group(2) or ""
    if word in ("i", "we", "je", "j'", "nous") or (word == "on" and _on_is_subject(following, lang)):
        return "first"
    if word in SECOND_PERSON:
        return "second"
    return None


def _past_decision(checked, lang):
    """Whether ``checked`` reports a past decision: owner subject, optional auxiliary, a verb of the list."""
    rest, positions = key(checked)
    match = _past_decision_pattern().match(rest)
    if not match:
        return False
    verb = match.group("verb")
    if verb in FRENCH_ONLY_PAST and lang != "fr":
        original = checked[positions[match.start("verb")]:positions[match.end("verb") - 1] + 1]
        return any(ord(c) > 127 for c in original)
    return True


_YEAR = re.compile(r"(?<!\d)\d{4}(?!\d)")


def carries_date(checked):
    """A year or a month name in the claim: its own date, which this core does not read."""
    return bool(_YEAR.search(checked)) or bool(_phrases(DATE_WORDS).search(key(checked)[0]))


@dataclass(frozen=True)
class Scoped:
    """What the scope step decided about one claim."""

    in_scope: bool
    reason: object
    wrapper: str
    marks: tuple
    text: str
    checked: str
    checked_start: int
    checked_end: int
    kind: str
    past_decision: bool
    time_sensitive: bool
    view: object
    dated: bool = False
    person: object = None


def _marks_of(view, start=None, end=None):
    """Marks the whole view, or the plain range [start, end) of one sentence, carries."""
    whole = start is None
    start, end = (0, len(view.text)) if whole else (start, end)
    if whole:
        lines = [line for line in view.lines if line.kind != markup.BLANK]
    else:
        lines = [line for line in view.lines if line.start < end and start < line.end]
    kinds = {line.kind for line in lines}
    marks = []
    visible = [p for p in range(start, end) if not view.text[p].isspace()]
    inline_code = bool(visible) and sum(1 for p in visible if p in view.code_chars) * 2 >= len(visible)
    if kinds & {markup.CODE, markup.FENCE} or inline_code:
        marks.append("code")
    if markup.HEADING in kinds:
        marks.append("heading")
    if markup.TABLE_ROW in kinds:
        marks.append("table_row")
    if whole and any(line.image for line in lines):
        marks.append("image")
    if not whole:
        ostart, oend = markup.to_original(view, start, end)
        if any(ostart <= a < oend for a, _ in view.images):
            marks.append("image")
    if (markup.UNPARSED in kinds or any(start <= p < end for p in view.html_chars)
            or any(start <= p < end for p in view.struck_chars)):
        marks.append("markup_unparsed")
    if markup.QUOTED in kinds:
        marks.append("quoted")
    if kinds & {markup.LIST_ITEM, markup.LIST_CONT}:
        marks.append("list_item")
    return marks


def analyse(claim, *, max_claim_chars):
    """Scope, wrapper and kind of one claim."""
    claim = as_claim(claim)
    view = None
    if claim.marks is None:
        view = markup.plain(claim.text)
        text = view.text.strip()
        marks = tuple(_marks_of(view))
    else:
        text = claim.text
        marks = claim.marks

    def out(reason, wrapper="none", checked=text, start=0, kind="world"):
        return Scoped(False, reason, wrapper, marks, text, checked, start, start + len(checked), kind,
                      False, False, view)

    for reason in MARKUP_REASONS:
        if reason in marks:
            return out(reason)
    if not _WORD.search(text):
        return out("empty")
    if len(text) > max_claim_chars:
        return out("too_long")
    if text.rstrip().rstrip(_CLOSERS).rstrip().endswith("?"):
        return out("question")

    folded, positions = key(text)
    wrapper = "none"
    start = 0
    opinion = re.match(r"^\s*(?:" + _phrases(OPINION_OPENERS).pattern + r")(?:\s+that(?![^\W_]))?[\s,:]*",
                       folded)
    advice = re.match(r"^\s*(?:" + _phrases(ADVICE_OPENERS).pattern + r")[\s,:]*", folded)
    if opinion:
        wrapper = "hedged"
        start = positions[opinion.end()] if opinion.end() < len(positions) else len(text)
    elif advice:
        wrapper = "advice"
        start = positions[advice.end()] if advice.end() < len(positions) else len(text)
    checked = text[start:]
    if not _WORD.search(checked):
        return out("empty", wrapper, checked, start)
    rest = key(checked)[0]
    if wrapper == "none" and _opening(IMPERATIVE_VERBS).match(rest):
        wrapper = "advice"
    if wrapper == "advice" and not (any(c.isdigit() for c in checked) or _named_past_first(checked)):
        return out("instruction", wrapper, checked, start)

    kind = claim.kind
    author = str(claim.origin.get("author", "")) if isinstance(claim.origin, dict) else ""
    subject_person = _person_of_subject(rest, claim.lang)
    if kind == "auto":
        decision = bool(_phrases(DECISION_MARKERS).search(rest))
        if decision and subject_person == "first":
            kind = "own"
        elif decision and author == "assistant" and subject_person == "second":
            kind = "own"
        else:
            kind = "world"
    if _pronoun_subject(rest) or _opening(ANAPHORIC_OPENERS).match(rest):
        return out("not_standalone", wrapper, checked, start, kind)
    if kind == "world" and (subject_person or _opening(PERSONAL_OPENERS).match(rest)):
        return out("not_standalone", wrapper, checked, start, kind)
    if kind == "world" and _opening(DETERMINERS).match(rest) and not _named_past_first(checked):
        return out("subject_unresolved", wrapper, checked, start, kind)
    past = kind == "own" and _past_decision(checked, claim.lang)
    sensitive = kind == "world" and bool(_phrases(TIME_SENSITIVE).search(rest))
    person = ("second" if author == "assistant" else "first") if kind == "own" else None
    return Scoped(True, None, wrapper, marks, text, checked, start, start + len(checked), kind, past,
                  sensitive, view, carries_date(checked), person)


def _pronoun_subject(rest):
    """A pronoun subject; a demonstrative counts only standing alone (a verb of the list, or nothing, after it)."""
    match = _opening(PRONOUN_SUBJECTS).match(rest)
    if not match:
        return False
    word = re.sub(r"\s+", " ", match.group(0).strip()).split(" ")[-1]
    if word not in DEMONSTRATIVES:
        return True
    tail = rest[match.end():]
    following = re.match(r"\s*(n'|[^\W_]+)", tail)
    return following is None or following.group(1) in AFTER_PRONOUN


def claims_from_text(text, *, lang="und", origin=None, max_claim_chars=600):
    """One claim per sentence of a markdown answer, in answer order, with offsets into it.

    What the subset marks out of scope comes back too, one entry per line (a
    heading, a table row, a code line, an image, markup not read), never dropped.
    """
    del max_claim_chars  # scope is decided by the check, which reads the limit
    origin = dict(origin or {})
    view = markup.plain(text)
    entries = []
    structural = {markup.HEADING: "heading", markup.TABLE_ROW: "table_row", markup.CODE: "code",
                  markup.UNPARSED: "markup_unparsed"}
    for line in view.lines:
        mark = structural.get(line.kind)
        if mark is None and line.image and not view.text[line.start:line.end].strip():
            mark = "image"
        if mark is None:
            continue
        if line.kind == markup.HEADING:
            body = view.text[line.start:line.end].strip()
            ostart, oend = markup.to_original(view, line.start, line.end)
        else:
            raw = text[line.ostart:line.oend]
            if not raw.strip():
                continue
            body = raw.strip()
            ostart = line.ostart + (len(raw) - len(raw.lstrip()))
            oend = ostart + len(body)
        entries.append(Claim(text=body, lang=lang, origin=origin, marks=(mark,), start=ostart, end=oend))
    for sentence in passage.sentences(view):
        body = view.text[sentence.start:sentence.end]
        ostart, oend = markup.to_original(view, sentence.start, sentence.end)
        while ostart - 1 in view.inline_removed:
            ostart -= 1
        while oend in view.inline_removed:
            oend += 1
        marks = _marks_of(view, sentence.start, sentence.end)
        entries.append(Claim(text=body, lang=lang, origin=origin, marks=tuple(marks), start=ostart,
                             end=oend))
    entries.sort(key=lambda c: (c.start, c.end))
    return entries
