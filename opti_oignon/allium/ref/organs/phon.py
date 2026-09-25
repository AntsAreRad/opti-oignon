"""Phonology: the sounds a being's genome allows, the forms it may say, and how it coins them.

Everything here is a pure function of its arguments: a 63-byte language
block (read from the genome's seven LEX records), a seed, a concept and its
iconic signs, and the lists a caller passes. There is no store, no journal
and no clock; the blocks that hold a lexicon pass it in.

* The alphabet is 27 symbols, one ASCII character per phoneme, with integer
  features in a table file (``tables/phon_v1.json``) the law pins by digest.
* ``decode`` turns a block into a phonology: the potential inventory (with
  universal floors: a, i, u, the reduction and epenthetic vowels, a stop, a
  nasal, six consonants), template and word-length weights, and the
  phonotactic genes.
* ``licit`` says whether a form is sayable, with the first failing reason,
  or its syllables and split points. It never raises.
* ``invent_case`` coins a form for a concept: keyed draws, an integer iconic
  bias, at most sixteen attempts, then a gesture. A rejected candidate is
  reported only by its digest, never spelled.
* The taboo filter holds SHA-256 digests only, matched on the whole form and
  every substring of three to eight letters; the work it charges does not
  depend on what the list holds.
* ``sas_list`` derives 2048 distinct sayable forms from a block, and
  ``sas_words`` renders the first 66 bits of a digest as six of them.
"""

import hashlib
import re

from ... import rng, wire
from ...wire import Refused
from .genome import CountingStream

checkpoint_before_apply = True

ALPHABET = "aeioupbtdkgq'cjfvszxhmnlrwy"
TEMPLATES = ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1))
SAS = {"bits": 11, "cap": 16384, "exclude": [12], "size": 2048, "syllables": [3, 4], "words": 6}
SON = (1, 2, 3, 4, 5, 5, 6, 7)
S_PHONEME = 17
LEX_BYTES = 63
WEIGHTS = 27

DOMAIN_FALLBACK = "lang.fallback"
DOMAIN_FIRST = "lang.first_sound"
DOMAIN_INVENT = "lang.invent"
DOMAIN_SAS = "lang.sas"
DOMAINS = (DOMAIN_FALLBACK, DOMAIN_FIRST, DOMAIN_INVENT, DOMAIN_SAS)
SAS_PREFIX = b"oo-allium-sas-v1"

REASONS = ("length", "alphabet", "inventory", "ocp", "long_vowel", "no_nucleus", "onset", "cluster", "coda")
FORM = re.compile(r"[a-z']{1,12}")
SHOWN = re.compile(r"[a-z']{1,24}")
FOLD_REPLACEMENT = re.compile(r"[a-z']{0,2}")
TABLE_KEYS = ("alphabet", "anchored_max", "anchored_total_max", "bias", "features", "first_sound_exclude",
              "floor_consonants", "floor_vowels", "fold", "form_max", "invent_tries", "lex_box", "name",
              "potential_min", "sas", "shown_max", "taboo_extra_max", "taboo_max", "taboo_window", "templates",
              "version")

# Feature columns.
F_CLASS, F_PLACE, F_MANNER, F_VOICE, F_SON, F_HEIGHT, F_BACK, F_ROUND = range(8)
FEATURE_RANGES = ((0, 1), (0, 8), (0, 7), (0, 1), (1, 7), (0, 2), (0, 2), (0, 1))


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _int_list(value, length=None):
    return isinstance(value, list) and (length is None or len(value) == length) and all(_is_int(v) for v in value)


def sha256_hex(data):
    return hashlib.sha256(bytes(data)).hexdigest()


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

class Table:
    """A validated phonology table, with its derived classes."""

    __slots__ = ("value", "features", "bias", "lex_box", "potential_min", "anchored_max", "anchored_total_max",
                 "taboo_max", "taboo_extra_max", "index", "vowels", "cons", "stops", "nasals", "manner", "son",
                 "first_exclude", "sas_exclude")

    def __init__(self, value):
        self.value = value
        self.features = tuple(tuple(row) for row in value["features"])
        self.bias = tuple(tuple(row) for row in value["bias"])
        self.lex_box = tuple((pair[0], pair[1]) for pair in value["lex_box"])
        self.potential_min = value["potential_min"]
        self.anchored_max = value["anchored_max"]
        self.anchored_total_max = value["anchored_total_max"]
        self.taboo_max = value["taboo_max"]
        self.taboo_extra_max = value["taboo_extra_max"]
        self.index = {char: p for p, char in enumerate(ALPHABET)}
        self.manner = tuple(row[F_MANNER] for row in self.features)
        self.son = tuple(row[F_SON] for row in self.features)
        self.vowels = tuple(p for p in range(WEIGHTS) if self.features[p][F_CLASS] == 0)
        self.cons = tuple(p for p in range(WEIGHTS) if self.features[p][F_CLASS] == 1)
        self.stops = tuple(p for p in range(WEIGHTS) if self.manner[p] == 0)
        self.nasals = tuple(p for p in range(WEIGHTS) if self.manner[p] == 3)
        self.first_exclude = tuple(value["first_sound_exclude"])
        self.sas_exclude = tuple(value["sas"]["exclude"])


def validate_table(value):
    """Every defect of a phonology table; ``[]`` when it is sound."""
    if not isinstance(value, dict) or len(value) != len(TABLE_KEYS) or any(k not in value for k in TABLE_KEYS):
        return ["phon: keys are not exactly the schema's"]
    out = []
    if value["alphabet"] != ALPHABET:
        out.append("phon: alphabet")
    templates = value["templates"]
    if not isinstance(templates, list) or not all(_int_list(row) for row in templates) \
            or templates != [list(t) for t in TEMPLATES]:
        out.append("phon: templates")
    for key, expected in (("invent_tries", 16), ("form_max", 12), ("shown_max", 24), ("floor_consonants", 6)):
        if not _is_int(value[key]) or value[key] != expected:
            out.append(f"phon: {key}")
    for key, expected in (("taboo_window", [3, 8]), ("floor_vowels", [0, 2, 4]), ("first_sound_exclude", [12])):
        if not _int_list(value[key]) or value[key] != expected:
            out.append(f"phon: {key}")
    if value["sas"] != SAS or not all(_is_int(v) for k, v in value["sas"].items() if k not in ("exclude", "syllables")):
        out.append("phon: sas")
    features = value["features"]
    if not isinstance(features, list) or len(features) != WEIGHTS or not all(_int_list(r, 8) for r in features):
        out.append("phon: features")
    else:
        for p, row in enumerate(features):
            for column, (low, high) in enumerate(FEATURE_RANGES):
                if not low <= row[column] <= high:
                    out.append(f"phon: feature {p}.{column} out of range")
            vowel = p <= 4
            if (row[F_CLASS] == 0) != vowel or (row[F_MANNER] == 7) != vowel or (row[F_PLACE] == 8) != vowel:
                out.append(f"phon: feature {p} vowel classes disagree")
            if 0 <= row[F_MANNER] <= 7 and row[F_SON] != SON[row[F_MANNER]]:
                out.append(f"phon: feature {p} sonority")
            if not vowel and (row[F_HEIGHT] or row[F_BACK] or row[F_ROUND]):
                out.append(f"phon: feature {p} vowel columns on a consonant")
        if not out:
            stops = [p for p in range(WEIGHTS) if features[p][F_MANNER] == 0]
            nasals = [p for p in range(WEIGHTS) if features[p][F_MANNER] == 3]
            if stops != list(range(5, 13)) or nasals != [21, 22] or ALPHABET[S_PHONEME] != "s":
                out.append("phon: stops, nasals or the s phoneme")
    bias = value["bias"]
    if not isinstance(bias, list) or len(bias) != WEIGHTS or not all(_int_list(r, 4) for r in bias) \
            or any(not -64 <= v <= 64 for r in bias for v in r):
        out.append("phon: bias")
    box = value["lex_box"]
    if not isinstance(box, list) or len(box) != LEX_BYTES or not all(_int_list(pair, 2) for pair in box) \
            or any(not 0 <= pair[0] <= pair[1] <= 255 for pair in box):
        out.append("phon: lex_box")
    elif box[45][1] > 2 or box[46][1] > 1:
        out.append("phon: lex_box onset or coda above what licit judges")
    elif box[52][1] > 4 or box[53][1] > 4:
        out.append("phon: lex_box reduction or epenthetic vowel beyond the five vowels")
    if not _is_int(value["potential_min"]) or not 1 <= value["potential_min"] <= 255:
        out.append("phon: potential_min")
    for key in ("anchored_max", "anchored_total_max", "taboo_max", "taboo_extra_max"):
        if not _is_int(value[key]) or not 1 <= value[key] <= 65536:
            out.append(f"phon: {key}")
    fold = value["fold"]
    previous = -1
    if not isinstance(fold, list):
        out.append("phon: fold")
    else:
        for entry in fold:
            if not (isinstance(entry, list) and len(entry) == 2 and _is_int(entry[0]) and isinstance(entry[1], str)
                    and previous < entry[0] <= 0x10FFFF and FOLD_REPLACEMENT.fullmatch(entry[1])):
                out.append("phon: fold")
                break
            previous = entry[0]
    if not isinstance(value["name"], str) or not _is_int(value["version"]):
        out.append("phon: name or version")
    return out


def validate_taboo(value, table):
    """Every defect of a taboo table against the phonology table's cap; ``[]`` when sound."""
    if not isinstance(value, dict) or sorted(value) != ["entries", "name", "version"]:
        return ["taboo: keys"]
    if not isinstance(value["name"], str) or not _is_int(value["version"]):
        return ["taboo: name or version"]
    entries = value["entries"]
    if not isinstance(entries, list) or len(entries) > table.taboo_max:
        return ["taboo: entries"]
    previous = None
    for entry in entries:
        if not taboo_pair(entry):
            return ["taboo: an entry is not [length, digest]"]
        key = (entry[0], entry[1])
        if previous is not None and key <= previous:
            return ["taboo: entries not strictly ascending"]
        previous = key
    return []


_HEX64 = re.compile(r"[0-9a-f]{64}")


def taboo_pair(entry):
    return (isinstance(entry, list) and len(entry) == 2 and _is_int(entry[0]) and 1 <= entry[0] <= 12
            and isinstance(entry[1], str) and _HEX64.fullmatch(entry[1]) is not None)


def taboo_set(entries, extra=()):
    """The taboo pairs as a dict keyed ``(length, digest bytes)``; only ever tested for membership."""
    out = {}
    for length, digest in list(entries) + list(extra):
        out[(length, bytes.fromhex(digest))] = True
    return out


def law_lex(table):
    """The law-level block: every byte at the top of its box."""
    return bytes(high for _low, high in table.lex_box)


# ---------------------------------------------------------------------------
# Decoding a block
# ---------------------------------------------------------------------------

class Phonology:
    __slots__ = ("table", "lex", "w", "genes", "member", "floors", "weight", "templates", "word_len",
                 "onset_max", "coda_max", "coda_classes", "s_exception", "length", "iconicity", "c1set",
                 "codaset")


GENE_BYTES = (("iconicity", 36), ("playfulness", 37), ("conformity", 38), ("openness", 39), ("conservatism", 40),
              ("chattiness", 41), ("critical_len", 42), ("voice_pitch", 43), ("voice_tempo", 44), ("onset_max", 45),
              ("coda_max", 46), ("coda_classes", 47), ("s_exception", 48), ("stress", 49), ("harmony", 50),
              ("length", 51), ("reduction_vowel", 52), ("epenthetic_vowel", 53), ("repair", 54), ("head_order", 55),
              ("dominance", 56))


def lex_ok(lex, table):
    return len(lex) == LEX_BYTES and all(low <= b <= high for b, (low, high) in zip(lex, table.lex_box))


def _best(candidates, member, w):
    kept = None
    for p in candidates:
        if member[p]:
            continue
        if kept is None or w[p] > w[kept]:
            kept = p
    return kept


def decode(lex, table, inventory=None):
    """The phonology of a validated 63-byte block; ``inventory`` (a mask) replaces the membership."""
    lex = bytes(lex)
    ph = Phonology()
    ph.table = table
    ph.lex = lex
    w = tuple(lex[0:WEIGHTS])
    ph.w = w
    genes = {name: lex[at] for name, at in GENE_BYTES}
    ph.genes = genes
    member = [w[p] >= table.potential_min for p in range(WEIGHTS)]
    floors = []
    for p in (0, 2, 4, genes["reduction_vowel"], genes["epenthetic_vowel"]):
        if not member[p]:
            member[p] = True
            floors.append(p)
    for group in (table.stops, table.nasals):
        if not any(member[p] for p in group):
            p = _best(group, member, w)
            member[p] = True
            floors.append(p)
    while sum(1 for p in table.cons if member[p]) < 6:
        p = _best(table.cons, member, w)
        member[p] = True
        floors.append(p)
    ph.floors = floors
    if inventory is not None:
        member = [bool((inventory >> p) & 1) for p in range(WEIGHTS)]
    ph.member = tuple(member)
    ph.weight = tuple(w[p] + 1 if member[p] else 0 for p in range(WEIGHTS))
    ph.onset_max = genes["onset_max"]
    ph.coda_max = genes["coda_max"]
    ph.coda_classes = genes["coda_classes"]
    ph.s_exception = genes["s_exception"]
    ph.length = genes["length"]
    ph.iconicity = genes["iconicity"]
    ph.c1set = tuple(c for c in table.cons if member[c] and any(member[d] and cluster_ok(ph, c, d) for d in table.cons))
    ph.codaset = tuple(c for c in table.cons if member[c] and coda_ok(ph, c))
    templates = []
    for t, (onset, coda) in enumerate(TEMPLATES):
        allowed = onset <= ph.onset_max and coda <= ph.coda_max and (onset < 2 or ph.c1set) \
            and (coda < 1 or ph.codaset)
        templates.append(lex[27 + t] + 1 if allowed else 0)
    ph.templates = tuple(templates)
    ph.word_len = tuple(lex[33 + n] + 1 for n in range(3))
    return ph


def cluster_ok(ph, c, d):
    table = ph.table
    if c == d:
        return False
    if table.son[d] - table.son[c] >= 2:
        return True
    return ph.s_exception == 1 and c == S_PHONEME and table.manner[d] == 0


def coda_ok(ph, c):
    return ((ph.coda_classes >> ph.table.manner[c]) & 1) == 1


def phonology_value(ph):
    """The wire form of a phonology (``phon_inventory``)."""
    inventory = [p for p in range(WEIGHTS) if ph.member[p]]
    mask = 0
    for p in inventory:
        mask |= 1 << p
    out = {name: ph.genes[name] for name, _at in GENE_BYTES}
    out.update({"floors": list(ph.floors), "inventory": inventory, "mask": mask, "templates": list(ph.templates),
                "weights": list(ph.weight), "word_len": list(ph.word_len)})
    return out


def fallback_lex(seed, table):
    """A block from the seed alone, each byte uniform in its box: ``(block, draws)``."""
    stream = CountingStream.from_key(rng.key(bytes(seed), DOMAIN_FALLBACK, ()))
    block = bytearray(LEX_BYTES)
    for i, (low, high) in enumerate(table.lex_box):
        block[i] = low if low == high else low + stream.below(high - low + 1)
    return bytes(block), stream.draws


def lex_from_tables(tables):
    """The block the seven LEX rows of compiled tables carry, placed by their offset field."""
    fields = wire.unpack_bulk(tables["reserved"]["lex"]["fields"])[1]
    if len(fields) != 70:
        raise Refused("engine_panic", "lex block")
    block = bytearray(LEX_BYTES)
    seen = []
    for i in range(7):
        offset = fields[10 * i]
        if offset not in (0, 9, 18, 27, 36, 45, 54) or offset in seen:
            raise Refused("engine_panic", "lex block")
        seen.append(offset)
        for k in range(9):
            value = fields[10 * i + 1 + k]
            if not 0 <= value <= 255:
                raise Refused("engine_panic", "lex block")
            block[offset + k] = value
    return bytes(block)


# ---------------------------------------------------------------------------
# Licit forms
# ---------------------------------------------------------------------------

def licit(form, ph):
    """``(None, syllables, splits)`` for a sayable form, else ``(reason, 0, [])``."""
    if not isinstance(form, str) or not 1 <= len(form) <= 12:
        return "length", 0, []
    if FORM.fullmatch(form) is None:
        return "alphabet", 0, []
    table = ph.table
    ps = [table.index[char] for char in form]
    if not all(ph.member[p] for p in ps):
        return "inventory", 0, []
    is_vowel = [table.features[p][F_CLASS] == 0 for p in ps]
    for i in range(1, len(ps)):
        if not is_vowel[i] and not is_vowel[i - 1] and ps[i] == ps[i - 1]:
            return "ocp", 0, []
    nuclei = []  # (start, length)
    i = 0
    n = len(ps)
    while i < n:
        if not is_vowel[i]:
            i += 1
            continue
        j = i
        while j < n and is_vowel[j] and ps[j] == ps[i]:
            j += 1
        run = j - i
        if run == 2 and ph.length != 1 or run >= 3:
            return "long_vowel", 0, []
        nuclei.append((i, run))
        i = j
    if not nuclei:
        return "no_nucleus", 0, []
    first = nuclei[0][0]
    initial = ps[:first]
    if len(initial) > ph.onset_max or len(initial) == 2 and not cluster_ok(ph, initial[0], initial[1]):
        return "onset", 0, []
    splits = []
    for (start, run), (next_start, _next_run) in zip(nuclei, nuclei[1:]):
        s0 = start + run
        m = next_start - s0
        if m == 0:
            splits.append(next_start)
            continue
        cluster = ps[s0:next_start]
        chosen = None
        for o in range(min(m, ph.onset_max), -1, -1):
            cd = m - o
            if cd > ph.coda_max:
                continue
            if cd and not coda_ok(ph, cluster[0]):
                continue
            if o >= 2 and not cluster_ok(ph, cluster[cd], cluster[cd + 1]):
                continue
            chosen = o
            break
        if chosen is None:
            return "cluster", 0, []
        splits.append(s0 + m - chosen)
    last_start, last_run = nuclei[-1]
    final = ps[last_start + last_run:]
    if len(final) > ph.coda_max or len(final) == 1 and not coda_ok(ph, final[0]):
        return "coda", 0, []
    return None, len(nuclei), splits


# ---------------------------------------------------------------------------
# Draws, taboo, invention
# ---------------------------------------------------------------------------

def pick(stream, weights):
    """An index drawn with probability ``weights[i] / sum``: first index whose running sum exceeds ``below(sum)``."""
    total = 0
    for value in weights:
        total += value
    if total < 1:
        raise Refused("engine_panic", "phon pick")
    r = stream.below(total)
    running = 0
    for index, value in enumerate(weights):
        running += value
        if running > r:
            return index
    raise Refused("engine_panic", "phon pick")


def taboo_work(n):
    work = 1
    for length in range(3, min(8, n - 1) + 1):
        work += n - length + 1
    return work


def taboo_hit(form, taboo):
    """True when the whole form, or a substring of three to eight letters, is listed."""
    data = form.encode("ascii")
    n = len(data)
    if (n, hashlib.sha256(data).digest()) in taboo:
        return True
    for length in range(3, min(8, n - 1) + 1):
        for i in range(n - length + 1):
            if (length, hashlib.sha256(data[i:i + length]).digest()) in taboo:
                return True
    return False


def near1(a, b):
    """Exact Levenshtein distance 1."""
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    if abs(len(a) - len(b)) != 1:
        return False
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    q = 0
    while q < len(shorter) and longer[q] == shorter[q]:
        q += 1
    return longer[q + 1:] == shorter[q:]


def iconic_weights(ph, signs):
    table = ph.table
    out = []
    for p in range(WEIGHTS):
        if not ph.member[p]:
            out.append(0)
            continue
        s = 0
        for f in range(4):
            s += signs[f] * table.bias[p][f]
        adjust = (ph.iconicity * s) // 256
        out.append(max(1, ph.weight[p] + adjust))
    return out


def _restricted(weights, allowed):
    return [weights[p] if allowed[p] else 0 for p in range(WEIGHTS)]


def _draw_form(stream, ph, wi, syllables):
    table = ph.table
    n = syllables if syllables else 1 + pick(stream, ph.word_len)
    cons = [ph.member[p] and table.features[p][F_CLASS] == 1 for p in range(WEIGHTS)]
    vowels = [ph.member[p] and table.features[p][F_CLASS] == 0 for p in range(WEIGHTS)]
    c1 = [p in ph.c1set for p in range(WEIGHTS)]
    coda = [p in ph.codaset for p in range(WEIGHTS)]
    letters = []
    for _ in range(n):
        onset, coda_len = TEMPLATES[pick(stream, ph.templates)]
        if onset == 1:
            letters.append(pick(stream, _restricted(wi, cons)))
        elif onset == 2:
            first = pick(stream, _restricted(wi, c1))
            # A cluster is two consonants: a vowel's sonority would pass the rise test, so it is excluded.
            second = [ph.member[d] and table.features[d][F_CLASS] == 1 and cluster_ok(ph, first, d)
                      for d in range(WEIGHTS)]
            letters.append(first)
            letters.append(pick(stream, _restricted(wi, second)))
        letters.append(pick(stream, _restricted(wi, vowels)))
        if coda_len:
            letters.append(pick(stream, _restricted(wi, coda)))
    return "".join(ALPHABET[p] for p in letters)


def invent_case(ph, seed, concept, coin, signs, epoch, syllables, anchored, others, taboo, tries_max=16):
    """``(case output, work)`` for one coinage."""
    wi = iconic_weights(ph, signs)
    known = {}
    for form in list(anchored) + list(others):
        known[form] = True
    work = LEX_BYTES + len(anchored) + len(others)
    tries = []
    for k in range(tries_max):
        stream = CountingStream.from_key(rng.key(bytes(seed), DOMAIN_INVENT, (concept, coin, k, epoch)))
        form = _draw_form(stream, ph, wi, syllables)
        work += stream.draws + len(form) + 1
        reason, count, _splits = licit(form, ph)
        if reason is None:
            work += 1
            if form in known:
                reason = "same"
            else:
                work += len(form) + 1
                if any(near1(form, a) for a in anchored):
                    reason = "near"
                else:
                    work += taboo_work(len(form))
                    if taboo_hit(form, taboo):
                        reason = "taboo"
        if reason is None:
            return {"form": form, "outcome": "word", "syllables": count, "tries": tries}, work
        work += 1
        tries.append([sha256_hex(form.encode("ascii")), reason])
    return {"outcome": "gesture", "tries": tries}, work


def first_sound(seed, ph):
    """``(phoneme, draws)``: one potential phoneme keyed on the seed and the block."""
    table = ph.table
    stream = CountingStream.from_key(rng.key(bytes(seed), DOMAIN_FIRST, ()))
    weights = [ph.weight[p] if ph.member[p] and p not in table.first_exclude else 0 for p in range(WEIGHTS)]
    return pick(stream, weights), stream.draws


# ---------------------------------------------------------------------------
# SAS words
# ---------------------------------------------------------------------------

def sas_list(ph, taboo):
    """``(list, candidates drawn, work)``: 2048 distinct sayable forms derived from the block alone."""
    table = ph.table
    seed = hashlib.sha256(SAS_PREFIX + b"\x00" + ph.lex).digest()
    stream = CountingStream.from_key(rng.key(seed, DOMAIN_SAS, ()))
    cons = [1 if ph.member[p] and table.features[p][F_CLASS] == 1 and p not in table.sas_exclude else 0
            for p in range(WEIGHTS)]
    vowels = [1 if ph.member[p] and table.features[p][F_CLASS] == 0 else 0 for p in range(WEIGHTS)]
    size, cap = SAS["size"], SAS["cap"]
    words = []
    seen = {}
    candidates = 0
    taboo_cost = 0
    for syllables in SAS["syllables"]:
        drawn = 0
        while len(words) < size and drawn < cap:
            letters = []
            for _ in range(syllables):
                letters.append(pick(stream, cons))
                letters.append(pick(stream, vowels))
            form = "".join(ALPHABET[p] for p in letters)
            drawn += 1
            candidates += 1
            if form in seen:
                continue
            taboo_cost += taboo_work(len(form))
            if taboo_hit(form, taboo):
                continue
            seen[form] = True
            words.append(form)
        if len(words) >= size:
            break
    if len(words) < size:
        raise Refused("limit", "sas list")
    return words, candidates, stream.draws + taboo_cost


def sas_indices(digest):
    x = int.from_bytes(bytes(digest)[0:9], "big") >> 6
    return [(x >> (11 * (5 - j))) & 0x7FF for j in range(6)], x


def sas_parse(words, positions):
    indices = [positions.get(word, -1) for word in words]
    if any(i < 0 for i in indices):
        return {"indices": indices, "value": None}
    x = 0
    for index in indices:
        x = (x << 11) | index
    return {"indices": indices, "value": x.to_bytes(9, "big").hex()}
