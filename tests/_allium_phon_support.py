#!/usr/bin/env python3
"""Shared support for the componion's phonology contracts: the two engines, blocks, an oracle.

Corpora are drawn from the chassis stream on fixed keys, never from
``random``. The oracle here judges a form by trying every split of every
consonant cluster independently -- written from the published rules, not
from the reference -- so an acceptance can be checked against something
that is not the engine itself.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_window import native_module  # noqa: E402

ALPHABET = "aeioupbtdkgq'cjfvszxhmnlrwy"
VOWELS = range(0, 5)
SON = (7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6)
MANNER = (7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 6)
INPUT_MAX = 1 << 20


class Engines:
    """The reference and (when asked) the native engine of one window."""

    def __init__(self, loaded, native=True):
        self.loaded = loaded
        self.wire = loaded["opti_oignon.allium.wire"]
        self.rng = loaded["opti_oignon.allium.rng"]
        self.lawfiles = loaded["opti_oignon.allium.lawfiles"]
        self.genome = loaded["opti_oignon.allium.ref.organs.genome"]
        self.phon = loaded["opti_oignon.allium.ref.organs.phon"]
        self.protocol = loaded["opti_oignon.allium.ref.protocol"]
        self.native = native_module(loaded) if native else None
        self.calls = {}
        self.compared = 0

    def request(self, obj):
        data = self.wire.emit(dict(obj, v=1))
        assert len(data) <= INPUT_MAX, f"request of {len(data)} bytes exceeds the input limit"
        return data

    def ask(self, obj):
        """The reference's answer, parsed."""
        return self.wire.parse(self.protocol.call(self.request(obj)))

    def raw(self, obj):
        return self.protocol.call(self.request(obj))

    def both(self, obj):
        """Both engines' answers must be the same bytes; the parsed answer."""
        data = self.request(obj)
        mine = self.protocol.call(data)
        op = obj.get("op")
        self.calls[op] = self.calls.get(op, 0) + 1
        theirs = bytes(self.native.allium_call(data))
        assert theirs == mine, f"{op}: {data[:160]!r}\n{mine[:240]!r}\n{theirs[:240]!r}"
        self.compared += 1
        return self.wire.parse(theirs)

    def table(self):
        value = self.lawfiles.table("phon_v1")
        return self.phon.Table(value)


def seeds(rng, domain, count):
    out = []
    for i in range(count):
        stream = rng.Stream(bytes(32), domain, i)
        out.append(b"".join(stream.next_u64().to_bytes(8, "big") for _ in range(4)))
    return out


def lo_block(table):
    return bytes(low for low, _high in table.lex_box)


def hi_block(table):
    return bytes(high for _low, high in table.lex_box)


def keyed_block(rng, table, domain, i):
    stream = rng.Stream(bytes(32), domain, i)
    return bytes(low if low == high else low + stream.below(high - low + 1) for low, high in table.lex_box)


def min_space_block(table):
    """Every weight 0 but the glottal stop's 127: the floors leave a i u, ', m and p b t d."""
    block = bytearray(lo_block(table))
    block[12] = 127
    return bytes(block)


def mask_of(letters):
    mask = 0
    for char in letters:
        mask |= 1 << ALPHABET.index(char)
    return mask


def sha(text):
    return hashlib.sha256(text.encode("ascii") if isinstance(text, str) else bytes(text)).hexdigest()


def listed(form, pairs):
    """Test-side taboo check with hashlib: the whole form or any substring of 3..=8 letters."""
    n = len(form)
    if (n, sha(form)) in pairs:
        return True
    for length in range(3, min(8, n - 1) + 1):
        for i in range(n - length + 1):
            if (length, sha(form[i:i + length])) in pairs:
                return True
    return False


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------

def _cluster_ok(c, d, s_exception):
    if c == d:
        return False
    if SON[d] - SON[c] >= 2:
        return True
    return s_exception == 1 and c == 17 and MANNER[d] == 0


def oracle(form, members, onset_max, coda_max, coda_classes, s_exception, length):
    """``None`` when the form is not sayable, else ``(syllables, splits)`` by maximal onset."""
    if not 1 <= len(form) <= 12 or any(ch not in ALPHABET for ch in form):
        return None
    ps = [ALPHABET.index(ch) for ch in form]
    if any(not members[p] for p in ps):
        return None
    vowel = [p < 5 for p in ps]
    for i in range(1, len(ps)):
        if not vowel[i] and ps[i] == ps[i - 1]:
            return None
    nuclei = []
    i = 0
    while i < len(ps):
        if not vowel[i]:
            i += 1
            continue
        j = i
        while j < len(ps) and vowel[j] and ps[j] == ps[i]:
            j += 1
        if j - i >= 3 or (j - i == 2 and length != 1):
            return None
        nuclei.append((i, j - i))
        i = j
    if not nuclei:
        return None

    def coda(c):
        return (coda_classes >> MANNER[c]) & 1 == 1

    initial = ps[:nuclei[0][0]]
    if len(initial) > onset_max or (len(initial) == 2 and not _cluster_ok(initial[0], initial[1], s_exception)):
        return None
    splits = []
    for (start, run), (nxt, _r) in zip(nuclei, nuclei[1:]):
        s0 = start + run
        cluster = ps[s0:nxt]
        m = len(cluster)
        if m == 0:
            splits.append(nxt)
            continue
        valid = []
        for o in range(0, min(m, 2) + 1):
            cd = m - o
            ok = o <= onset_max and cd <= coda_max and (cd == 0 or coda(cluster[0]))
            ok = ok and (o < 2 or _cluster_ok(cluster[cd], cluster[cd + 1], s_exception))
            if ok:
                valid.append(o)
        if not valid:
            return None
        splits.append(s0 + m - max(valid))
    last, run = nuclei[-1]
    final = ps[last + run:]
    if len(final) > coda_max or (len(final) == 1 and not coda(final[0])):
        return None
    return len(nuclei), splits
