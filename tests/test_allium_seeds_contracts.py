#!/usr/bin/env python3
"""Contracts for the componion's taboo displacement and its six-word lists.

  * SG11 -- a taboo hit moves a coinage deterministically: the attempts
    before it do not move, the hit attempt is reported by its digest, the
    word becomes a later attempt; the order or repetition of added entries
    changes nothing, an entry that matches nothing changes no byte (its work
    included), an entry labelled with the wrong length never matches, and a
    form hit at two lengths answers as if hit at one. The same in both
    engines.
  * SG12 -- a block's six-word list holds 2048 distinct sayable forms free
    of the taboo list; six words carry exactly the first 66 bits of a
    digest, one word per eleven bits; six words parse back to those bits,
    and an unknown word is pointed at; the list never depends on the
    moment it is asked; a taboo list that empties the short forms moves the
    list to its longer phase. The same in both engines.

Local-only; both contracts need the native artefact.
"""

import hashlib
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_phon_support as S  # noqa: E402
from _allium_window import open_allium  # noqa: E402

BUDGET_S = {
    "test_sg11_a_taboo_hit_moves_a_coinage_deterministically_and_additions_replay_identically": 2.0,
    "test_sg12_the_six_word_list_is_2048_sayable_forms_and_six_words_carry_66_bits": 2.0,
}


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


@pytest.fixture
def engines():
    loaded, restore = open_allium(native=True)
    try:
        yield S.Engines(loaded, native=True)
    finally:
        restore()


def test_sg11_a_taboo_hit_moves_a_coinage_deterministically_and_additions_replay_identically(engines):
    e = engines
    table = e.table()
    blocks = [S.hi_block(table)] + [S.keyed_block(e.rng, table, "test.sg11", i) for i in range(3)]
    cases = [{"anchored": [], "coin": 0, "concept": i, "epoch": 0, "lex": blocks[i % 4].hex(), "others": [],
              "seed": "%064x" % (i + 1), "signs": [((i + f) % 3) - 1 for f in range(4)],
              "syllables": 3 if i % 4 == 0 else 0} for i in range(160)]

    def ask(chosen, extra):
        request = {"cases": chosen, "law": "v0_1", "op": "phon_invent"}
        if extra is not None:
            request["taboo_extra"] = extra
        return e.both(request)

    plain = ask(cases, None)
    moved = both_lengths = 0
    eligible, firsts, pairs, wrongs = [], [], [], []
    for case, out in zip(cases, plain["out"]):
        if out["outcome"] != "word" or len(out["form"]) < 5:
            continue
        form, k = out["form"], len(out["tries"])
        subs = [[3, S.sha(form[1:4])], [len(form) - 1, S.sha(form[1:])] if len(form) - 1 <= 8 else [8, S.sha(form[1:9])],
                [len(form), S.sha(form)]]
        for entry in subs:
            answer = ask([case], [entry])["out"][0]
            assert answer["tries"][:k] == out["tries"], "an earlier attempt moved"
            assert answer["tries"][k] == [S.sha(form), "taboo"], (entry, answer)
            assert answer.get("form") != form
            moved += 1
        eligible.append(case)
        firsts.append(subs[0])
        pairs += [subs[0], subs[1]]
        wrongs.append([5, S.sha(form[1:4])])
        if len(form) >= 10:
            three, eight = [3, S.sha(form[1:4])], [8, S.sha(form[1:9])]
            assert ask([case], [three]) == ask([case], [eight]) == ask([case], [three, eight]), \
                "a form hit at two lengths answers apart"
            both_lengths += 1
    # One request each over every eligible case: an entry that cannot match
    # a case (another length, another form) must leave it alone as well.
    mislabelled = len(wrongs)
    assert ask(eligible, firsts + firsts) == ask(eligible, firsts), "a repeated entry changes the answer"
    assert ask(eligible, pairs) == ask(eligible, list(reversed(pairs))), "the order of added entries changes the answer"
    absent = [11, S.sha("q" * 11)]
    assert ask(cases, [absent]) == plain, "an entry matching nothing changed a byte"
    assert ask(eligible, wrongs) == ask(eligible, None), "an entry labelled with the wrong length matched"
    assert moved >= 150 and mislabelled >= 50 and both_lengths >= 3, (moved, mislabelled, both_lengths)


def _bits(digest):
    """Test-side bit numbering: bit i is (digest[i // 8] >> (7 - i % 8)) & 1."""
    return [(digest[i // 8] >> (7 - i % 8)) & 1 for i in range(len(digest) * 8)]


def _indices(digest):
    bits = _bits(digest)
    return [int("".join(str(b) for b in bits[11 * j:11 * j + 11]), 2) for j in range(6)]


def _flip(digest, bit):
    data = bytearray(digest)
    data[bit // 8] ^= 1 << (7 - bit % 8)
    return bytes(data)


def test_sg12_the_six_word_list_is_2048_sayable_forms_and_six_words_carry_66_bits(engines):
    e = engines
    table = e.table()
    phon = e.phon
    pairs = {(length, digest): True for length, digest in e.lawfiles.table("taboo_fixture_v1")["entries"]}
    blocks = [("fixture", S.lo_block(table)), ("fixture", S.hi_block(table)), ("fixture", S.min_space_block(table)),
              ("v0_1", S.min_space_block(table))]
    for seed in S.seeds(e.rng, "test.sg12", 3):
        genome = e.ask({"law": "fixture", "op": "genome_found", "seed": seed.hex()})["genome"]
        blocks.append(("fixture", bytes.fromhex(e.ask({"genome": genome, "law": "fixture", "op": "phon_lex"})["lex"])))
    blocks += [("fixture", S.keyed_block(e.rng, table, "test.sg12.block", i)) for i in range(2)]
    blocks.append(("v0_1", S.keyed_block(e.rng, table, "test.sg12.v0_1", 0)))
    base = hashlib.sha256(b"sg12").digest()
    digests = [base] + [_flip(base, bit) for bit in range(256)]
    for law, block in blocks:
        answer = e.both({"law": law, "lex": block.hex(), "op": "phon_sas"})
        words = answer["list"]
        assert len(words) == 2048 and len({w: True for w in words}) == 2048
        ph = phon.decode(block, table)
        for word in words:
            assert phon.licit(word, ph)[0] is None and phon.FORM.fullmatch(word), word
            assert "'" not in word
        if law == "fixture":
            assert not any(S.listed(w, pairs) for w in words)
        assert answer["sha256"] == hashlib.sha256(e.wire.emit(words)).hexdigest()
    law, block = blocks[1]
    request = {"digests": [d.hex() for d in digests], "law": law, "lex": block.hex(), "op": "phon_sas"}
    first = e.both(request)
    assert e.both(request) == first, "the list or the words moved between two identical calls"
    words = first["list"]
    base_row = first["indices"][0]
    for row, digest in zip(first["indices"], digests):
        assert row == _indices(digest), digest.hex()
    for bit in range(256):
        row = first["indices"][bit + 1]
        changed = sum(1 for a, b in zip(row, base_row) if a != b)
        assert changed == (1 if bit < 66 else 0), (bit, changed)
    assert first["words"][0] == [words[i] for i in base_row]
    phrases = [first["words"][0], first["words"][0][:3] + ["zzzzzz"] + first["words"][0][4:]]
    parsed = e.both({"law": law, "lex": block.hex(), "op": "phon_sas", "phrases": phrases})["parsed"]
    x = int.from_bytes(base[0:9], "big") >> 6
    assert parsed[0] == {"indices": base_row, "value": x.to_bytes(9, "big").hex()}
    assert parsed[1]["indices"][3] == -1 and parsed[1]["value"] is None
    law, block = blocks[2]
    kill = [[3, S.sha(v1 + c + v2)] for v1 in "aiu" for c in "pb" for v2 in "aiu"]
    longer = e.both({"law": law, "lex": block.hex(), "op": "phon_sas", "taboo_extra": kill})
    assert longer["candidates"] > 16384 and any(len(w) == 8 for w in longer["list"]), longer["candidates"]
    assert len({w: True for w in longer["list"]}) == 2048


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
