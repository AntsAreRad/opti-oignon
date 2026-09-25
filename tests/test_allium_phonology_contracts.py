#!/usr/bin/env python3
"""Contracts for the componion's phonology: its sounds, its sayable forms, its coinages.

A being's language starts from sixty-three bytes of its genome. They decide
which of twenty-seven sounds it can make, how it builds syllables, and how
it coins a word for a concept. Everything is a pure function of its inputs,
written twice (a Python reference, a Rust twin) and compared.

  * PN1 -- the alphabet and its features are one table: both engines export
    the same bytes, the export equals the file, its invariants hold, the
    authoring script reproduces the tables and the laws pin their digests.
  * PN2 -- the same block always gives the same phonology, and its floors
    always hold: the corner vowels, the reduction and epenthetic vowels, a
    stop, a nasal and six consonants, over ten thousand blocks.
  * PN3 -- the phonology reads the language block by its offsets, and only
    it: a change elsewhere in the genome changes nothing, and each byte
    changes the field it names.
  * PN4 -- ``licit`` is total and agrees with an oracle written apart, and
    every coined form is sayable.
  * PN5 -- no taboo form or substring is ever produced, and the witnesses
    prove the filter fires; a rejected candidate is shown only as a digest.
  * PN12 -- the two engines answer the same bytes for every phonology
    operation, and the cases the corpus exists to reach are reached.
  * PN13 -- the first sound is a sound the being can make, never the glottal
    stop, and depends only on its seed and block.
  * PN14 -- PN1's property with the table list compared to the list the
    engine is built from (PN1 is superseded). That list is where the export
    comes from, so the comparison held by construction; PN14 is superseded.
  * PN15 -- PN14's property with the table list read from the table files
    on disk, so a table file that the engines do not export is seen.

Local-only. The modules load through the shared isolation window; PN1, PN12
and PN13 also need the native artefact ``scripts/build_oo_core.sh`` builds.
"""

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_genome_corpus as corpus  # noqa: E402
import _allium_phon_support as S  # noqa: E402
from _allium_window import open_allium  # noqa: E402
from _isolation import REPO  # noqa: E402

TABLES_DIR = REPO / "opti_oignon" / "allium" / "tables"
SCRIPTS = REPO / "scripts"

BUDGET_S = {
    "test_pn1_the_alphabet_and_its_features_are_one_table_in_both_engines": 2.0,
    "test_pn2_the_same_block_gives_the_same_phonology_and_the_floors_always_hold": 2.0,
    "test_pn3_the_phonology_reads_the_language_block_by_offset_and_only_it": 2.0,
    "test_pn4_licit_is_total_agrees_with_an_oracle_and_every_coined_form_is_sayable": 2.0,
    "test_pn5_no_taboo_form_or_substring_is_produced_and_the_witnesses_fire": 2.0,
    "test_pn12_both_engines_answer_every_phonology_request_with_the_same_bytes": 2.0,
    "test_pn13_the_first_sound_is_a_sound_the_being_can_make_keyed_on_seed_and_block": 2.0,
    "test_pn14_the_alphabet_and_its_features_are_one_table_and_the_table_list_is_the_lawfiles": 2.0,
    "test_pn15_the_alphabet_and_its_features_are_one_table_and_the_table_list_is_the_files_on_disk": 2.0,
}


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


@pytest.fixture
def ref():
    loaded, restore = open_allium(native=False)
    try:
        yield S.Engines(loaded, native=False)
    finally:
        restore()


@pytest.fixture
def engines():
    loaded, restore = open_allium(native=True)
    try:
        yield S.Engines(loaded, native=True)
    finally:
        restore()


def _load_script(name, module_name):
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module


def _founder_lex(e, law, seed):
    genome = e.ask({"law": law, "op": "genome_found", "seed": seed.hex()})["genome"]
    return e.ask({"genome": genome, "law": law, "op": "phon_lex"})["lex"], genome


# ---------------------------------------------------------------------------
# PN1
# ---------------------------------------------------------------------------
def test_pn1_the_alphabet_and_its_features_are_one_table_in_both_engines(engines):
    e = engines
    file_value = e.lawfiles.table("phon_v1")
    for law in e.lawfiles.LAWS:
        export = e.both({"law": law, "op": "phon_table"})
        classify = e.wire.unpack_bulk(export["classify"])[1]
        assert len(classify) == 256
        for byte in range(256):
            expected = S.ALPHABET.index(chr(byte)) if chr(byte) in S.ALPHABET else -1
            assert classify[byte] == expected, byte
        for key in ("alphabet", "bias", "features", "first_sound_exclude", "floor_consonants", "floor_vowels",
                    "fold", "form_max", "invent_tries", "lex_box", "potential_min", "sas", "shown_max",
                    "taboo_extra_max", "taboo_max", "taboo_window", "templates", "anchored_max",
                    "anchored_total_max"):
            assert export[key] == file_value[key], key
        features = export["features"]
        for p, row in enumerate(features):
            vowel = p <= 4
            assert (row[0] == 0) == vowel and (row[2] == 7) == vowel and (row[1] == 8) == vowel, p
            assert row[4] == (1, 2, 3, 4, 5, 5, 6, 7)[row[2]], p
            assert row[4] == S.SON[p] and row[2] == S.MANNER[p], p
        assert [p for p, row in enumerate(features) if row[2] == 0] == list(range(5, 13))
        assert [p for p, row in enumerate(features) if row[2] == 3] == [21, 22]
        lang = e.lawfiles.law(law)["lang"]
        assert lang["phon"] == {"name": "phon_v1", "sha256": e.lawfiles.digest(file_value)}
        taboo_file = e.lawfiles.table(lang["taboo"]["name"])
        assert lang["taboo"]["sha256"] == e.lawfiles.digest(taboo_file) == export["taboo"]["sha256"]
        assert export["work"] == 27 + 256
    author = _load_script("allium_author_phon.py", "_pn1_author")
    for name, text in author.author().items():
        assert TABLES_DIR.joinpath(name).read_text(encoding="ascii") == text, name
    assert TABLES_DIR.joinpath("taboo_v1.json").read_text(encoding="ascii") == author.empty_taboo(), \
        "the full law's list is empty while its owner's list is owed"
    mine = e.wire.parse(e.protocol.engine_info())
    theirs = e.wire.parse(bytes(e.native.allium_engine()))
    assert theirs["tables"] == mine["tables"] and theirs["domains"] == mine["domains"]
    assert sorted(mine["tables"]) == ["phon_v1", "sine_q15_v1", "taboo_fixture_v1", "taboo_v1"]


# ---------------------------------------------------------------------------
# PN2
# ---------------------------------------------------------------------------
def _floors_rebuilt(block, potential_min, stops, nasals, cons):
    w = list(block[:27])
    member = [x >= potential_min for x in w]
    floors = []
    for p in (0, 2, 4, block[52], block[53]):
        if not member[p]:
            member[p] = True
            floors.append(p)

    def best(group):
        options = [p for p in group if not member[p]]
        top = max(w[p] for p in options)
        return min(p for p in options if w[p] == top)

    for group in (stops, nasals):
        if not any(member[p] for p in group):
            p = best(group)
            member[p] = True
            floors.append(p)
    while sum(member[p] for p in cons) < 6:
        p = best(cons)
        member[p] = True
        floors.append(p)
    return member, floors


def test_pn2_the_same_block_gives_the_same_phonology_and_the_floors_always_hold(ref):
    phon = ref.phon
    table = ref.table()
    stops, nasals, cons = list(range(5, 13)), [21, 22], list(range(5, 27))
    blocks = [S.lo_block(table), S.hi_block(table)]
    for i in range(9998):
        block = bytearray(S.keyed_block(ref.rng, table, "test.pn2", i))
        shape = i % 5
        if shape == 1:
            for p in range(27):
                block[p] = block[p] % 128
        elif shape == 2:
            for p in stops:
                block[p] = block[p] % 128
        elif shape == 3:
            for p in nasals:
                block[p] = block[p] % 128
        elif shape == 4:
            block[52] = i % 5
            block[53] = (i // 5) % 5
            for p in (block[52], block[53]):
                block[p] = block[p] % 128
        blocks.append(bytes(block))
    no_stop = no_nasal = 0
    for index, block in enumerate(blocks):
        ph = phon.decode(block, table)
        again = phon.decode(block, table)
        assert phon.phonology_value(ph) == phon.phonology_value(again), index
        member, floors = _floors_rebuilt(block, table.potential_min, stops, nasals, cons)
        assert list(ph.member) == member and list(ph.floors) == floors, (index, block.hex())
        for p in (0, 2, 4, block[52], block[53]):
            assert ph.member[p], (index, p)
        assert any(ph.member[p] for p in stops) and any(ph.member[p] for p in nasals)
        assert sum(ph.member[p] for p in cons) >= 6
        for p in range(27):
            if block[p] >= table.potential_min:
                assert ph.member[p]
        no_stop += not any(block[p] >= 128 for p in stops)
        no_nasal += not any(block[p] >= 128 for p in nasals)
    assert len(blocks) == 10000 and no_stop >= 500 and no_nasal >= 500, (no_stop, no_nasal)
    assert list(phon.decode(S.lo_block(table), table).floors) == [0, 2, 4, 5, 21, 6, 7, 8, 9]


# ---------------------------------------------------------------------------
# PN3
# ---------------------------------------------------------------------------
_BYTE_FIELD = {36: "iconicity", 37: "playfulness", 38: "conformity", 39: "openness", 40: "conservatism",
               41: "chattiness", 42: "critical_len", 43: "voice_pitch", 44: "voice_tempo", 45: "onset_max",
               46: "coda_max", 47: "coda_classes", 48: "s_exception", 49: "stress", 50: "harmony", 51: "length",
               52: "reduction_vowel", 53: "epenthetic_vowel", 54: "repair", 55: "head_order", 56: "dominance"}


def test_pn3_the_phonology_reads_the_language_block_by_offset_and_only_it(ref):
    phon = ref.phon
    table = ref.table()
    law = ref.lawfiles.law("fixture")
    lawview = ref.genome.view(law)
    digest = ref.lawfiles.digest(law)
    pool = ref.genome.pool_alleles(ref.lawfiles.founders("fixture"))
    compile_genome = ref.loaded["opti_oignon.allium.ref.organs.compile"].compile_genome
    inventories = {}
    founders = []
    for seed in S.seeds(ref.rng, "test.pn3", 128):
        data = ref.genome.found(seed, lawview, pool)[0]
        tables, _work = compile_genome(data, law, lawview, digest)
        rows = ref.wire.unpack_bulk(tables["reserved"]["lex"]["fields"])[1]
        expected = bytearray(63)
        for i in range(7):
            offset = rows[10 * i]
            expected[offset:offset + 9] = bytes(rows[10 * i + 1:10 * i + 10])
        answer = ref.ask({"genome": data.hex(), "law": "fixture", "op": "phon_lex"})
        assert answer["lex"] == expected.hex() and answer["source"] == "genome"
        inventories[phon.decode(bytes(expected), table).member] = True
        founders.append((data, bytes(expected)))
    assert len(inventories) >= 100, len(inventories)
    for data, expected in founders[:16]:
        pairs, chrom = corpus.split(data)
        chrom = [[corpus.with_field(r, "kcat", 20000) if corpus.locus_of(r) == 7 else r for r in c] for c in chrom]
        other = corpus.frame(chrom, pairs)
        assert other != data
        answer = ref.ask({"genome": other.hex(), "law": "fixture", "op": "phon_lex"})
        assert answer["lex"] == expected.hex(), "a change outside the language loci moved the block"
    for law_name in ("fixture", "v0_1"):
        entries = [e for e in ref.lawfiles.law(law_name)["genome"]["loci"] if e["kind"] == "lex"]
        kinds = {k["name"]: k for k in ref.lawfiles.law(law_name)["genome"]["kinds"]}
        for entry in entries:
            boxes = {f["name"]: (f["lo"], f["hi"]) for f in kinds["lex"]["fields"]}
            for override in entry["box"]:
                boxes[override["field"]] = (override["lo"], override["hi"])
            offset = boxes["offset"][0]
            for k in range(9):
                assert list(boxes[f"d{k}"]) == list(table.lex_box[offset + k]), (law_name, offset + k)
    base = S.lo_block(table)
    allowed = [w > 0 for w in phon.decode(base, table).templates]
    assert allowed == [True, True, False, False, False, False], "the lowest block speaks V and CV only"
    for at in range(57):
        low, high = table.lex_box[at]
        a = bytearray(base)
        b = bytearray(base)
        a[at], b[at] = low, high
        first = phon.phonology_value(phon.decode(bytes(a), table))
        second = phon.phonology_value(phon.decode(bytes(b), table))
        if at < 27:
            assert first["weights"] != second["weights"], at
        elif at < 33:
            # A template's byte moves its weight only when the template is allowed at all.
            assert (first["templates"] != second["templates"]) == allowed[at - 27], at
        elif at < 36:
            assert first["word_len"] != second["word_len"], at
        else:
            assert first[_BYTE_FIELD[at]] != second[_BYTE_FIELD[at]], at
        changed = [key for key in first if first[key] != second[key]]
        assert len(changed) <= 5, (at, changed)


# ---------------------------------------------------------------------------
# PN4
# ---------------------------------------------------------------------------
def _hostile(rng, i):
    stream = rng.Stream(bytes(32), "test.pn4", i)
    kind = stream.below(8)
    if kind == 0:
        return ""
    if kind == 1:
        return "".join(S.ALPHABET[stream.below(27)] for _ in range(13 + stream.below(4)))
    if kind == 2:
        return "ab\n"
    if kind == 3:
        return "".join("aeiou"[stream.below(5)] * (1 + stream.below(3)) for _ in range(1 + stream.below(3)))
    if kind == 4:
        return "a" + "".join("bdkmnlsrwy"[stream.below(10)] for _ in range(1 + stream.below(3)))
    if kind == 5:
        return "".join("Z$ -"[stream.below(4)] if stream.below(6) == 0 else S.ALPHABET[stream.below(27)]
                       for _ in range(1 + stream.below(8)))
    return "".join(S.ALPHABET[stream.below(27)] for _ in range(1 + stream.below(12)))


def test_pn4_licit_is_total_agrees_with_an_oracle_and_every_coined_form_is_sayable(ref):
    phon = ref.phon
    table = ref.table()
    blocks = [S.hi_block(table), S.lo_block(table)] + [S.keyed_block(ref.rng, table, "test.pn4.block", i)
                                                     for i in range(2)]
    phonologies = [phon.decode(b, table) for b in blocks]
    phonologies.append(phon.decode(blocks[0], table, S.mask_of("aiuptksmnl")))
    phonologies.append(phon.decode(blocks[2], table, S.mask_of("aeioubdgzmnrwy")))
    strings = [_hostile(ref.rng, i) for i in range(5000)]
    accepted = 0
    for ph in phonologies:
        genes = ph.genes
        for text in strings:
            reason, syllables, splits = phon.licit(text, ph)
            want = S.oracle(text, ph.member, genes["onset_max"], genes["coda_max"], genes["coda_classes"],
                            genes["s_exception"], genes["length"])
            if want is None:
                assert reason in phon.REASONS, (text, reason)
            else:
                assert reason is None and (syllables, splits) == want, (text, reason, syllables, splits, want)
                accepted += 1
    assert accepted >= 500, accepted
    formed = 0
    for i in range(3000):
        ph = phonologies[i % len(phonologies)]
        seed = (i % 7).to_bytes(32, "big")
        signs = [((i + f) % 3) - 1 for f in range(4)]
        out, _work = phon.invent_case(ph, seed, i, 0, signs, 0, i % 4, [], [], {})
        if out["outcome"] == "word":
            assert phon.licit(out["form"], ph)[0] is None and phon.FORM.fullmatch(out["form"]), out
            formed += 1
    assert formed >= 2900, formed


# ---------------------------------------------------------------------------
# PN5
# ---------------------------------------------------------------------------
def _witness_case(table, concept):
    return {"anchored": [], "coin": 0, "concept": concept, "epoch": 0, "lex": S.hi_block(table).hex(),
            "others": [], "seed": "00" * 32, "signs": [0, 0, 0, 0], "syllables": 0}


def test_pn5_no_taboo_form_or_substring_is_produced_and_the_witnesses_fire(ref):
    phon = ref.phon
    table = ref.table()
    author = _load_script("allium_author_phon.py", "_pn5_author")
    concepts, entries = author.witnesses(ref.lawfiles.table("phon_v1"))
    fixture_taboo = ref.lawfiles.table("taboo_fixture_v1")["entries"]
    assert entries == fixture_taboo and len(concepts) == 3
    pairs = {(length, digest): True for length, digest in fixture_taboo}
    for concept in concepts:
        unfiltered = ref.ask({"cases": [_witness_case(table, concept)], "law": "v0_1", "op": "phon_invent"})
        candidate = unfiltered["out"][0]["form"]
        assert unfiltered["out"][0]["tries"] == []
        filtered = ref.ask({"cases": [_witness_case(table, concept)], "law": "fixture", "op": "phon_invent"})
        out = filtered["out"][0]
        assert out["tries"][:1] == [[S.sha(candidate), "taboo"]], (concept, out)
        assert out.get("form") != candidate
    taboo = phon.taboo_set([tuple(e) for e in fixture_taboo])
    ph_hi = phon.decode(S.hi_block(table), table)
    blocks = [S.hi_block(table), S.lo_block(table)] + [S.keyed_block(ref.rng, table, "test.pn5", i)
                                                     for i in range(4)]
    phs = [phon.decode(b, table) for b in blocks]
    taboo_tries = 0
    for i in range(5000):
        ph = ph_hi if i % 2 == 0 else phs[i % len(phs)]
        out, _work = phon.invent_case(ph, bytes(32), i, 0, [0, 0, 0, 0], 0, 0, [], [], taboo)
        for entry in out["tries"]:
            assert isinstance(entry, list) and len(entry) == 2 and len(entry[0]) == 64, entry
            assert all(c in "0123456789abcdef" for c in entry[0]) and entry[1] in phon.REASONS + (
                "same", "near", "taboo"), entry
            taboo_tries += entry[1] == "taboo"
        if out["outcome"] == "word":
            assert not S.listed(out["form"], pairs), i
    assert taboo_tries >= 3, taboo_tries
    for ph in phs:
        words, _candidates, _work = phon.sas_list(ph, taboo)
        assert not any(S.listed(w, pairs) for w in words)


# ---------------------------------------------------------------------------
# PN12
# ---------------------------------------------------------------------------
_TWO_DEFECTS = (
    {"cases": 3, "law": "nope", "op": "phon_invent"},
    {"cases": [], "extra": 1, "law": "nope", "op": "phon_invent"},
    {"law": "fixture", "lex": "zz", "op": "phon_licit", "forms": 5},
    {"law": "fixture", "lex": ["zz"], "op": "phon_inventory"},
    {"genome": "00", "law": "fixture", "op": "phon_lex", "seed": "00"},
    {"law": "fixture", "op": "phon_lex"},
    {"law": "fixture", "op": "phon_lex", "seed": "0"},
    {"budget": -1, "cases": 1, "law": "fixture", "op": "phon_invent"},
    {"law": "fixture", "lex": "00", "op": "phon_sas", "digests": [1]},
    {"law": "fixture", "op": "phon_taboo", "strings": ["ABC"], "taboo_extra": 1},
    {"cases": [{"lex": "zz"}], "law": "fixture", "op": "phon_first_sound"},
    {"law": 1, "op": "phon_table"},
)


def test_pn12_both_engines_answer_every_phonology_request_with_the_same_bytes(engines):
    e = engines
    table = e.table()
    phon = e.phon
    hi, lo = S.hi_block(table), S.lo_block(table)
    blocks = [hi, lo, S.min_space_block(table)] + [S.keyed_block(e.rng, table, "test.pn12", i) for i in range(5)]
    founders = []
    for seed in S.seeds(e.rng, "test.pn12.founders", 6):
        lex, genome = _founder_lex(e, "fixture", seed)
        e.both({"genome": genome, "law": "fixture", "op": "phon_lex"})
        founders.append(bytes.fromhex(lex))
    for seed in S.seeds(e.rng, "test.pn12.fallback", 6):
        e.both({"law": "v0_1", "op": "phon_lex", "seed": seed.hex()})
    blocks += founders
    e.both({"law": "fixture", "lex": [b.hex() for b in blocks], "op": "phon_inventory"})
    reasons = {}
    carried = (_hostile(e.rng, i) for i in range(500))
    forms = [text for text in carried if all(" " <= ch <= "~" for ch in text)]
    forms += ["aa", "baab", "abba", "aaa", "stap", "tsap", "ab'"]
    for block, mask in ((hi, None), (lo, None), (blocks[3], S.mask_of("aiuptksmnl")), (blocks[4], None)):
        request = {"forms": forms, "law": "fixture", "lex": block.hex(), "op": "phon_licit"}
        if mask is not None:
            request["inventory"] = mask
        for out in e.both(request)["out"]:
            if "reason" in out:
                reasons[out["reason"]] = True
    assert sorted(reasons) == sorted(phon.REASONS), sorted(reasons)
    seen = {}
    cases = []
    for i in range(1500):
        block = blocks[i % len(blocks)]
        case = {"anchored": [], "coin": i % 3, "concept": i, "epoch": i % 5, "lex": block.hex(), "others": [],
                "seed": ((i * 7) % 256).to_bytes(1, "big").hex() * 32, "signs": [((i + f) % 3) - 1 for f in range(4)],
                "syllables": i % 4}
        if i % 11 == 0:
            case["inventory"] = S.mask_of("aiuptksmnl")
        cases.append(case)
    unfiltered = e.ask({"cases": cases[:60], "law": "v0_1", "op": "phon_invent"})["out"]
    for j, out in enumerate(unfiltered):
        if out["outcome"] != "word":
            continue
        form = out["form"]
        if j % 3 == 0:
            cases[j]["anchored"] = [form]
        elif j % 3 == 1:
            cases[j]["anchored"] = [form[:-1] if len(form) > 1 else form + "a"]
        else:
            cases[j]["others"] = [form]
    for start in range(0, len(cases), 500):
        answer = e.both({"cases": cases[start:start + 500], "law": "fixture", "op": "phon_invent",
                         "taboo_extra": [[3, S.sha("tik")], [3, S.sha("ana")]]})
        for out in answer["out"]:
            seen[out["outcome"]] = True
            for _digest, reason in out["tries"]:
                seen[reason] = True
            form = out.get("form", "")
            if len(form) >= 2 and form[0] not in "aeiou" and form[1] not in "aeiou":
                seen["cc onset"] = True
                if form[0] == "s" and S.MANNER[S.ALPHABET.index(form[1])] == 0:
                    seen["s stop"] = True
            if form and form[-1] not in "aeiou":
                seen["coda"] = True
            if any(form[k] == form[k + 1] and form[k] in "aeiou" for k in range(len(form) - 1)):
                seen["long vowel"] = True
    gesture_extra = sorted({(len(t), S.sha(t)): True for t in
                            [a for a in "aiupbtdkm"] + [a + b for a in "aiupbtdkm" for b in "aiupbtdkm"]
                            + [a + b + c for a in "aiupbtdkm" for b in "aiupbtdkm" for c in "aiupbtdkm"]})
    plain = {k: v for k, v in dict(cases[1], lex=lo.hex()).items() if k != "inventory"}
    gesture = e.both({"cases": [plain], "law": "v0_1", "op": "phon_invent",
                      "taboo_extra": [list(p) for p in gesture_extra]})
    assert gesture["out"][0]["outcome"] == "gesture" and len(gesture_extra) == 819
    refusal = e.both({"budget": 50, "cases": cases[:10], "law": "fixture", "op": "phon_invent"})
    assert refusal == {"detail": "phon invent", "refused": "budget"}
    for name in ("word", "same", "near", "taboo", "ocp", "long_vowel", "cc onset", "coda", "long vowel",
                 "s stop"):
        assert seen.get(name), (name, sorted(seen))
    firsts = [{"lex": blocks[i % len(blocks)].hex(), "seed": ("%02x" % (i % 256)) * 32} for i in range(300)]
    e.both({"cases": firsts, "law": "fixture", "op": "phon_first_sound"})
    for block in (hi, S.min_space_block(table)):
        e.both({"digests": [S.sha(str(i)) for i in range(8)], "law": "fixture", "lex": block.hex(), "op": "phon_sas",
                "phrases": [["a"] * 6]})
    shown = [text for text in (_hostile(e.rng, i) for i in range(400)) if phon.SHOWN.fullmatch(text)]
    e.both({"law": "fixture", "op": "phon_taboo", "strings": shown})
    for request in _TWO_DEFECTS:
        e.both(request)
    for op in ("phon_first_sound", "phon_invent", "phon_inventory", "phon_lex", "phon_licit", "phon_sas",
               "phon_table", "phon_taboo"):
        if op != "phon_table":
            assert e.calls.get(op, 0) >= 1, op
    assert e.compared >= 30


# ---------------------------------------------------------------------------
# PN13
# ---------------------------------------------------------------------------
def test_pn13_the_first_sound_is_a_sound_the_being_can_make_keyed_on_seed_and_block(engines):
    e = engines
    table = e.table()
    phon = e.phon
    blocks = [S.hi_block(table), S.lo_block(table), S.min_space_block(table),
              S.keyed_block(e.rng, table, "test.pn13", 0)]
    seeds = S.seeds(e.rng, "test.pn13.seeds", 1000)
    distinct_hi = {}
    for block in blocks:
        ph = phon.decode(block, table)
        cases = [{"lex": block.hex(), "seed": seed.hex()} for seed in seeds]
        first = e.both({"cases": cases, "law": "fixture", "op": "phon_first_sound"})
        e.both({"forms": ["a"], "law": "fixture", "lex": block.hex(), "op": "phon_licit"})
        again = e.both({"cases": cases, "law": "fixture", "op": "phon_first_sound"})
        assert first == again, "a first sound moved between two identical calls"
        for p in first["out"]:
            assert ph.member[p] and p != 12, p
            if block == blocks[0]:
                distinct_hi[p] = True
    assert len(distinct_hi) >= 20, len(distinct_hi)


# ---------------------------------------------------------------------------
# PN14
# ---------------------------------------------------------------------------
def test_pn14_the_alphabet_and_its_features_are_one_table_and_the_table_list_is_the_lawfiles(engines):
    e = engines
    file_value = e.lawfiles.table("phon_v1")
    for law in e.lawfiles.LAWS:
        export = e.both({"law": law, "op": "phon_table"})
        classify = e.wire.unpack_bulk(export["classify"])[1]
        assert len(classify) == 256
        for byte in range(256):
            expected = S.ALPHABET.index(chr(byte)) if chr(byte) in S.ALPHABET else -1
            assert classify[byte] == expected, byte
        for key in ("alphabet", "bias", "features", "first_sound_exclude", "floor_consonants", "floor_vowels",
                    "fold", "form_max", "invent_tries", "lex_box", "potential_min", "sas", "shown_max",
                    "taboo_extra_max", "taboo_max", "taboo_window", "templates", "anchored_max",
                    "anchored_total_max"):
            assert export[key] == file_value[key], key
        features = export["features"]
        for p, row in enumerate(features):
            vowel = p <= 4
            assert (row[0] == 0) == vowel and (row[2] == 7) == vowel and (row[1] == 8) == vowel, p
            assert row[4] == (1, 2, 3, 4, 5, 5, 6, 7)[row[2]], p
            assert row[4] == S.SON[p] and row[2] == S.MANNER[p], p
        assert [p for p, row in enumerate(features) if row[2] == 0] == list(range(5, 13))
        assert [p for p, row in enumerate(features) if row[2] == 3] == [21, 22]
        lang = e.lawfiles.law(law)["lang"]
        assert lang["phon"] == {"name": "phon_v1", "sha256": e.lawfiles.digest(file_value)}
        taboo_file = e.lawfiles.table(lang["taboo"]["name"])
        assert lang["taboo"]["sha256"] == e.lawfiles.digest(taboo_file) == export["taboo"]["sha256"]
        assert export["work"] == 27 + 256
    author = _load_script("allium_author_phon.py", "_pn14_author")
    for name, text in author.author().items():
        assert TABLES_DIR.joinpath(name).read_text(encoding="ascii") == text, name
    assert TABLES_DIR.joinpath("taboo_v1.json").read_text(encoding="ascii") == author.empty_taboo(), \
        "the full law's list is empty while its owner's list is owed"
    mine = e.wire.parse(e.protocol.engine_info())
    theirs = e.wire.parse(bytes(e.native.allium_engine()))
    assert theirs["tables"] == mine["tables"] and theirs["domains"] == mine["domains"]
    assert sorted(mine["tables"]) == sorted(e.lawfiles.TABLES)
    for name in ("phon_v1", "sine_q15_v1", "taboo_fixture_v1", "taboo_v1"):
        assert name in mine["tables"], name



# ---------------------------------------------------------------------------
# PN15
# ---------------------------------------------------------------------------
def test_pn15_the_alphabet_and_its_features_are_one_table_and_the_table_list_is_the_files_on_disk(engines):
    e = engines
    file_value = e.lawfiles.table("phon_v1")
    for law in e.lawfiles.LAWS:
        export = e.both({"law": law, "op": "phon_table"})
        classify = e.wire.unpack_bulk(export["classify"])[1]
        assert len(classify) == 256
        for byte in range(256):
            expected = S.ALPHABET.index(chr(byte)) if chr(byte) in S.ALPHABET else -1
            assert classify[byte] == expected, byte
        for key in ("alphabet", "bias", "features", "first_sound_exclude", "floor_consonants", "floor_vowels",
                    "fold", "form_max", "invent_tries", "lex_box", "potential_min", "sas", "shown_max",
                    "taboo_extra_max", "taboo_max", "taboo_window", "templates", "anchored_max",
                    "anchored_total_max"):
            assert export[key] == file_value[key], key
        features = export["features"]
        for p, row in enumerate(features):
            vowel = p <= 4
            assert (row[0] == 0) == vowel and (row[2] == 7) == vowel and (row[1] == 8) == vowel, p
            assert row[4] == (1, 2, 3, 4, 5, 5, 6, 7)[row[2]], p
            assert row[4] == S.SON[p] and row[2] == S.MANNER[p], p
        assert [p for p, row in enumerate(features) if row[2] == 0] == list(range(5, 13))
        assert [p for p, row in enumerate(features) if row[2] == 3] == [21, 22]
        lang = e.lawfiles.law(law)["lang"]
        assert lang["phon"] == {"name": "phon_v1", "sha256": e.lawfiles.digest(file_value)}
        taboo_file = e.lawfiles.table(lang["taboo"]["name"])
        assert lang["taboo"]["sha256"] == e.lawfiles.digest(taboo_file) == export["taboo"]["sha256"]
        assert export["work"] == 27 + 256
    author = _load_script("allium_author_phon.py", "_pn15_author")
    for name, text in author.author().items():
        assert TABLES_DIR.joinpath(name).read_text(encoding="ascii") == text, name
    assert TABLES_DIR.joinpath("taboo_v1.json").read_text(encoding="ascii") == author.empty_taboo(), \
        "the full law's list is empty while its owner's list is owed"
    mine = e.wire.parse(e.protocol.engine_info())
    theirs = e.wire.parse(bytes(e.native.allium_engine()))
    assert theirs["tables"] == mine["tables"] and theirs["domains"] == mine["domains"]
    on_disk = sorted(path.stem for path in TABLES_DIR.glob("*.json"))
    assert len(on_disk) >= 4 and all(name in on_disk for name in (
        "phon_v1", "sine_q15_v1", "taboo_fixture_v1", "taboo_v1")), ("witness: the glob reads the files", on_disk)
    assert sorted(mine["tables"]) == sorted(p.stem for p in TABLES_DIR.glob("*.json"))
    for name in ("phon_v1", "sine_q15_v1", "taboo_fixture_v1", "taboo_v1"):
        assert name in mine["tables"], name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
