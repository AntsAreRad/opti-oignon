#!/usr/bin/env python3
"""The fact-check core says five things, and "supported" means one of them exactly.

The core decides over evidence it is handed; it reaches no model, no store and
no network. These contracts pin its words and its two floors:

  * VK1 -- the vocabulary is closed and the core reaches no model: five
    verdicts, three bases, the closed codes of every verdict, refusal, flag
    and wrapper; a verdict built with another value, basis or reason, or a
    reason under the wrong verdict, raises. The package imports the standard
    library alone at load (class bodies and every block outside a function
    included), never the registry, a model client, a web client or a store
    anywhere, by statement or by name at run time, the native core only
    inside a function, and no module of the application outside the package
    imports it. No code is a word that claims truth.
  * VK2 -- without frames, "supported" is a whole source sentence restated
    verbatim under fold v1, and nothing else is: a claim inside a longer
    sentence (the tail cut after an initial or an abbreviation included), a
    moved negation, one digit, one added word and a splice of two sentences
    are never supported and never contradicted; an
    anchor-free paraphrase is "not enough evidence" with the reason that no
    judge is wired, every admitted source named as searched.
  * VK3 -- no evidence is "not enough evidence", never contradicted: no
    source, no admissible source and no source valid at the time each have
    their reason; every planted claim of the canary with its evidence
    blanked (an unrelated text, and a salad of the claim's own words) is
    never supported; the sources searched are every id handed in, refused
    ones included, each with its admission and validity (a source past its
    end recorded as such, with the date and how it was derived), in sorted
    order.

Loaded through the shared isolation window (``tests/_factcheck.py``), with the
native core unreachable. Nothing reaches a model, the network or the
maintainer's data.
"""

import ast
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _factcheck as F  # noqa: E402
from _isolation import REPO  # noqa: E402

BUDGET_S = {
    "test_vk1_the_vocabulary_is_closed_and_the_core_reaches_no_model": 2.0,
    "test_vk2_supported_is_a_whole_sentence_restated_verbatim_and_nothing_else": 2.0,
    "test_vk3_no_evidence_is_not_enough_evidence_and_names_what_was_searched": 2.0,
}

PACKAGE_DIR = REPO / "opti_oignon" / "factcheck"

SPEC_VERDICTS = ("supported", "contradicted", "conflicting", "not_enough_evidence", "out_of_scope")
SPEC_BASES = ("deterministic", "model_judged", "none")
SPEC_REASONS = {
    "supported": ("verbatim_sentence", "equivalent_restatement", "judged_entailment"),
    "contradicted": ("value_disjoint", "negation_flip", "quantifier_opposed", "superseded_state",
                     "judged_contradiction"),
    "conflicting": ("sources_disagree", "superseded"),
    "not_enough_evidence": (
        "no_sources", "no_admissible_source", "no_valid_source", "expired", "no_longer_held",
        "source_undated", "context_qualified", "context_incomplete", "attributed",
        "extraction_uncertain", "ingested_before_correction", "cites_retracted", "no_judge",
        "no_passage", "quote_not_found", "anchor_unmatched", "anchor_differs", "anchor_ambiguous",
        "anchor_incomparable", "anchor_binding_unverified", "role_order", "negation_differs",
        "negation_ambiguous", "quantifier_differs", "direction_differs", "evidence_hedged",
        "strength_exceeds", "conditional", "population_narrower", "judge_uncalibrated",
        "calibration_stale", "judge_abstained", "judge_failed", "judge_refused",
        "judged_contradiction_unanchored", "part_adds_anchor", "parts_incomplete",
        "parts_unfaithful",
    ),
    "out_of_scope": ("empty", "too_long", "question", "code", "heading", "table_row", "image",
                     "markup_unparsed", "instruction", "not_standalone", "subject_unresolved"),
}
SPEC_REFUSALS = ("author_model", "author_unknown", "author_model_quoted", "snippet", "no_consent",
                 "retracted", "successor_missing", "chunk_changed", "chunk_not_nfc",
                 "chunk_too_large", "over_budget")
SPEC_FLAGS = ("validity_unknown", "supersession_coarse", "truncated")
SPEC_WRAPPERS = ("hedged", "advice", "quoted")

TRUTH_WORDS = {"true", "false", "verified", "verify", "correct", "fact", "facts", "accurate",
               "proven", "confirmed", "vrai", "vraie", "faux", "fausse", "verifie", "verifiee",
               "correcte", "exact", "exacte", "confirme", "confirmee", "fait"}

FORBIDDEN_MODULES = ("inference_backend", "registry_clients", "ollama", "openai", "anthropic",
                     "llama_cpp", "requests", "httpx", "aiohttp", "urllib.request", "web_search",
                     "rag_store", "db_utils", "ledger_store", "vector_store", "conversations",
                     "notes_store", "memory", "untrusted_context")


@pytest.fixture
def fc():
    ns, restore = F.load()
    try:
        yield ns
    finally:
        restore()


def _resolve(module_name, node):
    """The dotted name an import statement reaches, relative imports resolved."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base = node.module or ""
    if node.level:
        parts = module_name.split(".")
        anchor = ".".join(parts[: len(parts) - node.level])
        base = f"{anchor}.{base}" if base else anchor
        if not node.module:
            return [f"{base}.{alias.name}" for alias in node.names]
    return [base]


def _load_time(tree):
    """Every node executed when the module loads: the tree without the bodies of functions and lambdas.

    Class bodies, loops, ``with``, ``try`` (and ``try*``), ``match`` and every
    other block run at load, so they are walked; a function body runs only
    when called, so it is not.
    """
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                stack.append(child)


def _dynamic_import(node):
    """The constant name an ``import_module(...)`` or ``__import__(...)`` call reaches, or None."""
    if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
        callee = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if callee in ("import_module", "__import__") and isinstance(node.args[0].value, str):
            return node.args[0].value
    return None


def _top_level_imports(tree):
    """Imports executed when the module loads, by statement: every block outside a function body."""
    return [node for node in _load_time(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]


def _modules(tree_dir):
    """Every module of the tree, the data directory left unopened."""
    found = []
    for folder, dirs, files in os.walk(tree_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not (
            Path(folder) == Path(tree_dir) and d == "data"))
        found.extend(Path(folder) / f for f in sorted(files) if f.endswith(".py"))
    return found


def census(package_dir, tree_dir):
    """What the package imports and who imports it, read from the syntax.

    Returns four lists: non-standard-library imports at module level; imports
    anywhere in the package of a model, web, registry or store module; the
    native core imported at module level; modules outside the package that
    import it. Also returns the counts scanned, so a census of nothing is seen.
    """
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    module_level, forbidden, native, importers = [], [], [], []
    scanned = 0
    for path in sorted(Path(package_dir).glob("*.py")):
        scanned += 1
        name = f"opti_oignon.factcheck.{path.stem}"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        loaded = [target for node in _top_level_imports(tree) for target in _resolve(name, node)]
        loaded += [target for node in _load_time(tree) if (target := _dynamic_import(node))]
        for target in loaded:
            own = target.startswith("opti_oignon.factcheck")
            if target.startswith("opti_oignon.native"):
                native.append((path.name, target))
            elif not own and target.split(".")[0] not in stdlib:
                module_level.append((path.name, target))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                targets = _resolve(name, node)
            elif (target := _dynamic_import(node)):
                targets = [target]
            for target in targets:
                segments = target.split(".")
                for bad in FORBIDDEN_MODULES:
                    pieces = bad.split(".")
                    if any(segments[i:i + len(pieces)] == pieces for i in range(len(segments))):
                        forbidden.append((path.name, target))
    outside = 0
    for path in _modules(tree_dir):
        if Path(package_dir) in path.parents:
            continue
        outside += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        if "factcheck" not in text:
            continue
        rel = path.relative_to(tree_dir).with_suffix("")
        name = ".".join(("opti_oignon", *rel.parts))
        if path.name == "__init__.py":
            name = name.rsplit(".", 1)[0] + ".__init__"
        tree = ast.parse(text)
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                targets = _resolve(name, node)
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                callee = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if callee in ("import_module", "__import__"):
                    targets = [str(node.args[0].value)]
            if any(t == "opti_oignon.factcheck" or t.startswith("opti_oignon.factcheck.") for t in targets):
                importers.append(str(rel))
    return module_level, forbidden, native, importers, scanned, outside


def test_vk1_the_vocabulary_is_closed_and_the_core_reaches_no_model(fc, tmp_path):
    vocab = fc.vocabulary

    # c3 -- no verdict value, basis or code is a word that claims truth.
    codes = set(vocab.all_codes())
    assert codes >= set(SPEC_VERDICTS) | set(SPEC_BASES), "the census of codes misses the verdicts"
    assert len(codes) >= 80, f"the census of codes found only {len(codes)} codes"
    words = {part for code in codes for part in code.split("_")}
    assert not (words & TRUTH_WORDS), sorted(words & TRUTH_WORDS)

    # c1 -- exactly five verdicts, three bases, the closed codes per verdict.
    assert tuple(vocab.VERDICTS) == SPEC_VERDICTS
    assert tuple(vocab.BASES) == SPEC_BASES
    assert {k: tuple(v) for k, v in vocab.REASONS.items()} == SPEC_REASONS
    assert tuple(vocab.REFUSALS) == SPEC_REFUSALS
    assert tuple(vocab.FLAGS) == SPEC_FLAGS
    assert tuple(vocab.WRAPPERS) == SPEC_WRAPPERS
    good = vocab.Verdict("supported", "deterministic", ("verbatim_sentence",))
    assert good.value == "supported" and good.leading == "verbatim_sentence"
    refused = [
        ("likely", "deterministic", ("verbatim_sentence",)),
        ("supported", "hunch", ("verbatim_sentence",)),
        ("supported", "deterministic", ("plausible",)),
        ("supported", "deterministic", ("no_judge",)),
        ("not_enough_evidence", "none", ("verbatim_sentence",)),
        ("conflicting", "deterministic", ("no_longer_held",)),
        ("out_of_scope", "none", ("superseded",)),
        ("not_enough_evidence", "none", ()),
        ("not_enough_evidence", "deterministic", ("no_judge",)),
    ]
    for value, basis, reasons in refused:
        with pytest.raises(ValueError):
            vocab.Verdict(value, basis, reasons)

    # c2 -- the census: standard library only at module level, nothing that
    # reaches a model, the web or a store, the native core only in a function,
    # and nothing outside the package imports it.
    tree_dir = REPO / "opti_oignon"
    module_level, forbidden, native, importers, scanned, outside = census(PACKAGE_DIR, tree_dir)
    assert scanned >= 10 and outside >= 100, (scanned, outside)
    assert module_level == [], module_level
    assert forbidden == [], forbidden
    assert native == [], native
    assert importers == [], importers

    # Witness: a planted import and a planted importer are found by the same helper.
    planted_pkg = tmp_path / "opti_oignon" / "factcheck"
    shutil.copytree(PACKAGE_DIR, planted_pkg, ignore=shutil.ignore_patterns("__pycache__"))
    (planted_pkg / "planted.py").write_text(
        "from .. import registry_clients\nimport yaml\nfrom ..native import load\n", encoding="utf-8")
    (tmp_path / "opti_oignon" / "planted_route.py").write_text(
        "from opti_oignon.factcheck import decide\n", encoding="utf-8")
    (planted_pkg / "planted_late.py").write_text(
        "import importlib\n\n\nclass _Late:\n    import yaml as _yaml\n\n\n"
        "def _reach():\n    return importlib.import_module(\"opti_oignon.inference_backend\")\n",
        encoding="utf-8")
    ml, fb, nt, im, _, _ = census(planted_pkg, tmp_path / "opti_oignon")
    assert ("planted.py", "yaml") in ml, ml
    assert ("planted.py", "opti_oignon.registry_clients") in fb, fb
    assert ("planted.py", "opti_oignon.native") in nt, nt
    assert im == ["planted_route"], im
    # A class body runs at load; a model reached by name at run time is still reached.
    assert ("planted_late.py", "yaml") in ml, ml
    assert ("planted_late.py", "opti_oignon.inference_backend") in fb, fb
    assert ("planted_late.py", "importlib") not in ml, ml


# Fold classes a verbatim restatement is read through, each met once.
def _fold_fixtures():
    rs, nb, sh, lq, rq, en, el, fi = (chr(0x2019), chr(0xA0), chr(0xAD), chr(0x201C), chr(0x201D),
                                      chr(0x2013), chr(0x2026), chr(0xFB01))
    return [
        ("apostrophe", f"Brittany{rs}s oldest lighthouse was first lit in 1836.",
         "Brittany's oldest lighthouse was first lit in 1836."),
        ("whitespace", f"Mount Etna erupted{nb}again in the spring of 2021.",
         "Mount Etna erupted again in the spring of 2021."),
        ("invisible", f"Lake Baikal is the deep{sh}est lake on Earth.",
         "Lake Baikal is the deepest lake on Earth."),
        ("double_quote", f"Verne called the submarine {lq}Nautilus{rq} in his novel.",
         'Verne called the submarine "Nautilus" in his novel.'),
        ("dash", f"The Paris{en}Rouen railway opened in 1843.",
         "The Paris-Rouen railway opened in 1843."),
        ("ellipsis", f"Hugo wrote the opening line in exile{el} and never changed it.",
         "Hugo wrote the opening line in exile... and never changed it."),
        ("ligature", f"Fernand Pouillon designed the {fi}rst tower of the square.",
         "Fernand Pouillon designed the first tower of the square."),
        ("wrapper_first_letter", "The Loire is the longest river in France.",
         "I believe the Loire is the longest river in France."),
        ("final_punctuation", "The Loire is the longest river in France.",
         "The Loire is the longest river in France"),
    ]


def test_vk2_supported_is_a_whole_sentence_restated_verbatim_and_nothing_else(fc):
    cfg = F.config(fc)

    # c1 -- a claim equal to a whole sentence under fold v1: supported,
    # deterministic, verbatim_sentence; the span is exactly that sentence.
    met = set()
    for fold_class, sentence, claim in _fold_fixtures():
        text = f"Travel notes from the spring. {sentence} More to follow soon."
        one = F.item(fc, "library:trip", text)
        verdict = F.run(fc, claim, [one], cfg=cfg)
        assert (verdict.value, verdict.basis, verdict.reasons) == (
            "supported", "deterministic", ("verbatim_sentence",)), (fold_class, verdict.reasons)
        spans = [s for s in verdict.record["spans"] if s["role"] == "support"]
        assert len(spans) == 1, (fold_class, spans)
        start, end = spans[0]["start"], spans[0]["end"]
        assert text[start:end] == sentence, (fold_class, text[start:end])
        met.add(fold_class)
    assert met == {f[0] for f in _fold_fixtures()}

    # c2 -- never supported and never contradicted.
    moon = "The Moon is 384,400 km away."
    bridge = "The Pont Neuf was completed in 1607 under Henri IV."
    never = [
        ("inside a longer sentence", "It is false that the Moon is 384,400 km away.", moon),
        ("negation moved", "Version 2 is not supported, version 3 is.",
         "Version 2 is supported, version 3 is not."),
        ("one digit", bridge, "The Pont Neuf was completed in 1608 under Henri IV."),
        ("one word added", bridge, "The Pont Neuf was finally completed in 1607 under Henri IV."),
        ("spliced", f"{bridge} It crosses the Seine at the western tip of the Ile de la Cite.",
         "The Pont Neuf was completed in 1607 under Henri IV and it crosses the Seine at the "
         "western tip of the Ile de la Cite."),
    ]
    for label, text, claim in never:
        verdict = F.run(fc, claim, [F.item(fc, "library:paris", text)], cfg=cfg)
        assert verdict.value == "not_enough_evidence", (label, verdict.value, verdict.reasons)
    # The tail of a sentence cut after an initial or an abbreviation is not a
    # sentence: the whole one negates or reports it. After an initial or an
    # abbreviation of the list the sentence stays whole, so nothing restates
    # the claim; after a period the splitter cannot place, the tail is read as
    # a cut context.
    e_acute = chr(0xE9)
    tails = [
        ("an initial", "Some believe that George W. Bush won the popular vote in 2000.",
         "Bush won the popular vote in 2000.", "en", "no_judge"),
        ("two initials", "It is often said that J. K. Rowling was born in Edinburgh.",
         "Rowling was born in Edinburgh.", "en", "no_judge"),
        ("a dotted abbreviation", "Critics say the U.S. Senate approved the treaty in 1920.",
         "Senate approved the treaty in 1920.", "en", "no_judge"),
        ("a French title", f"Rien ne prouve que le Pr. Raoult a gu{e_acute}ri 100 patients.",
         f"Raoult a gu{e_acute}ri 100 patients.", "fr", "no_judge"),
        ("a title of four letters", "Critics say Prof. Higgs predicted the boson in 1964.",
         "Higgs predicted the boson in 1964.", "en", "no_judge"),
        ("an abbreviation the list does not know", "Critics say Pvt. Manning leaked the files in 2010.",
         "Manning leaked the files in 2010.", "en", "context_incomplete"),
    ]
    for label, text, claim, lang, reason in tails:
        verdict = F.run(fc, fc.scope.Claim(text=claim, lang=lang), [F.item(fc, "library:tail", text, lang=lang)],
                        cfg=cfg)
        assert verdict.value == "not_enough_evidence", (label, verdict.value, verdict.reasons)
        assert reason in verdict.reasons, (label, verdict.reasons)

    # c3 -- an anchor-free paraphrase: not enough evidence, no_judge, every
    # admitted item named as searched.
    items = [
        F.item(fc, "library:alt", "Water boils at a lower temperature at high altitude."),
        F.item(fc, "note:alt", "Water boils at a lower temperature at high altitude.",
               kind="note", author="user"),
        F.item(fc, "library:model", "Water boils at a lower temperature at high altitude.",
               author="model"),
    ]
    verdict = F.run(fc, "At high altitude, water boils at a lower temperature.", items, cfg=cfg)
    assert verdict.value == "not_enough_evidence" and verdict.leading == "no_judge", verdict.reasons
    searched = {e["source_id"]: e for e in verdict.record["sources_searched"]}
    assert searched["library:alt"]["admission"] == "admitted"
    assert searched["note:alt"]["admission"] == "admitted"
    assert searched["library:model"]["admission"] == "author_model"
    assert "library:alt" in verdict.text and "note:alt" in verdict.text, verdict.text


def _salad(claim_text):
    words = [w.strip(".,;:!?") for w in claim_text.split()]
    return "Unordered words: " + " ".join(sorted(words, key=str.lower, reverse=True)) + "."


def test_vk3_no_evidence_is_not_enough_evidence_and_names_what_was_searched(fc):
    cfg = F.config(fc)
    claim = "Lake Baikal is the deepest lake on Earth."

    # c1 -- no items; every item refused; admitted but none valid at t.
    verdict = F.run(fc, claim, [], cfg=cfg)
    assert (verdict.value, verdict.leading) == ("not_enough_evidence", "no_sources"), verdict.reasons
    refused = [F.item(fc, "library:m", claim, author="model"),
               F.item(fc, "web:s", claim, kind="snippet")]
    verdict = F.run(fc, claim, refused, cfg=cfg)
    assert (verdict.value, verdict.leading) == ("not_enough_evidence", "no_admissible_source")
    stale = [F.item(fc, "library:later", "Lake Baikal holds a fifth of the fresh water on Earth.",
                    valid_from="2026-10-01"),
             F.item(fc, "library:gone", "Lake Baikal holds a fifth of the fresh water on Earth.",
                    valid_until="2026-01-01")]
    verdict = F.run(fc, claim, stale, cfg=cfg)
    assert (verdict.value, verdict.leading) == ("not_enough_evidence", "no_valid_source"), verdict.reasons

    # c2 -- every canary claim with its evidence blanked: never supported,
    # never contradicted, never conflicting.
    rows = fc.canary.rows()
    assert len(rows) >= 40, len(rows)
    in_scope_seen = 0
    for row in rows:
        claim_text = row.claim if isinstance(row.claim, str) else row.claim.text
        for filler in ("Gardening tools should be cleaned and oiled before winter storage.",
                       _salad(claim_text)):
            verdict = fc.decide.check(row.claim, F.blank(fc, row.items, filler), as_of=row.as_of,
                                      read_on=row.read_on, config=cfg)
            assert verdict.value in ("not_enough_evidence", "out_of_scope"), (row.name, verdict.value)
            if verdict.value == "not_enough_evidence":
                in_scope_seen += 1
    assert in_scope_seen >= 40, in_scope_seen

    # c3 -- sources_searched is every id handed in, refused ones included,
    # each with its admission and validity, in sorted order.
    handed = [
        F.item(fc, "note:b", claim, kind="note", author="user"),
        F.item(fc, "library:a", claim),
        F.item(fc, "web:c", claim, kind="snippet"),
        F.item(fc, "library:d", claim, author="unknown"),
        F.item(fc, "library:e", claim, consent=None),
    ]
    verdict = F.run(fc, claim, handed, cfg=cfg)
    entries = verdict.record["sources_searched"]
    assert [e["source_id"] for e in entries] == sorted(i.source_id for i in handed)
    admissions = {e["source_id"]: e["admission"] for e in entries}
    assert admissions == {"library:a": "admitted", "library:d": "author_unknown",
                          "library:e": "no_consent", "note:b": "admitted", "web:c": "snippet"}
    for entry in entries:
        assert set(entry["validity"]) >= {"start", "end", "valid_at"}, entry
    # A source past its end is recorded not valid, with its end and how it was derived.
    ended = F.item(fc, "library:ended", claim, valid_from="2025-01-01", valid_until="2026-01-01")
    entry = F.run(fc, claim, [ended], cfg=cfg).record["sources_searched"][0]
    assert entry["validity"]["valid_at"] is False, entry
    assert entry["validity"]["end"] == {"date": "2026-01-01", "from": "valid_until"}, entry
    assert entry["validity"]["start"] == {"date": "2025-01-01", "from": "valid_from"}, entry
