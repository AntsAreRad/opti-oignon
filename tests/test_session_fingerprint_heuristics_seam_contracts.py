#!/usr/bin/env python3
"""Context-fingerprint contracts: pure heuristics, the storage seam, and
fail-secure reads (``opti_oignon/session_fingerprint.py``).

The module under contract condenses an ongoing coding run into a compact,
token-bounded blob for prompt injection: ten dimensions, no model calls,
purely heuristic. A companion suite already pins its bounded collections,
its lean serialized output and the parameterised persistence round-trip.
What THIS suite pins is the surface underneath those guarantees:

  * the pure classifiers and extractors (task type, bug category, stack,
    identifier terms, TF-IDF, import graph, momentum) are deterministic and
    total over hostile or empty input -- exact values frozen from the
    reference environment;
  * the only persistent dimension reaches storage exclusively through the
    encrypted-connection helper, and that seam is EXERCISED at import time
    (the module-level manager opens its store eagerly), proven by a seeded
    connector with a call counter rather than assumed;
  * every storage read degrades to a safe default -- zero ratios, an empty
    mapping, a zero count -- when the connection helper refuses, and a
    refused connection at import does not raise: the store init is
    fail-secure, so the availability latch stays up (pinned as observed;
    the latch only trips on failures outside the store's own guard);
  * configuration loading survives a broken or missing file and lands on
    the documented defaults;
  * an abstract-syntax census freezes the storage statement inventory:
    every SQL site passes a constant string, exactly one carries bound
    parameters, and no site builds its statement from an f-string or
    ``format`` call; the single content-hash call is flagged as
    non-security use.

The load is NOT import-pure: ``fingerprint_config`` and
``fingerprint_manager`` construct at import and the manager opens its
database as it is built. The window therefore seeds a stand-in
``db_utils`` whose ``safe_connect`` redirects every requested path to a
throwaway file (or refuses outright, for the fail-secure reads). Without
the seed the real connector would reach the repository database that
shares the store's default file name, so the seed is load-bearing, not
decorative. Isolation goes through the shared window with ``ollama``
proven unreachable. The module itself is left byte-identical by this
suite.
"""

import ast
import hashlib
import os
import sqlite3
import tempfile
import types
import uuid

from _isolation import isolate, source

_FP = "opti_oignon.session_fingerprint"

# Reference values, read off the module and its shipped configuration in
# the reference environment and frozen here so a change to any of them must
# update this file. The shipped configuration file and the in-module
# defaults agree on every value below.
_CFG_SQLITE_PATH = "fingerprint.db"
_CFG_FORMAT = "yaml"
_CFG_TFIDF_INTERVAL = 5
_CFG_MAX_ANCHORS = 5
_CFG_MAX_HOT_FILES = 5
_CFG_MAX_DOMAIN_TERMS = 10
_CFG_MAX_DEP_CLUSTERS = 5
_CFG_MAX_BUG_HISTORY = 20

# Statement census snapshot: total SQL sites, how many carry bound
# parameters, and the content-hash call count. All frozen; f-string or
# ``format``-built SQL must stay at zero.
_SQL_SITES = 6
_SQL_PARAMETERIZED = 1
_MD5_SITES = 1


def _redirecting_db_utils(tmpdir, counter, requested):
    """A stand-in ``db_utils`` whose ``safe_connect`` never leaves ``tmpdir``.

    Equal requested paths map to one file (deterministic), distinct paths
    to distinct files. Every call bumps ``counter`` and journals the
    requested path in ``requested`` so a test can assert the connector was
    reached, and with what.
    """
    stub = types.ModuleType("opti_oignon.db_utils")

    def _safe_connect(db_path, **kwargs):
        counter["n"] += 1
        requested.append(str(db_path))
        key = hashlib.md5(
            str(db_path).encode(), usedforsecurity=False
        ).hexdigest()
        return sqlite3.connect(os.path.join(tmpdir, key + ".db"))

    stub.safe_connect = _safe_connect
    return stub


def _refusing_db_utils(counter):
    """A stand-in ``db_utils`` whose ``safe_connect`` always refuses."""
    stub = types.ModuleType("opti_oignon.db_utils")

    def _safe_connect(db_path, **kwargs):
        counter["n"] += 1
        raise sqlite3.OperationalError("connection refused by the seam")

    stub.safe_connect = _safe_connect
    return stub


def _load():
    """Open the shared window and load the fingerprint module from source.

    Returns ``(module, restore, counter, requested, tmpdir)``.
    ``counter['n']`` is the number of ``safe_connect`` calls made through
    the seeded connector; ``requested`` the paths it was asked to open.
    """
    tmpdir = tempfile.mkdtemp(prefix="fp_seam_")
    counter = {"n": 0}
    requested = []
    stub = _redirecting_db_utils(tmpdir, counter, requested)
    loaded, restore = isolate(
        targets={_FP: source("session_fingerprint.py")},
        blocked=("ollama",),
        seeded={"opti_oignon.db_utils": stub},
    )
    return loaded[_FP], restore, counter, requested, tmpdir


def _load_refusing():
    """Open the shared window with a connector that refuses every open."""
    counter = {"n": 0}
    stub = _refusing_db_utils(counter)
    loaded, restore = isolate(
        targets={_FP: source("session_fingerprint.py")},
        blocked=("ollama",),
        seeded={"opti_oignon.db_utils": stub},
    )
    return loaded[_FP], restore, counter


def _store(mod, tmpdir):
    """A preferences store on a database file no other store shares."""
    return mod.UserPreferencesStore(
        db_path=os.path.join(tmpdir, uuid.uuid4().hex + ".db")
    )


# --- Load posture: the seam is exercised, the singletons land ---------------


def test_b1_the_import_reaches_storage_through_the_seeded_seam():
    """b1: loading the module exercises the seeded encrypted-connection
    helper -- the module-level manager opens its store eagerly -- and the
    path it asks for is the configured store file name."""
    mod, restore, counter, requested, _tmp = _load()
    try:
        assert counter["n"] > 0, (
            "the seeded connector must be reached at import; the store "
            "init is eager, not lazy"
        )
        assert _CFG_SQLITE_PATH in requested, (
            f"the store must open its configured path, got {requested!r}"
        )
    finally:
        restore()


def test_b2_availability_latch_and_singletons_present():
    """b2: after a clean load the availability flag is up and both
    module-level singletons exist with their documented types."""
    mod, restore, _c, _r, _t = _load()
    try:
        assert mod.FINGERPRINT_AVAILABLE is True
        assert mod.fingerprint_manager is not None
        assert type(mod.fingerprint_manager).__name__ == "FingerprintManager"
        assert type(mod.fingerprint_config).__name__ == "FingerprintConfig"
    finally:
        restore()


def test_b3_loaded_configuration_matches_the_frozen_reference():
    """b3: the configuration the module actually loads in the reference
    tree carries exactly the frozen values, including the three dimension
    weights that are deliberately below full weight."""
    mod, restore, _c, _r, _t = _load()
    try:
        cfg = mod.fingerprint_config
        assert cfg.enabled is True
        assert cfg.sqlite_path == _CFG_SQLITE_PATH
        assert cfg.serialization_format == _CFG_FORMAT
        assert cfg.tfidf_refresh_interval == _CFG_TFIDF_INTERVAL
        assert cfg.max_anchors == _CFG_MAX_ANCHORS
        assert cfg.max_hot_files == _CFG_MAX_HOT_FILES
        assert cfg.max_domain_terms == _CFG_MAX_DOMAIN_TERMS
        assert cfg.max_dep_clusters == _CFG_MAX_DEP_CLUSTERS
        assert cfg.max_bug_history == _CFG_MAX_BUG_HISTORY
        weights = cfg.dimension_weights
        assert len(weights) == 10
        assert weights["domain_terms"] == 0.8
        assert weights["dep_clusters"] == 0.6
        assert weights["user_preferences"] == 0.9
    finally:
        restore()


# --- Task classification is total and exact ---------------------------------


def test_b4_empty_task_text_lands_on_the_unknown_floor():
    """b4: empty input yields the exact unknown/simple/zero-confidence
    floor rather than raising or guessing."""
    mod, restore, _c, _r, _t = _load()
    try:
        assert mod.TaskType.UNKNOWN == "unknown"
        assert mod.classify_task("") == {
            "type": mod.TaskType.UNKNOWN,
            "complexity": "simple",
            "confidence": 0.0,
        }
    finally:
        restore()


def test_b5_keyword_families_map_to_their_documented_types():
    """b5: representative keywords from three families classify to their
    documented task types with positive confidence."""
    mod, restore, _c, _r, _t = _load()
    try:
        bug = mod.classify_task("fix the crash, another bug slipped in")
        assert bug["type"] == "bug_fix" and bug["confidence"] > 0
        ref = mod.classify_task("refactor and simplify the helpers")
        assert ref["type"] == "refactor" and ref["confidence"] > 0
        tst = mod.classify_task("write a pytest spec with coverage")
        assert tst["type"] == "test" and tst["confidence"] > 0
    finally:
        restore()


def test_b6_text_without_any_keyword_stays_unknown():
    """b6: text matching no keyword family classifies as unknown with zero
    confidence -- the classifier does not invent a type."""
    mod, restore, _c, _r, _t = _load()
    try:
        got = mod.classify_task("zzz qqq wibble")
        assert got["type"] == "unknown"
        assert got["confidence"] == 0.0
    finally:
        restore()


def test_b7_complexity_ladder_over_word_count():
    """b7: word count walks the four complexity bands."""
    mod, restore, _c, _r, _t = _load()
    try:
        def _band(n):
            return mod.classify_task(" ".join(["w"] * n))["complexity"]

        assert _band(2) == "simple"
        assert _band(31) == "moderate"
        assert _band(100) == "complex"
        assert _band(161) == "very_complex"
    finally:
        restore()


def test_b8_complexity_boundaries_sit_exactly_where_documented():
    """b8: the band boundaries are exact -- the threshold word count
    itself already belongs to the next band."""
    mod, restore, _c, _r, _t = _load()
    try:
        def _band(n):
            return mod.classify_task(" ".join(["w"] * n))["complexity"]

        assert _band(19) == "simple"
        assert _band(20) == "moderate"
        assert _band(59) == "moderate"
        assert _band(60) == "complex"
        assert _band(149) == "complex"
        assert _band(150) == "very_complex"
    finally:
        restore()


def test_b9_confidence_is_the_winning_share_of_keyword_hits():
    """b9: with two bug keywords against one refactor keyword the type is
    the majority family and the confidence is its exact share."""
    mod, restore, _c, _r, _t = _load()
    try:
        got = mod.classify_task("fix the bug and refactor")
        assert got["type"] == "bug_fix"
        assert got["confidence"] == 0.667
    finally:
        restore()


# --- Bug classification is total, mapped, and order-stable ------------------


def test_b10_empty_or_unmatched_error_text_stays_unknown():
    """b10: empty input and text matching no failure family both land on
    the unknown category rather than raising or guessing."""
    mod, restore, _c, _r, _t = _load()
    try:
        assert mod.classify_bug("") == "unknown"
        assert mod.classify_bug("wibble wobble") == "unknown"
    finally:
        restore()


def test_b11_failure_families_map_to_their_documented_categories():
    """b11: representative failure texts classify into their documented
    categories."""
    mod, restore, _c, _r, _t = _load()
    try:
        assert mod.classify_bug("AssertionError: boom") == "assertion"
        assert (
            mod.classify_bug("ModuleNotFoundError: No module named foo")
            == "import"
        )
        assert mod.classify_bug("TypeError: expected int, got str") == "type"
        assert mod.classify_bug("KeyError: 'k'") == "index"
        assert mod.classify_bug("open failed: permission denied") == "io"
        assert mod.classify_bug("operation timed out") == "timeout"
    finally:
        restore()


def test_b12_first_matching_family_wins_when_texts_overlap():
    """b12: a text carrying markers of two families classifies as the
    family listed first, so the category is order-stable."""
    mod, restore, _c, _r, _t = _load()
    try:
        assert mod.classify_bug("assertionerror: no module named foo") == (
            "assertion"
        )
        assert mod.classify_bug("no module named foo") == "import"
    finally:
        restore()


# --- Stack detection is exact over extensions and imports -------------------


def test_b13_primary_language_follows_extension_frequency():
    """b13: extension counting picks the most frequent language as
    primary and reports exact counts."""
    mod, restore, _c, _r, _t = _load()
    try:
        got = mod.detect_stack(["a.py", "b.py", "c.js"])
        assert got["languages"] == {"python": 2, "javascript": 1}
        assert got["primary"] == "python"
    finally:
        restore()


def test_b14_no_input_lands_on_the_exact_unknown_shape():
    """b14: with nothing to look at the detector returns the exact empty
    shape with an unknown primary."""
    mod, restore, _c, _r, _t = _load()
    try:
        assert mod.detect_stack([]) == {
            "languages": {},
            "primary": "unknown",
            "frameworks": [],
        }
    finally:
        restore()


def test_b15_import_lines_count_and_frameworks_surface():
    """b15: content-based detection counts import lines on top of the
    extension and surfaces the framework named in an import."""
    mod, restore, _c, _r, _t = _load()
    try:
        got = mod.detect_stack(
            ["x.py"], {"x.py": "from fastapi import FastAPI\nimport os\n"}
        )
        assert got == {
            "languages": {"python": 3},
            "primary": "python",
            "frameworks": ["fastapi"],
        }
    finally:
        restore()


# --- Identifier terms and TF-IDF --------------------------------------------


def test_b16_terms_split_and_filter_exactly():
    """b16: camelCase and snake_case identifiers split into lower-case
    parts, stop words and short fragments are dropped, and nothing else
    slips through."""
    mod, restore, _c, _r, _t = _load()
    try:
        code = (
            "def computeScore():\n    return 1\n"
            "class DataHandler:\n    pass\n"
            "def foo_bar():\n    pass\n"
            "def is_ok():\n    pass\n"
            "def init_self_test():\n    pass\n"
        )
        terms = mod.extract_terms(code)
        assert terms == ["compute", "score", "data", "handler", "foo", "bar"]
        for dropped in ("is", "ok", "init", "self", "test"):
            assert dropped not in terms
    finally:
        restore()


def test_b17_tfidf_is_total_and_sorted_descending():
    """b17: an empty corpus scores to an empty list; a real corpus comes
    back strictly ordered by descending score with positive scores."""
    mod, restore, _c, _r, _t = _load()
    try:
        assert mod.compute_tfidf([]) == []
        scored = mod.compute_tfidf(
            [["alpha", "alpha", "alpha", "beta"], ["gamma"]], max_terms=10
        )
        assert scored, "a non-empty corpus must yield scored terms"
        assert all(s > 0 for _, s in scored)
        assert all(
            scored[i][1] >= scored[i + 1][1] for i in range(len(scored) - 1)
        ), f"scores must be non-increasing, got {scored!r}"
    finally:
        restore()


def test_b18_tfidf_honours_the_term_cap():
    """b18: the configured cap bounds the returned term list."""
    mod, restore, _c, _r, _t = _load()
    try:
        docs = [[f"term{i}"] * 2 for i in range(8)]
        assert len(mod.compute_tfidf(docs, max_terms=3)) == 3
    finally:
        restore()


# --- Import graph and clusters ----------------------------------------------


def test_b19_import_graph_takes_roots_and_excludes_self():
    """b19: a dotted import contributes its root module only, and a module
    importing itself contributes no self edge."""
    mod, restore, _c, _r, _t = _load()
    try:
        graph = mod.build_import_graph(
            {"mod_a.py": "import os\nfrom mod_b.inner import x\nimport mod_a\n"}
        )
        assert graph == {"mod_a": {"mod_b", "os"}}
    finally:
        restore()


def test_b20_clusters_are_components_largest_first_and_sorted():
    """b20: connected components come back largest first, each component
    sorted by name."""
    mod, restore, _c, _r, _t = _load()
    try:
        clusters = mod.find_clusters(
            {"a": {"b", "e"}, "c": {"d"}}, max_clusters=5
        )
        assert clusters == [["a", "b", "e"], ["c", "d"]]
    finally:
        restore()


def test_b21_cluster_cap_holds():
    """b21: the cluster cap bounds the returned list even when more
    components exist."""
    mod, restore, _c, _r, _t = _load()
    try:
        graph = {f"n{i}": {f"n{i}x"} for i in range(10)}
        clusters = mod.find_clusters(graph, max_clusters=3)
        assert len(clusters) == 3
        assert all(len(c) == 2 for c in clusters)
    finally:
        restore()


# --- Momentum arithmetic ----------------------------------------------------


def test_b22_step_counters_move_and_never_go_negative():
    """b22: completing a step moves both counters; the remaining count is
    floored at zero from both directions."""
    mod, restore, _c, _r, _t = _load()
    try:
        mom = mod.MomentumTracker()
        mom.set_total_steps(5)
        assert mom.steps_remaining == 5
        mom.complete_step()
        assert (mom.steps_completed, mom.steps_remaining) == (1, 4)

        floor = mod.MomentumTracker()
        floor.complete_step()
        assert floor.steps_remaining == 0, "no total set: floor holds"
        floor.complete_step()
        floor.set_total_steps(0)
        assert floor.steps_remaining == 0, (
            "a total below the completed count must floor at zero"
        )
    finally:
        restore()


def test_b23_progress_ratio_is_exact_and_total():
    """b23: the progress ratio is exact with work planned and zero with no
    work at all."""
    mod, restore, _c, _r, _t = _load()
    try:
        assert mod.MomentumTracker().progress_ratio == 0.0
        mom = mod.MomentumTracker()
        mom.set_total_steps(5)
        mom.complete_step()
        assert mom.progress_ratio == 0.2
    finally:
        restore()


def test_b24_velocity_guards_short_and_degenerate_histories():
    """b24: fewer than two timestamps or a zero elapsed span yields zero
    velocity; one minute between two steps yields exactly one step per
    minute."""
    mod, restore, _c, _r, _t = _load()
    try:
        assert mod.MomentumTracker().velocity == 0.0
        flat = mod.MomentumTracker()
        flat._step_timestamps = [7.0, 7.0]
        assert flat.velocity == 0.0
        paced = mod.MomentumTracker()
        paced._step_timestamps = [0.0, 60.0]
        assert paced.velocity == 1.0
    finally:
        restore()


# --- Hot files, refresh cadence, and manager wiring -------------------------


def test_b25_hot_files_rank_cap_and_average():
    """b25: the top list is ordered by touch count and capped, the unique
    count is exact, and the average size ignores unknown sizes."""
    mod, restore, _c, _r, _t = _load()
    try:
        hot = mod.HotFilesTracker()
        for _ in range(3):
            hot.touch("a.py", 10)
        hot.touch("b.py", 20)
        for _ in range(2):
            hot.touch("c.py", 5)
        top = hot.top(2)
        assert [f["path"] for f in top] == ["a.py", "c.py"]
        assert hot.file_count == 3
        assert hot.avg_file_size == 11
        assert set(hot.serialize(2)) == {"top", "file_count", "avg_file_size"}
    finally:
        restore()


def test_b26_term_refresh_cadence_and_reset():
    """b26: the refresh predicate trips exactly at the configured interval
    and a refresh resets the counter and caches terms."""
    mod, restore, _c, _r, _t = _load()
    try:
        tracker = mod.DomainTermsTracker()
        tracker._refresh_interval = 2
        tracker.update_file("a.py", "def alpha_one():\n    pass\n")
        assert tracker.should_refresh() is False
        tracker.update_file("b.py", "def beta_two():\n    pass\n")
        assert tracker.should_refresh() is True
        tracker.refresh()
        assert tracker.should_refresh() is False
        assert tracker.terms, "a refresh over real content must cache terms"
    finally:
        restore()


def test_b27_manager_hooks_wire_the_dimensions_together():
    """b27: steps feed the hot-file, stack and momentum dimensions; an
    incomplete step counts as a step but not as progress; the batch
    cluster computation runs over the contents the steps stored."""
    mod, restore, _c, _r, tmpdir = _load()
    try:
        mgr = mod.FingerprintManager(
            config=mod.FingerprintConfig(),
            preferences_store=_store(mod, tmpdir),
        )
        for _ in range(2):
            mgr.on_step(
                {"file_path": "pkg/alpha.py", "content": "import beta\n"}
            )
        mgr.on_step({"file_path": "pkg/beta.py", "content": "import os\n"})
        assert mgr.step_count == 3
        assert mgr._momentum.steps_completed == 3
        assert mgr._hot_files.top(1)[0]["path"] == "pkg/alpha.py"
        assert mgr._stack["primary"] == "python"

        mgr.on_step({"file_path": "pkg/x.py", "completed": False})
        assert mgr.step_count == 4
        assert mgr._momentum.steps_completed == 3, (
            "an incomplete step must not count as progress"
        )

        mgr.compute_dep_clusters()
        assert mgr._dep_clusters.clusters[0] == ["alpha", "beta", "os"]
    finally:
        restore()


# --- Fail-secure reads when the seam refuses --------------------------------


def test_b28_ratio_reads_degrade_to_zeroes_when_the_seam_refuses():
    """b28: with every connection refused the ratio read returns the exact
    zero shape instead of raising."""
    mod, restore, _counter = _load_refusing()
    try:
        store = mod.UserPreferencesStore(db_path="refused.db")
        assert store.get_ratios() == {
            "approve": 0.0,
            "modify": 0.0,
            "abort": 0.0,
        }
    finally:
        restore()


def test_b29_phase_and_count_reads_degrade_when_the_seam_refuses():
    """b29: with every connection refused the phase read returns an empty
    mapping and the total count reads as zero, without raising."""
    mod, restore, _counter = _load_refusing()
    try:
        store = mod.UserPreferencesStore(db_path="refused.db")
        assert store.get_phase_preferences() == {}
        assert store.total_decisions == 0
    finally:
        restore()


def test_b30_a_refused_import_does_not_raise_and_the_latch_stays_up():
    """b30: the module loads even when every connection is refused -- the
    store init is fail-secure -- and the availability latch stays up, as
    observed: the latch only trips on failures the store's own guard does
    not absorb. Writing through a refused seam is likewise absorbed."""
    mod, restore, counter = _load_refusing()
    try:
        assert counter["n"] > 0, "the refusing connector must be reached"
        assert mod.FINGERPRINT_AVAILABLE is True
        assert mod.fingerprint_manager is not None
        store = mod.UserPreferencesStore(db_path="refused.db")
        assert store.record("approve", "plan", "ctx") is None
    finally:
        restore()


# --- Configuration loading is fail-secure -----------------------------------


def test_b31_broken_or_missing_configuration_yields_safe_defaults():
    """b31: a syntactically broken configuration file and a missing one
    both land on the documented defaults without raising."""
    mod, restore, _c, _r, _t = _load()
    try:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False
        )
        handle.write("::: not : valid : yaml : {[}\n")
        handle.close()
        original = mod._CONFIG_PATH
        try:
            mod._CONFIG_PATH = handle.name
            broken = mod._load_config()
            os.unlink(handle.name)
            missing = mod._load_config()
        finally:
            mod._CONFIG_PATH = original
            if os.path.exists(handle.name):
                os.unlink(handle.name)
        for cfg in (broken, missing):
            assert cfg.enabled is True
            assert cfg.sqlite_path == _CFG_SQLITE_PATH
            assert cfg.serialization_format == _CFG_FORMAT
            assert cfg.max_anchors == _CFG_MAX_ANCHORS
            assert cfg.max_bug_history == _CFG_MAX_BUG_HISTORY
    finally:
        restore()


# --- Persistence round-trip details -----------------------------------------


def test_b32_phase_reads_exclude_the_empty_phase_and_count_everything():
    """b32: a decision recorded without a phase is excluded from the
    per-phase view but still counts toward ratios and totals."""
    mod, restore, _c, _r, tmpdir = _load()
    try:
        store = _store(mod, tmpdir)
        store.record("approve", "")
        store.record("approve", "plan")
        store.record("modify", "plan")
        assert store.get_phase_preferences() == {
            "plan": {"approve": 1, "modify": 1}
        }
        assert store.total_decisions == 3
        ratios = store.get_ratios()
        assert ratios == {"approve": 0.667, "modify": 0.333, "abort": 0.0}
    finally:
        restore()


# --- Statement census over the module's abstract syntax ---------------------


def _sql_sites(tree):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("execute", "executescript", "executemany")
    ]


def test_b33_every_storage_statement_is_a_constant_and_one_is_bound():
    """b33: the SQL site inventory is frozen -- every site passes a
    constant string (no f-string, no ``format`` call, no concatenation of
    caller data) and exactly one site carries bound parameters."""
    tree = ast.parse(source("session_fingerprint.py").read_text())
    sites = _sql_sites(tree)
    assert len(sites) == _SQL_SITES, (
        f"SQL site census moved: expected {_SQL_SITES}, found {len(sites)}"
    )
    for node in sites:
        assert node.args, "an SQL site must receive its statement"
        first = node.args[0]
        assert isinstance(first, ast.Constant) and isinstance(
            first.value, str
        ), (
            "every SQL statement must be a constant string; a dynamically "
            "built statement reddens this census"
        )
    bound = sum(1 for node in sites if len(node.args) >= 2)
    assert bound == _SQL_PARAMETERIZED, (
        f"expected exactly {_SQL_PARAMETERIZED} parameterised site, "
        f"found {bound}"
    )


def test_b34_the_content_hash_is_flagged_as_non_security_use():
    """b34: the single content-hash call in the module is explicitly
    flagged as non-security use, and no unflagged hash call exists."""
    tree = ast.parse(source("session_fingerprint.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "md5"
    ]
    assert len(calls) == _MD5_SITES
    flagged = sum(
        1
        for node in calls
        for kw in node.keywords
        if kw.arg == "usedforsecurity"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is False
    )
    assert flagged == _MD5_SITES, (
        "every content-hash call must carry the non-security flag"
    )
