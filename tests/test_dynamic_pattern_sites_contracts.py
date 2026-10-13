#!/usr/bin/env python3
"""Contracts for the patterns the package builds at run time.

The pattern cost guard measures each pattern written in the package; one
built at run time -- from a table, a configuration file, an argument -- it
can only count as dynamic. Each dynamic site the guard lists is held here by
module and function (lines move), in one of two ways:

  * driven -- a driver runs the site over every value the package ships for
    it (tables, rules, presets, configuration defaults) and, where a value
    can come from a person, a file or a model, over adversarial ones too;
    every pattern the site compiles is caught as ``re`` compiles it and
    measured by the guard with the methods the site reads it with;
  * named -- the site compiles a pattern it never runs over a text (to raise
    on an invalid one, to read its flags, to judge it before returning it),
    or one read only by ``match`` from a start another reader found; the
    reason is checked on the code.

  * PD1 -- the census: the dynamic sites the guard lists on the real tree are
    exactly the ones held here, as many per function; a site added, moved to
    another function or gone fails by name.
  * PD2 -- every driven site compiles at least one pattern under its driver,
    and no pattern a driven site compiles reads a long text in more than
    linear time with the methods the site reads it with.
  * PD3 -- the census, the capture and the measurement are not blind: a site
    the census does not hold is named, and a quadratic pattern a driven site
    compiles is caught at its site and found steep.
  * PD4 -- each named site holds its reason, read on the code.
  * PD5 -- a teacher's field is read from the start of its line: the value
    the pattern it replaced read, on enumerated and structured bodies, in
    linear time on a body of blank lines.
  * PD6 -- a fenced code block is read from the last line break after its
    fence: the blocks the patterns they replaced read, on enumerated and
    structured responses, in linear time on a fence left open; a language is
    read as written.
  * PD7 -- in place of PD4 (it stays word for word, deselected): each named
    site holds its reason, read on the code, and a pattern ``compile_linear``
    returns that the check refuses is the engine's reading of it, never a
    pattern ``re`` runs; with the engine left out it is refused as before.
"""
import ast
import collections
import contextlib
import importlib
import itertools
import os
import random
import re
import sys
import time
import types
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_pd1_the_dynamic_sites_the_guard_lists_are_the_ones_held_here": 20.0,
    "test_pd2_no_pattern_a_driven_site_compiles_reads_a_long_text_in_more_than_linear_time": 60.0,
    "test_pd3_an_unheld_site_is_named_and_a_quadratic_pattern_at_a_driven_site_is_found": 20.0,
    "test_pd4_each_named_site_holds_its_reason": 20.0,
    "test_pd5_a_teacher_field_is_read_from_the_start_of_its_line": 20.0,
    "test_pd6_a_fenced_code_block_is_read_from_the_last_line_break_after_its_fence": 30.0,
    "test_pd7_each_named_site_holds_its_reason_and_a_costly_pattern_is_returned_only_as_the_engine_reads_it": 20.0,
}

_GUARD = REPO / ".github" / "scripts" / "pattern_cost_guard.py"
_RE_DIR = os.path.dirname(re.__file__)
_STATE = {}

# The dynamic sites, by module::function, with how many the guard lists there. A driven site has its driver below;
# a named one says why it is not driven, and PD4 reads the reason on the code.
_DRIVEN = {
    "opti_oignon/benchmark_evaluator.py::_extract_code_block": 2,
    "opti_oignon/humanizer.py::_apply_vocabulary_replacements": 1,
    "opti_oignon/humanizer.py::_strip_banned_phrases": 1,
    "opti_oignon/humanizer.py::_apply_contractions": 1,
    "opti_oignon/humanizer.py::_reduce_hedging": 1,
    "opti_oignon/linear_pattern.py::_scan": 1,
    "opti_oignon/linear_pattern.py::_prefetch": 1,
    "opti_oignon/network_bind_guard.py::_check_ollama_ss": 1,
    "opti_oignon/presets.py::Preset.matches_keywords": 1,
    "opti_oignon/rag_sanitizer.py::RAGSanitizer.__init__": 1,
    "opti_oignon/rag_sanitizer.py::RAGSanitizer._neutralize_delimiters": 1,
    "opti_oignon/rag_store.py::_strip_boilerplate": 1,
    "opti_oignon/rag_store.py::RAGVectorStore._extract_html_text": 2,
    "opti_oignon/session_fingerprint.py::classify_task": 1,
    "opti_oignon/agent/skills.py::_extract_section": 1,
    "opti_oignon/agent/teacher.py::_field": 1,
    "opti_oignon/api/routes_governor.py::_set_yaml_scalar": 3,
    "opti_oignon/factcheck/scope.py::_phrases": 1,
    "opti_oignon/factcheck/scope.py::_opening": 1,
    "opti_oignon/factcheck/scope.py::denial": 1,
    "opti_oignon/factcheck/scope.py::_past_decision_pattern": 1,
    "opti_oignon/factcheck/scope.py::_person_of_subject": 1,
    "opti_oignon/factcheck/scope.py::analyse": 2,
    "opti_oignon/memory/peels.py::_probe_recall": 1,
    "opti_oignon/memory/peels.py::_bracketed": 1,
    "opti_oignon/memory/peels.py::_unmarked.opens_on_id": 1,
    "opti_oignon/memory/probes.py::answers": 1,
    "opti_oignon/memory/probes.py::_written_in": 1,
    "opti_oignon/memory/probes.py::_decision_split": 1,
    "opti_oignon/memory/probes.py::_unstitched": 1,
    "opti_oignon/plugins/code-guardian/entry_point.py::_check_python_pitfalls": 1,
    "opti_oignon/plugins/task-extractor/entry_point.py::_build_pattern_regex": 1,
    "opti_oignon/plugins/tone-shifter/entry_point.py::apply_rules": 1,
}
_NAMED = {
    # The whole of a joined pattern is read only by ``match`` from a start its head found, and for its groups: the
    # reader's linear time is held by its own contracts, never by a search of the whole.
    "opti_oignon/joined_pattern.py::JoinedPattern.__init__": (1, "match"),
    "opti_oignon/sandbox_manager.py::JoinedPattern.__init__": (1, "match"),
    # Compiled for its flags only; the pattern itself is made linear, or chained, or refused, elsewhere.
    "opti_oignon/sandbox_manager.py::CommandValidator._blocked": (1, "flags"),
    # Compiled so that an invalid pattern raises, then dropped.
    "opti_oignon/linear_pattern.py::verdict": (1, "dropped"),
    "opti_oignon/linear_engine.py::Pattern.__init__": (1, "dropped"),
    "opti_oignon/redteam/feedback.py::apply_suggestion_to_config": (1, "dropped"),
    # Returned only when the static check judges it linear.
    "opti_oignon/linear_pattern.py::compile_linear": (1, "judged"),
}

# Literals a person, a file or a model may hand a site that escapes them into a pattern: short runs of characters a
# pattern can tell apart (spaces and line breaks among them), metacharacters, a letter outside ASCII, a long word.
_ADVERSE = ("a", "aa", "a a", "aaa", " ", "  ", "a ", " a", ".", "a.", "-", "a-b", "'", "_", "1", "a1", "\n",
            "a\nb", chr(0xE9), "x" * 40, "(", "a+", "[a]", "\\", "*", "a*b")


def _guard():
    if "guard" not in _STATE:
        loaded, restore = isolate(targets={"pattern_cost_guard": _GUARD})
        restore()
        _STATE["guard"] = loaded["pattern_cost_guard"]
    return _STATE["guard"]


def _owner(tree, line):
    """The dotted name of the innermost function or class of ``tree`` that holds ``line``, or ``<module>``."""
    found = []

    def walk(node, stack):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child.lineno <= line <= child.end_lineno:
                    found[:] = stack + [child.name]
                walk(child, stack + [child.name])
            else:
                walk(child, stack)

    walk(tree, [])
    return ".".join(found) or "<module>"


def _dynamic_rows(root=REPO):
    """(key, module, line, column, methods) of each site under ``root`` the guard counts as dynamic, ``key`` being
    module::function."""
    if root not in _STATE:
        guard, trees, rows = _guard(), {}, []
        for module, line, column, pattern, _flags, methods in guard.sites(root):
            if pattern is not None or methods == guard.UNREAD:
                continue
            if module not in trees:
                trees[module] = ast.parse((root / module).read_text(encoding="utf-8"))
            rows.append((f"{module}::{_owner(trees[module], line)}", module, line, column, tuple(methods)))
        _STATE[root] = rows
    return _STATE[root]


def _census_gaps(rows):
    """(unheld, stale): the sites the guard lists that the census does not hold, and the census entries the guard
    does not list as many times; each as 'key: listed N, held M'."""
    listed = collections.Counter(key for key, *_rest in rows)
    held = dict(_DRIVEN, **{key: count for key, (count, _why) in _NAMED.items()})
    unheld = [f"{key}: listed {n}, held {held.get(key, 0)}" for key, n in sorted(listed.items()) if held.get(key) != n]
    stale = [f"{key}: listed {listed.get(key, 0)}, held {n}" for key, n in sorted(held.items()) if key not in listed]
    return unheld, stale


def _key_of(code):
    """module::function of a code object read from a file under the repository, or None. A generator, a lambda or a
    comprehension is its function's, as the guard reads the site."""
    try:
        module = Path(code.co_filename).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return None
    named = [part for part in getattr(code, "co_qualname", code.co_name).split(".") if not part.startswith("<")]
    return f"{module}::{'.'.join(named)}"


@contextlib.contextmanager
def _caught(keys):
    """{key: {(pattern, flags)}}: every pattern ``re`` compiles for a function of ``keys`` while the block runs."""
    seen = {key: set() for key in keys}
    real = re._compile

    def caught(pattern, flags):
        frame = sys._getframe(1)
        while frame is not None and (frame.f_code.co_filename.startswith(_RE_DIR)
                                     or frame.f_code.co_filename.endswith("functools.py")):
            frame = frame.f_back
        key = None if frame is None else _key_of(frame.f_code)
        if key in seen:
            if isinstance(pattern, re.Pattern):
                seen[key].add((pattern.pattern, int(pattern.flags)))
            else:
                seen[key].add((pattern, int(flags)))
        return real(pattern, flags)

    re._compile = caught
    try:
        yield seen
    finally:
        re._compile = real


def _steep(pattern, flags, methods):
    """The guard's reading of a pattern that reads a long text in more than linear time with ``methods``, else
    None: one reading, then the majority of readings when the first is steep."""
    guard = _guard()
    first = guard.growth(pattern, flags, methods)
    if not guard.superlinear(first):
        return None
    steep, reading = guard.steady(pattern, flags, methods, first)
    return reading if steep else None


def _values(tree, key):
    """Every value under ``key`` anywhere in a loaded YAML tree, a list read item by item."""
    found = []
    if isinstance(tree, dict):
        for name, value in tree.items():
            if name == key:
                found.extend(value if isinstance(value, list) else [value])
            else:
                found.extend(_values(value, key))
    elif isinstance(tree, list):
        for item in tree:
            found.extend(_values(item, key))
    return found


def _shipped(*parts):
    return yaml.safe_load(source(*parts).read_text(encoding="utf-8")) or {}


# The packages ``agent`` and ``memory`` read the maintainer's configuration when they load: their modules are loaded
# in a window, under stand-in parents, so that no driver reads it.
_TEACHER = {"opti_oignon.agent.untrusted_context": source("agent", "untrusted_context.py"),
            "opti_oignon.agent.teacher": source("agent", "teacher.py")}
_SKILLS = {"opti_oignon.agent.skills": source("agent", "skills.py")}
_MEMORY = {"opti_oignon.memory.probes": source("memory", "probes.py"),
           "opti_oignon.memory.peels": source("memory", "peels.py")}


@contextlib.contextmanager
def _window(targets, packages=(), seeded=None):
    loaded, restore = isolate(targets=targets, packages=packages, seeded=seeded)
    try:
        yield loaded
    finally:
        restore()


# ---------------------------------------------------------------------------
# The drivers: each runs its sites over every value the package ships for them, and over adversarial ones where a
# value can come from outside the code.
# ---------------------------------------------------------------------------
def _drive_benchmark(monkeypatch):
    evaluator = importlib.import_module("opti_oignon.benchmark_evaluator")
    languages = {"python", ""} | {str(lang) for lang in _values(_shipped("config", "benchmark_questions.yaml"),
                                                                  "language")} | set(_ADVERSE)
    for language in sorted(languages):
        evaluator._extract_code_block("no fence", language)


def _definitions(path, names):
    """The functions ``names`` of the file ``path``, defined alone from its own code and under its own name, the
    module itself left unloaded: (name -> function)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kept = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert sorted(node.name for node in kept) == sorted(names), (path, names)
    namespace = {"re": re}
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return {name: namespace[name] for name in names}


def _drive_humanizer(monkeypatch):
    # The module builds its engine as it loads, and the engine looks at the maintainer's data directory: the four
    # functions are defined alone, from the module's own code.
    found = _definitions(source("humanizer.py"), ("_apply_vocabulary_replacements", "_strip_banned_phrases",
                                                  "_reduce_hedging", "_apply_contractions"))
    raw = _shipped("config", "humanizer.yaml")
    assert raw.get("vocabulary_replacements") and raw.get("banned_phrases"), "the shipped tables were read"
    adverse = {word: "x" for word in _ADVERSE}
    found["_apply_vocabulary_replacements"]("text", {**raw.get("vocabulary_replacements", {}), **adverse})
    found["_strip_banned_phrases"]("text", list(raw.get("banned_phrases", [])) + list(_ADVERSE))
    found["_reduce_hedging"]("text", list(raw.get("hedging_excess", [])) + list(_ADVERSE))
    found["_apply_contractions"]("text", {**raw.get("contractions", {}), **adverse}, "casual")


def _drive_linear_pattern(monkeypatch):
    checker = importlib.import_module("opti_oignon.linear_pattern")
    checker._judged.cache_clear()
    monkeypatch.setattr(checker, "_RANGES", {})
    for pattern, flags in ((r"\bword\b", re.IGNORECASE), (r"[a-z]+\d", 0), (r"(?m)^\s*x", 0), (r"[^\W\d]+", 0),
                           (r"\w+@\w+\.\w+", re.IGNORECASE), (r"(?s)<[^>]*>.*?</b>", 0)):
        checker.verdict(pattern, flags)


def _drive_network(monkeypatch):
    guard = importlib.import_module("opti_oignon.network_bind_guard")
    for port in sorted({guard._OLLAMA_DEFAULT_PORT, 0, 1, 8001, 65535}):
        done = types.SimpleNamespace(returncode=0, stdout=f"LISTEN 0 4096 127.0.0.1:{port} 0.0.0.0:*\n")
        monkeypatch.setattr(guard, "subprocess", types.SimpleNamespace(
            run=lambda *_a, done=done, **_k: done, TimeoutExpired=TimeoutError))
        assert guard._check_ollama_ss(port) == "127.0.0.1", port


def _drive_presets(monkeypatch):
    # The real configuration reads the maintainer's own presets: a stand-in serves the module what it imports.
    stand_in = types.ModuleType("opti_oignon.config")
    stand_in.DATA_DIR, stand_in.config = Path(os.devnull), types.SimpleNamespace(get_all_presets=dict)
    stand_in.load_yaml = stand_in.save_yaml = lambda *_args, **_kwargs: {}
    with _window({"opti_oignon.presets": source("presets.py")}, seeded={"opti_oignon.config": stand_in}) as loaded:
        keywords = [str(word) for word in _values(_shipped("config", "presets.yaml"), "keywords")]
        assert keywords, "the shipped presets were read"
        preset = types.SimpleNamespace(keywords=keywords + list(_ADVERSE))
        loaded["opti_oignon.presets"].Preset.matches_keywords(preset, "text")


def _drive_rag_sanitizer(monkeypatch):
    sanitizer = importlib.import_module("opti_oignon.rag_sanitizer")
    held = importlib.import_module("opti_oignon.redteam.feedback").held_patterns()
    custom = [{"name": f"held{i}", "regex": pattern} for i, pattern in enumerate(sorted(held))]
    sanitizer.RAGSanitizer({"audit": {"enabled": False}, "sanitization": {"custom_patterns": custom}})
    shipped = (_shipped("config", "rag.yaml").get("injection_defense") or {}).get("separation") or {}
    assert shipped.get("system_tag"), "the shipped separation was read"
    chunk = types.SimpleNamespace(trust_level=types.SimpleNamespace(value="low"), source="s", sanitized_text="text")
    for separation in [shipped, {}] + [{"system_tag": w, "user_tag": w, "data_tag": w} for w in _ADVERSE]:
        made = sanitizer.RAGSanitizer({"audit": {"enabled": False}, "separation": separation})
        made._wrap_xml("system", "query", [chunk])
        made._wrap_separator("system", "query", [chunk])


def _drive_rag_store(monkeypatch):
    loaded, restore = isolate(targets={"opti_oignon.rag_store": source("rag_store.py")})
    try:
        store = loaded["opti_oignon.rag_store"]
        shipped = _shipped("config", "rag.yaml")
        tags = [str(tag) for tag in _values(shipped, "strip_tags")] + list(_ADVERSE)
        words = [str(word) for word in _values(shipped, "boilerplate_patterns")] + list(_ADVERSE)
        page = '<div class="x">kept</div><nav>menu</nav>'
        store.RAGVectorStore._extract_html_text(page, {})
        store.RAGVectorStore._extract_html_text(page, {"strip_tags": tags, "boilerplate_patterns": words})
    finally:
        restore()


def _drive_fingerprint(monkeypatch):
    importlib.import_module("opti_oignon.session_fingerprint").classify_task("fix the failing test")


def _written(loaded, function, position):
    """The values the code of the module ``loaded`` hands ``function`` at argument ``position``: a string it writes,
    or a module constant naming one; (values, calls) so that a caller can tell every call was read."""
    tree = ast.parse(Path(loaded.__file__).read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == function]
    values = []
    for call in calls:
        argument = call.args[position] if len(call.args) > position else None
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            values.append(argument.value)
        elif isinstance(argument, ast.Name) and argument.id.isupper() and isinstance(vars(loaded).get(argument.id), str):
            values.append(vars(loaded)[argument.id])
    return values, calls


def _drive_skills(monkeypatch):
    with _window(_SKILLS, packages=("opti_oignon.agent",)) as loaded:
        skills = loaded["opti_oignon.agent.skills"]
        headers, calls = _written(skills, "_extract_section", 1)
        # The header goes into the pattern escaped, but only a header the code writes is ever read: a section's name.
        assert calls and len(headers) == len(calls), "every header a section is read under is written by the code"
        for header in headers:
            skills._extract_section("## When\nbody", header)


def _drive_teacher(monkeypatch):
    with _window(_TEACHER, packages=("opti_oignon.agent",)) as loaded:
        teacher = loaded["opti_oignon.agent.teacher"]
        names, calls = _written(teacher, "_field", 1)
        # The field's name goes into the pattern as written: only a name the code writes is ever read.
        assert calls and len(names) == len(calls), "every field the teacher reads is named by the code"
        for name in names:
            teacher._field("name: x", name)


def _drive_governor(monkeypatch):
    routes = importlib.import_module("opti_oignon.api.routes_governor")
    text = source("config", "resource_governor.yaml").read_text(encoding="utf-8")
    for key in routes.WRITABLE_KEYS:
        try:
            routes._set_yaml_scalar(text, key, 1)
        except Exception:  # noqa: BLE001 -- a key the shipped file lacks is refused; only its pattern is wanted
            pass


_SCOPE_CLAIMS = (("I think we decided to use Docker.", "en"), ("We decided to use Docker.", "en"),
                 ("However, we chose the venue.", "en"), ("You should restart the server.", "en"),
                 ("No, not at all.", "en"), ("Restart the server.", "en"), ("It is raining.", "en"),
                 ("Je pense que nous avons choisi Docker.", "fr"), ("Non, pas du tout.", "fr"))


def _drive_scope(monkeypatch):
    scope = importlib.import_module("opti_oignon.factcheck.scope")
    monkeypatch.setattr(scope, "_COMPILED", {})
    scope.denial.cache_clear()
    scope._past_decision_pattern.cache_clear()
    for text, lang in _SCOPE_CLAIMS:
        scope.analyse(scope.Claim(text=text, lang=lang), max_claim_chars=600)
        scope.denial(text)
    scope.denial.cache_clear()
    scope._past_decision_pattern.cache_clear()


def _drive_memory(monkeypatch):
    with _window(_MEMORY, packages=("opti_oignon.memory",)) as loaded:
        peels, probes = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.probes"]
        gate = peels.load_gate()
        turns = [{"turn_id": "t1", "role": "user", "content": "keep the backups"}]
        peels._unmarked("A summary [t1] and [t 0001] and (x).", turns)
        for answer in ("2024-01-02", "42", "Docker") + _ADVERSE:
            probes.answers(probes.Probe(kind="date", question="q", answer=answer, turn_id="t1"), "text")
            probes._written_in(answer, ["text"])
        probes._decision_split(gate.directives, gate.lexicon)
        probes._unstitched("A summary. [t1] keep the backups", {("t1", "keep the backups")}, ("t1", "t2"))


def _drive_plugins(monkeypatch):
    loaded, restore = isolate(targets={
        "opti_oignon.linear_pattern": source("linear_pattern.py"),
        "tone_shifter_entry_point": source("plugins", "tone-shifter", "entry_point.py"),
        "task_extractor_entry_point": source("plugins", "task-extractor", "entry_point.py"),
        "code_guardian_entry_point": source("plugins", "code-guardian", "entry_point.py")})
    try:
        shifter = loaded["tone_shifter_entry_point"]
        for rules in shifter.MODE_RULES.values():
            shifter.apply_rules("Text, however, is here.", rules)
        extractor = loaded["task_extractor_entry_point"]
        extractor._build_pattern_regex(list(extractor.DEFAULT_PATTERNS) + list(_ADVERSE))
        guardian = loaded["code_guardian_entry_point"]
        code = "import os\nimport os.path\nimport a as b\nfrom x import y, z as w\nimport " + "x" * 40 + "\n"
        guardian._check_python_pitfalls(ast.parse(code), code)
    finally:
        restore()


_DRIVERS = (_drive_benchmark, _drive_humanizer, _drive_linear_pattern, _drive_network, _drive_presets,
            _drive_rag_sanitizer, _drive_rag_store, _drive_fingerprint, _drive_skills, _drive_teacher, _drive_governor,
            _drive_scope, _drive_memory, _drive_plugins)


def _measured(seen, rows):
    """(measured, steep): how many distinct patterns of ``seen`` were measured, and each one the guard finds steep
    with the methods its site reads it with, as 'key: pattern -- pump'."""
    methods = collections.defaultdict(set)
    for key, _module, _line, _column, how in rows:
        methods[key].update(how)
    done, steep = set(), []
    for key in sorted(seen):
        how = tuple(sorted(methods[key])) or ("search",)
        for pattern, flags in sorted(seen[key], key=repr):
            if (pattern, flags, how) in done:
                continue
            done.add((pattern, flags, how))
            reading = _steep(pattern, flags, how)
            if reading is not None:
                steep.append(f"{key}: {pattern!r:.80} -- {reading[2]!r:.40}")
    return len(done), steep


# ---------------------------------------------------------------------------
# PD1 -- the census
# ---------------------------------------------------------------------------
def test_pd1_the_dynamic_sites_the_guard_lists_are_the_ones_held_here():
    rows = _dynamic_rows()
    assert len(rows) >= 40, f"the guard lists {len(rows)} dynamic sites on the real tree"
    assert not set(_DRIVEN) & set(_NAMED), "a site is driven or named, never both"
    unheld, stale = _census_gaps(rows)
    assert not unheld, f"dynamic sites the census does not hold as listed: {unheld[:6]}"
    assert not stale, f"census entries the guard no longer lists: {stale[:6]}"


# ---------------------------------------------------------------------------
# PD2 -- the driven sites
# ---------------------------------------------------------------------------
def test_pd2_no_pattern_a_driven_site_compiles_reads_a_long_text_in_more_than_linear_time(monkeypatch):
    with _caught(set(_DRIVEN)) as seen:
        for drive in _DRIVERS:
            drive(monkeypatch)
    silent = sorted(key for key in _DRIVEN if not seen[key])
    assert not silent, f"driven sites that compiled nothing under their driver: {silent}"
    measured, steep = _measured(seen, _dynamic_rows())
    assert measured >= len(_DRIVEN), f"{measured} patterns measured for {len(_DRIVEN)} sites"
    assert not steep, f"{len(steep)} of {measured} patterns read a long text in more than linear time: {steep[:4]}"


# ---------------------------------------------------------------------------
# PD3 -- the census, the capture and the measurement are not blind
# ---------------------------------------------------------------------------
def test_pd3_an_unheld_site_is_named_and_a_quadratic_pattern_at_a_driven_site_is_found(tmp_path, monkeypatch):
    planted = tmp_path / "opti_oignon" / "planted.py"
    planted.parent.mkdir(parents=True)
    planted.write_text("import re\n\n\ndef planted(word):\n    return re.findall(word, 'text')\n", encoding="utf-8")
    unheld, stale = _census_gaps(_dynamic_rows(tmp_path))
    assert unheld == ["opti_oignon/planted.py::planted: listed 1, held 0"], unheld
    assert len(stale) == len(_DRIVEN) + len(_NAMED), "on a tree without them every entry is stale"
    fingerprint = importlib.import_module("opti_oignon.session_fingerprint")
    key = "opti_oignon/session_fingerprint.py::classify_task"
    steep_one = (r"\d+x", fingerprint._TASK_PATTERNS[0][1])
    monkeypatch.setattr(fingerprint, "_TASK_PATTERNS", list(fingerprint._TASK_PATTERNS) + [steep_one])
    with _caught({key}) as seen:
        fingerprint.classify_task("fix 123x")
    assert (r"\d+x", 0) in seen[key], sorted(seen[key])
    measured, steep = _measured(seen, _dynamic_rows())
    assert measured == len(fingerprint._TASK_PATTERNS), (measured, len(fingerprint._TASK_PATTERNS))
    assert len(steep) == 1 and steep[0].startswith(key + ": '\\\\d+x'"), steep


# ---------------------------------------------------------------------------
# PD4 -- the named sites
# ---------------------------------------------------------------------------
def _site(module, line, column):
    """(tree, parents, call): the module's tree, each node's parent, and the call the guard lists at the site."""
    tree = ast.parse((REPO / module).read_text(encoding="utf-8"))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and node.lineno == line and node.col_offset == column]
    assert len(calls) == 1, (module, line, column, len(calls))
    return tree, parents, calls[0]


def _holds(why, tree, parents, call):
    """True when the call at a site holds ``why`` on the code: its pattern dropped once compiled, read for its flags
    only, or kept as the whole of a joined pattern that is read by ``match`` and for its groups only."""
    parent = parents[call]
    if why == "dropped":
        return isinstance(parent, ast.Expr)
    if why == "flags":
        return isinstance(parent, ast.Attribute) and parent.attr == "flags"
    if why == "match":
        if not (isinstance(parent, ast.Assign) and [getattr(t, "attr", "") for t in parent.targets] == ["_whole"]):
            return False
        reads = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute) and node.attr == "_whole" and isinstance(node.ctx, ast.Load)]
        for node in reads:
            up = parents[node]
            if not isinstance(up, ast.Attribute) or up.attr not in ("match", "groups"):
                return False
            if up.attr == "match" and not (isinstance(parents[up], ast.Call) and parents[up].func is up):
                return False
        return bool(reads)
    return False


def test_pd4_each_named_site_holds_its_reason():
    rows = _dynamic_rows()
    named = [row for row in rows if row[0] in _NAMED]
    assert len(named) == sum(count for count, _why in _NAMED.values()), named
    for key, module, line, column, _methods in named:
        why = _NAMED[key][1]
        if why == "judged":
            continue
        assert _holds(why, *_site(module, line, column)), f"{key} (line {line}) does not hold its reason: {why}"
    checker = importlib.import_module("opti_oignon.linear_pattern")
    try:
        checker.compile_linear(r"(?:x+x+)+y")
    except checker.CostlyPattern:
        pass
    else:
        raise AssertionError("a costly pattern was returned by compile_linear")
    assert checker.compile_linear(r"\bword\b").pattern == r"\bword\b", "a linear pattern is returned compiled"
    # Not blind: a driven site holds none of the reasons.
    driven = next(row for row in rows if row[0] == "opti_oignon/agent/teacher.py::_field")
    for why in ("dropped", "flags", "match"):
        assert not _holds(why, *_site(*driven[1:4])), f"the teacher's field read as {why}"


# ---------------------------------------------------------------------------
# PD5 -- a teacher's field
# ---------------------------------------------------------------------------
def _cpu(read, text):
    """The least of three processor times of ``read(text)``."""
    best = None
    for _ in range(3):
        start = time.process_time()
        read(text)
        spent = time.process_time() - start
        best = spent if best is None else min(best, spent)
    return best


def _quadratic(read, make):
    """True when eight times the text takes far more than eight times the time, twice in a row (linear grows eight
    times, quadratic sixty-four)."""
    def once():
        small, large = _cpu(read, make(500)), _cpu(read, make(4000))
        return large > 0.002 and large > 24 * max(small, 2e-5)
    return once() and once()


def _pd5_old_field(body, field_name):
    found = re.search(rf"^\s*{field_name}\s*:\s*(.+?)\s*$", body, re.IGNORECASE | re.MULTILINE)
    return found.group(1).strip() if found else None


_PD5_PINNED = ("\n\n  name: v", "name:\nvalue", "name\n: x", "x\nname: a\nname: b", "  \t\nNAME :  spaced  \n",
               "name:", "name: \n\n", "\n \n category: c\nname: n", "noname: x\nname: y")


def test_pd5_a_teacher_field_is_read_from_the_start_of_its_line():
    with _window(_TEACHER, packages=("opti_oignon.agent",)) as loaded:
        _pd5_held(loaded["opti_oignon.agent.teacher"]._field)


def _pd5_held(field):
    for body in _PD5_PINNED:
        for name in ("name", "category"):
            assert field(body, name) == _pd5_old_field(body, name), (body, name)
    assert field("\n\n  name: v", "name") == "v" and field("name:\nvalue", "name") == "value", "pinned values read"
    tokens = ["\n", " ", "\t", "name", "Name", ":", "x", "\r", chr(0x2028)]
    checked = 0
    for size in range(6):
        for parts in itertools.product(tokens, repeat=size):
            body = "".join(parts)
            for name in ("name", "category"):
                assert field(body, name) == _pd5_old_field(body, name), (body, name)
                checked += 1
    rng = random.Random(5)
    pieces = ["\n", " ", "\t", "name", ":", " v ", "category", "\n\n", "x:", chr(0x85), "Name : w"]
    for _ in range(20000):
        body = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 14)))
        for name in ("name", "category"):
            assert field(body, name) == _pd5_old_field(body, name), (body, name)
            checked += 1
    assert checked > 150000, checked
    assert _quadratic(lambda body: _pd5_old_field(body, "name"), lambda n: "\n" * n), "the timing oracle sees it"
    assert not _quadratic(lambda body: field(body, "name"), lambda n: "\n" * n), "linear on a body of blank lines"
    assert not _quadratic(lambda body: field(body, "name"), lambda n: "name" + " \n" * n + ":"), "linear past it"


# ---------------------------------------------------------------------------
# PD6 -- a fenced code block
# ---------------------------------------------------------------------------
def _pd6_old_code_block(response, language="python"):
    for pattern in (rf"```(?:{language})?\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return "\n".join(matches)
    return response.strip()


_PD6_PINNED = ("```python\n\n\n  x = 1\n```", "```\n```", "```python \t\n x\n```", "```py\nx```", "```\n\n",
               "a ```python\nx``` b ```\ny```", "```python\n" + chr(0xA0) + "\nx```", "```bash\r\nls\n```")


def test_pd6_a_fenced_code_block_is_read_from_the_last_line_break_after_its_fence():
    extract = importlib.import_module("opti_oignon.benchmark_evaluator")._extract_code_block
    for text in _PD6_PINNED:
        for language in ("python", "bash", ""):
            assert extract(text, language) == _pd6_old_code_block(text, language), (text, language)
    assert extract(_PD6_PINNED[0]) == "  x = 1\n", "the first line keeps its indent"
    tokens = ["```", "`", "\n", " ", "python", "x", "\t", "``"]
    checked = 0
    for size in range(6):
        for parts in itertools.product(tokens, repeat=size):
            text = "".join(parts)
            for language in ("python", "bash", ""):
                assert extract(text, language) == _pd6_old_code_block(text, language), (text, language)
                checked += 1
    rng = random.Random(6)
    for _ in range(20000):
        parts = []
        for _ in range(rng.randint(1, 5)):
            parts.append(rng.choice(["```", "```python", "```bash", "```py"]) + "".join(
                rng.choice([" ", "\t", "\n", "\r", chr(0xA0)]) for _ in range(rng.randint(0, 5))))
            parts.append("".join(rng.choice(["x", " ", "\n", "  y = 1", "`", "``"]) for _ in range(rng.randint(0, 5))))
            if rng.random() < 0.7:
                parts.append("```" + rng.choice(["", "\n", " ", "x"]))
        text = "".join(parts)
        for language in ("python", "bash", ""):
            assert extract(text, language) == _pd6_old_code_block(text, language), (text, language)
            checked += 1
    assert checked > 150000, checked
    open_fence = lambda n: "```python" + "\n" * n  # noqa: E731
    assert _quadratic(lambda text: _pd6_old_code_block(text), open_fence), "the timing oracle sees it"
    assert not _quadratic(lambda text: extract(text), open_fence), "linear on a fence left open"
    assert not _quadratic(lambda text: extract(text), lambda n: "```" + " \n" * n), "linear on blank lines of spaces"
    assert extract("```c++\nint x;\n```", "c++") == "int x;\n", "a language is read as written"
    assert extract("```(\nx\n```", "(") == "x\n", "a language is never read as a pattern"


# ---------------------------------------------------------------------------
# PD7 -- the named sites, and what compile_linear returns for a costly pattern (in place of PD4)
# ---------------------------------------------------------------------------
def test_pd7_each_named_site_holds_its_reason_and_a_costly_pattern_is_returned_only_as_the_engine_reads_it():
    rows = _dynamic_rows()
    named = [row for row in rows if row[0] in _NAMED]
    assert len(named) == sum(count for count, _why in _NAMED.values()), named
    for key, module, line, column, _methods in named:
        why = _NAMED[key][1]
        if why == "judged":
            continue
        assert _holds(why, *_site(module, line, column)), f"{key} (line {line}) does not hold its reason: {why}"
    checker = importlib.import_module("opti_oignon.linear_pattern")
    engine = importlib.import_module("opti_oignon.linear_engine")
    costly = checker.compile_linear(r"(?:x+x+)+y")
    assert isinstance(costly, engine.Pattern) and not isinstance(costly, re.Pattern), type(costly)
    assert costly.search("xxxy").span() == (0, 4), "read as re reads it"
    try:
        checker.compile_linear(r"(?:x+x+)+y", engine=False)
    except checker.CostlyPattern:
        pass
    else:
        raise AssertionError("with the engine left out, a costly pattern was returned by compile_linear")
    assert checker.compile_linear(r"\bword\b").pattern == r"\bword\b", "a linear pattern is returned compiled"
    assert isinstance(checker.compile_linear(r"\bword\b"), re.Pattern)
    # Not blind: a driven site holds none of the reasons.
    driven = next(row for row in rows if row[0] == "opti_oignon/agent/teacher.py::_field")
    for why in ("dropped", "flags", "match"):
        assert not _holds(why, *_site(*driven[1:4])), f"the teacher's field read as {why}"
