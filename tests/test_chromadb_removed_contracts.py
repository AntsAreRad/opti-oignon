#!/usr/bin/env python3
"""Contracts for chromadb's departure: no module of the platform reads, imports, probes or installs it.

``db_encryption migrate`` reads ChromaDB's folders with the standard library
(``chroma_migration``, block 181): nothing of the package, its scripts or its
guards imports chromadb any more, by any spelling, and no file installs it
(block 182). What it brought that the platform uses, the platform declares.

  * RM1 -- NO DEPENDENCY FILE DECLARES CHROMADB: no requirement named
    ``chromadb``, ``chroma-hnswlib`` or ``chromadb-client``, by its normalized
    name, in pyproject's dependencies, any of its extras, its dependency
    groups or its build requires; in a requirements or constraints file --
    one named so (``*requirements*`` or ``constraints*``, ``.txt`` or
    ``.in``), or one a ``-r`` or a ``-c`` names, beside the file that names it
    or from the root --; or among the arguments of a ``pip install`` in any
    other file git carries outside ``tests/`` and ``CHANGELOG.md``: a line
    read as the shell reads it (quotes, ``;``, ``&``, ``|``, redirections,
    inline code's backquotes), pip under any of its names (``pip3.12``,
    ``python -m pip``, a path) with its general options before its command,
    a Dockerfile's exec form, a YAML file's strings and lists of strings, a
    Python source's string constants and its lists of them. A file read as
    requirements that is not text fails by name. A bare URL names no
    requirement: this census reads names.
  * RM2 -- NO MODULE IMPORTS CHROMADB BY ANY SPELLING: no ``import`` and no
    ``from ... import`` of chromadb or of a module below it, no call handed a
    constant that names it, by position or by name, whatever is called -- an
    import, a lookup, a probe, a read of its installation --, and no
    subscript or membership test of the modules with such a constant, in the
    package's modules, ``scripts/`` or ``.github/scripts/``. A name built at
    run time is no spelling this census can see.
  * RM3 -- THE REAL TREE IS GREEN AND CHROMADB IS NO CLASSIFIED LIBRARY: the
    egress census holds EC12's ratchet on EC12's own tables, and classifies
    every third-party name EC12 named but chromadb, which LIBRARIES no longer
    permits: an ``import chromadb`` in the package is refused by name. The
    census keeps its chromadb rules, as it keeps those of client libraries
    the package does not import: detectors. It supersedes EC12.
  * RM5 -- THE PACKAGE NAME SURVIVES ONLY IN DETECTORS AND HISTORY: the token
    ``chromadb``, in its own case, not inside a longer name, in the files git
    carries outside ``tests/`` and ``CHANGELOG.md`` -- a file that begins with
    UTF-16's mark read as its text --, counted file by file, is a closed
    list: the three guards that detect it, and pyproject's history of
    supersession. A path below a folder named ``data`` is never opened. pip
    reads a name in any case; RM1 holds what pip is told.
  * RM6 -- THE WEBSOCKET ROUTES KEEP THEIR SERVER STACK DECLARED: the package
    serves WebSocket routes, which uvicorn answers only with a WebSocket
    implementation installed, and its command-line client imports
    ``websockets``; chromadb was what installed both. pyproject declares
    exactly ``uvicorn[standard]>=0.24.0`` -- the server stack every install
    ran on -- and ``websockets>=10.4``, no marker on either, and
    ``requirements-backend.txt``, which the image installs, declares exactly
    pyproject's dependencies, specifier for specifier.
  * RM7 -- THE DOCS SAY HOW MIGRATE READS AND EMBEDS, AND NO PAGE NAMES
    CHROMADB: SECURITY.md's closed exception RS-01 and the encryption page say
    the migration reads ChromaDB's database with the standard library,
    read-only, and embeds each text again, with a model Ollama serves; the
    RAG guide says it embeds every text again with the models the
    configuration names, which Ollama must serve; no source of the frontend
    names ChromaDB, in any case.

Local-only (the public distribution ships no tests). RM2, RM6 and RM7 read
files; RM1 and RM5 ask git which files it carries; RM3 loads the census
through the shared isolation window.
"""

import ast
import codecs
import json
import posixpath
import re
import shlex
import sys
from pathlib import Path

import tomllib
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_census import modules as package_modules  # noqa: E402
from _frontend import _git, _split_z  # noqa: E402
from test_egress_census_guard_contracts import (  # noqa: E402
    _THIRD_PARTY,
    BIRTH,
    FLOOR,
    RETIRED,
    _load,
    _write,
)

_REPO = Path(__file__).resolve().parent.parent
BUDGET_S = {
    "test_rm1_no_dependency_file_declares_chromadb": 20.0,
    "test_rm2_no_module_imports_chromadb_by_any_spelling": 8.0,
    "test_rm3_the_real_tree_is_green_and_chromadb_is_no_classified_library": 8.0,
    "test_rm5_the_package_name_survives_only_in_detectors_and_history": 8.0,
    "test_rm6_the_websocket_routes_keep_their_server_stack_declared": 8.0,
    "test_rm7_the_docs_say_how_migrate_reads_and_embeds_and_no_page_names_chromadb": 8.0,
}

# The distributions chromadb ships as, by their normalized names.
_GONE = frozenset({"chromadb", "chroma-hnswlib", "chromadb-client"})
# pip, under the names it runs by: pip, pip3, pip3.12, a path to one of them.
_PIP = re.compile(r"(?:.*/)?pip(?:3(?:\.\d+)?)?")
# pip's general options that take the next word, written before its command.
_GENERAL_ARGUMENT = frozenset({
    "--log", "--log-file", "--cache-dir", "--proxy", "--retries", "--timeout", "--exists-action", "--trusted-host",
    "--cert", "--client-cert", "--python", "--keyring-provider", "--use-feature", "--use-deprecated",
})
# pip install's options that take the next word when not written with "=".
_TAKES_ARGUMENT = frozenset({
    "-r", "--requirement", "-c", "--constraint", "-e", "--editable", "-i", "--index-url", "--extra-index-url",
    "-f", "--find-links", "--prefix", "-t", "--target", "--root", "--trusted-host", "--platform",
    "--python-version", "--implementation", "--abi", "--cache-dir", "--src", "--upgrade-strategy", "--no-binary",
    "--only-binary", "--progress-bar", "--log", "--proxy", "--retries", "--timeout", "--exists-action", "--cert",
    "--client-cert", "--report", "-C", "--config-settings",
})
# The options that name a requirements or a constraints file.
_NAMES_A_FILE = ("-r", "--requirement", "-c", "--constraint")
# A file read as requirements by its name.
_REQUIREMENT_FILE = re.compile(r"(?:.*requirements.*|constraints.*)\.(?:txt|in)")
_COMMENT = re.compile(r"(?:^|\s)#.*$")
_ARRAY = re.compile(r"\[[^\[\]]*\]")
# The token, not inside a longer name.
_TOKEN = re.compile(rb"(?<![A-Za-z0-9_])chromadb(?![A-Za-z0-9_])")
# Where the name may still be written, and how many times: the guards that detect it, and pyproject's history.
_SURVIVORS = {
    # The census's sink rules for chromadb, kept as detectors.
    ".github/scripts/egress_census_guard.py": 16,
    # HEAVY: a library that must never load at `import opti_oignon`.
    ".github/scripts/import_footprint_guard.py": 1,
    # _DATABASE_LIBRARIES and its docstring: a database library is a writer.
    ".github/scripts/write_census_guard.py": 2,
    # The paragraphs of supersession above the deselect lines.
    "pyproject.toml": 8,
}


def _requirement(spec):
    """(PEP 503 name, extras, the rest without spaces) of a requirement, or None for a line that names none."""
    text = _COMMENT.sub("", spec).strip()
    match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?\s*(.*)", text)
    if text.startswith("-") or match is None:
        return None
    name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
    extras = frozenset(e.strip().lower() for e in (match.group(2) or "").split(",") if e.strip())
    return name, extras, re.sub(r"\s+", "", match.group(3))


def _named_file(word, following):
    """The file an option names -- ``-r FILE``, ``-rFILE``, ``--requirement=FILE``, and the same of ``-c`` --, or
    None."""
    for option in _NAMES_A_FILE:
        if word == option:
            return following
        if word.startswith(option + "="):
            return word[len(option) + 1:]
        if len(option) == 2 and word.startswith(option) and len(word) > 2:
            return word[2:]
    return None


def _install_arguments(words):
    """(requirements, files named) of each ``pip install`` among one command's words: pip under any of its names,
    its general options before its command passed over, install's options and their arguments dropped."""
    specs, named = [], []
    for at, word in enumerate(words):
        if not _PIP.fullmatch(word):
            continue
        rest, position = words[at + 1:], 0
        while position < len(rest) and rest[position].startswith("-"):
            position += 2 if rest[position] in _GENERAL_ARGUMENT else 1
        if position >= len(rest) or rest[position] != "install":
            continue
        arguments, skip = rest[position + 1:], False
        for index, item in enumerate(arguments):
            if skip:
                skip = False
            elif item.startswith("-"):
                file = _named_file(item, arguments[index + 1] if index + 1 < len(arguments) else None)
                if file:
                    named.append(file)
                skip = item in _TAKES_ARGUMENT
            else:
                specs.append(item)
    return specs, named


def _commands(script):
    """The commands of a shell script that run pip, each as its words, read as the shell reads them: continuations
    joined, comments and backquotes dropped, quotes taken, a line cut at ``;``, ``&``, ``|``, ``&&`` and ``||``, a
    redirection dropped with its target; a JSON array of strings on a line (a Dockerfile's exec form) is a command
    too."""
    commands = []
    for line in script.replace("\\\n", " ").splitlines():
        if "pip" not in line:
            continue
        for found in _ARRAY.finditer(line):
            try:
                array = json.loads(found.group(0))
            except ValueError:
                continue
            if isinstance(array, list) and array and all(isinstance(word, str) for word in array):
                commands.append(array)
        plain = _COMMENT.sub("", line.replace("`", " "))
        lexer = shlex.shlex(plain, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        try:
            words = list(lexer)
        except ValueError:  # an unbalanced quote: the words as they are written
            words = plain.split()
        command, skip = [], False
        for word in words:
            if skip:
                skip = False
            elif set(word) <= set(";&|"):
                commands.append(command)
                command = []
            elif set(word) <= set("<>&"):
                skip = True
            else:
                command.append(word)
        commands.append(command)
    return [command for command in commands if command]


def _installs(commands):
    specs, named = [], []
    for words in commands:
        found, files = _install_arguments(words)
        specs += found
        named += files
    return specs, named


def _from_commands(text):
    """(requirements, files named) of the ``pip install`` commands a file holds, read as shell lines."""
    return _installs(_commands(text))


def _strings(value):
    """Each string of a parsed YAML document, as a script, and each list of strings in it, as a command's words."""
    if isinstance(value, str):
        yield _commands(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        if value and all(isinstance(item, str) for item in value):
            yield [value]
        for item in value:
            yield from _strings(item)


def _from_yaml(text):
    """(requirements, files named) of a YAML file: each string read as a shell script -- a folded block is one line
    --, each list of strings as a command's words; a file YAML does not parse is read as shell lines."""
    try:
        documents = list(yaml.safe_load_all(text))
    except yaml.YAMLError:
        return _from_commands(text)
    return _installs([words for document in documents for commands in _strings(document) for words in commands])


def _from_python(text):
    """(requirements, files named) of a Python source: each string constant read as a shell script, each list or
    tuple as a command's words, a word that is no string constant standing unread."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _from_commands(text)
    commands = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            commands.append([e.value if isinstance(e, ast.Constant) and isinstance(e.value, str) else "?"
                             for e in node.elts])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            commands += _commands(node.value)
    return _installs(commands)


def _from_pyproject(text):
    """(every requirement pyproject declares -- its dependencies, each extra, each dependency group, the build's --,
    no file named)."""
    project = tomllib.loads(text)
    specs = list(project.get("project", {}).get("dependencies", []))
    for extra in project.get("project", {}).get("optional-dependencies", {}).values():
        specs += extra
    for group in project.get("dependency-groups", {}).values():
        specs += [item for item in group if isinstance(item, str)]
    specs += project.get("build-system", {}).get("requires", [])
    return specs, []


def _from_requirements(text):
    """(requirements, files named) of a requirements or constraints file: a line a requirement, an option line
    naming a file with ``-r`` or ``-c``."""
    specs, named = [], []
    for line in text.splitlines():
        words = _COMMENT.sub("", line).split()
        if not words:
            continue
        if words[0].startswith("-"):
            for index, word in enumerate(words):
                file = _named_file(word, words[index + 1] if index + 1 < len(words) else None)
                if file:
                    named.append(file)
            continue
        specs.append(" ".join(words))
    return specs, named


def _reader(rel):
    """How a carried file names what pip installs, by its name."""
    name = rel.rsplit("/", 1)[-1]
    if rel == "pyproject.toml":
        return _from_pyproject
    if _REQUIREMENT_FILE.fullmatch(name):
        return _from_requirements
    if name.endswith((".yml", ".yaml")):
        return _from_yaml
    if name.endswith(".py"):
        return _from_python
    return _from_commands


def _gone(texts):
    """{file: the gone names it gives pip} for {relative path: text}, each file read by its kind, and each file one
    of them names with ``-r`` or ``-c`` -- beside it, or from the root -- read as requirements."""
    readers = {rel: _reader(rel) for rel in texts}
    out, queue = {}, sorted(texts)
    while queue:
        rel = queue.pop()
        specs, named = readers[rel](texts[rel])
        names = sorted(found[0] for found in map(_requirement, specs) if found and found[0] in _GONE)
        if names:
            out[rel] = names
        else:
            out.pop(rel, None)
        for file in named:
            for place in {posixpath.normpath(posixpath.join(posixpath.dirname(rel), file)), posixpath.normpath(file)}:
                if place in texts and readers[place] is not _from_requirements:
                    readers[place] = _from_requirements
                    queue.append(place)
    return out


def _carried():
    """The files git carries, outside ``tests/`` and ``CHANGELOG.md``, none below a folder named ``data`` (never
    opened); a link, or a file gone from the disk, left out."""
    listed = _split_z(_git(_REPO, "ls-files", "-z", "--cached"))
    assert len(listed) > 500 and "opti_oignon/chroma_migration.py" in listed, ("control: the tree", len(listed))
    carried = []
    for rel in listed:
        parts = rel.split("/")
        if parts[0] == "tests" or rel == "CHANGELOG.md" or "data" in parts[:-1]:
            continue
        path = _REPO / rel
        if not path.is_symlink() and path.is_file():
            carried.append(rel)
    return carried


def _text(data):
    """Bytes as text: UTF-16 by its mark, else UTF-8 (its mark dropped); None for what is neither."""
    try:
        if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            return data.decode("utf-16")
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def test_rm1_no_dependency_file_declares_chromadb():
    texts, unread = {}, []
    for rel in _carried():
        text = _text((_REPO / rel).read_bytes())
        if text is not None:
            texts[rel] = text
        elif _reader(rel) in (_from_pyproject, _from_requirements):
            unread.append(rel)
    assert unread == [], ("read as requirements and not text: pip may read them, this census cannot", unread)
    assert {"requirements-backend.txt", "requirements-docs.txt", "constraints.txt", "Dockerfile.backend",
            "docker-compose.yml", ".github/workflows/ci.yml", "docs/getting-started/installation.md",
            "opti_oignon/cli/client.py"} <= set(texts), ("control: the files that declare or install", len(texts))
    assert [_reader(rel) for rel in ("requirements-docs.txt", ".github/workflows/ci.yml", "scripts/x.py", "README.md")] \
        == [_from_requirements, _from_yaml, _from_python, _from_commands]
    assert ("uvicorn", frozenset({"standard"}), ">=0.24.0") in map(_requirement, _from_pyproject(texts["pyproject.toml"])[0])
    assert "requirements-backend.txt" in _from_commands(texts["Dockerfile.backend"])[1], "control: a -r is read"
    assert "pytest" in _from_yaml(texts[".github/workflows/ci.yml"])[0], "control: the workflow's installs are read"
    assert _gone(texts) == {}, ("chromadb declared or installed", _gone(texts))
    # Control: every form a declaration or an install takes is found; an option's argument, a longer name, another
    # pip command, a comment and a bare URL are none.
    planted = {
        "pyproject.toml": (
            '[project]\ndependencies = ["fastapi", "chromadb>=0.4.0"]\n'
            '[project.optional-dependencies]\nstore = ["ChromaDB[server] >= 1.0"]\n'
            '[dependency-groups]\ndev = ["chromadb-client", {include-group = "x"}]\n'
            '[build-system]\nrequires = ["setuptools", "chroma_hnswlib==0.7.6"]\n'),
        "requirements.txt": "-r chromadb.txt\nchromadb==1.3.7 ; python_version >= '3.10'  # the store\nchromadb_like\n",
        "Dockerfile": (
            "RUN pip install --no-cache-dir --prefix /install chromadb-client \\\n    requests\n"
            "# RUN pip install chromadb\nRUN pip install -c chromadb requests\n"
            "RUN pip -q install --no-cache-dir -r deps/base.list ChromaDB\n"
            'RUN ["python3", "-m", "pip", "install", "chroma-hnswlib"]\n'),
        "deps/base.list": "fastapi\nchromadb>=1\n",
        "ci.yml": (
            "jobs:\n  t:\n    steps:\n      - run: |\n"
            "          pip install pip 2>/dev/null && pip install -U 'chromadb>=1'\n"
            "          python -m pip --disable-pip-version-check install 'fastapi >= 0.104' Chroma.HNSWLib\n"
            "      - run: >-\n          pip3.12 install\n          chromadb-client\n"),
        "setup_env.py": (
            'import subprocess, sys\nsubprocess.check_call([sys.executable, "-m", "pip", "install", "ChromaDB"])\n'
            'HINT = "pip install chromadb_like"\n'),
        "docs.md": ("Run `pip uninstall chromadb`, then `pip download chromadb`; never\n"
                    "`pip install git+https://example.org/chroma.git`.\n"),
    }
    assert _gone(planted) == {
        "pyproject.toml": ["chroma-hnswlib", "chromadb", "chromadb", "chromadb-client"],
        "requirements.txt": ["chromadb"],
        "Dockerfile": ["chroma-hnswlib", "chromadb", "chromadb-client"],
        "deps/base.list": ["chromadb"],
        "ci.yml": ["chroma-hnswlib", "chromadb", "chromadb-client"],
        "setup_env.py": ["chromadb"],
    }, _gone(planted)
    assert _text("chromadb\n".encode("utf-16")) == "chromadb\n" and _text(b"\xff\xfe\x00\xd8") is None


def _names_chromadb(value):
    return isinstance(value, str) and (value == "chromadb" or value.startswith("chromadb."))


def _spellings(tree):
    """Each way ``tree`` reaches chromadb, unparsed: an import of it or below it; a call handed a constant that names
    it, whatever is called (an import, a lookup, a probe, a read of its installation); a subscript, or a membership
    test, of the modules by such a constant."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(_names_chromadb(alias.name) for alias in node.names):
            found.append(ast.unparse(node))
        elif isinstance(node, ast.ImportFrom) and _names_chromadb(node.module or ""):
            found.append(ast.unparse(node))
        elif isinstance(node, ast.Call) and any(
                isinstance(arg, ast.Constant) and _names_chromadb(arg.value)
                for arg in [*node.args, *(keyword.value for keyword in node.keywords)]):
            found.append(ast.unparse(node))
        elif isinstance(node, ast.Subscript) and ast.unparse(node.value).endswith("modules") and (
                isinstance(node.slice, ast.Constant) and _names_chromadb(node.slice.value)):
            found.append(ast.unparse(node))
        elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) and _names_chromadb(
                node.left.value) and any(ast.unparse(right).endswith(("modules", "modules.keys()"))
                                         for right in node.comparators):
            found.append(ast.unparse(node))
    return found


def test_rm2_no_module_imports_chromadb_by_any_spelling():
    files = sorted({Path(path) for path in package_modules().values()}
                   | set((_REPO / "scripts").glob("*.py")) | set((_REPO / ".github" / "scripts").glob("*.py")))
    assert len(files) > 100 and _REPO / "opti_oignon" / "chroma_migration.py" in files, ("control: the tree", len(files))
    reached = {}
    for path in files:
        spellings = _spellings(ast.parse(path.read_text(encoding="utf-8")))
        if spellings:
            reached[path.relative_to(_REPO).as_posix()] = spellings
    assert reached == {}, ("chromadb reached", reached)
    # Control: every spelling the census looks for is found where it is written.
    planted = ("import chromadb\nimport chromadb.config as c\nfrom chromadb import PersistentClient\n"
               "from chromadb.config import Settings\nimport importlib\nimportlib.import_module('chromadb')\n"
               "__import__('chromadb.api')\nimportlib.util.find_spec('chromadb')\nsys.modules['chromadb']\n"
               "sys.modules.get('chromadb')\n__import__(name='chromadb')\nimportlib.import_module(name='chromadb')\n"
               "importlib.util.find_spec(name='chromadb')\n'chromadb' in sys.modules\n"
               "importlib.metadata.version('chromadb')\nversion('chromadb')\nmd.version('chromadb')\n"
               "'chromadb' in sys.modules.keys()\nsys.modules.__contains__('chromadb')\n"
               "sys.modules.setdefault('chromadb', None)\nsys.modules.pop('chromadb')\n"
               "importlib.metadata.distribution('chromadb')\n")
    assert sorted(_spellings(ast.parse(planted))) == sorted([
        "import chromadb", "import chromadb.config as c", "from chromadb import PersistentClient",
        "from chromadb.config import Settings", "importlib.import_module('chromadb')", "__import__('chromadb.api')",
        "importlib.util.find_spec('chromadb')", "sys.modules['chromadb']", "sys.modules.get('chromadb')",
        "__import__(name='chromadb')", "importlib.import_module(name='chromadb')",
        "importlib.util.find_spec(name='chromadb')", "'chromadb' in sys.modules",
        "importlib.metadata.version('chromadb')", "version('chromadb')", "md.version('chromadb')",
        "'chromadb' in sys.modules.keys()", "sys.modules.__contains__('chromadb')",
        "sys.modules.setdefault('chromadb', None)", "sys.modules.pop('chromadb')",
        "importlib.metadata.distribution('chromadb')"]), _spellings(ast.parse(planted))
    assert _spellings(ast.parse("x = 'chromadb is gone'\nimport chromadb_like\nfrom chroma import y\n")) == [], (
        "a string or another module is no spelling")


def test_rm3_the_real_tree_is_green_and_chromadb_is_no_classified_library(tmp_path):
    guard, restore = _load()
    try:
        result = guard.run(_REPO)
        # c1: the census on the repository is green, with its green line.
        assert result.code == 0, "\n".join(result.lines)
        assert result.lines[-1].startswith("Egress census OK: "), result.lines[-1]
        # c2: the ratchet, as properties.
        assert guard.find_home_proofs_missing(result.estate, result.census) == []
        assert set(guard.LEDGER) <= set(BIRTH), f"owed modules beyond the birth: {sorted(set(guard.LEDGER) - set(BIRTH))}"
        for rel, kinds in guard.LEDGER.items():
            for kind, count in kinds.items():
                assert count <= BIRTH[rel].get(kind, 0), f"{rel}: {kind} {count} is above its birth"
        assert guard.EXEMPT and all(isinstance(r, str) and r.strip() for r in guard.EXEMPT.values())
        # c3: the capacity anchor, less the sinks retired by name, each gone.
        retired = 0
        for rel, kinds in RETIRED.items():
            for kind, count in kinds.items():
                assert 0 < count <= BIRTH[rel].get(kind, 0), f"{rel}: {kind} was never owed {count}"
                assert guard.LEDGER.get(rel, {}).get(kind, 0) == 0, f"{rel}: {kind} is retired and still owed"
                left = [site.line for site in result.census.get(rel, ()) if site.kind == kind]
                assert left == [], f"{rel}: {kind} is retired and still sinks at {left}"
                retired += count
        total = sum(len(sites) for sites in result.census.values())
        assert FLOOR - retired > 0 and total >= FLOOR - retired, f"{total} sink site(s), floor {FLOOR} less {retired}"
        for rel in guard.LEDGER:
            assert len(result.census.get(rel, ())) >= 1, f"{rel} is owed and has no sink"
        # c4: every third-party name EC12 named is classified, but chromadb.
        named = tuple(name for name in _THIRD_PARTY if name != "chromadb")
        assert len(named) == 37
        assert [n for n in named if n not in guard.LIBRARIES] == []
        imported = guard.third_party_imports(result.estate)
        classified = [n for n in imported if n in guard.LIBRARIES]
        assert len(classified) >= 30, f"only {len(classified)} classified import(s) seen"
        # c5: chromadb is no classified library, and an import of it in the package is refused by name.
        assert "chromadb" not in guard.LIBRARIES and "chromadb" not in imported
        files = {"opti_oignon/__init__.py": "", "opti_oignon/reader.py": "import chromadb\n"}
        planted = guard.read_estate(_write(tmp_path / "planted", files))
        assert guard.find_unclassified_imports(planted) == [
            "chromadb: a third-party import that LIBRARIES does not classify"]
        # Control: the same estate importing a library LIBRARIES classifies is clean.
        files["opti_oignon/reader.py"] = "import yaml\n"
        assert guard.find_unclassified_imports(guard.read_estate(_write(tmp_path / "control", files))) == []
        # c6: the census still counts chromadb's sinks, a detector for the day it comes back.
        assert guard.count_sinks("import chromadb\n\ndef f():\n    return chromadb.HttpClient(host='h')\n") == 1
    finally:
        restore()


def _tokens(data):
    """How many times ``data`` writes the token: as its text when it begins with UTF-16's mark."""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = _text(data)
        if text is not None:
            data = text.encode("utf-8")
    return len(_TOKEN.findall(data))


def test_rm5_the_package_name_survives_only_in_detectors_and_history():
    found = {}
    for rel in _carried():
        count = _tokens((_REPO / rel).read_bytes())
        if count:
            found[rel] = count
    assert found == _SURVIVORS, ("where chromadb is written", found)
    # Control: the token is counted where it is written, never inside a longer name, in UTF-16 too.
    sample = b"import chromadb\nchromadb.Client()\n'chromadb'\ntest_chromadb_x chromadb_y ychromadb chromadbs\n"
    assert _tokens(sample) == 3 and _tokens(sample.decode("utf-8").encode("utf-16")) == 3


def _websocket_facts():
    """({module: its WebSocket routes}, [modules importing websockets]) of the package, from its sources."""
    routes, importers = {}, []
    for path in sorted({Path(path) for path in package_modules().values()}):
        rel = path.relative_to(_REPO).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and (
                            decorator.func.attr == "websocket"):
                        routes.setdefault(rel, []).append(node.name)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in (
                    "add_api_websocket_route", "add_websocket_route"):
                routes.setdefault(rel, []).append(ast.unparse(node.args[0]) if node.args else "?")
            elif (isinstance(node, ast.Import) and any(a.name.split(".")[0] == "websockets" for a in node.names)) or (
                    isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "websockets"):
                importers.append(rel)
    return routes, sorted(set(importers))


def test_rm6_the_websocket_routes_keep_their_server_stack_declared():
    routes, importers = _websocket_facts()
    # c1, control: the facts the declarations stand on -- the chat's stream among six WebSocket routes, and the
    # command-line client's import.
    assert sum(map(len, routes.values())) >= 6 and "opti_oignon/api/routes_chat.py" in routes, routes
    assert "opti_oignon/cli/client.py" in importers, importers
    # c2: pyproject declares, exactly, the server's standard stack, which carries its WebSocket implementation, and
    # the library the client imports: no marker that could leave either out, no bound below the stack's own.
    declared = {name: (extras, rest) for name, extras, rest in map(
        _requirement, tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"])}
    assert declared.get("uvicorn") == (frozenset({"standard"}), ">=0.24.0"), ("uvicorn", declared.get("uvicorn"))
    assert declared.get("websockets") == (frozenset(), ">=10.4"), ("websockets", declared.get("websockets"))
    # c3: the image installs what pip installs: requirements-backend.txt declares exactly pyproject's dependencies.
    backend = [found for found in map(_requirement, _from_requirements(
        (_REPO / "requirements-backend.txt").read_text(encoding="utf-8"))[0]) if found]
    project = [(name, extras, rest) for name, (extras, rest) in declared.items()]
    assert len(backend) == len({name for name, _e, _r in backend}), "a requirement named twice"
    assert sorted(backend) == sorted(project), (sorted(set(backend) ^ set(project)))


def _paragraph(text, opening):
    """The paragraph of ``text`` that begins with ``opening``, its lines joined by single spaces, or ""."""
    for paragraph in re.split(r"\n\s*\n", text):
        if paragraph.lstrip().startswith(opening):
            return " ".join(paragraph.split())
    return ""


def test_rm7_the_docs_say_how_migrate_reads_and_embeds_and_no_page_names_chromadb():
    security = _paragraph((_REPO / "SECURITY.md").read_text(encoding="utf-8"), "Closed exception (RS-01)")
    encryption = _paragraph((_REPO / "docs" / "security" / "encryption.md").read_text(encoding="utf-8"),
                            "The knowledge base --")
    guide = _paragraph((_REPO / "docs" / "user-guide" / "rag.md").read_text(encoding="utf-8"),
                       "Opti-Oignon's RAG system")
    assert security and encryption and guide, "control: the three paragraphs are found"
    # c1: the closed exception and the encryption page say how the migration reads, and that it embeds again.
    for name, said in (("SECURITY.md", security), ("encryption.md", encryption)):
        assert "with the standard library" in said and "read-only" in said, (name, "how it reads", said)
        assert re.search(r"embeds (?:each|every) text again|each text is embedded again", said), (name, said)
        assert "Ollama" in said, (name, "what serves the model", said)
    # c2: the RAG guide says it embeds again with the models the configuration names, which Ollama must serve.
    assert "embeds every text again" in guide and "rag.yaml" in guide and "Ollama must be running" in guide, guide
    # c3: no source of the frontend names ChromaDB, in any case.
    pages = [rel for rel in _carried() if rel.startswith("frontend/src/")]
    assert len(pages) > 50, ("control: the frontend's sources", len(pages))
    named = [rel for rel in pages if b"chromadb" in (_REPO / rel).read_bytes().lower()]
    assert named == [], ("ChromaDB named to the user", named)
