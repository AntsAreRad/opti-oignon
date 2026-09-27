#!/usr/bin/env python3
"""Public-language guard: the Python and Rust trees carry English prose only.

The published trees are English-only. Nothing enforced that. Its sibling
guard reads the added lines of a diff for internal nomenclature and has no
notion of language, so the rule the project states most strictly was the one
rule with no check behind it at all.

REACH, STATED PLAINLY. This guard reads two languages, one per tree. Under
the Python trees it reads ``.py`` files: comments through the tokeniser,
docstrings through the AST. Under ``rust/`` it reads ``.rs`` files: line,
doc and block comments and ``#[doc]`` strings, through the Rust lexer of the
comment-only guard, loaded by path so the two guards cannot disagree about
where a Rust comment ends. A tree is read for its one file kind and a file of
any other kind in it is never opened; Cargo's build directory, a ``target``
beside a ``Cargo.toml``, is skipped. Its sibling is a regex and covers every
published tree; this one cannot, and the difference is not cosmetic. A tree
placed in its perimeter with no file of its kind would be walked without a
single file being opened and would come out at zero, which is why
``unreadable_scan_paths`` treats that perimeter as a failure instead of a
silence -- and names ``rust/`` the same way when the Rust lexer cannot be
loaded. The prose of the other trees (TypeScript, Svelte, Kotlin, shell),
the comments of TOML files, any file under ``rust/`` that is not ``.rs``
(a module a ``#[path]`` attribute points at, a file an ``include!`` or an
``include_str!`` pulls in) and a doc a macro assembles out of its arguments
are a debt this guard does not cover and does not claim to.

WHAT COUNTS AS PROSE. Comments and docstrings are written for a human
reader, and they must be English. Text inside a string literal is DATA and
is never read: a classifier that recognises a French question carries
French patterns because its input is French, and a guard that charged those
would be wrong about the very code it polices. The line between the two is
structural, so it is drawn by parsing the post-image of each file -- not by
matching diff lines in isolation, which cannot tell a comment from a string.
In Rust a ``//`` comment is read line by line, as a Python comment is; a run
of ``///`` or of ``//!`` lines is one passage, as a docstring is; a block
comment, nested to any depth, is one passage; and a ``#[doc = "..."]`` string
is prose, though it is written as a string, because it becomes documentation
-- each string of the value, a ``concat!`` of them included, and only inside
an attribute: a ``format!`` argument named ``doc`` is data like any other.

DENSITY, NEVER A SINGLE WORD. An author's name, a borrowed noun, a domain
term: none of those makes a sentence French. A span is charged only when it
runs long enough to have a grammar at all, carries at least two French
function words, and carries more French than English. Half-translated prose
-- English tokens dropped into French grammar -- is charged, because it is
the shape the standing debt actually takes and it is not English.

TWO QUESTIONS, ONE DETECTOR, DIFFERENT DOMAINS. Added lines answer whether
NEW debt is arriving; the whole file answers whether the STANDING debt has
grown. The first lets the guard be adopted while the debt is still owed,
exactly as its sibling was. The second is the ratchet: every file that
carries debt carries a seal, and a seal may fall and may not rise. A file
paid down to nothing comes off the ledger rather than sitting on it at zero.

A file that does not parse -- or a Rust source whose comment, string or char
is never closed -- is REPORTED, never passed over. A guard that reads an
unreadable file as a clean file is worse than no guard.

The helpers are pure and import-safe, so they are unit-tested without
touching a filesystem; ``main`` scans the diff and the tree and exits
non-zero on any finding.
Usage: ``public_language_guard.py [BASE_REF]`` (default base ref:
origin/main).
"""

import ast
import bisect
import importlib.util
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

# The file kind a tree is read for unless it is named below. Stated as a
# constant rather than buried in a glob, because everything below depends on
# it: the comment pass tokenises Python and the docstring pass parses a Python
# AST, so a file of any other language is not merely unhandled, it is
# INVISIBLE. A neighbouring TypeScript file full of French prose raises the
# census by nothing at all.
_SCANNED_SUFFIX = ".py"

# The one other file kind this guard reads, and the trees read for it.
_RUST_SUFFIX = ".rs"
_TREE_SUFFIX = {"rust/": _RUST_SUFFIX}

# Where the Rust lexer lives: the comment-only guard, beside this script.
_RUST_READER_PATH = Path(__file__).resolve().parent / "comment_only_guard.py"

# Trees the guard scans. Nothing outside these is considered -- this script
# included, which is why the vocabularies below can be written in the clear.
#
# Only trees this guard can read belong here: the Python trees, and the Rust
# tree for its ``.rs`` files. Adding a tree of another language would not
# widen the guard; it would widen the CLAIM the guard makes at the end of a
# clean run, over files it never opened. ``unreadable_scan_paths`` refuses
# that state rather than trusting the reader to remember it.
_SCAN_PATHS = ("opti_oignon/", "tests/", "scripts/", "rust/")

_DEFAULT_BASE_REF = "origin/main"

# A span shorter than this has no grammar to judge.
_MIN_WORDS = 3

# French function words. Words spelled identically in English are excluded
# on purpose: a marker that fires on both languages measures nothing.
_FRENCH = (
    "le", "la", "les", "un", "une", "des", "du", "aux", "pour", "avec",
    "dans", "sur", "par", "est", "sont", "cette", "ces", "mais", "donc",
    "pas", "nombre", "fichier", "fichiers", "liste", "valeur", "valeurs",
    "retourne", "renvoie", "verifie", "vérifie", "calcule", "chaine",
    "chaîne", "ligne", "lignes", "taille", "entree", "entrees", "entrée",
    "entrées", "utilisateur", "contenu", "ici", "sans", "selon", "entre",
    "apres", "après", "avant", "toujours", "jamais", "chaque", "autre",
    "meme", "même", "deja", "déjà", "peut", "doit", "faut", "etre", "être",
    "avoir", "fait", "permet", "evite", "évite", "si", "ne", "que", "qui",
    "dont", "ou", "où", "cle", "clé", "cles", "clés", "repertoire",
    "répertoire", "recuperer", "récupérer", "creer", "créer", "supprimer",
    "ajouter", "charger", "sauvegarder", "gerer", "gérer", "tous", "toutes",
    "leur", "leurs", "notre", "elle", "ils", "elles", "sinon", "alors",
    "depuis", "vers", "aucun", "aucune", "plusieurs", "premier", "premiere",
    "première", "derniere", "dernière", "nouveau", "nouvelle", "champ",
    "champs", "requete", "requête", "reponse", "réponse", "essai",
)

# English function words, the counterweight.
_ENGLISH = (
    "the", "and", "for", "with", "this", "that", "from", "not", "are", "is",
    "to", "of", "in", "on", "it", "as", "be", "by", "or", "an", "a", "we",
    "when", "which", "its", "was", "were", "has", "have", "had", "so",
    "then", "than", "there", "here", "each", "any", "all", "no", "only",
    "into", "over", "under", "before", "after", "while", "because",
)


def _vocabulary(words):
    """Compile a whole-word, case-insensitive alternation over ``words``."""
    return re.compile(
        r"(?<![\w'-])(?:" + "|".join(words) + r")(?![\w'-])", re.IGNORECASE
    )


_FRENCH_RE = _vocabulary(_FRENCH)
_ENGLISH_RE = _vocabulary(_ENGLISH)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def is_french(text):
    """True when ``text`` reads as French prose rather than English.

    Density, not word-spotting: a span must be long enough to have a
    grammar, must carry at least two French function words, and must carry
    more French than English.
    """
    if len(_WORD_RE.findall(text)) < _MIN_WORDS:
        return False
    french = len(_FRENCH_RE.findall(text))
    english = len(_ENGLISH_RE.findall(text))
    return french >= 2 and french > english


def _comment_spans(source):
    """Yield ``(line, line, text)`` for every comment in ``source``."""
    spans = []
    try:
        readline = io.StringIO(source).readline
        for token in tokenize.generate_tokens(readline):
            if token.type == tokenize.COMMENT:
                body = token.string.lstrip("#").strip()
                spans.append((token.start[0], token.start[0], body))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # The AST pass reports an unreadable source; comments are best-effort.
        pass
    return spans


_DOCSTRING_OWNERS = (
    ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
)


def _docstring_spans(tree):
    """Yield ``(start, end, text)`` for every docstring in ``tree``."""
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        text = ast.get_docstring(node)
        if not text:
            continue
        holder = node.body[0]
        spans.append((holder.lineno, holder.end_lineno or holder.lineno, text))
    return spans


def _excerpt(text):
    """The line to quote in a report: the offending one, not merely the first.

    A docstring can open in English and turn French three lines down. Quoting
    its head would point a reader at an innocent line, so the first line that
    is itself French is preferred; a span that is French only in aggregate
    falls back to its head.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    for line in lines:
        if is_french(line):
            return line
    return lines[0]


# ------------------------------------------------------------------ Rust ---

_RUST_READERS = {}


def _rust_reader():
    """The comment-only guard, whose Rust lexer this guard reads Rust with.

    Loaded by path, once per path, and only when a Rust file is read, so no
    Python path depends on it. None when it cannot be loaded: the caller
    must then report the Rust tree unreadable, never read it as clean.
    """
    path = Path(_RUST_READER_PATH)
    key = str(path)
    if key not in _RUST_READERS:
        try:
            spec = importlib.util.spec_from_file_location(
                "_rust_reader_for_language_guard", path,
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name in ("rust_tokens", "rust_text", "RustUnterminated",
                         "RUST_COMMENTS", "RUST_DOCS"):
                getattr(module, name)
        except Exception:  # noqa: BLE001 -- any failure means no reader
            module = None
        _RUST_READERS[key] = module
    return _RUST_READERS[key]


def _rust_literal_text(raw):
    """The text of a Rust string literal, raw or not, quotes removed."""
    body = raw.lstrip("bcr")
    hashes = len(body) - len(body.lstrip("#"))
    body = body[hashes + 1:len(body) - hashes - 1]
    if raw.startswith(("r", "br", "cr")):
        return body
    # A line continuation or a whitespace escape separates words; any other
    # escape keeps its character.
    body = re.sub(r"\\\n\s*|\\[ntr0]", " ", body)
    return re.sub(r"\\(.)", r"\1", body)


def _rust_prose(kind, raw):
    """The prose of one Rust comment or doc string, its markers removed."""
    if kind in ("doc_outer", "doc_inner"):
        return raw[3:].strip()
    if kind == "line":
        return raw.lstrip("/").strip()
    if kind == "doc_attr":
        return _rust_literal_text(raw)
    inner = raw[2:-2]
    if kind != "block":
        inner = inner[1:]
    return "\n".join(
        line.strip().lstrip("*").strip() for line in inner.splitlines()
    )


def _rust_spans(reader, source):
    """``[(start, end, kind, text), ...]`` for the prose of a Rust source.

    A ``//`` comment is one span per line, as a Python comment is. Lines of
    ``///``, or of ``//!``, that stand alone on consecutive lines are one
    span, as a docstring is. A block comment is one span, and so is the
    string of a ``doc`` attribute. ``kind`` is ``comment`` for the plain
    comments and ``docstring`` for everything that becomes documentation.
    Raises the reader's ``RustUnterminated``.
    """
    starts = [0]
    for index, char in enumerate(source):
        if char == "\n":
            starts.append(index + 1)
    spans = []
    run = None  # (token kind, last line, stood alone) of the previous token
    for kind, start, end in reader.rust_tokens(source):
        if kind not in reader.RUST_COMMENTS and kind not in reader.RUST_DOCS:
            continue
        first = bisect.bisect_right(starts, start)
        last = first + source.count("\n", start, end)
        text = _rust_prose(kind, source[start:end])
        alone = not source[starts[first - 1]:start].strip()
        label = "docstring" if kind in reader.RUST_DOCS else "comment"
        if (kind in ("doc_outer", "doc_inner") and alone and run
                and run == (kind, first - 1, True)):
            head, _end, held, joined = spans[-1]
            spans[-1] = (head, last, held, joined + "\n" + text)
        else:
            spans.append((first, last, label, text))
        run = (kind, last, alone)
    return spans


def _find_rust_violations(source, added_lines):
    """``find_violations`` for a Rust source."""
    reader = _rust_reader()
    if reader is None:
        return [(1, "unparsable", "the Rust reader could not be loaded")]
    try:
        spans = _rust_spans(reader, source)
    except reader.RustUnterminated as exc:
        # Unfiltered, as for Python: an unreadable source is a finding
        # whatever the diff touched.
        line = source.count("\n", 0, exc.offset) + 1
        return [(line, "unparsable", str(exc))]
    found = []
    for start, end, kind, text in spans:
        if added_lines is not None and not any(
            line in added_lines for line in range(start, end + 1)
        ):
            continue
        if is_french(text):
            found.append((start, kind, _excerpt(text)))
    return sorted(found, key=lambda item: (item[0], item[1]))


# --------------------------------------------------------------- reading ---

def find_violations(source, added_lines=None, suffix=_SCANNED_SUFFIX):
    """Return ``[(line, kind, text), ...]`` for French prose in ``source``.

    ``added_lines`` is a set of 1-based line numbers to restrict the scan to,
    or None for the whole source. A span is kept when any of its lines was
    added. ``kind`` is ``comment``, ``docstring``, or ``unparsable``.
    ``suffix`` is the file kind ``source`` is read as: ``.py`` or ``.rs``.
    """
    if suffix == _RUST_SUFFIX:
        return _find_rust_violations(source, added_lines)
    if suffix != _SCANNED_SUFFIX:
        raise ValueError(f"no reader for {suffix!r} files")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        # Unfiltered on purpose: a file the guard cannot read is a finding
        # whatever the diff touched, and must never read as a clean file.
        return [(exc.lineno or 1, "unparsable", (exc.msg or "").strip())]

    found = []
    for kind, spans in (
        ("comment", _comment_spans(source)),
        ("docstring", _docstring_spans(tree)),
    ):
        for start, end, text in spans:
            if added_lines is not None and not any(
                line in added_lines for line in range(start, end + 1)
            ):
                continue
            if is_french(text):
                found.append((start, kind, _excerpt(text)))
    return sorted(found, key=lambda item: (item[0], item[1]))


def census(source, suffix=_SCANNED_SUFFIX):
    """Number of French prose spans in ``source``, added lines or not."""
    return len(
        [item for item in find_violations(source, suffix=suffix)
         if item[1] != "unparsable"]
    )


def _suffix_for(scan_path):
    """The file kind a perimeter entry is read for."""
    return _TREE_SUFFIX.get(scan_path.rstrip("/") + "/", _SCANNED_SUFFIX)


def _in_cargo_build(tree, path):
    """True when ``path`` lies in a Cargo build directory under ``tree``.

    Cargo writes its build output, third-party generated sources included,
    into a ``target`` directory beside a ``Cargo.toml``. That directory is
    the machine's, never the product's. A module directory that happens to be
    called ``target`` has no ``Cargo.toml`` beside it and is read.
    """
    parts = path.relative_to(tree).parts[:-1]
    for depth, part in enumerate(parts):
        if part == "target" and (tree.joinpath(*parts[:depth])
                                 / "Cargo.toml").is_file():
            return True
    return False


def _tree_files(root, scan_path):
    """The files of one perimeter entry this guard reads, as the walk finds them.

    Lazy on purpose: asking whether a tree holds any readable file stops at
    the first one, as it always has, instead of walking the whole tree.
    """
    suffix = _suffix_for(scan_path)
    tree = Path(root) / scan_path.rstrip("/")
    for path in tree.rglob("*" + suffix):
        if suffix == _RUST_SUFFIX and _in_cargo_build(tree, path):
            continue
        yield path


def _source_of(path, suffix):
    """The text of ``path`` read as a file of kind ``suffix``.

    A Rust file is read through the Rust reader's ``rust_text``, as the
    compiler reads it: only a CR LF pair is a line break, so a lone CR stays
    in the comment that holds it and cannot push a sentence out of it. Raises
    ``OSError`` or ``UnicodeDecodeError`` as ``read_text`` does.
    """
    path = Path(path)
    if suffix == _RUST_SUFFIX:
        data = path.read_bytes()
        reader = _rust_reader()
        # Without its reader the file is reported unparsable by
        # ``find_violations`` whatever its text; decoding is all that is left.
        return data.decode("utf-8") if reader is None else reader.rust_text(data)
    return path.read_text(encoding="utf-8")


def _read_as(path):
    """The file kind a repository path is read as, or None when it is not.

    A path is read only inside the perimeter and only for its tree's kind:
    a ``.py`` file under ``rust/`` or a ``.rs`` file under a Python tree is
    not opened, exactly as the census walks them.
    """
    for scan_path in _SCAN_PATHS:
        if path.startswith(scan_path.rstrip("/") + "/"):
            suffix = _suffix_for(scan_path)
            return suffix if path.endswith(suffix) else None
    return None


def unreadable_scan_paths(repo, scan_paths=None):
    """Perimeter entries carrying no file this guard is able to read.

    A tree of TypeScript or Kotlin inside the perimeter would be walked,
    matched against nothing, and counted as zero -- and the run would then
    print a clean bill for it. That is the one failure this guard refuses by
    name: reporting on a file it never opened is worse than not scanning it.
    Each tree is asked for its own file kind, and the Rust tree is named as
    well when the Rust lexer cannot be loaded. Returned so the caller can
    fail rather than sign.
    """
    missing = []
    for scan_path in (_SCAN_PATHS if scan_paths is None else scan_paths):
        if _suffix_for(scan_path) == _RUST_SUFFIX and _rust_reader() is None:
            missing.append(scan_path)
        elif not any(True for _path in _tree_files(repo, scan_path)):
            missing.append(scan_path)
    return tuple(missing)


def census_tree(repo, scan_paths=None):
    """Map every scanned file that carries debt to how much it carries.

    Files at zero are omitted: a file paid down to nothing comes off the
    ledger rather than sitting on it at zero. Each tree is read for its own
    file kind only; see ``unreadable_scan_paths``.
    """
    counts = {}
    root = Path(repo)
    for scan_path in (_SCAN_PATHS if scan_paths is None else scan_paths):
        suffix = _suffix_for(scan_path)
        for path in sorted(_tree_files(root, scan_path)):
            try:
                source = _source_of(path, suffix)
            except (OSError, UnicodeDecodeError):
                continue
            found = census(source, suffix)
            if found:
                counts[path.relative_to(root).as_posix()] = found
    return counts


# Standing debt, sealed file by file. MAY ONLY SHRINK. A file paid down to
# nothing comes off this ledger; a file that grows is a regression.
LEDGER = {
    "opti_oignon/api/routes_artifacts.py": 5,
    "opti_oignon/api/routes_chat.py": 17,
    "opti_oignon/api/routes_conversations.py": 4,
    "opti_oignon/api/routes_pipelines.py": 10,
    "opti_oignon/api/routes_presets.py": 7,
    "opti_oignon/api/schemas.py": 18,
    "opti_oignon/conversation.py": 16,
    "opti_oignon/memory/legacy.py": 13,
    "opti_oignon/performance_benchmark.py": 15,
    "opti_oignon/pipelines.py": 17,
    "opti_oignon/rag/chunkers.py": 31,
    "opti_oignon/rag/indexer.py": 14,
    "opti_oignon/reasoning.py": 24,
    "opti_oignon/response_cache.py": 9,
    "opti_oignon/search_integration.py": 33,
    "opti_oignon/self_correction.py": 31,
    "opti_oignon/structured_output.py": 15,
    "opti_oignon/verification.py": 16,
}


def find_ledger_regressions(counts, ledger=None):
    """Return ``[(path, sealed, actual), ...]`` for debt that grew.

    A file the ledger does not name is sealed at zero, so any debt it
    carries is new debt. Falling below a seal is never a finding.
    """
    ledger = LEDGER if ledger is None else ledger
    regressions = []
    for path, actual in sorted(counts.items()):
        sealed = ledger.get(path, 0)
        if actual > sealed:
            regressions.append((path, sealed, actual))
    return regressions


def _added_lines_by_path(base_ref):
    """Return ``{path: {line, ...}}`` for the diff over the scan trees.

    The diff is read as the clean guard reads it: as bytes, cut at newlines
    alone, and with a file header only before a file's first hunk. Split
    anywhere else, or with an added line opening with ``++ `` taken for a
    header, the rest of the diff could be handed to another path.
    """
    cmd = [
        "git", "diff", "--unified=0", "--no-color", base_ref,
        "--", *_SCAN_PATHS,
    ]
    # Fixed argv, no shell: safe.
    result = subprocess.run(cmd, capture_output=True, check=False)

    hunk = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    by_path = {}
    current = None
    in_hunk = False
    for line in result.stdout.decode("utf-8").split("\n"):
        if line.startswith("diff --git "):
            in_hunk = False
            continue
        if not in_hunk and line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            current = None if path == "/dev/null" else path
            continue
        match = hunk.match(line)
        if match:
            in_hunk = True
        if match and current:
            start = int(match.group(1))
            count = int(match.group(2) or 1)
            by_path.setdefault(current, set()).update(
                range(start, start + count)
            )
    return by_path


def added_violations(repo, added):
    """``[(path, line, kind, text), ...]`` for prose on the added lines.

    ``added`` maps a repository path to the set of its added line numbers,
    as ``_added_lines_by_path`` returns it. Each file is read as the kind
    its tree is read for; a path outside the perimeter, a file of another
    kind, or a file that is gone is not opened.
    """
    root = Path(repo)
    found = []
    for path, lines in sorted(added.items()):
        suffix = _read_as(path)
        full = root / path
        if suffix is None or not full.is_file():
            continue
        try:
            source = _source_of(full, suffix)
        except (OSError, UnicodeDecodeError):
            continue
        for line, kind, text in find_violations(
            source, added_lines=lines, suffix=suffix,
        ):
            found.append((path, line, kind, text))
    return found


def main(argv=None):
    """Scan the diff and the tree; exit non-zero on any finding."""
    argv = list(sys.argv[1:] if argv is None else argv)
    base_ref = argv[0] if argv else _DEFAULT_BASE_REF
    repo = Path(__file__).resolve().parent.parent.parent

    failed = False

    opaque = unreadable_scan_paths(repo)
    if opaque:
        print(
            "public-language guard: FAILED -- the perimeter names trees this "
            "guard cannot read, and a clean run would sign for them:"
        )
        for scan_path in opaque:
            suffix = _suffix_for(scan_path)
            if suffix == _RUST_SUFFIX and _rust_reader() is None:
                print(f"  {scan_path}: the Rust lexer could not be loaded "
                      f"from {Path(_RUST_READER_PATH).name}")
            else:
                print(f"  {scan_path}: carries no {suffix} file to read")
        failed = True

    added = _added_lines_by_path(base_ref)
    for path, line, kind, text in added_violations(repo, added):
        if not failed:
            print(
                "public-language guard: FAILED -- non-English prose on "
                "added lines:"
            )
            failed = True
        print(f"  {path}:{line} [{kind}]: {text}")

    regressions = find_ledger_regressions(census_tree(repo))
    if regressions:
        print("public-language guard: FAILED -- the standing debt grew:")
        for path, sealed, actual in regressions:
            print(f"  {path}: sealed at {sealed}, now carries {actual}")
        failed = True

    if failed:
        return 1

    total = sum(LEDGER.values())
    trees = {}
    for scan_path in _SCAN_PATHS:
        trees.setdefault(_suffix_for(scan_path), []).append(scan_path)
    scanned = " and ".join(
        f"{suffix} files under {' '.join(paths)}"
        for suffix, paths in trees.items()
    )
    print(
        f"public-language guard: added lines are English (base {base_ref}); "
        f"standing debt {total} span(s) across {len(LEDGER)} file(s), sealed "
        f"and falling only; read {scanned}, Cargo build output skipped -- "
        "no other tree or file kind, no TOML comment, no string literal, no "
        "file a Rust source includes and no doc a macro builds is covered by "
        "this line"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
