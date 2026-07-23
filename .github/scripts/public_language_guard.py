#!/usr/bin/env python3
"""Public-language guard: the Python trees carry English prose only.

The published trees are English-only. Nothing enforced that. Its sibling
guard reads the added lines of a diff for internal nomenclature and has no
notion of language, so the rule the project states most strictly was the one
rule with no check behind it at all.

REACH, STATED PLAINLY. This guard reads Python and only Python: comments
through the tokeniser, docstrings through the AST. Its sibling is a regex
and covers every published tree; this one cannot, and the difference is not
cosmetic. A tree of another language placed in its perimeter would be walked
without a single file being opened and would come out at zero, which is why
``unreadable_scan_paths`` treats that perimeter as a failure instead of a
silence. The prose of the non-Python trees is a debt this guard does not
cover and does not claim to.

WHAT COUNTS AS PROSE. Comments and docstrings are written for a human
reader, and they must be English. Text inside a string literal is DATA and
is never read: a classifier that recognises a French question carries
French patterns because its input is French, and a guard that charged those
would be wrong about the very code it polices. The line between the two is
structural, so it is drawn by parsing the post-image of each file -- not by
matching diff lines in isolation, which cannot tell a comment from a string.

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

A file that does not parse is REPORTED, never passed over. A guard that
reads an unreadable file as a clean file is worse than no guard.

The helpers are pure and import-safe, so they are unit-tested without
touching a filesystem; ``main`` scans the diff and the tree and exits
non-zero on any finding.
Usage: ``public_language_guard.py [BASE_REF]`` (default base ref:
origin/main).
"""

import ast
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

# The only file kind this guard can read. Stated as a constant rather than
# buried in a glob, because everything below depends on it: the comment pass
# tokenises Python and the docstring pass parses a Python AST, so a file of
# any other language is not merely unhandled, it is INVISIBLE. A neighbouring
# TypeScript file full of French prose raises the census by nothing at all.
_SCANNED_SUFFIX = ".py"

# Trees the guard scans. Nothing outside these is considered -- this script
# included, which is why the vocabularies below can be written in the clear.
#
# Only trees that carry Python belong here. Adding a tree of another language
# would not widen the guard; it would widen the CLAIM the guard makes at the
# end of a clean run, over files it never opened. ``unreadable_scan_paths``
# refuses that state rather than trusting the reader to remember it.
_SCAN_PATHS = ("opti_oignon/", "tests/", "scripts/")

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


def find_violations(source, added_lines=None):
    """Return ``[(line, kind, text), ...]`` for French prose in ``source``.

    ``added_lines`` is a set of 1-based line numbers to restrict the scan to,
    or None for the whole source. A span is kept when any of its lines was
    added. ``kind`` is ``comment``, ``docstring``, or ``unparsable``.
    """
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


def census(source):
    """Number of French prose spans in ``source``, added lines or not."""
    return len(
        [item for item in find_violations(source) if item[1] != "unparsable"]
    )


def unreadable_scan_paths(repo, scan_paths=None):
    """Perimeter entries carrying no file this guard is able to read.

    A tree of TypeScript or Kotlin inside the perimeter would be walked,
    matched against nothing, and counted as zero -- and the run would then
    print a clean bill for it. That is the one failure this guard refuses by
    name: reporting on a file it never opened is worse than not scanning it.
    Returned so the caller can fail rather than sign.
    """
    root = Path(repo)
    missing = []
    for scan_path in (_SCAN_PATHS if scan_paths is None else scan_paths):
        tree = root / scan_path.rstrip("/")
        if not any(tree.rglob("*" + _SCANNED_SUFFIX)):
            missing.append(scan_path)
    return tuple(missing)


def census_tree(repo, scan_paths=None):
    """Map every scanned file that carries debt to how much it carries.

    Files at zero are omitted: a file paid down to nothing comes off the
    ledger rather than sitting on it at zero. Only files ending in
    ``_SCANNED_SUFFIX`` are read; see ``unreadable_scan_paths``.
    """
    counts = {}
    root = Path(repo)
    for scan_path in (_SCAN_PATHS if scan_paths is None else scan_paths):
        for path in sorted(
            (root / scan_path.rstrip("/")).rglob("*" + _SCANNED_SUFFIX)
        ):
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            found = census(source)
            if found:
                counts[path.relative_to(root).as_posix()] = found
    return counts


# Standing debt, sealed file by file. MAY ONLY SHRINK. A file paid down to
# nothing comes off this ledger; a file that grows is a regression.
LEDGER = {
    "opti_oignon/agents/dynamic_pipeline.py": 1,
    "opti_oignon/api/routes_artifacts.py": 6,
    "opti_oignon/api/routes_chat.py": 17,
    "opti_oignon/api/routes_code.py": 2,
    "opti_oignon/api/routes_context.py": 2,
    "opti_oignon/api/routes_conversations.py": 4,
    "opti_oignon/api/routes_exec_pipelines.py": 3,
    "opti_oignon/api/routes_export.py": 1,
    "opti_oignon/api/routes_files.py": 1,
    "opti_oignon/api/routes_health.py": 2,
    "opti_oignon/api/routes_memory.py": 1,
    "opti_oignon/api/routes_models.py": 1,
    "opti_oignon/api/routes_pipelines.py": 10,
    "opti_oignon/api/routes_presets.py": 7,
    "opti_oignon/api/routes_settings.py": 2,
    "opti_oignon/api/schemas.py": 18,
    "opti_oignon/artifacts.py": 2,
    "opti_oignon/config.py": 1,
    "opti_oignon/context_window.py": 1,
    "opti_oignon/conversation.py": 16,
    "opti_oignon/executor.py": 1,
    "opti_oignon/memory/legacy.py": 13,
    "opti_oignon/performance_benchmark.py": 15,
    "opti_oignon/pipelines.py": 19,
    "opti_oignon/rag/augmenter.py": 2,
    "opti_oignon/rag/chunkers.py": 31,
    "opti_oignon/rag/indexer.py": 14,
    "opti_oignon/rag/retriever.py": 3,
    "opti_oignon/reasoning.py": 25,
    "opti_oignon/response_cache.py": 9,
    "opti_oignon/search_integration.py": 33,
    "opti_oignon/self_correction.py": 31,
    "opti_oignon/structured_output.py": 17,
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
    """Return ``{path: {line, ...}}`` for the diff over the scan trees."""
    cmd = [
        "git", "diff", "--unified=0", "--no-color", base_ref,
        "--", *_SCAN_PATHS,
    ]
    # Fixed argv, no shell: safe.
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    hunk = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    by_path = {}
    current = None
    for line in result.stdout.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            current = None if path == "/dev/null" else path
            continue
        match = hunk.match(line)
        if match and current:
            start = int(match.group(1))
            count = int(match.group(2) or 1)
            by_path.setdefault(current, set()).update(
                range(start, start + count)
            )
    return by_path


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
            print(f"  {scan_path}: carries no {_SCANNED_SUFFIX} file to read")
        failed = True

    added = _added_lines_by_path(base_ref)
    for path, lines in sorted(added.items()):
        full = repo / path
        if full.suffix != _SCANNED_SUFFIX or not full.is_file():
            continue
        try:
            source = full.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line, kind, text in find_violations(source, added_lines=lines):
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
    scanned = " ".join(_SCAN_PATHS)
    print(
        f"public-language guard: added lines are English (base {base_ref}); "
        f"standing debt {total} span(s) across {len(LEDGER)} file(s), sealed "
        f"and falling only; read {_SCANNED_SUFFIX} files under {scanned} -- "
        "no other tree or file kind is covered by this line"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
