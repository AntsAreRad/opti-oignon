#!/usr/bin/env python3
"""Turn the maintainer's list of forbidden words into the full law's taboo table.

``python3 scripts/allium_taboo_digest.py PATH`` reads a word list from a file
OUTSIDE the repository (one word per line, UTF-8), or, with no path, asks for
the words one at a time with the echo off, until an empty line. The words
are never printed, logged, written to disk or passed on the command line;
only the SHA-256 of each folded word reaches ``tables/taboo_v1.json``.

Each line is decoded as strict UTF-8, stripped, normalised to NFC, folded
by the phonology table's fold list alone (accented Latin letters to their
base letter, capitals to lower case, the typographic apostrophe to ``'``,
spaces and hyphens removed), and kept only if the result is 1 to 12
letters of the componion's alphabet. Dropped lines are reported by cause and
line number, never by content.

Before writing, the script builds the six-word lists (SAS) the new table
would leave for the law's own block, the lowest block and sixty-four
minimum-space blocks, and refuses to write if any of them needs the second,
longer phase or more than 8192 candidates: a taboo list must never leave a
being without words for its recovery phrase.

The digests are unsalted and publish each word's length: they keep a text
scanner and a casual reader from seeing the words, not a determined one.

The table may change only while the full law is provisional and before any
being is sown. After it changes, the law must be re-pinned
(``scripts/allium_author_genome.py``), the native core rebuilt and the
goldens that carry the law's digest re-recorded; the script prints the list.
"""

import argparse
import getpass
import hashlib
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT.joinpath("opti_oignon", "allium", "tables")
LAWS_DIR = ROOT.joinpath("opti_oignon", "allium", "laws")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opti_oignon.allium import rng, wire  # noqa: E402
from opti_oignon.allium.ref.organs import phon  # noqa: E402

MIN_SPACE_BLOCKS = 64
MAX_CANDIDATES = 8192


def fold(text, folds):
    """The alphabet spelling of ``text``, or None when a character has no fold."""
    out = []
    for char in unicodedata.normalize("NFC", text):
        if char in phon.ALPHABET:
            out.append(char)
        elif ord(char) in folds:
            out.append(folds[ord(char)])
        else:
            return None
    return "".join(out)


def digest_lines(lines, folds):
    """``(entries, dropped)``: sorted unique ``[length, sha256]`` pairs and ``{cause: [line numbers]}``."""
    entries = {}
    dropped = {"not utf-8": [], "outside the alphabet": [], "empty or longer than 12": []}
    for number, raw in enumerate(lines, start=1):
        try:
            text = raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else raw
        except UnicodeDecodeError:
            dropped["not utf-8"].append(number)
            continue
        text = text.rstrip("\r\n")
        if not text.strip():
            continue
        folded = fold(text.strip(), folds)
        if folded is None:
            dropped["outside the alphabet"].append(number)
            continue
        if phon.FORM.fullmatch(folded) is None:
            dropped["empty or longer than 12"].append(number)
            continue
        entries[(len(folded), hashlib.sha256(folded.encode("ascii")).hexdigest())] = True
    return [[length, digest] for length, digest in sorted(entries)], dropped


def capacity(table, entries):
    """The largest candidate count any tested block needs; raises when a list needs its second phase."""
    taboo = phon.taboo_set([tuple(e) for e in entries])
    blocks = [phon.law_lex(table), bytes(low for low, _high in table.lex_box)]
    for i in range(MIN_SPACE_BLOCKS):
        stream = rng.Stream(bytes(32), "test.sas_min", i)
        block = bytearray(low for low, _high in table.lex_box)
        for p in range(phon.WEIGHTS):
            block[p] = stream.below(128)
        blocks.append(bytes(block))
    worst = 0
    for block in blocks:
        words, candidates, _work = phon.sas_list(phon.decode(block, table), taboo)
        if candidates > phon.SAS["cap"] or candidates > MAX_CANDIDATES or any(len(w) != 6 for w in words):
            raise RuntimeError("this list would leave a block without its short six-word list")
        worst = max(worst, candidates)
    return worst


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", help="a word list outside the repository; omit to type the words")
    args = parser.parse_args(argv)
    table_value = wire.parse(TABLES_DIR.joinpath("phon_v1.json").read_bytes(), lenient=True)
    if phon.validate_table(table_value):
        print("the phonology table does not validate", file=sys.stderr)
        return 2
    table = phon.Table(table_value)
    folds = {code: replacement for code, replacement in table_value["fold"]}
    law = wire.parse(LAWS_DIR.joinpath("v0_1.json").read_bytes(), lenient=True)
    if law.get("provisional") is not True:
        print("the full law is no longer provisional: a new list is a new table under a new law", file=sys.stderr)
        return 2
    if args.path:
        path = Path(args.path).resolve()
        if path == ROOT or ROOT in path.parents:
            print("refused: the word list must live outside the repository", file=sys.stderr)
            return 2
        lines = path.read_bytes().splitlines()
    else:
        lines = []
        print("Type one word per prompt; an empty line ends. Nothing you type is shown or kept.")
        while True:
            word = getpass.getpass("word: ")
            if not word:
                break
            lines.append(word)
    entries, dropped = digest_lines(lines, folds)
    if len(entries) > table.taboo_max:
        print(f"refused: {len(entries)} entries, the table holds at most {table.taboo_max}", file=sys.stderr)
        return 2
    try:
        worst = capacity(table, entries)
    except RuntimeError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    value = {"entries": entries, "name": "taboo_v1", "version": 1}
    if phon.validate_taboo(value, table):
        print("refused: the table would not validate", file=sys.stderr)
        return 2
    sys.path.insert(0, str(ROOT.joinpath("scripts")))
    from allium_author_phon import render
    TABLES_DIR.joinpath("taboo_v1.json").write_text(render(value) + "\n", encoding="ascii")
    print(f"taboo_v1.json: {len(entries)} digests; the six-word lists need at most {worst} candidates")
    for cause, numbers in dropped.items():
        if numbers:
            print(f"dropped, {cause}: {len(numbers)} line(s): {', '.join(str(n) for n in numbers[:40])}")
    print("next: python3 scripts/allium_author_genome.py; bash scripts/build_oo_core.sh; re-record the goldens that "
          "carry the full law's digest (tests/allium_golden/v1/{genome,phon,chassis}.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
