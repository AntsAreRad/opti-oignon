"""The map's deterministic skeleton (P20, phase 2, stage 1: the skeleton first, free, with no model): a corpus as the
tree of its folders and files, each node said in a few lines drawn from the store alone -- how many files and parts,
which languages; a file's first sentence (its module's docstring, its first heading), what it defines, the settings
it reads, the tests that name what it defines, the docs that name it, the files that call into it. No model writes
any of it: it is never wrong about what it says, and it is as fresh as the last indexing pass.

  folder(store, source, path)     a folder: its sub-folders and files, biggest first
  file_card(store, source, path)  a file: its parts and its links to the rest of the corpus, or None
  branch(store, source, path)     the node at ``path`` in a few lines, bounded, for a model or a person

The agent walks it with ``mycelium_map`` (``tool.py``) instead of loading everything (P20, point 5). Reads the store
only; writes nothing; every query is bound by parameters, never by text built from a path.
"""
from collections import Counter, defaultdict

MAX_CHILDREN = 25       # sub-folders, and files, said of one folder
MAX_PARTS = 30          # parts said of one file
MAX_LINKS = 8           # files said for each kind of link
MAX_SETTINGS = 15       # settings said of one file
MAX_SENTENCE = 110      # characters of a file's first sentence in a folder's listing


class NoSource(KeyError):
    """The store holds no source of that name."""


def specific_setting(store, name, sid):
    """Whether ``name`` names a setting for sure in the source ``sid``: a key with an underscore, or of 8 letters or
    more, defined in at most 3 places of that source (``call_timeout_s``, ``span_turns``; not ``code``, ``enabled``,
    ``context``, keys of a dozen files). Another source of the store is never read."""
    if "_" not in name and len(name) < 8:
        return False
    held = store.db.execute(
        "SELECT COUNT(*) FROM symbol s JOIN part p ON p.id = s.part JOIN file f ON f.id = p.file "
        "WHERE s.name = ? AND s.kind = 'setting' AND f.source = ?", (name, sid)).fetchone()
    return 0 < held[0] <= 3


def _source_id(store, source):
    """The id of ``source``, read now and only while it is public or private: a source marked secret since, or gone,
    is ``NoSource`` (the search filters the same way at each query)."""
    row = store.db.execute("SELECT id FROM source WHERE name = ? AND sensitivity IN ('public', 'private')",
                           (source,)).fetchone()
    if row is None:
        raise NoSource(source)
    return row[0]


def _first_sentence(store, file_id):
    row = store.db.execute("SELECT module_doc FROM file WHERE id = ?", (file_id,)).fetchone()
    if row and row[0]:
        return row[0]
    head = store.db.execute("SELECT symbol FROM part WHERE file = ? AND kind = 'section' ORDER BY first LIMIT 1",
                            (file_id,)).fetchone()
    return head[0] if head else ""


def _escaped(prefix):
    """``prefix`` for a LIKE pattern: its wildcards and the escape character taken literally."""
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def folder(store, source, path=""):
    """A folder of ``source``: (sub-folders [(name, files, parts, languages)], files [(name, parts, sentence,
    file id)]), each biggest first. Raises ``NoSource`` when the store holds no such source."""
    sid = _source_id(store, source)
    prefix = path.strip("/")
    prefix = prefix + "/" if prefix else ""
    rows = store.db.execute(
        "SELECT f.id, f.path, f.lang, COUNT(p.id) FROM file f LEFT JOIN part p ON p.file = f.id "
        "WHERE f.source = ? AND f.path LIKE ? ESCAPE '\\' AND substr(f.path, 1, ?) = ? GROUP BY f.id",
        (sid, _escaped(prefix) + "%", len(prefix), prefix)).fetchall()     # LIKE ignores case: the prefix exact
    folders, files = defaultdict(lambda: [0, 0, Counter()]), []
    for file_id, rel, lang, parts in rows:
        rest = rel[len(prefix):]
        if "/" in rest:
            name = rest.split("/", 1)[0]
            folders[name][0] += 1
            folders[name][1] += parts
            folders[name][2][lang] += 1
        else:
            files.append((rest, parts, _first_sentence(store, file_id), file_id))
    subs = sorted(((name, n, p, ", ".join(f"{k} {v}" for k, v in langs.most_common(3)))
                   for name, (n, p, langs) in folders.items()), key=lambda r: (-r[1], r[0]))
    files.sort(key=lambda r: (-r[1], r[0]))
    return subs, files


def file_card(store, source, path):
    """A file of ``source``: its parts (kind, symbol, first line, last line), what it defines, the settings it reads,
    and the files that name what it defines -- tests, docs, the others -- each with how many of their parts do; None
    when no such file is indexed. Raises ``NoSource`` when the store holds no such source."""
    sid = _source_id(store, source)
    row = store.db.execute("SELECT id FROM file WHERE source = ? AND path = ?", (sid, path)).fetchone()
    if row is None:
        return None
    file_id = row[0]
    parts = store.db.execute("SELECT id, kind, symbol, first, last FROM part WHERE file = ? ORDER BY first",
                             (file_id,)).fetchall()
    ids = [p[0] for p in parts]
    marks = ",".join("?" * len(ids)) or "NULL"
    defined = [r[0] for r in store.db.execute(
        f"SELECT DISTINCT name FROM symbol WHERE part IN ({marks}) AND kind != 'module' ORDER BY name", ids)]
    settings = sorted({r[0] for r in store.db.execute(
        f"SELECT DISTINCT name FROM ref WHERE part IN ({marks}) AND kind = 'string'", ids)
        if specific_setting(store, r[0], sid)})
    users = store.db.execute(
        f"SELECT f.path, COUNT(DISTINCT r.part) FROM ref r JOIN part p ON p.id = r.part JOIN file f ON f.id = p.file "
        f"WHERE r.name IN (SELECT name FROM symbol WHERE part IN ({marks}) AND kind != 'module') AND f.id != ? "
        f"AND f.source = ? AND (SELECT COUNT(*) FROM symbol s2 JOIN part p2 ON p2.id = s2.part "
        f"JOIN file f2 ON f2.id = p2.file WHERE s2.name = r.name AND f2.source = ?) <= 6 "
        f"GROUP BY f.path ORDER BY 2 DESC, 1", ids + [file_id, sid, sid]).fetchall()
    tests = [(p, n) for p, n in users if p.startswith("tests/") or "/tests/" in p]
    docs = [(p, n) for p, n in users if p.endswith(".md")]
    callers = [(p, n) for p, n in users if (p, n) not in tests and (p, n) not in docs]
    return {"path": path, "sentence": _first_sentence(store, file_id),
            "parts": [(k, s, a, b) for _i, k, s, a, b in parts], "defines": defined, "reads_settings": settings,
            "tested_by": tests, "documented_by": docs, "used_by": callers}


def branch(store, source, path=""):
    """The node at ``path`` of ``source`` -- a file, or a folder -- in a few lines, bounded, for a model or a person:
    what the store holds, said; nothing indexed there, said. Raises ``NoSource`` when the store holds no such source."""
    where = path.strip("/")
    card = file_card(store, source, where) if where else None
    if card is not None:
        out = [f"{card['path']}: {card['sentence']}" if card["sentence"] else card["path"]]
        out.append(f"parts ({len(card['parts'])}):")
        for kind, symbol, first, last in card["parts"][:MAX_PARTS]:
            out.append(f"  {first}-{last} {kind.split('~')[0]} {symbol}")
        if len(card["parts"]) > MAX_PARTS:
            out.append(f"  ... {len(card['parts']) - MAX_PARTS} more")
        if card["reads_settings"]:
            out.append("reads settings: " + ", ".join(card["reads_settings"][:MAX_SETTINGS])
                       + (f", ... {len(card['reads_settings']) - MAX_SETTINGS} more"
                          if len(card["reads_settings"]) > MAX_SETTINGS else ""))
        for label, key in (("tested by", "tested_by"), ("documented by", "documented_by"), ("used by", "used_by")):
            if card[key]:
                out.append(f"{label}: " + ", ".join(f"{p} ({n})" for p, n in card[key][:MAX_LINKS])
                           + (f", ... {len(card[key]) - MAX_LINKS} more" if len(card[key]) > MAX_LINKS else ""))
        return "\n".join(out)
    subs, files = folder(store, source, where)
    if not subs and not files:
        return f"{source}:{where or '/'}: nothing indexed here."
    out = [f"{source}:{where or '/'}"]
    for name, n, parts, langs in subs[:MAX_CHILDREN]:
        out.append(f"  {name}/  {n} files, {parts} parts ({langs})")
    if len(subs) > MAX_CHILDREN:
        out.append(f"  ... {len(subs) - MAX_CHILDREN} more folders")
    for name, parts, sentence, _id in files[:MAX_CHILDREN]:
        out.append(f"  {name}  {parts} parts" + (f": {sentence[:MAX_SENTENCE]}" if sentence else ""))
    if len(files) > MAX_CHILDREN:
        out.append(f"  ... {len(files) - MAX_CHILDREN} more files")
    return "\n".join(out)
