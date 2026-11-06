"""Finding in the Mycelium: a question -> cited spans, each with why it is there.

Two lists, fused by their reciprocal ranks (K = 60): the identifiers of the question looked up in the symbol table
(exact, free), and the full-text index (FTS5, bm25 over the path, the text, the words inside its identifiers and
what the code says to a person -- raised, logged, printed), a part of the tests keeping a share of its bm25 only (a
test's name is a sentence that holds the words of any question). ``lexical_query`` is what both lists are asked (a
question put into English, say). Every result names its part, its lines and the sha256 of its text: ``verify``
says whether the source still says that."""
import hashlib

from .lexis import identifiers, match_expression, question_terms

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

K = 60
DEPTH = 100
MAX_DEFINITIONS = 6
DOCUMENTS = "documents:"


class Searcher:
    # bm25 weights of the token index's columns: path, body, words of the identifiers, what the code says.
    WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    # The share of its bm25 a part of the tests keeps in the full text's ranking (1.0: no discount).
    TESTS_KEEP = 0.7

    def __init__(self, store, allow=("public", "private"), lexical_query=None):
        self.store = store
        self.allow = tuple(allow)
        self.lexical_query = lexical_query

    def _filter(self):
        """The sensitivity filter as SQL over the part's source (``s``), and its values: applied in each stage's own
        query, before its limit, so a secret source neither shows a part nor pushes a readable one out."""
        return f"s.sensitivity IN ({','.join('?' * len(self.allow))})", tuple(self.allow)

    def lexical(self, query, n=DEPTH):
        terms = question_terms(query)
        if not terms:
            return []
        weights = ", ".join(str(float(w)) for w in self.WEIGHTS)
        allowed, values = self._filter()
        rows = self.store.db.execute(
            f"SELECT fts.rowid, bm25(fts, {weights}) AS r FROM fts JOIN part p ON p.id = fts.rowid "
            f"JOIN file f ON f.id = p.file JOIN source s ON s.id = f.source WHERE fts MATCH ? AND {allowed} "
            f"ORDER BY r LIMIT ?", (match_expression(terms), *values, n * 2)).fetchall()
        found = [(r[0], -r[1]) for r in rows]
        if self.TESTS_KEEP != 1.0 and found:
            marks = ",".join("?" * len(found))
            tests = {r[0] for r in self.store.db.execute(
                f"SELECT p.id FROM part p JOIN file f ON f.id = p.file WHERE p.id IN ({marks}) AND "
                f"(f.path LIKE 'tests/%' OR f.path LIKE '%/tests/%')", [i for i, _s in found])}
            found = sorted(((i, s * self.TESTS_KEEP if i in tests else s) for i, s in found), key=lambda x: -x[1])
        return found[:n]

    def symbols(self, query, n=20):
        found = []
        allowed, values = self._filter()
        for name in identifiers(query):
            # A name defined too often says little: counted over the definitions this caller may read alone.
            rows = self.store.db.execute(
                f"SELECT y.part FROM symbol y JOIN part p ON p.id = y.part JOIN file f ON f.id = p.file "
                f"JOIN source s ON s.id = f.source WHERE y.name = ? AND {allowed} LIMIT ?",
                (name, *values, MAX_DEFINITIONS + 1)).fetchall()
            if len(rows) <= MAX_DEFINITIONS:
                found.extend(r[0] for r in rows if r[0] not in found)
        return [(i, 1.0) for i in found][:n]

    def search(self, query, k=10, stages=("symbols", "lexical")):
        worded = self.lexical_query(query) if self.lexical_query else query
        lists = {}
        if "symbols" in stages:
            lists["symbols"] = self.symbols(worded)
        if "lexical" in stages:
            lists["lexical"] = self.lexical(worded)
        fused, why = {}, {}
        for stage, ranked in lists.items():
            for place, (part_id, _score) in enumerate(ranked, 1):
                fused[part_id] = fused.get(part_id, 0.0) + 1.0 / (K + place)
                why.setdefault(part_id, []).append(f"{stage}#{place}")
        order = sorted(fused, key=lambda i: (-fused[i], i))
        cited = (self.cite(part_id, fused[part_id], why[part_id]) for part_id in order[:k * 4])
        return [c for c in cited if c is not None]

    def cite(self, part_id, score, why):
        """The citation of a part, or None when a pass removed it since a stage found it."""
        row = self.store.db.execute(
            "SELECT f.path, p.first, p.last, p.kind, p.symbol, p.sha256, s.root, s.name FROM part p JOIN file f "
            "ON f.id = p.file JOIN source s ON s.id = f.source WHERE p.id = ?", (part_id,)).fetchone()
        if row is None:
            return None
        path, first, last, kind, symbol, sha, root, source = row
        return {"part": part_id, "source": source, "root": root, "path": path, "first": first, "last": last,
                "kind": kind, "symbol": symbol, "sha256": sha, "score": round(score, 6), "why": why}


def verify(citation, readers=None):
    """'fresh' when the source's lines still hash to the citation's sha256; 'modified' when they do not; 'gone'
    when the file is not there. A source of documents is read through ``readers[NAME]`` (``Indexer.readers``);
    without one, nothing can be checked: 'unchecked', never 'fresh'."""
    from .index import UNKNOWN

    root = citation["root"]
    if root.startswith(DOCUMENTS):
        reader = (readers or {}).get(root[len(DOCUMENTS):])
        if reader is None or reader(citation["path"]) is UNKNOWN:
            return "unchecked"      # no reader, or one of an indexer that never held this document
    lines = source_lines(citation, readers)
    if lines is None:
        return "gone"
    body = "\n".join(lines[citation["first"] - 1:citation["last"]])
    return "fresh" if hashlib.sha256(body.encode("utf-8")).hexdigest() == citation["sha256"] else "modified"


def source_lines(citation, readers=None):
    """The lines of a citation's file as its source holds them now, read as the index read them (``\\r`` kept),
    or None when they cannot be had: a link, a path out of the root, not a regular file, past the size bound
    (``index.file_lines``)."""
    from .index import UNKNOWN, file_lines

    root = citation["root"]
    if root.startswith(DOCUMENTS):
        reader = (readers or {}).get(root[len(DOCUMENTS):])
        text = reader(citation["path"]) if reader is not None else None
        return None if text is None or text is UNKNOWN else text.split("\n")
    return file_lines(root, citation["path"])
