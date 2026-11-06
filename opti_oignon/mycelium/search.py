"""Finding in the Mycelium: a question -> cited spans, each with why it is there.

Three lists, fused by their reciprocal ranks (K = 60): the identifiers of the question looked up in the symbol
table (exact, free); the full-text index (FTS5, bm25 over the path, the text, the words inside its identifiers and
what the code says to a person -- raised, logged, printed), a part of the tests keeping a share of its bm25 only (a
test's name is a sentence that holds the words of any question); and the vectors of the active encoder (exact
cosine over the matrix, held while the store's generation holds). With a bridge, the first two are asked the
question in English when it is French, and the full text is asked a second time with the words the question's code
would use (a fourth list, ``rewritten``); without one, ``lexical_query`` is what the first two are asked; the
vectors are asked the question as it was typed. A search may ask the reranker to judge its first passages, read
from the fused list as deep as the reranker judges, then cut to k. A stage that cannot answer -- no encoder active,
the engine missing, the governor holding the question, the bridge's or the reranker's model away -- is silent, and
``silent`` says why: the search answers by the others. Every stage reads only the sources ``allow`` names. Every
result names its part, its lines and the sha256 of its text: ``verify`` says whether the source still says that."""
import hashlib
import itertools

from .lexis import identifiers, match_expression, question_terms
from .vectors import Embedder, Encoder, VectorError, from_blobs

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

    def __init__(self, store, allow=("public", "private"), lexical_query=None, timeout_s=120.0, bridge=None,
                 reranker=None, readers=None):
        """``bridge`` (``bridge.Bridge``) gives the words-matching lists their English and a second, rewritten,
        full-text list; ``reranker`` (``rerank.Reranker``) judges the first passages when a search asks it;
        ``readers`` read a source of documents (``Indexer.readers``)."""
        if bridge is not None and lexical_query is not None:
            raise ValueError("a searcher takes a bridge or a lexical_query, not both: the bridge gives the words")
        self.store = store
        self.allow = tuple(allow)
        self.lexical_query = lexical_query
        self.timeout_s = timeout_s
        self.bridge, self.reranker, self.readers = bridge, reranker, readers
        self.silent = {}
        self._held = (None, None, None)

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

    def _matrix(self):
        """(encoder's {key, card}, part ids, matrix) of the active encoder, read again when the store changed or
        the sensitivities this searcher may read did."""
        held = (self.store.generation(), self.allow)     # what is read, and who may read it
        if self._held[0] != held:
            active = self.store.active_encoder()
            ids, matrix = [], None
            if active is not None:
                allowed, values = self._filter()
                rows = self.store.db.execute(
                    "SELECT p.id, v.data FROM part p JOIN vector v ON v.embed_sha = p.embed_sha AND v.encoder = ? "
                    f"JOIN file f ON f.id = p.file JOIN source s ON s.id = f.source WHERE {allowed} ORDER BY p.id",
                    (active["key"], *values)).fetchall()
                ids, matrix = [r[0] for r in rows], from_blobs([r[1] for r in rows], active["card"]["dim"])
            self._held = (held, active, (ids, matrix))
        return self._held[1], *self._held[2]

    def dense(self, query, n=DEPTH):
        import numpy as np

        active, ids, matrix = self._matrix()
        if active is None:
            self.silent["dense"] = "no encoder is active: no vector pass has completed"
            return []
        if not ids:
            self.silent["dense"] = f"the active encoder {active['key']} has no vector this search may read"
            return []
        try:
            vector = Embedder(Encoder(**active["card"]), self.timeout_s).query(query)
        except VectorError as exc:
            self.silent["dense"] = str(exc)
            return []
        # The question's vector comes as its unit vector (normalized in double precision): a cast never overflows.
        scores = matrix @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores, kind="stable")[:n]
        return [(ids[j], float(scores[j])) for j in order]

    def search(self, query, k=10, stages=("symbols", "lexical", "dense"), rerank=False):
        """The fused, cited passages, k at most; with ``rerank``, the first ones judged by the reranker."""
        self.silent = silent = {}
        worded = query
        if self.bridge is not None and ("symbols" in stages or "lexical" in stages):
            worded = self.bridge.english(query, silent)
        elif self.lexical_query:
            worded = self.lexical_query(query)
        lists = {}
        if "symbols" in stages:
            lists["symbols"] = self.symbols(worded)
        if "lexical" in stages:
            lists["lexical"] = self.lexical(worded)
            said = self.bridge.rewritten(query, silent) if self.bridge is not None else ""
            if said and said != worded:
                lists["rewritten"] = self.lexical(said)
        if "dense" in stages:
            lists["dense"] = self.dense(query)
        fused, why = {}, {}
        for stage, ranked in lists.items():
            for place, (part_id, _score) in enumerate(ranked, 1):
                fused[part_id] = fused.get(part_id, 0.0) + 1.0 / (K + place)
                why.setdefault(part_id, []).append(f"{stage}#{place}")
        order = sorted(fused, key=lambda i: (-fused[i], i))
        # A part gone since the index was read gives no citation: the next ones take its place, k at most -- or as
        # many as the reranker judges, before the cut to k.
        judging = rerank and self.reranker is not None
        if rerank and self.reranker is None:
            silent["rerank"] = "no reranker: the passages keep their fused order"
        want = 0 if k <= 0 else max(k, self.reranker.settings.depth if judging else 0)
        cited = (self.cite(part_id, fused[part_id], why[part_id]) for part_id in order[:want * 4])
        hits = list(itertools.islice((c for c in cited if c is not None), want))
        if judging and hits:
            hits = self.reranker(query, hits, self.read, silent)
        return hits[:max(0, k)]

    def read(self, citation):
        """The text of a citation's lines while its source still says it (``verify``), else None."""
        lines = source_lines(citation, self.readers)
        if lines is None:
            return None
        body = "\n".join(lines[citation["first"] - 1:citation["last"]])
        return body if hashlib.sha256(body.encode("utf-8")).hexdigest() == citation["sha256"] else None

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
