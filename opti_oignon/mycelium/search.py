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
the search says why with its passages (``find``: ``Found(hits, silent)``, the silences its own; ``search`` gives the
passages and keeps the last silences on the searcher): the search answers by the others. Every stage reads only the
sources of the sensitivities ``allow`` names -- and, with ``sources``, only the sources named. The vectors' matrix is
the store's, one per set of sensitivities and of sources read, shared by every searcher of the store and read again
when its generation moves. Every result names its part, its lines and the
sha256 of its text: ``Searcher.verify`` says whether the source still says that, a citation read through the store
(``Searcher.lines``) -- never through the root or the path it carries, which a model may have written."""
import hashlib
import itertools
import threading
from collections import namedtuple

from .lexis import identifiers, match_expression, question_terms
from .vectors import Embedder, Encoder, VectorError, from_blobs

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

K = 60
DEPTH = 100
MAX_DEFINITIONS = 6
DOCUMENTS = "documents:"

# A search's passages, and why each silent stage was silent: its own, whatever other searches run meanwhile.
Found = namedtuple("Found", "hits silent")


class Searcher:
    # bm25 weights of the token index's columns: path, body, words of the identifiers, what the code says.
    WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    # The share of its bm25 a part of the tests keeps in the full text's ranking (1.0: no discount).
    TESTS_KEEP = 0.7

    def __init__(self, store, allow=("public", "private"), lexical_query=None, timeout_s=120.0, bridge=None,
                 reranker=None, readers=None, sources=None):
        """``bridge`` (``bridge.Bridge``) gives the words-matching lists their English and a second, rewritten,
        full-text list; ``reranker`` (``rerank.Reranker``) judges the first passages when a search asks it;
        ``readers`` read a source of documents (``Indexer.readers``); ``sources``, when given, the names of the only
        sources every stage and every citation read back may read (the agent's tool: those of the run's workspace)."""
        if bridge is not None and lexical_query is not None:
            raise ValueError("a searcher takes a bridge or a lexical_query, not both: the bridge gives the words")
        self.store = store
        self.allow = tuple(allow)
        self.sources = None if sources is None else tuple(sources)
        self.lexical_query = lexical_query
        self.timeout_s = timeout_s
        self.bridge, self.reranker, self.readers = bridge, reranker, readers
        self.silent = {}
        self._searching = threading.local()     # the silences of the search this thread runs (``find``)

    def _filter(self):
        """The sensitivity filter as SQL over the part's source (``s``), and its values -- with ``sources``, the
        sources named too: applied in each stage's own query, before its limit, so a source this searcher may not
        read neither shows a part nor pushes a readable one out."""
        clause, values = f"s.sensitivity IN ({','.join('?' * len(self.allow))})", tuple(self.allow)
        if self.sources is not None:
            clause += f" AND s.name IN ({','.join('?' * len(self.sources))})" if self.sources else " AND 0"
            values += self.sources
        return clause, values

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
        """(encoder's {key, card}, part ids, matrix) of the active encoder, as the store holds it for the sets of
        sensitivities and of sources this searcher may read (``Store.matrices``, whatever their order), read again when
        the store's
        generation moved; None when a pass of this process holds the store past the searcher's ``timeout_s`` (its
        rows are read under the store's lock, never in the middle of a pass)."""
        if not self.store.lock.acquire(timeout=self.timeout_s):
            return None
        try:
            generation = self.store.generation()
            key = (tuple(sorted(set(self.allow))),
                   None if self.sources is None else tuple(sorted(set(self.sources))))
            held = self.store.matrices.get(key)
            if held is None or held[0] != generation:
                active = self.store.active_encoder()
                ids, matrix = [], None
                if active is not None:
                    allowed, values = self._filter()
                    rows = self.store.db.execute(
                        "SELECT p.id, v.data FROM part p JOIN vector v ON v.embed_sha = p.embed_sha AND v.encoder = "
                        f"? JOIN file f ON f.id = p.file JOIN source s ON s.id = f.source WHERE {allowed} ORDER BY "
                        "p.id", (active["key"], *values)).fetchall()
                    ids, matrix = [r[0] for r in rows], from_blobs([r[1] for r in rows], active["card"]["dim"])
                held = (generation, active, ids, matrix)
                self.store.matrices[key] = held
        finally:
            self.store.lock.release()
        return held[1:]

    def dense(self, query, n=DEPTH):
        import numpy as np

        silent = getattr(self._searching, "silent", None)
        silent = self.silent if silent is None else silent
        held = self._matrix()
        if held is None:
            silent["dense"] = f"a pass holds the store past {self.timeout_s:g} s: the vectors are left out"
            return []
        active, ids, matrix = held
        if active is None:
            silent["dense"] = "no encoder is active: no vector pass has completed"
            return []
        if not ids:
            silent["dense"] = f"the active encoder {active['key']} has no vector this search may read"
            return []
        try:
            vector = Embedder(Encoder(**active["card"]), self.timeout_s).query(query)
        except VectorError as exc:
            silent["dense"] = str(exc)
            return []
        # The question's vector comes as its unit vector (normalized in double precision): a cast never overflows.
        scores = matrix @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores, kind="stable")[:n]
        return [(ids[j], float(scores[j])) for j in order]

    def search(self, query, k=10, stages=("symbols", "lexical", "dense"), rerank=False):
        """The fused, cited passages, k at most (``find``'s); the silences kept on the searcher, the last search's."""
        found = self.find(query, k, stages, rerank)
        self.silent = found.silent
        return found.hits

    def find(self, query, k=10, stages=("symbols", "lexical", "dense"), rerank=False):
        """``Found``: the fused, cited passages, k at most -- with ``rerank``, the first ones judged by the
        reranker --, and the silences of this search's own stages."""
        silent = {}
        outer = getattr(self._searching, "silent", None)     # a search run inside another's, on this thread
        self._searching.silent = silent
        try:
            return self._find(query, k, stages, rerank, silent)
        finally:
            self._searching.silent = outer

    def _find(self, query, k, stages, rerank, silent):
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
        return Found(hits[:max(0, k)], silent)

    def held_root(self, citation):
        """The root the store holds for the source a citation names, when the citation is one the store holds: the
        path one of that source's files, the source one this searcher may read, its lines whole numbers in order
        (1 <= first <= last), its sha256 a text; else None. Never the root the citation carries: a model writes the
        citations it hands back, and a file the walk skipped (a secret one) or a source it may not read is never read
        for it."""
        try:
            source, path, first, last, sha = (citation[key] for key in ("source", "path", "first", "last", "sha256"))
        except (KeyError, TypeError):
            return None
        if not all(type(n) is int for n in (first, last)) or not 1 <= first <= last:
            return None
        if not all(isinstance(text, str) for text in (source, path, sha)):
            return None
        try:
            source.encode("utf-8")
            path.encode("utf-8")
        except UnicodeEncodeError:
            return None         # a lone surrogate (JSON can carry one): no name the store holds
        allowed, values = self._filter()
        row = self.store.db.execute(
            f"SELECT s.root FROM file f JOIN source s ON s.id = f.source WHERE s.name = ? AND f.path = ? AND "
            f"{allowed}", (source, path, *values)).fetchone()
        return None if row is None else row[0]

    def lines(self, citation):
        """The lines of a citation's file, read through the store (``held_root``) as the index read them; None for a
        citation the store does not hold, or a file that cannot be read now."""
        root = self.held_root(citation)
        return None if root is None else source_lines({"root": root, "path": citation["path"]}, self.readers)

    def read(self, citation):
        """The text of a citation's lines while its source still says it, read through the store (``lines``), else
        None."""
        lines = self.lines(citation)
        if lines is None or citation["last"] > len(lines):
            return None         # lines the file does not hold are no text of it
        body = "\n".join(lines[citation["first"] - 1:citation["last"]])
        return body if hashlib.sha256(body.encode("utf-8")).hexdigest() == citation["sha256"] else None

    def verify(self, citation):
        """``verify`` of a citation read through the store: 'unknown' when the store holds no such file of a source
        this searcher may read, or its lines are not whole numbers in order, or it carries no sha256 (``held_root``);
        'modified' when the file holds fewer lines than it cites; else 'fresh', 'modified', 'gone' or 'unchecked',
        read under the store's root."""
        root = self.held_root(citation)
        if root is None:
            return "unknown"
        lines = self.lines(citation)
        if lines is not None and citation["last"] > len(lines):
            return "modified"
        return verify(dict(citation, root=root), self.readers)

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
    without one, nothing can be checked: 'unchecked', never 'fresh'. The citation's root and path are read as it
    carries them: for the caller's own citations; one a model handed back goes through ``Searcher.verify``."""
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
