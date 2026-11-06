"""Contracts of the Mycelium's second layer: the vectors, asked through the registry under the governor.

MV1 vectors come from the registry's embed_many, admitted as background work and sent under that admission's
    ticket; a question is admitted as interactive work; the real seams exist by these names and these parameters
MV2 an embedded text is asked once: parts that embed the same text share it, a second pass asks nothing and reads
    no file, a changed file asks only its changed parts, another cutter asks nothing new; a file changed since its
    index pass is left to the next one
MV3 a refused text is cut out of its batch and counted; a vector that is not finite, or that points nowhere, is
    left out, and a finite one of any size is kept as a unit vector; another dimension than the card's, or a batch
    of which nothing comes back, stops the pass by name
MV4 two encoders never mix: the active one is read, and asked the question, until another covers every part; then
    the old one's vectors go; a changed card embeds again
MV5 a missing encoder, an engine without an embedding endpoint, a held question, an answer that is not numbers,
    or an active encoder with no vector the search may read leave the search to its other stages, silently for
    the chat, and say why
MV6 under SQLCipher the store holds no vector's bytes
MV7 a pass held by the governor keeps what it wrote, writes only numbers in its episode; the next asks only the rest
MV8 a store of schema 1 opens at schema 2 and its next pass cuts every file again, with embedded texts
MV9 a source marked again with another sensitivity leaves the vectors at once, the held matrix read again; so
    does a searcher's narrowed allow
MV10 an encoder is never read while a part lacks its vector; a header change cuts every file again, once
MV11 an engine that cannot be reached stops a vector pass at its first request
MV12 a search gives k citations at most
MV13 the vectors' settings come from mycelium.yaml, refused by name when malformed (its YAML, its shape, its
    types and bounds); the card fits the store
MV14 the move from schema 1 is one transaction: one that fails leaves schema 1 whole, opened again later
MV15 a vector pass holds the store's lock only to plan and to write: a write from another thread lands while the
    engine answers, and a text removed meanwhile gets no vector
MV16 a document an indexer cannot read again (it never held the whole source) is left without a vector, the
    encoder not made the one read, the pass never stopped
MV17 only a text the engine refuses (400) is cut out of its batch: a model the engine does not serve, an engine
    that fails, or an answer that cannot be read stops the pass at its first request, by name, the old encoder
    still read with its vectors
MV18 a refusal of the governor that no wait lifts stops the pass as missing, never as held
MV19 two openers of a schema-1 store move it once
MV20 an encoder that covers nothing is never made the one read, nor one that lacks a part while another is read;
    on a store with no encoder read, the texts refused in that pass aside
MV21 a text refused, then gone while the engine answered, never stands for a text added meanwhile
MV22 a question's vector of any finite size is scored as a unit vector
MV23 a busy engine, or a refusal of the engine's own gate that a wait lifts, holds the pass; a final one stops it
MV24 the governor and the engine are asked under the name the engine serving the encoder knows it by: Ollama's
    for a name with no tag is its :latest (the tag after the last colon that follows the last slash); another
    engine's names are its own; the governor is told which engine serves it; the store keeps the encoder as the
    settings name it
MV25 on the real governor, the background pass of an encoder the settings name without its tag, loaded in Ollama,
    is admitted and loads nothing (it was refused for good as of unknown cost: BACKLOG 239 f)
MV26 the texts of a pass and a question are asked of no encoder off this machine: one served elsewhere, or by a
    backend that forwards its requests or cannot say where it sends
"""

import hashlib
import inspect
import re
import sqlite3
import threading

import pytest

from opti_oignon.mycelium import index as index_module
from opti_oignon.mycelium import vectors
from opti_oignon.mycelium.index import Indexer
from opti_oignon.mycelium.search import Searcher
from opti_oignon.mycelium.store import SCHEMA_VERSION, Store, StoreError
from opti_oignon.mycelium.vectors import Embedder, Encoder, VectorError, load_settings

BUDGET_S = {
    "test_mv1_vectors_come_from_embed_many_admitted_as_background_under_the_admissions_ticket": 1.5,
    "test_mv2_an_embedded_text_is_asked_once": 1.5,
    "test_mv3_a_refused_text_is_cut_out_and_a_wrong_vector_never_kept": 1.5,
    "test_mv4_two_encoders_never_mix_and_the_old_one_stays_until_the_new_covers_every_part": 1.5,
    "test_mv5_a_missing_encoder_or_a_held_question_leaves_the_search_to_its_other_stages": 1.5,
    "test_mv6_under_sqlcipher_the_store_holds_no_vectors_bytes": 1.5,
    "test_mv7_a_held_pass_keeps_what_it_wrote_and_the_next_asks_only_the_rest": 1.5,
    "test_mv8_a_schema_1_store_opens_at_schema_2_and_is_cut_again": 1.5,
    "test_mv9_a_source_marked_again_leaves_the_vectors_at_once": 1.5,
    "test_mv10_an_encoder_is_never_read_while_a_part_lacks_its_vector_and_a_header_change_cuts_again_once": 1.5,
    "test_mv11_an_engine_that_cannot_be_reached_stops_the_pass_at_once": 1.5,
    "test_mv12_a_search_gives_k_citations_at_most": 1.5,
    "test_mv13_the_settings_come_from_mycelium_yaml_and_the_card_fits_the_store": 1.5,
    "test_mv14_the_move_from_schema_1_is_one_transaction": 1.5,
    "test_mv15_a_vector_pass_holds_the_lock_only_to_plan_and_to_write": 3.0,
    "test_mv16_a_document_that_cannot_be_read_again_is_left_without_a_vector": 1.5,
    "test_mv17_only_a_refused_text_is_cut_out_any_other_failure_stops_the_pass_at_once": 1.5,
    "test_mv18_a_final_refusal_of_the_governor_stops_the_pass_as_missing": 1.5,
    "test_mv19_two_openers_of_a_schema_1_store_move_it_once": 1.5,
    "test_mv20_an_encoder_that_covers_nothing_or_lacks_a_part_beside_another_is_never_made_the_one_read": 1.5,
    "test_mv21_a_refused_text_gone_meanwhile_never_stands_for_a_text_added": 3.0,
    "test_mv22_a_question_vector_of_any_finite_size_is_scored_as_a_unit_vector": 1.5,
    "test_mv23_a_busy_engine_holds_the_pass_and_a_final_refusal_of_its_gate_stops_it": 1.5,
    "test_mv24_the_governor_and_the_engine_are_asked_under_the_engines_name_for_the_encoder": 1.5,
    "test_mv25_the_real_governor_admits_the_pass_of_an_encoder_named_without_its_tag": 1.5,
    "test_mv26_the_texts_and_the_question_never_leave_this_machine[remote]": 1.5,
    "test_mv26_the_texts_and_the_question_never_leave_this_machine[forwards]": 1.5,
    "test_mv26_the_texts_and_the_question_never_leave_this_machine[mute]": 1.5,
}

DIM = 16
A = Encoder("enc-a", DIM, "query: ", "text: ")
B = Encoder("enc-b", DIM, "q: ", "t: ")

PEELS = '''"""Peels: summaries of evicted spans."""


def select_peels(conversation_id, query):
    """A peel scores the share of the query's terms what it holds."""
    chosen = []
    for peel in held_peels(conversation_id):
        if share_of_terms(peel, query) > 0:
            chosen.append(peel)
    return chosen


def held_peels(conversation_id):
    """The peels a conversation holds, oldest first."""
    return []
'''

GATE = '''"""The eviction gate: a summary is checked against its sources before it replaces them."""


def probe_summary(summary, sources):
    """Entity, number and date probes: each one the sources name must be in the summary."""
    missing = [name for name in sources if name not in summary]
    if missing:
        raise ValueError("the summary drops what its sources name")
    return True
'''

HELPER = '''"""A helper module."""


def same_helper(value):
    """Return the value doubled, the same in two files."""
    doubled = value * 2
    checked = doubled + 0
    kept = checked
    return kept
'''


def bag(text, dim=DIM):
    """A deterministic vector: the text's words hashed into ``dim`` slots."""
    v = [0.0] * dim
    for word in re.findall(r"[a-z]+", text.lower()):
        v[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    v[0] += 0.01
    return v


class Decision:
    def __init__(self, admitted, reason=""):
        self.admitted, self.reason = admitted, reason


class Governor:
    """Admits everything, or holds the index requests after the first ``hold_after``, or every request of
    ``hold_callers``; records who asked and the decision it gave."""

    def __init__(self, hold_after=None, hold_callers=()):
        self.asked, self.decisions = [], []
        self.hold_after, self.hold_callers = hold_after, set(hold_callers)

    def admit(self, model, requested_ctx=None, caller="chat", **_kw):
        self.asked.append((model, caller))
        index_asks = sum(1 for _m, c in self.asked if c == "index")
        held = caller in self.hold_callers or (
            self.hold_after is not None and caller == "index" and index_asks > self.hold_after)
        self.decisions.append(Decision(not held, "busy" if held else ""))
        return self.decisions[-1]


class ResponseError(Exception):
    """As the engine's client raises it: the engine's words and the HTTP status."""

    def __init__(self, error, status_code=-1):
        super().__init__(error)
        self.status_code = status_code


class Engine:
    """A stand-in encoder on this machine (``where`` places it elsewhere): its calls and the tickets they were made
    under."""

    forwards_requests = False

    def __init__(self, dim=DIM, refuse=(), nan=(), endpoint=True, down=None, where="http://127.0.0.1:11434"):
        self.dim, self.refuse, self.nan, self.serves, self.down = dim, tuple(refuse), tuple(nan), endpoint, down
        self.where = where
        self.calls, self.tickets = [], []

    def endpoint(self):
        return self.where

    def embed_many(self, model, texts, timeout=None):
        from opti_oignon.resource_governor import get_active_ticket

        self.calls.append((model, list(texts)))
        self.tickets.append(get_active_ticket())
        if not self.serves:
            return None
        if self.down is not None:
            raise self.down
        if any(marker in text for text in texts for marker in self.refuse):
            raise ResponseError("the input length exceeds the context length", 400)
        out = []
        for text in texts:
            v = bag(text, self.dim)
            if any(marker in text for marker in self.nan):
                v[1] = float("nan")
            out.append(v)
        return out

    def texts(self):
        return [t for _m, ts in self.calls for t in ts]


class Registry:
    def __init__(self, engines):
        self.engines = dict(engines)

    def resolve_backend(self, model):
        return self.engines.get(model)


@pytest.fixture(autouse=True)
def no_engine(monkeypatch):
    """No contract reaches the machine's engine or governor: until a contract installs its stand-ins, the registry
    refuses by name and there is no governor."""

    def refused():
        raise RuntimeError("a contract asked the real inference registry")

    monkeypatch.setattr(vectors, "_registry", refused)
    monkeypatch.setattr(vectors, "_governor", lambda: None)


@pytest.fixture
def seams(monkeypatch):
    def install(registry, governor):
        monkeypatch.setattr(vectors, "_registry", lambda: registry)
        monkeypatch.setattr(vectors, "_governor", lambda: governor)

    return install


def folder(tmp_path, name, files):
    root = tmp_path / name
    root.mkdir()
    for rel, text in files.items():
        (root / rel).write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def corpus(tmp_path):
    root = folder(tmp_path, "src", {"peels.py": PEELS, "gate.py": GATE})
    store = Store(tmp_path / "corpus.myc", sqlite3.connect)
    Indexer(store).index("oo", root)
    yield {"store": store, "root": root, "tmp": tmp_path}
    store.close()


def distinct_texts(store):
    return store.db.execute("SELECT COUNT(DISTINCT embed_sha) FROM part").fetchone()[0]


def texts_without_vector(store, encoder):
    return store.db.execute(
        "SELECT COUNT(DISTINCT embed_sha) FROM part WHERE embed_sha NOT IN "
        "(SELECT embed_sha FROM vector WHERE encoder = ?)", (encoder.key,)).fetchone()[0]


def vectors_under(store, encoder):
    return store.db.execute("SELECT COUNT(*) FROM vector WHERE encoder = ?", (encoder.key,)).fetchone()[0]


def test_mv1_vectors_come_from_embed_many_admitted_as_background_under_the_admissions_ticket(corpus, seams):
    """MV1 -- every text goes through the registry's backend for the encoder, in a call the governor admitted as
    the caller ``index`` and made under that very decision's ticket, with the card's text prefix; the question is
    admitted as ``chat`` with the query prefix. Another way to the engine, no admission, a call outside the ticket,
    a prefix lost -> RED. The real registry and governor are reached by these names, with these parameters."""
    engine, governor = Engine(), Governor()
    seams(Registry({"enc-a": engine}), governor)
    store = corpus["store"]
    counts = Indexer(store).embed(Embedder(A), batch=3)
    assert counts["embedded"] == distinct_texts(store) > 3 and counts["held"] is False and counts["missing"] is False
    assert len(engine.calls) == len(governor.asked) >= 2
    assert all(model == "enc-a" for model, _t in engine.calls)
    assert all(asked == ("enc-a", "index") for asked in governor.asked)
    assert engine.tickets == governor.decisions and all(t.admitted for t in engine.tickets)
    assert all(text.startswith("text: ") for text in engine.texts())
    assert store.active_encoder()["key"] == A.key
    searcher = Searcher(store)
    found = searcher.search("probe summary sources entity number date", stages=("dense",))
    assert governor.asked[-1] == ("enc-a", "chat")
    assert engine.calls[-1][1] == ["query: probe summary sources entity number date"]
    assert engine.tickets[-1] is governor.decisions[-1]
    assert found and found[0]["path"] == "gate.py" and searcher.silent == {}
    from opti_oignon import inference_backend, resource_governor

    registry = inference_backend.get_backend_registry
    assert callable(registry) and "resolve_backend" in dir(inference_backend.BackendRegistry)
    assert "timeout" in inspect.signature(inference_backend.InferenceBackend.embed_many).parameters
    assert "caller" in inspect.signature(resource_governor.ResourceGovernor.admit).parameters
    assert all(callable(f) for f in (resource_governor.get_resource_governor, resource_governor.ticket_scope,
                                     resource_governor.get_active_ticket))


def test_mv2_an_embedded_text_is_asked_once(tmp_path, seams, monkeypatch):
    """MV2 -- two files that embed the same texts (no header) are asked those texts once; a second pass asks
    nothing; a changed file asks only the parts whose text changed; a store cut again by another cutter asks
    nothing new; a file changed again after its index pass gives none of its new lines (they would be kept under
    the old text's name). A vector asked per part, again for a text that has one, or under another text's name
    -> RED."""
    root = folder(tmp_path, "src", {"a.py": HELPER, "b.py": HELPER, "peels.py": PEELS})
    store = Store(tmp_path / "c.myc", sqlite3.connect)
    indexer = Indexer(store, header="none")
    indexer.index("oo", root)
    engine = Engine()
    seams(Registry({"enc-a": engine}), Governor())
    indexer.embed(Embedder(A))
    sent = engine.texts()
    assert len(sent) == len(set(sent))
    parts = store.db.execute("SELECT COUNT(*) FROM part").fetchone()[0]
    assert len(sent) == distinct_texts(store) < parts
    engine.calls.clear()
    read = []
    reader = index_module.file_lines
    monkeypatch.setattr(index_module, "file_lines", lambda *a: read.append(a) or reader(*a))
    assert indexer.embed(Embedder(A))["embedded"] == 0 and engine.calls == [] and read == []
    (root / "peels.py").write_text(PEELS.replace("oldest first", "newest first"), encoding="utf-8")
    indexer.index("oo", root)
    indexer.embed(Embedder(A))
    assert len(engine.texts()) == 1 and "newest first" in engine.texts()[0]
    engine.calls.clear()
    monkeypatch.setattr(index_module, "CHUNKER_VERSION", "another")
    assert indexer.index("oo", root)["recut"] == 3
    assert indexer.embed(Embedder(A))["embedded"] == 0 and engine.calls == []
    assert texts_without_vector(store, A) == 0
    (root / "peels.py").write_text(PEELS.replace("oldest first", "first of all"), encoding="utf-8")
    indexer.index("oo", root)
    (root / "peels.py").write_text(PEELS.replace("oldest first", "last of all"), encoding="utf-8")
    indexer.embed(Embedder(A))
    assert not any("last of all" in t for t in engine.texts()) and texts_without_vector(store, A) == 1
    store.close()


def test_mv3_a_refused_text_is_cut_out_and_a_wrong_vector_never_kept(corpus, seams):
    """MV3 -- a text the engine refuses is cut out of its batch (the others get their vectors) and counted; a
    vector with a NaN, or of zeros, leaves its text without one, and a finite vector of components past float32's
    range is kept as a unit vector; a vector of another dimension than the card's stops the pass by name, as does a
    batch of which no text comes back. A batch dropped whole, a refused text retried forever, a NaN, a zero vector,
    an overflow kept as infinities, or a wrong dimension kept -> RED."""
    store = corpus["store"]
    engine = Engine(refuse=("drops what its sources name",), nan=("oldest first",))
    seams(Registry({"enc-a": engine}), Governor())
    counts = Indexer(store).embed(Embedder(A), batch=8)
    distinct = distinct_texts(store)
    assert counts["refused"] == 2 and counts["embedded"] == distinct - 2 > 0 and counts["missing"] is False
    assert texts_without_vector(store, A) == 2
    refused = {t for t in engine.texts() if "drops what its sources name" in t}
    assert len(refused) == 1 and list(refused) in [ts for _m, ts in engine.calls]
    assert len(engine.calls) < 2 * distinct
    for (data,) in store.db.execute("SELECT data FROM vector"):
        assert len(data) == DIM * 2
    seams(Registry({"enc-b": Engine(dim=DIM + 1)}), Governor())
    indexer = Indexer(store)
    counts = indexer.embed(Embedder(B))
    assert counts["missing"] is True and counts["embedded"] == 0
    assert "17 dimensions" in indexer.last_stop and "16" in indexer.last_stop
    assert vectors_under(store, B) == 0
    down = Engine(refuse=("",))       # it answers, and refuses every text (one that is not reached: MV11)
    seams(Registry({"enc-b": down}), Governor())
    counts = indexer.embed(Embedder(B), batch=4)
    assert counts["missing"] is True and "no vector for any text of a batch of 4" in indexer.last_stop
    assert len(down.calls) == 7 and store.active_encoder()["key"] == A.key

    class Huge(Engine):
        """A finite vector of components past float32's range for one text, a vector of zeros for another."""

        def embed_many(self, model, texts, timeout=None):
            out = super().embed_many(model, texts, timeout)
            return [[1e39] + v[1:] if "oldest first" in t else [0.0] * DIM if "drops what" in t else v
                    for t, v in zip(texts, out)]

    huge = Encoder("enc-c", DIM)
    seams(Registry({"enc-c": Huge()}), Governor())
    counts = Indexer(store).embed(Embedder(huge))
    assert counts["refused"] == 1 and counts["missing"] is False and texts_without_vector(store, huge) == 1
    import numpy as np

    for (data,) in store.db.execute("SELECT data FROM vector"):
        kept = np.frombuffer(data, dtype=np.float16)
        assert np.isfinite(kept).all() and abs(float(np.linalg.norm(kept.astype(np.float64))) - 1.0) < 0.01


def test_mv4_two_encoders_never_mix_and_the_old_one_stays_until_the_new_covers_every_part(corpus, seams):
    """MV4 -- with A active, a pass under B held halfway leaves A read and asked the question; B's vectors are
    never read; once B covers every part it is the one read and asked, and A's vectors go. A card changed (a
    prefix) is another key: everything is embedded again. A question asked to B while A's vectors are read, B
    active before it covers every part, or A's vectors read beside B's -> RED."""
    store = corpus["store"]
    engine_a, engine_b = Engine(), Engine()
    registry = Registry({"enc-a": engine_a, "enc-b": engine_b})
    seams(registry, Governor())
    Indexer(store).embed(Embedder(A))
    seams(registry, Governor(hold_after=1))
    counts = Indexer(store).embed(Embedder(B), batch=2)
    assert counts["held"] is True and counts["embedded"] == 2 and store.active_encoder()["key"] == A.key
    engine_a.calls.clear()
    engine_b.calls.clear()
    searcher = Searcher(store)
    found = searcher.search("probe summary sources", stages=("dense",))
    assert engine_a.calls and not engine_b.calls and found
    active, ids, matrix = searcher._matrix()
    parts = store.db.execute("SELECT COUNT(*) FROM part").fetchone()[0]
    assert active["key"] == A.key and len(ids) == len(set(ids)) == parts == matrix.shape[0]
    seams(registry, Governor())
    counts = Indexer(store).embed(Embedder(B))
    assert counts.get("activated") and store.active_encoder()["key"] == B.key
    assert vectors_under(store, A) == 0
    engine_a.calls.clear()
    searcher.search("probe summary sources", stages=("dense",))
    assert engine_b.calls[-1][1] == ["q: probe summary sources"] and not engine_a.calls
    renamed = Encoder("enc-b", DIM, "q2: ", "t2: ")
    assert renamed.key != B.key
    engine_b.calls.clear()
    counts = Indexer(store).embed(Embedder(renamed))
    assert counts["embedded"] == distinct_texts(store) and all(t.startswith("t2: ") for t in engine_b.texts())


def test_mv5_a_missing_encoder_or_a_held_question_leaves_the_search_to_its_other_stages(corpus, seams):
    """MV5 -- no backend for the encoder, a backend without an embedding endpoint: the pass writes nothing and says
    so, by name, in the status and the log (its episode a flag); a search answers by symbols and full text, and
    ``silent`` says why the vectors did not; so with a question the governor holds, or an engine that fails on it.
    An exception to the chat, or a silence that does not say why -> RED."""
    store = corpus["store"]
    seams(Registry({}), Governor())
    indexer = Indexer(store)
    counts = indexer.embed(Embedder(A))
    assert counts["missing"] is True and counts["embedded"] == 0
    status = indexer.vector_status(A)
    assert status["active"] is None and status["with_vector"] == 0 and status["missing"] is True
    assert "no backend serves enc-a" in status["why"]
    searcher = Searcher(store)
    found = searcher.search("select_peels share of terms")
    assert found and found[0]["path"] == "peels.py" and "no encoder is active" in searcher.silent["dense"]
    seams(Registry({"enc-a": Engine(endpoint=False)}), Governor())
    assert indexer.embed(Embedder(A))["missing"] is True and "no embedding endpoint" in indexer.last_stop
    seams(Registry({"enc-a": Engine()}), Governor())
    indexer.embed(Embedder(A))
    seams(Registry({"enc-a": Engine()}), Governor(hold_callers=("chat",)))
    found = searcher.search("select_peels share of terms")
    assert found and "held by the governor" in searcher.silent["dense"]
    seams(Registry({"enc-a": Engine(down=ConnectionError("the engine is down"))}), Governor())
    found = searcher.search("select_peels share of terms")
    assert found and "did not answer the question" in searcher.silent["dense"]
    seams(Registry({}), Governor())
    assert searcher.search("select_peels share of terms") and "no backend" in searcher.silent["dense"]

    class Garbled(Engine):
        def embed_many(self, model, texts, timeout=None):
            return [["not", "a", "number"] + [0.0] * (DIM - 3) for _t in texts]

    seams(Registry({"enc-a": Garbled()}), Governor())
    found = searcher.search("select_peels share of terms")
    assert found and "not numbers" in searcher.silent["dense"]
    seams(Registry({"enc-a": Engine()}), Governor())
    Indexer(store, header="none").index("oo", corpus["root"])
    assert store.active_encoder()["key"] == A.key
    found = searcher.search("select_peels share of terms")
    assert found and "no vector" in searcher.silent["dense"]


def test_mv6_under_sqlcipher_the_store_holds_no_vectors_bytes(tmp_path, seams):
    """MV6 -- the bytes of a vector written are in a plain store's file (the probe sees them) and not in a store
    opened under SQLCipher. A vector kept beside the store, or in the clear -> RED."""
    root = folder(tmp_path, "src", {"gate.py": GATE})
    seams(Registry({"enc-a": Engine()}), Governor())
    plain = Store(tmp_path / "plain.myc", sqlite3.connect)
    Indexer(plain).index("oo", root)
    Indexer(plain).embed(Embedder(A))
    blobs = [r[0] for r in plain.db.execute("SELECT data FROM vector")]
    plain.close()
    raw = (tmp_path / "plain.myc").read_bytes()
    assert blobs and all(blob in raw for blob in blobs)
    sqlcipher = pytest.importorskip("sqlcipher3.dbapi2")

    def connect(path):
        db = sqlcipher.connect(path)
        db.execute("PRAGMA key = \"x'" + "cd" * 32 + "'\"")
        return db

    sealed = Store(tmp_path / "sealed.myc", connect)
    Indexer(sealed).index("oo", root)
    Indexer(sealed).embed(Embedder(A))
    held = [r[0] for r in sealed.db.execute("SELECT data FROM vector")]
    sealed.close()
    raw = (tmp_path / "sealed.myc").read_bytes()
    assert held == blobs and not any(blob in raw for blob in held)
    assert not raw.startswith(b"SQLite format 3")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plain.myc", "sealed.myc", "src"]


def test_mv7_a_held_pass_keeps_what_it_wrote_and_the_next_asks_only_the_rest(corpus, seams):
    """MV7 -- the governor holds the second batch: the first batch's vectors are kept, the episode says held (a
    flag: why is said in the log and the status, never written), the encoder is not active yet; the next pass asks
    only the texts without a vector, and then the encoder is read. A held pass that loses its batch, asks a text
    twice, writes the governor's words, or activates a half-covered encoder -> RED."""
    store = corpus["store"]
    engine = Engine()
    seams(Registry({"enc-a": engine}), Governor(hold_after=1))
    indexer = Indexer(store)
    counts = indexer.embed(Embedder(A), batch=2)
    assert counts["held"] is True and "busy" in indexer.last_stop and counts["embedded"] == 2
    assert store.active_encoder() is None
    detail = store.episodes(None, 1)[0]["detail"]
    assert detail["held"] is True and all(type(v) in (int, float, bool) for v in detail.values())
    first = list(engine.texts())
    seams(Registry({"enc-a": engine}), Governor())
    counts = indexer.embed(Embedder(A), batch=2)
    rest = engine.texts()[len(first):]
    assert not set(first) & set(rest) and counts["embedded"] == len(rest)
    assert store.active_encoder()["key"] == A.key and texts_without_vector(store, A) == 0


# Schema 1 as block 234 wrote it (its store.py's SCHEMA, verbatim), with the rows its indexer writes for a file.
SCHEMA_1 = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS source(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, root TEXT NOT NULL, kind TEXT NOT NULL,
  sensitivity TEXT NOT NULL CHECK (sensitivity IN ('public', 'private', 'secret')),
  generation INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS file(
  id INTEGER PRIMARY KEY, source INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE, path TEXT NOT NULL,
  size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL, lang TEXT NOT NULL,
  generation INTEGER NOT NULL, module_doc TEXT NOT NULL DEFAULT '', UNIQUE(source, path));
CREATE INDEX IF NOT EXISTS file_sha ON file(sha256);
CREATE TABLE IF NOT EXISTS part(
  id INTEGER PRIMARY KEY, file INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE, kind TEXT NOT NULL,
  symbol TEXT NOT NULL, first INTEGER NOT NULL, last INTEGER NOT NULL, sha256 TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS part_file ON part(file);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(path, body, words, said, content='', contentless_delete=1,
  tokenize="porter unicode61 remove_diacritics 2 tokenchars '_'");
CREATE TABLE IF NOT EXISTS symbol(
  part INTEGER NOT NULL REFERENCES part(id) ON DELETE CASCADE, name TEXT NOT NULL, qualname TEXT NOT NULL,
  kind TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS symbol_name ON symbol(name);
CREATE INDEX IF NOT EXISTS symbol_part ON symbol(part);
CREATE TABLE IF NOT EXISTS ref(
  part INTEGER NOT NULL REFERENCES part(id) ON DELETE CASCADE, name TEXT NOT NULL, kind TEXT NOT NULL,
  module TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ref_name ON ref(name);
CREATE INDEX IF NOT EXISTS ref_part ON ref(part);
CREATE TABLE IF NOT EXISTS episode(
  id INTEGER PRIMARY KEY, at TEXT NOT NULL, action TEXT NOT NULL, source INTEGER, detail TEXT NOT NULL);
"""


def schema_1_store(tmp_path):
    """A store as block 234 left it: one source, one file, its part, its cutter and pass stamps."""
    root = folder(tmp_path, "src", {"gate.py": GATE})
    path = tmp_path / "old.myc"
    db = sqlite3.connect(path)
    db.executescript(SCHEMA_1)
    db.execute("INSERT INTO meta VALUES ('schema', '1')")
    db.execute("INSERT INTO source(id, name, root, kind, sensitivity, generation) "
               "VALUES (1, 'oo', ?, 'folder', 'private', 1)", (str(root),))
    st = (root / "gate.py").stat()
    sha = hashlib.sha256((root / "gate.py").read_bytes()).hexdigest()
    db.execute("INSERT INTO file VALUES (1, 1, 'gate.py', ?, ?, ?, 'python', 1, '')", (st.st_size, st.st_mtime_ns,
                                                                                      sha))
    db.execute("INSERT INTO part VALUES (1, 1, 'function', 'probe_summary', 4, 9, 'x')")
    db.execute("INSERT INTO meta VALUES ('cutter:1', '5')")
    db.execute("INSERT INTO meta VALUES ('pass:1', '1')")
    db.commit()
    db.close()
    return root, path


def test_mv8_a_schema_1_store_opens_at_schema_2_and_is_cut_again(tmp_path, seams):
    """MV8 -- a store written at schema 1 (its tables as block 234 made them, a file and its part in them) opens
    at schema 2: the part keeps its row, the next pass cuts every file again and each part has its embedded text,
    and the vectors come. A schema-1 store refused, or kept with parts that have no embedded text -> RED."""
    root, path = schema_1_store(tmp_path)
    store = Store(path, sqlite3.connect)
    assert store.meta("schema") == str(SCHEMA_VERSION) == "2"
    assert store.db.execute("SELECT embed_sha FROM part WHERE id = 1").fetchone() == ("",)
    counts = Indexer(store).index("oo", root)
    assert counts["recut"] == 1
    assert store.db.execute("SELECT COUNT(*) FROM part WHERE embed_sha = ''").fetchone()[0] == 0
    seams(Registry({"enc-a": Engine()}), Governor())
    assert Indexer(store).embed(Embedder(A))["embedded"] > 0 and texts_without_vector(store, A) == 0
    store.close()


def test_mv9_a_source_marked_again_leaves_the_vectors_at_once(tmp_path, seams):
    """MV9 -- a private corpus marked secret: a searcher that held its matrix reads it again and finds nothing by
    its vectors, as does a new searcher; one allowed secrets finds it. A matrix kept from before, or the vectors
    read without the sources' sensitivity -> RED."""
    root = folder(tmp_path, "notes", {"gate.py": GATE})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("notes", root)
    seams(Registry({"enc-a": Engine()}), Governor())
    Indexer(store).embed(Embedder(A))
    searcher = Searcher(store)
    assert searcher.search("probe summary sources", stages=("dense",))
    Indexer(store).index("notes", root, sensitivity="secret")
    assert searcher.search("probe summary sources", stages=("dense",)) == [] and searcher._matrix()[1] == []
    assert Searcher(store).search("probe summary sources", stages=("dense",)) == []
    allowed = Searcher(store, allow=("public", "private", "secret"))
    assert allowed.search("probe summary sources", stages=("dense",))
    allowed.allow = ("public", "private")
    assert allowed.search("probe summary sources", stages=("dense",)) == []
    store.close()


def test_mv10_an_encoder_is_never_read_while_a_part_lacks_its_vector_and_a_header_change_cuts_again_once(
        tmp_path, seams):
    """MV10 -- A read; a pass under B by an indexer that cannot rebuild the texts (another header) embeds nothing:
    B is not made the one read and A's vectors stay; the next index pass with that header cuts every file again,
    the one after it nothing, and B then covers every part and is read. A half-covered encoder read, the old one's
    vectors dropped, a header change unseen, or a header that cuts again at every pass -> RED."""
    root = folder(tmp_path, "src", {"gate.py": GATE})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    seams(Registry({"enc-a": Engine(), "enc-b": Engine()}), Governor())
    Indexer(store).index("oo", root)
    Indexer(store).embed(Embedder(A))
    assert store.active_encoder()["key"] == A.key
    counts = Indexer(store, header="none").embed(Embedder(B))
    assert store.active_encoder()["key"] == A.key and not counts.get("activated")
    assert vectors_under(store, A) > 0 and counts["uncovered"] > 0
    assert Indexer(store, header="none").index("oo", root)["recut"] == 1
    assert Indexer(store, header="none").index("oo", root)["recut"] == 0
    Indexer(store, header="none").embed(Embedder(B))
    assert store.active_encoder()["key"] == B.key
    store.close()


def test_mv11_an_engine_that_cannot_be_reached_stops_the_pass_at_once(tmp_path, seams):
    """MV11 -- the engine times out or cannot be reached: one request, the pass stops and says the engine did not
    answer; no cutting in two (2N - 1 requests, each its timeout). -> RED otherwise."""
    root = folder(tmp_path, "src", {"gate.py": GATE, "peels.py": PEELS})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("oo", root)
    for failure in (TimeoutError("timed out"), ConnectionRefusedError("refused")):
        engine = Engine(down=failure)
        seams(Registry({"enc-a": engine}), Governor())
        indexer = Indexer(store)
        counts = indexer.embed(Embedder(A), batch=4)
        assert len(engine.calls) == 1 and counts["missing"] is True and "did not answer" in indexer.last_stop
    store.close()


def test_mv12_a_search_gives_k_citations_at_most(tmp_path):
    """MV12 -- a question many parts answer: k citations, no more (a part gone since takes no place). A search that
    gives up to four times k -> RED."""
    root = folder(tmp_path, "src", {"peels.py": PEELS, "gate.py": GATE, "a.py": HELPER, "b.py": HELPER})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("oo", root)
    searcher = Searcher(store)
    question = "peels summary sources helper value conversation query"
    many = searcher.search(question, k=10)
    assert len(many) > 2
    assert len(searcher.search(question, k=2)) == 2 and searcher.search(question, k=0) == []
    store.close()


def test_mv13_the_settings_come_from_mycelium_yaml_and_the_card_fits_the_store(tmp_path, seams):
    """MV13 -- the tree's mycelium.yaml gives the vectors' settings (its encoder, the card's dimension, the
    header); a header it does not know, a missing card or a card longer than the store keeps (accented prefixes:
    their stored form is longer than their text) is refused by name; a long card the settings take is kept as the
    store's active encoder. Settings read elsewhere, a malformed file taken, or a card the store refuses halfway
    through a pass -> RED."""
    settings = load_settings()
    assert settings.enabled and settings.header == "full" and settings.batch > 0
    assert settings.encoder.model == "embeddinggemma-2" and settings.encoder.dim == 768
    assert settings.encoder.query_prefix and settings.encoder.text_prefix
    good = "vectors:\n  encoder: e\n  encoders:\n    e:\n      dim: 4\n"
    (tmp_path / "ok.yaml").write_text(good, encoding="utf-8")
    assert load_settings(tmp_path / "ok.yaml").encoder == Encoder("e", 4)
    bad = {"header.yaml": good + "  header: whole\n", "card.yaml": "vectors:\n  encoder: f\n  encoders: {}\n",
           "prefix.yaml": good + "      query_prefix: " + chr(233) * 700 + "\n"}
    bad.update({"huge.yaml": good.replace("dim: 4", "dim: " + "9" * 400),
                "prefix_flag.yaml": good + "      query_prefix: false\n",
                "syntax.yaml": "vectors: [unclosed\n", "list.yaml": "- vectors\n",
                "timeout.yaml": good + "  timeout_s: -1\n", "zero.yaml": good + "  timeout_s: 0\n",
                "nan.yaml": good + "  timeout_s: .nan\n", "enabled.yaml": good + "  enabled: 'false'\n",
                "dim.yaml": good.replace("dim: 4", "dim: true"), "batch.yaml": good + "  batch: 2.5\n"})
    for name, text in bad.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
        with pytest.raises(VectorError, match="mycelium.yaml"):
            load_settings(tmp_path / name)
    long = "vectors:\n  encoder: " + "m" * 100 + "\n  encoders:\n    " + "m" * 100 + ":\n      dim: 16\n"
    long += "      query_prefix: " + chr(233) * 300 + "\n      text_prefix: " + "t" * 300 + "\n"
    (tmp_path / "long.yaml").write_text(long, encoding="utf-8")
    wide = load_settings(tmp_path / "long.yaml").encoder
    root = folder(tmp_path, "src", {"gate.py": GATE})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("oo", root)
    seams(Registry({wide.model: Engine()}), Governor())
    assert Indexer(store).embed(Embedder(wide)).get("activated") and store.active_encoder()["key"] == wide.key
    store.close()


class FailingMove:
    """A connection whose move to schema 2 fails at its last step (the new stamp)."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)

    def execute(self, sql, *args):
        if sql.startswith("UPDATE meta SET value = '2'"):
            raise sqlite3.OperationalError("disk I/O error")
        return self.db.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.db, name)


def test_mv14_the_move_from_schema_1_is_one_transaction(tmp_path):
    """MV14 -- the move to schema 2 fails at its last step: the store is refused, and opened again it moves whole
    (schema 2, its parts with the new column, cut again at the next pass). A move that leaves its column behind
    and the stamp at 1 (the next open then fails on a column that exists) -> RED."""
    root, path = schema_1_store(tmp_path)
    failing = FailingMove(path)
    with pytest.raises(sqlite3.OperationalError):
        Store(path, lambda _p: failing)
    failing.db.close()
    store = Store(path, sqlite3.connect)
    assert store.meta("schema") == "2" and store.meta("cutter:1") is None
    assert Indexer(store).index("oo", root)["recut"] == 1
    store.close()


def test_mv15_a_vector_pass_holds_the_lock_only_to_plan_and_to_write(tmp_path, seams):
    """MV15 -- while a vector pass waits on the engine, another thread marks the source again and runs an index pass
    that removes a file: both land at once; the pass then writes no vector for the removed file's texts, and the
    encoder covers what is left. A pass that holds the store's lock across the engine's answers (every write of
    the store waits up to its batches' timeouts), or one that writes the vector of a text gone meanwhile -> RED."""
    root = folder(tmp_path, "src", {"gate.py": GATE, "peels.py": PEELS})
    store = Store(tmp_path / "s.myc", lambda p: sqlite3.connect(p, check_same_thread=False))
    Indexer(store).index("oo", root)
    inside, release, wrote = threading.Event(), threading.Event(), threading.Event()

    class Slow(Engine):
        def embed_many(self, model, texts, timeout=None):
            inside.set()
            release.wait(10)
            return super().embed_many(model, texts, timeout)

    seams(Registry({"enc-a": Slow()}), Governor())
    passing = threading.Thread(target=lambda: Indexer(store).embed(Embedder(A), batch=100))
    passing.start()
    assert inside.wait(10)

    def write():
        store.source("oo", root, sensitivity="public")
        (root / "peels.py").unlink()
        Indexer(store).index("oo", root)
        wrote.set()

    writer = threading.Thread(target=write)
    writer.start()
    try:
        assert wrote.wait(5)
    finally:
        release.set()
        passing.join(10)
        writer.join(10)
    assert store.db.execute("SELECT COUNT(*) FROM vector WHERE embed_sha NOT IN (SELECT embed_sha FROM part)"
                            ).fetchone()[0] == 0
    assert vectors_under(store, A) == distinct_texts(store) > 0 and store.active_encoder()["key"] == A.key
    store.close()


def test_mv16_a_document_that_cannot_be_read_again_is_left_without_a_vector(tmp_path, seams):
    """MV16 -- an indexer given part of a source of documents it never held whole: the documents it was given get
    their vectors; one it cannot read again is left without, the encoder not made the one read, and the pass goes
    on; the indexer that holds the source covers it. A pass stopped by a document it cannot read, or an encoder
    read while a document has no vector -> RED."""
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    seams(Registry({"enc-a": Engine()}), Governor())
    whole = Indexer(store)
    whole.index_documents("notes", {"a.md": "# Peels\n\nA peel holds a summary.\n",
                                    "b.md": "# Gate\n\nThe gate checks a summary.\n"})
    partial = Indexer(store)
    partial.index_documents("notes", {"a.md": "# Peels\n\nA peel holds a shorter summary.\n"}, complete=False)
    counts = partial.embed(Embedder(A))
    assert counts["embedded"] >= 1 and counts["uncovered"] > 0 and store.active_encoder() is None
    whole.index_documents("notes", {"a.md": "# Peels\n\nA peel holds a shorter summary.\n",
                                    "b.md": "# Gate\n\nThe gate checks a summary.\n"})
    assert whole.embed(Embedder(A)).get("activated") and texts_without_vector(store, A) == 0
    store.close()


def test_mv17_only_a_refused_text_is_cut_out_any_other_failure_stops_the_pass_at_once(corpus, seams):
    """MV17 -- with A read: a model the engine does not serve (404), an engine that fails (500), an answer the
    client cannot read (no status), or one of another count than the texts: one request, the pass stops by name
    (missing), nothing counted refused, B never made the one read and A's vectors kept. A failure taken for a
    refused text (cut down to single texts, then counted aside, the encoder made the one read and the old vectors
    dropped) -> RED."""
    store = corpus["store"]
    seams(Registry({"enc-a": Engine()}), Governor())
    Indexer(store).embed(Embedder(A))
    kept = vectors_under(store, A)
    failures = (ResponseError("model 'enc-b' not found", 404), ResponseError("llama runner process has terminated",
                                                                           500),
                type("RemoteProtocolError", (Exception,), {})("peer closed the connection"),
                ValueError("enc-b answered 1 vector(s) for 4 text(s): refused"))
    for failure in failures:
        engine = Engine(down=failure)
        seams(Registry({"enc-b": engine}), Governor())
        indexer = Indexer(store)
        counts = indexer.embed(Embedder(B), batch=4)
        assert len(engine.calls) == 1 and counts["missing"] is True and counts["refused"] == 0, failure
        assert str(failure) in indexer.last_stop and not counts.get("activated")
        assert store.active_encoder()["key"] == A.key and vectors_under(store, A) == kept


def test_mv18_a_final_refusal_of_the_governor_stops_the_pass_as_missing(corpus, seams):
    """MV18 -- the governor refuses the encoder for good (its cost unknown: a model the engine does not hold): the
    pass stops as missing, saying the governor's reason, never as held (which the next pass would only meet
    again); a refusal a wait lifts stays held. The real governor names a final refusal by these words. A final
    refusal reported as held -> RED."""
    from opti_oignon import resource_governor

    class Final(Governor):
        def admit(self, model, requested_ctx=None, caller="chat", **_kw):
            self.asked.append((model, caller))
            self.decisions.append(Decision(False, "background_cost_unknown"))
            return self.decisions[-1]

    store, engine = corpus["store"], Engine()
    seams(Registry({"enc-a": engine}), Final())
    indexer = Indexer(store)
    counts = indexer.embed(Embedder(A))
    assert counts["missing"] is True and counts["held"] is False and engine.calls == []
    assert "background_cost_unknown" in indexer.last_stop and store.active_encoder() is None
    assert resource_governor.refusal_is_final(Decision(False, "background_cost_unknown"))
    assert not resource_governor.refusal_is_final(Decision(False, "busy"))
    seams(Registry({"enc-a": engine}), Governor(hold_after=0))
    assert indexer.embed(Embedder(A))["held"] is True


class SecondOpener:
    """A connection that lets another opener move the store to schema 2 just before it begins its own move."""

    def __init__(self, path):
        self.db, self.path, self.done = sqlite3.connect(path), path, False

    def execute(self, sql, *args):
        if sql.startswith("BEGIN") and not self.done:
            self.done = True
            Store(self.path, sqlite3.connect).close()
        return self.db.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.db, name)


def test_mv19_two_openers_of_a_schema_1_store_move_it_once(tmp_path):
    """MV19 -- an opener reads schema 1; another opener moves the store before the first begins its move: the first
    sees the move done and opens the store at schema 2, its column there once. A move that trusts the schema read
    before its transaction (the column added twice: the open fails) -> RED."""
    root, path = schema_1_store(tmp_path)
    second = SecondOpener(path)
    store = Store(path, lambda _p: second)
    assert second.done and store.meta("schema") == "2"
    columns = [r[1] for r in store.db.execute("PRAGMA table_info(part)")]
    assert columns.count("embed_sha") == 1
    assert Indexer(store).index("oo", root)["recut"] == 1
    store.close()


def test_mv20_an_encoder_that_covers_nothing_or_lacks_a_part_beside_another_is_never_made_the_one_read(corpus,
                                                                                                    seams):
    """MV20 -- the engine answers 400 to every request, one text each (a request it refuses whole: the client never
    asks it to keep a text whole, so a 400 may be about the request): every text is counted refused; with A read, B
    is not made the one read and A's vectors stay; on a store with no encoder read, none is made. A text B truly
    refuses while A is read keeps A read too; on a fresh store the same pass makes B the one read, that text aside.
    An encoder that covers nothing, or that replaces another while it lacks a part, made the one read -> RED."""
    store, root = corpus["store"], corpus["root"]
    seams(Registry({"enc-a": Engine()}), Governor())
    Indexer(store).embed(Embedder(A))
    kept = vectors_under(store, A)
    seams(Registry({"enc-b": Engine(refuse=("",))}), Governor())
    counts = Indexer(store).embed(Embedder(B), batch=1)
    assert counts["refused"] == distinct_texts(store) and counts["missing"] is False and not counts.get("activated")
    assert store.active_encoder()["key"] == A.key and vectors_under(store, A) == kept
    seams(Registry({"enc-b": Engine(refuse=("drops what its sources name",))}), Governor())
    counts = Indexer(store).embed(Embedder(B))
    assert counts["refused"] == 1 and counts["embedded"] > 0 and not counts.get("activated")
    assert store.active_encoder()["key"] == A.key and vectors_under(store, A) == kept
    fresh = Store(corpus["tmp"] / "fresh.myc", sqlite3.connect)
    Indexer(fresh).index("oo", root)
    seams(Registry({"enc-b": Engine(refuse=("",))}), Governor())
    assert Indexer(fresh).embed(Embedder(B), batch=1)["refused"] > 0 and fresh.active_encoder() is None
    seams(Registry({"enc-b": Engine(refuse=("drops what its sources name",))}), Governor())
    counts = Indexer(fresh).embed(Embedder(B))
    assert counts["refused"] == 1 and counts.get("activated") and fresh.active_encoder()["key"] == B.key
    fresh.close()


def test_mv21_a_refused_text_gone_meanwhile_never_stands_for_a_text_added(tmp_path, seams):
    """MV21 -- on a store with no encoder read, the engine refuses a text of gate.py; while it answers, gate.py goes
    and new.py comes, indexed: the pass's refused text is no longer in the store, new.py's text was never asked, so
    the encoder is not made the one read. A count of refusals subtracted from a count of texts lacking (the two
    cancel out) -> RED."""
    root = folder(tmp_path, "src", {"gate.py": GATE, "peels.py": PEELS})
    store = Store(tmp_path / "s.myc", lambda p: sqlite3.connect(p, check_same_thread=False))
    Indexer(store).index("oo", root)
    inside, release = threading.Event(), threading.Event()

    class Slow(Engine):
        def embed_many(self, model, texts, timeout=None):
            if not inside.is_set():
                inside.set()
                assert release.wait(10)
            return super().embed_many(model, texts, timeout)

    seams(Registry({"enc-a": Slow(refuse=("drops what its sources name",))}), Governor())
    out = {}
    passing = threading.Thread(target=lambda: out.setdefault("counts", Indexer(store).embed(Embedder(A), batch=100)))
    passing.start()
    try:
        assert inside.wait(10)
        (root / "gate.py").unlink()
        (root / "new.py").write_text("X = 1\n", encoding="utf-8")
        Indexer(store).index("oo", root)
    finally:
        release.set()
        passing.join(10)
    counts = out["counts"]
    assert counts["refused"] == 1 and counts["uncovered"] > 0 and not counts.get("activated")
    assert texts_without_vector(store, A) > 0 and store.active_encoder() is None
    store.close()


def test_mv22_a_question_vector_of_any_finite_size_is_scored_as_a_unit_vector(corpus, seams):
    """MV22 -- the engine answers the question with a finite vector whose components are past float32's range:
    the dense stage scores it as the unit vector it points to, every score finite, the best part first. A question
    vector cast before it is normalized (every score NaN, parts given in id order as if found) -> RED."""
    store = corpus["store"]
    seams(Registry({"enc-a": Engine()}), Governor())
    Indexer(store).embed(Embedder(A))
    searcher = Searcher(store)
    plain = searcher.dense("probe summary sources entity number date")

    class Huge(Engine):
        def embed_many(self, model, texts, timeout=None):
            return [[x * 1e300 for x in v] for v in super().embed_many(model, texts, timeout)]

    seams(Registry({"enc-a": Huge()}), Governor())
    scaled = searcher.dense("probe summary sources entity number date")
    assert all(score == score and abs(score) <= 1.01 for _id, score in scaled) and searcher.silent == {}
    assert [i for i, _s in scaled] == [i for i, _s in plain]


def test_mv23_a_busy_engine_holds_the_pass_and_a_final_refusal_of_its_gate_stops_it(corpus, seams):
    """MV23 -- the engine answers 503 (its queue full) or 429: the pass is held, not missing, and says the engine is
    busy; the engine's own gate refuses with the governor's GovernorRefusal: held when a wait lifts it, missing when
    none does. A busy engine reported missing, or a gate's final refusal reported held -> RED."""
    from opti_oignon.resource_governor import GovernorRefusal

    class GateDecision(Decision):
        def refusal_payload(self):
            return {"message": f"admission refused: {self.reason}"}

    store = corpus["store"]
    cases = ((ResponseError("server busy, please try again", 503), "held"),
             (ResponseError("too many requests", 429), "held"),
             (GovernorRefusal(GateDecision(False, "busy")), "held"),
             (GovernorRefusal(GateDecision(False, "background_cost_unknown")), "missing"))
    for failure, verdict in cases:
        engine = Engine(down=failure)
        seams(Registry({"enc-a": engine}), Governor())
        indexer = Indexer(store)
        counts = indexer.embed(Embedder(A), batch=4)
        assert len(engine.calls) == 1 and counts[verdict] is True, failure
        assert counts["held" if verdict == "missing" else "missing"] is False and str(failure) in indexer.last_stop
    assert store.active_encoder() is None


class Ollama(Engine):
    """An engine stand-in that names itself as the registry's Ollama backend does."""

    name = "ollama"


class LlamaServer(Engine):
    name = "llama_server"


class Told(Governor):
    """The admitting stand-in, recording the engine each admission is told."""

    def __init__(self):
        super().__init__()
        self.engines = []

    def admit(self, model, requested_ctx=None, caller="chat", **kw):
        self.engines.append(kw.get("engine"))
        return super().admit(model, requested_ctx, caller, **kw)


def test_mv24_the_governor_and_the_engine_are_asked_under_the_engines_name_for_the_encoder(tmp_path, seams):
    """MV24 -- Ollama shows every loaded model with its tag and resolves a name with no tag to its ``:latest``: an
    encoder the settings name ``enc`` is admitted and asked as ``enc:latest`` when Ollama serves it, for a pass and
    for a question alike; ``enc:q8`` and a registry's ``reg.local:5000/enc`` (a port is no tag) are resolved by the
    same rule; another engine is asked under the name as given. The governor is told the engine, which its own
    cache of the registry's routes, kept under the settings' name, no longer answers for the engine's name. The
    store keeps the encoder as the settings name it. The settings' name sent to Ollama's governor, a tag added for
    another engine, or no engine told -> RED."""
    for engine, model, asked in ((Ollama(), "enc", "enc:latest"), (Ollama(), "enc:q8", "enc:q8"),
                                 (Ollama(), "reg.local:5000/enc", "reg.local:5000/enc:latest"),
                                 (LlamaServer(), "enc", "enc")):
        root = folder(tmp_path, f"src{len(asked)}{engine.name}", {"gate.py": GATE})
        store = Store(tmp_path / f"{len(asked)}{engine.name}.myc", sqlite3.connect)
        try:
            Indexer(store).index("oo", root)
            governor = Told()
            seams(Registry({model: engine}), governor)
            encoder = Encoder(model, DIM, "query: ", "text: ")
            counts = Indexer(store).embed(Embedder(encoder), batch=8)
            Searcher(store).search("probe summary sources", stages=("dense",))
            assert counts["embedded"] > 0 and counts["missing"] is False, (model, counts)
            assert {name for name, _c in governor.asked} == {asked}, (model, governor.asked)
            assert {name for name, _t in engine.calls} == {asked}, (model, engine.calls)
            assert [c for _n, c in governor.asked][-1] == "chat"
            assert governor.engines and set(governor.engines) == {engine.name}, (model, governor.engines)
            assert store.active_encoder()["key"] == encoder.key
        finally:
            store.close()


def test_mv25_the_real_governor_admits_the_pass_of_an_encoder_named_without_its_tag(corpus, seams, tmp_path,
                                                                                    monkeypatch):
    """MV25 -- the real resource governor, its view of the engine set by hand and its store a plain SQLite file:
    Ollama holds ``enc:latest`` at the context it was loaded with, on a card with room. The pass of the encoder the
    settings name ``enc`` asks as the background and is admitted under a ticket that loads nothing, every text
    embedded. Asking under ``enc``: the governor sees no resident and refuses the background a load of unknown cost,
    for good -> RED."""
    from opti_oignon import resource_governor as rg

    monkeypatch.setattr(rg, "_safe_connect", lambda path, **kw: sqlite3.connect(
        str(path), check_same_thread=kw.get("check_same_thread", False)))
    gov = rg.ResourceGovernor(config_path=str(tmp_path / "no-governor.yaml"), db_path=str(tmp_path / "governor.db"),
                              warmup=None, registry=None, meminfo_path=str(tmp_path / "no-meminfo"), hardware=None)
    gov._config = rg.GovernorConfig(total_vram_gb=10.0, safety_margin_gb=1.5, kv_coefficient=0.5)
    held = rg.LoadedModelView(name="enc:latest", size_vram_bytes=512 * 1024 ** 2, context_length=2048)
    gov._snapshot = rg.ResourceSnapshot(taken_at=gov._clock(), ttl_s=9999.0, loaded=[held], capacity_gb=10.0,
                                        vram_in_use_gb=0.5, ram_available_mb=64000.0)
    engine = Ollama()
    seams(Registry({"enc": engine}), gov)
    store = corpus["store"]
    counts = Indexer(store).embed(Embedder(Encoder("enc", DIM, "query: ", "text: ")), batch=8)
    assert counts["embedded"] == distinct_texts(store) and counts["missing"] is False and counts["held"] is False, \
        counts
    assert engine.tickets and all(t.admitted and t.load_expected is False for t in engine.tickets)
    assert {t.model for t in engine.tickets} == {"enc:latest"}


class Mute(Engine):
    def endpoint(self):
        raise RuntimeError("cannot tell")


@pytest.mark.parametrize("case", ["remote", "forwards", "mute"])
def test_mv26_the_texts_and_the_question_never_leave_this_machine(corpus, seams, case):
    """MV26 -- an encoder served at another machine's address, by a backend that forwards its requests, or by one
    that cannot say where it sends: a pass sends nothing and stops as missing, saying the encoder is not on this
    machine; a question sends nothing and the search answers by its other stages, saying so. A text or a question
    sent wherever the registry points -> RED."""
    engine = {"remote": lambda: Engine(where="http://10.0.0.9:11434"),
              "forwards": lambda: type("Forwarding", (Engine,), {"forwards_requests": True})(),
              "mute": Mute}[case]()
    store = corpus["store"]
    seams(Registry({"enc-a": Engine()}), Governor())
    indexer = Indexer(store)
    assert indexer.embed(Embedder(A))["embedded"] > 0 and store.active_encoder()["key"] == A.key
    seams(Registry({"enc-a": engine}), Governor())
    searcher = Searcher(store)
    found = searcher.search("select_peels share of terms")
    assert found and found[0]["path"] == "peels.py"
    assert "not served on this machine" in searcher.silent["dense"], searcher.silent
    (corpus["root"] / "helper.py").write_text(HELPER, encoding="utf-8")
    Indexer(store).index("oo", corpus["root"])
    counts = indexer.embed(Embedder(A))
    assert counts["missing"] is True and counts["embedded"] == 0 and "not served on this machine" in indexer.last_stop
    assert engine.calls == [], ("nothing sent off the machine", engine.calls)
