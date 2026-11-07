"""Contracts of the Mycelium's third layer: the bridge's words and the reranker's verdicts.

MB1 the bridge asks the registry's backend (generate, thinking off, temperature 0) as interactive work, under its
    admission's ticket and at the context it names (none when it names none); the real names exist
MB2 a French question is put into English for the symbol table and the full text; the vectors and the rewrite get
    it as typed; an English question asks no translation
MB3 the rewrite is a second full-text list, named in each hit's why; a rewrite equal to the worded question adds
    none
MB4 a model away, held, failing or answering nothing leaves the question as typed and the search answering, and
    says why; a question asked again asks nothing; the kept questions are bounded
MB5 the reranker puts the passages judged "yes" first, then the unknown, then the "no", each group in its fused
    order; the passages beyond its depth follow as they came; each judged passage carries its verdict
MB6 the judge is shown each passage as a document: its path and lines, at most max_chars, one token asked; a
    passage whose source changed is not judged
MB7 a verdict that cannot be had is unknown; none at all leaves the order as fused, and silent says why; ``found``
    says whether the first passage answers
MB8 every call carries the settings' timeout, under the name the engine reads it by
MB9 the reranker judges as deep as its depth, past k, before the search is cut to k
MB10 a search that does not ask the bridge says nothing the bridge said for another search
MB11 the bridge's and the reranker's settings come from mycelium.yaml, refused by name when malformed or past
    their bounds
MB12 ``found`` is True only when the first passage was judged to answer, False only when every passage judged says
    no, None when one could not be judged; a verdict is read as a word, in English or French
MB13 a thinking model's reasoning is never taken for its answer: a closed block is dropped, an open one is nothing
MB14 a call carries the admission's keep_alive, and a llama-server backend its one-token bound as max_tokens
MB15 a call past its timeout is given up whatever the backend does, and while it lives no call to that model starts
MB16 French is told by its words, apostrophes split, an acronym or a loanword's accents not counted
MB17 each search keeps its own silences; k=0 judges nothing; the question is bounded in every prompt; a final
    refusal is named; a reranker asked and absent is said; a bridge and a lexical_query together are refused; a
    setting the code does not read is refused
MB18 the bridge and the reranker never send to a backend that is not on this machine, in any mode
MB19 a call does not start when a call given up came to live while it was being resolved
MB20 a call whose thread cannot start leaves the question as typed and hands its admission back
MB21 a call's resolution and answer share one timeout, and a reranking's judges share one
"""

import inspect
import sqlite3

import pytest

from opti_oignon.mycelium import bridge as bridge_module
from opti_oignon.mycelium import rerank as rerank_module
from opti_oignon.mycelium import vectors
from opti_oignon.mycelium.bridge import TRANSLATE, Bridge, BridgeSettings, is_french
from opti_oignon.mycelium.index import Indexer
from opti_oignon.mycelium.rerank import SYSTEM, Reranker, RerankSettings
from opti_oignon.mycelium.search import Searcher

BUDGET_S = {
    "test_mb1_the_bridge_asks_the_registry_as_interactive_work_under_its_ticket_at_the_admitted_context": 1.5,
    "test_mb2_a_french_question_is_put_into_english_for_the_words_and_typed_for_the_vectors": 1.5,
    "test_mb3_the_rewrite_is_a_second_full_text_list_named_in_the_why": 1.5,
    "test_mb4_a_model_that_cannot_be_asked_leaves_the_question_as_typed_and_says_why": 1.5,
    "test_mb5_yes_first_then_unknown_then_no_each_in_its_fused_order": 1.5,
    "test_mb6_the_judge_sees_each_passage_as_a_bounded_document_and_never_a_stale_one": 1.5,
    "test_mb7_an_unjudged_passage_is_unknown_none_judged_keeps_the_fused_order_and_found_says_it": 1.5,
    "test_mb8_every_call_carries_the_settings_timeout_under_the_engines_name": 1.5,
    "test_mb9_the_reranker_judges_past_k_before_the_cut_to_k": 1.5,
    "test_mb10_a_search_that_does_not_ask_the_bridge_says_nothing_it_said_before": 1.5,
    "test_mb11_the_bridge_and_the_reranker_settings_are_read_strictly": 1.5,
    "test_mb12_found_is_true_false_or_unknown_and_a_verdict_is_read_as_a_word": 1.5,
    "test_mb13_a_thinking_models_reasoning_is_never_taken_for_its_answer": 1.5,
    "test_mb14_a_call_carries_the_admissions_keep_alive_and_llama_server_its_bound": 1.5,
    "test_mb15_a_call_past_its_timeout_is_given_up_and_none_starts_while_it_lives": 3.0,
    "test_mb16_french_is_told_by_its_words": 1.5,
    "test_mb17_each_search_keeps_its_own_silences_and_the_edges_are_said": 1.5,
    "test_mb18_the_bridge_and_the_reranker_never_send_off_this_machine": 1.5,
    "test_mb19_a_call_does_not_start_beside_a_call_given_up_meanwhile": 3.0,
    "test_mb20_a_call_that_cannot_start_hands_its_admission_back": 1.5,
    "test_mb21_a_call_and_a_reranking_each_keep_one_timeout": 3.0,
}

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

FRENCH_QUESTION = "comment le resume est-il verifie contre ses sources ?"
ENGLISH = "how is the summary checked against its sources"
REWRITE = "probe_summary eviction gate summary sources"


class Decision:
    def __init__(self, admitted, reason="", num_ctx=None, keep_alive=None):
        self.admitted, self.reason, self.num_ctx, self.keep_alive = admitted, reason, num_ctx, keep_alive


class Governor:
    def __init__(self, held=False, num_ctx=None):
        self.held, self.num_ctx = held, num_ctx
        self.asked, self.decisions = [], []

    def admit(self, model, requested_ctx=None, caller="chat", **_kw):
        self.asked.append((model, caller))
        self.decisions.append(Decision(not self.held, "busy" if self.held else "", self.num_ctx))
        return self.decisions[-1]


class Said:
    def __init__(self, content):
        self.content = content


class Model:
    """A stand-in chat backend on this machine: answers the translation, the rewrite and the judge by what it is
    asked."""

    name = "ollama"
    forwards_requests = False

    def __init__(self, verdicts=None, fail=False, empty=False, judge_fails=()):
        self.calls, self.tickets = [], []
        self.verdicts, self.fail, self.empty, self.judge_fails = verdicts or {}, fail, empty, tuple(judge_fails)

    def endpoint(self):
        return "http://127.0.0.1:11434"

    def generate(self, model, messages, options=None, keep_alive="30m", think=None, images=None):
        from opti_oignon.resource_governor import get_active_ticket

        self.calls.append({"model": model, "messages": messages, "options": dict(options or {}), "think": think,
                           "keep_alive": keep_alive})
        self.tickets.append(get_active_ticket())
        if self.fail:
            raise ConnectionError("the engine is down")
        if self.empty:
            return Said("")
        prompt = messages[-1]["content"]
        if prompt.startswith("<Instruct>"):
            if any(marker in prompt for marker in self.judge_fails):
                raise RuntimeError("judge failed")
            for marker, verdict in self.verdicts.items():
                if marker in prompt:
                    return Said(verdict)
            return Said("no")
        if prompt.startswith("You are a professional French"):
            return Said(ENGLISH)
        return Said(REWRITE + "\nand a second line")

    def judged(self):
        return [c for c in self.calls if c["messages"][-1]["content"].startswith("<Instruct>")]


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


@pytest.fixture(autouse=True)
def no_call_left_given_up():
    """A contract that fails before its given-up call returns leaves that call in ``bridge._hung``: emptied after
    each contract, so the next one is never told that a call of another contract still lives."""
    yield
    bridge_module._hung.clear()


@pytest.fixture
def seams(monkeypatch):
    def install(registry, governor):
        monkeypatch.setattr(vectors, "_registry", lambda: registry)
        monkeypatch.setattr(vectors, "_governor", lambda: governor)

    return install


@pytest.fixture
def corpus(tmp_path):
    from opti_oignon.mycelium.store import Store

    root = tmp_path / "src"
    root.mkdir()
    (root / "peels.py").write_text(PEELS, encoding="utf-8")
    (root / "gate.py").write_text(GATE, encoding="utf-8")
    store = Store(tmp_path / "corpus.myc", sqlite3.connect)
    Indexer(store).index("oo", root)
    yield {"store": store, "root": root}
    store.close()


def bridge(cache=256, translate=True, rewrite=True):
    return Bridge(BridgeSettings("m-4b", translate, rewrite, 30.0, cache))


def reranker(depth=20, max_chars=3000):
    return Reranker(RerankSettings("m-4b", depth, 4, max_chars, 60.0))


def test_mb1_the_bridge_asks_the_registry_as_interactive_work_under_its_ticket_at_the_admitted_context(seams):
    """MB1 -- the bridge's calls go to the registry's backend for its model, admitted as ``chat``, made under that
    decision's ticket, thinking off, temperature 0; the context sent is the one the admission names, and none when
    it names none. Another way to the engine, no admission, a call outside the ticket, a context of its own -> RED."""
    model, governor = Model(), Governor(num_ctx=8192)
    seams(Registry({"m-4b": model}), governor)
    b = bridge()
    assert b.english(FRENCH_QUESTION) == ENGLISH
    assert b.rewritten(FRENCH_QUESTION) == REWRITE
    assert governor.asked == [("m-4b", "chat"), ("m-4b", "chat")]
    assert model.tickets == governor.decisions
    assert all(c["model"] == "m-4b" and c["think"] is False and c["options"]["temperature"] == 0.0
               and c["options"]["num_ctx"] == 8192 for c in model.calls)
    model2, governor2 = Model(), Governor()
    seams(Registry({"m-4b": model2}), governor2)
    bridge().english(FRENCH_QUESTION)
    assert "num_ctx" not in model2.calls[0]["options"]
    assert model2.calls[0]["messages"][-1]["content"] == TRANSLATE.format(text=FRENCH_QUESTION)
    from opti_oignon.inference_backend import InferenceBackend

    parameters = inspect.signature(InferenceBackend.generate).parameters
    assert all(name in parameters for name in ("model", "messages", "options", "think"))


def test_mb2_a_french_question_is_put_into_english_for_the_words_and_typed_for_the_vectors(corpus, seams):
    """MB2 -- a French question: the symbol table and the full text are asked its English, the rewrite is asked
    the question as typed; an English question asks no translation; the detection reads function words and
    accents. The vectors asked the English, the full text asked the French -> RED."""
    model = Model()
    seams(Registry({"m-4b": model}), Governor())
    asked = {}
    searcher = Searcher(corpus["store"], bridge=bridge())
    searcher.lexical = lambda q, n=100, _orig=searcher.lexical: asked.setdefault("lexical", []).append(q) or _orig(q, n)
    searcher.symbols = lambda q, n=20, _orig=searcher.symbols: asked.setdefault("symbols", []).append(q) or _orig(q, n)
    searcher.dense = lambda q, n=100: asked.setdefault("dense", []).append(q) or []
    searcher.search(FRENCH_QUESTION)
    assert asked["symbols"] == [ENGLISH] and asked["lexical"] == [ENGLISH, REWRITE]
    assert asked["dense"] == [FRENCH_QUESTION]
    rewrite_prompts = [c["messages"][-1]["content"] for c in model.calls if "Question:" in c["messages"][-1]["content"]]
    assert rewrite_prompts and rewrite_prompts[0].endswith("Question: " + FRENCH_QUESTION)
    model.calls.clear()
    searcher.search("how does the gate probe a summary")
    assert not [c for c in model.calls if c["messages"][-1]["content"].startswith("You are a professional French")]
    assert is_french("comment \u00e7a marche ?") and is_french("o\u00f9 est la porte")
    assert not is_french("where is the gate")


def test_mb3_the_rewrite_is_a_second_full_text_list_named_in_the_why(corpus, seams):
    """MB3 -- the rewrite's words are a list of their own (``rewritten#n`` in a hit's why) that brings a part the
    question's words do not reach; a rewrite equal to the worded question adds no list. The rewrite dropped, or
    merged into the first list -> RED."""
    seams(Registry({"m-4b": Model()}), Governor())
    searcher = Searcher(corpus["store"], bridge=bridge())
    hits = searcher.search("why would my text lose its names", stages=("symbols", "lexical"))
    gate = [h for h in hits if h["path"] == "gate.py"]
    assert gate and any(w.startswith("rewritten#") for h in gate for w in h["why"])
    plain = Searcher(corpus["store"]).search("why would my text lose its names", stages=("symbols", "lexical"))
    assert not any(w.startswith("rewritten#") for h in plain for w in h["why"])

    class Same(Model):
        def generate(self, model, messages, **kw):
            super().generate(model, messages, **kw)
            return Said("select_peels share of terms")

    seams(Registry({"m-4b": Same()}), Governor())
    hits = Searcher(corpus["store"], bridge=bridge()).search("select_peels share of terms",
                                                             stages=("symbols", "lexical"))
    assert hits and not any(w.startswith("rewritten#") for h in hits for w in h["why"])


def test_mb4_a_model_that_cannot_be_asked_leaves_the_question_as_typed_and_says_why(corpus, seams):
    """MB4 -- no backend, a held admission, an engine down, an empty answer: the words-matching lists get the
    question as typed, no rewritten list, the search answers, ``silent`` says why; a question asked again asks
    nothing; the kept questions are bounded. An exception to the chat, a silence that does not say why, a cache
    that grows without end -> RED."""
    for registry, governor, reason in ((Registry({}), Governor(), "no backend serves m-4b"),
                                       (Registry({"m-4b": Model()}), Governor(held=True), "held by the governor"),
                                       (Registry({"m-4b": Model(fail=True)}), Governor(), "did not answer"),
                                       (Registry({"m-4b": Model(empty=True)}), Governor(), "answered nothing")):
        seams(registry, governor)
        searcher = Searcher(corpus["store"], bridge=bridge())
        hits = searcher.search("comment select_peels choisit les pelures ?", stages=("symbols", "lexical"))
        assert hits and hits[0]["path"] == "peels.py"
        assert reason in searcher.silent["translate"] and reason in searcher.silent["rewrite"]
        assert not any(w.startswith("rewritten#") for h in hits for w in h["why"])
    model = Model()
    seams(Registry({"m-4b": model}), Governor())
    b = bridge(cache=2)
    b.english(FRENCH_QUESTION)
    b.english(FRENCH_QUESTION)
    assert len(model.calls) == 1
    for n in range(5):
        b.english(f"comment le module {n} est-il fait ?")
    assert len(b._kept) == 2


def judged_order(hits):
    return [(h["path"], h["first"], h.get("verdict")) for h in hits]


def test_mb5_yes_first_then_unknown_then_no_each_in_its_fused_order(corpus, seams):
    """MB5 -- over a search's first passages: those judged "yes" first, then the unknown, then the "no", each group
    in the fused order; beyond the depth, as they came; each judged passage carries its verdict in its why. A sort
    that breaks a group's fused order, passages beyond the depth judged or moved -> RED."""
    model = Model(verdicts={"eviction gate": "yes", "oldest first": "Maybe"})
    seams(Registry({"m-4b": model}), Governor())
    store = corpus["store"]
    plain = Searcher(store).search("summary peels sources terms query", stages=("lexical",))
    assert len(plain) >= 4
    hits = Searcher(store, reranker=reranker(depth=3)).search("summary peels sources terms query",
                                                              stages=("lexical",), rerank=True)
    head = hits[:3]
    verdicts = [h["verdict"] for h in head]
    assert verdicts == sorted(verdicts, key=lambda v: {"yes": 0, "unknown": 1, "no": 2}[v])
    fused = {(h["path"], h["first"]): n for n, h in enumerate(plain)}
    for group in ("yes", "unknown", "no"):
        places = [fused[(h["path"], h["first"])] for h in head if h["verdict"] == group]
        assert places == sorted(places)
    assert {(h["path"], h["first"]) for h in head} == {(h["path"], h["first"]) for h in plain[:3]}
    assert [(h["path"], h["first"]) for h in hits[3:]] == [(h["path"], h["first"]) for h in plain[3:]]
    assert all("verdict" not in h for h in hits[3:]) and len(model.judged()) == 3
    assert all(f"rerank:{h['verdict']}" in h["why"] for h in head)
    assert "yes" in verdicts or "unknown" in verdicts


def test_mb6_the_judge_sees_each_passage_as_a_bounded_document_and_never_a_stale_one(corpus, seams):
    """MB6 -- the judge is asked with the card's system instruction, one token, the question and a document that
    starts with the passage's path and lines and holds at most max_chars; a passage whose source changed since its
    pass is not shown. An unbounded passage, a stale text judged, more than one token asked -> RED."""
    model = Model()
    seams(Registry({"m-4b": model}), Governor())
    store = corpus["store"]
    searcher = Searcher(store, reranker=reranker(depth=5, max_chars=60))
    hits = searcher.search("summary peels sources", stages=("lexical",), rerank=True)
    for call in model.judged():
        system, user = call["messages"][0]["content"], call["messages"][-1]["content"]
        document = user.split("<Document>: ", 1)[1]
        assert system == SYSTEM and call["options"]["num_predict"] == 1
        assert "<Query>: summary peels sources" in user and len(document) <= 60
        assert document.split("\n", 1)[0].split(":")[0] in ("peels.py", "gate.py")
    assert len(model.judged()) == min(5, len(hits))
    (corpus["root"] / "gate.py").write_text(GATE.replace("Entity, number", "Names, numbers"), encoding="utf-8")
    model.calls.clear()
    hits = searcher.search("summary peels sources", stages=("lexical",), rerank=True)
    shown = [c["messages"][-1]["content"].split("<Document>: ", 1)[1] for c in model.judged()]
    assert shown and not any("Entity, number" in d or "Names, numbers" in d for d in shown)
    stale = [h for h in hits[:5] if h["path"] == "gate.py" and h["symbol"] == "probe_summary"]
    assert stale and stale[0]["verdict"] == "unknown"


def test_mb7_an_unjudged_passage_is_unknown_none_judged_keeps_the_fused_order_and_found_says_it(corpus, seams):
    """MB7 -- a passage whose judge fails is unknown and the others are judged; every judge failing (or the model
    away) leaves the order as fused and no passage with a verdict, ``silent`` says why, ``found`` is None; ``found``
    is True when the first is a "yes", False when no passage is. A failure that drops a passage, reorders, or
    reads as "no" answer found -> RED."""
    store = corpus["store"]
    plain = Searcher(store).search("summary peels sources", stages=("lexical",))
    seams(Registry({"m-4b": Model(verdicts={"eviction gate": "yes"}, judge_fails=("oldest first",))}), Governor())
    searcher = Searcher(store, reranker=reranker(depth=10))
    hits = searcher.search("summary peels sources", stages=("lexical",), rerank=True)
    assert len(hits) == len(plain) and "unjudged" in searcher.silent["rerank"]
    assert any(h.get("verdict") == "unknown" for h in hits) and Reranker.found(hits) is True
    seams(Registry({}), Governor())
    hits = searcher.search("summary peels sources", stages=("lexical",), rerank=True)
    assert [(h["path"], h["first"]) for h in hits] == [(h["path"], h["first"]) for h in plain]
    assert "no backend serves m-4b" in searcher.silent["rerank"] and Reranker.found(hits) is None
    assert all("verdict" not in h for h in hits)
    seams(Registry({"m-4b": Model()}), Governor())
    hits = searcher.search("summary peels sources", stages=("lexical",), rerank=True)
    assert all(h.get("verdict") == "no" for h in hits[:10]) and Reranker.found(hits) is False


def test_mb8_every_call_carries_the_settings_timeout_under_the_engines_name(corpus, seams):
    """MB8 -- the translation, the rewrite and each verdict are asked with the bridge's or the reranker's timeout in
    their options (what is left of it), under the name the engine's backends read a call's deadline by; so a model
    that hangs is left after that many seconds. A timeout kept in the settings and never sent (the engine's own
    default then) -> RED."""
    from opti_oignon import inference_backend

    model = Model()
    seams(Registry({"m-4b": model}), Governor())
    searcher = Searcher(corpus["store"], bridge=bridge(), reranker=Reranker(RerankSettings("m-4b", 3, 2, 3000, 7.0)))
    searcher.search(FRENCH_QUESTION, stages=("symbols", "lexical"), rerank=True)
    assert bridge_module.TIMEOUT_OPTION == inference_backend.TIMEOUT_OPTION
    words = [c for c in model.calls if not c["messages"][-1]["content"].startswith("<Instruct>")]
    assert len(words) == 2 and all(29.0 < c["options"][bridge_module.TIMEOUT_OPTION] <= 30.0 for c in words)
    assert model.judged() and all(6.0 < c["options"][bridge_module.TIMEOUT_OPTION] <= 7.0 for c in model.judged())


def test_mb9_the_reranker_judges_past_k_before_the_cut_to_k(corpus, seams):
    """MB9 -- k=2 and a reranker of depth 6: the six first fused passages are judged, and a passage the fusion put
    past the second place but the judge says answers comes within the two given. A reranker that only sees the k
    passages the search would give without it -> RED."""
    store = corpus["store"]
    plain = Searcher(store).search("summary peels sources terms query", k=6, stages=("lexical",))
    assert len(plain) >= 4
    late = plain[3]
    model = Model(verdicts={f"{late['path']}:{late['first']}-{late['last']}\n": "yes"})
    seams(Registry({"m-4b": model}), Governor())
    hits = Searcher(store, reranker=reranker(depth=6)).search("summary peels sources terms query", k=2,
                                                              stages=("lexical",), rerank=True)
    assert len(hits) == 2 and len(model.judged()) == min(6, len(plain))
    assert (hits[0]["path"], hits[0]["first"]) == (late["path"], late["first"]) and hits[0]["verdict"] == "yes"


def test_mb10_a_search_that_does_not_ask_the_bridge_says_nothing_it_said_before(corpus, seams):
    """MB10 -- a search whose bridge failed says why; the next search, by the vectors alone, does not ask the
    bridge and says nothing of it. A bridge's silence carried into a search that never asked it -> RED."""
    seams(Registry({}), Governor())
    searcher = Searcher(corpus["store"], bridge=bridge())
    searcher.search(FRENCH_QUESTION, stages=("symbols", "lexical"))
    assert "no backend serves m-4b" in searcher.silent["translate"]
    searcher.search(FRENCH_QUESTION, stages=("dense",))
    assert "translate" not in searcher.silent and "rewrite" not in searcher.silent


def _write(tmp_path, name, text):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path / name


def test_mb11_the_bridge_and_the_reranker_settings_are_read_strictly(tmp_path):
    """MB11 -- the tree's mycelium.yaml gives the bridge's and the reranker's settings; a quoted flag, a timeout that
    is no positive number or past its bound, a cache below 0 or past its bound, a depth, parallel or max_chars past
    theirs, an empty model, are refused by name. A malformed file taken, or a bound that lets a setting make a
    search judge or keep without end -> RED."""
    b, r = bridge_module.load_settings(), rerank_module.load_settings()
    assert b.model == r.model == "qwen3.5:4b" and b.translate and b.rewrite and b.timeout_s == 30.0
    assert (r.depth, r.parallel, r.max_chars, r.timeout_s) == (20, 4, 3000, 60.0)
    good_b = "bridge:\n  model: m\n"
    good_r = "rerank:\n  model: m\n"
    assert bridge_module.load_settings(_write(tmp_path, "b.yaml", good_b)).cache == 256
    assert rerank_module.load_settings(_write(tmp_path, "r.yaml", good_r)).depth == 20
    bad_b = (good_b + "  translate: 'false'\n", good_b + "  timeout_s: 0\n", good_b + "  timeout_s: 100000\n",
             good_b + "  cache: -1\n", good_b + "  cache: 100000\n", "bridge:\n  model: ' '\n", "- bridge\n")
    for n, text in enumerate(bad_b):
        with pytest.raises(ValueError, match="mycelium.yaml"):
            bridge_module.load_settings(_write(tmp_path, f"bb{n}.yaml", text))
    bad_r = (good_r + "  depth: 1000\n", good_r + "  parallel: 64\n", good_r + "  max_chars: 0\n",
             good_r + "  timeout_s: .nan\n", good_r + "  depth: true\n", "rerank: [model]\n")
    for n, text in enumerate(bad_r):
        with pytest.raises(ValueError, match="mycelium.yaml"):
            rerank_module.load_settings(_write(tmp_path, f"rb{n}.yaml", text))


def test_mb12_found_is_true_false_or_unknown_and_a_verdict_is_read_as_a_word(corpus, seams):
    """MB12 -- the judge of the first fused passage fails and the others say no: ``found`` is None (the answer may
    be the passage never judged), never False; every judge saying no: False, unless the search gives passages past
    the reranker's depth, never judged (None); "Oui" and "**Yes**" read yes, "Non"
    no, "Not", "None", "Nope" and "<think>" unknown; every answer unreadable: the order as fused, ``found`` None, and
    ``silent`` says the verdicts could not be read. "No passage answers" said while a passage went unjudged, or a
    French yes lost while its no is kept -> RED."""
    from opti_oignon.mycelium.rerank import verdict_of

    store = corpus["store"]
    plain = Searcher(store).search("summary peels sources", stages=("lexical",))
    first = f"{plain[0]['path']}:{plain[0]['first']}-{plain[0]['last']}\n"
    seams(Registry({"m-4b": Model(judge_fails=(first,))}), Governor())
    searcher = Searcher(store, reranker=reranker(depth=3))
    hits = searcher.search("summary peels sources", stages=("lexical",), rerank=True)
    assert [h.get("verdict") for h in hits[:3]].count("unknown") == 1 and Reranker.found(hits) is None
    seams(Registry({"m-4b": Model()}), Governor())
    judged_all = searcher.search("summary peels sources", k=3, stages=("lexical",), rerank=True)
    assert Reranker.found(judged_all) is False
    shallow = Searcher(store, reranker=reranker(depth=1)).search("summary peels sources", k=5, stages=("lexical",),
                                                                 rerank=True)
    assert len(shallow) > 1 and shallow[0]["verdict"] == "no" and Reranker.found(shallow) is None
    assert [verdict_of(t) for t in ("Oui", "**Yes**", "yes.", "Non", "no", "Not", "None", "Nope", "<think>")] == [
        "yes", "yes", "yes", "no", "no", "unknown", "unknown", "unknown", "unknown"]

    class Unsure(Model):
        def generate(self, model, messages, **kw):
            super().generate(model, messages, **kw)
            return Said("Maybe")

    seams(Registry({"m-4b": Unsure()}), Governor())
    hits = searcher.search("summary peels sources", stages=("lexical",), rerank=True)
    assert [(h["path"], h["first"]) for h in hits] == [(h["path"], h["first"]) for h in plain]
    assert Reranker.found(hits) is None and "could not be read" in searcher.silent["rerank"]


def test_mb13_a_thinking_models_reasoning_is_never_taken_for_its_answer(corpus, seams):
    """MB13 -- a model that thinks aloud: its closed <think> block is dropped from the translation, the rewrite and
    the verdict; an answer that is only an open <think> block is nothing (the question as typed, said why, nothing
    kept); an answer whose block the prompt opened (only its close in the answer) keeps what follows the close.
    Reasoning sent to the full text as words, or kept for the next search -> RED."""

    class Aloud(Model):
        def __init__(self, closed=True):
            super().__init__(verdicts={"eviction gate": "yes"})
            self.closed = closed

        def generate(self, model, messages, **kw):
            said = super().generate(model, messages, **kw).content
            return Said(f"<think>\nOkay, the user wants it.\n</think>\n\n{said}" if self.closed
                        else "<think>\nOkay, the user wants it, so")

    seams(Registry({"m-4b": Aloud()}), Governor())
    b = bridge()
    assert b.english(FRENCH_QUESTION) == ENGLISH and b.rewritten(FRENCH_QUESTION) == REWRITE
    hits = Searcher(corpus["store"], reranker=reranker(depth=3)).search("summary peels sources",
                                                                        stages=("lexical",), rerank=True)
    assert any(h.get("verdict") == "yes" for h in hits)
    seams(Registry({"m-4b": Aloud(closed=False)}), Governor())
    b = bridge()
    assert b.english(FRENCH_QUESTION) == FRENCH_QUESTION and "answered nothing" in b.silent["translate"]
    assert b._kept == {}

    class Opened(Model):
        """A chat template that opens the block in the prompt: the answer holds only its close."""

        def generate(self, model, messages, **kw):
            said = super().generate(model, messages, **kw).content
            return Said(f"Okay, the user asks; I will translate.\n</think>\n\n{said}")

    seams(Registry({"m-4b": Opened()}), Governor())
    b = bridge()
    assert b.english(FRENCH_QUESTION) == ENGLISH and list(b._kept.values()) == [ENGLISH]


class Llama(Model):
    name = "llama_server"


def test_mb14_a_call_carries_the_admissions_keep_alive_and_llama_server_its_bound(corpus, seams):
    """MB14 -- the admission names a keep_alive (the governor's pressure relief): the call carries it; with none
    named, the backend's own; a llama-server backend, which reads a call's bound as max_tokens, gets the number
    asked (one token for a verdict), an Ollama one never sees the key. A keep_alive of the bridge's own over the
    governor's, or a verdict a llama-server lets run to its own limit -> RED."""

    class Soft(Governor):
        def admit(self, model, requested_ctx=None, caller="chat", **_kw):
            decision = super().admit(model, requested_ctx, caller)
            decision.keep_alive = "5m"
            return decision

    model = Model()
    seams(Registry({"m-4b": model}), Soft())
    bridge().english(FRENCH_QUESTION)
    assert model.calls[-1]["keep_alive"] == "5m" and "max_tokens" not in model.calls[-1]["options"]
    seams(Registry({"m-4b": model}), Governor())
    bridge().english(FRENCH_QUESTION)
    assert model.calls[-1]["keep_alive"] == "30m"
    llama = Llama()
    seams(Registry({"m-4b": llama}), Governor())
    Searcher(corpus["store"], bridge=bridge(), reranker=reranker(depth=2)).search(
        FRENCH_QUESTION, stages=("symbols", "lexical"), rerank=True)
    judged = llama.judged()
    assert judged and all(c["options"]["max_tokens"] == 1 for c in judged)
    assert all(c["options"]["max_tokens"] == c["options"]["num_predict"] for c in llama.calls)


def test_mb15_a_call_past_its_timeout_is_given_up_and_none_starts_while_it_lives(corpus, seams):
    """MB15 -- a backend that never returns and reads no timeout: the bridge gives the call up at its timeout, the
    search answers as typed and says so; while the call given up lives, the next search asks that model nothing --
    not even the registry for its backend -- and says why at once; once it has returned, the model is asked again.
    A search held as long as the backend holds, or a second call piled on a hung one -> RED."""
    import threading
    import time

    from opti_oignon.mycelium import bridge as module

    release = threading.Event()

    class Hung(Model):
        def generate(self, model, messages, **kw):
            super().generate(model, messages, **kw)
            release.wait(10)
            return Said(ENGLISH)

    class Counting(Registry):
        resolved = 0

        def resolve_backend(self, model):
            Counting.resolved += 1
            return super().resolve_backend(model)

    hung = Hung()
    seams(Counting({"m-4b": hung}), Governor())
    b = Bridge(BridgeSettings("m-4b", True, False, 0.3, 256))
    try:
        started = time.perf_counter()
        assert b.english(FRENCH_QUESTION) == FRENCH_QUESTION and time.perf_counter() - started < 2.0
        assert "gave no answer within 0.3" in b.silent["translate"]
        started = time.perf_counter()
        assert b.english("comment le module est-il fait ?") == "comment le module est-il fait ?"
        assert time.perf_counter() - started < 0.2 and len(hung.calls) == 1 and Counting.resolved == 1
        assert "still lives" in b.silent["translate"]
    finally:
        release.set()
    deadline = time.time() + 5
    while module._hanging("m-4b") and time.time() < deadline:
        time.sleep(0.02)
    seams(Registry({"m-4b": Model()}), Governor())
    assert bridge().english(FRENCH_QUESTION) == ENGLISH


def test_mb16_french_is_told_by_its_words(corpus):
    """MB16 -- "qu'est-ce que c'est ?" and "pourquoi select_peels renvoie rien" are French (an apostrophe splits
    words; "rien" is a French word); "how does the DES cipher plus padding work", "does LE mode use DES keys" and
    "why is the naive resume parser slow" with its accents are English (an acronym is not a French word, nor "plus";
    a loanword's accents alone make no French); a word of code (an underscore, a digit) is no word, and a hyphen
    joins French words only when each part is one ("est-ce", never "de-serialize"). -> RED otherwise."""
    assert is_french("qu'est-ce que c'est ?") and is_french("pourquoi select_peels renvoie rien")
    assert is_french("comment \u00e7a marche ?") and is_french("o\u00f9 est la porte")
    assert not is_french("how does the DES cipher plus padding work")
    assert not is_french("does LE mode use DES keys")
    for english in ("how do I de-serialize a C struct", "why does to_le_bytes fail in the C api",
                    "is ce_loss computed by de_dup", "how does un-escape handle la_config",
                    "does the un-zip step de-dup files",
                    "why is the le_la flag set by un_des"):
        assert not is_french(english), english
    assert not is_french("why is the na\u00efve r\u00e9sum\u00e9 parser slow")


def test_mb17_each_search_keeps_its_own_silences_and_the_edges_are_said(corpus, seams, tmp_path):
    """MB17 -- two searchers sharing one bridge: one search's failure is not in the other's silences; with k=0 and
    rerank, nothing is judged; a question past the bound is cut in every prompt; a final refusal is named as one; a
    search that asks a reranker it has not says so; a bridge given with a lexical_query is refused; a setting the
    code does not read is refused by name. -> RED otherwise."""
    from opti_oignon.mycelium import bridge as module

    shared = bridge()
    seams(Registry({}), Governor())
    failed = Searcher(corpus["store"], bridge=shared)
    failed.search(FRENCH_QUESTION, stages=("symbols", "lexical"))
    seams(Registry({"m-4b": Model()}), Governor())
    other = Searcher(corpus["store"], bridge=shared)
    other.search("comment les pelures sont-elles choisies ?", stages=("symbols", "lexical"))
    assert "translate" in failed.silent and "translate" not in other.silent
    model = Model()
    seams(Registry({"m-4b": model}), Governor())
    assert Searcher(corpus["store"], reranker=reranker()).search("summary", k=0, rerank=True) == []
    assert model.judged() == []
    long = "comment le resume " + "est-il verifie " * 2000
    bridge().english(long)
    assert len(model.calls[-1]["messages"][-1]["content"]) <= len(module.TRANSLATE) + module.MAX_QUESTION

    class Final(Governor):
        def admit(self, model, requested_ctx=None, caller="chat", **_kw):
            self.decisions.append(Decision(False, "background_cost_unknown"))
            return self.decisions[-1]

    seams(Registry({"m-4b": Model()}), Final())
    b = bridge()
    b.english(FRENCH_QUESTION)
    assert "no wait lifts it" in b.silent["translate"]
    seams(Registry({"m-4b": Model()}), Governor())
    b.english("comment les pelures sont-elles faites ?")
    assert b.silent == {}
    asked = Searcher(corpus["store"])
    asked.search("summary", rerank=True)
    assert "no reranker" in asked.silent["rerank"]
    with pytest.raises(ValueError, match="lexical_query"):
        Searcher(corpus["store"], bridge=bridge(), lexical_query=str.upper)
    (tmp_path / "typo.yaml").write_text("bridge:\n  model: m\n  timout_s: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="timout_s"):
        module.load_settings(tmp_path / "typo.yaml")


def test_mb18_the_bridge_and_the_reranker_never_send_off_this_machine(corpus, seams, monkeypatch):
    """MB18 -- a backend whose endpoint is another machine, one that forwards its requests to another process, or
    one that cannot say where it sends: nothing is sent, the question stays as typed, the order as fused, and
    ``silent`` says the model is not on this machine; the real locality check is the one asked. A private passage
    or question sent off the machine -> RED."""

    class Remote(Model):
        def endpoint(self):
            return "http://10.0.0.5:11434"

    class Forwarding(Model):
        forwards_requests = True

    class Mute(Model):
        def endpoint(self):
            raise RuntimeError("cannot tell")

    for backend in (Remote(), Forwarding(), Mute()):
        seams(Registry({"m-4b": backend}), Governor())
        searcher = Searcher(corpus["store"], bridge=bridge(), reranker=reranker(depth=3))
        hits = searcher.search(FRENCH_QUESTION, stages=("symbols", "lexical"), rerank=True)
        assert backend.calls == [] and all("verdict" not in h for h in hits), type(backend).__name__
        assert "not on this machine" in searcher.silent["translate"]
    from opti_oignon import inference_backend

    assert inference_backend.is_on_this_machine("http://127.0.0.1:11434")
    assert not inference_backend.is_on_this_machine("http://10.0.0.5:11434")
    asked = []
    monkeypatch.setattr(inference_backend, "is_on_this_machine", lambda url: asked.append(url) or False)
    local = Model()
    seams(Registry({"m-4b": local}), Governor())
    b = bridge()
    assert b.english(FRENCH_QUESTION) == FRENCH_QUESTION and local.calls == []
    assert asked == ["http://127.0.0.1:11434"] and "not on this machine" in b.silent["translate"]


def test_mb19_a_call_does_not_start_beside_a_call_given_up_meanwhile(seams):
    """MB19 -- a first call hangs; a second is being resolved while the first is given up: the second, resolved,
    sees the call given up still alive and does not start. A check made only before the resolution (the second
    call piled on the hung one) -> RED."""
    import threading
    import time

    from opti_oignon.mycelium import bridge as module

    release = threading.Event()

    class Hung(Model):
        def generate(self, model, messages, **kw):
            super().generate(model, messages, **kw)
            release.wait(10)
            return Said(ENGLISH)

    class Slow(Registry):
        def __init__(self, engines):
            super().__init__(engines)
            self.resolved = 0

        def resolve_backend(self, model):
            self.resolved += 1
            if self.resolved == 2:
                deadline = time.time() + 5
                while not module._hanging(model) and time.time() < deadline:
                    time.sleep(0.01)
            return super().resolve_backend(model)

    hung = Hung()
    seams(Slow({"m-4b": hung}), Governor())
    errors = {}

    def run(name, asker):
        try:
            asker.ask("x", 4)
        except module.AskError as exc:
            errors[name] = str(exc)

    first = threading.Thread(target=run, args=("first", module.Asker("m-4b", 0.3)))
    second = threading.Thread(target=run, args=("second", module.Asker("m-4b", 3.0)))
    try:
        first.start()
        deadline = time.time() + 5
        while not hung.calls and time.time() < deadline:
            time.sleep(0.01)
        second.start()
        first.join(5)
        second.join(5)
        assert "gave no answer" in errors["first"] and "still lives" in errors.get("second", "")
        assert len(hung.calls) == 1
    finally:
        release.set()
    deadline = time.time() + 5
    while module._hanging("m-4b") and time.time() < deadline:
        time.sleep(0.02)


def test_mb20_a_call_that_cannot_start_hands_its_admission_back(seams, monkeypatch):
    """MB20 -- the thread of an admitted call cannot start (the process is out of threads): the question stays as
    typed, ``silent`` says why, and the admission is handed back to the governor (``end_pending_load``) rather
    than left counted until it expires. An exception to the chat, or an admission kept -> RED."""
    import threading
    import types

    from opti_oignon.mycelium import bridge as module

    starts = []

    class Failing(threading.Thread):
        def start(self):
            starts.append(self.name)
            if len(starts) >= 2:
                raise RuntimeError("can't start new thread")
            super().start()

    monkeypatch.setattr(module, "threading", types.SimpleNamespace(Thread=Failing, Lock=threading.Lock))

    class Ticketed(Governor):
        def __init__(self):
            super().__init__()
            self.handed = []

        def admit(self, model, requested_ctx=None, caller="chat", **_kw):
            decision = super().admit(model, requested_ctx, caller)
            decision.ticket_id = f"t{len(self.decisions)}"
            return decision

        def end_pending_load(self, ticket_id):
            self.handed.append(ticket_id)

    governor, model = Ticketed(), Model()
    seams(Registry({"m-4b": model}), governor)
    b = bridge()
    assert b.english(FRENCH_QUESTION) == FRENCH_QUESTION and model.calls == []
    assert "could not start" in b.silent["translate"] and governor.handed == ["t1"]


def test_mb21_a_call_and_a_reranking_each_keep_one_timeout(corpus, seams):
    """MB21 -- a call whose resolution took part of its timeout is sent with what is left, not the whole again;
    a reranking whose judges are slow stops judging once its timeout is spent, the rest unknown and said, the
    deadline each judge is sent shrinking. A call waiting twice its timeout, or a reranking waiting its timeout
    once per round -> RED."""
    import time

    from opti_oignon.mycelium import bridge as module

    class SlowResolve(Registry):
        def resolve_backend(self, model):
            time.sleep(0.3)
            return super().resolve_backend(model)

    model = Model()
    seams(SlowResolve({"m-4b": model}), Governor())
    b = Bridge(BridgeSettings("m-4b", True, False, 0.6, 256))
    assert b.english(FRENCH_QUESTION) == ENGLISH
    assert model.calls[-1]["options"][module.TIMEOUT_OPTION] <= 0.35

    class Steady(Model):
        def generate(self, model, messages, **kw):
            out = super().generate(model, messages, **kw)
            time.sleep(0.2)
            return out

    steady = Steady()
    seams(Registry({"m-4b": steady}), Governor())
    searcher = Searcher(corpus["store"], reranker=Reranker(RerankSettings("m-4b", 6, 1, 3000, 0.7)))
    started = time.perf_counter()
    hits = searcher.search("summary peels sources terms query", stages=("lexical",), rerank=True)
    elapsed = time.perf_counter() - started
    sent = [c["options"][module.TIMEOUT_OPTION] for c in steady.judged()]
    assert len(hits) >= 5 and len(sent) <= 4 and elapsed < 1.6, (sent, elapsed)
    assert sent == sorted(sent, reverse=True) and sent[0] <= 0.7
    assert "unjudged" in searcher.silent["rerank"] and Reranker.found(hits) is not False
    deadline = time.time() + 5
    while module._hanging("m-4b") and time.time() < deadline:
        time.sleep(0.02)
