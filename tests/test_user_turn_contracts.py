#!/usr/bin/env python3
"""Contracts for the user turn: the words typed, the files attached, and who wrote each.

A chat turn carries the words the user typed and, beside them, the files the
user attached. The executor joins each file after the words, under a line it
writes itself, and saves the turn with segments: the words are typed, or
refined when a hook or the vision step rewrote them; each file is a document
segment of its own; the lines the executor writes belong to no one. The chat
route composes the turn once, as a claim, and the turn carries that claim to
every path that saves it -- the executor, the agentic pipelines, the
execution pipelines and the coding agent. A claim vouches for its own text
and nothing else: a prompt a pipeline or an agent composes from it is saved
as legacy. A retry re-creates the turn with the origin and segments it was
saved with.

  * UT1 -- the composer joins the question and each document after a line
    the executor writes; a document of no name is joined as before, word for
    word, and each bound points at its document's text.
  * UT2 -- a claim is typed when the words sent are the words typed and
    refined when they were rewritten; each document with text is a segment
    of its own, a turn of documents alone is a document, one of no words at
    all is legacy, and every claim lies inside the grammar.
  * UT3 -- a claim vouches for its own text and nothing else.
  * UT4 -- the executor saves named documents one segment each, and sends
    the model the turn it saves.
  * UT5 -- a question the vision step rewrote is saved refined, its
    documents still segments of their own.
  * UT6 -- the turn's claim reaches the executor through the run: the
    claimed text is saved with the claim's parts, a text composed from it as
    legacy.
  * UT7 -- the agentic executor saves the user turn by its run's claim, and
    typed as before when no claim rides the run.
  * UT8 -- the pipeline runner hands every step the turn's claim: the first
    step's prompt is the claimed text, a later step's composed prompt is
    legacy.
  * UT9 -- the coding agent saves its turn by the claim it was handed, and
    drops the claim when the turn ends.
  * UT10 -- the attachment bounds are read from config/chat.yaml, each
    refusal names the file and the rule, and an unreadable file refuses
    every attachment.
  * UT11 -- a request over the bounds is refused before anything runs or is
    written; files with no typed words make a turn; nothing at all is
    refused.
  * UT12 -- the route hands the executor the typed words and the files
    apart, routes on both, leaves the context check as it was, and the saved
    turn has one segment per file, refined when a hook rewrote the words.
  * UT13 -- the route hands the agentic executor and the pipeline runner the
    composed turn, its claim riding the turn.
  * UT14 -- the coding path reads its directives from the typed words only,
    hands the session the composed turn and its claim, and the agent's own
    model calls save their prompts as legacy.
  * UT15 -- a retry reads the stored turn's origin and segments before it
    removes the turn, and the turn it re-creates carries them; a rewrite
    demotes them.
  * UT16 -- the page sends the attached files as documents and the typed
    words as the message; nothing wraps a file into the message.
  * UT17 -- the page's composer joins a turn exactly as the executor does.
  * UT18 -- the request carries the documents and drops an empty list, and
    the chat store shows the turn as it will be saved.
  * UT19 -- the coding session reads its directives from the typed words,
    and every phase that reads them is handed the whole turn, files
    included.
  * UT20 -- a retry on the coding path sends the stored turn as it was,
    reads its directives from its typed words only, and keeps its claim.
  * UT21 -- a pre_inference hook is shown the files and may rewrite their
    text, which stays a document; words no hook touched stay typed.
  * UT22 -- a stored turn whose labels lie outside the grammar, or whose
    text is not the turn's, comes back legacy.
  * UT23 -- a refused name says which rule it breaks, and the length rule
    names the file that sets it.
  * UT24 -- a hook without the inference-content permission is never shown
    a file's text, as it is never shown the words.
  * UT25 -- a hook that writes a file in place and then fails changes
    nothing, and a rewrite over the bounds config/chat.yaml sets, or of no
    valid text, is set aside.
  * UT26 -- on the coding path too, the hook of each model call hides the
    words and the files from a plugin without the permission.

Local-only (the public distribution ships no tests). The node halves need
Node >= 22.6.
"""

import asyncio
import json
import re
import sqlite3
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _frontend import read, run_ts  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder
# from the junit file (the suite reads the frontend through its helper).
BUDGET_S = {
    "test_ut1_the_composer_joins_each_document_after_a_line_of_its_own_and_bounds_its_text": 1.0,
    "test_ut2_a_claim_is_typed_or_refined_with_one_segment_per_document_and_lies_inside_the_grammar": 1.0,
    "test_ut3_a_claim_vouches_for_its_own_text_and_nothing_else": 1.0,
    "test_ut4_the_executor_saves_named_documents_one_segment_each_and_sends_the_turn_it_saves": 1.0,
    "test_ut5_a_question_the_vision_step_rewrote_is_saved_refined_and_its_documents_keep_their_segments": 1.0,
    "test_ut6_the_executor_saves_the_claimed_text_with_the_claims_parts_and_a_text_composed_from_it_as_legacy":
        1.0,
    "test_ut7_the_agentic_executor_saves_the_user_turn_by_its_runs_claim": 1.0,
    "test_ut8_the_pipeline_runner_hands_each_step_the_claim_and_a_later_steps_composed_prompt_is_legacy": 1.0,
    "test_ut9_the_coding_agent_saves_its_turn_by_the_claim_it_was_handed_and_drops_it_after": 1.0,
    "test_ut10_the_attachment_bounds_come_from_the_yaml_and_each_refusal_names_the_file_and_the_rule": 1.0,
    "test_ut11_a_request_over_the_bounds_is_refused_before_anything_runs_and_files_alone_make_a_turn": 1.0,
    "test_ut12_the_route_hands_the_executor_words_and_files_apart_and_the_saved_turn_has_a_segment_per_file":
        2.0,
    "test_ut13_the_route_hands_the_agentic_executor_and_the_pipeline_runner_the_composed_turn_and_its_claim":
        2.0,
    "test_ut14_the_coding_path_reads_directives_from_the_typed_words_and_hands_the_session_the_claimed_turn":
        2.0,
    "test_ut15_a_retry_re_creates_the_turn_with_its_stored_origin_and_segments_and_a_rewrite_demotes_them": 3.0,
    "test_ut16_the_page_sends_the_files_as_documents_and_never_wraps_them_into_the_message": 1.0,
    "test_ut17_the_pages_composer_joins_a_turn_exactly_as_the_executor_does": 2.0,
    "test_ut18_the_request_carries_the_documents_and_the_store_shows_the_turn_as_it_will_be_saved": 2.0,
    "test_ut19_the_coding_session_reads_its_directives_from_the_typed_words_and_every_phase_gets_the_files": 2.0,
    "test_ut20_a_retry_on_the_coding_path_sends_the_stored_turn_as_it_was_and_reads_directives_from_its_words":
        3.0,
    "test_ut21_a_pre_inference_hook_sees_the_files_and_may_rewrite_their_text_which_stays_a_document": 2.0,
    "test_ut22_a_stored_turn_outside_the_grammar_or_of_another_text_comes_back_legacy": 2.0,
    "test_ut23_a_refused_name_says_which_rule_it_breaks": 1.0,
    "test_ut24_a_hook_without_the_inference_content_permission_is_never_shown_a_files_text": 2.0,
    "test_ut25_a_hook_that_writes_a_file_in_place_and_fails_changes_nothing_and_a_rewrite_out_of_bounds_is_set_aside":
        3.0,
    "test_ut26_on_the_coding_path_the_hook_of_each_model_call_hides_the_files_from_a_plugin_without_the_permission":
        2.0,
}

_QUESTION = "Where does Alice meet Bob on 2024-03-15?"
_DOCS = [
    ("venue.txt", "Contoso opens the venue to 40 guests."),
    ("notes.md", "Bob arrives at noon.\nAlice brings the keys."),
]
_HOOKED = "Think step by step. "
_HEAD = "\n\n---\nDocument provided:"
_SEEN = "[Image: a red door]\n\n"
_EXECUTOR = "opti_oignon.executor"
_PAGE = "frontend/src/routes/(app)/(use)/chat/[id]/+page.svelte"
_USER_TURN_TS = "frontend/src/lib/chat/userTurn.ts"
_REQUEST_FIELDS = "frontend/src/lib/chat/requestFields.ts"
_CHAT_STORE = "frontend/src/lib/stores/chat.ts"
_CHAT_YAML = REPO / "opti_oignon" / "config" / "chat.yaml"


def _documents(pairs=_DOCS):
    """The documents field of a chat request: one object per file."""
    return [{"filename": name, "content": text} for name, text in pairs]


def _parts(content, segments):
    """Each segment as the text it covers and its base."""
    return [(content[start:end], base) for start, end, base in segments]


def _segments(claim):
    return [list(segment) for segment in claim.segments]


def _saved_users(store, conversation_id="conv-1"):
    return [m for m in store.saved if m["role"] == "user" and m["conversation_id"] == conversation_id]


def _drive(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def _run(claim):
    """A run as the chat route hands it on: its stop, its results, the turn's claim."""
    return SimpleNamespace(stop=threading.Event(), results={}, steps=None, user_turn=claim)


def _routing():
    return SimpleNamespace(model="test-model:1b", task_type="general", temperature=0.2,
                           prompt_variant="standard", timeout=30, routing_reason="contract", images=None)


# ---------------------------------------------------------------------------
# The executor's window, alone or under the chat routes
# ---------------------------------------------------------------------------
class _Scripted:
    """The model behind the registry: records what it is sent, answers in two chunks."""

    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return iter([{"message": {"content": "At the "}}, {"message": {"content": "venue."}, "done": True}])


class _Store:
    """A conversation store that keeps what it is handed and reads it back as the route and the executor do."""

    def __init__(self, turns=(), model=None):
        self.saved = [dict(turn) for turn in turns]
        self.model = model
        self.created = []

    def add_message(self, conversation_id, role, content, model=None, metadata=None, *, origin="legacy",
                    segments=()):
        self.saved.append({"conversation_id": conversation_id, "role": role, "content": content, "model": model,
                           "origin": origin, "segments": [list(s) for s in segments]})

    def _of(self, conversation_id):
        return [m for m in self.saved if m["conversation_id"] == conversation_id]

    def get_context_messages(self, conversation_id, **kwargs):
        return [{"role": m["role"], "content": m["content"]} for m in self._of(conversation_id)]

    def get_mirror_messages(self, conversation_id):
        return [{"role": m["role"], "content": m["content"], "origin": m["origin"], "segments": m["segments"]}
                for m in self._of(conversation_id)]

    def get_messages(self, conversation_id):
        return [SimpleNamespace(role=m["role"], content=m["content"], model=m["model"])
                for m in self._of(conversation_id)]

    def delete_last_message(self, conversation_id, role=None):
        for index in range(len(self.saved) - 1, -1, -1):
            if self.saved[index]["conversation_id"] == conversation_id:
                if role is not None and self.saved[index]["role"] != role:
                    return False
                del self.saved[index]
                return True
        return False

    def get_conversation(self, conversation_id):
        return SimpleNamespace(id=conversation_id, model=self.model,
                               messages=self.get_context_messages(conversation_id), metadata={})

    def create_conversation(self, title=""):
        self.created.append(title)
        return SimpleNamespace(id="conv-new")

    def update_conversation_metadata(self, *args, **kwargs):
        return None


_GRAMMAR = []
_PERMITTED = ("careless-editor", "trusted-editor")


def _store_grammar():
    """The store's grammar of a turn's origin, as conversation.py holds it: probes.py's copy, one text with it by OT4."""
    if not _GRAMMAR:
        loaded, close = isolate(targets={"opti_oignon.memory.probes": source("memory", "probes.py")},
                                packages=("opti_oignon.memory",))
        try:
            _GRAMMAR.append(loaded["opti_oignon.memory.probes"]._origin_defect)
        finally:
            close()
    return _GRAMMAR[0]


def _seeds(store, scripted):
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(get_model=lambda *a, **k: "test-model:1b", get_temperature=lambda *a, **k: 0.2)
    router = types.ModuleType("opti_oignon.router")
    router.RoutingResult = type("RoutingResult", (), {})
    retrieval = types.ModuleType("opti_oignon.memory.retrieval")
    retrieval.build_memory_block = lambda *a, **k: ""
    retrieval.working_memory_block = lambda *a, **k: ""
    conversation = types.ModuleType("opti_oignon.conversation")
    conversation.conversation_manager = store
    conversation._origin_defect = _store_grammar()
    manifest = types.ModuleType("opti_oignon.plugin_manifest")
    manifest.VALID_HOOKS = frozenset({"pre_inference"})
    manifest.plugin_registry = SimpleNamespace(get=lambda name: SimpleNamespace(
        manifest=SimpleNamespace(permissions=["inference_content"])) if name in _PERMITTED else None)
    librarian = types.ModuleType("opti_oignon.memory.librarian")
    librarian.onion_enabled = lambda path=None: False
    librarian.memory_block = lambda conversation_id, question=None, **kwargs: ""
    librarian.maybe_curate = lambda conversation_id, messages, **kwargs: False
    seeded = {
        "opti_oignon.config": cfg,
        "opti_oignon.router": router,
        "opti_oignon.memory.retrieval": retrieval,
        "opti_oignon.conversation": conversation,
        "opti_oignon.memory.librarian": librarian,
        "opti_oignon.plugin_manifest": manifest,
    }
    seed_registry(seeded, scripted)
    return seeded


def _stub_ollama(scripted):
    stub = types.ModuleType("ollama")
    stub.chat = scripted.chat
    had, prev = "ollama" in sys.modules, sys.modules.get("ollama")
    sys.modules["ollama"] = stub

    def put_back():
        if had:
            sys.modules["ollama"] = prev
        else:
            sys.modules.pop("ollama", None)

    return put_back


_EXECUTOR_TARGETS = (
    ("opti_oignon.agent.untrusted_context", ("agent", "untrusted_context.py")),
    ("opti_oignon.context_dedup", ("context_dedup.py",)),
    (_EXECUTOR, ("executor.py",)),
)


def _open(store, extra_targets=(), extra_seeds=None, packages=()):
    scripted = _Scripted()
    seeded = _seeds(store, scripted)
    seeded.update(extra_seeds or {})
    targets = {name: source(*parts) for name, parts in _EXECUTOR_TARGETS + tuple(extra_targets)}
    put_back = _stub_ollama(scripted)
    try:
        loaded, close = isolate(targets=targets, seeded=seeded,
                                packages=("opti_oignon.agent", "opti_oignon.memory") + tuple(packages))
    except BaseException:
        put_back()
        raise

    def restore():
        close()
        put_back()

    return loaded, scripted, restore


def _executor(store=None):
    """executor.py in its window: a scripted model, a store that keeps what it is handed."""
    store = store if store is not None else _Store()
    loaded, scripted, restore = _open(store)
    return loaded[_EXECUTOR], scripted, store, restore


def _deps_stub():
    deps = types.ModuleType("opti_oignon.api.deps")
    for name in ("ANALYZER_AVAILABLE", "CONVERSATION_AVAILABLE", "EXECUTOR_AVAILABLE", "PRESET_AVAILABLE",
                 "ROUTER_AVAILABLE"):
        setattr(deps, name, False)
    for name in ("analyzer", "conversation_manager", "executor", "preset_manager", "router"):
        setattr(deps, name, None)
    return deps


def _auth_stub():
    auth = types.ModuleType("opti_oignon.api.routes_auth")

    async def authenticate_websocket(websocket):
        return {"username": "local"}

    auth.authenticate_websocket = authenticate_websocket
    return auth


def _world(store=None):
    """The chat routes over the real executor in one window: the route composes with the executor's composer."""
    store = store if store is not None else _Store()
    loaded, scripted, restore = _open(
        store,
        extra_targets=(("opti_oignon.plugin_hooks", ("plugin_hooks.py",)),
                       ("opti_oignon.api.schemas", ("api", "schemas.py")),
                       ("opti_oignon.api.routes_chat", ("api", "routes_chat.py"))),
        extra_seeds={"opti_oignon.api.deps": _deps_stub(), "opti_oignon.api.routes_auth": _auth_stub()},
        packages=("opti_oignon.api",),
    )
    rc, ex = loaded["opti_oignon.api.routes_chat"], loaded[_EXECUTOR]
    rc.EXECUTOR_AVAILABLE = True
    rc.executor = ex.Executor()
    rc._emergency_stop = None
    routed = []
    rc._resolve_model_and_route = lambda message, request: (routed.append(message), (_routing(), None))[1]
    world = SimpleNamespace(rc=rc, ex=ex, schemas=loaded["opti_oignon.api.schemas"], scripted=scripted,
                            store=store, routed=routed, hooks=loaded["opti_oignon.plugin_hooks"])
    return world, restore


def _routes_alone():
    """The chat routes alone, beside stand-in seams: the attachment bounds need nothing else."""
    loaded, close = isolate(
        targets={"opti_oignon.api.schemas": source("api", "schemas.py"),
                 "opti_oignon.api.routes_chat": source("api", "routes_chat.py")},
        seeded={"opti_oignon.api.deps": _deps_stub(), "opti_oignon.api.routes_auth": _auth_stub()},
        packages=("opti_oignon.api",),
    )
    return loaded["opti_oignon.api.routes_chat"], close


class _Socket:
    """A chat socket stand-in: the payload it hands over, the frames it was sent."""

    def __init__(self, payload=None):
        self.payload = payload
        self.sent = []

    async def accept(self):
        return None

    async def receive_json(self):
        return self.payload

    async def send_json(self, data):
        self.sent.append(data)

    async def close(self, code=None):
        return None


def _errors(ws):
    return [frame.get("content", "") for frame in ws.sent if frame.get("type") == "error"]


def _stream(world, message, **fields):
    """One turn through the route's stream, as the socket endpoint opens it."""
    request = world.schemas.ChatRequest(conversation_id="conv-1", message=message, **fields)
    ws = _Socket()
    asyncio.run(world.rc._stream_response(ws, "conv-1", message, request))
    return ws


class _Hooks:
    """A pre_inference hook that puts words in front of the message, as the chain-of-thought plugin does."""

    def has_hooks(self, kind):
        return kind == "pre_inference"

    def execute(self, kind, **kwargs):
        message = (kwargs.get("data") or {}).get("message", "")
        return SimpleNamespace(final_data={"message": _HOOKED + message}, results=[])


# ---------------------------------------------------------------------------
# UT1 -- the composer
# ---------------------------------------------------------------------------
def test_ut1_the_composer_joins_each_document_after_a_line_of_its_own_and_bounds_its_text():
    mod, scripted, store, restore = _executor()
    try:
        content, bounds = mod.compose_user_turn(_QUESTION, [(None, _DOCS[0][1])])
        assert content == _QUESTION + _HEAD + "\n" + _DOCS[0][1], "a document of no name is joined as before"
        assert [content[start:end] for start, end in bounds] == [_DOCS[0][1]]
        content, bounds = mod.compose_user_turn(_QUESTION, _DOCS)
        assert content == (_QUESTION + _HEAD + " venue.txt\n" + _DOCS[0][1]
                           + _HEAD + " notes.md\n" + _DOCS[1][1])
        assert [content[start:end] for start, end in bounds] == [text for _name, text in _DOCS]
        content, bounds = mod.compose_user_turn(_QUESTION, [("empty.txt", ""), _DOCS[1]])
        assert [content[start:end] for start, end in bounds] == [_DOCS[1][1]], "an empty document has no bound"
        assert _HEAD + " empty.txt\n" in content, "its line stays: the model is told the file was empty"
        assert mod.compose_user_turn(_QUESTION, []) == (_QUESTION, [])
    finally:
        restore()


# ---------------------------------------------------------------------------
# UT2 -- the claim and the grammar
# ---------------------------------------------------------------------------
def _grammar(tmp_path):
    """conversation.py's own judge of an origin, loaded alone over plain sqlite."""
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda path, **kw: sqlite3.connect(path, check_same_thread=False)
    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = Path(tmp_path)
    loaded, close = isolate(targets={"opti_oignon.conversation": source("conversation.py")},
                            seeded={"opti_oignon.db_utils": db, "opti_oignon.config": cfg})
    return loaded["opti_oignon.conversation"]._origin_defect, close


def test_ut2_a_claim_is_typed_or_refined_with_one_segment_per_document_and_lies_inside_the_grammar(tmp_path):
    mod, scripted, store, restore = _executor()
    try:
        typed = mod.user_turn(_QUESTION, _QUESTION, _DOCS)
        hooked = mod.user_turn(_QUESTION, _HOOKED + _QUESTION, _DOCS)
        files_only = mod.user_turn("", "", _DOCS)
        plain = mod.user_turn(_QUESTION, _QUESTION, [])
        nothing = mod.user_turn("", "", [("empty.txt", "")])
    finally:
        restore()
    assert typed.content.startswith(_QUESTION) and typed.content.endswith(_DOCS[1][1]), "control: the turn joins all"
    assert typed.origin == "typed"
    assert _parts(typed.content, typed.segments) == [
        (_QUESTION, "typed"), (_DOCS[0][1], "document"), (_DOCS[1][1], "document")]
    gap = typed.content[len(_QUESTION):typed.segments[1][0]]
    assert "Document provided: venue.txt" in gap, "the line naming the file belongs to no one"
    assert hooked.origin == "refined"
    assert _parts(hooked.content, hooked.segments)[0] == (_HOOKED + _QUESTION, "refined")
    assert files_only.origin == "document"
    assert [base for _text, base in _parts(files_only.content, files_only.segments)] == ["document", "document"]
    assert (plain.content, plain.origin, _segments(plain)) == (_QUESTION, "typed", [])
    assert (nothing.origin, _segments(nothing)) == ("legacy", []), "a turn of no words at all vouches for none"
    defect, close = _grammar(tmp_path)
    try:
        for claim in (typed, hooked, files_only, plain, nothing):
            assert defect("user", claim.origin, _segments(claim), len(claim.content)) is None, claim
    finally:
        close()


# ---------------------------------------------------------------------------
# UT3 -- a claim vouches for its own text
# ---------------------------------------------------------------------------
def test_ut3_a_claim_vouches_for_its_own_text_and_nothing_else():
    mod, scripted, store, restore = _executor()
    try:
        claim = mod.user_turn(_QUESTION, _QUESTION, _DOCS)
    finally:
        restore()
    assert claim.parts_for(claim.content) == ("typed", _segments(claim))
    step = ("Based on the following previous analysis:\n\n---\nAt the venue.\n---\n\n"
            "Original question: " + claim.content)
    assert claim.parts_for(step) == ("legacy", []), "a prompt composed from the turn is no one's"
    assert claim.parts_for(claim.content + "\n") == ("legacy", [])
    assert claim.parts_for(_QUESTION) == ("legacy", []), "its words without its files are not the turn"


# ---------------------------------------------------------------------------
# UT4 -- named documents through the executor
# ---------------------------------------------------------------------------
def test_ut4_the_executor_saves_named_documents_one_segment_each_and_sends_the_turn_it_saves():
    mod, scripted, store, restore = _executor()
    try:
        expected = mod.compose_user_turn(_QUESTION, _DOCS)[0]
        _drive(mod.Executor().execute(_QUESTION, _routing(), refine=False, conversation_id="conv-1",
                                      documents=_DOCS))
        users = _saved_users(store)
        assert len(users) == 1, f"control: the turn was saved: {store.saved}"
        user = users[0]
        assert user["content"] == expected
        assert user["origin"] == "typed"
        assert _parts(user["content"], user["segments"]) == [
            (_QUESTION, "typed"), (_DOCS[0][1], "document"), (_DOCS[1][1], "document")]
        assert scripted.calls, "control: the model was called"
        assert expected in scripted.calls[-1]["messages"][-1]["content"], "the model is sent the turn saved"
    finally:
        restore()


# ---------------------------------------------------------------------------
# UT5 -- the vision step's rewrite is refined
# ---------------------------------------------------------------------------
class _Vision:
    """A vision step that writes its description of the image in front of the question."""

    def detect_needs_delegation(self, message, images, current_model):
        return False

    def process(self, message, images, current_model, on_status=None):
        return _SEEN + message, None, {"delegated": True}


def test_ut5_a_question_the_vision_step_rewrote_is_saved_refined_and_its_documents_keep_their_segments():
    for documents in (_DOCS, []):
        mod, scripted, store, restore = _executor()
        try:
            mod.VISION_PIPELINE_AVAILABLE = True
            mod._vision_pipeline = _Vision()
            _drive(mod.Executor().execute(_QUESTION, _routing(), refine=False, conversation_id="conv-1",
                                          images=["aW1n"], documents=documents))
            users = _saved_users(store)
            assert len(users) == 1, f"control: the turn was saved: {store.saved}"
            user = users[0]
            assert user["content"].startswith(_SEEN + _QUESTION), "control: the vision step rewrote the question"
            assert user["origin"] == "refined", f"the model's description is not typed ({len(documents)} files)"
            if documents:
                assert _parts(user["content"], user["segments"]) == [
                    (_SEEN + _QUESTION, "refined"), (_DOCS[0][1], "document"), (_DOCS[1][1], "document")]
            else:
                assert user["segments"] == []
        finally:
            restore()


# ---------------------------------------------------------------------------
# UT6 -- the claim reaches the executor through the run
# ---------------------------------------------------------------------------
def test_ut6_the_executor_saves_the_claimed_text_with_the_claims_parts_and_a_text_composed_from_it_as_legacy():
    mod, scripted, store, restore = _executor()
    try:
        hooked = mod.user_turn(_QUESTION, _HOOKED + _QUESTION, [])
        _drive(mod.Executor().execute(_HOOKED + _QUESTION, _routing(), refine=False, conversation_id="conv-1",
                                      run=_run(hooked)))
        composed = mod.user_turn(_QUESTION, _QUESTION, _DOCS)
        _drive(mod.Executor().execute(composed.content, _routing(), refine=False, conversation_id="conv-2",
                                      run=_run(composed)))
        step = "Now continue with: " + composed.content
        _drive(mod.Executor().execute(step, _routing(), refine=False, conversation_id="conv-3",
                                      run=_run(composed)))
    finally:
        restore()
    saved = [_saved_users(store, cid) for cid in ("conv-1", "conv-2", "conv-3")]
    assert all(len(users) == 1 for users in saved), f"control: each turn was saved: {store.saved}"
    first, second, third = (users[0] for users in saved)
    assert (first["origin"], first["segments"]) == ("refined", []), "a hook's rewrite the executor never saw"
    assert (second["content"], second["origin"], second["segments"]) == (
        composed.content, "typed", _segments(composed)), "a composed turn keeps the parts it was claimed with"
    assert (third["content"], third["origin"], third["segments"]) == (step, "legacy", [])


# ---------------------------------------------------------------------------
# UT7 -- the agentic executor saves by its run's claim
# ---------------------------------------------------------------------------
class _AgentStore:
    """The store's keyword surface as the agentic executor calls it."""

    def __init__(self):
        self.saved = []

    def add_message(self, conv_id=None, role=None, content=None, model=None, metadata=None, *, origin="legacy",
                    segments=()):
        self.saved.append({"conv_id": conv_id, "role": role, "content": content, "origin": origin,
                           "segments": [list(s) for s in segments]})


class _Cascade:
    enabled = True

    def cascade(self, query, task_type=None):
        return SimpleNamespace(final_response="At the venue.", model="fake-model", tier_index=0, tier_name="fast",
                               score=0.9, total_latency_ms=1.0, draft_accepted=True, iterations=1,
                               convergence_score=1.0)


def test_ut7_the_agentic_executor_saves_the_user_turn_by_its_runs_claim():
    mod, scripted, store, restore = _executor()
    try:
        claim = mod.user_turn(_QUESTION, _QUESTION, _DOCS)
    finally:
        restore()
    step = "Original question: " + claim.content
    agent_store = _AgentStore()
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = agent_store
    loaded, close = isolate(targets={"opti_oignon.agentic_executor": source("agentic_executor.py")},
                            seeded={"opti_oignon.conversation": conv})
    try:
        ae = loaded["opti_oignon.agentic_executor"]
        ae.CASCADING_INFERENCE_AVAILABLE = True
        agent = ae.AgenticExecutor(executor=SimpleNamespace(), cascading_inference=_Cascade(), default_model="m")
        for message, run in ((claim.content, _run(claim)), (step, _run(claim)), (_QUESTION, None)):
            list(agent.execute(message=message, routing=SimpleNamespace(model="m", task_type=None),
                               conversation_id="conv-a", cascading=True, **({"run": run} if run else {})))
    finally:
        close()
    users = [(m["content"], m["origin"], m["segments"]) for m in agent_store.saved if m["role"] == "user"]
    assert users == [
        (claim.content, "typed", _segments(claim)),
        (step, "legacy", []),
        (_QUESTION, "typed", []),
    ], users


# ---------------------------------------------------------------------------
# UT8 -- the pipeline runner's steps
# ---------------------------------------------------------------------------
def test_ut8_the_pipeline_runner_hands_each_step_the_claim_and_a_later_steps_composed_prompt_is_legacy():
    mod, scripted, store, restore = _executor()
    try:
        claim = mod.user_turn(_QUESTION, _QUESTION, _DOCS)
    finally:
        restore()
    loaded, close = isolate(targets={"opti_oignon.pipelines": source("pipelines.py")}, seeded={},
                            packages=("opti_oignon",))
    try:
        pl = loaded["opti_oignon.pipelines"]
        pl._resolve_emergency_stop = lambda: None
        pl._resolve_resource_governor = lambda: None

        class Recording:
            def __init__(self):
                self.steps = []

            def execute(self, **kwargs):
                self.steps.append(kwargs["run"].user_turn.parts_for(kwargs["message"]))
                yield "At the venue."

        router = SimpleNamespace(enabled=False, override_routing=lambda routing, step_type: routing)
        pipe = pl.ExecutionPipeline(id="p", name="P", steps=[
            pl.ExecutionStep("direct", label="one"), pl.ExecutionStep("direct", label="two"),
        ])
        agent = Recording()
        list(pl.PipelineRunner(agentic_executor=agent, smart_router=router).execute(
            pipe, claim.content, SimpleNamespace(model="m"), run=_run(claim),
        ))
    finally:
        close()
    assert agent.steps == [("typed", _segments(claim)), ("legacy", [])], agent.steps


# ---------------------------------------------------------------------------
# UT9 -- the coding agent saves by the claim it was handed
# ---------------------------------------------------------------------------
class _CodingStore:
    def __init__(self):
        self.saved = []

    def add_message(self, conv_id, role, content, model=None, metadata=None, *, origin="legacy", segments=()):
        self.saved.append({"role": role, "content": content, "origin": origin,
                           "segments": [list(s) for s in segments]})


def test_ut9_the_coding_agent_saves_its_turn_by_the_claim_it_was_handed_and_drops_it_after():
    files = [("test_app.py", "assert add(1, 2) == 4")]
    mod, scripted, store, restore = _executor()
    try:
        claim = mod.user_turn("Fix the failing test.", "Fix the failing test.", files)
    finally:
        restore()
    plan = "Plan the change:\n" + claim.content
    coding_store = _CodingStore()
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = coding_store
    loaded, close = isolate(targets={"opti_oignon.chat_coding_agent": source("chat_coding_agent.py")},
                            seeded={"opti_oignon.conversation": conv})
    try:
        cc = loaded["opti_oignon.chat_coding_agent"]
        assert cc.CONVERSATION_AVAILABLE is True, "control: the agent reaches the store"
        session = object.__new__(cc.ChatCodingSession)
        session._conversation_id = "conv-code"
        session._turn_lock = threading.Lock()
        held = []

        def locked(message, model, directives, images, web_search, think):
            held.append(getattr(session, "_turn_user_turn", None))
            session._save_turn_to_conversation(message, "Fixed: the expected sum.", model)
            session._save_turn_to_conversation(plan, "A plan.", model)
            return {"turn": 1}
            yield  # a generator, as the body it stands in for

        session._execute_task_locked = locked
        list(session.execute_task(claim.content, "coder:7b", user_turn=claim))
        after = getattr(session, "_turn_user_turn", "absent")
    finally:
        close()
    assert len(held) == 1 and held[0] is claim, "the turn's claim is held while the turn runs"
    assert after is None, "and dropped when the turn ends"
    users = [(m["content"], m["origin"], m["segments"]) for m in coding_store.saved if m["role"] == "user"]
    assert users == [(claim.content, "typed", _segments(claim)), (plan, "legacy", [])], users


# ---------------------------------------------------------------------------
# UT10 -- the attachment bounds and their refusals
# ---------------------------------------------------------------------------
def test_ut10_the_attachment_bounds_come_from_the_yaml_and_each_refusal_names_the_file_and_the_rule(tmp_path):
    shipped = yaml.safe_load(_CHAT_YAML.read_text(encoding="utf-8"))["attachments"]
    rc, close = _routes_alone()
    try:
        keys = ("max_documents", "max_document_bytes", "max_filename_chars")
        assert rc._attachment_bounds() == {key: shipped[key] for key in keys}, "the shipped file sets the bounds"
        small = tmp_path / "chat.yaml"
        small.write_text("attachments:\n  max_documents: 2\n  max_document_bytes: 8\n  max_filename_chars: 6\n",
                         encoding="utf-8")
        rc._CHAT_CONFIG = small
        e = chr(0xE9)
        assert rc._attachment_refusal([("a.txt", "x" * 8), ("b.txt", e * 4)]) is None, "at the bounds, in bytes"
        too_many = rc._attachment_refusal([("a.txt", "x")] * 3)
        assert too_many and "3 documents" in too_many and "limit of 2" in too_many and "chat.yaml" in too_many
        too_big = rc._attachment_refusal([("ok.txt", "x"), ("big.md", e * 5)])
        assert too_big and "'big.md'" in too_big and "10 bytes" in too_big and "limit of 8" in too_big, too_big
        for bad in ("", "x" * 7, "a" + chr(10) + "b", "a" + chr(0), "a" + chr(0x202E) + "txt"):
            refusal = rc._attachment_refusal([("ok.txt", "x"), (bad, "x")])
            assert refusal and "document 2" in refusal, ascii(bad)
        broken = tmp_path / "broken.yaml"
        broken.write_text("attachments: [1, 2\n", encoding="utf-8")
        rc._CHAT_CONFIG = broken
        refused = rc._attachment_refusal([("a.txt", "x")])
        assert refused and "chat.yaml" in refused, "an unreadable file refuses every attachment"
        assert rc._attachment_refusal([]) is None, "a turn without attachments needs no bounds"
        rc._CHAT_CONFIG = tmp_path / "absent.yaml"
        assert rc._attachment_refusal([("a.txt", "x")]), "an absent file refuses them too"
    finally:
        close()


# ---------------------------------------------------------------------------
# UT11 -- refused before anything runs
# ---------------------------------------------------------------------------
def test_ut11_a_request_over_the_bounds_is_refused_before_anything_runs_and_files_alone_make_a_turn():
    rc, close = _routes_alone()
    try:
        store = _Store()
        rc.CONVERSATION_AVAILABLE = True
        rc.conversation_manager = store
        rc._emergency_stop = None
        streamed = []

        async def spy(websocket, conversation_id, message, request):
            streamed.append((conversation_id, message, [(d.filename, d.content) for d in request.documents or ()]))

        rc._stream_response = spy

        def send(payload):
            ws = _Socket(payload)
            asyncio.run(rc.chat_stream(ws))
            return ws

        over = send({"message": _QUESTION, "documents": _documents([("a.txt", "x")] * 11)})
        assert any("Invalid request" in e and "11 documents" in e for e in _errors(over)), over.sent
        assert streamed == [] and store.created == [], "a refused request runs nothing and creates nothing"
        files_only = send({"conversation_id": "conv-1", "message": "", "documents": _documents()})
        assert streamed == [("conv-1", "", _DOCS)], f"files with no typed words make a turn: {files_only.sent}"
        for payload in ({"message": "  "}, {"message": "", "documents": _documents([("empty.txt", "")])}):
            ws = send(payload)
            assert "Empty message" in _errors(ws), (payload, ws.sent)
        assert len(streamed) == 1 and store.created == []
    finally:
        close()


# ---------------------------------------------------------------------------
# UT12 -- the route and the executor
# ---------------------------------------------------------------------------
def test_ut12_the_route_hands_the_executor_words_and_files_apart_and_the_saved_turn_has_a_segment_per_file():
    world, restore = _world()
    try:
        calls = []
        real = world.rc.executor.execute

        def recording(**kwargs):
            calls.append(kwargs)
            return real(**kwargs)

        world.rc.executor.execute = recording
        ws = _stream(world, _QUESTION, documents=_documents())
        assert calls, f"control: the executor was called: {ws.sent}"
        call = calls[0]
        assert call["question"] == _QUESTION, "the typed words travel alone"
        assert [tuple(d) for d in call["documents"]] == _DOCS, "the files travel beside them"
        assert call["validate_context"] is False, "the context check stays as it was for an attached file"
        assert world.routed and all(text in world.routed[0] for _name, text in _DOCS), "routing reads the files"
        expected = world.ex.compose_user_turn(_QUESTION, _DOCS)[0]
        users = _saved_users(world.store)
        assert len(users) == 1, f"control: the turn was saved: {world.store.saved}"
        assert (users[0]["content"], users[0]["origin"]) == (expected, "typed")
        assert _parts(users[0]["content"], users[0]["segments"]) == [
            (_QUESTION, "typed"), (_DOCS[0][1], "document"), (_DOCS[1][1], "document")]
        assert expected in world.scripted.calls[-1]["messages"][-1]["content"], "the model reads the files"
    finally:
        restore()
    world, restore = _world()
    try:
        world.rc.PLUGIN_HOOKS_AVAILABLE = True
        world.rc._hook_manager = _Hooks()
        _stream(world, _QUESTION, documents=_documents())
        users = _saved_users(world.store)
        assert len(users) == 1, f"control: the turn was saved: {world.store.saved}"
        assert users[0]["origin"] == "refined", "words a hook rewrote are not typed"
        assert _parts(users[0]["content"], users[0]["segments"]) == [
            (_HOOKED + _QUESTION, "refined"), (_DOCS[0][1], "document"), (_DOCS[1][1], "document")]
    finally:
        restore()


# ---------------------------------------------------------------------------
# UT13 -- the route, the agentic executor and the pipeline runner
# ---------------------------------------------------------------------------
class _Agent:
    """An executor that records each message it is handed and what the run's claim says of it."""

    available = True

    def __init__(self):
        self.calls = []

    def execute(self, **kwargs):
        claim = getattr(kwargs.get("run"), "user_turn", None)
        message = kwargs["message"]
        self.calls.append((message, claim.parts_for(message) if claim is not None else None))
        yield "At the venue."


def test_ut13_the_route_hands_the_agentic_executor_and_the_pipeline_runner_the_composed_turn_and_its_claim():
    world, restore = _world()
    try:
        expected = world.ex.user_turn(_QUESTION, _QUESTION, _DOCS)
        agent = _Agent()
        world.rc.AGENTIC_EXECUTOR_AVAILABLE = True
        world.rc._agentic_executor = agent
        world.rc.PIPELINE_DIRECT = "direct"  # what the guarded import brings beside the executor
        _stream(world, _QUESTION, documents=_documents())
        runner = _Agent()
        world.rc.EXEC_PIPELINES_AVAILABLE = True
        world.rc.get_pipeline_store = lambda: SimpleNamespace(get=lambda pipeline_id: SimpleNamespace(id=pipeline_id))
        world.rc.get_pipeline_runner = lambda: runner
        _stream(world, _QUESTION, documents=_documents(), exec_pipeline="review")
    finally:
        restore()
    claimed = (expected.content, ("typed", _segments(expected)))
    assert agent.calls == [claimed], agent.calls
    assert runner.calls == [claimed], runner.calls


# ---------------------------------------------------------------------------
# UT14 -- the coding path
# ---------------------------------------------------------------------------
class _CodingSession:
    """A coding session that runs one turn: it asks the model for a plan through the turn's callback."""

    session_id = "s-1"
    turn_count = 0

    def __init__(self):
        self.sandbox_state = SimpleNamespace(files=[])
        self.calls = []

    def execute_task(self, **kwargs):
        self.calls.append(kwargs)
        prompt = [{"role": "user", "content": "Plan the change:\n" + kwargs["message"]}]
        kwargs["llm_call"](prompt, "test-model:1b", None)
        yield SimpleNamespace(event_type="coding_done", data={"turn": 1}, content="Fixed.")

    def get_sandbox_files_for_ui(self):
        return []


class _CodingManager:
    enabled = True
    available = True

    def __init__(self, session):
        self.session = session

    def get_or_create_session(self, conversation_id, llm_call):
        return self.session


def test_ut14_the_coding_path_reads_directives_from_the_typed_words_and_hands_the_session_the_claimed_turn():
    files = [("test_app.py", "assert add(1, 2) == 4")]
    world, restore = _world()
    try:
        session = _CodingSession()
        parsed = []

        def parse(message):
            parsed.append(message)
            return SimpleNamespace(cleaned_message=message.replace(" --plan-only", ""))

        world.rc.CHAT_CODING_AVAILABLE = True
        world.rc._chat_coding_manager = _CodingManager(session)
        world.rc._parse_coding_directives = parse
        world.rc.LLMCallResult = lambda: SimpleNamespace(thinking="", text="", tool_calls=[], error=None,
                                                         vision_meta=None, plugin_annotations=[])
        _stream(world, "Fix the failing test. --plan-only", documents=_documents(files), chat_coding=True)
        expected = world.ex.user_turn("Fix the failing test.", "Fix the failing test.", files)
    finally:
        restore()
    assert parsed == ["Fix the failing test. --plan-only"], "directives are read from the typed words only"
    assert session.calls, "control: the session ran the turn"
    call = session.calls[0]
    assert call["message"] == expected.content
    assert call["user_turn"].parts_for(expected.content) == ("typed", _segments(expected))
    prompts = _saved_users(world.store)
    assert prompts, f"control: the agent's model call saved its prompt: {world.store.saved}"
    assert all((m["origin"], m["segments"]) == ("legacy", []) for m in prompts), prompts


# ---------------------------------------------------------------------------
# UT15 -- a retry keeps the stored parts
# ---------------------------------------------------------------------------
def _stored(content, origin, segments):
    return [
        {"conversation_id": "conv-1", "role": "user", "content": content, "model": None, "origin": origin,
         "segments": [list(s) for s in segments]},
        {"conversation_id": "conv-1", "role": "assistant", "content": "At the venue.", "model": "test-model:1b",
         "origin": "assistant", "segments": []},
    ]


def _retry(store, hooks=False):
    world, restore = _world(store)
    try:
        world.rc.CONVERSATION_AVAILABLE = True
        world.rc.conversation_manager = store
        if hooks:
            world.rc.PLUGIN_HOOKS_AVAILABLE = True
            world.rc._hook_manager = _Hooks()
        ws = _Socket({"conversation_id": "conv-1"})
        asyncio.run(world.rc.chat_retry(ws))
        return ws
    finally:
        restore()


def test_ut15_a_retry_re_creates_the_turn_with_its_stored_origin_and_segments_and_a_rewrite_demotes_them():
    mod, scripted, store, restore = _executor()
    try:
        claim = mod.user_turn(_QUESTION, _QUESTION, _DOCS)
        plain = mod.user_turn(_QUESTION, _QUESTION, [])
        demoted = [
            plain.rewritten(_HOOKED + _QUESTION),
            claim.rewritten(claim.content),
            claim.rewritten(_HOOKED + claim.content),
            mod.UserTurn("Old words.", "legacy", ()).rewritten("New words."),
            mod.UserTurn(_QUESTION, "refined", ()).rewritten(_HOOKED + _QUESTION),
        ]
    finally:
        restore()
    assert [(d.origin, _segments(d)) for d in demoted] == [
        ("refined", []), ("typed", _segments(claim)), ("legacy", []), ("legacy", []), ("refined", [])]
    assert demoted[0].content == _HOOKED + _QUESTION
    cases = (
        (_stored(claim.content, "typed", claim.segments), False, (claim.content, "typed", _segments(claim))),
        (_stored("Where was it, again?", "legacy", []), False, ("Where was it, again?", "legacy", [])),
        (_stored(_QUESTION, "typed", []), True, (_HOOKED + _QUESTION, "refined", [])),
    )
    for turns, hooks, expected in cases:
        store = _Store(turns)
        ws = _retry(store, hooks)
        users = [(m["content"], m["origin"], m["segments"]) for m in _saved_users(store)]
        assert users == [expected], f"{expected[1]}: {users} {ws.sent}"


# ---------------------------------------------------------------------------
# UT16 -- the page sends the files beside the words
# ---------------------------------------------------------------------------
def test_ut16_the_page_sends_the_files_as_documents_and_never_wraps_them_into_the_message():
    page = read(_PAGE)
    script = "\n".join(re.findall(r"<script\b[^>]*>(.*?)</script>", page, re.S))
    assert "function handleSend" in script, "control: the page's send handler is read"
    assert re.search(r"options\.documents\s*=\s*attachedFiles\.map\(\s*\(\s*\{\s*filename\s*,\s*content\s*\}\s*\)"
                     r"\s*=>\s*\(\s*\{\s*filename\s*,\s*content\s*\}\s*\)\s*\)", script), (
        "the files travel as documents, their name and their text and nothing else")
    assert re.search(r"sendMessage\(\s*convId\s*,\s*event\.detail\.text\s*,\s*options\s*\)", script), (
        "the message is the words typed")
    assert "[File:" not in script, "no file is wrapped into the message"
    assert not re.search(r"attachedFiles[^;]*\.content\b", script), "no file's text is read into the message"


# ---------------------------------------------------------------------------
# UT17 and UT18 -- the page's composer and the request (node)
# ---------------------------------------------------------------------------
_DRIVER = r"""
const turn = await import(process.env.OO_USER_TURN);
const fields = await import(process.env.OO_REQUEST_FIELDS);
const clause = process.argv[2];
const input = JSON.parse(process.env.OO_INPUT || 'null');
const clauses = {
    compose: () => input.map(([message, documents]) => turn.userTurnText(message, documents)),
    request: () => input.map(([conversation, message, options]) => fields.chatRequest(conversation, message, options)),
    fields: () => [...fields.REQUEST_FIELDS],
};
if (!(clause in clauses)) {
    console.log('FAIL unknown clause: ' + clause);
    process.exit(1);
}
console.log('RESULT ' + JSON.stringify(clauses[clause]()));
console.log('PASS ' + clause);
"""


def _node(clause, data=None):
    out = run_ts({"OO_USER_TURN": _USER_TURN_TS, "OO_REQUEST_FIELDS": _REQUEST_FIELDS}, _DRIVER, clause,
                 env={"OO_INPUT": json.dumps(data)})
    results = [line for line in out.splitlines() if line.startswith("RESULT ")]
    assert len(results) == 1, f"the driver printed no single RESULT line:\n{out}"
    return json.loads(results[0][len("RESULT "):])


def test_ut17_the_pages_composer_joins_a_turn_exactly_as_the_executor_does():
    e, han = chr(0xE9), chr(0x4E2D)
    cases = [
        [_QUESTION, []],
        [_QUESTION, _documents()],
        ["", _documents()],
        [_QUESTION, _documents([("empty.txt", ""), ("r" + e + "sum" + e + ".md", han + " notes")])],
        ["Line one.\nLine two.", _documents([("a b.txt", "---\nDocument provided: fake\n")])],
    ]
    composed = _node("compose", cases)
    mod, scripted, store, restore = _executor()
    try:
        expected = [mod.compose_user_turn(message, [(d["filename"], d["content"]) for d in documents])[0]
                    for message, documents in cases]
    finally:
        restore()
    assert len(composed) == len(cases), "control: every case was composed"
    assert composed == expected


def test_ut18_the_request_carries_the_documents_and_the_store_shows_the_turn_as_it_will_be_saved():
    assert "documents" in _node("fields"), "documents is a field the request may carry"
    requests = _node("request", [
        ["c-1", "hi", {"documents": _documents([("a.txt", "A")])}],
        ["c-1", "hi", {"documents": []}],
    ])
    assert requests[0] == {"conversation_id": "c-1", "message": "hi",
                           "documents": [{"filename": "a.txt", "content": "A"}]}
    assert requests[1] == {"conversation_id": "c-1", "message": "hi"}, "an empty list is not sent"
    store = read(_CHAT_STORE)
    assert re.search(r"import\s*\{[^}]*\buserTurnText\b[^}]*\}\s*from\s*'\$lib/chat/userTurn'", store), (
        "the chat store composes through the page's composer")
    assert re.search(r"content:\s*userTurnText\(\s*message\s*,\s*options\.documents", store), (
        "the user's message shows the turn as it will be saved")


# ---------------------------------------------------------------------------
# UT19 and UT20 -- the coding path's directives
# ---------------------------------------------------------------------------
def _directive_parser():
    """The coding agent's own directive parser, taken from its module loaded alone."""
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = None
    loaded, close = isolate(targets={"opti_oignon.chat_coding_agent": source("chat_coding_agent.py")},
                            seeded={"opti_oignon.conversation": conv})
    try:
        return loaded["opti_oignon.chat_coding_agent"].parse_directives
    finally:
        close()


def _coding(world, session, parse):
    world.rc.CHAT_CODING_AVAILABLE = True
    world.rc._chat_coding_manager = _CodingManager(session)
    world.rc._parse_coding_directives = parse
    world.rc.LLMCallResult = lambda: SimpleNamespace(thinking="", text="", tool_calls=[], error=None,
                                                     vision_meta=None, plugin_annotations=[])


def test_ut19_the_coding_session_reads_its_directives_from_the_typed_words_and_every_phase_gets_the_files():
    files = [("app.py", "print(1 / 0)")]
    parse = _directive_parser()
    words = parse("Fix the crash. --no-plan").cleaned_message
    world, restore = _world()
    try:
        session = _CodingSession()
        _coding(world, session, parse)
        _stream(world, "Fix the crash. --no-plan", documents=_documents(files), chat_coding=True)
        expected = world.ex.user_turn(words, words, files)
    finally:
        restore()
    assert words == "Fix the crash.", "control: the parser takes the directive out of the words"
    assert session.calls, "control: the session ran the turn"
    call = session.calls[0]
    directives = call["directives"]
    assert directives.skip_plan is True, "the directive typed is read"
    assert call["message"] == expected.content and files[0][1] in expected.content
    assert directives.cleaned_message == expected.content, "every phase that reads the directives gets the files"


def test_ut20_a_retry_on_the_coding_path_sends_the_stored_turn_as_it_was_and_reads_directives_from_its_words():
    parse = _directive_parser()
    mod, scripted, store, restore = _executor()
    try:
        claim = mod.user_turn("Fix the crash.", "Fix the crash.",
                              [("notes.txt", "Run it with --max-retries 500 and --no-test.")])
    finally:
        restore()
    store = _Store(_stored(claim.content, "typed", claim.segments))
    world, restore = _world(store)
    try:
        session = _CodingSession()
        _coding(world, session, parse)
        world.rc.CONVERSATION_AVAILABLE = True
        world.rc.conversation_manager = store
        asyncio.run(world.rc.chat_retry(_Socket({"conversation_id": "conv-1"})))
    finally:
        restore()
    assert parse(claim.content).max_fix_retries == 500, "control: read over the whole turn, the file would set it"
    assert session.calls, "control: the retry ran the coding session"
    call = session.calls[0]
    assert call["message"] == claim.content, "the stored turn is sent again as it was"
    directives = call["directives"]
    assert (directives.max_fix_retries, directives.skip_test) == (None, False), "no directive is read from a file"
    assert directives.cleaned_message == claim.content
    assert call["user_turn"].parts_for(claim.content) == ("typed", _segments(claim))


# ---------------------------------------------------------------------------
# UT21 -- the hooks are shown the files
# ---------------------------------------------------------------------------
class _RedactingHooks:
    """A pre_inference hook that redacts a phrase in each file it is shown and leaves the words alone."""

    def __init__(self):
        self.seen = []

    def has_hooks(self, kind):
        return kind == "pre_inference"

    def execute(self, kind, **kwargs):
        data = kwargs.get("data") or {}
        self.seen.append(data)
        documents = [{"filename": d["filename"], "content": d["content"].replace("40 guests", "[redacted]")}
                     for d in data.get("documents") or ()]
        return SimpleNamespace(final_data={**data, "documents": documents}, results=[])


def test_ut21_a_pre_inference_hook_sees_the_files_and_may_rewrite_their_text_which_stays_a_document():
    world, restore = _world()
    try:
        hooks = _RedactingHooks()
        world.rc.PLUGIN_HOOKS_AVAILABLE = True
        world.rc._hook_manager = hooks
        _stream(world, _QUESTION, documents=_documents())
        users = _saved_users(world.store)
    finally:
        restore()
    assert hooks.seen and hooks.seen[0].get("documents") == _documents(), "the hook is shown the files"
    assert len(users) == 1, f"control: the turn was saved: {users}"
    assert users[0]["origin"] == "typed", "words no hook touched stay typed"
    assert _parts(users[0]["content"], users[0]["segments"]) == [
        (_QUESTION, "typed"), (_DOCS[0][1].replace("40 guests", "[redacted]"), "document"), (_DOCS[1][1], "document")]


# ---------------------------------------------------------------------------
# UT22 -- a stored turn read as the grammar admits it
# ---------------------------------------------------------------------------
class _Mirror:
    def __init__(self, turns):
        self.turns = turns

    def get_mirror_messages(self, conversation_id):
        return list(self.turns)


def test_ut22_a_stored_turn_outside_the_grammar_or_of_another_text_comes_back_legacy():
    world, restore = _world()
    try:
        claim = world.ex.user_turn(_QUESTION, _QUESTION, _DOCS)
        kept = _segments(claim)
        end = len(claim.content)
        cases = (
            (claim.content, "typed", kept, ("typed", kept)),
            (claim.content + " ", "typed", kept, ("legacy", [])),
            (claim.content, "typed", [[0, end + 5, "typed"]], ("legacy", [])),
            (claim.content, "typed", [kept[1], kept[0]], ("legacy", [])),
            (claim.content, "typed", [[0, 5, "assistant"]], ("legacy", [])),
            (claim.content, "typed", [[3, 3, "typed"]], ("legacy", [])),
            (claim.content, "assistant", [], ("legacy", [])),
        )
        read = []
        for stored, origin, segments, _expected in cases:
            world.rc.conversation_manager = _Mirror([{"role": "user", "content": stored, "origin": origin,
                                                      "segments": segments}])
            got = world.rc._stored_user_turn("conv-1", claim.content)
            read.append((got.content, got.origin, _segments(got)))
    finally:
        restore()
    assert read[0] == (claim.content, "typed", kept), "control: a stored turn inside the grammar is kept"
    for (stored, origin, segments, expected), got in zip(cases, read):
        assert got[0] == claim.content and got[1:] == expected, (origin, segments, got)


# ---------------------------------------------------------------------------
# UT23 -- a refused name names its rule
# ---------------------------------------------------------------------------
def test_ut23_a_refused_name_says_which_rule_it_breaks(tmp_path):
    rc, close = _routes_alone()
    try:
        small = tmp_path / "chat.yaml"
        small.write_text("attachments:\n  max_documents: 4\n  max_document_bytes: 8\n  max_filename_chars: 6\n",
                         encoding="utf-8")
        rc._CHAT_CONFIG = small
        empty = rc._attachment_refusal([("ok.txt", "x"), ("", "x")])
        long = rc._attachment_refusal([("ok.txt", "x"), ("x" * 7, "x")])
        odd = rc._attachment_refusal([("ok.txt", "x"), ("a" + chr(10) + "b", "x")])
    finally:
        close()
    assert empty and "document 2" in empty and "no name" in empty, empty
    assert long and "document 2" in long and "7 characters" in long and "limit of 6" in long, long
    assert "chat.yaml" in long, long
    assert odd and "document 2" in odd and "not printable" in odd and "'a\\nb'" in odd, odd
    assert "not printable" not in long and "no name" not in odd, "each refusal names its own rule"


# ---------------------------------------------------------------------------
# UT24 and UT25 -- the files shown to the hooks, through the real manager
# ---------------------------------------------------------------------------
_PAYROLL = [("payroll.csv", "alice,92000")]


def test_ut24_a_hook_without_the_inference_content_permission_is_never_shown_a_files_text():
    world, restore = _world()
    try:
        manager = world.hooks.HookManager()
        seen = []
        manager.register("pre_inference", "watcher", lambda ctx: seen.append(json.dumps(ctx.data)) or None)
        world.rc.PLUGIN_HOOKS_AVAILABLE = True
        world.rc._hook_manager = manager
        _stream(world, _QUESTION, documents=_documents(_PAYROLL))
        users = _saved_users(world.store)
        placeholder = world.hooks.REDACTED_PLACEHOLDER
    finally:
        restore()
    assert seen, "control: the hook ran"
    shown = json.loads(seen[0])
    assert shown["message"] == placeholder, "control: the words are hidden from it"
    assert shown.get("documents") == placeholder, f"so are the files: {shown.get('documents')!r}"
    assert "alice,92000" not in seen[0]
    assert len(users) == 1 and users[0]["content"].endswith("alice,92000"), "the file still reaches the turn"


def test_ut25_a_hook_that_writes_a_file_in_place_and_fails_changes_nothing_and_a_rewrite_out_of_bounds_is_set_aside():
    def careless(ctx):
        ctx.data["documents"][0]["content"] = "INJECTED"
        raise RuntimeError("the plugin fails after writing")

    def oversize(ctx):
        return {"documents": [dict(d, content=d["content"] + "x" * 600_000) for d in ctx.data["documents"]]}

    def surrogate(ctx):
        return {"documents": [dict(d, content=d["content"] + chr(0xD800)) for d in ctx.data["documents"]]}

    kept = {}
    for plugin, callback in (("careless-editor", careless), ("trusted-editor", oversize),
                             ("trusted-editor", surrogate)):
        world, restore = _world()
        try:
            manager = world.hooks.HookManager()
            manager.register("pre_inference", plugin, callback)
            world.rc.PLUGIN_HOOKS_AVAILABLE = True
            world.rc._hook_manager = manager
            _stream(world, _QUESTION, documents=_documents(_PAYROLL))
            users = _saved_users(world.store)
            kept[callback.__name__] = [users[0]["content"][s:e] for s, e, base in users[0]["segments"]
                                       if base == "document"] if users else None
        finally:
            restore()
    assert kept == {name: ["alice,92000"] for name in ("careless", "oversize", "surrogate")}, kept


def test_ut26_on_the_coding_path_the_hook_of_each_model_call_hides_the_files_from_a_plugin_without_the_permission():
    world, restore = _world()
    try:
        manager = world.hooks.HookManager()
        seen = []
        manager.register("pre_inference", "watcher", lambda ctx: seen.append(json.dumps(ctx.data)) or None)
        world.rc.PLUGIN_HOOKS_AVAILABLE = True
        world.rc._hook_manager = manager
        session = _CodingSession()
        _coding(world, session, lambda message: SimpleNamespace(cleaned_message=message, raw_message=message))
        _stream(world, "Fix the payroll export.", documents=_documents(_PAYROLL), chat_coding=True)
        placeholder = world.hooks.REDACTED_PLACEHOLDER
    finally:
        restore()
    assert session.calls, "control: the coding session ran and called its model"
    assert seen, "control: the model call's hook ran"
    assert all("alice,92000" not in data for data in seen), "no file's text reaches the plugin"
    assert all(json.loads(data)["message"] == placeholder for data in seen), "nor do the words"
