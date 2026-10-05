#!/usr/bin/env python3
"""What the chat path promises about the material a summary is made of.

A summary of a summary drifts: each pass restates the last one in fresh
words, keeps whatever an earlier pass let in, and loses the trace of where
it came from. The manual pipeline did exactly that. The history it loads is
the whole archive, yet it also re-inserted the stored summary of that same
archive at the head, then merged that summary with the oldest turns on every
overflow, so the same turns were folded in again on every request. The tier
layer did it one level up, rolling the segment summaries into a summary of
summaries.

These contracts pin the other rule. A summary stands in for turns that left
the window, never beside turns that are still in it. Every summarizer call
reads archived turns and nothing else. The frozen segments cover the oldest
part of an evicted span, only segments wholly inside that span are composed,
and the live part is summarized from the turns after them. The composition
keeps the first segment, where a conversation usually states its task, and
the newest ones that fit its budget, and says how many it left out; there is
no rollup. Out of step with the archive, a summary covers the evicted turns
alone.

A recording summarizer stands in for the model: it is the seam, and every
assertion reads what the pipeline chose to hand it.
"""

import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_EXECUTOR = "opti_oignon.executor"
_WRAPPER = "opti_oignon.agent.untrusted_context"
_TIERS = "opti_oignon.context_summary_tiers"

_OLD = "OLD-SUMMARY-SENTINEL the user said the bed floods"
_QUESTION = "What should I plant next?"
_HEAD = "You are the assistant."

_BLOCK_RE = re.compile(
    r'<untrusted_data source="([a-z0-9_\-]+)" trusted="false">\n(.*?)\n</untrusted_data>',
    re.DOTALL,
)


class _Summarizer:
    """The model seam of the live summarizer: records every call."""

    SUMMARY_THRESHOLD = 4
    MAX_SUMMARY_TOKENS = 40
    available = True

    def __init__(self):
        self.calls = []

    def summarize_messages(self, messages, existing_summary=None, **kwargs):
        self.calls.append(
            {"messages": [dict(m) for m in messages], "existing": existing_summary}
        )
        return f"LIVE-SUMMARY over {len(messages)} turns"

    def create_summary_message(self, summary):
        return {"role": "system", "content": f"[Summary of earlier conversation]\n{summary}"}


def _is_legacy_summary(message):
    return message.get("role") == "system" and "[Summary" in message.get("content", "")


def _extract_legacy(message):
    if not _is_legacy_summary(message):
        return None
    parts = message.get("content", "").split("\n", 1)
    return parts[1].strip() if len(parts) > 1 else parts[0].strip()


def _archive(pairs, start_id=1):
    out = []
    next_id = start_id
    for i in range(pairs):
        filler = " loam" * 10
        out.append({"id": next_id, "role": "user", "content": f"turn {i} user: which onion for clay soil, part {i}{filler}"})
        out.append({"id": next_id + 1, "role": "assistant", "content": f"turn {i} assistant: a red onion copes with clay, part {i}{filler}"})
        next_id += 2
    return out


class _Conv:
    def __init__(self, archive, metadata=None):
        self.archive = [dict(m) for m in archive]
        self.metadata = dict(metadata or {})

    def get_messages(self, cid):
        return [SimpleNamespace(**m) for m in self.archive]

    def get_context_messages(self, cid):
        return [{"role": m["role"], "content": m["content"]} for m in self.archive]

    def get_conversation(self, cid):
        return SimpleNamespace(metadata=dict(self.metadata))

    def update_conversation_metadata(self, cid, metadata=None, **k):
        self.metadata.update(metadata or {})

    def add_message(self, *a, **k):
        pass


def _load(conv):
    summarizer = _Summarizer()
    summary_mod = types.ModuleType("opti_oignon.context_summary")
    summary_mod.context_summarizer = summarizer
    summary_mod.is_summary_message = _is_legacy_summary
    summary_mod.extract_summary_text = _extract_legacy
    convmod = types.ModuleType("opti_oignon.conversation")
    convmod.conversation_manager = conv
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(
        get_model=lambda *a, **k: "test-model:1b",
        get_temperature=lambda *a, **k: 0.2,
    )
    router = types.ModuleType("opti_oignon.router")
    router.RoutingResult = type("RoutingResult", (), {})
    loaded, restore = isolate(
        targets={
            _WRAPPER: source("agent", "untrusted_context.py"),
            _TIERS: source("context_summary_tiers.py"),
            # The executor imports the deduplicator plainly; it is pure and
            # standard-library only, so it rides along as a real target.
            "opti_oignon.context_dedup": source("context_dedup.py"),
            _EXECUTOR: source("executor.py"),
        },
        seeded={
            "opti_oignon.context_summary": summary_mod,
            "opti_oignon.conversation": convmod,
            "opti_oignon.config": cfg,
            "opti_oignon.router": router,
        },
        packages=("opti_oignon.agent",),
    )
    mod = loaded[_EXECUTOR]
    # A small window: sixteen exchanges overflow its soft limit, and what a
    # summary frees keeps the result under its hard limit, so no forced cut
    # follows the summary path.
    mod.CONTEXT_MANAGER_AVAILABLE = True
    mod.cm_get_model_limits = lambda model: SimpleNamespace(context_window=900, max_output=100)
    mod.cm_estimate_tokens = lambda text, model=None: len(text) // 4
    world = SimpleNamespace(
        mod=mod, tiers=loaded[_TIERS], wrapper=loaded[_WRAPPER], summarizer=summarizer, conv=conv
    )
    return world, restore


def _build(world):
    ex = world.mod.Executor()
    msgs, _tokens, _stats = ex._build_conversation_messages(
        system_prompt=_HEAD,
        conversation_id="conv-1",
        current_message=_QUESTION,
        model="test-model:1b",
        prompt_budget=None,
    )
    return msgs


def _segment(tiers, archive, first_index, last_index, text):
    span = archive[first_index:last_index + 1]
    return tiers.SegmentRecord(
        first_id=span[0]["id"],
        last_id=span[-1]["id"],
        digest=tiers.span_digest(span),
        text=text,
    )


def _with_segments(tiers, segments):
    state = tiers.TierState(segments=list(segments))
    return {tiers.TIERS_METADATA_KEY: state.to_metadata_value()}


def _kept_ids(msgs, archive):
    """Archive ids of the turns the window still carries word for word."""
    by_content = {m["content"]: m["id"] for m in archive}
    ids = []
    for m in msgs:
        for piece in m["content"].split("\n\n"):
            if piece in by_content:
                ids.append(by_content[piece])
    return ids


def _input_texts(world):
    return [m["content"] for call in world.summarizer.calls for m in call["messages"]]


# ---------------------------------------------------------------------------
# fs1 -- across two overflows, no summarizer input is a summary
# ---------------------------------------------------------------------------

def test_fs1_no_summarizer_input_is_a_summary_across_two_overflows():
    archive = _archive(16)
    world, restore = _load(_Conv(archive, metadata={"context_summary": _OLD}))
    try:
        _build(world)
        _build(world)
        assert len(world.summarizer.calls) >= 2, "both requests must overflow into the summarizer"
        assert [c["existing"] for c in world.summarizer.calls if c["existing"]] == []
        archived = {m["content"] for m in archive}
        foreign = [t[:40] for t in _input_texts(world) if t not in archived]
        assert foreign == [], "a summarizer read something that is not an archived turn"
    finally:
        restore()


# ---------------------------------------------------------------------------
# fs2 -- a stored summary is never restored beside the archive it summarizes
# ---------------------------------------------------------------------------

def test_fs2_a_stored_summary_is_never_restored_beside_the_archive():
    archive = _archive(1)
    world, restore = _load(_Conv(archive, metadata={"context_summary": _OLD}))
    try:
        msgs = _build(world)
        assert any(archive[0]["content"] in m["content"] for m in msgs), "the archive itself must be there"
        assert [m["role"] for m in msgs if _OLD in m["content"]] == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# fs3 -- the live part starts after the last frozen segment inside the span
# ---------------------------------------------------------------------------

def test_fs3_the_live_part_of_an_evicted_span_is_summarized_from_the_turns_after_its_segments():
    archive = _archive(16)
    conv = _Conv(archive)
    world, restore = _load(conv)
    try:
        conv.metadata.update(
            _with_segments(world.tiers, [_segment(world.tiers, archive, 0, 3, "SEGMENT-ONE")])
        )
        msgs = _build(world)
        kept = _kept_ids(msgs, archive)
        assert kept, "some turns must stay word for word"
        first_kept = min(kept)
        assert first_kept > archive[3]["id"], "the span must reach past the frozen segment"
        expected = [m["content"] for m in archive if archive[3]["id"] < m["id"] < first_kept]
        assert expected, "the evicted span must have a live part"
        live_calls = [c for c in world.summarizer.calls if c["messages"]]
        assert [m["content"] for m in live_calls[-1]["messages"]] == expected
    finally:
        restore()


# ---------------------------------------------------------------------------
# fs4 -- a segment reaching into the kept turns is not composed
# ---------------------------------------------------------------------------

def test_fs4_a_segment_reaching_into_the_kept_turns_is_not_composed():
    archive = _archive(16)
    conv = _Conv(archive)
    world, restore = _load(conv)
    try:
        conv.metadata.update(
            _with_segments(
                world.tiers,
                [
                    _segment(world.tiers, archive, 0, 3, "SEGMENT-ONE"),
                    _segment(world.tiers, archive, 4, 29, "SEGMENT-REACHING"),
                ],
            )
        )
        msgs = _build(world)
        kept = _kept_ids(msgs, archive)
        assert kept and min(kept) <= archive[29]["id"], "the second segment must overlap kept turns"
        assert any("SEGMENT-ONE" in m["content"] for m in msgs), "the inner segment is composed"
        assert [m["role"] for m in msgs if "SEGMENT-REACHING" in m["content"]] == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# fs5 -- advancing the tiers never builds a rollup
# ---------------------------------------------------------------------------

def test_fs5_advancing_the_tiers_never_builds_a_rollup():
    loaded, restore = isolate(targets={_TIERS: source("context_summary_tiers.py")})
    try:
        tiers = loaded[_TIERS]
        archive = _archive(20)
        inputs = []

        def summarize(messages):
            inputs.append([dict(m) for m in messages])
            return f"segment of {len(messages)}"

        manager = tiers.TierManager(
            lambda cid: archive, segment_budget_tokens=20, tail_keep_messages=2
        )
        update = manager.advance("conv-1", {}, summarize_fn=summarize)
        state = tiers.TierState.from_metadata(update)
        assert len(state.segments) >= 3, "several segments must freeze"
        assert state.rollup is None
        assert [m for call in inputs for m in call if "id" not in m] == [], (
            "every summarizer input is an archived turn"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# fs6 -- the first segment and the newest that fit, the rest named
# ---------------------------------------------------------------------------

def test_fs6_composition_keeps_the_first_segment_and_the_newest_that_fit_and_names_the_rest():
    loaded, restore = isolate(targets={_TIERS: source("context_summary_tiers.py")})
    try:
        tiers = loaded[_TIERS]
        archive = _archive(10)
        texts = [f"SEG{i} " + "x" * 36 for i in range(5)]
        segments = [
            _segment(tiers, archive, 4 * i, 4 * i + 3, texts[i]) for i in range(5)
        ]
        manager = tiers.TierManager(lambda cid: archive, estimate=lambda t: len(t))
        out = manager.compose(
            "conv-1", _with_segments(tiers, segments),
            # Three segments and the line counting the omitted ones.
            budget_tokens=3 * 40 + 10 + len(tiers.omission_line(len(segments))),
        )
        assert [t.split()[0] for t in texts if t in out] == ["SEG0", "SEG3", "SEG4"]
        assert out.index("SEG0") < out.index("SEG3") < out.index("SEG4")
        assert re.search(r"\b2 earlier summary segments? omitted", out)
    finally:
        restore()


# ---------------------------------------------------------------------------
# fs7 -- out of step with the archive, only the evicted turns are summarized
# ---------------------------------------------------------------------------

def test_fs7_out_of_step_with_the_archive_the_summary_covers_the_evicted_turns_only():
    archive = _archive(16)
    world, restore = _load(_Conv(archive))
    try:
        block = world.wrapper.untrusted_message(
            "[Summary of earlier conversation]\nCOMPRESSED-SENTINEL", source="memory"
        )
        history = [block] + [{"role": m["role"], "content": m["content"]} for m in archive[4:]]
        ex = world.mod.Executor()
        total = sum(ex._estimate_tokens(m["content"], "test-model:1b") for m in history)
        done = ex._summarize_old_messages(history, total, total // 2, "test-model:1b", "conv-1")
        assert done is True
        texts = _input_texts(world)
        assert texts, "the summarizer must have been asked"
        assert [t for t in texts if "COMPRESSED-SENTINEL" in t] == []
        assert set(texts) <= {m["content"] for m in archive[4:]}
        assert any("COMPRESSED-SENTINEL" in m["content"] for m in history), (
            "the block that was there stays there"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# fs8 -- the summary the pipeline writes is memory data in the user role
# ---------------------------------------------------------------------------

def test_fs8_the_live_summary_rides_the_user_role_as_memory_data():
    archive = _archive(16)
    world, restore = _load(_Conv(archive))
    try:
        msgs = _build(world)
        holders = [m for m in msgs if "LIVE-SUMMARY over" in m["content"]]
        assert holders, "the summary must reach the window"
        assert [m["role"] for m in holders] == ["user"]
        assert any("LIVE-SUMMARY over" in b for b in
                   [body for label, body in _BLOCK_RE.findall(holders[0]["content"]) if label == "memory"])
    finally:
        restore()


# ---------------------------------------------------------------------------
# fs9 -- with frozen segments in the record, no turn leaves unrepresented
# ---------------------------------------------------------------------------

def test_fs9_with_frozen_segments_every_turn_is_kept_summarized_or_counted_as_omitted():
    archive = _archive(16)
    conv = _Conv(archive)
    world, restore = _load(conv)
    try:
        segments = [
            _segment(world.tiers, archive, 0, 3, "SEGMENT-A " + "a" * 590),
            _segment(world.tiers, archive, 4, 7, "SEGMENT-B " + "b" * 590),
        ]
        conv.metadata.update(_with_segments(world.tiers, segments))
        msgs = _build(world)
        kept = sorted(_kept_ids(msgs, archive))
        assert kept, "some turns must stay word for word"
        last = archive[-1]["id"]
        assert kept == list(range(kept[0], last + 1)), "the verbatim turns run unbroken to the newest"
        text = "\n".join(m["content"] for m in msgs)
        composed = [s for s in segments if s.text in text]
        match = re.search(r"\[(\d+) earlier summary segments? omitted", text)
        omitted = int(match.group(1)) if match else 0
        assert len(composed) + omitted == len(segments), "every segment is composed or counted as omitted"
        by_content = {m["content"]: m["id"] for m in archive}
        live = {by_content[t] for t in _input_texts(world) if t in by_content}
        covered = {i for s in segments for i in range(s.first_id, s.last_id + 1)} | live
        assert set(range(1, kept[0])) <= covered, (
            "a turn before the verbatim ones is neither in a segment nor in the live summary"
        )
        assert live, "control: turns evicted after the segments went to the live summarizer"
        assert "LIVE-SUMMARY" in text, "the live summary those turns went into reached the window"
    finally:
        restore()


# ---------------------------------------------------------------------------
# fs11 -- out of step with the archive, no room is taken for segments
# ---------------------------------------------------------------------------

def test_fs11_when_no_segment_can_be_composed_the_composition_bound_evicts_nothing_more():
    archive = _archive(16)
    evicted = []
    for share in (0.01, 1.0):
        world, restore = _load(_Conv(archive))
        try:
            world.mod.load_tier_settings = lambda path=None, s=share: world.tiers.TierSettings(
                segment_budget_tokens=1200, tail_keep_messages=4, compose_budget_tokens=5000, compose_share=s
            )
            block = world.wrapper.untrusted_message(
                "[Summary of earlier conversation]\nCOMPRESSED-SENTINEL", source="memory"
            )
            history = [block] + [{"role": m["role"], "content": m["content"]} for m in archive[4:]]
            ex = world.mod.Executor()
            total = sum(ex._estimate_tokens(m["content"], "test-model:1b") for m in history)
            assert ex._summarize_old_messages(history, total, total // 2, "test-model:1b", "conv-1") is True
            evicted.append(len(_input_texts(world)))
        finally:
            restore()
    assert evicted[0] > 0, "control: some turns were summarized"
    assert evicted[0] == evicted[1], "the segment bound moved the eviction on a path that composes no segment"


# ---------------------------------------------------------------------------
# fs12 -- a live summary longer than its cap costs no verbatim turn
# ---------------------------------------------------------------------------

class _LongSummarizer(_Summarizer):
    """A model that writes past the cap it was given: the estimate it was budgeted at is wrong."""

    def summarize_messages(self, messages, existing_summary=None, **kwargs):
        super().summarize_messages(messages, existing_summary=existing_summary, **kwargs)
        # Past its cap and the unused room for segments, short of the hard limit.
        return "LONG-SUMMARY " + "w" * 1200


def test_fs12_a_live_summary_longer_than_its_cap_costs_no_verbatim_turn():
    archive = _archive(16)
    world, restore = _load(_Conv(archive))
    try:
        long = _LongSummarizer()
        world.summarizer = long
        world.mod.context_summarizer = long
        msgs = _build(world)
        assert any("LONG-SUMMARY" in m["content"] for m in msgs), "control: the summary reached the window"
        kept = sorted(_kept_ids(msgs, archive))
        assert kept, "some turns must stay word for word"
        assert kept == list(range(kept[0], archive[-1]["id"] + 1)), "the verbatim turns run unbroken to the newest"
        by_content = {m["content"]: m["id"] for m in archive}
        live = {by_content[t] for t in _input_texts(world) if t in by_content}
        assert set(range(1, kept[0])) <= live, "a turn the summary was written for is missing from it and from the window"
    finally:
        restore()


# ---------------------------------------------------------------------------
# fs10 -- the same, when the engine counts tokens unlike the tier manager
# ---------------------------------------------------------------------------

def _code_heavy(text, model=None):
    """A real engine's count: denser than four characters a token, and more again for code."""
    if not text:
        return 0
    estimate = len(text) / 3.2
    if "import " in text or "```" in text or "def " in text:
        estimate *= 1.15
    return max(1, int(estimate))


def test_fs10_with_an_engine_that_counts_code_heavier_no_turn_leaves_unrepresented():
    archive = _archive(16)
    conv = _Conv(archive)
    world, restore = _load(conv)
    try:
        world.mod.cm_estimate_tokens = _code_heavy
        filler = "import x; " * 60
        segments = [
            _segment(world.tiers, archive, 0, 3, "SEGMENT-A " + filler[:20]),
            _segment(world.tiers, archive, 4, 7, "SEGMENT-B " + filler[:530]),
        ]
        conv.metadata.update(_with_segments(world.tiers, segments))
        ex = world.mod.Executor()
        msgs, total, _stats = ex._build_conversation_messages(
            system_prompt=_HEAD, conversation_id="conv-1", current_message=_QUESTION,
            model="test-model:1b", prompt_budget=None,
        )
        soft = int((900 - 100) * ex.CONTEXT_SOFT_LIMIT)
        assert total <= soft, f"the window lands at {total} tokens, over its soft limit {soft}"
        kept = sorted(_kept_ids(msgs, archive))
        assert kept, "some turns must stay word for word"
        assert kept == list(range(kept[0], archive[-1]["id"] + 1)), "the verbatim turns run unbroken to the newest"
        text = "\n".join(m["content"] for m in msgs)
        composed = [s for s in segments if s.text in text]
        match = re.search(r"\[(\d+) earlier summary segments? omitted", text)
        omitted = int(match.group(1)) if match else 0
        assert len(composed) + omitted == len(segments), "every segment is composed or counted as omitted"
        by_content = {m["content"]: m["id"] for m in archive}
        live = {by_content[t] for t in _input_texts(world) if t in by_content}
        covered = {i for s in segments for i in range(s.first_id, s.last_id + 1)} | live
        assert set(range(1, kept[0])) <= covered, (
            "a turn before the verbatim ones is neither in a segment nor in the live summary"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# fs13 -- segments that grow when defanged are fitted as they are placed
# ---------------------------------------------------------------------------

_FORGED = "[/data]" * 20


def test_fs13_segments_whose_markers_grow_when_defanged_are_fitted_as_placed():
    """A recalled segment can carry forged frame markers. The wrapper defangs each
    into a marker three times its length, which no estimate of the parts sees."""
    archive = _archive(16)
    conv = _Conv(archive)
    world, restore = _load(conv)
    try:
        segments = [
            _segment(world.tiers, archive, 0, 3, "SEGMENT-A " + _FORGED),
            _segment(world.tiers, archive, 4, 7, "SEGMENT-B " + _FORGED),
        ]
        conv.metadata.update(_with_segments(world.tiers, segments))
        ex = world.mod.Executor()
        msgs, total, _stats = ex._build_conversation_messages(
            system_prompt=_HEAD, conversation_id="conv-1", current_message=_QUESTION,
            model="test-model:1b", prompt_budget=None,
        )
        text = "\n".join(m["content"] for m in msgs)
        assert "SEGMENT-" in text and "[/data]" not in text, "control: a segment is placed, its markers defanged"
        soft = int((900 - 100) * ex.CONTEXT_SOFT_LIMIT)
        assert total <= soft, f"the window lands at {total} tokens, over its soft limit {soft}"
        kept = sorted(_kept_ids(msgs, archive))
        assert kept, "some turns must stay word for word"
        assert kept == list(range(kept[0], archive[-1]["id"] + 1)), "the verbatim turns run unbroken to the newest"
        composed = [s for s in segments if s.text.split(" ", 1)[0] in text]
        match = re.search(r"\[(\d+) earlier summary segments? omitted", text)
        omitted = int(match.group(1)) if match else 0
        assert len(composed) + omitted == len(segments), "every segment is composed or counted as omitted"
    finally:
        restore()


# ---------------------------------------------------------------------------
# fs14 -- the tier manager weighs each segment as the engine counts
# ---------------------------------------------------------------------------

def test_fs14_the_tier_manager_weighs_each_segment_as_the_engine_counts():
    archive = _archive(16)
    conv = _Conv(archive)
    world, restore = _load(conv)
    try:
        weighed = []

        def engine_count(text, model=None):
            weighed.append(text)
            return _code_heavy(text, model)

        world.mod.cm_estimate_tokens = engine_count
        filler = "import x; " * 60
        segments = [
            _segment(world.tiers, archive, 0, 3, "SEGMENT-A " + filler[:20]),
            _segment(world.tiers, archive, 4, 7, "SEGMENT-B " + filler[:530]),
        ]
        conv.metadata.update(_with_segments(world.tiers, segments))
        _build(world)
        assert any("SEGMENT-A" in t for t in weighed), "control: the placed summary was weighed by the engine"
        assert segments[0].text in weighed, "the first segment was weighed on its own by the engine's count"
        assert segments[1].text in weighed, "the newest segment was weighed on its own by the engine's count"
    finally:
        restore()
