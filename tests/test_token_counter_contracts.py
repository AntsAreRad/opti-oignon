#!/usr/bin/env python3
"""What the token counter promises about exactness, and about admitting less.

Every token figure this codebase produced before this module was a
character-ratio estimate. The counter adds a path to a real tokenizer and
with it one promise that carries everything else: a count is labelled
``exact`` only when the tokenizer actually answered. These contracts pin
that promise from every side it can be approached.

The exact path runs through one injectable seam -- a transport callable
taking a path and a payload -- and the wire shape is pinned here: the text
travels under ``content`` to ``/tokenize`` and the answer's ``tokens``
list is counted by length. Every way that round trip can fail is a
demotion, never a lie: a raising transport, a shape that is not a mapping,
a ``tokens`` that is not a list -- each falls back to the estimate and
says so in the method label. A disabled counter goes further: it never
consults the transport at all, even one handed to it, so the default
configuration cannot emit a byte of traffic.

The estimated floor has its own ladder. With the family-calibrated
estimator present the fallback delegates to it, model name and all --
proven by a witness value the length ratio could never produce. With that
module absent the floor is a plain character ratio with a minimum of one.
Empty input short-circuits to a zero labelled estimated, because no
tokenizer was consulted and no exactness may be claimed.

Aggregation over messages keeps the same honesty at scale: the total is
exact only when every counted part was exact, empty contents are skipped,
and the first exact failure closes the transport for the rest of the
batch -- a dead server is paid for once, not once per message. Endpoint
resolution is a fixed ladder (own config, then the backends file's
llama_server host, then the conventional local default), and the shared
module-level counter is a plain singleton with an explicit reset.

Loaded through the shared isolation window. The family estimator is the
only project seam the module reaches; it is seeded or blocked per
contract, and the transport is always caller-supplied except where a
contract deliberately exercises the built-in one against a closed local
port.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.token_counter"

# A value the character-ratio floor can never produce for our inputs, so a
# contract that sees it knows the fallback delegated to the seeded seam.
_CM_TOKENS = 777


_NO_PAYLOAD = object()


class _ScriptedTransport:
    """Answers a fixed payload and records every call."""

    def __init__(self, payload=_NO_PAYLOAD, error=None):
        self.payload = {"tokens": [1, 2, 3]} if payload is _NO_PAYLOAD else payload
        self.error = error
        self.calls = []

    def __call__(self, path, payload=None):
        self.calls.append((path, payload))
        if self.error is not None:
            raise self.error
        return self.payload


def _load(*, seed_cm=False, cm_capture=None):
    """Load the counter module in isolation.

    seed_cm    -- when true, a context-manager stand-in whose
                  ``estimate_tokens_calibrated`` returns ``_CM_TOKENS`` is
                  seeded (recording the model it was handed into
                  ``cm_capture`` when given); when false the name is
                  blocked and the character-ratio floor runs instead.
    cm_capture -- optional list receiving ``(text, model)`` per delegated
                  estimate.
    """
    seeded = {}
    blocked = []
    if seed_cm:
        cm = types.ModuleType("opti_oignon.context_manager")

        def _estimate(text, model=""):
            if cm_capture is not None:
                cm_capture.append((text, model))
            return _CM_TOKENS

        cm.estimate_tokens_calibrated = _estimate
        seeded["opti_oignon.context_manager"] = cm
    else:
        blocked.append("opti_oignon.context_manager")

    loaded, restore = isolate(
        targets={_TARGET: source("token_counter.py")},
        blocked=blocked,
        seeded=seeded,
    )
    return loaded[_TARGET], restore


# ---------------------------------------------------------------------------
# The honesty bit
# ---------------------------------------------------------------------------

def test_t1_disabled_counter_never_consults_an_injected_transport():
    mod, restore = _load(seed_cm=True)
    try:
        transport = _ScriptedTransport()
        counter = mod.TokenCounter(
            config={"enabled": False}, transport=transport
        )
        result = counter.count("some text worth counting", "qwen3:32b")
        assert result.method == mod.METHOD_ESTIMATED
        assert result.tokens == _CM_TOKENS
        assert transport.calls == []
        assert counter.exact_enabled is False
    finally:
        restore()


def test_t2_exact_count_is_the_token_list_length_and_the_wire_shape_is_pinned():
    mod, restore = _load(seed_cm=True)
    try:
        transport = _ScriptedTransport(payload={"tokens": [9, 8, 7, 6, 5]})
        counter = mod.TokenCounter(
            config={"enabled": True}, transport=transport
        )
        result = counter.count("hello there", "llama3.2:8b")
        assert result.tokens == 5
        assert result.method == mod.METHOD_EXACT
        assert result.source == mod.SOURCE_TOKENIZE
        assert transport.calls == [("/tokenize", {"content": "hello there"})]
    finally:
        restore()


def test_t3_raising_transport_demotes_to_the_estimate():
    capture = []
    mod, restore = _load(seed_cm=True, cm_capture=capture)
    try:
        transport = _ScriptedTransport(error=RuntimeError("server down"))
        counter = mod.TokenCounter(
            config={"enabled": True}, transport=transport
        )
        result = counter.count("hello there", "llama3.2:8b")
        assert result.method == mod.METHOD_ESTIMATED
        assert result.source == mod.SOURCE_FAMILY
        assert result.tokens == _CM_TOKENS
        assert len(transport.calls) == 1
        assert capture == [("hello there", "llama3.2:8b")]
    finally:
        restore()


def test_t4_malformed_answers_demote_to_the_estimate():
    mod, restore = _load(seed_cm=True)
    try:
        for bad in (["not", "a", "dict"], {"tokens": "not-a-list"}, {"other": 1}, None):
            counter = mod.TokenCounter(
                config={"enabled": True},
                transport=_ScriptedTransport(payload=bad),
            )
            result = counter.count("hello", "qwen3:32b")
            assert result.method == mod.METHOD_ESTIMATED, bad
            assert result.tokens == _CM_TOKENS, bad
    finally:
        restore()


def test_t5_empty_input_is_zero_estimated_and_never_consults_the_transport():
    mod, restore = _load(seed_cm=True)
    try:
        transport = _ScriptedTransport()
        counter = mod.TokenCounter(
            config={"enabled": True}, transport=transport
        )
        result = counter.count("", "qwen3:32b")
        assert result.tokens == 0
        assert result.method == mod.METHOD_ESTIMATED
        assert result.source == mod.SOURCE_EMPTY
        assert transport.calls == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# The estimated floor
# ---------------------------------------------------------------------------

def test_t6_fallback_delegates_to_the_family_estimator_with_the_model():
    capture = []
    mod, restore = _load(seed_cm=True, cm_capture=capture)
    try:
        counter = mod.TokenCounter(config={"enabled": False})
        result = counter.count("some text", "deepseek-r1:32b")
        assert result.tokens == _CM_TOKENS
        assert result.source == mod.SOURCE_FAMILY
        assert capture == [("some text", "deepseek-r1:32b")]
    finally:
        restore()


def test_t7_absent_family_estimator_falls_to_the_character_ratio_floor():
    mod, restore = _load(seed_cm=False)
    try:
        counter = mod.TokenCounter(config={"enabled": False})
        result = counter.count("abcdefghij", "any-model")
        assert result.tokens == len("abcdefghij") // 4
        assert result.method == mod.METHOD_ESTIMATED
        assert result.source == mod.SOURCE_CHAR_RATIO
        tiny = counter.count("ab", "any-model")
        assert tiny.tokens == 1
        assert tiny.source == mod.SOURCE_CHAR_RATIO
    finally:
        restore()


# ---------------------------------------------------------------------------
# Aggregation over messages
# ---------------------------------------------------------------------------

def test_t8_aggregate_is_exact_only_when_every_part_was_and_skips_empties():
    mod, restore = _load(seed_cm=True)
    try:
        transport = _ScriptedTransport(payload={"tokens": [1, 2, 3, 4]})
        counter = mod.TokenCounter(
            config={"enabled": True}, transport=transport
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "ask"},
        ]
        result = counter.count_messages(messages, "llama3")
        assert result.tokens == 8
        assert result.method == mod.METHOD_EXACT
        assert result.source == mod.SOURCE_TOKENIZE
        assert len(transport.calls) == 2
    finally:
        restore()


def test_t9_first_exact_failure_closes_the_transport_and_demotes_the_batch():
    mod, restore = _load(seed_cm=True)
    try:
        transport = _ScriptedTransport(error=RuntimeError("down"))
        counter = mod.TokenCounter(
            config={"enabled": True}, transport=transport
        )
        messages = [
            {"role": "system", "content": "one"},
            {"role": "user", "content": "two"},
            {"role": "user", "content": "three"},
        ]
        result = counter.count_messages(messages, "llama3")
        assert result.method == mod.METHOD_ESTIMATED
        assert result.tokens == 3 * _CM_TOKENS
        assert len(transport.calls) == 1
    finally:
        restore()


def test_t10_aggregate_of_nothing_countable_is_zero_estimated():
    mod, restore = _load(seed_cm=True)
    try:
        counter = mod.TokenCounter(config={"enabled": True},
                                   transport=_ScriptedTransport())
        for empty in (None, [], [{"role": "user", "content": ""}], [{}]):
            result = counter.count_messages(empty, "llama3")
            assert result.tokens == 0, empty
            assert result.method == mod.METHOD_ESTIMATED, empty
            assert result.source == mod.SOURCE_EMPTY, empty
    finally:
        restore()


# ---------------------------------------------------------------------------
# Configuration and resolution
# ---------------------------------------------------------------------------

def test_t11_config_defaults_and_junk_values_never_break_construction(tmp_path):
    mod, restore = _load(seed_cm=True)
    try:
        cfg = mod._load_config(tmp_path / "absent.yaml")
        assert cfg == {"enabled": False, "endpoint": "", "timeout_s": 0.5}
        for junk in ({"enabled": False, "timeout_s": "abc"},
                     {"enabled": False, "timeout_s": -3}):
            counter = mod.TokenCounter(config=junk)
            result = counter.count("text", "llama3")
            assert result.method == mod.METHOD_ESTIMATED
        partial = tmp_path / "partial.yaml"
        partial.write_text("enabled: true\n", encoding="utf-8")
        loaded = mod._load_config(partial)
        assert loaded["enabled"] is True
        assert loaded["timeout_s"] == 0.5
    finally:
        restore()


def test_t12_endpoint_resolution_ladder_own_then_backends_then_default(tmp_path):
    mod, restore = _load(seed_cm=True)
    try:
        own = mod._resolve_endpoint({"endpoint": "http://10.0.0.5:9999/"})
        assert own == "http://10.0.0.5:9999"

        backends = tmp_path / "backends.yaml"
        backends.write_text(
            "llama_server:\n  host: http://127.0.0.1:9090\n",
            encoding="utf-8",
        )
        original = mod._BACKENDS_CONFIG_PATH
        mod._BACKENDS_CONFIG_PATH = backends
        try:
            assert mod._resolve_endpoint({"endpoint": ""}) == "http://127.0.0.1:9090"
        finally:
            mod._BACKENDS_CONFIG_PATH = original

        mod._BACKENDS_CONFIG_PATH = tmp_path / "absent.yaml"
        try:
            assert mod._resolve_endpoint({"endpoint": ""}) == mod._DEFAULT_ENDPOINT
        finally:
            mod._BACKENDS_CONFIG_PATH = original
    finally:
        restore()


def test_t13_built_in_transport_fails_closed_against_a_dead_local_port():
    mod, restore = _load(seed_cm=True)
    try:
        counter = mod.TokenCounter(
            config={
                "enabled": True,
                "endpoint": "http://127.0.0.1:9",
                "timeout_s": 0.05,
            }
        )
        assert counter.exact_enabled is True
        result = counter.count("hello", "llama3")
        assert result.method == mod.METHOD_ESTIMATED
        assert result.tokens == _CM_TOKENS
    finally:
        restore()


def test_t14_shared_counter_is_a_singleton_with_an_explicit_reset():
    mod, restore = _load(seed_cm=True)
    try:
        first = mod.get_token_counter()
        again = mod.get_token_counter()
        assert first is again
        assert first.exact_enabled is False
        mod.reset_token_counter()
        rebuilt = mod.get_token_counter()
        assert rebuilt is not first
    finally:
        restore()
