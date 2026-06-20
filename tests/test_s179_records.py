#!/usr/bin/env python3
"""Tests for S179 Goal 1 -- the Veilid record encoding (Theme 4 / Veilid Sync).

Covers opti_oignon/veilid/records.py:

- The record kinds and their allowlist frozenset.
- The content hash: stable for the same content, order-independent on payload
  keys, sensitive to kind / id / payload / tombstone, and independent of the
  metadata (clock, device, timestamp).
- The encode / decode round-trip, including through the JSON wire helpers, with
  every field preserved.
- Defensive decoding: a non-mapping, a wrong format version, an unknown kind, a
  bad identity, a bool / negative / non-int clock, a bad device, a missing hash, a
  non-mapping payload, a non-bool tombstone, and a tampered payload (hash
  mismatch) are all rejected as None and never raise; a batch separates the
  parseable from the rejected; bad JSON yields an empty result.

Loaded via spec_from_file_location with opti_oignon stubbed. records.py is pure
and domain-free (standard library only), so no security-mode or audit-log stub is
needed here; the package stubs only let the dotted module name resolve.
"""

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"


def _ensure_stubs() -> None:
    for name, sub in (("opti_oignon", OO), ("opti_oignon.veilid", VEILID)):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(sub)]
            sys.modules[name] = mod


def _load(name: str):
    full = f"opti_oignon.veilid.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(VEILID / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod  # register before exec (3.12+ dataclass processing)
    spec.loader.exec_module(mod)
    return mod


_ensure_stubs()
records = _load("records")
RecordKind = records.RecordKind
SyncRecord = records.SyncRecord


def _rec(**over):
    base = dict(
        kind=RecordKind.CONVERSATION,
        record_id="c-1",
        payload={"title": "hello", "turns": [1, 2, 3]},
        device="dev-A",
        clock=5,
    )
    base.update(over)
    return records.new_record(**base)


# Kinds


class TestKinds:
    def test_four_kinds(self):
        values = {k.value for k in RecordKind}
        assert values == {
            "conversation",
            "memory_canonical",
            "memory_archive",
            "skill",
        }

    def test_two_memory_tiers(self):
        assert RecordKind.MEMORY_CANONICAL.value == "memory_canonical"
        assert RecordKind.MEMORY_ARCHIVE.value == "memory_archive"

    def test_allowlist_matches(self):
        assert records.RECORD_KINDS == frozenset(k.value for k in RecordKind)
        assert isinstance(records.RECORD_KINDS, frozenset)


# Content hash


class TestContentHash:
    def test_stable_for_same_content(self):
        a = records.content_hash_for(RecordKind.SKILL, "s-1", {"x": 1}, False)
        b = records.content_hash_for(RecordKind.SKILL, "s-1", {"x": 1}, False)
        assert a == b
        assert len(a) == 64  # sha256 hex

    def test_order_independent_on_payload_keys(self):
        a = records.content_hash_for("conversation", "c", {"a": 1, "b": 2}, False)
        b = records.content_hash_for("conversation", "c", {"b": 2, "a": 1}, False)
        assert a == b

    def test_sensitive_to_payload(self):
        a = records.content_hash_for("skill", "s", {"x": 1}, False)
        b = records.content_hash_for("skill", "s", {"x": 2}, False)
        assert a != b

    def test_sensitive_to_tombstone(self):
        live = records.content_hash_for("skill", "s", {"x": 1}, False)
        gone = records.content_hash_for("skill", "s", {"x": 1}, True)
        assert live != gone

    def test_sensitive_to_kind_and_id(self):
        h_kind_a = records.content_hash_for("skill", "s", {}, False)
        h_kind_b = records.content_hash_for("conversation", "s", {}, False)
        h_id = records.content_hash_for("skill", "t", {}, False)
        assert h_kind_a != h_kind_b
        assert h_kind_a != h_id

    def test_independent_of_metadata(self):
        # Two records, same content but different clock/device/timestamp, share a hash.
        r1 = _rec(device="dev-A", clock=1, updated_at="2024-01-01T00:00:00Z")
        r2 = _rec(device="dev-Z", clock=99, updated_at="2025-12-31T23:59:59Z")
        assert r1.content_hash == r2.content_hash


# new_record


class TestNewRecord:
    def test_computes_hash(self):
        r = _rec()
        assert records.verify_record_hash(r)

    def test_key_of(self):
        r = _rec(kind=RecordKind.MEMORY_CANONICAL, record_id="m-7")
        assert records.key_of(r) == ("memory_canonical", "m-7")

    def test_defaults(self):
        r = _rec()
        assert r.deleted is False
        assert r.updated_at == ""

    @pytest.mark.parametrize(
        "over",
        [
            {"kind": "nonsense"},
            {"record_id": ""},
            {"record_id": 123},
            {"clock": True},
            {"clock": -1},
            {"clock": "5"},
            {"device": ""},
            {"deleted": "yes"},
        ],
    )
    def test_rejects_bad_producer_input(self, over):
        with pytest.raises(ValueError):
            _rec(**over)


# Round-trip


class TestRoundTrip:
    def test_encode_decode_record(self):
        r = _rec(deleted=False, updated_at="2024-06-01T10:00:00Z")
        back = records.decode_record(records.encode_record(r))
        assert back == r

    def test_round_trip_tombstone(self):
        r = _rec(payload={}, deleted=True)
        back = records.decode_record(records.encode_record(r))
        assert back is not None
        assert back.deleted is True
        assert back == r

    def test_wire_json_round_trip(self):
        rs = [_rec(record_id="c-1"), _rec(record_id="c-2", kind=RecordKind.SKILL)]
        result = records.from_wire_json(records.to_wire_json(rs))
        assert result.rejected == 0
        assert result.records == rs

    def test_list_payload_survives(self):
        # A list payload round-trips through JSON and the hash still matches.
        r = _rec(payload={"turns": [{"role": "user"}, {"role": "assistant"}]})
        back = records.decode_record(records.encode_record(r))
        assert back is not None
        assert records.verify_record_hash(back)


# Defensive decoding


class TestDefensiveDecode:
    def _wire(self, **over):
        w = records.encode_record(_rec())
        w.update(over)
        return w

    @pytest.mark.parametrize(
        "obj",
        [
            None,
            42,
            "not a dict",
            ["list"],
        ],
    )
    def test_non_mapping_rejected(self, obj):
        assert records.decode_record(obj) is None

    def test_wrong_version_rejected(self):
        assert records.decode_record(self._wire(v=999)) is None
        assert records.decode_record(self._wire(v="1")) is None

    def test_unknown_kind_rejected(self):
        assert records.decode_record(self._wire(kind="bogus")) is None
        assert records.decode_record(self._wire(kind=7)) is None

    def test_bad_id_rejected(self):
        assert records.decode_record(self._wire(id="")) is None
        assert records.decode_record(self._wire(id=None)) is None

    def test_bad_clock_rejected(self):
        assert records.decode_record(self._wire(clock=True)) is None
        assert records.decode_record(self._wire(clock=-3)) is None
        assert records.decode_record(self._wire(clock="5")) is None

    def test_bad_device_rejected(self):
        assert records.decode_record(self._wire(device="")) is None
        assert records.decode_record(self._wire(device=None)) is None

    def test_missing_hash_rejected(self):
        assert records.decode_record(self._wire(hash="")) is None
        assert records.decode_record(self._wire(hash=None)) is None

    def test_non_mapping_payload_rejected(self):
        assert records.decode_record(self._wire(payload=[1, 2])) is None
        assert records.decode_record(self._wire(payload="x")) is None

    def test_non_bool_deleted_rejected(self):
        assert records.decode_record(self._wire(deleted="true")) is None
        assert records.decode_record(self._wire(deleted=1)) is None

    def test_tampered_payload_rejected(self):
        # The hash no longer matches the (mutated) content: reject, do not trust.
        w = records.encode_record(_rec())
        w["payload"] = {"title": "TAMPERED"}
        assert records.decode_record(w) is None

    def test_decode_never_raises(self):
        # A pathological mapping must not blow up the decoder.
        class Boom(dict):
            def get(self, *_a, **_k):
                raise RuntimeError("boom")

        assert records.decode_record(Boom()) is None


# Batch decode


class TestBatchDecode:
    def test_separates_good_and_rejected(self):
        good1 = records.encode_record(_rec(record_id="a"))
        good2 = records.encode_record(_rec(record_id="b"))
        bad1 = {"v": 999}
        bad2 = "not a record"
        result = records.decode_records([good1, bad1, good2, bad2])
        assert {r.record_id for r in result.records} == {"a", "b"}
        assert result.rejected == 2

    def test_non_iterable_yields_empty(self):
        result = records.decode_records(123)
        assert result.records == []
        assert result.rejected == 0

    def test_bad_json_yields_empty(self):
        assert records.from_wire_json("{ not json").records == []
        assert records.from_wire_json("{ not json").rejected == 0

    def test_json_non_array_yields_empty(self):
        assert records.from_wire_json('{"v": 1}').records == []


# Integrity helper


class TestVerifyHash:
    def test_true_for_good_record(self):
        assert records.verify_record_hash(_rec()) is True

    def test_false_for_corrupted_hash(self):
        r = _rec()
        bad = dataclasses.replace(r, content_hash="0" * 64)
        assert records.verify_record_hash(bad) is False
