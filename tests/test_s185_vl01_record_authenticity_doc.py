"""S185 audit fix VL-01 -- document the record-authenticity gap (Option A).

Veilid sync records carry a content hash (integrity), not a signature
(authenticity), and the in-record ``device`` provenance is self-asserted -- not
bound to the authenticated peer identity. Signing records per device is a
wire-format change touching records, pairing, and reconcile (a migration for
existing unsigned records), so it is design-scale. S185 takes the documentation
route: VEILID_SPEC records the gap, an in-code note sits on verify_record_hash,
and the per-device-signing work is recorded as a future sync-authenticity cycle.

These are source-content assertions that lock that documentation in place so the
known gap and the planned remediation cannot be silently dropped. There is no
runtime change to assert (live Veilid is out of scope for this pass).
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8").lower()


def test_veilid_spec_documents_record_authenticity_gap():
    text = _read("VEILID_SPEC.md")
    assert "vl-01" in text
    assert "authenticity" in text
    # The hash-is-not-authenticity point and the self-asserted device provenance.
    assert "self-asserted" in text or "self asserted" in text
    assert "signature" in text
    assert "sync-authenticity cycle" in text


def test_records_py_has_in_code_note():
    text = _read("opti_oignon/veilid/records.py")
    assert "vl-01" in text
    # The note must sit on the integrity check and point to per-device signing.
    body = text.split("def verify_record_hash", 1)[1].split("\ndef ", 1)[0]
    assert "integrity" in body
    assert "authenticity" in body
    assert "verify_record_signature" in body


def test_roadmap_records_the_sync_authenticity_cycle():
    text = _read("ROADMAP_POST_S183.md")
    assert "sync-authenticity cycle" in text
    assert "vl-01" in text
    # The cycle describes per-device signing of the wire record.
    assert "signature" in text
    assert "per-device signing" in text
