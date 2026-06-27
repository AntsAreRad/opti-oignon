#!/usr/bin/env python3
"""Tests for the vault blob on-demand transfer (SYN-01, lot 3d-2).

The full fetch logic is exercised in isolation with producer and receiver
:class:`NotesBlobStore` instances holding DIFFERENT master keys (the receiver
re-seals under its own key), and the wire reduced to direct ``pull_chunk``
calls. ``blob_store.py`` loads with a stub ``opti_oignon.encryption`` backed by
real AES-256-GCM; ``blob_transfer.py`` loads standalone (it imports nothing
from the package).

Proven here:

  * a multi-frame blob transfers end-to-end -- producer streams off disk, the
    receiver re-seals under its own key, and the reassembled plaintext matches;
  * the content-hash check DISCARDS a transfer whose bytes were corrupted on
    the wire (the end-to-end integrity gate, independent of at-rest framing),
    and a wrong expected hash never lands;
  * the producer gate refuses a non-served attachment, the session is bound to
    its peer (another peer cannot pull it), the cursor tolerates a retry, and
    the concurrent-session bound is enforced.

Local-only. Runs under pytest or the __main__ runner.
"""

import hashlib
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"
_KEY_A = bytes(range(32))
_KEY_B = bytes(range(32, 64))


def _load_blob_store():
    keys = ("opti_oignon", "opti_oignon.notes", "opti_oignon.encryption",
            "opti_oignon.notes.blob_store")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon"); pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    npkg = types.ModuleType("opti_oignon.notes"); npkg.__path__ = []
    sys.modules["opti_oignon.notes"] = npkg

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    enc = types.ModuleType("opti_oignon.encryption")
    enc.encrypt_bytes = lambda key, pt: bytes([1]) + (
        lambda n: n + AESGCM(key).encrypt(n, pt, None)
    )(os.urandom(12))
    enc.decrypt_bytes = lambda key, blob: AESGCM(key).decrypt(blob[1:13], blob[13:], None)
    enc.get_encryption_key = lambda: None
    sys.modules["opti_oignon.encryption"] = enc

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.notes.blob_store", _OO / "notes" / "blob_store.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.notes.blob_store"] = mod
    spec.loader.exec_module(mod)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod, restore


def _load_transfer():
    spec = importlib.util.spec_from_file_location(
        "blob_transfer_under_test", _OO / "veilid" / "blob_transfer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _env():
    bs, restore = _load_blob_store()
    tx = _load_transfer()
    tx.reset_blob_transfer()
    return bs, tx, restore


def test_end_to_end_transfer():
    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        bs, tx, restore = _env()
        try:
            producer = bs.NotesBlobStore(root=ta, master_key=_KEY_A)
            receiver = bs.NotesBlobStore(root=tb, master_key=_KEY_B)
            payload = os.urandom(200_000)
            h = hashlib.sha256(payload).hexdigest()
            # Seal on the producer in small frames to force a multi-chunk stream.
            producer.seal_stream("a1", [payload], chunk_size=4096)

            assert tx.open_fetch(
                "peerX", "req1", "a1", blob_store=producer, serve_ok=lambda i: True
            ) is True
            ok = tx.receive_blob(
                tx.pull_iter("peerX", "req1"),
                attachment_id="a1", dest_store=receiver, expected_hash=h,
            )
            assert ok is True
            # Re-sealed under the receiver's OWN key, it opens back to the original.
            assert b"".join(receiver.open_stream("a1")) == payload
            assert tx.active_fetches() == 0   # session closed on done
        finally:
            tx.reset_blob_transfer()
            restore()


def test_wrong_hash_never_lands():
    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        bs, tx, restore = _env()
        try:
            producer = bs.NotesBlobStore(root=ta, master_key=_KEY_A)
            receiver = bs.NotesBlobStore(root=tb, master_key=_KEY_B)
            producer.seal_stream("a1", [os.urandom(50_000)], chunk_size=4096)
            tx.open_fetch("p", "r", "a1", blob_store=producer, serve_ok=lambda i: True)
            ok = tx.receive_blob(
                tx.pull_iter("p", "r"),
                attachment_id="a1", dest_store=receiver,
                expected_hash=hashlib.sha256(b"WRONG").hexdigest(),
            )
            assert ok is False
            assert receiver.exists("a1") is False   # discarded, not left behind
        finally:
            tx.reset_blob_transfer()
            restore()


def test_corrupted_wire_caught_by_hash():
    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        bs, tx, restore = _env()
        try:
            producer = bs.NotesBlobStore(root=ta, master_key=_KEY_A)
            receiver = bs.NotesBlobStore(root=tb, master_key=_KEY_B)
            payload = os.urandom(60_000)
            h = hashlib.sha256(payload).hexdigest()
            producer.seal_stream("a1", [payload], chunk_size=4096)
            tx.open_fetch("p", "r", "a1", blob_store=producer, serve_ok=lambda i: True)

            def _corrupt(src):
                first = True
                for c in src:
                    if first and c:
                        b = bytearray(c); b[0] ^= 0x01; c = bytes(b)
                        first = False
                    yield c

            ok = tx.receive_blob(
                _corrupt(tx.pull_iter("p", "r")),
                attachment_id="a1", dest_store=receiver, expected_hash=h,
            )
            assert ok is False                       # plaintext hash mismatch
            assert receiver.exists("a1") is False
        finally:
            tx.reset_blob_transfer()
            restore()


def test_serve_gate_refuses():
    with tempfile.TemporaryDirectory() as ta:
        bs, tx, restore = _env()
        try:
            producer = bs.NotesBlobStore(root=ta, master_key=_KEY_A)
            producer.seal_stream("a1", [b"data"], chunk_size=4096)
            assert tx.open_fetch(
                "p", "r", "a1", blob_store=producer, serve_ok=lambda i: False
            ) is False
            assert tx.active_fetches() == 0
            assert tx.pull_chunk("p", "r", 0) is None
        finally:
            tx.reset_blob_transfer()
            restore()


def test_peer_binding_and_cursor_retry():
    with tempfile.TemporaryDirectory() as ta:
        bs, tx, restore = _env()
        try:
            producer = bs.NotesBlobStore(root=ta, master_key=_KEY_A)
            producer.seal_stream("a1", [os.urandom(12_000)], chunk_size=4096)
            assert tx.open_fetch("peerA", "r", "a1", blob_store=producer) is True

            # Another peer cannot pull this session.
            assert tx.pull_chunk("peerB", "r", 0) is None

            c0 = tx.pull_chunk("peerA", "r", 0)
            assert c0 is not None and c0["done"] is False and c0["cursor"] == 1
            # Retry of the same cursor re-serves the same chunk (lost-ack case).
            c0b = tx.pull_chunk("peerA", "r", 0)
            assert c0b["content_b64"] == c0["content_b64"]
            c1 = tx.pull_chunk("peerA", "r", 1)
            assert c1["content_b64"] != c0["content_b64"]
        finally:
            tx.reset_blob_transfer()
            restore()


def test_session_bound_enforced():
    with tempfile.TemporaryDirectory() as ta:
        bs, tx, restore = _env()
        try:
            producer = bs.NotesBlobStore(root=ta, master_key=_KEY_A)
            producer.seal_stream("a1", [b"x"], chunk_size=4096)
            for i in range(tx.MAX_FETCH_SESSIONS):
                assert tx.open_fetch(
                    "p", "req%d" % i, "a1", blob_store=producer
                ) is True
            assert tx.active_fetches() == tx.MAX_FETCH_SESSIONS
            # One past the bound is refused rather than evicting a live session.
            assert tx.open_fetch("p", "overflow", "a1", blob_store=producer) is False
        finally:
            tx.reset_blob_transfer()
            restore()


def test_absent_blob_refused():
    with tempfile.TemporaryDirectory() as ta:
        bs, tx, restore = _env()
        try:
            producer = bs.NotesBlobStore(root=ta, master_key=_KEY_A)
            assert tx.open_fetch(
                "p", "r", "missing", blob_store=producer, serve_ok=lambda i: True
            ) is False
        finally:
            tx.reset_blob_transfer()
            restore()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'OK' if failures == 0 else 'FAILED'} - {failures} failure(s)")
    sys.exit(1 if failures else 0)
