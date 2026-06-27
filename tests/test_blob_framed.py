#!/usr/bin/env python3
"""Tests for the framed AEAD blob format (SYN-01, vault on-demand transfer).

``NotesBlobStore.seal_stream`` / ``open_stream`` seal and open a blob in
CONSTANT memory as a sequence of independently-sealed frames, so a multi-GB
file never sits whole in RAM. ``blob_store.py`` is loaded in isolation with a
stub ``opti_oignon.encryption`` backed by REAL AES-256-GCM (the
``version||nonce||ct||tag`` layout the module expects), and a deterministic
master key is injected.

Proven here:

  * a stream of any piece sizes round-trips byte-exactly through the framed
    container -- multi-chunk, exact-multiple, single-chunk, empty, and a large
    payload fed from a lazy generator (streamed both ways);
  * the framed magic is detected and a legacy single-seal blob still opens
    through ``open_stream``;
  * every tamper is caught: a flipped byte (GCM tag), reordered frames,
    a dropped final frame (truncation), a duplicated frame, and opening under
    the wrong attachment id (wrong subkey).

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
_KEY = bytes(range(32))  # deterministic 32-byte AES-256 master key


def _load():
    keys = ("opti_oignon", "opti_oignon.notes", "opti_oignon.encryption",
            "opti_oignon.notes.blob_store")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    notes_pkg = types.ModuleType("opti_oignon.notes")
    notes_pkg.__path__ = []
    sys.modules["opti_oignon.notes"] = notes_pkg

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    enc = types.ModuleType("opti_oignon.encryption")

    def encrypt_bytes(key, plaintext):
        nonce = os.urandom(12)
        ct_tag = AESGCM(key).encrypt(nonce, plaintext, None)
        return bytes([1]) + nonce + ct_tag

    def decrypt_bytes(key, blob):
        return AESGCM(key).decrypt(blob[1:13], blob[13:], None)

    enc.encrypt_bytes = encrypt_bytes
    enc.decrypt_bytes = decrypt_bytes
    enc.get_encryption_key = lambda: None  # production refuses; tests inject
    sys.modules["opti_oignon.encryption"] = enc

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.notes.blob_store", _OO / "notes" / "blob_store.py",
    )
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


def _store(mod, td):
    return mod.NotesBlobStore(root=td, master_key=_KEY)


def _read_frames(mod, path):
    """Parse a framed file into (header_bytes, [frame_bytes, ...])."""
    raw = Path(path).read_bytes()
    hsize = mod._FRAMED_HEADER.size
    header, body = raw[:hsize], raw[hsize:]
    frames = []
    i = 0
    while i < len(body):
        (flen,) = mod._FRAME_LEN.unpack(body[i:i + 4])
        i += 4
        frames.append(body[i:i + flen])
        i += flen
    return header, frames


def _write_frames(mod, path, header, frames):
    out = bytearray(header)
    for fr in frames:
        out += mod._FRAME_LEN.pack(len(fr))
        out += fr
    Path(path).write_bytes(bytes(out))


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def test_stream_roundtrip_variants():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = _store(mod, td)
            cases = {
                "multichunk": os.urandom(200),       # 64,64,64,8 at chunk 64
                "exact_multiple": os.urandom(128),    # 64,64
                "single": os.urandom(10),
                "empty": b"",
            }
            for name, payload in cases.items():
                store.seal_stream(name, [payload], chunk_size=64)
                out = b"".join(store.open_stream(name))
                assert out == payload, name
        finally:
            restore()


def test_stream_roundtrip_large_from_generator():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = _store(mod, td)
            # 1 MiB fed as many small lazy pieces, framed at 1 KiB -> ~1024
            # frames, streamed both ways; assert by hash, never the whole bytes.
            piece = os.urandom(4096)
            n = 256  # 1 MiB total
            h = hashlib.sha256()
            for _ in range(n):
                h.update(piece)
            store.seal_stream(
                "big", (piece for _ in range(n)), chunk_size=1024
            )
            out = hashlib.sha256()
            total = 0
            for chunk in store.open_stream("big"):
                out.update(chunk)
                total += len(chunk)
            assert total == n * len(piece)
            assert out.hexdigest() == h.hexdigest()
        finally:
            restore()


def test_framed_detected_and_legacy_still_opens():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = _store(mod, td)
            store.seal_stream("framed1", [b"hello world"], chunk_size=4)
            assert store.is_framed("framed1") is True

            # Legacy single-seal path is untouched and has no framed magic.
            store.seal("legacy1", b"legacy payload")
            assert store.is_framed("legacy1") is False
            # open_stream reads the legacy blob uniformly (one chunk).
            assert b"".join(store.open_stream("legacy1")) == b"legacy payload"
            assert store.open("legacy1") == b"legacy payload"
        finally:
            restore()


def test_tamper_flip_byte_is_caught():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = _store(mod, td)
            store.seal_stream("a", [os.urandom(200)], chunk_size=64)
            path = store._blob_path("a")
            header, frames = _read_frames(mod, path)
            fr = bytearray(frames[1])
            fr[-1] ^= 0x01  # flip a byte inside the second frame
            frames[1] = bytes(fr)
            _write_frames(mod, path, header, frames)
            assert _raises(lambda: list(store.open_stream("a")))  # GCM tag
        finally:
            restore()


def test_tamper_reorder_is_caught():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = _store(mod, td)
            store.seal_stream("a", [os.urandom(200)], chunk_size=64)
            path = store._blob_path("a")
            header, frames = _read_frames(mod, path)
            frames[0], frames[1] = frames[1], frames[0]  # swap order
            _write_frames(mod, path, header, frames)
            assert _raises(lambda: list(store.open_stream("a")))  # out-of-order
        finally:
            restore()


def test_tamper_truncate_final_is_caught():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = _store(mod, td)
            store.seal_stream("a", [os.urandom(200)], chunk_size=64)
            path = store._blob_path("a")
            header, frames = _read_frames(mod, path)
            _write_frames(mod, path, header, frames[:-1])  # drop final frame
            assert _raises(lambda: list(store.open_stream("a")))  # no final
        finally:
            restore()


def test_tamper_duplicate_is_caught():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = _store(mod, td)
            store.seal_stream("a", [os.urandom(200)], chunk_size=64)
            path = store._blob_path("a")
            header, frames = _read_frames(mod, path)
            frames = [frames[0]] + frames  # duplicate first frame
            _write_frames(mod, path, header, frames)
            assert _raises(lambda: list(store.open_stream("a")))  # index gap
        finally:
            restore()


def test_wrong_attachment_id_fails():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = _store(mod, td)
            store.seal_stream("a1", [os.urandom(200)], chunk_size=64)
            # The blob is bound to its id by the per-attachment subkey; opening
            # the same bytes under another id fails the very first frame's tag.
            os.replace(store._blob_path("a1"), store._blob_path("a2"))
            assert _raises(lambda: list(store.open_stream("a2")))
        finally:
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
