#!/usr/bin/env python3
"""The download digest pin: the one check here that is not trust-on-first-use.

The SSRF defence on this path proves WHERE the bytes came from -- a routable
public host, re-validated and IP-pinned across every redirect hop. It says
nothing about WHAT they are. A compromised mirror, a hostile CDN edge, or a
transfer corrupted in flight all satisfy it perfectly.

``expected_sha256`` is the answer, and its value comes entirely from where the
caller got it: out of band, from the model card, NOT from the server currently
serving the file. That is what makes it different in kind from every other
check on this path. A pin taken from the same host that serves the bytes is
just trust-on-first-use with extra steps; a pin taken from the model card is
a claim the serving host cannot satisfy if it substitutes the weights.

Contracts pinned:

  * C1 a digest mismatch discards the download and NO loadable .gguf is ever
    created. The hash is taken on the .part file, before the rename, so the
    refusal cannot leave a substituted model sitting in a model directory
    where the load seam would later find it.
  * C2 a matching pin promotes the file and enrols it in the provenance
    manifest, so the load seam can verify the same bytes later.
  * C3 (control) a caller that supplies no pin is not broken. This is what
    makes the pin adoptable: it is an offer, not a new mandatory field.
  * C4 the digest primitive is a true streaming sha256, chunk-independent.
    It is deliberately a local, stdlib-only copy rather than a call into the
    provenance module: the check that refuses a substituted file must not be
    reachable only through an optional import, because a check that can be
    skipped when a module is missing is not a check.

The SSRF guard itself is stubbed out here rather than re-tested; it has its
own contracts, and neutering it in this window keeps a probe on either surface
from reddening the other.

Local-only. Runs under pytest or the __main__ runner.
"""

import hashlib
import sys
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_BODY = b"GGUF" + bytes(range(256)) * 40
_NAME = "pinned-model-Q4_K_M.gguf"
_URL = f"https://models.example.com/{_NAME}"


class _FakeResponse:
    """The subset of the response object download_model touches."""

    def __init__(self, body: bytes):
        self._body = body
        self._offset = 0
        self.headers = {"Content-Length": str(len(body))}

    def read(self, size: int) -> bytes:
        block = self._body[self._offset:self._offset + size]
        self._offset += len(block)
        return block

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _open_front_door():
    """The front door, answering open: the downloader's own checks are what is read."""
    egress = types.ModuleType("opti_oignon.egress")

    class EgressRefused(RuntimeError):
        pass

    egress.EgressRefused = EgressRefused
    egress.require_web = lambda label: None
    egress.web_refusal = lambda label="Web request": None
    return egress


def _public_addresses():
    """The page fetch's address check, finding nothing: the downloader's own rule decides."""
    web_search = types.ModuleType("opti_oignon.web_search")
    web_search.address_refusal = lambda address, hop=0: None
    return web_search


def _load(*, provenance=None, body: bytes = _BODY):
    """Load model_manager in the shared isolation window; returns (module, restore).

    The network seam and the SSRF pre-check are replaced: this suite is about
    the digest, and re-testing the SSRF guard here would only make a probe on
    one surface redden the other. The front door answers open. The provenance
    module is the stand-in given, or it is proven unreachable.
    """
    seeded = {"opti_oignon.egress": _open_front_door(), "opti_oignon.web_search": _public_addresses()}
    blocked = ()
    if provenance is not None:
        seeded["opti_oignon.model_provenance"] = provenance
    else:
        blocked = ("opti_oignon.model_provenance",)
    loaded, restore = isolate(
        targets={"opti_oignon.model_manager": source("model_manager.py")},
        seeded=seeded,
        blocked=blocked,
    )
    mod = loaded["opti_oignon.model_manager"]

    mod.urlopen_ssrf_safe = lambda url, **kw: _FakeResponse(body)
    mod.ModelManager._validate_download_url = staticmethod(lambda url: None)

    return mod, restore


def _provenance_stub():
    """A stand-in provenance module recording what gets enrolled."""
    stub = types.ModuleType("opti_oignon.model_provenance")
    stub.recorded = []

    def record_model(path, **kwargs):
        stub.recorded.append(Path(path))
        return {"model": Path(path).name, "sha256": "x", "scheme": "hmac-sha512"}

    stub.record_model = record_model
    return stub


# ---------------------------------------------------------------------------
# C1: a mismatch never becomes a loadable file
# ---------------------------------------------------------------------------


def test_c1_digest_mismatch_leaves_no_loadable_model(tmp_path):
    mod, restore = _load()
    try:
        manager = mod.ModelManager(default_dir=str(tmp_path))

        # A well-formed digest that simply is not this file's: the substituting
        # mirror. The pin came from the model card, so the mirror cannot
        # satisfy it.
        wrong = hashlib.sha256(b"the weights the card promised").hexdigest()
        result = manager.download_model(_URL, expected_sha256=wrong)

        assert result["status"] == "error"
        assert "sha256" in result["message"].lower()

        # The hash was taken on the .part file, so the refusal cannot leave a
        # substituted model sitting where the load seam would later find it.
        assert not (tmp_path / _NAME).exists()
        assert not (tmp_path / f"{Path(_NAME).stem}.gguf.part").exists()
        assert list(tmp_path.iterdir()) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# C2: a matching pin promotes and enrols
# ---------------------------------------------------------------------------


def test_c2_matching_pin_promotes_and_enrols(tmp_path):
    prov = _provenance_stub()
    mod, restore = _load(provenance=prov)
    try:
        manager = mod.ModelManager(default_dir=str(tmp_path))

        source = tmp_path / "source.bin"
        source.write_bytes(_BODY)
        expected = mod._sha256_file(source)
        source.unlink()

        result = manager.download_model(_URL, expected_sha256=expected)

        assert result["status"] == "completed"
        assert result["sha256"] == expected
        assert (tmp_path / _NAME).exists()

        # Enrolled, so the load seam can verify these same bytes later. Without
        # this the model would download cleanly and then be refused at load
        # time for being unpinned -- fail-secure, but useless.
        assert result["provenance"]["recorded"] is True
        assert prov.recorded == [tmp_path / _NAME]
    finally:
        restore()


# ---------------------------------------------------------------------------
# C3: the pin is an offer, not a new mandatory field
# ---------------------------------------------------------------------------


def test_c3_absent_pin_does_not_break_the_caller(tmp_path):
    mod, restore = _load()
    try:
        manager = mod.ModelManager(default_dir=str(tmp_path))

        result = manager.download_model(_URL)

        assert result["status"] == "completed"
        assert (tmp_path / _NAME).exists()
        assert result["sha256"] == mod._sha256_file(tmp_path / _NAME)
    finally:
        restore()


# ---------------------------------------------------------------------------
# C4: the primitive the refusal rests on
# ---------------------------------------------------------------------------


def test_c4_digest_primitive_is_a_true_streaming_sha256(tmp_path):
    mod, restore = _load()
    try:
        path = tmp_path / "blob.bin"
        path.write_bytes(_BODY)
        expected = hashlib.sha256(_BODY).hexdigest()

        # Chunk size must not change the answer: the pin published on a model
        # card is a plain sha256 of the file, and anything else would only
        # ever agree with itself.
        assert mod._sha256_file(path, chunk_size=7) == expected
        assert mod._sha256_file(path, chunk_size=1 << 20) == expected
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _main() -> int:
    import tempfile

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(Path(tmp))
                print(f"PASS {fn.__name__}")
            except Exception:
                failed += 1
                print(f"FAIL {fn.__name__}")
                traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
