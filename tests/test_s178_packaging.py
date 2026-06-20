#!/usr/bin/env python3
"""Tests for S178 Goal 2 -- the veilid-server packaging script.

Covers scripts/fetch_veilid_server.py:

- No import-time network: importing the module touches neither urlopen nor a
  socket; the network lives only inside download(), called from stage()/main().
- Platform detection maps the supported hosts and rejects the rest.
- Checksum verification: a match passes; a mismatch raises ChecksumMismatch; an
  empty expectation raises ChecksumUnset.
- stage() is fail-secure: with a fake downloader, a good checksum stages the
  binary with an exec bit; a bad checksum stages nothing and leaves no temp file;
  an unset checksum refuses before any download is attempted.
- download() refuses a non-HTTPS URL without reaching the network.
- plan() / --print-plan / --verify are network-free and behave as documented.
- Signature verification is layered: unconfigured is a skip, and a required but
  unmet signature raises.

Loaded via spec_from_file_location, registered in sys.modules before exec (the
frozen dataclass is processed under Python 3.12+, which reads
sys.modules[__module__]). The only opti_oignon import in the script is a guarded,
lazy DATA_DIR lookup, so the script is fully isolatable.
"""

import hashlib
import importlib.util
import socket
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "fetch_veilid_server.py"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec (3.12+ dataclass processing)
    spec.loader.exec_module(mod)
    return mod


fvs = _load("s178_fetch_veilid_server")

_CONTENT = b"veilid-server fake payload bytes"
_SHA = hashlib.sha256(_CONTENT).hexdigest()


def _writer(content: bytes):
    """A fake downloader that writes fixed bytes instead of hitting the network."""

    def _dl(url: str, dest) -> None:
        Path(dest).write_bytes(content)

    return _dl


def _recording_downloader():
    calls: list[str] = []

    def _dl(url: str, dest) -> None:  # pragma: no cover - asserts it is never called
        calls.append(url)

    return _dl, calls


# Sentinels


class TestSentinels:
    def test_flags(self):
        assert fvs.checkpoint_before_apply is True
        assert fvs.FEATURE_AVAILABLE is True

    def test_version_and_https_base(self):
        assert fvs.VEILID_SERVER_VERSION
        assert fvs.RELEASE_BASE_URL.startswith("https://")


# No import-time network


class TestNoImportTimeNetwork:
    def test_import_makes_no_network(self, monkeypatch):
        net: list[str] = []
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: net.append("urlopen"))
        monkeypatch.setattr(
            socket.socket, "connect", lambda self, *a, **k: net.append("connect")
        )
        mod = _load("s178_fetch_veilid_server_reimport")
        assert net == []
        # The module is usable without ever having opened a connection.
        assert callable(mod.stage)


# Platform detection


class TestPlatformDetection:
    @pytest.mark.parametrize(
        "system,machine,expected",
        [
            ("Linux", "x86_64", "linux-x86_64"),
            ("Linux", "amd64", "linux-x86_64"),
            ("Linux", "aarch64", "linux-aarch64"),
            ("Darwin", "arm64", "macos-aarch64"),
            ("Darwin", "x86_64", "macos-x86_64"),
        ],
    )
    def test_supported(self, monkeypatch, system, machine, expected):
        monkeypatch.setattr(fvs.platform, "system", lambda: system)
        monkeypatch.setattr(fvs.platform, "machine", lambda: machine)
        assert fvs.detect_platform() == expected

    def test_unsupported_os(self, monkeypatch):
        monkeypatch.setattr(fvs.platform, "system", lambda: "Windows")
        monkeypatch.setattr(fvs.platform, "machine", lambda: "x86_64")
        with pytest.raises(fvs.UnsupportedPlatform):
            fvs.detect_platform()

    def test_artifact_for_unknown_key(self):
        with pytest.raises(fvs.UnsupportedPlatform):
            fvs.artifact_for("plan9-pdp11")

    def test_artifact_url_is_https(self):
        art = fvs.artifact_for("linux-x86_64")
        assert art.url.startswith("https://")
        assert art.filename == fvs.BINARY_NAME


# Checksum verification


class TestChecksum:
    def test_sha256_file(self, tmp_path):
        p = tmp_path / "blob"
        p.write_bytes(_CONTENT)
        assert fvs.sha256_file(p) == _SHA

    def test_verify_match(self, tmp_path):
        p = tmp_path / "blob"
        p.write_bytes(_CONTENT)
        assert fvs.verify_checksum(p, _SHA) is True

    def test_verify_match_case_insensitive(self, tmp_path):
        p = tmp_path / "blob"
        p.write_bytes(_CONTENT)
        assert fvs.verify_checksum(p, _SHA.upper()) is True

    def test_verify_mismatch(self, tmp_path):
        p = tmp_path / "blob"
        p.write_bytes(_CONTENT)
        with pytest.raises(fvs.ChecksumMismatch):
            fvs.verify_checksum(p, "00" * 32)

    def test_verify_unset(self, tmp_path):
        p = tmp_path / "blob"
        p.write_bytes(_CONTENT)
        with pytest.raises(fvs.ChecksumUnset):
            fvs.verify_checksum(p, "")


# stage(): fail-secure


class TestStage:
    def test_stage_good_checksum(self, tmp_path):
        target = fvs.stage(
            "linux-x86_64", dest=tmp_path, expected_sha=_SHA, downloader=_writer(_CONTENT)
        )
        assert target.exists()
        assert target.read_bytes() == _CONTENT
        assert target.name == "veilid-server"

    def test_stage_sets_exec_bit(self, tmp_path):
        import os
        import stat as _stat

        target = fvs.stage(
            "linux-x86_64", dest=tmp_path, expected_sha=_SHA, downloader=_writer(_CONTENT)
        )
        assert os.stat(target).st_mode & _stat.S_IXUSR

    def test_stage_bad_checksum_stages_nothing(self, tmp_path):
        with pytest.raises(fvs.ChecksumMismatch):
            fvs.stage(
                "linux-x86_64",
                dest=tmp_path,
                expected_sha="11" * 32,
                downloader=_writer(_CONTENT),
            )
        # No staged binary and no leftover temp file.
        assert not (tmp_path / "veilid-server").exists()
        assert list(tmp_path.glob(".veilid-server-*")) == []

    def test_stage_unset_checksum_does_not_download(self, tmp_path):
        dl, calls = _recording_downloader()
        with pytest.raises(fvs.ChecksumUnset):
            fvs.stage("linux-x86_64", dest=tmp_path, expected_sha="", downloader=dl)
        assert calls == []
        assert not (tmp_path / "veilid-server").exists()

    def test_default_pins_are_unset_so_real_stage_refuses(self, tmp_path):
        # The shipped artifacts have empty pinned checksums on purpose.
        dl, calls = _recording_downloader()
        with pytest.raises(fvs.ChecksumUnset):
            fvs.stage("linux-x86_64", dest=tmp_path, downloader=dl)
        assert calls == []


# download(): HTTPS-only, no network on refusal


class TestDownloadGuard:
    def test_refuses_non_https(self, tmp_path, monkeypatch):
        net: list[str] = []
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: net.append("urlopen"))
        with pytest.raises(fvs.InsecureURL):
            fvs.download("http://example.com/veilid-server", tmp_path / "out")
        assert net == []


# plan() and the CLI (network-free paths)


class TestPlanAndCli:
    def test_plan_shape(self):
        p = fvs.plan("linux-x86_64", dest=Path("/tmp/x"))
        for key in ("platform", "version", "url", "expected_sha256", "target"):
            assert key in p
        assert p["platform"] == "linux-x86_64"
        assert "unset" in p["expected_sha256"]

    def test_print_plan_no_network(self, monkeypatch, capsys):
        net: list[str] = []
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: net.append("urlopen"))
        rc = fvs.main(["--print-plan", "--platform", "linux-x86_64", "--dest", "/tmp/x"])
        assert rc == 0
        assert net == []
        assert "platform: linux-x86_64" in capsys.readouterr().out

    def test_verify_cli_match(self, tmp_path):
        p = tmp_path / "veilid-server"
        p.write_bytes(_CONTENT)
        assert fvs.main(["--verify", str(p), "--sha", _SHA]) == 0

    def test_verify_cli_mismatch_returns_1(self, tmp_path):
        p = tmp_path / "veilid-server"
        p.write_bytes(_CONTENT)
        assert fvs.main(["--verify", str(p), "--sha", "22" * 32]) == 1


# Signature: layered, optional


class TestSignature:
    def test_unconfigured_is_a_skip(self, tmp_path):
        p = tmp_path / "veilid-server"
        p.write_bytes(_CONTENT)
        assert fvs.verify_signature(p) == "skipped-not-configured"

    def test_required_but_unmet_raises(self, tmp_path):
        p = tmp_path / "veilid-server"
        p.write_bytes(_CONTENT)
        with pytest.raises(fvs.SignatureError):
            fvs.verify_signature(p, require=True)

    def test_required_signature_aborts_stage(self, tmp_path):
        # No pinned key -> signature is unmet -> a required signature refuses,
        # and nothing is staged.
        with pytest.raises(fvs.SignatureError):
            fvs.stage(
                "linux-x86_64",
                dest=tmp_path,
                expected_sha=_SHA,
                downloader=_writer(_CONTENT),
                require_signature=True,
            )
        assert not (tmp_path / "veilid-server").exists()
        assert list(tmp_path.glob(".veilid-server-*")) == []
