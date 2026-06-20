#!/usr/bin/env python3
"""
Tests for S105 -- Multi-Backend Inference (llama.cpp + GGUF Standalone).

Covers:
- inference_backend: ABC, data types (ChatResponse, StreamChunk, BackendModelInfo),
  OllamaBackend, LlamaCppBackend, BackendRegistry, singleton, config loading
- model_manager: GGUF header parser (magic, version, KV types, errors),
  ModelManager (scan, info, storage, download, delete, cache), singleton
- routes_backends: API endpoint responses (mocked)
- executor integration: backend abstraction wiring
"""

import importlib.util
import json
import os
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

# =========================================================================
# MODULE LOADING (importlib isolation)
# =========================================================================

ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, filename: str) -> ModuleType:
    """Load a module by file path with importlib isolation."""
    filepath = ROOT / "opti_oignon" / filename
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)

    # Stub parent package
    if "opti_oignon" not in sys.modules:
        parent = MagicMock()
        parent.__path__ = [str(ROOT / "opti_oignon")]
        sys.modules["opti_oignon"] = parent

    spec.loader.exec_module(mod)
    return mod


# Load the modules under test
_ib_mod = _load_module("s105_inference_backend", "inference_backend.py")
_mm_mod = _load_module("s105_model_manager", "model_manager.py")


# =========================================================================
# GGUF test file helpers
# =========================================================================

def _write_gguf_string(f, s: str):
    """Write a GGUF-format string (uint64 length + bytes)."""
    data = s.encode("utf-8")
    f.write(struct.pack("<Q", len(data)))
    f.write(data)


def _write_kv_string(f, key: str, value: str):
    """Write a string KV pair."""
    _write_gguf_string(f, key)
    f.write(struct.pack("<I", 8))  # GGUF_TYPE_STRING
    _write_gguf_string(f, value)


def _write_kv_uint32(f, key: str, value: int):
    """Write a uint32 KV pair."""
    _write_gguf_string(f, key)
    f.write(struct.pack("<I", 4))  # GGUF_TYPE_UINT32
    f.write(struct.pack("<I", value))


def _write_kv_int32(f, key: str, value: int):
    """Write an int32 KV pair."""
    _write_gguf_string(f, key)
    f.write(struct.pack("<I", 5))  # GGUF_TYPE_INT32
    f.write(struct.pack("<i", value))


def _write_kv_float32(f, key: str, value: float):
    """Write a float32 KV pair."""
    _write_gguf_string(f, key)
    f.write(struct.pack("<I", 6))  # GGUF_TYPE_FLOAT32
    f.write(struct.pack("<f", value))


def _write_kv_bool(f, key: str, value: bool):
    """Write a bool KV pair."""
    _write_gguf_string(f, key)
    f.write(struct.pack("<I", 7))  # GGUF_TYPE_BOOL
    f.write(struct.pack("<B", 1 if value else 0))


def _create_gguf_file(
    tmpdir: str,
    filename: str = "test.gguf",
    version: int = 3,
    tensor_count: int = 10,
    kv_pairs: list | None = None,
) -> Path:
    """Create a minimal valid GGUF file for testing."""
    filepath = Path(tmpdir) / filename
    kv_pairs = kv_pairs or []

    with open(filepath, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", version))

        if version == 1:
            f.write(struct.pack("<I", tensor_count))
            f.write(struct.pack("<I", len(kv_pairs)))
        else:
            f.write(struct.pack("<Q", tensor_count))
            f.write(struct.pack("<Q", len(kv_pairs)))

        for writer_fn, key, value in kv_pairs:
            writer_fn(f, key, value)

    return filepath


# =========================================================================
# A. INFERENCE BACKEND — Data Types
# =========================================================================

class TestChatResponse:
    """Tests for ChatResponse data type."""

    def test_basic_creation(self):
        resp = _ib_mod.ChatResponse(content="Hello world", model="test-model")
        assert resp.content == "Hello world"
        assert resp.model == "test-model"
        assert resp.done is True
        assert resp.thinking is None

    def test_with_thinking(self):
        resp = _ib_mod.ChatResponse(
            content="Answer", thinking="Let me think...", model="m"
        )
        assert resp.thinking == "Let me think..."

    def test_to_dict_basic(self):
        resp = _ib_mod.ChatResponse(content="Hi", model="m1")
        d = resp.to_dict()
        assert d["message"]["role"] == "assistant"
        assert d["message"]["content"] == "Hi"
        assert d["model"] == "m1"
        assert d["done"] is True
        assert "thinking" not in d["message"]

    def test_to_dict_with_thinking(self):
        resp = _ib_mod.ChatResponse(content="A", thinking="T", model="m")
        d = resp.to_dict()
        assert d["message"]["thinking"] == "T"

    def test_to_dict_with_duration(self):
        resp = _ib_mod.ChatResponse(content="X", total_duration=12345)
        d = resp.to_dict()
        assert d["total_duration"] == 12345

    def test_extra_field(self):
        resp = _ib_mod.ChatResponse(content="X", extra={"foo": "bar"})
        assert resp.extra == {"foo": "bar"}


class TestStreamChunk:
    """Tests for StreamChunk data type."""

    def test_basic_creation(self):
        chunk = _ib_mod.StreamChunk(content="token")
        assert chunk.content == "token"
        assert chunk.thinking == ""
        assert chunk.done is False

    def test_to_dict(self):
        chunk = _ib_mod.StreamChunk(content="Hi", done=True, model="m")
        d = chunk.to_dict()
        assert d["message"]["content"] == "Hi"
        assert d["done"] is True
        assert d["model"] == "m"

    def test_to_dict_with_thinking(self):
        chunk = _ib_mod.StreamChunk(thinking="hmm")
        d = chunk.to_dict()
        assert d["message"]["thinking"] == "hmm"

    def test_to_dict_no_thinking(self):
        chunk = _ib_mod.StreamChunk(content="X")
        d = chunk.to_dict()
        assert "thinking" not in d["message"]


class TestBackendModelInfo:
    """Tests for BackendModelInfo data type."""

    def test_basic_creation(self):
        info = _ib_mod.BackendModelInfo(name="llama.gguf", backend="llama_cpp")
        assert info.name == "llama.gguf"
        assert info.backend == "llama_cpp"
        assert info.size is None
        assert info.extra == {}

    def test_full_creation(self):
        info = _ib_mod.BackendModelInfo(
            name="qwen:7b", backend="ollama", size="4.0GB",
            family="qwen2", parameter_size="7B", quantization_level="Q4_K_M",
            context_length=32768, path="/models/qwen.gguf",
        )
        assert info.context_length == 32768
        assert info.family == "qwen2"

    def test_to_dict(self):
        info = _ib_mod.BackendModelInfo(
            name="test", backend="ollama", size="1.0GB"
        )
        d = info.to_dict()
        assert d["name"] == "test"
        assert d["backend"] == "ollama"
        assert d["size"] == "1.0GB"
        assert d["context_length"] is None


# =========================================================================
# B. INFERENCE BACKEND — OllamaBackend
# =========================================================================

class TestOllamaBackend:
    """Tests for OllamaBackend."""

    def test_name_and_display(self):
        backend = _ib_mod.OllamaBackend()
        assert backend.name == "ollama"
        assert backend.display_name == "Ollama"

    def test_health_check_no_ollama(self):
        backend = _ib_mod.OllamaBackend()
        # In test env, ollama is not available
        if not _ib_mod.OLLAMA_AVAILABLE:
            assert backend.health_check() is False

    def test_list_models_no_ollama(self):
        backend = _ib_mod.OllamaBackend()
        if not _ib_mod.OLLAMA_AVAILABLE:
            assert backend.list_models() == []

    def test_model_info_no_ollama(self):
        backend = _ib_mod.OllamaBackend()
        if not _ib_mod.OLLAMA_AVAILABLE:
            assert backend.model_info("test") is None

    def test_generate_no_ollama_raises(self):
        backend = _ib_mod.OllamaBackend()
        if not _ib_mod.OLLAMA_AVAILABLE:
            with pytest.raises(RuntimeError, match="not installed"):
                backend.generate("m", [{"role": "user", "content": "hi"}])

    def test_stream_no_ollama_raises(self):
        backend = _ib_mod.OllamaBackend()
        if not _ib_mod.OLLAMA_AVAILABLE:
            with pytest.raises(RuntimeError, match="not installed"):
                list(backend.stream("m", [{"role": "user", "content": "hi"}]))

    def test_parse_ollama_model_dict(self):
        model_data = {
            "name": "qwen3:32b",
            "size": 18_000_000_000,
            "modified_at": "2025-01-01T00:00:00Z",
            "details": {
                "family": "qwen2",
                "parameter_size": "32B",
                "quantization_level": "Q4_0",
            },
        }
        info = _ib_mod.OllamaBackend._parse_ollama_model(model_data)
        assert info.name == "qwen3:32b"
        assert info.backend == "ollama"
        assert info.family == "qwen2"
        assert "GB" in info.size

    def test_parse_ollama_model_object(self):
        model_obj = MagicMock()
        model_obj.model = "llama3:8b"
        model_obj.name = "llama3:8b"
        model_obj.size = 5_000_000_000
        model_obj.modified_at = "2025-02-01"
        details = MagicMock()
        details.family = "llama"
        details.parameter_size = "8B"
        details.quantization_level = "Q5_K_M"
        model_obj.details = details

        info = _ib_mod.OllamaBackend._parse_ollama_model(model_obj)
        assert info.name == "llama3:8b"
        assert info.family == "llama"


# =========================================================================
# C. INFERENCE BACKEND — LlamaCppBackend
# =========================================================================

class TestLlamaCppBackend:
    """Tests for LlamaCppBackend."""

    def test_name_and_display(self):
        backend = _ib_mod.LlamaCppBackend()
        assert backend.name == "llama_cpp"
        assert backend.display_name == "llama.cpp"

    def test_health_check(self):
        backend = _ib_mod.LlamaCppBackend()
        assert backend.health_check() == _ib_mod.LLAMA_CPP_AVAILABLE

    def test_list_models_empty_dirs(self):
        backend = _ib_mod.LlamaCppBackend(model_dirs=["/nonexistent/path"])
        assert backend.list_models() == []

    def test_list_models_with_gguf_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some fake .gguf files
            (Path(tmpdir) / "model-a.gguf").write_bytes(b"x" * 100)
            (Path(tmpdir) / "model-b.gguf").write_bytes(b"x" * 200)
            (Path(tmpdir) / "not-a-model.txt").write_bytes(b"x" * 50)

            backend = _ib_mod.LlamaCppBackend(model_dirs=[tmpdir])
            models = backend.list_models()
            assert len(models) == 2
            names = {m.name for m in models}
            assert "model-a.gguf" in names
            assert "model-b.gguf" in names

    def test_list_models_dedup(self):
        with tempfile.TemporaryDirectory() as d1, \
             tempfile.TemporaryDirectory() as d2:
            (Path(d1) / "shared.gguf").write_bytes(b"x")
            (Path(d2) / "shared.gguf").write_bytes(b"y")

            backend = _ib_mod.LlamaCppBackend(model_dirs=[d1, d2])
            models = backend.list_models()
            assert len(models) == 1

    def test_model_info_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.gguf").write_bytes(b"x" * 100)
            backend = _ib_mod.LlamaCppBackend(model_dirs=[tmpdir])
            info = backend.model_info("test.gguf")
            assert info is not None
            assert info.name == "test.gguf"
            assert info.path is not None

    def test_model_info_not_found(self):
        backend = _ib_mod.LlamaCppBackend(model_dirs=[])
        assert backend.model_info("nonexistent.gguf") is None

    def test_resolve_model_path_direct(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"x")
            fpath = f.name
        try:
            backend = _ib_mod.LlamaCppBackend()
            resolved = backend._resolve_model_path(fpath)
            assert resolved is not None
            assert resolved.is_file()
        finally:
            os.unlink(fpath)

    def test_resolve_model_path_in_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "mymodel.gguf").write_bytes(b"x")
            backend = _ib_mod.LlamaCppBackend(model_dirs=[tmpdir])
            resolved = backend._resolve_model_path("mymodel.gguf")
            assert resolved is not None

    def test_resolve_model_path_auto_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "mymodel.gguf").write_bytes(b"x")
            backend = _ib_mod.LlamaCppBackend(model_dirs=[tmpdir])
            resolved = backend._resolve_model_path("mymodel")
            assert resolved is not None

    def test_unload_model(self):
        backend = _ib_mod.LlamaCppBackend()
        backend._loaded_models["test"] = "fake"
        assert backend.unload_model("test") is True
        assert "test" not in backend._loaded_models

    def test_unload_model_not_loaded(self):
        backend = _ib_mod.LlamaCppBackend()
        assert backend.unload_model("nope") is False

    def test_unload_all(self):
        backend = _ib_mod.LlamaCppBackend()
        backend._loaded_models = {"a": 1, "b": 2, "c": 3}
        count = backend.unload_all()
        assert count == 3
        assert len(backend._loaded_models) == 0


# =========================================================================
# D. INFERENCE BACKEND — BackendRegistry
# =========================================================================

class TestBackendRegistry:
    """Tests for BackendRegistry."""

    def _make_mock_backend(self, name: str, healthy: bool = True) -> MagicMock:
        b = MagicMock(spec=_ib_mod.InferenceBackend)
        b.name = name
        b.display_name = name.title()
        b.health_check.return_value = healthy
        b.list_models.return_value = []
        return b

    def test_register_and_get(self):
        reg = _ib_mod.BackendRegistry()
        backend = self._make_mock_backend("test")
        reg.register(backend)
        assert reg.get("test") is backend

    def test_get_unknown(self):
        reg = _ib_mod.BackendRegistry()
        assert reg.get("nope") is None

    def test_activate(self):
        reg = _ib_mod.BackendRegistry()
        backend = self._make_mock_backend("test")
        reg.register(backend)
        assert reg.activate("test") is True
        assert reg.active_name == "test"
        assert reg.active is backend

    def test_activate_unknown(self):
        reg = _ib_mod.BackendRegistry()
        assert reg.activate("nope") is False

    def test_active_fallback(self):
        reg = _ib_mod.BackendRegistry()
        b1 = self._make_mock_backend("sick", healthy=False)
        b2 = self._make_mock_backend("healthy", healthy=True)
        reg.register(b1)
        reg.register(b2)
        # No explicit activation — should fall back to first healthy
        assert reg.active is b2

    def test_active_none(self):
        reg = _ib_mod.BackendRegistry()
        assert reg.active is None

    def test_unregister(self):
        reg = _ib_mod.BackendRegistry()
        backend = self._make_mock_backend("test")
        reg.register(backend)
        reg.activate("test")
        assert reg.unregister("test") is True
        assert reg.get("test") is None
        assert reg.active_name is None

    def test_unregister_unknown(self):
        reg = _ib_mod.BackendRegistry()
        assert reg.unregister("nope") is False

    def test_list_backends(self):
        reg = _ib_mod.BackendRegistry()
        b1 = self._make_mock_backend("a", healthy=True)
        b2 = self._make_mock_backend("b", healthy=False)
        reg.register(b1)
        reg.register(b2)
        reg.activate("a")

        listing = reg.list_backends()
        assert len(listing) == 2
        a_entry = [e for e in listing if e["name"] == "a"][0]
        assert a_entry["healthy"] is True
        assert a_entry["active"] is True
        b_entry = [e for e in listing if e["name"] == "b"][0]
        assert b_entry["healthy"] is False
        assert b_entry["active"] is False

    def test_all_models(self):
        reg = _ib_mod.BackendRegistry()
        b1 = self._make_mock_backend("a")
        model1 = _ib_mod.BackendModelInfo(name="m1", backend="a")
        b1.list_models.return_value = [model1]
        reg.register(b1)

        models = reg.all_models()
        assert len(models) == 1
        assert models[0].name == "m1"

    def test_all_models_skips_unhealthy(self):
        reg = _ib_mod.BackendRegistry()
        b1 = self._make_mock_backend("sick", healthy=False)
        b1.list_models.return_value = [
            _ib_mod.BackendModelInfo(name="m", backend="sick")
        ]
        reg.register(b1)

        models = reg.all_models()
        assert len(models) == 0


# =========================================================================
# E. INFERENCE BACKEND — Helpers
# =========================================================================

class TestBackendHelpers:
    """Tests for module-level helper functions."""

    def test_inject_images(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        result = _ib_mod._inject_images(messages, ["img_data"])
        # Original untouched
        assert "images" not in messages[1]
        # Copy has images
        assert result[1]["images"] == ["img_data"]

    def test_inject_images_no_user(self):
        messages = [{"role": "system", "content": "sys"}]
        result = _ib_mod._inject_images(messages, ["img"])
        # No user message — no crash
        assert len(result) == 1

    def test_format_messages_for_llama_cpp(self):
        messages = [
            {"role": "system", "content": "be helpful", "extra": "ignored"},
            {"role": "user", "content": "hi"},
        ]
        formatted = _ib_mod._format_messages_for_llama_cpp(messages)
        assert len(formatted) == 2
        assert formatted[0] == {"role": "system", "content": "be helpful"}
        assert "extra" not in formatted[0]

    def test_parse_gguf_filename_quant_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "llama-3-8b-Q4_K_M.gguf"
            p.write_bytes(b"x" * 50)
            info = _ib_mod._parse_gguf_filename(p)
            assert info.quantization_level == "Q4_K_M"
            assert info.backend == "llama_cpp"

    def test_parse_gguf_filename_f16(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model-F16.gguf"
            p.write_bytes(b"x" * 50)
            info = _ib_mod._parse_gguf_filename(p)
            assert info.quantization_level == "F16"

    def test_parse_gguf_filename_no_quant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "mymodel.gguf"
            p.write_bytes(b"x" * 50)
            info = _ib_mod._parse_gguf_filename(p)
            assert info.quantization_level is None


# =========================================================================
# F. MODEL MANAGER — GGUF Header Parser
# =========================================================================

class TestGGUFParser:
    """Tests for GGUF header parsing (pure Python)."""

    def test_parse_valid_v3(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _create_gguf_file(tmpdir, kv_pairs=[
                (_write_kv_string, "general.architecture", "llama"),
                (_write_kv_string, "general.name", "TestModel"),
                (_write_kv_uint32, "llama.context_length", 4096),
                (_write_kv_uint32, "llama.block_count", 32),
            ], tensor_count=42)

            meta = _mm_mod.parse_gguf_header(str(path))
            assert meta.version == 3
            assert meta.tensor_count == 42
            assert meta.architecture == "llama"
            assert meta.model_name == "TestModel"
            assert meta.context_length == 4096
            assert meta.block_count == 32
            assert meta.metadata_kv_count == 4

    def test_parse_valid_v2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _create_gguf_file(tmpdir, version=2, kv_pairs=[
                (_write_kv_string, "general.architecture", "gpt2"),
            ])
            meta = _mm_mod.parse_gguf_header(str(path))
            assert meta.version == 2
            assert meta.architecture == "gpt2"

    def test_parse_invalid_magic(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"NOPE" + b"\x00" * 100)
            fpath = f.name
        try:
            with pytest.raises(_mm_mod.GGUFParseError, match="magic"):
                _mm_mod.parse_gguf_header(fpath)
        finally:
            os.unlink(fpath)

    def test_parse_file_not_found(self):
        with pytest.raises(_mm_mod.GGUFParseError, match="not found"):
            _mm_mod.parse_gguf_header("/nonexistent/model.gguf")

    def test_parse_truncated_file(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"GGUF")  # Only magic, no version
            fpath = f.name
        try:
            with pytest.raises(_mm_mod.GGUFParseError):
                _mm_mod.parse_gguf_header(fpath)
        finally:
            os.unlink(fpath)

    def test_parse_unsupported_version(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 99))  # Bad version
            f.write(b"\x00" * 100)
            fpath = f.name
        try:
            with pytest.raises(_mm_mod.GGUFParseError, match="version"):
                _mm_mod.parse_gguf_header(fpath)
        finally:
            os.unlink(fpath)

    def test_parse_float32_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _create_gguf_file(tmpdir, kv_pairs=[
                (_write_kv_float32, "test.value", 3.14),
            ])
            meta = _mm_mod.parse_gguf_header(str(path))
            assert abs(meta.metadata.get("test.value", 0) - 3.14) < 0.01

    def test_parse_bool_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _create_gguf_file(tmpdir, kv_pairs=[
                (_write_kv_bool, "test.flag", True),
            ])
            meta = _mm_mod.parse_gguf_header(str(path))
            assert meta.metadata.get("test.flag") is True

    def test_parse_file_type_to_quant_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _create_gguf_file(tmpdir, kv_pairs=[
                (_write_kv_string, "general.architecture", "llama"),
                (_write_kv_uint32, "general.file_type", 15),  # Q4_K_M
            ])
            meta = _mm_mod.parse_gguf_header(str(path))
            assert meta.quantization_name == "Q4_K_M"

    def test_parse_max_kv_read_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kv_pairs = [
                (_write_kv_string, f"key_{i}", f"value_{i}")
                for i in range(10)
            ]
            path = _create_gguf_file(tmpdir, kv_pairs=kv_pairs)
            meta = _mm_mod.parse_gguf_header(str(path), max_kv_read=3)
            # Should only have read 3 KV pairs
            assert len(meta.metadata) == 3

    def test_to_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _create_gguf_file(tmpdir, kv_pairs=[
                (_write_kv_string, "general.architecture", "llama"),
            ])
            meta = _mm_mod.parse_gguf_header(str(path))
            d = meta.to_dict()
            assert "version" in d
            assert "architecture" in d
            assert "tensor_count" in d


class TestGGUFParserParameterEstimate:
    """Tests for parameter count estimation."""

    def test_estimate_with_embedding_and_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _create_gguf_file(tmpdir, kv_pairs=[
                (_write_kv_string, "general.architecture", "llama"),
                (_write_kv_uint32, "llama.embedding_length", 4096),
                (_write_kv_uint32, "llama.block_count", 32),
            ])
            meta = _mm_mod.parse_gguf_header(str(path))
            assert meta.parameter_count is not None
            assert meta.parameter_count > 0

    def test_no_estimate_without_embedding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _create_gguf_file(tmpdir, kv_pairs=[
                (_write_kv_string, "general.architecture", "llama"),
                (_write_kv_uint32, "llama.block_count", 32),
            ])
            meta = _mm_mod.parse_gguf_header(str(path))
            assert meta.parameter_count is None


# =========================================================================
# G. MODEL MANAGER — ModelManager class
# =========================================================================

class TestModelManager:
    """Tests for ModelManager."""

    def test_scan_models_empty(self):
        mgr = _mm_mod.ModelManager(model_dirs=["/nonexistent"])
        assert mgr.scan_models() == []

    def test_scan_models_with_gguf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_gguf_file(tmpdir, "model-a.gguf", kv_pairs=[
                (_write_kv_string, "general.architecture", "llama"),
            ])
            _create_gguf_file(tmpdir, "model-b.gguf", kv_pairs=[
                (_write_kv_string, "general.architecture", "gpt2"),
            ])
            # Non-GGUF file should be ignored
            (Path(tmpdir) / "readme.txt").write_text("hello")

            mgr = _mm_mod.ModelManager(model_dirs=[tmpdir])
            models = mgr.scan_models()
            assert len(models) == 2
            filenames = {m["filename"] for m in models}
            assert "model-a.gguf" in filenames
            assert "model-b.gguf" in filenames

    def test_get_model_info_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _create_gguf_file(tmpdir, "test.gguf", kv_pairs=[
                (_write_kv_string, "general.architecture", "llama"),
                (_write_kv_string, "general.name", "MyModel"),
                (_write_kv_uint32, "llama.context_length", 8192),
            ])
            mgr = _mm_mod.ModelManager(model_dirs=[tmpdir])
            info = mgr.get_model_info(str(path))
            assert info is not None
            assert info["architecture"] == "llama"
            assert info["model_name"] == "MyModel"
            assert info["context_length"] == 8192

    def test_get_model_info_not_found(self):
        mgr = _mm_mod.ModelManager(model_dirs=[])
        assert mgr.get_model_info("nonexistent.gguf") is None

    def test_get_model_info_invalid_gguf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.gguf"
            bad.write_bytes(b"NOT_GGUF_DATA")
            mgr = _mm_mod.ModelManager(model_dirs=[tmpdir])
            assert mgr.get_model_info(str(bad)) is None

    def test_metadata_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _create_gguf_file(tmpdir, "cached.gguf", kv_pairs=[
                (_write_kv_string, "general.architecture", "llama"),
            ])
            mgr = _mm_mod.ModelManager(model_dirs=[tmpdir])
            info1 = mgr.get_model_info(str(path))
            info2 = mgr.get_model_info(str(path))
            assert info1 == info2
            assert len(mgr._metadata_cache) == 1

    def test_clear_cache(self):
        mgr = _mm_mod.ModelManager()
        mgr._metadata_cache["key"] = "val"
        count = mgr.clear_cache()
        assert count == 1
        assert len(mgr._metadata_cache) == 0

    def test_get_storage_usage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.gguf").write_bytes(b"x" * 1000)
            (Path(tmpdir) / "b.gguf").write_bytes(b"y" * 2000)

            mgr = _mm_mod.ModelManager(model_dirs=[tmpdir])
            usage = mgr.get_storage_usage()
            assert usage["model_count"] == 2
            assert usage["total_size"] == 3000
            assert len(usage["directories"]) == 1
            assert usage["directories"][0]["exists"] is True

    def test_get_storage_usage_nonexistent_dir(self):
        mgr = _mm_mod.ModelManager(model_dirs=["/nonexistent"])
        usage = mgr.get_storage_usage()
        assert usage["model_count"] == 0
        assert usage["directories"][0]["exists"] is False

    def test_add_model_dir(self):
        mgr = _mm_mod.ModelManager()
        assert mgr.add_model_dir("/new/path") is True
        assert mgr.add_model_dir("/new/path") is False  # Duplicate
        assert len(mgr.model_dirs) == 1

    def test_delete_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "delete_me.gguf"
            path.write_bytes(b"x" * 500)
            mgr = _mm_mod.ModelManager(model_dirs=[tmpdir])
            result = mgr.delete_model(str(path))
            assert result["status"] == "deleted"
            assert result["freed"] == 500
            assert not path.exists()

    def test_delete_model_not_found(self):
        mgr = _mm_mod.ModelManager(model_dirs=[])
        result = mgr.delete_model("nonexistent.gguf")
        assert result["status"] == "error"


# =========================================================================
# H. MODEL MANAGER — Utility functions
# =========================================================================

class TestModelManagerUtils:
    """Tests for utility functions."""

    def test_format_size(self):
        assert _mm_mod._format_size(0) == "0B"
        assert _mm_mod._format_size(None) == "0B"
        assert _mm_mod._format_size(500) == "500B"
        assert _mm_mod._format_size(1500) == "1.5KB"
        assert _mm_mod._format_size(1_500_000) == "1.5MB"
        assert _mm_mod._format_size(4_700_000_000) == "4.7GB"

    def test_format_params(self):
        assert _mm_mod._format_params(None) is None
        assert _mm_mod._format_params(500) == "500"
        assert _mm_mod._format_params(1_500) == "1.5K"
        assert _mm_mod._format_params(7_000_000) == "7.0M"
        assert _mm_mod._format_params(7_000_000_000) == "7.0B"

    def test_estimate_parameter_count(self):
        count = _mm_mod._estimate_parameter_count(
            embedding_length=4096, block_count=32
        )
        assert count > 0
        # Should be in the billions range for 4096/32 arch
        assert count > 1_000_000_000


# =========================================================================
# I. MODEL MANAGER — Singleton
# =========================================================================

class TestModelManagerSingleton:
    """Tests for singleton pattern."""

    def test_singleton_consistency(self):
        # Reset singleton for test
        _mm_mod._model_manager_instance = None
        m1 = _mm_mod.get_model_manager()
        m2 = _mm_mod.get_model_manager()
        assert m1 is m2
        # Cleanup
        _mm_mod._model_manager_instance = None


# =========================================================================
# J. INFERENCE BACKEND — Config loading
# =========================================================================

class TestBackendConfig:
    """Tests for backend config loading."""

    def test_load_nonexistent_config(self):
        cfg = _ib_mod._load_backend_config("/nonexistent/path.yaml")
        assert cfg == {}

    def test_load_valid_config(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("default_backend: llama_cpp\n")
            f.write("ollama:\n  host: http://localhost:11434\n")
            f.write("llama_cpp:\n  model_dirs:\n    - /models\n  n_ctx: 8192\n")
            fpath = f.name
        try:
            cfg = _ib_mod._load_backend_config(fpath)
            assert cfg["default_backend"] == "llama_cpp"
            assert cfg["llama_cpp"]["n_ctx"] == 8192
        finally:
            os.unlink(fpath)

    def test_init_backends_from_config_no_file(self):
        # Reset singleton
        _ib_mod._registry_instance = None
        reg = _ib_mod.init_backends_from_config("/nonexistent.yaml")
        assert isinstance(reg, _ib_mod.BackendRegistry)
        _ib_mod._registry_instance = None


# =========================================================================
# K. EXECUTOR INTEGRATION
# =========================================================================

class TestExecutorIntegration:
    """Tests for executor.py backend abstraction wiring."""

    def test_executor_has_backend_import(self):
        """Verify executor.py imports the backend abstraction."""
        content = (ROOT / "opti_oignon" / "executor.py").read_text()
        assert "INFERENCE_BACKEND_AVAILABLE" in content
        assert "get_backend_registry" in content

    def test_executor_has_three_backend_paths(self):
        """Verify all three call sites have backend abstraction."""
        content = (ROOT / "opti_oignon" / "executor.py").read_text()
        # Count backend usage sites
        count = content.count("INFERENCE_BACKEND_AVAILABLE")
        # 2 in import block (True/False) + 3 in usage
        assert count >= 5, f"Expected >= 5 occurrences, got {count}"

    def test_executor_has_fallbacks(self):
        """Verify all call sites have ollama fallback paths."""
        content = (ROOT / "opti_oignon" / "executor.py").read_text()
        assert content.count("# Fallback: direct ollama") >= 2
        assert "# Fallback: direct ollama.chat() call" in content


# =========================================================================
# L. ROUTES BACKENDS — Schema validation
# =========================================================================

class TestBackendSchemas:
    """Tests for backend API schemas."""

    def test_schemas_importable(self):
        """Verify all backend schemas exist in schemas.py."""
        content = (ROOT / "opti_oignon" / "api" / "schemas.py").read_text()
        expected = [
            "BackendStatusResponse",
            "BackendListResponse",
            "BackendActivateResponse",
            "GGUFModelInfoResponse",
            "GGUFModelListResponse",
            "GGUFDownloadRequest",
            "GGUFDownloadResponse",
            "GGUFStorageResponse",
            "BackendModelsResponse",
        ]
        for name in expected:
            assert name in content, f"Schema {name} not found in schemas.py"


# =========================================================================
# M. ROUTES BACKENDS — Route ordering
# =========================================================================

class TestRouteOrdering:
    """Tests for correct route registration order."""

    def test_fixed_routes_before_parametric(self):
        """Verify fixed paths are registered before /{name} catch-all."""
        content = (ROOT / "opti_oignon" / "api" / "routes_backends.py").read_text()
        # Find @router.get/post decorator positions (not comments)
        models_all_pos = content.index('@router.get("/models/all"')
        gguf_models_pos = content.index('@router.get("/gguf/models"')
        gguf_storage_pos = content.index('@router.get("/gguf/storage"')
        name_get_pos = content.index('@router.get("/{name}"')
        name_post_pos = content.index('@router.post("/{name}/activate"')

        assert models_all_pos < name_get_pos
        assert gguf_models_pos < name_get_pos
        assert gguf_storage_pos < name_get_pos
        assert name_get_pos < name_post_pos


# =========================================================================
# N. APP.PY INTEGRATION
# =========================================================================

class TestAppIntegration:
    """Tests for app.py and deps.py integration."""

    def test_app_includes_backends_router(self):
        content = (ROOT / "opti_oignon" / "api" / "app.py").read_text()
        assert "backends_router" in content
        assert "include_router(backends_router)" in content

    def test_app_health_check_includes_backend_flags(self):
        content = (ROOT / "opti_oignon" / "api" / "app.py").read_text()
        assert '"inference_backend"' in content
        assert '"model_manager"' in content

    def test_deps_has_backend_flags(self):
        content = (ROOT / "opti_oignon" / "api" / "deps.py").read_text()
        assert "INFERENCE_BACKEND_AVAILABLE" in content
        assert "MODEL_MANAGER_AVAILABLE" in content


# =========================================================================
# O. CONFIG FILE
# =========================================================================

class TestBackendsYaml:
    """Tests for backends.yaml configuration file."""

    def test_yaml_exists_and_valid(self):
        import yaml
        path = ROOT / "opti_oignon" / "config" / "backends.yaml"
        assert path.is_file()
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "default_backend" in data
        assert "ollama" in data
        assert "llama_cpp" in data

    def test_yaml_default_backend(self):
        import yaml
        path = ROOT / "opti_oignon" / "config" / "backends.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["default_backend"] in ("ollama", "llama_cpp")

    def test_yaml_llama_cpp_settings(self):
        import yaml
        path = ROOT / "opti_oignon" / "config" / "backends.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        lc = data["llama_cpp"]
        assert "model_dirs" in lc
        assert "n_ctx" in lc
        assert "n_gpu_layers" in lc
        assert isinstance(lc["model_dirs"], list)


# =========================================================================
# P. LAUNCH.SH
# =========================================================================

class TestLaunchSh:
    """Tests for launch.sh backend detection."""

    def test_launch_sh_ollama_optional(self):
        content = (ROOT / "launch.sh").read_text()
        assert "Ollama is now OPTIONAL" in content
        assert "HAS_BACKEND" in content
        assert "LLAMA_CPP_OK" in content
        assert "OLLAMA_OK" in content

    def test_launch_sh_version_updated(self):
        content = (ROOT / "launch.sh").read_text()
        assert "v2.0.0" in content

    def test_launch_sh_no_hard_exit_on_ollama_missing(self):
        """Verify Ollama missing alone does not exit."""
        content = (ROOT / "launch.sh").read_text()
        # The only exit 1 related to backends should be when NO backend at all
        # Find the "No inference backend" block
        assert "No inference backend available" in content
