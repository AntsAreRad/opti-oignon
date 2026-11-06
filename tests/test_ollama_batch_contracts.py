#!/usr/bin/env python3
"""Contracts of a fixed batch per model (D14), sent by oo's Ollama engine.
Ollama reads a prompt in batches of ``num_batch`` (512 of its own); a larger batch reads the onion's window several
times faster, and a model asked at another batch than it holds is loaded again, as for a context. Every request of
oo's Ollama engine that may load a model therefore sends the batch ``backends.yaml`` gives it.
  * BA1 -- every head that may load a model (generate, stream, embed, embed_many) sends the model's batch beside the
    call's own options; a model the table does not name sends none; a call that names its own keeps it (the tuner's
    sweep); the table's names and the call's compare as Ollama resolves them.
  * BA2 -- the table is read from ``backends.yaml`` by ``init_backends_from_config``; an entry that is not a model's
    name and a whole number above zero is refused by its name, and so is a second entry for a model already named,
    the others kept; a value that is no table is refused whole; the shipped file gives the two models measured 2048.
Local-only. The engine module is loaded through the shared isolation window, the client library faked.
"""
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_ba1_every_head_sends_the_models_batch_and_a_callers_own_wins": 3.0,
    "test_ba2_the_table_comes_from_backends_yaml_and_a_bad_entry_is_refused_by_name": 3.0,
}

_BACKEND = "opti_oignon.inference_backend"
_MSGS = [{"role": "user", "content": "hi"}]


class _Fake:
    """The client library: every chat's and every embedding's arguments kept."""

    def __init__(self):
        self.sent = []

    def chat(self, **kwargs):
        self.sent.append(("chat", kwargs))
        if kwargs.get("stream"):
            return iter([{"message": {"content": "x"}, "done": True}])
        return {"message": {"content": "ok"}}

    def embed(self, **kwargs):
        self.sent.append(("embed", kwargs))
        texts = kwargs.get("input")
        return {"embeddings": [[0.5]] * (len(texts) if isinstance(texts, list) else 1)}


def _open_backend():
    loaded, restore = isolate(targets={_BACKEND: source("inference_backend.py")}, packages=("opti_oignon",))
    mod = loaded[_BACKEND]
    fake = _Fake()
    mod.OLLAMA_AVAILABLE = True
    mod._ollama_module = fake
    return mod, fake, restore


def test_ba1_every_head_sends_the_models_batch_and_a_callers_own_wins():
    """BA1 -- the table names ``big`` 2048, ``small:4b`` 1024 and ``enc`` 256. Generate, the stream and both
    embedding heads send the model's batch, beside the call's own options; ``big:latest`` is ``big`` to Ollama; a
    model the table does not name sends no batch; a call that names one keeps it. A head that sends none, a table
    name compared as written, or the caller's batch overwritten -> RED."""
    mod, fake, restore = _open_backend()
    try:
        mod._governor_admission = lambda model, options, unsplittable=None: None
        backend = mod.OllamaBackend()
        backend._num_batch = mod._read_num_batch({"big": 2048, "small:4b": 1024, "enc": 256})
        assert backend.generate("big", _MSGS).content == "ok"
        assert "".join(c.content for c in backend.stream("big:latest", _MSGS)) == "x"
        backend.generate("small:4b", _MSGS, options={"num_ctx": 4096})
        backend.generate("other", _MSGS)
        backend.generate("big", _MSGS, options={"num_batch": 256, "num_ctx": 2048})
        assert backend.embed("enc", "a") == [0.5]
        assert backend.embed_many("enc:latest", ["a", "b"]) == [[0.5], [0.5]]
        assert backend.embed("other", "a") == [0.5]
        sent = [(head, kw["model"], kw.get("options") or {}) for head, kw in fake.sent]
        assert sent == [("chat", "big", {"num_batch": 2048}), ("chat", "big:latest", {"num_batch": 2048}),
                        ("chat", "small:4b", {"num_ctx": 4096, "num_batch": 1024}), ("chat", "other", {}),
                        ("chat", "big", {"num_batch": 256, "num_ctx": 2048}),
                        ("embed", "enc", {"num_batch": 256}), ("embed", "enc:latest", {"num_batch": 256}),
                        ("embed", "other", {})], sent
        assert backend.batch_for("big:latest") == 2048 and backend.batch_for("small") is None
    finally:
        restore()


def test_ba2_the_table_comes_from_backends_yaml_and_a_bad_entry_is_refused_by_name(tmp_path, caplog):
    """BA2 -- entries that are not a model's name and a whole number above zero are each refused by name, the others
    kept under the name Ollama resolves; ``big:latest`` after ``big`` names the same model and is refused by both
    names, the first kept; a list is refused whole, an absent table is no table; the file reaches the engine through
    ``init_backends_from_config``, and a file read again whose Ollama section names no table leaves none; the
    shipped file gives qwen3.8:27b and qwen3.5:4b 2048. A bad entry kept, a second entry for one model silently
    winning, a refusal unsaid, an earlier table kept, or the shipped batches changed -> RED."""
    mod, _fake, restore = _open_backend()
    try:
        caplog.set_level(logging.WARNING, logger=_BACKEND)
        table = mod._read_num_batch({"big": 2048, "flag": True, "half": 1.5, "zero": 0, "neg": -2, "word": "2048",
                                     " ": 512, 7: 512, "enc:latest": 64, "big:latest": 512})
        assert table == {"big:latest": 2048, "enc:latest": 64}, table
        said = caplog.text
        assert all(name in said for name in ("flag", "half", "zero", "neg", "word", "' '", "7")), said
        assert "names big again as big:latest" in said, said
        caplog.clear()
        assert mod._read_num_batch(["big", 2048]) == {} and "not a table" in caplog.text
        caplog.clear()
        assert mod._read_num_batch(None) == {} and caplog.text == ""
        config = tmp_path / "backends.yaml"
        config.write_text(yaml.safe_dump({"ollama": {"num_batch": {"big": 2048}}}), encoding="utf-8")
        registry = mod.init_backends_from_config(str(config))
        assert registry.get("ollama").batch_for("big") == 2048
        config.write_text(yaml.safe_dump({"ollama": {"timeout": 30}}), encoding="utf-8")
        assert mod.init_backends_from_config(str(config)).get("ollama").batch_for("big") is None
        shipped = yaml.safe_load(Path(source("config", "backends.yaml")).read_text(encoding="utf-8"))
        assert shipped["ollama"]["num_batch"] == {"qwen3.8:27b": 2048, "qwen3.5:4b": 2048}
    finally:
        restore()
