#!/usr/bin/env python3
"""Contracts that a directory written with ``~`` in a configuration file is
the user's home directory.

``backends.yaml`` ships ``~/models/gguf`` as the llama.cpp model directory
and as the download directory. A path taken as written keeps the tilde as
a folder name under the working directory: the model directory is never
found, so llama.cpp lists nothing, and a download lands in a literal ``~``
folder wherever the process started.

  * HP1 -- llama.cpp scans a model directory configured as ``~/...`` in the
    home directory, not in a folder named ``~`` under the working directory.
  * HP2 -- the model manager reads its scan directories and its download
    directory from ``backends.yaml`` the same way, and so does its
    constructor.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window; ``HOME`` and the working directory are seams, both
set under the test's own directory.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home directory and a separate working directory, both under the test's own."""
    home = tmp_path / "home"
    (home / "models" / "gguf").mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    return home


# ---------------------------------------------------------------------------
# HP1 -- llama.cpp
# ---------------------------------------------------------------------------
def test_hp1_llama_cpp_scans_a_tilde_directory_in_the_home_directory(home):
    (home / "models" / "gguf" / "served-7b.Q4_K_M.gguf").write_bytes(b"GGUF")
    loaded, restore = isolate(targets={"opti_oignon.inference_backend": source("inference_backend.py")}, packages=("opti_oignon",))
    try:
        mod = loaded["opti_oignon.inference_backend"]
        listed = mod.LlamaCppBackend(model_dirs=["~/models/gguf"]).list_models()
        assert listed is not None, "the configured directory exists: it is in the home directory"
        assert [m.name for m in listed] == ["served-7b.Q4_K_M.gguf"]
        assert not (Path.cwd() / "~").exists(), "nothing reads or makes a folder named ~ in the working directory"
    finally:
        restore()


# ---------------------------------------------------------------------------
# HP2 -- the model manager
# ---------------------------------------------------------------------------
def test_hp2_the_model_manager_puts_its_directories_in_the_home_directory(home, tmp_path):
    config = tmp_path / "backends.yaml"
    config.write_text(
        "llama_cpp:\n  model_dirs:\n    - \"~/models/gguf\"\n  default_download_dir: \"~/downloads/gguf\"\n",
        encoding="utf-8",
    )
    loaded, restore = isolate(targets={"opti_oignon.model_manager": source("model_manager.py")}, packages=("opti_oignon",))
    try:
        mm = loaded["opti_oignon.model_manager"]
        manager = mm.init_model_manager(str(config))
        assert manager.model_dirs == [home / "models" / "gguf"], manager.model_dirs
        assert manager.default_dir == home / "downloads" / "gguf", "a download lands in the home directory"
        built = mm.ModelManager(model_dirs=["~/models/gguf"], default_dir="~/downloads/gguf")
        assert built.model_dirs == [home / "models" / "gguf"] and built.default_dir == home / "downloads" / "gguf"
        assert built.add_model_dir("~/more") is True and built.model_dirs[-1] == home / "more"
    finally:
        restore()
