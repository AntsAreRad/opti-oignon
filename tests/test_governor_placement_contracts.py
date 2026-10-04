#!/usr/bin/env python3
"""What the governor promises when it places a split model layer by layer.

A split admission used to stop at a share: so many GiB on the GPU, the rest
in RAM, and the layers left to the engine. Ollama then placed them by its own
estimate, which knows nothing of the admission, and llama.cpp in process put
every layer on the GPU, so the gate refused it. Both engines put the LAST
layers on the GPU when told how many. What the governor lacked was the size
of each layer, which only the model file's tensor table holds.

  The tensor table, read from the file (model_manager):
    * GP1 -- each of the 34 ggml tensor types reads to its exact bytes, as
      ggml lays its blocks out; any other type id is refused by name.
    * GP2 -- the bytes are summed per block (blk.N.), outside the blocks, and
      for the output head (output.weight, else the tied token embedding).
    * GP3 -- every metadata value type is skipped without being kept, nested
      arrays included; the declared alignment places the data.
    * GP4 -- a corpus of damaged files: each refusal reason, by name, and no
      reason outside the closed set.
    * GP5 -- a file cut at any byte is refused by name, never otherwise.
    * GP6 -- a file with any byte inverted is read or refused by name.

  The plan (resource_governor):
    * GP7 -- the KV geometry carries each layer's bytes per token, and a
      split carries each layer's KV with it, uniform under an operator
      override.
    * GP8 -- the table of the file llama.cpp names is read once per file
      identity, again when the file changes, and never for a whole load.
    * GP9 -- Ollama's model file is the blob its modelfile names (FROM); the
      projector is set aside, and a path that is not a blob is never opened.
    * GP10 -- the last blocks that fit the free VRAM go to the GPU, with the
      cost the tensors and the KV do not explain; num_gpu = gpu_layers = N.
    * GP11 -- blocks of unequal sizes: the count is taken from the end.
    * GP12 -- a cost priced under the model's own tensors is raised to them,
      and a draft loading beside it is priced on top.
    * GP13 -- when no planned split holds, the refusal names what the plan
      lacked, in the words used before, and names a split where not one
      layer fits.
    * GP14 -- without a plan the split is the one given before: no table, no
      context told, or weights the operator names for the model.

  The pin and the gate:
    * GP15 -- a split load pins its layer count at its context, unless the
      caller names its own; a resident's decision at that context carries it;
      a load, an eviction, or the model leaving the loaded view clears it.
    * GP16 -- the gate returns the decision it acted on: the ticket, the
      backstop's, or None when the governor is disabled.
    * GP17 -- a split is refused to an engine that cannot split only when it
      carries no layer count.

  The engines (inference_backend):
    * GP18 -- Ollama is sent the decision's num_gpu, generating or streaming,
      unless the caller names its own; a whole load, or a call that tells no
      context, is sent none.
    * GP19 -- the pinned count rides the calls that follow the split load.
    * GP20 -- llama.cpp loads at the admitted context and layer count; the
      operator's explicit count is a ceiling a counted split can lower.
    * GP21 -- a held model is loaded again only for a longer context on the
      GPU alone, and the old one is closed first.
    * GP22 -- the backend's loaded view carries the context each model is
      held at.

  The expected speed:
    * GP23 -- the expected slowdown: the bytes read per token on each side
      over its bandwidth, against all of them on the GPU.
    * GP24 -- a split slower than max_slowdown does not hold: the ladder goes
      on, the final refusal names it, and an unknown speed never refuses.
    * GP25 -- the split_speed block parses, holds its ranges, and ships null.
    * GP26 -- the config routes show and write its three keys, in range only.
    * GP27 -- the decision's dictionary carries num_gpu, gpu_layers and the
      expected slowdown.

  The count and its context, and what the contracts above left open:
    * GP28 -- the count rides only with the context it was priced for.
    * GP29 -- a split load pins its count only when the count rode with it.
    * GP30 -- Ollama is told the count only while it keeps one sequence, the
      KV the plan prices (num_parallel named 1).
    * GP31 -- llama.cpp keeps the count whatever Ollama's parallelism.
    * GP32 -- a refusal after a planned split is figured on the plan's total.
    * GP33 -- the reader refuses what is not a regular file as such, closes
      every descriptor it opened, and does not follow a link.
    * GP34 -- weights an engine declares leave no count to tell from the file.
    * GP35 -- a model without a KV cache is placed layer by layer with no
      context told.
    * GP36 -- a declared state is counted once in the planned total.
    * GP37 -- a resident at an unknown context carries no pin to a call that
      tells no context.
    * GP38 -- a split without a count serves a model llama.cpp holds as held.
    * GP39 -- a relative FROM is never opened, even where it names a blob.
    * GP40 -- one call through the governor at another context than the one
      priced is sent no count and pins nothing; at that context, both.
    * GP41 -- a stream refused for its options was never admitted.

Everything here is proven in the container, on files the contracts write
(holes where they are large), scripted snapshots and fakes loaded through the
shared isolation window: no card, no socket, no model. Where an engine really
puts the layers, what a reload costs and how fast a split runs are owed to the
machine.
"""

import dataclasses
import os
import sqlite3
import struct
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_RG = "opti_oignon.resource_governor"
_MM = "opti_oignon.model_manager"
_BACKEND = "opti_oignon.inference_backend"
_ROUTES = "opti_oignon.api.routes_governor"
# The two seams the governor resolves at the call: unreachable when the window
# opens, so neither an emergency stop nor a model window enters a decision.
_SEAMS = ("opti_oignon.context_manager", "opti_oignon.emergency_stop")
_GIB = 1024 ** 3
_CLOSERS = []

# ggml's block layout per tensor type, id -> (elements per block, bytes per
# block), as ggml's type traits define them (checked against libggml 0.11.1).
_GGML = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66),
    17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4),
    27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54),
    35: (256, 66), 39: (32, 17), 40: (64, 36), 41: (128, 18),
}
_F32, _Q4_0, _Q4_K, _Q6_K = 0, 2, 12, 14
_REFUSALS = {
    "bad_magic", "unsupported_version", "too_many_tensors", "too_many_pairs",
    "name_too_long", "string_too_long", "array_too_long", "nesting_too_deep",
    "unknown_value_type", "bad_alignment", "bad_dimensions",
    "unknown_tensor_type", "partial_block", "misaligned_offset",
    "past_end_of_file", "duplicate_tensor", "truncated",
}


def _db_utils():
    """A db_utils stand-in whose safe_connect is plain sqlite."""
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda p, **kw: sqlite3.connect(
        str(p), check_same_thread=kw.get("check_same_thread", False)
    )
    return db


def _open(*extra, packages=()):
    """The governor and ``extra`` (name, path) modules after it, in one window."""
    targets = {_RG: source("resource_governor.py")}
    for name, path in extra:
        targets[name] = path
    loaded, restore = isolate(
        targets=targets,
        blocked=_SEAMS,
        seeded={"opti_oignon.db_utils": _db_utils()},
        packages=packages,
    )
    _CLOSERS.append(restore)
    return loaded


def _with_mm():
    return _open((_MM, source("model_manager.py")))


def _project_modules():
    """Every project entry of the module cache, by identity."""
    return {k: v for k, v in sys.modules.items() if k == "opti_oignon" or k.startswith("opti_oignon.")}


@pytest.fixture(autouse=True)
def _left_as_found():
    """No contract may leave a project module or the HTTP transport changed."""
    before = _project_modules()
    urlopen = urllib.request.urlopen
    yield
    urllib.request.urlopen = urlopen
    while _CLOSERS:
        _CLOSERS.pop()()
    assert _project_modules() == before, "every project module is left as the contract found it"


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="gp-"))


# ---------------------------------------------------------------------------
# GGUF files written by the contracts
# ---------------------------------------------------------------------------


class _Typed:
    """A metadata value written with an explicit GGUF type id and raw bytes."""

    def __init__(self, kind, raw):
        self.kind = kind
        self.raw = raw


def _str(text):
    raw = text if isinstance(text, bytes) else text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(value):
    """(GGUF type id, bytes) of a metadata value."""
    if isinstance(value, _Typed):
        return value.kind, value.raw
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, int):
        return 4, struct.pack("<I", value)
    if isinstance(value, float):
        return 6, struct.pack("<f", value)
    if isinstance(value, (str, bytes)):
        return 8, _str(value)
    if isinstance(value, list):
        kind = _value(value[0])[0] if value else 4
        body = b"".join(_value(item)[1] for item in value)
        return 9, struct.pack("<IQ", kind, len(value)) + body
    raise TypeError(value)


def _nbytes(kind, shape):
    per_block, size = _GGML.get(kind, (1, 0))
    count = 1
    for n in shape:
        count *= n
    return count // per_block * size


def _layout(tensors, meta, *, version=3, alignment=32, offsets=None, counts=None):
    """The header ggml would write, its data offset, and the file length."""
    tally = counts if counts is not None else (len(tensors), len(meta))
    out = b"GGUF" + struct.pack("<IQQ", version, *tally)
    for key, value in meta.items():
        kind, raw = _value(value)
        out += _str(key) + struct.pack("<I", kind) + raw
    at = 0
    for index, (name, kind, shape) in enumerate(tensors):
        offset = at if offsets is None else offsets[index]
        out += _str(name) + struct.pack("<I", len(shape))
        out += b"".join(struct.pack("<Q", n) for n in shape)
        out += struct.pack("<IQ", kind, offset)
        at += -(-_nbytes(kind, shape) // alignment) * alignment
    data = -(-len(out) // alignment) * alignment
    return out, data, data + at


def _gguf(path, tensors, meta=None, *, arch="llama", version=3, alignment=None):
    """A GGUF file holding ``tensors`` (name, type, shape) after ``meta``.

    The data section is a hole: a file of many GiB costs no disk.
    """
    fields = {"general.architecture": arch} if arch is not None else {}
    if alignment is not None:
        fields["general.alignment"] = alignment
    fields.update(meta or {})
    header, _data, length = _layout(tensors, fields, version=version, alignment=alignment or 32)
    path = Path(path)
    path.write_bytes(header)
    os.truncate(path, length)
    return path


def _model(path, blocks_gib, *, embd_gib=0.5, head_gib=0.5, kv_heads=8):
    """A llama model file whose block i weighs ``blocks_gib[i]`` GiB.

    One F32 tensor per block, a token embedding and an output head, and the
    geometry keys: 32 heads of 128, ``kv_heads`` KV heads (an int, or one per
    layer), so 512 bytes per token per KV head and layer.
    """

    def f32(gib):
        return (int(gib * 2 ** 28),)

    tensors = [("token_embd.weight", _F32, f32(embd_gib))]
    tensors += [(f"blk.{i}.ffn_up.weight", _F32, f32(gib)) for i, gib in enumerate(blocks_gib)]
    if head_gib:
        tensors.append(("output.weight", _F32, f32(head_gib)))
    meta = {
        "llama.block_count": len(blocks_gib),
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": kv_heads,
        "llama.embedding_length": 4096,
        "llama.attention.key_length": 128,
        "llama.attention.value_length": 128,
    }
    return _gguf(path, tensors, meta)


def _llama_info(layers, kv_heads=8):
    """The geometry keys of ``_model`` as Ollama's model_info reports them."""
    return {
        "general.architecture": "llama",
        "llama.block_count": layers,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": kv_heads,
        "llama.embedding_length": 4096,
        "llama.attention.key_length": 128,
        "llama.attention.value_length": 128,
    }


def _untabled(path, layers):
    """A header the geometry reads, over a tensor table that is refused."""
    meta = {k: v for k, v in _llama_info(layers).items() if k != "general.architecture"}
    return _gguf(path, [("blk.0.weight", 4, (32,))], meta)


def _refused(mm, path):
    with pytest.raises(mm.GGUFTableError) as refused:
        mm.read_gguf_tensors(path)
    return refused.value.reason


# ---------------------------------------------------------------------------
# The governor's fakes
# ---------------------------------------------------------------------------


class _Clock:
    """A fixed monotonic stand-in: a hand-set snapshot never goes stale."""

    def __call__(self) -> float:
        return 1000.0


def _meminfo(directory: Path, ram_mb: float) -> str:
    path = directory / "meminfo"
    path.write_text(f"MemAvailable:   {int(ram_mb * 1024)} kB\n", encoding="utf-8")
    return str(path)


def _config(rg, *, weights=None, capacity=10.0, **fields):
    """A ``capacity`` GiB card with a 1.5 GiB margin: 8.5 GiB free on 10."""
    cfg = rg.GovernorConfig(
        total_vram_gb=capacity,
        safety_margin_gb=1.5,
        kv_coefficient=0.5,
        ctx_ladder=[8192, 4096, 2048],
        ctx_floor={"chat": 2048},
    )
    cfg.weights_override_models = dict(weights or {})
    for name, value in fields.items():
        setattr(cfg, name, value)
    return cfg


def _governor(rg, config=None, *, registry=None, warmup=None, ram_mb=64000.0):
    tmp = _tmp()
    gov = rg.ResourceGovernor(
        config_path=str(tmp / "missing.yaml"),
        db_path=str(tmp / "governor.db"),
        warmup=warmup,
        registry=registry,
        clock=_Clock(),
        meminfo_path=_meminfo(tmp, ram_mb),
        vram_probe=None,
    )
    if config is not None:
        gov._config = config
    return gov


def _snapshot(rg, gov, *, capacity=10.0, in_use=0.0, loaded=None, ram_mb=64000.0):
    gov._snapshot = rg.ResourceSnapshot(
        taken_at=1000.0,
        ttl_s=9999.0,
        loaded=list(loaded or []),
        capacity_gb=capacity,
        vram_in_use_gb=in_use,
        ram_available_mb=ram_mb,
    )
    return gov._snapshot


class _Warmup:
    """The warmup as the governor reads it: a keep_alive and a loaded set."""

    def __init__(self, loaded=None, keep_alive="10m"):
        self.keep_alive = keep_alive
        self._loaded = list(loaded or [])

    def get_loaded_models(self):
        return list(self._loaded)


def _resident(name, size_vram, size=None, context_length=None):
    return types.SimpleNamespace(
        name=name,
        size_vram=size_vram,
        size=size,
        expires_at=None,
        context_length=context_length,
        digest=None,
    )


class _Info:
    """A model_info answer: the metadata and the modelfile under ``extra``."""

    def __init__(self, mapping=None, path=None, modelfile=None):
        self.extra = {}
        if mapping is not None:
            self.extra["model_info"] = dict(mapping)
        if modelfile is not None:
            self.extra["modelfile"] = modelfile
        self.path = path
        self.parameter_size = None
        self.quantization_level = None
        self.size = None


class _Backend:
    def __init__(self, infos, name="ollama"):
        self.name = name
        self._infos = infos

    def model_info(self, model):
        return self._infos.get(model)


class _Registry:
    def __init__(self, *backends):
        self._backends = list(backends)

    def backends(self):
        return list(self._backends)


def _estimating(gov, figures):
    """The weights estimator answers ``figures`` (GiB by model), as for a
    model it has learned; an operator's override is another thing (GP14)."""
    gov.estimate_model_vram_gb = lambda model, digest=None, engine=None: (figures.get(model), "learned")
    return gov


def _plan(rg, path, weights, *, capacity=10.0, ram_mb=64000.0, name="llama_cpp", override=None, **fields):
    """A governor whose one engine names ``path`` for model "m", estimated
    at ``weights`` GiB (or overridden by the operator at ``override``)."""
    info = _Info(path=str(path) if path is not None else None)
    gov = _governor(
        rg,
        _config(rg, weights={"m": override} if override is not None else None, capacity=capacity, **fields),
        registry=_Registry(_Backend({"m": info}, name=name)),
        ram_mb=ram_mb,
    )
    _snapshot(rg, gov, capacity=capacity, ram_mb=ram_mb)
    return _estimating(gov, {"m": weights, "draft": 1.0})


def _counting(mm):
    """Record every path the tensor-table reader is asked to open."""
    reads = []
    real = mm.read_gguf_tensors

    def reader(path):
        reads.append(str(path))
        return real(path)

    mm.read_gguf_tensors = reader
    return reads


def _blob_store(tmp):
    blobs = tmp / "models" / "blobs"
    blobs.mkdir(parents=True)
    return blobs


class _GateGovernor:
    """What backend_admission_gate touches, with its admission scripted."""

    def __init__(self, decision):
        self.config = types.SimpleNamespace(enabled=True)
        self._decision = decision
        self.recorded = []
        self.loads = []
        self.pins = []

    def admit(self, model, requested, caller="direct"):
        return self._decision

    def _record_admission(self, decision):
        self.recorded.append(decision)

    def invalidate_on_load(self, model, num_ctx):
        self.loads.append((model, num_ctx))

    def pin_layers(self, model, layers, num_ctx):
        self.pins.append((model, layers, num_ctx))

    def _honour_conditional_eviction(self, decision):  # pragma: no cover
        raise AssertionError("a split is never conditional on an eviction")


def _split(rg, model, *, num_gpu=None, num_ctx=4096, num_parallel=1):
    """A split decision; Ollama keeps one sequence, as the plan prices, unless
    ``num_parallel`` says otherwise (GP30)."""
    return rg.AdmissionDecision(
        admitted=True,
        model=model,
        num_ctx=num_ctx,
        num_gpu=num_gpu,
        num_parallel=num_parallel,
        action="admit",
        reason="partial_offload",
        caller="chat",
        requested_ctx=num_ctx,
        load_expected=True,
        gpu_share=8.5 / 11.0,
        vram_cost_gb=8.5,
        ram_cost_gb=2.5,
        gpu_layers=num_gpu,
    )


def _whole(rg, model, *, num_ctx=4096):
    return rg.AdmissionDecision(
        admitted=True,
        model=model,
        num_ctx=num_ctx,
        action="admit",
        reason="fits",
        caller="chat",
        requested_ctx=num_ctx,
        load_expected=True,
    )


class _Ollama:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([{"message": {"content": "local"}, "done": True}])
        return {"message": {"content": "local"}}


class _Llama:
    """The in-process engine's model class; constructing it is a load."""

    built = []
    events = []

    def __init__(self, **kwargs):
        _Llama.built.append(kwargs)
        _Llama.events.append(("load", kwargs.get("n_ctx")))
        self.held_ctx = kwargs.get("n_ctx")

    def close(self):
        _Llama.events.append(("close", self.held_ctx))

    def create_chat_completion(self, **kwargs):
        if kwargs.get("stream"):
            return iter([{"choices": [{"delta": {"content": "served"}, "finish_reason": "stop"}]}])
        return {"choices": [{"message": {"content": "served"}}]}


def _llama_engine(loaded):
    """The backend module with the fake engine class and a model directory."""
    mod = loaded[_BACKEND]
    mod._live_mode = lambda: "daily"
    mod.LLAMA_CPP_AVAILABLE = True
    # The file's integrity gate is not what these contracts are about.
    mod._provenance_guard = lambda path: None
    _Llama.built = []
    _Llama.events = []
    mod._LlamaCpp = _Llama
    models = _tmp()
    (models / "big.gguf").write_bytes(b"GGUF")
    return mod, [str(models)]


_MESSAGES = [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# GP1-GP6 -- the tensor table, read from the file
# ---------------------------------------------------------------------------


def test_gp1_each_ggml_type_reads_to_its_exact_bytes_and_any_other_id_is_refused_by_name():
    mm = _with_mm()[_MM]
    tmp = _tmp()
    kinds = sorted(_GGML)
    # Two blocks per row and three rows of each type, one type per block.
    tensors = [(f"blk.{i}.weight", kind, (_GGML[kind][0] * 2, 3)) for i, kind in enumerate(kinds)]
    table = mm.read_gguf_tensors(_gguf(tmp / "types.gguf", tensors))
    expected = {i: _GGML[kind][1] * 6 for i, kind in enumerate(kinds)}
    assert len(expected) == 34
    assert dict(table.blocks) == expected
    assert table.total_bytes == sum(expected.values())
    assert table.other_bytes == 0
    # Ids ggml retired, ids past its last type, and the largest id.
    for kind in (4, 5, 31, 32, 33, 36, 37, 38, 42, 255, 2 ** 32 - 1):
        path = _gguf(tmp / f"type-{kind}.gguf", [("blk.0.weight", kind, (256, 1))])
        assert _refused(mm, path) == "unknown_tensor_type", kind


def test_gp2_the_bytes_are_summed_per_block_outside_the_blocks_and_for_the_output_head():
    mm = _with_mm()[_MM]
    tmp = _tmp()
    tensors = [
        ("token_embd.weight", _Q4_K, (512, 8)),  # 16 blocks of 144: 2304
        ("blk.0.attn_q.weight", _Q4_K, (256, 4)),  # 576
        ("blk.0.ffn_up.weight", _Q6_K, (256, 2)),  # 420
        ("blk.1.attn_q.weight", _Q4_K, (256, 8)),  # 1152
        ("blk.1.attn_norm.weight", _F32, (256,)),  # 1024
        ("output_norm.weight", _F32, (256,)),  # 1024
        ("output.weight", _Q6_K, (256, 8)),  # 1680
        # Names that only look like a block's are outside the blocks.
        ("blk.x.weight", _F32, (8,)),
        ("blk.01.weight", _F32, (8,)),
        ("v.blk.0.weight", _F32, (8,)),
        ("blk.2", _F32, (8,)),
    ]
    table = mm.read_gguf_tensors(_gguf(tmp / "head.gguf", tensors))
    assert dict(table.blocks) == {0: 576 + 420, 1: 1152 + 1024}
    assert table.other_bytes == 2304 + 1024 + 1680 + 4 * 32
    assert table.total_bytes == 576 + 420 + 1152 + 1024 + 2304 + 1024 + 1680 + 4 * 32
    assert (table.output_bytes, table.output_tied) == (1680, False)
    assert (table.architecture, table.alignment, table.tensor_count, table.version) == ("llama", 32, 11, 3)
    # No output.weight: the token embedding is the head, tied.
    tied = [t for t in tensors if t[0] != "output.weight"]
    table = mm.read_gguf_tensors(_gguf(tmp / "tied.gguf", tied))
    assert (table.output_bytes, table.output_tied) == (2304, True)
    # Neither: no head.
    bare = [t for t in tied if t[0] != "token_embd.weight"]
    table = mm.read_gguf_tensors(_gguf(tmp / "bare.gguf", bare))
    assert (table.output_bytes, table.output_tied) == (0, False)
    # Version 2 lays the table out the same way.
    table = mm.read_gguf_tensors(_gguf(tmp / "v2.gguf", tensors, version=2))
    assert (table.version, dict(table.blocks)) == (2, {0: 996, 1: 2176})


def test_gp3_every_metadata_value_is_skipped_unkept_and_the_alignment_places_the_data():
    mm = _with_mm()[_MM]
    tmp = _tmp()
    meta = {
        "t.u8": _Typed(0, b"\xff"),
        "t.i8": _Typed(1, b"\x80"),
        "t.u16": _Typed(2, b"\xff\xff"),
        "t.i16": _Typed(3, b"\x00\x80"),
        "t.u32": 7,
        "t.i32": _Typed(5, struct.pack("<i", -7)),
        "t.f32": 1.5,
        # A boolean byte that is neither 0 nor 1.
        "t.bool": _Typed(7, b"\x02"),
        "t.str": "text",
        "t.u64": _Typed(10, struct.pack("<Q", 2 ** 64 - 1)),
        "t.i64": _Typed(11, struct.pack("<q", -1)),
        "t.f64": _Typed(12, struct.pack("<d", float("nan"))),
        # Strings a strict decoder refuses, a long array and a long template.
        "tokenizer.ggml.tokens": [b"\xff\xfe", b"", "a"] * 20000,
        "tokenizer.ggml.scores": [0.5] * 1000,
        "tokenizer.chat_template": "x" * (2 * 1024 * 1024),
        "general.name": b"\xc3\x28",
        # Four levels of arrays, and an empty one.
        "t.nested": [[[[1, 2], [3]]], [[[4]]]],
        "t.empty": [],
    }
    tensors = [("blk.0.weight", _F32, (8,)), ("blk.1.weight", _Q4_0, (32, 3))]
    table = mm.read_gguf_tensors(_gguf(tmp / "meta.gguf", tensors, meta, alignment=64))
    assert dict(table.blocks) == {0: 32, 1: 54}
    assert (table.architecture, table.alignment) == ("llama", 64)
    assert table.data_offset % 64 == 0
    # The table keeps what the plan needs, and no metadata mapping.
    assert {f.name for f in dataclasses.fields(table)} == {
        "version", "architecture", "alignment", "data_offset", "tensor_count",
        "blocks", "other_bytes", "output_bytes", "output_tied", "total_bytes",
    }
    # Without the key the alignment is ggml's default, 32.
    table = mm.read_gguf_tensors(_gguf(tmp / "plain.gguf", tensors))
    assert (table.alignment, table.data_offset % 32) == (32, 0)


def _holes(path, pieces):
    """Write bytes, and an int as a hole of that many bytes."""
    with open(path, "wb") as out:
        for piece in pieces:
            if isinstance(piece, int):
                out.seek(piece, os.SEEK_CUR)
            else:
                out.write(piece)
        out.truncate()
    return path


def test_gp4_a_corpus_of_damaged_files_is_refused_each_by_its_name():
    mm = _with_mm()[_MM]
    tmp = _tmp()
    arch = {"general.architecture": "llama"}
    base = [("blk.0.weight", _F32, (8, 4))]  # 128 bytes

    def file(tensors=base, meta=arch, **kw):
        # A header whose data no file can hold is written alone.
        header, _data, length = _layout(tensors, meta, **kw)
        return header, length if length < 2 ** 40 else None

    good, length = file()
    _header, data, _end = _layout(base + base[:1], arch)
    corpus = [
        ("bad_magic", b"GGUX" + good[4:], length),
        ("unsupported_version", *file(version=1)),
        ("unsupported_version", *file(version=4)),
        ("unsupported_version", *file(version=0x03000000)),
        ("too_many_tensors", *file(counts=(2 ** 32, 1))),
        ("too_many_pairs", *file(counts=(1, 2 ** 32))),
        ("name_too_long", *file(tensors=[("blk.0." + "w" * 58, _F32, (8,))])),
        ("name_too_long", *file(meta={**arch, "k" * 65536: 1})),
        ("string_too_long", *file(meta={"general.architecture": "a" * 257})),
        ("array_too_long", *file(meta={**arch, "big": _Typed(9, struct.pack("<IQ", 0, 2 ** 24 + 1))})),
        ("nesting_too_deep", *file(meta={**arch, "deep": [[[[[1]]]]]})),
        ("unknown_value_type", *file(meta={**arch, "odd": _Typed(13, b"")})),
        ("unknown_value_type", *file(meta={**arch, "odd": _Typed(9, struct.pack("<IQ", 13, 1))})),
        ("bad_alignment", *file(meta={**arch, "general.alignment": 0})),
        ("bad_alignment", *file(meta={**arch, "general.alignment": 48})),
        ("bad_alignment", *file(meta={**arch, "general.alignment": "32"})),
        ("bad_alignment", *file(meta={**arch, "general.alignment": _Typed(10, struct.pack("<Q", 32))})),
        ("bad_dimensions", *file(tensors=[("blk.0.weight", _F32, (1, 1, 1, 1, 1))])),
        ("bad_dimensions", *file(tensors=[("blk.0.weight", _F32, (2 ** 63,))])),
        ("bad_dimensions", *file(tensors=[("blk.0.weight", _F32, (2 ** 32, 2 ** 32))])),
        ("unknown_tensor_type", *file(tensors=[("blk.0.weight", 4, (32,))])),
        ("partial_block", *file(tensors=[("blk.0.weight", _Q4_0, (33,))])),
        ("misaligned_offset", *file(offsets=[16])),
        ("past_end_of_file", good, length - 1),
        # Two tensors over the same bytes claim more than the file holds.
        ("past_end_of_file", file(tensors=base + [("blk.1.weight", _F32, (8, 4))], offsets=[0, 0])[0], data + 128),
        ("duplicate_tensor", *file(tensors=base + base)),
        ("truncated", good[:20], None),
    ]
    reasons = set()
    for index, (reason, header, size) in enumerate(corpus):
        path = tmp / f"damaged-{index}.gguf"
        path.write_bytes(header)
        if size is not None:
            os.truncate(path, size)
        assert _refused(mm, path) == reason, (index, reason)
        reasons.add(reason)
    assert reasons == _REFUSALS
    assert set(mm.GGUF_TABLE_REFUSALS) == _REFUSALS
    # What cannot name a file at all is a parse error, not another kind.
    for path in ("bad" + chr(0) + ".gguf", None, 3.5):
        with pytest.raises(mm.GGUFParseError):
            mm.read_gguf_tensors(path)
    # The element bound counts every array of the header together: two
    # arrays of 2 ** 23 bytes reach it, one element more passes it.
    head = b"GGUF" + struct.pack("<IQQ", 3, 0, 3) + _str("general.architecture") + struct.pack("<I", 8) + _str("llama")

    def u8_array(key, n):
        return _str(key) + struct.pack("<IIQ", 9, 0, n)

    pieces = [head, u8_array("a", 2 ** 23), 2 ** 23, u8_array("b", 2 ** 23), 2 ** 23]
    at_bound = mm.read_gguf_tensors(_holes(tmp / "bound.gguf", pieces))
    assert (at_bound.tensor_count, dict(at_bound.blocks)) == (0, {})
    over = b"GGUF" + struct.pack("<IQQ", 3, 0, 4) + head[24:]
    path = _holes(tmp / "over.gguf", [over] + pieces[1:] + [u8_array("c", 1), b"\x00"])
    assert _refused(mm, path) == "array_too_long"


def _small(tmp):
    """A small whole file: three tensors, the last one ending the file."""
    tensors = [
        ("token_embd.weight", _F32, (8, 2)),  # 64 bytes
        ("blk.0.attn_q.weight", _Q4_0, (32, 2)),  # 36 bytes, padded to 64
        ("output.weight", _F32, (8,)),  # 32 bytes, ends the file
    ]
    meta = {"general.alignment": 32, "tokenizer.ggml.tokens": ["a", "bb", "ccc"], "llama.block_count": 1}
    return _gguf(tmp / "whole.gguf", tensors, meta)


def test_gp5_a_file_cut_at_any_byte_is_refused_by_name_and_never_otherwise():
    mm = _with_mm()[_MM]
    tmp = _tmp()
    whole = _small(tmp).read_bytes()
    assert mm.read_gguf_tensors(tmp / "whole.gguf").total_bytes == 64 + 36 + 32
    cut = tmp / "cut.gguf"
    reasons = set()
    for size in range(len(whole)):
        cut.write_bytes(whole[:size])
        try:
            mm.read_gguf_tensors(cut)
        except mm.GGUFTableError as refused:
            reasons.add(refused.reason)
            continue
        pytest.fail(f"a file cut at byte {size} of {len(whole)} was read")
    assert reasons == {"truncated", "past_end_of_file"}


def test_gp6_a_file_with_any_byte_inverted_is_read_or_refused_by_name():
    mm = _with_mm()[_MM]
    tmp = _tmp()
    whole = _small(tmp).read_bytes()
    flipped = tmp / "flipped.gguf"
    outcomes = {"read": 0, "refused": 0}
    for at in range(len(whole)):
        damaged = bytearray(whole)
        damaged[at] ^= 0xFF
        flipped.write_bytes(bytes(damaged))
        try:
            mm.read_gguf_tensors(flipped)
        except mm.GGUFTableError as refused:
            assert refused.reason in _REFUSALS, (at, refused.reason)
            outcomes["refused"] += 1
        else:
            outcomes["read"] += 1
    assert outcomes["read"] > 0 and outcomes["refused"] > 0, outcomes


# ---------------------------------------------------------------------------
# GP7-GP14 -- the plan
# ---------------------------------------------------------------------------


def test_gp7_the_geometry_carries_each_layer_s_kv_and_a_split_carries_it_with_its_layer():
    loaded = _with_mm()
    rg = loaded[_RG]
    # 8 KV heads of 128 + 128, f16: 4096 bytes per token in each layer.
    geometry = rg._kv_geometry_from_metadata(_llama_info(4))
    assert geometry["kv_bytes_per_layer"] == (4096,) * 4
    assert geometry["kv_bytes_per_token"] == 4 * 4096
    # A KV head count per layer: each layer its own.
    geometry = rg._kv_geometry_from_metadata(_llama_info(4, kv_heads=[0, 8, 0, 16]))
    assert geometry["kv_bytes_per_layer"] == (0, 4096, 0, 8192)
    assert geometry["kv_bytes_per_token"] == 12288
    tmp = _tmp()
    # Four blocks of 1 GiB; at 8192 tokens the last two layers hold 0.125
    # GiB of KV each and the first two none; a 2.3 GiB budget.
    path = _model(tmp / "m.gguf", [1.0] * 4, kv_heads=[0, 0, 32, 32])
    gov = _plan(rg, path, 5.0, capacity=3.8)
    decision = gov.admit("m", requested_ctx=8192, caller="chat")
    # The last two blocks and their KV: 2 x 1.125 on the GPU, the rest (the
    # first two blocks, the embedding and the head) in RAM.
    assert (decision.num_gpu, decision.vram_cost_gb, decision.ram_cost_gb) == (2, 2.25, 3.0)
    # The operator's coefficient prices the same 0.25 GiB, spread evenly.
    gov = _plan(rg, path, 5.0, capacity=3.8, kv_override_models={"m": 0.03125})
    decision = gov.admit("m", requested_ctx=8192, caller="chat")
    assert (decision.num_gpu, decision.vram_cost_gb, decision.ram_cost_gb) == (2, 2.125, 3.125)


def test_gp8_the_llama_cpp_file_is_read_once_per_identity_again_when_it_changes_never_for_a_whole_load():
    loaded = _with_mm()
    rg, mm = loaded[_RG], loaded[_MM]
    reads = _counting(mm)
    tmp = _tmp()
    path = _model(tmp / "m.gguf", [1.0] * 8)
    gov = _plan(rg, path, 10.0)
    assert gov.admit("m", requested_ctx=4096, caller="chat").num_gpu == 7
    assert gov.admit("m", requested_ctx=4096, caller="chat").num_gpu == 7
    assert reads == [str(path)]
    assert gov.tensor_table("m").total_bytes == 9 * _GIB
    assert reads == [str(path)]
    # The file changes: read again, and the plan follows it. Two blocks of
    # 2 GiB at the end now: 2 x 2.015625 + 4 x 1.015625 fit 8.5 GiB.
    _model(path, [1.0] * 6 + [2.0, 2.0])
    assert gov.admit("m", requested_ctx=4096, caller="chat").num_gpu == 6
    assert reads == [str(path)] * 2
    # Same size, another modification time: another file.
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert gov.admit("m", requested_ctx=4096, caller="chat").num_gpu == 6
    assert reads == [str(path)] * 3
    # A model the GPU holds whole reads no table.
    whole = _plan(rg, path, 4.0)
    decision = whole.admit("m", requested_ctx=4096, caller="chat")
    assert (decision.partial_offload, decision.num_gpu) == (False, None)
    assert reads == [str(path)] * 3


def test_gp9_ollama_s_model_file_is_the_blob_its_modelfile_names_and_no_other_path_is_opened():
    loaded = _with_mm()
    rg, mm = loaded[_RG], loaded[_MM]
    reads = _counting(mm)
    tmp = _tmp()
    blobs = _blob_store(tmp)
    model = _model(blobs / ("sha256-" + "a" * 64), [1.0] * 8)
    # A blob of another architecture, with blocks of its own: set aside for
    # its architecture alone.
    projector = _gguf(blobs / ("sha256-" + "b" * 64), [("blk.0.attn.weight", _F32, (8,))], arch="clip")
    decoy = _model(tmp / "decoy.gguf", [1.0] * 2)
    link = blobs / ("sha256-" + "c" * 64)
    link.symlink_to(model)
    modelfile = "\n".join(
        [
            '# Modelfile generated by "ollama show"',
            f"# FROM {model}",
            f"FROM {decoy}",
            "FROM llama3:8b",
            f"FROM {blobs}/../blobs/sha256-{'a' * 64}",
            f"FROM models/blobs/sha256-{'a' * 64}",
            f"FROM {link}",
            f"FROM {projector}",
            f"FROM {model}",
            'TEMPLATE """{{ .Prompt }}"""',
        ]
    )
    info = _Info(_llama_info(8), modelfile=modelfile)
    # Ollama's parallelism named, one sequence: unnamed, no count is told (GP31).
    config = _config(rg, ollama_num_parallel=1)
    gov = _estimating(_governor(rg, config, registry=_Registry(_Backend({"m": info}))), {"m": 10.0})
    _snapshot(rg, gov)
    table = gov.tensor_table("m")
    assert (len(table.blocks), table.architecture) == (8, "llama")
    # The projector is read to learn it is one; nothing else but the model.
    assert reads == [str(projector), str(model)]
    decision = gov.admit("m", requested_ctx=4096, caller="chat")
    assert (decision.num_gpu, decision.vram_cost_gb) == (7, 8.109)
    # Without the model's own blob, only the projector: no table.
    info.extra["modelfile"] = f"FROM {projector}\n"
    assert _governor(rg, _config(rg), registry=_Registry(_Backend({"m": info}))).tensor_table("m") is None


def test_gp10_the_last_blocks_that_fit_go_to_the_gpu_with_the_unexplained_cost():
    rg = _with_mm()[_RG]
    tmp = _tmp()
    # Eight blocks of 1 GiB, a 0.5 GiB embedding and a 0.5 GiB head: 9 GiB.
    path = _model(tmp / "m.gguf", [1.0] * 8)
    gov = _plan(rg, path, 10.0)
    decision = gov.admit("m", requested_ctx=4096, caller="chat")
    # 10.0 + 0.125 of KV is priced; the tensors and the KV explain 9.125,
    # and the 1.0 left goes to the GPU. Each block brings 1.015625 with its
    # KV: 1.0 + 7 x 1.015625 = 8.109375 fits the 8.5 free, 8 do not.
    assert decision.admitted is True
    assert (decision.reason, decision.num_ctx) == ("partial_offload", 4096)
    assert (decision.num_gpu, decision.gpu_layers) == (7, 7)
    assert decision.vram_cost_gb == 8.109
    assert decision.ram_cost_gb == 2.016
    assert decision.cost_gb == 10.125
    assert decision.gpu_share == pytest.approx(8.109375 / 10.125)
    assert decision.expected_slowdown is None  # no bandwidth is known


def test_gp11_blocks_of_unequal_sizes_are_counted_from_the_end():
    rg = _with_mm()[_RG]
    tmp = _tmp()
    # Six blocks of 0.5 GiB, then two of 2 GiB, each with 0.015625 of KV at
    # 4096 tokens; 6.25 GiB free.
    path = _model(tmp / "m.gguf", [0.5] * 6 + [2.0, 2.0])
    gov = _plan(rg, path, 9.0, capacity=7.75)
    decision = gov.admit("m", requested_ctx=4096, caller="chat")
    # From the end: 1.0 unexplained + 2 x 2.015625 + 2 x 0.515625 = 6.0625;
    # one more block is 6.578125. From the first block, seven would fit.
    assert decision.num_gpu == 4
    assert (decision.vram_cost_gb, decision.ram_cost_gb) == (round(6.0625, 3), round(3.0625, 3))


def test_gp12_a_cost_priced_under_the_model_s_tensors_is_raised_to_them():
    rg = _with_mm()[_RG]
    tmp = _tmp()
    # Eight blocks of 1.5 GiB and 1 GiB outside them: 13 GiB of tensors,
    # priced at 9.0.
    path = _model(tmp / "m.gguf", [1.5] * 8)
    gov = _plan(rg, path, 9.0)
    decision = gov.admit("m", requested_ctx=4096, caller="chat")
    # 13.0 + 0.125 of KV; 5 x 1.515625 = 7.578125 fits, 6 do not.
    assert decision.cost_gb == 13.125
    assert (decision.num_gpu, decision.vram_cost_gb, decision.ram_cost_gb) == (5, 7.578, 5.547)
    # A draft of 1.0 loads beside it: priced on top of the tensors, not
    # inside the room the floor makes. 1.0 + 4 x 1.515625 fits, 5 do not.
    decision = gov.admit("m", requested_ctx=4096, caller="chat", extra_models=["draft"])
    assert (decision.cost_gb, decision.num_gpu, decision.vram_cost_gb) == (14.125, 4, 7.062)


def test_gp13_when_no_planned_split_holds_the_refusal_names_what_the_plan_lacked():
    rg = _with_mm()[_RG]
    tmp = _tmp()
    # Blocks of 4 GiB and 3.0 GiB free: not one layer fits.
    path = _model(tmp / "big.gguf", [4.0] * 8)
    gov = _plan(rg, path, 33.0, capacity=4.5)
    decision = gov.admit("m", requested_ctx=4096, caller="chat")
    assert (decision.admitted, decision.num_gpu) == (False, None)
    assert decision.reason == "vram_insufficient+no_layer_fits"
    # The last step of the ladder, 2048: 33.0 + 0.0625 against 3.0 free.
    assert decision.shortfall_gb == round(33.0625 - 3.0, 3)
    assert decision.ram_shortfall_gb == 0.0
    payload = decision.refusal_payload()
    assert "VRAM on the GPU alone" in payload["message"]
    assert "not one of its layers fits" in payload["message"]
    # The same model without its table is split as before, evenly.
    untabled = _untabled(tmp / "untabled.gguf", 8)
    shared = _plan(rg, untabled, 33.0, capacity=4.5).admit("m", requested_ctx=4096, caller="chat")
    assert (shared.admitted, shared.partial_offload) == (True, True)
    # RAM short for the planned split, where the even one would hold: at
    # 2048, 1.0 + 7 x 1.0078125 on the GPU and 2.0078125 in RAM, where
    # 6000 MB less the 4 GiB reserve leaves 1.859375.
    path = _model(tmp / "m.gguf", [1.0] * 8)
    gov = _plan(rg, path, 10.0, ram_mb=6000.0)
    decision = gov.admit("m", requested_ctx=4096, caller="chat")
    assert decision.reason == "vram_insufficient+ram_insufficient"
    assert decision.ram_shortfall_gb == round(2.0078125 - 1.859375, 3)
    shared = _plan(rg, untabled, 10.0, ram_mb=6000.0).admit("m", requested_ctx=4096, caller="chat")
    assert (shared.admitted, shared.ram_cost_gb) == (True, 1.625)


def test_gp14_a_split_without_a_plan_is_the_one_given_before():
    loaded = _with_mm()
    rg = loaded[_RG]
    tmp = _tmp()
    # The geometry is read and the table refused: 8.5 on the GPU, the rest
    # in RAM, the layers in proportion, and no count sent.
    decision = _plan(rg, _untabled(tmp / "odd.gguf", 8), 10.0, **_SPEED).admit("m", requested_ctx=4096, caller="chat")
    shape = (decision.vram_cost_gb, decision.ram_cost_gb, decision.cost_gb, decision.reason)
    assert shape == (8.5, 1.625, 10.125, "partial_offload")
    assert (decision.num_gpu, decision.gpu_layers, decision.expected_slowdown) == (None, int(8.5 / 10.125 * 8), None)
    # No file named: the flat KV coefficient, 2.0 GiB at 4096, and no count.
    decision = _plan(rg, None, 10.0).admit("m", requested_ctx=4096, caller="chat")
    assert (decision.vram_cost_gb, decision.ram_cost_gb, decision.cost_gb) == (8.5, 3.5, 12.0)
    assert (decision.num_gpu, decision.gpu_layers) == (None, None)
    # An engine that names no file at all, Ollama without a modelfile.
    info = _Info(_llama_info(8))
    gov = _estimating(_governor(rg, _config(rg), registry=_Registry(_Backend({"m": info}))), {"m": 10.0})
    _snapshot(rg, gov)
    decision = gov.admit("m", requested_ctx=4096, caller="chat")
    assert (decision.num_gpu, decision.gpu_layers, decision.vram_cost_gb) == (None, 6, 8.5)
    # A readable table, but no context told: the engine would hold its own
    # default context's KV on layers priced without it.
    path = _model(tmp / "m.gguf", [1.0] * 8)
    reads = _counting(loaded[_MM])
    decision = _plan(rg, path, 10.0).admit("m", requested_ctx=None, caller="chat")
    assert (decision.vram_cost_gb, decision.ram_cost_gb, decision.num_gpu, decision.gpu_layers) == (8.5, 1.5, None, 6)
    # The operator's weights for the model replace the file's: no count.
    decision = _plan(rg, path, 10.0, override=10.0).admit("m", requested_ctx=4096, caller="chat")
    assert (decision.vram_cost_gb, decision.ram_cost_gb, decision.num_gpu) == (8.5, 1.625, None)
    assert reads == []


# ---------------------------------------------------------------------------
# GP15-GP17 -- the pin and the gate
# ---------------------------------------------------------------------------


def test_gp15_a_split_load_pins_its_layers_a_resident_carries_them_and_loads_evictions_and_the_view_clear_them():
    rg = _with_mm()[_RG]
    tmp = _tmp()
    # Ollama's parallelism named, one sequence: the count may be told (GP30).
    gov = _plan(rg, _model(tmp / "m.gguf", [1.0] * 8), 10.0, ollama_num_parallel=1)
    decision = gov.admit("m", requested_ctx=4096, caller="chat")
    rg._governor = gov
    try:
        with rg.ticket_scope(decision):
            rg.backend_admission_gate("m", {"num_ctx": 4096})
        assert gov.pinned_layers("m", 4096) == 7
        # Resident at the context it holds: the decision carries the pin.
        view = rg.LoadedModelView(
            name="m", size_vram_bytes=int(8.1 * _GIB), size_bytes=int(10.1 * _GIB), context_length=4096
        )
        _snapshot(rg, gov, in_use=8.1, loaded=[view])
        held = gov.admit("m", requested_ctx=4096, caller="chat")
        assert (held.reason, held.load_expected, held.num_gpu) == ("fits_resident", False, 7)
        # Held at another context than the pin was priced at: not carried.
        view.context_length = 8192
        held = gov.admit("m", requested_ctx=8192, caller="chat")
        assert (held.reason, held.num_ctx, held.num_gpu) == ("fits_resident", 8192, None)
        assert gov.pinned_layers("m", 8192) is None
        # A load clears it; the next split load pins its own count.
        gov.invalidate_on_load("m", 4096)
        assert gov.pinned_layers("m", 4096) is None
        # A caller that names its own num_gpu pins nothing.
        decision.load_expected = True
        with rg.ticket_scope(decision):
            rg.backend_admission_gate("m", {"num_ctx": 4096, "num_gpu": 3})
        assert gov.pinned_layers("m", 4096) is None
        gov.pin_layers("m", 7, 4096)
        gov.invalidate_on_evict("m")
        assert gov.pinned_layers("m", 4096) is None
        # An eviction that names no model may have unloaded any of them.
        gov.pin_layers("m", 7, 4096)
        gov.invalidate_on_evict(None)
        assert gov.pinned_layers("m", 4096) is None
        # Leaving the loaded view clears it, once the view has shown it: a
        # view that lags the load does not.
        warmup = _Warmup()
        gov._warmup = warmup
        gov.pin_layers("m", 7, 4096)
        gov.refresh(force=True)
        assert gov.pinned_layers("m", 4096) == 7
        warmup._loaded = [_resident("m", int(8.1 * _GIB), size=int(10.1 * _GIB), context_length=4096)]
        gov.refresh(force=True)
        assert gov.pinned_layers("m", 4096) == 7
        warmup._loaded = []
        gov.refresh(force=True)
        assert gov.pinned_layers("m", 4096) is None
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gp16_the_gate_returns_the_decision_it_acted_on():
    rg = _open()[_RG]
    backstop = _split(rg, "big", num_gpu=12)
    fake = _GateGovernor(backstop)
    rg._governor = fake
    try:
        assert rg.backend_admission_gate("big", {"num_ctx": 4096}) is backstop
        ticket = _split(rg, "big", num_gpu=12)
        with rg.ticket_scope(ticket):
            assert rg.backend_admission_gate("big", None) is ticket
        assert ticket.load_expected is False
        # Another model's ticket is not this call's: the backstop answers.
        with rg.ticket_scope(_split(rg, "other")):
            assert rg.backend_admission_gate("big", None) is backstop
        # Only the call that told the context the count was priced for pins
        # it; the two that told none were placed by the engine (GP29).
        assert fake.pins == [("big", 12, 4096)]
        fake.config.enabled = False
        assert rg.backend_admission_gate("big", None) is None
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gp17_a_split_is_refused_to_an_engine_that_cannot_split_only_without_a_layer_count():
    rg = _open()[_RG]
    name = "llama.cpp in process (n_gpu_layers -1 puts every layer on the GPU)"
    counted = _split(rg, "big", num_gpu=12)
    fake = _GateGovernor(counted)
    rg._governor = fake
    try:
        assert rg.backend_admission_gate("big", None, unsplittable=name) is counted
        ticket = _split(rg, "big", num_gpu=12)
        with rg.ticket_scope(ticket):
            assert rg.backend_admission_gate("big", None, unsplittable=name) is ticket
        assert fake.recorded == []
        assert fake.loads == [("big", 4096)] * 2
        # No layer count: refused by name, before any accounting.
        fake._decision = _split(rg, "big")
        with pytest.raises(rg.GovernorRefusal) as refused:
            rg.backend_admission_gate("big", None, unsplittable=name)
        assert refused.value.decision.reason == "partial_offload_unsupported"
        with rg.ticket_scope(_split(rg, "big")):
            with pytest.raises(rg.GovernorRefusal):
                rg.backend_admission_gate("big", None, unsplittable=name)
        assert [d.reason for d in fake.recorded] == ["partial_offload_unsupported"] * 2
        assert fake.loads == [("big", 4096)] * 2
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


# ---------------------------------------------------------------------------
# GP18-GP22 -- the engines
# ---------------------------------------------------------------------------


def test_gp18_ollama_is_sent_the_decision_s_num_gpu_unless_the_caller_names_its_own():
    loaded = _open((_BACKEND, source("inference_backend.py")))
    rg, mod = loaded[_RG], loaded[_BACKEND]
    mod._live_mode = lambda: "daily"
    fake = _GateGovernor(_split(rg, "big", num_gpu=12))
    rg._governor = fake
    try:
        client = _Ollama()
        mod.OLLAMA_AVAILABLE = True
        mod._ollama_module = client
        backend = mod.OllamaBackend()
        options = {"num_ctx": 4096}
        assert backend.generate("big", _MESSAGES, options=options).content == "local"
        assert client.calls[-1]["options"] == {"num_ctx": 4096, "num_gpu": 12}
        assert options == {"num_ctx": 4096}  # the caller's options stay its own
        assert [chunk.content for chunk in backend.stream("big", _MESSAGES, options=options)] == ["local"]
        assert client.calls[-1]["options"] == {"num_ctx": 4096, "num_gpu": 12}
        # No context told: Ollama would load at a default of its own, whose KV
        # the count never placed; it places the layers itself.
        backend.generate("big", _MESSAGES)
        assert client.calls[-1]["options"] == {}
        # The caller's own count is kept.
        backend.generate("big", _MESSAGES, options={"num_ctx": 4096, "num_gpu": 3})
        assert client.calls[-1]["options"] == {"num_ctx": 4096, "num_gpu": 3}
        list(backend.stream("big", _MESSAGES, options={"num_gpu": 0}))
        assert client.calls[-1]["options"] == {"num_gpu": 0}
        # A whole load: nothing beyond the context.
        fake._decision = _whole(rg, "big")
        backend.generate("big", _MESSAGES, options={"num_ctx": 4096})
        assert client.calls[-1]["options"] == {"num_ctx": 4096}
        list(backend.stream("big", _MESSAGES, options={"num_ctx": 4096}))
        assert client.calls[-1]["options"] == {"num_ctx": 4096}
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gp19_the_pinned_count_rides_the_calls_that_follow_the_split_load():
    loaded = _open((_MM, source("model_manager.py")), (_BACKEND, source("inference_backend.py")))
    rg, mod = loaded[_RG], loaded[_BACKEND]
    mod._live_mode = lambda: "daily"
    tmp = _tmp()
    model = _model(_blob_store(tmp) / ("sha256-" + "d" * 64), [1.0] * 8)
    info = _Info(_llama_info(8), modelfile=f"FROM {model}\n")
    # Ollama's parallelism named, one sequence: unnamed, no count is told (GP31).
    config = _config(rg, ollama_num_parallel=1)
    gov = _estimating(_governor(rg, config, registry=_Registry(_Backend({"m": info}))), {"m": 10.0})
    _snapshot(rg, gov)
    rg._governor = gov
    try:
        client = _Ollama()
        mod.OLLAMA_AVAILABLE = True
        mod._ollama_module = client
        backend = mod.OllamaBackend()
        backend.generate("m", _MESSAGES, options={"num_ctx": 4096})
        assert client.calls[0]["options"] == {"num_ctx": 4096, "num_gpu": 7}
        assert gov.pinned_layers("m", 4096) == 7
        # Ollama now reports it loaded, split, at the context it holds.
        view = rg.LoadedModelView(
            name="m", size_vram_bytes=int(8.1 * _GIB), size_bytes=int(10.1 * _GIB), context_length=4096
        )
        _snapshot(rg, gov, in_use=8.1, loaded=[view])
        backend.generate("m", _MESSAGES, options={"num_ctx": 4096})
        list(backend.stream("m", _MESSAGES, options={"num_ctx": 4096}))
        assert [call["options"].get("num_gpu") for call in client.calls] == [7, 7, 7]
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gp20_llama_cpp_loads_at_the_admitted_context_and_layers_under_the_operator_s_count():
    loaded = _open((_BACKEND, source("inference_backend.py")))
    rg = loaded[_RG]
    mod, dirs = _llama_engine(loaded)
    fake = _GateGovernor(_split(rg, "big.gguf", num_gpu=12, num_ctx=8192))
    rg._governor = fake

    def placed():
        return [(built["n_ctx"], built["n_gpu_layers"]) for built in _Llama.built]

    try:
        assert mod.LlamaCppBackend(model_dirs=dirs).generate("big.gguf", _MESSAGES).content == "served"
        assert placed() == [(8192, 12)]
        assert fake.recorded == []  # a counted split is no refusal
        # The operator's count is a ceiling a split priced at the admitted
        # context can lower, never raise.
        mod.LlamaCppBackend(model_dirs=dirs, n_gpu_layers=20).generate("big.gguf", _MESSAGES)
        assert placed()[-1] == (8192, 12)
        mod.LlamaCppBackend(model_dirs=dirs, n_gpu_layers=8).generate("big.gguf", _MESSAGES)
        assert placed()[-1] == (8192, 8)
        assert [c.content for c in mod.LlamaCppBackend(model_dirs=dirs).stream("big.gguf", _MESSAGES)] == ["served"]
        assert placed()[-1] == (8192, 12)
        # A whole admission: its context, every layer on the GPU, or the
        # operator's count.
        fake._decision = _whole(rg, "big.gguf", num_ctx=6144)
        mod.LlamaCppBackend(model_dirs=dirs).generate("big.gguf", _MESSAGES)
        assert placed()[-1] == (6144, -1)
        mod.LlamaCppBackend(model_dirs=dirs, n_gpu_layers=20).generate("big.gguf", _MESSAGES)
        assert placed()[-1] == (6144, 20)
        # No decision at all: what the backend was configured with.
        fake.config.enabled = False
        mod.LlamaCppBackend(model_dirs=dirs, n_ctx=3072).generate("big.gguf", _MESSAGES)
        assert placed()[-1] == (3072, -1)
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gp21_a_held_model_is_loaded_again_only_whole_for_a_longer_context_the_old_one_closed_first():
    loaded = _open((_BACKEND, source("inference_backend.py")))
    rg = loaded[_RG]
    mod, dirs = _llama_engine(loaded)
    fake = _GateGovernor(_whole(rg, "big.gguf", num_ctx=4096))
    rg._governor = fake
    try:
        backend = mod.LlamaCppBackend(model_dirs=dirs)
        backend.generate("big.gguf", _MESSAGES)
        fake._decision = _whole(rg, "big.gguf", num_ctx=2048)
        backend.generate("big.gguf", _MESSAGES)
        assert _Llama.events == [("load", 4096)]
        # A split for a held model counts the model's own memory against it
        # (the governor does not see what this backend holds): served as held.
        fake._decision = _split(rg, "big.gguf", num_gpu=12, num_ctx=8192)
        list(backend.stream("big.gguf", _MESSAGES))
        assert _Llama.events == [("load", 4096)]
        fake._decision = _whole(rg, "big.gguf", num_ctx=8192)
        list(backend.stream("big.gguf", _MESSAGES))
        assert _Llama.events == [("load", 4096), ("close", 4096), ("load", 8192)]
        assert _Llama.built[-1]["n_gpu_layers"] == -1
        # A decision without a context, or none at all, loads nothing.
        fake._decision = _whole(rg, "big.gguf", num_ctx=None)
        backend.generate("big.gguf", _MESSAGES)
        fake.config.enabled = False
        backend.generate("big.gguf", _MESSAGES)
        assert len(_Llama.events) == 3
        # A model held at a context the backend never saw is served as held.
        fake.config.enabled = True
        fake._decision = _whole(rg, "other.gguf", num_ctx=32768)
        backend._loaded_models["other.gguf"] = _Llama.__new__(_Llama)
        assert backend.generate("other.gguf", _MESSAGES).content == "served"
        assert len(_Llama.events) == 3
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gp22_the_loaded_view_carries_the_context_each_model_is_held_at():
    loaded = _open((_BACKEND, source("inference_backend.py")))
    rg = loaded[_RG]
    mod, dirs = _llama_engine(loaded)
    fake = _GateGovernor(_whole(rg, "big.gguf", num_ctx=8192))
    rg._governor = fake
    try:
        backend = mod.LlamaCppBackend(model_dirs=dirs)
        backend.generate("big.gguf", _MESSAGES)
        backend._loaded_models["other.gguf"] = _Llama.__new__(_Llama)
        view = sorted((m.name, m.backend, m.context_length) for m in backend.loaded_models())
        assert view == [("big.gguf", "llama_cpp", 8192), ("other.gguf", "llama_cpp", None)]
        assert backend.unload_model("big.gguf") is True
        assert [(m.name, m.context_length) for m in backend.loaded_models()] == [("other.gguf", None)]
        # Loaded again without a decision: the configured context.
        fake.config.enabled = False
        backend.generate("big.gguf", _MESSAGES)
        assert ("big.gguf", 4096) in [(m.name, m.context_length) for m in backend.loaded_models()]
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


# ---------------------------------------------------------------------------
# GP23-GP27 -- the expected speed
# ---------------------------------------------------------------------------


_SPEED = {"split_gpu_bandwidth_gbs": 1000.0, "split_ram_bandwidth_gbs": 100.0}


def test_gp23_the_expected_slowdown_is_each_side_s_bytes_over_its_bandwidth_against_the_gpu_alone():
    rg = _with_mm()[_RG]
    tmp = _tmp()
    path = _model(tmp / "m.gguf", [1.0] * 8)
    decision = _plan(rg, path, 10.0, **_SPEED).admit("m", requested_ctx=4096, caller="chat")
    # Read per token: on the GPU, 7 blocks and their KV, 7.109375 GiB; in
    # RAM, block 0 and its KV, 1.015625, and the output head, 0.5.
    gpu, ram = 7 * 1.015625, 1.015625 + 0.5
    assert decision.expected_slowdown == round((gpu / 1000.0 + ram / 100.0) / ((gpu + ram) / 1000.0), 3)
    assert decision.expected_slowdown == 2.582
    # A tied head is the token embedding, read in RAM.
    tied = _model(tmp / "tied.gguf", [1.0] * 8, embd_gib=1.0, head_gib=0.0)
    decision = _plan(rg, tied, 10.0, **_SPEED).admit("m", requested_ctx=4096, caller="chat")
    gpu, ram = 7 * 1.015625, 1.015625 + 1.0
    assert decision.expected_slowdown == round((gpu + ram * 10.0) / (gpu + ram), 3)
    # One bandwidth unknown, a whole load, or no table: no figure.
    assert _plan(rg, path, 10.0, split_gpu_bandwidth_gbs=1000.0).admit(
        "m", requested_ctx=4096, caller="chat"
    ).expected_slowdown is None
    assert _plan(rg, path, 4.0, **_SPEED).admit("m", requested_ctx=4096, caller="chat").expected_slowdown is None
    assert _plan(rg, None, 10.0, **_SPEED).admit("m", requested_ctx=4096, caller="chat").expected_slowdown is None


def test_gp24_a_split_slower_than_max_slowdown_does_not_hold_and_an_unknown_speed_never_refuses():
    rg = _with_mm()[_RG]
    tmp = _tmp()
    # 32 KV heads: 0.125 GiB of KV per layer at 8192 tokens, 0.0625 at 4096.
    path = _model(tmp / "m.gguf", [1.0] * 8, kv_heads=32)
    # At 8192 the split takes 7 blocks of 1.125 and runs (7.875 + 1.625 x
    # 10) / 9.5 = 2.54 times slower; at 4096 the GPU alone holds 8.0 + 0.5.
    slow = round((7.875 + 16.25) / 9.5, 3)
    decision = _plan(rg, path, 8.0, split_max_slowdown=3.0, **_SPEED).admit("m", requested_ctx=8192, caller="chat")
    assert (decision.num_ctx, decision.num_gpu, decision.expected_slowdown) == (8192, 7, slow)
    decision = _plan(rg, path, 8.0, split_max_slowdown=2.0, **_SPEED).admit("m", requested_ctx=8192, caller="chat")
    assert (decision.admitted, decision.action, decision.num_ctx) == (True, "downsize", 4096)
    assert decision.partial_offload is False
    # A speed that cannot be told never refuses.
    decision = _plan(rg, path, 8.0, split_max_slowdown=2.0).admit("m", requested_ctx=8192, caller="chat")
    assert (decision.num_ctx, decision.num_gpu, decision.expected_slowdown) == (8192, 7, None)
    # Too slow at every step and too large for the GPU alone: named.
    decision = _plan(rg, path, 12.0, split_max_slowdown=2.0, **_SPEED).admit("m", requested_ctx=8192, caller="chat")
    assert decision.admitted is False
    assert decision.reason == "vram_insufficient+split_too_slow"
    assert "slower" in decision.refusal_payload()["message"]


def _speed(cfg):
    return (cfg.split_gpu_bandwidth_gbs, cfg.split_ram_bandwidth_gbs, cfg.split_max_slowdown)


def test_gp25_the_split_speed_block_parses_holds_its_ranges_and_ships_null():
    rg = _open()[_RG]
    tmp = _tmp()
    path = tmp / "resource_governor.yaml"
    path.write_text(
        "split_speed:\n  gpu_bandwidth_gbs: 936.0\n  ram_bandwidth_gbs: 89.6\n  max_slowdown: 3.0\n",
        encoding="utf-8",
    )
    assert _speed(rg.load_config(path)) == (936.0, 89.6, 3.0)
    assert _speed(rg.load_config(tmp / "missing.yaml")) == (None, None, None)
    for bad in ("0.0", "-5", ".inf", ".nan", "fast", "true"):
        path.write_text(
            f"split_speed:\n  gpu_bandwidth_gbs: {bad}\n  ram_bandwidth_gbs: {bad}\n  max_slowdown: {bad}\n",
            encoding="utf-8",
        )
        assert _speed(rg.load_config(path)) == (None, None, None), bad
    path.write_text("split_speed:\n  max_slowdown: 0.5\n", encoding="utf-8")
    assert _speed(rg.load_config(path)) == (None, None, None)
    path.write_text("split_speed:\n  max_slowdown: 1.0\n  gpu_bandwidth_gbs: null\n", encoding="utf-8")
    assert _speed(rg.load_config(path)) == (None, None, 1.0)
    shipped = Path(source("config", "resource_governor.yaml"))
    assert _speed(rg.load_config(shipped)) == (None, None, None)
    raw = yaml.safe_load(shipped.read_text(encoding="utf-8"))
    assert raw["split_speed"] == {"gpu_bandwidth_gbs": None, "ram_bandwidth_gbs": None, "max_slowdown": None}


def test_gp26_the_config_routes_show_and_write_the_three_keys_in_range_only():
    loaded = _open((_ROUTES, source("api", "routes_governor.py")), packages=("opti_oignon.api",))
    rg, routes = loaded[_RG], loaded[_ROUTES]
    gov = _governor(rg, rg.GovernorConfig())
    view = routes.config_read_payload(gov)
    assert view["config"]["split_speed"] == {"gpu_bandwidth_gbs": None, "ram_bandwidth_gbs": None, "max_slowdown": None}
    keys = {"split_speed.gpu_bandwidth_gbs", "split_speed.ram_bandwidth_gbs", "split_speed.max_slowdown"}
    assert keys <= set(view["writable_keys"])
    path = _tmp() / "resource_governor.yaml"
    path.write_text(Path(source("config", "resource_governor.yaml")).read_text(encoding="utf-8"), encoding="utf-8")
    written = {"split_speed.gpu_bandwidth_gbs": 936.0, "split_speed.ram_bandwidth_gbs": 89.6, "split_speed.max_slowdown": 2.5}
    out = routes.config_write_payload(gov.config, written, path, lambda: None, lambda changes: None)
    assert out["applied"]["split_speed.max_slowdown"] == {"old": None, "new": 2.5}
    assert _speed(rg.load_config(path)) == (936.0, 89.6, 2.5)
    routes.config_write_payload(
        rg.load_config(path), {"split_speed.max_slowdown": None}, path, lambda: None, lambda changes: None
    )
    assert _speed(rg.load_config(path)) == (936.0, 89.6, None)
    before = path.read_bytes()
    for bad in (
        {"split_speed.gpu_bandwidth_gbs": 0.0},
        {"split_speed.gpu_bandwidth_gbs": -1.0},
        {"split_speed.gpu_bandwidth_gbs": float("inf")},
        {"split_speed.ram_bandwidth_gbs": 0.0},
        {"split_speed.ram_bandwidth_gbs": True},
        {"split_speed.max_slowdown": 0.99},
        {"split_speed.max_slowdown": float("inf")},
        {"split_speed.max_slowdown": "fast"},
    ):
        with pytest.raises(routes.ConfigWriteError) as refused:
            routes.config_write_payload(gov.config, bad, path, lambda: None, lambda changes: None)
        assert refused.value.status_code == 400, bad
    assert path.read_bytes() == before


def test_gp27_the_decision_s_dictionary_carries_num_gpu_gpu_layers_and_the_expected_slowdown():
    rg = _with_mm()[_RG]
    tmp = _tmp()
    path = _model(tmp / "m.gguf", [1.0] * 8)
    decision = _plan(rg, path, 10.0, **_SPEED).admit("m", requested_ctx=4096, caller="chat")
    carried = decision.to_dict()
    assert (carried["num_gpu"], carried["gpu_layers"], carried["expected_slowdown"]) == (7, 7, 2.582)
    whole = _plan(rg, path, 4.0, **_SPEED).admit("m", requested_ctx=4096, caller="chat").to_dict()
    assert (whole["num_gpu"], whole["gpu_layers"], whole["expected_slowdown"]) == (None, None, None)


# ---------------------------------------------------------------------------
# GP28-GP39 -- the count and its context, and what the contracts above left open
# ---------------------------------------------------------------------------


def _ollama_plan(rg, *, kv_heads=8, ram_mb=64000.0, **fields):
    """A governor whose Ollama names model "m"'s blob in its modelfile: the
    eight 1 GiB blocks of GP10, estimated at 10.0 GiB, with 8.5 GiB free."""
    model = _model(_blob_store(_tmp()) / ("sha256-" + "e" * 64), [1.0] * 8, kv_heads=kv_heads)
    info = _Info(_llama_info(8, kv_heads=kv_heads), modelfile=f"FROM {model}\n")
    gov = _governor(rg, _config(rg, **fields), registry=_Registry(_Backend({"m": info})), ram_mb=ram_mb)
    _snapshot(rg, gov, ram_mb=ram_mb)
    return _estimating(gov, {"m": 10.0})


class _Declaring(_Backend):
    """llama.cpp, declaring what one request to its model costs."""

    def __init__(self, infos, declared):
        super().__init__(infos, name="llama_cpp")
        self._declared = dict(declared)

    def cost_model(self, model):
        return dict(self._declared)


def _declaring(rg, path, declared):
    """A governor whose llama.cpp names ``path`` for model "m", estimated at
    10.0 GiB, and declares ``declared`` for it (admitted with engine named)."""
    registry = _Registry(_Declaring({"m": _Info(path=str(path))}, declared))
    gov = _governor(rg, _config(rg), registry=registry)
    _snapshot(rg, gov)
    return _estimating(gov, {"m": 10.0})


def test_gp28_the_count_rides_only_with_the_context_it_was_priced_for():
    loaded = _open((_BACKEND, source("inference_backend.py")))
    rg, mod = loaded[_RG], loaded[_BACKEND]
    mod._live_mode = lambda: "daily"
    rg._governor = _GateGovernor(_split(rg, "big", num_gpu=12))
    try:
        client = _Ollama()
        mod.OLLAMA_AVAILABLE = True
        mod._ollama_module = client
        backend = mod.OllamaBackend()
        # Priced at 4096: at another context Ollama holds another KV on the
        # layers the count was chosen for, so it places them itself.
        backend.generate("big", _MESSAGES, options={"num_ctx": 8192})
        assert client.calls[-1]["options"] == {"num_ctx": 8192}
        list(backend.stream("big", _MESSAGES, options={"num_ctx": 2048, "temperature": 0.2}))
        assert client.calls[-1]["options"] == {"num_ctx": 2048, "temperature": 0.2}
        list(backend.stream("big", _MESSAGES, options={"num_ctx": 4096, "temperature": 0.2}))
        assert client.calls[-1]["options"] == {"num_ctx": 4096, "temperature": 0.2, "num_gpu": 12}
        # Admitted below what was asked (a ladder step, a clamp): the call still
        # tells the context it asked, which is not the one the count was priced for.
        clamped = _split(rg, "big", num_gpu=12)
        clamped.requested_ctx = 8192
        rg._governor._decision = clamped
        backend.generate("big", _MESSAGES, options={"num_ctx": 8192})
        assert client.calls[-1]["options"] == {"num_ctx": 8192}
        backend.generate("big", _MESSAGES, options={"num_ctx": 4096})
        assert client.calls[-1]["options"] == {"num_ctx": 4096, "num_gpu": 12}
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gp29_a_split_load_pins_its_count_only_when_the_count_rode_with_it():
    rg = _open()[_RG]
    fake = _GateGovernor(_split(rg, "big", num_gpu=12))
    rg._governor = fake
    try:
        # A load that tells no context, or another, is placed by the engine:
        # the count placed nothing, so nothing is pinned. Each is accounted.
        for options in (None, {}, {"num_predict": 0}, {"num_ctx": 8192}):
            with rg.ticket_scope(_split(rg, "big", num_gpu=12)):
                rg.backend_admission_gate("big", options)
            rg.backend_admission_gate("big", options)
        assert (fake.pins, len(fake.loads)) == ([], 8)
        # A shorter context, or the one asked when the admission went below it.
        clamped = _split(rg, "big", num_gpu=12)
        clamped.requested_ctx = 8192
        for ticket, options in ((_split(rg, "big", num_gpu=12), {"num_ctx": 2048}), (clamped, {"num_ctx": 8192})):
            with rg.ticket_scope(ticket):
                rg.backend_admission_gate("big", options)
        assert (fake.pins, len(fake.loads)) == ([], 10)
        with rg.ticket_scope(_split(rg, "big", num_gpu=12)):
            rg.backend_admission_gate("big", {"num_ctx": 4096})
        assert fake.pins == [("big", 12, 4096)]
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gp30_ollama_is_told_the_count_only_while_it_keeps_one_sequence():
    loaded = _open((_MM, source("model_manager.py")), (_BACKEND, source("inference_backend.py")))
    rg, mod = loaded[_RG], loaded[_BACKEND]
    # Ollama keeps num_ctx x num_parallel tokens of KV, and the plan prices one
    # sequence: named 1, the count is what Ollama will hold, and it is told.
    one = _ollama_plan(rg, ollama_num_parallel=1).admit("m", requested_ctx=4096, caller="chat")
    assert (one.num_gpu, one.num_parallel, one.ollama_layers({"num_ctx": 4096})) == (7, 1, 7)
    # Unnamed (as shipped), more than one, or no parallelism at all: priced
    # alike, the count is kept, and Ollama is told none.
    for other in (None, 4, 0, -2, True):
        decision = _ollama_plan(rg, ollama_num_parallel=other).admit("m", requested_ctx=4096, caller="chat")
        assert (decision.num_gpu, decision.vram_cost_gb, decision.cost_gb) == (7, 8.109, 10.125)
        assert decision.ollama_layers({"num_ctx": 4096}) is None
    # The head sends what the decision lets it tell.
    mod._live_mode = lambda: "daily"
    rg._governor = _GateGovernor(_split(rg, "big", num_gpu=12, num_parallel=None))
    try:
        client = _Ollama()
        mod.OLLAMA_AVAILABLE = True
        mod._ollama_module = client
        mod.OllamaBackend().generate("big", _MESSAGES, options={"num_ctx": 4096})
        assert client.calls[-1]["options"] == {"num_ctx": 4096}
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gp31_llama_cpp_keeps_the_count_whatever_ollama_s_parallelism():
    loaded = _open((_MM, source("model_manager.py")), (_BACKEND, source("inference_backend.py")))
    rg = loaded[_RG]
    mod, dirs = _llama_engine(loaded)
    # Placed from the blob Ollama names while its parallelism is unnamed (as
    # shipped): Ollama would be told no count; llama.cpp, should it be the
    # engine that serves (one sequence), loads with the count, not refuses it.
    decision = _ollama_plan(rg).admit("m", requested_ctx=4096, caller="chat")
    assert (decision.num_gpu, decision.ollama_layers({"num_ctx": 4096})) == (7, None)
    fake = _GateGovernor(decision)
    rg._governor = fake
    try:
        assert mod.LlamaCppBackend(model_dirs=dirs).generate("big.gguf", _MESSAGES).content == "served"
        assert (_Llama.built[-1]["n_ctx"], _Llama.built[-1]["n_gpu_layers"]) == (4096, 7)
        assert fake.recorded == []
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gp32_a_refusal_after_a_planned_split_is_figured_on_the_plan_s_total():
    rg = _with_mm()[_RG]
    # Eight blocks of 4 GiB, 33 GiB of tensors priced at 20.0: 3.0 GiB free.
    path = _model(_tmp() / "big.gguf", [4.0] * 8)
    decision = _plan(rg, path, 20.0, capacity=4.5).admit("m", requested_ctx=4096, caller="chat")
    assert (decision.admitted, decision.reason) == (False, "vram_insufficient+no_layer_fits")
    # The last step of the ladder, 2048: the tensors and 0.0625 of KV, not
    # the 20.0 priced under them, in the decision and in what it tells.
    assert (decision.cost_gb, decision.shortfall_gb) == (round(33.0625, 3), round(33.0625 - 3.0, 3))
    assert decision.refusal_payload()["shortfall_gb"] == round(33.0625 - 3.0, 3)


def test_gp33_what_is_not_a_regular_file_is_refused_as_such_and_a_link_is_not_followed():
    mm = _with_mm()[_MM]
    tmp = _tmp()
    model = _model(tmp / "m.gguf", [1.0] * 2)
    fifo = tmp / "fifo"
    os.mkfifo(fifo)
    link = tmp / "link.gguf"
    link.symlink_to(model)
    open_fds = len(os.listdir("/proc/self/fd"))
    for path in (tmp, fifo):
        with pytest.raises(mm.GGUFParseError) as refused:
            mm.read_gguf_tensors(path)
        assert type(refused.value) is mm.GGUFParseError  # not a table refusal
    with pytest.raises(OSError):
        mm.read_gguf_tensors(link)
    assert len(os.listdir("/proc/self/fd")) == open_fds  # every descriptor opened is closed
    assert len(mm.read_gguf_tensors(model).blocks) == 2


def test_gp34_weights_an_engine_declares_leave_no_count_to_tell_from_the_file():
    loaded = _with_mm()
    rg = loaded[_RG]
    path = _model(_tmp() / "m.gguf", [1.0] * 8)
    reads = _counting(loaded[_MM])
    gov = _declaring(rg, path, {"kind": "generator", "weights_gb": 10.0})
    decision = gov.admit("m", requested_ctx=4096, caller="chat", engine="llama_cpp")
    # The engine's 10.0 (experts it keeps in RAM, say) is not the file's 9.0:
    # the even split of GP14, and the table is never read.
    shape = (decision.reason, decision.num_gpu, decision.vram_cost_gb, decision.ram_cost_gb)
    assert shape == ("partial_offload", None, 8.5, 1.625)
    assert reads == []


def test_gp35_a_model_without_a_kv_cache_is_placed_layer_by_layer_with_no_context_told():
    rg = _with_mm()[_RG]
    path = _model(_tmp() / "m.gguf", [1.0] * 8)
    gov = _declaring(rg, path, {"kind": "encoder"})
    decision = gov.admit("m", requested_ctx=None, caller="chat", engine="llama_cpp")
    # 10.0 priced and no KV: 1.0 unexplained + 7 x 1.0 = 8.0 fits the 8.5 free.
    assert (decision.reason, decision.num_ctx, decision.num_gpu) == ("partial_offload", None, 7)
    assert (decision.vram_cost_gb, decision.ram_cost_gb, decision.cost_gb) == (8.0, 2.0, 10.0)


def test_gp36_a_declared_state_is_counted_once_in_the_planned_total():
    rg = _with_mm()[_RG]
    path = _model(_tmp() / "m.gguf", [1.0] * 8)
    gov = _declaring(rg, path, {"kind": "generator", "state_gb": 0.5})
    decision = gov.admit("m", requested_ctx=4096, caller="chat", engine="llama_cpp")
    # 10.0 + 0.5 of state + 0.125 of KV. The state stays on the GPU with what
    # the tensors do not explain: 1.5 + 6 x 1.015625 fits, a seventh does not.
    assert decision.cost_gb == 10.625
    gpu = 1.5 + 6 * 1.015625
    assert (decision.num_gpu, decision.vram_cost_gb, decision.ram_cost_gb) == (6, round(gpu, 3), round(10.625 - gpu, 3))


def test_gp37_a_resident_at_an_unknown_context_carries_no_pin_to_a_call_that_tells_none():
    rg = _with_mm()[_RG]
    gov = _plan(rg, _model(_tmp() / "m.gguf", [1.0] * 8), 10.0)
    gov.pin_layers("m", 7, 4096)
    # The engine reports no context for it: what it holds may not be what
    # the pin was priced at, and a call that tells none carries none.
    view = rg.LoadedModelView(
        name="m", size_vram_bytes=int(8.1 * _GIB), size_bytes=int(10.1 * _GIB), context_length=None
    )
    _snapshot(rg, gov, in_use=8.1, loaded=[view])
    held = gov.admit("m", requested_ctx=None, caller="chat")
    assert (held.admitted, held.load_expected, held.num_gpu) == (True, False, None)
    assert gov.pinned_layers("m", None) is None


def test_gp38_a_split_without_a_count_serves_a_model_llama_cpp_holds_as_held():
    loaded = _open((_BACKEND, source("inference_backend.py")))
    rg = loaded[_RG]
    mod, dirs = _llama_engine(loaded)
    fake = _GateGovernor(_whole(rg, "big.gguf", num_ctx=4096))
    rg._governor = fake
    try:
        backend = mod.LlamaCppBackend(model_dirs=dirs)
        backend.generate("big.gguf", _MESSAGES)
        # A split the governor could not count, at a longer context: the held
        # model is never loaded again with every layer on the GPU.
        fake._decision = _split(rg, "big.gguf", num_ctx=8192)
        assert backend.generate("big.gguf", _MESSAGES).content == "served"
        assert [chunk.content for chunk in backend.stream("big.gguf", _MESSAGES)] == ["served"]
        assert _Llama.events == [("load", 4096)]
        assert fake.recorded == []  # held: not refused either
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gp39_a_relative_from_is_never_opened_even_where_it_names_a_blob():
    loaded = _with_mm()
    rg, mm = loaded[_RG], loaded[_MM]
    reads = _counting(mm)
    tmp = _tmp()
    blob = _model(_blob_store(tmp) / ("sha256-" + "f" * 64), [1.0] * 8)
    relative = os.path.relpath(blob, tmp)
    info = _Info(_llama_info(8), modelfile=f"FROM {relative}\n")
    gov = _governor(rg, _config(rg), registry=_Registry(_Backend({"m": info})))
    here = os.getcwd()
    os.chdir(tmp)
    try:
        # From here the relative path names the blob; it is still not opened.
        assert Path(relative).is_file()
        assert gov.tensor_table("m") is None
    finally:
        os.chdir(here)
    assert reads == []


def test_gp40_one_call_through_the_governor_at_another_context_is_sent_no_count_and_pins_nothing():
    loaded = _open((_MM, source("model_manager.py")), (_BACKEND, source("inference_backend.py")))
    rg, mod = loaded[_RG], loaded[_BACKEND]
    mod._live_mode = lambda: "daily"
    # 32 KV heads and 6400 MB of RAM, 2.25 GiB past the reserve: at 8192 the
    # split needs 3.25 GiB of RAM and does not hold; at 4096 it needs 2.0625.
    gov = _ollama_plan(rg, kv_heads=32, ram_mb=6400.0, ollama_num_parallel=1)
    rg._governor = gov
    try:
        client = _Ollama()
        mod.OLLAMA_AVAILABLE = True
        mod._ollama_module = client
        backend = mod.OllamaBackend()
        # A funnel asks 8192 for chat and is admitted 4096; its call still tells
        # 8192, so the head sends no count and the gate pins none.
        decision = gov.admit("m", requested_ctx=8192, caller="chat")
        assert (decision.action, decision.num_ctx, decision.num_gpu) == ("downsize", 4096, 7)
        with rg.ticket_scope(decision):
            backend.generate("m", _MESSAGES, options={"num_ctx": 8192})
        assert (client.calls[-1]["options"], gov.pinned_layers("m", 4096)) == ({"num_ctx": 8192}, None)
        # At the context it was priced for, the count rides and is pinned.
        with rg.ticket_scope(gov.admit("m", requested_ctx=4096, caller="chat")):
            backend.generate("m", _MESSAGES, options={"num_ctx": 4096})
        assert (client.calls[-1]["options"], gov.pinned_layers("m", 4096)) == ({"num_ctx": 4096, "num_gpu": 7}, 7)
        # Held there now, a keepalive that tells no context is sent none.
        view = rg.LoadedModelView(
            name="m", size_vram_bytes=int(8.4 * _GIB), size_bytes=int(10.5 * _GIB), context_length=4096
        )
        _snapshot(rg, gov, in_use=8.4, loaded=[view], ram_mb=6400.0)
        backend.generate("m", _MESSAGES, options={"num_predict": 0})
        assert client.calls[-1]["options"] == {"num_predict": 0}
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gp41_a_stream_refused_for_its_options_was_never_admitted():
    loaded = _open((_BACKEND, source("inference_backend.py")))
    rg, mod = loaded[_RG], loaded[_BACKEND]
    mod._live_mode = lambda: "daily"
    fake = _GateGovernor(_split(rg, "big", num_gpu=12))
    rg._governor = fake
    try:
        client = _Ollama()
        mod.OLLAMA_AVAILABLE = True
        mod._ollama_module = client
        backend = mod.OllamaBackend()
        # As generate does: a malformed schema is refused before any admission,
        # so no load is accounted and no count is pinned for a request never sent.
        with pytest.raises(ValueError):
            list(backend.stream("big", _MESSAGES, options={"num_ctx": 4096, mod.SCHEMA_OPTION: "no schema"}))
        assert (fake.loads, fake.pins, client.calls) == ([], [], [])
        list(backend.stream("big", _MESSAGES, options={"num_ctx": 4096}))
        assert (len(fake.loads), fake.pins) == (1, [("big", 12, 4096)])
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()
