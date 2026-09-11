#!/usr/bin/env python3
"""Contracts that a swept axis actually varies the request it is swept over.

The tuner sweeps one axis at a time and then reports which value won. That
report is only worth reading if the values it compared reached the backend as
different requests. They did not: the Ollama path set ``num_batch`` from the
batch size and then overwrote it with ``min(batch_size, ubatch_size)``, and the
sweep always carries both keys, so every batch-size point sent a byte-identical
request. The analyser then compared their timings and announced one value as a
multiple of another, at medium confidence, from run-to-run noise alone. The
llama.cpp path had the mirror defect: it never mapped the micro-batch at all,
so its micro-batch points were the identical ones instead.

These contracts pin the axis, not the tuning. They assert nothing about which
value is faster -- that is machine work, and it is owed. They assert that two
distinct values on an axis leave as two distinct requests, so that a later
comparison between them is a comparison of something.

  * AX1 -- distinct batch sizes reach the transport as distinct batch options.
  * AX2 -- a micro-batch value does not overwrite the batch value.
  * AX3 -- distinct micro-batch sizes reach the llama.cpp backend as distinct
    options, rather than collapsing to one.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window, with the inference backend declared unreachable and proven so
before anything runs: both paths are driven by an injected transport here.
"""

import sys
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _open():
    """Open the shared window on the tuner alone."""
    loaded, restore = isolate(
        targets={"opti_oignon.auto_tuner": source("auto_tuner.py")},
        blocked=("opti_oignon.inference_backend",),
    )
    return loaded["opti_oignon.auto_tuner"], restore


# ---------------------------------------------------------------------------
# Injected transports
# ---------------------------------------------------------------------------
_OLLAMA_PAYLOAD = {
    "eval_count": 128,
    "eval_duration": 4_000_000_000,
    "prompt_eval_count": 32,
    "prompt_eval_duration": 500_000_000,
    "message": {"content": "stand-in reply"},
}


class _FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return _OLLAMA_PAYLOAD


class _FakeRequests(types.ModuleType):
    """Stand-in ``requests`` module that records the options it was sent."""

    def __init__(self):
        super().__init__("requests")
        self.options = []

    def post(self, url, json=None, timeout=None):
        self.options.append(dict((json or {}).get("options", {})))
        return _FakeResponse()


class _FakeChatResponse:
    def __init__(self, content):
        self.content = content


class _FakeLlamaCppBackend:
    """A llama.cpp stand-in that records the options it was handed."""

    def __init__(self):
        self.options = []

    def generate(self, model=None, messages=None, options=None):
        self.options.append(dict(options or {}))
        return _FakeChatResponse("x" * 400)


def _install_requests():
    saved = sys.modules.get("requests")
    fake = _FakeRequests()
    sys.modules["requests"] = fake

    def restore():
        if saved is None:
            sys.modules.pop("requests", None)
        else:
            sys.modules["requests"] = saved

    return fake, restore


def _batch_axis(mod):
    """The sweep points that differ only in batch size, as the tuner builds them."""
    tuner = mod.AutoTuner(
        config=mod.TunerConfig(warmup_runs=0, trials_per_param=1),
        param_space=mod.ParameterSpace(),
        benchmark_fn=lambda params: mod.BenchmarkResult(params=params),
    )
    defaults = tuner._default_params()
    return [
        dict(defaults, batch_size=size)
        for size in mod.ParameterSpace().batch_size
    ]


# ---------------------------------------------------------------------------
# AX1 -- distinct batch sizes leave as distinct requests
# ---------------------------------------------------------------------------
def test_ax1_the_batch_axis_varies_what_is_sent():
    mod, restore = _open()
    fake, restore_requests = _install_requests()
    try:
        points = _batch_axis(mod)
        assert len({p["batch_size"] for p in points}) > 1, (
            "the axis under test really does carry more than one value"
        )
        bench = mod.create_ollama_benchmark_fn("stand-in-model")
        for params in points:
            bench(params)
        assert len(fake.options) == len(points), (
            "every point reached the transport, so this is not vacuous"
        )
        sent = [opts.get("num_batch") for opts in fake.options]
        assert len(set(sent)) == len(set(p["batch_size"] for p in points)), (
            "as many distinct batch options leave as there are distinct batch "
            "sizes: the axis varies the request rather than the record only"
        )
        assert sent == [p["batch_size"] for p in points], (
            "each point sends its own batch size, unmodified"
        )
    finally:
        restore_requests()
        restore()


# ---------------------------------------------------------------------------
# AX2 -- the micro-batch does not overwrite the batch
# ---------------------------------------------------------------------------
def test_ax2_the_micro_batch_does_not_overwrite_the_batch():
    mod, restore = _open()
    fake, restore_requests = _install_requests()
    try:
        bench = mod.create_ollama_benchmark_fn("stand-in-model")
        bench({"batch_size": 4096, "ubatch_size": 256, "threads": 6})
        assert len(fake.options) == 1, "the transport was exercised once"
        opts = fake.options[0]
        assert opts["num_batch"] == 4096, (
            "the batch option is the batch size, not the smaller of the two"
        )
        assert opts["num_batch"] != 256, (
            "a micro-batch never silently becomes the batch"
        )
        assert 256 in opts.values(), (
            "the micro-batch is still carried for backends that read it, "
            "rather than being dropped to protect the batch"
        )
    finally:
        restore_requests()
        restore()


# ---------------------------------------------------------------------------
# AX3 -- the micro-batch axis varies the llama.cpp request
# ---------------------------------------------------------------------------
def test_ax3_the_micro_batch_axis_varies_the_llamacpp_request():
    mod, restore = _open()
    try:
        backend = _FakeLlamaCppBackend()
        bench = mod.create_llamacpp_benchmark_fn(
            "stand-in-model", backend=backend,
        )
        sizes = mod.ParameterSpace().ubatch_size
        assert len(set(sizes)) > 1, (
            "the axis under test really does carry more than one value"
        )
        for size in sizes:
            bench({"batch_size": 2048, "ubatch_size": size, "threads": 6})
        assert len(backend.options) == len(sizes), (
            "every point reached the backend, so this is not vacuous"
        )
        distinct = {
            tuple(sorted(opts.items())) for opts in backend.options
        }
        assert len(distinct) == len(set(sizes)), (
            "as many distinct requests leave as there are distinct micro-batch "
            "sizes, instead of one request repeated"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("AX1 batch axis varies what is sent", test_ax1_the_batch_axis_varies_what_is_sent),
        ("AX2 micro-batch does not overwrite batch", test_ax2_the_micro_batch_does_not_overwrite_the_batch),
        ("AX3 micro-batch axis varies llama.cpp request", test_ax3_the_micro_batch_axis_varies_the_llamacpp_request),
    ]
    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
