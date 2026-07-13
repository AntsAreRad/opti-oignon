#!/usr/bin/env python3
"""Contracts for the inference profiler as a governed telemetry consumer.

The profiler already implements the telemetry consumer protocol and already
has a REST surface. What was missing is the WIRING, and above all who is
allowed to arm it. Before these contracts, the profiler subscribed to the
bus as a SIDE EFFECT of the first read of its own route: nothing collected
until someone looked, the first look always answered empty, and the arming
bypassed the consumer policy the bus keeps for its three other consumers.

These contracts pin the position of the switch:

  * Contract P1 -- with no explicitly true toggle (key absent, consumers
    section absent or malformed, configuration unreadable) the profiler is
    NOT registered on the bus. Inert by default: wiring never widens the
    collected surface on its own.
  * Contract P2 -- with the toggle true the profiler IS registered and
    receives the start / token / end events, producing real profiles.
  * Contract P3 -- the read never arms the collection: with the toggle
    explicitly false, calling get_profiler() leaves the bus untouched and
    the profiler records nothing.
  * Contract P4 -- a non-boolean toggle value ("true", 1, []) keeps the
    profiler off. A malformed configuration must not arm a collector by
    truthiness.
  * Contract P5 -- an unavailable profiler module leaves the bus intact
    with its other consumers and raises nothing: the factory answers None.

The subscription in get_profiler() and the built-in registration are one
single mechanism, not two: a bus that builds its consumers while the
consumer accessor calls back into the bus deadlocks on the singleton locks.
Only one of the two can exist, so only one clause pins each direction.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the house idiom: canonical
dotted names, deterministic recorder stand-ins for the sibling consumers,
and a meta-path guard sealing the isolation window.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


class _IsolationGuard:
    """Refuse every project submodule the test did not seed.

    A stand-in package whose ``__path__`` is empty isolates the tree only
    while the parent path is the sole way to resolve a submodule. That
    assumption breaks wherever the project is installed in editable mode:
    such an install registers a finder that answers on the module NAME and
    ignores the parent path, so a real submodule resolves behind the
    test's back -- silently importing live code. This guard sits ahead of
    every finder and refuses the names that were not seeded, so a load
    behaves identically whether the project is installed or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


# ---------------------------------------------------------------------------
# Deterministic stand-ins for the sibling consumers (recorders)
# ---------------------------------------------------------------------------
class _Recorder:
    """Records every event batch a sibling consumer is handed."""

    def __init__(self):
        self.batches: list[list] = []


def _live_metrics_module(rec: _Recorder) -> types.ModuleType:
    mod = types.ModuleType("opti_oignon.live_metrics")

    class _Collector:
        def start_generation(self, model=""):
            rec.batches.append(["start"])

        def record_token(self, count=1):
            rec.batches.append(["token"])

        def end_generation(self, eval_time_ms=0.0):
            rec.batches.append(["end"])

    mod.get_live_metrics = lambda: _Collector()
    return mod


def _performance_monitor_module() -> types.ModuleType:
    mod = types.ModuleType("opti_oignon.performance_monitor")

    class _Monitor:
        def record_execution(self, **kwargs):
            return None

    mod.performance_monitor = _Monitor()
    return mod


def _speculative_decoding_module() -> types.ModuleType:
    mod = types.ModuleType("opti_oignon.speculative_decoding")

    class _Manager:
        def record_acceptance(self, **kwargs):
            return None

    mod.get_speculative_decoding_manager = lambda: _Manager()
    return mod


# ---------------------------------------------------------------------------
# Isolated loading
# ---------------------------------------------------------------------------
def _load(tmp_dir: Path, yaml_text: str | None, seed_profiler: bool = True):
    """Load telemetry (and optionally the real profiler) in isolation.

    ``yaml_text`` is written to a scratch config file handed to the loader.
    ``None`` means no file at all. When ``seed_profiler`` is false the
    profiler module is left unseeded, so the meta-path guard refuses it --
    the unavailable-module case, without touching the tree.
    """
    keys = (
        "opti_oignon", "opti_oignon.telemetry", "opti_oignon.inference_profiler",
        "opti_oignon.live_metrics", "opti_oignon.performance_monitor",
        "opti_oignon.speculative_decoding",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    live_rec = _Recorder()
    for dotted, mod in (
        ("opti_oignon.live_metrics", _live_metrics_module(live_rec)),
        ("opti_oignon.performance_monitor", _performance_monitor_module()),
        ("opti_oignon.speculative_decoding", _speculative_decoding_module()),
    ):
        sys.modules[dotted] = mod
        setattr(pkg, dotted.rsplit(".", 1)[1], mod)

    def _real(dotted: str, path: Path):
        spec = importlib.util.spec_from_file_location(dotted, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    profiler_mod = None
    if seed_profiler:
        profiler_mod = _real(
            "opti_oignon.inference_profiler", _OO / "inference_profiler.py",
        )
        pkg.inference_profiler = profiler_mod

    telemetry_mod = _real("opti_oignon.telemetry", _OO / "telemetry.py")
    pkg.telemetry = telemetry_mod

    cfg_path = None
    if yaml_text is not None:
        cfg_path = tmp_dir / "telemetry.yaml"
        cfg_path.write_text(yaml_text)

    def restore():
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return telemetry_mod, profiler_mod, cfg_path, live_rec, restore


# ---------------------------------------------------------------------------
# Local material
# ---------------------------------------------------------------------------
MODEL = "test-model"

_YAML_ON = """
enabled: true
buffer:
  flush_interval_ms: 0
consumers:
  live_metrics: true
  performance_monitor: true
  speculative_decoding: true
  inference_profiler: true
"""

_YAML_OFF = """
enabled: true
buffer:
  flush_interval_ms: 0
consumers:
  live_metrics: true
  performance_monitor: true
  speculative_decoding: true
  inference_profiler: false
"""

_YAML_SILENT = """
enabled: true
buffer:
  flush_interval_ms: 0
consumers:
  live_metrics: true
  performance_monitor: true
  speculative_decoding: true
"""

_YAML_NO_SECTION = """
enabled: true
buffer:
  flush_interval_ms: 0
"""

_YAML_MALFORMED_SECTION = """
enabled: true
buffer:
  flush_interval_ms: 0
consumers:
  - live_metrics
  - inference_profiler
"""

_YAML_UNREADABLE = """
enabled: true
consumers:
  inference_profiler: [unclosed
"""


def _scratch() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix="oo_telemetry_"))


def _profiler_registered(collector, profiler_mod) -> bool:
    """True when the bus holds the profiler singleton's consumer callback."""
    profiler = profiler_mod._profiler
    if profiler is None:
        return False
    bound = profiler.consume
    return any(
        c == bound or getattr(c, "__self__", None) is profiler
        for c in collector._consumers
    )


def _drive_one_request(telemetry_mod, collector) -> None:
    """Push a full start / token / end cycle through the bus."""
    rid = collector.on_inference_start(model=MODEL)
    collector.on_token_generated(request_id=rid, count=3)
    collector.on_inference_end(
        request_id=rid,
        model=MODEL,
        tokens_in=5,
        tokens_out=3,
        latency_ms=120.0,
    )
    collector.flush()


# ---------------------------------------------------------------------------
# Contract P1 -- no explicitly true toggle keeps the profiler off the bus
# ---------------------------------------------------------------------------
def test_p1_absent_or_unreadable_toggle_keeps_the_profiler_unregistered():
    for label, text in (
        ("key silent", _YAML_SILENT),
        ("no consumers section", _YAML_NO_SECTION),
        ("malformed consumers section", _YAML_MALFORMED_SECTION),
        ("unreadable configuration", _YAML_UNREADABLE),
        ("no configuration file", None),
    ):
        tmp = _scratch()
        telemetry_mod, profiler_mod, cfg_path, _rec, restore = _load(tmp, text)
        try:
            cfg = telemetry_mod._load_config(cfg_path)
            assert cfg.consumer_inference_profiler is False, (
                f"{label}: the profiler toggle must default to off, "
                f"got {cfg.consumer_inference_profiler!r}"
            )

            collector = telemetry_mod.TelemetryCollector(config_path=cfg_path)
            assert not _profiler_registered(collector, profiler_mod), (
                f"{label}: the profiler must not be registered on the bus"
            )
            # The bus is not crippled: its other consumers are still there.
            assert len(collector._consumers) == 3, (
                f"{label}: the sibling consumers must stay registered, "
                f"got {len(collector._consumers)}"
            )

            _drive_one_request(telemetry_mod, collector)
            if profiler_mod._profiler is not None:
                assert profiler_mod._profiler.total_profiled == 0, (
                    f"{label}: an unregistered profiler must record nothing"
                )
        finally:
            restore()


# ---------------------------------------------------------------------------
# Contract P2 -- a true toggle registers the profiler and it profiles
# ---------------------------------------------------------------------------
def test_p2_enabled_toggle_registers_the_profiler_and_it_profiles():
    tmp = _scratch()
    telemetry_mod, profiler_mod, cfg_path, _rec, restore = _load(tmp, _YAML_ON)
    try:
        cfg = telemetry_mod._load_config(cfg_path)
        assert cfg.consumer_inference_profiler is True

        collector = telemetry_mod.TelemetryCollector(config_path=cfg_path)

        assert _profiler_registered(collector, profiler_mod), (
            "an enabled toggle must register the profiler on the bus"
        )
        assert len(collector._consumers) == 4, (
            "the profiler joins the sibling consumers, it does not replace "
            f"them; got {len(collector._consumers)} consumers"
        )

        _drive_one_request(telemetry_mod, collector)

        profiler = profiler_mod._profiler
        assert profiler is not None
        assert profiler.total_profiled == 1, (
            "a registered profiler must record the completed request, "
            f"got total_profiled={profiler.total_profiled}"
        )
        recent = profiler.get_recent(5)
        assert len(recent) == 1
        assert recent[0]["model"] == MODEL
        summary = profiler.get_summary()
        assert any(s["model"] == MODEL for s in summary), (
            "the completed request must surface in the per-model summary"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract P3 -- the read never arms the collection
# ---------------------------------------------------------------------------
def test_p3_reading_the_profiler_never_subscribes_it_to_the_bus():
    tmp = _scratch()
    telemetry_mod, profiler_mod, cfg_path, _rec, restore = _load(tmp, _YAML_OFF)
    try:
        collector = telemetry_mod.get_telemetry(config_path=cfg_path)
        before = len(collector._consumers)
        assert before == 3

        # This is exactly what the REST route does. It must stay a read.
        profiler = profiler_mod.get_profiler()
        assert profiler is not None

        assert len(collector._consumers) == before, (
            "get_profiler() must not subscribe the profiler to the bus: "
            f"consumer count went {before} -> {len(collector._consumers)}"
        )
        assert not _profiler_registered(collector, profiler_mod), (
            "reading the profiler must never arm the collection"
        )

        _drive_one_request(telemetry_mod, collector)
        assert profiler.total_profiled == 0, (
            "a profiler that was merely read must record nothing, "
            f"got total_profiled={profiler.total_profiled}"
        )
    finally:
        telemetry_mod.reset_telemetry()
        restore()


# ---------------------------------------------------------------------------
# Contract P4 -- a non-boolean toggle value never arms the profiler
# ---------------------------------------------------------------------------
def test_p4_non_boolean_toggle_value_keeps_the_profiler_off():
    for literal in ('"true"', '"yes"', "1", "[]", '"on"'):
        text = (
            "enabled: true\n"
            "buffer:\n"
            "  flush_interval_ms: 0\n"
            "consumers:\n"
            "  live_metrics: true\n"
            "  performance_monitor: true\n"
            "  speculative_decoding: true\n"
            f"  inference_profiler: {literal}\n"
        )
        tmp = _scratch()
        telemetry_mod, profiler_mod, cfg_path, _rec, restore = _load(tmp, text)
        try:
            cfg = telemetry_mod._load_config(cfg_path)
            assert cfg.consumer_inference_profiler is False, (
                f"toggle {literal}: a non-boolean value must not arm the "
                f"profiler, got {cfg.consumer_inference_profiler!r}"
            )

            collector = telemetry_mod.TelemetryCollector(config_path=cfg_path)
            assert not _profiler_registered(collector, profiler_mod), (
                f"toggle {literal}: truthiness must never register a consumer"
            )
            assert len(collector._consumers) == 3
        finally:
            restore()


# ---------------------------------------------------------------------------
# Contract P5 -- an unavailable profiler module leaves the bus intact
# ---------------------------------------------------------------------------
def test_p5_unavailable_profiler_module_leaves_the_bus_intact():
    tmp = _scratch()
    telemetry_mod, _absent, cfg_path, _rec, restore = _load(
        tmp, _YAML_ON, seed_profiler=False,
    )
    try:
        # The factory must answer None rather than let the import escape.
        consumer = telemetry_mod._create_inference_profiler_consumer()
        assert consumer is None, (
            "an unavailable profiler module must yield no consumer, "
            f"got {consumer!r}"
        )

        # Building the bus must not raise, and must keep the other consumers.
        collector = telemetry_mod.TelemetryCollector(config_path=cfg_path)
        assert len(collector._consumers) == 3, (
            "an unavailable profiler must not cost the bus its other "
            f"consumers; got {len(collector._consumers)}"
        )

        # And the bus still works end to end.
        _drive_one_request(telemetry_mod, collector)
        assert collector.get_stats()["total_requests"] == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner (pytest picks up the test_ functions; direct execution works too)
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    names = sorted(n for n in globals() if n.startswith("test_"))
    selected = [
        n for n in names if not argv or any(fragment in n for fragment in argv)
    ]
    failures = 0
    for name in selected:
        try:
            globals()[name]()
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"{len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
