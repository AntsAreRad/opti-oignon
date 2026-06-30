#!/usr/bin/env python3
"""Channel abuse-control guarantees for the remote-inference streaming buffer.

The desktop buffers a borrowed-model reply server-side and lets the paired phone
pull it chunk by chunk. The same buffer module carries the channel's abuse
controls. This suite pins their load-bearing safety properties, exercised
against the real (standard-library-only) buffer:

  * the per-device fixed-window rate gate refuses a device that exceeds its
    window -- the request is not absorbed and an alert is recorded -- and the
    window resets once it has elapsed;
  * a single buffered reply is bounded by total bytes, so a pathological
    generation cannot grow memory without bound;
  * the session registry is bounded in count, the oldest evicted past the cap;
  * the live half of a revoke drops exactly the target device's in-flight
    buffers and leaves every other device's untouched.

The module is loaded directly from source (it imports only the standard
library). The registry and the rate windows are reset before each test that
needs a clean slate.

Local-only. Runs under pytest or the ``__main__`` runner.
"""

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load():
    """Load the streaming buffer module fresh from source (stdlib-only)."""
    spec = importlib.util.spec_from_file_location(
        "remote_streaming_under_test", _OO / "veilid" / "remote_streaming.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rate_gate_refuses_a_device_over_its_window():
    """A device past its fixed window is refused, not served; the window resets."""
    rs = _load()
    rs.reset_for_tests()
    # Within one window, requests up to the limit are allowed; the next is a
    # breach -- refused and NOT counted against a served budget.
    allowed = [
        rs.check_rate("phoneA", now=0.0, limit=2, window=10.0) for _ in range(3)
    ]
    assert allowed == [True, True, False]
    # The breach surfaces as an alert for the desktop control surface.
    assert rs.telemetry("phoneA")["alerts"] == 1
    # Once the window has elapsed, the same device is allowed again.
    assert rs.check_rate("phoneA", now=10.0, limit=2, window=10.0) is True


def test_a_buffered_reply_is_bounded_by_total_bytes():
    """A pathological generation is truncated at the byte cap (a memory guard)."""
    rs = _load()
    # Chunks sized so the byte cap trips well before the per-count cap.
    chunk = "x" * 2000
    huge = [chunk] * 5000  # ~10 MB raw, above the byte cap
    bounded = rs._bounded_chunks(huge)
    total = sum(len(s.encode("utf-8")) for s in bounded)
    assert total <= rs.MAX_SESSION_BYTES
    assert len(bounded) < len(huge)  # genuinely truncated, not passed through


def test_the_session_registry_is_bounded_in_count():
    """Past the cap the oldest session is evicted; the registry stays bounded."""
    rs = _load()
    rs.reset_for_tests()
    for i in range(rs.MAX_SESSIONS + 5):
        rs.open_session("phoneA", f"req-{i}", ["chunk"])
    # The count never exceeds the cap, no matter how many were opened.
    assert rs.active_session_count() == rs.MAX_SESSIONS
    # The newest survive; the oldest were the ones evicted.
    assert rs.pull("phoneA", f"req-{rs.MAX_SESSIONS + 4}", 0) is not None
    assert rs.pull("phoneA", "req-0", 0) is None


def test_revoke_drops_only_the_target_devices_sessions():
    """The live revoke kills the target device's buffers and no others."""
    rs = _load()
    rs.reset_for_tests()
    rs.open_session("phoneA", "r1", ["a"])
    rs.open_session("phoneA", "r2", ["b"])
    rs.open_session("phoneB", "r3", ["c"])
    dropped = rs.kill_sessions_for_device("phoneA")
    assert dropped == 2
    # phoneA's streams no longer pull -- its next continuation misses.
    assert rs.pull("phoneA", "r1", 0) is None
    assert rs.pull("phoneA", "r2", 0) is None
    # phoneB is untouched and still reads its own chunk.
    out = rs.pull("phoneB", "r3", 0)
    assert out is not None and out["content"] == "c"


if __name__ == "__main__":
    import sys

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
