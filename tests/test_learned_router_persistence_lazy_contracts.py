#!/usr/bin/env python3
"""The replacement for the persistence contract, in a file of its own.

It supersedes a contract that lives in a suite the isolation seal carries by
byte digest. Adding this beside it would have changed that digest and broken
the seal, and regenerating a seal to quiet a guard is the habit the ratchet
exists to break. So the original suite is left byte-identical and the
replacement lives here, deselecting the old one by name as usual.

What it supersedes pinned that the training store is opened through the
encrypted helper, and that a SQL-control payload in a label is stored
verbatim rather than executed. Both still hold. Only the moment moved:
construction opens nothing now, so the helper is reached at the first write.

  * LP1 -- construction opens nothing, the first write goes through the
    encrypted helper, and a SQL-control payload is bound rather than run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_learned_router_integrity_contracts import (  # noqa: E402
    _SAFE_CONNECT_CALLS,
    _load,
    _tmp,
)


def test_lp1_persistence_reaches_the_helper_on_first_use_and_binds_sql():
    mod, _enc, restore = _load()
    tmp = _tmp()
    try:
        mod.SKLEARN_AVAILABLE = False  # keep init light, no model load

        _SAFE_CONNECT_CALLS["n"] = 0
        router = mod.LearnedRouter(
            config_path=tmp / "absent.yaml",
            db_path=tmp / "lr.db",
            model_path=tmp / "lr.pkl",
        )
        assert _SAFE_CONNECT_CALLS["n"] == 0, (
            "construction must open nothing; the cost and the encryption "
            "posture both belong to the first caller who asks"
        )

        payload = "\'; DROP TABLE training_samples; --"
        router.log_sample("some query text", payload, source="router")

        assert _SAFE_CONNECT_CALLS["n"] > 0, (
            "the training store must be opened via the encrypted helper, and "
            "the first write is where that now happens"
        )
        assert router.get_sample_count() == 1, (
            "the table must survive a SQL-control payload -- queries are bound"
        )
        dist = router.get_class_distribution()
        assert dist.get(payload) == 1, (
            "the label is persisted literally, proving the query is "
            "parameterized"
        )
    finally:
        restore()
