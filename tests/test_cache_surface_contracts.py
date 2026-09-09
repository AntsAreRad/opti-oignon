#!/usr/bin/env python3
"""Contracts for the semantic cache's published surface.

The cache is reachable from outside the process at two coupled points: the
HTTP paths the server publishes and the client code that calls them. Before
these contracts existed, every one of those paths and schema names could be
renamed without a single test objecting -- the suite stayed entirely green
while the interface broke. These contracts make the surface load-bearing:

  * CS1 -- the five cache routes are published under the semcache prefix.
  * CS2 -- the four cache schemas are published under their SemCache names.
  * CS3 -- the serialized published surface carries no trace of the
    retired legacy token, in either case.
  * CS4 -- the frontend cache client speaks the published prefix and
    carries no trace of the legacy token: the client/server contract is
    asserted on both sides, not assumed.

The legacy token is assembled from fragments so it never appears in this
file's source. Local-only. Runs under pytest; imports the application the
way the release-docs contracts do.
"""

import json
from pathlib import Path

from opti_oignon.api.app import app

REPO = Path(__file__).resolve().parent.parent

# The retired token, assembled so a scan of this file stays clean.
_LEGACY = "s" + "68"

_PREFIX = "/api/cache/semcache"

_ROUTES = (
    _PREFIX + "/status",
    _PREFIX + "/toggle",
    _PREFIX + "/config",
    _PREFIX + "/clear",
    _PREFIX + "/expire",
)

_SCHEMAS = (
    "SemCacheStatusResponse",
    "SemCacheStatsSchema",
    "SemCacheConfigUpdate",
    "SemCacheClearRequest",
)

_CLIENT_FILES = (
    REPO / "frontend" / "src" / "lib" / "api" / "semanticCache.ts",
    REPO / "frontend" / "src" / "lib" / "types.ts",
    REPO / "frontend" / "src" / "lib" / "components" / "panels"
    / "CacheStatsPanel.svelte",
    REPO / "frontend" / "src" / "lib" / "components" / "chat"
    / "ChatControlBar.svelte",
)


def test_cs1_the_five_cache_routes_are_published():
    published = set(app.openapi()["paths"])
    missing = [route for route in _ROUTES if route not in published]
    assert not missing, f"unpublished cache routes: {missing}"


def test_cs2_the_four_cache_schemas_are_published():
    components = set(app.openapi()["components"]["schemas"])
    missing = [name for name in _SCHEMAS if name not in components]
    assert not missing, f"unpublished cache schemas: {missing}"


def test_cs3_the_published_surface_carries_no_legacy_token():
    blob = json.dumps(app.openapi(), sort_keys=True, ensure_ascii=False)
    for token in (_LEGACY, _LEGACY.upper()):
        assert token not in blob, (
            f"legacy token {token!r} still published somewhere in the "
            "serialized surface"
        )


def test_cs4_the_frontend_cache_client_speaks_the_published_prefix():
    api_layer = _CLIENT_FILES[0].read_text(encoding="utf-8")
    control_bar = _CLIENT_FILES[3].read_text(encoding="utf-8")
    for source, name in ((api_layer, "api layer"), (control_bar, "control bar")):
        assert _PREFIX + "/" in source, (
            f"the {name} does not call the published prefix"
        )
    for path in _CLIENT_FILES:
        text = path.read_text(encoding="utf-8")
        for token in (_LEGACY, _LEGACY.upper()):
            assert token not in text, (
                f"legacy token {token!r} survives in {path.name}"
            )
