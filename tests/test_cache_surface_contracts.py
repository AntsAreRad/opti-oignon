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
  * CS8 -- the same property, wherever the cache switch is shown: the API
    layer speaks the published prefix; every component that shows the
    switch reaches it through the API layer and spells no path of its own
    (the list is computed, and it is never empty); and no file of the
    frontend carries the legacy token, in either case.

The legacy token is assembled from fragments so it never appears in this
file's source. Local-only. Runs under pytest; imports the application the
way the release-docs contracts do.
"""

import json
import re
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


# The component files of the frontend, and the ones that show the cache
# switch: those that spell the cache's paths, call its API functions, or
# label a control with its name.
_FRONTEND = REPO / "frontend" / "src"
_TEXT_KINDS = (".svelte", ".ts", ".js", ".css", ".html")
_SHOWS_SWITCH = re.compile(
    re.escape(_PREFIX) + r"|\b(?:toggleSemCache|getSemCacheStatus)\b"
    r"|aria-label\s*=\s*[\"'][^\"']*semantic cache",
    re.IGNORECASE,
)
_API_IMPORT = re.compile(r"""from\s*['"]\$lib/api/semanticCache['"]""")


def test_cs8_every_view_of_the_cache_switch_goes_through_the_api_layer():
    api_layer = _CLIENT_FILES[0].read_text(encoding="utf-8")
    assert _PREFIX + "/" in api_layer, "the api layer does not call the published prefix"

    sample = (
        "<button aria-label=\"Toggle semantic cache\" on:click={t}>Cache</button>\n"
        "<script>await fetch('" + _PREFIX + "/toggle');</script>"
    )
    assert _SHOWS_SWITCH.search(sample) and not _API_IMPORT.search(sample), (
        "the census reads a view of the switch, and a raw path is not the api layer"
    )
    components = sorted(
        path for path in _FRONTEND.rglob("*.svelte")
        if _SHOWS_SWITCH.search(path.read_text(encoding="utf-8"))
    )
    assert components, "no component shows the cache switch: the census went blind"
    for path in components:
        text = path.read_text(encoding="utf-8")
        name = path.relative_to(REPO).as_posix()
        assert _API_IMPORT.search(text), f"{name} does not reach the cache through the api layer"
        assert _PREFIX not in text, f"{name} spells the cache's path rather than calling the api layer"

    texts = [
        path for path in _FRONTEND.rglob("*")
        if path.is_file() and path.suffix in _TEXT_KINDS
    ]
    assert len(texts) > 100, f"the census reads the frontend's files: {len(texts)}"
    for path in texts:
        text = path.read_text(encoding="utf-8")
        for token in (_LEGACY, _LEGACY.upper()):
            assert token not in text, (
                f"legacy token {token!r} survives in {path.relative_to(REPO).as_posix()}"
            )
