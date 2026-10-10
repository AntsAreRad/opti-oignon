#!/usr/bin/env python3
"""Contracts for the interface of the adoption of memory.

A fact the user did not write, type or accept lowers every answer it reaches
until they adopt it, by the digest of the text they were shown. The panel of
memories is where they read it and decide.

  * TL44 -- the panel shows each fact awaiting adoption whole, every
    character a screen hides written as its escape, and adopts it with the
    digest of the text shown, one request per fact; the client lists and
    adopts on routes the server serves, and the server reads exactly an id
    and a digest per fact.

Local-only (the public distribution ships no tests). The client and the panel
are read as text; the router is loaded through the isolation window.
"""

import re
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _frontend import read  # noqa: E402
from _isolation import isolate, source  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder
# from the junit file.
BUDGET_S = {
    "test_tl44_the_panel_shows_each_unendorsed_fact_whole_and_adopts_it_by_the_digest_shown": 6.0,
}

_CLIENT = "frontend/src/lib/api/memories.ts"
_PANEL = "frontend/src/lib/components/panels/MemoriesPanel.svelte"


def test_tl44_the_panel_shows_each_unendorsed_fact_whole_and_adopts_it_by_the_digest_shown():
    client, panel = read(_CLIENT), read(_PANEL)
    assert "apiGet<UnendorsedMemory[]>('/api/memories/unendorsed')" in client, "the client lists the facts awaiting"
    assert "apiPost<{ adopted: string[]; refused: string[] }>('/api/memories/adopt', { items })" in client, (
        "the client adopts with the items it is handed, in one request"
    )
    assert re.search(r"\{escapeAll\(fact\.text\)\}", panel), "each fact is shown with every hidden character escaped"
    assert "{fact.text}" not in panel, "never the bare text"
    assert re.search(r"adoptMemories\(\[\{ id: fact\.id, digest: fact\.digest \}\]\)", panel), (
        "an adoption carries the digest of the text shown"
    )
    assert re.search(r"\{fact\.digest\.slice\(0, 16\)\}", panel), "the digest is shown beside the text"

    deps = types.ModuleType("opti_oignon.api.deps")
    deps.MEMORY_AVAILABLE = False
    deps.memory_manager = None
    loaded, restore = isolate(
        targets={"opti_oignon.api.schemas": source("api", "schemas.py"),
                 "opti_oignon.api.routes_memory": source("api", "routes_memory.py")},
        seeded={"opti_oignon.api.deps": deps},
        packages=("opti_oignon.api",),
    )
    try:
        routes = loaded["opti_oignon.api.routes_memory"]
        schemas = loaded["opti_oignon.api.schemas"]
        served = {(method, route.path) for route in routes.memories_router.routes for method in route.methods}
        item_fields = sorted(schemas.MemoryAdoptItem.model_fields)
        request_fields = sorted(schemas.MemoryAdoptRequest.model_fields)
    finally:
        restore()
    assert {("GET", "/api/memories/unendorsed"), ("POST", "/api/memories/adopt")} <= served, sorted(served)
    assert (item_fields, request_fields) == (["digest", "id"], ["items"]), (item_fields, request_fields)
