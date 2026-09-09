# Browser tests

Playwright specs for the desktop application. They drive a real browser
against a served frontend and a running backend, so they are deliberately
kept out of the continuous-integration gate: a run needs both processes,
and one of them needs a model host that no runner provides.

## Running them

    ./scripts/run_e2e.sh              # headless chromium
    ./scripts/run_e2e.sh --headed     # visible browser
    ./scripts/run_e2e.sh --ui         # interactive

The configuration starts the frontend dev server itself. The backend is
not started for you; bring it up first on port 8001, which is where the
dev server proxies `/api`:

    uvicorn opti_oignon.api.app:app --port 8001

## What is covered, and what is not

The specs here cover paths that hold up without a model host: the status
page and its health call, the security mode the origin reports, and the
refusal of bad credentials. Anything that reaches a model -- a chat turn,
a benchmark, a routing decision -- is left out on purpose. Those need a
local inference host and a loaded model, so they belong to a measurement
run on a real workstation rather than to this directory.
