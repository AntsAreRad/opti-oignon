#!/usr/bin/env bash
# Build the onion memory's native core and place it where the loader looks.
#
# The artefact is never tracked: it is rebuilt from the pinned crate
# (rust/oo_core, Cargo.lock committed) on every machine that wants it, and
# the Python reference runs wherever it is absent. Requires cargo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATE="$ROOT/rust/oo_core"
DEST="$ROOT/opti_oignon/native"
cargo build --release --locked --manifest-path "$CRATE/Cargo.toml"
built="$CRATE/target/release/liboo_core.so"
[ -f "$built" ] || { echo "no artefact at $built" >&2; exit 1; }
cp "$built" "$DEST/oo_core.so"
python3 - <<'PY'
from opti_oignon.native import load
core = load()
print("oo_core", core.VERSION if core else "NOT LOADABLE")
raise SystemExit(0 if core else 1)
PY
