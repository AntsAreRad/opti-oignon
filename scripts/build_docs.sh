#!/usr/bin/env bash
# scripts/build_docs.sh -- Build the MkDocs documentation site (S161)
#
# Usage:
#   ./scripts/build_docs.sh          # build only
#   ./scripts/build_docs.sh serve    # build and serve locally
#   ./scripts/build_docs.sh check    # validate links only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQUIREMENTS="$PROJECT_ROOT/requirements-docs.txt"
MKDOCS_YML="$PROJECT_ROOT/mkdocs.yml"
DOCS_DIR="$PROJECT_ROOT/docs"
SITE_DIR="$PROJECT_ROOT/site"

# -- helpers ---------------------------------------------------------------

log() {
    echo "[build_docs] $*"
}

die() {
    echo "[build_docs] ERROR: $*" >&2
    exit 1
}

# -- preflight checks ------------------------------------------------------

if [ ! -f "$MKDOCS_YML" ]; then
    die "mkdocs.yml not found at $MKDOCS_YML"
fi

if [ ! -f "$REQUIREMENTS" ]; then
    die "requirements-docs.txt not found at $REQUIREMENTS"
fi

if [ ! -d "$DOCS_DIR" ]; then
    die "docs/ directory not found at $DOCS_DIR"
fi

# -- install dependencies --------------------------------------------------

log "Installing documentation dependencies..."
pip install -q -r "$REQUIREMENTS" --break-system-packages 2>/dev/null \
    || pip install -q -r "$REQUIREMENTS"

# -- validate nav references -----------------------------------------------

log "Checking that all nav-referenced files exist..."
NAV_FILES=$(grep -oP ':\s+\K[a-z].*\.md' "$MKDOCS_YML" || true)
MISSING=0
for f in $NAV_FILES; do
    if [ ! -f "$DOCS_DIR/$f" ]; then
        echo "  MISSING: docs/$f"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    die "$MISSING nav-referenced file(s) missing"
fi
log "All nav references OK."

# -- check internal links in markdown --------------------------------------

log "Checking internal links in markdown files..."
BROKEN_LINKS=0
while IFS= read -r mdfile; do
    # Extract markdown links: [text](target.md) or [text](../path/file.md)
    # Skip external URLs (http/https), anchors-only (#), and empty links
    links=$(grep -oP '\[(?:[^\]]*)\]\(\K[^)#][^)]*\.md(?:#[^)]*)?' "$mdfile" 2>/dev/null || true)
    for link in $links; do
        # Strip anchor fragment
        target="${link%%#*}"
        # Resolve relative to the file's directory
        filedir="$(dirname "$mdfile")"
        resolved="$filedir/$target"
        if [ ! -f "$resolved" ]; then
            echo "  BROKEN: $mdfile -> $target"
            BROKEN_LINKS=$((BROKEN_LINKS + 1))
        fi
    done
done < <(find "$DOCS_DIR" -name "*.md" -type f)

if [ "$BROKEN_LINKS" -gt 0 ]; then
    log "WARNING: $BROKEN_LINKS broken internal link(s) found."
else
    log "All internal links OK."
fi

# -- early exit for check mode ---------------------------------------------

if [ "${1:-}" = "check" ]; then
    if [ "$BROKEN_LINKS" -gt 0 ]; then
        exit 1
    fi
    log "Link check passed."
    exit 0
fi

# -- build -----------------------------------------------------------------

log "Building documentation site..."
cd "$PROJECT_ROOT"
mkdocs build --strict --site-dir "$SITE_DIR"
log "Site built at $SITE_DIR"

# -- serve mode ------------------------------------------------------------

if [ "${1:-}" = "serve" ]; then
    log "Starting development server on http://127.0.0.1:8001 ..."
    mkdocs serve --dev-addr 127.0.0.1:8001
fi
