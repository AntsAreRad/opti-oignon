#!/usr/bin/env bash
# =============================================================================
# verify_release.sh — Verify GPG signature of an Opti-Oignon release archive
#
# Usage:
#   ./scripts/verify_release.sh <archive.zip> [--key <KEY_ID>] [--strict]
#
# Expects:
#   - <archive.zip>.sig  (detached GPG signature)
#   - <archive.zip>.sha256 (SHA-256 checksum, optional but checked if present)
#
# Options:
#   --key <KEY_ID>   Require signature from this specific key
#   --strict         Exit non-zero on any warning (missing checksum, etc.)
#
# Exit codes:
#   0  — Signature valid (and checksum matches if present)
#   1  — Signature invalid or missing
#   2  — Checksum mismatch
#   3  — Missing dependencies or files
#
# See SECURITY.md § "Release Signing" for full documentation.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo -e "${GREEN}[verify]${NC} $*"; }
warn()  { echo -e "${YELLOW}[verify]${NC} $*"; }
error() { echo -e "${RED}[verify]${NC} $*" >&2; }
die()   { error "$@"; exit "${2:-1}"; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
ARCHIVE=""
KEY_ID=""
STRICT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --key)
            KEY_ID="$2"
            shift 2
            ;;
        --strict)
            STRICT=1
            shift
            ;;
        --help|-h)
            head -25 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            if [[ -z "$ARCHIVE" ]]; then
                ARCHIVE="$1"
            else
                die "Unknown argument: $1"
            fi
            shift
            ;;
    esac
done

[[ -n "$ARCHIVE" ]] || die "Usage: $0 <archive.zip> [--key <KEY_ID>] [--strict]" 3
[[ -f "$ARCHIVE" ]] || die "File not found: $ARCHIVE" 3

# ---------------------------------------------------------------------------
# Check GPG availability
# ---------------------------------------------------------------------------
if ! command -v gpg &>/dev/null; then
    die "gpg not found. Install GnuPG: sudo apt install gnupg" 3
fi

# ---------------------------------------------------------------------------
# Step 1: Verify GPG signature
# ---------------------------------------------------------------------------
SIG_FILE="${ARCHIVE}.sig"

if [[ ! -f "$SIG_FILE" ]]; then
    die "Signature file not found: $SIG_FILE" 1
fi

info "Verifying GPG signature..."
info "  Archive:   $ARCHIVE"
info "  Signature: $SIG_FILE"

GPG_OUTPUT=$(gpg --batch --status-fd 1 --verify "$SIG_FILE" "$ARCHIVE" 2>&1) || true

# Check for GOODSIG in status output
if echo "$GPG_OUTPUT" | grep -q "GOODSIG\|Good signature"; then
    info "GPG signature: ${GREEN}VALID${NC}"

    # Extract signer info
    SIGNER=$(echo "$GPG_OUTPUT" | grep -oP 'Good signature from "\K[^"]+' || echo "unknown")
    info "  Signed by: $SIGNER"

    # If --key specified, verify it matches
    if [[ -n "$KEY_ID" ]]; then
        if echo "$GPG_OUTPUT" | grep -qi "$KEY_ID"; then
            info "  Key match: ${GREEN}$KEY_ID${NC}"
        else
            die "Signature valid but NOT from expected key $KEY_ID" 1
        fi
    fi
else
    error "GPG verification output:"
    echo "$GPG_OUTPUT" >&2
    die "GPG signature: INVALID" 1
fi

# ---------------------------------------------------------------------------
# Step 2: Verify SHA-256 checksum (if present)
# ---------------------------------------------------------------------------
CHECKSUM_FILE="${ARCHIVE}.sha256"

if [[ -f "$CHECKSUM_FILE" ]]; then
    info "Verifying SHA-256 checksum..."

    if sha256sum --check --quiet "$CHECKSUM_FILE" 2>/dev/null; then
        info "SHA-256 checksum: ${GREEN}MATCH${NC}"
    else
        die "SHA-256 checksum: MISMATCH — file may be corrupted or tampered" 2
    fi
else
    if [[ "$STRICT" -eq 1 ]]; then
        die "Checksum file not found: $CHECKSUM_FILE (strict mode)" 2
    else
        warn "Checksum file not found: $CHECKSUM_FILE (skipping)"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
info "=== Verification PASSED ==="
info "  Archive is authentic and intact."
echo ""

exit 0
