#!/usr/bin/env bash
# =============================================================================
# verify_release.sh -- Verify GPG signature of an Opti-Oignon release archive
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
#   0  -- Signature valid (and checksum matches if present)
#   1  -- Signature invalid or missing
#   2  -- Checksum mismatch
#   3  -- Missing dependencies or files
#
# See SECURITY.md section "Release Signing" for full documentation.
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
die()   { error "$1"; exit "${2:-1}"; }

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
# Pinned project key
# ---------------------------------------------------------------------------
# A fingerprint recorded next to this script names THE release key. When no
# --key is given and the pin exists, the pin is the expected key. Without
# either, a valid signature only proves the archive matches some key in the
# local keyring -- integrity, not identity.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN_FILE="${SCRIPT_DIR}/release_key.fpr"
if [[ -z "$KEY_ID" && -f "$PIN_FILE" ]]; then
    KEY_ID="$(tr -d '[:space:]' < "$PIN_FILE")"
    [[ -n "$KEY_ID" ]] || die "Pinned key file is empty: $PIN_FILE" 3
    info "Expected key: pinned project fingerprint ($(basename "$PIN_FILE"))"
fi

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

# The verdict is taken from the machine-readable status channel, written to
# its own file, and never from the rendered output. A signer chooses their
# own user id, so anything printed for humans is partly the signer's text:
# searching it for words like "Good signature" lets whoever signed the
# archive write the sentence that judges it.
STATUS_FILE=$(mktemp)
trap 'rm -f "$STATUS_FILE"' EXIT

GPG_OUTPUT=$(gpg --batch --status-file "$STATUS_FILE" \
    --verify "$SIG_FILE" "$ARCHIVE" 2>&1) || true

# GOODSIG is emitted only for a signature that covers the bytes on disk.
if grep -q '^\[GNUPG:\] GOODSIG ' "$STATUS_FILE"; then
    info "GPG signature: ${GREEN}VALID${NC}"

    # VALIDSIG carries the full fingerprint of the signing key.
    # Extraction must not decide anything: a missing line is reported as
    # unknown, never as a failure that happens to stop the script. The
    # verdict above is the only thing allowed to refuse.
    SIGNING_FPR=$(grep -m1 '^\[GNUPG:\] VALIDSIG ' "$STATUS_FILE" \
        | awk '{print $3}' || true)
    SIGNER=$(grep -m1 '^\[GNUPG:\] GOODSIG ' "$STATUS_FILE" \
        | cut -d' ' -f4- || true)
    info "  Signed by: ${SIGNER:-unknown}"
    info "  Key fingerprint: ${SIGNING_FPR:-unknown}"

    # If --key specified, it must match the signing key itself. A hex string
    # that merely appears somewhere in the output is not a match.
    if [[ -z "$KEY_ID" ]]; then
        warn "No pinned key: this proves integrity for a key in the local"
        warn "keyring, not the identity of the project. Record the release"
        warn "fingerprint in scripts/release_key.fpr or pass --key."
    fi
    if [[ -n "$KEY_ID" ]]; then
        WANTED="${KEY_ID^^}"
        WANTED="${WANTED#0X}"
        HAVE="${SIGNING_FPR^^}"
        if [[ -n "$HAVE" && "$HAVE" == *"$WANTED" ]]; then
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
        die "Checksum file not found: $CHECKSUM_FILE (strict mode)" 3
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
