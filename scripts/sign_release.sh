#!/usr/bin/env bash
# =============================================================================
# sign_release.sh — GPG-sign an Opti-Oignon release archive
#
# Usage:
#   ./scripts/sign_release.sh <archive.zip> [--key <KEY_ID>]
#
# Creates a detached signature file: <archive.zip>.sig
# Also generates a SHA-256 checksum file: <archive.zip>.sha256
#
# Requirements:
#   - gpg (GnuPG) installed
#   - A GPG signing key available (or specify --key <KEY_ID>)
#
# Key management:
#   Generate a key:   gpg --full-generate-key
#   List keys:        gpg --list-secret-keys --keyid-format=long
#   Export public:    gpg --armor --export <KEY_ID> > opti-oignon-release.pub
#   Import public:    gpg --import opti-oignon-release.pub
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
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo -e "${GREEN}[sign]${NC} $*"; }
warn()  { echo -e "${YELLOW}[sign]${NC} $*"; }
error() { echo -e "${RED}[sign]${NC} $*" >&2; }
die()   { error "$@"; exit 1; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
ARCHIVE=""
KEY_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --key)
            KEY_ID="$2"
            shift 2
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

[[ -n "$ARCHIVE" ]] || die "Usage: $0 <archive.zip> [--key <KEY_ID>]"
[[ -f "$ARCHIVE" ]] || die "File not found: $ARCHIVE"

# ---------------------------------------------------------------------------
# Check GPG availability
# ---------------------------------------------------------------------------
if ! command -v gpg &>/dev/null; then
    die "gpg not found. Install GnuPG: sudo apt install gnupg"
fi

GPG_VERSION=$(gpg --version | head -1)
info "Using $GPG_VERSION"

# ---------------------------------------------------------------------------
# Check for available signing keys
# ---------------------------------------------------------------------------
if [[ -z "$KEY_ID" ]]; then
    KEY_COUNT=$(gpg --list-secret-keys --keyid-format=long 2>/dev/null | grep -c '^sec' || true)
    if [[ "$KEY_COUNT" -eq 0 ]]; then
        die "No GPG secret keys found. Generate one with: gpg --full-generate-key"
    elif [[ "$KEY_COUNT" -eq 1 ]]; then
        KEY_ID=$(gpg --list-secret-keys --keyid-format=long 2>/dev/null \
            | grep '^sec' | head -1 \
            | sed -E 's/.*\/([A-F0-9]+).*/\1/')
        info "Auto-selected key: $KEY_ID"
    else
        warn "Multiple keys found. Specify one with --key <KEY_ID>"
        gpg --list-secret-keys --keyid-format=long 2>/dev/null
        die "Ambiguous key selection"
    fi
fi

info "Signing: $ARCHIVE"
info "Key:     $KEY_ID"

# ---------------------------------------------------------------------------
# Step 1: Generate SHA-256 checksum
# ---------------------------------------------------------------------------
CHECKSUM_FILE="${ARCHIVE}.sha256"
sha256sum "$ARCHIVE" > "$CHECKSUM_FILE"
info "Checksum: $CHECKSUM_FILE"
cat "$CHECKSUM_FILE"

# ---------------------------------------------------------------------------
# Step 2: Create detached GPG signature
# ---------------------------------------------------------------------------
SIG_FILE="${ARCHIVE}.sig"

# Remove existing sig if present
[[ -f "$SIG_FILE" ]] && rm -f "$SIG_FILE"

gpg --batch --yes --local-user "$KEY_ID" \
    --armor --detach-sign \
    --output "$SIG_FILE" \
    "$ARCHIVE"

info "Signature: $SIG_FILE"

# ---------------------------------------------------------------------------
# Step 3: Verify the signature we just created (sanity check)
# ---------------------------------------------------------------------------
info "Verifying signature..."
if gpg --batch --verify "$SIG_FILE" "$ARCHIVE" 2>&1; then
    info "Verification: ${GREEN}PASSED${NC}"
else
    die "Self-verification FAILED — signature may be corrupt"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
info "=== Release signed successfully ==="
info "  Archive:   $ARCHIVE"
info "  Signature: $SIG_FILE"
info "  Checksum:  $CHECKSUM_FILE"
info ""
info "To verify on another machine:"
info "  1. Import the public key:  gpg --import opti-oignon-release.pub"
info "  2. Verify signature:       ./scripts/verify_release.sh $ARCHIVE"
echo ""
