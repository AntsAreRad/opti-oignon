#!/usr/bin/env python3
"""
reseal_model_manifest.py -- Re-seal the model provenance manifest.

Fortress mode REQUIRES a post-quantum signature. It is a property of the mode,
like the socket bind, and no configuration file switches it off. A host whose
manifest is sealed with a symmetric MAC is therefore a host that refuses every
model it owns: enforcement reads the MAC as a downgrade, and it is right to.

The seal used to be written only as a SIDE EFFECT of enrolling a model, so the
one way to change it was to download every model again. This is the handle that
was missing.

Nothing is re-pinned. The digests already in the manifest are carried across
verbatim -- this changes the SCHEME, never the CLAIM. A re-seal that re-hashed
the files on disk would bless whatever is sitting there now, which is precisely
the substitution the manifest exists to refuse.

The full migration onto a fortress, in order:

    1. python -c "from opti_oignon.pqc_signatures import *; \
           save_pqc_keypair(*generate_pqc_keypair())"
       (or: POST /api/security/pqc/generate-keys)
    2. python scripts/reseal_model_manifest.py
    3. Escalate to Bulbe. The escalation refuses until 1 and 2 are done.

Usage:
    python scripts/reseal_model_manifest.py [--manifest PATH] [--dry-run]

Exit code 0 on success, 1 on refusal.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opti_oignon.model_provenance import (  # noqa: E402
    ProvenanceError,
    load_manifest,
    manifest_seal_scheme,
    reseal_manifest,
    resolve_seal_keys,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--manifest", default=None, help="Path to the manifest (default: data/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change and write nothing",
    )
    args = parser.parse_args()

    path = Path(args.manifest) if args.manifest else None

    manifest = load_manifest(path)
    if not manifest:
        print("No model provenance manifest. Nothing to re-seal.")
        print("A manifest is written when a model is enrolled; enrol one first.")
        return 1

    current = manifest_seal_scheme(path)
    entries = manifest.get("entries") or {}
    print(f"Manifest: {len(entries)} model(s), sealed with {current or 'nothing'}")

    keys = resolve_seal_keys()
    if keys is None:
        print()
        print("REFUSED: no key material this host is allowed to seal with.")
        print("A signature is required here and there is nothing to sign with.")
        print("Mint a keypair first -- and note that substituting a MAC would")
        print("hand back a manifest the fortress rejects while you believed the")
        print("migration had happened.")
        return 1

    if current == keys.scheme:
        print(f"Already sealed with {keys.scheme}. Nothing to do.")
        return 0

    if args.dry_run:
        print(f"Would re-seal: {current or 'nothing'} -> {keys.scheme}")
        print(f"Would carry {len(entries)} pin(s) across unchanged.")
        return 0

    try:
        result = reseal_manifest(manifest_path=path)
    except ProvenanceError as exc:
        print(f"REFUSED: {exc}")
        return 1

    print(
        f"Re-sealed: {result['previous_scheme'] or 'nothing'} -> "
        f"{result['scheme']} ({result['entries']} pin(s) carried across "
        f"unchanged)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
