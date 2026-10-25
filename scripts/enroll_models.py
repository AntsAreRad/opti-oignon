#!/usr/bin/env python3
"""
enroll_models.py -- Pin the GGUF models already on disk in the provenance manifest.

Under enforcement -- which every mode that is not Daily applies -- the load seam
refuses any model whose bytes are not pinned in the manifest. A fresh host has
no manifest at all, so the instant it enters the fortress every model it owns is
refused as unpinned. Nothing crashes; it simply loads nothing.

The pin used to be written only as a SIDE EFFECT of downloading a model, so a
host that already held its weights had no way to enrol them short of fetching
them again. This is the handle that was missing. It hashes the bytes on disk,
pins each one, and seals the manifest once with the strongest scheme this host
is allowed to use.

Enrolment decides these bytes are the right ones, so it HASHES them. That is why
it differs from re-sealing, which renews a decision already made and must never
re-hash. Run this when the models are the ones you meant to have.

It carries only what the manifest's seal vouches for. A manifest whose seal does
not verify here -- one a process without the key wrote into, whose seal was
deleted, or sealed under a key this host cannot check -- is refused by its
reason, and nothing is written. Once the weights are checked against where they
came from, --start-again sets that manifest aside, untouched, and pins the models
on disk from their bytes, saying how many entries it dropped. A pinned model
whose bytes changed is never re-pinned on the way: it is named, nothing is
written, and --repin NAME re-pins it once its bytes are checked.

The full migration onto a fortress, in order:

    1. python -c "from opti_oignon.pqc_signatures import *; \
           save_pqc_keypair(*generate_pqc_keypair())"
       (or: POST /api/security/pqc/generate-keys)
    2. python scripts/enroll_models.py         # pins on-disk weights, sealed PQC
    3. Escalate to Bulbe. The escalation refuses until the models are pinned.

If models were enrolled earlier under Daily (an HMAC seal) and you mint a key
afterwards, run scripts/reseal_model_manifest.py to lift the seal to a signature
without re-hashing.

Usage:
    python scripts/enroll_models.py [--manifest PATH] [--backends PATH] [--dry-run]
                                    [--start-again] [--repin NAME ...]

Exit code 0 on success, 1 on refusal.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opti_oignon.model_provenance import (  # noqa: E402
    REASON_KEY_UNAVAILABLE,
    REASON_MANIFEST_MISSING,
    REASON_VERIFIED,
    ProvenanceError,
    enroll_models,
    read_manifest,
    resolve_seal_keys,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BACKENDS_YAML = _PROJECT_ROOT / "opti_oignon" / "config" / "backends.yaml"


def _model_dirs(backends_path: Path) -> list[Path]:
    """The directories the load seam scans, read from backends.yaml.

    Discovery lives HERE and not in the provenance module: that module keeps a
    stdlib-only import surface on purpose, because the load seam treats an
    unimportable provenance module as a refusal. A YAML read belongs in the
    handle, which is free to fail loudly.
    """
    try:
        import yaml

        with open(backends_path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001 - report and carry on with nothing
        print(f"backends.yaml could not be read: {exc}")
        return []

    llama = cfg.get("llama_cpp", {}) or {}
    dirs = list(llama.get("model_dirs", []) or [])
    default_dir = llama.get("default_download_dir")
    if default_dir:
        dirs.append(default_dir)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for d in dirs:
        p = Path(d).expanduser()
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def _discover(backends_path: Path) -> list[Path]:
    found: dict[str, Path] = {}
    for d in _model_dirs(backends_path):
        if not d.is_dir():
            continue
        for gguf in sorted(d.glob("*.gguf")):
            found.setdefault(gguf.name, gguf)
    return list(found.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--manifest", default=None, help="Path to the manifest (default: data/)"
    )
    parser.add_argument(
        "--backends",
        default=None,
        help="Path to backends.yaml (default: opti_oignon/config/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be enrolled and write nothing",
    )
    parser.add_argument(
        "--start-again",
        action="store_true",
        help="Set a manifest that does not verify aside and pin the models on disk afresh",
    )
    parser.add_argument(
        "--repin",
        action="append",
        default=[],
        metavar="NAME",
        help="Re-pin this pinned model whose bytes changed, once they are checked (repeatable)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else None
    backends_path = Path(args.backends) if args.backends else _BACKENDS_YAML

    models = _discover(backends_path)
    if not models:
        print("No GGUF models found in the configured directories.")
        print(f"Looked under the dirs named in {backends_path}.")
        print("Nothing to enrol. Add models or check llama_cpp.model_dirs.")
        return 0

    keys = resolve_seal_keys()
    if keys is None:
        print("REFUSED: no key material this host is allowed to seal with.")
        print("A signature is required here and there is nothing to sign with.")
        print("Mint a keypair first: POST /api/security/pqc/generate-keys")
        print("Enrolling without it would leave no manifest, and the escalation")
        print("would keep refusing -- correctly.")
        return 1

    reading = read_manifest(manifest_path, keys=keys)
    verified = reading.status == REASON_VERIFIED
    pinned = set(reading.entries) if verified else set()
    fresh = [m for m in models if m.name not in pinned]
    print(f"Found {len(models)} model(s); {len(fresh)} not yet pinned.")
    for m in models:
        mark = " (already pinned)" if m.name in pinned else ""
        print(f"  {m.name}{mark}")

    missing = reading.status == REASON_MANIFEST_MISSING
    if args.start_again and (verified or missing or not reading.read):
        print()
        why = ("it verifies here: enrol without --start-again" if verified else
               "there is none: enrol without --start-again" if missing else
               f"{reading.detail}: make it readable first")
        print(f"REFUSED ({reading.status}): nothing to start again from -- {why}.")
        return 1
    if not (verified or missing or args.start_again):
        print()
        print(f"REFUSED ({reading.status}) -- the manifest: {reading.detail}.")
        print("Its entries vouch for nothing; none is carried, and nothing is written.")
        if reading.status == REASON_KEY_UNAVAILABLE:
            print("Set OPTI_KEYFILE_PASSPHRASE, or put back the keypair that sealed it; or,")
        print("once the weights are checked against where they came from, start again:")
        print("  python scripts/enroll_models.py --start-again")
        return 1

    if args.dry_run:
        print()
        print(f"Manifest: {reading.status}.")
        if args.start_again:
            print(f"Would set it aside and pin {len(models)} model(s) afresh, sealed with {keys.scheme}.")
        else:
            print(f"Would pin {len(fresh)} new model(s), sealed with {keys.scheme}.")
            print("A pinned model whose bytes changed is found only by hashing: the real run refuses it.")
        return 0

    try:
        result = enroll_models(
            models, manifest_path=manifest_path, keys=keys, start_again=args.start_again, repin=args.repin
        )
    except ProvenanceError as exc:
        print()
        print(f"REFUSED: {exc}")
        return 1

    print()
    for item in result["enrolled"]:
        print(f"  pinned {item['model']} {item['sha256']}")
    print(
        f"Enrolled {result['count']} model(s), sealed with {result['scheme']}; "
        f"{len(result['kept'])} kept as pinned."
    )
    if result["repin_unused"]:
        print(f"Not re-pinned (no changed pin by that name): {', '.join(result['repin_unused'])}")
    if not result["written"]:
        print("Nothing new: the manifest is left as it is.")
    dropped = result.get("dropped")
    if dropped:
        count = "an unknown number of" if dropped["count"] is None else dropped["count"]
        print(f"Started again: {count} entr(ies) dropped ({dropped['reason']}); the old manifest is set aside at")
        print(f"  {result['set_aside']}")
        for name in dropped["names"]:
            print(f"  dropped {name!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
