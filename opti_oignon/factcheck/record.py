"""The record of a verdict: canonical JSON, a digest, and a replay.

Every verdict carries a record: the claim, every source handed in with its
admission and validity, every span the host located with its window, every
check, the verdict with all its reasons, the text shown, and the digests of the
rules and of the configuration that decided it. It is written as canonical JSON
(sorted keys, no whitespace, UTF-8, floats refused), and its id is the SHA-256
of that JSON. The creation time and the path that computed it (the Python
reference, or later the native core with its artefact's digest) sit outside
the digest, so two paths proven equal replay to the same id.

Reproducible, not tamper-evident: the id is a digest, not a signature, and an
edited record can be digested again; the record says so ("digest_only").
``replay`` recomputes a record from its sources on the current path and says
whether it reproduced the id, or why not: a chunk whose text no longer matches
its hash ("source_changed"), rules that changed since ("rules_changed"), or a
record that differs for another reason ("record_differs").
"""

import hashlib
import json
from dataclasses import dataclass

checkpoint_before_apply = True

RECORD_VERSION = 1
OUTSIDE = ("record_id", "created_at", "computed_by")


def _refuse_floats(value, path="$"):
    if isinstance(value, float):
        raise TypeError(f"a record holds no float: {path} is {value!r}")
    if isinstance(value, dict):
        for name, inner in value.items():
            if not isinstance(name, str):
                raise TypeError(f"a record's keys are strings: {path} has {name!r}")
            _refuse_floats(inner, f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, inner in enumerate(value):
            _refuse_floats(inner, f"{path}[{index}]")


def canonical(value):
    """Canonical JSON bytes: sorted keys, no whitespace, UTF-8, floats refused."""
    _refuse_floats(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def inside(record):
    """The part of a record its id is the digest of."""
    return {name: value for name, value in record.items() if name not in OUTSIDE}


def seal(record):
    """The record with its id, and the fields that sit outside the digest."""
    sealed = dict(record)
    sealed["record_id"] = digest(inside(record))
    sealed["created_at"] = None
    sealed["computed_by"] = {"path": "reference"}
    return sealed


@dataclass(frozen=True)
class Replay:
    """What a replay found, and the path it ran on."""

    outcome: str
    path: str
    record_id: object


def replay(record, chunks, *, check=None):
    """Recompute ``record`` from the texts of its chunks, on the current path."""
    from . import decide, scope, sources

    check = check or decide.check
    by_hash = {}
    for text in chunks:
        by_hash.setdefault(hashlib.sha256(text.encode("utf-8")).hexdigest(), text)
    items = []
    for entry in record["sources_searched"]:
        rebuilt = []
        for stored in entry["chunks"]:
            text = by_hash.get(stored["text_sha256"])
            if text is None:
                return Replay("source_changed", "reference", None)
            rebuilt.append(sources.Chunk(
                text=text, sha256=stored["sha256"], ingested_at=stored["ingested_at"],
                offset_in_source=stored["offset_in_source"], locator=stored["locator"],
                extraction=stored["extraction"],
                model_quoted=tuple(tuple(pair) for pair in stored["model_quoted"])))
        items.append(sources.SourceItem(
            source_id=entry["source_id"], kind=entry["kind"], author=entry["author"], chunks=tuple(rebuilt),
            title=entry["title"], lang=entry["lang"], valid_from=entry["valid_from"],
            valid_until=entry["valid_until"], superseded_by=entry["superseded_by"],
            recorded_at=entry["recorded_at"], source_date=entry["source_date"], flags=entry["flags"],
            consent=entry["consent"]))
    if record["rules_digest"] != decide.rules_digest():
        return Replay("rules_changed", "reference", None)
    given = record["claim"]["input"]
    claim = scope.Claim(text=given["text"], lang=given["lang"], kind=given["kind"], origin=given["origin"],
                        marks=given["marks"], start=given["start"], end=given["end"])
    verdict = check(claim, items, as_of=record["claim"]["as_of"], read_on=record["claim"]["read_on"],
                    config=record["config"], canary=record["canary"])
    recomputed = verdict.record["record_id"]
    outcome = "reproduced" if recomputed == record["record_id"] else "record_differs"
    return Replay(outcome, "reference", recomputed)
