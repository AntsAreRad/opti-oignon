"""Sources as a store hands them in: items, chunks, who wrote them, and when they held.

A source item is a unit a store can name: its id, its kind, its author, its
dates, its flags, the owner's consent, and its chunks, each chunk the text
exactly as the store keeps it with the SHA-256 recorded at ingest.

Admission refuses by name, and every refusal is recorded, never dropped: a
search engine's excerpt, a model's words (an assistant turn, a model-extracted
memory, an agent's note), an unknown author (the core never guesses), no
consent, a retracted source (kept as a flag), a fact whose successor was not
handed in; per chunk, a hash that no longer matches, a text that is not NFC,
a chunk over the size limit; and every item past the budgets.

Validity at a date t: a source starts at its "valid from", else at the date it
was recorded ("recorded on"), else it is treated as started and flagged
"validity unknown". For a claim about the present ("the current CEO"), a
source does not speak for any date before its own: its start is its own date
when that is later. It ends at its "valid until", or, for a fact-level item (a
ledger decision, a Core entry) replaced by an item handed in, at the start of
its successor when that comes first, because the drift ledger links a
replaced fact to its successor and never writes an end date. When the
successor's start is unknown, or earlier than the replaced fact's own, the
dates cannot say when the replaced fact held: it is valid at no date, and
both are flagged "validity unknown". A multi-statement item (a note, a
document) carrying a successor is flagged coarse and the link is ignored, so a
replacement of one of its facts never marks the others. It is valid at t when
it has started by t and not ended by t.
"""

import dataclasses
import datetime
import unicodedata
from dataclasses import dataclass, field

from . import passage
from . import vocabulary as V

checkpoint_before_apply = True

SOURCES_VERSION = 1

TIER_BY_KIND = {"ledger": "own_decision", "core": "own_decision", "note": "own_note",
                "library": "library", "web": "web"}


def rules():
    return {"version": SOURCES_VERSION, "tiers": dict(TIER_BY_KIND), "refusals": list(V.REFUSALS),
            "fact_level": list(V.FACT_LEVEL_KINDS)}


def day(value, name="date", *, required=False):
    """An ISO date (or the date part of an ISO timestamp), or None when empty."""
    if value is None or value == "":
        if required:
            raise ValueError(f"{name} is required, as an ISO date")
        return None
    if not isinstance(value, str) or len(value) < 10:
        raise ValueError(f"{name} {value!r} is not an ISO date")
    try:
        return datetime.date.fromisoformat(value[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} {value!r} is not an ISO date") from exc


@dataclass(frozen=True)
class Chunk:
    """A chunk's text exactly as stored, with the hash recorded at ingest."""

    text: str
    sha256: str
    ingested_at: str = ""
    offset_in_source: object = None
    locator: dict = field(default_factory=dict)
    extraction: object = None
    model_quoted: tuple = ()

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError("a chunk's text is a string")
        if not (isinstance(self.sha256, str) and len(self.sha256) == 64):
            raise ValueError("a chunk carries the SHA-256 of its text, 64 hex characters")
        day(self.ingested_at, "ingested_at")
        if self.offset_in_source is not None and (
                isinstance(self.offset_in_source, bool) or not isinstance(self.offset_in_source, int)):
            raise ValueError("offset_in_source is a number of code points, or None")
        object.__setattr__(self, "locator", dict(self.locator or {}))
        if self.extraction is not None:
            extraction = dict(self.extraction)
            flags = tuple(extraction.get("flags") or ())
            stray = [f for f in flags if f not in V.EXTRACTION_FLAGS]
            if stray:
                raise ValueError(f"unknown extraction flags {stray}")
            extraction["flags"] = flags
            object.__setattr__(self, "extraction", extraction)
        ranges = []
        for pair in self.model_quoted or ():
            start, end = pair
            if isinstance(start, bool) or isinstance(end, bool) or not (0 <= start <= end):
                raise ValueError(f"a model-quoted range is two code-point offsets, not {pair!r}")
            ranges.append((int(start), int(end)))
        object.__setattr__(self, "model_quoted", tuple(ranges))


@dataclass(frozen=True)
class SourceItem:
    """A source a store can name, with its chunks."""

    source_id: str
    kind: str
    author: str
    chunks: tuple = ()
    title: str = ""
    lang: str = "und"
    valid_from: str = ""
    valid_until: str = ""
    superseded_by: object = None
    recorded_at: str = ""
    source_date: object = None
    flags: dict = field(default_factory=dict)
    consent: object = None

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("a source item needs an id")
        if self.kind not in V.SOURCE_KINDS:
            raise ValueError(f"unknown source kind {self.kind!r}")
        if self.author not in V.AUTHORS:
            raise ValueError(f"unknown author {self.author!r}")
        if self.lang not in V.LANGS:
            raise ValueError(f"unknown language {self.lang!r}")
        for name in ("valid_from", "valid_until", "recorded_at", "source_date"):
            day(getattr(self, name), name)
        flags = dict(self.flags or {})
        for name, value in flags.items():
            if name not in V.ITEM_FLAGS:
                raise ValueError(f"unknown source flag {name!r}")
            if name in V.DATED_ITEM_FLAGS:
                day(value, name, required=True)
            flags[name] = value if value is not None else ""
        object.__setattr__(self, "flags", flags)
        chunks = tuple(self.chunks)
        if not all(isinstance(c, Chunk) for c in chunks):
            raise TypeError("an item's chunks are Chunk objects")
        object.__setattr__(self, "chunks", chunks)


def tier(item):
    return TIER_BY_KIND.get(item.kind)


def start_of(item):
    """(date, how it was derived)."""
    if item.valid_from:
        return day(item.valid_from), "valid_from"
    if item.recorded_at:
        return day(item.recorded_at), "recorded_at"
    return None, "unknown"


@dataclass(frozen=True)
class Admission:
    """An item's admission, and each of its chunks' in sorted order."""

    result: str
    chunks: tuple


def sorted_chunks(item):
    return sorted(item.chunks, key=lambda c: (c.sha256, c.offset_in_source if c.offset_in_source is not None
                                              else -1, c.text))


def admit(items, *, limits):
    """Admission of every item, in the order given (the caller sorts by id)."""
    by_id = {item.source_id: item for item in items}
    out = {}
    total = 0
    counted = 0
    exhausted = False
    for item in items:
        refusal = None
        if item.kind == "snippet":
            refusal = "snippet"
        elif item.author == "model":
            refusal = "author_model"
        elif item.author == "unknown":
            refusal = "author_unknown"
        elif not item.consent:
            refusal = "no_consent"
        elif "retracted" in item.flags:
            refusal = "retracted"
        elif (item.kind in V.FACT_LEVEL_KINDS and item.superseded_by
              and item.superseded_by not in by_id):
            refusal = "successor_missing"
        results = []
        for index, chunk in enumerate(sorted_chunks(item)):
            if index >= limits["max_chunks_per_item"]:
                result = "over_budget"
            elif passage.sha256(chunk.text) != chunk.sha256:
                result = "chunk_changed"
            elif not unicodedata.is_normalized("NFC", chunk.text):
                result = "chunk_not_nfc"
            elif len(chunk.text) > limits["max_chunk_chars"]:
                result = "chunk_too_large"
            else:
                result = "admitted"
            results.append((chunk, result))
        if refusal is None:
            counted += 1
            size = sum(len(c.text) for c, r in results if r == "admitted")
            if exhausted or counted > limits["max_items"] or total + size > limits["max_total_chars"]:
                refusal = "over_budget"
                exhausted = True
            else:
                total += size
        if refusal is None and results and all(r != "admitted" for _, r in results):
            refusal = next(r for _, r in results)
        out[item.source_id] = Admission(refusal or "admitted", tuple(results))
    return out


@dataclass(frozen=True)
class Validity:
    """When an item held, how each end was derived, and whether it held at t."""

    start: object
    start_from: str
    end: object
    end_from: object
    valid: bool
    flags: tuple
    successor: object


def validity(item, by_id, as_of, *, time_sensitive=False):
    flags = []
    start, start_from = start_of(item)
    if time_sensitive and item.source_date:
        own = day(item.source_date)
        if start is None or own > start:
            start, start_from = own, "source_date"
    if start is None:
        flags.append("validity_unknown")
    successor = None
    if item.superseded_by:
        if item.kind in V.FACT_LEVEL_KINDS:
            if item.superseded_by in by_id:
                successor = item.superseded_by
        else:
            flags.append("supersession_coarse")
    end, end_from = None, None
    never = False
    if item.valid_until:
        end, end_from = day(item.valid_until), "valid_until"
    if successor is not None:
        following, _ = start_of(by_id[successor])
        if following is None:
            never, end_from = True, "successor_undated"
        elif start is not None and following < start:
            never, end, end_from = True, following, "successor_earlier"
        elif end is None or following < end:
            end, end_from = following, "successor_start"
        if never and "validity_unknown" not in flags:
            flags.append("validity_unknown")
    valid = not never and (start is None or start <= as_of) and (end is None or as_of < end)
    return Validity(start, start_from, end, end_from, valid, tuple(flags), successor)


def validities(items, by_id, as_of, *, time_sensitive=False):
    """Every item's validity at ``as_of``; a successor dated before what it replaced is flagged too."""
    out = {item.source_id: validity(item, by_id, as_of, time_sensitive=time_sensitive) for item in items}
    for held in list(out.values()):
        if held.end_from == "successor_earlier" and "validity_unknown" not in out[held.successor].flags:
            following = out[held.successor]
            out[held.successor] = dataclasses.replace(following, flags=following.flags + ("validity_unknown",))
    return out
