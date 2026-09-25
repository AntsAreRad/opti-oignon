"""The being's chain in its store: links, the local digest, the keyed anchor, verification, redaction.

Each fact has an eid computed by the engine from its envelope alone, and
each has a local link, ``link = sha256(OCJ({"eid", "prev", "seq"}))``, the
previous link being ``"0" * 64`` at genesis. The link and the seq never leave
the device; the eid is the same on every device that holds the fact.

The store anchor (``meta.anchor``) is rewritten inside every committing
transaction. Its MAC, under a key derived from the master key (a plain
SHA-256, advisory only, in a glass jar), covers the head, the write
generation, the owner, the soil and a digest of the local tables outside the
journal: the key names and the payload references, the rhythm layer's row
count and highest ``rseq``, the heard row count of each season, each
checkpoint's key and ``state_hash``, the drop count, the rhythm floor, the
ended seasons and the destruction records not yet anchored. So an old page
written back into the file is refused even where the file's own page checks
pass. Row contents are not in that digest: a key's bytes, a rhythm or heard
ciphertext and a checkpoint blob edited in place rest on their own checks --
a ciphertext opens only under its own key (AEAD), and a checkpoint blob is
read back against its ``state_hash`` -- not on the anchor.

Verification reads everything it needs in one short read transaction, then
computes outside it, so another writer is never kept waiting on the hashes.
The first failure decides, and a break in the chain is refused with the
event it breaks at. A refused verification writes nothing.

A resume, which only a person asks for, sets aside what failed to verify
and never rewrites what it keeps: the tail above the last sound event goes,
every recorded destruction is enforced again, and a forget whose forgetter
went with the tail is issued again.
"""

import hashlib
import hmac

checkpoint_before_apply = True

ZERO = "0" * 64
CHUNK = 3000
SOILS = ("encrypted", "glass")
ANCHOR_KEYS = ("gen", "key", "mac", "seq")
PENDING_KEYS = ("destroyed", "ended", "forgot", "rhythm_floor")
_HEX = "0123456789abcdef"
_BAD = object()

ROWS = ("SELECT l.seq, l.prev, l.eid, l.link, f.eid, f.being, f.t, f.kind, f.origin, f.oseq, f.laws, "
        "f.body_sha256, b.body, b.redacted_by FROM links l LEFT JOIN facts f ON f.eid = l.eid "
        "LEFT JOIN bodies b ON b.eid = l.eid")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex(value, length):
    if not isinstance(value, str) or len(value) != length:
        return False
    for char in value:
        if char not in _HEX:
            return False
    return True


def _exactly(value, names):
    if not isinstance(value, dict):
        return False
    for name in value:
        if name not in names:
            return False
    for name in names:
        if name not in value:
            return False
    return True


# ---------------------------------------------------------------------------
# Links, the local digest, the anchor
# ---------------------------------------------------------------------------
def link(eid, prev, seq):
    """The local link of a fact: ``sha256(OCJ({"eid", "prev", "seq"}))``."""
    from . import wire

    return hashlib.sha256(wire.emit({"eid": eid, "prev": prev, "seq": seq})).hexdigest()


def parse_meta(rows):
    """``meta`` rows as values; a value that is not OCJ is kept as a marker no check accepts."""
    from . import wire

    meta = {}
    for key, value in rows:
        try:
            meta[key] = wire.parse(bytes(value))
        except Exception:  # noqa: BLE001 - an unreadable value is refused where it is checked
            meta[key] = _BAD
    return meta


def read_local(conn):
    """The local tables outside the journal, as the local digest and the structural checks read them."""
    keys = [(row[0], row[1]) for row in conn.execute("SELECT name, key FROM keys ORDER BY name")]
    payloads = [(row[0], row[1]) for row in conn.execute("SELECT ref, eid FROM payloads ORDER BY ref")]
    rhythm = conn.execute("SELECT COUNT(*), MAX(rseq), MIN(rseq) FROM rhythm_facts").fetchone()
    heard = [(row[0], row[1]) for row in
             conn.execute("SELECT season, COUNT(*) FROM heard GROUP BY season ORDER BY season")]
    checkpoints = [tuple(row) for row in conn.execute(
        "SELECT t, engine, laws, through, state_hash FROM checkpoints ORDER BY t, engine, laws, through, state_hash")]
    dropped = conn.execute("SELECT COALESCE(SUM(dropped), 0) FROM overflow").fetchone()[0]
    return {"checkpoints": checkpoints, "dropped": dropped, "heard": heard, "keys": keys, "payloads": payloads,
            "rhythm": (rhythm[0], rhythm[1], rhythm[2])}


def local_digest(local, meta):
    """The digest of the local tables and the local meta values the anchor covers."""
    from . import wire

    rhythm_max = local["rhythm"][1]
    value = {
        "checkpoints": sorted([list(row) for row in local["checkpoints"]]),
        "cross_pending": meta.get("cross_pending", []),
        "dropped": local["dropped"],
        "heard": sorted([[season, count] for season, count in local["heard"]]),
        "heard_ended": meta.get("heard_ended", []),
        "keys": sorted(name for name, _ in local["keys"]),
        "payloads": sorted([[ref, eid] for ref, eid in local["payloads"]]),
        "rhythm": [local["rhythm"][0], -1 if rhythm_max is None else rhythm_max],
        "rhythm_floor": meta.get("rhythm_floor", 0),
    }
    return hashlib.sha256(wire.emit(value)).hexdigest()


def anchor_input(*, being, gen, head, local, owner, seq, soil):
    """The bytes the store anchor's MAC covers."""
    from . import wire

    return wire.emit({"being": being, "gen": gen, "head": head, "local": local, "owner": owner, "seq": seq,
                      "soil": soil})


def anchor_value(*, being, gen, head_seq, head_link, local, owner, soil, key_id, anchor_key):
    """``meta.anchor``: ``{"gen", "key", "mac", "seq"}`` over the head and the local digest."""
    from . import anchors

    data = anchor_input(being=being, gen=gen, head=head_link, local=local, owner=owner, seq=head_seq, soil=soil)
    return {"gen": gen, "key": key_id, "mac": anchors.mac(anchor_key, data), "seq": head_seq}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_state(conn, since=None):
    """Everything verification reads, in one short read transaction; rows from ``since`` when given."""
    conn.execute("BEGIN")
    done = False
    try:
        if since is None:
            rows = conn.execute(ROWS + " ORDER BY l.seq").fetchall()
        else:
            rows = conn.execute(ROWS + " WHERE l.seq >= ? ORDER BY l.seq", (since,)).fetchall()
        state = {
            "checkpoints": conn.execute("SELECT c.through, l.seq FROM checkpoints c "
                                        "LEFT JOIN links l ON l.eid = c.through").fetchall(),
            "local": read_local(conn),
            "meta": parse_meta(conn.execute("SELECT key, value FROM meta").fetchall()),
            "orphan": conn.execute("SELECT eid FROM facts EXCEPT SELECT eid FROM links LIMIT 1").fetchall(),
            "payloads": conn.execute("SELECT p.ref, p.eid, b.body, l.seq FROM payloads p "
                                     "LEFT JOIN bodies b ON b.eid = p.eid "
                                     "LEFT JOIN links l ON l.eid = p.eid ORDER BY p.ref").fetchall(),
            "rows": rows,
            "top": conn.execute("SELECT MAX(seq) FROM links").fetchone()[0],
        }
        done = True
    finally:
        try:
            conn.execute("COMMIT" if done else "ROLLBACK")
        except Exception:  # noqa: BLE001 - the read's own failure is the one to report
            if done:
                raise
    return state


def target_info(conn, eid):
    """``(seq, kind, body_is_null, redacted_by)`` of a linked fact, or ``None``."""
    row = conn.execute("SELECT l.seq, f.kind, b.body IS NULL, b.redacted_by FROM links l "
                       "JOIN facts f ON f.eid = l.eid LEFT JOIN bodies b ON b.eid = l.eid WHERE l.eid = ?",
                       (eid,)).fetchone()
    return None if row is None else (row[0], row[1], bool(row[2]), row[3])


def link_at(conn, seq):
    row = conn.execute("SELECT link FROM links WHERE seq = ?", (seq,)).fetchone()
    return None if row is None else row[0]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _strict_object(data):
    from . import wire

    try:
        raw = bytes(data)
        value = wire.parse(raw)
    except Exception:  # noqa: BLE001 - bytes that do not parse are not a body
        return None
    if not isinstance(value, dict) or wire.emit(value) != raw:
        return None
    return value


def _ask(envelopes):
    from . import engine, wire

    try:
        answer = wire.parse(engine.call(wire.emit({"envelopes": envelopes, "op": "fact_envelope", "v": 1})))
    except Exception:  # noqa: BLE001 - an envelope the codec refuses has no eid
        return None
    eids = answer.get("eids") if isinstance(answer, dict) else None
    if not isinstance(eids, list) or len(eids) != len(envelopes):
        return None
    return eids


def engine_eids(envelopes):
    """The engine's eid of each envelope, in chunks; ``None`` for an envelope it refuses."""
    out = []
    for start in range(0, len(envelopes), CHUNK):
        chunk = envelopes[start:start + CHUNK]
        eids = _ask(chunk)
        if eids is None:
            eids = []
            for envelope in chunk:
                one = _ask([envelope])
                eids.append(None if one is None else one[0])
        out.extend(eids)
    return out


class Verified:
    """A store that passed verification, as the verification read it."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def refusal(self, seq, reason, detail=""):
        """A refusal after every row passed: the head is what is kept."""
        from .store import ChainRefused

        through = max(self.through, self.head)
        return ChainRefused(seq, reason, kept_seq=self.head, kept_t=self.head_t, through=through,
                            discarded=through - self.head, detail=detail, birth=self.birth, gen=self.meta["gen"])


def _genesis_ok(body, table):
    from . import membrane

    if body is None:
        return False
    try:
        membrane.check_body(table["kinds"]["genesis"]["body"], body)
    except membrane.MembraneRefused:
        return False
    return True


def _pin_for(rows):
    """The journal pin of the law the genesis names, or of the first carried law when it cannot be read.

    A genesis that reads -- the first row, of kind ``genesis``, its bytes
    hashing to its digest -- and names a law this engine does not carry is
    refused ``law``: its being is never read under another law's kinds and
    budgets. A genesis that cannot be read is left to the per-row checks,
    which refuse it at event #0.
    """
    from . import lawfiles, membrane
    from .store import StoreRefused

    name = None
    if rows and rows[0][12] is not None:
        body = _strict_object(rows[0][12])
        laws = body.get("laws") if isinstance(body, dict) else None
        name = laws.get("name") if isinstance(laws, dict) else None
        sound = (rows[0][0] == 0 and rows[0][7] == "genesis"
                 and hashlib.sha256(bytes(rows[0][12])).hexdigest() == rows[0][11])
        if sound and isinstance(name, str) and name not in lawfiles.LAWS:
            raise StoreRefused("law", "the genesis names a law this engine does not carry")
    if isinstance(name, str) and name in lawfiles.LAWS:
        return membrane.law_pin(name)
    return membrane.law_pin(lawfiles.LAWS[0])


def verify(conn, *, name_tag, name_soil, key_id, anchor_key, cross_seq=None, prior=None):
    """Steps 1 to 7 of verification; a ``Verified`` or the first refusal met.

    ``prior`` is the ``Verified`` of an earlier pass on the same connection:
    only the rows above its head are checked again, from its link, unless
    the segment moved (another process resumed) or its head is no longer
    there, in which case the whole chain is.
    """
    state = read_state(conn, None if prior is None else prior.head)
    if prior is not None:
        rows = state["rows"]
        first = rows[0] if rows else None
        if (first is None or first[0] != prior.head or first[3] != prior.head_link
                or state["meta"].get("segment") != prior.segment):
            return verify(conn, name_tag=name_tag, name_soil=name_soil, key_id=key_id, anchor_key=anchor_key,
                          cross_seq=cross_seq, prior=None)
    return _check(conn, state, name_tag=name_tag, name_soil=name_soil, key_id=key_id, anchor_key=anchor_key,
                  cross_seq=cross_seq, prior=prior)


def _check(conn, state, *, name_tag, name_soil, key_id, anchor_key, cross_seq, prior):
    from . import anchors
    from .store import ChainRefused, StoreRefused

    meta = state["meta"]
    rows = state["rows"]
    top = state["top"] if _is_int(state["top"]) else -1
    keys = state["local"]["keys"]
    anchor = meta.get("anchor")
    context = {"birth": None}

    def through():
        values = [top]
        if isinstance(anchor, dict) and _is_int(anchor.get("seq")):
            values.append(anchor["seq"])
        cross = meta.get("cross")
        if isinstance(cross, dict) and _is_int(cross.get("seq")):
            values.append(cross["seq"])
        if cross_seq is not None:
            secret = dict((name, value) for name, value in keys if isinstance(name, str)).get("being_secret")
            if isinstance(secret, bytes) and len(secret) == 32:
                values.append(cross_seq(anchors.being_tag(secret)))
        return max(values)

    def refuse(seq, reason, kept_seq, kept_t, detail=""):
        last = through()
        return ChainRefused(seq, reason, kept_seq=kept_seq, kept_t=kept_t, through=last,
                            discarded=last - kept_seq, detail=detail, birth=context["birth"], gen=meta.get("gen"))

    # 1. Meta.
    checks = (
        ("schema", lambda v: v == 1 and _is_int(v)),
        ("being", lambda v: _is_hex(v, 32)),
        ("owner", lambda v: _is_hex(v, 32)),
        ("soil", lambda v: v in SOILS),
        ("origin", lambda v: _is_hex(v, 16)),
        ("gen", lambda v: _is_int(v) and v >= 1),
        ("segment", lambda v: _is_int(v) and v >= 0),
    )
    for key, good in checks:
        value = meta.get(key, _BAD)
        if value is _BAD or not good(value):
            raise StoreRefused("local", f"meta {key}")
    if "anchor" not in meta:
        raise refuse(0, "anchor", top, 0, "the store has no anchor")
    # 2. The file name binds the owner and the soil.
    if name_tag != meta["owner"]:
        raise StoreRefused("owner", "the file name names another owner")
    if name_soil != meta["soil"]:
        raise StoreRefused("soil", "the file name names another soil")
    # 3. An empty chain never passes.
    if not rows:
        raise refuse(0, "genesis", -1, 0, "the chain is empty")

    if prior is None:
        pin = _pin_for(rows)
        check_rows = rows
        expected_seq, expected_prev = 0, ZERO
        kept_seq, kept_t = -1, 0
        genesis, owner_to, gaps = None, None, []
    else:
        pin = prior.pin
        check_rows = rows[1:]
        expected_seq, expected_prev = prior.head + 1, prior.head_link
        kept_seq, kept_t = prior.head, prior.head_t
        genesis, owner_to, gaps = prior.genesis, prior.owner, list(prior.gaps)
        context["birth"] = genesis["birth"]["wall"]
    table = pin["table"]
    kinds = table["kinds"]

    envelopes = []
    positions = []
    for index, row in enumerate(check_rows):
        if row[4] is not None:
            envelopes.append({"being": row[5], "body": row[11], "kind": row[7], "laws": row[10],
                              "origin": row[8], "oseq": row[9], "t": row[6]})
            positions.append(index)
    computed = engine_eids(envelopes)
    eid_at = {}
    for index, value in zip(positions, computed):
        eid_at[index] = value

    seen = {}
    pending = {}
    times = {}
    for index, row in enumerate(check_rows):
        seq, prev, leid, stored_link, feid, being, t, kind, _origin, oseq, laws, digest, body, redacted_by = row
        at = seq if _is_int(seq) else expected_seq
        if not _is_int(seq) or not (seq == expected_seq or (kind == "resumed" and seq > expected_seq)):
            raise refuse(at, "seq", kept_seq, kept_t)
        if seq > expected_seq:
            gaps.append([expected_seq - 1, seq])
        if prev != expected_prev:
            raise refuse(seq, "link", kept_seq, kept_t)
        if feid is None:
            raise refuse(seq, "eid", kept_seq, kept_t, "no fact for this link")
        try:
            recomputed = link(leid, prev, seq)
        except Exception:  # noqa: BLE001 - a stored eid the codec refuses breaks the link
            recomputed = None
        if recomputed != stored_link:
            raise refuse(seq, "link", kept_seq, kept_t)
        entry = kinds.get(kind) if isinstance(kind, str) else None
        parsed = None
        if body is not None:
            parsed = _strict_object(body)
            if parsed is None or hashlib.sha256(bytes(body)).hexdigest() != digest:
                raise refuse(seq, "body", kept_seq, kept_t)
        elif entry is None or entry["redact_by"] is None or redacted_by is None:
            raise refuse(seq, "redaction", kept_seq, kept_t, "a body is missing without a redaction")
        else:
            pending[leid] = seq
        if eid_at.get(index) != leid:
            raise refuse(seq, "eid", kept_seq, kept_t)
        if being != meta["being"]:
            raise refuse(seq, "being", kept_seq, kept_t)
        # The first row defines the law version the others are checked against, so there
        # the genesis is read before the law; after it, the law comes first.
        if genesis is None:
            if kind != "genesis" or oseq != 0 or t != 0 or not _genesis_ok(parsed, table):
                raise refuse(seq, "genesis", kept_seq, kept_t, "the first event is not a genesis")
            genesis = parsed
            context["birth"] = genesis["birth"]["wall"]
            if laws != genesis["laws"]["v"]:
                raise refuse(seq, "laws", kept_seq, kept_t)
        else:
            if laws != genesis["laws"]["v"]:
                raise refuse(seq, "laws", kept_seq, kept_t)
            if kind == "genesis":
                raise refuse(seq, "genesis", kept_seq, kept_t, "a second genesis")
        if kind == "lang_forget":
            named = parsed.get("target") if isinstance(parsed, dict) else None
            info = seen.get(named) if isinstance(named, str) else None
            if info is None and isinstance(named, str) and prior is not None:
                info = target_info(conn, named)
                if info is not None and info[0] > prior.head:
                    info = None
            if (info is None or kinds.get(info[1], {}).get("redact_by") != "lang_forget" or not info[2]
                    or info[3] != leid):
                raise refuse(seq, "redaction", kept_seq, kept_t, "a forget names no forgotten fact")
            pending.pop(named, None)
        if kind == "owner" and isinstance(parsed, dict) and _is_hex(parsed.get("to"), 32):
            owner_to = parsed["to"]
        seen[leid] = (seq, kind, body is None, redacted_by)
        times[seq] = t
        kept_seq, kept_t = seq, t
        expected_seq, expected_prev = seq + 1, stored_link
    if pending:
        first = min(pending.values())
        before = first - 1
        raise refuse(first, "redaction", before, times.get(before, 0), "a redacted body no forget names")

    head = kept_seq
    head_link = expected_prev
    head_t = kept_t
    # 4. The genesis names the owner and the soil the store has.
    owner_now = owner_to if owner_to is not None else genesis["owner"]
    if owner_now != meta["owner"]:
        raise StoreRefused("owner", "the genesis names another owner")
    if genesis["soil"] != meta["soil"]:
        raise StoreRefused("soil", "the genesis names another soil")
    # 5. The store anchor.
    if (not _exactly(anchor, ANCHOR_KEYS) or not _is_int(anchor["gen"]) or not _is_int(anchor["seq"])
            or not isinstance(anchor["key"], str) or not _is_hex(anchor["mac"], 64)):
        raise refuse(head, "anchor", head, head_t, "the anchor is not whole")
    if anchor["seq"] > head:
        raise refuse(head + 1, "truncated", head, head_t,
                     f"the anchor names event #{anchor['seq']}; the record ends at #{head}")
    if anchor["seq"] < head:
        raise refuse(head, "anchor", head, head_t, "the anchor names an earlier head")
    if anchor["key"] != key_id:
        raise refuse(head, "anchor", head, head_t, "the anchor is under another key")
    if anchor["gen"] != meta["gen"]:
        raise refuse(head, "anchor", head, head_t, "the anchor names another generation")
    try:
        local = local_digest(state["local"], meta)
        expected = anchors.mac(anchor_key, anchor_input(
            being=meta["being"], gen=anchor["gen"], head=head_link, local=local, owner=meta["owner"],
            seq=anchor["seq"], soil=meta["soil"]))
    except Exception:  # noqa: BLE001 - local tables that cannot be summed do not match any anchor
        expected = None
    if expected is None or not hmac.compare_digest(expected, anchor["mac"]):
        raise refuse(head, "anchor", head, head_t, "the anchor does not verify")
    # 6. No fact outside the chain.
    if state["orphan"]:
        raise refuse(head + 1, "orphan", head, head_t, "a fact no link names")
    # 7. The local tables.
    defect = _local_defect(state, meta)
    if defect is not None:
        raise refuse(head, "local", head, head_t, defect)
    secret = dict(keys)["being_secret"]
    return Verified(
        being_tag=anchors.being_tag(secret), birth=genesis["birth"]["wall"], gaps=gaps, genesis=genesis,
        head=head, head_link=head_link, head_t=head_t, keys=keys, meta=meta, owner=owner_now, pin=pin,
        segment=meta["segment"], through=through())


def _canonical_int(text):
    if not text or not text.isdigit() or not text.isascii():
        return None
    value = int(text)
    return value if str(value) == text else None


def _local_defect(state, meta):
    """What is wrong with the local tables, or ``None``."""
    from . import wire

    names = []
    for name, value in state["local"]["keys"]:
        if not isinstance(name, str) or not isinstance(value, bytes) or len(value) != 32:
            return "a key row"
        if not (name in ("being_secret", "rhythm") or (name.startswith("payload:") and _is_hex(name[8:], 32))
                or (name.startswith("heard:") and _canonical_int(name[6:]) is not None)):
            return "a key name"
        names.append(name)
    if "being_secret" not in names:
        return "no being secret"
    references = sorted(name[8:] for name in names if name.startswith("payload:"))
    rows = state["payloads"]
    if references != [row[0] for row in rows]:
        return "payload keys and payload rows differ"
    for ref, _eid, body, seq in rows:
        if seq is None or body is None or bytes(body) != wire.emit({"payload": ref}):
            return "a payload whose fact is not a live body naming it"
    floor = meta.get("rhythm_floor", 0)
    ended = meta.get("heard_ended", [])
    if not _is_int(floor) or not isinstance(ended, list) or not all(_is_int(item) for item in ended):
        return "the rhythm floor or the ended seasons"
    count, _most, least = state["local"]["rhythm"]
    if count and "rhythm" not in names:
        return "rhythm rows without a rhythm key"
    if count and (not _is_int(least) or least < floor):
        return "a rhythm row below the floor"
    for season, _count in state["local"]["heard"]:
        if f"heard:{season}" not in names or season in ended:
            return "heard rows without a key of their season, or of an ended season"
    for name in names:
        if name.startswith("heard:") and _canonical_int(name[6:]) in ended:
            return "a key of an ended season"
    for _through, seq in state["checkpoints"]:
        if seq is None:
            return "a checkpoint through an event that is not linked"
    return None


# ---------------------------------------------------------------------------
# After the audit: cross-anchors and destruction records (steps 9 and 10)
# ---------------------------------------------------------------------------
def in_gap(seq, gaps):
    for low, high in gaps:
        if low < seq < high:
            return True
    return False


def check_cross(verified, latest, link_of):
    """Step 9: an anchor newer than the store is ``older``; one that disagrees with a link is ``fork``."""
    if latest is None:
        return
    gen, seq, head = latest
    if gen > verified.meta["gen"]:
        raise verified.refusal(verified.head, "older", "this onion is older than its own history")
    if seq <= verified.head and not in_gap(seq, verified.gaps):
        known = link_of(seq)
        if known is None or known[:16] != head:
            raise verified.refusal(verified.head, "fork", f"event #{seq} is not the one anchored")


def check_destroyed(verified, union, body_is_null):
    """Step 10: every recorded destruction, anchored or pending, is still in effect."""
    from . import anchors

    total = union
    pending = verified.meta.get("cross_pending", [])
    if not isinstance(pending, list):
        raise verified.refusal(verified.head, "destroyed", "the pending records are not a list")
    for item in pending:
        if not _pending_shaped(item):
            raise verified.refusal(verified.head, "destroyed", "a pending record is not whole")
        total = anchors.merge(total, item)
    for eid in total["forgot"]:
        if body_is_null(eid) is False:
            raise verified.refusal(verified.head, "destroyed", "a forgotten body is back")
    live = {}
    for _name, key in verified.keys:
        live[anchors.fingerprint(key)] = True
    for gone in total["destroyed"]:
        if gone in live:
            raise verified.refusal(verified.head, "destroyed", "a destroyed key is back")
    if verified.meta.get("rhythm_floor", 0) < total["rhythm_floor"]:
        raise verified.refusal(verified.head, "destroyed", "the rhythm floor is below its record")
    ended = verified.meta.get("heard_ended", [])
    for season in total["ended"]:
        if season not in ended:
            raise verified.refusal(verified.head, "destroyed", "an ended season is open again")


def _pending_shaped(item):
    """A destruction record not yet anchored, whole: fingerprints, seasons, eids and a floor."""
    if not (_exactly(item, PENDING_KEYS) and isinstance(item["destroyed"], list) and isinstance(item["ended"], list)
            and isinstance(item["forgot"], list) and _is_int(item["rhythm_floor"])):
        return False
    return (all(_is_hex(value, 16) for value in item["destroyed"]) and all(_is_int(value) for value in item["ended"])
            and all(_is_hex(value, 64) for value in item["forgot"]))


def pending_whole(item):
    """Whether a pending destruction record can be enforced as it stands."""
    return _pending_shaped(item)


def body_is_null(conn, eid):
    """``True`` when a linked fact's body is gone, ``False`` when it is there, ``None`` when not linked."""
    row = conn.execute("SELECT b.body IS NULL FROM links l LEFT JOIN bodies b ON b.eid = l.eid WHERE l.eid = ?",
                       (eid,)).fetchone()
    return None if row is None else bool(row[0])


# ---------------------------------------------------------------------------
# Redaction side effects, applied inside the forgetting transaction
# ---------------------------------------------------------------------------
def forget_target(conn, *, target, forgetter, kinds):
    """Redact a ``lang_teach`` (or other redactable) body and destroy its payload key.

    Returns the destruction: ``{"destroyed": [fingerprint], "forgot": [eid]}``.
    """
    from . import anchors, wire
    from .membrane import MembraneRefused

    row = conn.execute("SELECT f.kind, f.t, b.body FROM facts f JOIN links l ON l.eid = f.eid "
                       "LEFT JOIN bodies b ON b.eid = f.eid WHERE f.eid = ?", (target,)).fetchone()
    if row is None:
        raise MembraneRefused("target", "unknown")
    kind, t, body = row
    if kinds.get(kind, {}).get("redact_by") != "lang_forget":
        raise MembraneRefused("target", "not forgettable")
    if body is None:
        raise MembraneRefused("target", "already forgotten")
    conn.execute("UPDATE bodies SET body = NULL, redacted_by = ? WHERE eid = ?", (forgetter, target))
    destroyed = []
    ref = wire.parse(bytes(body)).get("payload")
    if isinstance(ref, str):
        name = "payload:" + ref
        key = conn.execute("SELECT key FROM keys WHERE name = ?", (name,)).fetchone()
        if key is not None:
            destroyed.append(anchors.fingerprint(key[0]))
        conn.execute("DELETE FROM keys WHERE name = ?", (name,))
        conn.execute("DELETE FROM payloads WHERE ref = ? AND eid = ?", (ref, target))
    conn.execute("DELETE FROM checkpoints WHERE t >= ?", (t,))
    return {"destroyed": destroyed, "forgot": [target]}


def next_rseq(conn, floor):
    """The next rhythm seq: never reused, not even after the rows are gone."""
    most = conn.execute("SELECT MAX(rseq) FROM rhythm_facts").fetchone()[0]
    after = 0 if most is None else most + 1
    return after if after > floor else floor


def forget_rhythm(conn, *, floor):
    """Destroy the rhythm key, delete every rhythm row and every checkpoint; raise the floor.

    Returns the destruction: ``{"destroyed": [fingerprint], "rhythm_floor": n}``.
    """
    from . import anchors

    key = conn.execute("SELECT key FROM keys WHERE name = 'rhythm'").fetchone()
    new_floor = next_rseq(conn, floor)
    conn.execute("DELETE FROM keys WHERE name = 'rhythm'")
    conn.execute("DELETE FROM rhythm_facts")
    conn.execute("DELETE FROM checkpoints")
    return {"destroyed": [] if key is None else [anchors.fingerprint(key[0])], "rhythm_floor": new_floor}


def end_season(conn, season):
    """Destroy a heard season's key and delete every row of it.

    Returns the destruction: ``{"destroyed": [fingerprint], "rows": n}``.
    """
    from . import anchors

    name = f"heard:{season}"
    key = conn.execute("SELECT key FROM keys WHERE name = ?", (name,)).fetchone()
    rows = conn.execute("SELECT COUNT(*) FROM heard WHERE season = ?", (season,)).fetchone()[0]
    conn.execute("DELETE FROM keys WHERE name = ?", (name,))
    conn.execute("DELETE FROM heard WHERE season = ?", (season,))
    return {"destroyed": [] if key is None else [anchors.fingerprint(key[0])], "rows": rows}


# ---------------------------------------------------------------------------
# Restore, never repair: what a resume sets aside, inside its own transaction
# ---------------------------------------------------------------------------
def _key_name_ok(name):
    return isinstance(name, str) and (
        name in ("being_secret", "rhythm") or (name.startswith("payload:") and _is_hex(name[8:], 32))
        or (name.startswith("heard:") and _canonical_int(name[6:]) is not None))


def _fingerprint_of(value):
    from . import anchors

    if isinstance(value, (bytes, bytearray, memoryview)):
        return anchors.fingerprint(bytes(value))
    return None


def resume_discard(conn, *, kept_seq, kinds, union, floor, ended):
    """Steps 1 to 4 of a resume: set the tail aside, forget again, enforce the records, drop the checkpoints.

    ``union`` is every destruction record, anchored or still pending. Rows
    are deleted, never rewritten -- except that a body a record says was
    forgotten is forgotten again. Nothing that failed to verify is kept:

    1. the links above ``kept_seq``, their facts, bodies, payload rows and
       payload keys, and every fact no link names;
    2. the targets to forget again, in seq order: a forgotten body whose
       forgetter was set aside, and a recorded forgotten fact whose body is
       back (its body goes, with its payload row and key);
    3. every key a record says was destroyed, with the rows it sealed; every
       key and row the structural checks would refuse (an unknown key name,
       a payload row without its key or its live body, rhythm rows without
       their key or below the floor, heard rows or keys of an ended season);
    4. every checkpoint.

    Returns ``{"removed": [eid], "reissue": [eid], "destroyed": [fingerprint],
    "rhythm_floor": n, "ended": [season]}``; the fingerprints name every key
    this pass deleted whose bytes no key it keeps holds, so a stray row
    carrying a kept key's bytes never records that key as destroyed. A
    record, anchored or pending, that names a key the pass would keep -- the
    being's secret first -- refuses the resume (``local``): the record and
    the key cannot both stand.
    """
    from . import wire
    from .store import StoreRefused

    destroyed = []

    def drop_key(name):
        row = conn.execute("SELECT key FROM keys WHERE name = ?", (name,)).fetchone()
        if row is None:
            return
        fingerprint = _fingerprint_of(row[0])
        if fingerprint is not None:
            destroyed.append(fingerprint)
        conn.execute("DELETE FROM keys WHERE name = ?", (name,))

    def drop_payload(ref):
        drop_key("payload:" + str(ref))
        conn.execute("DELETE FROM payloads WHERE ref = ?", (ref,))

    # 1. The tail, and every fact no link names.
    removed = [row[0] for row in conn.execute("SELECT eid FROM links WHERE seq > ? ORDER BY seq", (kept_seq,))]
    conn.execute("DELETE FROM links WHERE seq > ?", (kept_seq,))
    unlinked = [row[0] for row in conn.execute("SELECT eid FROM facts EXCEPT SELECT eid FROM links ORDER BY 1")]
    for eid in unlinked:
        for (ref,) in conn.execute("SELECT ref FROM payloads WHERE eid = ? ORDER BY ref", (eid,)).fetchall():
            drop_payload(ref)
        conn.execute("DELETE FROM bodies WHERE eid = ?", (eid,))
        conn.execute("DELETE FROM facts WHERE eid = ?", (eid,))
    conn.execute("DELETE FROM bodies WHERE eid NOT IN (SELECT eid FROM facts)")

    # 2. What must be forgotten again.
    rows = conn.execute("SELECT l.seq, l.eid, f.kind, b.body, b.redacted_by FROM links l JOIN facts f "
                        "ON f.eid = l.eid LEFT JOIN bodies b ON b.eid = l.eid ORDER BY l.seq").fetchall()
    linked = {}
    for row in rows:
        linked[row[1]] = True
    recorded = {}
    for eid in union["forgot"]:
        recorded[eid] = True
    reissue = []
    for _seq, eid, kind, body, redacted_by in rows:
        dangling = body is None and redacted_by not in linked
        again = body is not None and eid in recorded
        if not (dangling or again):
            continue
        entry = kinds.get(kind) if isinstance(kind, str) else None
        if not isinstance(entry, dict) or entry.get("redact_by") != "lang_forget":
            raise StoreRefused("local", "a forgotten fact is of a kind no forget redacts")
        if again:
            parsed = _strict_object(body)
            ref = parsed.get("payload") if isinstance(parsed, dict) else None
            if ref is not None:
                drop_payload(ref)
            conn.execute("UPDATE bodies SET body = NULL, redacted_by = NULL WHERE eid = ?", (eid,))
        reissue.append(eid)

    # 3. The records, enforced; and what the structural checks would refuse, discarded.
    gone = {}
    for fingerprint in union["destroyed"]:
        gone[fingerprint] = True
    new_floor = floor if floor >= union["rhythm_floor"] else union["rhythm_floor"]
    ends = {}
    for season in list(ended) + list(union["ended"]):
        ends[season] = True
    new_ended = sorted(ends)
    names = {}
    for name, value in conn.execute("SELECT name, key FROM keys ORDER BY name").fetchall():
        whole = isinstance(value, bytes) and len(value) == 32
        if name == "being_secret":
            if not whole:
                raise StoreRefused("local", "the being's secret is not whole: nothing can be resumed")
            if _fingerprint_of(value) in gone:
                raise StoreRefused("local", "a destruction record names the being's secret: nothing can be resumed")
            names[name] = True
            continue
        season = _canonical_int(name[6:]) if isinstance(name, str) and name.startswith("heard:") else None
        if _key_name_ok(name) and whole and _fingerprint_of(value) not in gone and season not in ends:
            names[name] = True
            continue
        drop_key(name)
        if name == "rhythm":
            conn.execute("DELETE FROM rhythm_facts")
        elif isinstance(name, str) and name.startswith("payload:"):
            conn.execute("DELETE FROM payloads WHERE ref = ?", (name[8:],))
        elif season is not None:
            conn.execute("DELETE FROM heard WHERE season = ?", (season,))
    if "being_secret" not in names:
        raise StoreRefused("local", "the being's secret is gone: nothing can be resumed")
    for ref, eid in conn.execute("SELECT ref, eid FROM payloads ORDER BY ref").fetchall():
        name = "payload:" + ref if _is_hex(ref, 32) else None
        body = conn.execute("SELECT b.body FROM links l JOIN bodies b ON b.eid = l.eid WHERE l.eid = ?",
                            (eid,)).fetchone()
        live = (name is not None and name in names and body is not None and body[0] is not None
                and bytes(body[0]) == wire.emit({"payload": ref}))
        if not live:
            conn.execute("DELETE FROM payloads WHERE ref = ?", (ref,))
            if name is not None and name in names:
                drop_key(name)
                del names[name]
    refs = {}
    for (ref,) in conn.execute("SELECT ref FROM payloads").fetchall():
        refs[ref] = True
    for name in sorted(names):
        if name.startswith("payload:") and name[8:] not in refs:
            drop_key(name)
            del names[name]
    if "rhythm" in names:
        conn.execute("DELETE FROM rhythm_facts WHERE rseq < ?", (new_floor,))
    else:
        conn.execute("DELETE FROM rhythm_facts")
    for (season,) in conn.execute("SELECT DISTINCT season FROM heard ORDER BY season").fetchall():
        if f"heard:{season}" not in names or season in ends:
            conn.execute("DELETE FROM heard WHERE season = ?", (season,))

    # What is kept is never recorded as destroyed, and no record may name it.
    kept = {}
    for (value,) in conn.execute("SELECT key FROM keys ORDER BY name").fetchall():
        fingerprint = _fingerprint_of(value)
        if fingerprint in gone:
            raise StoreRefused("local", "a destruction record names a key the resume keeps")
        kept[fingerprint] = True
    destroyed = [fingerprint for fingerprint in destroyed if fingerprint not in kept]

    # 4. The checkpoints.
    conn.execute("DELETE FROM checkpoints")
    return {"destroyed": destroyed, "ended": new_ended, "reissue": reissue, "removed": removed,
            "rhythm_floor": new_floor}
