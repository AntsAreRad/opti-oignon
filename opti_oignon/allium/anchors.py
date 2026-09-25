"""The being's anchors outside its store: tags, fingerprints, and entries in the signed audit log.

Three kinds of entry, one event type each (only the event type is filtered
by the audit log itself; everything else is read here):

* ``allium_sow`` -- written once at every birth, before the store is linked
  into place: ``{"being": being_tag, "key": key_id, "mac": E, "owner":
  owner_tag}``. There is never a being without one.
* ``allium_wipe`` -- the same four fields, written by the vault when a store
  is composted or moved; it closes the birth it names.
* ``allium_anchor`` -- after the first write of each day of life, after
  every destruction and after every resume: the head, the write generation
  and the destruction record, sealed under a key derived from the anchor key
  on a keyed install (``{"key", "sealed", "tag"}``), in clear only in a
  keyless glass jar (``{"key": "nokey", "record", "tag"}``).

``E`` is an HMAC under the anchor key (a plain SHA-256, advisory only, in a
glass jar). The owner appears in clear as ``owner_tag`` -- an unkeyed hash of
the account id -- so a birth under a foreign key blocks only its own owner.
``being_tag`` is keyed under the being's own secret, so nothing links it to a
being without the store file.

What these anchors cannot see is said plainly: a consistent rollback of the
store together with the audit log, and an owner who edits both.
"""

import hashlib
import hmac

checkpoint_before_apply = True

OWNER_LABEL = b"oo-allium-owner-v1:"
TAG_LABEL = b"oo-allium-tag-v1"
FP_LABEL = b"oo-allium-fp-v1:"
SEAL_LABEL = b"oo-allium-anchor-seal-v1"
NOKEY = "nokey"
PAGE = 256
WHYS = ("day", "destroy", "resume")
RECORD_KEYS = ("destroyed", "ended", "forgot", "gen", "head", "rhythm_floor", "seq", "why")
_HEX = "0123456789abcdef"


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex(value, length):
    if not isinstance(value, str) or len(value) != length:
        return False
    for char in value:
        if char not in _HEX:
            return False
    return True


def _refused(code, detail):
    from .store import StoreRefused

    return StoreRefused(code, detail)


def owner_tag(user):
    """The tag of an account: the first 32 hex characters of an unkeyed SHA-256 of its id."""
    return hashlib.sha256(OWNER_LABEL + user.encode("utf-8")).hexdigest()[:32]


def being_tag(being_secret):
    """The being's tag, keyed under its own secret: nothing links it to a being without the store."""
    tag_key = hmac.new(bytes(being_secret), TAG_LABEL, hashlib.sha256).digest()
    return hmac.new(tag_key, b"being", hashlib.sha256).hexdigest()[:32]


def fingerprint(key):
    """What a destruction record names a destroyed key by: reveals nothing of a random key."""
    return hashlib.sha256(FP_LABEL + bytes(key)).hexdigest()[:16]


def seal_key(anchor_key):
    """The key the anchor records are sealed under."""
    return hmac.new(bytes(anchor_key), SEAL_LABEL, hashlib.sha256).digest()


def mac(anchor_key, data):
    """HMAC-SHA256 under the anchor key; without one (a glass jar), a plain SHA-256, advisory only."""
    if anchor_key is None:
        return hashlib.sha256(data).hexdigest()
    return hmac.new(bytes(anchor_key), data, hashlib.sha256).hexdigest()


def sow_mac(anchor_key, being, key, owner):
    from . import wire

    return mac(anchor_key, wire.emit({"being": being, "key": key, "owner": owner}))


def sow_details(anchor_key, being, key, owner):
    """The details of an ``allium_sow`` (and ``allium_wipe``) entry."""
    return {"being": being, "key": key, "mac": sow_mac(anchor_key, being, key, owner), "owner": owner}


def write_sow(audit, *, being, key_id, anchor_key, owner):
    """Write the seed anchor; ``True`` only when the audit log accepted the entry."""
    written = audit.append_event("allium_sow", source="allium", action="sow", severity="INFO",
                                 details=sow_details(anchor_key, being, key_id, owner))
    return written is not None


def record(*, destroyed, ended, forgot, gen, head, rhythm_floor, seq, why):
    """A cross-anchor's record: the head, the generation and what was destroyed."""
    return {"destroyed": list(destroyed), "ended": list(ended), "forgot": list(forgot), "gen": gen,
            "head": head[:16], "rhythm_floor": rhythm_floor, "seq": seq, "why": why}


def write_anchor(audit, *, tag, key_id, seal, seal_fn, entry):
    """Write a cross-anchor, sealed on a keyed install; ``True`` only when the audit log accepted it."""
    from . import wire

    if key_id == NOKEY:
        details = {"key": NOKEY, "record": entry, "tag": tag}
    else:
        details = {"key": key_id, "sealed": bytes(seal_fn(seal, wire.emit(entry))).hex(), "tag": tag}
    written = audit.append_event("allium_anchor", source="allium", action="anchor", severity="INFO",
                                 details=details)
    return written is not None


def events(audit, event_type):
    """Every entry of one event type, newest first, a page at a time until the log is exhausted."""
    offset = 0
    while True:
        page = audit.get_events(limit=PAGE, offset=offset, event_type=event_type)
        if not isinstance(page, list):
            raise _refused("audit", "the audit log answered no list")
        yield from page
        if len(page) < PAGE:
            return
        offset += PAGE


def _entry_id(entry):
    value = entry.get("id") if isinstance(entry, dict) else None
    if not _is_int(value):
        raise _refused("audit", "an entry has no id")
    return value


def _details(entry):
    details = entry.get("details") if isinstance(entry, dict) else None
    return details if isinstance(details, dict) else {}


def _sow_shaped(details):
    return (_is_hex(details.get("being"), 32) and isinstance(details.get("key"), str)
            and _is_hex(details.get("mac"), 64) and _is_hex(details.get("owner"), 32))


def _verifies(details, owner, key_id, anchor_key):
    """An entry under the current key verifies its HMAC; a ``nokey`` entry its advisory digest."""
    key = details["key"]
    if key == NOKEY:
        expected = sow_mac(None, details["being"], key, owner)
    elif key == key_id and anchor_key is not None:
        expected = sow_mac(anchor_key, details["being"], key, owner)
    else:
        return False
    return hmac.compare_digest(expected, details["mac"])


def sow_state(audit, owner, key_id, anchor_key):
    """``("open", being_tag)``, ``("none",)`` or ``("foreign",)`` for one owner, from the audit alone.

    It is ``sow_birth`` without the key; see there.
    """
    return sow_birth(audit, owner, key_id, anchor_key)[0]


def sow_birth(audit, owner, key_id, anchor_key):
    """``(state, key)`` for one owner, from the audit alone: ``sow_state``'s answer and the key of its birth.

    ``state`` is ``("open", being_tag)``, ``("none",)`` or ``("foreign",)``;
    ``key`` is the key the deciding birth was sown under (``"nokey"`` for a
    glass jar), ``None`` when there is none.

    The latest birth for this owner decides. A later wipe of the same owner
    and being, under the same key, closes it. Under a key that is neither
    the current one nor ``nokey``, an unclosed birth is foreign: nothing
    under that key can be checked here, and only that owner is blocked.
    Otherwise the birth must verify, and so must the wipe that closes it;
    one that does not is refused ``audit``.
    """
    latest = None
    for entry in events(audit, "allium_sow"):
        details = _details(entry)
        if details.get("owner") != owner:
            continue
        if not _sow_shaped(details):
            raise _refused("audit", "an entry for this account is malformed")
        if latest is None or _entry_id(entry) > _entry_id(latest):
            latest = entry
    if latest is None:
        return ("none",), None
    sown = _details(latest)
    after = _entry_id(latest)
    checkable = sown["key"] == key_id or sown["key"] == NOKEY
    wiped = False
    for entry in events(audit, "allium_wipe"):
        details = _details(entry)
        if (details.get("owner") != owner or details.get("being") != sown["being"]
                or details.get("key") != sown["key"] or _entry_id(entry) < after):
            continue
        if not _sow_shaped(details) or (checkable and not _verifies(details, owner, key_id, anchor_key)):
            raise _refused("audit", "a wipe for this account does not verify")
        wiped = True
    if not checkable:
        return (("none",), None) if wiped else (("foreign",), sown["key"])
    if not _verifies(sown, owner, key_id, anchor_key):
        raise _refused("audit", "an entry for this account does not verify")
    return (("none",), None) if wiped else (("open", sown["being"]), sown["key"])


def _record_shaped(value):
    if not isinstance(value, dict) or sorted(value) != list(RECORD_KEYS):
        return False
    if value["why"] not in WHYS or not _is_hex(value["head"], 16):
        return False
    if not (_is_int(value["gen"]) and _is_int(value["seq"]) and _is_int(value["rhythm_floor"])):
        return False
    lists = (value["destroyed"], value["ended"], value["forgot"])
    if not all(isinstance(items, list) for items in lists):
        return False
    return (all(_is_hex(item, 16) for item in value["destroyed"]) and all(_is_int(item) for item in value["ended"])
            and all(_is_hex(item, 64) for item in value["forgot"]))


def empty_union():
    return {"destroyed": [], "ended": [], "forgot": [], "rhythm_floor": 0}


def merge(union, entry):
    """The union of destruction records: lists joined without repeats, the rhythm floor at its highest."""
    out = {}
    for name in ("destroyed", "ended", "forgot"):
        seen = {}
        for item in list(union[name]) + list(entry[name]):
            seen[item] = True
        out[name] = sorted(seen)
    out["rhythm_floor"] = max(union["rhythm_floor"], entry["rhythm_floor"])
    return out


def anchors(audit, tag, required_key, seal, open_fn, known=None):
    """The cross-anchors of one being: ``{"latest", "latest_id", "seen", "union"}``.

    ``latest`` is ``(gen, seq, head)`` of the newest entry for the tag (its
    id ``latest_id``), ``union`` the union of every record, and ``seen`` the
    highest id of any anchor entry read. Given ``known`` -- an earlier
    answer for the same tag -- only the entries newer than its ``seen`` are
    read, newest first, and merged into it: ids only grow, so the read stops
    at the first entry already seen. An audit log whose newest anchor is
    older than one already read has gone backwards and is refused ``audit``.

    Every entry for the tag must be under the key the being's soil requires;
    another key is refused ``foreign``. A sealed record that does not open,
    or a record that is not whole, is refused ``audit``.
    """
    from . import wire

    after = known.get("seen") if isinstance(known, dict) else None
    if after is None:
        latest, latest_id, union = None, None, empty_union()
    else:
        latest, latest_id, union = known["latest"], known["latest_id"], known["union"]
    seen = after
    newest = None
    for entry in events(audit, "allium_anchor"):
        entry_id = _entry_id(entry)
        if newest is None:
            newest = entry_id
        if after is not None and entry_id <= after:
            break
        if seen is None or entry_id > seen:
            seen = entry_id
        details = _details(entry)
        if details.get("tag") != tag:
            continue
        key = details.get("key")
        if key != required_key:
            raise _refused("foreign", "an anchor of this being is under another key")
        if key == NOKEY:
            value = details.get("record")
        else:
            try:
                value = wire.parse(bytes(open_fn(seal, bytes.fromhex(details.get("sealed")))))
            except Exception:  # noqa: BLE001 - a record that does not open under the key is not trusted
                raise _refused("audit", "a sealed anchor does not open under this key") from None
        if not _record_shaped(value):
            raise _refused("audit", "an anchor record is not whole")
        if latest_id is None or entry_id > latest_id:
            latest_id = entry_id
            latest = (value["gen"], value["seq"], value["head"])
        union = merge(union, value)
    if after is not None and (newest is None or newest < after):
        raise _refused("audit", "the audit log holds fewer anchors than were already read")
    return {"latest": latest, "latest_id": latest_id, "seen": seen, "union": union}
