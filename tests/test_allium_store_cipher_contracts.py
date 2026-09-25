#!/usr/bin/env python3
"""Contracts for the componion's store on real SQLCipher: the header, the pages and a splice.

  * AS11 -- on real SQLCipher, an unkeyed connection on a new file is
    refused and a raw key is accepted, its header no longer the database
    magic; a flipped byte in a table page makes the store unreadable, never
    ready and never an empty being, and the first codec error closes the
    connection; a flipped index page, which no read of the store touches, is
    caught by the page check alone; and old root pages written back under
    the same key, which SQLCipher accepts, are refused by the store's anchor.

Local-only, and skipped where the SQLCipher binding is not installed. The
platform loads through the shared isolation window with the platform's
configuration, keys, mode, audit log and user modules proven unreachable;
the key is a raw key made up for the contract.
"""

import hashlib
import sys
import threading
from pathlib import Path

import pytest

sqlcipher3 = pytest.importorskip("sqlcipher3")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_store_support as support  # noqa: E402

BUDGET_S = {
    "test_as11_on_real_sqlcipher_the_header_the_pages_and_a_splice_are_refused": 2.0,
}


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


@pytest.fixture
def p():
    platform, restore = support.open_platform()
    try:
        yield platform
    finally:
        restore()


def _status(p, given, user="local"):
    target = support.store(p, given)
    try:
        return target.status(user)
    finally:
        target.close()


def test_as11_on_real_sqlcipher_the_header_the_pages_and_a_splice_are_refused(p, tmp_path):
    key_hex = hashlib.sha256(b"a raw key for the contract").hexdigest()
    tag = p.anchors.owner_tag("local")

    unkeyed = support.seams(p, tmp_path.joinpath("unkeyed"), suite="as11", probe=None,
                            connect=support.keyed_connect(key_hex, keyed=False))
    target = support.store(p, unkeyed)
    with pytest.raises(p.store.PlaintextRefused) as info:
        support.sow(p, target)
    assert "not keyed" in str(info.value), str(info.value)
    assert sorted(support.directory(unkeyed).glob(tag + "*")) == []
    target.close()

    given = support.seams(p, tmp_path.joinpath("keyed"), suite="as11", index=1, probe=None,
                          connect=support.keyed_connect(key_hex))
    sower = support.store(p, given)
    being = support.sow(p, sower)
    path = support.store_path(p, given)
    assert path.read_bytes()[:16] != support.MAGIC, "a raw key writes no database magic"
    support.build(being, 10, support.cli(p))
    sower.close()
    good = path.read_bytes()
    keyed = support.keyed_connect(key_hex)
    facts_root, size = support.root_page(path, "facts", keyed)
    index_root, _ = support.root_page(path, "facts_budget", keyed)
    assert facts_root == 3 and index_root > facts_root

    def flip(page):
        data = bytearray(good)
        data[(page - 1) * size + size // 2] ^= 0xFF
        path.write_bytes(bytes(data))

    # A table page: a fresh process refuses it at the page check.
    flip(facts_root)
    status = _status(p, given)
    assert (status.status, status.reason) == ("unreadable", "pages"), status
    fresh = support.store(p, given)
    with pytest.raises(p.store.StoreRefused) as info:
        fresh.open("local")
    assert info.value.code == "pages"
    fresh.close()
    # The process that already checked the pages meets the codec error, and drops that connection.
    status = sower.status("local")
    assert (status.status, status.reason) == ("unreadable", "pages"), status
    with pytest.raises(sqlcipher3.ProgrammingError):
        given["connect"].connections[-1].execute("SELECT 1")
    sower.close()

    # An index page no read touches: only the page check sees it.
    path.write_bytes(good)
    checked = support.store(p, given)
    assert checked.status("local").status == "alive"
    checked.close()
    flip(index_root)
    assert checked.status("local").status == "alive", "witness: the store's reads never touch that page"
    checked.close()
    status = _status(p, given)
    assert (status.status, status.reason) == ("unreadable", "pages"), status

    # Old root pages written back under the same key: SQLCipher accepts them, the anchor does not.
    path.write_bytes(good)
    target = support.store(p, given)
    taught = target.open("local").append("lang_teach", {}, transport=support.cli(p), payload="chive")
    target.close()
    before_forget = path.read_bytes()
    target = support.store(p, given)
    target.open("local").append("lang_forget", {"target": taught.eid}, transport=support.cli(p))
    target.close()
    for table in ("keys", "payloads"):
        support.splice(path, before_forget, table, keyed)
    conn = keyed(str(path))
    try:
        names = [row[0] for row in conn.execute("SELECT name FROM keys ORDER BY name")]
        assert any(name.startswith("payload:") for name in names), "the spliced key is back"
        assert conn.execute("PRAGMA cipher_integrity_check").fetchall() == [], "the pages themselves check"
    finally:
        conn.close()
    status = _status(p, given)
    assert (status.status, status.reason) == ("unreadable", "anchor"), status
    assert status.offer is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
