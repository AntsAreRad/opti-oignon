#!/usr/bin/env python3
"""The drift ledger's table: facts that are superseded, never edited, with
their contradiction census taken at every write.

The harness gave the ledger its semantics in memory; this is the same
record on disk. A fact is a row that no SQL in this module ever updates
except to link it to its successor. Every write runs the deterministic
contradiction templates against the active facts and records what they
find beside the fact, so drift is a count the store carries rather than an
impression: contradictions per thousand turns, unknown when there were no
turns. The LLM judge stays on the host; a statement the templates cannot
read contradicts nothing here, and that absence is reported as an empty
list, never as agreement.

Connections go through the repository's ``safe_connect``, so the table is
encrypted where the rest of the memory is. A store built where that seam
is unreachable refuses and creates no file. Where the seam is reachable
but opens plaintext -- no SQLCipher or no key, outside Bulbe mode -- this
store follows the seam and writes plaintext with the seam's warning,
unless built with ``require_encryption=True``, which refuses by name; the
onion store, which holds conversation text, requires it by default. The
migration from the canonical facts table is a mapping here; running it
against a real data directory is a host step.
"""

import json
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

checkpoint_before_apply = True

TABLE = "memory_ledger"
CENSUS = "memory_ledger_contradictions"

_KIND_BY_CATEGORY = {
    "fact": "fact", "identity": "fact", "contact": "fact",
    "preference": "preference", "project": "project", "correction": "correction",
}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LedgerStore:
    """Supersession-only facts with a contradiction census, on one SQLite file."""

    def __init__(self, path, *, connect=None, require_encryption=False):
        self._path = Path(path)
        if connect is None:
            from ..db_utils import safe_connect

            connect = safe_connect
        self._connect = connect
        self._require_encryption = bool(require_encryption)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self):
        conn = self._connect(self._path)
        if self._require_encryption:
            try:
                encrypted = bool(conn.execute("PRAGMA cipher_version").fetchall())
            except Exception:  # noqa: BLE001 - a client without the pragma is a plain one
                encrypted = False
            if not encrypted:
                conn.close()
                raise RuntimeError(
                    f"the drift ledger at {self._path} would be written in plaintext; refused by request"
                )
        return closing(conn)

    def _init_db(self):
        with self._lock, self._conn() as conn:
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    id TEXT PRIMARY KEY,
                    statement TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    valid_from TEXT NOT NULL DEFAULT '',
                    valid_until TEXT NOT NULL DEFAULT '',
                    superseded_by TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS {CENSUS} (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_id TEXT NOT NULL,
                    new_fact_id TEXT NOT NULL,
                    template TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                """
            )
            conn.commit()

    @staticmethod
    def _to_fact(row):
        from .drift import LedgerFact

        return LedgerFact(
            id=row[0], statement=row[1], kind=row[2], provenance=json.loads(row[3]),
            confidence=row[4], valid_from=row[5], valid_until=row[6], superseded_by=row[7],
        )

    def _rows(self, where="", params=()):
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                f"SELECT id, statement, kind, provenance, confidence, valid_from, valid_until, "
                f"superseded_by FROM {TABLE} {where} ORDER BY rowid",
                params,
            )
            return [self._to_fact(r) for r in cur.fetchall()]

    def add(self, fact):
        """Hold ``fact`` and record every held fact it contradicts. Returns (id, contradictions)."""
        from .drift import find_contradictions, validate_fact

        errors = validate_fact(fact)
        if errors:
            raise ValueError("; ".join(errors))
        if self._rows("WHERE id = ?", (fact.id,)):
            raise ValueError(f"fact {fact.id!r} is already held; supersede it, never overwrite it")
        found = find_contradictions(self.active(), fact.statement)
        stamp = _now()
        with self._lock, self._conn() as conn:
            conn.execute(
                f"INSERT INTO {TABLE} (id, statement, kind, provenance, confidence, valid_from, "
                f"valid_until, superseded_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fact.id, fact.statement, fact.kind, json.dumps(list(fact.provenance)),
                    float(fact.confidence), fact.valid_from or "", fact.valid_until or "",
                    fact.superseded_by, stamp,
                ),
            )
            for c in found:
                conn.execute(
                    f"INSERT INTO {CENSUS} (fact_id, new_fact_id, template, detail, statement, recorded_at) "
                    f"VALUES (?, ?, ?, ?, ?, ?)",
                    (c.fact_id, fact.id, c.template, c.detail, fact.statement, stamp),
                )
            conn.commit()
        return fact.id, found

    def supersede(self, old_id, new_fact):
        """Hold ``new_fact`` and link the old row to it. The old statement is untouched."""
        old = self.get(old_id)
        if old.superseded_by:
            raise ValueError(f"fact {old_id!r} is already superseded by {old.superseded_by!r}")
        new_id, _found = self.add(new_fact)
        with self._lock, self._conn() as conn:
            conn.execute(f"UPDATE {TABLE} SET superseded_by = ? WHERE id = ?", (new_id, old_id))
            conn.commit()
        return new_id

    def get(self, fact_id):
        rows = self._rows("WHERE id = ?", (fact_id,))
        if not rows:
            raise KeyError(fact_id)
        return rows[0]

    def head(self, fact_id):
        fact = self.get(fact_id)
        seen = {fact_id}
        while fact.superseded_by:
            if fact.superseded_by in seen:
                raise ValueError(f"supersession cycle at {fact.superseded_by!r}")
            seen.add(fact.superseded_by)
            fact = self.get(fact.superseded_by)
        return fact

    def active(self):
        return self._rows("WHERE superseded_by IS NULL")

    def all(self):
        return self._rows()

    def contradictions(self):
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                f"SELECT fact_id, new_fact_id, template, detail, statement, recorded_at FROM {CENSUS} ORDER BY seq"
            )
            keys = ("fact_id", "new_fact_id", "template", "detail", "statement", "recorded_at")
            return [dict(zip(keys, r)) for r in cur.fetchall()]

    def drift_rate(self, *, turns):
        """Contradictions per thousand turns; None when there were no turns to rate."""
        turns = int(turns)
        if turns <= 0:
            return None
        return round(len(self.contradictions()) * 1000 / turns, 4)


def migrate_facts(rows):
    """Canonical ``memory_facts`` rows as ledger facts. Every row maps; none is dropped.

    The old table has no supersession, so an inactive row is closed by its
    last update date rather than linked to a successor it never had. An
    unknown category lands on ``fact``: the migration records, it does not
    judge.
    """
    from .drift import LedgerFact

    facts = []
    for row in rows:
        get = row.get if isinstance(row, dict) else lambda k, _r=row: getattr(_r, k, None)
        source = str(get("source") or "").strip() or "unknown"
        active = get("active")
        active = True if active is None else bool(active)
        facts.append(LedgerFact(
            id=str(get("id")),
            statement=str(get("text") or ""),
            kind=_KIND_BY_CATEGORY.get(str(get("category") or "").lower(), "fact"),
            provenance=[f"memory_facts:{source}"],
            confidence=1.0,
            valid_from=str(get("created_at") or ""),
            valid_until="" if active else str(get("updated_at") or get("created_at") or ""),
        ))
    return facts
