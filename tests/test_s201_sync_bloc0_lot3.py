#!/usr/bin/env python3
"""S201 -- sync cycle Bloc 0 lot 3: the skills producer.

One tight test group per fix (SYN-01, ROADMAP_SYNC_CYCLE Bloc 0, the lot-1/2
precedents applied to ``agent/skills.py``):

- Clock discipline through the hook: an unseen skill key mints clock 1;
  clocks stay strictly monotonic over local writes AND past a journalled
  remote winner (PRT-02 journals adoptions, so ``current_clock`` reflects
  them).

- Identity (the mandatory lot-3 checks, documented as tests): the key is the
  ``category/name`` slug join -- the path identity. ``_safe_segment`` keeps
  only ``[a-z0-9_-]``, so the ``/`` separator can never appear inside a
  segment and distinct category/name pairs map to distinct keys; a raw input
  carrying ``/`` sanitises into a single segment and cannot forge the join.
  There is no rename path (asserted structurally); scoping rides the payload
  (``user_id``, the ``effective_user_id`` pattern), never the key.

- Drafts are device-local: creating, patching or deleting a draft journals
  nothing; only ``publish`` (the human-approved transition) journals the
  first state, and ``publish``'s internal cleanup of the promoted draft
  journals nothing either.

- Usage is device-local telemetry: ``increment_usage`` publishes nothing and
  ``uses`` / ``last_used`` never appear in a payload.

- Payload shape: ``user_id`` hoisted top-level; the nested skill carries the
  frontmatter metadata plus ``markdown`` -- the EXACT on-disk bytes, so a
  receiver rebuilds the file byte-faithfully without re-serialising
  frontmatter.

- Hook contract: the domain commit (the completed file write / unlink)
  happens first and a publish or payload failure never breaks the write;
  no-op when the veilid framework is absent (and pays nothing then -- the
  payload is provably never built); mode-free (publishing works under Bulbe,
  only the wire is Daily-gated).

Loader idiom: lot-1/2's (spec_from_file_location, sys.modules registration
BEFORE exec_module, package stubs). skills.py's guarded deps are stubbed:
signed_audit_log (no-op chain_log, so no runtime audit DB), config (tmp
DATA_DIR), user_isolation. The veilid modules are the real ones over tmp
feeds; the engine singleton is injected per test via ``set_sync_engine``.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"
AGENT = OO / "agent"

_MODE = {"fn": (lambda: "daily")}


def set_mode(value: str = "daily") -> None:
    def _gm() -> str:
        return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


_SESSION_DATA_DIR = Path(tempfile.mkdtemp(prefix="oo_s201_data_"))


def _ensure_stubs() -> None:
    for name, sub in (
        ("opti_oignon", OO),
        ("opti_oignon.veilid", VEILID),
        ("opti_oignon.agent", AGENT),
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(sub)]
            sys.modules[name] = mod
    if "opti_oignon.security_mode" not in sys.modules:
        sm = types.ModuleType("opti_oignon.security_mode")
        sm.get_current_mode = _MODE["fn"]  # type: ignore[attr-defined]
        sys.modules["opti_oignon.security_mode"] = sm
    if "opti_oignon.signed_audit_log" not in sys.modules:
        al = types.ModuleType("opti_oignon.signed_audit_log")
        al.chain_log = lambda **kwargs: None  # type: ignore[attr-defined]
        sys.modules["opti_oignon.signed_audit_log"] = al
    if "opti_oignon.config" not in sys.modules:
        cfg = types.ModuleType("opti_oignon.config")
        cfg.DATA_DIR = _SESSION_DATA_DIR  # type: ignore[attr-defined]
        sys.modules["opti_oignon.config"] = cfg
    if "opti_oignon.user_isolation" not in sys.modules:
        ui = types.ModuleType("opti_oignon.user_isolation")
        ui.DEFAULT_LOCAL_USER = "local"  # type: ignore[attr-defined]
        ui.effective_user_id = (  # type: ignore[attr-defined]
            lambda user_id=None, single_user_mode=True: "local" if user_id is None else user_id
        )
        sys.modules["opti_oignon.user_isolation"] = ui


def _load(name: str, base: Path = VEILID, package: str = "opti_oignon.veilid"):
    full = f"{package}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(base / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod  # register before exec (3.12+ dataclass processing)
    spec.loader.exec_module(mod)
    return mod


_ensure_stubs()
guard = _load("guard")
records = _load("records")
reconcile = _load("reconcile")
change_feed = _load("change_feed")
protocol = _load("protocol")
peers = _load("peers")
producers = _load("producers")
sync_engine = _load("sync_engine")
skills = _load("skills", base=AGENT, package="opti_oignon.agent")
RecordKind = records.RecordKind

_REAL_VEILID_AVAILABLE = guard.veilid_available
_REAL_SKILL_PAYLOAD = skills._skill_payload


@pytest.fixture(autouse=True)
def _daily_reset_and_available():
    set_mode("daily")
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    skills.reset_skill_registry()
    # The container has no veilid framework; the hook gates on this probe, so
    # force it on by default and let the no-op tests force it off.
    guard.veilid_available = lambda: True
    yield
    guard.veilid_available = _REAL_VEILID_AVAILABLE
    skills._skill_payload = _REAL_SKILL_PAYLOAD
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    skills.reset_skill_registry()
    set_mode("daily")


def _install_engine(tmp_path, device="dev-a"):
    f = change_feed.ChangeFeed(root=tmp_path / "feed")
    eng = sync_engine.SyncEngine(device=device, feed=f)
    sync_engine.set_sync_engine(eng)
    return f, eng


def _registry(tmp_path):
    return skills.SkillRegistry(root=tmp_path / "skills")


BODY_V1 = (
    "# When to Use\nWhen cleaning tabular data.\n\n"
    "# Procedure\n1. Inspect the columns.\n\n"
    "# Pitfalls\nNone known.\n\n"
    "# Verification\nRe-run the inspection.\n"
)
BODY_V2 = BODY_V1.replace("Inspect the columns.", "Inspect the columns twice.")


def _rows(feed, record_id=None, kind=RecordKind.SKILL):
    out = []
    for r in feed.current_records():
        if record_id is not None and r.record_id != record_id:
            continue
        if kind is not None and str(getattr(r.kind, "value", r.kind)) != str(
            getattr(kind, "value", kind)
        ):
            continue
        out.append(r)
    return out


def _latest(feed, record_id):
    matches = _rows(feed, record_id=record_id)
    if not matches:
        return None
    return max(matches, key=lambda r: r.clock)


def _remote(record_id, clock, *, deleted=False):
    return records.new_record(
        kind=RecordKind.SKILL,
        record_id=record_id,
        payload={} if deleted else {"v": clock},
        device="remote",
        clock=clock,
        deleted=deleted,
    )


# --- Clock discipline through the hook (SYN-01) ------------------------------


class TestClockDiscipline:
    def test_first_publish_mints_clock_one(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("clean-data", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        latest = _latest(f, "data/clean-data")
        assert latest is not None
        assert latest.clock == 1
        assert latest.deleted is False

    def test_clock_monotonic_over_local_writes(self, tmp_path):
        # The feed's read view collapses to latest-per-key (state-based LWW),
        # so monotonicity is asserted as the latest clock stepping by exactly
        # one after each successive write (the lot-2 precedent).
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("clean-data", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        assert _latest(f, "data/clean-data").clock == 1
        reg.update("clean-data", "data", body=BODY_V2)
        assert _latest(f, "data/clean-data").clock == 2
        reg.patch("clean-data", "data", "twice", "three times")
        assert _latest(f, "data/clean-data").clock == 3
        reg.delete("clean-data", "data")
        assert _latest(f, "data/clean-data").clock == 4

    def test_clock_continues_past_a_remote_winner(self, tmp_path):
        # apply_record_batch journals winners (PRT-02 included), so the feed
        # is the merged latest view; a journalled remote row stands in for an
        # applied winner here, and the next local mint must out-clock it.
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("clean-data", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        f.record(_remote("data/clean-data", 7))
        updated = reg.update("clean-data", "data", body=BODY_V2)
        assert updated is not None
        assert _latest(f, "data/clean-data").clock == 8


# --- Identity: the category/name slug join (the mandatory lot-3 checks) ------


class TestIdentityKey:
    def test_distinct_pairs_map_to_distinct_keys(self, tmp_path):
        # ("data", "clean-up") vs ("data-clean", "up"): a naive join without a
        # safe separator could collide these; the slug join cannot, because
        # "/" never survives sanitisation inside a segment.
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("clean-up", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        reg.add("up", "data-clean", BODY_V1, status=skills.STATUS_PUBLISHED)
        assert skills._skill_sync_key("data", "clean-up") == "data/clean-up"
        assert skills._skill_sync_key("data-clean", "up") == "data-clean/up"
        assert _latest(f, "data/clean-up").clock == 1
        assert _latest(f, "data-clean/up").clock == 1
        assert len(f.current_records()) == 2

    def test_separator_cannot_be_forged_through_raw_input(self, tmp_path):
        # A raw name carrying "/" sanitises into ONE slug segment; the key
        # tracks the storage identity (the same path the registry writes).
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        s = reg.add("clean/up", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        assert s.name == "clean-up"
        assert skills._skill_sync_key("data", "clean/up") == "data/clean-up"
        assert _latest(f, "data/clean-up").clock == 1
        assert _latest(f, "data/clean/up") is None

    def test_key_idempotent_on_sanitised_slugs(self):
        assert skills._skill_sync_key("data", "clean-up") == skills._skill_sync_key(
            "Data!", "Clean Up"
        )

    def test_no_rename_path_exists(self):
        # The lot-3 mandatory check (b), structural: the registry exposes no
        # rename; update/patch preserve name and category, so a path-derived
        # key can never orphan history. A future rename would be delete+add
        # (tombstone + new key) by construction.
        assert not hasattr(skills.SkillRegistry, "rename")
        assert "rename" not in skills._WRITE_ACTIONS

    def test_user_id_rides_the_payload_not_the_key(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("clean-data", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        latest = _latest(f, "data/clean-data")
        assert latest.payload["user_id"] == "local"
        assert "user_id" not in latest.payload["skill"]
        assert "/" not in latest.payload["user_id"]


# --- Drafts are device-local --------------------------------------------------


class TestDraftsJournalNothing:
    def test_draft_add_journals_nothing(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("clean-data", "data", BODY_V1, status=skills.STATUS_DRAFT)
        assert f.current_records() == []

    def test_publish_journals_the_first_state(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("clean-data", "data", BODY_V1, status=skills.STATUS_DRAFT)
        published = reg.publish("clean-data", "data")
        assert published is not None
        rows = _rows(f, record_id="data/clean-data")
        assert [r.clock for r in rows] == [1]
        assert rows[0].deleted is False
        assert rows[0].payload["skill"]["status"] == skills.STATUS_PUBLISHED
        # publish()'s internal cleanup of the promoted draft journalled no
        # tombstone: the single row above is the whole journal for the key.
        assert reg.exists("clean-data", "data", draft=True) is False

    def test_republish_after_new_draft_journals_new_state(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("clean-data", "data", BODY_V1, status=skills.STATUS_DRAFT)
        reg.publish("clean-data", "data")
        reg.add("clean-data", "data", BODY_V2, status=skills.STATUS_DRAFT)
        assert _latest(f, "data/clean-data").clock == 1  # the draft was silent
        republished = reg.publish("clean-data", "data")
        assert republished.version == 2
        latest = _latest(f, "data/clean-data")
        assert latest.clock == 2
        assert latest.payload["skill"]["version"] == 2

    def test_draft_delete_journals_nothing(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("clean-data", "data", BODY_V1, status=skills.STATUS_DRAFT)
        assert reg.delete("clean-data", "data", draft=True) is True
        assert f.current_records() == []


# --- Tombstone on a published delete ------------------------------------------


class TestTombstone:
    def test_delete_published_publishes_a_tombstone(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("clean-data", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        assert reg.delete("clean-data", "data") is True
        latest = _latest(f, "data/clean-data")
        assert latest.clock == 2
        assert latest.deleted is True
        assert latest.payload == {}

    def test_delete_missing_publishes_nothing(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        assert reg.delete("ghost", "data") is False
        assert f.current_records() == []


# --- Usage is device-local telemetry ------------------------------------------


class TestUsageSilence:
    def test_increment_usage_publishes_nothing(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("clean-data", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        before = _latest(f, "data/clean-data").clock
        usage = reg.increment_usage("clean-data", "data")
        assert usage.uses == 1  # the counter moved locally
        assert _latest(f, "data/clean-data").clock == before  # the journal did not
        assert len(_rows(f, record_id="data/clean-data")) == 1

    def test_usage_numbers_absent_from_payloads(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("clean-data", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        reg.increment_usage("clean-data", "data")
        reg.update("clean-data", "data", body=BODY_V2)
        nested = _latest(f, "data/clean-data").payload["skill"]
        assert "uses" not in nested
        assert "last_used" not in nested

    def test_reads_publish_nothing(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("clean-data", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        reg.get("clean-data", "data")
        reg.list()
        reg.search("clean")
        reg.view("clean-data", "data")
        reg.index()
        assert len(_rows(f, record_id="data/clean-data")) == 1


# --- Payload shape: byte-faithful markdown -------------------------------------


class TestPayloadShape:
    def test_exact_field_set(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add(
            "clean-data", "data", BODY_V1,
            source=skills.SOURCE_MANUAL, status=skills.STATUS_PUBLISHED,
        )
        payload = _latest(f, "data/clean-data").payload
        assert set(payload.keys()) == {"user_id", "skill"}
        assert set(payload["skill"].keys()) == {
            "name", "category", "status", "version", "source",
            "created_at", "updated_at", "markdown",
        }
        assert "summary" not in payload["skill"]  # the derived to_dict field

    def test_markdown_round_trips_byte_faithfully(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        s = reg.add("clean-data", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        on_disk = (
            tmp_path / "skills" / "data" / "clean-data" / skills.SKILL_FILENAME
        ).read_text(encoding="utf-8")
        wire = _latest(f, "data/clean-data").payload["skill"]["markdown"]
        assert wire == on_disk
        assert wire == s.to_markdown()
        # A receiver writing the field verbatim rebuilds the identical file:
        # the round-trip re-parses to the same skill.
        meta, body = skills._parse_frontmatter(wire)
        assert meta["name"] == "clean-data"
        assert meta["category"] == "data"
        assert body == s.body.strip()


# --- Hook contract --------------------------------------------------------------


class TestHookContract:
    def test_publish_failure_never_breaks_the_write(self, tmp_path):
        f, eng = _install_engine(tmp_path)

        def _boom(*a, **kw):
            raise RuntimeError("journal append failed (test)")

        eng.publish_skill = _boom  # type: ignore[assignment]
        reg = _registry(tmp_path)
        s = reg.add("resilient", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        assert reg.get("resilient", "data") is not None  # the file write held
        assert f.current_records() == []
        # at-least-once: the next write (engine restored) re-journals state.
        eng2 = sync_engine.SyncEngine(device="dev-a", feed=f)
        sync_engine.set_sync_engine(eng2)
        reg.update("resilient", "data", body=BODY_V2)
        assert _latest(f, "data/resilient").clock == 1
        assert _latest(f, "data/resilient").payload["skill"]["version"] == s.version + 1

    def test_payload_failure_never_breaks_the_write(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        reg = _registry(tmp_path)

        def _boom(skill):
            raise RuntimeError("payload build failed (test)")

        skills._skill_payload = _boom  # type: ignore[assignment]
        s = reg.add("fragile", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        assert s.version == 1
        assert reg.get("fragile", "data") is not None  # the file write held
        assert f.current_records() == []

    def test_delete_failure_isolation(self, tmp_path):
        f, eng = _install_engine(tmp_path)
        reg = _registry(tmp_path)
        reg.add("doomed", "data", BODY_V1, status=skills.STATUS_PUBLISHED)

        def _boom(*a, **kw):
            raise RuntimeError("journal append failed (test)")

        eng.publish_skill = _boom  # type: ignore[assignment]
        assert reg.delete("doomed", "data") is True  # the unlink held
        assert reg.get("doomed", "data") is None
        assert [r.clock for r in _rows(f, record_id="data/doomed")] == [1]

    def test_noop_when_veilid_unavailable_pays_nothing(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        guard.veilid_available = lambda: False
        reg = _registry(tmp_path)

        def _boom(skill):
            raise AssertionError("payload built while sync was absent")

        skills._skill_payload = _boom  # type: ignore[assignment]
        s = reg.add("offline", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        assert s.version == 1
        reg.update("offline", "data", body=BODY_V2)
        assert reg.delete("offline", "data") is True
        assert f.current_records() == []

    def test_publish_is_mode_free_and_works_in_bulbe(self, tmp_path):
        # Producing + journalling are local-disk operations, permitted in any
        # mode (the producers.py posture); only the wire is Daily-gated.
        f, _ = _install_engine(tmp_path)
        set_mode("bulbe")
        reg = _registry(tmp_path)
        reg.add("bulbe-local", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        latest = _latest(f, "data/bulbe-local")
        assert latest is not None
        assert latest.clock == 1

    def test_uninstalled_engine_write_holds_and_injected_feed_untouched(self, tmp_path):
        # No engine was installed: get_sync_engine auto-constructs the
        # guarded singleton over the default feed root (the stubbed tmp
        # DATA_DIR). The properties that matter at the seam: the domain
        # write holds whatever the singleton path does, and a feed that was
        # never installed sees nothing.
        f = change_feed.ChangeFeed(root=tmp_path / "feed")
        sync_engine.reset_sync_engine()
        reg = _registry(tmp_path)
        s = reg.add("loner", "data", BODY_V1, status=skills.STATUS_PUBLISHED)
        assert s.version == 1
        assert reg.get("loner", "data") is not None
        assert f.current_records() == []
