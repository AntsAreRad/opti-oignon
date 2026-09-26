"""The garden as the terminal and the API use it: every look and every act on the componion goes through here.

``Garden`` is the one facade over the componion's store. Every public
method is one action: the switch is read first (anything but ``"on"`` is
the ``disabled`` form, and no store is built), then the emergency stop
(``stopped``, again with no store), then ``Store.action()`` pins the mode,
the single-user reading and the wall clock for the length of the action, so
every store call of the action shares them and the next action reads them
again. An attended verb then asks the terminal test, once; the look comes
next, and the verb last.

The served state machine. A look derives exactly one status, the first
rule that matches (``STATUSES``, in order): the switch, the stop, a store
that cannot be opened (or a terminal with no account, or a view refused
after the store said alive), a seed with no soil, a store ready to sow, a
glass jar sealed outside Daily, a store missing after its birth, a
prototype whose law was retired, a record that does not verify, a rest (no
state of this version holds one), and the being alive. Its labels open
every form about a being, in a fixed order: a prototype (or retired), a
glass jar, a view catching up, a view frozen on an engine fault, and the
mode when it is not Daily. The mode changes the served form, never the
life: the same record and the same minute give the same view under Daily,
Bulbe and a mode that cannot be read.

Looks write nothing in the being's store: ``look``, ``verify`` and ``laws``
open, verify and read it only. A write the terminal makes (a name, a
gesture, a law update, a pin, a resume) is followed, in the same action, by
a settle of the being's caches; a settle that fails is logged and changes
nothing of what the write said, and a refusal raised once the write has
returned is marked ``written``: it says the write was done. A view the
engine stops on with a
fault is served from the state it starts from, labelled ``frozen``, and
nothing records the fault. Deep verification (``verify``) replays every
kept state and the state served now on the reference engine.

A sowing names only the newest law of ``CHANNEL`` this engine carries and
has not retired (``sowing_law``, never the test law), writes that the
rhythm is not consented (``RHYTHM_CONSENT``), and is shown first as a card
(``sow_card``) that says everything before its two questions. A name is a
permanent fact, in the closed alphabet of ``check_name``.

The records a look and each act return are closed ``NamedTuple``s; the
terminal renders them through ``describe`` and prints nothing else. A
refusal is named: ``ServiceRefused`` with a code of its own, or the store's,
the membrane's, the laws writer's, the engine's or a deep verification's,
passed through with the look it was refused on. ``transport`` is the one
place a ``membrane.Transport`` is built; ``terminal_attended`` is the strong
terminal test, and ``platform_single_user`` the single-user reader the
production store is built with (``Garden.production``): it writes nothing
in the auth store it reads.

The API's garden (``Garden.production(terminal=False, ...)``) serves the
same looks to the web and nothing else it can reach: no attended verb, a
caller given to every method (a missing one is a ``TypeError``, a defect of
its caller, never a status, since a default caller would read as the
terminal), the emergency stop the API process holds, the single-user rule
that process already runs, and a cap on each view's work,
``api_view_cap``: the ``api`` settings' cap for the reference engine, or
for the native core when it answers, chosen at each look after the store
said alive. A view past the cap is served ``catching_up`` from its last
kept state. The terminal's garden caps nothing and has a default caller.

``gated`` is the Bulbe gate of a capability: it calls the capability only
when the switch is on, nothing is stopped and the policy of the pinned mode
allows it (``habitat``). No capability exists yet to call through it.

Nothing is imported at module level but the standard library: the store,
the membrane, the life and the habitat are imported when an action needs
them.
"""

import contextlib
import logging
import os
import threading
from pathlib import Path
from typing import NamedTuple

checkpoint_before_apply = True

logger = logging.getLogger(__name__)

DAY = 1440
# Every status a look serves, in the order they are derived: the first rule that matches wins.
STATUSES = ("disabled", "stopped", "unavailable", "awaiting_soil", "ready", "sealed_bulbe", "missing",
            "retired_prototype", "unreadable", "resting", "alive")
# Every sowing writes that the rhythm is not consented: a consent frozen in the genesis could not be withdrawn.
RHYTHM_CONSENT = False
# The laws a production sowing may name, oldest first; the test law is never among them.
CHANNEL = ("v0_1",)
_SWITCHES = ("on", "off", "unreadable")
# The statuses whose record exists but is not opened: the prototype label is said when every law of the
# channel is provisional.
_UNOPENED = ("sealed_bulbe", "missing", "unreadable")
# What a verb that restores (resume, finish) is refused with, by the status of the look before it.
_RESTORE_FORM = ("unavailable", "sealed_bulbe")
# A person's name: its characters, its length, and the word a name may not start with.
NAME_MAX = 32
_NAME_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -'.")
_NAME_FIRST = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
_SPEAKER = "beetle"
# The kinds a law history shows.
_LAW_KINDS = ("evolve", "laws_pin", "laws_unpin")
# The single-user reader's one-way latch: once a second account was seen, never single-user again.
_MULTI_USER = {"latched": False}
# The auth settings the reader falls back to when the file cannot be read, as the auth module does.
_AUTH_DEFAULTS = {"single_user_mode": True, "db_path": "data/auth.db"}


class ServiceRefused(ValueError):
    """An action the garden refuses by name; ``look`` is the look it was refused on, ``detail`` a log string."""

    CODES = ("off", "stopped", "status", "exists", "channel", "name", "offset", "attended", "consent", "budget")

    def __init__(self, code, look=None, detail=""):
        if code not in self.CODES:
            raise ValueError(f"unknown service refusal: {code}")
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.look = look
        self.detail = detail


class Look(NamedTuple):
    """What one look served: the status and its labels, the mode and habitat, and what was read to say it."""

    status: str
    labels: tuple
    mode: object
    habitat: object
    view: object
    felt: object
    being: object
    reason: object
    offer: object
    seq: object
    exit: int
    glass_allowed: bool


class BeingInfo(NamedTuple):
    """Who the being is: its name, soil, genesis law and digest, tag, count of events and weather."""

    name: object
    soil: str
    law: str
    v: int
    provisional: bool
    digest: str
    tag: str
    events: int
    weather: str


class Card(NamedTuple):
    """The sowing card: what would be sown, under which law and in which soil, and where each field came from."""

    look: Look
    law: str
    provisional: bool
    soil: str
    place: tuple


class Sown(NamedTuple):
    """A sowing: the look after it, the genome's and the being's short prints, and whether it was named."""

    look: Look
    genome: str
    tag: str
    named: bool
    name_refusal: object


class Acted(NamedTuple):
    """A write: the look before it, what was appended, the law update it carried, and whether a settle followed.

    ``carried`` is ``(law, version, day)`` of the law update the store wrote
    right after a gesture; for a law update written with its code
    (``laws_apply``), it is that update itself.
    """

    look: Look
    appended: object
    carried: object
    settled: bool


class LawsView(NamedTuple):
    """The laws of a being's world: its law state, the proposal's diff, the law history, what is due, the offset."""

    look: Look
    state: object
    diff: object
    history: tuple
    due: object
    offset_now: object


class Verified(NamedTuple):
    """A deep verification: the look it ran on, and what it compared (``life.Deep``)."""

    look: Look
    deep: object


class Diffed(NamedTuple):
    """A law diff: the look it was read on, and what a law update from the proposal would change."""

    look: Look
    diff: object


def transport(channel, *, attended=False, principal=None):
    """The one ``membrane.Transport`` the garden builds: ``attended`` only when it is exactly ``True``."""
    from . import membrane

    return membrane.Transport(channel, principal=principal, attended=attended is True)


def terminal_attended(_os=None):
    """Whether a person sits at this terminal, in the foreground: all three conditions, else ``False``.

    Standard input is a terminal; this process group is the terminal's
    foreground group; and the controlling terminal opens. An ``OSError`` at
    any step is ``False``. ``_os`` stands in for the ``os`` module.
    """
    system = os if _os is None else _os
    try:
        if not system.isatty(0):
            return False
        if system.tcgetpgrp(0) != system.getpgrp():
            return False
        system.close(system.open("/dev/tty", system.O_RDONLY))
    except OSError:
        return False
    return True


def _auth_settings(path):
    """The auth settings as the auth module reads them: the file's mapping, else its defaults."""
    try:
        if not path.exists():
            return dict(_AUTH_DEFAULTS)
        import yaml

        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:  # noqa: BLE001 - a file that cannot be read gives the auth module's defaults
        return dict(_AUTH_DEFAULTS)


def _auth_connect(path):
    """A connection to the auth store at ``path`` that leaves every byte of its files as it found them.

    The platform's keyed connector opens it read-only when a WAL file or a
    rollback journal lies beside it: a read-only connection never
    checkpoints pending frames into the store, never deletes a WAL file and
    never rolls back a hot journal (it refuses, and the reader then answers
    ``False``). With neither beside it, a read-only open of a store kept in
    WAL mode would leave a WAL file and an index behind, so it is opened as
    usual, with SQL writes refused (``query_only``): its transient files are
    removed when it closes and the store's bytes do not move.
    """
    from opti_oignon import db_utils

    beside = (Path(str(path) + "-wal"), Path(str(path) + "-journal"))
    if any(file.exists() for file in beside):
        return db_utils.safe_connect(str(path), read_only=True)
    conn = db_utils.safe_connect(str(path))
    conn.execute("PRAGMA query_only = ON")
    return conn


# The largest ``work.ceilings.awake_day`` among the laws this engine carries, read once; the caps said raised.
_awake_day = {"value": None}
_raised = set()


def _day_of_work():
    """The work of one awake day under the costliest law this engine carries (its ``awake_day`` ceiling)."""
    if _awake_day["value"] is None:
        from . import lawfiles

        _awake_day["value"] = max(int(lawfiles.law(name)["work"]["ceilings"]["awake_day"]) for name in lawfiles.LAWS)
    return _awake_day["value"]


def api_view_cap(path=None):
    """The API's cap on a view's work: ``api.native_cap`` when the native core answers, else ``api.python_cap``.

    Read at each look, so a change of the settings file or of the engine in
    use is followed. An engine that cannot say which it is counts as the
    reference; settings that cannot be read give the reference's default
    cap. A cap below one awake day of the costliest law this engine carries
    is raised to that day, and the raise is logged once per value: under it,
    the view after a write made in ``oo garden`` would still be catching up,
    and the line a capped view says (the next write computes it) would be
    false. Never raises.
    """
    from . import engine, settings

    try:
        native = engine.native_in_use() is True
    except Exception:  # noqa: BLE001 - an engine that cannot say is the reference
        native = False
    try:
        read = settings.api(path)
    except Exception:  # noqa: BLE001 - settings that cannot be read give the reference's default
        return settings.DEFAULT_PYTHON_CAP
    cap = read.native_cap if native else read.python_cap
    try:
        day = _day_of_work()
    except Exception:  # noqa: BLE001 - laws that cannot be read leave the cap as written
        return cap
    if cap < day:
        if (cap, day) not in _raised:
            _raised.add((cap, day))
            logger.warning("the API's view cap %d is below one awake day of work (%d): %d is used", cap, day, day)
        return day
    return cap


def platform_single_user(config=None, root=None):
    """Whether the platform runs for one person, read without importing the auth module; ``False`` when unsure.

    The auth settings file (``config``, default the package's
    ``config/auth.yaml``): an explicit false single-user mode is ``False``;
    once a second account has been seen, ``False`` for the rest of the
    process. The auth store (``db_path``, relative to ``root``, default the
    project root) is looked at only if it exists: no directory is created,
    no connection is opened without it, and one that exists is opened
    through the platform's keyed connector so that nothing in it is written
    (``_auth_connect``), counted and closed. More than one account is
    ``False`` and latches; anything that fails is ``False``.
    """
    try:
        package = Path(__file__).resolve().parent.parent
        settings = _auth_settings(Path(config) if config is not None else package.joinpath("config", "auth.yaml"))
        if not settings.get("single_user_mode", True):
            return False
        if _MULTI_USER["latched"]:
            return False
        configured = Path(settings.get("db_path", _AUTH_DEFAULTS["db_path"]))
        base = Path(root) if root is not None else package.parent
        found = configured if configured.is_absolute() else base.joinpath(configured)
        if not found.exists():
            return True
        conn = _auth_connect(found)
        try:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        finally:
            conn.close()
        if count > 1:
            _MULTI_USER["latched"] = True
            return False
        return True
    except Exception:  # noqa: BLE001 - a count that cannot be read requires an account
        return False


def check_name(text):
    """A person's name as typed, its line end and outer spaces removed; ``ServiceRefused("name")`` if not one.

    1 to 32 characters of ``A-Z a-z 0-9``, space, hyphen, apostrophe and
    period; a letter or a digit first; no two spaces in a row; not starting
    with the beetle's word, in any case.
    """
    if not isinstance(text, str):
        raise ServiceRefused("name", detail="not text")
    name = text.rstrip("\r\n").strip(" ")
    if not 1 <= len(name) <= NAME_MAX:
        raise ServiceRefused("name", detail="length")
    if not set(name) <= _NAME_CHARS or name[0] not in _NAME_FIRST or "  " in name:
        raise ServiceRefused("name", detail="characters")
    if name.lower().startswith(_SPEAKER):
        raise ServiceRefused("name", detail="the speaker's word")
    return name


def sowing_law():
    """The newest law of ``CHANNEL`` this engine carries and has not retired; ``ServiceRefused("channel")``.

    Never the test law, and no fallback.
    """
    from . import lawfiles

    try:
        retired = {(entry["name"], entry["sha256"]) for entry in lawfiles.retired()}
        for name in reversed(CHANNEL):
            if name not in lawfiles.LAWS:
                continue
            if (name, lawfiles.digest(lawfiles.law(name))) in retired:
                continue
            return name
    except Exception:  # noqa: BLE001 - a register or a law that cannot be read names no law to sow
        raise ServiceRefused("channel", detail="the channel cannot be read") from None
    raise ServiceRefused("channel", detail="no law of the channel is carried and not retired")


def _channel_provisional():
    """Whether every law of ``CHANNEL`` is carried and provisional; ``False`` when it cannot be said."""
    from . import lawfiles, membrane

    try:
        return bool(CHANNEL) and all(
            name in lawfiles.LAWS and membrane.law_pin(name)["provisional"] is True for name in CHANNEL)
    except Exception:  # noqa: BLE001 - unknown: the label is left out
        return False


def _made(status, *, labels=(), mode=None, habitat=None, view=None, felt=None, being=None, reason=None,
          offer=None, seq=None, exit=None, glass_allowed=False):
    """A ``Look`` of ``status``, its labels in their fixed order, its exit the status's unless given."""
    from . import describe

    order = describe.LABEL_ORDER
    labels = tuple(sorted(dict.fromkeys(labels), key=order.index))
    if exit is None:
        exit = describe.STATUS_FORMS[status][1]
    return Look(status, labels, mode, habitat, view, felt, being, reason, offer, seq, exit, glass_allowed is True)


def _refusals():
    """The refusal classes a verb passes through: the store's, the membrane's, the laws writer's, the life's."""
    from . import evolution, life, membrane, store, wording

    return (store.StoreRefused, store.ResumeRefused, membrane.MembraneRefused, evolution.LawsRefused,
            life.LifeRefused, life.Diverged, wording.CopyRefused)


@contextlib.contextmanager
def _about(look):
    """A refusal raised inside, of a class the garden names, carries ``look`` when it has none."""
    try:
        yield
    except _refusals() as exc:
        if getattr(exc, "look", None) is None:
            exc.look = look
        raise


@contextlib.contextmanager
def _after_write(look):
    """A refusal raised inside, once a write has returned, is marked ``written``: the write itself was done.

    The reads and the settle that follow a write (the law update it
    carried, the view of minute 0, the look after) can still be refused;
    the refusal then says so, and never that nothing was written.
    """
    try:
        yield
    except _refusals() as exc:
        exc.written = True
        if getattr(exc, "look", None) is None:
            exc.look = look
        raise


def _view_reason(exc):
    """The reason a view refused after the store said alive is served with, or ``None`` if it is not one."""
    from . import life, membrane, store, wording

    if isinstance(exc, life.LifeRefused):
        return "engine"
    if isinstance(exc, membrane.MembraneRefused):
        return "clock" if exc.code == "clock" else None
    if isinstance(exc, store.StoreRefused) and not isinstance(exc, store.ChainRefused):
        return exc.code if exc.code in wording.REASONS else None
    return None


def _latest_name(ordered):
    name = None
    for _seq, _eid, fact in ordered:
        body = fact.get("body")
        if fact.get("kind") == "name" and isinstance(body, dict) and isinstance(body.get("name"), str):
            name = body["name"]
    return name


def _being_info(being, ordered):
    genesis = ordered[0][2]["body"]
    laws = genesis["laws"]
    return BeingInfo(_latest_name(ordered), being.soil, laws["name"], laws["v"], laws["provisional"],
                     laws["sha256"][:12], being.being_tag[:8], len(ordered), genesis["weather"])


def _mode_label(mode):
    return {"bulbe": "bulbe", "unknown": "mode_unknown"}.get(mode)


class Garden:
    """The componion's garden for one process: its seams, and one store built at most once, when first needed.

    ``view_cap`` (a callable of no argument answering a cap, or ``None``)
    bounds the engine's work of each view when no cap is asked for; it is
    called once per look, after the store said alive. ``caller_required``:
    every method that takes a caller refuses a missing one (``TypeError``).
    """

    def __init__(self, *, store_factory, switch, stopped=None, attended=None, law=None, view_cap=None,
                 caller_required=False):
        if not callable(store_factory) or not callable(switch):
            raise TypeError("store_factory and switch are callables")
        if view_cap is not None and not callable(view_cap):
            raise TypeError("view_cap is a callable or None")
        self.store_factory = store_factory
        self.switch = switch
        self.stopped = stopped
        self.attended = attended
        self.law = law
        self.view_cap = view_cap
        self.caller_required = caller_required is True
        self._store = None
        self._lock = threading.Lock()

    @classmethod
    def production(cls, *, stopped=None, terminal=True, view_cap=None, single_user=None):
        """This process's garden: the settings' switch and a single-user reader; the terminal's by default.

        The terminal's (the defaults): the strong terminal test, no emergency
        stop (the terminal's process holds none), no cap on a view, and
        ``platform_single_user``. With ``terminal`` anything but ``True``, the
        API's: no attended verb, a caller required on every method, the
        ``stopped`` seam, the ``view_cap`` and the ``single_user`` rule the
        API process gives. The law sown is the channel's. The store is built
        at the first action that needs it, with the single-user reader and
        every other seam at the store's own default.
        """
        from . import settings

        def build():
            from . import store

            reader = platform_single_user if single_user is None else single_user
            return store.Store(single_user=reader)

        terminal = terminal is True
        return cls(store_factory=build, switch=settings.switch, stopped=stopped,
                   attended=terminal_attended if terminal else None, law=None, view_cap=view_cap,
                   caller_required=not terminal)

    # -- the seams ------------------------------------------------------------

    def _given(self, caller):
        """A garden that requires its caller refuses a missing one: a defect of the caller, never a status."""
        if caller is None and self.caller_required:
            raise TypeError("this garden serves only a caller that is given")

    def store(self):
        """This process's one ``Store``, built on the first call."""
        with self._lock:
            if self._store is None:
                self._store = self.store_factory()
            return self._store

    def close(self):
        """Close the store if one was built."""
        with self._lock:
            store, self._store = self._store, None
        if store is not None:
            store.close()

    def _switch(self):
        """``"on"``, ``"off"`` or ``"unreadable"``; any other answer, or a switch that raises, is unreadable."""
        try:
            value = self.switch()
        except Exception:  # noqa: BLE001 - a switch that cannot be read is off
            return "unreadable"
        return value if isinstance(value, str) and value in _SWITCHES else "unreadable"

    def _stopped(self):
        """``None`` when nothing is stopped; ``"on"`` when the stop is on; ``"unknown"`` when it cannot be read."""
        if self.stopped is None:
            return None
        try:
            value = self.stopped()
        except Exception:  # noqa: BLE001 - a stop that cannot be read stops
            return "unknown"
        return "on" if value is True else None

    def _early(self):
        """The ``disabled`` or ``stopped`` look, decided before any store is built; ``None`` when neither."""
        switch = self._switch()
        if switch != "on":
            return _made("disabled", reason="unreadable" if switch == "unreadable" else None)
        stop = self._stopped()
        if stop is not None:
            return _made("stopped", reason="unknown" if stop == "unknown" else None)
        return None

    def _begin(self):
        """Refuse a verb the switch or the stop rules out, with that look; else nothing."""
        early = self._early()
        if early is not None:
            raise ServiceRefused("off" if early.status == "disabled" else "stopped", look=early)

    def _read_attended(self):
        if self.attended is None:
            return False
        try:
            return self.attended() is True
        except Exception:  # noqa: BLE001 - a terminal test that raises says no
            return False

    def _attended(self, caller):
        """The caller of an attended verb: the terminal's own when it is attended, else ``ServiceRefused``.

        A caller that is given must be a terminal's (``cli``) and attended;
        the terminal's own caller asks the terminal test once.
        """
        if caller is not None:
            if getattr(caller, "channel", None) != "cli" or getattr(caller, "attended", False) is not True:
                raise ServiceRefused("attended", detail="an attended terminal only")
            return caller
        if not self._read_attended():
            raise ServiceRefused("attended", detail="no person at a foreground terminal")
        return transport("cli", attended=True)

    # -- the look -------------------------------------------------------------

    def _look(self, store, action, caller, cap=None):
        """``(Look, being or None, Seen or None)`` of one look inside an action; writes nothing."""
        from . import habitat, membrane
        from .store import ChainRefused

        try:
            user = store.account(caller)
        except membrane.MembraneRefused:
            return _made("unavailable", mode=action.mode, reason="account"), None, None
        seen = store.look(user)
        status = seen.status.status
        mode = action.mode
        if status == "alive":
            return self._alive(seen, action, cap), seen.being, seen
        labels = []
        # A record that exists but is not opened: a prototype when every law of the channel is one; an unavailable
        # store counts when its file is there.
        unopened = status in _UNOPENED or (status == "unavailable" and seen.soil is not None)
        if unopened and _channel_provisional():
            labels.append("prototype")
        if status == "retired_prototype":
            labels.append("retired")
        if seen.soil == "glass" and status in ("sealed_bulbe", "retired_prototype", "unavailable"):
            labels.append("glass_jar")
        place = None
        if status in ("sealed_bulbe", "retired_prototype") and seen.soil is not None:
            place = habitat.layer(seen.soil, mode)
        known = status == "sealed_bulbe" or (seen.soil == "encrypted" and status in ("retired_prototype",
                                                                                     "unreadable"))
        if known and _mode_label(mode) is not None:
            labels.append(_mode_label(mode))
        seq = seen.refusal.seq if isinstance(seen.refusal, ChainRefused) else None
        look = _made(status, labels=labels, mode=mode, habitat=place, reason=seen.status.reason,
                     offer=seen.status.offer, seq=seq, glass_allowed=seen.glass_allowed)
        return look, None, seen

    def _alive(self, seen, action, cap):
        """The look of a being the store opened: its view, what ``show`` reads of it, who it is."""
        from . import describe, habitat, lawfiles, life

        being = seen.being
        mode = action.mode
        labels = list(seen.status.labels)
        label = _mode_label(mode) if being.soil == "encrypted" else None
        if label is not None:
            labels.append(label)
        # The store opened this being: an unavailable look about it carries its habitat, and says it cannot be
        # computed now rather than that its store cannot be opened.
        place = habitat.layer(being.soil, mode)
        try:
            ordered = being.in_order()
        except Exception as exc:  # noqa: BLE001 - a refusal the served state machine names is a status
            reason = _view_reason(exc)
            if reason is None:
                raise
            return _made("unavailable", labels=labels, mode=mode, habitat=place, reason=reason,
                         glass_allowed=seen.glass_allowed)
        info = _being_info(being, ordered)
        if cap is None and self.view_cap is not None:
            cap = self.view_cap()
        frozen = False
        try:
            view = being.view(cap=cap)
        except life.LifeRefused as exc:
            view = None
            if exc.code == "engine_panic":
                try:
                    view = life.stored(being, cap=cap)
                    frozen = True
                except Exception as fault:  # noqa: BLE001 - even the kept state refused: the engine's reason
                    if _view_reason(fault) is None:
                        raise
            if view is None:
                return _made("unavailable", labels=labels, mode=mode, habitat=place, being=info, reason="engine",
                             glass_allowed=seen.glass_allowed)
        except Exception as exc:  # noqa: BLE001 - a view refused after alive is served as unavailable
            reason = _view_reason(exc)
            if reason is None:
                raise
            return _made("unavailable", labels=labels, mode=mode, habitat=place, being=info, reason=reason,
                         glass_allowed=seen.glass_allowed)
        if view.status == "catching_up":
            labels.append("catching_up")
        if frozen:
            labels.append("frozen")
        constants = lawfiles.law(view.law["name"])["constants"]
        look = _made("alive", labels=labels, mode=mode, habitat=place, view=view, being=info,
                     exit=1 if frozen else 0, glass_allowed=seen.glass_allowed)
        felt = describe.felt(view, name=info.name, weather=info.weather, soil=being.soil, layer=place[1],
                             labels=look.labels, constants=constants)
        return look._replace(felt=felt)

    def _unavailable_after(self, look, reason):
        """The look of a being whose laws or view the engine refused after the look said alive; its habitat kept."""
        return look._replace(status="unavailable", view=None, felt=None, reason=reason, exit=1,
                             labels=tuple(label for label in look.labels if label not in ("catching_up", "frozen")))

    def _opened(self, store, action, caller):
        """``(Look, being)`` of an alive being; ``ServiceRefused("status")`` with the look otherwise."""
        look, being, _seen = self._look(store, action, caller)
        if look.status != "alive":
            raise ServiceRefused("status", look=look)
        return look, being

    def look(self, caller=None, cap=None):
        """What the being is now, as a ``Look``: every outcome a status; nothing is written."""
        self._given(caller)
        early = self._early()
        if early is not None:
            return early
        store = self.store()
        with store.action() as action:
            return self._look(store, action, caller if caller is not None else transport("cli"), cap)[0]

    # -- sowing ---------------------------------------------------------------

    def _card(self, store, look, seen, hemisphere, band, weather):
        """``(Card, offset)`` for a store the look says is ready; refused by name otherwise."""
        from . import evolution, membrane

        if look.status != "ready":
            identified = look.status in ("sealed_bulbe", "missing", "retired_prototype", "unreadable", "alive")
            raise ServiceRefused("exists" if identified else "status", look=look)
        law = self.law if self.law is not None else sowing_law()
        with _about(look):
            pin = membrane.law_pin(law)
            raw = store._laws_settings()
            given = {"hemisphere": hemisphere, "band": band, "weather": weather}
            chosen = evolution.sowing(pin, raw, **given)
            evolution.proposal(pin, raw)
        proposed = raw.get("sowing") if isinstance(raw, dict) else None
        proposed = proposed if isinstance(proposed, dict) else {}
        place = []
        for field in ("hemisphere", "band", "weather"):
            if given[field] is not None:
                source = "option"
            elif proposed.get(field) is not None:
                source = "file"
            else:
                source = "default"
            place.append((field, chosen[field], source))
        _wall, offset = store.reading()
        if offset is None:
            raise ServiceRefused("offset", look=look, detail="the local offset cannot be read")
        card = Card(look, law, pin["provisional"], seen.soil, tuple(place))
        return card, offset

    def sow_card(self, caller=None, *, hemisphere=None, band=None, weather=None):
        """The sowing card, every check a sowing makes done first; nothing is written."""
        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            caller = self._attended(caller)
            look, _being, seen = self._look(store, action, caller)
            return self._card(store, look, seen, hemisphere, band, weather)[0]

    def sow(self, caller=None, *, name=None, hemisphere=None, band=None, weather=None):
        """Sow the one seed of this garden, name it when a name is given, and say what was sown (``Sown``)."""
        from . import membrane

        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            caller = self._attended(caller)
            clean = None if name is None else check_name(name)
            look, _being, seen = self._look(store, action, caller)
            card, offset = self._card(store, look, seen, hemisphere, band, weather)
            with _about(look):
                being = store.sow(transport=caller, law=card.law, tz_minutes=offset, rhythm_consent=RHYTHM_CONSENT,
                                  hemisphere=hemisphere, band=band, weather=weather)
            with _after_write(look):
                named, refusal = False, None
                if clean is not None:
                    try:
                        appended = being.append("name", {"name": clean}, transport=caller)
                    except _refusals() as exc:
                        refusal = getattr(exc, "code", "other")
                    else:
                        if isinstance(appended, membrane.Dropped):
                            refusal = appended.reason
                        else:
                            named = True
                self._settle(being)
                zero = being.view(to=0)
                after = self._look(store, action, caller)[0]
            return Sown(after, zero.state["genome"][:12], being.being_tag[:8], named, refusal)

    # -- writes ---------------------------------------------------------------

    def _settle(self, being):
        """A settle of the being's caches after a write; whatever it raises is logged and changes nothing said.

        The person's write has succeeded: a refusal, or any other failure of
        the settle, is logged by its code or class alone and the verb's
        output and exit are those of the write.
        """
        try:
            return being.settle().done is True
        except Exception as exc:  # noqa: BLE001 - a settle never changes what the write said
            logger.warning("the settle after a write was refused: %s", getattr(exc, "code", type(exc).__name__))
            return False

    def _carried(self, being, appended):
        """``(law, version, day)`` of the law update the store wrote right after ``appended``, or ``None``."""
        if being.head()[0] == appended.seq:
            return None
        for seq, _eid, fact in being.in_order():
            if (seq == appended.seq + 1 and fact["kind"] == "evolve" and fact["t"] == appended.t
                    and fact["origin"] == being.origin and fact["oseq"] == appended.oseq + 1):
                to = fact["body"]["to"]
                return (to["name"], to["v"], fact["body"]["effective_from"] // DAY)
        return None

    def _written(self, being, appended):
        """``(law, version, day)`` of the law update ``appended`` is; refused ``local`` when it is not read back."""
        from .store import StoreRefused

        for seq, _eid, fact in being.in_order():
            if seq == appended.seq and fact["kind"] == "evolve":
                to = fact["body"]["to"]
                return (to["name"], to["v"], fact["body"]["effective_from"] // DAY)
        raise StoreRefused("local", "the law update just written is not read back")

    def _journaled(self, look, being, write, *, carried="gesture"):
        """Run one journaled write about ``look``'s being, then the settle: ``Acted``.

        ``carried`` says which law update the record names: the one a
        ``gesture`` carried, the ``update`` the write itself is, or none.
        """
        from . import membrane

        with _about(look):
            appended = write()
            if isinstance(appended, membrane.Dropped):
                raise ServiceRefused("budget", look=look, detail=appended.reason)
        with _after_write(look):
            if carried == "gesture":
                update = self._carried(being, appended)
            elif carried == "update":
                update = self._written(being, appended)
            else:
                update = None
            settled = self._settle(being)
        return Acted(look, appended, update, settled)

    def act(self, act, caller=None):
        """A gesture (``greet``, ``water``, ``warm``, ``play``): one fact, and the law update it may carry."""
        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            caller = caller if caller is not None else transport("cli")
            look, being = self._opened(store, action, caller)
            return self._journaled(look, being, lambda: being.append("act", {"act": act}, transport=caller))

    def name_card(self, caller=None):
        """The checks a name goes through before its question is asked: the look of an alive being."""
        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            caller = self._attended(caller)
            return self._opened(store, action, caller)[0]

    def name(self, text, caller=None):
        """Name the being: one permanent ``name`` fact, from an attended terminal."""
        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            caller = self._attended(caller)
            look, being = self._opened(store, action, caller)
            clean = check_name(text)
            return self._journaled(look, being, lambda: being.append("name", {"name": clean}, transport=caller))

    def laws_apply(self, confirm, caller=None):
        """Write the law update the diff confirmed, with its code: ``Acted``, ``carried`` the update itself."""
        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            caller = self._attended(caller)
            look, being = self._opened(store, action, caller)
            return self._journaled(look, being, lambda: being.laws_apply(caller, confirm), carried="update")

    def laws_pin(self, caller=None):
        """Pin the laws in force, from an attended terminal."""
        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            caller = self._attended(caller)
            look, being = self._opened(store, action, caller)
            return self._journaled(look, being, lambda: being.laws_pin(caller), carried=None)

    def laws_unpin(self, caller=None):
        """Lift the pin, from an attended terminal."""
        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            caller = self._attended(caller)
            look, being = self._opened(store, action, caller)
            return self._journaled(look, being, lambda: being.laws_unpin(caller), carried=None)

    # -- reading the record ---------------------------------------------------

    def verify(self, caller=None):
        """Deep verification of an alive being (``life.deep_verify``): ``Verified``; nothing is written."""
        from . import life

        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            look, being = self._opened(store, action, caller if caller is not None else transport("cli"))
            with _about(look):
                return Verified(look, life.deep_verify(being))

    def laws(self, caller=None):
        """The laws of the being's world (``LawsView``); a proposal that cannot be used is kept, not raised."""
        from . import evolution

        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            look, being = self._opened(store, action, caller if caller is not None else transport("cli"))
            try:
                with _about(look):
                    state = being.law_state()
                    try:
                        diff = being.laws_diff()
                    except evolution.LawsRefused as refusal:
                        diff = refusal
                    due = evolution.due(being)
            except Exception as exc:  # noqa: BLE001 - a laws screen the engine refused is served as unavailable
                reason = _view_reason(exc)
                if reason is None:
                    raise
                raise ServiceRefused("status", look=self._unavailable_after(look, reason)) from None
            history = tuple((seq, fact["t"], fact["kind"], fact["body"]) for seq, _eid, fact in being.in_order()
                            if fact["kind"] in _LAW_KINDS)
            _wall, offset = store.reading()
            return LawsView(look, state, diff, history, due, offset)

    def laws_diff(self, caller=None):
        """What a law update from the proposal would change (``Diffed``); a proposal that is not one is refused."""
        from . import evolution

        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            look, being = self._opened(store, action, caller if caller is not None else transport("cli"))
            try:
                with _about(look):
                    return Diffed(look, being.laws_diff())
            except evolution.LawsRefused:
                raise
            except Exception as exc:  # noqa: BLE001 - a diff the engine refused is served as unavailable
                reason = _view_reason(exc)
                if reason is None:
                    raise
                raise ServiceRefused("status", look=self._unavailable_after(look, reason)) from None

    # -- restore, never repair ------------------------------------------------

    def resume(self, kept, discarded, caller=None):
        """Resume a record that failed its verification, with the numbers its look showed; the look after."""
        from .store import ResumeRefused

        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            caller = self._attended(caller)
            look, _being, _seen = self._look(store, action, caller)
            if look.status in _RESTORE_FORM + ("retired_prototype",):
                raise ServiceRefused("status", look=look)
            offer = look.offer
            if look.status != "unreadable" or getattr(offer, "kept_seq", -1) < 0:
                refusal = ResumeRefused("nothing", "there is nothing to resume")
                refusal.look = look
                raise refusal
            with _about(look):
                being = store.resume(transport=caller, confirm=(kept, discarded))
            with _after_write(look):
                self._settle(being)
                return self._look(store, action, caller)[0]

    def finish(self, tag, caller=None):
        """Finish an interrupted sowing, with the tag its look showed; the look after."""
        from .store import ResumeRefused

        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            caller = self._attended(caller)
            look, _being, _seen = self._look(store, action, caller)
            if look.status in _RESTORE_FORM:
                raise ServiceRefused("status", look=look)
            if look.status != "missing" or getattr(look.offer, "being", None) is None:
                refusal = ResumeRefused("nothing", "there is no interrupted sowing to finish")
                refusal.look = look
                raise refusal
            with _about(look):
                store.finish_sowing(transport=caller, confirm=tag)
            with _after_write(look):
                return self._look(store, action, caller)[0]

    # -- consent --------------------------------------------------------------

    def confirm_share(self, code, caller=None):
        """Confirm a consent request: none can be opened in this version, so this is always refused.

        An alive being first; then the membrane's own test of an attended
        terminal; then ``ServiceRefused("consent")``. Nothing is written.
        """
        from . import membrane

        self._given(caller)
        self._begin()
        store = self.store()
        with store.action() as action:
            if caller is not None and getattr(caller, "channel", None) != "cli":
                raise ServiceRefused("attended", detail="an attended terminal only")
            look, _being = self._opened(store, action, caller if caller is not None else transport("cli"))
            asking = caller if caller is not None else transport("cli", attended=self._read_attended())
            with _about(look):
                membrane.permit("grant", asking, action.wall)
            raise ServiceRefused("consent", look=look, detail="no consent request is open")

    # -- the gate -------------------------------------------------------------

    def gated(self, capability, call):
        """``call()`` when the policy of this action's mode allows ``capability``; ``None``, uncalled, otherwise.

        One action of its own, or the one in progress. A garden switched off
        or stopped builds no store and calls nothing; ``KeyError`` for a
        capability the policy does not name.
        """
        from . import habitat

        if capability not in habitat.CAPABILITIES:
            raise KeyError(capability)
        if self._switch() != "on" or self._stopped() is not None:
            return None
        with self.store().action() as action:
            if not habitat.allows(action.mode, capability):
                return None
            return call()
