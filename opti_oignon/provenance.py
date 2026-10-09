#!/usr/bin/env python3
"""Where each argument of a tool call came from, and what a call may carry out of the machine.

A tool call is the model's choice, and so is every argument it carries: words
the model read in a page, a document, a tool result or the memory reach an
argument as easily as the user's own. A call that reaches the network can
carry those words out of the machine -- an exfiltration by planted
instruction, where the attacker acts only through content the model reads.
This module is the gate both passage points apply to every call, the chat's
``ToolExecutor._execute_tool`` and the agent's ``dispatch_tool_call``, on the
arguments as they execute: after their names are repaired, their defaults
filled and their values coerced.

* LABELS. Each argument carries one. ``typed`` when, folded (Unicode NFC,
  each run of white space one space, the ends stripped), it equals a part
  the user typed in the current turn, whole -- the unit the pending writes
  endorse with, never a sentence or a line of it; for an argument that does
  not leave the machine, only when it is that part character for character,
  so "you typed it" stays literally true where nothing replaces the value.
  ``default`` when it equals, in type and in value, what the tool's own
  declaration gives it. ``unendorsed`` otherwise. A label is a fact about
  the value, never a reading of its words.
* CLASSES. Every tool has an effect class from a closed table: ``none``;
  ``session``, state of the run alone; ``sandbox``, inside the disposable
  sandbox, the net for files and processes; ``deferred``, a write held for
  the user's review; ``network``; ``approved``, behind a human approval of
  its own. A name the table does not know is held as ``network``, the class
  the gate holds tightest, and so is a tool declared as reaching the
  network, whatever its name.
* MODES. Daily permits every class. Bulbe permits ``none``, ``session`` and
  ``sandbox`` alone, and asks the user before each call that is neither
  ``none`` nor ``session``.
* THE NETWORK POLICY. In Daily, ``config/provenance.yaml`` says what a
  network call that carries an unendorsed argument does: ``free``, the
  default, goes out; ``ask_unendorsed`` waits for the user, who is shown the
  call as it would run and the label of each argument -- every value whole,
  so a network value longer than the surfaces can show (``SHOWN_LIMIT``) is
  refused rather than asked; ``refuse_unendorsed`` is refused. Only an
  absent file, or one that says ``free``, is free: a file that is present but
  cannot be read, names no policy, or gives a value outside the three reads
  as ``refuse_unendorsed`` -- an unreadable policy is an unknown, not a
  default. A policy given with a turn can only tighten the file's. An
  endorsed network argument is handed to its sink as the typed part's own
  characters, never the model's spelling of them, so not one of them is the
  model's choice (the search may still scrub personal data from it before
  it leaves).
* FAIL CLOSED. A call that must be asked and has no way to reach the user is
  refused, and the model is told why; so is a call of a tool the turn did
  not offer, or of a class the machine's mode does not permit, read at each
  call.

The provenance of a turn travels with each call, never on a shared object:
two turns that overlap on one executor are each judged by their own words.

What this cannot see: the user's own words, sent whole to the network by a
planted instruction, still go -- they are the user's, and the turn's. A
search the user asked for in other words than its query is unendorsed, so
the asking and refusing policies ask or refuse it: the cost of the strict
settings is that gesture or that refusal. Words a user pasted into the web
chat are told apart and endorse nothing, nor do the words typed beside them;
words pasted into a terminal's prompt, an agent run's task and a message a
client posts whole count as typed as their caller vouches. How many calls a
turn makes, and when, can still signal a little. The gate holds what a call
can carry, not what the model says.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Collection, Mapping

from opti_oignon.pending_writes import Endorsers, typed_units

logger = logging.getLogger(__name__)

checkpoint_before_apply = True

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
TYPED = "typed"
DEFAULT = "default"
UNENDORSED = "unendorsed"
LABELS = (TYPED, DEFAULT, UNENDORSED)

# No default: the declaration gives the argument no value of its own.
MISSING = object()

# ---------------------------------------------------------------------------
# Effect classes
# ---------------------------------------------------------------------------
NONE = "none"
SESSION = "session"
SANDBOX = "sandbox"
DEFERRED = "deferred"
NETWORK = "network"
APPROVED = "approved"
EFFECTS = (NONE, SESSION, SANDBOX, DEFERRED, NETWORK, APPROVED)

# The closed table: every tool the agent and the chat declare, with its class.
TOOL_EFFECTS: Mapping[str, str] = MappingProxyType({
    # The agent: the sandboxed seven, the network search, the three writes
    # held for review (a skill is always proposed), and the run's own two.
    "bash": SANDBOX,
    "view": SANDBOX,
    "create_file": SANDBOX,
    "str_replace": SANDBOX,
    "grep": SANDBOX,
    "glob": SANDBOX,
    "ls": SANDBOX,
    "web_search": NETWORK,
    "manage_memory": DEFERRED,
    "manage_notes": DEFERRED,
    "manage_skills": DEFERRED,
    "todo": SESSION,
    "task": SESSION,
    # The chat: the four built-ins that run only in a sandbox session, and
    # the sandbox session's own four.
    "execute_code": SANDBOX,
    "read_file": SANDBOX,
    "write_file": SANDBOX,
    "list_files": SANDBOX,
    "sandbox_bash": SANDBOX,
    "sandbox_view": SANDBOX,
    "sandbox_create_file": SANDBOX,
    "sandbox_str_replace": SANDBOX,
})

MODE_DAILY = "daily"
MODE_BULBE = "bulbe"
_PERMITTED = {
    MODE_DAILY: frozenset(EFFECTS),
    MODE_BULBE: frozenset({NONE, SESSION, SANDBOX}),
}
# Classes a Bulbe call runs without asking: they act on nothing outside the run.
_UNASKED = frozenset({NONE, SESSION})

# ---------------------------------------------------------------------------
# The network policy
# ---------------------------------------------------------------------------
FREE = "free"
ASK_UNENDORSED = "ask_unendorsed"
REFUSE_UNENDORSED = "refuse_unendorsed"
# Loosest first: a policy is tightened by moving right, never left.
NETWORK_POLICIES = (FREE, ASK_UNENDORSED, REFUSE_UNENDORSED)

POLICY_FILE = Path(__file__).resolve().parent / "config" / "provenance.yaml"
_POLICY_KEY = "network_daily"

# The longest value, in characters, the approval surfaces show whole
# (``tool_call_approval.SHOWN_CHARS``, held equal by a contract). A person
# asked about a network call must be able to see all that would leave, so a
# longer unendorsed network value is refused, never asked.
SHOWN_LIMIT = 2000

# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------
ALLOWED = "allowed"
NOT_OFFERED = "not_offered"
NOT_PERMITTED = "not_permitted"
NO_CHANNEL = "no_channel"
DENIED = "denied"
UNENDORSED_REFUSED = "unendorsed"
UNSHOWABLE = "unshowable"

# The words the chat's approval gate has always said on a denial.
_DENIED_BY_GATE = "Tool call denied by approval gate"


def effect_of(tool_name: str, *, network: bool | None = None) -> str:
    """A tool's class: the table's, and ``network`` for a name it does not know or a tool declared networked."""
    if network:
        return NETWORK
    return TOOL_EFFECTS.get(str(tool_name), NETWORK)


def permitted(effect: str, mode: str) -> bool:
    """Whether ``mode`` permits a call of class ``effect``; a mode the table does not know is held as Bulbe."""
    return effect in _PERMITTED.get(mode, _PERMITTED[MODE_BULBE])


def machine_mode() -> str:
    """The machine's security mode at this moment, Bulbe when it cannot be read."""
    try:
        from opti_oignon.agent import allowlists

        return allowlists.current_mode()
    except Exception:
        return MODE_BULBE


def read_network_policy(path: str | Path | None = None) -> str:
    """Daily's network policy as the file says it.

    A missing file is ``free``, the shipped default. A file that is present
    but cannot be read or parsed, names no policy (empty, or another setting
    only), or gives a value outside the three is ``refuse_unendorsed``, and
    the reason is logged: a zero-byte file left by a failed write must not
    loosen a stricter choice.
    """
    target = Path(path) if path is not None else Path(POLICY_FILE)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return FREE
    except Exception as exc:
        logger.warning("provenance policy %s cannot be read (%s): refuse_unendorsed", target.name, exc)
        return REFUSE_UNENDORSED
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception as exc:
        logger.warning("provenance policy %s cannot be parsed (%s): refuse_unendorsed", target.name, exc)
        return REFUSE_UNENDORSED
    if not isinstance(data, dict):
        logger.warning("provenance policy %s is not a mapping: refuse_unendorsed", target.name)
        return REFUSE_UNENDORSED
    if _POLICY_KEY not in data:
        logger.warning("provenance policy %s names no %s: refuse_unendorsed", target.name, _POLICY_KEY)
        return REFUSE_UNENDORSED
    value = data[_POLICY_KEY]
    if isinstance(value, str) and value in NETWORK_POLICIES:
        return value
    logger.warning("provenance policy %s: %s %r is not one of %s; refuse_unendorsed", target.name, _POLICY_KEY,
                   value, ", ".join(NETWORK_POLICIES))
    return REFUSE_UNENDORSED


def _tighter(first: str, second: str) -> str:
    rank = {policy: index for index, policy in enumerate(NETWORK_POLICIES)}
    return first if rank[first] >= rank[second] else second


def network_policy(mode: str, *, explicit: str | None = None, path: str | Path | None = None) -> str:
    """The policy a network call obeys in ``mode``: the file's, tightened by ``explicit``; Bulbe refuses."""
    if mode != MODE_DAILY:
        return REFUSE_UNENDORSED
    filed = read_network_policy(path)
    if explicit is None:
        return filed
    given = explicit if explicit in NETWORK_POLICIES else REFUSE_UNENDORSED
    return _tighter(filed, given)


# ---------------------------------------------------------------------------
# A turn's provenance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TurnProvenance:
    """One turn's provenance: the parts its user typed, whole; its way to ask the user; its policy, if any.

    ``demand`` is called as ``demand(tool_name, arguments, labels=..., effect=...)``
    and answers whether the user allows the call; None means the turn has no
    way to ask. ``policy`` tightens the file's network policy for this turn.
    """

    units: tuple[str, ...] = ()
    demand: Callable[..., bool] | None = field(default=None, compare=False)
    policy: str | None = None
    _endorsers: Endorsers = field(default=None, init=False, compare=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "units", tuple(self.units))
        object.__setattr__(self, "_endorsers", Endorsers(self.units))

    @classmethod
    def of_turn(cls, content: str, origin: str = "legacy", segments: Collection = (), *,
                demand: Callable[..., bool] | None = None, policy: str | None = None) -> TurnProvenance:
        """The provenance of a turn whose words, of ``origin``, are ``content`` with ``segments``."""
        units = typed_units(content, origin, segments) if content else ()
        return cls(units=units, demand=demand, policy=policy)

    @classmethod
    def of_user_turn(cls, claim: Any, *, demand: Callable[..., bool] | None = None,
                     policy: str | None = None) -> TurnProvenance:
        """The provenance of the turn a caller composed (its content, origin and segments); None endorses nothing."""
        if claim is None:
            return cls(demand=demand, policy=policy)
        return cls.of_turn(str(getattr(claim, "content", "") or ""), str(getattr(claim, "origin", "legacy")),
                           tuple(getattr(claim, "segments", ()) or ()), demand=demand, policy=policy)

    def endorse(self, value: Any) -> str | None:
        """The typed part ``value`` equals once both are folded, or None."""
        return self._endorsers.endorse(value)


NO_PROVENANCE = TurnProvenance()


def label(value: Any, provenance: TurnProvenance | None = None, *, default: Any = MISSING) -> str:
    """The label of one argument value: typed, default (its declared value, type and all), or unendorsed."""
    turn = provenance if provenance is not None else NO_PROVENANCE
    if turn.endorse(value) is not None:
        return TYPED
    if default is not MISSING and type(value) is type(default) and value == default:
        return DEFAULT
    return UNENDORSED


def label_arguments(arguments: Mapping[str, Any] | None, provenance: TurnProvenance | None = None,
                    defaults: Mapping[str, Any] | None = None) -> dict[str, str]:
    """The label of each argument given, by name."""
    declared = defaults or {}
    return {str(name): label(value, provenance, default=declared.get(name, MISSING))
            for name, value in (arguments or {}).items()}


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Assessment:
    """A call as it would run: its class, the label of each argument, and the arguments themselves."""

    effect: str
    labels: Mapping[str, str]
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Verdict:
    """What the gate decided for one call, why, and what the model is told when it is refused."""

    allowed: bool
    reason: str
    message: str
    assessment: Assessment
    asked: bool = False

    @property
    def effect(self) -> str:
        return self.assessment.effect

    @property
    def labels(self) -> dict[str, str]:
        return dict(self.assessment.labels)

    @property
    def arguments(self) -> dict[str, Any]:
        return dict(self.assessment.arguments)

    def metadata(self) -> dict[str, Any]:
        """The provenance a result carries: the class, each argument's label, the decision, and whether a person was asked."""
        return {"effect": self.effect, "labels": self.labels, "decision": self.reason, "asked": self.asked}


def assess(tool_name: str, arguments: Mapping[str, Any] | None, provenance: TurnProvenance | None = None, *,
           defaults: Mapping[str, Any] | None = None, network: bool | None = None) -> Assessment:
    """Label a call's arguments as they would run.

    An endorsed network argument takes the typed part's own characters; an
    argument that stays on the machine runs as given, so it is typed only
    when it already is the typed part, character for character.
    """
    turn = provenance if provenance is not None else NO_PROVENANCE
    effect = effect_of(tool_name, network=network)
    executed = dict(arguments or {})
    labels = label_arguments(executed, turn, defaults)
    for name, tag in labels.items():
        if tag != TYPED:
            continue
        unit = turn.endorse(executed[name])
        if effect == NETWORK:
            executed[name] = unit
        elif executed[name] != unit:
            labels[name] = UNENDORSED
    return Assessment(effect=effect, labels=MappingProxyType(dict(labels)),
                      arguments=MappingProxyType(executed))


def _shown_length(value: Any) -> int:
    """How many characters a value takes on the approval surfaces before any cut."""
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=True, default=str))
    except Exception:
        return len(str(value))


def accepts_labels(fn: Callable[..., Any]) -> bool:
    """Whether an approval callable takes the labels and the class by keyword."""
    try:
        parameters = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return True
    return {"labels", "effect"} <= {p.name for p in parameters}


def _ask(fn: Callable[..., Any], tool_name: str, assessment: Assessment) -> bool:
    """Put the call, as it would run, to a person; anything but an explicit yes is a no."""
    arguments = dict(assessment.arguments)
    try:
        if accepts_labels(fn):
            return bool(fn(tool_name, arguments, labels=dict(assessment.labels), effect=assessment.effect))
        return bool(fn(tool_name, arguments))
    except Exception as exc:
        logger.warning("asking about %s failed (%s): refused", tool_name, exc)
        return False


def _named(names: Collection[str]) -> str:
    return ", ".join(f"{name} ({UNENDORSED})" for name in names)


def decide(tool_name: str, assessment: Assessment, provenance: TurnProvenance | None = None, *, mode: str,
           offered: Collection[str] | None = None, approval: Callable[..., Any] | None = None,
           approval_required: bool = False) -> Verdict:
    """Decide one assessed call: offered, permitted in ``mode``, within the network policy, asked when it must be."""
    turn = provenance if provenance is not None else NO_PROVENANCE
    effect = assessment.effect

    def refused(reason: str, message: str, asked: bool = False) -> Verdict:
        logger.info("tool call refused: %s (%s, %s)", tool_name, effect, reason)
        return Verdict(False, reason, message, assessment, asked)

    if offered is not None and tool_name not in offered:
        return refused(NOT_OFFERED, f"Refused: the tool '{tool_name}' was not offered in this turn; nothing ran.")
    if not permitted(effect, mode):
        return refused(NOT_PERMITTED,
                       f"Refused: the tool '{tool_name}' ({effect}) is not permitted in {mode} mode; nothing ran.")
    unendorsed = [name for name, tag in assessment.labels.items() if tag == UNENDORSED]
    policy = network_policy(mode, explicit=turn.policy) if effect == NETWORK and unendorsed else FREE
    if policy == REFUSE_UNENDORSED:
        return refused(UNENDORSED_REFUSED,
                       f"Refused: the call to '{tool_name}' would send {_named(unendorsed)}, which the user did not "
                       "type in this turn; the network policy refuses such calls, and nothing was sent.")
    asking = approval_required or policy == ASK_UNENDORSED
    if effect == NETWORK and asking and any(_shown_length(assessment.arguments.get(name)) > SHOWN_LIMIT
                                            for name in unendorsed):
        return refused(UNSHOWABLE,
                       f"Refused: the call to '{tool_name}' would send {_named(unendorsed)} longer than the "
                       f"{SHOWN_LIMIT} characters the user can be shown, so it cannot be put to them; nothing was "
                       "sent.")
    asked = False
    if approval_required:
        channel = approval if approval is not None else turn.demand
        if channel is None:
            return refused(NO_CHANNEL, f"Refused: the tool '{tool_name}' needs the user's approval, and this turn "
                                       "has no way to ask for it; nothing ran.")
        asked = True
        if not _ask(channel, tool_name, assessment):
            return refused(DENIED, _DENIED_BY_GATE, asked=True)
    if policy == ASK_UNENDORSED and not asked:
        if turn.demand is None:
            return refused(NO_CHANNEL,
                           f"Refused: the call to '{tool_name}' would send {_named(unendorsed)}, which the user did "
                           "not type in this turn, and this turn has no way to ask the user; nothing was sent.")
        asked = True
        if not _ask(turn.demand, tool_name, assessment):
            return refused(DENIED, f"Refused: the user did not allow the call to '{tool_name}'; nothing was sent.",
                           asked=True)
    return Verdict(True, ALLOWED, "", assessment, asked)


def check(tool_name: str, arguments: Mapping[str, Any] | None, provenance: TurnProvenance | None = None, *,
          defaults: Mapping[str, Any] | None = None, network: bool | None = None, mode: str | None = None,
          offered: Collection[str] | None = None, approval: Callable[..., Any] | None = None,
          approval_required: bool | None = None) -> Verdict:
    """Assess and decide one call; the mode is the machine's, read now, unless given.

    With ``approval_required`` left unset, a call is put to ``approval`` when
    one is given, and in any mode but Daily when its class acts outside the
    run.
    """
    assessment = assess(tool_name, arguments, provenance, defaults=defaults, network=network)
    resolved = mode if mode is not None else machine_mode()
    if approval_required is None:
        approval_required = approval is not None or (resolved != MODE_DAILY and assessment.effect not in _UNASKED)
    return decide(tool_name, assessment, provenance, mode=resolved, offered=offered, approval=approval,
                  approval_required=approval_required)
