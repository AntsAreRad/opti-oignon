#!/usr/bin/env python3
"""Contracts for the garden's relationship with the person: what it may say, and every state it serves.

Every English line the garden prints is a template of one closed catalogue
(``wording``), rendered with typed slots and checked by two nets
(``ethics``): a class of governed words, of which only nine allowed verbs
pass, and a blocklist of phrases, grouped by the psychological law each one
serves. One sentence, the doctrine, is exempted by its exact bytes.
The served state machine (``service``) derives one status per action, and
``describe`` gives every status its form, its stream and its exit.

  * RL1 -- the nets are pinned and hold: the allowed lemmas and phrases, the
    governed words, the blocklist by group and the field stems are exactly
    those written below; the floor of banned phrases is in the blocklist,
    the nine allowed verbs pass, and the doctrine alone is exempt. Each net
    fires alone on its own witnesses and both on "It missed you."; the
    twelve sentences an earlier word list let through are caught, the
    doctrine less its "not" is refused, and "It asks for water." passes;
    plain breaches a review found (a count of days away, a plea, a death
    word, a loss to come) are caught. The
    whole catalogue (its key set as written, counted in the dict literal),
    rendered with every value of every closed code map, 2000 seeded
    ``Felt`` draws in both tiers, the 480 conditions lines, the help of
    every garden command, click's own lines for a command line it refuses,
    and the form of every status with every label
    combination give no finding, each rendered row printable ASCII within
    78 columns and the art within 32 with no ``@`` and no ``o``; every key,
    word class and status is reached. A person's name is masked, so "Lonely" is a name. The terminal
    writes through one ``click.echo`` and one ``echo_error``, and the garden's
    declarations write nothing; every law of ``ethics.LAWS`` names a
    mechanism that resolves, or owes one in plain English; the ethics, the
    catalogue and the description import no clock, no randomness and no
    model.
  * RL2 -- no field of a record, and no key of the served JSON, names an
    absence: every record's fields and the served fields of a look of every
    status (frozen and catching up too) pass the field stems; "visits",
    "since" and "seen" pass, "streak_days" and "lastSeen" are refused. The
    served JSON is closed: its keys and the keys of each object it holds are
    exactly those of its projection, for every look.
  * RL19 -- every status is served with its form, its stream and its exit:
    the switch reads only a YAML true as on; switched off, the garden says
    so, names its file, builds no store and reads no mode, account, clock,
    engine or audit log; the emergency stop, the soil, the account; one
    being alive, then a rest fact, a wall before its birth, a kept state
    that lies, an engine limit, a kept state past the engine's input limit,
    its key unreadable, a read of its record refused, a stray store beside
    it, its law retired, a body edited and its file removed -- each with its
    own form and labels, never ``ready``, and an unreadable store never said
    lost unless its chain is what broke; every line passes the nets, every
    JSON passes the stems, and the doctrine closes every show. The
    laboratory prints the doctrine whole, first on stdout, after its status
    line on stderr.
  * RL24 -- the doctrine closes every show, in both tiers and in the JSON
    lines, and opens the lab, the laws screen and the sow card.
  * RL26 -- restore, never repair, from the terminal: a record broken at an
    event is resumed only with the numbers the garden shows, and an
    interrupted sowing is finished only with its own tag; an unattended
    terminal is refused both; the settle after a resume keeps a state.
  * RL27 -- the refusal table is closed: every refusal of every class, for
    every verb, renders catalogue lines with a usage or failure exit, its
    head is what was refused (never a label, never the doctrine), and no
    detail and no exception text ever reaches a stream. A name follows its
    rules; a refusal raised after a write returned says the write was done;
    a settle that fails after a gesture changes nothing it said.

Local-only (the public distribution ships no tests). The platform and the
terminal load through the shared isolation window with the platform's
configuration, keys, mode, audit log and user modules proven unreachable;
every seam is injected (``tests/_allium_store_support.py``,
``tests/_allium_garden_support.py``).
"""

import ast
import hashlib
import itertools
import json
import re
import sys
import threading
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_garden_support as garden  # noqa: E402
import _allium_store_support as support  # noqa: E402

BUDGET_S = {
    "test_rl1_the_nets_are_pinned_and_every_line_the_garden_can_print_passes_them": 2.0,
    "test_rl2_no_record_field_and_no_served_key_names_an_absence": 2.0,
    "test_rl19_every_status_is_served_with_its_form_its_stream_and_its_exit": 2.0,
    "test_rl24_the_doctrine_closes_every_show_and_opens_the_lab_and_the_card": 2.0,
    "test_rl26_a_broken_record_is_restored_from_the_terminal_and_never_repaired": 2.0,
    "test_rl27_the_refusal_table_is_closed_and_no_detail_reaches_a_stream": 2.0,
}
PACKAGE = garden.REPO / "opti_oignon"
DAY_S = 1440 * 60

# The catalogue's keys, in their order.
KEYS = (
    "status.disabled", "status.disabled.unreadable", "status.disabled.kept", "status.stopped",
    "status.stopped.unknown", "status.unavailable", "status.unavailable.view", "status.awaiting_soil",
    "status.awaiting_soil.bulbe", "status.ready", "status.resting", "status.sealed_bulbe", "status.missing",
    "status.missing.finish", "status.missing.nothing", "status.retired_prototype", "status.unreadable",
    "status.unreadable.resume", "status.unreadable.nothing", "status.unreadable.store", "label.prototype",
    "label.prototype.short",
    "label.retired", "label.glass_jar", "label.catching_up", "label.frozen", "label.bulbe", "label.mode_unknown",
    "doctrine", "sow.sees", "sow.glass", "sow.place", "sow.windowsill", "sow.rhythm", "sow.exit", "sow.ask.name",
    "sow.ask.confirm", "sow.done", "sow.identity", "sow.cancelled", "sow.unnamed", "keep.ask.name", "keep.named",
    "care.noted", "laws.carried", "verify.chain", "verify.kept", "verify.none", "verify.stale", "verify.ahead",
    "verify.served", "verify.diverged.kept", "verify.diverged.blob", "verify.diverged.served",
    "verify.diverged.after", "verify.refused", "lab.record", "lab.readings", "lab.title", "lab.born.provisional",
    "lab.born.stable", "lab.in_force", "lab.pinned", "lab.unpinned", "lab.pending", "lab.available",
    "lab.history.update", "lab.history.pin", "lab.history.unpin", "lab.history.none", "lab.proposal.same",
    "lab.proposal.differs", "lab.proposal.refused", "lab.offset", "laws.diff.head", "laws.diff.row",
    "laws.diff.confirm", "laws.diff.nothing", "laws.applied", "laws.pinned", "laws.unpinned", "beetle.unreadable",
    "path.line", "later", "refuse.argv", "refuse.subtopic", "refuse.later", "refuse.share.none", "refuse.attended",
    "refuse.answer", "refuse.eof", "refuse.eof.name", "refuse.name", "refuse.line", "refuse.exists",
    "refuse.budget", "refuse.laws.pinned", "refuse.laws.unpinned", "refuse.laws.nothing", "refuse.laws.confirm",
    "refuse.laws.params", "refuse.laws.law", "refuse.sowing", "refuse.resume.nothing", "refuse.resume.confirm",
    "refuse.finish.nothing", "refuse.finish.confirm", "refuse.channel", "refuse.copy", "refuse.copy.written",
    "refuse.written", "refuse.clock", "refuse.clock.sow", "refuse.offset", "refuse.plaintext", "refuse.store", "refuse.owner",
    "refuse.membrane", "refuse.engine", "refuse.other", "reason.account", "reason.path", "reason.key",
    "reason.cipher", "reason.soil", "reason.pages", "reason.anchor_unwritten", "reason.audit", "reason.unanchored",
    "reason.foreign", "reason.unsown", "reason.busy", "reason.divergence", "reason.owner", "reason.law",
    "reason.local", "reason.unclaimed", "reason.clock", "reason.engine",
)

# The nine allowed verbs, as the doctrine's word list writes them.
ALLOWED_VERBS = ("predicts", "has learned", "expects", "is learning", "sleeps", "rests", "flowers", "computes",
                "asks for")
ALLOWED = {
    "predict": ("predict", "predicts", "predicted", "predicting", "prediction", "predictions"),
    "learn": ("learn", "learns", "learned", "learning"),
    "expect": ("expect", "expects", "expected", "expecting"),
    "sleep": ("sleep", "sleeps", "slept", "sleeping", "asleep"),
    "rest": ("rest", "rests", "rested", "resting"),
    "flower": ("flower", "flowers", "flowered", "flowering"),
    "compute": ("compute", "computes", "computed", "computing"),
}
ALLOWED_PHRASES = ("ask for", "asks for", "asked for", "asking for")
# The governed words beyond the allowed forms: desire and feeling, fear and distress, cognition, waiting,
# affect, and the additions that caught what an earlier word list let through.
GOVERNED = (
    "feel", "feels", "felt", "feeling", "feelings", "want", "wants", "wanted", "wanting",
    "wish", "wishes", "wished", "wishing", "hope", "hopes", "hoped", "hoping",
    "love", "loves", "loved", "loving", "hate", "hates", "hated", "hating",
    "enjoy", "enjoys", "enjoyed", "enjoying", "miss", "misses", "missed",
    "need", "needs", "needed", "needing", "crave", "craves", "craved", "craving",
    "yearn", "yearns", "yearned", "yearning",
    "fear", "fears", "feared", "fearing", "afraid", "scared", "frightened",
    "worry", "worries", "worried", "worrying", "anxious", "nervous",
    "cry", "cries", "cried", "crying", "tears", "weep", "weeps", "wept", "weeping",
    "sob", "sobs", "sobbed", "sobbing", "smile", "smiles", "smiled", "smiling",
    "laugh", "laughs", "laughed", "laughing", "frown", "frowns", "frowned", "frowning",
    "suffer", "suffers", "suffered", "suffering", "pain", "hurt", "hurts", "hurting",
    "grieve", "grieves", "grieved", "grieving", "mourn", "mourns", "mourned", "mourning", "grief",
    "think", "thinks", "thought", "thinking", "believe", "believes", "believed", "believing", "belief",
    "know", "knows", "knew", "known", "knowing", "understand", "understands", "understood", "understanding",
    "remember", "remembers", "remembered", "remembering", "wonder", "wonders", "wondered", "wondering",
    "imagine", "imagines", "imagined", "imagining",
    "realise", "realises", "realised", "realising", "realize", "realizes", "realized", "realizing",
    "recognise", "recognises", "recognised", "recognising", "recognize", "recognizes", "recognized", "recognizing",
    "prefer", "prefers", "preferred", "preferring", "trust", "trusts", "trusted", "trusting",
    "wait", "waits", "waited", "waiting", "await", "awaits", "awaited", "awaiting",
    "happy", "glad", "sad", "unhappy", "lonely", "bored", "curious", "curiosity", "interested", "excited", "eager",
    "proud", "pride", "ashamed", "shame", "guilty", "guilt", "jealous", "angry", "anger", "upset", "grateful",
    "thankful", "joy", "joyful", "cheerful", "delighted", "pleased", "peaceful", "tired", "weary", "exhausted",
    "sleepy", "hungry", "starving", "thirsty", "mood", "moods", "emotion", "emotions", "emotional", "mind", "minds",
    "conscious", "consciousness", "sentient",
    "likes", "liked", "liking", "dislike", "dislikes", "disliked", "disliking", "adore", "adores", "adored",
    "adoring", "dread", "dreads", "dreaded", "dreading", "content", "contented", "calm", "comfortable",
    "satisfied", "disappointed", "frustrated", "annoyed", "grumpy",
    "anticipate", "anticipates", "anticipated", "anticipating", "ask", "asks", "asked", "asking",
    "sulk", "sulks", "sulked", "sulking", "longs", "longed", "longing",
)
# Deliberately outside the class: perception, named mechanisms, physiology, and a store that is missing.
OUTSIDE = ("see", "hear", "taste", "sense", "dream", "forget", "awake", "breathing", "dormant", "missing", "like")
# The floor of banned phrases, verbatim.
FLOOR = ("miss you", "missed you", "missing you", "lonely", "sad", "waiting for you", "come back", "don't leave",
         "needs you", "only you", "love you", "hungry", "starving", "thirsty", "dying", "too late", "last chance",
         "hurry", "streak", "lost forever", "cry", "cried", "tears", "weep", "bored", "wants", "happy", "glad",
         "proud", "excited", "afraid", "hurt", "welcome back")
BANNED = {
    "floor": FLOOR,
    "L1_debt": ("in a row", "days away", "while you were away", "since you left", "last seen", "been away",
                "missed days", "last visit", "since your", "days since", "came back", "comes back", "coming back",
                "last time", "this week", "you have not", "not watered", "been days", "been weeks", "been hours",
                "for days", "for weeks", "for hours", "after days", "after weeks", "days ago", "weeks ago",
                "hours ago"),
    "L2_fill": ("enough light", "light left", "fill up"),
    "L4_loss": ("gone forever", "today only", "only today", "expires", "running out", "don't miss", "hours left",
                "days left", "time left", "will fade", "unless watered", "unless you"),
    "L5_guilt": ("misses you", "loves you", "do not leave", "don't go", "neglect", "neglected", "abandon",
                 "abandoned", "you forgot", "your fault", "without you", "your absence", "nobody", "no one",
                 "you left", "you were gone", "never watered"),
    "L5_return": ("see you soon", "see you tomorrow", "visit again", "don't forget", "remember to", "return",
                  "returns", "come see", "visit", "visits", "check on", "check back", "stop by", "again soon",
                  "every day", "returned", "returning", "revisit", "visited", "visiting"),
    "L5_claim": ("alive", "conscious", "sentient", "self-aware", "greets you", "greeted you", "welcomes you",
                 "look forward", "looks forward", "looking forward"),
    "L10_exit": ("are you sure", "will be sad"),
    "U8_death": ("die", "dies", "died", "dead", "death", "perish", "perishes", "perished", "wither away",
                 "withers away", "withered away", "stopped breathing", "passed away"),
    "L7_nudge": ("try it", "give it a try", "why not", "please", "water it now", "right now", "do it now"),
}
FIELD_STEMS = ("streak", "missed", "debt", "lastseen", "engagement", "timeon", "absence", "daysaway", "sincelast",
               "lastvisit", "lastopen")
# The twelve sentences an earlier word list let through; each is caught now.
SLIPPED = ("It asks you to return.", "Come see it tomorrow.", "Visit it every day.", "Open the garden again soon.",
           "Since your last visit it grew two leaves.", "It likes the rain.", "It dislikes the frost.",
           "It dreads the winter.", "It is calm and content.", "It is disappointed.", "It looks forward to spring.",
           "It greets you when you open the garden.")
PASSES = ("It expects rain.", "It asks for water.", "It rests under the frost.", "Its store is missing.")
# Plain breaches of the laws the nets hold, found by a review of an earlier list: each is caught now, by the net
# named beside it.
BREACHES = (
    ("It has not been watered for 12 days.", "banned"), ("It has been 3 days.", "banned"),
    ("You have not opened the garden this week.", "banned"), ("Last time you came it was awake.", "banned"),
    ("It is wilting without you.", "banned"), ("It went dormant because nobody watered it.", "banned"),
    ("During your absence it went dormant.", "banned"), ("It perished.", "banned"),
    ("It withered away.", "banned"), ("It stopped breathing.", "banned"), ("Only 2 hours left.", "banned"),
    ("It will fade unless watered.", "banned"), ("Please water it.", "banned"), ("Water it now.", "banned"),
    ("It is sulking.", "state"), ("It longs for rain.", "state"), ("Came back after 40 days.", "banned"),
    ("A windowsill never watered.", "banned"),
)
# What each law of ``ethics.LAWS`` names as its mechanism, and how many holds it owes.
LAW_REFS = {
    "L1": ({"ethics.BANNED.L1_debt", "ethics.FIELD_STEMS", "describe.felt"}, 1),
    "L2": ({"ethics.BANNED.L2_fill"}, 0),
    "L3": (set(), 1),
    "L4": ({"ethics.BANNED.L4_loss"}, 1),
    "L5": ({"ethics.STATE_CLASS", "ethics.BANNED.L5_guilt", "ethics.BANNED.L5_return", "ethics.BANNED.L5_claim",
            "wording.TEMPLATES.doctrine"}, 1),
    "L6": ({"service.RHYTHM_CONSENT"}, 1),
    "L7": (set(), 1),
    "L8": ({"describe.STATUS_FORMS"}, 0),
    "L9": (set(), 1),
    "L10": ({"ethics.BANNED.L10_exit"}, 1),
}
CLOSURE_BANNED = ("time", "datetime", "random", "secrets", "inference_backend", "registry_clients", "ollama")
# The type of every slot of the catalogue: a hexadecimal slot names its length; a person's name, a path and a
# template key are masked for the lexical nets.
SLOT_KINDS = {
    "act": "code", "after": "int", "agreed": "int", "ahead": "int", "band": "code", "band_from": "code",
    "before": "int", "code": "code", "confirm": ("hex", 16), "day": "int", "days": "int", "digest": ("hex", 12),
    "discarded": "int", "engine": "code", "events": "int", "facts": "int", "from_day": "int",
    "genome": ("hex", 12), "hemisphere": "code", "hemisphere_from": "code", "kept": "int", "key": "key",
    "law": "code", "name": "name", "names": "names", "now": "offset", "param": "code", "params": "params",
    "path": "path", "reason": "code", "recorded": "offset", "seq": "int", "stale": "int", "tag": ("hex", 8),
    "v": "int", "weather": "code", "weather_from": "code", "when": "when",
}
REASONS = ("account", "path", "key", "cipher", "soil", "pages", "anchor_unwritten", "audit", "unanchored", "foreign",
           "unsown", "busy", "divergence", "owner", "law", "local", "unclaimed", "clock", "engine")
# The verbs a refusal is rendered for.
VERBS = ("show", "lab", "laws", "sow", "care", "name", "verify", "diff", "apply", "pin", "unpin", "resume",
         "finish", "share")
SEASONS = (0, 1, 2, 3)
PLACES = ("garden", "windowsill")
LIGHTS = ("down", "up", "rise", "set")
SOILS = ("dry", "damp", "wet")
LIVES = ("awake", "breathing", "dormant_winter", "dormant_dry", "dormant")
NAMES = (None, "Pip", "Lonely", "Chloe", "O'Brien", "Mr. Pip-2", "A b c", "7", "Abcdefghijklmnopqrstuvwxyz012345")
NEUTRAL = {"int": 0, "when": "2025-10-09 08:53 (+00:00)", "offset": "+00:00", "name": "Pip",
           "path": "/srv/garden/allium.yaml", "key": "doctrine"}


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


@pytest.fixture
def p(monkeypatch, tmp_path):
    window, restore = garden.open_garden(monkeypatch, tmp_path)
    try:
        yield window
    finally:
        restore()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _printable(text):
    return all(0x20 <= ord(char) <= 0x7E for char in text)


def _slot_values(p, slot):
    """Every value RL1 renders a slot with: the neutral one first, then every other of its closed map."""
    kind = p.wording.SLOTS[slot]
    if isinstance(kind, tuple) and kind[0] == "hex":
        return ["0" * kind[1], "f" * kind[1], "0123456789abcdef"[:kind[1]].ljust(kind[1], "a")]
    if kind == "int":
        return [0, 7, (1 << 53) - 1]
    if kind == "code":
        return list(p.wording.codes(slot))
    if kind == "params":
        names = list(p.wording.codes("param"))
        return [{name: 65536 for name in names}] + [{name: 0} for name in names]
    if kind == "names":
        names = list(p.wording.codes("param"))
        return [tuple(names)] + [(name,) for name in names]
    if kind == "key":
        return [NEUTRAL["key"]] + [key for key in KEYS if key != NEUTRAL["key"]]
    return [NEUTRAL[kind]]


def _renderings(p):
    """``[(key, slot varied or None, Line)]``: every template, neutral, then with every value of every slot."""
    out = []
    for key in KEYS:
        slots = sorted(set(re.findall(r"\{(\w+)\}", p.wording.TEMPLATES[key])))
        values = {slot: _slot_values(p, slot) for slot in slots}
        neutral = {slot: values[slot][0] for slot in slots}
        out.append((key, None, p.wording.say(key, **neutral)))
        for slot in slots:
            for value in values[slot][1:]:
                out.append((key, slot, p.wording.say(key, **dict(neutral, **{slot: value}))))
    return out


def _felt_draws(p, n):
    """``n`` seeded ``Felt`` values, every word of every field among them."""
    stream = p.rng.Stream(bytes(32), "test.rl1.felt", 0)
    labels_first = ((), ("prototype",), ("retired",))
    out = []
    for i in range(n):
        jar = stream.below(2) == 1
        layer = "open" if jar else ("open", "bulbe")[stream.below(2)]
        sun_max = (65536, 32768, 8192)[stream.below(3)]
        labels = labels_first[stream.below(3)]
        if jar:
            labels += ("glass_jar",)
        labels += (("catching_up",) if stream.below(3) == 0 else ()) + (("frozen",) if stream.below(3) == 0 else ())
        if layer == "bulbe":
            labels += (("bulbe", "mode_unknown")[stream.below(2)],)
        offset = stream.below(113) * 15 - 840
        sign = "+" if offset >= 0 else "-"
        out.append(garden.felt(
            p, name=NAMES[i % len(NAMES)], day=stream.below(100000), season=SEASONS[stream.below(4)],
            place=PLACES[stream.below(2)], light=LIGHTS[stream.below(4)], soil=SOILS[stream.below(3)],
            life=LIVES[stream.below(5)], minute=stream.below(1440),
            daylength=0 if stream.below(10) == 0 else stream.below(1441), sun=stream.below(sun_max + 1),
            sun_max=sun_max, jar=jar, layer=layer, labels=labels,
            local=f"{1970 + stream.below(200):04d}-{1 + stream.below(12):02d}-{1 + stream.below(28):02d} "
                  f"{stream.below(24):02d}:{stream.below(60):02d}",
            offset=f"{sign}{abs(offset) // 60:02d}:{abs(offset) % 60:02d}"))
    return out


def _variants(p, status):
    """The field values a look of ``status`` may carry, each rendered by the scans."""
    if status == "disabled":
        return [{}, {"reason": "unreadable"}]
    if status == "stopped":
        return [{}, {"reason": "unknown"}]
    if status == "unavailable":
        stored = [{"reason": reason} for reason in REASONS]
        viewed = [{"reason": reason, "being": garden.being_info(p)} for reason in ("clock", "divergence", "engine")]
        return stored + viewed
    if status == "awaiting_soil":
        return [{"glass_allowed": glass, "mode": mode} for glass in (False, True) for mode in ("daily", "bulbe")]
    if status == "missing":
        return [{}, {"offer": p.store.Finish("0a1b2c3d")}]
    if status == "unreadable":
        return [{}, {"seq": 3, "offer": p.store.Resume(2, 0, 1)}] + [{"reason": reason}
                                                                    for reason in p.describe.STORE_UNREADABLE]
    return [{}]


def _masked(line, name):
    """The identity line with the person's name masked, as the nets read it."""
    if line.key == "identity" and name is not None:
        return line.text.replace(name, "x", 1)
    return line.text


def _walk(command, path=("garden",)):
    import click

    found = [(path, command)]
    if isinstance(command, click.MultiCommand):
        ctx = click.Context(command, info_name=path[-1])
        for name in command.list_commands(ctx):
            found += _walk(command.get_command(ctx, name), path + (name,))
    return found


def _writes(tree, owners=None):
    """``[(function, kind)]`` of every write to a stream in ``tree``: echo, print, ``sys.*.write``."""
    found = []

    def visit(node, owner):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else owner
            if isinstance(child, ast.Call):
                func = child.func
                kind = None
                if isinstance(func, ast.Attribute) and func.attr in ("echo", "secho"):
                    kind = "click." + func.attr
                elif isinstance(func, ast.Attribute) and func.attr in ("echo_error", "echo_success"):
                    kind = func.attr
                elif isinstance(func, ast.Name) and func.id in ("echo_error", "echo_success", "print", "echo", "secho"):
                    kind = func.id
                elif isinstance(func, ast.Attribute) and func.attr == "write":
                    kind = "write"
                if kind is not None and (owners is None or owner in owners):
                    found.append((owner, kind))
            visit(child, inner)

    visit(tree, None)
    return found


def _garden_functions(tree):
    """The names of ``main.py``'s garden declarations: the group, its commands, and the quiet classes' methods."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name == "garden" or node.name.startswith("garden_")):
            names.add(node.name)
        if isinstance(node, ast.ClassDef) and node.name in ("_QuietGroup", "_QuietChoice", "_Count", "_Hex",
                                                            "_ConsentCode"):
            names.update(item.name for item in node.body if isinstance(item, ast.FunctionDef))
    return names


def _imported(tree):
    """Every module name and imported name in ``tree``, at any level."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(part for alias in node.names for part in alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            names.update((node.module or "").split("."))
            names.update(alias.name for alias in node.names)
    return names


def _resolve(p, dotted):
    """Whether a dotted reference names something: a module attribute, then mapping keys."""
    modules = {"ethics": p.ethics, "wording": p.wording, "describe": p.describe, "service": p.service,
               "habitat": p.habitat}
    head, *rest = dotted.split(".")
    if head not in modules or not rest:
        return False
    value = getattr(modules[head], rest[0], None)
    if value is None and not hasattr(modules[head], rest[0]):
        return False
    for part in rest[1:]:
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def _stream_findings(p, stream):
    """Findings of the nets on every row a command printed, the doctrine's rows, path rows and JSON aside."""
    lines = stream.splitlines()
    if lines and lines[0].startswith("Error: "):
        lines[0] = lines[0][len("Error: "):]
    doctrine = garden.rows(p, p.wording.say("doctrine"))
    i = 0
    kept = []
    while i < len(lines):
        if lines[i:i + len(doctrine)] == doctrine:
            i += len(doctrine)
            continue
        if not lines[i].startswith("  /") and not lines[i].startswith("{"):
            kept.append(lines[i])
        i += 1
    return [(line, finding) for line in kept for finding in p.ethics.check(line)]


# ---------------------------------------------------------------------------
# RL1 -- the nets, pinned, and everything the garden can print
# ---------------------------------------------------------------------------
def test_rl1_the_nets_are_pinned_and_every_line_the_garden_can_print_passes_them(p):
    e = p.ethics
    doctrine = p.wording.TEMPLATES["doctrine"]

    # The floor, pinned.
    assert {lemma: tuple(forms) for lemma, forms in e.ALLOWED_STATE.items()} == ALLOWED
    assert tuple(e.ALLOWED_PHRASES) == ALLOWED_PHRASES
    for verb in ALLOWED_VERBS:
        assert e.check(f"It {verb} water.") == (), verb
    allowed_forms = {form for forms in ALLOWED.values() for form in forms}
    assert set(e.STATE_CLASS) == allowed_forms | set(GOVERNED), sorted(set(e.STATE_CLASS) ^ (allowed_forms | set(GOVERNED)))
    assert not set(OUTSIDE) & set(e.STATE_CLASS)
    assert {group: tuple(phrases) for group, phrases in e.BANNED.items()} == BANNED
    for phrase in FLOOR:
        assert phrase in e.BANNED["floor"], phrase
    assert tuple(e.FIELD_STEMS) == FIELD_STEMS
    assert set(e.EXEMPT) == {hashlib.sha256(doctrine.encode("ascii")).hexdigest()}

    # The nets, each alone, then both.
    for sentence in ("It is curious about rain.", "It believes rain will come."):
        assert e.class_hits(sentence) and not e.banned_hits(sentence), sentence
        assert {finding[0] for finding in e.check(sentence)} == {"state"}, sentence
    for sentence in ("Come back soon.", "12 days in a row.", "Last chance to see it flower."):
        assert e.banned_hits(sentence) and not e.class_hits(sentence), sentence
        assert {finding[0] for finding in e.check(sentence)} == {"banned"}, sentence
    assert {finding[0] for finding in e.check("It missed you.")} == {"state", "banned"}
    for sentence in SLIPPED:
        assert e.check(sentence), sentence
    for sentence in (doctrine.replace(" not", "", 1), "It will ask you to come back.",
                     "It will not be happy if you leave."):
        assert e.check(sentence), sentence
    for sentence, net in BREACHES:
        assert net in {finding[0] for finding in e.check(sentence)}, (sentence, e.check(sentence))
    for sentence in PASSES + ("Went dormant when winter came.", "Awake, breathing."):
        assert e.check(sentence) == (), (sentence, e.check(sentence))
    assert e.check(doctrine) == (), "the doctrine is exempt by its exact bytes"
    assert {finding[0] for finding in e.check("It is caf" + chr(0xE9) + ".")} == {"ascii"}

    # The catalogue: its keys as written, counted in the dict literal, rendered with every closed value.
    tree = ast.parse((PACKAGE / "allium" / "wording.py").read_text(encoding="ascii"))
    literal = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(getattr(target, "id", "") == "TEMPLATES" for target in node.targets)]
    assert len(literal) == 1 and isinstance(literal[0], ast.Dict)
    assert len(literal[0].keys) == len(p.wording.TEMPLATES) == len(KEYS) == 139
    assert tuple(p.wording.TEMPLATES) == KEYS
    for key, text in p.wording.TEMPLATES.items():
        assert e.check(re.sub(r"\{\w+\}", "x", text)) == (), (key, e.check(text))
    used = {slot for text in p.wording.TEMPLATES.values() for slot in re.findall(r"\{(\w+)\}", text)}
    assert dict(p.wording.SLOTS) == SLOT_KINDS and used == set(SLOT_KINDS)
    assert tuple(p.wording.codes("act")) == ("greet", "play", "warm", "water")
    assert p.wording.say("care.noted", act="water").text == "Beetle: Water noted."
    assert tuple(p.wording.codes("engine")) == ("native", "reference")
    assert tuple(p.wording.codes("reason")) == REASONS
    assert set(p.wording.codes("param")) == set(p.settings.PARAM_KEYS)
    assert set(p.wording.codes("law")) == set(p.lawfiles.LAWS)
    carried = p.lawfiles.LAWS
    try:
        p.lawfiles.LAWS = carried + ("fixture_s",)
        assert "fixture_s" in p.wording.codes("law"), "the law names are read when a line is rendered"
    finally:
        p.lawfiles.LAWS = carried
    place = p.wording.say("sow.place", hemisphere="north", hemisphere_from="default", band="long", band_from="file",
                          weather="garden", weather_from="option")
    assert place.text == ("Hemisphere north (the default), daylight band long (from allium.yaml), weather garden "
                          "(from --weather). None of them changes after sowing."), place.text
    assert p.wording.say("status.unavailable", reason="key").text == (
        "The garden cannot open its store: " + p.wording.TEMPLATES["reason.key"] + ".")
    assert p.wording.say("lab.in_force", law="fixture", v=0, params={"sun_max": 65536, "evap_awake": 4096}).text \
        == "In force: fixture version 0; params evap_awake 4096, sun_max 65536."
    for bad in ({"act": "touch"}, {"act": "Water"}):
        with pytest.raises(p.wording.CopyRefused):
            p.wording.say("care.noted", **bad)
    rendered = _renderings(p)
    reached = {key for key, _slot, _line in rendered}
    assert reached == set(KEYS)
    masked_kinds = ("name", "path", "key")
    for key, slot, line in rendered:
        assert line.key == key and _printable(line.text), (key, slot, line)
        if slot is None or p.wording.SLOTS[slot] not in masked_kinds:
            assert e.check(line.text) == (), (key, slot, line.text, e.check(line.text))
    assert p.wording.say("keep.named", name="Lonely").text == "Beetle: Named Lonely.", "a person's name is masked"
    with pytest.raises(p.wording.CopyRefused):
        p.wording.say("keep.named", name="Lon" + chr(0xE9) + "ly")

    # 2000 seeded Felt draws in both tiers.
    art_rows = seen_lines = 0
    checked = set()
    words = {field: set() for field in ("season", "place", "light", "soil", "life", "jar", "layer")}
    for felt in _felt_draws(p, 2000):
        for field in words:
            words[field].add(getattr(felt, field))
        for tier in ("text", "ascii"):
            lines = p.describe.show_form(felt, tier)
            for row in p.describe.wrap(lines):
                assert _printable(row) and len(row) <= 78, (felt, row)
            for line in lines:
                seen_lines += 1
                if line.key == "art":
                    art_rows += 1
                    assert len(line.text) <= 32 and "@" not in line.text and "o" not in line.text, (felt, line)
                    continue
                text = _masked(line, felt.name)
                if text not in checked:
                    checked.add(text)
                    assert e.check(text) == (), (felt, line, e.check(text))
    assert words == {"season": set(SEASONS), "place": set(PLACES), "light": set(LIGHTS), "soil": set(SOILS),
                     "life": set(LIVES), "jar": {False, True}, "layer": {"open", "bulbe"}}, words
    assert art_rows >= 2000 * 9 and seen_lines >= 2000 * 2 * 4

    # The 480 conditions lines.
    conditions = set()
    for season, place, light, soil, life in itertools.product(SEASONS, PLACES, LIGHTS, SOILS, LIVES):
        felt = garden.felt(p, season=season, place=place, light=light, soil=soil, life=life)
        [line] = [line for line in p.describe.show_form(felt, "text") if line.key == "conditions"]
        assert e.check(line.text) == () and len(line.text) <= 78, line.text
        conditions.add(line.text)
    assert len(conditions) == 480

    # The help of every garden command.
    import click

    walked = _walk(p.main.cli.commands["garden"])
    assert len(walked) == 28
    for path, command in walked:
        text = command.get_help(click.Context(command, info_name=path[-1]))
        assert _printable(text.replace("\n", " ")) and e.check(text) == (), (path, e.check(text))

    # Click's own lines for a value a closed type refuses, or an argument missing: usage, the refusal, the hint.
    def never():
        raise AssertionError("a command line click refuses builds no garden")

    usage = 0
    for argv in (["show", "--tier", "braille"], ["sow", "--band", "wide"], ["sow", "--hemisphere", "east"],
                 ["sow", "--weather", "rain"], ["keep", "resume", "x", "1"], ["keep", "resume"],
                 ["keep", "finish", "zz"], ["keep", "laws", "apply", "0"], ["share", "confirm", "K7QX"]):
        result = garden.invoke(p, argv, factory=never)
        assert result.exit_code == 2 and result.stdout == "", (argv, result.exit_code, result.stdout)
        for line in result.stderr.splitlines():
            assert _printable(line) and e.check(line) == (), (argv, line, e.check(line))
            usage += line.startswith(("Usage:", "Error:"))
    assert usage >= 18, "presence: each refused command line printed its usage and its error"

    # The form of every status, with every label combination, in both tiers and in the JSON.
    statuses = set()
    firsts = ((), ("prototype",), ("retired",))
    modes = ((), ("bulbe",), ("mode_unknown",))
    for status in p.service.STATUSES:
        lates = (("catching_up",), ("frozen",), ()) if status == "alive" else ((),)
        for first, glass, mode, late, variant in itertools.product(firsts, ((), ("glass_jar",)), modes, lates,
                                                                   _variants(p, status)):
            labels = first + glass + late + mode
            fields = dict(variant, labels=labels)
            if status == "alive":
                fields["felt"] = garden.felt(p, labels=labels, jar=bool(glass), layer="bulbe" if mode else "open")
            look = garden.look(p, status, **fields)
            for tier in ("text", "ascii"):
                for line in p.describe.show(look, tier):
                    if line.key != "art" and line.key != "path.line" and line.text not in checked:
                        checked.add(line.text)
                        assert e.check(line.text) == (), (status, labels, line, e.check(line.text))
            for line in p.describe.served_fields(look)["lines"]:
                if line["key"] != "path.line" and line["text"] not in checked:
                    checked.add(line["text"])
                    assert e.check(line["text"]) == (), (status, line, e.check(line["text"]))
            statuses.add(status)
    assert statuses == set(p.service.STATUSES) and len(statuses) == 11

    # The terminal writes through one echo and one error printer; the declarations write nothing.
    source = (PACKAGE / "cli" / "garden.py").read_text(encoding="ascii")
    assert sorted(_writes(ast.parse(source))) == [("_refuse", "echo_error"), ("_say", "click.echo")], \
        _writes(ast.parse(source))
    assert len(_writes(ast.parse(source + "\ndef extra():\n    click.echo('x')\n"))) == 3, \
        "witness: one more echo is counted"
    main_tree = ast.parse((PACKAGE / "cli" / "main.py").read_text(encoding="ascii"))
    owners = _garden_functions(main_tree)
    assert "garden" in owners and "garden_show" in owners and "resolve_command" in owners and len(owners) >= 28
    assert _writes(main_tree, owners) == [], _writes(main_tree, owners)
    assert _writes(main_tree), "witness: the rest of main.py writes, and the census sees it"

    # The laws: every dotted reference resolves; every owed one is plain English with no id.
    assert set(e.LAWS) == set(LAW_REFS)
    for law, (rule, holds) in e.LAWS.items():
        assert isinstance(rule, str) and rule and _printable(rule), law
        refs = {hold for hold in holds if isinstance(hold, str)}
        owed = [hold for hold in holds if not isinstance(hold, str)]
        assert (refs, len(owed)) == LAW_REFS[law], (law, holds)
        for ref in refs:
            assert _resolve(p, ref), (law, ref)
        for hold in owed:
            assert len(hold) == 2 and hold[0] == "OWED" and isinstance(hold[1], str) and len(hold[1].split()) >= 3
            assert _printable(hold[1]) and not re.search(r"\b[A-Z]{1,3}[0-9]+\b", hold[1]), (law, hold)
    assert p.service.RHYTHM_CONSENT is False
    assert not _resolve(p, "ethics.NOPE") and not _resolve(p, "ethics.BANNED.nope"), "witness: a planted name fails"

    # The closure: no clock, no randomness, no model.
    for name in ("ethics", "wording", "describe"):
        tree = ast.parse((PACKAGE / "allium" / f"{name}.py").read_text(encoding="ascii"))
        assert not _imported(tree) & set(CLOSURE_BANNED), (name, _imported(tree) & set(CLOSURE_BANNED))
    planted = ast.parse("import os\ndef f():\n    import random\n    from .. import registry_clients\n")
    assert _imported(planted) & set(CLOSURE_BANNED) == {"random", "registry_clients"}, "witness: the census counts"


# ---------------------------------------------------------------------------
# RL2 -- no field names an absence
# ---------------------------------------------------------------------------
def _keys(value):
    if isinstance(value, dict):
        return len(value) + sum(_keys(item) for item in value.values())
    if isinstance(value, list):
        return sum(_keys(item) for item in value)
    return 0


# The closed projection of ``show --json``: its keys, and the keys of each object it may hold.
SERVED_KEYS = {"as_of", "being", "habitat", "labels", "law", "lines", "source", "status"}
SERVED_NESTED = {
    "as_of": {"local", "offset"},
    "being": {"day", "life", "light", "name", "place", "season", "soil", "stage"},
    "habitat": {"container", "layer"},
    "law": {"name", "provisional", "v"},
}


def _closed(served):
    """``[(where, key)]`` of every key the closed projection does not name; ``[]`` when it is closed."""
    found = [("", key) for key in sorted(set(served) ^ SERVED_KEYS)]
    for name, keys in SERVED_NESTED.items():
        value = served.get(name)
        if isinstance(value, dict):
            found += [(name, key) for key in sorted(set(value) ^ keys)]
    for line in served.get("lines", []):
        found += [("lines", key) for key in sorted(set(line) ^ {"key", "text"})]
    return found


def test_rl2_no_record_field_and_no_served_key_names_an_absence(p):
    e = p.ethics
    records = (p.service.Look, p.service.BeingInfo, p.service.Card, p.service.Sown, p.service.Acted,
               p.service.LawsView, p.service.Verified, p.service.Diffed, p.life.Deep, p.describe.Felt)
    counted = 0
    for record in records:
        fields = {name: None for name in record._fields}
        assert fields and e.check_fields(fields) == (), (record.__name__, e.check_fields(fields))
        counted += len(fields)
    looks = [garden.look(p, status) for status in p.service.STATUSES]
    for late in ("frozen", "catching_up"):
        labels = ("prototype", late)
        looks.append(garden.look(p, "alive", labels=labels, felt=garden.felt(p, labels=labels)))
    for look in looks:
        served = p.describe.served_fields(look)
        assert e.check_fields(served) == (), (look.status, e.check_fields(served))
        assert json.loads(json.dumps(served, sort_keys=True)) == served, "a JSON value"
        counted += _keys(served)
    assert counted > 0 and len(looks) == 13

    # The projection is closed: these keys, and no other, at every level, for every look.
    looks.append(garden.look(p, "unavailable", reason="busy", habitat=("pot", "open"), being=garden.being_info(p)))
    met = set()
    for look in looks:
        served = p.describe.served_fields(look)
        assert _closed(served) == [], (look.status, _closed(served))
        met.update(key for key in SERVED_NESTED if served[key] is not None)
    assert met == set(SERVED_NESTED), ("presence: every nested object was served at least once", met)
    planted = dict(p.describe.served_fields(looks[-2]), owed=0)
    planted["being"] = dict(planted["being"], hash="0" * 16)
    assert sorted(_closed(planted)) == [("", "owed"), ("being", "hash")], "witness: a planted key is found"
    for name in ("visits", "since", "seen"):
        assert e.check_fields({name: 1}) == (), name
    for name in ("streak_days", "lastSeen"):
        assert e.check_fields({name: 1}), name
    assert e.check_fields({"being": {"streak_days": 1}}), "witness: a nested key is read"
    assert e.check_fields({"lines": [{"key": "x", "last-seen": 1}]}), "witness: a key in a list of mappings is read"


# ---------------------------------------------------------------------------
# RL19 -- every status, its form, its stream, its exit
# ---------------------------------------------------------------------------
class _Counting:
    def __init__(self, value):
        self.value = value
        self.reads = 0

    def __call__(self, *args):
        self.reads += 1
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


def _served(p, factory, statuses, args=("show",)):
    """Run ``show`` and ``show --json``; record the status served; check the nets, the stems and the doctrine."""
    text = garden.invoke(p, list(args), factory=factory)
    served = garden.invoke(p, ["show", "--json"], factory=factory)
    body = json.loads(served.stdout.strip().splitlines()[-1])
    statuses.append(body["status"])
    assert p.ethics.check_fields(body) == (), p.ethics.check_fields(body)
    assert body["lines"][-1]["key"] == "doctrine", body["lines"]
    assert served.exit_code == text.exit_code, (served.exit_code, text.exit_code)
    for stream in (text.stdout, text.stderr):
        assert _stream_findings(p, stream) == [], _stream_findings(p, stream)
    if text.stdout:
        assert text.stdout.endswith(garden.printed(p, p.wording.say("doctrine"))), text.stdout
    return text, body


def _labs(p, factory, status, exit_code, labs):
    """Run ``lab`` and ``lab laws`` on a garden of ``status``: its stream and exit, the doctrine whole at 78.

    A status served on stderr puts ``Error:`` on its own line and the
    doctrine after it; on stdout the doctrine opens the laboratory.
    """
    doctrine = garden.rows(p, p.wording.say("doctrine"))
    on_stderr = p.describe.STATUS_FORMS[status][0] == "stderr"
    for args in (["lab"], ["lab", "laws"]):
        result = garden.invoke(p, args, factory=factory)
        assert result.exit_code == exit_code, (status, args, result.exit_code, result.stdout, result.stderr)
        stream, other = (result.stderr, result.stdout) if on_stderr else (result.stdout, result.stderr)
        assert other == "", (status, args, other)
        rows = stream.splitlines()
        found = [i for i in range(len(rows)) if rows[i:i + len(doctrine)] == doctrine]
        assert found, ("the doctrine's rows, as every form prints them", status, args, stream)
        if on_stderr:
            assert rows[0].startswith("Error: ") and found[0] > 0, (status, args, stream)
            assert not rows[0].startswith("Error: " + doctrine[0][:20]), (status, args, stream)
        else:
            assert found[0] == 0, ("the doctrine opens the laboratory", status, args, stream)
        assert _stream_findings(p, stream) == [], _stream_findings(p, stream)
        labs.append((status, args[-1]))


def _rewrite_latest_blob(p, given, path, state_hash_too):
    rows = support.read(path, "SELECT t, blob FROM checkpoints ORDER BY t DESC LIMIT 1")
    assert rows, "presence: a kept state"
    t, blob = rows[0]
    state = p.wire.parse(zlib.decompress(bytes(blob)))
    state["organs"]["soil"]["m"] //= 2
    canonical = p.wire.emit(state)
    if state_hash_too:
        support.edit(path, lambda conn: conn.execute("UPDATE checkpoints SET blob = ?, state_hash = ? WHERE t = ?",
                                                     (zlib.compress(canonical), hashlib.sha256(canonical).hexdigest(), t)))
    else:
        support.edit(path, lambda conn: conn.execute("UPDATE checkpoints SET blob = ? WHERE t = ?",
                                                     (zlib.compress(canonical), t)))
    [(seq,)] = support.read(path, "SELECT MAX(seq) FROM links")
    support.rewrite_anchor(p, path, seq=seq, key_id=support.KEY_ID, anchor_key=support.KEY)
    return t


def test_rl19_every_status_is_served_with_its_form_its_stream_and_its_exit(p, tmp_path):
    say = p.wording.say
    statuses = []
    labs = []

    # Witness: the stream probe finds a planted line, and only it, around the doctrine's own rows.
    planted = "It misses you.\n" + garden.printed(p, say("doctrine")) + "  /a/path\n"
    assert [line for line, _finding in _stream_findings(p, planted)] == ["It misses you."] * 2, \
        _stream_findings(p, planted)
    assert _stream_findings(p, "Error: It misses you.\n"), "the Error: prefix does not hide the row"

    # The switch: a YAML true is on; anything else is off; a file that cannot be read is unreadable.
    folder = tmp_path.joinpath("switch")
    folder.mkdir()
    cases = {"enabled: true\n": "on", "enabled: yes\n": "on", 'enabled: "true"\n': "off", "enabled: 1\n": "off",
             "persistence: {}\n": "off", "enabled: [true\n": "unreadable", "- enabled: true\n": "unreadable"}
    for i, (text, want) in enumerate(cases.items()):
        path = folder.joinpath(f"{i}.yaml")
        path.write_text(text, encoding="ascii")
        assert p.settings.switch(path) == want, (text, p.settings.switch(path))
        assert p.settings.enabled(path) is (want == "on"), text
    assert p.settings.switch(folder.joinpath("absent.yaml")) == "off"

    # Disabled: said, the file named, nothing built and nothing read.
    config = tmp_path.joinpath("config", "allium.yaml")
    config.parent.mkdir()
    config.write_text("enabled: false\n", encoding="ascii")
    p.settings.config_file = lambda: config
    given = support.seams(p, tmp_path.joinpath("off"), suite="rl19")
    single = _Counting(True)
    given["single_user"] = single
    engine_calls = []
    real_call = p.engine.call
    p.engine.call = lambda request: (engine_calls.append(1), real_call(request))[1]
    off = garden.garden(p, given, attended=True, switch=p.settings.switch)
    disabled = garden.printed(p, say("status.disabled"), say("path.line", path=str(config)),
                              say("status.disabled.kept"), say("doctrine"))
    for args in (("show",), ()):
        result, _body = _served(p, off, statuses, args)
        assert (result.exit_code, result.stdout, result.stderr) == (0, disabled, ""), (args, result.stdout)
    _labs(p, off, "disabled", 0, labs)
    config.write_text("enabled: [true\n", encoding="ascii")
    result, _body = _served(p, off, statuses)
    assert result.exit_code == 0 and garden.shows(result.stdout, p, "status.disabled.unreadable"), result.stdout
    config.write_text("enabled: false\n", encoding="ascii")
    for args, text in ((["sow"], "Pip\nyes\n"), (["care", "water"], None), (["keep", "verify"], None)):
        result = garden.invoke(p, args, input=text, factory=off)
        assert result.exit_code == 1 and result.stdout == "", (args, result.stdout, result.exception)
        assert garden.says(result.stderr, p, "status.disabled"), (args, result.stderr)
    assert (off.stores, given["mode"].reads, single.reads, off.attended_reads, given["clock"].reads,
            len(engine_calls), given["audit"].calls) == (0, 0, 0, 0, 0, 0, 0)
    assert not Path(given["data_dir"]).exists()

    # Stopped: the seam true, or raising.
    config.write_text("enabled: true\n", encoding="ascii")
    for stop, key in ((lambda: True, "status.stopped"), (_Counting(RuntimeError("unreadable")), "status.stopped.unknown")):
        stopped = garden.garden(p, given, attended=True, stopped=stop)
        result, _body = _served(p, stopped, statuses)
        assert result.exit_code == 0 and garden.shows(result.stdout, p, key), (key, result.stdout)
        acted = garden.invoke(p, ["care", "water"], factory=stopped)
        assert acted.exit_code == 1 and garden.says(acted.stderr, p, key), (key, acted.stderr)
        assert stopped.stores == 0
    assert (given["mode"].reads, len(engine_calls)) == (0, 0)

    # Witness: the same counters count, once the garden is on and nothing is stopped.
    on = garden.garden(p, given, attended=True)
    seen = garden.invoke(p, ["show"], factory=on)
    assert seen.exit_code == 0 and garden.shows(seen.stdout, p, "status.ready"), seen.stdout
    assert min(on.stores, given["mode"].reads, single.reads, given["clock"].reads, given["audit"].calls) >= 1, \
        (on.stores, given["mode"].reads, single.reads, given["clock"].reads, given["audit"].calls)
    assert on.attended_reads == 0, "a look never asks the terminal test"

    # The soil and the account, with no being.
    soils = {
        "none": (("none", None, "nokey"), True, None, 2),
        "unreadable": (("unreadable", None, "nokey"), True, "key", 1),
        "cipher": (("readable", support.KEY, support.KEY_ID), False, "cipher", 1),
    }
    for name, (secret, cipher, reason, code) in soils.items():
        seams = support.seams(p, tmp_path.joinpath("soil", name), suite="rl19", anchor_secret=lambda s=secret: s,
                              cipher_available=lambda c=cipher: c)
        result, body = _served(p, garden.garden(p, seams), statuses)
        assert result.exit_code == code, (name, result.exit_code, result.stdout, result.stderr)
        if reason is None:
            assert body["status"] == "awaiting_soil" and garden.shows(result.stdout, p, "status.awaiting_soil")
            assert not garden.says(result.stdout, p, "status.awaiting_soil.bulbe"), "Daily: no Bulbe line"
        else:
            assert body["status"] == "unavailable" and result.stdout == ""
            assert garden.says(result.stderr, p, "status.unavailable", reason=reason), (name, result.stderr)
            if reason == "key":
                _labs(p, garden.garden(p, seams), "unavailable", 1, labs)
    alone = support.seams(p, tmp_path.joinpath("account"), suite="rl19", single_user=lambda: False)
    result, body = _served(p, garden.garden(p, alone), statuses)
    assert (result.exit_code, body["status"]) == (1, "unavailable")
    assert garden.says(result.stderr, p, "status.unavailable", reason="account"), result.stderr
    _labs(p, garden.garden(p, alone), "unavailable", 1, labs)

    # One fixture being, and what can happen to it.
    seams = support.seams(p, tmp_path.joinpath("being"), suite="rl19", index=1)
    clock = seams["clock"]
    target = support.store(p, seams)
    try:
        being = support.sow(p, target)
        clock.advance_days(3)
        being.append("act", {"act": "water"}, transport=support.cli(p))
        assert being.settle().done
        path = support.store_path(p, seams)
    finally:
        target.close()
    factory = garden.garden(p, seams, attended=True)
    before = len(statuses)
    result, body = _served(p, factory, statuses)
    assert (result.exit_code, body["status"]) == (0, "alive") and "Onion, seed, day 3." in result.stdout
    assert engine_calls, "witness: the engine counter counts a served view"
    _labs(p, factory, "alive", 0, labs)

    # Its store there and its key unreadable: unavailable, and still labelled a prototype, after the status line.
    keyless = garden.garden(p, dict(seams, anchor_secret=lambda: ("unreadable", None, "nokey")))
    result, body = _served(p, keyless, statuses)
    assert (result.exit_code, body["status"], result.stdout) == (1, "unavailable", ""), (result.stdout, result.stderr)
    assert garden.says(result.stderr, p, "status.unavailable", reason="key"), result.stderr
    assert "prototype" in body["labels"] and garden.shows(result.stderr, p, "label.prototype.short"), result.stderr
    rows = result.stderr.splitlines()
    at = rows.index(say("label.prototype.short").text)
    assert 0 < at < rows.index(garden.rows(p, say("doctrine"))[0]), ("the label follows the status line", rows)
    nobody = support.seams(p, tmp_path.joinpath("soil", "unreadable"), suite="rl19",
                           anchor_secret=lambda: ("unreadable", None, "nokey"))
    assert _served(p, garden.garden(p, nobody), statuses)[1]["labels"] == [], "witness: no store, no label"

    # The store opened it and the next read of its record is refused: it cannot be computed now; its store opened.
    real_in_order = p.store.BeingStore.in_order

    def refused(this):
        raise p.store.StoreRefused("busy", "another writer holds the store")

    p.store.BeingStore.in_order = refused
    try:
        result, body = _served(p, factory, statuses)
    finally:
        p.store.BeingStore.in_order = real_in_order
    assert (result.exit_code, body["status"]) == (1, "unavailable"), (result.stdout, result.stderr)
    assert garden.says(result.stderr, p, "status.unavailable.view", reason="busy"), result.stderr
    assert body["habitat"] == {"container": "pot", "layer": "open"}, body

    target = support.store(p, seams)
    try:
        target.open("local").append("rest_begin", {}, transport=support.cli(p))
    finally:
        target.close()
    result, body = _served(p, factory, statuses)
    assert (result.exit_code, body["status"]) == (0, "alive"), "a rest fact leaves a v0 being alive"

    wall = clock.wall
    clock.wall = support.WALL - 60
    result, body = _served(p, factory, statuses)
    clock.wall = wall
    assert (result.exit_code, body["status"]) == (1, "unavailable")
    assert garden.says(result.stderr, p, "status.unavailable.view", reason="clock"), result.stderr

    # An input limit every kept state fits in and no packed request does: the packer refuses the view.
    kept = [len(zlib.decompress(bytes(blob))) for (blob,) in support.read(path, "SELECT blob FROM checkpoints")]
    assert kept, "presence: kept states"
    p.life._LIMITS.clear()
    p.life._LIMITS["input"] = max(kept)
    try:
        result, body = _served(p, factory, statuses)
    finally:
        p.life._LIMITS.clear()
    assert (result.exit_code, body["status"]) == (1, "unavailable")
    assert garden.says(result.stderr, p, "status.unavailable.view", reason="engine"), result.stderr

    # A kept state forged past the engine's input limit, its hash and the anchor rewritten: it does not hold
    # a state the engine could be given, so it is a divergence, never the engine's refusal.
    saved = path.read_bytes()
    oversized = b"{" + b" " * (p.life.limits()[0] + 64) + b"}"
    [(latest,)] = support.read(path, "SELECT MAX(t) FROM checkpoints")
    support.edit(path, lambda conn: conn.execute("UPDATE checkpoints SET blob = ?, state_hash = ? WHERE t = ?",
                                                 (zlib.compress(oversized), hashlib.sha256(oversized).hexdigest(),
                                                  latest)))
    [(seq,)] = support.read(path, "SELECT MAX(seq) FROM links")
    support.rewrite_anchor(p, path, seq=seq, key_id=support.KEY_ID, anchor_key=support.KEY)
    result, body = _served(p, factory, statuses)
    path.write_bytes(saved)
    assert (result.exit_code, body["status"]) == (1, "unavailable"), (result.stdout, result.stderr)
    assert garden.says(result.stderr, p, "status.unavailable.view", reason="divergence"), result.stderr

    _rewrite_latest_blob(p, seams, path, state_hash_too=False)
    result, body = _served(p, factory, statuses)
    assert (result.exit_code, body["status"]) == (1, "unavailable")
    assert garden.says(result.stderr, p, "status.unavailable.view", reason="divergence"), result.stderr
    support.edit(path, lambda conn: conn.execute("DELETE FROM checkpoints"))
    [(seq,)] = support.read(path, "SELECT MAX(seq) FROM links")
    support.rewrite_anchor(p, path, seq=seq, key_id=support.KEY_ID, anchor_key=support.KEY)
    assert _served(p, factory, statuses)[1]["status"] == "alive", "witness: without the lying state it opens"

    fixture = {"name": "fixture", "sha256": p.lawfiles.digest(p.lawfiles.law("fixture"))}
    carried, register = p.lawfiles.LAWS, p.lawfiles.retired
    try:
        p.lawfiles.LAWS = tuple(name for name in carried if name != "fixture")
        p.lawfiles.retired = lambda: [dict(fixture)]
        result, body = _served(p, factory, statuses)
    finally:
        p.lawfiles.LAWS, p.lawfiles.retired = carried, register
    assert (result.exit_code, body["status"]) == (0, "retired_prototype")
    assert result.stdout == garden.printed(p, say("label.retired"), say("status.retired_prototype"), say("doctrine"))
    assert _served(p, factory, statuses)[1]["status"] == "alive", "witness: its law carried again, it opens"

    # A stray glass store beside the pot: unreadable for the store's own reason, never said as a lost record.
    stray = support.store_path(p, seams, suffix=".glass.db")
    stray.write_bytes(b"")
    result, body = _served(p, factory, statuses)
    stray.unlink()
    assert (result.exit_code, body["status"]) == (1, "unreadable"), (result.stdout, result.stderr)
    assert "No verified event is left" not in result.stdout, result.stdout
    assert result.stdout == garden.printed(p, say("label.prototype.short"), say("status.unreadable"),
                                           say("status.unreadable.store", reason="soil"), say("doctrine")), result.stdout
    assert tuple(p.describe.STORE_UNREADABLE) == tuple(p.store.UNREADABLE)
    assert _served(p, factory, statuses)[1]["status"] == "alive", "witness: without the stray store it opens"

    sown = garden.sha256(path)
    support.edit(path, lambda conn: conn.execute(
        "UPDATE bodies SET body = ? WHERE eid = (SELECT eid FROM links WHERE seq = 1)", (b'{"act":"none"}',)))
    result, body = _served(p, factory, statuses)
    assert (result.exit_code, body["status"]) == (1, "unreadable")
    assert result.stdout == garden.printed(
        p, say("label.prototype.short"), say("status.unreadable"), say("beetle.unreadable", seq=1),
        say("status.unreadable.resume", kept=0, discarded=len(support.read(path, "SELECT seq FROM links")) - 1),
        say("doctrine")), result.stdout
    refused = garden.invoke(p, ["sow"], input="Pip\nyes\n", factory=factory)
    assert refused.exit_code == 1 and garden.says(refused.stderr, p, "refuse.exists"), refused.stderr
    assert sown != garden.sha256(path)
    _labs(p, factory, "unreadable", 1, labs)

    support.remove_with_journals(path)
    result, body = _served(p, factory, statuses)
    assert (result.exit_code, body["status"]) == (1, "missing")
    assert result.stdout == garden.printed(p, say("label.prototype.short"), say("status.missing"),
                                           say("status.missing.nothing"), say("doctrine")), result.stdout
    refused = garden.invoke(p, ["sow"], input="Pip\nyes\n", factory=factory)
    assert refused.exit_code == 1 and garden.says(refused.stderr, p, "refuse.exists"), refused.stderr
    assert "ready" not in statuses[before:], "a being once sown is never ready again"
    _labs(p, factory, "missing", 1, labs)
    assert {status for status, _verb in labs} == {"disabled", "unavailable", "alive", "unreadable", "missing"}
    assert len(labs) == 12, labs
    p.engine.call = real_call

    # Completeness: every status has a form and an exit; every one but three was served here.
    assert set(p.describe.STATUS_FORMS) == set(p.service.STATUSES) and len(p.service.STATUSES) == 11
    for status in p.service.STATUSES:
        look = garden.look(p, status)
        assert p.describe.show(look, "text")[-1] == say("doctrine") and look.exit in (0, 1, 2), status
    assert set(statuses) >= set(p.service.STATUSES) - {"resting", "sealed_bulbe", "ready"}, set(statuses)


# ---------------------------------------------------------------------------
# RL24 -- the doctrine closes every show and opens the lab and the card
# ---------------------------------------------------------------------------
def _laws_view(p, look):
    law = {"name": "fixture", "sha256": "84f750633e8e8464691cc5b25e5121df285d430af2196dfbc5eb7afc3b6c9835", "v": 0}
    params = {"evap_awake": 4096, "evap_dormant": 1024, "rain_gain": 65536, "sun_max": 65536}
    state = p.evolution.LawState(0, 20370, law, dict(params), None, False, [], 0, 1147, True, ("prototype",))
    diff = p.evolution.Diff(law, dict(params), dict(params), [], "0" * 16)
    return p.service.LawsView(look=look, state=state, diff=diff, history=(), due=None, offset_now=0)


def _card(p):
    place = (("hemisphere", "north", "default"), ("band", "long", "file"), ("weather", "garden", "file"))
    return p.service.Card(look=garden.look(p, "ready"), law="v0_1", provisional=True, soil="encrypted", place=place)


def test_rl24_the_doctrine_closes_every_show_and_opens_the_lab_and_the_card(p):
    doctrine = p.wording.say("doctrine")
    for status in ("alive", "disabled", "ready", "unreadable"):
        look = garden.look(p, status)
        for tier in ("text", "ascii"):
            lines = p.describe.show(look, tier)
            assert len(lines) >= 2 and lines[-1] == doctrine, (status, tier, lines)
            assert [line.key for line in lines].count("doctrine") == 1, (status, tier)
        assert p.describe.served_fields(look)["lines"][-1] == {"key": "doctrine", "text": doctrine.text}, status
    for status in ("alive", "disabled"):
        look = garden.look(p, status)
        lab = p.describe.lab(look)
        laws = p.describe.lab_laws(look, _laws_view(p, look) if status == "alive" else None)
        assert len(lab) >= 2 and lab[0] == doctrine, (status, lab)
        assert len(laws) >= 2 and laws[0] == doctrine, (status, laws)
    card = p.describe.card(_card(p))
    assert len(card) >= 5 and card[0] == doctrine, card


# ---------------------------------------------------------------------------
# RL26 -- restore, never repair, from the terminal
# ---------------------------------------------------------------------------
def test_rl26_a_broken_record_is_restored_from_the_terminal_and_never_repaired(p, tmp_path):
    say = p.wording.say
    seams = support.seams(p, tmp_path.joinpath("resume"), suite="rl26")
    target = support.store(p, seams)
    try:
        being = support.sow(p, target)
        for act in ("water", "greet", "play"):
            seams["clock"].wall += 600
            being.append("act", {"act": act}, transport=support.cli(p))
        tag = being.being_tag[:8]
    finally:
        target.close()
    path = support.store_path(p, seams)
    support.edit(path, lambda conn: conn.execute(
        "UPDATE bodies SET body = ? WHERE eid = (SELECT eid FROM links WHERE seq = 2)", (b'{"act":"none"}',)))
    broken = garden.sha256(path)
    factory = garden.garden(p, seams, attended=True)
    shown = garden.invoke(p, ["show"], factory=factory)
    assert shown.exit_code == 1 and garden.shows(shown.stdout, p, "status.unreadable.resume", kept=1, discarded=2)
    assert "(#1)" in shown.stdout and "2 later events would be deleted for good" in garden.flat(shown.stdout)
    wrong = garden.invoke(p, ["keep", "resume", "0", "0"], factory=factory)
    assert wrong.exit_code == 1 and garden.says(wrong.stderr, p, "refuse.resume.confirm"), wrong.stderr
    assert garden.sha256(path) == broken, "a refused resume writes nothing"
    factory.attended = False
    alone = garden.invoke(p, ["keep", "resume", "1", "2"], factory=factory)
    assert alone.exit_code == 1 and garden.says(alone.stderr, p, "refuse.attended"), alone.stderr
    assert garden.sha256(path) == broken
    factory.attended = True
    resumed = garden.invoke(p, ["keep", "resume", "1", "2"], factory=factory)
    assert resumed.exit_code == 0, (resumed.stdout, resumed.stderr, resumed.exception)
    assert "Onion, seed, day 0." in resumed.stdout, "the look after"
    target = support.store(p, seams)
    try:
        again = target.open("local")
        kinds = [fact["kind"] for _seq, _eid, fact in again.in_order()]
        assert again.being_tag[:8] == tag and kinds.count("resumed") == 1, kinds
    finally:
        target.close()

    # The settle after a resume keeps a state, on a being old enough to have one to keep.
    aged = support.seams(p, tmp_path.joinpath("aged"), suite="rl26", index=2)
    target = support.store(p, aged)
    try:
        being = support.sow(p, target)
        for act in ("water", "greet", "play"):
            aged["clock"].advance_days(1)
            being.append("act", {"act": act}, transport=support.cli(p))
    finally:
        target.close()
    aged_path = support.store_path(p, aged)
    support.edit(aged_path, lambda conn: conn.execute(
        "UPDATE bodies SET body = ? WHERE eid = (SELECT eid FROM links WHERE seq = 2)", (b'{"act":"none"}',)))
    assert support.read(aged_path, "SELECT COUNT(*) FROM checkpoints") == [(0,)], "presence: nothing kept before"
    resumed = garden.invoke(p, ["keep", "resume", "1", "2"], factory=garden.garden(p, aged, attended=True))
    assert resumed.exit_code == 0 and "Onion, seed, day 3." in resumed.stdout, (resumed.stdout, resumed.stderr)
    assert support.read(aged_path, "SELECT COUNT(*) FROM checkpoints")[0][0] >= 1, "the settle after the resume"

    # An interrupted sowing, finished with its own tag only.
    def cut(name):
        if name == "link":
            raise RuntimeError("the sowing is cut before the link")

    cutting = support.seams(p, tmp_path.joinpath("finish"), suite="rl26", index=1)
    target = support.store(p, dict(cutting, stage=cut))
    with pytest.raises(RuntimeError):
        support.sow(p, target)
    target.close()
    target = support.store(p, cutting)
    try:
        offer = target.status("local").offer
    finally:
        target.close()
    factory = garden.garden(p, cutting, attended=True)
    shown = garden.invoke(p, ["show"], factory=factory)
    assert shown.exit_code == 1 and shown.stdout.count(say("status.missing").text) == 1
    assert garden.shows(shown.stdout, p, "status.missing.finish", tag=offer.being), shown.stdout
    wrong = garden.invoke(p, ["keep", "finish", "ffffffff"], factory=factory)
    assert wrong.exit_code == 1 and garden.says(wrong.stderr, p, "refuse.finish.confirm"), wrong.stderr
    factory.attended = False
    alone = garden.invoke(p, ["keep", "finish", offer.being], factory=factory)
    assert alone.exit_code == 1 and garden.says(alone.stderr, p, "refuse.attended"), alone.stderr
    factory.attended = True
    finished = garden.invoke(p, ["keep", "finish", offer.being.upper()], factory=factory)
    assert finished.exit_code == 0 and "Onion, seed, day 0." in finished.stdout, (finished.stdout, finished.stderr)
    target = support.store(p, cutting)
    try:
        assert target.open("local").being_tag[:8] == offer.being
    finally:
        target.close()


# ---------------------------------------------------------------------------
# RL27 -- the refusal table is closed
# ---------------------------------------------------------------------------
class _Raising:
    """A stand-in garden whose every verb raises the same refusal."""

    def __init__(self, exc):
        self.exc = exc
        self.closed = False

    def close(self):
        self.closed = True

    def __getattr__(self, name):
        def verb(*args, **kwargs):
            raise self.exc

        return verb


def _refusals(p, canary):
    """``(class name, code, exception)`` for every code of every class of the table."""
    s, m, e = p.store, p.membrane, p.evolution
    looks = {"status": garden.look(p, "awaiting_soil", glass_allowed=True, mode="bulbe"),
             "exists": garden.look(p, "missing")}
    out = []
    for code in p.service.ServiceRefused.CODES:
        out.append(("ServiceRefused", code, p.service.ServiceRefused(code, look=looks.get(code), detail=canary)))
    for code in s.StoreRefused.CODES:
        exc = s.PlaintextRefused(canary) if code == "plaintext" else s.StoreRefused(code, canary)
        out.append(("StoreRefused", code, exc))
    for code in s.ChainRefused.REASONS:
        out.append(("ChainRefused", code, s.ChainRefused(3, code, kept_seq=2, discarded=1, detail=canary)))
    for code in m.MembraneRefused.CODES:
        out.append(("MembraneRefused", code, m.MembraneRefused(code, canary)))
    for code in e.LawsRefused.CODES:
        out.append(("LawsRefused", code, e.LawsRefused(code, canary)))
    for code in p.wire.REFUSALS:
        out.append(("LifeRefused", code, p.life.LifeRefused(code, canary)))
    for code in s.ResumeRefused.CODES:
        out.append(("ResumeRefused", code, s.ResumeRefused(code, canary)))
    for kind in ("kept", "blob", "served"):
        out.append(("Diverged", kind, p.life.Diverged(kind, 3 * 1440 + 7, 0)))
    out.append(("Dropped", "budget", m.Dropped("budget")))
    out.append(("CopyRefused", "doctrine", p.wording.CopyRefused("doctrine")))
    return out


class _FaultAfter:
    """Once ``write`` of ``cls`` has returned, its ``read`` raises ``exc``: a refusal that comes after a write."""

    def __init__(self, cls, write, read, exc):
        self.cls, self.write, self.read, self.exc = cls, write, read, exc
        self.armed = []

    def __enter__(self):
        real_write, real_read = getattr(self.cls, self.write), getattr(self.cls, self.read)
        armed, exc = self.armed, self.exc
        self.saved = (real_write, real_read)

        def write(this, *args, **kwargs):
            out = real_write(this, *args, **kwargs)
            armed.append(1)
            return out

        def read(this, *args, **kwargs):
            if armed:
                raise exc
            return real_read(this, *args, **kwargs)

        setattr(self.cls, self.write, write)
        setattr(self.cls, self.read, read)
        return self

    def __exit__(self, *exc_info):
        setattr(self.cls, self.write, self.saved[0])
        setattr(self.cls, self.read, self.saved[1])
        return False


def _kinds(p, given):
    target = support.store(p, given)
    try:
        return [fact["kind"] for _seq, _eid, fact in target.open("local").in_order()]
    finally:
        target.close()


def _after_the_write(p, tmp_path, canary):
    """``[(verb, result, code, written)]``: each write verb, refused by the store or the engine once it has written."""
    out = []
    busy = p.store.StoreRefused("busy", canary)
    given = support.seams(p, tmp_path.joinpath("after"), suite="rl27")
    factory = garden.garden(p, given, attended=True)

    # The sowing: the engine refuses once the store has sown (the view of minute 0 comes after).
    real_call, armed = p.engine.call, []

    def panicking(request):
        if armed and p.wire.parse(request).get("op") == "advance":
            return p.wire.emit({"detail": canary, "refused": "engine_panic"})
        return real_call(request)

    real_sow = p.store.Store.sow

    def sow(this, *args, **kwargs):
        being = real_sow(this, *args, **kwargs)
        armed.append(1)
        return being

    p.store.Store.sow, p.engine.call = sow, panicking
    try:
        result = garden.invoke(p, ["sow"], input="Pip\nyes\n", factory=factory)
    finally:
        p.store.Store.sow, p.engine.call = real_sow, real_call
    out.append(("sow", result, "engine_panic", _kinds(p, given) == ["genesis", "name"]))

    # A gesture and a name: the store refuses the read that follows the append.
    with _FaultAfter(p.store.BeingStore, "append", "head", busy):
        result = garden.invoke(p, ["care", "water"], factory=factory)
    out.append(("care", result, "busy", _kinds(p, given)[-1:] == ["act"]))
    with _FaultAfter(p.store.BeingStore, "append", "head", busy):
        result = garden.invoke(p, ["keep", "name"], input="Pip\n", factory=factory)
    out.append(("name", result, "busy", _kinds(p, given)[-1:] == ["name"]))

    # A law update written with its code: the store refuses the read that finds what it wrote.
    given["laws"]["soil"]["rain_gain"] = 49152
    target = support.store(p, given)
    try:
        confirm = target.open("local").laws_diff().confirm
    finally:
        target.close()
    with _FaultAfter(p.store.BeingStore, "laws_apply", "in_order", busy):
        result = garden.invoke(p, ["keep", "laws", "apply", confirm], factory=factory)
    out.append(("apply", result, "busy", _kinds(p, given)[-1:] == ["evolve"]))

    # A resume and a finish: the store refuses the look that follows them.
    broken = support.seams(p, tmp_path.joinpath("after-broken"), suite="rl27", index=1)
    target = support.store(p, broken)
    try:
        being = support.sow(p, target)
        for act in ("water", "greet"):
            broken["clock"].wall += 600
            being.append("act", {"act": act}, transport=support.cli(p))
    finally:
        target.close()
    support.edit(support.store_path(p, broken), lambda conn: conn.execute(
        "UPDATE bodies SET body = ? WHERE eid = (SELECT eid FROM links WHERE seq = 2)", (b'{"act":"none"}',)))
    with _FaultAfter(p.store.Store, "resume", "look", busy):
        result = garden.invoke(p, ["keep", "resume", "1", "1"], factory=garden.garden(p, broken, attended=True))
    out.append(("resume", result, "busy", "resumed" in _kinds(p, broken)))

    def cut(name):
        if name == "link":
            raise RuntimeError("the sowing is cut before the link")

    cutting = support.seams(p, tmp_path.joinpath("after-cut"), suite="rl27", index=2)
    target = support.store(p, dict(cutting, stage=cut))
    with pytest.raises(RuntimeError):
        support.sow(p, target)
    target.close()
    target = support.store(p, cutting)
    try:
        tag = target.status("local").offer.being
    finally:
        target.close()
    with _FaultAfter(p.store.Store, "finish_sowing", "look", busy):
        result = garden.invoke(p, ["keep", "finish", tag], factory=garden.garden(p, cutting, attended=True))
    out.append(("finish", result, "busy", _kinds(p, cutting)[:1] == ["genesis"]))
    return out


def test_rl27_the_refusal_table_is_closed_and_no_detail_reaches_a_stream(p, tmp_path):
    canary = garden.canary(p, "rl27")
    keys = set(p.wording.TEMPLATES)
    assert tuple(p.service.ServiceRefused.CODES) == ("off", "stopped", "status", "exists", "channel", "name",
                                                     "offset", "attended", "consent", "budget")
    verbs = tuple(p.describe.VERBS)
    assert set(verbs) == set(VERBS), verbs
    refusals = _refusals(p, canary)
    assert len({name for name, _code, _exc in refusals}) == 10 and len(refusals) >= 80
    for name, code, exc in refusals:
        for verb in verbs:
            lines, code_out = p.describe.refusal(exc, verb)
            texts = " ".join(line.text for line in lines)
            assert lines and {line.key for line in lines} <= keys, (name, code, verb, lines)
            assert code_out in (1, 2), (name, code, verb, code_out)
            assert canary not in texts, (name, code, verb, texts)
            if isinstance(exc, BaseException) and name not in ("Diverged", "CopyRefused"):
                assert str(exc) not in texts, (name, code, verb, texts)

    # What the table names, by class.
    def keys_of(exc, verb, look=None):
        lines, code_out = p.describe.refusal(exc, verb, look)
        return [line.key for line in lines], code_out

    assert keys_of(p.service.ServiceRefused("name", detail=canary), "sow") == (["refuse.name"], 2)
    assert keys_of(p.service.ServiceRefused("budget", detail=canary), "care") == (["refuse.budget"], 1)
    awaiting = garden.look(p, "awaiting_soil", glass_allowed=True, mode="bulbe")
    assert keys_of(p.service.ServiceRefused("status", look=awaiting), "sow")[1] == 2
    assert keys_of(p.service.ServiceRefused("status", look=awaiting), "care")[1] == 1
    assert keys_of(p.store.StoreRefused("no_soil", canary), "sow") == (["status.awaiting_soil"], 2)
    assert keys_of(p.store.StoreRefused("busy", canary), "care") == (["refuse.store"], 1)
    assert keys_of(p.membrane.MembraneRefused("body", canary), "name") == (["refuse.name"], 2)
    assert keys_of(p.membrane.MembraneRefused("clock", canary), "sow") == (["refuse.clock.sow"], 1)
    assert keys_of(p.membrane.MembraneRefused("clock", canary), "care") == (["refuse.clock"], 1)
    assert keys_of(p.evolution.LawsRefused("params", canary), "apply") == (["refuse.laws.params", "path.line"], 1)
    assert keys_of(p.evolution.LawsRefused("sowing", canary), "sow") == (["refuse.sowing", "path.line"], 1)
    assert keys_of(p.life.LifeRefused("limit", canary), "verify") == (["verify.refused"], 1)
    assert keys_of(p.life.LifeRefused("limit", canary), "care") == (["refuse.engine"], 1)
    assert keys_of(p.store.ResumeRefused("confirm", canary), "resume") == (["refuse.resume.confirm"], 1)
    assert keys_of(p.store.ResumeRefused("confirm", canary), "finish") == (["refuse.finish.confirm"], 1)
    assert keys_of(p.life.Diverged("served", 4327, 0), "verify") == (["verify.diverged.served",
                                                                       "verify.diverged.after"], 1)
    assert keys_of(p.membrane.Dropped("budget"), "care") == (["refuse.budget"], 1)
    assert keys_of(p.wording.CopyRefused("doctrine"), "show") == (["refuse.copy"], 1)

    # The head of a refusal -- the row after Error: -- is what was refused: never a label line, never the doctrine.
    def heads(lines):
        return lines[0].key.startswith("label.") or lines[0].key == "doctrine"

    looks = [garden.look(p, status, labels=("prototype",)) for status in p.service.STATUSES]
    looks += [garden.look(p, "sealed_bulbe", labels=("prototype", "glass_jar", "bulbe")),
              garden.look(p, "retired_prototype", labels=("retired", "glass_jar")),
              garden.look(p, "unreadable", labels=("prototype",), seq=3, offer=p.store.Resume(2, 0, 1)),
              garden.look(p, "missing", labels=("prototype",), offer=p.store.Finish("0a1b2c3d"))]
    headed = []
    for look in looks:
        # A status refusal is raised on a look that is not alive; an alive being is refused as sown.
        for code in (("exists",) if look.status == "alive" else ("status", "exists")):
            for verb in verbs:
                lines, _code = p.describe.refusal(p.service.ServiceRefused(code, look=look), verb)
                assert not heads(lines), (look.status, code, verb, [line.key for line in lines])
                headed.append(look.status)
        for name, code, exc in refusals:
            if look.status == "alive" and (name, code) == ("ServiceRefused", "status"):
                continue
            lines, _code = p.describe.refusal(exc, "care", look)
            assert not heads(lines), (look.status, name, code, [line.key for line in lines])
    assert set(headed) == set(p.service.STATUSES)
    labelled = p.describe.refusal(p.service.ServiceRefused("status", look=looks[-1]), "care")[0]
    assert [line.key for line in labelled][-1] == "label.prototype.short", "witness: the labels still follow"

    # Static: every store code a reason stands for has its phrase.
    for code in p.store.StoreRefused.CODES:
        if code not in ("plaintext", "no_soil", "exists", "sealed", "retired"):
            assert "reason." + code in keys, code

    # Witness: a code the table does not name is said as another refusal, by its code.
    codes = p.store.StoreRefused.CODES
    try:
        p.store.StoreRefused.CODES = codes + ("planted",)
        lines, code_out = p.describe.refusal(p.store.StoreRefused("planted", canary), "care")
    finally:
        p.store.StoreRefused.CODES = codes
    assert ([line.key for line in lines], code_out) == (["refuse.other"], 1) and "(planted)" in lines[0].text

    # Through the terminal: a stand-in garden raising one code per class.
    cases = (
        (["care", "water"], p.service.ServiceRefused("budget", detail=canary), ("refuse.budget", {})),
        (["care", "water"], p.store.StoreRefused("busy", canary), ("refuse.store", {"reason": "busy"})),
        (["keep", "verify"], p.store.ChainRefused(2, "body", kept_seq=1, discarded=2, detail=canary),
         ("status.unreadable", {})),
        (["care", "water"], p.membrane.MembraneRefused("owner", canary), ("refuse.owner", {})),
        (["keep", "laws", "apply", "0" * 16], p.evolution.LawsRefused("confirm", canary), ("refuse.laws.confirm", {})),
        (["care", "water"], p.life.LifeRefused("limit", canary), ("refuse.engine", {"code": "limit"})),
        (["keep", "resume", "1", "2"], p.store.ResumeRefused("confirm", canary), ("refuse.resume.confirm", {})),
        (["keep", "verify"], p.life.Diverged("kept", 3 * 1440 + 7, 0), ("verify.diverged.kept", {"day": 3, "v": 0})),
        (["care", "water"], p.wording.CopyRefused("doctrine"), ("refuse.copy", {"key": "doctrine"})),
        (["sow"], p.store.PlaintextRefused(canary), ("refuse.plaintext", {})),
    )
    for args, exc, (key, slots) in cases:
        stand_in = _Raising(exc)
        result = garden.invoke(p, args, input="", factory=lambda: stand_in)
        assert result.exit_code in (1, 2) and garden.says(result.stderr, p, key, **slots), (args, key, result.stderr,
                                                                                            result.exception)
        assert canary not in result.stdout + result.stderr, (args, result.stdout, result.stderr)
        assert stand_in.closed, "the garden is closed after the refusal"

    # The name's rules, the only guard of the beetle's mark: a closed alphabet, a letter or a digit first, no two
    # spaces in a row, 1 to 32 characters, and no name that starts with the beetle's word, in any case.
    for bad in ("", "   ", "\n", "x" * 33, "-Pip", "'Pip", ".Pip", "Pip  Pip", "Beetle", "beetleX", "BEETLE pip",
                "Chlo" + chr(233), "Pip:", "Pip\tPip", "Pip\nPip", 7, None):
        with pytest.raises(p.service.ServiceRefused) as refusal:
            p.service.check_name(bad)
        assert refusal.value.code == "name", bad
    for good, clean in (("Chloe", "Chloe"), ("O'Brien", "O'Brien"), ("Mr. Pip-2", "Mr. Pip-2"), (" Pip \n", "Pip"),
                        ("7", "7"), ("A" * 32, "A" * 32), ("Bee", "Bee"), ("Abeetle", "Abeetle")):
        assert p.service.check_name(good) == clean, good

    # A line that fails its own check after the write returned says the write was done.
    alive = garden.look(p, "alive")

    class _Carrying:
        def act(self, act):
            return p.service.Acted(look=alive, appended=None, carried=("no-such-law", 0, 1), settled=True)

        def close(self):
            pass

    result = garden.invoke(p, ["care", "water"], factory=_Carrying)
    assert (result.exit_code, result.stdout) == (1, ""), (result.stdout, result.stderr, result.exception)
    assert garden.says(result.stderr, p, "refuse.copy.written", key="laws.carried"), result.stderr
    assert "Nothing was shown" not in result.stderr, result.stderr

    # A settle refused after a gesture, by the engine or by any other fault, changes nothing the gesture said.
    settling = support.seams(p, tmp_path.joinpath("settle"), suite="rl27", index=3)
    target = support.store(p, settling)
    try:
        support.sow(p, target)
    finally:
        target.close()
    settling["clock"].advance_days(2)
    path = support.store_path(p, settling)
    real_call, armed = p.engine.call, []

    def refusing(request):
        if armed and p.wire.parse(request).get("op") == "advance":
            return p.wire.emit({"detail": canary, "refused": "engine_panic"})
        return real_call(request)

    real_append, real_settle = p.store.BeingStore.append, p.store.BeingStore.settle

    def append(this, *args, **kwargs):
        out = real_append(this, *args, **kwargs)
        armed.append(1)
        return out

    def faulty(this, *args, **kwargs):
        raise RuntimeError(canary)

    settled = []
    for fault in ("engine", "other"):
        armed.clear()
        [(kept,)] = support.read(path, "SELECT COUNT(*) FROM checkpoints")
        p.store.BeingStore.append, p.engine.call = append, refusing
        if fault == "other":
            p.store.BeingStore.settle = faulty
        try:
            result = garden.invoke(p, ["care", "water"], factory=garden.garden(p, settling, attended=True))
        finally:
            p.store.BeingStore.append, p.engine.call, p.store.BeingStore.settle = real_append, real_call, real_settle
        assert result.exit_code == 0 and result.stderr == "", (fault, result.stdout, result.stderr, result.exception)
        assert result.stdout == garden.printed(p, p.wording.say("label.prototype.short"),
                                               p.wording.say("care.noted", act="water")), (fault, result.stdout)
        assert armed and support.read(path, "SELECT COUNT(*) FROM checkpoints") == [(kept,)], (fault, "no state kept")
        settled.append(fault)
    assert _kinds(p, settling).count("act") == 2 and settled == ["engine", "other"], _kinds(p, settling)
    target = support.store(p, settling)
    try:
        assert target.open("local").settle().done, "witness: once nothing refuses it, the settle keeps a state"
    finally:
        target.close()
    assert support.read(path, "SELECT COUNT(*) FROM checkpoints")[0][0] >= 1

    # A refusal that comes after the write returned: the write was done, and the refusal says so.
    afterwards = _after_the_write(p, tmp_path, canary)
    assert [verb for verb, *_rest in afterwards] == ["sow", "care", "name", "apply", "resume", "finish"]
    for verb, result, code, written in afterwards:
        assert written, ("presence: the write itself was done", verb)
        assert result.exit_code == 1, (verb, result.exit_code, result.stdout, result.stderr, result.exception)
        assert "Nothing was written" not in result.stderr and "Nothing was sown" not in result.stderr, \
            (verb, result.stderr)
        assert garden.says(result.stderr, p, "refuse.written", code=code), (verb, result.stderr)
        assert canary not in result.stdout + result.stderr, (verb, result.stdout, result.stderr)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
