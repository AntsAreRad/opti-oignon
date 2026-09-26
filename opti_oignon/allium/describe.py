"""The one description of a being: what a look shows, as lines of the catalogue.

``felt(view, ...)`` projects a served view onto what ``show`` may read, as a
``Felt``: the day of life, the season, the place, the light, the soil's
water against the law's thresholds, whether it is awake or breathing or
went dormant and why, the sky, the habitat and the local time. It never
reads a count, the soil's store of water, the chemistry, the clocks, the
genome or the reducer's bookkeeping: what an absence could be counted from
is not read at all.

Every form is a list of ``Line(key, text)``: the catalogue's lines
(``wording.say``) and three of the description's own -- ``art``, a row of
the drawing; ``identity``, who and which day; ``conditions``, the season,
the place, the light, the soil and the life. ``identity`` is checked by the
nets with the person's name masked, ``conditions`` in full, and the drawing
by its characters alone. Lines are unwrapped: ``wrap`` breaks them at 78
columns for the terminal, and leaves the drawing and a path whole.

The drawing (``art``) is 32 columns by 9 rows of printable ASCII: the sky
(its sun, or the sun down), a pot -- its lid closed under Bulbe's rules --
or a glass jar, the soil (wet, damp or dry, the first row frosted in
winter), the seed, eyes open when awake and closed when dormant, and the
ground of the garden or the windowsill. It holds no ``@`` (real spike
activity) and no ``o`` (an unlit core): v0 has neither. It reads no gene.

``show(look, tier)`` is the form of every status; the doctrine closes each
one. ``lab`` and ``lab_laws`` open with the doctrine, as the sow card does.
``refusal(exc, verb, look)`` renders the closed refusal table of
``wording.REFUSALS``, never an exception's text or detail; a refusal the
service marked ``written`` (raised after its verb's write returned) is
said as a refusal after the write, never as nothing written.
``served_fields(look)`` is the closed projection ``show --json`` prints:
every value a closed code but the lines' text. ``web_fields(look)`` is the
one the API serves: the same, except that the switched-off form carries no
line at all -- the terminal's ends in the settings file's path, which a
page or a phone never sees -- and that a view the API capped says, right
after its ``catching_up`` label, the API's own line of what computes it.

Pure: no clock, no randomness, no environment, no terminal width, no locale.
Nothing is imported at module level but the standard library.
"""

from typing import NamedTuple

checkpoint_before_apply = True

DAY = 1440
WIDTH = 78
ART_WIDTH = 32
ART_ROWS = 9

# The verbs a refusal is rendered for.
VERBS = ("show", "lab", "laws", "sow", "care", "name", "verify", "diff", "apply", "pin", "unpin", "resume",
         "finish", "share")
# The stream and the exit of the form of each status (a frozen view exits 1: the look says so).
STATUS_FORMS = {
    "disabled": ("stdout", 0),
    "stopped": ("stdout", 0),
    "unavailable": ("stderr", 1),
    "awaiting_soil": ("stdout", 2),
    "ready": ("stdout", 0),
    "sealed_bulbe": ("stdout", 0),
    "missing": ("stdout", 1),
    "retired_prototype": ("stdout", 0),
    "unreadable": ("stdout", 1),
    "resting": ("stdout", 0),
    "alive": ("stdout", 0),
}
# The label lines, in the order they open a form.
LABEL_ORDER = ("prototype", "retired", "glass_jar", "catching_up", "frozen", "mode_unknown", "bulbe")
# The statuses of a being that is identified: a refusal about one carries its labels.
IDENTIFIED = ("alive", "sealed_bulbe", "retired_prototype")
# The lines ``wrap`` never breaks: a row of the drawing, a path, the JSON line.
UNSPLIT = ("art", "path.line", "json")
# The store's own reasons for a record it does not open (``store.UNREADABLE``): its chain is not what failed, so
# the form names the reason and never says that no verified event is left.
STORE_UNREADABLE = ("soil", "pages", "owner", "unsown", "local")

SEASON_CODES = ("winter", "spring", "summer", "autumn")
SEASONS = ("Winter", "Spring", "Summer", "Autumn")
PLACES = {"garden": "in the garden", "windowsill": "on a windowsill"}
LIGHTS = {"down": "Sun down", "up": "Sun up", "rise": "Sunrise", "set": "Sunset"}
SOILS = {"dry": "Dry", "damp": "Damp", "wet": "Wet"}
LIVES = {"awake": "Awake", "breathing": "Awake, breathing", "dormant_winter": "Went dormant when winter came",
         "dormant_dry": "Went dormant in a dry spell", "dormant": "Dormant"}
_DORMANT = {"winter": "dormant_winter", "dry": "dormant_dry"}


class Felt(NamedTuple):
    """What ``show`` may read of a being at one minute, and its habitat."""

    name: object
    day: int
    season: int
    place: str
    light: str
    soil: str
    life: str
    minute: int
    daylength: int
    sun: int
    sun_max: int
    jar: bool
    layer: str
    labels: tuple
    local: str
    offset: str


def _wording():
    from . import wording

    return wording


def _say(key, /, **slots):
    return _wording().say(key, **slots)


def _line(key, text):
    return _wording().Line(key, text)


def offset_text(minutes):
    """``+HH:MM`` for an offset of ``minutes`` east of UTC."""
    sign = "+" if minutes >= 0 else "-"
    minutes = abs(minutes)
    return f"{sign}{minutes // 60:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------------------
# What show reads
# ---------------------------------------------------------------------------
def felt(view, *, name, weather, soil, layer, labels, constants):
    """The ``Felt`` of a served view: ``name`` its latest name, ``weather`` its genesis's, ``soil`` its store's.

    ``constants`` are the law's constants in force; ``stage.theta_dry`` and
    ``stage.theta_wet`` split the soil's water into dry, damp and wet.
    """
    state = view.state
    env = view.env
    bus = state["bus"]
    params = state["params"]
    sun = env["sun"]
    minute = env["minute"]
    if sun <= 0:
        light = "down"
    elif sun >= params["sun_max"]:
        light = "up"
    else:
        light = "rise" if minute < 720 else "set"
    stage = constants["stage"]
    moisture = bus["moisture"]
    if moisture < stage["theta_dry"]:
        water = "dry"
    elif moisture < stage["theta_wet"]:
        water = "damp"
    else:
        water = "wet"
    if bus["dormant"] == 0:
        life = "breathing" if bus["metab"] > 0 else "awake"
    else:
        life = _DORMANT.get(state["organs"]["stage"]["cause"], "dormant")
    year, month, day = env["civil"]
    return Felt(name=name, day=state["at"] // DAY, season=env["season"], place=weather, light=light, soil=water,
                life=life, minute=minute, daylength=env["daylength"], sun=sun, sun_max=params["sun_max"],
                jar=soil == "glass", layer=layer, labels=tuple(labels),
                local=f"{year:04d}-{month:02d}-{day:02d} {minute // 60:02d}:{minute % 60:02d}",
                offset=offset_text(state["tz"]))


# ---------------------------------------------------------------------------
# The drawing
# ---------------------------------------------------------------------------
def _fill(felt, start, width, first):
    """A row of soil ``width`` wide from absolute column ``start``."""
    if first and felt.season == 0:
        return "'" * width
    if felt.soil == "wet":
        return "~" * width
    if felt.soil == "damp":
        return ":" * width
    return "".join("." if (start + j) % 2 == 0 else " " for j in range(width))


def _seeded(fill, seed):
    at = (len(fill) - 4) // 2
    return fill[:at] + seed + fill[at + 4:]


def art(felt):
    """The drawing of ``felt``: 9 rows of at most 32 printable columns, right-stripped."""
    rows = ["", ""]
    if felt.sun <= 0 or felt.daylength <= 0:
        rows[0] = " " * 26 + "( )"
    else:
        rise = 720 - felt.daylength // 2
        pos = min(max(felt.minute - rise, 0), felt.daylength)
        x = 3 + (pos * 23) // felt.daylength
        rows[0 if felt.sun >= felt.sun_max else 1] = " " * x + "(*)"
    seed = "(..)" if felt.life in ("awake", "breathing") else "(--)"
    if felt.jar:
        rows.append("     ." + "-" * 20 + ".")
        rows.append("     |" + " " * 20 + "|")
        for i in range(3):
            fill = _fill(felt, 6, 20, i == 0)
            rows.append("     |" + (_seeded(fill, seed) if i == 1 else fill) + "|")
        rows.append("     '" + "-" * 20 + "'")
    else:
        lid = "=" if felt.layer == "bulbe" else "-"
        rows.append("   ." + lid * 24 + ".")
        for i in range(4):
            fill = _fill(felt, 5 + i, 22 - 2 * i, i == 0)
            rows.append(" " * (4 + i) + "\\" + (_seeded(fill, seed) if i == 2 else fill) + "/")
        rows.append(" " * 8 + "`" + "-" * 14 + "'")
    rows.append("_" * ART_WIDTH if felt.place == "windowsill" else ('"  ' * 11)[:ART_WIDTH].rstrip())
    rows = tuple(row.rstrip() for row in rows)
    for row in rows:
        if len(row) > ART_WIDTH or not all(0x20 <= ord(char) <= 0x7E for char in row):
            raise _wording().CopyRefused("art")
    return rows


# ---------------------------------------------------------------------------
# Lines of a being
# ---------------------------------------------------------------------------
def _when(felt):
    return f"{felt.local} ({felt.offset})"


def label_lines(labels, felt=None, *, short=True):
    """The label lines of ``labels``, in their fixed order; ``short``: the one-line prototype label."""
    labels = tuple(labels or ())
    out = []
    for code in LABEL_ORDER:
        if code not in labels:
            continue
        if code == "prototype":
            if "retired" not in labels:
                out.append(_say("label.prototype.short" if short else "label.prototype"))
        elif code in ("catching_up", "frozen"):
            if felt is not None:
                out.append(_say("label." + code, when=_when(felt)))
        else:
            out.append(_say("label." + code))
    return out


def identity(felt):
    """``{who}, seed, day {day}.``, checked by the nets with the name masked."""
    from . import ethics

    who = felt.name if felt.name is not None else "Onion"
    text = f"{who}, seed, day {felt.day}."
    masked = f"{'x' if felt.name is not None else who}, seed, day {felt.day}."
    if ethics.check(masked) or not all(0x20 <= ord(char) <= 0x7E for char in text):
        raise _wording().CopyRefused("identity")
    return _line("identity", text)


def conditions(felt):
    """``{Season} {place}. {Light}. {Soil} soil. {Life}.``, checked by the nets in full."""
    from . import ethics

    try:
        text = (f"{SEASONS[felt.season]} {PLACES[felt.place]}. {LIGHTS[felt.light]}. {SOILS[felt.soil]} soil. "
                f"{LIVES[felt.life]}.")
    except (IndexError, KeyError, TypeError):
        raise _wording().CopyRefused("conditions") from None
    if ethics.check(text):
        raise _wording().CopyRefused("conditions")
    return _line("conditions", text)


def show_form(felt, tier):
    """The form of an alive being: its labels, the drawing (``ascii`` tier), who, its conditions, the doctrine."""
    if tier not in ("text", "ascii"):
        raise ValueError(f"unknown tier: {tier}")
    lines = label_lines(felt.labels, felt)
    if tier == "ascii":
        lines += [_line("art", row) for row in art(felt)]
    lines += [identity(felt), conditions(felt), _say("doctrine")]
    return lines


def path_line():
    """The settings file's path, two spaces in, escaped to ASCII, never split."""
    from . import settings

    return _say("path.line", path=str(settings.config_file()))


# ---------------------------------------------------------------------------
# The form of every status
# ---------------------------------------------------------------------------
def _disabled(look):
    unreadable = look is not None and look.reason == "unreadable"
    return [_say("status.disabled.unreadable" if unreadable else "status.disabled"), path_line(),
            _say("status.disabled.kept")]


def _stopped(look):
    unknown = look is not None and look.reason == "unknown"
    return [_say("status.stopped.unknown" if unknown else "status.stopped")]


def _unreadable(seq, offer, reason=None):
    lines = [_say("status.unreadable")]
    chained = isinstance(seq, int) and not isinstance(seq, bool)
    if not chained and reason in STORE_UNREADABLE:
        return lines + [_say("status.unreadable.store", reason=reason)]
    if chained and seq >= 0:
        lines.append(_say("beetle.unreadable", seq=seq))
    kept = getattr(offer, "kept_seq", None)
    if isinstance(kept, int) and not isinstance(kept, bool) and kept >= 0:
        lines.append(_say("status.unreadable.resume", kept=kept, discarded=offer.discarded))
    else:
        lines.append(_say("status.unreadable.nothing"))
    return lines


def status_lines(look, labels_after=False):
    """The lines of a look's status, without the doctrine: the labels that open it, then what it says.

    ``unavailable`` goes to stderr, where its own line comes first and the
    being's labels follow, as after any refusal; ``labels_after`` renders
    any status that way, for a form said as a refusal.
    """
    status = look.status
    labels = label_lines(look.labels, look.felt)
    if labels_after and status not in ("unavailable", "alive"):
        return [line for line in status_lines(look._replace(labels=())) if not line.key.startswith("label.")] + labels
    if status == "disabled":
        return _disabled(look)
    if status == "stopped":
        return _stopped(look)
    if status == "unavailable":
        # A being the store opened (its habitat known) cannot be computed now; otherwise the store cannot be opened.
        opened = look.being is not None or look.habitat is not None
        key = "status.unavailable.view" if opened else "status.unavailable"
        return [_say(key, reason=look.reason)] + labels
    if status == "awaiting_soil":
        lines = [_say("status.awaiting_soil")]
        if look.mode != "daily" and look.glass_allowed is True:
            lines.append(_say("status.awaiting_soil.bulbe"))
        return labels + lines
    if status == "ready":
        return labels + [_say("status.ready")]
    if status == "sealed_bulbe":
        return labels + [_say("status.sealed_bulbe")]
    if status == "missing":
        tag = getattr(look.offer, "being", None)
        tail = _say("status.missing.finish", tag=tag) if tag is not None else _say("status.missing.nothing")
        return labels + [_say("status.missing"), tail]
    if status == "retired_prototype":
        return labels + [_say("status.retired_prototype")]
    if status == "unreadable":
        return labels + _unreadable(look.seq, look.offer, look.reason)
    if status == "resting":
        return labels + [_say("status.resting")]
    if status == "alive":
        return labels
    raise ValueError(f"unknown status: {status}")


def show(look, tier):
    """The ``show`` form of any look, in ``tier``; the doctrine closes it."""
    if look.status == "alive":
        if look.felt is None:
            raise ValueError("an alive look carries what show reads")
        return show_form(look.felt, tier)
    if tier not in ("text", "ascii"):
        raise ValueError(f"unknown tier: {tier}")
    return status_lines(look) + [_say("doctrine")]


# ---------------------------------------------------------------------------
# The laboratory and the sow card
# ---------------------------------------------------------------------------
def _lab_header(look):
    return ([_say("doctrine"), _say("lab.record", events=look.being.events)]
            + label_lines(look.labels, look.felt, short=False))


def _lab_status(look):
    """The laboratory of a being that is not alive: the doctrine, then the status form.

    A status served on stderr opens with its own line, after ``Error:``,
    and the doctrine follows it: the doctrine is never an error's headline.
    """
    if STATUS_FORMS[look.status][0] == "stderr":
        return status_lines(look) + [_say("doctrine")]
    return [_say("doctrine")] + status_lines(look)


def lab(look):
    """``oo garden lab``: the doctrine, the record, the labels, where the readings are; else the status form."""
    if look.status != "alive":
        return _lab_status(look)
    return _lab_header(look) + [_say("lab.readings")]


def _history(entry):
    seq, t, kind, body = entry
    if kind == "evolve":
        return _say("lab.history.update", day=t // DAY, seq=seq, law=body["to"]["name"], v=body["to"]["v"],
                    from_day=body["effective_from"] // DAY)
    if kind == "laws_pin":
        return _say("lab.history.pin", day=t // DAY, seq=seq)
    if kind == "laws_unpin":
        return _say("lab.history.unpin", day=t // DAY, seq=seq)
    raise ValueError(f"not a law fact: {kind}")


def lab_laws(look, laws):
    """``oo garden lab laws``: the header, then the laws of its world (``laws``, a ``LawsView``); else the form."""
    if look.status != "alive" or laws is None:
        return _lab_status(look)
    being = look.being
    state = laws.state
    lines = _lab_header(look) + [_say("lab.title")]
    born = "lab.born.provisional" if being.provisional else "lab.born.stable"
    lines.append(_say(born, law=being.law, v=being.v, digest=being.digest))
    lines.append(_say("lab.in_force", law=state.law["name"], v=state.law["v"], params=dict(state.params)))
    lines.append(_say("lab.pinned" if state.pinned else "lab.unpinned"))
    if state.pending is not None:
        to = state.pending["to"]
        lines.append(_say("lab.pending", law=to["name"], v=to["v"], day=state.pending["effective_from"] // DAY))
    if laws.due is not None:
        lines.append(_say("lab.available", law=laws.due["name"], v=laws.due["v"]))
    history = [_history(entry) for entry in laws.history]
    lines += history if history else [_say("lab.history.none")]
    diff = laws.diff
    if not hasattr(diff, "changed"):
        # The proposal could not be read as one (a refusal kept, never raised): no detail, the file named.
        lines += [_say("lab.proposal.refused"), path_line()]
    elif diff.changed:
        lines.append(_say("lab.proposal.differs", names=tuple(diff.changed)))
    else:
        lines.append(_say("lab.proposal.same"))
    if laws.offset_now is not None and laws.offset_now != state.tz:
        lines.append(_say("lab.offset", recorded=offset_text(state.tz), now=offset_text(laws.offset_now)))
    return lines


def card(record):
    """The sow card (a ``service.Card``): the doctrine, every disclosure, then the two questions, before any read."""
    place = {field: (value, source) for field, value, source in record.place}
    lines = [_say("doctrine")]
    if record.provisional:
        lines.append(_say("label.prototype"))
    if record.soil == "glass":
        lines.append(_say("sow.glass"))
    lines.append(_say("sow.sees"))
    lines.append(_say("sow.place", hemisphere=place["hemisphere"][0], hemisphere_from=place["hemisphere"][1],
                      band=place["band"][0], band_from=place["band"][1], weather=place["weather"][0],
                      weather_from=place["weather"][1]))
    if place["weather"][0] == "windowsill":
        lines.append(_say("sow.windowsill"))
    lines += [_say("sow.rhythm"), _say("sow.exit"), _say("sow.ask.name"), _say("sow.ask.confirm")]
    return lines


# ---------------------------------------------------------------------------
# What an act says
# ---------------------------------------------------------------------------
def _being_labels(look):
    if look is None:
        return []
    return label_lines(look.labels, look.felt)


def sown(result):
    """After a sowing (a ``service.Sown``): the being's labels, the beetle's line, the genome's and being's prints."""
    lines = _being_labels(result.look) + [_say("sow.done"),
                                          _say("sow.identity", genome=result.genome, tag=result.tag)]
    if not result.named and result.name_refusal is not None:
        lines.append(_say("sow.unnamed"))
    return lines


def acted(result, key, /, **slots):
    """After a write (a ``service.Acted``): the being's labels, the acknowledgement ``key``, the update it carried."""
    lines = _being_labels(result.look) + [_say(key, **slots)]
    if result.carried is not None:
        law, v, day = result.carried
        lines.append(_say("laws.carried", law=law, v=v, day=day))
    return lines


def applied(result):
    """After a law update written with its code (a ``service.Acted`` whose ``carried`` is that update)."""
    _law, _v, day = result.carried
    return _being_labels(result.look) + [_say("laws.applied", day=day)]


def deep(result, look=None):
    """``keep verify`` when nothing disagrees (a ``life.Deep``): the chain, the kept states, the skipped, the served."""
    lines = _being_labels(look) + [_say("verify.chain", facts=result.facts, days=result.days)]
    if result.kept > 0:
        lines.append(_say("verify.kept", agreed=result.agreed, kept=result.kept))
    else:
        lines.append(_say("verify.none"))
    if result.stale > 0:
        lines.append(_say("verify.stale", stale=result.stale))
    if result.ahead > 0:
        lines.append(_say("verify.ahead", ahead=result.ahead))
    lines.append(_say("verify.served", engine=result.engine))
    return lines


def diff_form(diff, look=None):
    """``keep laws diff``: nothing to apply, or each changed param and the code that writes the update."""
    lines = _being_labels(look)
    if not diff.changed:
        return lines + [_say("laws.diff.nothing")]
    lines.append(_say("laws.diff.head", law=diff.law["name"], v=diff.law["v"]))
    for param in diff.changed:
        lines.append(_say("laws.diff.row", param=param, before=diff.current[param], after=diff.proposed[param]))
    lines.append(_say("laws.diff.confirm", confirm=diff.confirm))
    return lines


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def _refusal_class(exc):
    """The first class of ``exc``'s lineage the table names, or ``None``."""
    table = _wording().REFUSALS
    for klass in type(exc).__mro__:
        if klass.__name__ in table:
            return klass.__name__
    return None


def _refusal_code(name, exc):
    if name == "ChainRefused":
        return getattr(exc, "reason", None)
    if name == "Diverged":
        return getattr(exc, "kind", None)
    if name == "Dropped":
        return getattr(exc, "reason", None)
    if name == "CopyRefused":
        return getattr(exc, "key", None)
    return getattr(exc, "code", None)


def _entry(name, code, verb):
    w = _wording()
    found = w.BY_VERB.get((name, code, verb)) or w.BY_VERB.get((name, "*", verb))
    if found is not None:
        return found
    codes = w.REFUSALS[name]
    if code in codes:
        return codes[code]
    found = codes.get("*")
    if found is not None and name == "StoreRefused" and "reason." + str(code) not in w.TEMPLATES:
        return None
    return found


def _refusal_line(key, name, code, exc):
    """One line of a refusal: its key with the slots the refusal fills."""
    if key == "refuse.store":
        return _say(key, reason=code)
    if key in ("refuse.membrane", "refuse.engine", "refuse.other", "verify.refused"):
        return _say(key, code=code)
    if key in ("verify.diverged.kept", "verify.diverged.blob"):
        return _say(key, day=exc.day, v=exc.laws)
    if key == "verify.diverged.served":
        engine = exc.engine if exc.engine in ("native", "reference") else "reference"
        return _say(key, day=exc.day, v=exc.laws, engine=engine)
    if key == "refuse.copy":
        return _say("refuse.copy.written" if getattr(exc, "written", False) is True else key, key=code)
    return _say(key)


def refusal(exc, verb, look=None):
    """``(lines, exit)`` of a refusal of ``verb``, from the closed table; ``TypeError`` for what it does not name.

    ``look`` (else the refusal's own) is the look it was refused on: a
    status refusal is said with that look's form, and a refusal about an
    identified being carries the being's label lines after its own.
    """
    if verb not in VERBS:
        raise ValueError(f"unknown verb: {verb}")
    name = _refusal_class(exc)
    if name is None:
        raise TypeError(f"not a refusal the garden names: {type(exc).__name__}")
    if look is None:
        look = getattr(exc, "look", None)
    code = _refusal_code(name, exc)
    identified = look is not None and look.status in IDENTIFIED
    if name != "CopyRefused" and getattr(exc, "written", False) is True:
        # Raised after the verb's write returned: the write was done, and the refusal never says otherwise.
        return [_say("refuse.written", code=code)] + (_being_labels(look) if identified else []), 1
    entry = _entry(name, code, verb)
    if entry is None:
        return [_say("refuse.other", code=code)] + (_being_labels(look) if identified else []), 1
    keys, exit_code = entry
    lines = []
    formed = False
    for key in keys:
        if key == "@disabled":
            lines += _disabled(look)
            formed = True
        elif key == "@stopped":
            lines += _stopped(look)
            formed = True
        elif key == "@form":
            # The look's own form, its own lines first and the labels after them; a status refusal with no look
            # to say is said as another refusal.
            if look is not None:
                lines += status_lines(look, labels_after=True)
                formed = True
        elif key == "@unreadable":
            lines += _unreadable(getattr(exc, "seq", None), exc if getattr(exc, "kept_seq", -1) >= 0 else None)
        elif key == "@labels":
            lines += _being_labels(look)
            formed = True
        elif key == "@path":
            lines.append(path_line())
        else:
            lines.append(_refusal_line(key, name, code, exc))
    if not lines:
        return [_say("refuse.other", code=code)] + (_being_labels(look) if identified else []), 1
    if not formed and identified:
        lines += _being_labels(look)
    if name == "ServiceRefused" and code == "status" and verb == "sow" and look.status == "awaiting_soil":
        exit_code = 2
    return lines, exit_code


# ---------------------------------------------------------------------------
# The served JSON
# ---------------------------------------------------------------------------
def served_fields(look):
    """The closed projection ``show --json`` prints: codes, the text tier's lines unwrapped, the doctrine last."""
    felt = look.felt if look.status == "alive" else None
    being = look.being
    return {
        "as_of": None if felt is None else {"local": felt.local, "offset": felt.offset},
        "being": None if felt is None else {
            "day": felt.day, "life": felt.life, "light": felt.light, "name": felt.name, "place": felt.place,
            "season": SEASON_CODES[felt.season], "soil": felt.soil, "stage": "seed"},
        "habitat": None if look.habitat is None else {"container": look.habitat[0], "layer": look.habitat[1]},
        "labels": list(look.labels),
        "law": None if being is None else {"name": being.law, "provisional": being.provisional, "v": being.v},
        "lines": [{"key": line.key, "text": line.text} for line in show(look, "text")],
        "source": "simulation",
        "status": look.status,
    }


def _web_disabled():
    """The switched-off form the API serves: every field empty, no line, so no path of this machine."""
    return {"as_of": None, "being": None, "habitat": None, "labels": [], "law": None, "lines": [],
            "source": "simulation", "status": "disabled"}


def web_fields(look):
    """The closed projection the API serves: ``served_fields``, the switched-off form emptied, never a path.

    ``disabled`` is built directly, with no line: its terminal form is never
    computed. Every other status is ``served_fields(look)``, with the API's
    own ``web.catching_up`` line inserted right after ``label.catching_up``.
    A line keyed ``path.line`` is refused (``CopyRefused("path.line")``),
    and so is a key that names an absence (``CopyRefused("json")``).
    """
    from . import ethics

    w = _wording()
    if look.status == "disabled":
        payload = _web_disabled()
    else:
        payload = served_fields(look)
        lines = []
        for line in payload["lines"]:
            lines.append(line)
            if line["key"] == "label.catching_up":
                said = w.web("web.catching_up")
                lines.append({"key": said.key, "text": said.text})
        payload["lines"] = lines
    if any(line["key"] == "path.line" for line in payload["lines"]):
        raise w.CopyRefused("path.line")
    if ethics.check_fields(payload):
        raise w.CopyRefused("json")
    return payload


# ---------------------------------------------------------------------------
# Wrapping, for the terminal
# ---------------------------------------------------------------------------
def _wrap_text(text, width):
    rows = []
    while len(text) > width:
        cut = text.rfind(" ", 0, width + 1)
        if cut <= 0 or not text[:cut].strip():
            rows.append(text[:width])
            text = text[width:]
            continue
        rows.append(text[:cut].rstrip(" "))
        text = text[cut:].lstrip(" ")
    if text or not rows:
        rows.append(text)
    return rows


def wrap(lines, width=WIDTH):
    """The rows the terminal prints for ``lines``: each broken at spaces within ``width``, a longer word hard.

    A row of the drawing, a path line and the JSON line are never broken.
    """
    rows = []
    for line in lines:
        if line.key in UNSPLIT:
            rows.append(line.text)
        else:
            rows.extend(_wrap_text(line.text, width))
    return rows
