#!/usr/bin/env python3
"""
The bodies of ``oo garden`` -- Opti-Oignon CLI.

``cli/main.py`` declares the group and its subtopics; each command body
imports its runner from here when it runs, so importing the CLI never
imports the garden. The garden itself (``opti_oignon.allium.service``) is
imported inside the runners, never at module load.

Every runner refuses words left on the command line first, without
repeating them and before any garden is built; then it builds one
``Garden`` from the root object's ``allium_service`` factory (the
production one when none is given), asks it one thing, renders what it
answers through ``describe`` whole, prints it, and closes the garden. While
the garden is open, the platform's log records never reach the terminal
(``_Session``): ``oo`` configures no logging, and a record would otherwise
print outside the catalogue. A refusal the garden names is said through
the closed refusal table, never with an exception's text; anything else is
a defect and is raised.

Text is read from stdin only, one line at a time and never echoed: a
line of 256 bytes without its end, a line that is not UTF-8, or no line at
all, is refused.
Every question is printed on stdout before its line is read, and only an
attended verb asks one.

The terminal writes through exactly two places: ``_say`` prints a form's
lines on one stream, wrapped at 78 columns at spaces (a row of the drawing,
a path line and the JSON line are never broken), after checking that every
character is printable ASCII and every row fits; ``_refuse`` prints a
refusal on stderr, its first row after the ``Error:`` prefix, and exits
with the refusal's code. Nothing else here writes.
"""

import json

import click

from .output import echo_error

checkpoint_before_apply = True

WIDTH = 78
# The width of a refusal's first row, so that ``Error: `` and the row fit in ``WIDTH``.
ERROR_WIDTH = WIDTH - len("Error: ")
# The longest answer read, in bytes, its line end included.
LINE_MAX = 256


def _rows(lines, width=WIDTH):
    """The rows of ``lines`` as the terminal prints them; ``ValueError`` if one cannot be printed as it is."""
    from opti_oignon.allium import describe

    rows = describe.wrap(list(lines), width)
    unsplit = [line for line in lines if line.key in describe.UNSPLIT]
    for row in rows:
        if not all(0x20 <= ord(char) <= 0x7E for char in row):
            raise ValueError("a row that is not printable ASCII")
        if len(row) > width and not any(row == line.text for line in unsplit):
            raise ValueError("a row wider than the terminal's width")
    return rows


def _say(lines, err=False):
    """Print ``lines`` (``Line`` records) on stdout, or stderr with ``err``, in one write."""
    rows = _rows(lines)
    if rows:
        click.echo("\n".join(rows), err=err)


def _refuse(ctx, lines, code):
    """Print a refusal on stderr -- its first row after the ``Error:`` prefix -- and exit with ``code``."""
    from opti_oignon.allium import wording

    head = _rows(lines[:1], ERROR_WIDTH) if lines else [""]
    rest = [wording.Line(lines[0].key, row) for row in head[1:]] + list(lines[1:])
    rows = _rows(rest)
    echo_error(head[0])
    if rows:
        _say(rest, err=True)
    ctx.exit(code)


def _line(key, /, **slots):
    from opti_oignon.allium import wording

    return wording.say(key, **slots)


def _unargued(ctx):
    """Refuse words left on the command line, without repeating them, before any garden is built."""
    if ctx.args:
        _refuse(ctx, [_line("refuse.argv")], 2)


def refuse_subtopic(ctx):
    """Refuse a subtopic the group does not have, without repeating the words given."""
    _refuse(ctx, [_line("refuse.subtopic")], 2)


def _garden(ctx):
    """The garden of this command: the root object's ``allium_service`` factory, else the production one."""
    obj = ctx.find_root().obj
    factory = obj.get("allium_service") if isinstance(obj, dict) else None
    if factory is None:
        from opti_oignon.allium import service

        factory = service.Garden.production
    return factory()


class _Session:
    """The garden of one command, closed at the end; the platform's log records stay off the terminal meanwhile.

    ``oo`` configures no logging, so a record no handler takes would reach
    stderr through logging's last resort, outside the catalogue and its
    checks. A handler that drops them is attached to the package's logger
    for the length of the command and taken off when it ends; a handler the
    embedding process configured still receives them.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.gardener = None
        self.guard = None

    def __enter__(self):
        import logging

        self.guard = logging.NullHandler()
        logging.getLogger("opti_oignon").addHandler(self.guard)
        try:
            self.gardener = _garden(self.ctx)
        except BaseException:
            self._unguard()
            raise
        return self.gardener

    def _unguard(self):
        import logging

        logging.getLogger("opti_oignon").removeHandler(self.guard)

    def __exit__(self, kind, exc, _trace):
        try:
            self.gardener.close()
        finally:
            self._unguard()
        return False


class _Refusals:
    """Say a refusal the garden names, raised inside, through the closed table, and exit with its code.

    ``done`` is a mapping whose ``written`` says a write was made before
    the refusal: a line that then fails its own check says the write was
    done. Anything the table does not name is raised as it is.
    """

    def __init__(self, ctx, verb, done=None):
        self.ctx = ctx
        self.verb = verb
        self.done = done

    def __enter__(self):
        return self

    def __exit__(self, kind, exc, _trace):
        from opti_oignon.allium import describe, wording

        if exc is None or not isinstance(exc, Exception):
            return False
        if isinstance(exc, (click.exceptions.Exit, click.exceptions.Abort, click.ClickException)):
            return False
        if isinstance(exc, wording.CopyRefused) and self.done is not None and self.done.get("written") \
                and not exc.written:
            exc = wording.CopyRefused(exc.key, written=True)
        try:
            lines, code = describe.refusal(exc, self.verb)
        except TypeError:
            return False
        _refuse(self.ctx, lines, code)
        return False


def _ask(ctx, end_key, bad_key):
    """One line of stdin, its end included; refused when it is too long, ``end_key`` when there is none.

    The line is read as bytes and decoded alone, so a line that is not
    UTF-8 is refused ``bad_key`` -- the refusal of what that line was to
    hold -- and never blamed on another line, nor repeated.
    """
    raw = click.get_binary_stream("stdin").readline(LINE_MAX)
    try:
        line = raw.decode("utf-8")
    except UnicodeDecodeError:
        _refuse(ctx, [_line(bad_key)], 2)
    if line == "":
        _refuse(ctx, [_line(end_key)], 2)
    if len(line) >= LINE_MAX and not line.endswith("\n"):
        _refuse(ctx, [_line("refuse.line")], 2)
    return line


def _form(ctx, look, lines):
    """Print a form on the stream its status serves, and exit with the look's exit."""
    from opti_oignon.allium import describe

    if describe.STATUS_FORMS[look.status][0] == "stderr":
        _refuse(ctx, lines, look.exit)
    _say(lines)
    if look.exit:
        ctx.exit(look.exit)


# ---------------------------------------------------------------------------
# Looks
# ---------------------------------------------------------------------------
def run_show(ctx, *, tier, as_json):
    """``oo garden show``: the form of what the being is now; ``--json`` its closed projection, on stdout."""
    from opti_oignon.allium import describe, ethics, wording

    _unargued(ctx)
    with _Session(ctx) as gardener:
        with _Refusals(ctx, "show"):
            look = gardener.look()
            if as_json:
                served = describe.served_fields(look)
                if ethics.check_fields(served):
                    raise wording.CopyRefused("json")
                lines = [wording.Line("json", json.dumps(served, sort_keys=True, separators=(",", ":")))]
            else:
                lines = describe.show(look, tier)
        if as_json:
            _say(lines)
            if look.exit:
                ctx.exit(look.exit)
        else:
            _form(ctx, look, lines)


def run_lab(ctx):
    """``oo garden lab``: the doctrine, the record and the labels; else the doctrine and the status form."""
    from opti_oignon.allium import describe

    _unargued(ctx)
    with _Session(ctx) as gardener:
        with _Refusals(ctx, "lab"):
            look = gardener.look()
            lines = describe.lab(look)
        _form(ctx, look, lines)


def run_lab_laws(ctx):
    """``oo garden lab laws``: the laws of its world; else the doctrine and the status form."""
    from opti_oignon.allium import describe, service

    _unargued(ctx)
    with _Session(ctx) as gardener:
        with _Refusals(ctx, "laws"):
            try:
                laws = gardener.laws()
            except service.ServiceRefused as refusal:
                if refusal.code not in ("off", "stopped", "status") or refusal.look is None:
                    raise
                look, laws = refusal.look, None
            else:
                look = laws.look
            lines = describe.lab_laws(look, laws)
        _form(ctx, look, lines)


def run_verify(ctx):
    """``oo garden keep verify``: the chain, every kept state and the state served now, on the reference."""
    from opti_oignon.allium import describe

    _unargued(ctx)
    with _Session(ctx) as gardener:
        with _Refusals(ctx, "verify"):
            result = gardener.verify()
            lines = describe.deep(result.deep, result.look)
        _say(lines)


def run_laws_diff(ctx):
    """``oo garden keep laws diff``: what a law update from the proposal would change, and its code."""
    from opti_oignon.allium import describe

    _unargued(ctx)
    with _Session(ctx) as gardener:
        with _Refusals(ctx, "diff"):
            result = gardener.laws_diff()
            lines = describe.diff_form(result.diff, result.look)
        _say(lines)


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
def run_sow(ctx, *, hemisphere, band, weather):
    """``oo garden sow``: the card, then a name and a yes read from stdin, then what was sown."""
    from opti_oignon.allium import describe, service

    _unargued(ctx)
    done = {"written": False}
    with _Session(ctx) as gardener:
        with _Refusals(ctx, "sow"):
            card = gardener.sow_card(hemisphere=hemisphere, band=band, weather=weather)
            lines = describe.card(card)
        _say(lines)
        typed = _ask(ctx, "refuse.eof", "refuse.name").rstrip("\r\n").strip(" ")
        name = None
        if typed:
            with _Refusals(ctx, "sow"):
                name = service.check_name(typed)
        answer = _ask(ctx, "refuse.eof", "refuse.answer").strip().lower()
        if answer == "no":
            _say([_line("sow.cancelled")])
            ctx.exit(1)
        if answer != "yes":
            _refuse(ctx, [_line("refuse.answer")], 2)
        with _Refusals(ctx, "sow", done):
            result = gardener.sow(name=name, hemisphere=hemisphere, band=band, weather=weather)
            done["written"] = True
            lines = describe.sown(result)
        _say(lines)


def run_care(ctx, act):
    """``oo garden care ACT``: one gesture noted, and the law update it carried."""
    from opti_oignon.allium import describe

    _unargued(ctx)
    done = {"written": False}
    with _Session(ctx) as gardener:
        with _Refusals(ctx, "care", done):
            result = gardener.act(act)
            done["written"] = True
            lines = describe.acted(result, "care.noted", act=act)
        _say(lines)


def run_name(ctx):
    """``oo garden keep name``: the question, one line read from stdin, then the name noted."""
    from opti_oignon.allium import describe, service

    _unargued(ctx)
    done = {"written": False}
    with _Session(ctx) as gardener:
        with _Refusals(ctx, "name"):
            gardener.name_card()
        _say([_line("keep.ask.name")])
        typed = _ask(ctx, "refuse.eof.name", "refuse.name")
        with _Refusals(ctx, "name", done):
            name = service.check_name(typed)
            result = gardener.name(name)
            done["written"] = True
            lines = describe.acted(result, "keep.named", name=name)
        _say(lines)


def run_laws_apply(ctx, confirm):
    """``oo garden keep laws apply CONFIRM``: the law update the diff showed, written with its code."""
    from opti_oignon.allium import describe

    _unargued(ctx)
    done = {"written": False}
    with _Session(ctx) as gardener:
        with _Refusals(ctx, "apply", done):
            result = gardener.laws_apply(confirm)
            done["written"] = True
            lines = describe.applied(result)
        _say(lines)


def _pinning(ctx, verb, key):
    from opti_oignon.allium import describe

    _unargued(ctx)
    done = {"written": False}
    with _Session(ctx) as gardener:
        with _Refusals(ctx, verb, done):
            result = gardener.laws_pin() if verb == "pin" else gardener.laws_unpin()
            done["written"] = True
            lines = describe.acted(result, key)
        _say(lines)


def run_laws_pin(ctx):
    """``oo garden keep laws pin``: the laws in force kept."""
    _pinning(ctx, "pin", "laws.pinned")


def run_laws_unpin(ctx):
    """``oo garden keep laws unpin``: the pin lifted."""
    _pinning(ctx, "unpin", "laws.unpinned")


def _restored(ctx, verb, call):
    from opti_oignon.allium import describe

    _unargued(ctx)
    done = {"written": False}
    with _Session(ctx) as gardener:
        with _Refusals(ctx, verb, done):
            look = call(gardener)
            done["written"] = True
            lines = describe.show(look, "ascii")
        _form(ctx, look, lines)


def run_resume(ctx, kept, discarded):
    """``oo garden keep resume KEPT DISCARDED``: the record resumed from its last verified event; the look after."""
    _restored(ctx, "resume", lambda gardener: gardener.resume(kept, discarded))


def run_finish(ctx, tag):
    """``oo garden keep finish TAG``: the interrupted sowing linked into place; the look after."""
    _restored(ctx, "finish", lambda gardener: gardener.finish(tag))


def run_share_confirm(ctx, code):
    """``oo garden share confirm CODE``: no consent request can be opened, so none is confirmed."""
    _unargued(ctx)
    with _Session(ctx) as gardener:
        with _Refusals(ctx, "share"):
            gardener.confirm_share(code)


# ---------------------------------------------------------------------------
# Not in this version
# ---------------------------------------------------------------------------
def run_later(ctx):
    """A subtopic of a later version: said, and nothing read or written."""
    _unargued(ctx)
    _say([_line("later")])


def run_later_refused(ctx):
    """A verb of a later version: refused before anything is read."""
    _unargued(ctx)
    _refuse(ctx, [_line("refuse.later")], 1)
