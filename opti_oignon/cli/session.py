#!/usr/bin/env python3
"""The interactive chat session behind ``oo chat``.

One object, one entry point: ``handle(line)`` turns a line into a turn or
a command and yields printable events. A line that does not start with a
slash is a turn: analysed, routed, and streamed through the executor
under the session's conversation id, so the history, the memory block and
the librarian's mirror follow it the way they follow a turn from the API.
A line that starts with a slash is a user action:

  /open ID          find a persisted onion again and continue that conversation
  /close            evict the whole Flesh through the gate, save, and end the conversation
  /pin TEXT         pin a statement to the conversation's Core, as the user
  /recall KEY       show the verbatim span behind a receipt, which marks it resolved
  /skill NAME ARGS  run ARGS as a turn with a published skill as the system suffix
  /help             list the commands
  /quit             end the session

Every refusal is an event carrying its reason by name; nothing here
raises into the loop that prints. The onion commands are refused while
the onion is switched off. A draft or an unknown skill is refused; a
published skill is human-approved text the user names on purpose, so its
body rides the turn's system prompt, where the skills the agent consults
on its own stay wrapped as untrusted data.

Executor, router, analyser, librarian, skill registry and the
conversation factory are seams, resolved lazily when not injected; the
module imports nothing from the package at load. Inference goes through
the registry the executor asks: this module holds no client of its own.
"""

from dataclasses import dataclass

checkpoint_before_apply = True

TOKEN = "token"
THINKING = "thinking"
INFO = "info"
REFUSAL = "refusal"
QUIT = "quit"

HELP = (
    "/open ID          find a persisted onion again and continue that conversation\n"
    "/close            evict the whole Flesh through the gate, save, and end the conversation\n"
    "/pin TEXT         pin a statement to the conversation's Core, as the user\n"
    "/recall KEY       show the verbatim span behind a receipt (this marks the receipt resolved)\n"
    "/skill NAME ARGS  run ARGS as a turn with a published skill as the system suffix\n"
    "/help             list the commands\n"
    "/quit             end the session"
)


@dataclass(frozen=True)
class Event:
    kind: str
    text: str


def _refusal(text):
    return Event(REFUSAL, text)


def _info(text):
    return Event(INFO, text)


def _default_executor():
    from opti_oignon.executor import executor

    return executor


def _default_analyze(question):
    from opti_oignon.analyzer import analyze

    return analyze(question)


def _default_route(analysis, priority, model):
    from opti_oignon.router import route

    return route(analysis, priority, model)


def _default_librarian():
    from opti_oignon.memory import librarian

    return librarian


def _default_skills():
    from opti_oignon.agent.skills import get_skill_registry

    return get_skill_registry()


def _default_new_conversation(title, model):
    from opti_oignon.conversation import create_conversation

    return create_conversation(title=title, model=model).id


class ChatSession:
    """A conversation driven line by line; every command is a user action."""

    def __init__(self, *, conversation_id=None, model=None, priority="balanced", refine=False,
                 executor=None, analyze=None, route=None, librarian=None, skills=None,
                 new_conversation=None):
        self.conversation_id = conversation_id
        self.model = model
        self.priority = priority
        self.refine = refine
        self._executor = executor
        self._analyze = analyze or _default_analyze
        self._route = route or _default_route
        self._librarian = librarian
        self._skills = skills
        self._new_conversation = new_conversation or _default_new_conversation
        self.closed = False

    # -- seams ------------------------------------------------------------

    def _executor_seam(self):
        if self._executor is None:
            self._executor = _default_executor()
        return self._executor

    def _librarian_seam(self):
        if self._librarian is None:
            self._librarian = _default_librarian()
        return self._librarian

    def _skills_seam(self):
        if self._skills is None:
            self._skills = _default_skills()
        return self._skills

    # -- entry point --------------------------------------------------------

    def handle(self, line):
        """The events for one line: a turn's stream, a command's outcome, or a refusal."""
        text = (line or "").strip()
        if not text:
            return
        if self.closed:
            yield _refusal("the session has ended")
            return
        if not text.startswith("/"):
            yield from self._turn(text)
            return
        name, _, rest = text[1:].partition(" ")
        rest = rest.strip()
        handler = {
            "open": self._open,
            "close": self._close,
            "pin": self._pin,
            "recall": self._recall,
            "skill": self._skill,
            "help": self._help,
            "quit": self._quit,
        }.get(name)
        if handler is None:
            yield _refusal(f"unknown command /{name}: /help lists the commands")
            return
        try:
            yield from handler(rest)
        except Exception as exc:  # noqa: BLE001 - every refusal is printed by name
            yield _refusal(f"/{name} refused: {_named(exc)}")

    # -- the turn -----------------------------------------------------------

    def _turn(self, question, suffix=None):
        try:
            analysis = self._analyze(question)
            routing = self._route(analysis, self.priority, self.model)
            if self.conversation_id is None:
                self.conversation_id = self._new_conversation(question[:60], getattr(routing, "model", None))
                yield _info(f"conversation {self.conversation_id}")
            stream = self._executor_seam().execute(
                question, routing, None, self.refine,
                conversation_id=self.conversation_id,
                system_prompt_suffix=suffix,
            )
            for chunk in stream:
                if isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == THINKING:
                    yield Event(THINKING, str(chunk[1]))
                else:
                    yield Event(TOKEN, str(chunk))
        except Exception as exc:  # noqa: BLE001 - a failed turn is said, and the session goes on
            yield _refusal(f"turn refused: {_named(exc)}")

    # -- commands -----------------------------------------------------------

    def _onion(self):
        librarian = self._librarian_seam()
        if not librarian.onion_enabled():
            raise _Refused("the onion memory is switched off (onion.yaml: enabled)")
        return librarian

    def _require_conversation(self, command):
        if self.conversation_id is None:
            raise _Refused(f"{command} needs a conversation: send a turn or /open one first")
        return self.conversation_id

    def _open(self, rest):
        if not rest:
            raise _Refused("/open needs a conversation id")
        opening = self._onion().open_onion(rest)
        self.conversation_id = opening.conversation_id
        yield _info(
            f"opened {opening.conversation_id}: {opening.flesh_turns} turn(s) in the Flesh, "
            f"{opening.peels} peel(s), Core root {opening.core_root[:12]}"
        )
        if opening.digest:
            yield _info("open receipts:\n" + opening.digest)

    def _close(self, rest):
        cid = self._require_conversation("/close")
        closing = self._onion().close_onion(cid)
        yield _info(
            f"closed {cid}: {closing.evicted} span(s) evicted, {closing.remaining} turn(s) left verbatim, "
            f"Core root {closing.core_root[:12]}, {'saved' if closing.saved else 'not saved: no persistence path'}"
        )
        if closing.refusal:
            yield _refusal(f"the gate refused the next span, which stays verbatim: {closing.refusal}")
        if closing.digest:
            yield _info("open receipts:\n" + closing.digest)
        self.conversation_id = None

    def _pin(self, rest):
        cid = self._require_conversation("/pin")
        if not rest:
            raise _Refused("/pin needs the text to pin")
        entry_id = self._onion().pin(cid, rest, actor="user")
        yield _info(f"pinned {entry_id[:12]}")

    def _recall(self, rest):
        cid = self._require_conversation("/recall")
        if not rest:
            raise _Refused("/recall needs a receipt key")
        span = self._onion().recall(cid, rest)
        lines = [f"[{t.get('turn_id', '')}] {t.get('role', '')}: {t.get('text', '')}" for t in span]
        yield _info("\n".join(lines) + "\n(the receipt is now resolved)")

    def _skill(self, rest):
        ref, _, args = rest.partition(" ")
        args = args.strip()
        if not ref:
            raise _Refused("/skill needs a skill name and a request")
        skill = self._published_skill(ref)
        if not args:
            raise _Refused(f"/skill {ref} needs a request to run the skill on")
        suffix = f"\n\nApply the skill {skill.name} ({skill.category}) v{skill.version}:\n{skill.body.strip()}"
        yield _info(f"skill {skill.category}/{skill.name} v{skill.version}")
        yield from self._turn(args, suffix=suffix)

    def _published_skill(self, ref):
        registry = self._skills_seam()
        category, _, name = ref.rpartition("/")
        if category:
            skill = registry.get(name, category)
            if skill is not None:
                return skill
            if registry.exists(name, category, draft=True):
                raise _Refused(f"skill {ref} is a draft, not published: a draft never runs")
            raise _Refused(f"no published skill {ref}")
        matches = [s for s in registry.list() if s.name == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            where = ", ".join(sorted(f"{s.category}/{s.name}" for s in matches))
            raise _Refused(f"skill {name} is published in more than one category ({where}): name it as category/name")
        if any(s.name == name for s in registry.list(include_drafts=True)):
            raise _Refused(f"skill {name} is a draft, not published: a draft never runs")
        raise _Refused(f"no published skill {name}")

    def _help(self, rest):
        yield _info(HELP)

    def _quit(self, rest):
        self.closed = True
        yield Event(QUIT, "bye")


class _Refused(Exception):
    """A command refused by the session itself; its message is the whole reason."""


def _named(exc):
    if isinstance(exc, _Refused):
        return str(exc)
    message = str(exc).strip("'\"") if isinstance(exc, KeyError) else str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
