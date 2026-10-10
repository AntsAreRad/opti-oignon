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
  /recall KEY       show the verbatim span behind a receipt; the receipt stays open
  /recall code:KEY  show the code block behind a [code:KEY] marker
  /resolve KEY      close a receipt, as the user: it leaves the digest
  /dropped          list the sentences the repair dropped from a summary, never their words
  /status           show what the onion's queue counted, by event and motive
  /skill NAME ARGS  run ARGS as a turn with a published skill as the system suffix
  /adopt NAME [DIGEST]  show a skill's bytes not adopted here, then adopt those bytes
  /review [ID [DIGEST|decline]]  the agent's writes waiting for you: list, show one, accept it by its digest
  /help             list the commands
  /quit             end the session

Every refusal is an event carrying its reason by name; nothing here
raises into the loop that prints. The onion commands are refused while
the onion is switched off. A draft or an unknown skill is refused. A
published skill's body rides the turn's system prompt only when its bytes
are the user's on this device -- written here by hand, or named by their
digest here: a proposal of the agent or its teacher accepted with the
digest of its text, or bytes adopted -- the very bytes read once to judge
and to run them. The skills the agent consults on its own are admitted by
the same rule and stay wrapped as untrusted data. A skill received from a
paired device, which the sync gate let through on its provenance without
showing its text, and the agent's text approved on its name alone before
proposals, are refused until ``/adopt`` has shown their bytes here and the
user has named their digest. ``/review`` is where the user reads the
agent's waiting writes whole, every character a screen hides written as
its escape, and accepts one by naming its digest.

Executor, router, analyser, librarian, skill registry and the
conversation factory are seams, resolved lazily when not injected; the
module imports nothing from the package at load. Inference goes through
the registry the executor asks: this module holds no client of its own.
"""

import threading
from dataclasses import dataclass
from types import SimpleNamespace

checkpoint_before_apply = True

TOKEN = "token"
THINKING = "thinking"
INFO = "info"
REFUSAL = "refusal"
QUIT = "quit"

HELP = (
    "/open ID          find a persisted onion again and continue that conversation\n"
    "/close            evict the whole Flesh through the queue, save, and end the conversation\n"
    "/pin TEXT         pin a statement to the conversation's Core, as the user\n"
    "/recall KEY       show the verbatim span behind a receipt (the receipt stays open)\n"
    "/recall code:KEY  show the code block behind a [code:KEY] marker\n"
    "/resolve KEY      close a receipt, as the user: it leaves the digest\n"
    "/proposals        list the decisions you typed that the queue offers the Core, word for word\n"
    "/accept ID        pin one proposal's exact words to the Core, as the user\n"
    "/decline ID       set one proposal aside; the Core does not change\n"
    "/dropped          list the sentences the repair dropped from a summary: peel, receipt, motive, digest\n"
    "/status           show what the onion's queue counted, by event and motive; no word of a conversation\n"
    "/skill NAME ARGS  run ARGS as a turn with a published skill as the system suffix\n"
    "/adopt NAME [DIGEST]  show a skill's bytes not adopted here, then adopt those bytes\n"
    "/adopt-memory [ID DIGEST]  the facts of memory you have not endorsed: list them whole, adopt one by its digest\n"
    "/withdraw KIND VALUE  withdraw a source (a web address, a document's text, a fact's id, or kind:digest):\n"
    "                  every turn it reached is lowered\n"
    "/review [ID [DIGEST|decline]]  the agent's writes waiting for you: list, show one, accept it by its digest\n"
    "/help             list the commands\n"
    "/quit             end the session"
)

# The commands a line read from a pipe or a file may give: they act as no
# one and change nothing. Every other command acts as the user, and is
# typed at the keyboard.
_PIPED_COMMANDS = frozenset({"help", "quit"})


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


def _default_pending():
    from opti_oignon.pending_writes import get_pending_store

    return get_pending_store()


def _default_memory():
    from opti_oignon.memory.dedup import get_memory_store

    return get_memory_store()


def _default_withdraw(source):
    from opti_oignon.source_withdrawal import withdraw

    return withdraw(source)


def _default_mirror(conversation_id):
    from opti_oignon.conversation import conversation_manager

    return conversation_manager.get_mirror_messages(conversation_id)


def _visible(text):
    """``text`` as the approval drawer shows it: every character a screen hides written as its escape."""
    try:
        from opti_oignon.tool_call_approval import _visible as drawn
    except Exception:  # noqa: BLE001 - no drawer to agree with: escape all but printable ASCII and newlines
        return "".join("\\\\" if ch == "\\" else ch if ch == "\n" or " " <= ch <= "~" else ascii(ch)[1:-1]
                       for ch in str(text))
    return drawn(str(text))


_WRITERS = {"agent": "the agent", "teacher-escalation": "the teacher model"}


def _unadopted(registry, skill):
    """Why a published skill's bytes are not admitted here, in a few words, for one not received by sync."""
    if registry.rewritten(skill):
        return "was rewritten through the registry before its text was shown, and never adopted here"
    writer = _WRITERS.get(skill.source, f"the source '{skill.source}'")
    return f"was written by {writer} and its text was never adopted here"


def _default_new_conversation(title, model):
    from opti_oignon.conversation import create_conversation

    return create_conversation(title=title, model=model).id


class ChatSession:
    """A conversation driven line by line; every command is a user action."""

    def __init__(self, *, conversation_id=None, model=None, priority="balanced", refine=False,
                 executor=None, analyze=None, route=None, librarian=None, skills=None,
                 new_conversation=None, pending=None, memory=None, withdraw=None, mirror=None):
        self.conversation_id = conversation_id
        self.model = model
        self.priority = priority
        self.refine = refine
        self._executor = executor
        self._analyze = analyze or _default_analyze
        self._route = route or _default_route
        self._librarian = librarian
        self._skills = skills
        self._pending = pending
        self._memory = memory
        self._withdraw = withdraw or _default_withdraw
        self._mirror = mirror or _default_mirror
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

    def _pending_seam(self):
        if self._pending is None:
            self._pending = _default_pending()
        return self._pending

    def _memory_seam(self):
        if self._memory is None:
            self._memory = _default_memory()
        return self._memory

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
        name = text[1:].partition(" ")[0]
        if getattr(self, "pasted_input", False) and name not in _PIPED_COMMANDS:
            # A command acts as the user -- it pins, accepts, declines,
            # closes -- and a line read from a pipe or a file is not the
            # user's typing: a mail piped in could carry "/accept".
            yield _refusal("command refused: commands are typed at the keyboard, and this line was read from a pipe "
                           "or a file")
            return
        name, _, rest = text[1:].partition(" ")
        rest = rest.strip()
        handler = {
            "open": self._open,
            "close": self._close,
            "pin": self._pin,
            "recall": self._recall,
            "resolve": self._resolve,
            "proposals": self._proposals,
            "accept": self._accept,
            "decline": self._decline,
            "dropped": self._dropped,
            "status": self._status,
            "skill": self._skill,
            "adopt": self._adopt,
            "adopt-memory": self._adopt_memory,
            "withdraw": self._withdraw_source,
            "review": self._review,
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
            # A line that did not come from a keyboard (a pipe, a file) is
            # pasted: its turn is a document, whose words endorse nothing.
            pasted_run = None
            if getattr(self, "pasted_input", False):
                from opti_oignon.executor import user_turn

                pasted_run = SimpleNamespace(stop=threading.Event(), results={}, steps=None,
                                             user_turn=user_turn(question, question, (), pasted=[[0, len(question)]]))
            stream = self._executor_seam().execute(
                question, routing, None, self.refine,
                conversation_id=self.conversation_id,
                system_prompt_suffix=suffix,
                **({"run": pasted_run} if pasted_run is not None else {}),
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
        onion = self._onion()
        # The conversation as the store reads it, so each turn leaves with
        # the context it declares; a read that fails closes as before.
        try:
            messages = self._mirror(cid)
        except Exception:  # noqa: BLE001 - the close is never stopped by its read
            messages = None
        closing = onion.close_onion(cid, **({"messages": messages} if isinstance(messages, list) and messages else {}))
        yield _info(
            f"closed {cid}: {closing.evicted} span(s) evicted, {closing.remaining} turn(s) left verbatim, "
            f"Core root {closing.core_root[:12]}, {'saved' if closing.saved else 'not saved: no persistence path'}"
            + ("; ended without the model, on the rungs that need none" if closing.without_model else "")
        )
        if closing.refusal:
            yield _refusal(f"the close stopped at a span it could not commit, which stays verbatim: {closing.refusal}")
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
        if rest.startswith("code:"):
            block = self._onion().recall_code(cid, rest)
            yield _info(f"{block['key']} ({block['language'] or 'no language named'}):\n{block['code']}")
            return
        span = self._onion().recall(cid, rest)
        lines = [f"[{t.get('turn_id', '')}] {t.get('role', '')}: {t.get('text', '')}" for t in span]
        yield _info("\n".join(lines) + "\n(the receipt stays open; /resolve KEY closes it)")

    def _resolve(self, rest):
        cid = self._require_conversation("/resolve")
        if not rest:
            raise _Refused("/resolve needs a receipt key")
        self._onion().resolve_receipt(cid, rest, actor="user")
        yield _info(f"receipt {rest[:12]} closed: it leaves the digest and stays in the ledger")

    def _proposals(self, rest):
        cid = self._require_conversation("/proposals")
        offered = self._onion().proposals(cid)
        if not offered:
            yield _info("no open proposal")
            return
        lines = [f"{p['id'][:12]} [{p['turn_id']}, {p['origin']}, {p['made_on']}] {p['text']}" for p in offered]
        yield _info("open proposals (/accept ID pins one to the Core, /decline ID sets it aside):\n" + "\n".join(lines))

    def _dropped(self, rest):
        """The sentences the repair dropped from the peels of the conversation: by peel, receipt, motive and digest."""
        cid = self._require_conversation("/dropped")
        found = self._onion().dropped_sentences(cid)
        if not found:
            yield _info("no sentence dropped by the repair")
            return
        lines = [f"peel {d['peel'][:12]} over receipt {d['receipt'][:12]}: {d['motive']} ({d['sha256'][:12]})"
                 for d in found]
        yield _info("sentences the repair dropped from a summary (no peel keeps their words; /recall KEY shows "
                    "the span):\n" + "\n".join(lines))

    def _one_proposal(self, cid, command, rest):
        """The one open proposal ``rest`` begins the id of; anything else is refused by name."""
        if not rest:
            raise _Refused(f"{command} needs a proposal id")
        ids = [p["id"] for p in self._onion().proposals(cid) if p["id"].startswith(rest)]
        if len(ids) != 1:
            raise _Refused(f"{command}: {rest[:24]!r} names {len(ids)} open proposal(s); one is needed")
        return ids[0]

    def _accept(self, rest):
        cid = self._require_conversation("/accept")
        pid = self._one_proposal(cid, "/accept", rest)
        entry_id = self._onion().accept_proposal(cid, pid, actor="user")
        yield _info(f"proposal {pid[:12]} pinned to the Core as {entry_id[:12]}")

    def _decline(self, rest):
        cid = self._require_conversation("/decline")
        pid = self._one_proposal(cid, "/decline", rest)
        self._onion().decline_proposal(cid, pid, actor="user")
        yield _info(f"proposal {pid[:12]} declined: the Core does not change")

    def _status(self, rest):
        """What the onion's queue counted, by event and motive, and whether the onion is on; shown with it off too."""
        librarian = self._librarian_seam()
        totals = librarian.counter_totals()
        switch = "on" if librarian.onion_enabled() else "off (onion.yaml: enabled)"
        kept = f"kept across sessions since {totals['since']}" if totals["persisted"] else "kept in this process only"
        if totals["refused"]:
            kept += f" ({totals['refused']})"
        lines = [f"the onion memory is {switch}; its counts, {kept}:"]
        for event in sorted(totals["counts"]):
            motives = totals["counts"][event]
            lines.append(f"  {event}: " + ", ".join(f"{motive}={n}" for motive, n in sorted(motives.items())))
        if len(lines) == 1:
            lines.append("  nothing counted yet")
        yield _info("\n".join(lines))

    def _skill(self, rest):
        ref, _, args = rest.partition(" ")
        args = args.strip()
        if not ref:
            raise _Refused("/skill needs a skill name and a request")
        # One read: the bytes judged here are the bytes that ride the prompt.
        skill = self._published_skill(ref)
        registry = self._skills_seam()
        if not registry.admits(skill):
            where = f"{skill.category}/{skill.name}"
            if registry.received(skill):
                raise _Refused(
                    f"skill {where} arrived from a paired device and these bytes were never adopted here: "
                    f"/adopt {where} shows them"
                )
            raise _Refused(f"skill {where} {_unadopted(registry, skill)}: /adopt {where} shows it")
        if not args:
            raise _Refused(f"/skill {ref} needs a request to run the skill on")
        suffix = f"\n\nApply the skill {skill.name} ({skill.category}) v{skill.version}:\n{skill.body.strip()}"
        yield _info(f"skill {skill.category}/{skill.name} v{skill.version}")
        yield from self._turn(args, suffix=suffix)

    def _adopt(self, rest):
        ref, _, digest = rest.partition(" ")
        digest = digest.strip()
        if not ref:
            raise _Refused("/adopt needs a skill name")
        skill = self._published_skill(ref)
        registry = self._skills_seam()
        where = f"{skill.category}/{skill.name}"
        received = registry.received(skill)
        if registry.admits(skill):
            if not received and skill.source == "manual":
                yield _info(f"skill {where} was written on this device: there is nothing to adopt")
            else:
                yield _info(f"skill {where}: these bytes are already adopted on this device")
            return
        if not digest:
            origin = "arrived from a paired device" if received else _unadopted(registry, skill)
            yield _info(f"skill {where} {origin}; its text, as it is on disk, every character a screen hides "
                        f"written as its escape:\n{_visible(skill.raw)}")
            yield _info(f"digest {skill.file_digest[:16]}: /adopt {where} {skill.file_digest[:16]} adopts exactly "
                        "these bytes")
            return
        adopted = registry.adopt(skill.name, skill.category, digest)
        if adopted is None:
            raise _Refused(f"the digest {digest} does not name the bytes of {where} on disk now: /adopt {where} shows them again")
        yield _info(f"adopted {where} ({adopted[:16]}): /skill runs it now")

    def _adopt_memory(self, rest):
        """The facts of memory the user has not endorsed: list them whole, or adopt one by the digest shown.

        A fact kept from before endorsements, or received from a peer, lowers
        every turn it is placed in until it is adopted. The digest shown is
        the first sixteen characters of the digest of the fact's text; an
        adoption whose digest no longer names that text is refused.
        """
        from opti_oignon.memory.canonical_store import fact_digest

        fact_id, _, digest = rest.partition(" ")
        fact_id, digest = fact_id.strip(), digest.strip()
        store = self._memory_seam()
        records = store.unendorsed()
        if not fact_id:
            if not records:
                yield _info("every fact of memory is endorsed: there is nothing to adopt")
                return
            lines = [f"{len(records)} fact(s) of memory you have not endorsed; each lowers the turns it is placed "
                     "in until you adopt it. Every character a screen hides is written as its escape:"]
            for record in records:
                lines.append(f"  {record.id}  digest {fact_digest(record.text)[:16]}  [{record.category}]  "
                             f"{_visible(record.text)}")
            lines.append("/adopt-memory ID DIGEST adopts one fact's text exactly as shown")
            yield _info("\n".join(lines))
            return
        record = next((r for r in records if r.id == fact_id), None)
        if record is None:
            raise _Refused(f"no fact {fact_id} waits for your adoption: /adopt-memory lists them")
        full = fact_digest(record.text)
        if len(digest) < 16 or not full.startswith(digest):
            raise _Refused(f"the digest {digest or '(none)'} does not name the text of fact {fact_id} as it reads now: "
                           "/adopt-memory shows it again")
        if not store.adopt(fact_id, full):
            raise _Refused(f"fact {fact_id} changed before it could be adopted: /adopt-memory shows it again")
        yield _info(f"adopted fact {fact_id} ({full[:16]}): it no longer lowers the turns it is placed in")

    def _withdraw_source(self, rest):
        """Withdraw a source: every turn and branch message it reached is lowered; a fact of memory is set aside."""
        from opti_oignon.source_withdrawal import source_for

        kind, _, value = rest.partition(" ")
        kind, value = kind.strip(), value.strip()
        if not kind:
            raise _Refused("/withdraw needs a kind and a value (web URL, document TEXT, memory ID) or kind:digest")
        if value:
            try:
                source = source_for(kind, value)
            except ValueError as exc:
                raise _Refused(str(exc)) from None
        else:
            source = kind
        try:
            outcome = self._withdraw(source)
        except ValueError as exc:
            raise _Refused(f"withdrawal refused: {exc}") from None
        aside = "; the fact is set aside (restorable)" if outcome.get("fact_set_aside") else ""
        yield _info(f"withdrew {source}: {outcome.get('turns', 0)} turn(s) and "
                    f"{outcome.get('branch_messages', 0)} branch message(s) lowered{aside}")
        if value and kind not in ("web", "memory") and not outcome.get("turns") and not outcome.get("branch_messages"):
            # A text is named by its exact bytes, which one typed line may
            # not hold (a line break, a space at an edge): a miss is said.
            yield _info("no turn carried that text: a document's text is named by its exact bytes; "
                        "name it as kind:digest (the SHA-256 of the exact text) to be sure")

    def _review(self, rest):
        """The agent's writes waiting for the user: list them, show one whole, accept it by its digest, or decline."""
        from opti_oignon import pending_writes

        pid, _, word = rest.partition(" ")
        pid, word = pid.strip(), word.strip()
        queue = self._pending_seam()
        if not pid:
            records = queue.list(status="pending")
            if not records:
                yield _info("nothing waits for your review")
                return
            lines = [f"{len(records)} waiting for your review; /review ID shows one whole:"]
            for record in records:
                lines.append(f"  {record.id}  {_review_summary(record)}  "
                             f"digest {pending_writes.shown_digest(record)[:16]}")
            yield _info("\n".join(lines))
            return
        record = queue.get(pid)
        if record is None or record.status != "pending":
            raise _Refused(f"no proposal {pid} waits for your review: /review lists them")
        if not word:
            yield _info(self._review_text(record, pending_writes.shown_digest(record)))
            return
        if word == "decline":
            result = pending_writes.decline([pid], pending=queue)[0]
            if not result.get("declined"):
                raise _Refused(f"proposal {pid} not declined: {result.get('reason', '')}")
            yield _info(f"declined {pid}: nothing was written")
            return
        result = pending_writes.accept([pid], pending=queue, digests={pid: word},
                                       skills_registry=self._skills_seam())[0]
        if not result.get("applied"):
            reason = str(result.get("reason", ""))
            if "digest" in reason:
                raise _Refused(f"the digest {word} does not name what proposal {pid} shows: /review {pid} shows it "
                               "again")
            raise _Refused(f"proposal {pid} not applied: {reason}")
        yield _info(f"accepted {pid}: {result.get('outcome', '')}")

    def _review_text(self, record, digest):
        """One proposal whole: what it would do, each word as the drawer shows it, and the digest that accepts it."""
        arguments = record.arguments
        who = {"agent": "the agent", "teacher": "the teacher model", "extraction": "the extraction"}.get(
            str(record.provenance.get("source", "")), "the agent")
        lines = [f"proposal {record.id}: {_review_summary(record)}, proposed by {who}"]
        if record.store == "skills":
            lines.append("risk: high -- a skill's text reaches a system prompt")
            base = arguments.get("base_sha256")
            target = self._skills_seam().get(arguments.get("name", ""), arguments.get("category", ""),
                                             draft=bool(arguments.get("draft")))
            if record.action == "delete":
                kind = "draft" if arguments.get("draft") else "published skill"
                now = _visible(target.canonical()) if target is not None else "(no longer there)"
                lines.append(f"it deletes the {kind} whose text, as it is now, every character a screen hides "
                             f"written as its escape, is:\n{now}")
            else:
                lines.append("its whole text, every character a screen hides written as its escape:")
                lines.append(_visible(arguments.get("text", "")))
                if base:
                    lines.append(f"it replaces the published text whose digest is {base}")
            if (target.digest() if target is not None else None) != base:
                lines.append("what it changes changed since it was proposed: accepting it will be refused")
        else:
            for key, value in arguments.items():
                if value is not None:
                    lines.append(f"{key}: {_visible(value) if isinstance(value, str) else value}")
        untyped = record.provenance.get("untyped") or []
        if untyped:
            lines.append("not typed by you: " + ", ".join(str(name) for name in untyped))
        read = record.provenance.get("read") or []
        if read:
            lines.append("read before proposing: " + ", ".join(str(name) for name in read))
        lines.append(f"digest {digest}: /review {record.id} {digest[:16]} accepts exactly this; "
                     f"/review {record.id} decline declines it")
        return "\n".join(lines)

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
        self.end()
        yield Event(QUIT, "bye")

    def end(self):
        """Write what the onion counted in this session; the front end calls it at /quit, at the end of its input and on an interruption. Never raises."""
        try:
            self._librarian_seam().flush_counters()
        except Exception:  # noqa: BLE001 - the counts stay unwritten, the session still ends
            pass


class _Refused(Exception):
    """A command refused by the session itself; its message is the whole reason."""


def _review_summary(record):
    """What accepting a proposal would do, on one line."""
    arguments = record.arguments
    if record.store == "skills":
        what = {"add": "add the skill", "edit": "change the skill", "delete": "delete the skill"}.get(
            record.action, f"{record.action} the skill")
        draft = " (its draft)" if arguments.get("draft") else ""
        return f"{what} {arguments.get('category', '')}/{arguments.get('name', '')}{draft}"
    return f"{record.action} in {record.store}"


def _named(exc):
    if isinstance(exc, _Refused):
        return str(exc)
    message = str(exc).strip("'\"") if isinstance(exc, KeyError) else str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
