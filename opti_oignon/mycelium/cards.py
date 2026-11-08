"""The map's file cards (P20, phase 2, stage 2): two sentences a local model writes of a file -- what it does, when the
rest of the corpus uses it --, CHECKED before they are kept, by the citation gate's own rule for a claim's words: each
identifier and number a sentence gives (a dotted name, a decimal kept whole: a sentence ends at a stop followed by a
space) must be a whole token of the file (a dotted name by each of its parts, a part of the file's own path counting),
and each path or address it gives must be in the file, else the sentence is dropped and counted. Prose is not checked:
a sentence of plain words passes as the model wrote it -- the map says the card is a model's. A card left with no
sentence is kept empty: the map shows nothing of it, and the file is not asked again until it changes.

A card is written from the file as the store last indexed it -- its bytes read again and their sha256 the store's,
else the file is passed over until the next index pass -- and holds that sha256: the index pass that changes the file
removes its card (``index.py``), so the store keeps no word of a file's old text past the rewrite that pass owes; the
next pass of the cards writes it again. Only the folders of the public and private sources are carded, never a secret
one: the source's sensitivity is read again before each file is read and inside the transaction that writes its card,
and a source marked secret loses its cards at once (``Store.source``). The model is asked through the bridge's asker
(the registry, the governor, as background work, ``CALLER``); the first refusal -- the model, the governor, a call
given up -- ends the pass, what it wrote kept, the refusal said. The store's lock is held to plan and to write, never
across the model's answer; a card replaced owes the store a rewrite (its old text stays in the pages' free space until
one), as a pass that changes a file does.

The map shows a card as a model's, its names checked against the file, never as something to cite (``map.py``). No
card is searched: on the bench the cards lowered the search's fusion (0.633 against 0.658, P20, 4 November); they serve
the map alone.

  write(store, settings, *, source=None, limit=None, asker=None)  one pass; its counts
"""
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from . import gate, vectors
from .bridge import MAX_TIMEOUT_S, Asker, AskError, answer_of
from .index import file_bytes

logger = logging.getLogger(__name__)

CALLER = "index"            # the governor's background caller, as the vector pass asks
MAX_DEFINES = 40            # names a file defines, given to the model
MAX_HEAD_LINES = 60         # a file's first lines, given to the model
MAX_HEAD_CHARS = 4000
MAX_SENTENCES = 2
MAX_CARD = 400              # characters of a kept card
MAX_PER_PASS = 2000
NUM_PREDICT = 160
PROMPT = ("Here is a file of a software project: its path, its first lines and the list of what it defines.\n\n"
          "PATH: {path}\nDEFINES: {defines}\nFIRST LINES:\n{head}\n\n"
          "In two short sentences, say what this file does and when the rest of the project uses it. Name only "
          "functions, classes or settings that appear above. No preamble.")
PATH_WORD = re.compile(r"[A-Za-z0-9]+")
EDGES = "`'\"()[]{}<>,;:!?"


@dataclass(frozen=True)
class CardSettings:
    model: str
    per_pass: int
    timeout_s: float


def load_settings(path=None):
    """The cards' settings from ``mycelium.yaml`` (section ``cards``), refused by name (ValueError) when malformed: a
    model that is no name, a count of cards per pass that is not one (1 to ``MAX_PER_PASS``), a timeout that is not a
    positive number of seconds (at most ``MAX_TIMEOUT_S``). No section: the defaults (the small model of the bench)."""
    section = vectors.Section("cards", path, error=ValueError)
    section.only("model", "per_pass", "timeout_s", where="cards")
    raw, typed = section.raw, section.typed
    model = typed(raw.get("model", "qwen3.5:4b"), (str,), "cards.model")
    if not model.strip():
        raise ValueError("mycelium.yaml: cards.model names no model")
    per_pass = typed(raw.get("per_pass", 200), (int,), "cards.per_pass")
    if isinstance(per_pass, bool) or not 1 <= per_pass <= MAX_PER_PASS:
        raise ValueError(f"mycelium.yaml: cards.per_pass is a count from 1 to {MAX_PER_PASS}, not {str(per_pass)[:40]}")
    return CardSettings(model, per_pass, float(section.positive(raw.get("timeout_s", 60), (int, float),
                                                                "cards.timeout_s", MAX_TIMEOUT_S)))


def _sentences(text):
    """``text``'s sentences, as the citation gate ends one (``gate.SENTENCE_END``: a stop followed by a space or the
    end), so a dotted name or a decimal stays whole."""
    out, start = [], 0
    for match in gate.SENTENCE_END.finditer(text):
        out.append(text[start:match.end()].strip())
        start = match.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


def _unbacked(sentence, body):
    """Whether ``sentence`` gives a path or an address the file does not hold (the gate leaves both unread)."""
    for token in sentence.split():
        word = token.strip(EDGES).rstrip(".")
        if ("/" in word or "://" in word or word.startswith("~")) and word not in body:
            return True
    return False


def checked(text, body, path):
    """``(the card: its sentences whose every identifier and number is a whole token of ``body`` or of the file's
    path, and every path or address in ``body``, two at most, on one line, MAX_CARD characters at most; how many
    sentences were dropped)``."""
    names, numbers = gate._tokens(body)
    path_words = frozenset(PATH_WORD.findall(path))
    kept, dropped = [], 0
    for sentence in _sentences(" ".join(answer_of(text or "").split())):
        if len(sentence) < 3:
            continue
        if not _unbacked(sentence, body) and all(gate._found(word, names, numbers, path_words)
                                                 for word in gate._words_of(sentence)):
            if len(kept) < MAX_SENTENCES:
                kept.append(sentence)
        else:
            dropped += 1
    card = " ".join(kept)
    return (card if len(card) <= MAX_CARD else card[:MAX_CARD - 3].rstrip() + "..."), dropped


def _wanted(store, source, limit):
    """The files of the folders of the public and private sources (``source`` alone when named) that have no card
    of their sha256: ``[(file id, root, path, sha256)]``, by path; ``limit`` at most. A source of documents is never
    carded (its texts are no files to read again)."""
    query = ("SELECT f.id, s.root, f.path, f.sha256 FROM file f JOIN source s ON s.id = f.source "
             "LEFT JOIN card c ON c.file = f.id WHERE s.sensitivity IN ('public', 'private') AND s.kind = 'folder' "
             "AND (c.file IS NULL OR c.sha256 != f.sha256)")
    args = []
    if source is not None:
        query += " AND s.name = ?"
        args.append(source)
    query += " ORDER BY f.path LIMIT ?"
    args.append(limit)
    return list(store.db.execute(query, args))


def _readable(store, file_id):
    """Whether the file's source may be carded now: public or private (read again: a pass marks it at once)."""
    row = store.db.execute("SELECT s.sensitivity FROM file f JOIN source s ON s.id = f.source WHERE f.id = ?",
                           (file_id,)).fetchone()
    return row is not None and row[0] in ("public", "private")


def _prompt(store, file_id, path, body):
    names = [r[0] for r in store.db.execute(
        "SELECT DISTINCT s.name FROM symbol s JOIN part p ON p.id = s.part WHERE p.file = ? AND s.kind != 'module' "
        "ORDER BY s.name LIMIT ?", (file_id, MAX_DEFINES))]
    head = "\n".join(body.split("\n")[:MAX_HEAD_LINES])[:MAX_HEAD_CHARS]
    return PROMPT.format(path=path, defines=", ".join(names) or "(nothing)", head=head)


def write(store, settings, *, source=None, limit=None, asker=None):
    """One pass of the cards over ``store`` (see the module's doc): ``{"written", "replaced", "empty", "dropped",
    "passed", "stopped"}`` -- cards written, cards replaced (each owes a rewrite), files whose card kept no sentence,
    sentences dropped, files passed over (changed since their index, unreadable), and why the pass stopped ("" when
    it ran to its end). ``asker`` stands in for the bridge's (tests)."""
    from .store import owed_mark

    asker = asker or Asker(settings.model, settings.timeout_s, caller=CALLER)
    limit = settings.per_pass if limit is None else max(0, min(int(limit), MAX_PER_PASS))
    counts = {"written": 0, "replaced": 0, "empty": 0, "dropped": 0, "passed": 0, "stopped": ""}
    with store.lock:
        wanted = _wanted(store, source, limit)
    for file_id, root, path, sha in wanted:
        with store.lock:
            if not _readable(store, file_id):
                counts["passed"] += 1       # marked secret, or gone, since the pass was planned: never read
                continue
        data = file_bytes(root, path)
        body = None
        if data is not None and b"\0" not in data[:8192]:
            try:
                body = data.decode("utf-8")
            except UnicodeDecodeError:
                body = None
        if body is None or hashlib.sha256(data).hexdigest() != sha:
            counts["passed"] += 1       # unreadable, or changed since its index: the next index pass comes first
            continue
        with store.lock:
            prompt = _prompt(store, file_id, path, body)
        try:
            text = asker.ask(prompt, NUM_PREDICT)
        except AskError as exc:
            counts["stopped"] = str(exc)
            break
        card, dropped = checked(text, body, path)
        counts["dropped"] += dropped
        with store.lock:
            store.begin()
            try:
                row = store.db.execute("SELECT sha256 FROM file WHERE id = ?", (file_id,)).fetchone()
                if row is None or row[0] != sha or not _readable(store, file_id):
                    store.db.rollback()
                    counts["passed"] += 1       # changed, removed or marked secret while the model wrote
                    continue
                held = store.db.execute("SELECT 1 FROM card WHERE file = ?", (file_id,)).fetchone() is not None
                store.db.execute("INSERT OR REPLACE INTO card(file, sha256, model, text, dropped, at) "
                                 "VALUES (?, ?, ?, ?, ?, ?)",
                                 (file_id, sha, settings.model, card, dropped,
                                  datetime.now(timezone.utc).isoformat(timespec="seconds")))
                if held:
                    store.meta("rewrite", owed_mark())  # the old card's text stays in the pages until a rewrite
                store.db.commit()
            except BaseException:
                store.db.rollback()
                raise
        counts["replaced"] += int(held)
        if card:
            counts["written"] += 1
        else:
            counts["empty"] += 1
    if counts["stopped"]:
        logger.info("mycelium: the cards' pass stopped (%s), %d written", counts["stopped"], counts["written"])
    return counts
