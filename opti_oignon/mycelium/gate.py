"""The citation gate: the citations an agent's final answer gives, checked against the lines the run's
``mycelium_search`` gave the model -- mechanically, with no model.

A citation is ``[path:first-last]`` or ``[path:line]``, the form the tool gives; a path the search showed is read as
it was given, spaces and brackets included. Each citation gets one verdict, on its PROVENANCE only:

* ``unseen``: no passage the Mycelium's search gave this run holds these lines of this path (lines the run read
  with another tool are not counted);
* ``gone``: the file cannot be read now under the root it was shown from (``index.file_lines``: never a link,
  never a path out of the root);
* ``modified``: a cited line no longer reads as it did when it was shown;
* ``empty``: the cited lines hold no word;
* ``shown`` otherwise: the citation points to lines the run was shown, unchanged.

The gate never says a claim is right. Beside the verdict it gives an ALERT: the identifiers and numbers of the
citing claim that are in none of the lines its claim cites (a word the gate does not read is not compared, so a
missing alert proves nothing). The claim of a citation is the whole sentence that holds it (an abbreviation's period
does not end one), the citations of that sentence checked together against all their lines; a citation that stands
alone carries the sentence before it, or the code block just before it. Read as words: names with an underscore, an
inner capital, a leading acronym, or dots (each part found, in the lines or the cited path); backquoted code token by
token; numbers of two digits or more and decimals, digit separators and thousands groups dropped. Not read: one-digit
integers, paths and URLs, file names, domains, abbreviations (``e.g.``), the line numbers a sentence names or a
citation's range starts and ends at, a number after ``path:``.

What a run was shown is kept by its handler (``Shown``), bounded; a run past the bound is said, never guessed. A line
counts as shown only once the agent loop handed the model the text that holds it whole -- the answer as the loop's
message holds it, clipped by its caps and defanged by the untrusted wrapper (``Shown.delivered``): an answer cut keeps
the lines before the cut or the first change, an answer that never reached the model counts nothing. A line longer
than the tool's MAX_LINE_CHARS is given cut, its cut said, and counts as given. What the engine does with a prompt
longer than its context is not seen."""
import bisect
import hashlib
import re
import threading

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

MAX_TEXT = 200_000          # characters of an answer read for citations
MAX_CHECKED = 50            # citations checked; the others counted
MAX_KEPT_LINES = 20_000     # shown lines a run keeps (a hash each)
MAX_PATHS = 2_000           # shown paths a run keeps
MAX_PENDING = 64            # answers a run's search gave that wait for the loop to say what reached the model
MAX_BODY = 200_000          # characters of a claim's cited lines read for its words
MAX_CLAIM = 2_000           # characters around a citation read for its sentence, on each side
MAX_NAMED_WORDS = 5         # missing words named in a verdict
MAX_SAID = 4000             # characters of the verdict paragraph

CITATION = re.compile(r"\[([^\[\]\s:]{1,300}):(\d{1,7})(?:-(\d{1,7}))?\]")
LINES = re.compile(r":(\d{1,7})(?:-(\d{1,7}))?\]")
BACKQUOTED = re.compile(r"`([^`\n]{1,200})`")
CODE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|\d[\d_]*(?:\.\d+)?")
IDENTIFIER = re.compile(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)+[A-Za-z_][A-Za-z0-9_]*\b|\b[A-Za-z0-9]*_[A-Za-z0-9_]+\b"
                        r"|\b[a-z]+[A-Z][A-Za-z0-9]*\b|\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b"
                        r"|\b[A-Z]{2,}[a-z]{2,}[A-Za-z0-9]*\b")
NUMBER = re.compile(r"(?<![A-Za-z0-9_.])\d[\d_]*(?:\.\d+)?")
BODY_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
BODY_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])\d[\d_]*(?:\.\d+)?")     # not the digits of a name
SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")
LINE_WORDS = re.compile(r"\b(?:lines?|lignes?|l\.)\s*\d{1,7}(?:\s*-\s*\d{1,7})?", re.IGNORECASE)
# A period after one of these does not end a sentence.
ABBREVIATIONS = frozenset({"i.e", "e.g", "etc", "cf", "vs", "resp", "approx", "ie", "eg", "fig", "no", "nr"})
# A dotted name that is a file's name is no word to find in its lines: the closed list of the suffixes the tool
# shows.
FILE_SUFFIXES = (".py", ".md", ".ts", ".js", ".json", ".yaml", ".yml", ".toml", ".txt", ".rs", ".svelte", ".sh",
                 ".html", ".css", ".cfg", ".ini", ".rst", ".tsx", ".jsx", ".sql")
FENCE = "```"

NO_CITATION = ("This answer gives no citation in the form [path:first-last]: nothing of it was checked against the "
               "passages this run's Mycelium search gave it.")
VERDICTS = ("shown", "empty", "modified", "gone", "unseen")
REASONS = {
    "unseen": "no passage this run's Mycelium search gave holds these lines (lines read with another tool are not "
              "counted)",
    "gone": "the file cannot be read now",
    "modified": "the lines changed since they were shown",
    "empty": "the cited lines hold no word",
}
# Top-level domains: a dotted word ending with one is an address, not a name of the code.
DOMAINS = frozenset({"com", "org", "net", "io", "dev", "fr", "de", "uk", "eu", "ai", "app", "edu", "gov"})
THOUSANDS = re.compile(r"(?<![\d.,])(\d{1,3})((?:[ ,]\d{3})+)(?![\d,])")


def _hash(line):
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


class Shown:
    """What one run's ``mycelium_search`` gave the model: per (root, path), the hash of each line given. A search's
    answer is OFFERED by the handler (``offer``) and its lines count only once the agent loop says what of it the
    model was given (``delivered``: the answer as the loop's message holds it): a line counts when that text holds
    it whole, its newline too; an answer the loop cut or changed keeps the lines before the cut or the change, and an
    answer never delivered counts nothing. Bounded: past MAX_KEPT_LINES lines or MAX_PATHS paths nothing more is
    kept, and ``full`` says so; past MAX_PENDING answers waiting, the oldest is dropped (it counts nothing)."""

    def __init__(self):
        self.lines = {}         # (root, path) -> {line number: hash}
        self.count = 0
        self.full = False
        self.pending = []       # (answer, [(root, path, first, lines, ends)]), oldest first
        self.lock = threading.Lock()

    def offer(self, answer, records):
        """An answer the handler gave, and its lines: ``records`` of ``(root, path, first, lines, ends)``, ``ends``
        the offset in ``answer`` just past each line as given. Nothing counts before ``delivered``."""
        with self.lock:
            self.pending.append((answer, list(records)))
            del self.pending[:-MAX_PENDING]

    def delivered(self, answer, given):
        """The loop gave the model ``given`` for the handler's ``answer``: the whole answer, or its head before a
        cut and the loop's note. The lines ``given`` holds whole are kept; an answer the handler never offered
        (another tool's text) is ignored."""
        with self.lock:
            at = next((n for n, (text, _records) in enumerate(self.pending) if text == answer), None)
            if at is None:
                return
            _text, records = self.pending.pop(at)
        held = len(answer) if given == answer else _common(answer, given)
        for root, path, first, lines, ends in records:
            # A line given whole: its newline given too, or it ends the answer given whole.
            whole = [line for line, end in zip(lines, ends) if end < held or end == held == len(answer)]
            if whole:
                self.keep(root, path, first, whole)

    def keep(self, root, path, first, lines):
        with self.lock:
            if (root, path) not in self.lines and len(self.lines) >= MAX_PATHS:
                self.full = True
                return
            held = self.lines.setdefault((root, path), {})
            for number, line in enumerate(lines, first):
                if number in held:
                    held[number] = _hash(line)
                    continue
                if self.count >= MAX_KEPT_LINES:
                    self.full = True
                    return
                held[number] = _hash(line)
                self.count += 1

    def any(self):
        with self.lock:
            return bool(self.lines)

    def paths(self):
        with self.lock:
            return {path for _root, path in self.lines}

    def find(self, path, first, last):
        """``(root, {number: hash})`` of the shown file named ``path`` that holds every line first..last, or None."""
        with self.lock:
            for (root, shown_path), held in self.lines.items():
                if shown_path == path and all(n in held for n in range(first, last + 1)):
                    return root, {n: held[n] for n in range(first, last + 1)}
        return None


def _common(a, b):
    """The length of the longest common head of ``a`` and ``b``."""
    low, high = 0, min(len(a), len(b))
    while low < high:
        mid = (low + high + 1) // 2
        if a[:mid] == b[:mid]:
            low = mid
        else:
            high = mid - 1
    return low


def citations(text, known=()):
    """The citations of ``text`` (MAX_TEXT characters read): ``(start, end, path, first, last)``, lines in order. A
    path in ``known`` (the paths the search showed) is read as it is spelled there; any other by ``CITATION``."""
    text = text[:MAX_TEXT]
    known = {path for path in known if path}
    longest = max(map(len, known), default=0)
    # Where a known path may end: just before a ``:lines]`` (no two of them overlap: each holds one colon).
    ends = [match.start() for match in LINES.finditer(text)] if known else []
    out = []
    at = text.find("[")
    while at != -1:
        found = None
        if ends:
            low = bisect.bisect_right(ends, at + 1)
            high = bisect.bisect_right(ends, at + 1 + longest)
            for end in reversed(ends[low:high]):        # the longest known path first
                if text[at + 1:end] in known:
                    lines = LINES.match(text, end)
                    found = (at, lines.end(), text[at + 1:end], lines.group(1), lines.group(2))
                    break
        if found is None:
            match = CITATION.match(text, at)
            if match:
                found = (at, match.end(), match.group(1), match.group(2), match.group(3))
        if found is not None:
            start, end, path, first, last = found
            first = int(first)
            last = int(last or first)
            if 1 <= first <= last:
                out.append((start, end, path, first, last))
                at = text.find("[", end)
                continue
        at = text.find("[", at + 1)
    return out


def _ends(text, low, high):
    """The sentence ends in text[low:high]: positions just past a ``.``, ``!`` or ``?`` before a space (not after an
    abbreviation or a single letter), or past a newline."""
    out = []
    for match in SENTENCE_END.finditer(text, low, high):
        if match.group(0) == ".":
            word = text[max(low, match.start() - 8):match.start()].split()
            word = word[-1].lower().lstrip("(") if word else ""
            if word in ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                continue
        out.append(match.end())
    return out


def _sentence(text, start, end):
    """``(low, high)``: the whole sentence that holds text[start:end]."""
    before = _ends(text, max(0, start - MAX_CLAIM), start)
    low = before[-1] if before else max(0, start - MAX_CLAIM)
    after = _ends(text, end, min(len(text), end + MAX_CLAIM))
    high = after[0] if after else min(len(text), end + MAX_CLAIM)
    return low, high


def _words_of(text):
    """The identifiers and numbers a claim's text says, in order, once each: backquoted code read token by token
    (a call's parentheses dropped), the rest by ``IDENTIFIER`` and ``NUMBER``; no path's words, no line number
    named as one, no file's name."""
    words = []
    for match in BACKQUOTED.finditer(text):
        code = match.group(1)
        if "/" in code or code.lower().rstrip(".,;:)").endswith(FILE_SUFFIXES):
            continue        # a path between backquotes
        words += [m.group(0) for m in CODE_TOKEN.finditer(_grouped(code)) if len(m.group(0)) >= 2]
    rest = _grouped(LINE_WORDS.sub(" ", BACKQUOTED.sub(" ", text)))
    kept = []
    for token in rest.split():
        if "://" in token or token.startswith(("/", "./", "../", "~/")):
            continue
        if "/" in token:
            if any(part.lower().rstrip(".,;:)").endswith(FILE_SUFFIXES) for part in token.split("/")):
                continue        # a path
            token = token.replace("/", " ")
        kept.append(token)
    rest = " ".join(kept)
    words += [m.group(0) for m in IDENTIFIER.finditer(rest)]
    words += [m.group(0) for m in NUMBER.finditer(rest) if not _after_colon(rest, m.start())]
    seen, out = set(), []
    for word in words:
        if word.lower().endswith(FILE_SUFFIXES) or word in seen:
            continue
        if word[0].isdigit() and "." not in word and len(word.replace("_", "")) < 2:
            continue        # a one-digit integer: a list's, a count's
        if "." in word and not word[0].isdigit():
            parts = word.split(".")
            if all(len(part) <= 2 for part in parts) or parts[-1].lower() in DOMAINS:
                continue    # an abbreviation (e.g., i.e.), a domain
        seen.add(word)
        out.append(word)
    return out


def _grouped(text):
    """``text`` with its thousands groups joined: 4,096 and 4 096 read 4096."""
    return THOUSANDS.sub(lambda m: m.group(1) + m.group(2).replace(",", "").replace(" ", ""), text)


def _after_colon(text, at):
    """Whether the number at ``at`` follows ``path:`` -- a file's name and a colon -- as a line number does."""
    if at <= 0 or text[at - 1] != ":":
        return False
    head = text[max(0, at - 301):at - 1].split()
    return bool(head) and head[-1].lower().endswith(FILE_SUFFIXES)


def _number(word):
    return word.replace("_", "")


def _tokens(body):
    names = set(BODY_NAME.findall(body))
    # Both readings of a body: its numbers as written (a list's "100,200") and with thousands joined ("4,096").
    numbers = {_number(n) for text in (body, _grouped(body)) for n in BODY_NUMBER.findall(text)}
    return names, numbers


def _found(word, names, numbers, path_words=frozenset()):
    """Whether a word of the claim is a whole token of the cited lines: a number with its separators dropped; a
    dotted name by each of its parts, a part found in the lines or in the cited paths (``budget.turn_budget`` of
    budget.py)."""
    if word[0].isdigit():
        return _number(word) in numbers
    if "." in word:
        return all(part in names or part in path_words for part in word.split("."))
    return word in names


def _claims(text, found):
    """The citations grouped by the sentence that holds them: ``[(indexes, claim text)]``, in order. A sentence
    whose own words are none (its citations taken out) takes the sentence before it, or the code block before it."""
    groups = []
    for index, (start, end, *_rest) in enumerate(found):
        low, high = _sentence(text, start, end)
        if groups and groups[-1][1] == (low, high):
            groups[-1][0].append(index)
        else:
            groups.append(([index], (low, high)))
    out = []
    for indexes, (low, high) in groups:
        pieces, at = [], low
        for index in indexes:
            start, end = found[index][:2]
            pieces.append(text[at:start])
            at = end
        pieces.append(text[at:high])
        claim = " ".join(pieces)
        if not re.search(r"[A-Za-z0-9]", claim):
            claim = _before(text, low, found)
        out.append((indexes, claim))
    return out


def _before(text, low, found):
    """The claim a lone citation carries: the code block that closes just before it, else the sentence before."""
    floor = max(0, low - MAX_CLAIM)
    head = text[floor:low].rstrip()
    if head.endswith(FENCE):
        opening = head.rfind(FENCE, 0, len(head) - len(FENCE))
        if opening != -1:
            return head[opening + len(FENCE):len(head) - len(FENCE)]
    stop = floor + len(head)        # past the sentence before's last character, its end mark included
    ends = _ends(text, floor, max(floor, stop - 1))
    start = ends[-1] if ends else floor
    for c_start, _c_end, *_rest in found:
        if start <= c_start < stop:
            return ""       # the sentence before has a citation of its own
    return text[start:stop]


def check(text, shown, read_lines):
    """The verdicts of ``text``'s citations against ``shown``; ``read_lines(root, path)`` gives a file's lines now
    (None when it cannot be read). ``{"checked": [...], "more": n, "full": bool}``."""
    text = text if isinstance(text, str) else ""
    found = citations(text, shown.paths())
    kept = found[:MAX_CHECKED]
    items, bodies = [], {}
    for _start, _end, path, first, last in kept:
        item = {"citation": f"{path}:{first}-{last}" if last != first else f"{path}:{first}"}
        held = shown.find(path, first, last)
        if held is None:
            item["verdict"] = "unseen"
        else:
            root, hashes = held
            lines = read_lines(root, path)
            if lines is None or last > len(lines):
                item["verdict"] = "gone" if lines is None else "modified"
            else:
                cited = lines[first - 1:last]
                if any(_hash(line) != hashes[n] for n, line in enumerate(cited, first)):
                    item["verdict"] = "modified"
                elif not any(re.search(r"\w", line) for line in cited):
                    item["verdict"] = "empty"
                else:
                    item["verdict"] = "shown"
                    bodies[len(items)] = "\n".join(cited)[:MAX_BODY]
        items.append(item)
    for indexes, claim in _claims(text, kept):
        open_ = [i for i in indexes if i in bodies]
        if not open_:
            continue
        names, numbers, path_words = set(), set(), set()
        for i in open_:
            more_names, more_numbers = _tokens(bodies[i])
            names |= more_names
            numbers |= more_numbers
            path_words |= set(re.split(r"[/.]", kept[i][2]))
        cited_numbers = set()
        for i in indexes:
            cited_numbers |= {str(kept[i][3]), str(kept[i][4])}      # a range's own ends, not the numbers inside it
        words = [w for w in _words_of(claim) if not (w[0].isdigit() and _number(w) in cited_numbers)]
        missing = [w for w in words if not _found(w, names, numbers, path_words)]
        for i in open_:
            items[i]["compared"] = len(words)
            if missing:
                items[i]["missing"] = missing[:MAX_NAMED_WORDS]
    return {"checked": items, "more": max(0, len(found) - MAX_CHECKED), "full": shown.full}


def said(result, shown_any=False):
    """The paragraph a run's final answer is given after it: how many citations point to lines the run was shown,
    unchanged, each other one with its reason; then, apart, the words of a citing sentence its cited lines do not
    hold. It never says an answer is right. When the answer cites none and the run was shown passages, that it cites
    none; else ''."""
    checked = result["checked"]
    if not checked:
        return NO_CITATION if shown_any else ""
    good = sum(1 for item in checked if item["verdict"] == "shown")
    out = [f"Citations checked against the lines this run's Mycelium search gave it: {good} of {len(checked)} point to "
           "lines the search gave, unchanged (where they point, not whether the answer is right; a line read with "
           "another tool is not counted)."]
    for item in checked:
        if item["verdict"] != "shown":
            out.append(f"- {item['citation']}: {REASONS[item['verdict']]}")
    alerts = [item for item in checked if item.get("missing")]
    if alerts:
        out.append("Words of a citing sentence that its cited lines do not hold (a mechanical alert; a word not "
                   "listed may not have been compared):")
        out += [f"- {item['citation']}: {', '.join(item['missing'])}" for item in alerts]
    if result["more"]:
        out.append(f"{result['more']} more citations were not checked.")
    if result["full"]:
        out.append("This run was given more lines than the gate keeps: a citation of the last ones reads unseen.")
    paragraph = "\n".join(out)
    return paragraph if len(paragraph) <= MAX_SAID else paragraph[:MAX_SAID] + " [...]"
