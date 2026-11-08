"""What an answer keeps of its project's context: closed forms, never a name.

A turn's project context is said in passing (the executor's status lines, which the chat's loader replaces one by
one); the answer keeps its facts with it -- the source of the passages kept, their number, and the form of what the
turn said of its context --, so that the chat can say them again under the answer, and after a reload.

What is kept is closed: a source from ``SOURCES``; a count; a form from ``SAID`` with its numbers (the seconds a
search ran past, the files left to the knowledge base, counted). What the turn says can name the project's files
("... (scan.pdf) ..."); kept as it was said, a name would outlive the file's removal from the project: it is never
kept. A form this table does not know is kept as ``unknown``, without its text. The chat's loader
(``frontend/src/lib/chat/loaderWords.ts``, ``PROJECT_SAID``) holds the same forms, in the same order."""
import math
import re

# Where the passages kept came from, or why there are none: the passages' line names the first three; "cut" is a
# text fitted to its share of the window (it names no source), "withheld" a text nothing could wrap as data,
# "left_out" a retrieval the window had no room for, "none" a project turn that kept no passage.
SOURCES = ("mycelium", "rag", "mycelium+rag", "cut", "withheld", "left_out", "none")

# What the turn says of its context, each closed form whole (``ProjectContext.said``), and the numbers it keeps.
SAID = (
    (re.compile(r"\Athe project's Mycelium store cannot be read \(\w+\): no project passage\Z", re.ASCII), "unreadable", ()),
    (re.compile(r"\Athe project's Mycelium search failed \(\w+\)\Z", re.ASCII), "search_failed", ()),
    (re.compile(r"\Athe project holds files its Mycelium store does not: the knowledge base serves it\Z", re.ASCII),
     "rag_serves", ()),
    (re.compile(r"\Athe search took longer than (\d+(?:\.\d+)?) s: the lexical stages answered\Z", re.ASCII), "lexical",
     ("s",)),
    (re.compile(r"\Athe project holds (\d+) file\(s\) its Mycelium store does not \([\s\S]*\): those are left to "
                r"the knowledge base\Z", re.ASCII), "left", ("files",)),
    (re.compile(r"\Athe search took longer than (\d+(?:\.\d+)?) s: the lexical stages answered; the project holds "
                r"(\d+) file\(s\) its Mycelium store does not \([\s\S]*\): those are left to the knowledge base\Z", re.ASCII),
     "lexical_left", ("s", "files")),
)
# Each pattern is anchored at both ends (``\A``, ``\Z``), as the loader's are (``^``, ``$``): a search for one tries
# one start, whatever text it is given; and read in ASCII (``\w``, ``\d``), as the loader's are, having no flag.
FORMS = tuple(form for _pattern, form, _names in SAID) + ("unknown",)


def said_form(said: str) -> dict | None:
    """The closed form of what a turn says of its context, with its numbers; None when it says nothing; ``unknown``
    (no text) for a form the table does not know."""
    if not said:
        return None
    for pattern, form, names in SAID:
        match = pattern.fullmatch(said)
        if match:
            return {"form": form, **{n: _number(v) for n, v in zip(names, match.groups())}}
    return {"form": "unknown"}


def _number(text: str) -> int | float:
    return float(text) if "." in text else int(text)


def facts(source: str, passages: int, said: str | dict | None = None) -> dict:
    """The facts an answer keeps: ``said`` given as the turn's text (put in its form) or as a form already."""
    if source not in SOURCES:
        raise ValueError(f"no such project context source: {source!r}")
    form = said if isinstance(said, dict) or said is None else said_form(said)
    if form is not None and _said(form) != form:
        raise ValueError(f"no such said form: {form!r}")
    return {"source": source, "passages": max(int(passages), 0), "said": form}


def clean(value) -> dict | None:
    """Stored facts read back: kept only in their closed shape (a source of the table, a count, a known form and its
    numbers); anything else -- an older row, another writer -- reads as nothing."""
    if not isinstance(value, dict) or value.get("source") not in SOURCES:
        return None
    passages = value.get("passages")
    if type(passages) is not int or passages < 0:
        return None
    said = value.get("said")
    if said is None:
        return {"source": value["source"], "passages": passages, "said": None}
    kept = _said(said)
    return None if kept is None else {"source": value["source"], "passages": passages, "said": kept}


def _said(said) -> dict | None:
    """A said form in its closed shape (a known form, each of its numbers finite and not negative), or None."""
    if not isinstance(said, dict) or said.get("form") not in FORMS:
        return None
    names = next((n for _p, form, n in SAID if form == said["form"]), ())
    kept = {"form": said["form"]}
    for name in names:
        number = said.get(name)
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            return None
        if name == "files" and type(number) is not int:
            return None
        kept[name] = number
    return kept
