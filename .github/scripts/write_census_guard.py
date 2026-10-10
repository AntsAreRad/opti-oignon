#!/usr/bin/env python3
"""Write census guard: every write into a store a model reads back is housed, gated, exempt, or owed.

A model's context is assembled from stores: the user's facts, notes and
skills, the Core, the transcript and its summary, the response caches, the
projects and their documents, the prompt templates, the coding agent's
working memory. Whatever is written into one of them comes back to a model
later, so a write is where the platform decides what a model will be told.
The platform puts gates on those writes -- the user's approval of a skill,
the review queue that holds what the user did not type whole, the capture
that keeps typed words only. A gate protects only the writes that go through
it. This guard proves, on every commit, that every write site of every store
it knows is one of:

  (a) housed -- inside the store itself, or inside a store built on it;
  (b) gated -- behind a named gate, proven site by site;
  (c) exempt by name, with its reason, and a predicate that checks the reason
      wherever one can;
  (d) owed, in a ledger of counts by module and by store, sealed by the
      module's digest, that MAY ONLY SHRINK.

And that the list of stores is closed: every module that opens a database --
it imports sqlite3, aiosqlite, chromadb, a vector store client, or a
connection helper of the package -- is a store's house, or is named in
``OUTSIDE`` with the reason the census leaves it out; and that the methods of
a store are closed too: a public method of a store's class that runs an SQL
write, or calls one of its class's writes, is a write in the table or is set
aside with the reason no model reads what it writes.

THE STORES. ``STORES`` names each store: the module that is its house, the
classes whose objects are the store, the functions and the module-level
objects of the house that hand it out, the methods that write it, the house
functions that write it on a caller's behalf, which of its writes add no
content (a touch, a deletion), the methods that read it back, the keywords
that bind a write to its whole context, the keyword that places a new store
object somewhere of its caller's choosing, the stores it is built on, the
keywords that carry the content a filter must cover, and the methods set
aside.

WHAT COUNTS AS A SITE. A reference, called or not, to a write method on an
object the census resolves to a store, or to a house function that writes;
``getattr`` on a store object with a write method's name, or with a name it
cannot read; a private member of a store object reached outside its house,
read or not: its connection, its collection, its rows; any attribute of a
store object set or deleted outside its house, ``setattr`` and ``delattr``
on one, and what rebuilds one in place (``__init__``, ``__setattr__``,
``__dict__``); and ``getattr`` on a module that exports a write function. A
reference that is only tested -- compared, negated, or used as a condition
-- is not a site: it hands nothing on. Prose never counts.

BINDINGS FOLLOWED. Imports and aliases, absolute or relative, at module level
or in a function, in a ``try`` whose handler binds the name to ``None``; star
imports, and what a star re-exports -- a name, a module, the definition a
caller's argument reaches; module aliases and their submodules, a plain
``import a.b`` among them; a module looked up by a constant name
(``sys.modules.get``, ``sys.modules[...]``, ``modules`` imported from ``sys``,
``import_module``, relative with its package, by position or keyword,
``__import__``, which hands back the leaf with a ``fromlist`` and the top
package without, either when the ``fromlist`` is no literal), bound to a name
or to an attribute, through ``or``, conditional expressions and containers
too; ``getattr`` of a package's submodule, and a module a package or a plain
module binds to a name -- by an import, each branch of a ``try`` kept, or by a
lookup by name at its module level; a package's lazy export table, a
dictionary of name to (module, attribute) that its ``__getattr__`` serves,
read as a re-export, a name that is also a submodule read both ways; every
place a name may be defined -- both branches of a ``try``, a definition and an
import of the same name; nothing under ``if TYPE_CHECKING:``; assignments to
names and to attributes, unpacking, the walrus, ``for`` and ``with`` targets,
a match capture, ``setattr`` with a constant name, parameter defaults -- a
provider handed uncalled as one too -- and annotations that name a dependency
(``Annotated[Jot, Depends(get_jot)]``), ``*args`` and ``**kwargs`` as
containers of what they gather, ``or`` and conditional expressions,
containers and comprehensions, subscripts, an element read out of a container
of store objects, named or literal (``get``, ``pop``, ``setdefault``,
``values``, ``items``, ``copy``, ``next``), calls that hand back what they are
handed (``cast()``, ``nullcontext()``, ``enumerate``, ``zip``, ``sorted``,
``reversed``, ``list``, ``iter``, ``filter``, ``closing``, ``enter_context``
and their kind); a function, a method, a property or a lambda that returns or
yields a store object -- a dependency, a context manager, a generator; a
class's own factory (``cls()``, ``Jot.instance()``), ``super()`` in a method, an
object called through its ``__call__``, ``partial(provider)()``; a call given
such a function uncalled (a dependency provider); an argument passed to a
function, a method or a class of the same module, or to what a dispatch table
holds -- a function, a lambda, a bound method, or a function of another
module -- (a container literal bound to a name or an attribute, called through
a subscript, ``get``, or a name bound to either), or handed with a function
that is called with it later (``add_task(f, store)``, ``submit``,
``to_thread``, ``Thread(target=f, args=(store,))``, ``map``, ``partial``), or
by a decorator of the module that calls the function it decorates with it;
``getattr(x, "f")(...)`` read as ``x.f(...)``; until nothing new binds;
``self`` in a method, as an object of its class -- a store, a subclass of one,
or a carrier. Across modules: what a module binds at module level -- a store
object, a function that returns one, a class, a write function -- is what
another imports from it; an argument passed to a function or a class of
another package module binds its parameter where it is defined, re-exports
followed; an argument passed to a method of an object whose class the census
does not follow binds the parameter of every method of that name in the
modules the caller imports; and an object of any class whose methods bind
``self.x`` to a store object, or whose methods and properties hand one back,
or a class built from its fields -- a dataclass, an attrs class, a named
tuple, a model -- whose constructor is handed one, carries that store at ``x``
or behind that method, wherever the object goes, subclasses included. Names
are not scoped: a name bound to a store object anywhere in a module is one
everywhere in it, a name bound to several modules names them all, and a name
bound to a module somewhere is read too as the object it may hold elsewhere,
which can only over-count. A star import binds, once more, every name its
module reads. A module is read by its syntax tree, never by its text, when it
spells a symbol a store module exports or another module hands it a store
object, and is walked only when what it imports, or is handed, holds one; a
module read only for its imports keeps no tree.

HOUSED. A site in a method of one of the store's own classes, or in one of
its house functions or accessors, writing that store or a store it is built
on; or a module-level alias of the house that hands one of its writes out.

GATE PRESENCE IS PER SITE, BY DOMINANCE. A gate in ``GATES`` has a kind:

  * ``verdict``: a call whose truth decides. The site sits in the body of an
    ``if`` whose test is the call, or an ``and`` holding it; or after, in the
    same block or an enclosing one, an ``if`` whose test is its negation, or
    an ``or`` holding the negation, and whose body always leaves (return,
    raise, continue, break; inside a ``with``, a raise does not count, the
    context may swallow it); or in the ``else`` of an ``if`` whose test is
    its negation; or after an ``if`` whose test is the call and whose
    ``else`` always leaves. A name stands for the call when the innermost
    function that binds it -- its parameters included -- binds it exactly
    once, to the call, and no nested scope declares it ``nonlocal`` or
    ``global``;
  * ``approval``: a parameter of the function whose truth decides, read the
    same way, and never bound again;
  * ``raises``: a call that refuses by raising: a statement that is the call,
    or an assignment of it, before the site in the same block or an
    enclosing one;
  * ``filter``: a call whose result is all that is written: every positional
    argument of the site, and every keyword the store names as content, is
    the filter's call or a name bound once to it, and that name is read
    nowhere but as such an argument, in a test, or by ``len``;
  * ``decision``: a function whose body is the gate, defined once at home: a
    site in its own body is the gate's own effect -- never one in a function,
    a lambda or a generator expression nested in it, which runs whenever it
    is called -- and so is a function it binds with ``functools.partial`` and
    calls itself, at once or through a name of its own body bound once and
    only ever called there; never one it returns, keeps or hands on;
  * ``acceptance``: a decision that is the user's acceptance;
  * ``house``: a store whose class binds each write method once, through
    every block of its body and by any form, as a method that first calls
    the gate on ``self`` with a parameter, and binds the gate at most once;
    the class, like a subclass anywhere, carries no class decorator or
    metaclass and no hook through which a class answers for its attributes
    (``__getattribute__``, ``__init_subclass__``, ``__new__``...). A subclass
    anywhere has no base besides the store's class, no binding of the gate's
    name, and every write it binds calls the gate first or hands straight,
    through a ``super()`` with no arguments, to the method it overrides; an
    attribute of a store's class set anywhere is a site, and one that rebinds
    its gate or a write is no house's, in the house itself. A call of one of
    those methods on an object that is a store object on every path, and no
    module, or a ``getattr`` of one by its name, is gated by the store
    itself -- a house function that only shares a write's name, even reached
    as an attribute of the house module, a private member, or a ``getattr``
    the census cannot read is not.

The gate itself is called by its own name -- bound in the module only by its
definition at home, or only by imports that all lead to the gate -- or as an
attribute of its home module and of no other; and its home defines one
function of that name, outside ``if TYPE_CHECKING:``, and binds the name by
nothing else -- no walrus, match capture or handler, and no ``global`` of it
in a function. A decision is bound once, by its definition, decorated by
nothing but a method marker. "Before" is dominance on the syntax tree, never line order:
a gate in a branch not taken, a verdict nobody reads, or a gate in a ``try``
the site sits outside of, does not count. A function's header -- decorators,
defaults, annotations, a class's bases -- runs where it is defined, in the
scope around it, and is read there. A nested function is not gated by the
code around its definition: it runs whenever it is called, and is gated by
its own callers, though the check in its own body may read a verdict the
function around it bound once; a write inside a lambda or a generator
expression is never gated by its position. A filter, which covers data and
not control, still covers a name a nested function or a lambda reads from
the function around it, unless a name the lambda binds itself -- a
parameter, a comprehension target, a walrus -- stands in front of it. A
method is not gated by its class body; a site at module level sits in no
function and is never gated. A function is gated by its callers when every
reference to it, anywhere in the package, is a call from a gated position --
a reference read through every import and re-export of it, a star import, a
``getattr`` with its name, a decorator that is handed it (a builtin or
standard-library method marker, or the setter of a property of the same
class, excepted), and any ``x.name`` whose ``x`` the census cannot name for a
module, in a module that imports it, unless the census knows ``x`` for a
store object on every path: a name every binding of which is one -- a call
whose function, by every binding of its name, leads to a store's class, a
subclass of one or a declared accessor; a conditional whose every branch is
one -- an attribute when every assignment of it, in the carrier classes the
receiver may be an object of and their subclasses (in every class when the
census knows none) and on any other object, binds one, a class the receiver
may be built from directly besides them binds it to one wherever it binds it,
and no ``setattr``, ``__setattr__``, ``__delattr__``, ``__dict__`` or
``vars()`` reaches an object the census types as one of theirs; a container
or a subscript never. A call of a coroutine or a generator function runs
nothing: it counts only where it is awaited, iterated or drained on the spot.
A reference that
is not a call, or one call outside the gates, leaves it ungated. A gate
whose authority comes from its caller -- an approval, the Core's actor, an
acceptance -- also needs the write it stands in front of, through whatever
helper, to be reached only from the user's gesture: the body of a route
handler of a router or an application the module builds or imports, whose
attributes it never replaces, decorated by nothing else but a method marker
and called by no code but its own body (a closure the handler defines is
reached by its own callers: handed to an agent run, it is the model's to
call), or a function or a method -- never a closure -- of an entry module
the gate names. A gate is defined where ``GATES`` says, and names the contracts
that prove it, read by a closed grammar of tests that surely run: a function
``test_*``, undecorated and bound once, directly in the suite's body, or an
undecorated method ``test_*`` of an undecorated class ``Test*`` with no base,
no ``__init__`` or ``__new__``, and no ``__test__`` or ``pytestmark``; in a
suite that binds neither at module level, names no silent skip anywhere
(``skip``, ``skipif``, ``xfail``, ``importorskip``, ``SkipTest``), defines no
``pytest_generate_tests``, runs none of its own functions, raises nothing at
import and names no way to end the session (``pytest.exit``, ``sys.exit``,
``os._exit``) outside its ``__main__`` test, and rebinds no test's name (a
walrus, a handler, a match capture, a ``global``); that the selection rule
neither ignores nor deselects as pytest reads it (``addopts`` split as a shell
splits, a ``--deselect`` a prefix of node ids, letter for letter); and that
imports the gate's home or spells it in a string that is not a docstring. A
selection rule holding anything else -- another option, another pytest
configuration file or table, the tests' folder's own included, a setting that
changes what pytest collects, a ``conftest.py`` collection hook, skip,
``collect_ignore``, ``pytest_plugins`` or write to ``config.option`` -- is a
finding.

EXEMPTIONS. An exemption names its kind. Six are checked: ``route`` (every
site sits in the body of a route handler, or in a function only ever called
from one), ``unread`` (no method that reads the store back is referenced
outside its house), ``keyed`` (every site passes a keyword that binds the
write to its whole context, computed by the function that writes from what
it is handed: a key that is a constant on any path -- a name bound at module
level, a parameter with a constant default, a literal, a container, an
element of a constant container, an f-string, a digest, a conversion, a
string or container method of constants, a function of the module or a
lambda handed only constants, one side of an ``or`` or of a conditional, a
walrus or a ``with`` of one -- is no live key; an augmented assignment keeps
a live key live and a constant one constant; a name a nested scope declares
``nonlocal`` or ``global`` is no live key; and a false sentinel is no binding
when every binding of the name stands before the test of it the write sits
behind, in no loop around it, and no augmentation by a constant carries it
on; a site the predicate refuses may be argued apart, by function and
method, in ``ARGUED_SITES``), ``quiet`` (every
site adds no content: a touch, a deletion), ``script`` (every site runs only
when the module runs as a program: under its ``__main__`` test, or in a
program module no other module imports, looks up by name or runs with
``runpy``, where every function around it is reached from inside the
program alone), and ``instance`` (every write's receiver is a store object
the module built itself, from one of the store's own classes, at a place
named by the place keyword and made by ``tempfile`` -- ``mkdtemp`` or
``mkstemp``, a ``TemporaryDirectory`` bound by ``with``, ``Path`` or ``str``
of a place, or a place joined to constant relative names -- through names
bound to nothing else, never a parameter, an import, a loop, a walrus, an
augmented assignment or a pattern, and never set, deleted or rebuilt after
it was built). The others are reasons in prose: each covers the sites
``ARGUED_SITES`` lists for it by function and method -- a new site, or one
swapped for another, is a finding -- and the green line counts them apart.

THE QUESTIONS, with disjoint domains so no one can cover for another:

  * ``find_unclassified``        -- a site nobody houses, gates, exempts or owes.
  * ``find_conflicts``           -- a pair classified twice.
  * ``find_ungated``             -- a site of a gated pair no rule gates.
  * ``find_failed_exemptions``   -- a site its exemption's predicate refuses.
  * ``find_ledger_growth``       -- an owed pair with more sites than it owes.
  * ``find_broken_seals``        -- an owed module whose bytes moved.
  * ``find_stale_entries``       -- an entry, a gate or an owed count the census
                                    no longer finds.
  * ``find_unknown_gates``       -- a gate or an exemption kind no table defines,
                                    or a gate its home does not define.
  * ``find_gate_proofs_missing`` -- a gate whose contract names no test
                                    function that runs, or whose suite never
                                    names its home.
  * ``find_table_drift``         -- a store whose house does not define what the
                                    table names, a store method that writes
                                    unlisted, or a house gate not called first.
  * ``find_unclassified_stores`` -- a module that opens a database and is no
                                    house, no helper, and not outside by name.

The green line carries its denominators: the modules read and censused, the
sites by class, the pairs, the exemptions checked and argued, the debt, the
gates and the stores outside.

WHAT THE CENSUS DOES NOT SEE, said here rather than claimed covered:

  * a name assembled at run time, or read through ``vars``, ``__dict__``,
    ``globals``, ``operator.attrgetter``, ``methodcaller``, ``exec`` or
    ``eval``; a builtin rebound or partially applied (``_set = setattr``,
    ``partial(setattr, ...)``); a module looked up by a name that is not a
    constant, or through an import function renamed on import;
  * a store object handed through a dispatch table that is not a literal, or
    stored in a container or a closure of another module, or carried by an
    exception; an argument unpacked from a container (``f(*stores)``) binds
    one parameter only; and a store a decorator of another module injects;
  * a method's callers beyond its name: a method is reached by attribute
    under its name, whatever the object; and a module object handed to a
    module that does not import it, which calls a function through it, or
    an attribute bound to a module by ``setattr`` or in a module the census
    does not walk;
  * a context key computed by a function of another module, or of the
    standard library, handed only constants, when it is no digest,
    conversion or method ``keyed`` knows, or read from ambient state; a
    parameter is live by being handed, whatever its callers hand it;
  * ``__import__`` with a ``level``, and annotations evaluated lazily (Python
    3.14) or the value of ``type X = ...``, which are read as headers;
  * a write a third-party library makes on the census's behalf (a callback
    the census cannot name);
  * values: gate presence is read from the syntax, so a verdict bound once
    in a loop and read on a later turn of it, or a verdict on one value
    guarding the write of another, counts as a gate;
  * control flow inside expressions: a gate in a conditional expression or a
    short-circuit is not a gate, a refusal inside a ``with`` block before the
    site is not one either, and a verdict carried by a field of a decision
    object counts only through the ``decision`` kind;
  * a store method that writes in a way its class does not show -- neither
    an SQL write a string of its body, of its module or of its class spells,
    nor a write or a private helper of its class it reaches;
  * a store that is not a database and is not in ``STORES``: a file the
    platform writes and a model reads back is listed by hand, never found;
  * user-installed plugins under ``opti_oignon/data/``, which it never opens:
    every directory named ``data`` is pruned before listing;
  * the maintainer scripts outside the package (``scripts/``,
    ``.github/scripts/``, ``tests/``).

THREAT MODEL. The census reads the package's own code, statically, and it is
the drift of that code it catches: a change that writes around the gates, by
mistake or in haste, whatever form the write takes among those listed above.
Python can do anything at run time, and code written to hide a write from a
static reader -- a class rebuilt by a metaclass or by ``type(...)``, a name
assembled from strings, a gate patched from a module the census cannot name
-- is beyond it, and is listed here rather than claimed covered. Where the
census cannot tell, it refuses: a name it cannot prove a store object is
read as possibly the module, a proof it cannot read runs nothing. A context
key is read the other way, and said so: refused where the census proves it
constant, taken as live where it cannot, those places listed above. A
witness at run time, which sees the writes that happen, is the static
census's complement, and the next work.

The helpers are pure and import-safe; ``main`` scans the repository and exits
non-zero on any finding. Usage: ``write_census_guard.py [REPO_ROOT]``.
"""

import ast
import hashlib
import os
import re
import shlex
import sys
from collections import Counter, namedtuple
from pathlib import Path

_PACKAGE_DIR = "opti_oignon"
_PRUNED = frozenset({"data", "__pycache__"})

Store = namedtuple(
    "Store", "house classes accesses instances writes functions quiet reads keys place backs reason content aside",
    defaults=((), None),
)
Gate = namedtuple("Gate", "kind home name contracts reason entries", defaults=((),))
Site = namedtuple("Site", "store method line function")
Estate = namedtuple("Estate", "root modules unread")
Result = namedtuple("Result", "code lines estate census")


def _store(house, *, classes=(), accesses=(), instances=(), writes=(), functions=(), quiet=(), reads=(),
           keys=(), place=None, backs=(), reason="", content=(), aside=None):
    return Store(house, tuple(classes), tuple(accesses), tuple(instances), tuple(writes), tuple(functions),
                 tuple(quiet), tuple(reads), tuple(keys), place, tuple(backs), reason, tuple(content),
                 dict(aside or {}))


# ---------------------------------------------------------------------------
# The tables.
# ---------------------------------------------------------------------------
_PURGE = "it removes entries and adds nothing"
_BOOKKEEPING = "a run's bookkeeping for the history panel and its analytics; no model reads it back"
_WITHDRAWAL = ("the user's withdrawal of a source: it adds the withdrawn kind to the context of each turn the "
               "source reached and records the source; it writes no text, and a context never reaches a model")

# Every store a model reads back. A write method that adds no content is
# also named under ``quiet``. An adoption writes no text, yet it raises how
# every turn a fact is placed in is labelled: it is never quiet, so a module
# exempt as quiet cannot adopt.
STORES = {
    "facts": _store(
        "opti_oignon/memory/dedup.py", classes=("MemoryStore",), accesses=("get_memory_store",),
        writes=("add", "update", "touch", "soft_delete", "restore", "hard_delete", "adopt"),
        quiet=("touch", "soft_delete", "hard_delete"), backs=("facts_canonical", "facts_vectors"),
        reason="the user's facts, composed into the memory block of every turn",
    ),
    "facts_canonical": _store(
        "opti_oignon/memory/canonical_store.py", classes=("CanonicalMemoryStore",),
        accesses=("get_canonical_store",),
        writes=("add", "update", "touch", "soft_delete", "restore", "hard_delete", "clear",
                "apply_synced_memory_canonical", "adopt"),
        quiet=("touch", "soft_delete", "hard_delete", "clear"),
        reason="the rows behind the facts, which the retriever reads",
    ),
    "facts_vectors": _store(
        "opti_oignon/memory/vector_store.py", classes=("MemoryVectorStore",), accesses=("get_vector_store",),
        writes=("add", "update", "delete", "clear"), quiet=("delete", "clear"),
        reason="the embeddings behind the facts, which the retriever searches",
    ),
    "legacy": _store(
        "opti_oignon/memory/legacy.py", classes=("MemoryManager",), instances=("memory_manager",),
        writes=("add_fact", "update_fact", "activate_fact", "deactivate_fact", "delete_fact", "clear_all",
                "deduplicate", "extract_and_store"),
        quiet=("deactivate_fact", "delete_fact", "clear_all", "deduplicate"), place="db_path",
        reason="the frozen flat memory, read as the memory block's fallback and through the retriever's bridge",
    ),
    "extraction": _store(
        "opti_oignon/memory/extraction.py", classes=("FactExtractor",), accesses=("get_extractor",),
        writes=("extract_and_store", "aextract_and_store"), functions=("extract_and_store", "schedule_extraction"),
        backs=("facts",), content=("messages",),
        reason="the extractor, which stores the facts a model draws from the messages it is given",
    ),
    "notes": _store(
        "opti_oignon/notes/notes_store.py", classes=("NotesStore",), accesses=("get_notes_store",),
        writes=("add_note", "update_note", "delete_note", "add_attachment", "update_attachment",
                "delete_attachment", "apply_synced_note"),
        quiet=("delete_note", "delete_attachment"),
        aside={"set_mobile_allowed": "a note's permission flag for the phone; it carries no content"},
        reason="the user's notes and their attachments' text, which the agent reads",
    ),
    "skills": _store(
        "opti_oignon/agent/skills.py", classes=("SkillRegistry",), accesses=("get_skill_registry",),
        writes=("add", "update", "patch", "publish", "delete", "apply_synced_skill", "adopt_synced",
                "write_accepted", "publish_draft", "delete_named", "adopt"),
        quiet=("delete", "delete_named"),
        reason="the skills, whose text re-enters the prompt once admitted: written here by hand, or its bytes "
               "named by their digest on this device",
    ),
    "core": _store(
        "opti_oignon/memory/core_store.py", classes=("CoreStore",), writes=("add", "supersede"),
        reason="the Core: the entries the user pinned, read into every turn of the conversation",
    ),
    "peels": _store(
        "opti_oignon/memory/peels.py", classes=("PeelTree",), writes=("add",),
        reason="the onion's peels: the librarian model's summaries of evicted spans, read into the memory block "
               "when the onion is enabled",
    ),
    "receipts": _store(
        "opti_oignon/memory/receipts.py", classes=("ReceiptLedger",), writes=("append", "supersede"),
        reason="the onion's receipts: pointers into the evicted spans, rendered into the memory block when the "
               "onion is enabled",
    ),
    "cellar": _store(
        "opti_oignon/memory/receipts.py", classes=("Cellar",), writes=("store",),
        reason="the onion's cellar: the evicted spans, verbatim, whose anchors enter the memory block when the "
               "onion is enabled",
    ),
    "flesh": _store(
        "opti_oignon/memory/receipts.py", classes=("Flesh",),
        writes=("append", "take_back", "evict_oldest", "evict_span", "evict_until_fits"),
        quiet=("take_back", "evict_oldest", "evict_span", "evict_until_fits"), backs=("cellar", "receipts"),
        reason="the onion's live turns, mirrored from the conversation until they are evicted",
    ),
    "conversation": _store(
        "opti_oignon/conversation.py", classes=("ConversationManager",), instances=("conversation_manager",),
        writes=("add_message", "apply_synced_conversation", "create_conversation", "update_conversation_metadata",
                "delete_last_message", "delete_conversation", "migrate_json_history"),
        functions=("add_message", "create_conversation", "delete_last_message"),
        quiet=("delete_last_message", "delete_conversation"), place="db_path",
        aside={
            "rename_conversation": "the title, shown in the lists and the exports; no model reads it back",
            "withdraw_source": _WITHDRAWAL,
        },
        reason="the transcript and its metadata, the summary of its older messages included, read back as "
               "each turn's history",
    ),
    "branches": _store(
        "opti_oignon/conversation_branches.py", classes=("ConversationBranchManager",),
        writes=("fork", "add_branch_message", "update_branch", "merge_messages", "delete_branch",
                "delete_all_branches"),
        quiet=("delete_branch", "delete_all_branches"), place="db_path",
        aside={"withdraw_source": _WITHDRAWAL},
        reason="the conversation's branches, read back as the history of the branch the user is on",
    ),
    "response_cache": _store(
        "opti_oignon/response_cache.py", classes=("ResponseCache",), instances=("response_cache",),
        writes=("put", "warm"), keys=("explicit_key",), place="db_path",
        aside={
            "get": "a read that counts its hit and its last use",
            "invalidate": _PURGE, "invalidate_model": _PURGE, "clear": _PURGE,
        },
        reason="answers replayed for the same request",
    ),
    "semantic_cache": _store(
        "opti_oignon/semantic_cache.py", classes=("SemanticCache",), instances=("semantic_cache",),
        writes=("put", "store_embedding", "put_with_embedding"), keys=("context_fingerprint",), place="db_path",
        backs=("response_cache",),
        aside={
            "get": "a read that counts its hit and its last use",
            "update_config": "the cache's own settings; no content",
            "invalidate": _PURGE, "expire_stale": _PURGE, "remove_embedding": _PURGE,
            "remove_embeddings_for_model": _PURGE, "clear": _PURGE, "cleanup_orphans": _PURGE,
        },
        reason="answers replayed for a similar request under the same context",
    ),
    "coding_memory": _store(
        "opti_oignon/coding_history.py", classes=("CodingHistoryStore",),
        writes=("save_working_memory", "record_checkpoint"), reads=("load_working_memory", "get_last_checkpoint"),
        aside={
            "record_task_start": "the user's own task text and the run's start, shown in the history panel and "
                                 "handed back with a checkpoint on resume: the user's words, unchanged",
            "update_task_status": _BOOKKEEPING, "record_step": _BOOKKEEPING, "record_test": _BOOKKEEPING,
            "delete_working_memory": _PURGE, "delete_task": _PURGE, "prune": _PURGE,
            "batch_delete_by_ids": _PURGE, "batch_delete_before_date": _PURGE,
        },
        reason="the coding agent's working memory and checkpoints, kept for a resume",
    ),
    "projects": _store(
        "opti_oignon/projects.py", classes=("ProjectStore",), instances=("project_store",),
        writes=("create_project", "update_project", "delete_project", "add_file", "remove_file", "add_output",
                "remove_output", "link_conversation", "unlink_conversation"),
        quiet=("delete_project", "remove_file", "remove_output", "unlink_conversation"),
        reason="projects: their system instructions and files, read into the turns of a linked conversation",
    ),
    "project_index": _store(
        "opti_oignon/project_context.py", classes=("ProjectIndexer",), instances=("project_indexer",),
        writes=("index_file", "reindex_project", "remove_file_from_index", "delete_project_index"),
        quiet=("remove_file_from_index", "delete_project_index"),
        reason="the chunks of a project's files, retrieved into the turns of a linked conversation",
    ),
    "documents": _store(
        "opti_oignon/rag_store.py", classes=("RAGVectorStore",), accesses=("get_rag_store",),
        writes=("ingest_file", "ingest_text", "ingest_url", "store_chunked", "delete_document",
                "delete_collection"),
        quiet=("delete_document", "delete_collection"),
        reason="the ingested documents, retrieved into an augmented prompt",
    ),
    "document_index": _store(
        "opti_oignon/rag/indexer.py", classes=("DocumentIndexer",), functions=("quick_index",),
        writes=("index_file", "index_directory", "remove_file", "clear_index"),
        quiet=("remove_file", "clear_index"),
        reason="the indexed chunks the document retriever searches",
    ),
    "prompt_templates": _store(
        "opti_oignon/prompt_optimization.py", classes=("PromptTemplateEngine",),
        instances=("prompt_template_engine",),
        writes=("set_runtime_override", "clear_runtime_override", "clear_all_runtime_overrides"),
        quiet=("clear_runtime_override", "clear_all_runtime_overrides"),
        reason="the system prompt templates, with the overrides set at run time",
    ),
}

# The gates. A kind, the module that defines it, the name it is called or
# defined by there, and the contracts that prove it.
GATES = {
    "typed.turns": Gate(
        "filter", "opti_oignon/memory/auto_capture.py", "typed_turns", ("ac1", "pw8"),
        "facts are drawn from the words the user typed, and from nothing else",
    ),
    "review.endorsement": Gate(
        "decision", "opti_oignon/pending_writes.py", "WriteGate.write", ("pw1", "pw2"),
        "an agent write goes through only when the user typed its words whole; anything else is proposed",
    ),
    "review.acceptance": Gate(
        "acceptance", "opti_oignon/pending_writes.py", "accept", ("pw5", "pw14", "ap11", "ap18"),
        "a proposal is written only once the user accepts it, in a route or at the terminal; a skill only by the "
        "digest of the text shown, its bytes hashed again as they are written",
        ("opti_oignon/cli/session.py",),
    ),
    "review.recovery": Gate(
        "decision", "opti_oignon/pending_writes.py", "recover", ("pw27",),
        "an acceptance cut short is finished, once, never redone",
    ),
    "peel.eviction": Gate(
        "decision", "opti_oignon/memory/peels.py", "evict_gated", ("gf1", "gf51"),
        "a span leaves the live turns, and its peel enters the tree, only when the peel answers its probes",
    ),
    "peel.ladder": Gate(
        "decision", "opti_oignon/memory/peels.py", "advance", ("dv44", "dv53", "oq8", "oq10"),
        "the eviction ladder keeps a peel only once the gate accepts it, faithful and holding no order but the "
        "user's own; otherwise the span stays verbatim in the cellar",
    ),
    "peel.leaf": Gate(
        "decision", "opti_oignon/memory/peels.py", "build_leaf", ("gf14", "dv61"),
        "a leaf peel enters the tree only when the gate accepts it",
    ),
    "peel.parent": Gate(
        "decision", "opti_oignon/memory/peels.py", "build_parent", ("gf14", "dv61"),
        "a parent peel enters the tree only when the gate accepts it",
    ),
    "core.user": Gate(
        "house", "opti_oignon/memory/core_store.py", "_require_user", ("rk3",),
        "the Core refuses every writer but the user, by the actor its caller names; only a route or the "
        "terminal client reaches a Core write",
        ("opti_oignon/cli/session.py",),
    ),
}

# (module, store) -> the gates its sites pass.
GATED = {
    ("opti_oignon/api/routes_memory.py", "extraction"): ("typed.turns",),
    ("opti_oignon/memory/auto_capture.py", "extraction"): ("typed.turns",),
    ("opti_oignon/memory/librarian.py", "core"): ("core.user",),
    ("opti_oignon/memory/peels.py", "flesh"): ("peel.eviction", "peel.ladder"),
    ("opti_oignon/memory/peels.py", "peels"): ("peel.eviction", "peel.ladder", "peel.leaf", "peel.parent"),
    ("opti_oignon/pending_writes.py", "facts"): ("review.endorsement", "review.acceptance", "review.recovery"),
    ("opti_oignon/pending_writes.py", "notes"): ("review.endorsement", "review.acceptance", "review.recovery"),
    ("opti_oignon/pending_writes.py", "skills"): ("review.endorsement", "review.acceptance", "review.recovery"),
}

_TRANSCRIPT = "the conversation's own messages, written as they happen and read back by role, each with its origin"
_RESTORE = "it rebuilds the onion from its own saved snapshot, every row re-hashed against its id: nothing new"
_ROUTE = "the user's own gesture, made in the interface"
_INSTANCE = "it times or plants into store objects of its own, at a place it names, never the user's"

# (module, store) -> (kind, reason).
EXEMPT = {
    ("opti_oignon/agent_eval/fidelity.py", "conversation"): (
        "evaluation", "the needle evaluation plants its haystacks in a conversation store at the path its caller "
                      "gives, with sync publishing off; the user's own store is never its default"),
    ("opti_oignon/agentic_executor.py", "conversation"): ("transcript", _TRANSCRIPT),
    ("opti_oignon/api/routes_agent.py", "skills"): (
        "route", "the user publishes a draft, deletes a draft or a skill, or adopts a skill's bytes, each named by "
                 "the digest of what they were shown"),
    ("opti_oignon/api/routes_branches.py", "branches"): ("route", _ROUTE),
    ("opti_oignon/api/routes_cache.py", "semantic_cache"): (
        "route", "the user turns the cache on or off from its settings route; the switch adds no content"),
    ("opti_oignon/api/routes_chat.py", "conversation"): ("route", _ROUTE),
    ("opti_oignon/api/routes_conversations.py", "conversation"): ("route", _ROUTE),
    ("opti_oignon/api/routes_memory.py", "facts"): ("route", "the user adds, edits, archives and restores their facts"),
    ("opti_oignon/api/routes_notes.py", "notes"): ("route", _ROUTE),
    ("opti_oignon/api/routes_notes_attachments.py", "notes"): ("route", _ROUTE),
    ("opti_oignon/api/routes_projects.py", "project_index"): ("route", _ROUTE),
    ("opti_oignon/api/routes_projects.py", "projects"): ("route", _ROUTE),
    ("opti_oignon/api/routes_prompt.py", "prompt_templates"): ("route", "the user sets or clears a template override"),
    ("opti_oignon/api/routes_rag.py", "documents"): ("route", "the user ingests or deletes a document they chose"),
    ("opti_oignon/artifacts.py", "conversation"): (
        "display", "it keeps the artifact list the artifact routes serve for download; no model reads it back"),
    ("opti_oignon/chat_coding_agent.py", "conversation"): ("transcript", _TRANSCRIPT),
    ("opti_oignon/cli/session.py", "conversation"): (
        "gesture", "the terminal client opens a conversation when its user starts one"),
    ("opti_oignon/cli/session.py", "skills"): (
        "gesture", "the user adopts a skill's bytes not admitted here, named by their digest, on the command "
                   "line"),
    ("opti_oignon/cli/session.py", "facts"): (
        "gesture", "the user adopts a fact of memory, its text named by its digest, on the command line"),
    ("opti_oignon/conversation.py", "conversation"): ("script", "the module's self-test, run only as a program"),
    ("opti_oignon/conversation_wipe.py", "conversation"): ("quiet", "the wipe deletes conversations"),
    ("opti_oignon/executor.py", "conversation"): (
        "transcript", "the turn's messages, their model and task type, and the summary of the conversation's own "
                      "older messages, read back in their place"),
    ("opti_oignon/executor.py", "response_cache"): (
        "keyed", "an answer is replayed only under the same model, system prompt and history"),
    ("opti_oignon/executor.py", "semantic_cache"): (
        "keyed", "an answer is replayed only under the fingerprint of the same assembled context; the cascade and "
                 "the speculative paths, argued apart, key theirs to the no-context sentinel: they hand the model "
                 "the question alone (cascade(query=question, task_type=...), generate(query=question, ...))"),
    ("opti_oignon/memory/curation.py", "facts"): ("quiet", "curation touches and retires facts; it never adds one"),
    ("opti_oignon/source_withdrawal.py", "facts"): (
        "quiet", "the user's withdrawal of a fact sets it aside, restorable; it never adds one"),
    ("opti_oignon/memory/migration.py", "facts"): (
        "migration", "it moves the legacy rows into the store, at start-up or when the user asks again: nothing new"),
    ("opti_oignon/memory/librarian.py", "flesh"): (
        "transcript", "the onion mirrors the conversation's own turns, and takes back the turns it took back"),
    ("opti_oignon/memory/librarian.py", "peels"): (
        "copy", "it copies the peels already in the tree, less the superseded, into a view for one composition"),
    ("opti_oignon/memory/librarian.py", "receipts"): (
        "transcript", "taking back the conversation's last turns supersedes the receipts that pointed at them"),
    ("opti_oignon/memory/onion_store.py", "cellar"): ("restore", _RESTORE),
    ("opti_oignon/memory/onion_store.py", "core"): ("restore", _RESTORE),
    ("opti_oignon/memory/onion_store.py", "flesh"): ("restore", _RESTORE),
    ("opti_oignon/memory/onion_store.py", "peels"): ("restore", _RESTORE),
    ("opti_oignon/memory/onion_store.py", "receipts"): ("restore", _RESTORE),
    ("opti_oignon/project_context.py", "projects"): (
        "gesture", "it stores the extractive summary and key terms of a file the user added to the project"),
    ("opti_oignon/rag_hybrid_search.py", "documents"): (
        "reading", "the keyword search reads the chunks through the store's collection; it writes nothing there"),
    ("opti_oignon/memory/retrieval.py", "facts_canonical"): ("quiet", "retrieval touches the facts it read"),
    ("opti_oignon/performance_benchmark.py", "conversation"): ("instance", _INSTANCE),
    ("opti_oignon/performance_benchmark.py", "legacy"): ("instance", _INSTANCE),
    ("opti_oignon/performance_benchmark.py", "response_cache"): ("instance", _INSTANCE),
    ("opti_oignon/performance_benchmark.py", "semantic_cache"): ("instance", _INSTANCE),
    ("opti_oignon/rag/__main__.py", "document_index"): ("script", "the document indexer's command line"),
    ("opti_oignon/rag/augmenter.py", "document_index"): (
        "gesture", "it indexes the folder the user names on the command line (main.py, cmd_rag)"),
    ("opti_oignon/rag/batch_ingest.py", "documents"): (
        "gesture", "it stores the files of a batch the user started on a folder they chose"),
    ("opti_oignon/rag/indexer.py", "document_index"): ("script", "the indexer's self-run, only as a program"),
    ("opti_oignon/user_data_manager.py", "conversation"): ("quiet", "the user's own erasure"),
    ("opti_oignon/user_data_manager.py", "documents"): ("quiet", "the user's own erasure"),
    ("opti_oignon/user_data_manager.py", "facts_canonical"): ("quiet", "the user's own erasure"),
    ("opti_oignon/user_data_manager.py", "facts_vectors"): ("quiet", "the user's own erasure"),
}

# The sites each exemption argued in prose covers, by function and method. No
# predicate checks a reason in prose, so it covers the sites it was written
# for, and no other: a new site is a finding, so is a site swapped for
# another, and a site the census no longer finds is taken off.
ARGUED_SITES = {
    ("opti_oignon/agent_eval/fidelity.py", "conversation"): (
        ("_plant", "add_message"), ("_plant", "create_conversation"),),
    ("opti_oignon/agentic_executor.py", "conversation"): (
        ("AgenticExecutor._save_to_conversation", "add_message"),
        ("AgenticExecutor._save_to_conversation", "add_message"),),
    ("opti_oignon/artifacts.py", "conversation"): (
        ("ArtifactManager._save_to_metadata", "update_conversation_metadata"),),
    ("opti_oignon/chat_coding_agent.py", "conversation"): (
        ("ChatCodingSession._save_turn_to_conversation", "add_message"),
        ("ChatCodingSession._save_turn_to_conversation", "add_message"),),
    ("opti_oignon/cli/session.py", "conversation"): (
        ("_default_new_conversation", "create_conversation"),),
    ("opti_oignon/cli/session.py", "skills"): (
        ("ChatSession._adopt", "adopt"),),
    ("opti_oignon/cli/session.py", "facts"): (
        ("ChatSession._adopt_memory", "adopt"),),
    ("opti_oignon/executor.py", "semantic_cache"): (
        ("Executor.execute_cascade", "put"), ("Executor.execute_speculative", "put"),),
    ("opti_oignon/executor.py", "conversation"): (
        ("Executor._summarize_old_messages", "update_conversation_metadata"), ("Executor.execute", "add_message"),
        ("Executor.execute", "add_message"), ("Executor.execute", "add_message"),
        ("Executor.execute", "add_message"), ("Executor.execute", "update_conversation_metadata"),),
    ("opti_oignon/memory/librarian.py", "flesh"): (
        ("OnionState._take_back", "take_back"), ("OnionState._take_back", "take_back"),
        ("OnionState.mirror", "append"),),
    ("opti_oignon/memory/librarian.py", "peels"): (
        ("_compose_block", "add"),),
    ("opti_oignon/memory/librarian.py", "receipts"): (
        ("OnionState._take_back", "supersede"),),
    ("opti_oignon/memory/migration.py", "facts"): (
        ("migrate_legacy_to_store", "add"),),
    ("opti_oignon/memory/onion_store.py", "cellar"): (
        ("_rebuild", "store"),),
    ("opti_oignon/memory/onion_store.py", "core"): (
        ("_rebuild", "add"), ("_rebuild", "supersede"),),
    ("opti_oignon/memory/onion_store.py", "flesh"): (
        ("_rebuild", "append"),),
    ("opti_oignon/memory/onion_store.py", "peels"): (
        ("_rebuild", "add"),),
    ("opti_oignon/memory/onion_store.py", "receipts"): (
        ("_rebuild", "append"),),
    ("opti_oignon/project_context.py", "projects"): (
        ("ProjectIndexer._update_file_record", "_get_conn"),),
    ("opti_oignon/rag/augmenter.py", "document_index"): (
        ("ContexteurRAGIntegration.clear", "clear_index"),
        ("ContexteurRAGIntegration.index_folder", "index_directory"),),
    ("opti_oignon/rag/batch_ingest.py", "documents"): (
        ("BatchIngestEngine._store_file", "store_chunked"),),
    ("opti_oignon/rag_hybrid_search.py", "documents"): (
        ("HybridSearchEngine._keyword_search", "_build_where"), ("HybridSearchEngine._keyword_search", "_chroma"),),
}

# Debt found when this guard was written: module -> its seal and its counts by
# store. MAY ONLY SHRINK: a new site of an owed store, or of a store it did not
# owe, is a finding, and so is a count the census no longer finds. Every owed
# module carries the digest of its text as the debt was enumerated: touch it,
# and you pay it.
LEDGER = {
    # A checkpoint's plan, written by the coding agent's model, is handed back
    # to the interface when the user resumes the task, which may give it to
    # the agent again: its provenance travels with nothing (three sites). The
    # fourth, the working memory, is read by nothing, and owed with them.
    "opti_oignon/coding_agent.py": {
        "seal": "4aabdb7a7c4bd4a3d9f471e23a506d78ab46f865ff4dffdd184305389e66ea3f",
        "owes": {"coding_memory": 4},
    },
    # A caption, or a transcript, is written back on a request that carries the
    # user's approval, and that request runs the model again: the text written
    # is not the text the user approved (contracts wb1, wb2 hold the approval
    # itself).
    "opti_oignon/notes/caption.py": {
        "seal": "bcd0ffdf995e6a452b07e753eea49388a25e25416648f8b45c69dcf09ce50590",
        "owes": {"notes": 1},
    },
    "opti_oignon/notes/transcription.py": {
        "seal": "b08eaa0727e6679b1bfdb60cca51a908c7ea54000babdfb22024d6f945789922",
        "owes": {"notes": 1},
    },
    # The pre-cache stores answers to its common queries with no context
    # fingerprint, so an answer may be replayed under any context.
    "opti_oignon/pre_cache.py": {
        "seal": "e1f010eb9d7101d90a890ce846d9c9f8df2108a04f79874f0793f6079ff709ba",
        "owes": {"semantic_cache": 1},
    },
    # The auto-refresh re-ingests a document whose file changed on disk, with
    # no gate: a file changed by anything lands in the documents.
    "opti_oignon/rag_dashboard.py": {
        "seal": "b9de3eddf8a464e346f7f33bd2a87af85671f595ac763b8a2f39db6b8bcbcd0f",
        "owes": {"documents": 2},
    },
    # A synced fact, note or skill lands as its peer sent it: the gate that
    # admitted it there travels with nothing. One more site of facts and of
    # notes is an over-count the census owns, not a debt of the module: the
    # engine reads ``getattr(self._store, "_root")`` on its peer store, typed
    # a facts and notes store because module-level helpers bind a local of the
    # same name to those stores (names are not scoped). A typing that scopes a
    # parameter to its own function would take them off.
    "opti_oignon/veilid/sync_engine.py": {
        "seal": "eed2e0e669ae6fe9725272b8304c8047278670c2a97e9583bf7318bcb6c839f2",
        "owes": {"conversation": 1, "facts_canonical": 2, "notes": 2, "skills": 1},
    },
}

_UNREAD = "no read of it reaches a model's context"

# Modules that open a database and are no store's house: the reason the
# census leaves each out.
OUTSIDE = {
    "opti_oignon/admin_audit.py": f"the administrators' audit events; {_UNREAD}",
    "opti_oignon/agent_eval/store.py": f"the evaluation runs and their scores; {_UNREAD}",
    "opti_oignon/allium/service.py": f"the garden simulation's service, over its own state; {_UNREAD}",
    "opti_oignon/allium/store.py": f"the garden simulation's state, shown read-only; {_UNREAD}",
    "opti_oignon/analytics.py": f"routing and latency metrics for the dashboard; {_UNREAD}",
    "opti_oignon/api/deps.py": "it wires the routes' dependencies and opens nothing of its own",
    "opti_oignon/api/routes_security.py": "the security administration routes; they hold no content",
    "opti_oignon/audit_anchor_export.py": f"signs and checks audit anchors; {_UNREAD}",
    "opti_oignon/auth.py": f"users, sessions and tokens; {_UNREAD}",
    "opti_oignon/auth_2fa.py": f"second-factor secrets; {_UNREAD}",
    "opti_oignon/benchmark_history.py": f"benchmark runs for the dashboard; {_UNREAD}",
    "opti_oignon/benchmark_judge.py": "the judge's scores; its prompt is built from its arguments, never from the store",
    "opti_oignon/benchmark_recommendations.py": f"benchmark recommendations for the dashboard; {_UNREAD}",
    "opti_oignon/benchmark_runner.py": f"benchmark runs for the dashboard; {_UNREAD}",
    "opti_oignon/context_ledger.py": f"the per-request routing ledger for the dashboard; {_UNREAD}",
    "opti_oignon/feedback.py": f"the user's ratings; read by the dashboard and an offline export; {_UNREAD}",
    "opti_oignon/fine_tune_tracker.py": f"fine-tune variants and comparisons; {_UNREAD}",
    "opti_oignon/humanizer.py": "the humanizer's ratings; its prompt is built from the text it is given",
    "opti_oignon/learned_router.py": "routing samples; the router predicts from its trained model, not from the rows",
    "opti_oignon/memory/ledger_store.py": "a fact ledger nothing imports: no writer and no reader",
    "opti_oignon/memory/onion_store.py": (
        "the onion's own snapshot, written only by the librarian and re-hashed on every load; its writes into "
        "the Core are censused under the Core"),
    "opti_oignon/notes/note_updates_store.py": "the editor's replay log for a note; the note itself is read, not its log",
    "opti_oignon/pending_writes.py": (
        "the review queue: what the agent proposed, shown to the user; no model reads it back, and its writes "
        "into the facts and the notes are censused under them"),
    "opti_oignon/performance_monitor.py": f"execution metrics for the dashboard; {_UNREAD}",
    "opti_oignon/plugin_index.py": f"the plugin marketplace index; {_UNREAD}",
    "opti_oignon/plugin_manifest.py": f"the plugin registry; {_UNREAD}",
    "opti_oignon/plugin_reviews.py": f"plugin ratings; {_UNREAD}",
    "opti_oignon/plugin_user_config.py": f"per-user plugin settings; {_UNREAD}",
    "opti_oignon/plugins/github-connector/entry_point.py": (
        "the connector's account token, sent to GitHub only; its hook replies under a key the chat never reads"),
    "opti_oignon/plugins/scratchpad/entry_point.py": (
        "the plugin's own notes; its hook replies under a key the chat never reads"),
    "opti_oignon/plugins/task-extractor/entry_point.py": (
        "the plugin's extracted tasks; its hook replies under a key the chat never reads"),
    "opti_oignon/rag/batch_ingest.py": "ingestion job status, no text; the documents it stores are censused under them",
    "opti_oignon/rag/pool_integration.py": "a connection pool; it holds no content",
    "opti_oignon/rag/retriever.py": "it reads the documents; it writes nothing",
    "opti_oignon/rag_dashboard.py": (
        "its own refresh bookkeeping; the documents its auto-refresh re-ingests are censused under them"),
    "opti_oignon/rag_external.py": (
        "connectors to vector stores the user fills by their own means; the platform only connects to and "
        "queries them, and writes nothing there"),
    "opti_oignon/rag_sanitizer.py": f"the log of chunks it flagged; {_UNREAD}",
    "opti_oignon/resource_governor.py": "admission costs, read by the governor to admit a model, never by a model",
    "opti_oignon/sandbox_manager.py": f"the sandbox audit log; {_UNREAD}",
    "opti_oignon/session_fingerprint.py": "a coding session's fingerprint; its compact form, built for injection, has "
                                          "no caller",
    "opti_oignon/signed_audit_log.py": f"the signed audit chain; {_UNREAD}",
    "opti_oignon/sync_queue.py": "the user's own queries queued offline; no path resubmits them to a model",
    "opti_oignon/telemetry_history.py": f"latency and usage telemetry; {_UNREAD}",
    "opti_oignon/user_isolation.py": f"per-user interface settings; {_UNREAD}",
    "opti_oignon/user_key_manager.py": "key salts and derived keys, kept inside the encryption layer",
    "opti_oignon/veilid/change_feed.py": "the sync change log; what it carries lands in the stores censused here",
    "opti_oignon/veilid/deferred_ledger.py": "sync records held back; what lands is censused where it lands",
    "opti_oignon/veilid/peers.py": "peer identities and grants, used to authorize, never shown to a model",
}

# The connection helpers themselves: they open what their callers name.
HELPERS = {
    "opti_oignon/db_utils.py": "the connection helper every store opens its database through",
    "opti_oignon/connection_pool.py": "the connection pool the stores borrow connections from",
    "opti_oignon/db_encryption.py": "the encrypted connection the stores open their database with",
}

_DATABASE_LIBRARIES = frozenset({"sqlite3", "chromadb", "pysqlcipher3", "sqlcipher3", "aiosqlite", "qdrant_client",
                                 "weaviate", "pinecone", "lancedb", "duckdb"})
_HELPER_MODULES = frozenset({"db_utils", "connection_pool", "db_encryption"})

_GATE_KINDS = frozenset({"verdict", "approval", "raises", "filter", "decision", "acceptance", "house"})
_AUTHORITY = frozenset({"approval", "acceptance", "house"})
_CHECKED = frozenset({"route", "unread", "keyed", "quiet", "script", "instance"})
_ARGUED = frozenset({"transcript", "migration", "restore", "gesture", "display", "reading", "copy", "evaluation"})
_ROUTE_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options", "api_route", "websocket"})
# Calls that hand back what they are handed, or an element of it.
_PASS_THROUGH = frozenset({"cast", "nullcontext", "enumerate", "zip", "sorted", "reversed", "list", "tuple", "set",
                           "frozenset", "iter", "filter", "dict", "closing", "enter_context", "enter_async_context",
                           "chain", "islice", "deque", "copy", "deepcopy", "proxy", "ChainMap"})
# Decorators and bases whose classes take their fields as constructor arguments.
_FIELD_DECORATORS = frozenset({"dataclass", "define", "frozen", "mutable", "attrs"})
_FIELD_BASES = frozenset({"NamedTuple", "BaseModel", "TypedDict"})
_CONTAINER_READS = frozenset({"get", "pop", "setdefault", "values", "items", "copy"})
_CONTAINERS = (ast.Dict, ast.List, ast.Tuple, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_BINDERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.Assign, ast.AnnAssign,
            ast.AugAssign, ast.NamedExpr, ast.For, ast.AsyncFor, ast.comprehension, ast.withitem, ast.Call, ast.Match)
_KINDS = ("handle", "class", "returner", "sink")
_BLOCKS = ("body", "orelse", "finalbody")
_LEAVES = (ast.Return, ast.Raise, ast.Continue, ast.Break)


def digest(text):
    """The seal of a module, taken on the text this guard reads."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Syntax helpers.
# ---------------------------------------------------------------------------
def _leaf(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _target_names(target):
    """The names and the attributes an assignment target binds."""
    names, attributes = set(), set()
    stack = [target]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Subscript):
            node = node.value
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            attributes.add(node.attr)
        elif isinstance(node, (ast.Tuple, ast.List)):
            stack.extend(node.elts)
        elif isinstance(node, ast.Starred):
            stack.append(node.value)
    return names, attributes


def _parameters(args):
    """The positional parameters, then the keyword-only ones, of a signature."""
    return list(getattr(args, "posonlyargs", [])) + list(args.args), list(args.kwonlyargs)


def _name_bindings(index):
    """name -> [(kind, value)] for every binding of it in a module, by any form Python has; names are not scoped.

    ``value`` is the expression bound by an assignment, a walrus or a ``with``
    item, the statement for an import, and the binding node otherwise. A bare
    annotation binds nothing.
    """
    if "bindings" in index:
        return index["bindings"]
    out = {}

    def bind(names, kind, value):
        for name in names:
            out.setdefault(name, []).append((kind, value))

    for node in index["nodes"]:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bind([a.asname or a.name.split(".")[0] for a in node.names if a.name != "*"], "import", node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                bind(_target_names(target)[0], "assign", node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind(_target_names(node.target)[0], "assign", node.value)
        elif isinstance(node, ast.NamedExpr):
            bind(_target_names(node.target)[0], "walrus", node.value)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            bind(_target_names(node.optional_vars)[0], "with", node.context_expr)
        elif isinstance(node, ast.AugAssign):
            bind(_target_names(node.target)[0], "augmented", node)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            bind(_target_names(node.target)[0], "loop", node)
        elif isinstance(node, (*_FUNCTIONS, ast.Lambda)):
            args = node.args
            params = sum(_parameters(args), []) + [p for p in (args.vararg, args.kwarg) if p is not None]
            bind([p.arg for p in params], "parameter", node)
            if not isinstance(node, ast.Lambda):
                bind([node.name], "definition", node)
        elif isinstance(node, ast.ClassDef):
            bind([node.name], "definition", node)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bind([node.name], "handler", node)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bind([node.name], "pattern", node)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bind([node.rest], "pattern", node)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bind(node.names, "declaration", node)
        elif type(node).__name__ in ("TypeAlias", "TypeVar", "ParamSpec", "TypeVarTuple"):
            name = getattr(node, "name", None)
            bind([name.id if isinstance(name, ast.Name) else name] if name else [], "definition", node)
    # A star import may bind any name its source defines: every name the
    # module reads is bound once more, by something the census does not read.
    star = next((n for n in index["nodes"] if isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)),
                None)
    if star is not None:
        bind({n.id for n in index["nodes"] if isinstance(n, ast.Name)}, "star", star)
    index["bindings"] = out
    return out


def _dotted(index, node, may=False):
    """The dotted name a name or an attribute chain stands for through the module's imports, or None.

    Every binding of the root, anywhere in the module, must be an absolute
    import of the same thing: ``os.path.join`` after ``import os``, ``join``
    after ``from os.path import join``. A root bound by anything else stands
    for nothing known. With ``may``, a star import is no reason to doubt: the
    reading that refuses more, where a doubt would let a write through.
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    roots = set()
    for kind, stmt in _name_bindings(index).get(node.id, ()):
        if may and kind == "star":
            continue
        if kind != "import" or (isinstance(stmt, ast.ImportFrom) and (stmt.level or not stmt.module)):
            return None
        for alias in stmt.names:
            if (alias.asname or alias.name.split(".")[0]) == node.id:
                if isinstance(stmt, ast.Import):
                    roots.add(alias.name if alias.asname else alias.name.split(".")[0])
                else:
                    roots.add(f"{stmt.module}.{alias.name}")
    if len(roots) != 1:
        return None
    return ".".join([roots.pop()] + list(reversed(parts)))


def _top_level(tree):
    """Statements that run at module level: the body, and the bodies of its if, try and with blocks -- never the
    body of ``if TYPE_CHECKING:``, which runs for a type checker alone."""
    out, stack = [], list(tree.body)
    while stack:
        stmt = stack.pop()
        out.append(stmt)
        if isinstance(stmt, ast.If):
            stack.extend(stmt.orelse if _leaf(stmt.test) == "TYPE_CHECKING" else stmt.body + stmt.orelse)
        elif isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
            stack.extend(stmt.body + stmt.orelse + stmt.finalbody)
            for handler in stmt.handlers:
                stack.extend(handler.body)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            stack.extend(stmt.body)
    return out


def _only_tested(node, parents):
    """True when a reference is only tested -- compared, negated, or a condition -- never handed on."""
    current, parent = node, parents.get(id(node))
    while isinstance(parent, ast.BoolOp):
        current, parent = parent, parents.get(id(parent))
    if isinstance(parent, ast.Compare):
        return True
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
        return True
    return isinstance(parent, (ast.If, ast.While, ast.IfExp, ast.Assert)) and parent.test is current


_ROUTER_TYPES = frozenset({"fastapi.APIRouter", "fastapi.FastAPI", "fastapi.routing.APIRouter"})


_BENIGN_NAMES = frozenset({"abc.abstractmethod", "functools.cached_property", "typing.override",
                           "typing_extensions.override"})


def _benign(index, function, decorator):
    """True when a decorator only marks a method: a builtin marker no binding of the module shadows, a marker
    imported from the standard library, or the ``setter``, ``getter`` or ``deleter`` of a property the same class
    defines. A registry's method of the same name is handed the function."""
    if isinstance(decorator, ast.Name):
        if decorator.id in ("staticmethod", "classmethod", "property"):
            return decorator.id not in _name_bindings(index)
        return _dotted(index, decorator) in _BENIGN_NAMES
    if isinstance(decorator, ast.Attribute):
        if decorator.attr in ("setter", "getter", "deleter") and isinstance(decorator.value, ast.Name):
            scope = index["parents"].get(id(function))
            return isinstance(scope, ast.ClassDef) and any(
                isinstance(s, _FUNCTIONS) and s.name == decorator.value.id
                and any(isinstance(d, ast.Name) and d.id == "property" for d in s.decorator_list) for s in scope.body)
        return _dotted(index, decorator) in _BENIGN_NAMES
    return False


def _route_handler(census, rel, function):
    """True when a function is decorated as a route of a router or an application, and by nothing else but a
    method marker: a decorator named like a route on any other object -- a registry of model tools -- makes no
    route, and another decorator may hand the function elsewhere."""
    index = census.package.index(rel)
    routed = False
    for decorator in getattr(function, "decorator_list", ()):
        func = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(func, ast.Attribute) and func.attr in _ROUTE_METHODS and isinstance(func.value, ast.Name) \
                and _router(census, rel, func.value.id):
            routed = True
        elif not _benign(index, function, decorator):
            return False
    return routed


def _router(census, rel, name, depth=0):
    """True when a module binds ``name`` only ever to a router or an application it builds (or to None, a
    fallback), at least once; or imports it from a package module where it is one."""
    if depth > 8 or rel not in census.package.texts or census.package.trees[rel] is None:
        return False
    index = census.package.index(rel)
    # A router whose attribute is replaced (``router.post = tools.post``) may
    # route anywhere.
    if any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == name
           and not isinstance(n.ctx, ast.Load) for n in index["nodes"]) or any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("setattr", "delattr")
            and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == name for n in index["calls"]) \
            or _router_rewired(census, rel, name):
        return False
    built = False
    for kind, value in _name_bindings(index).get(name, ()):
        if kind == "assign" and isinstance(value, ast.Constant) and value.value is None:
            continue
        if kind == "assign" and isinstance(value, ast.Call) and _dotted(index, value.func) in _ROUTER_TYPES:
            built = True
            continue
        if kind == "import" and isinstance(value, ast.ImportFrom):
            source = census.package.source_of(rel, value)
            alias = next((a for a in value.names if (a.asname or a.name) == name), None)
            if source is not None and alias is not None and _router(census, source, alias.name, depth + 1):
                built = True
                continue
        return False
    return built


def _router_rewired(census, rel, name):
    """True when another module of the package sets or deletes an attribute of the router ``rel`` binds to
    ``name``, reached through an import of it or as an attribute of ``rel``: its routes may go anywhere."""
    cache = census.__dict__.setdefault("_rewired_routers", {})
    if (rel, name) in cache:
        return cache[(rel, name)]
    found = False
    for other in sorted(census.package.spelling({name})):
        if other == rel or found or census.package.trees[other] is None:
            continue
        index = census.package.index(other)
        for node in index["nodes"]:
            if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
                found = found or _names_router(census, other, node.value, rel, name)
            elif isinstance(node, ast.Call) and _leaf(node.func) in ("setattr", "delattr") and node.args:
                found = found or _names_router(census, other, node.args[0], rel, name)
    cache[(rel, name)] = found
    return found


def _names_router(census, other, expr, rel, name):
    """True when an expression of module ``other`` may be the object ``rel`` binds to ``name``: a name imported
    from it, by name or by a star, or the attribute of a name for it."""
    package = census.package
    if isinstance(expr, ast.Name):
        for kind, stmt in _name_bindings(package.index(other)).get(expr.id, ()):
            if kind == "import" and isinstance(stmt, ast.ImportFrom) and package.source_of(other, stmt) == rel \
                    and any((a.asname or a.name) == expr.id and a.name == name for a in stmt.names):
                return True
            if kind == "star" and expr.id == name and package.source_of(other, stmt) == rel:
                return True
        return False
    if isinstance(expr, ast.Attribute) and expr.attr == name:
        return rel in (census._value_modules(expr.value, census._module_aliases(other)) or ())
    return False


def _main_guarded(node, parents):
    """True when a node runs only under ``if __name__ == "__main__":`` at module level."""
    current = node
    while True:
        parent = parents.get(id(current))
        if parent is None:
            return False
        if isinstance(parent, ast.If) and isinstance(parents.get(id(parent)), ast.Module) \
                and current in parent.body and _is_main_test(parent.test):
            return True
        current = parent


def _is_main_test(test):
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)):
        return False
    sides = [test.left, test.comparators[0]]
    names = [s for s in sides if isinstance(s, ast.Name) and s.id == "__name__"]
    consts = [s for s in sides if isinstance(s, ast.Constant) and s.value == "__main__"]
    return bool(names) and bool(consts)


def _leaves(block, raising=True):
    """True when a block always leaves: its last statement returns, raises, continues or breaks.

    Inside a ``with``, a raise does not count: the context manager may
    swallow it, and control goes on past the block.
    """
    if not block:
        return False
    last = block[-1]
    if isinstance(last, ast.Raise):
        return raising
    if isinstance(last, _LEAVES):
        return True
    if isinstance(last, ast.If):
        return _leaves(last.body, raising) and _leaves(last.orelse, raising)
    if isinstance(last, (ast.With, ast.AsyncWith)):
        return _leaves(last.body, raising=False)
    return False


# ---------------------------------------------------------------------------
# The package: texts, trees parsed on first use, and what each module spells.
# ---------------------------------------------------------------------------
class _Trees(dict):
    """Syntax trees parsed on first use: a module the census never needs is never parsed."""

    def __init__(self, texts):
        super().__init__()
        self.texts = texts
        self.unparsed = {}

    def __missing__(self, rel):
        try:
            tree = ast.parse(self.texts[rel])
        except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
            # A text too deep for the parser does not parse either: it is
            # named, never a crash.
            self.unparsed[rel] = f"{type(exc).__name__}: {exc}"
            tree = None
        self[rel] = tree
        return tree


class Package:
    """The modules of the package, how a dotted name resolves to one, and what the census reads of each."""

    _WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    _LOOKUP = re.compile(r"\b(?:modules|import_module|__import__)\b")
    _IMPORT_SPAN = re.compile(r"\bimport\b(?:[ \t]*\([^)]*\)|(?:\\\n|[^\n])*)")

    def __init__(self, modules):
        self.texts = dict(modules)
        self.trees = _Trees(self.texts)
        self._imports, self._index, self._words, self._definitions = {}, {}, None, {}
        self._submodules, self._import_words, self._relayed, self._lookups = {}, {}, {}, {}

    def resolve(self, dotted):
        if not dotted:
            return None
        path = dotted.replace(".", "/")
        for candidate in (path + "/__init__.py", path + ".py"):
            if candidate in self.texts:
                return candidate
        return None

    def base(self, rel, level):
        parts = rel[:-3].split("/")[:-1]
        if level > 1:
            parts = parts[: len(parts) - (level - 1)]
        return ".".join(parts)

    def submodules(self, rel, name, depth=0):
        """Every module ``rel.name`` may be: a package's submodule, or a module ``rel`` binds to that name by an
        import -- each branch of a ``try`` kept, a module a plain module relays, a re-export of a re-export, a
        star import's."""
        if not depth and (rel, name) in self._submodules:
            return set(self._submodules[(rel, name)])
        out = set()
        if rel.endswith("/__init__.py"):
            found = self.resolve(rel[:-12].replace("/", ".") + "." + name)
            if found is not None:
                out.add(found)
        if depth > 4 or rel not in self.texts or not self._may_import(rel, name):
            if not depth:
                self._submodules[(rel, name)] = frozenset(out)
            return out
        for node in self.imports(rel):
            for alias in node.names:
                if isinstance(node, ast.Import):
                    if alias.asname == name or (not alias.asname and alias.name.split(".")[0] == name):
                        target = self.resolve(alias.name if alias.asname else name)
                        if target is not None:
                            out.add(target)
                    continue
                source = self.source_of(rel, node)
                if source is not None and (alias.name == "*" or (alias.asname or alias.name) == name):
                    out |= self.submodules(source, name if alias.name == "*" else alias.name, depth + 1)
        # A module looked up by its name and bound at module level is relayed too.
        out |= self.module_lookups(rel).get(name, set())
        if not depth:
            self._submodules[(rel, name)] = frozenset(out)
        return out

    def module_lookups(self, rel):
        """name -> the package modules a module binds the name to at module level by a lookup by name
        (``storage = import_module("opti_oignon.jot")``)."""
        if rel not in self._lookups:
            out = {}
            if rel in self.texts and self._LOOKUP.search(self.texts[rel]) and self.trees[rel] is not None:
                for stmt in _top_level(self.trees[rel]):
                    if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
                        modules = _module_calls(self, stmt.value)
                        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                        for target in targets if modules else ():
                            for local in _target_names(target)[0]:
                                out.setdefault(local, set()).update(modules)
            self._lookups[rel] = out
        return self._lookups[rel]

    def _may_import(self, rel, name):
        """False only when a module surely binds ``name`` to no module: no import statement of its text spells
        the name, it has no star import, and no lookup by name binds it. A superset: whatever it lets through is
        read by its syntax tree."""
        self.spelling(())
        if rel in self._stars:
            return True
        words = self._import_words.get(rel)
        if words is None:
            # Every word an import statement spells after its ``import``, to the end of its logical line or of its
            # parentheses.
            words = self._import_words[rel] = frozenset(
                word for span in self._IMPORT_SPAN.findall(self.texts[rel]) for word in self._WORD.findall(span))
        return name in words or name in self.module_lookups(rel)

    def top_names(self, rel):
        """The names a module binds at its top level, and whether a star import of its own may bind any other."""
        if rel not in self.texts or self.trees[rel] is None:
            return frozenset(), True
        top = _top_level(self.trees[rel])
        names = set().union(*(_statement_names(s) | _expression_names(s) for s in top)) if top else set()
        return frozenset(names - {"*"}), any(isinstance(s, ast.ImportFrom) and any(a.name == "*" for a in s.names)
                                             for s in top)

    def _binds(self, rel, name):
        """True when a module may bind ``name`` at its top level: by a statement, a lazy export, or a star import."""
        names, starred = self.top_names(rel)
        return starred or name in names or name in self.lazy_exports(rel)

    def definitions(self, origin):
        """Every place a name a module imports may be defined: re-exports followed -- named, renamed, through a
        star import, through a lazy export table -- and each alternative kept: the two branches of a ``try``, a
        definition and an import of the same name, and a module's own assignment of the name, which answers for
        it there. Each place is visited once, whatever the shape of the graph of re-exports."""
        if origin in self._definitions:
            return self._definitions[origin]
        out, seen, frontier = set(), {origin}, [origin]
        for _depth in range(9):
            following = []
            for module, name in frontier:
                if module not in self.texts or self.trees[module] is None:
                    out.add((module, name))
                    continue
                # A fallback to None (``except ImportError: X = None``) offers
                # no other implementation: a call of it fails, it writes nowhere.
                own = any((name in _statement_names(stmt) and not isinstance(stmt, (ast.Import, ast.ImportFrom))
                           and not (isinstance(stmt, (ast.Assign, ast.AnnAssign))
                                    and isinstance(stmt.value, ast.Constant) and stmt.value.value is None))
                          or name in _expression_names(stmt) for stmt in _top_level(self.trees[module]))
                targets = []
                for node in self.imports(module):
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    source = self.source_of(module, node)
                    if source is None:
                        continue
                    for alias in node.names:
                        if alias.name == "*":
                            if self._binds(source, name):
                                targets.append((source, name))
                        elif (alias.asname or alias.name) == name and not self.submodules(source, alias.name):
                            targets.append((source, alias.name))
                entry = self.lazy_exports(module).get(name)
                if entry is not None:
                    targets.append(entry)
                if own or not targets:
                    out.add((module, name))
                for target in targets:
                    if target not in seen:
                        seen.add(target)
                        following.append(target)
            frontier = following
            if not frontier:
                break
        # Beyond the depth followed, a place is kept as it stands.
        out |= set(frontier)
        out = out or {origin}
        self._definitions[origin] = out
        return out

    def lazy_exports(self, rel):
        """A package's lazy export table: name -> (package module, attribute), read from its literal.

        A package ``__init__`` that serves its names through ``__getattr__``
        keeps them in a module-level dictionary whose every value is a pair of
        strings, a module (relative to the package, or dotted from the root)
        and the attribute it holds. A table the census cannot read as such a
        literal is not one.
        """
        if not rel.endswith("/__init__.py") or self.trees[rel] is None:
            return {}
        cached = self._lazy.get(rel) if hasattr(self, "_lazy") else None
        if cached is not None:
            return cached
        if not hasattr(self, "_lazy"):
            self._lazy = {}
        package = rel[: -len("/__init__.py")].replace("/", ".")
        out = {}
        for stmt in _top_level(self.trees[rel]):
            value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
            if not isinstance(value, ast.Dict) or not value.keys:
                continue
            entries = {}
            for key, item in zip(value.keys, value.values):
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)
                        and isinstance(item, ast.Tuple) and len(item.elts) == 2
                        and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in item.elts)):
                    entries = None
                    break
                module, attr = item.elts[0].value, item.elts[1].value
                dotted = package + module if module.startswith(".") else module
                source = self.resolve(dotted)
                if source is not None:
                    entries[key.value] = (source, attr)
            if entries:
                out.update(entries)
        self._lazy[rel] = out
        return out

    def source_of(self, rel, node):
        """The package module an ``ImportFrom`` reads from, or None."""
        if node.level:
            base = self.base(rel, node.level)
            dotted = base + ("." + node.module if node.module else "") if base else (node.module or "")
        else:
            dotted = node.module or ""
        return self.resolve(dotted)

    def stars(self):
        """Every module with a star import: it may bring in any name, spelling none."""
        self.spelling(())
        return self._stars

    def spelling(self, names):
        """The modules whose text spells one of ``names`` as a word, and every module with a star import."""
        if self._words is None:
            self._words = {r: frozenset(self._WORD.findall(t)) for r, t in self.texts.items()}
            # ``import*`` needs no space, and a continuation may stand before
            # the star: a superset, which can only wake a module for nothing.
            self._stars = {r for r, t in self.texts.items() if re.search(r"\bimport(?:\s|\\)*\*", t)}
        names = frozenset(names)
        return {rel for rel, words in self._words.items() if not words.isdisjoint(names)} | self._stars

    def imports(self, rel):
        """Every import statement of a module, found by walking statements only. A module read for nothing but its
        imports keeps no tree; one that does not parse is named as the census names it."""
        if rel not in self._imports:
            tree = self.trees[rel] if rel in self.trees else None
            if rel not in self.trees:
                try:
                    tree = ast.parse(self.texts[rel])
                except (SyntaxError, ValueError, MemoryError, RecursionError):
                    tree = self.trees[rel]
            found, stack = [], list(tree.body) if tree is not None else []
            while stack:
                stmt = stack.pop()
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    found.append(stmt)
                    continue
                for field in _BLOCKS:
                    stack.extend(getattr(stmt, field, ()) or ())
                for handler in getattr(stmt, "handlers", ()) or ():
                    stack.extend(handler.body)
                for case in getattr(stmt, "cases", ()) or ():
                    stack.extend(case.body)
            self._imports[rel] = found
        return self._imports[rel]

    def relayed(self, rel, seen=frozenset()):
        """The package modules a module binds to a name by its imports -- ``import a.b as x``, ``import a``,
        ``from a import b`` when ``b`` is a module, a star import's: what ``rel.x`` may be."""
        out = set()
        if rel in seen:
            return out
        if not seen and rel in self._relayed:
            return set(self._relayed[rel])
        top, seen = not seen, seen | {rel}
        for node in self.imports(rel):
            for alias in node.names:
                if isinstance(node, ast.Import):
                    target = self.resolve(alias.name if alias.asname else alias.name.split(".")[0])
                    if target is not None:
                        out.add(target)
                    continue
                source = self.source_of(rel, node)
                if source is None:
                    continue
                if alias.name == "*":
                    # A star relays every module its source binds to a name.
                    out |= self.relayed(source, seen)
                    continue
                out |= self.submodules(source, alias.name)
        # A module looked up by its name and bound at module level is relayed too.
        for modules in self.module_lookups(rel).values():
            out |= modules
        if top:
            self._relayed[rel] = frozenset(out)
        return out

    def imported_modules(self, rel):
        """Every package module a module imports from."""
        out = set()
        for node in self.imports(rel):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    for i in range(1, len(parts) + 1):
                        target = self.resolve(".".join(parts[:i]))
                        if target is not None:
                            out.add(target)
            else:
                source = self.source_of(rel, node)
                if source is not None:
                    out.add(source)
                    for alias in node.names:
                        if alias.name != "*":
                            out |= self.submodules(source, alias.name)
        return out

    def index(self, rel):
        """One walk of a module: every node with its function, its dotted name, its class and its parent."""
        if rel in self._index:
            return self._index[rel]
        tree = self.trees[rel]
        # One walk, children read field by field: the parent of every node,
        # the class a ``self.x =`` sits in, and every return value, charged
        # to each function it sits in (a nested function's return is read as
        # its enclosing functions' too). Owners are found by climbing parents.
        nodes, returns, parents, klass = [], {}, {}, {}
        stack = [(tree, (), None, False)]
        AST, function_types, class_type, return_type = ast.AST, _FUNCTIONS, ast.ClassDef, ast.Return
        while stack:
            node, enclosing, cls, in_function = stack.pop()
            nodes.append(node)
            if isinstance(node, function_types):
                returns[id(node)] = []
                enclosing = enclosing + (id(node),)
                in_function = True
            elif isinstance(node, class_type):
                cls, in_function = node.name, False
            elif isinstance(node, (return_type, ast.Yield, ast.YieldFrom)) and node.value is not None:
                # What a generator yields is what its callers get: a dependency
                # that yields a store object, a context manager, an iterator.
                for function in enclosing:
                    returns[function].append(node)
            elif in_function and isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                klass[id(node)] = cls
            elif in_function and isinstance(node, ast.Name) and node.id in ("self", "cls"):
                klass[id(node)] = cls
            elif in_function and isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in ("super", "setattr"):
                klass[id(node)] = cls
            for field in node._fields:
                value = getattr(node, field, None)
                if isinstance(value, list):
                    for child in value:
                        if isinstance(child, AST):
                            parents[id(child)] = node
                            stack.append((child, enclosing, cls, in_function))
                elif isinstance(value, AST):
                    parents[id(value)] = node
                    stack.append((value, enclosing, cls, in_function))
        signatures, top_defs, methods, properties = {}, {}, {}, set()
        for node in _top_level(tree):
            if isinstance(node, _FUNCTIONS):
                top_defs[node.name] = (node.args, 0)
            elif isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, _FUNCTIONS) and stmt.name == "__init__":
                        top_defs[node.name] = (stmt.args, 1)
        for node in nodes:
            if isinstance(node, _FUNCTIONS):
                signatures.setdefault(node.name, []).append((node.args, None))
                if any(_leaf(d) in ("property", "cached_property") for d in node.decorator_list):
                    properties.add(node.name)
            elif isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, _FUNCTIONS):
                        methods.setdefault("." + stmt.name, []).append((stmt.args, 1))
                        if stmt.name == "__init__":
                            signatures.setdefault(node.name, []).append((stmt.args, 1))
        # A class built from its fields takes them as its constructor's parameters.
        fielded = _fielded(tree, nodes)
        for name, (args, top, _names) in fielded.items():
            signatures.setdefault(name, []).append((args, 1))
            if top and name not in top_defs:
                top_defs[name] = (args, 1)
        index = {
            "nodes": nodes, "returns": returns, "parents": parents, "klass": klass,
            "binders": [n for n in nodes if isinstance(n, _BINDERS)],
            "references": [n for n in nodes if isinstance(n, (ast.Attribute, ast.Name, ast.Call))],
            "calls": [n for n in nodes if isinstance(n, ast.Call)],
            "class_body": {id(s) for n in nodes if isinstance(n, ast.ClassDef) for s in n.body},
            "signatures": signatures, "top_defs": top_defs, "methods": methods, "properties": properties,
            "tables": _dispatch_tables(nodes), "method_of": {
                id(stmt): n.name for n in nodes if isinstance(n, ast.ClassDef) for stmt in n.body
                if isinstance(stmt, _FUNCTIONS)},
            "fields": {name: names for name, (_args, _top, names) in fielded.items()},
        }
        self._index[rel] = index
        return index


def _fielded(tree, nodes):
    """name -> (signature, defined at top level, fields) for each class built from its fields with no ``__init__``
    of its own -- a dataclass, an attrs class, a named tuple, a model -- and each ``namedtuple(...)`` bound to a
    name: the fields, in order, are the constructor's parameters."""
    top = {id(n) for n in _top_level(tree)}
    out = {}
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            if any(isinstance(s, _FUNCTIONS) and s.name == "__init__" for s in node.body):
                continue
            marks = {_leaf(d.func if isinstance(d, ast.Call) else d) for d in node.decorator_list}
            marks |= {_leaf(b) for b in node.bases}
            if not marks & (_FIELD_DECORATORS | _FIELD_BASES):
                continue
            names = [s.target.id for s in node.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
                     and _leaf(s.annotation.value if isinstance(s.annotation, ast.Subscript) else s.annotation)
                     != "ClassVar"]
            out[node.name] = (_field_signature(names), id(node) in top, names)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and _leaf(node.value.func) in ("namedtuple", "NamedTuple") and len(node.value.args) >= 2:
            spec = node.value.args[1]
            if isinstance(spec, ast.Constant) and isinstance(spec.value, str):
                names = [n for n in re.split(r"[,\s]+", spec.value) if n]
            elif isinstance(spec, (ast.List, ast.Tuple)):
                names = [e.value if isinstance(e, ast.Constant) else e.elts[0].value
                         if isinstance(e, ast.Tuple) and e.elts and isinstance(e.elts[0], ast.Constant) else None
                         for e in spec.elts]
                names = [n for n in names if isinstance(n, str)]
            else:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = (_field_signature(names), id(node) in top, names)
    return out


def _field_signature(names):
    """A constructor's signature that takes ``names`` after ``self``."""
    return ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")] + [ast.arg(arg=n) for n in names], vararg=None,
                         kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])


def _handed(call):
    """Functions a call hands over uncalled, each with what it will be called with: ``g(f, a, b, k=v)`` hands
    ``f`` what follows it -- a scheduler, a pool, a thread, ``map``, ``partial`` -- and ``Thread(target=f,
    args=(a,), kwargs={"k": v})`` hands ``f`` its ``args`` and ``kwargs``."""
    out = []
    keywords = [(kw.arg, kw.value) for kw in call.keywords if kw.arg]
    for i, arg in enumerate(call.args):
        if isinstance(arg, (ast.Name, ast.Attribute)):
            out.append((arg, list(call.args[i + 1:]), keywords))
    target = next((kw.value for kw in call.keywords if kw.arg in ("target", "func", "fn", "function", "callback")),
                  None)
    if isinstance(target, (ast.Name, ast.Attribute)):
        args = next((kw.value for kw in call.keywords if kw.arg == "args"), None)
        kwargs = next((kw.value for kw in call.keywords if kw.arg == "kwargs"), None)
        positional = list(args.elts) if isinstance(args, (ast.Tuple, ast.List)) else []
        named = [(k.value, v) for k, v in zip(kwargs.keys, kwargs.values)
                 if isinstance(k, ast.Constant) and isinstance(k.value, str)] if isinstance(kwargs, ast.Dict) else []
        out.append((target, positional, named))
    return out


def _injections(index, name):
    """The calls a decorator the module defines under ``name`` makes of the function it is handed -- directly, or
    from the decorator a factory of that name returns: each call's arguments reach the decorated function."""
    if not name:
        return []
    cache = index.setdefault("injections", {})
    if name not in cache:
        calls = []
        for node in index["nodes"]:
            if not (isinstance(node, _FUNCTIONS) and node.name == name):
                continue
            for scope in [node] + [n for n in ast.walk(node) if isinstance(n, _FUNCTIONS) and n is not node]:
                positional = _parameters(scope.args)[0]
                if not positional:
                    continue
                handed = positional[0].arg
                calls.extend(n for n in ast.walk(scope) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id == handed)
        cache[name] = calls
    return cache[name]


def _in_body(scope, child):
    """True when ``child`` sits in a scope's body. What a scope's header holds -- its decorators, its defaults,
    its annotations, its bases -- runs when the scope is defined, in the scope around it."""
    if isinstance(scope, ast.Lambda):
        return child is scope.body
    return isinstance(scope, (*_FUNCTIONS, ast.ClassDef)) and child in scope.body


def _owner(index, node):
    """The innermost function whose body a node sits in, and the dotted names of the functions and classes whose
    bodies hold it."""
    parents = index["parents"]
    owner, names, child = None, [], node
    while True:
        current = parents.get(id(child))
        if current is None:
            break
        if isinstance(current, (*_FUNCTIONS, ast.ClassDef)) and _in_body(current, child):
            if isinstance(current, _FUNCTIONS) and owner is None:
                owner = current
            names.append(current.name)
        child = current
    return owner, tuple(reversed(names))


def _type_checking_only(index, node):
    """True when a node sits in the body of an ``if TYPE_CHECKING:``, which runs for a type checker alone."""
    parents, child = index["parents"], node
    while True:
        current = parents.get(id(child))
        if current is None:
            return False
        if isinstance(current, ast.If) and _leaf(current.test) == "TYPE_CHECKING" and child in current.body:
            return True
        child = current


def _header(scope):
    """The expressions of a scope's header: decorators, defaults, annotations, bases and class keywords."""
    if isinstance(scope, ast.ClassDef):
        return list(scope.decorator_list) + list(scope.bases) + [k.value for k in scope.keywords]
    args = scope.args
    out = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
    if isinstance(scope, _FUNCTIONS):
        params = sum(_parameters(args), []) + [p for p in (args.vararg, args.kwarg) if p is not None]
        out += list(scope.decorator_list) + [p.annotation for p in params if p.annotation is not None]
        if scope.returns is not None:
            out.append(scope.returns)
    return out


def _home_of(index, node):
    """The function whose own body a node runs in, or None: at module level, in a class body, or inside a lambda
    or a generator expression, which run whenever their value is used, by whoever holds it."""
    parents = index["parents"]
    child = node
    while True:
        current = parents.get(id(child))
        if current is None or isinstance(current, (ast.Module, ast.GeneratorExp)):
            return None
        if isinstance(current, (ast.ClassDef, ast.Lambda)) and _in_body(current, child):
            return None
        if isinstance(current, _FUNCTIONS) and _in_body(current, child):
            return current
        child = current


def _lazy(function):
    """How a call of a function runs its body: 'coroutine' for an ``async def``, 'asyncgen' for one that also
    yields, 'generator' for a function with its own ``yield``, None when the call runs it on the spot."""
    stack, yields = list(function.body), False
    while stack and not yields:
        node = stack.pop()
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            yields = True
        elif not isinstance(node, (*_FUNCTIONS, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))
    if isinstance(function, ast.AsyncFunctionDef):
        return "asyncgen" if yields else "coroutine"
    return "generator" if yields else None


# What rebuilds an object in place when it is read and called or written through; an assignment to any attribute,
# ``__class__`` included, is a site of its own.
_REBUILDING = frozenset({"__init__", "__setattr__", "__delattr__", "__dict__", "__setstate__"})

# Builtins that drain an iterable on the spot.
_DRAINS = frozenset({"list", "tuple", "set", "frozenset", "sorted", "sum", "min", "max", "any", "all", "dict"})


def _consumed(index, call, lazy):
    """True when a call of a coroutine or a generator function runs its body on the spot: awaited, iterated by a
    ``for`` or a comprehension that is no generator expression, delegated to by ``yield from``, or drained by a
    builtin no binding of the module shadows."""
    parents = index["parents"]
    parent = parents.get(id(call))
    if lazy == "coroutine":
        return isinstance(parent, ast.Await)
    if lazy == "asyncgen":
        return (isinstance(parent, ast.AsyncFor) and parent.iter is call) or (
            isinstance(parent, ast.comprehension) and parent.is_async and parent.iter is call
            and not isinstance(parents.get(id(parent)), ast.GeneratorExp))
    if isinstance(parent, ast.For) and parent.iter is call:
        return True
    if isinstance(parent, ast.comprehension) and parent.iter is call:
        return not isinstance(parents.get(id(parent)), ast.GeneratorExp)
    if isinstance(parent, ast.YieldFrom):
        return True
    return isinstance(parent, ast.Call) and parent.args and parent.args[0] is call \
        and isinstance(parent.func, ast.Name) and parent.func.id in _DRAINS \
        and parent.func.id not in _name_bindings(index)


def _dispatch_tables(nodes):
    """key -> what a container literal bound to it holds, a dispatch table: function names, lambdas, bound
    methods (``self._add``) and functions of other modules (``helper.add``).

    The key is the name the literal is bound to, or ``.attr`` for an
    attribute (``self.TABLE = {...}``).
    """
    out = {}
    for node in nodes:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        items = value.values if isinstance(value, ast.Dict) else (
            value.elts if isinstance(value, (ast.List, ast.Tuple, ast.Set)) else [])
        held = [item.id if isinstance(item, ast.Name) else item for item in items
                if isinstance(item, (ast.Name, ast.Lambda, ast.Attribute))]
        if not held:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                # A class attribute is read as ``self.TABLE`` or ``C.TABLE``.
                out.setdefault(target.id, []).extend(held)
                out.setdefault("." + target.id, []).extend(held)
            elif isinstance(target, ast.Attribute):
                out.setdefault("." + target.attr, []).extend(held)
    return out


def _table_key(node):
    """The dispatch table a subscript or a ``get`` reads: its name, or ``.attr`` for an attribute."""
    if isinstance(node, ast.Subscript):
        node = node.value
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        node = node.func.value
    else:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return "." + node.attr
    return None


def _module_calls(package, node):
    """The package modules a constant ``sys.modules`` lookup or ``import_module`` call may name; empty for none.

    ``__import__`` with a dotted name hands back its leaf with a non-empty
    ``fromlist`` and its top package without: both, when the ``fromlist`` is
    not a literal.
    """
    found = None
    if isinstance(node, ast.Call):
        func = node.func
        arg = node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg == "name"), None)
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
            return set()
        name = arg.value
        if _leaf(func) == "import_module":
            if name.startswith("."):
                anchor = node.args[1] if len(node.args) > 1 else next(
                    (kw.value for kw in node.keywords if kw.arg == "package"), None)
                if not (isinstance(anchor, ast.Constant) and isinstance(anchor.value, str)):
                    return set()
                level = len(name) - len(name.lstrip("."))
                parts = anchor.value.split(".")
                base = parts[: len(parts) - (level - 1)] if level > 1 else parts
                name = ".".join(base) + "." + name.lstrip(".") if name.lstrip(".") else ".".join(base)
            found = {package.resolve(name)}
        elif _leaf(func) == "__import__":
            fromlist = node.args[3] if len(node.args) > 3 else next(
                (kw.value for kw in node.keywords if kw.arg == "fromlist"), None)
            if fromlist is None or isinstance(fromlist, ast.Constant) or (
                    isinstance(fromlist, (ast.List, ast.Tuple, ast.Set))
                    and not any(isinstance(e, ast.Starred) for e in fromlist.elts)):
                leaf = isinstance(fromlist, (ast.List, ast.Tuple, ast.Set)) and bool(fromlist.elts) or (
                    isinstance(fromlist, ast.Constant) and bool(fromlist.value))
                found = {package.resolve(name if leaf else name.split(".")[0])}
            else:
                found = {package.resolve(name), package.resolve(name.split(".")[0])}
        elif isinstance(func, ast.Attribute) and func.attr == "get" and _modules_table(func.value):
            found = {package.resolve(name)}
    elif isinstance(node, ast.Subscript) and _modules_table(node.value):
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            found = {package.resolve(key.value)}
    return {module for module in found or () if module is not None}


def _modules_table(node):
    """True for ``sys.modules``, or ``modules`` imported from ``sys`` under its name."""
    return (isinstance(node, ast.Attribute) and node.attr == "modules") or (
        isinstance(node, ast.Name) and node.id == "modules")


def _seed(rel):
    """The symbols a store's house defines for itself."""
    out = {}
    for name, store in STORES.items():
        if store.house != rel:
            continue
        for c in store.classes:
            out.setdefault(c, set()).add(("class", frozenset({name})))
        for a in store.accesses:
            out.setdefault(a, set()).add(("returner", frozenset({name})))
        for i in store.instances:
            out.setdefault(i, set()).add(("handle", frozenset({name})))
        for f in store.functions:
            out.setdefault(f, set()).add(("sink", frozenset({name})))
    return out


class _World:
    """What the modules tell each other: exports, what carriers carry, parameters bound by callers."""

    def __init__(self, package):
        self.package = package
        self.exports = {}
        for rel in package.texts:
            seeded = _seed(rel)
            if seeded:
                self.exports[rel] = seeded
        self.members = {}
        self.params = {}
        # Whether any carrier hands a store object back from ``__call__``.
        self.callable = False


# ---------------------------------------------------------------------------
# One module's bindings and sites.
# ---------------------------------------------------------------------------
class ModuleCensus:
    """Bindings and sites of one module, given what the other modules export."""

    def __init__(self, world, rel):
        self.world = world
        self.package = world.package
        self.rel = rel
        self.tree = self.package.trees[rel]
        self.handles, self.attrs, self.classes, self.returners, self.sinks = {}, {}, {}, {}, {}
        self.carried = {}
        self.modules = {}
        self.origins = {}
        self.calls_out = {}
        self.containers = set()
        self.more_modules = {}
        self.attr_modules = {}
        self.table_aliases = {}
        self.sites = []
        if self.tree is None:
            return
        for name, entries in _seed(rel).items():
            for kind, stores in entries:
                self._table(kind).setdefault(name, set()).update(stores)
        self._imports()
        # Every binding derives from a store symbol the module defines,
        # imports, or is handed by a caller: with none of the three, nothing
        # can ever bind, and the module is not walked.
        # A module alias counts when it names an exporting module or a package
        # that holds one (``import opti_oignon.x`` binds the package), and so
        # does a lookup of a module by its name at run time.
        exporting = set(self.world.exports)
        bound = set(self.modules.values()).union(*self.more_modules.values()) if self.more_modules else set(
            self.modules.values())
        # A module also hands out modules it imports itself -- a package what
        # it imports from its own folder or from anywhere, a plain module what
        # it binds to a name: ``pkg.alias`` or ``relay.storage`` may be a
        # store module.
        for _depth in range(3):
            bound |= {m for module in list(bound) for m in (
                self.package.imported_modules(module) if module.endswith("/__init__.py")
                else self.package.relayed(module))}
        packages = {m[: -len("__init__.py")] for m in bound if m.endswith("/__init__.py")}
        self.inert = not (self.handles or self.classes or self.returners or self.sinks) and not any(
            target == rel for target, _name in self.world.params) and not any(
            module in exporting for module in bound) and not any(
            key.startswith(package) for package in packages for key in exporting) and not re.search(
            r"\b(?:modules|import_module|__import__)\b", self.package.texts[rel])
        if not self.inert:
            self._fixed_point()

    def _table(self, kind):
        return {"handle": self.handles, "class": self.classes, "returner": self.returners,
                "sink": self.sinks}[kind]

    def _imports(self):
        for node in self.package.imports(self.rel):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = self.package.resolve(alias.name)
                    if alias.asname:
                        if target is not None:
                            self._alias(alias.asname, target)
                    else:
                        head = alias.name.split(".")[0]
                        top = self.package.resolve(head)
                        if top is not None:
                            self._alias(head, top)
                continue
            source = self.package.source_of(self.rel, node)
            if source is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    for name, entries in self.world.exports.get(source, {}).items():
                        for kind, stores in entries:
                            self._table(kind).setdefault(name, set()).update(stores)
                    # Every name the source binds may be called here, and an
                    # argument handed to it reaches the source's definition;
                    # a module the source binds to a name is bound here too.
                    names, _starred = self.package.top_names(source)
                    for name in sorted(names):
                        self.origins.setdefault(name, set()).add((source, name))
                        for sub in sorted(self.package.submodules(source, name)):
                            self._alias(name, sub)
                    continue
                local = alias.asname or alias.name
                subs = sorted(self.package.submodules(source, alias.name))
                for sub in subs:
                    self._alias(local, sub)
                # A package may hand out an object under a submodule's name (a
                # lazy export table): both readings are kept.
                exported = self.world.exports.get(source, {}).get(alias.name, ())
                if subs and not exported:
                    continue
                self.origins.setdefault(local, set()).add((source, alias.name))
                for kind, stores in exported:
                    self._table(kind).setdefault(local, set()).update(stores)
        # A package's lazy export table names what its attributes resolve to.
        self.lazy = set()
        for name, (source, attr) in self.package.lazy_exports(self.rel).items():
            self.lazy.add(name)
            self.origins.setdefault(name, set()).add((source, attr))
            for sub in sorted(self.package.submodules(source, attr)):
                self._alias(name, sub)
            for kind, stores in self.world.exports.get(source, {}).get(attr, ()):
                self._table(kind).setdefault(name, set()).update(stores)

    def _exported(self, owner, name, kind):
        out = set()
        for k, stores in self.world.exports.get(owner, {}).get(name, ()):
            if k == kind:
                out |= stores
        return out

    def symbol(self, node, kind):
        """Stores of a name of ``kind``, local or on a module alias -- and, names not being scoped, the method of
        that name too when the receiver may be something else."""
        if isinstance(node, ast.Name):
            out = set(self._table(kind).get(node.id, ()))
            if kind == "class" and node.id == "cls":
                # ``cls`` in a class method is the class itself.
                cls = self.package.index(self.rel)["klass"].get(id(node))
                if cls is not None:
                    out |= self.classes.get(cls, set())
            return out
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" \
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
                and isinstance(node.args[1].value, str):
            # ``getattr(mod, "Jot")`` names what ``mod.Jot`` names.
            return self.symbol(ast.Attribute(value=node.args[0], attr=node.args[1].value, ctx=ast.Load()), kind)
        if isinstance(node, ast.Attribute):
            out = set().union(*(self._exported(owner, node.attr, kind) for owner in self.owners(node.value)))
            if kind == "returner":
                out |= set(self.returners.get(node.attr, ()))
            return out
        return set()

    def stores_of(self, value):
        """The stores, and the carriers, an expression may evaluate to an object of."""
        if value is None:
            return set()
        if isinstance(value, ast.Name):
            out = set(self.handles.get(value.id, ()))
            if value.id in ("self", "cls"):
                # ``self`` in a method is an object of its class: a store, a
                # subclass of one, or a carrier.
                cls = self.package.index(self.rel)["klass"].get(id(value))
                if cls is not None:
                    out |= self.classes.get(cls, set())
            return out
        if isinstance(value, ast.Attribute):
            # Both readings: a name bound to a module somewhere may hold a
            # carrier elsewhere, names not being scoped.
            out = set().union(*(self._exported(owner, value.attr, "handle") for owner in self.owners(value.value)))
            out |= set(self.attrs.get(value.attr, ()))
            for store in self.stores_of(value.value):
                out |= self.world.members.get(store, {}).get(value.attr, set())
                out |= self.carried.get(store, {}).get(value.attr, set())
            return out
        if isinstance(value, ast.Call):
            func = value.func
            out = self.symbol(func, "class") | self.symbol(func, "returner")
            if isinstance(func, ast.Name) and func.id == "getattr" and len(value.args) >= 2:
                name = value.args[1]
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    out |= self.stores_of(ast.Attribute(value=value.args[0], attr=name.value, ctx=ast.Load()))
                for default in value.args[2:3]:
                    out |= self.stores_of(default)
            # A default handed back when the key or the item is missing.
            if isinstance(func, ast.Attribute) and func.attr in ("get", "pop", "setdefault") and len(value.args) >= 2:
                out |= self.stores_of(value.args[1])
            # ``next`` hands back an element of what it is handed, or its default.
            if isinstance(func, ast.Name) and func.id == "next":
                for arg in value.args[:2]:
                    out |= self.stores_of(arg)
            if _leaf(func) in _PASS_THROUGH:
                for arg in list(value.args) + [kw.value for kw in value.keywords]:
                    out |= self.stores_of(arg)
            for arg in list(value.args) + [kw.value for kw in value.keywords]:
                if isinstance(arg, (ast.Name, ast.Attribute)):
                    out |= self.symbol(arg, "returner")
            index = self.package.index(self.rel)
            # ``super()`` in a method is an object of the method's class.
            if isinstance(func, ast.Name) and func.id == "super" and "super" not in _name_bindings(index):
                cls = index["klass"].get(id(value))
                if cls is not None:
                    out |= self.classes.get(cls, set())
            # ``partial(provider)()`` calls the provider.
            if isinstance(func, ast.Call) and _leaf(func.func) == "partial" and func.args:
                out |= self.symbol(func.args[0], "returner") | self.symbol(func.args[0], "class")
            # An object called hands back what its ``__call__`` does.
            if self.world.callable or any("()__call__" in members for members in self.carried.values()):
                for store in self.stores_of(func):
                    out |= self.world.members.get(store, {}).get("()__call__", set())
                    out |= self.carried.get(store, {}).get("()__call__", set())
            if isinstance(func, ast.Attribute):
                # A method of a carrier, or of a class -- a factory -- that hands a store object back.
                held = self.stores_of(func.value) | self.symbol(func.value, "class")
                for store in held:
                    out |= self.world.members.get(store, {}).get("()" + func.attr, set())
                    out |= self.carried.get(store, {}).get("()" + func.attr, set())
                # An element read out of a container of store objects, named or literal.
                if func.attr in _CONTAINER_READS and (self._container(func.value)
                                                      or isinstance(func.value, _CONTAINERS)):
                    out |= held
            return out
        if isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return self.stores_of(value.elt)
        if isinstance(value, ast.DictComp):
            return self.stores_of(value.value)
        if isinstance(value, ast.BoolOp):
            return set().union(*(self.stores_of(v) for v in value.values))
        if isinstance(value, ast.IfExp):
            return self.stores_of(value.body) | self.stores_of(value.orelse)
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return set().union(*(self.stores_of(e) for e in value.elts)) if value.elts else set()
        if isinstance(value, ast.Dict):
            return set().union(*(self.stores_of(v) for v in value.values)) if value.values else set()
        if isinstance(value, (ast.Starred, ast.Await, ast.NamedExpr, ast.Subscript)):
            return self.stores_of(value.value)
        return set()

    def _alias(self, name, module):
        """Bind a name to a module; a name bound to several keeps them all, never flipping between two."""
        first = self.modules.get(name)
        if first is None:
            self.modules[name] = module
            self.grown = True
        elif first != module and module not in self.more_modules.setdefault(name, set()):
            self.more_modules[name].add(module)
            self.grown = True

    def _alias_attr(self, attr, module):
        """Bind an attribute name to a module: ``self.mod = sys.modules.get(...)``."""
        if module not in self.attr_modules.setdefault(attr, set()):
            self.attr_modules[attr].add(module)
            self.grown = True

    def _module_values(self, value):
        """Every package module an assigned value may be: through ``or`` and conditional expressions too."""
        if isinstance(value, ast.BoolOp):
            return set().union(*(self._module_values(v) for v in value.values))
        if isinstance(value, ast.IfExp):
            return self._module_values(value.body) | self._module_values(value.orelse)
        # A module held in a container, and read back out of it.
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return set().union(*(self._module_values(e) for e in value.elts))
        if isinstance(value, ast.Dict):
            return set().union(*(self._module_values(v) for v in value.values))
        if isinstance(value, ast.Subscript):
            return self.owners(value) | self._module_values(value.value)
        if isinstance(value, (ast.Call, ast.Name, ast.Attribute)):
            return self.owners(value)
        return set()

    def owners(self, node):
        """Every package module an expression may name: a module alias bound more than once names each, and an
        attribute chain is followed through every module its root names."""
        if isinstance(node, ast.Name):
            first = self.modules.get(node.id)
            return ({first} if first else set()) | self.more_modules.get(node.id, set())
        if isinstance(node, ast.Attribute):
            parents = self.owners(node.value)
            if parents:
                return {sub for parent in parents for sub in self.package.submodules(parent, node.attr)}
            return set(self.attr_modules.get(node.attr, set()))
        if isinstance(node, (ast.Call, ast.Subscript)):
            out = _module_calls(self.package, node)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" \
                    and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
                    and isinstance(node.args[1].value, str):
                # ``getattr(pkg, "jot")`` names the submodule ``pkg.jot`` does.
                out = out | {sub for parent in self.owners(node.args[0])
                             for sub in self.package.submodules(parent, node.args[1].value)}
            return out
        return set()

    def _container(self, node):
        """True when a name or an attribute was bound as a container of store objects."""
        if isinstance(node, ast.Name):
            return node.id in self.containers
        if isinstance(node, ast.Attribute):
            return node.attr in self.containers
        return False

    def write_ref(self, node):
        """The stores a reference writes to: a write method on a store object, or a write function."""
        if isinstance(node, ast.Attribute):
            out = {s for s in self.stores_of(node.value) if s in STORES and node.attr in STORES[s].writes}
            # ``Jot.put(j, x)``: a write reached through the class itself.
            out |= {s for s in self.symbol(node.value, "class") if s in STORES and node.attr in STORES[s].writes}
            if self.owners(node.value):
                out |= self.symbol(node, "sink")
            return out
        if isinstance(node, ast.Name):
            return set(self.sinks.get(node.id, ()))
        return set()

    def _fixed_point(self):
        index = self.package.index(self.rel)
        class_body, signatures, top_defs = index["class_body"], index["signatures"], index["top_defs"]
        returns, klass = index["returns"], index["klass"]
        store_classes = {c: s for s, spec in STORES.items() if spec.house == self.rel for c in spec.classes}
        for (rel, name), bound in self.world.params.items():
            if rel != self.rel:
                continue
            targets = index["methods"].get(name, []) if name.startswith(".") else (
                [top_defs[name]] if name in top_defs else [])
            for args, offset in targets:
                positional, keyword_only = _parameters(args)
                for key, stores in bound.items():
                    if isinstance(key, int) and key + offset < len(positional):
                        self.handles.setdefault(positional[key + offset].arg, set()).update(stores)
                    elif isinstance(key, int) and args.vararg is not None:
                        self.handles.setdefault(args.vararg.arg, set()).update(stores)
                        self.containers.add(args.vararg.arg)
                    elif isinstance(key, str):
                        params = [p for p in positional + keyword_only if p.arg == key]
                        for param in params:
                            self.handles.setdefault(key, set()).update(stores)
                        if not params and args.kwarg is not None:
                            self.handles.setdefault(args.kwarg.arg, set()).update(stores)
                            self.containers.add(args.kwarg.arg)
            # A class built from its fields, built in another module: its
            # objects carry what each field was handed there.
            fields = index["fields"].get(name, ())
            for key, stores in bound.items():
                field = fields[key] if isinstance(key, int) and key < len(fields) else (
                    key if isinstance(key, str) and key in fields else None)
                if field is not None:
                    carrier = store_classes.get(name) or f"{self.rel}:{name}"
                    self.carried.setdefault(carrier, {}).setdefault(field, set()).update(stores)
                    if carrier not in STORES:
                        self.classes.setdefault(name, set()).add(carrier)
        self.grown = True

        def add(table, name, stores):
            if stores and not stores <= table.get(name, set()):
                table.setdefault(name, set()).update(stores)
                self.grown = True

        def bind_target(target, value, in_class=False):
            if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) \
                    and len(target.elts) == len(value.elts) \
                    and not any(isinstance(e, ast.Starred) for e in target.elts + value.elts):
                for t, v in zip(target.elts, value.elts):
                    bind_target(t, v, in_class)
                return
            names, attributes = _target_names(target)
            if isinstance(value, ast.Lambda):
                for n in names | attributes:
                    add(self.returners, n, self.stores_of(value.body))
                return
            # Sorted: the first module a name is bound to must not depend on
            # the order of a set.
            for module in sorted(self._module_values(value)):
                for n in names:
                    self._alias(n, module)
                for a in attributes:
                    self._alias_attr(a, module)
            key = _table_key(value) if isinstance(value, (ast.Subscript, ast.Call)) else None
            if key is not None and key in index["tables"]:
                # ``fn = TABLE.get(action)``: fn calls what the table holds.
                for n in names:
                    if key not in self.table_aliases.setdefault(n, set()):
                        self.table_aliases[n].add(key)
                        self.grown = True
            if isinstance(value, (ast.Name, ast.Attribute)):
                for kind in ("class", "returner"):
                    stores = self.symbol(value, kind)
                    for n in names:
                        add(self._table(kind), n, stores)
                written = self.write_ref(value)
                for n in names:
                    add(self.sinks, n, written)
            stores = self.stores_of(value)
            for n in names:
                add(self.handles, n, stores)
            for a in attributes | (names if in_class else set()):
                add(self.attrs, a, stores)
            if stores and (isinstance(target, ast.Subscript) or isinstance(value, _CONTAINERS)):
                for n in names | attributes:
                    if n not in self.containers:
                        self.containers.add(n)
                        self.grown = True

        def carry(node, target, value):
            """``self.x = store object`` in a method of class C: C's objects carry it at ``x``."""
            if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id in ("self", "cls")):
                return
            cls = klass.get(id(node))
            if cls is None:
                return
            stores = self.stores_of(value)
            if not stores:
                return
            carrier = store_classes.get(cls) or f"{self.rel}:{cls}"
            add(self.carried.setdefault(carrier, {}), target.attr, stores)
            if carrier not in STORES:
                add(self.classes, cls, {carrier})

        def gather(name, stores):
            # ``*args`` and ``**kwargs`` are containers of what they gather.
            add(self.handles, name, stores)
            if stores and name not in self.containers:
                self.containers.add(name)
                self.grown = True

        def bind_parameters(args, values, keywords, offset=0):
            positional, keyword_only = _parameters(args)
            for i, value in enumerate(values):
                if i + offset < len(positional):
                    if positional[i + offset].arg not in ("self", "cls"):
                        add(self.handles, positional[i + offset].arg, self.stores_of(value))
                elif args.vararg is not None:
                    gather(args.vararg.arg, self.stores_of(value))
            for name, value in keywords:
                params = [p for p in positional + keyword_only if p.arg == name and name not in ("self", "cls")]
                for param in params:
                    add(self.handles, param.arg, self.stores_of(value))
                if not params and args.kwarg is not None:
                    gather(args.kwarg.arg, self.stores_of(value))

        while self.grown:
            self.grown = False
            for node in index["binders"]:
                if isinstance(node, _FUNCTIONS):
                    stores = set()
                    for r in returns[id(node)]:
                        stores |= self.stores_of(r.value)
                    add(self.returners, node.name, stores)
                    # A decorator of the module that calls the function it is
                    # handed hands it what it calls it with: an injected store.
                    for decorator in node.decorator_list:
                        name = _leaf(decorator.func if isinstance(decorator, ast.Call) else decorator)
                        for call in _injections(index, name):
                            bind_parameters(node.args, call.args, [(kw.arg, kw.value) for kw in call.keywords if kw.arg])
                    cls = index["method_of"].get(id(node))
                    if cls is not None and stores:
                        # A method or a property of class C that hands a store
                        # object back: C's objects carry it, wherever they go.
                        carrier = store_classes.get(cls) or f"{self.rel}:{cls}"
                        key = node.name if node.name in index["properties"] else "()" + node.name
                        add(self.carried.setdefault(carrier, {}), key, stores)
                        if carrier not in STORES:
                            add(self.classes, cls, {carrier})
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        add(self.classes, node.name, self.symbol(base, "class"))
                if isinstance(node, (*_FUNCTIONS, ast.Lambda)):
                    positional, keyword_only = _parameters(node.args)
                    defaults = list(node.args.defaults)
                    for param, default in list(zip(positional[len(positional) - len(defaults):], defaults)) + list(
                            zip(keyword_only, node.args.kw_defaults)):
                        add(self.handles, param.arg, self.stores_of(default))
                        if isinstance(default, (ast.Name, ast.Attribute)):
                            # A provider handed uncalled as a default: ``maker=get_jot``.
                            for kind in ("class", "returner"):
                                add(self._table(kind), param.arg, self.symbol(default, kind))
                            add(self.sinks, param.arg, self.write_ref(default))
                    # A dependency an annotation names: ``Annotated[Jot, Depends(get_jot)]``.
                    for param in positional + keyword_only:
                        for sub in ast.walk(param.annotation) if param.annotation is not None else ():
                            if isinstance(sub, ast.Call):
                                add(self.handles, param.arg, self.stores_of(sub))
                elif isinstance(node, ast.Match):
                    # A capture binds what the subject holds.
                    stores = self.stores_of(node.subject)
                    for case in node.cases:
                        for sub in ast.walk(case.pattern):
                            if isinstance(sub, (ast.MatchAs, ast.MatchStar)) and sub.name:
                                add(self.handles, sub.name, stores)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        bind_target(target, node.value, id(node) in class_body)
                        carry(node, target, node.value)
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
                    bind_target(node.target, node.value, id(node) in class_body)
                    carry(node, node.target, node.value)
                elif isinstance(node, ast.NamedExpr):
                    bind_target(node.target, node.value)
                elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                    bind_target(node.target, node.iter)
                elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                    bind_target(node.optional_vars, node.context_expr)
                elif isinstance(node, ast.Call):
                    keywords = [(kw.arg, kw.value) for kw in node.keywords if kw.arg]
                    func = node.func
                    if isinstance(func, ast.Call) and isinstance(func.func, ast.Name) and func.func.id == "getattr" \
                            and len(func.args) >= 2 and isinstance(func.args[1], ast.Constant) \
                            and isinstance(func.args[1].value, str):
                        # ``getattr(x, "name")(...)`` calls what ``x.name(...)`` calls.
                        func = ast.Attribute(value=func.args[0], attr=func.args[1].value, ctx=ast.Load())
                    if isinstance(func, ast.Lambda):
                        bind_parameters(func.args, node.args, keywords)
                    elif isinstance(func, (ast.Name, ast.Attribute)):
                        name = func.id if isinstance(func, ast.Name) else func.attr
                        for args, skipped in signatures.get(name, ()):
                            positional = _parameters(args)[0]
                            method = bool(positional) and positional[0].arg in ("self", "cls")
                            if skipped is None and method != isinstance(func, ast.Attribute):
                                continue
                            offset = skipped if skipped is not None else (1 if method else 0)
                            bind_parameters(args, node.args, keywords, offset)
                        # A class built from its fields carries what each field is handed.
                        fields = index["fields"].get(name, ())
                        given = dict(zip(fields, node.args))
                        given.update((k, v) for k, v in keywords if k in fields)
                        for field, value in given.items():
                            stores = self.stores_of(value)
                            if stores:
                                carrier = store_classes.get(name) or f"{self.rel}:{name}"
                                add(self.carried.setdefault(carrier, {}), field, stores)
                                if carrier not in STORES:
                                    add(self.classes, name, {carrier})
                    if isinstance(node.func, ast.Name) and node.func.id == "setattr" and len(node.args) >= 3 \
                            and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                        # ``setattr(x, "name", value)`` binds ``x.name``.
                        target = ast.Attribute(value=node.args[0], attr=node.args[1].value, ctx=ast.Store())
                        bind_target(target, node.args[2])
                        carry(node, target, node.args[2])
                    # A function handed over uncalled, with what it will be called with.
                    for handed, args, handed_keywords in _handed(node):
                        hname = handed.id if isinstance(handed, ast.Name) else handed.attr
                        for sargs, skipped in signatures.get(hname, ()):
                            positional = _parameters(sargs)[0]
                            method = bool(positional) and positional[0].arg in ("self", "cls")
                            if skipped is None and method != isinstance(handed, ast.Attribute):
                                continue
                            bind_parameters(sargs, args, handed_keywords,
                                            skipped if skipped is not None else (1 if method else 0))
                    if isinstance(func, (ast.Subscript, ast.Call, ast.Name)):
                        # A dispatch table: ``T[key](...)``, ``T.get(key)(...)``,
                        # ``self.T[key](...)``, or a name bound to one of those,
                        # calls one of the functions or lambdas its literal holds.
                        for held in self._held(index, func):
                            if isinstance(held, ast.Lambda):
                                bind_parameters(held.args, node.args, keywords, 0)
                                continue
                            if isinstance(held, ast.Attribute):
                                # A bound method of this module's classes; a
                                # function of another module is bound across.
                                if not self.owners(held.value):
                                    for args, offset in index["methods"].get("." + held.attr, ()):
                                        bind_parameters(args, node.args, keywords, offset)
                                continue
                            for args, skipped in signatures.get(held, ()):
                                positional = _parameters(args)[0]
                                if skipped is None and not (positional and positional[0].arg in ("self", "cls")):
                                    bind_parameters(args, node.args, keywords, 0)
        self._calls_out(index)

    def _calls_out(self, index):
        """Store objects handed to another module: a function or a class by name, or a method by its name."""
        imported = None
        for node in index["calls"]:
            # A function handed over uncalled receives what it will be called with.
            for handed, args, handed_keywords in _handed(node):
                self._hand_out(handed, [(i, self.stores_of(a)) for i, a in enumerate(args)]
                               + [(k, self.stores_of(v)) for k, v in handed_keywords])
            passed = [(i, self.stores_of(a)) for i, a in enumerate(node.args)]
            passed += [(kw.arg, self.stores_of(kw.value)) for kw in node.keywords if kw.arg]
            passed = [(key, stores) for key, stores in passed if stores]
            if not passed:
                continue
            origins = []
            func = node.func
            if isinstance(func, ast.Call) and isinstance(func.func, ast.Name) and func.func.id == "getattr" \
                    and len(func.args) >= 2 and isinstance(func.args[1], ast.Constant) \
                    and isinstance(func.args[1].value, str):
                # ``getattr(mod, "f")(...)`` calls what ``mod.f(...)`` calls.
                func = ast.Attribute(value=func.args[0], attr=func.args[1].value, ctx=ast.Load())
            if isinstance(func, ast.Name) and func.id in self.origins:
                origins = sorted(self.origins[func.id])
            elif isinstance(func, ast.Attribute):
                # Every module the receiver may name: a name bound to several
                # modules hands the store object to each.
                origins = [(owner, func.attr) for owner in sorted(self.owners(func.value))]
            # A dispatch table hands it to what it holds of other modules: an
            # imported function, a module's function, or a method by its name.
            methods = []
            for held in self._held(index, node.func) if isinstance(node.func, (ast.Subscript, ast.Call, ast.Name)) \
                    else ():
                if isinstance(held, str) and held in self.origins:
                    origins.extend(sorted(self.origins[held]))
                elif isinstance(held, ast.Attribute):
                    owners = self.owners(held.value)
                    origins.extend((owner, held.attr) for owner in sorted(owners))
                    if not owners:
                        # ``self._drop`` too: the method may come from a base
                        # class of another module.
                        methods.append(held.attr)
            if methods:
                if imported is None:
                    imported = self._reached_modules()
                for module in sorted(imported):
                    for attr in methods:
                        for key, stores in passed:
                            self.calls_out.setdefault((module, "." + attr), {}).setdefault(key, set()).update(stores)
            if origins:
                # Each place the function may be defined receives the store.
                for origin in sorted({d for o in origins for d in self.package.definitions(o)}):
                    if origin[0] != self.rel:
                        for key, stores in passed:
                            self.calls_out.setdefault(origin, {}).setdefault(key, set()).update(stores)
                continue
            if isinstance(func, ast.Attribute):
                if imported is None:
                    imported = self._reached_modules()
                for module in sorted(imported):
                    for key, stores in passed:
                        self.calls_out.setdefault((module, "." + func.attr), {}).setdefault(
                            key, set()).update(stores)

    def _hand_out(self, func, passed):
        """Store objects a function of another module will be called with: each place it may be defined receives
        them; a method on a receiver the census cannot name, every method of that name in the modules reached."""
        passed = [(key, stores) for key, stores in passed if stores]
        if not passed:
            return
        if isinstance(func, ast.Name):
            origins = sorted(self.origins.get(func.id, ()))
        elif isinstance(func, ast.Attribute) and self.owners(func.value):
            origins = [(owner, func.attr) for owner in sorted(self.owners(func.value))]
        elif isinstance(func, ast.Attribute):
            for module in sorted(self._reached_modules()):
                for key, stores in passed:
                    self.calls_out.setdefault((module, "." + func.attr), {}).setdefault(key, set()).update(stores)
            return
        else:
            return
        for origin in sorted({d for o in origins for d in self.package.definitions(o)}):
            if origin[0] != self.rel:
                for key, stores in passed:
                    self.calls_out.setdefault(origin, {}).setdefault(key, set()).update(stores)

    def _reached_modules(self):
        """The package modules this module imports, or looks up by a constant name, bound to a name or not."""
        out = set(self.package.imported_modules(self.rel)) | set(self.modules.values())
        for modules in list(self.more_modules.values()) + list(self.attr_modules.values()):
            out |= modules
        for node in self.package.index(self.rel)["nodes"]:
            if isinstance(node, (ast.Call, ast.Subscript)):
                out |= _module_calls(self.package, node)
        return out - {self.rel}

    def _held(self, index, func):
        """What the dispatch tables a call's function reads hold: ``T[key]``, ``T.get(key)``, a name bound to
        either."""
        keys = self.table_aliases.get(func.id, set()) if isinstance(func, ast.Name) else {_table_key(func)} - {None}
        return [h for key in sorted(keys) for h in index["tables"].get(key, ())]

    def collect_sites(self):
        """Every site of the module, as (Site, node, owner function)."""
        self.sites = []
        if self.tree is None or self.inert:
            return
        index = self.package.index(self.rel)

        def foreign(receiver):
            return sorted(s for s in self.stores_of(receiver) if s in STORES and STORES[s].house != self.rel)

        def classes(receiver):
            # A store's class itself: setting one of its attributes rewires
            # every object of it, wherever it is done.
            return {s for s in self.symbol(receiver, "class") if s in STORES}

        for child in index["references"]:
            owner, names = _owner(index, child)
            qual = ".".join(names) if owner is not None else None
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) \
                    and child.func.attr in ("__setattr__", "__delattr__") and isinstance(child.func.value, ast.Name) \
                    and child.func.value.id in ("object", "type") and child.args:
                # ``object.__setattr__(j, ...)`` is ``setattr(j, ...)``.
                for store in sorted(set(foreign(child.args[0])) | classes(child.args[0])):
                    self.sites.append((Site(store, child.func.attr, child.lineno, qual), child, owner))
                continue
            if isinstance(child, ast.Attribute) and not isinstance(child.ctx, ast.Load):
                # An attribute of a store object set or deleted outside its
                # house rewires the store: ``j._db_path = user_path`` moves it.
                for store in sorted(set(foreign(child.value)) | classes(child.value)):
                    self.sites.append((Site(store, child.attr, child.lineno, qual), child, owner))
            elif isinstance(child, (ast.Attribute, ast.Name)) and isinstance(child.ctx, ast.Load):
                if _only_tested(child, index["parents"]):
                    continue
                label = child.attr if isinstance(child, ast.Attribute) else child.id
                for store in sorted(self.write_ref(child)):
                    self.sites.append((Site(store, label, child.lineno, qual), child, owner))
                if isinstance(child, ast.Attribute) and (child.attr in _REBUILDING or (
                        child.attr.startswith("_") and not child.attr.startswith("__"))):
                    # A private member of a store, reached outside its house:
                    # its connection, its rows, its cache; or what rebuilds it
                    # in place -- ``Jot.__init__(j, path)`` through the class
                    # too. Nothing outside the house may write through it
                    # unseen.
                    rebuilt = classes(child.value) if child.attr in _REBUILDING else set()
                    for store in sorted(set(foreign(child.value)) | rebuilt):
                        self.sites.append((Site(store, child.attr, child.lineno, qual), child, owner))
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name) \
                    and child.func.id in ("getattr", "setattr", "delattr") and len(child.args) >= 2:
                name = child.args[1]
                constant = isinstance(name, ast.Constant) and isinstance(name.value, str)
                label = name.value if constant else f"<{child.func.id}>"
                if child.func.id != "getattr":
                    for store in sorted(classes(child.args[0]) - {s for s in self.stores_of(child.args[0])}):
                        self.sites.append((Site(store, label, child.lineno, qual), child, owner))
                for store in sorted(s for s in self.stores_of(child.args[0]) if s in STORES):
                    outside = STORES[store].house != self.rel
                    if child.func.id != "getattr":
                        if outside:
                            self.sites.append((Site(store, label, child.lineno, qual), child, owner))
                    elif not constant or name.value in STORES[store].writes or (outside and (
                            name.value in _REBUILDING or (name.value.startswith("_")
                                                          and not name.value.startswith("__")))):
                        self.sites.append((Site(store, label, child.lineno, qual), child, owner))
                if child.func.id == "getattr" and self.owners(child.args[0]):
                    # ``getattr(mod, "put_line")`` on a module that exports a
                    # write function reaches it as ``mod.put_line`` does.
                    if constant:
                        written = self.symbol(ast.Attribute(value=child.args[0], attr=name.value, ctx=ast.Load()),
                                              "sink")
                    else:
                        written = set().union(*(stores for owner in self.owners(child.args[0])
                                                for kind, stores in set().union(
                                                    *self.world.exports.get(owner, {}).values() or [set()])
                                                if kind == "sink"))
                    for store in sorted(written):
                        self.sites.append((Site(store, label, child.lineno, qual), child, owner))

    def exported(self):
        """Module-level symbols another module may import from this one."""
        top = set()
        for stmt in _top_level(self.tree) if self.tree is not None else ():
            if isinstance(stmt, (*_FUNCTIONS, ast.ClassDef)):
                top.add(stmt.name)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for target in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]):
                    top |= _target_names(target)[0]
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for alias in stmt.names:
                    if alias.name != "*":
                        top.add(alias.asname or alias.name.split(".")[0])
                    elif isinstance(stmt, ast.ImportFrom):
                        # A star re-exports what its source exports.
                        source = self.package.source_of(self.rel, stmt)
                        top |= set(self.world.exports.get(source, {})) if source is not None else set()
        top |= getattr(self, "lazy", set())
        out = {}
        for kind in _KINDS:
            for name, stores in self._table(kind).items():
                if stores and name in top:
                    out.setdefault(name, set()).add((kind, frozenset(stores)))
        return out


def _merge(table, key, bound):
    """Merge ``bound`` into ``table[key]``; True when anything was added."""
    current = table.setdefault(key, {})
    grew = False
    for k, stores in bound.items():
        have = current.setdefault(k, set())
        if not set(stores) <= have:
            have.update(stores)
            grew = True
    return grew


class Census:
    """The census of an estate: every module that can reach a store, its sites, and the package around it."""

    def __init__(self, estate):
        self.estate = estate
        self.package = Package(estate.modules)
        self.world = _World(self.package)
        self.modules = {}
        self._references, self._callers, self._roots, self._reach = {}, {}, {}, {}
        self._attr_proofs, self._builders, self._attr_sites = {}, None, None
        package, world = self.package, self.world
        houses = sorted({s.house for s in STORES.values()} & set(package.texts))
        seeds = set()
        for exports in world.exports.values():
            seeds |= set(exports)
        queue = houses + sorted(package.spelling(seeds) - set(houses)) if package.texts else []
        queued = set(queue)
        while queue:
            rel = queue.pop(0)
            queued.discard(rel)
            # Every module that spells a symbol a store module exports is read
            # by its syntax tree, never by its text: whether it imports a store
            # module is the census's to say (``inert``), and it is woken again
            # if one of its sources starts exporting.
            if package.trees[rel] is None:
                continue
            census = ModuleCensus(world, rel)
            self.modules[rel] = census
            wake = set()
            new = census.exported()
            if new:
                merged = {n: set(e) for n, e in world.exports.get(rel, {}).items()}
                for n, e in new.items():
                    merged.setdefault(n, set()).update(e)
                if merged != world.exports.get(rel):
                    changed = {n for n in merged if merged[n] != world.exports.get(rel, {}).get(n)}
                    world.exports[rel] = merged
                    wake |= package.spelling(changed) - {rel}
            for origin, bound in census.calls_out.items():
                if _merge(world.params, origin, bound):
                    wake.add(origin[0])
            for store, members in census.carried.items():
                grown = {key for key, stores in members.items()
                         if not set(stores) <= world.members.get(store, {}).get(key, set())}
                world.callable = world.callable or "()__call__" in members
                if _merge(world.members, store, members):
                    # Only a module that spells the attribute or the method can
                    # reach what a carrier holds there.
                    names = {key[2:] if key.startswith("()") else key for key in grown}
                    wake |= package.spelling(names) - {rel}
            for other in sorted(wake):
                if other not in queued and other in package.texts:
                    queue.append(other)
                    queued.add(other)
        for census in self.modules.values():
            census.collect_sites()

    def reached(self, rel):
        """Every package module a module imports, looks up by a constant name, or runs with ``runpy`` (a package
        run so runs its ``__main__``), read on its syntax tree."""
        if rel in self._reach:
            return self._reach[rel]
        out = set()
        if rel in self.package.texts and self.package.trees[rel] is not None:
            out |= self.package.imported_modules(rel)
            index = self.package.index(rel)
            for node in index["nodes"]:
                if isinstance(node, (ast.Call, ast.Subscript)):
                    out |= _module_calls(self.package, node)
                if not isinstance(node, ast.Call):
                    continue
                name = node.args[0] if node.args else next(
                    (kw.value for kw in node.keywords if kw.arg in ("name", "mod_name", "path_name")), None)
                name = name.value if isinstance(name, ast.Constant) and isinstance(name.value, str) else None
                if _leaf(node.func) in ("import_module", "__import__") and name and not name.startswith("."):
                    # Importing ``a.b.c`` runs ``a``, ``a.b`` and ``a.b.c``.
                    parts = name.split(".")
                    out |= {m for m in (self.package.resolve(".".join(parts[:i])) for i in range(1, len(parts) + 1))
                            if m is not None}
                    if _leaf(node.func) == "__import__":
                        out |= self._fromlist_modules(node, name)
                if _dotted(index, node.func, may=True) in ("runpy.run_module", "runpy.run_path"):
                    if name is None:
                        # A program run by a name the census cannot read may
                        # be any program of the package.
                        out |= {rel for rel in self.package.texts if rel.endswith("/__main__.py")}
                        continue
                    target = self.package.resolve(name.strip("/").removesuffix(".py").replace("/", "."))
                    if target is not None:
                        out.add(target)
                        main = target[: -len("__init__.py")] + "__main__.py" if target.endswith("__init__.py") else None
                        if main in self.package.texts:
                            out.add(main)
        self._reach[rel] = out
        return out

    def _fromlist_modules(self, call, name):
        """The submodules an ``__import__`` of ``name`` imports through its fromlist: each constant item that names
        one; for a fromlist the census cannot read -- a name, a star, an item that is no constant string -- every
        module under ``name``."""
        fromlist = call.args[3] if len(call.args) > 3 else next(
            (kw.value for kw in call.keywords if kw.arg == "fromlist"), None)
        if fromlist is None or (isinstance(fromlist, ast.Constant) and not fromlist.value):
            return set()
        items = fromlist.elts if isinstance(fromlist, (ast.List, ast.Tuple, ast.Set)) else None
        if items is not None and all(isinstance(i, ast.Constant) and isinstance(i.value, str) and i.value != "*"
                                     for i in items):
            return {m for m in (self.package.resolve(f"{name}.{i.value}") for i in items) if m is not None}
        prefix = name.replace(".", "/") + "/"
        return {m for m in self.package.texts if m.startswith(prefix)}

    @property
    def unparsed(self):
        """Every module the census needed and could not parse, so far."""
        return dict(self.package.trees.unparsed)

    def sites(self):
        """rel -> [Site] for every module with a site."""
        return {rel: [s for s, _n, _o in c.sites] for rel, c in sorted(self.modules.items()) if c.sites}

    def records(self, rel):
        census = self.modules.get(rel)
        return census.sites if census is not None else []

    def references(self, name):
        """Every reference to ``name`` in the package: (rel, node, owner, qualified name, is a call)."""
        if name in self._references:
            return self._references[name]
        found = []
        for rel in sorted(self.package.spelling({name})):
            if self.package.trees[rel] is None:
                # A module that spells the name and does not parse may call it
                # from anywhere: one reference no rule can gate, and the module
                # is named among the unparsed.
                found.append((rel, None, None, None, False))
                continue
            index = self.package.index(rel)
            for node in index["references"]:
                if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
                    pass
                elif isinstance(node, ast.Attribute) and node.attr == name and isinstance(node.ctx, ast.Load):
                    pass
                elif self._getattr_named(node, {name}):
                    # ``getattr(x, "name")`` reaches it as ``x.name`` does.
                    pass
                else:
                    continue
                owner, qual = _owner(index, node)
                parent = index["parents"].get(id(node))
                is_call = isinstance(parent, ast.Call) and parent.func is node
                found.append((rel, node, owner, ".".join(qual) if owner is not None else None, is_call))
        self._references[name] = found
        return found


    def callers(self, rel, function):
        """Every reference to one function, resolved by where it is defined.

        A nested function is reached by its name inside the function that
        defines it; a method, by attribute, anywhere, under its name; a
        module-level function, by its name in its module, and elsewhere by the
        name an import from a package module binds it to, or as an attribute of
        an imported package module. Each is (rel, node, owner, qualified name,
        is a call); a module that may hold one and does not parse is one
        reference no rule can gate.
        """
        key = (rel, id(function))
        if key in self._callers:
            return self._callers[key]
        index = self.package.index(rel)
        parents = index["parents"]
        scope = parents.get(id(function))
        while scope is not None and not isinstance(scope, (ast.Module, ast.ClassDef, *_FUNCTIONS)):
            scope = parents.get(id(scope))
        name = function.name
        if isinstance(scope, _FUNCTIONS) and any(isinstance(n, ast.Global) and name in n.names for n in ast.walk(scope)):
            # Declared global, a nested definition binds a module-level name:
            # read it as one.
            scope = self.package.trees[rel]
        found = []

        def record(r_rel, node):
            r_index = self.package.index(r_rel)
            owner, qual = _owner(r_index, node)
            parent = r_index["parents"].get(id(node))
            is_call = isinstance(parent, ast.Call) and parent.func is node
            found.append((r_rel, node, owner, ".".join(qual) if owner is not None else None, is_call))

        # A decorator that is not one of the plain method markers is handed the
        # function itself: a reference no rule can gate.
        if not all(_benign(index, function, d) for d in function.decorator_list):
            found.append((rel, None, None, None, False))
        if isinstance(scope, _FUNCTIONS):
            for node in ast.walk(scope):
                if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
                    record(rel, node)
        elif isinstance(scope, ast.ClassDef):
            # A method: by attribute, anywhere, under its name; and by name in
            # its class body (an alias of it there).
            for node in ast.walk(scope):
                if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
                    record(rel, node)
            for r_rel in sorted(self.package.spelling({name})):
                if self.package.trees[r_rel] is None:
                    found.append((r_rel, None, None, None, False))
                    continue
                for node in self.package.index(r_rel)["references"]:
                    if isinstance(node, ast.Attribute) and node.attr == name and isinstance(node.ctx, ast.Load):
                        record(r_rel, node)
                    elif self._getattr_named(node, {name}):
                        found.append(self._reference(r_rel, node, call=False))
        else:
            aliases = self._aliases(rel, name)
            spelled = {alias for _module, alias in aliases}
            for r_rel in sorted(self.package.spelling(spelled)):
                if self.package.trees[r_rel] is None:
                    found.append((r_rel, None, None, None, False))
                    continue
                names = {alias for module, alias in aliases if module == r_rel}
                if r_rel in self.package.stars():
                    names |= spelled
                holders = {}
                for module, alias in aliases:
                    holders.setdefault(alias, set()).add(module)
                roots = self._module_aliases(r_rel)
                importing = r_rel in {m for m, _a in aliases} or bool(
                    self.reached(r_rel) & {m for m, _a in aliases})
                typed = self.modules.get(r_rel)
                for node in self.package.index(r_rel)["references"]:
                    if isinstance(node, ast.Name) and node.id in names and isinstance(node.ctx, ast.Load):
                        record(r_rel, node)
                    elif isinstance(node, ast.Attribute) and node.attr in holders and isinstance(node.ctx, ast.Load):
                        # ``m.f`` reaches f when m names a module that holds it.
                        # A receiver the census cannot name may be that module,
                        # in a module that imports it, unless the census knows
                        # the receiver for a store object or a carrier.
                        modules = self._value_modules(node.value, roots)
                        if modules is None:
                            if importing and not self._typed_receiver(typed, r_rel, node.value):
                                record(r_rel, node)
                        elif modules & holders[node.attr]:
                            record(r_rel, node)
                    elif self._getattr_named(node, spelled) and importing and not self._typed_receiver(
                            typed, r_rel, node.args[0]):
                        found.append(self._reference(r_rel, node, call=False))
        self._callers[key] = found
        return found

    def _typed_receiver(self, typed, rel, receiver):
        """True when the census knows a receiver for a store object or a carrier, hence not a module.

        A name is known so only when every binding of it, anywhere in the
        module, assigns a store object or a carrier: names are not scoped, and
        a name a parameter, a loop, a lambda or another assignment binds may be
        handed the module itself. An attribute or a call is known so when the
        census resolves it to one, and no module binds an attribute of that
        name to a module.
        """
        if typed is None:
            return False
        if isinstance(receiver, ast.Attribute):
            return self._attribute_proof(typed, rel, receiver)
        return self._must_store(typed, rel, receiver, set())

    @staticmethod
    def _carriers(typed, value):
        """The carrier classes, as (module, class), an expression may be an object of."""
        return frozenset(tuple(c.rsplit(":", 1)) for c in typed.stores_of(value) if c not in STORES and ":" in c)

    def _attribute_proof(self, typed, rel, receiver):
        """True when ``x.attr`` is a store object on every path. When the census types ``x`` as an object of carrier
        classes, what those classes bind, and what is set on one of their objects, is read -- and a class ``x``
        may be built from directly besides them (``Holder() if flag else Other()``) binds the attribute to a store
        object wherever it binds it; otherwise every binding of the attribute anywhere is read."""
        carriers = self._carriers(typed, receiver.value)
        if carriers:
            family = self._carrier_family(carriers)
            for other in sorted(self._constructed(rel, receiver.value, set()) - family):
                if not self._class_binds_store(other, receiver.attr):
                    return False
        return self._must_attribute(receiver.attr, carriers)

    def _constructed(self, rel, value, seen):
        """(module, class) for every class an expression may be built from directly on some path: a construction,
        through the names every assignment binds and the branches of a conditional or an ``or``."""
        index = self.package.index(rel)
        if isinstance(value, ast.Name):
            if value.id in seen:
                return set()
            seen = seen | {value.id}
            return set().union(*(self._constructed(rel, v, seen) for kind, v in _name_bindings(index).get(value.id, ())
                                 if kind in ("assign", "walrus")))
        if isinstance(value, ast.IfExp):
            return self._constructed(rel, value.body, seen) | self._constructed(rel, value.orelse, seen)
        if isinstance(value, ast.BoolOp):
            return set().union(*(self._constructed(rel, v, seen) for v in value.values))
        if not isinstance(value, ast.Call):
            return set()
        func, places = value.func, set()
        if isinstance(func, ast.Name):
            for kind, stmt in _name_bindings(index).get(func.id, ()):
                if kind == "definition":
                    places.add((rel, func.id))
                elif kind == "import" and isinstance(stmt, ast.ImportFrom):
                    source = self.package.source_of(rel, stmt)
                    alias = next((a for a in stmt.names if (a.asname or a.name) == func.id), None)
                    if source is not None and alias is not None:
                        places |= self.package.definitions((source, alias.name))
        elif isinstance(func, ast.Attribute):
            for module in self._value_modules(func.value, self._module_aliases(rel)) or ():
                places |= self.package.definitions((module, func.attr))
        return {(m, n) for m, n in places if m in self.package.texts and self.package.trees[m] is not None and any(
            isinstance(s, ast.ClassDef) and s.name == n for s in self.package.index(m)["nodes"])}

    def _class_binds_store(self, place, attr):
        """True when a class binds an attribute only to store objects, wherever it binds it -- ``self.attr`` in its
        methods, its body -- or binds it nowhere."""
        rel, name = place
        module = self.modules.get(rel)
        index = self.package.index(rel)
        for store_rel, node in self._attribute_sites()["stores"].get(attr, ()):
            parent = index["parents"].get(id(node)) if store_rel == rel else None
            if parent is None or not (isinstance(node.value, ast.Name) and node.value.id == "self") \
                    or index["klass"].get(id(parent)) != name:
                continue
            if not (isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is not None and module is not None
                    and self._must_store(module, rel, parent.value, set())):
                return False
        for node in index["nodes"]:
            if isinstance(node, ast.ClassDef) and node.name == name:
                for stmt in _class_bindings(node).get(attr, ()):
                    if not (isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None
                            and module is not None and self._must_store(module, rel, stmt.value, set())):
                        return False
        return True

    def _carrier_family(self, carriers):
        """The carrier classes and every class of the census built on one of them, as (module, class)."""
        family = set(carriers)
        ids = {f"{rel}:{name}" for rel, name in carriers}
        for rel, module in self.modules.items():
            family |= {(rel, name) for name, held in module.classes.items() if held & ids}
        return frozenset(family)

    def _must_attribute(self, attr, carriers=frozenset()):
        """True when every assignment of an attribute of this name surely binds a store object, and there is one
        (``self.core = CoreStore(...)``). With carrier classes known -- the receiver one of their objects on every
        path -- every ``self.attr`` in them and their subclasses is read, every assignment of it on any other
        object, and a ``setattr``, a ``__setattr__``, a ``__delattr__``, a ``__dict__`` or a ``vars()`` on an
        object the census types as one of theirs undoes the proof; with none known, every assignment of it in
        every class, and any of those on any object. An assignment that is no plain one -- augmented, unpacked, a
        loop or a ``with`` target, a method of that name -- may bind anything."""
        key = (attr, carriers)
        if key in self._attr_proofs:
            return self._attr_proofs[key]
        self._attr_proofs[key] = False
        family = self._carrier_family(carriers) if carriers else frozenset()
        ids = {f"{rel}:{name}" for rel, name in family}
        sites = self._attribute_sites()
        found, ok = False, True
        for rel, node in sites["stores"].get(attr, ()):
            index = self.package.index(rel)
            parent = index["parents"].get(id(node))
            on_self = isinstance(node.value, ast.Name) and node.value.id == "self"
            if family and on_self and (rel, index["klass"].get(id(parent))) not in family:
                continue
            targets = parent.targets if isinstance(parent, ast.Assign) else (
                [parent.target] if isinstance(parent, ast.AnnAssign) and parent.value is not None else [])
            if any(t is node for t in targets):
                found = True
                ok = ok and self._must_store(self.modules[rel], rel, parent.value, set())
            else:
                ok = False
        for rel, receiver, name in sites["dynamic"] if ok else ():
            if isinstance(name, ast.Constant) and name.value != attr:
                continue
            if not family or self._may_carry(rel, receiver, ids):
                ok = False
        for rel, receiver in sites["dicts"] if ok else ():
            if not family or self._may_carry(rel, receiver, ids):
                ok = False
        for rel, node in sites["classes"] if ok else ():
            if family and (rel, node.name) not in family:
                continue
            for stmt in _class_bindings(node).get(attr, ()):
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
                    found = True
                    ok = ok and self._must_store(self.modules[rel], rel, stmt.value, set())
                else:
                    ok = False
        self._attr_proofs[key] = found and ok
        return found and ok

    def _may_carry(self, rel, receiver, ids):
        """True when the census types an object as one of the carrier classes', on some path: a parameter is typed
        by every call the census reads that hands it one."""
        module = self.modules.get(rel)
        return bool(module is not None and module.stores_of(receiver) & ids)

    def _attribute_sites(self):
        """Every node an attribute proof reads, found in one walk of the modules the census reads: an attribute
        set or deleted, by its name; a ``setattr``, ``delattr``, ``__setattr__`` or ``__delattr__``, with its
        receiver and the name it sets; a ``__dict__`` or a ``vars()`` of an object; every class."""
        if self._attr_sites is None:
            stores, dynamic, dicts, classes = {}, [], [], []
            for rel, module in sorted(self.modules.items()):
                if module.tree is None:
                    continue
                for node in self.package.index(rel)["nodes"]:
                    if isinstance(node, ast.Attribute):
                        if not isinstance(node.ctx, ast.Load):
                            stores.setdefault(node.attr, []).append((rel, node))
                        if node.attr == "__dict__":
                            dicts.append((rel, node.value))
                    elif isinstance(node, ast.Call):
                        func = node.func
                        if isinstance(func, ast.Name) and func.id in ("setattr", "delattr") and len(node.args) >= 2:
                            dynamic.append((rel, node.args[0], node.args[1]))
                        elif isinstance(func, ast.Attribute) and func.attr in ("__setattr__", "__delattr__"):
                            if isinstance(func.value, ast.Name) and func.value.id in ("object", "type"):
                                if len(node.args) >= 2:
                                    dynamic.append((rel, node.args[0], node.args[1]))
                            elif node.args:
                                dynamic.append((rel, func.value, node.args[0]))
                        elif isinstance(func, ast.Name) and func.id == "vars" and node.args:
                            dicts.append((rel, node.args[0]))
                    elif isinstance(node, ast.ClassDef):
                        classes.append((rel, node))
            self._attr_sites = {"stores": stores, "dynamic": dynamic, "dicts": dicts, "classes": classes}
        return self._attr_sites

    def _must_store(self, typed, rel, value, seen):
        """True when an expression is a store object on every path: a construction of a store's class, a call of
        a declared accessor, a name every binding of which is one, or a conditional whose every branch is. A
        container, a subscript, or a function that may hand back something else is no proof."""
        if isinstance(value, ast.Name):
            if value.id in seen:
                return False
            seen = seen | {value.id}
            bound = _name_bindings(self.package.index(rel)).get(value.id, ())
            return bool(bound) and all(kind in ("assign", "walrus") and self._must_store(typed, rel, v, seen)
                                       for kind, v in bound)
        if isinstance(value, ast.IfExp):
            return self._must_store(typed, rel, value.body, seen) and self._must_store(typed, rel, value.orelse, seen)
        if isinstance(value, ast.BoolOp):
            return all(self._must_store(typed, rel, v, seen) for v in value.values)
        if not isinstance(value, ast.Call):
            return False
        # The function, by every binding of its name, is a store's class, a
        # subclass of one, or a declared accessor: a parameter or a local of
        # the same name may be anything.
        targets = self._store_builders()
        func = value.func
        if isinstance(func, ast.Name):
            bound = _name_bindings(self.package.index(rel)).get(func.id, ())
            for kind, stmt in bound:
                if kind == "definition" and (rel, func.id) in targets:
                    continue
                if kind == "import" and isinstance(stmt, ast.ImportFrom):
                    source = self.package.source_of(rel, stmt)
                    alias = next((a for a in stmt.names if (a.asname or a.name) == func.id), None)
                    if source is not None and alias is not None \
                            and self.package.definitions((source, alias.name)) <= targets:
                        continue
                return False
            return bool(bound)
        if isinstance(func, ast.Attribute):
            modules = self._value_modules(func.value, self._module_aliases(rel))
            return bool(modules) and all(self.package.definitions((m, func.attr)) <= targets for m in modules)
        return False

    def _store_builders(self):
        """(module, name) of every store's class and accessor at its house, and of every class a module of the
        census defines on a store's class."""
        if self._builders is None:
            out = {(spec.house, n) for spec in STORES.values() for n in spec.classes + spec.accesses}
            for rel, module in self.modules.items():
                if module.tree is None:
                    continue
                for node in _top_level(module.tree):
                    if isinstance(node, ast.ClassDef) and module.classes.get(node.name, set()) & set(STORES):
                        out.add((rel, node.name))
            self._builders = out
        return self._builders

    def _reference(self, rel, node, call):
        owner, qual = _owner(self.package.index(rel), node)
        return (rel, node, owner, ".".join(qual) if owner is not None else None, call)

    @staticmethod
    def _getattr_named(node, names):
        """True when a node is ``getattr(x, "name")`` with one of ``names`` as its constant."""
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" \
            and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value in names

    def _aliases(self, rel, name):
        """Every (module, name) a module-level function is bound to: its own, and each import or re-export of it."""
        aliases = {(rel, name)}
        grown = True
        while grown:
            grown = False
            for module in sorted(self.package.spelling({alias for _m, alias in aliases})):
                if self.package.trees[module] is None:
                    continue
                for node in self.package.imports(module):
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    source = self.package.source_of(module, node)
                    for alias in node.names if source is not None else ():
                        if alias.name == "*":
                            # A star re-exports every name its source binds.
                            for held in [a for m, a in aliases if m == source and (module, a) not in aliases]:
                                aliases.add((module, held))
                                grown = True
                        elif (source, alias.name) in aliases and (module, alias.asname or alias.name) not in aliases:
                            aliases.add((module, alias.asname or alias.name))
                            grown = True
                for local, entry in self.package.lazy_exports(module).items():
                    if entry in aliases and (module, local) not in aliases:
                        aliases.add((module, local))
                        grown = True
        return aliases

    def _module_aliases(self, rel):
        """name -> the package modules a module binds it to, by its imports or a lookup by name.

        Only a name every binding of which is such an import or lookup is
        known: names are not scoped, and a name a parameter, a loop, a lambda
        or another assignment binds anywhere in the module may hold any object,
        the module that holds a function included.
        """
        if rel in self._roots:
            return self._roots[rel]
        out, resolved = {}, {}

        def add(local, *targets):
            out.setdefault(local, set()).update(targets)
            resolved[local] = resolved.get(local, 0) + 1

        for node in self.package.imports(rel):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        target = self.package.resolve(alias.name)
                        local = alias.asname
                    else:
                        local = alias.name.split(".")[0]
                        target = self.package.resolve(local)
                    if target is not None:
                        add(local, target)
            else:
                source = self.package.source_of(rel, node)
                for alias in node.names if source is not None else ():
                    subs = self.package.submodules(source, alias.name) if alias.name != "*" else set()
                    if subs:
                        add(alias.asname or alias.name, *sorted(subs))
        index = self.package.index(rel)
        for node in index["binders"]:
            if isinstance(node, ast.Assign):
                modules = _module_calls(self.package, node.value)
                if modules:
                    for target in node.targets:
                        for local in _target_names(target)[0]:
                            add(local, *sorted(modules))
        bindings = _name_bindings(index)
        known = {name: modules for name, modules in out.items() if len(bindings.get(name, ())) == resolved[name]}
        self._roots[rel] = known
        return known

    def _value_modules(self, node, roots):
        """The package modules an expression names, or None when its root is no module alias at all."""
        if isinstance(node, ast.Name):
            return set(roots[node.id]) if node.id in roots else None
        if isinstance(node, ast.Attribute):
            parents = self._value_modules(node.value, roots)
            if parents is None:
                return None
            return {sub for parent in parents for sub in self.package.submodules(parent, node.attr)}
        if isinstance(node, (ast.Call, ast.Subscript)):
            return _module_calls(self.package, node) or None
        return None


def _root_name(node):
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def take_census(estate):
    return Census(estate)


# ---------------------------------------------------------------------------
# Classification.
# ---------------------------------------------------------------------------
def _housed(census, rel, site, node, owner):
    """True when a site is a store writing itself or a store it is built on, from its own code.

    Its own code is a method of one of its classes, one of its house
    functions or accessors, or a module-level alias that hands one of its
    write methods out under a name of the house.
    """
    houses = [name for name, spec in STORES.items()
              if spec.house == rel and (site.store == name or site.store in spec.backs)]
    if not houses:
        return False
    # The house rebinding its own class's gate or a write -- ``Jot._check =
    # ...``, ``setattr(Jot, "put", ...)`` -- writes around its gate: no house
    # covers that.
    rebinding = (isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load)) or (
        isinstance(node, ast.Call) and _leaf(node.func) in ("setattr", "delattr", "__setattr__", "__delattr__"))
    gates = {g.name for g in GATES.values() if g.kind == "house" and g.home == rel}
    if rebinding and (site.method in gates or any(site.method in STORES[n].writes for n in houses)):
        return False
    if owner is None:
        parents = census.package.index(rel)["parents"]
        parent = parents.get(id(node))
        return isinstance(parent, ast.Assign) and parent.value is node \
            and all(isinstance(t, ast.Name) for t in parent.targets) \
            and parent in _top_level(census.package.trees[rel])
    head = (site.function or "").split(".")[0]
    return any(head in STORES[n].classes or head in STORES[n].functions or head in STORES[n].accesses
               for n in houses)


def _pairs(census):
    """(rel, store) -> [(Site, node, owner)] for every site no house covers."""
    cached = getattr(census, "_pairs", None)
    if cached is not None:
        return cached
    out = {}
    for rel, module in sorted(census.modules.items()):
        for site, node, owner in module.sites:
            if not _housed(census, rel, site, node, owner):
                out.setdefault((rel, site.store), []).append((site, node, owner))
    census._pairs = out
    return out


def _housed_count(census):
    return sum(1 for rel, m in census.modules.items() for s, n, o in m.sites if _housed(census, rel, s, n, o))


class _Gates:
    """Gate presence for the sites of one pair."""

    def __init__(self, census, gates):
        self.census = census
        self.gates = [(name, GATES[name]) for name in gates if name in GATES]
        self.memo = {}

    def _resolves(self, rel, func, gate):
        """True when a call's function is the gate, as the module names it: its name, bound once in the module
        by the gate's definition or by its import, or an attribute of the gate's home module and of no other."""
        package = self.census.package
        if not self._defined_once(gate):
            return False
        if isinstance(func, ast.Name) and func.id == gate.name:
            bound = _name_bindings(package.index(rel)).get(func.id, ())
            if rel == gate.home:
                return len(bound) == 1 and bound[0][0] == "definition"
            # Every binding an import that leads to the gate: a name imported
            # twice, or through a re-export, is still the gate.
            return bool(bound) and all(kind == "import" and isinstance(stmt, ast.ImportFrom) and any(
                (a.asname or a.name) == func.id and package.source_of(rel, stmt) is not None
                and package.definitions((package.source_of(rel, stmt), a.name)) == {(gate.home, gate.name)}
                for a in stmt.names) for kind, stmt in bound)
        if isinstance(func, ast.Attribute) and func.attr == gate.name:
            return self.census._value_modules(func.value, self.census._module_aliases(rel)) == {gate.home}
        return False

    def _defined_once(self, gate):
        """True when the gate's home defines one function of the gate's name -- outside ``if TYPE_CHECKING:`` --
        and binds the name at module level by nothing else."""
        package = self.census.package
        if gate.home not in package.texts or package.trees[gate.home] is None:
            return False
        index = package.index(gate.home)
        defined = [n for n in index["nodes"] if isinstance(n, _FUNCTIONS) and n.name == gate.name
                   and not _type_checking_only(index, n)]
        top = [s for s in _top_level(package.trees[gate.home])
               if gate.name in _statement_names(s) | _expression_names(s)]
        # A function that declares the name global may bind it anew.
        return len(defined) == 1 and len(top) <= 1 and all(isinstance(s, _FUNCTIONS) for s in top) and not any(
            isinstance(n, ast.Global) and gate.name in n.names for n in index["nodes"])

    def _gate_call(self, rel, node, kinds):
        return isinstance(node, ast.Call) and any(
            g.kind in kinds and self._resolves(rel, node.func, g) for _n, g in self.gates)

    def _scope(self, rel, function, kinds):
        """(counts, gated): how often each name is bound in the function's own scope, its parameters
        included, and which of them a binding ties to a gate call of ``kinds``.

        A nested function, lambda or class is a scope of its own and is not
        read here; a name declared global or nonlocal counts as bound beyond
        reach.
        """
        counts, gated = {}, set()
        params = _parameters(function.args)
        extras = [function.args.vararg, function.args.kwarg]
        for param in params[0] + params[1] + [p for p in extras if p is not None]:
            counts[param.arg] = counts.get(param.arg, 0) + 1
        stack = list(function.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (*_FUNCTIONS, ast.Lambda, ast.ClassDef)):
                if isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
                    counts[node.name] = counts.get(node.name, 0) + 1
                # A nested scope that declares a name nonlocal or global may
                # rebind this one behind its back.
                for inner in ast.walk(node):
                    if isinstance(inner, (ast.Nonlocal, ast.Global)):
                        for name in inner.names:
                            counts[name] = counts.get(name, 0) + 99
                # Its header runs here: a walrus in a default, a decorator or
                # a base binds in this scope.
                stack.extend(_header(node))
                continue
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                for name in node.names:
                    counts[name] = counts.get(name, 0) + 99
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for name in _target_names(target)[0]:
                        counts[name] = counts.get(name, 0) + 1
                        value = getattr(node, "value", None)
                        if isinstance(target, ast.Name) and not isinstance(node, ast.AugAssign) \
                                and self._gate_call(rel, value, kinds):
                            gated.add(name)
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension, ast.withitem, ast.ExceptHandler)):
                target = node.optional_vars if isinstance(node, ast.withitem) else getattr(node, "target", None)
                names = _target_names(target)[0] if target is not None else set()
                if isinstance(node, ast.ExceptHandler) and node.name:
                    names = {node.name}
                for name in names:
                    counts[name] = counts.get(name, 0) + 2
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    counts[local] = counts.get(local, 0) + 1
            elif isinstance(node, (ast.MatchAs, ast.MatchStar, ast.MatchMapping)):
                # A capture pattern binds a name: ``case ok:`` rebinds a verdict.
                name = node.rest if isinstance(node, ast.MatchMapping) else node.name
                if name:
                    counts[name] = counts.get(name, 0) + 2
            elif type(node).__name__ == "TypeAlias" and isinstance(getattr(node, "name", None), ast.Name):
                counts[node.name.id] = counts.get(node.name.id, 0) + 2
            stack.extend(ast.iter_child_nodes(node))
        return counts, gated

    def _usable(self, rel, functions, kinds):
        """Names that stand for a gate of ``kinds`` at a node inside ``functions`` (innermost first).

        The name is read in the innermost function that binds it: there it
        must be bound exactly once, to a gate call; an approval parameter
        must never be bound again.
        """
        usable, shadowed = set(), set()
        for function in functions:
            counts, gated = self._scope(rel, function, kinds)
            for name in gated:
                if name not in shadowed and counts.get(name) == 1:
                    usable.add(name)
            if "approval" in kinds:
                params = _parameters(function.args)
                for param in params[0] + params[1]:
                    if param.arg not in shadowed and counts.get(param.arg) == 1 and any(
                            g.kind == "approval" and g.name == param.arg and g.home == rel for _n, g in self.gates):
                        usable.add(param.arg)
            shadowed |= set(counts)
        return usable

    def _verdicts(self, rel, functions):
        """What stands for a verdict or an approval in these functions: names bound once to a gate, parameters."""
        return self._usable(rel, functions, {"verdict", "approval"})

    def _is_verdict(self, rel, node, names):
        if self._gate_call(rel, node, {"verdict"}):
            return True
        return isinstance(node, ast.Name) and node.id in names

    def _positive(self, rel, test, names):
        if self._is_verdict(rel, test, names):
            return True
        return isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) and any(
            self._positive(rel, v, names) for v in test.values)

    def _negative(self, rel, test, names):
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return self._positive(rel, test.operand, names)
        return isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or) and any(
            self._negative(rel, v, names) for v in test.values)

    def _frames(self, rel, node):
        """(container, field, block, position) from the node's statement outward, up to the body of the function
        it runs in: a nested function runs whenever it is called, and the code around its definition decides
        nothing for it; a method is not gated by its class, nor a function by module-level code."""
        index = self.census.package.index(rel)
        parents = index["parents"]
        frames, child = [], node
        while True:
            parent = parents.get(id(child))
            if parent is None or isinstance(parent, ast.Module) or (
                    isinstance(parent, ast.ClassDef) and _in_body(parent, child)):
                return frames
            for field in _BLOCKS:
                block = getattr(parent, field, None)
                if isinstance(block, list) and child in block:
                    frames.append((parent, field, block, block.index(child)))
                    break
            if isinstance(parent, _FUNCTIONS) and _in_body(parent, child):
                return frames
            child = parent

    def _lambdas(self, rel, node):
        """Every lambda whose body holds a node, out to the module."""
        parents = self.census.package.index(rel)["parents"]
        out, child = [], node
        while True:
            current = parents.get(id(child))
            if current is None:
                return out
            if isinstance(current, ast.Lambda) and _in_body(current, child):
                out.append(current)
            child = current

    def _home(self, rel, node):
        """The function whose own body a node runs in, or None (see ``_home_of``)."""
        return _home_of(self.census.package.index(rel), node)

    def _enclosing(self, rel, node):
        """The functions whose bodies hold a node, innermost first, out to a class body or the module."""
        index = self.census.package.index(rel)
        out, child = [], node
        parents = index["parents"]
        while True:
            current = parents.get(id(child))
            if current is None or isinstance(current, ast.Module) or (
                    isinstance(current, ast.ClassDef) and _in_body(current, child)):
                return out
            if isinstance(current, _FUNCTIONS) and _in_body(current, child):
                out.append(current)
            child = current

    def dominated(self, rel, node):
        """True when a verdict, an approval or a raising gate dominates ``node`` in the function it runs in."""
        home = self._home(rel, node)
        if home is None:
            return False
        # The check must stand in the function's own body; the verdict it reads
        # may be a name the function around it bound once to the gate.
        names = self._verdicts(rel, self._enclosing(rel, node))
        for container, field, block, position in self._frames(rel, node):
            if isinstance(container, ast.If) and field == "body" and self._positive(rel, container.test, names):
                return True
            # ``if not gate(): ... else: write`` -- the else runs when the gate said yes.
            if isinstance(container, ast.If) and field == "orelse" and self._negative(rel, container.test, names):
                return True
            for stmt in block[:position]:
                if isinstance(stmt, ast.If) and self._negative(rel, stmt.test, names) and _leaves(stmt.body):
                    return True
                # ``if gate(): ... else: return`` -- past it, the gate said yes.
                if isinstance(stmt, ast.If) and self._positive(rel, stmt.test, names) and _leaves(stmt.orelse):
                    return True
                value = stmt.value if isinstance(stmt, (ast.Expr, ast.Assign, ast.AnnAssign)) else None
                if self._gate_call(rel, value, {"raises"}):
                    return True
        return False

    def filtered(self, rel, node, store):
        """True when everything the site writes comes out of a filter gate.

        The site is a call: each of its positional arguments, and each keyword
        the store names as content, is the filter's call or a name bound once
        to it; and that name is read nowhere but as such an argument, in a
        test, or measured with ``len`` -- never extended, mutated or handed on.
        """
        index = self.census.package.index(rel)
        call = index["parents"].get(id(node))
        if not (isinstance(call, ast.Call) and call.func is node):
            return False
        # What a filter covers is data: a nested function or a lambda that
        # writes a name its enclosing function bound once to the filter writes
        # that filtered value, whoever calls it -- unless a name the lambda
        # binds itself (a parameter, a comprehension target, a walrus) stands
        # in front of it.
        functions = self._enclosing(rel, node)
        lambdas = set()
        for scope in self._lambdas(rel, node):
            args = scope.args
            lambdas |= {p.arg for p in sum(_parameters(args), []) + [p for p in (args.vararg, args.kwarg) if p]}
            lambdas |= {n.id for n in ast.walk(scope.body) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        names = self._usable(rel, functions, {"filter"}) - lambdas
        content = set(STORES[store].content) if store in STORES else set()
        written = list(call.args) + [kw.value for kw in call.keywords if kw.arg in content]
        if not written:
            return False
        for arg in written:
            if self._gate_call(rel, arg, {"filter"}):
                continue
            if not (isinstance(arg, ast.Name) and arg.id in names and self._clean(rel, functions, arg.id)):
                return False
        return True

    def _clean(self, rel, functions, name):
        """True when a filtered name is read only as a write's argument, in a test, or by ``len``."""
        module = self.census.modules.get(rel)
        parents = self.census.package.index(rel)["parents"]
        binder = next((f for f in functions if name in self._scope(rel, f, {"filter"})[0]), None)
        if binder is None or module is None:
            return False
        for node in ast.walk(binder):
            if not (isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)):
                continue
            if _only_tested(node, parents):
                continue
            parent = parents.get(id(node))
            if isinstance(parent, ast.keyword):
                parent = parents.get(id(parent))
            if isinstance(parent, ast.Call) and parent.func is not node and (
                    module.write_ref(parent.func) or _leaf(parent.func) == "len"):
                continue
            if isinstance(parent, ast.IfExp) and parent.test is node:
                continue
            return False
        return True

    def decided(self, rel, node):
        """True when the node runs in a decision gate's own body: never in a function, a lambda or a generator
        expression nested in it, which runs whenever it is called, by whoever holds it."""
        home = self._home(rel, node)
        if home is None:
            return False
        index = self.census.package.index(rel)
        dotted = ".".join(_owner(index, home)[1] + (home.name,))
        if not any(g.kind in ("decision", "acceptance") and g.home == rel and g.name == dotted for _n, g in self.gates):
            return False
        # A decision defined twice, bound again, or handed to a decorator that
        # may replace it, is no one body: the other may be what runs.
        if sum(1 for n in index["nodes"] if isinstance(n, _FUNCTIONS)
               and ".".join(_owner(index, n)[1] + (n.name,)) == dotted) != 1:
            return False
        if not all(_benign(index, home, d) for d in home.decorator_list):
            return False
        scope = index["parents"].get(id(home))
        if isinstance(scope, ast.ClassDef):
            return len(_class_bindings(scope).get(home.name, ())) == 1 and not any(
                isinstance(n, ast.Attribute) and n.attr == home.name and isinstance(n.value, ast.Name)
                and n.value.id == scope.name and not isinstance(n.ctx, ast.Load) for n in index["nodes"])
        top = [s for s in _top_level(self.census.package.trees[rel])
               if home.name in _statement_names(s) | _expression_names(s)]
        return len(top) == 1 and not any(isinstance(n, ast.Global) and home.name in n.names for n in index["nodes"])

    def housed_gate(self, rel, store, method, node):
        """True when a house gate covers the site: a call of a write method on an object the census knows for the
        store and for no module, or a ``getattr`` of one by its constant name -- never a house function that only
        shares a write's name, a private member, or a ``getattr`` the census cannot read."""
        spec = STORES.get(store)
        module = self.census.modules.get(rel)
        if spec is None or module is None or method not in spec.writes:
            return False
        receiver = node.value if isinstance(node, ast.Attribute) else (
            node.args[0] if isinstance(node, ast.Call) and node.args else None)
        if receiver is None or store not in module.stores_of(receiver) or module.owners(receiver):
            return False
        # A store object on every path: a parameter handed the house module
        # would reach its function of the same name, ungated.
        census = self.census
        if not (census._attribute_proof(module, rel, receiver)
                if isinstance(receiver, ast.Attribute) else census._must_store(module, rel, receiver, set())):
            return False
        return any(g.kind == "house" and spec.house == g.home for _n, g in self.gates)

    def position(self, rel, node):
        """True when a node's position is gated: decided, dominated, or in a function gated by its callers."""
        if self.decided(rel, node) or self.dominated(rel, node):
            return True
        home = self._home(rel, node)
        return home is not None and self.function(rel, home)

    def function(self, rel, function):
        """True when every reference to the function, in the package, is a call from a gated position.

        A reference that is not a call counts only as the function of a
        ``functools.partial`` a decision gate builds and calls itself: handed
        on anywhere else, the function could be called from anywhere.
        """
        key = (rel, id(function))
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = False
        references = self.census.callers(rel, function)
        # Calling a coroutine or a generator function runs nothing: its body
        # runs where the call is awaited or iterated, and only that counts.
        lazy = _lazy(function)
        ok = bool(references) and all(
            owner is not None and (
                self.position(r_rel, r_node) and (lazy is None or _consumed(
                    self.census.package.index(r_rel), self.census.package.index(r_rel)["parents"].get(id(r_node)),
                    lazy))
                if is_call else self._called_in_place(r_rel, r_node, lazy))
            for r_rel, r_node, owner, _q, is_call in references)
        self.memo[key] = ok
        return ok

    def _called_in_place(self, rel, node, lazy=None):
        """True when a reference is the function of a ``functools.partial`` a decision gate builds in its own body
        and calls there itself: at once, or through a name of its own body, bound once, only ever called there --
        and, for a coroutine or a generator function, each call consumed on the spot. Returned, kept or handed on,
        the partial could be called from anywhere."""
        index = self.census.package.index(rel)
        parents = index["parents"]
        build = parents.get(id(node))
        if not (isinstance(build, ast.Call) and build.args and build.args[0] is node
                and _dotted(index, build.func) == "functools.partial" and self.decided(rel, build)):
            return False
        holder = parents.get(id(build))
        if isinstance(holder, ast.Call) and holder.func is build:
            return lazy is None or _consumed(index, holder, lazy)
        if isinstance(holder, ast.Assign) and holder.value is build and len(holder.targets) == 1 \
                and isinstance(holder.targets[0], ast.Name):
            name = holder.targets[0].id
        elif isinstance(holder, ast.AnnAssign) and holder.value is build and isinstance(holder.target, ast.Name):
            name = holder.target.id
        else:
            return False
        home = self._home(rel, build)
        if self._scope(rel, home, ())[0].get(name) != 1:
            return False
        for sub in ast.walk(home):
            if isinstance(sub, ast.Name) and sub.id == name and not isinstance(sub.ctx, ast.Store):
                call = parents.get(id(sub))
                if not (isinstance(call, ast.Call) and call.func is sub and self.decided(rel, call)
                        and (lazy is None or _consumed(index, call, lazy))):
                    return False
        return True

    def site(self, rel, site, node, owner):
        if self.housed_gate(rel, site.store, site.method, node):
            return True
        if self.filtered(rel, node, site.store):
            return True
        return owner is not None and self.position(rel, node)


class _Routes:
    """Whether a position is reached only from route handlers, or from the functions of named entry modules."""

    def __init__(self, census, entries=()):
        self.census = census
        self.entries = frozenset(entries)
        self.memo = {}

    def position(self, rel, node):
        """True when a node runs in the body of a route handler or of a function of an entry module, or in a
        function only ever called from such a position. A nested function, a lambda or a generator expression is
        reached by its own callers, never by the handler around its definition: a closure a handler hands to an
        agent run is the model's to call."""
        index = self.census.package.index(rel)
        home = _home_of(index, node)
        if home is None:
            return False
        if rel in self.entries and _owner(index, home)[0] is None:
            # A function or a method of an entry module, never a closure one
            # of them hands on.
            return True
        if _route_handler(self.census, rel, home) and not any(
                r_node is not None and r_owner is not home for _r, r_node, r_owner, *_rest in self.census.callers(
                    rel, home)):
            # Only the framework calls a handler: called from code, the
            # approval it carries is that code's (its own body excepted).
            return True
        return self.function(rel, home)

    def function(self, rel, function):
        if rel in self.entries and _owner(self.census.package.index(rel), function)[0] is None:
            # A function or a method of an entry module is itself the user's entry point; a closure one of them
            # defines is reached by its own callers.
            return True
        key = (rel, id(function))
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = False
        references = self.census.callers(rel, function)
        ok = bool(references) and all(
            is_call and owner is not None and self.position(r_rel, r_node)
            for r_rel, r_node, owner, _q, is_call in references)
        self.memo[key] = ok
        return ok


def _exemption_failures(census, rel, store, kind, records):
    """Why each site of an exempt pair fails its predicate; empty when every one holds."""
    spec = STORES[store]
    out = []
    if kind == "route":
        routes = _Routes(census)
        for site, node, owner in records:
            if owner is None or not routes.position(rel, node):
                where = f"in {site.function}()" if site.function else "at module level"
                out.append(f"{rel}:{site.line}: a {store} write {where} that no route handler alone reaches")
    elif kind == "unread":
        for read in spec.reads:
            for r_rel, r_node, _o, qual, _c in census.references(read):
                if r_rel != spec.house:
                    line = r_node.lineno if r_node is not None else "?"
                    out.append(f"{r_rel}:{line}: reads {store} back ({read}), so its writes are read")
        if not spec.reads:
            out.append(f"{rel}: {store} names no method that reads it back; 'unread' cannot be checked")
    elif kind == "keyed":
        index = census.package.index(rel)
        for site, node, owner in records:
            call = index["parents"].get(id(node))
            keyed = isinstance(call, ast.Call) and call.func is node and any(
                kw.arg in spec.keys and _live_key(census, rel, call, kw.value) for kw in call.keywords)
            if not keyed:
                out.append(f"{rel}:{site.line}: a {store} write that binds no context key "
                           f"({', '.join(spec.keys) or 'none named'})")
    elif kind == "quiet":
        for site, node, owner in records:
            if site.method not in spec.quiet:
                out.append(f"{rel}:{site.line}: {site.method} adds content to {store}")
    elif kind == "script":
        parents = census.package.index(rel)["parents"]
        # A program module another module imports runs, on that import,
        # everything outside its __main__ test: it is a program no more.
        package = census.package
        program = rel.endswith("/__main__.py") and not any(
            rel in census.reached(other)
            for other in sorted(package.spelling({"__main__", "run_module", "run_path"}) - {rel}))
        for site, node, owner in records:
            if _main_guarded(node, parents):
                continue
            # A program module runs as a program: its writes are its own, as
            # long as nothing outside it calls the function they sit in.
            if program and (owner is None or all(
                    r_rel == rel
                    for function in _Gates(census, ())._enclosing(rel, node)
                    for r_rel, *_rest in census.callers(rel, function))):
                # Every function around the write is reached only from inside
                # the program (a command handed to its own parser included):
                # none is called, or handed, from another module.
                continue
            out.append(f"{rel}:{site.line}: a {store} write outside the module's __main__ block")
    elif kind == "instance":
        own = _own_objects(census, rel, store)
        module = census.modules.get(rel)
        for site, node, owner in records:
            receiver = node.value if isinstance(node, ast.Attribute) and module is not None \
                and not module.owners(node.value) else None
            if isinstance(node, ast.Call):
                receiver = node.args[0] if node.args else None
            if receiver is None or not _own_receiver(census, rel, store, receiver, own):
                out.append(f"{rel}:{site.line}: a {store} write on an object the module did not build itself "
                           f"at a temporary place")
    return out


def _live_key(census, rel, call, value):
    """True when a context key is computed by the function that writes, on every path that reaches the write.

    A name bound once to a false constant -- an empty key, None -- and
    computed elsewhere is live when the write sits under a test that the name
    is true, and nothing binds it between that test and the write.
    """
    index = census.package.index(rel)
    constant_name = _constant_names(census, rel, call)
    if not _constant(value, constant_name, index):
        return True
    if not isinstance(value, ast.Name):
        return False
    functions = _Gates(census, ())._enclosing(rel, call)
    function = next((f for f in functions if _bindings_in(f, value.id)), None)
    if function is None:
        return False
    local = _bindings_in(function, value.id)
    sentinels = [b for b in local if b[0] in ("assign", "walrus") and isinstance(b[1], ast.Constant) and not b[1].value]
    rest = [b for b in local if not any(b is s for s in sentinels)]
    if not sentinels or not rest or not _tested_true(census, rel, call, value.id):
        return False
    # The sentinels never reach the write: the name is live when no other
    # binding of it is a constant.
    for kind, bound in rest:
        if kind in ("declaration", "import"):
            return False
        if kind == "parameter" and bound is not None and _constant(bound, constant_name, index):
            return False
        if kind in ("assign", "walrus") and _constant(bound, constant_name, index):
            return False
        if kind == "loop" and _constant(bound.iter, constant_name, index):
            return False
        # An augmentation by a constant may carry the sentinel on: ``'' + '-v2'``.
        if kind == "augmented" and _constant(bound, constant_name, index):
            return False
        if kind == "with" and _constant(bound.context_expr, constant_name, index):
            return False
    return True


def _tested_true(census, rel, node, name):
    """True when ``node`` runs only once a test found ``name`` true, every binding of the name in the function
    stands before that test and in no loop around it, and no nested scope declares it ``nonlocal`` or ``global``."""
    gates = _Gates(census, ())
    index = census.package.index(rel)
    parents = index["parents"]
    function = gates._home(rel, node)
    guard = None
    for container, field, block, position in gates._frames(rel, node):
        if isinstance(container, ast.If) and (
                (field == "body" and gates._positive(rel, container.test, {name}))
                or (field == "orelse" and gates._negative(rel, container.test, {name}))):
            guard = container
        else:
            guard = next((stmt for stmt in block[:position] if isinstance(stmt, ast.If)
                          and gates._negative(rel, stmt.test, {name}) and _leaves(stmt.body)), None)
        if guard is not None:
            break
    if function is None or guard is None:
        return False
    loops, current = set(), guard
    while current is not None and current is not function:
        current = parents.get(id(current))
        if isinstance(current, (ast.For, ast.AsyncFor, ast.While)):
            loops.add(id(current))
    for sub in ast.walk(function):
        if isinstance(sub, (ast.Global, ast.Nonlocal)) and name in sub.names:
            return False
        if isinstance(sub, ast.Name) and sub.id == name and not isinstance(sub.ctx, ast.Load):
            if sub.lineno >= guard.lineno:
                return False
            current = sub
            while current is not None and current is not function:
                current = parents.get(id(current))
                if id(current) in loops:
                    return False
    return True


# Calls whose value is a constant when everything they are handed is.
_PURE_BUILTINS = frozenset({"str", "bytes", "int", "float", "bool", "repr", "len", "tuple", "frozenset", "sorted",
                            "abs", "round", "min", "max", "sum", "hash", "range", "enumerate", "reversed", "zip",
                            "iter", "next", "list", "dict", "set", "chr", "ord", "divmod", "pow", "format"})
_PURE_METHODS = frozenset({"hexdigest", "digest", "upper", "lower", "strip", "lstrip", "rstrip", "title", "casefold",
                           "encode", "decode", "format", "join", "replace", "zfill", "split", "rsplit", "splitlines",
                           "partition", "rpartition", "removeprefix", "removesuffix", "capitalize", "swapcase",
                           "center", "ljust", "rjust", "get", "keys", "values", "items", "copy", "index", "count"})
# Functions of the standard library whose value is a constant when everything they are handed is.
_PURE_DOTTED = frozenset({"contextlib.nullcontext", "json.dumps", "copy.copy", "copy.deepcopy", "itertools.chain",
                          "base64.b64encode", "base64.urlsafe_b64encode", "binascii.hexlify", "zlib.crc32",
                          "uuid.uuid3", "uuid.uuid5", "operator.add", "functools.reduce"})


def _constant(node, constant_name, index):
    """True for an expression that holds the same value on every call: a literal, a container, an f-string, an
    arithmetic or a comparison of constants, a constant subscript, a digest, a conversion or a string method of
    constants, or a name -- or an attribute of a name -- that ``constant_name`` says holds one."""
    def c(n):
        return _constant(n, constant_name, index)
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return constant_name(node.id)
    if isinstance(node, ast.Attribute):
        root = node.value
        while isinstance(root, ast.Attribute):
            root = root.value
        return isinstance(root, ast.Name) and constant_name(root.id)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(c(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and c(k) for k in node.keys) and all(c(v) for v in node.values)
    if isinstance(node, ast.JoinedStr):
        return all(c(v.value if isinstance(v, ast.FormattedValue) else v) for v in node.values)
    if isinstance(node, ast.UnaryOp):
        return c(node.operand)
    if isinstance(node, ast.BinOp):
        return c(node.left) and c(node.right)
    # A key that is a constant on one path is no live key: ``fp or NOCTX``.
    if isinstance(node, ast.BoolOp):
        return any(c(v) for v in node.values)
    if isinstance(node, ast.Compare):
        return c(node.left) and all(c(x) for x in node.comparators)
    if isinstance(node, ast.IfExp):
        return c(node.body) or c(node.orelse)
    if isinstance(node, ast.NamedExpr):
        return c(node.value)
    if isinstance(node, ast.Subscript):
        # Any element of a constant container is a constant.
        return c(node.value)
    if isinstance(node, ast.Slice):
        return all(n is None or c(n) for n in (node.lower, node.upper, node.step))
    if isinstance(node, ast.Starred):
        return c(node.value)
    if isinstance(node, ast.Call):
        if not (all(c(a) for a in node.args) and all(c(k.value) for k in node.keywords)):
            return False
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _PURE_METHODS and c(func.value):
            return True
        # Read the way that refuses more: a star import is no reason to doubt
        # that ``hashlib`` or ``str`` are what they look like.
        dotted = _dotted(index, func, may=True)
        if dotted is not None and (dotted.split(".")[0] == "hashlib" or dotted in _PURE_DOTTED):
            return True
        # A lambda handed constants hands back what its body makes of them and
        # of the names around it; a function the module defines, handed
        # constants, depends on nothing the function that writes was handed.
        if isinstance(func, ast.Lambda):
            return c(func.body)
        bound = _name_bindings(index).get(func.id, ()) if isinstance(func, ast.Name) else ()
        if any(kind == "definition" and isinstance(value, _FUNCTIONS) for kind, value in bound):
            return True
        # A name bound to a lambda is that lambda.
        if any(kind == "assign" and isinstance(value, ast.Lambda) and c(value.body) for kind, value in bound):
            return True
        return isinstance(func, ast.Name) and func.id in _PURE_BUILTINS and not any(
            kind != "star" for kind, _value in _name_bindings(index).get(func.id, ()))
    return False


def _constant_names(census, rel, node):
    """name -> True when a name read at ``node`` holds the same value on every call of the function around it.

    A key is live only when the function that writes computes it from what it
    is handed: a name bound at module level -- by assignment, import or
    anything else -- is the same on every call; a name the function binds is
    a constant as soon as one of its bindings is (a constant value, a
    parameter whose default is constant, a loop over constants), and a name it
    declares ``global`` or ``nonlocal``, or imports, is one too.
    """
    index = census.package.index(rel)
    functions = _Gates(census, ())._enclosing(rel, node)
    seen = set()
    # Names a scope nested in each function declares ``nonlocal`` or
    # ``global``: it may bind them anew, behind the function's back.
    declared = {id(f): {name for n in ast.walk(f) if isinstance(n, (ast.Nonlocal, ast.Global)) for name in n.names}
                for f in functions}

    def constant_name(name):
        if name in seen:
            return False
        for function in functions:
            local = _bindings_in(function, name)
            if not local:
                continue
            if name in declared[id(function)] or any(kind in ("declaration", "import") for kind, _value in local):
                return True
            seen.add(name)
            try:
                for kind, value in local:
                    if kind == "parameter" and value is not None and _constant(value, constant_name, index):
                        return True
                    # ``key += "-v2"`` keeps a live key live: an augmented
                    # assignment brings no constant of its own.
                    if kind in ("assign", "walrus") and _constant(value, constant_name, index):
                        return True
                    if kind == "loop" and _constant(value.iter, constant_name, index):
                        return True
                    if kind == "with" and _constant(value.context_expr, constant_name, index):
                        return True
                return False
            finally:
                seen.discard(name)
        return True

    return constant_name


def _bindings_in(function, name):
    """(kind, value) for every binding of ``name`` in a function's own scope: a parameter with its default (None
    when it has none), an assignment with its value, and the binding node for the other forms. A nested scope's
    body is its own; its header runs here."""
    out = []
    args = function.args
    positional, keyword_only = _parameters(args)
    defaults = dict(zip([p.arg for p in positional][len(positional) - len(args.defaults):], args.defaults))
    defaults.update({p.arg: d for p, d in zip(keyword_only, args.kw_defaults)})
    for param in positional + keyword_only + [p for p in (args.vararg, args.kwarg) if p is not None]:
        if param.arg == name:
            out.append(("parameter", defaults.get(param.arg)))
    stack = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (*_FUNCTIONS, ast.Lambda, ast.ClassDef)):
            if not isinstance(node, ast.Lambda) and node.name == name:
                out.append(("definition", node))
            stack.extend(_header(node))
            continue
        if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            out.append(("declaration", node))
        elif isinstance(node, ast.Assign) and any(name in _target_names(t)[0] for t in node.targets):
            out.append(("assign", node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None and name in _target_names(node.target)[0]:
            out.append(("assign", node.value))
        elif isinstance(node, ast.AugAssign) and name in _target_names(node.target)[0]:
            out.append(("augmented", node.value))
        elif isinstance(node, ast.NamedExpr) and name in _target_names(node.target)[0]:
            out.append(("walrus", node.value))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and name in _target_names(node.target)[0]:
            out.append(("loop", node))
        elif isinstance(node, ast.withitem) and node.optional_vars is not None \
                and name in _target_names(node.optional_vars)[0]:
            out.append(("with", node))
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            out.append(("handler", node))
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (a.asname or a.name.split(".")[0]) == name for a in node.names):
            out.append(("import", node))
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == name:
            out.append(("pattern", node))
        stack.extend(ast.iter_child_nodes(node))
    return out


# What makes a place of the module's own: a fresh directory or file of its
# own. ``gettempdir()`` is the shared temporary root, a predictable place
# anyone may have filled first.
_TEMPORARY = frozenset({"tempfile.mkdtemp", "tempfile.mkstemp"})
# A place only as the target of its ``with``: assigned, it is an object.
_TEMPORARY_CONTEXTS = frozenset({"tempfile.TemporaryDirectory"})
_PATH_TYPES = frozenset({"pathlib.Path", "pathlib.PurePath", "pathlib.PosixPath", "pathlib.PurePosixPath"})
_JOINS = frozenset({"os.path.join", "posixpath.join"})


def _temporary_names(census, rel):
    """Names a module binds only ever to a temporary place.

    Names are not scoped: a name bound to ``mkdtemp()`` in one function and to
    the user's path in another is not temporary, and neither is a name a
    parameter, an import, a loop, a walrus, an augmented assignment or a
    pattern binds anywhere in the module.
    """
    index = census.package.index(rel)
    bindings = _name_bindings(index)
    temporary, grown = set(), True
    while grown:
        grown = False
        for name, bound in bindings.items():
            if name not in temporary and all(
                    kind in ("assign", "with") and _temporary_place(index, value, temporary) or (
                        kind == "with" and isinstance(value, ast.Call)
                        and _dotted(index, value.func) in _TEMPORARY_CONTEXTS) for kind, value in bound):
                temporary.add(name)
                grown = True
    return temporary


def _temporary_place(index, value, temporary):
    """True when an expression is a temporary place, by a closed grammar: a ``tempfile`` call, a name bound only
    to a place, ``Path`` or ``str`` of a place, or a place joined to constant relative names by ``/``, ``Path``
    or ``os.path.join``. An expression that merely holds a temporary path somewhere is no place: joined to
    ``name``, any place becomes ``name`` when ``name`` is absolute."""
    if isinstance(value, ast.Name):
        return value.id in temporary
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div):
        return _temporary_place(index, value.left, temporary) and _relative_name(value.right)
    if not isinstance(value, ast.Call):
        return False
    dotted = _dotted(index, value.func)
    if dotted in _TEMPORARY:
        return True
    if value.keywords or not value.args:
        return False
    if dotted in _PATH_TYPES or dotted in _JOINS or (
            len(value.args) == 1 and isinstance(value.func, ast.Name) and value.func.id == "str"
            and "str" not in _name_bindings(index)):
        return _temporary_place(index, value.args[0], temporary) and all(_relative_name(a) for a in value.args[1:])
    return False


def _relative_name(node):
    """True for a constant relative name that stays inside the place it is joined to."""
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value):
        return False
    text = node.value
    return not text.startswith(("/", "\\", "~")) and ":" not in text and chr(0) not in text \
        and ".." not in re.split(r"[/\\]+", text)


def _own_class(census, rel, store, func):
    """True when a call's function is one of the store's own classes, as the module names it: bound once, by
    its definition in the house or an import that leads there, or an attribute of the house alone. A subclass
    may do anything with the place it is handed."""
    spec = STORES[store]
    module = census.modules.get(rel)
    if module is None:
        return False
    if isinstance(func, ast.Name) and func.id in spec.classes:
        bound = _name_bindings(census.package.index(rel)).get(func.id, ())
        if len(bound) != 1:
            return False
        if rel == spec.house:
            return bound[0][0] == "definition"
        origins = module.origins.get(func.id)
        return bool(origins) and all(census.package.definitions(o) == {(spec.house, func.id)} for o in origins)
    if isinstance(func, ast.Attribute) and func.attr in spec.classes:
        return census._value_modules(func.value, census._module_aliases(rel)) == {spec.house}
    return False


def _placed(census, rel, store, call, temporary):
    """True when a call builds one of the store's own classes at a temporary place, named by its keyword.

    A place passed by position, or through ``*args`` or ``**kwargs``, is
    refused: which parameter receives it is the class's affair.
    """
    spec = STORES[store]
    if not spec.place or not _own_class(census, rel, store, call.func):
        return False
    place = next((kw.value for kw in call.keywords if kw.arg == spec.place), None)
    return place is not None and _temporary_place(census.package.index(rel), place, temporary)


def _own_objects(census, rel, store):
    """Names the module binds to store objects, every binding of the name, anywhere in the module, an assignment
    of a store built at a temporary place, and no binding of any other kind."""
    module = census.modules.get(rel)
    if module is None or module.tree is None:
        return set()
    temporary = _temporary_names(census, rel)
    index = census.package.index(rel)
    # An object whose attributes are set, deleted or rebuilt after it was
    # built may be placed anywhere: ``j._db_path = user_path`` moves it.
    mutated = {n.value.id for n in index["nodes"] if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
               and (not isinstance(n.ctx, ast.Load) or n.attr in _REBUILDING)}
    mutated |= {n.args[0].id for n in index["calls"] if isinstance(n.func, ast.Name)
                and n.func.id in ("setattr", "delattr") and n.args and isinstance(n.args[0], ast.Name)}
    # ``object.__setattr__(j, ...)`` and ``Jot.__init__(j, ...)`` rebuild ``j``.
    mutated |= {n.args[0].id for n in index["calls"] if isinstance(n.func, ast.Attribute)
                and n.func.attr in _REBUILDING and n.args and isinstance(n.args[0], ast.Name)}
    out = set()
    for name, bound in _name_bindings(index).items():
        if name in mutated:
            continue
        # Names are not scoped: a name bound in one function to a cache built
        # in a temporary place, and in another to a second store built the
        # same way, holds own objects only.
        if all(kind in ("assign", "walrus") and isinstance(value, ast.Call) and any(
                _placed(census, rel, other, value, temporary) for other in STORES) for kind, value in bound):
            out.add(name)
    return out


def _own_receiver(census, rel, store, receiver, own):
    """True when a write's receiver is an object of the store the module built itself, at a temporary place."""
    if isinstance(receiver, ast.Name):
        return receiver.id in own
    if isinstance(receiver, ast.Call):
        return _placed(census, rel, store, receiver, _temporary_names(census, rel))
    return False


# ---------------------------------------------------------------------------
# The questions.
# ---------------------------------------------------------------------------
def _owed_pairs():
    return {(rel, store) for rel, entry in LEDGER.items() for store in entry.get("owes", {})}


def find_unclassified(census):
    """A site nobody houses, gates, exempts or owes."""
    owed = _owed_pairs()
    out = []
    for (rel, store), records in sorted(_pairs(census).items()):
        if (rel, store) in GATED or (rel, store) in EXEMPT or (rel, store) in owed:
            continue
        first = min(site.line for site, _n, _o in records)
        out.append(f"{rel}: {len(records)} write site(s) of {store} nobody houses, gates, exempts or owes; "
                   f"first at line {first}")
    return out


def find_conflicts(census):
    """A pair classified in more than one table."""
    owed = _owed_pairs()
    out = []
    for pair in sorted(set(GATED) | set(EXEMPT) | owed):
        tables = [name for name, keys in (("GATED", GATED), ("EXEMPT", EXEMPT), ("LEDGER", owed)) if pair in keys]
        if len(tables) > 1:
            out.append(f"{pair[0]}: {pair[1]} is classified in {' and '.join(tables)}")
    return out


def find_ungated(census):
    """A site of a gated pair that no rule gates."""
    pairs = _pairs(census)
    out = []
    for (rel, store), gates in sorted(GATED.items()):
        checker = _Gates(census, gates)
        for site, node, owner in pairs.get((rel, store), []):
            if not checker.site(rel, site, node, owner):
                where = f"in {site.function}()" if site.function else "at module level"
                out.append(f"{rel}:{site.line}: a {store} write ({site.method}) {where} that none of its gates "
                           f"({', '.join(gates)}) covers")
        out.extend(_authority_failures(census, rel, store, gates, pairs.get((rel, store), [])))
    return out


def _find_function(census, rel, dotted):
    """The function a module defines under a dotted name (``Class.method`` or ``function``), or None."""
    if rel not in census.estate.modules or census.package.trees[rel] is None:
        return None
    index = census.package.index(rel)
    for node in index["nodes"]:
        if isinstance(node, _FUNCTIONS) and ".".join(_owner(index, node)[1] + (node.name,)) == dotted:
            return node
    return None


def _authority_failures(census, rel, store, gates, records):
    """Sites whose gate takes its authority from a caller that no route or named entry alone reaches.

    An approval is a parameter, the Core's actor is an argument, an acceptance
    is a call: each is only the user's when the user's own gesture is what
    reaches it -- a route handler, or a function of an entry module the gate
    names (the terminal client).
    """
    out = set()
    # A site another gate of the pair covers on its own needs no authority.
    plain = _Gates(census, [g for g in gates if g in GATES and GATES[g].kind not in _AUTHORITY])
    for name in gates:
        gate = GATES.get(name)
        if gate is None or gate.kind not in _AUTHORITY:
            continue
        reach = _Routes(census, gate.entries)
        if gate.kind == "acceptance":
            function = _find_function(census, gate.home, gate.name)
            if function is None or not (_route_handler(census, gate.home, function)
                                        or reach.function(gate.home, function)):
                out.add(f"{gate.home}: {gate.name} carries the user's acceptance, and something other than a route "
                        f"or a named entry reaches it")
            continue
        for site, node, owner in records:
            if plain.gates and plain.site(rel, site, node, owner):
                continue
            # The write itself, through whatever helper, must be reached only
            # from the user's gesture: the approval or the actor it stands
            # behind is only the user's there.
            if owner is None or not reach.position(rel, node):
                out.add(f"{rel}:{site.line}: a {store} write whose {gate.kind} gate ({name}) takes its authority "
                        f"from a caller that no route or named entry alone reaches")
    return sorted(out)


def find_failed_exemptions(census):
    """A site its exemption's predicate refuses, or a site more than a reason in prose argues for.

    A reason in prose is checked by no predicate, so it is counted: it covers
    the sites it was written for, and a new one is a finding.
    """
    pairs = _pairs(census)
    out = []
    for (rel, store), entry in sorted(EXEMPT.items()):
        kind = entry[0]
        if kind in _CHECKED and (rel, store) in pairs:
            # A key predicate may meet a site its reason argues apart, by
            # function and method: such a site, and only one the predicate
            # refuses, is the reason's to carry.
            refused, _absorbed = _keyed_apart(census, rel, store, kind, pairs[(rel, store)])
            out.extend(refused)
        elif kind in _ARGUED:
            argued = ARGUED_SITES.get((rel, store))
            if not argued:
                out.append(f"{rel}: {store} is exempt by a reason alone and names none of the sites it covers")
                continue
            found = Counter((s.function or "", s.method) for s, _n, _o in pairs.get((rel, store), ()))
            for function, method in sorted((found - Counter(argued)).elements()):
                where = f"in {function}()" if function else "at module level"
                out.append(f"{rel}: a {store} write ({method}) {where} that its reason was not written for")
    return out


def _keyed_apart(census, rel, store, kind, records):
    """(refusals, absorbed): what a checked exemption's predicate refuses, once the sites a ``keyed`` pair's
    reason argues apart have absorbed a refusal each -- an argued site the predicate does not refuse absorbs
    nothing -- and the number absorbed."""
    argued = ARGUED_SITES.get((rel, store), ()) if kind == "keyed" else ()
    if not argued:
        return _exemption_failures(census, rel, store, kind, records), 0
    left, refusals, absorbed = Counter(argued), [], 0
    for record in records:
        failed = _exemption_failures(census, rel, store, kind, [record])
        key = (record[0].function or "", record[0].method)
        if failed and left[key] > 0:
            left[key] -= 1
            absorbed += 1
            continue
        refusals.extend(failed)
    return refusals, absorbed


def find_ledger_growth(census):
    """An owed pair with more sites than it owes."""
    pairs = _pairs(census)
    out = []
    for rel, entry in sorted(LEDGER.items()):
        for store, count in sorted(entry.get("owes", {}).items()):
            found = len(pairs.get((rel, store), []))
            if found > count:
                out.append(f"{rel}: {found} write site(s) of {store}, above the {count} it owes")
    return out


def find_broken_seals(census):
    """An owed module whose bytes moved while it still owes."""
    pairs = _pairs(census)
    out = []
    for rel, entry in sorted(LEDGER.items()):
        text = census.estate.modules.get(rel)
        owing = any(pairs.get((rel, store)) for store in entry.get("owes", {}))
        if text is not None and owing and digest(text) != entry.get("seal"):
            out.append(f"{rel}: owes, and its bytes moved since its debt was sealed: touch it, and you pay it")
    return out


def find_stale_entries(census):
    """An entry, a gate or an owed count the census no longer finds."""
    pairs = _pairs(census)
    out = []
    for table, name in ((GATED, "gated"), (EXEMPT, "exempt")):
        for rel, store in sorted(table):
            if not pairs.get((rel, store)):
                state = "it is gone" if rel not in census.estate.modules else f"it has no {store} write site"
                out.append(f"{rel}: {name} for {store}, but {state}; take the entry off")
    for (rel, store), argued in sorted(ARGUED_SITES.items()):
        entry = EXEMPT.get((rel, store))
        if entry is None or entry[0] not in _ARGUED | {"keyed"}:
            out.append(f"{rel}: sites argued for {store}, which no reason in prose exempts; take them off")
            continue
        found = Counter((s.function or "", s.method) for s, _n, _o in pairs.get((rel, store), ()))
        if not found:
            continue
        if entry[0] == "keyed":
            _refused, absorbed = _keyed_apart(census, rel, store, "keyed", pairs[(rel, store)])
            if absorbed < len(argued):
                out.append(f"{rel}: its reason argues apart {len(argued)} {store} write(s) its key predicate would "
                           f"refuse, and the predicate refuses {absorbed} of them; take the others off")
            continue
        for function, method in sorted((Counter(argued) - found).elements()):
            where = f"in {function}()" if function else "at module level"
            out.append(f"{rel}: its reason argues for a {store} write ({method}) {where} the census no longer "
                       f"finds; take it off")
    for rel, entry in sorted(LEDGER.items()):
        owes = entry.get("owes", {})
        if not any(pairs.get((rel, store)) for store in owes):
            state = "it is gone" if rel not in census.estate.modules else "it has no owed site"
            out.append(f"{rel}: owed, but {state}; take it off the ledger")
            continue
        for store, count in sorted(owes.items()):
            found = len(pairs.get((rel, store), []))
            if found < count:
                out.append(f"{rel}: owes {count} write site(s) of {store}, the census finds {found}; lower it")
    named = {gate for gates in GATED.values() for gate in gates}
    for gate in sorted(set(GATES) - named):
        out.append(f"gate {gate}: no gated pair names it; take it off")
    for rel in sorted(OUTSIDE):
        if rel not in census.estate.modules:
            out.append(f"{rel}: outside the census, but it is gone; take it off")
        elif rel not in _store_openers(census):
            out.append(f"{rel}: outside the census, but it opens no database; take it off")
    return out


def _defines(census, rel, name, kind):
    """True when module ``rel`` defines what a gate of ``kind`` is called by."""
    if rel not in census.estate.modules or census.package.trees[rel] is None:
        return False
    nodes = census.package.index(rel)["nodes"]
    if kind in ("decision", "acceptance"):
        for node in nodes:
            if isinstance(node, _FUNCTIONS):
                owner, qual = _owner(census.package.index(rel), node)
                if ".".join(qual + (node.name,)) == name:
                    return True
        return False
    if kind == "approval":
        return any(isinstance(n, _FUNCTIONS) and any(p.arg == name for p in sum(_parameters(n.args), []))
                   for n in nodes)
    return any(isinstance(n, _FUNCTIONS) and n.name == name for n in nodes)


def find_unknown_gates(census):
    """A gate or an exemption kind no table defines, or a gate its home does not define."""
    out = []
    for (rel, store), gates in sorted(GATED.items()):
        for gate in gates:
            if gate not in GATES:
                out.append(f"{rel}: {store} names the gate {gate}, which GATES does not define")
    for name, gate in sorted(GATES.items()):
        if gate.kind not in _GATE_KINDS:
            out.append(f"gate {name}: kind {gate.kind!r} is none of {', '.join(sorted(_GATE_KINDS))}")
        elif not _defines(census, gate.home, gate.name, gate.kind):
            out.append(f"gate {name}: {gate.home} does not define {gate.name}")
    for (rel, store), entry in sorted(EXEMPT.items()):
        kind, reason = entry[0], entry[1]
        if kind not in _CHECKED | _ARGUED:
            out.append(f"{rel}: {store} is exempt as {kind!r}, a kind no predicate checks and no reason argues")
        if not str(reason).strip():
            out.append(f"{rel}: {store} is exempt with no reason")
    for rel, reason in sorted(OUTSIDE.items()):
        if not str(reason).strip():
            out.append(f"{rel}: outside the census with no reason")
    return out


class SuitesUnreadable(Exception):
    """A test suite the census reads for its gate proofs could not be read."""


def _suites(root, contracts):
    """suite -> (tree, its test functions) for every suite that may hold one of ``contracts``.

    A suite is parsed only when its text spells ``def test_<id>`` for one of
    them; a test function is one its syntax tree defines, never a line a
    string happens to hold.
    """
    tests = Path(root) / "tests"
    out, unread = {}, []
    if not tests.is_dir() or not contracts:
        return out
    spelled = re.compile(r"\bdef\s+test_(?:" + "|".join(sorted(map(re.escape, contracts))) + r")(?:_|\()")
    for path in sorted(tests.glob("test_*.py")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            unread.append(f"{rel}: cannot be read as UTF-8 text ({type(exc).__name__})")
            continue
        if not spelled.search(text):
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError) as exc:
            unread.append(f"{rel}: does not parse ({type(exc).__name__})")
            continue
        out[rel] = (tree, _collected_tests(tree))
    if unread:
        raise SuitesUnreadable("; ".join(unread))
    return out


def _collected_tests(tree):
    """The tests that surely run, as ``name`` or ``Class::name``, by a closed grammar: a function ``test_*``,
    undecorated and bound once, directly in the module's body; or an undecorated method ``test_*``, bound once,
    of an undecorated class ``Test*`` bound once directly in the module's body, with no base, no ``__init__`` or
    ``__new__``, and no binding of ``__test__`` or ``pytestmark``. A suite that binds ``pytestmark`` or
    ``__test__``, names a silent skip (``skip``, ``skipif``, ``xfail``, ``importorskip``, ``SkipTest``), defines
    ``pytest_generate_tests``, runs one of its own functions or raises at import, or rebinds a test's name vouches
    for no test; an ``async`` test needs a plugin to run, and a decorator may be a skip under another name."""
    # The module's statements as pytest imports it: what runs only as a
    # program, under the ``__main__`` test, does not run there.
    top, stack = [], list(tree.body)
    while stack:
        stmt = stack.pop()
        top.append(stmt)
        if isinstance(stmt, ast.If):
            stack.extend(stmt.orelse if _is_main_test(stmt.test) else stmt.body + stmt.orelse)
        elif isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
            stack.extend(stmt.body + stmt.orelse + stmt.finalbody)
            for handler in stmt.handlers:
                stack.extend(handler.body)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            stack.extend(stmt.body)
    marked = {"pytestmark", "__test__"}
    # A suite that names a way to skip anywhere -- a mark, a call, an alias,
    # an exception -- vouches for no test: it may skip any of them.
    for node in ast.walk(tree):
        names = {node.id} if isinstance(node, ast.Name) else {node.attr} if isinstance(node, ast.Attribute) else {
            a.asname or a.name for a in node.names} | {a.name for a in node.names} if isinstance(
                node, (ast.Import, ast.ImportFrom)) else set()
        if names & _SKIPPING:
            return set()
    # A suite that may exit -- ``pytest.exit``, ``sys.exit``, ``os._exit``,
    # ``exit()`` -- outside its ``__main__`` test may end the session green,
    # with its tests unrun.
    main_only = {id(n) for stmt in tree.body if isinstance(stmt, ast.If) and _is_main_test(stmt.test)
                 for s in stmt.body for n in ast.walk(s)}
    for node in ast.walk(tree):
        if id(node) in main_only:
            continue
        names = {node.id} if isinstance(node, ast.Name) else {node.attr} if isinstance(node, ast.Attribute) else {
            a.name for a in node.names} if isinstance(node, (ast.Import, ast.ImportFrom)) else set()
        if names & _EXITS:
            return set()
    local = {s.name for s in top if isinstance(s, _FUNCTIONS)}
    if "pytest_generate_tests" in local:
        return set()
    bound = Counter()
    for stmt in top:
        if isinstance(stmt, ast.Raise):
            return set()
        value = getattr(stmt, "value", None)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in local:
            # A function of the suite run at import may raise anything.
            return set()
        names = _statement_names(stmt)
        if names & marked:
            return set()
        bound.update(names)
        # A walrus or a handler at module level binds a name too.
        for sub in ast.walk(stmt) if not isinstance(stmt, (*_FUNCTIONS, ast.ClassDef)) else ():
            if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
                bound[sub.target.id] += 1
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                bound[sub.name] += 1
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            bound.update(node.names)
    out = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_") and not stmt.decorator_list \
                and bound[stmt.name] == 1:
            out.add(stmt.name)
        elif isinstance(stmt, ast.ClassDef) and stmt.name.startswith("Test") and not stmt.decorator_list \
                and not stmt.bases and not stmt.keywords and bound[stmt.name] == 1:
            members = Counter()
            for member in stmt.body:
                members.update(_statement_names(member))
            if members.keys() & (marked | {"__init__", "__new__"}):
                continue
            out |= {f"{stmt.name}::{m.name}" for m in stmt.body if isinstance(m, ast.FunctionDef)
                    and m.name.startswith("test_") and not m.decorator_list and members[m.name] == 1}
    return out


def _statement_names(stmt):
    """The names one statement binds where it stands: a definition, assignment targets, imports, ``del``,
    loop and ``with`` targets."""
    if isinstance(stmt, (*_FUNCTIONS, ast.ClassDef)):
        return {stmt.name}
    targets = []
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
        targets = [stmt.target]
    elif isinstance(stmt, ast.Delete):
        targets = stmt.targets
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        targets = [item.optional_vars for item in stmt.items if item.optional_vars is not None]
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return {a.asname or a.name.split(".")[0] for a in stmt.names}
    return set().union(*(_target_names(t)[0] for t in targets))


# The names a suite skips by, in silence: marks, calls, the exception. A
# ``fail`` is loud, and a common word besides.
_SKIPPING = frozenset({"skip", "skipif", "xfail", "importorskip", "SkipTest"})
# The names a suite may end the session by, with a code of its choosing.
_EXITS = frozenset({"exit", "_exit", "quit"})

# What changes what pytest collects, besides ``addopts``.
_SELECTION_KEYS = frozenset({"python_files", "python_classes", "python_functions", "testpaths", "norecursedirs"})
_COLLECTION_HOOKS = frozenset({"pytest_collection", "pytest_collection_modifyitems", "pytest_ignore_collect",
                               "pytest_collect_file", "pytest_pycollect_makemodule", "pytest_pycollect_makeitem",
                               "pytest_generate_tests", "pytest_itemcollected", "pytest_deselected"})


def _deselected(root):
    """(removed, unread): what the selection rule removes -- ('deselect', a node id prefix) or ('ignore', a path),
    as pytest reads them from the root -- and what it holds that the census cannot read: an option other than
    ``--deselect`` and ``--ignore``, a word no option takes, a setting that changes what pytest collects, a
    ``conftest.py`` hook that does, or a skip a ``conftest.py`` calls."""
    out, unread = set(), []
    # pytest reads these before pyproject.toml: the census reads none of them.
    for name in ("pytest.ini", ".pytest.ini", "pytest.toml", ".pytest.toml"):
        if (Path(root) / name).is_file():
            unread.append(name)
    for name, section in (("tox.ini", "[pytest]"), ("setup.cfg", "[tool:pytest]")):
        try:
            if section in (Path(root) / name).read_text(encoding="utf-8"):
                unread.append(f"{name} {section}")
        except (OSError, UnicodeDecodeError):
            pass
    # A suite run by its path (``pytest tests/x.py``) finds a configuration in
    # its own folder first: any there is one the census does not read.
    for name in ("pytest.ini", ".pytest.ini", "pytest.toml", ".pytest.toml", "tox.ini", "setup.cfg", "pyproject.toml"):
        if (Path(root) / "tests" / name).is_file():
            unread.append(f"tests/{name}")
    try:
        import tomllib
        table = tomllib.loads((Path(root) / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]
        options = table.get("ini_options", {})
        unread.extend(sorted(f"[tool.pytest] {key}" for key in set(table) - {"ini_options"}))
    except (OSError, KeyError, ValueError, ImportError, AttributeError):
        options = {}
    unread.extend(sorted(set(options) & _SELECTION_KEYS))
    addopts = options.get("addopts", "")
    try:
        words = shlex.split(addopts) if isinstance(addopts, str) else list(addopts)
    except ValueError:
        words, unread = [], unread + ["addopts, which does not split as a shell would"]
    i = 0
    while i < len(words):
        flag, eq, value = words[i].partition("=")
        if flag in ("--deselect", "--ignore"):
            if not eq:
                i += 1
                value = words[i] if i < len(words) else ""
            # pytest reads an ignore as a path, a deselect as a node id prefix,
            # letter for letter.
            out.add(("deselect", value) if flag == "--deselect" else ("ignore", _normal(value)))
        elif words[i].startswith("-") and not eq and i + 1 < len(words) and not words[i + 1].startswith("-"):
            unread.append(f"{words[i]} {words[i + 1]}")
            i += 1
        else:
            unread.append(words[i])
        i += 1
    for conftest in (Path(root) / "conftest.py", Path(root) / "tests" / "conftest.py"):
        if not conftest.is_file():
            continue
        name = conftest.relative_to(root).as_posix()
        try:
            tree = ast.parse(conftest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
            unread.append(f"{name}, which cannot be read")
            continue
        for node in ast.walk(tree):
            if isinstance(node, _FUNCTIONS) and node.name in _COLLECTION_HOOKS:
                unread.append(f"{name}: {node.name}")
            elif isinstance(node, ast.Call) and _leaf(node.func) in ("skip", "importorskip", "xfail"):
                unread.append(f"{name}: {_leaf(node.func)}()")
            elif isinstance(node, ast.Raise) and node.exc is not None and "Skip" in ast.unparse(node.exc):
                unread.append(f"{name}: raise {ast.unparse(node.exc)[:40]}")
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    names, attributes = _target_names(target)
                    if names & {"collect_ignore", "collect_ignore_glob", "pytest_plugins"}:
                        unread.append(f"{name}: {', '.join(sorted(names))}")
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Attribute) \
                            and target.value.attr in ("option", "args"):
                        unread.append(f"{name}: {ast.unparse(target)[:40]}")
    return out, unread


def _normal(path):
    """A selection path as pytest reads it from the root: ``./`` and a trailing slash dropped, a node id kept."""
    head, sep, rest = path.partition("::")
    return (os.path.normpath(head) if head else head) + sep + rest


def _removed(suite, test, removed):
    """True when the selection rule removes a test: a ``--deselect`` that is a prefix of its node id (pytest
    matches prefixes: ``test_k1`` removes ``test_k10_x`` too), or an ``--ignore`` of its suite or a directory
    above it."""
    nodeid = f"{suite}::{test}"
    return any((kind == "deselect" and nodeid.startswith(entry))
               or (kind == "ignore" and (suite == entry or suite.startswith(entry + "/"))) for kind, entry in removed)


def _answers(function, contract):
    """True when a test function answers to a contract id: ``test_<id>_...`` or ``test_<id>``."""
    return function == f"test_{contract}" or function.startswith(f"test_{contract}_")


def find_gate_proofs_missing(census):
    """A gate whose contract names no test function that runs, or whose suite never names its home."""
    out = []
    wanted = {contract for gate in GATES.values() for contract in gate.contracts}
    suites = None
    deselected = None
    for name, gate in sorted(GATES.items()):
        if not gate.contracts:
            out.append(f"gate {name}: names no contract")
            continue
        if suites is None:
            suites = _suites(census.estate.root, wanted)
            deselected, unread = _deselected(census.estate.root)
            if unread:
                out.append(f"the selection rule holds what the census cannot read ({'; '.join(unread)}): no gate "
                           f"proof can be read under it")
        for contract in gate.contracts:
            owners = [suite for suite, (_tree, functions) in suites.items()
                      if any(_answers(f.rsplit("::", 1)[-1], contract) and not _removed(suite, f, deselected)
                             for f in functions)]
            if not owners:
                out.append(f"gate {name}: contract {contract} names no test function that runs")
            elif not any(_names_home(suites[s][0], gate.home) for s in owners):
                out.append(f"gate {name}: the suite of contract {contract} ({', '.join(owners)}) never names "
                           f"{gate.home}")
    return out


def _names_home(tree, home):
    """True when a suite names a module: an import of it, or a string that spells its dotted path, its file
    path, or its file name alone as a window does. A comment is not a name."""
    dotted = home[:-3].replace("/", ".")
    package, stem = dotted.rsplit(".", 1) if "." in dotted else ("", dotted)
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.ClassDef, *_FUNCTIONS)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    for node in ast.walk(tree):
        if id(node) in docstrings:
            continue
        if isinstance(node, ast.Import) and any(a.name == dotted or a.name.startswith(dotted + ".")
                                                for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and not node.level and (
                node.module == dotted or (node.module == package and any(a.name == stem for a in node.names))):
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and (
                dotted in node.value or home in node.value or node.value == stem):
            return True
    return False


def find_table_drift(census):
    """A store whose house does not define what the table names, or a house gate not called first."""
    out = []
    for name, spec in sorted(STORES.items()):
        if spec.house not in census.estate.modules:
            out.append(f"store {name}: its house {spec.house} is gone")
            continue
        tree = census.package.trees[spec.house]
        if tree is None:
            continue
        classes = {n.name: n for n in census.package.index(spec.house)["nodes"] if isinstance(n, ast.ClassDef)}
        functions = {n.name for n in _top_level(tree) if isinstance(n, _FUNCTIONS)}
        top = set()
        for stmt in _top_level(tree):
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                for target in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]):
                    top |= _target_names(target)[0]
        methods = set()
        for cls in spec.classes:
            if cls not in classes:
                out.append(f"store {name}: {spec.house} defines no class {cls}")
                continue
            methods |= {s.name for s in classes[cls].body if isinstance(s, _FUNCTIONS)}
        for access in spec.accesses + spec.functions:
            if access not in functions:
                out.append(f"store {name}: {spec.house} defines no function {access}")
        for instance in spec.instances:
            if instance not in top:
                out.append(f"store {name}: {spec.house} binds no {instance} at module level")
        for method in spec.writes + spec.reads:
            if spec.classes and method not in methods:
                out.append(f"store {name}: none of {', '.join(spec.classes)} defines {method}")
        for method in spec.quiet:
            if method not in spec.writes:
                out.append(f"store {name}: {method} is quiet but not a write")
        aside = dict(spec.aside or {})
        for method, reason in sorted(aside.items()):
            if not str(reason).strip():
                out.append(f"store {name}: {method} is set aside with no reason")
            elif method in spec.writes:
                out.append(f"store {name}: {method} is both a write and set aside")
        for cls in spec.classes:
            node = classes.get(cls)
            sql = _sql_constants(tree, node) if node is not None else set()
            shapes = _method_shapes(node, spec.writes, cls, sql) if node is not None else {}
            for stmt in node.body if node is not None else ():
                if not isinstance(stmt, _FUNCTIONS) or stmt.name.startswith("_") \
                        or stmt.name in spec.writes or stmt.name in aside:
                    continue
                shape = shapes.get(stmt.name)
                if shape:
                    out.append(f"store {name}: {cls}.{stmt.name} {shape}, and is neither a write nor set aside "
                               f"with a reason")
        for back in spec.backs:
            if back not in STORES:
                out.append(f"store {name}: it is built on {back}, which STORES does not name")
    for gname, gate in sorted(GATES.items()):
        if gate.kind != "house":
            continue
        stores = [n for n, s in STORES.items() if s.house == gate.home]
        for store in stores:
            spec = STORES[store]
            classes = {n.name: n for n in census.package.index(spec.house)["nodes"]
                       if isinstance(n, ast.ClassDef)} if census.package.trees.get(spec.house) else {}
            for cls in spec.classes:
                node = classes.get(cls)
                if node is None:
                    continue
                # Every binding of the class body, through its if, try, with
                # and for blocks: a second ``put`` would be the one that runs.
                bound = _class_bindings(node)
                if len(bound.get(gate.name, ())) > 1:
                    out.append(f"gate {gname}: {cls} binds {gate.name} more than once: only one can be the gate")
                # The house's own class, like any subclass, carries nothing
                # that may rebuild it or reroute its methods.
                if node.decorator_list or node.keywords:
                    out.append(f"gate {gname}: {cls} carries a class decorator or a metaclass, which may rebuild it")
                for hook in sorted(set(bound) & _CLASS_HOOKS):
                    out.append(f"gate {gname}: {cls} defines {hook}, which may reroute any of its methods")
                for method in spec.writes:
                    nodes = bound.get(method, [])
                    if len(nodes) > 1:
                        out.append(f"gate {gname}: {cls}.{method} is bound more than once in its class: only one "
                                   f"body is gated")
                    elif nodes and not (isinstance(nodes[0], _FUNCTIONS) and _calls_first(nodes[0], gate)):
                        out.append(f"gate {gname}: {cls}.{method} writes before it calls {gate.name}")
            # A subclass anywhere must leave the gate where it is: no other base,
            # no decorator or metaclass that may rebuild it, no hook that may
            # reroute its methods, no new binding of the gate's name; and every
            # write it binds calls the gate first, or hands straight to the
            # method it overrides.
            for rel, module in sorted(census.modules.items()):
                if module.tree is None:
                    continue
                for node in census.package.index(rel)["nodes"]:
                    if not isinstance(node, ast.ClassDef) or store not in module.classes.get(node.name, ()) or (
                            rel == spec.house and node.name in spec.classes):
                        continue
                    where = f"gate {gname}: {rel}: {node.name}"
                    if not all(store in module.symbol(base, "class") for base in node.bases):
                        out.append(f"{where} has a base besides {store}'s class, which may run before or around "
                                   f"its writes")
                    if node.decorator_list or node.keywords:
                        out.append(f"{where} carries a class decorator or a metaclass, which may rebuild it")
                    bound = _class_bindings(node)
                    if gate.name in bound:
                        out.append(f"{where} binds {gate.name}, the gate of {store}, anew")
                    for hook in sorted(set(bound) & _CLASS_HOOKS):
                        out.append(f"{where} defines {hook}, which may reroute any of its methods")
                    for method in sorted(set(bound) & set(spec.writes)):
                        nodes = bound[method]
                        if len(nodes) > 1 or not (isinstance(nodes[0], _FUNCTIONS) and (
                                _calls_first(nodes[0], gate) or _calls_super_first(nodes[0]))):
                            out.append(f"{where}.{method} overrides a write of {store} and writes before it "
                                       f"calls {gate.name}")
    return out


# Hooks through which a class may answer for any of its attributes.
_CLASS_HOOKS = frozenset({"__getattribute__", "__getattr__", "__setattr__", "__delattr__", "__init_subclass__",
                          "__new__", "__set_name__", "__class_getitem__"})


def _expression_names(stmt):
    """Names a statement binds in its own scope through its expressions, patterns and handlers: a walrus, a match
    capture, an exception handler's name. A nested function's, lambda's or class's body is its own scope; its
    header is read."""
    out, stack = set(), []
    for field, value in ast.iter_fields(stmt):
        if field in _BLOCKS:
            continue
        if field == "handlers":
            out |= {handler.name for handler in value if handler.name}
            continue
        if field == "cases":
            stack.extend(case.pattern for case in value)
            stack.extend(case.guard for case in value if case.guard is not None)
            continue
        stack.extend(value if isinstance(value, list) else [value])
    while stack:
        node = stack.pop()
        if not isinstance(node, ast.AST):
            continue
        if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            out.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            out.add(node.rest)
        if isinstance(node, ast.Lambda):
            stack.extend(list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None])
            continue
        if isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
            stack.extend(_header(node))
            continue
        stack.extend(ast.iter_child_nodes(node))
    return out


def _class_bindings(cls):
    """name -> [the statement that binds it] for a class body, through its if, try, with, for and match blocks,
    by any form -- a walrus or a match capture too; a method's body and a nested class's body are their own."""
    out, stack = {}, list(cls.body)
    while stack:
        stmt = stack.pop()
        for name in _statement_names(stmt) | _expression_names(stmt):
            out.setdefault(name, []).append(stmt)
        if isinstance(stmt, (*_FUNCTIONS, ast.ClassDef)):
            continue
        for field in _BLOCKS:
            stack.extend(getattr(stmt, field, ()) or ())
        for handler in getattr(stmt, "handlers", ()) or ():
            stack.extend(handler.body)
        for case in getattr(stmt, "cases", ()) or ():
            stack.extend(case.body)
    return out


def _calls_super_first(function):
    """True when the function's first statement after its docstring hands to the method it overrides, through a
    ``super()`` with no arguments: ``super(Base, self)`` may skip the store's own method."""
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    first = body[0] if body else None
    call = first.value if isinstance(first, (ast.Expr, ast.Return)) else None
    return isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == function.name \
        and isinstance(call.func.value, ast.Call) and isinstance(call.func.value.func, ast.Name) \
        and call.func.value.func.id == "super" and not call.func.value.args and not call.func.value.keywords


# An SQL write anywhere in a string: after a comment, inside a script or a common table expression too.
_SQL_WRITE = re.compile(r"\b(?:(?:INSERT|REPLACE)\s+(?:OR\s+\w+\s+)?INTO|UPDATE\s+\S+\s+SET|DELETE\s+FROM|UPSERT)\b",
                        re.I)


def _sql_constants(tree, cls):
    """Names a module or a class binds, in its body, to a value whose strings spell an SQL write."""
    out = set()
    for stmt in list(_top_level(tree)) + list(cls.body):
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None and any(
                isinstance(n, ast.Constant) and isinstance(n.value, str) and _SQL_WRITE.search(n.value)
                for n in ast.walk(stmt.value)):
            for target in stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]:
                out |= _target_names(target)[0]
    return out


def _write_shape(function, writes, cls_name="", sql=frozenset()):
    """How a store method's body writes, or None: an SQL write it spells -- in a string of its body that is no
    docstring, or in a string constant of its module or its class it names -- or a write of its class it reaches,
    called or not: through ``self``, ``cls``, the class's name, or a ``getattr`` of its name."""
    body = function.body
    docstring = body[0].value if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
        else None
    owners = {"self", "cls", cls_name}
    for node in ast.walk(function):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node is not docstring \
                and _SQL_WRITE.search(node.value):
            return "runs an SQL write"
        if (isinstance(node, ast.Name) and node.id in sql) or (
                isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in owners
                and node.attr in sql):
            return "runs an SQL write"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in owners \
                and node.attr in writes and isinstance(node.ctx, ast.Load):
            return f"calls the write {node.attr}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" \
                and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id in owners \
                and isinstance(node.args[1], ast.Constant) and node.args[1].value in writes:
            return f"calls the write {node.args[1].value}"
    return None


def _method_shapes(cls_node, writes, cls_name, sql):
    """method -> how it writes, for each method of a store's class: its own body's shape, or that of a private
    helper of its class it reaches through ``self``, ``cls`` or the class's name, helpers of helpers followed."""
    methods = {s.name: s for s in cls_node.body if isinstance(s, _FUNCTIONS)}
    shapes = {name: _write_shape(m, writes, cls_name, sql) for name, m in methods.items()}
    owners = {"self", "cls", cls_name}
    reached = {name: {n.attr for n in ast.walk(m) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                      and n.value.id in owners and n.attr in methods and n.attr != name
                      and n.attr.startswith("_") and not n.attr.startswith("__")}
               for name, m in methods.items()}
    grown = True
    while grown:
        grown = False
        for name in methods:
            if shapes[name] is None:
                helper = next((h for h in sorted(reached[name]) if shapes[h] is not None), None)
                if helper is not None:
                    shapes[name] = f"calls {helper}, which writes"
                    grown = True
    return shapes


def _calls_first(function, gate):
    """True when the function's first statement after its docstring calls the gate on ``self`` with a parameter.

    The gate reads what the caller passed: a constant would make the check
    say yes to everyone, and another object's gate is not this store's.
    """
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        return False
    first = body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)):
        return False
    call = first.value
    func = call.func
    on_self = isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
        and func.value.id in ("self", "cls") and func.attr == gate.name
    params = {p.arg for p in sum(_parameters(function.args), [])} - {"self", "cls"}
    return on_self and bool(call.args) and isinstance(call.args[0], ast.Name) and call.args[0].id in params


def _store_openers(census):
    """Modules that import a database library or a connection helper of the package."""
    cached = getattr(census, "_openers", None)
    if cached is not None:
        return cached
    out = set()
    # Every helper the table names opens what its callers name, whatever its stem.
    helpers = _HELPER_MODULES | {Path(rel).stem for rel in HELPERS}
    spelled = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, _DATABASE_LIBRARIES | helpers))) + r")\b")
    for rel, text in census.estate.modules.items():
        if not spelled.search(text):
            continue
        tree = census.package.trees[rel]
        if tree is None:
            continue
        for node in census.package.imports(rel):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if parts[0] in _DATABASE_LIBRARIES or (
                            parts[-1] in helpers and census.package.resolve(alias.name) is not None):
                        out.add(rel)
            else:
                module = node.module or ""
                if module.split(".")[0] in _DATABASE_LIBRARIES and not node.level:
                    out.add(rel)
                elif census.package.source_of(rel, node) is not None and (
                        module.split(".")[-1] in helpers or any(a.name in helpers for a in node.names)):
                    out.add(rel)
        # A library imported by its name at run time opens a database too.
        for node in census.package.index(rel)["calls"]:
            if _leaf(node.func) not in ("import_module", "__import__"):
                continue
            arg = node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg == "name"), None)
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                parts = arg.value.split(".")
                if parts[0] in _DATABASE_LIBRARIES or (parts[-1] in helpers and census.package.resolve(arg.value)):
                    out.add(rel)
    census._openers = out
    return out


def find_unclassified_stores(census):
    """A module that opens a database and is no store's house, no helper, and not outside by name."""
    houses = {s.house for s in STORES.values()}
    return [f"{rel}: opens a database, and is no store's house, no helper, and not named in OUTSIDE"
            for rel in sorted(_store_openers(census)) if rel not in houses and rel not in HELPERS
            and rel not in OUTSIDE]


_QUESTIONS = (
    ("unclassified", find_unclassified),
    ("conflicts", find_conflicts),
    ("ungated", find_ungated),
    ("failed exemptions", find_failed_exemptions),
    ("ledger growth", find_ledger_growth),
    ("broken seals", find_broken_seals),
    ("stale entries", find_stale_entries),
    ("unknown gates", find_unknown_gates),
    ("gate proofs missing", find_gate_proofs_missing),
    ("table drift", find_table_drift),
    ("unclassified stores", find_unclassified_stores),
)


def find_all(census):
    """Every question asked, in order: name -> its findings."""
    return {name: question(census) for name, question in _QUESTIONS}


# ---------------------------------------------------------------------------
# The estate, the green line, the entry point.
# ---------------------------------------------------------------------------
def read_estate(root):
    """Every module of the package, read strictly: what cannot be listed or read is named, never skipped."""
    root = Path(root)
    package = root / _PACKAGE_DIR
    unread, modules = [], {}
    if not package.is_dir():
        return Estate(root, modules, unread)

    def unlisted(error):
        rel = Path(os.path.relpath(error.filename, root)).as_posix()
        unread.append(f"{rel}: cannot be listed ({error.strerror})")

    found = []
    for current, dirs, names in os.walk(package, onerror=unlisted):
        dirs[:] = sorted(d for d in dirs if d not in _PRUNED)
        found.extend(Path(current) / name for name in names if name.endswith(".py"))
    for path in sorted(found):
        rel = path.relative_to(root).as_posix()
        try:
            modules[rel] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            unread.append(f"{rel}: cannot be read as UTF-8 text ({type(exc).__name__})")
    return Estate(root, modules, unread)


def green_line(census):
    pairs = _pairs(census)
    gated = sum(len(pairs.get(p, ())) for p in GATED)
    exempt_checked = sum(len(pairs.get(p, ())) - (_keyed_apart(census, *p, "keyed", pairs[p])[1]
                                                  if entry[0] == "keyed" and p in pairs else 0)
                         for p, entry in EXEMPT.items() if entry[0] in _CHECKED)
    exempt_argued = sum(len(pairs.get(p, ())) for p in EXEMPT) - exempt_checked
    owed = sum(len(pairs.get(p, ())) for p in _owed_pairs())
    sites = sum(len(m.sites) for m in census.modules.values())
    return (
        f"Write census OK: {len(census.estate.modules)} module(s) read, {len(census.modules)} censused, "
        f"{len(STORES)} store(s); {sites} write site(s): {_housed_count(census)} housed, {gated} gated in "
        f"{len(GATED)} pair(s) behind {len(GATES)} proven gate(s), {exempt_checked + exempt_argued} exempt in "
        f"{len(EXEMPT)} pair(s) ({exempt_checked} by a checked predicate, {exempt_argued} by a reason alone), "
        f"{owed} owed in {len(LEDGER)} module(s); {len(OUTSIDE)} module(s) that open a database are outside "
        f"the census by name; the ledger may only shrink."
    )


def run(root):
    """The census of the repository at ``root``: its exit code, its lines, and what it read."""
    estate = read_estate(root)
    if estate.unread:
        return Result(1, [f"Write census: FAILED -- {len(estate.unread)} part(s) of the estate could not be "
                          f"read, and what was not read was not counted:"]
                      + [f"  {reason}" for reason in estate.unread], estate, None)
    if not estate.modules:
        return Result(1, [f"Write census: nothing was scanned under {Path(root)}: no Python module in "
                          f"{_PACKAGE_DIR}/."], estate, None)
    try:
        census = take_census(estate)
        _store_openers(census)
    except RecursionError as exc:
        return Result(1, [f"Write census: FAILED -- an expression nests too deep to be followed ({exc}); what "
                          f"was not followed was not counted"], estate, None)
    if census.unparsed:
        return Result(1, ["Write census: FAILED -- these modules do not parse, so their writes cannot be "
                          "counted:"] + [f"  {rel}: {reason}" for rel, reason in sorted(census.unparsed.items())],
                      estate, census)
    try:
        found = find_all(census)
    except SuitesUnreadable as exc:
        return Result(1, [f"Write census: FAILED -- a suite that proves a gate could not be read: {exc}"],
                      estate, census)
    except RecursionError as exc:
        return Result(1, [f"Write census: FAILED -- a chain of callers nests too deep to be followed ({exc}); "
                          f"what was not followed was not judged"], estate, census)
    if census.unparsed:
        # A module the questions had to read, to follow a caller, did not parse.
        return Result(1, ["Write census: FAILED -- these modules do not parse, so the callers they may hold "
                          "cannot be read:"] + [f"  {rel}: {reason}" for rel, reason in sorted(census.unparsed.items())],
                      estate, census)
    lines = []
    for name, findings in found.items():
        if findings:
            lines.append(f"Write census: {name} ({len(findings)}):")
            lines.extend(f"  {finding}" for finding in findings)
    if lines:
        return Result(1, lines, estate, census)
    return Result(0, [green_line(census)], estate, census)


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[2]
    result = run(root)
    for line in result.lines:
        print(line)
    return result.code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
