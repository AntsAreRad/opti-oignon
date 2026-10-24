#!/usr/bin/env python3
"""Child environment census guard: the backend's process starts, held to the environment built for its children.

The backend is handed two secrets in its environment, the key file's passphrase and the master key. Its entry
takes them out first thing (``opti_oignon/process_secrets.py``, ``take()``), with what a shell runs code from, and
every program it starts directly is handed an environment built for it. ``take()`` also installs a check, at run
time, that refuses a process start Python audits whose environment or arguments hold a secret, or whose
environment names a shell variable, and refuses a secret's value or name, or a shell variable, put into the
process's environment, which the starts Python does not audit inherit. This
census is the developer's view of the same rules, on every commit, over the spellings and the bindings it follows
(listed below; what neither it nor the check sees is listed too):

  (a) every DIRECT START -- a call that runs a program and can be handed an environment -- is handed one built by a
      builder: a call to ``child_env`` imported from ``opti_oignon.process_secrets`` and bound to nothing else in
      its module, its extra environment a mapping the census reads; an untrusted tier's allowlist named in
      ``BUILDERS`` and defined once in the module that calls it; a built one with a literal mapping merged by
      ``|``; a name its function binds once, to such a call; or, in a function that takes the environment as a
      parameter and hands it on (a launcher), that parameter, every caller of the function in the package --
      its module's and the others' -- filling it with a built environment. A built environment may be read, its
      length taken, handed as the environment of a start or a launcher (a method through ``self`` or ``cls`` in
      its own class, when no other class of the module defines a method of that name), given a key the census
      reads, or updated with a literal mapping of such keys or with a parameter (a
      merger) whose default and every caller in the package pass one -- a merger handing its parameter on to
      another is a merger too, a retry no caller --; a key is never a secret's name nor a shell variable, and a
      value never reads
      ``os.environ`` nor a secret. Anything else -- an alias, a function it is handed to (as an argument or by
      ``env=``), a function nested in the one that starts, a key read from a variable -- is a finding, and so is
      a merger called nowhere in the package;
  (b) no start is handed a copy of the process's environment, nor an environment that anything derived from
      ``os.environ`` was merged into: ``None``, ``os.environ``, ``dict(os.environ)``, ``os.environ.copy()``,
      ``{**os.environ}``, ``os.environ | {...}``, a comprehension over it, a name bound to one of these, or an
      update with any expression that reads it, a name its function bound to one included;
  (c) every INHERITING START -- one no environment can be handed: a fork, a program run through a shell string, a
      subinterpreter, which runs without the audit hooks of the interpreter that made it, a multiprocessing child,
      process class, spawner, queue, lock, shared memory or resource tracker, a
      ``ProcessPoolExecutor``, a ``joblib.Parallel``, a ``webbrowser`` open, ``ctypes.util.find_library``, a C
      library's ``system``, ``fork`` or ``exec*`` through ctypes, the ``platform`` calls that may run ``uname -p``
      or ``file``, ``uuid.getnode``, an import that runs programs (``FORKING_IMPORTS``) -- is listed in
      ``INHERITING`` by module, function and spelling, with its reason and the contracts that prove what its
      children hold; an unlisted one is a finding, and so is a listed one the census no longer finds;
  (d) a secret's name appears in only two places outside ``process_secrets``: the first argument of one of the
      holder's readers (``secret``, ``withdraw``, ``encryption._secret``; by position, or as ``name=``), its name
      -- or the module it is read from, dotted or not -- bound where it is called by imports of the holder alone,
      in the scope that decides: the function's own (an assignment, a parameter, a nested definition binding it
      otherwise), else each function it is nested in, else the module's -- the name bound nowhere that chain
      does not read: the class body it is called in, a ``global`` or ``nonlocal``, a comprehension, a lambda's
      parameter, a ``match`` capture, a type parameter --; and the one module-level constant that names it, bound
      once
      (``IDENT = "<name>"``, or a tuple assignment pairing names with literals). Every other place is a finding
      -- a read or a write in the environment, a name handed to another function, compared, made a parameter's
      default, a keyword, a mapping's key, bound in a function or to another constant --, wherever the name comes
      from: a literal (text or bytes; escapes and implicit concatenation read as the parser reads them), a
      constant of the module or of another, read as an attribute or through ``getattr``, imported under any name
      or handed on through other modules (by name, under an alias, by a star, in the top level's ``try``), an item
      of the holder's ``SECRET_NAMES``, itself handed on. A mapping's value and a log call's argument are places
      like any other: a status or a log line that names the variable says it in a sentence, which is no place
      (an f-string with words of its own). And no secret's VALUE -- a read of a secret, or a name bound to one
      where it is read: by its function, a function it is nested in, or the module, anywhere at its top level or
      under ``global`` -- is put into the process's environment, as a variable's value or its name (a key of
      ``os.environ``, a tuple's element too, ``|=``, ``+=``, an update, a setdefault, ``os.putenv``), which every
      child handed none inherits;
  (e) every process sink the egress census charges is a start this census counts, on the same line and in the
      same function. The twin runs one way: a start this census counts that the egress census does not charge
      is not checked here;
  (f) what cannot be read fails the census by name: an estate with no module, a file that is not UTF-8 text, a
      module that does not parse, a module deeper than the levels whose names it follows, a directory the walk
      cannot list, a suite read for a proof, a ``process_secrets.py`` that does not write its names.

Every module of the package is read whole. The secrets' names, the shell's variables and the prefix of exported
functions are read from ``SECRET_NAMES``, ``SHELL_HAZARDS`` and ``FUNCTION_PREFIX`` in
``opti_oignon/process_secrets.py``, never written here. Modules exempt by name (``EXEMPT``) are held to no rule,
with their reason. The bundled plugins run in the plugin worker, whose environment is an allowlist that holds no
secret: they are not the backend's children.

SPELLINGS AND BINDINGS FOLLOWED: imports and aliases (relative ones included), star imports, ``import_module`` and
``__import__`` with a constant, a ``sys.modules`` lookup, ``getattr`` with a constant, a start module reached as an
attribute of another module (``webbrowser.subprocess``, a module of the package) or imported from one, ``posix``
for ``os``, ``asyncio.subprocess``, an assignment to a name, to a tuple of names or to an attribute, a class
attribute, the walrus, a ``with ... as`` or ``for`` target, a class whose base is a start's class,
``functools.partial``, ``staticmethod`` and ``classmethod`` of a start, a dictionary, tuple or list of starts
indexed, a C library's function read by its name, and a function that returns what it was handed. An event loop's
``subprocess_exec`` and ``subprocess_shell`` count in every module, whatever the loop is bound to. Names are not
scoped: a name or an attribute bound to a start anywhere in a module is one everywhere in it, which can only
over-count.

THE QUESTIONS, with disjoint domains:

  * ``find_unbuilt_starts``       -- a direct start whose environment no builder made, or that is used past what
                                     the census reads, a launcher's or a merger's callers in other modules
                                     included (a).
  * ``find_copied_environments``  -- a start handed a copy of the process's environment, or one merged with it (b).
  * ``find_unlisted_inheriting``  -- an inheriting start ``INHERITING`` does not list (c).
  * ``find_stale_inheriting``     -- a listed inheriting start the census no longer finds (c).
  * ``find_inheriting_unproven``  -- a listed contract that names no test function, or whose suite does not
                                     name the module (c).
  * ``find_secret_reads``         -- a secret's name anywhere but its two places, or a secret's value put into the
                                     process's environment (d).
  * ``find_twin_gaps``            -- an egress process sink this census does not count (e).
  * ``find_stale_exemptions``     -- an exempt module with no start, or gone.

The green line carries its denominator: the modules scanned, the direct starts and what built each environment,
the inheriting starts listed and the modules that hold them, the modules exempt, the egress process sinks
matched.

WHAT THE CENSUS DOES NOT SEE, said here rather than claimed covered -- the check ``take()`` installs refuses at run
time, of these, a start Python audits that would hand a child a secret's value or name or a shell variable, and a
secret's value or name, or a shell variable, put into the process's environment; what neither sees is marked
(neither):

  * a name assembled at run time, or read through ``vars``, ``__dict__``, ``globals``, ``exec`` or ``eval``;
  * a start function received from another module as a value, or handed to a function that calls it
    (``asyncio.to_thread(subprocess.run, ...)``), a start reached through a container's ``get`` or by
    iterating it, and an environment a ``functools.partial`` carries: the partial's start is counted, its
    environment is not read;
  * an environment hidden in ``**kwargs`` or ``*args``: such a start is a finding, never taken for a built one;
  * what a builder returns: ``child_env`` and the allowlists are proven by their contracts, not read here;
  * control flow: a name bound once to a builder is trusted on every path;
  * a secret's value that reaches an environment through a function, or a name bound outside the function that
    starts: a value is read where it is put, a reader's call or a name bound to one in that function;
  * a secret's name computed, or reached through a loop over a collection of names;
  * the C library reached through ctypes other than by ``CDLL``, ``PyDLL`` or ``cdll``, and a raw ``syscall``; a C
    function, called through ctypes or a library's own C, that starts a process or writes the environment
    (neither);
  * a direct call of ``_posixsubprocess.fork_exec`` (``subprocess._fork_exec``) that hands its child an explicit
    environment holding a value the census cannot read: counted here as a start, never audited at run time
    (neither, past the count);
  * code a subinterpreter runs: counted here as an inheriting start where one is made, never checked at run time
    -- the hook is the interpreter's own -- (neither, past the count);
  * an ``os.environ`` update the check refuses midway: the variables before the refused one are set;
  * a launcher's method called from a module that imports neither the launcher's module nor one that hands a name
    of it on -- an instance handed in from elsewhere --, a launcher's method that a class of another module
    overrides, which a call through ``self`` in the launcher's class may reach, and a launcher or a merger called
    from a module that defines a function, or binds an attribute, of the same name: that module is taken to call
    its own;
  * what a child does once it runs, and the children of third-party code beyond ``FORKING_IMPORTS``;
  * code outside the package (``scripts/``, ``.github/scripts/``, ``tests/``) and the frontend;
  * every directory named ``data``, which the walk never opens.

The helpers are pure and import-safe; ``main`` scans the repository and exits non-zero on any finding.
Usage: ``child_env_census_guard.py [REPO_ROOT]``.
"""

import ast
import importlib.util
import sys
from collections import Counter, namedtuple
from pathlib import Path

_PACKAGE_DIR = "opti_oignon"
_SECRETS_MODULE = "opti_oignon/process_secrets.py"
_HOLDER = "opti_oignon.process_secrets"
_CHILD_ENV = _HOLDER + ".child_env"
# The holder's reads, which a secret's name may be handed to.
_SECRET_READERS = frozenset({_HOLDER + ".secret", _HOLDER + ".withdraw", "opti_oignon.encryption._secret"})
_MODULE_LEVEL = "<module>"

Inheriting = namedtuple("Inheriting", "sites reason contracts")
Start = namedtuple("Start", "kind line function qual")
Result = namedtuple("Result", "code lines estate census")
Names = namedtuple("Names", "secrets hazards prefix ordered", defaults=((),))
_NO_NAMES = Names(frozenset(), frozenset(), "\0")


# ---------------------------------------------------------------------------
# The tables.
# ---------------------------------------------------------------------------
# The untrusted tiers' environments, each an allowlist defined in its own module: name -> that module.
BUILDERS = {
    "_build_plugin_env": "opti_oignon/plugin_subprocess.py",
    "_tempdir_env": "opti_oignon/sandbox_manager.py",
}

# Starts no environment can be handed: module -> {function: the spellings it holds}, why they hold no secret, and
# the contracts that prove it.
INHERITING = {
    "opti_oignon/auto_tuner.py": Inheriting(
        sites={"get_hardware_fingerprint": ("platform.processor",)},
        reason=(
            "platform.processor() runs `uname -p`, once per process, when a tuning run's result is recorded: "
            "long after the entry took the secrets out of the environment"
        ),
        contracts=("xe20",),
    ),
    "opti_oignon/background_pool.py": Inheriting(
        sites={"_spawn_executor": ("concurrent.futures.ProcessPoolExecutor",),
               "submit_alone": ("multiprocessing.get_context().SimpleQueue",),
               "_start": ("multiprocessing.get_context().SimpleQueue",)},
        reason=(
            "the pool's spawn-context workers, and the queues their pids come back on, which start the "
            "multiprocessing resource tracker: each starts with the environment as it is then, out of which the "
            "entry took the secrets before any request ran"
        ),
        contracts=("xe12",),
    ),
    "opti_oignon/conversation_wipe.py": Inheriting(
        sites={_MODULE_LEVEL: ("ctypes.util.find_library",)},
        reason=(
            "find_library('c') at import runs ldconfig with an environment of its own (LC_ALL and LANG); the C "
            "library is in its cache, so no compiler is run"
        ),
        contracts=("xe19",),
    ),
    "opti_oignon/pqc_signatures.py": Inheriting(
        sites={"_liboqs_candidates": ("ctypes.util.find_library",), _MODULE_LEVEL: ("import oqs",),
               "generate_pqc_keypair": ("import oqs",), "sign_bytes": ("import oqs",),
               "verify_bytes": ("import oqs",)},
        reason=(
            "liboqs is looked up with find_library -- the compiler and the linker when ldconfig does not know "
            "the name -- and liboqs-python looks again when it is imported; both run when the module is "
            "imported, after the entry took the secrets out of the environment. The functions import the module "
            "the import at load found, out of the module cache, which runs nothing"
        ),
        contracts=("xe17", "xe19"),
    ),
    "opti_oignon/secure_bytes.py": Inheriting(
        sites={_MODULE_LEVEL: ("ctypes.util.find_library",)},
        reason=(
            "find_library('c') at import, as conversation_wipe.py; the encryption module imports it, after the "
            "entry took both secrets out of the environment"
        ),
        contracts=("xe19",),
    ),
}

# Not the backend's children, by name, with the reason.
EXEMPT = {
    "opti_oignon/ui.py": (
        "the launcher: it takes both secrets out of its own environment and zeroes them there first (UL8, "
        "xe25), hands them to the backend alone, and its other children -- npm, the frontend, the browser -- "
        "need the user's session as it is"
    ),
}

# Imports that run programs: the third-party module -> what it runs.
FORKING_IMPORTS = {
    "oqs": (
        "liboqs-python looks liboqs up with ctypes.util.find_library at import -- ldconfig, then the compiler "
        "and the linker -- and clones and builds liboqs from GitHub when it finds none"
    ),
}


# ---------------------------------------------------------------------------
# What counts.
# ---------------------------------------------------------------------------
# An event loop's own starts, in every module, whatever the loop is bound to; one taken as a value -- bound, wrapped,
# read through getattr -- is a start whose environment cannot be read.
_LOOP = "asyncio.loop."
_LOOP_STARTS = frozenset({"subprocess_exec", "subprocess_shell"})
_AS_VALUE = ", taken as a value"
_ENV_KEYWORD = frozenset({
    "subprocess.run", "subprocess.Popen", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
    "asyncio.create_subprocess_exec", "asyncio.create_subprocess_shell",
    "asyncio.subprocess.create_subprocess_exec", "asyncio.subprocess.create_subprocess_shell",
    _LOOP + "subprocess_exec", _LOOP + "subprocess_shell",
})
# Popen's own positional order puts the environment eleventh.
_POPEN_ENV_INDEX = 10
# The os calls that take the environment as a positional argument: name -> its index, None for the last.
_ENV_POSITIONAL = {
    "os.execve": 2, "os.execvpe": 2, "os.execle": None, "os.execlpe": None,
    "os.spawnve": 3, "os.spawnvpe": 3, "os.spawnle": None, "os.spawnlpe": None,
    "os.posix_spawn": 2, "os.posix_spawnp": 2,
    # The stdlib's own start, which subprocess and multiprocessing call and which Python does not audit.
    "_posixsubprocess.fork_exec": 5, "subprocess._fork_exec": 5,
}
_INHERITING_CALLS = frozenset({
    "subprocess.getoutput", "subprocess.getstatusoutput", "os.system", "os.popen",
    "os.execv", "os.execl", "os.execlp", "os.execvp", "os.spawnv", "os.spawnl", "os.spawnlp", "os.spawnvp",
    "os.fork", "os.forkpty", "pty.fork", "pty.spawn",
    "concurrent.futures.ProcessPoolExecutor", "concurrent.futures.process.ProcessPoolExecutor",
    "ctypes.util.find_library",
    "platform.processor", "platform.platform", "platform.architecture",
    "uuid.getnode", "uuid.uuid1",
    "joblib.Parallel",
    "multiprocessing.resource_tracker.register",
    # A subinterpreter runs code without this interpreter's audit hooks, the check take() installs among them.
    "_interpreters.create", "_interpreters.exec", "_interpreters.run_string", "_xxsubinterpreters.create",
    "_xxsubinterpreters.run_string", "interpreters.create", "concurrent.interpreters.create",
    "concurrent.futures.InterpreterPoolExecutor",
})
# What multiprocessing builds or calls that starts a process, or its resource tracker.
_MULTIPROCESSING = frozenset({
    "Process", "Pool", "Manager", "SyncManager", "BaseManager", "Queue", "SimpleQueue", "JoinableQueue", "Lock",
    "RLock", "Semaphore", "BoundedSemaphore", "Condition", "Event", "Barrier", "Value", "Array", "SharedMemory",
    "ShareableList", "ensure_running", "spawnv_passfds", "Popen",
})
_BROWSER_OPENS = frozenset({"open", "open_new", "open_new_tab"})
# What a C library reached through ctypes starts.
_C_STARTS = frozenset({
    "system", "popen", "fork", "vfork", "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp",
    "posix_spawn", "posix_spawnp", "clone", "daemon",
})
_C_LIBRARIES = ("ctypes.CDLL(", "ctypes.PyDLL(", "ctypes.cdll.")
# A start module, wherever it is reached from: ``webbrowser.subprocess.Popen`` is ``subprocess.Popen``.
_ROOT_SEGMENTS = frozenset({
    "subprocess", "os", "posix", "pty", "asyncio", "multiprocessing", "concurrent", "webbrowser", "ctypes",
    "platform", "uuid", "joblib", "_posixsubprocess", "_interpreters", "_xxsubinterpreters", "interpreters",
})
_UNAME = "platform.uname"
# The fields of platform.uname() that never run a program; the processor field, or the whole result, may.
_UNAME_FIELDS = frozenset({"system", "node", "release", "version", "machine"})
_IMPORT_FUNCTIONS = frozenset({"importlib.import_module", "builtins.__import__", "__import__"})
_WRAPPERS = frozenset({"functools.partial", "staticmethod", "classmethod", "builtins.staticmethod",
                       "builtins.classmethod"})
# Calls whose result the census follows besides a class's: they build the object a start is later made on.
_FACTORIES = frozenset({"multiprocessing.get_context", "ctypes.CDLL", "ctypes.PyDLL", "ctypes.cdll.LoadLibrary",
                        "webbrowser.get", "platform.uname"})

# The nodes a type parameter is (Python 3.12 and later): each binds its name in a scope of its own.
_TYPE_PARAMS = tuple(getattr(ast, kind) for kind in ("TypeVar", "ParamSpec", "TypeVarTuple") if hasattr(ast, kind))

_ENVIRON = frozenset({"os.environ", "os.environb"})
_READS = frozenset({"get", "__getitem__", "__contains__"})
_WRITES = frozenset({"pop", "setdefault", "__setitem__", "__delitem__"})
_ENV_FUNCTIONS = {"os.getenv": "reads", "os.getenvb": "reads", "os.putenv": "writes", "os.unsetenv": "writes"}
# What a built environment may be asked without a finding: it reads it, or empties it.
_BUILT_READS = frozenset({"get", "keys", "items", "values", "pop", "clear", "__contains__", "__getitem__"})

# The roots a start, an environment or a builder can be reached from: a name is bound only to what starts at one,
# or at a module of the package, through which a start module can be reached too.
_WATCHED = frozenset({
    "subprocess", "os", "posix", "asyncio", "pty", "multiprocessing", "concurrent", "webbrowser", "ctypes",
    "platform", "uuid", "importlib", "builtins", "__import__", "functools", "joblib", "staticmethod",
    "classmethod", "_posixsubprocess", "_interpreters", "_xxsubinterpreters", "interpreters", "logging", _PACKAGE_DIR,
}) | frozenset(FORKING_IMPORTS)
_DEPTH = 10
# A name of the package is followed through its module and two attributes past it -- a constant, a start module and
# its function --; a module deeper than that fails the census by name (``run``), its names not followed.
_PACKAGE_DEPTH = 8
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _watched(q):
    root = q.split(".", 1)[0].split("(", 1)[0]
    if root == _PACKAGE_DIR:
        return q.count(".") <= _PACKAGE_DEPTH
    return root in _WATCHED and q.count(".") < _DEPTH


def _carried(values):
    """What an attribute or a loop's target may stand for: a start, never an object of the package -- the holder
    of the secrets aside --, whose chains of attributes would grow without telling anything."""
    return {q for q in values if not q.startswith(_PACKAGE_DIR + ".") or q.startswith(_HOLDER)}


def _leaf(q):
    return q.rsplit(".", 1)[-1]


def _called_name(func):
    """The name a call is made by: the bare name or the attribute."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _candidates(q):
    """``q`` and what it stands for: the start module reached through another module, ``posix`` read as ``os``."""
    parts = q.split(".")
    out = [q] + [".".join(parts[i:]) for i in range(1, len(parts)) if parts[i] in _ROOT_SEGMENTS]
    return out + ["os." + c[len("posix."):] for c in out if c.startswith("posix.")]


def _norm(quals):
    """Every name ``quals`` stand for, read through the modules that reach a start module."""
    return {c for q in quals for c in _candidates(q)}


def _kind_of(q):
    if q in _ENV_KEYWORD or q in _ENV_POSITIONAL:
        return "direct"
    if q in _INHERITING_CALLS:
        return "inheriting"
    leaf = _leaf(q)
    if "(" in leaf:
        return None
    if q.startswith("multiprocessing.") and (leaf in _MULTIPROCESSING or leaf.endswith("Process")):
        return "inheriting"
    if q.startswith("webbrowser.") and leaf in _BROWSER_OPENS:
        return "inheriting"
    if q.startswith(_C_LIBRARIES) and leaf in _C_STARTS:
        return "inheriting"
    return None


def _classify(quals):
    """``(kind, qual)`` of the start among ``quals`` -- a direct one first, which is held to a built environment --,
    or ``(None, None)``."""
    found = []
    for q in quals:
        for candidate in _candidates(q):
            kind = _kind_of(candidate)
            if kind:
                found.append((kind, candidate))
                break
    for wanted in ("direct", "inheriting"):
        quals_of = sorted(candidate for kind, candidate in found if kind == wanted)
        if quals_of:
            return wanted, quals_of[0]
    return None, None


def _target_ids(target):
    """The names a target binds: a subscript or an attribute changes an object, it binds no name."""
    out, stack = set(), [target]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            stack.extend(node.elts)
        elif isinstance(node, ast.Starred):
            stack.append(node.value)
    return out


def _literal_key(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        return node.value.decode("utf-8", "replace") if isinstance(node.value, bytes) else node.value
    return None


# ---------------------------------------------------------------------------
# One module.
# ---------------------------------------------------------------------------
class _Module:
    """What one module starts, what each start is handed, and where it reads a secret's name in the environment.

    The syntax is read once, when the module is built, and let go unless ``keep``: a census of hundreds of modules
    that held every tree alive at once would spend more time in the garbage collector than in the census.
    """

    def __init__(self, rel, text, names=None, constants=None, keep=False):
        self.rel = rel
        self.starts, self.verdicts, self.secret_uses, self.launchers, self.unparsed = [], [], [], [], None
        self.mergers, self.definitions, self.reexports, self.identifiers = [], {}, [], set()
        self.strings, self.keyword_names, self.imported_names = set(), set(), set()
        self.names_of, self.tree = names, None
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError) as exc:
            # Recorded, and the census fails by name: a module that does not parse counts no start, which is
            # not the same as having none.
            self.unparsed = f"{type(exc).__name__}: {exc}"
            return
        dotted = rel[:-3].replace("/", ".")
        is_package = dotted.endswith(".__init__")
        self.dotted = dotted[: -len(".__init__")] if is_package else dotted
        self.package = self.dotted if is_package else self.dotted.rpartition(".")[0]
        self.tree = tree
        self._collect()
        self._bind()
        self._take_starts()
        self.start_calls = {id(node) for start, node in self.nodes.items() if start.kind == "direct"}
        self.launcher_params = self._launcher_params()
        self.verdicts = [_verdict_of(self, start) for start in self.starts if start.kind == "direct"]
        self.identifiers = set(self.loaded)
        if names is not None and rel != _SECRETS_MODULE and rel not in EXEMPT:
            self.secret_uses = _secret_value_puts(self)
            if self.strings & names.secrets:
                self.definitions = _definitions(tree, names.secrets)[0]
            self.reexports = _reexports(self)
            if constants is not None:
                self.secret_uses = sorted(set(self.secret_uses) | set(_secret_uses(self, names, constants)))
            # A module that spells a secret's name keeps its syntax for the name rule, which needs every constant.
            keep = keep or bool(self.strings & names.secrets or self.keyword_names & names.secrets)
        if not keep:
            self.release()

    def release(self):
        """Let the syntax go; what the questions read is kept."""
        for field in ("parent", "owner", "calls", "imports", "binders", "returns_of", "subscripts",
                      "compares", "augassigns", "loaded", "binding_map", "names", "attrs", "returns", "nodes",
                      "_memo", "bound", "class_body", "defaults", "defined_in", "globals_named"):
            self.__dict__.pop(field, None)
        self.tree = None

    # -- one pass over the tree -----------------------------------------
    def _collect(self):
        self.parent, self.owner = {}, {}
        self.calls, self.imports, self.binders, self.subscripts, self.compares, self.augassigns = [], [], [], [], [], []
        self.returns_of = {}
        self.loaded, self.binding_map, self.bound, self.class_body, self.defaults = {}, {}, {}, set(), []
        self.strings, self.keyword_names, self.imported_names = set(), set(), set()
        self.defined_in, self.methods, self.aside_bound, self.globals_named = {}, {}, set(), {}
        self.module_functions = {node.name for node in self.tree.body if isinstance(node, _FUNCTIONS)}
        stack = [(self.tree, None)]
        while stack:
            node, owner = stack.pop()
            if isinstance(node, ast.ClassDef):
                self.class_body.update(id(statement) for statement in node.body)
            for child in ast.iter_child_nodes(node):
                self.parent[id(child)] = node
                self.owner[id(child)] = owner
                kind = type(child)
                if kind in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef) and not isinstance(node, ast.ClassDef):
                    # A method binds no name of the scope its class sits in.
                    self.defined_in.setdefault((id(owner), child.name), []).append(child)
                if kind in (ast.FunctionDef, ast.AsyncFunctionDef) and isinstance(node, ast.ClassDef):
                    self.methods.setdefault(child.name, set()).add(node.name)
                self._note_aside(child)
                if kind in (ast.FunctionDef, ast.AsyncFunctionDef):
                    self._bound(child.name, "def")
                    arguments = child.args
                    for argument in arguments.posonlyargs + arguments.args + arguments.kwonlyargs + \
                            [arguments.vararg, arguments.kwarg]:
                        if argument is not None:
                            self._bound(argument.arg, "parameter")
                    self.defaults.extend(list(arguments.defaults) + [d for d in arguments.kw_defaults if d is not None])
                elif kind is ast.ClassDef:
                    self._bound(child.name, "class")
                elif kind is ast.Call:
                    self.calls.append(child)
                elif kind in (ast.Import, ast.ImportFrom):
                    self.imports.append(child)
                    for alias in child.names:
                        self._bound(alias.asname or alias.name.split(".")[0], "import")
                    if kind is ast.ImportFrom:
                        self.imported_names.update(alias.name for alias in child.names)
                elif kind is ast.Constant and isinstance(child.value, (str, bytes)):
                    self.strings.add(child.value if isinstance(child.value, str)
                                     else child.value.decode("utf-8", "replace"))
                elif kind is ast.keyword and child.arg:
                    self.keyword_names.add(child.arg)
                elif kind is ast.Subscript:
                    self.subscripts.append(child)
                elif kind is ast.Compare:
                    self.compares.append(child)
                elif kind is ast.Return and owner is not None and child.value is not None:
                    self.returns_of.setdefault(owner.name, []).append(child.value)
                elif kind is ast.Name and isinstance(child.ctx, ast.Load):
                    self.loaded.setdefault(child.id, []).append(child)
                elif kind is ast.Attribute and isinstance(child.ctx, ast.Load):
                    self.loaded.setdefault(child.attr, []).append(child)
                if kind in (ast.Assign, ast.AnnAssign, ast.NamedExpr, ast.withitem, ast.ClassDef, ast.For,
                            ast.AsyncFor, ast.comprehension):
                    self.binders.append(child)
                if kind is ast.AugAssign:
                    self.augassigns.append(child)
                self._note_binding(child, owner)
                stack.append((child, child if kind in (ast.FunctionDef, ast.AsyncFunctionDef) else owner))

    def _bound(self, name, how):
        self.bound.setdefault(name, []).append(how)

    def _note_aside(self, node):
        """A name bound where the scope chain does not read: a ``global`` or ``nonlocal``, a comprehension, a
        lambda's parameter, a ``match`` capture, a type parameter."""
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            self.aside_bound.update(node.names)
            if isinstance(node, ast.Global):
                for name in node.names:
                    self.globals_named.setdefault(name, []).append(node)
        elif isinstance(node, ast.comprehension):
            self.aside_bound.update(_target_ids(node.target))
        elif isinstance(node, ast.Lambda):
            args = node.args
            self.aside_bound.update(arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs
                                    + [args.vararg, args.kwarg] if arg is not None)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            self.aside_bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            self.aside_bound.add(node.rest)
        elif isinstance(node, _TYPE_PARAMS):
            self.aside_bound.add(node.name)

    def _note_binding(self, node, owner):
        if isinstance(node, ast.Assign):
            ids = set().union(*(_target_ids(target) for target in node.targets))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.For, ast.AsyncFor)):
            ids = _target_ids(node.target)
        elif isinstance(node, ast.withitem):
            ids = _target_ids(node.optional_vars) if node.optional_vars is not None else set()
        elif isinstance(node, ast.ExceptHandler):
            ids = {node.name} if node.name else set()
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            ids = set(node.names)
        elif isinstance(node, ast.Delete):
            ids = set().union(*(_target_ids(target) for target in node.targets))
        else:
            return
        targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else \
            [getattr(node, "target", None) or getattr(node, "optional_vars", None)]
        for target in targets:
            if isinstance(target, ast.Attribute):
                self._bound(target.attr, "attribute")
        for name in ids:
            self._bound(name, "assignment")
            # The module's own (owner None, its class bodies with it) are read by the value rule alone.
            self.binding_map.setdefault((id(owner), name), []).append(node)

    def function_of(self, node):
        owner = self.owner.get(id(node))
        return owner.name if owner is not None else _MODULE_LEVEL

    def owner_of(self, node):
        return self.owner.get(id(node))

    def bindings(self, owner, name):
        """Every statement of ``owner`` (not of a function nested in it) that binds ``name``."""
        return self.binding_map.get((id(owner), name), [])

    @staticmethod
    def parameters(owner):
        args = owner.args
        return [arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs]

    def sole_import(self, name):
        """Whether ``name`` is bound in this module by imports alone."""
        hows = self.bound.get(name, [])
        return bool(hows) and all(how == "import" for how in hows)

    # -- bindings -------------------------------------------------------
    def _absolute(self, node):
        if not node.level:
            return node.module
        base = self.package.split(".") if self.package else []
        if node.level > 1:
            base = base[: max(0, len(base) - (node.level - 1))]
        return ".".join(base + ([node.module] if node.module else [])) or None

    def _add(self, table, key, values):
        values = {q for q in values if _watched(q)}
        if not values:
            return
        known = table.setdefault(key, set())
        if not values <= known:
            known |= values
            self._grown = True

    def _bind_target(self, target, value):
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) \
                and len(target.elts) == len(value.elts) \
                and not any(isinstance(element, ast.Starred) for element in target.elts + value.elts):
            for sub_target, sub_value in zip(target.elts, value.elts):
                self._bind_target(sub_target, sub_value)
            return
        values = self.resolve(value)
        if isinstance(target, ast.Attribute):
            # Unscoped: every attribute of that name in the module stands for it, which can only over-count.
            self._add(self.attrs, target.attr, _carried(values))
            return
        for name in _target_ids(target):
            self._add(self.names, name, values)

    def _bind(self):
        self.names, self.attrs, self.returns, self.stars, self._memo = {}, {}, {}, set(), {}
        self.aliases = {}
        for node in self.imports:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self._add(self.names, alias.asname, {alias.name})
                    else:
                        top = alias.name.split(".")[0]
                        self._add(self.names, top, {top})
                continue
            module = self._absolute(node)
            for alias in node.names if module else ():
                if alias.name == "*":
                    self.stars.add(module)
                else:
                    self._add(self.names, alias.asname or alias.name, {module + "." + alias.name})
                    if alias.asname:
                        self.aliases[alias.asname] = alias.name
        self._grown = True
        while self._grown:
            self._grown = False
            self._memo = {}
            for name, values in self.returns_of.items():
                self._add(self.returns, name, set().union(*(self.resolve(value) for value in values)))
            for node in self.binders:
                if isinstance(node, ast.ClassDef):
                    # A subclass of a start's class starts what its base starts.
                    self._add(self.names, node.name, set().union(*(self.resolve(base) for base in node.bases)))
                elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                    for name in _target_ids(node.target):
                        self._add(self.names, name, _carried(self.resolve(node.iter)))
                elif isinstance(node, ast.withitem):
                    if node.optional_vars is not None:
                        self._bind_target(node.optional_vars, node.context_expr)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        self._bind_target(target, node.value)
                        if id(node) in self.class_body and isinstance(target, ast.Name):
                            # A class attribute, called through the class or an instance.
                            self._add(self.attrs, target.id, _carried(self.resolve(node.value)))
                elif node.value is not None:
                    self._bind_target(node.target, node.value)
        self._memo = {}

    def resolve(self, node):
        """Every dotted name an expression may stand for; ``()`` marks what a call returns."""
        if node is None:
            return frozenset()
        key = id(node)
        memo = self._memo
        if key in memo:
            return memo[key]
        memo[key] = frozenset()
        out = frozenset(q for q in self._resolve(node) if _watched(q))
        memo[key] = out
        return out

    def _resolve(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.names:
                return self.names[node.id]
            if node.id in ("__import__", "staticmethod", "classmethod"):
                return {node.id}
            return {module + "." + node.id for module in self.stars}
        if isinstance(node, ast.Attribute):
            return {q + "." + node.attr for q in self.resolve(node.value)} | self.attrs.get(node.attr, set())
        if isinstance(node, ast.Call):
            return self._call_result(node)
        if isinstance(node, (ast.NamedExpr, ast.Await, ast.Starred)):
            return self.resolve(node.value)
        if isinstance(node, ast.BoolOp):
            return set().union(*(self.resolve(value) for value in node.values))
        if isinstance(node, ast.IfExp):
            return self.resolve(node.body) | self.resolve(node.orelse)
        if isinstance(node, ast.Dict):
            return set().union(*(self.resolve(value) for value in node.values)) if node.values else set()
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return set().union(*(self.resolve(element) for element in node.elts)) if node.elts else set()
        if isinstance(node, ast.Subscript):
            container, index = node.value, node.slice
            if isinstance(container, ast.Attribute) and container.attr == "modules" \
                    and isinstance(index, ast.Constant) and isinstance(index.value, str):
                return {index.value}
            # An item of a container of starts is a start; an item of the environment is no environment; a C
            # library's function read by its name is that function: ``CDLL(None)["system"]``.
            quals = set(self.resolve(container))
            if isinstance(index, ast.Constant) and isinstance(index.value, str):
                quals |= {q + "." + index.value for q in quals if q.startswith(_C_LIBRARIES)}
            return quals - _ENVIRON
        return set()

    def _call_result(self, call):
        func = call.func
        quals = self.resolve(func)
        first = call.args[0] if call.args else None
        literal = isinstance(first, ast.Constant) and isinstance(first.value, str)
        if quals & _IMPORT_FUNCTIONS and literal:
            return {first.value}
        if isinstance(func, ast.Name) and func.id == "getattr" and len(call.args) >= 2:
            name = call.args[1]
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                return {q + "." + name.value for q in self.resolve(call.args[0])}
        if isinstance(func, ast.Attribute) and func.attr == "get" and isinstance(func.value, ast.Attribute) \
                and func.value.attr == "modules" and literal:
            return {first.value}
        if quals & _WRAPPERS and call.args:
            # A partial, a static or a class method stands for the function it wraps.
            return set(self.resolve(call.args[0]))
        # What a class or a factory builds is followed; what any other call returns is not -- a result called in
        # turn would grow a chain without end.
        out = {q + "()" for q in quals if not q.endswith(")")
               and (_leaf(q)[:1].isupper() or any(c in _FACTORIES for c in _candidates(q)))}
        name = _called_name(func)
        if name in self.returns and isinstance(func, (ast.Name, ast.Attribute)):
            out |= self.returns[name]
        return out

    # -- the starts -----------------------------------------------------
    def _take_starts(self):
        found = []
        for node in self.calls:
            quals = sorted(self.resolve(node.func))
            kind, qual = _classify(quals)
            if kind is None and isinstance(node.func, ast.Attribute) and node.func.attr in _LOOP_STARTS:
                kind, qual = "direct", _LOOP + node.func.attr
            if kind is None and isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2 \
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value in _LOOP_STARTS:
                kind, qual = "direct", _LOOP + node.args[1].value + _AS_VALUE
            if kind is None and _UNAME in _norm(quals) and not self._uname_field(node):
                kind, qual = "inheriting", _UNAME + "()"
            if kind is None and set(quals) & _IMPORT_FUNCTIONS and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str) \
                        and first.value.split(".")[0] in FORKING_IMPORTS:
                    kind, qual = "inheriting", "import " + first.value
            if kind is not None:
                found.append((Start(kind, node.lineno, self.function_of(node), qual), node))
        for attr in _LOOP_STARTS:
            for node in self.loaded.get(attr, ()):
                parent = self.parent.get(id(node))
                if isinstance(node, ast.Attribute) and not (isinstance(parent, ast.Call) and parent.func is node):
                    found.append((Start("direct", node.lineno, self.function_of(node), _LOOP + attr + _AS_VALUE), node))
        for node in self.imports:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in FORKING_IMPORTS:
                        found.append((Start("inheriting", node.lineno, self.function_of(node),
                                            "import " + alias.name), node))
            elif not node.level and node.module and node.module.split(".")[0] in FORKING_IMPORTS:
                found.append((Start("inheriting", node.lineno, self.function_of(node), "import " + node.module),
                              node))
        found.sort(key=lambda pair: (pair[0].line, pair[0].qual, pair[0].kind))
        self.starts = [start for start, _node in found]
        self.nodes = dict(found)

    def _launcher_params(self):
        """function name -> [(its environment parameter, whether it is a method, its class)]: each function that
        hands one of its parameters on as a direct start's environment."""
        out = {}
        for start, node in self.nodes.items():
            if start.kind != "direct" or start.qual.endswith(_AS_VALUE):
                continue
            value, _how = self.env_argument(start, node)
            owner = self.owner_of(node)
            if isinstance(value, ast.Name) and owner is not None and value.id in self.parameters(owner) \
                    and not self.bindings(owner, value.id):
                params = [arg.arg for arg in owner.args.posonlyargs + owner.args.args]
                out.setdefault(owner.name, []).append((value.id, params[:1] in (["self"], ["cls"]),
                                                       _class_of(self, owner)))
        return out

    def _uname_field(self, call):
        """Whether a uname() call is read only for a field that never runs a program."""
        parent = self.parent.get(id(call))
        return isinstance(parent, ast.Attribute) and parent.value is call and parent.attr in _UNAME_FIELDS

    # -- what a start is handed -----------------------------------------
    def env_argument(self, start, call=None):
        """``(node, how)``: the environment a direct start is handed, and how it was given."""
        call = self.nodes[start] if call is None else call
        if start.qual in _ENV_KEYWORD:
            for keyword in call.keywords:
                if keyword.arg == "env":
                    return keyword.value, "env="
            if any(keyword.arg is None for keyword in call.keywords):
                return None, "**kwargs"
            if any(isinstance(arg, ast.Starred) for arg in call.args):
                return None, "*args"
            if start.qual == "subprocess.Popen" and len(call.args) > _POPEN_ENV_INDEX:
                return call.args[_POPEN_ENV_INDEX], "positional"
            return None, "absent"
        for keyword in call.keywords:
            if keyword.arg == "env":
                return keyword.value, "env="
        if any(isinstance(arg, ast.Starred) for arg in call.args):
            return None, "*args"
        index = _ENV_POSITIONAL[start.qual]
        if index is None:
            return (call.args[-1], "positional") if call.args else (None, "absent")
        return (call.args[index], "positional") if len(call.args) > index else (None, "absent")

    def builder_of(self, value):
        """``(builder, why)``: which builder made ``value`` -- "child_env" or "allowlist" --, or None and why not."""
        if not isinstance(value, ast.Call):
            return None, "what no builder made"
        func = value.func
        name = _called_name(func)
        if name == "child_env":
            if not (_CHILD_ENV in _norm(self.resolve(func)) or (self.rel == _SECRETS_MODULE
                                                                and isinstance(func, ast.Name))):
                return None, "a child_env that is not opti_oignon.process_secrets'"
            through = func if isinstance(func, ast.Name) else func.value
            while isinstance(through, ast.Attribute):
                through = through.value
            if isinstance(through, ast.Name) and self.rel != _SECRETS_MODULE and not self.sole_import(through.id):
                return None, f"{through.id}, which this module binds to something else as well"
            return "child_env", None
        if name in BUILDERS:
            if isinstance(func, ast.Name) and BUILDERS[name] == self.rel and name in self.module_functions \
                    and self.bound.get(name) == ["def"]:
                return "allowlist", None
            return None, f"{name}, not the one {BUILDERS[name]} defines once"
        return None, "what no builder made"


# ---------------------------------------------------------------------------
# What a start is handed.
# ---------------------------------------------------------------------------
def _environ(module, node):
    return bool(_norm(module.resolve(node)) & _ENVIRON)


def _derived(module, node, seen=()):
    """Whether anything in ``node`` reads the process's environment: os.environ itself, or a name its function binds
    to an expression that does -- ``env = dict(os.environ)``, then ``env[...]``."""
    for sub in ast.walk(node):
        if not isinstance(sub, (ast.Name, ast.Attribute)):
            continue
        if _environ(module, sub):
            return True
        owner = module.owner_of(sub) if isinstance(sub, ast.Name) and sub.id not in seen else None
        if owner is not None:
            for binding in module.bindings(owner, sub.id):
                value = getattr(binding, "value", None)
                if isinstance(binding, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and value is not None \
                        and _derived(module, value, seen + (sub.id,)):
                    return True
    return False


def _copy_reason(module, value, owner, seen=()):
    """Why ``value`` is a copy of the process's environment, or None."""
    if isinstance(value, ast.Constant) and value.value is None:
        return "None, the whole environment"
    if _environ(module, value):
        return "os.environ itself"
    if isinstance(value, ast.Name):
        if owner is not None and value.id not in seen:
            for binding in module.bindings(owner, value.id):
                bound = getattr(binding, "value", None)
                if isinstance(binding, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and bound is not None:
                    why = _copy_reason(module, bound, owner, seen + (value.id,))
                    if why:
                        return f"{value.id}, bound to {why}"
        return None
    if _derived(module, value):
        return f"an expression that reads os.environ (line {value.lineno})"
    return None


def _withheld(names, key):
    return key in names.secrets or key in names.hazards or key.startswith(names.prefix)


def _reader(module, call):
    """Whether ``call`` reads a secret: one of the holder's readers, however it is bound, or the encryption
    module's own ``_secret``."""
    if module.rel == _ENCRYPTION_MODULE and isinstance(call.func, ast.Name) and call.func.id == "_secret":
        return True
    return bool(_norm(module.resolve(call.func)) & _SECRET_READERS)


def _values_of(module, name, owner):
    """The expressions ``name`` may be bound to where ``owner`` (a function, or None for the module) reads it: those
    of the first function, from ``owner`` outwards, that binds it -- a ``nonlocal`` read through to the one it
    names --, else the module's, anywhere at its top level and in every function that declares it ``global``."""
    values, scope = [], owner
    while scope is not None:
        bindings = module.bindings(scope, name)
        declared = [binding for binding in bindings if isinstance(binding, (ast.Global, ast.Nonlocal))]
        values += [getattr(binding, "value", None) for binding in bindings if binding not in declared]
        if any(isinstance(binding, ast.Global) for binding in declared):
            break
        args = scope.args
        params = {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]
                  if arg is not None}
        if (bindings and not declared) or name in params:
            return values
        scope = module.owner.get(id(scope))
    values += [getattr(binding, "value", None) for binding in module.bindings(None, name)]
    for declaration in module.globals_named.get(name, ()):
        values += [getattr(binding, "value", None) for binding in module.bindings(module.owner_of(declaration), name)
                   if not isinstance(binding, ast.Global)]
    return values


def _reads_secret(module, node, owner):
    """Whether ``node`` holds a read of a secret, or a name bound to an expression that does where it is read: a
    secret's value, whatever name it is put under."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and _reader(module, sub):
            return True
        if isinstance(sub, ast.Name) and any(
                value is not None and any(isinstance(call, ast.Call) and _reader(module, call)
                                          for call in ast.walk(value))
                for value in _values_of(module, sub.id, owner)):
            return True
    return False


def _may_call(module, reference, home, method):
    """Whether ``reference`` -- a name or an attribute spelled like a function of the module ``home`` -- may stand
    for it: a method only through an attribute, a function only through a name bound to nothing else; an
    attribute whose receiver the census resolves stands for it only when that receiver is of ``home``."""
    if isinstance(reference, ast.Name):
        return not method and all(q.startswith(home + ".") for q in module.resolve(reference))
    receiver = reference.value
    quals = module.resolve(receiver)
    if quals:
        return any(q == home or q.startswith(home + ".") for q in quals)
    # A receiver bound by its import alone is a module this census does not follow, not an instance.
    return not (isinstance(receiver, ast.Name) and module.sole_import(receiver.id))


def _mapping_reason(module, node, names, owner, what):
    """Why the mapping ``node`` cannot go into a built environment, or None: its keys must be literals the census
    reads, never a secret's name nor a shell variable, and nothing in it may read the process's environment or a
    secret."""
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if key is None:
                return f"{what} spreads a mapping this census cannot read"
            literal = _literal_key(key)
            if literal is None:
                return f"{what} sets a key this census cannot read"
            if _withheld(names, literal):
                return f"{what} sets {literal}, which a child's environment never carries"
            if _derived(module, value):
                return f"{what} sets {literal} from os.environ"
            if _reads_secret(module, value, owner):
                return f"{what} sets {literal} to a secret's value, which a child never holds"
        return None
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    if isinstance(node, ast.BoolOp):
        # ``extra or {}``: each operand is merged in turn.
        for value in node.values:
            why = _mapping_reason(module, value, names, owner, what)
            if why:
                return why
        return None
    if isinstance(node, ast.Name) and owner is not None and node.id in module.parameters(owner) \
            and not module.bindings(owner, node.id):
        return _parameter_callers(module, owner, node.id, names, what)
    if _derived(module, node):
        return f"{what} merges os.environ"
    return f"{what} merges what this census cannot read"


def _default_of(function, parameter):
    """The default ``parameter`` of ``function`` takes, or None."""
    args = function.args
    positional = args.posonlyargs + args.args
    for arg, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
        if arg.arg == parameter:
            return default
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if arg.arg == parameter:
            return default
    return None


def _argument_reason(module, call, bound_method, parameter, params, keyword_only, names, what):
    """Why the mapping ``call`` hands a merger's ``parameter`` cannot go into a built environment, or None."""
    argument = next((keyword.value for keyword in call.keywords if keyword.arg == parameter), None)
    if argument is None and any(keyword.arg is None for keyword in call.keywords):
        return f"{what}: the call at line {call.lineno} hands {parameter} in **kwargs"
    if argument is None and not keyword_only and parameter in params:
        index = params.index(parameter) - (1 if bound_method and params[:1] in (["self"], ["cls"]) else 0)
        if 0 <= index < len(call.args):
            argument = call.args[index]
    if argument is None:
        return None
    return _mapping_reason(module, argument, names, module.owner_of(call), what)


def _parameter_callers(module, function, parameter, names, what):
    """A parameter merged into a built environment (a merger): its default, and the mapping every caller hands it
    -- here in its module, in the others by ``_foreign_calls`` --, pass what the census reads, and it is called. A
    merger that hands the parameter on to itself -- a retry -- adds no caller: the reading under way covers them."""
    merging = module.__dict__.setdefault("_merging", set())
    if (id(function), parameter) in merging:
        return None
    merging.add((id(function), parameter))
    try:
        return _parameter_callers_read(module, function, parameter, names, what)
    finally:
        merging.discard((id(function), parameter))


def _parameter_callers_read(module, function, parameter, names, what):
    params = [arg.arg for arg in function.args.posonlyargs + function.args.args]
    keyword_only = parameter in [arg.arg for arg in function.args.kwonlyargs]
    method = params[:1] in (["self"], ["cls"])
    default = _default_of(function, parameter)
    if default is not None:
        why = _mapping_reason(module, default, names, module.owner_of(function), f"{what}, by its default,")
        if why:
            return why
    calls = 0
    for reference in module.loaded.get(function.name, ()):
        if not _may_call(module, reference, module.dotted, method):
            continue
        call = module.parent.get(id(reference))
        if not (isinstance(call, ast.Call) and call.func is reference):
            return f"{what}: {function.name}() is handed on as a value, its callers cannot be followed"
        if not _within(module, call, function):
            # A retry, the merger calling itself, is read for what it hands on, but is no caller.
            calls += 1
        why = _argument_reason(module, call, isinstance(reference, ast.Attribute), parameter, params, keyword_only,
                               names, what)
        if why:
            return why
    module.mergers.append((function.name, parameter, tuple(params), keyword_only, what, calls, function.lineno))
    return None


def _within(module, node, function):
    """Whether ``node`` sits in ``function``, or in a function nested in it."""
    owner = module.owner_of(node)
    while owner is not None:
        if owner is function:
            return True
        owner = module.owner.get(id(owner))
    return False


def _class_of(module, node):
    """The name of the class ``node`` sits in, or None."""
    parent = module.parent.get(id(node))
    while parent is not None and not isinstance(parent, ast.ClassDef):
        parent = module.parent.get(id(parent))
    return parent.name if parent is not None else None


def _env_call(module, call):
    """Whether ``call`` is a direct start of this module, or a call of one of its launchers -- a function called by
    its name, a method through ``self`` or ``cls`` in its own class --: a call whose environment the census reads
    where it is made."""
    if not isinstance(call, ast.Call):
        return False
    if id(call) in module.start_calls:
        return True
    for _parameter, method, cls in module.launcher_params.get(_called_name(call.func), ()):
        if not method:
            if isinstance(call.func, ast.Name) and _may_call(module, call.func, module.dotted, False):
                return True
            continue
        if not isinstance(call.func, ast.Attribute):
            continue
        receiver = call.func.value
        # Through self or cls in its own class, no other class of the module defining a method of that name: a
        # subclass, an instance or a class of another name could stand for another method.
        if isinstance(receiver, ast.Name) and receiver.id in ("self", "cls") and _class_of(module, call) == cls \
                and module.methods.get(call.func.attr) == {cls}:
            return True
    return False


def _in_signature(module, node, function):
    """Whether ``node`` sits in the signature of ``function`` -- a default, an annotation --, which the enclosing
    scope evaluates."""
    parent = module.parent.get(id(node))
    while parent is not None and parent is not function:
        if isinstance(parent, ast.arguments):
            return True
        parent = module.parent.get(id(parent))
    return False


def _nested_in(module, node, owner):
    """The function ``node`` sits in when that function is nested in ``owner``, or None."""
    inner = module.owner_of(node)
    function = inner
    while inner is not None and inner is not owner:
        inner = module.owner.get(id(inner))
    return function if inner is owner and function is not owner else None


# What a built environment may be handed to without a finding: builtins that read it.
_READING_BUILTINS = frozenset({"len", "sorted", "bool", "list", "tuple", "set", "frozenset"})


def _uses(module, name, owner, names, start_call):
    """Why the built environment ``name`` of ``owner`` is used past what the census reads, or None."""
    for reference in module.loaded.get(name, ()):
        if not isinstance(reference, ast.Name):
            continue
        if module.owner_of(reference) is not owner:
            inner = _nested_in(module, reference, owner)
            if inner is None:
                continue
            if not _in_signature(module, reference, inner) and (
                    name in module.parameters(inner) or module.bindings(inner, name)):
                continue
            return (f"{name} used in {inner.name}(), a function nested in {owner.name}() (line {reference.lineno}), "
                    "where this census cannot follow it")
        parent = module.parent.get(id(reference))
        if isinstance(parent, ast.keyword) and parent.arg == "env":
            call = module.parent.get(id(parent))
            if call is start_call or _env_call(module, call):
                continue
            return (f"{name} handed as env= to {_called_name(call.func) or 'a call'}() (line {parent.lineno}), "
                    "which is no start this census reads")
        if isinstance(parent, ast.Call) and reference in parent.args and (
                parent is start_call or _env_call(module, parent)):
            continue
        if isinstance(parent, ast.Compare):
            continue
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) \
                and parent.func.id in _READING_BUILTINS and parent.func.id not in module.bound \
                and reference in parent.args:
            continue
        if isinstance(parent, ast.Subscript) and parent.value is reference:
            if not isinstance(parent.ctx, ast.Store):
                continue
            key = _literal_key(parent.slice)
            if key is None:
                return f"{name}[...] set with a key this census cannot read (line {parent.lineno})"
            if _withheld(names, key):
                return f"{name}[{key!r}] set (line {parent.lineno}): a child's environment never carries it"
            assignment = module.parent.get(id(parent))
            if isinstance(assignment, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and assignment.value is not None:
                if _derived(module, assignment.value):
                    return f"{name}[{key!r}] set from os.environ (line {parent.lineno})"
                if _reads_secret(module, assignment.value, owner):
                    return f"{name}[{key!r}] set to a secret's value (line {parent.lineno}), which a child never holds"
            continue
        if isinstance(parent, ast.Attribute) and parent.value is reference:
            call = module.parent.get(id(parent))
            if not (isinstance(call, ast.Call) and call.func is parent):
                return f"{name}.{parent.attr} used where this census cannot follow it (line {parent.lineno})"
            if parent.attr in _BUILT_READS:
                continue
            if parent.attr == "update":
                what = f"{name}.update() (line {call.lineno})"
                for argument in call.args:
                    why = _mapping_reason(module, argument, names, owner, what)
                    if why:
                        return why
                for keyword in call.keywords:
                    if keyword.arg is None:
                        return f"{what} spreads a mapping this census cannot read"
                    if _withheld(names, keyword.arg):
                        return f"{what} sets {keyword.arg}, which a child's environment never carries"
                    if _derived(module, keyword.value):
                        return f"{what} sets {keyword.arg} from os.environ"
                    if _reads_secret(module, keyword.value, owner):
                        return f"{what} sets {keyword.arg} to a secret's value, which a child never holds"
                continue
            if parent.attr == "setdefault" and call.args:
                what = f"{name}.setdefault() (line {call.lineno})"
                key = _literal_key(call.args[0])
                if key is None:
                    return f"{what} sets a key this census cannot read"
                if _withheld(names, key):
                    return f"{what} sets {key}, which a child's environment never carries"
                for argument in call.args[1:]:
                    if _derived(module, argument):
                        return f"{what} sets {key} from os.environ"
                    if _reads_secret(module, argument, owner):
                        return f"{what} sets {key} to a secret's value, which a child never holds"
                continue
            return (f"{name}.{parent.attr}() (line {call.lineno}): the built environment used where this census "
                    f"cannot follow it")
        return f"{name} handed where this census cannot follow it (line {reference.lineno})"
    return None


def _extra_reason(module, call, names, owner):
    """Why the extra environment a ``child_env()`` call is handed cannot go into a child's, or None."""
    what = f"child_env() (line {call.lineno})"
    for argument in call.args:
        why = _mapping_reason(module, argument, names, owner, what)
        if why:
            return why
    for keyword in call.keywords:
        why = _mapping_reason(module, keyword.value, names, owner, what) if keyword.arg in (None, "extra") \
            else f"{what} is handed {keyword.arg}=, which it does not take"
        if why:
            return why
    return None


def _built(module, value, builder, names, owner):
    """``(kind, why, builder)`` for a value a builder made: child_env's extra environment read too."""
    if builder == "child_env" and isinstance(value, ast.Call) and (value.args or value.keywords):
        why = _extra_reason(module, value, names, owner)
        if why:
            return ("copy" if "os.environ" in why else "unbuilt"), why, None
    return None, None, builder


def _verdict(module, value, owner, names, seen=(), start_call=None):
    """``(kind, why, builder)``: kind None for a built environment, "copy" for a copy of the process's or one
    merged with it, "unbuilt" otherwise; ``why`` says what it is, ``builder`` what built it."""
    why = _copy_reason(module, value, owner)
    if why:
        return "copy", why, None
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr):
        # A built environment with a mapping merged into it: ``child_env() | {...}``.
        kind, why, builder = _verdict(module, value.left, owner, names, seen, start_call)
        if kind is not None:
            return kind, why, builder
        why = _mapping_reason(module, value.right, names, owner, f"the mapping merged at line {value.lineno}")
        if why:
            return ("copy" if "os.environ" in why else "unbuilt"), why, None
        return None, None, builder
    builder, why = module.builder_of(value)
    if builder:
        return _built(module, value, builder, names, owner)
    if not isinstance(value, ast.Name):
        return "unbuilt", why, None
    if owner is None:
        return "unbuilt", f"{value.id}, a name at module level", None
    bindings = module.bindings(owner, value.id)
    if value.id in module.parameters(owner) and not bindings:
        kind, why, builder = _callers_fill(module, owner, value.id, names, seen)
        if kind is None:
            used = _uses(module, value.id, owner, names, start_call)
            if used:
                return ("copy" if "os.environ" in used else "unbuilt"), used, None
        return kind, why, builder
    if not bindings:
        return "unbuilt", f"{value.id}, bound nowhere in {owner.name}(): a name of the module, or one received", None
    binding = bindings[0] if len(bindings) == 1 else None
    if binding is None or not isinstance(binding, (ast.Assign, ast.AnnAssign)) \
            or (isinstance(binding, ast.Assign) and len(binding.targets) != 1) \
            or not isinstance(binding.targets[0] if isinstance(binding, ast.Assign) else binding.target, ast.Name):
        return "unbuilt", f"{value.id}, bound {len(bindings)} time(s) in {owner.name}()", None
    builder, why = module.builder_of(binding.value)
    if not builder:
        return "unbuilt", f"{value.id}, bound to {why}", None
    kind, why, builder = _built(module, binding.value, builder, names, owner)
    if kind is not None:
        return kind, why, builder
    used = _uses(module, value.id, owner, names, start_call)
    if used:
        return ("copy" if "os.environ" in used else "unbuilt"), used, None
    return None, None, builder


def _callers_fill(module, launcher, parameter, names, seen):
    """A launcher's parameter: every caller in its module fills it with a built environment. The launcher is
    recorded, so that its callers in other modules are held to it too."""
    name = launcher.name
    if name in seen:
        return "unbuilt", f"{name}() hands its {parameter} on to itself", None
    params = [arg.arg for arg in launcher.args.posonlyargs + launcher.args.args]
    keyword_only = parameter in [arg.arg for arg in launcher.args.kwonlyargs]
    method = params[:1] in (["self"], ["cls"])
    if not seen:
        module.launchers.append((name, parameter, tuple(params), keyword_only))
    calls, values = [], 0
    for reference in module.loaded.get(name, ()):
        if not _may_call(module, reference, module.dotted, method):
            continue
        parent = module.parent.get(id(reference))
        if isinstance(parent, ast.Call) and parent.func is reference:
            calls.append((parent, isinstance(reference, ast.Attribute)))
        else:
            values += 1
    if values:
        return "unbuilt", f"{name}() is handed on as a value: its callers cannot be followed", None
    if not calls:
        return "unbuilt", f"{name}() takes {parameter} and is called nowhere in its module", None
    for call, bound_method in sorted(calls, key=lambda pair: pair[0].lineno):
        kind, why = _caller_verdict(module, call, bound_method, parameter, params, keyword_only, names, name,
                                    seen + (name,))
        if kind is not None:
            return kind, why, None
    return None, None, "launcher"


def _caller_verdict(module, call, bound_method, parameter, params, keyword_only, names, name, seen):
    argument = next((keyword.value for keyword in call.keywords if keyword.arg == parameter), None)
    if argument is None and any(keyword.arg is None for keyword in call.keywords):
        return "unbuilt", f"the call of {name}() at line {call.lineno} hands {parameter} in **kwargs"
    if argument is None and not keyword_only and parameter in params:
        index = params.index(parameter) - (1 if bound_method and params[:1] in (["self"], ["cls"]) else 0)
        if 0 <= index < len(call.args):
            argument = call.args[index]
    if argument is None:
        return "unbuilt", f"the call of {name}() at line {call.lineno} leaves {parameter} to its default"
    kind, why, _builder = _verdict(module, argument, module.owner_of(call), names, seen, start_call=call)
    if kind is not None:
        return kind, f"the call of {name}() at line {call.lineno} hands {parameter} {why}"
    return None, None


def _verdict_of(module, start):
    """``(start, kind, why, builder)`` for a direct start."""
    if start.qual.endswith(_AS_VALUE):
        return start, "unbuilt", "nothing: it is taken as a value, and the environment it is handed cannot be read", None
    value, how = module.env_argument(start)
    if value is None:
        why = ("no env=: the child is handed the whole environment" if how == "absent"
               else f"an environment hidden in {how}, which this census cannot read")
        return start, "unbuilt", why, None
    call = module.nodes[start]
    kind, why, builder = _verdict(module, value, module.owner_of(call), module.names_of or _NO_NAMES,
                                  start_call=call)
    return start, kind, why, builder


# ---------------------------------------------------------------------------
# A secret's name, outside the holder: a closed rule.
# ---------------------------------------------------------------------------
_HOLDER_NAMES = _HOLDER + ".SECRET_NAMES"
# What the constants table holds for the holder's own names, and every name they are handed on under.
_HOLDER_LIST = "\0SECRET_NAMES"
_A_VALUE = "a secret's value"
_ENCRYPTION_MODULE = "opti_oignon/encryption.py"
_ANY_SECRET = "a secret's name"


def _dotted(rel):
    dotted = rel[:-3].replace("/", ".")
    return dotted[: -len(".__init__")] if dotted.endswith(".__init__") else dotted


def _top_bound(statement, imports=True):
    """``[(identifier, line)]``: what a module-level statement binds, the control flow at the top level read
    through, the bodies of functions and classes not; its imports too unless ``imports`` is false."""
    out, stack = [], [statement]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append((node.name, node.lineno))
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if imports:
                out.extend(((alias.asname or alias.name.split(".")[0]), node.lineno) for alias in node.names)
            continue
        targets = node.targets if isinstance(node, ast.Assign) else \
            [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)) else \
            [item.optional_vars for item in node.items if item.optional_vars is not None] \
            if isinstance(node, (ast.With, ast.AsyncWith)) else []
        for target in targets:
            out.extend((ident, node.lineno) for ident in _target_ids(target))
        for field in ("body", "orelse", "finalbody", "handlers"):
            stack.extend(getattr(node, field, None) or ())
    return out


def _definitions(tree, secrets):
    """``({identifier: name}, {id of each literal they bind}, {identifier: line it is bound again})``: a module's
    constants that name a secret -- ``IDENT = "<a secret's name>"`` at the top level, or a tuple assignment there
    pairing names with such literals."""
    named, literals, lines = {}, set(), {}
    for statement in tree.body:
        for ident, line in _top_bound(statement):
            lines.setdefault(ident, []).append(line)
        if isinstance(statement, (ast.Assign, ast.AnnAssign)) and statement.value is not None:
            value = statement.value
            for target in statement.targets if isinstance(statement, ast.Assign) else [statement.target]:
                pairs = list(zip(target.elts, value.elts)) if isinstance(target, (ast.Tuple, ast.List)) \
                    and isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(value.elts) \
                    else [(target, value)]
                for one, it in pairs:
                    if isinstance(one, ast.Name) and isinstance(it, ast.Constant) and it.value in secrets:
                        named[one.id] = it.value
                        literals.add(id(it))
    again = {ident: sorted(lines[ident])[1] for ident in named if len(lines.get(ident, ())) > 1}
    return named, literals, again


def _binds_in(module, scope, name):
    """Whether ``scope`` (a function, or None for the module) binds ``name`` otherwise than by an import: an
    assignment, a parameter, a function or a class it defines."""
    if (id(scope), name) in module.defined_in:
        return True
    if scope is None:
        return any(ident == name for statement in module.tree.body
                   for ident, _line in _top_bound(statement, imports=False))
    args = scope.args
    params = {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg] if arg}
    return name in params or bool(module.bindings(scope, name))


def _scope_quals(module, name, owner):
    """What ``name`` stands for where ``owner`` (a function, or None for the module) reads it, when imports alone bind
    it in the scope that decides -- the function's own, else each function it is nested in, else the module's --:
    the dotted names those import statements bring in; None when that scope binds it otherwise, or nothing binds it."""
    scope = owner
    while True:
        imports = [node for node in module.imports if module.owner_of(node) is scope
                   and any((alias.asname or alias.name.split(".")[0]) == name for alias in node.names)]
        others = _binds_in(module, scope, name)
        if imports or others:
            if others:
                return None
            quals = set()
            for node in imports:
                for alias in node.names:
                    if (alias.asname or alias.name.split(".")[0]) != name:
                        continue
                    if isinstance(node, ast.Import):
                        quals.add(alias.name if alias.asname else alias.name.split(".")[0])
                    elif module._absolute(node):
                        quals.add(f"{module._absolute(node)}.{alias.name}")
            return quals
        if scope is None:
            return None
        scope = module.owner.get(id(scope))


def _class_binds(module, node, name):
    """Whether ``node`` sits in a class body -- not in a function of it -- that binds ``name``."""
    parent = module.parent.get(id(node))
    while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        if isinstance(parent, ast.ClassDef):
            return any(ident == name for statement in parent.body for ident, _line in _top_bound(statement))
        parent = module.parent.get(id(parent))
    return False


def _is_reader_call(module, call):
    """Whether ``call`` is one of the holder's readers: its name, or the module it is read from, bound where it is
    called by imports of the holder alone -- or, in the encryption module, its own ``_secret``, defined once."""
    func = call.func
    if module.rel == _ENCRYPTION_MODULE and isinstance(func, ast.Name) and func.id == "_secret":
        return module.bound.get("_secret") == ["def"]
    owner = module.owner_of(call)
    attrs, root = [], func
    while isinstance(root, ast.Attribute):
        attrs.insert(0, root.attr)
        root = root.value
    if not isinstance(root, ast.Name):
        return False
    if root.id in module.aside_bound or _class_binds(module, call, root.id):
        # Bound where the scope chain does not read: the name may stand for something else where it is called.
        return False
    quals = _scope_quals(module, root.id, owner)
    tail = "".join("." + attr for attr in attrs)
    return bool(quals) and {qual + tail for qual in quals} <= _SECRET_READERS


def _occurrence(module, node, names, constants, own):
    """The secret's name ``node`` spells -- a literal, a constant of this module or of another, an item of the
    holder's names --, or None."""
    if isinstance(node, ast.Constant):
        value = node.value.decode("utf-8", "replace") if isinstance(node.value, bytes) else node.value
        return value if isinstance(value, str) and value in names.secrets else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
        # A constant read through getattr, its identifier a string.
        found = next((constants[q] for q in sorted(module.resolve(node)) if q in constants), None)
        return _ANY_SECRET if found == _HOLDER_LIST else found
    if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
        if isinstance(node, ast.Name) and node.id in own:
            return own[node.id]
        quals = module.resolve(node)
        found = next((constants[q] for q in sorted(quals) if q in constants), None)
        return _ANY_SECRET if found == _HOLDER_LIST else found
    if isinstance(node, ast.Subscript) and any(constants.get(qual) == _HOLDER_LIST
                                               for qual in module.resolve(node.value)):
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, int) and names.ordered \
                and -len(names.ordered) <= index.value < len(names.ordered):
            return names.ordered[index.value]
        return _ANY_SECRET
    return None


def _allowed(module, node):
    """Whether a secret's name sits where it may: the first argument of one of the holder's readers (by position or
    as ``name=``), or a sentence it is interpolated into (an f-string with words of its own). A mapping's value
    and a log call's argument are no exceptions: a mapping is read back however it is handed, and a log call's
    receiver may be bound to anything."""
    parent = module.parent.get(id(node))
    if isinstance(parent, ast.Call) and parent.args and parent.args[0] is node and _is_reader_call(module, parent):
        return True
    if isinstance(parent, ast.keyword) and parent.arg == "name":
        call = module.parent.get(id(parent))
        return isinstance(call, ast.Call) and _is_reader_call(module, call)
    if isinstance(parent, ast.FormattedValue):
        joined = module.parent.get(id(parent))
        return isinstance(joined, ast.JoinedStr) and any(
            isinstance(part, ast.Constant) and str(part.value).strip() for part in joined.values)
    return False


def _context(module, node):
    """``(verb, how)``: what a secret's name is used for where it may not be."""
    child, parent = node, module.parent.get(id(node))
    while isinstance(parent, (ast.Tuple, ast.List, ast.Set, ast.Starred, ast.Dict)):
        child, parent = parent, module.parent.get(id(parent))
    if isinstance(parent, ast.keyword):
        child, parent = parent, module.parent.get(id(parent))
    if isinstance(parent, ast.Call) and child is not parent.func:
        func = parent.func
        if isinstance(func, ast.Attribute) and _derived(module, func.value):
            verb = "reads" if func.attr in _READS else "writes" if func.attr in _WRITES | {"update"} else "uses"
            return verb, f"os.environ.{func.attr}"
        reader = next((q for q in sorted(_norm(module.resolve(func))) if q in _ENV_FUNCTIONS), None)
        if reader:
            return _ENV_FUNCTIONS[reader], reader
        return "hands", f"to {_called_name(func) or 'a call'}(), which may read the environment with it"
    if isinstance(parent, ast.Subscript) and child is parent.slice:
        if _derived(module, parent.value):
            return ("reads" if isinstance(parent.ctx, ast.Load) else "writes"), "os.environ[...]"
        return "uses", "as a key"
    if isinstance(parent, ast.Compare):
        if child is parent.left and any(isinstance(op, (ast.In, ast.NotIn)) and _derived(module, comparator)
                                        for op, comparator in zip(parent.ops, parent.comparators)):
            return "reads", "in os.environ"
        return "compares", "a value with a secret's name"
    if isinstance(parent, ast.MatchValue):
        return "compares", "a value with a secret's name, in a match"
    if isinstance(parent, ast.arguments):
        return "defaults", "a parameter to a secret's name"
    if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr, ast.AugAssign)):
        return "binds", "to a name, where only the one constant that names it may"
    return "uses", "where only the holder's readers may"


def _may_name(module, names, idents):
    """Whether a module can name a secret at all, read from its parsed syntax -- escapes and implicit concatenation
    resolved --: every spelling ``_occurrence`` reads is a string that is a secret's name, a keyword named like one,
    or a name, an attribute, an imported name or a string that is the identifier of a constant naming one, or
    ``SECRET_NAMES``. A module that holds none of them names no secret."""
    if module.strings & names.secrets or module.keyword_names & names.secrets:
        return True
    return bool(set(idents) & (module.identifiers | module.imported_names | module.strings))


def _reexports(module):
    """``[(module, identifier, alias)]``: what a module's top level imports from a module of the package, which it
    may hand on to another under the alias."""
    out, stack = [], list(module.tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.ImportFrom):
            source = module._absolute(node)
            if source and source.split(".")[0] == _PACKAGE_DIR:
                out.extend((source, alias.name, alias.asname or alias.name) for alias in node.names)
        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for field in ("body", "orelse", "finalbody", "handlers"):
                stack.extend(getattr(node, field, None) or ())
    return out


def _assigned(module, target):
    """The expression assigned to ``target``, a target of an assignment or an element of a tuple of them, or None."""
    node, parent = target, module.parent.get(id(target))
    path = []
    while isinstance(parent, (ast.Tuple, ast.List)):
        path.insert(0, next(index for index, element in enumerate(parent.elts) if element is node))
        node, parent = parent, module.parent.get(id(parent))
    if not isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign)) or parent.value is None:
        return None
    value = parent.value
    for index in path:
        if not isinstance(value, (ast.Tuple, ast.List)) or index >= len(value.elts):
            return parent.value
        value = value.elts[index]
    return value


def _secret_value_puts(module):
    """``[(line, verb, name, how)]``: each secret's value put into this process's environment, as a variable's value
    or its name -- a key of ``os.environ`` set (a tuple's element too, ``+=`` too), ``|=``, an update, a setdefault,
    an ``os.putenv`` --, the value a read of a secret or a name bound to one where it is read."""
    out = []
    for node in module.subscripts:
        if isinstance(node.ctx, ast.Store) and _environ(module, node.value):
            owner, value = module.owner_of(node), _assigned(module, node)
            if _reads_secret(module, node.slice, owner) or (value is not None
                                                             and _reads_secret(module, value, owner)):
                out.append((node.lineno, "puts", _A_VALUE, "into os.environ[...]"))
    for node in module.augassigns:
        if isinstance(node.op, ast.BitOr) and _environ(module, node.target) \
                and _reads_secret(module, node.value, module.owner_of(node)):
            out.append((node.lineno, "puts", _A_VALUE, "into os.environ |="))
    for call in module.calls:
        func, owner = call.func, module.owner_of(call)
        values = list(call.args) + [keyword.value for keyword in call.keywords]
        if isinstance(func, ast.Attribute) and func.attr in ("update", "setdefault", "__setitem__", "__ior__") \
                and _environ(module, func.value):
            if any(_reads_secret(module, value, owner) for value in values):
                out.append((call.lineno, "puts", _A_VALUE, f"into os.environ.{func.attr}"))
        elif "os.putenv" in _norm(module.resolve(func)) and any(_reads_secret(module, value, owner)
                                                               for value in values):
            out.append((call.lineno, "puts", _A_VALUE, "into os.putenv"))
    return out


def _secret_uses(module, names, constants):
    """``[(line, verb, name, how)]``: each place a secret's name appears but the two it may -- the first argument of
    one of the holder's readers, or the one module-level constant that names it, bound once."""
    own, defining, again = _definitions(module.tree, names.secrets)
    out = [(line, "binds", own[ident], f"{ident} again, where a constant that names a secret is bound once")
           for ident, line in again.items()]
    inside = set()
    for node in ast.walk(module.tree):
        if isinstance(node, ast.keyword) and node.arg in names.secrets:
            verb, how = _context(module, node)
            out.append((node.lineno, verb, node.arg, how + ", as a keyword"))
            continue
        if id(node) in defining or id(node) in inside:
            continue
        name = _occurrence(module, node, names, constants, own)
        if name is None:
            continue
        if isinstance(node, ast.Subscript):
            inside.add(id(node.value))
        if not _allowed(module, node):
            verb, how = _context(module, node)
            out.append((node.lineno, verb, name, how))
    return sorted(set(out))


# ---------------------------------------------------------------------------
# The estate.
# ---------------------------------------------------------------------------
_EGRESS = None


def _egress():
    """The egress census, loaded from its file beside this one: its estate reader, its sinks, its suites."""
    global _EGRESS
    if _EGRESS is None:
        path = Path(__file__).resolve().with_name("egress_census_guard.py")
        spec = importlib.util.spec_from_file_location("_child_env_census_egress", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EGRESS = module
    return _EGRESS


def read_estate(root):
    """The package's modules under ``root``, read as the egress census reads them: what could not be read is
    named in ``unread``, never read as empty, and no directory named ``data`` is opened."""
    return _egress().read_estate(root)


def _plugin(rel):
    parts = rel.split("/")
    return len(parts) >= 4 and parts[0] == _PACKAGE_DIR and parts[1] == "plugins"


class SecretNamesUnread(Exception):
    """``process_secrets.py`` does not write the names this census reads."""


def secret_names(estate):
    """``Names(secrets, hazards, prefix)``, as ``opti_oignon/process_secrets.py`` writes them in ``SECRET_NAMES``,
    ``SHELL_HAZARDS`` and ``FUNCTION_PREFIX``."""
    text = estate.modules.get(_SECRETS_MODULE)
    if text is None:
        raise SecretNamesUnread(f"{_SECRETS_MODULE} is not in the estate")
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as exc:
        raise SecretNamesUnread(f"{_SECRETS_MODULE} does not parse ({type(exc).__name__})") from None
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    secrets, hazards, prefix = found.get("SECRET_NAMES"), found.get("SHELL_HAZARDS"), found.get("FUNCTION_PREFIX")
    if not (isinstance(secrets, tuple) and secrets and all(isinstance(name, str) and name for name in secrets)):
        raise SecretNamesUnread(f"{_SECRETS_MODULE} writes no SECRET_NAMES tuple of names")
    if not (isinstance(hazards, tuple) and all(isinstance(name, str) and name for name in hazards)):
        raise SecretNamesUnread(f"{_SECRETS_MODULE} writes no SHELL_HAZARDS tuple of names")
    if not (isinstance(prefix, str) and prefix):
        raise SecretNamesUnread(f"{_SECRETS_MODULE} writes no FUNCTION_PREFIX")
    return Names(frozenset(secrets), frozenset(hazards), prefix, tuple(secrets))


def _constants(census):
    """``{"<module>.<identifier>": name}``: every module-level constant that names a secret, as ``_definitions``
    reads it in the parsed module, and every name a module's top level imports it under, however many modules it
    is handed on through. A constant bound to another constant is not one -- it is a finding where it is bound."""
    out = {_HOLDER_NAMES: _HOLDER_LIST}
    for module in census.values():
        out.update({f"{module.dotted}.{ident}": name for ident, name in module.definitions.items()})
    grown = True
    while grown:
        grown = False
        for module in census.values():
            for source, ident, alias in module.reexports:
                if ident == "*":
                    handed = {qual[len(source) + 1:]: name for qual, name in out.items()
                              if qual.startswith(source + ".") and "." not in qual[len(source) + 1:]}
                else:
                    handed = {alias: out[f"{source}.{ident}"]} if f"{source}.{ident}" in out else {}
                for handed_as, name in handed.items():
                    key = f"{module.dotted}.{handed_as}"
                    if key not in out:
                        out[key] = name
                        grown = True
    return out


def _name_uses(estate, census, names):
    """The name rule over every module that can name a secret, once every constant is known."""
    constants = _constants(census)
    idents = {qual.rsplit(".", 1)[1] for qual in constants}
    for rel, module in census.items():
        if module.unparsed or rel == _SECRETS_MODULE or rel in EXEMPT or not _may_name(module, names, idents):
            continue
        reader = module if module.tree is not None else _Module(rel, estate.modules[rel], names, keep=True)
        module.secret_uses = sorted(set(module.secret_uses) | set(_secret_uses(reader, names, constants)))
    for module in census.values():
        if not module.unparsed:
            module.release()


def take_census(estate):
    """module -> its ``_Module``, for every module of the estate but the bundled plugins."""
    cached = estate.cache.get("child_env_census")
    if cached is None:
        try:
            names = secret_names(estate)
        except SecretNamesUnread as exc:
            estate.cache["child_env_names"] = exc
            names = None
        else:
            estate.cache["child_env_names"] = names
        cached = {rel: _Module(rel, text, names) for rel, text in estate.modules.items() if not _plugin(rel)}
        if names is not None:
            _name_uses(estate, cached, names)
        estate.cache["child_env_census"] = cached
    return cached


# ---------------------------------------------------------------------------
# The questions.
# ---------------------------------------------------------------------------
def _checked(census):
    """The modules the rules hold: every one but the exempt."""
    return {rel: module for rel, module in census.items() if rel not in EXEMPT}


def _imports_module(module, dotted):
    """Whether ``module`` imports the module ``dotted``, or a name of it."""
    package, _, leaf = dotted.rpartition(".")
    for node in module.imports:
        if isinstance(node, ast.Import):
            if any(alias.name == dotted or alias.name.startswith(dotted + ".") for alias in node.names):
                return True
            continue
        source = module._absolute(node)
        if source == dotted or (source or "").startswith(dotted + ".") \
                or (source == package and any(alias.name in (leaf, "*") for alias in node.names)):
            return True
    return False


def _handed_on(census, dotted, name):
    """``{(module, name)}``: a function of the module ``dotted`` as it may be reached -- its module and its name,
    and each module whose top level imports it, under the name it binds, however many modules it goes through."""
    out, grown = {(dotted, name)}, True
    while grown:
        grown = False
        for module in census.values():
            for source, ident, alias in module.reexports:
                for known_source, known in list(out):
                    if source == known_source and ident in (known, "*"):
                        pair = (module.dotted, known if ident == "*" else alias)
                        if pair not in out:
                            out.add(pair)
                            grown = True
    return out


def _foreign_calls(estate, census):
    """``[(rel, line, why)]``: a launcher or a merger called from another module with an environment, or a mapping,
    not built there, or taken there as a value -- bound, read through ``getattr`` --; and a merger called nowhere in
    the package."""
    names = estate.cache.get("child_env_names")
    names = names if isinstance(names, Names) else _NO_NAMES
    targets = [(rel, "launcher", entry) for rel, module in _checked(census).items() for entry in module.launchers]
    targets += [(rel, "merger", entry) for rel, module in _checked(census).items() for entry in module.mergers]
    out, read = [], {}
    queued = {(home, role, entry[0], entry[1]) for home, role, entry in targets}
    while targets:
        home, role, entry = targets.pop(0)
        name, parameter, params, keyword_only = entry[:4]
        dotted, method = _dotted(home), params[:1] in (("self",), ("cls",))
        calls = entry[5] if role == "merger" else 1
        # A method is reached through its class, which a module may import from one that hands it on.
        sources = {dotted} | {module.dotted for module in census.values()
                              if any(source == dotted for source, _ident, _alias in module.reexports)}
        handed = _handed_on(census, dotted, name) if not method else {(source, name) for source in sources}
        spellings = {alias for _source, alias in handed}
        leaves = {source.rsplit(".", 1)[-1] for source, _alias in handed}
        for rel, text in sorted(estate.modules.items()):
            # A function is reached from another module only through its module, or one that hands it on, which
            # that module's text names however it imports it; a method through an instance, which no text needs to
            # name.
            if rel == home or rel in EXEMPT or _plugin(rel) or not any(spelling in text for spelling in spellings) \
                    or (not method and not any(leaf in text for leaf in leaves)):
                continue
            if rel not in read:
                read[rel] = _Module(rel, text, names, keep=True)
            module = read[rel]
            # A module that defines a function of that name, or binds an attribute of it, calls its own. A
            # launcher's method is looked for where its module is imported -- an instance of its class made or
            # named there --: a caller that leaves the environment to its default is a finding, and a method name
            # as common as ``run`` would make one of every call. A merger's method is looked for everywhere: its
            # callers are findings only for what they hand it.
            if module.tree is None or (method and role == "launcher"
                                       and not any(_imports_module(module, source) for source in sources)):
                continue
            references = [(source, reference) for source, alias in sorted(handed)
                          if not {"def", "attribute"} & set(module.bound.get(alias, ()))
                          for reference in module.loaded.get(alias, ())]
            for source, reference in references:
                spelled = reference.id if isinstance(reference, ast.Name) else reference.attr
                if not _may_call(module, reference, source, method) or (
                        isinstance(reference, ast.Name) and f"{source}.{spelled}" not in _norm(module.resolve(reference))):
                    continue
                call = module.parent.get(id(reference))
                if not (isinstance(call, ast.Call) and call.func is reference):
                    out.append((rel, reference.lineno, f"{name}() of {home} taken as a value: its callers cannot "
                                                       "be followed"))
                    continue
                calls += 1
                if role == "launcher":
                    kind, why = _caller_verdict(module, call, isinstance(reference, ast.Attribute), parameter,
                                                list(params), keyword_only, names, name, ())
                else:
                    why = _argument_reason(module, call, isinstance(reference, ast.Attribute), parameter,
                                           list(params), keyword_only, names, entry[4])
                    kind = why and "unbuilt"
                if kind is not None and why:
                    out.append((rel, call.lineno, why))
            for call in module.calls:
                if isinstance(call.func, ast.Name) and call.func.id == "getattr" and len(call.args) >= 2 \
                        and isinstance(call.args[1], ast.Constant) and call.args[1].value in spellings:
                    out.append((rel, call.lineno, f"{name}() of {home} read through getattr: its callers cannot be "
                                                  "followed"))
            # A launcher or a merger that hands its parameter on to this one is itself one: its callers are read in
            # turn.
            for found_role, found in [("launcher", entry) for entry in module.launchers] + \
                    [("merger", entry) for entry in module.mergers]:
                if (rel, found_role, found[0], found[1]) not in queued:
                    queued.add((rel, found_role, found[0], found[1]))
                    targets.append((rel, found_role, found))
        if role == "merger" and not calls:
            out.append((home, entry[6], f"{entry[4]}: {name}() takes {parameter} and is called nowhere in the "
                                        "package, so what it merges cannot be read"))
    return sorted(set(out))


def find_unbuilt_starts(estate, census=None):
    """A direct start whose environment no builder made, or that it uses past what the census reads, a launcher
    called from another module included."""
    census = take_census(estate) if census is None else census
    out = [f"{rel}:{start.line}: {start.qual} in {start.function}() is handed {why}"
           for rel, module in sorted(_checked(census).items())
           for start, kind, why, _builder in module.verdicts if kind == "unbuilt"]
    return out + [f"{rel}:{line}: {why}" for rel, line, why in _foreign_calls(estate, census)]


def find_copied_environments(estate, census=None):
    """A start handed a copy of the process's environment, or one merged with it."""
    census = take_census(estate) if census is None else census
    return [f"{rel}:{start.line}: {start.qual} in {start.function}() is handed a copy of the process's "
            f"environment: {why}"
            for rel, module in sorted(_checked(census).items())
            for start, kind, why, _builder in module.verdicts if kind == "copy"]


def _inheriting(census):
    """module -> {function: Counter of the spellings of its inheriting starts}, for the modules the rules hold."""
    out = {}
    for rel, module in _checked(census).items():
        for start in module.starts:
            if start.kind == "inheriting":
                out.setdefault(rel, {}).setdefault(start.function, Counter())[start.qual] += 1
    return out


def find_unlisted_inheriting(estate, census=None):
    """An inheriting start ``INHERITING`` does not list, or more of a spelling in a function than it lists."""
    census = take_census(estate) if census is None else census
    out = []
    for rel, sites in sorted(_inheriting(census).items()):
        listed = INHERITING[rel].sites if rel in INHERITING else {}
        for function, found in sorted(sites.items()):
            allowed = Counter(listed.get(function, ()))
            for qual, n in sorted(found.items()):
                if n > allowed[qual]:
                    out.append(f"{rel}: {n} inheriting start(s) {qual} in {function}, {allowed[qual]} listed")
    return out


def find_stale_inheriting(estate, census=None):
    """A listed inheriting start the census no longer finds, or a listed module that is gone."""
    census = take_census(estate) if census is None else census
    found = _inheriting(census)
    out = []
    for rel, entry in sorted(INHERITING.items()):
        if rel not in census:
            out.append(f"{rel}: listed as inheriting, but it is gone")
            continue
        for function, quals in sorted(entry.sites.items()):
            have = found.get(rel, {}).get(function, Counter())
            for qual, n in sorted(Counter(quals).items()):
                if have[qual] < n:
                    out.append(f"{rel}: {n} inheriting start(s) {qual} listed in {function}, the census finds "
                               f"{have[qual]}; lower the entry")
    return out


def find_inheriting_unproven(estate, census=None):
    """A listed contract that names no test function, or whose suite does not name the module."""
    out = []
    suites = None
    for rel, entry in sorted(INHERITING.items()):
        if not entry.contracts or not str(entry.reason).strip():
            out.append(f"{rel}: an inheriting entry with no reason or no contract")
            continue
        if suites is None:
            suites = _egress()._suite_ids(estate)
        dotted = rel[:-3].replace("/", ".")
        for contract in entry.contracts:
            owners = [suite for suite, (_text, ids) in suites.items() if contract in ids]
            if not owners:
                out.append(f"{rel}: contract {contract} names no test function")
            elif not any(dotted in suites[suite][0] or rel in suites[suite][0] for suite in owners):
                out.append(f"{rel}: the suite of contract {contract} ({', '.join(owners)}) never names {dotted}")
    return out


def find_secret_reads(estate, census=None):
    """A secret's name anywhere but its two places, or a secret's value put into the process's environment,
    anywhere but process_secrets."""
    census = take_census(estate) if census is None else census
    return [f"{rel}:{line}: puts {name} {how}, which every child inherits; a secret's value never goes into the "
            "environment" if verb == "puts" else
            f"{rel}:{line}: {verb} {name} ({how}); the one read is opti_oignon.process_secrets.secret()"
            for rel, module in sorted(_checked(census).items()) if rel != _SECRETS_MODULE
            for line, verb, name, how in module.secret_uses]


def find_twin_gaps(estate, census=None):
    """An egress process sink this census does not count, on its line and in its function."""
    census = take_census(estate) if census is None else census
    out = []
    for rel, sites in sorted(_egress().take_census(estate).items()):
        module = census.get(rel)
        if module is None:
            continue
        have = {(start.line, start.function) for start in module.starts}
        for site in sites:
            if site.kind == "process" and (site.line, site.function or _MODULE_LEVEL) not in have:
                out.append(f"{rel}:{site.line}: a process sink of the egress census in "
                           f"{site.function or _MODULE_LEVEL} that this census does not count")
    return out


def find_stale_exemptions(estate, census=None):
    """An exempt module with no start, or gone."""
    census = take_census(estate) if census is None else census
    return [f"{rel}: exempt, but {'it is gone' if rel not in census else 'it starts no process'}"
            for rel in sorted(EXEMPT) if rel not in census or not census[rel].starts]


_QUESTIONS = (
    ("unbuilt_starts", find_unbuilt_starts),
    ("copied_environments", find_copied_environments),
    ("unlisted_inheriting", find_unlisted_inheriting),
    ("stale_inheriting", find_stale_inheriting),
    ("inheriting_unproven", find_inheriting_unproven),
    ("secret_reads", find_secret_reads),
    ("twin_gaps", find_twin_gaps),
    ("stale_exemptions", find_stale_exemptions),
)


def find_all(estate, census=None):
    """Every question asked, in order: name -> its findings."""
    census = take_census(estate) if census is None else census
    return {name: question(estate, census) for name, question in _QUESTIONS}


def green_line(estate, census):
    verdicts = [verdict for module in _checked(census).values() for verdict in module.verdicts]
    built = {builder: sum(1 for verdict in verdicts if verdict[3] == builder)
             for builder in ("child_env", "allowlist", "launcher")}
    inheriting = sum(sum(found.values()) for sites in _inheriting(census).values() for found in sites.values())
    twins = sum(1 for sites in _egress().take_census(estate).values() for site in sites if site.kind == "process")
    return (
        f"Child environment census OK: {len(estate.modules)} module(s) scanned, {len(verdicts)} direct start(s) "
        f"each handed a built environment ({built['child_env']} by child_env, {built['allowlist']} by an "
        f"untrusted tier's allowlist, {built['launcher']} through a launcher whose callers build it), "
        f"{inheriting} inheriting start(s) listed in {len(INHERITING)} module(s), {len(EXEMPT)} module(s) exempt "
        f"by name, no secret's name outside its two places (the holder's readers, its one constant), {twins} "
        f"egress process sink(s) counted here too."
    )


def run(root):
    """The census of the repository at ``root``: its exit code, its lines, and what it read."""
    estate = read_estate(root)
    if estate.unread:
        return Result(1, [f"Child environment census: FAILED -- {len(estate.unread)} part(s) of the estate could "
                          "not be read, and what was not read was not counted:"]
                      + [f"  {reason}" for reason in estate.unread], estate, {})
    if not estate.modules:
        return Result(1, [f"Child environment census: nothing was scanned under {root}: no Python module in "
                          f"{_PACKAGE_DIR}/."], estate, {})
    census = take_census(estate)
    unparsed = sorted((rel, module.unparsed) for rel, module in census.items() if module.unparsed)
    if unparsed:
        return Result(1, ["Child environment census: FAILED -- these modules do not parse, so their starts "
                          "cannot be counted:"] + [f"  {rel}: {reason}" for rel, reason in unparsed],
                      estate, census)
    deep = sorted(rel for rel, module in census.items() if module.dotted.count(".") > _PACKAGE_DEPTH - 2)
    if deep:
        return Result(1, [f"Child environment census: FAILED -- these modules sit deeper than the {_PACKAGE_DEPTH - 2} "
                          "levels whose names the census follows, so what they define or start cannot be read "
                          "where it is used:"] + [f"  {rel}" for rel in deep], estate, census)
    names = estate.cache.get("child_env_names")
    if isinstance(names, SecretNamesUnread):
        return Result(1, [f"Child environment census: FAILED -- {names}: the secrets' names cannot be told "
                          "apart from any other."], estate, census)
    egress = _egress()
    try:
        found = find_all(estate, census)
    except egress.SuitesUnreadable as exc:
        return Result(1, [f"Child environment census: FAILED -- a suite that proves an inheriting start could "
                          f"not be read: {exc}"], estate, census)
    lines = []
    for name, findings in found.items():
        if findings:
            lines.append(f"Child environment census: {name.replace('_', ' ')} ({len(findings)}):")
            lines.extend(f"  {finding}" for finding in findings)
    if lines:
        return Result(1, lines, estate, census)
    return Result(0, [green_line(estate, census)], estate, census)


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[2]
    result = run(root)
    for line in result.lines:
        print(line)
    return result.code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
