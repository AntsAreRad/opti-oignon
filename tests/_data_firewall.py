#!/usr/bin/env python3
"""The test process, and the Python children it starts, see the maintainer's data as a fresh checkout would.

Real personal data lives in three kinds of place inside the tree: ``data/``
(the master key, the security mode, the signed audit chain),
``opti_oignon/data/`` (conversations, plugins, the governor's decisions,
the sync change feed) and database files wherever they lie. A contract must
never read or write any of it. Measured before this module existed, 253 of
them reached for it through real modules -- the master key by its
existence check, the audit chain by an append -- and a sandbox that hid
only ``data/`` answered differently from the maintainer's machine.

While a ``DataFirewall`` is installed, every path inside those places is
redirected to a mirror under a temporary directory. The mirror starts with
the tracked files of the data places as HEAD holds them, taken from git and
never from the places themselves, and nothing else: a read finds only what
this session wrote, a write lands in the mirror, and the real place is
never touched. A path is normalised before it is judged, so ``..``, a
doubled slash and a relative path that climbs out and back in name the
place they reach. The redirection covers the file functions the standard
library routes through: ``open``; the ``os`` calls that take a path,
including a listing with no path, which lists the working directory, and
the calls that make links, pipes and nodes; the two calls glob's string
globber bound when its class was made (every ``pathlib`` glob and rglob
lists through them); the ``open`` that ``tarfile``, ``bz2`` and
``tokenize`` bound under a name of their own; and every SQLite and
SQLCipher connect -- each package's and each DB-API module's -- by path or
by a ``file:`` URI whose query is kept. A listing through the globber
names what it found where the caller asked; ``os.scandir`` itself hands
back the mirror's entries, whose ``path`` lies in the mirror (so
``Path.iterdir`` yields mirror paths). A path relative to a directory
descriptor is left alone, since it names something the descriptor already
reached.

``cover_children`` carries the firewall to the Python child processes this
process starts from then on. It sets two variables -- the root of the tree
and its mirror, each a list joined with ``os.pathsep`` so that a firewall
covering inside a covered process adds to them -- and puts
``tests/_firewall_site`` first on ``PYTHONPATH``, then the tree's own root,
so that a child started from any directory imports the package under test
and not whatever checkout an install points at. A child that inherits them
runs that directory's ``usercustomize`` before any code of its own: it
installs one firewall per root, all on the same mirrors, and refuses to run
when it cannot. A child firewall installs lazily: it loads no module the
interpreter had not loaded without it, and wraps the connects, the globber
and the bound opens when their modules are executed. ``cover_children``
then starts one child to see the firewall installed there, and names the
reason when it is not, so a session can refuse to run uncovered.

In a covered process the project's package is refused (``ImportError``)
when a finder other than the path's -- an editable install -- would load it
from outside every covered tree: that checkout's data places are covered by
no firewall here. A package found along the path, a scratch copy a contract
runs on purpose, is left alone.

Each covered child leaves, in ``CHILDREN_REPORT`` in the mirror, a marker
when its firewall is installed and each path it keeps off, the first time,
in files named by its process and a token of its own; a fork writes its own.
A covered process also names every launch its firewall does not reach: a
Python child run with ``-I``, ``-E``, ``-s`` or ``-S``, with
``PYTHONNOUSERSITE``, without the two variables, or with a ``PYTHONPATH``
that does not hold the site -- the import footprint guard's probe, whose
``PYTHONPATH`` is replaced on purpose to measure a fresh interpreter, is one
-- and a process that is not Python started inside a data place. The summary
line counts all of it. Not covered, and not counted: a child that is not
Python outside the data places, an interpreter whose user site is disabled
by its own build (a virtual environment without the system site), a child
launched by a route that raises no launch event, and a path spelled through
a symbolic link -- an alias of a data place, ``/proc/self/cwd`` -- since
the firewall judges spellings, not files. Nor are the files SQLite opens by
itself from SQL (``ATTACH``, ``VACUUM INTO``) or a ``sqlite3.Connection``
built directly.

Each redirected path is counted against the running test, so the figure can
be driven down suite by suite as suites become hermetic.

Import-safe and stdlib-only. At module level it imports only what every
interpreter has loaded before its site hooks run, so that a child pays
nothing for it.
"""

import builtins
import io
import os
import sys

# The data places, relative to the root of the tree, as git spells them.
PLACES = ("data", "opti_oignon/data")
DATABASE_SUFFIXES = (".db", ".db-journal", ".db-wal", ".db-shm", ".sqlite", ".sqlite3")
# The project's package: in a covered process it loads from a covered tree.
PACKAGE = "opti_oignon"

# What carries the firewall to a child process. The usercustomize of SITE
# spells the two variable names itself: it reads them before this module is
# loaded.
ROOTS_VARIABLE = "OO_TEST_FIREWALL_ROOT"
MIRRORS_VARIABLE = "OO_TEST_FIREWALL_MIRROR"
SITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_firewall_site")
# Inside a mirror: per covered process, a marker, the paths it kept off and
# the launches its firewall did not reach.
CHILDREN_REPORT = ".firewall-children"
# The status of a process that refuses to run uncovered (EX_SOFTWARE).
REFUSED = 70

# The os functions whose first argument is a path, the ones that take two,
# and the one whose second argument is the path it makes.
_ONE_PATH = (
    "open", "stat", "lstat", "access", "listdir", "scandir", "mkdir", "rmdir",
    "remove", "unlink", "chmod", "utime", "truncate", "readlink", "chown", "lchown",
    "mkfifo", "mknod",
)
_TWO_PATHS = ("rename", "replace", "link")
_NEW_LINK = ("symlink",)
# The listings whose path, left out or None, is the working directory.
_LISTINGS = ("listdir", "scandir")
_NOT_A_FILE = (":memory:", "")
# The connect functions wrapped when their module is executed: each DB-API
# module before its package, so the package's star import copies the
# wrapped one. In the test process the packages were executed before the
# firewall, so each of the four is wrapped by name.
_CONNECT_MODULES = ("sqlite3.dbapi2", "sqlite3", "sqlcipher3.dbapi2", "sqlcipher3")
# Modules that bind the built-in open under a name of their own when they
# are executed, so replacing builtins.open does not reach them.
_BOUND_OPENS = {"tarfile": "bltn_open", "bz2": "_builtin_open", "tokenize": "_builtin_open"}
# glob's string globber binds these two os calls as its class is created, so
# replacing them on os does not reach it.
_GLOBBER_CALLS = ("scandir", "lstat")
# The audit events of a process launch, and the interpreter flags that leave
# PYTHONPATH or the user site unread.
_LAUNCH_EVENTS = frozenset(("subprocess.Popen", "os.posix_spawn", "os.exec"))
_UNCOVERING_FLAGS = "IEsS"
# What a child started to check prints: the roots its firewall covers.
_START_CHECK = "import sys; print('\\n'.join(getattr(sys.modules.get('usercustomize'), 'COVERED', ())))"

# The covering firewalls of this process that watch its launches, whether
# the audit hook that feeds them was added, and the arguments of the last
# subprocess launch (which may go on to raise os.posix_spawn for itself).
_LAUNCH_WATCHES = []
_AUDITING = []
_LAST_POPEN = [None]


class _EntryNamed:
    """A directory entry read in the mirror, named under the path the caller listed."""

    __slots__ = ("_entry", "name", "path")

    def __init__(self, entry, path):
        self._entry = entry
        self.name = entry.name
        self.path = path

    def __getattr__(self, name):
        if name == "_entry":
            raise AttributeError(name)
        return getattr(self._entry, name)

    def __fspath__(self):
        return self.path

    def __repr__(self):
        return f"<DirEntry {self.name!r}>"


class _ListingNamed:
    """A scandir listing of the mirror whose entries carry the path the caller gave."""

    def __init__(self, listing, path):
        self._listing = listing
        self._path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._listing.close()

    def __iter__(self):
        for entry in self._listing:
            yield _EntryNamed(entry, os.path.join(self._path, entry.name))

    def close(self):
        self._listing.close()


class _PatchingLoader:
    """A module loader that runs ``patch`` on the module right after it is executed."""

    def __init__(self, loader, patch):
        self._loader = loader
        self._patch = patch

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self._loader.exec_module(module)
        self._patch(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _AfterImport:
    """A meta path entry that has the named modules patched once they are executed.

    It finds nothing itself: it asks the finders behind it and hands their
    spec on with its loader wrapped. A second firewall's entry in front of
    this one is asked in turn, so each patch runs once.
    """

    def __init__(self, patches):
        self._patches = patches
        self._finding = set()

    def find_spec(self, name, path=None, target=None):
        patch = self._patches.get(name)
        if patch is None or name in self._finding:
            return None
        self._finding.add(name)
        try:
            spec = None
            for finder in list(sys.meta_path):
                find = getattr(finder, "find_spec", None)
                if finder is self or find is None:
                    continue
                spec = find(name, path, target)
                if spec is not None:
                    break
        finally:
            self._finding.discard(name)
        if spec is None or not hasattr(spec.loader, "exec_module"):
            return spec
        spec.loader = _PatchingLoader(spec.loader, patch)
        return spec


class _PackageHome:
    """A meta path entry that refuses the project's package when an install finds it outside every covered tree.

    An editable install maps the package's name to one checkout. Run from
    another -- a second worktree -- a process whose path does not reach its
    own tree would load that checkout's package, whose data places no
    firewall of this process covers. A package found along the path (the
    working directory, ``PYTHONPATH``, the interpreter's directories) is
    left alone: a contract that runs a copy of the package does it on
    purpose. One found by any other finder, whose file lies outside every
    covered tree, is refused before any of its code runs. One entry serves a
    process: a firewall of another copy of this module adds its trees to the
    same list, which it finds by its attribute.
    """

    def __init__(self):
        self.data_firewall_homes = []
        self._finding = False

    def find_spec(self, name, path=None, target=None):
        if name != PACKAGE or not self.data_firewall_homes or self._finding:
            return None
        self._finding = True
        try:
            return self._judged(name, path, target)
        finally:
            self._finding = False

    def _judged(self, name, path, target):
        """Ask the finders behind this one, as the import would; refuse what an install finds outside every tree."""
        along_the_path = sys.modules["_frozen_importlib_external"].PathFinder
        for finder in list(sys.meta_path):
            find = getattr(finder, "find_spec", None)
            if find is None or getattr(finder, "data_firewall_homes", None) is not None:
                continue
            spec = find(name, path, target)
            if spec is None:
                continue
            origin = spec.origin
            if finder is along_the_path or not isinstance(origin, str) or not os.path.isabs(origin):
                return spec
            home = os.path.realpath(os.path.dirname(os.path.dirname(origin)))
            if home in {os.path.realpath(tree) for tree in self.data_firewall_homes}:
                return spec
            raise ImportError(
                f"data firewall: {PACKAGE} would load from {origin}, found by an install and in no covered "
                f"tree ({os.pathsep.join(self.data_firewall_homes)}); that checkout's data places are not "
                "covered here, so the import is refused",
                name=name,
            )
        return None


def _package_home():
    """This process's one ``_PackageHome`` entry, put first on ``sys.meta_path`` when there is none."""
    for finder in sys.meta_path:
        if isinstance(getattr(finder, "data_firewall_homes", None), list):
            return finder
    home = _PackageHome()
    sys.meta_path.insert(0, home)
    return home


def _in_place(root, full):
    """The path of the normalised ``full`` inside ``root`` when it lies in a data place or is a database; else None."""
    if not full.startswith(root + os.sep):
        return None
    relative = full[len(root) + 1:]
    for place in PLACES:
        if relative == place or relative.startswith(place + os.sep):
            return relative
    if relative.endswith(DATABASE_SUFFIXES):
        return relative
    return None


def _normalised(text):
    """``text`` as an absolute path with ``..``, ``.`` and doubled slashes resolved, by spelling alone."""
    full = os.path.normpath(text if os.path.isabs(text) else os.path.join(os.getcwd(), text))
    if full.startswith("//"):
        full = "/" + full.lstrip("/")
    return full


def _environ_value(env, name):
    """``env[name]`` as text, whether ``env`` is keyed by text or by bytes; None when unset."""
    try:
        value = env.get(name)
    except (AttributeError, TypeError):
        return None
    if value is None:
        try:
            value = env.get(os.fsencode(name))
        except (TypeError, ValueError):
            value = None
    return os.fsdecode(value) if value is not None else None


def _is_python(program):
    """Whether ``program`` names a Python interpreter, or a Python script run by its own line."""
    try:
        name = os.path.basename(os.fsdecode(os.fspath(program)))
    except TypeError:
        return False
    if name.endswith(".py"):
        return True
    return name.startswith("python") and all(ch.isdigit() or ch == "." for ch in name[len("python"):])


def _uncovering_flags(argv):
    """The interpreter flags in ``argv`` that leave ``PYTHONPATH`` or the user site unread."""
    found = []
    rest = [os.fsdecode(arg) for arg in argv[1:]]
    while rest:
        arg = rest.pop(0)
        if arg in ("-", "--") or not arg.startswith("-"):
            break
        if arg.startswith("--"):
            if arg == "--check-hash-based-pycs" and rest:
                rest.pop(0)
            continue
        letters = arg[1:]
        for index, letter in enumerate(letters):
            if letter in _UNCOVERING_FLAGS:
                found.append("-" + letter)
            if letter in "cm":
                return found
            if letter in "XW":
                if index == len(letters) - 1 and rest:
                    rest.pop(0)
                break
    return found


def launch_uncovered(program, argv, cwd, env, root, site=SITE):
    """Why a launch leaves the firewall of ``root`` behind, in a few words; None when it does not.

    ``program``, ``argv``, ``cwd`` and ``env`` are what a launch event
    carries (``env`` None: this process's environment). A process that is
    not Python is named only when it starts inside a data place, since that
    is the one way it reaches one without naming it.
    """
    argv = [argv] if isinstance(argv, (str, bytes)) else list(argv or ())
    if program is None:
        program = argv[0] if argv else ""
    if not _is_python(program):
        try:
            here = _normalised(os.fsdecode(os.fspath(cwd)) if cwd is not None else os.getcwd())
        except (OSError, TypeError):
            return None
        return "not Python, in a data place" if _in_place(root, here) is not None else None
    flags = _uncovering_flags(argv)
    if flags:
        return flags[0]
    env = os.environ if env is None else env
    if _environ_value(env, "PYTHONNOUSERSITE"):
        return "PYTHONNOUSERSITE"
    if root not in (_environ_value(env, ROOTS_VARIABLE) or "").split(os.pathsep):
        return "without the firewall's variables"
    if site not in (_environ_value(env, "PYTHONPATH") or "").split(os.pathsep):
        return "PYTHONPATH without the site"
    return None


def _audit(event, args):
    """Hand each launch this process makes to the firewalls that watch; never breaks a launch."""
    if event not in _LAUNCH_EVENTS or not _LAUNCH_WATCHES:
        return
    try:
        if event == "subprocess.Popen":
            program, argv, cwd, env = args
            _LAST_POPEN[0] = argv
        else:
            program, argv, env = args
            cwd = None
            if event == "os.posix_spawn" and _LAST_POPEN[0] is not None and list(_LAST_POPEN[0]) == list(argv):
                _LAST_POPEN[0] = None
                return  # the same launch, already seen as subprocess.Popen
        for firewall in list(_LAUNCH_WATCHES):
            firewall._launched(program, argv, cwd, env)
    except Exception:  # noqa: BLE001 - counting never stops a process from starting
        pass


def _watch(firewall):
    if firewall not in _LAUNCH_WATCHES:
        _LAUNCH_WATCHES.append(firewall)
    if not _AUDITING:
        sys.addaudithook(_audit)
        _AUDITING.append(True)


class DataFirewall:
    """Redirects the data places of one tree to a mirror while it is installed."""

    def __init__(self, root, *, mirror=None, seed=True, report=None):
        self.root = os.path.normpath(os.path.abspath(os.fspath(root)))
        self.mirror = os.fspath(mirror) if mirror is not None else None
        self._own_mirror = mirror is None
        self._seed = seed
        self.report = os.fspath(report) if report is not None else None
        self._saved = {}
        self._patched = []
        self._finder = None
        self._open = None
        self._token = None
        self._at_fork = False
        self._children = None
        self._children_report = None
        self._home = None
        self._site = SITE
        self.children_uncovered = None
        self.launched = []
        self.current = None
        self.redirected = {}

    # -- which paths ------------------------------------------------------------

    def _relative(self, full):
        """The path of ``full`` inside the tree when it must be redirected, else None."""
        return _in_place(self.root, full)

    def target(self, path):
        """``path`` itself, or where it lies in the mirror when it is in a data place."""
        if path is None or isinstance(path, int):
            return path
        try:
            raw = os.fspath(path)
        except TypeError:
            return path
        relative = self._relative(_normalised(os.fsdecode(raw)))
        if relative is None:
            return path
        kept = self.redirected.setdefault(self.current or "(while collecting)", set())
        if relative not in kept:
            kept.add(relative)
            if self.report is not None:
                self._report_line("paths", relative)
        mirrored = os.path.join(self.mirror, relative)
        return os.fsencode(mirrored) if isinstance(raw, bytes) else mirrored

    # -- the report of a covered process ----------------------------------------

    def _report_line(self, kind, line):
        """Append ``line`` to this process's file of ``kind`` in the report directory; counting never breaks a run."""
        try:
            name = os.path.join(self.report, f"{os.getpid()}-{self._token}.{kind}")
            opened = self._saved.get("os.open", os.open)
            handle = opened(name, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                if line is not None:
                    os.write(handle, os.fsencode(line + "\n"))
            finally:
                os.close(handle)
        except (OSError, TypeError, ValueError):
            pass

    def _mark(self):
        """A fresh token for this process, and its marker: this firewall is installed here."""
        self._token = os.urandom(6).hex()
        self._report_line("covered", None)

    def _forked(self):
        """In a fork: count it as a covered process of its own, with a report of its own."""
        if not self._saved:
            return
        if self.report is None and self._children_report is not None and self._children is not None:
            self.report = self._children_report
            self.redirected = {}
            self.launched = []
        if self.report is not None:
            self._mark()

    def _launched(self, program, argv, cwd, env):
        reason = launch_uncovered(program, argv, cwd, env, self.root, self._site)
        if reason is None:
            return
        if self.report is not None:
            self._report_line("launches", reason)
        else:
            self.launched.append(reason)

    # -- the mirror -------------------------------------------------------------

    def _seed_from_head(self):
        """Copy the tracked files of the data places as HEAD holds them, if git can say."""
        import subprocess
        import tarfile

        git = ["git", "-C", self.root]
        try:
            listed = subprocess.run(
                [*git, "ls-tree", "-r", "-z", "--name-only", "HEAD", "--", *PLACES],
                capture_output=True, check=True, timeout=30,
            ).stdout
            names = [name for name in listed.decode("utf-8", "surrogateescape").split("\0") if name]
            if not names:
                return
            archive = subprocess.run(
                [*git, "archive", "--format=tar", "HEAD", "--", *names],
                capture_output=True, check=True, timeout=60,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return  # not a git tree: the mirror stays empty
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(self.mirror, filter="data")

    # -- install ----------------------------------------------------------------

    def _one_path(self, original, listing=False):
        target = self.target

        def redirected(*args, **kwargs):
            if kwargs.get("dir_fd") is None:
                if args:
                    first = "." if listing and args[0] is None else args[0]
                    args = (target(first), *args[1:])
                elif "path" in kwargs:
                    first = "." if listing and kwargs["path"] is None else kwargs["path"]
                    kwargs["path"] = target(first)
                elif listing:
                    args = (target("."),)
            return original(*args, **kwargs)

        return redirected

    def _two_paths(self, original):
        target = self.target

        def redirected(src, dst, *args, **kwargs):
            if kwargs.get("src_dir_fd") is None:
                src = target(src)
            if kwargs.get("dst_dir_fd") is None:
                dst = target(dst)
            return original(src, dst, *args, **kwargs)

        return redirected

    def _new_link(self, original):
        target = self.target

        def redirected(src, dst, *args, **kwargs):
            if kwargs.get("dir_fd") is None:
                dst = target(dst)
            return original(src, dst, *args, **kwargs)

        return redirected

    def _named_scandir(self, original):
        target = self.target

        def scandir(path):
            moved = target(path)
            if moved is path:
                return original(path)
            return _ListingNamed(original(moved), path)

        return scandir

    def _uri_target(self, database):
        """A ``file:`` URI whose file lies in a data place, rewritten to the mirror with its query kept; else as given.

        The query (``mode=ro``, say) and the fragment travel unchanged, so a
        read-only open of the mirror stays read-only. An in-memory URI, one
        that is not ``file:``, and one naming a file elsewhere pass untouched.
        """
        from urllib.parse import quote, unquote, urlsplit

        text = os.fsdecode(os.fspath(database))
        if not text.startswith("file:"):
            return database
        parts = urlsplit(text)
        path = unquote(parts.path)
        if path in _NOT_A_FILE or path.startswith(":memory:"):
            return database
        moved = self.target(path)
        if moved is path:
            return database
        os.makedirs(os.path.dirname(moved), exist_ok=True)
        rebuilt = "file:" + quote(moved) + ("?" + parts.query if parts.query else "")
        rebuilt += "#" + parts.fragment if parts.fragment else ""
        return os.fsencode(rebuilt) if isinstance(database, bytes) else rebuilt

    def _connect(self, original):
        target = self.target
        uri_target = self._uri_target

        def redirected(database, *args, **kwargs):
            named = isinstance(database, (str, bytes, os.PathLike))
            if named and kwargs.get("uri"):
                database = uri_target(database)
            elif named and os.fsdecode(os.fspath(database)) not in _NOT_A_FILE:
                moved = target(database)
                if moved is not database:
                    os.makedirs(os.path.dirname(os.fsdecode(moved)), exist_ok=True)
                database = moved
            return original(database, *args, **kwargs)

        redirected.data_firewall = self
        return redirected

    def _wrap_connect(self, module):
        """Wrap ``module.connect``, once per firewall."""
        original = getattr(module, "connect", None)
        if original is None or getattr(original, "data_firewall", None) is self:
            return
        self._patched.append((module, "connect", original))
        module.connect = self._connect(original)

    def _cover_bound_open(self, module):
        """Point the ``open`` a module bound under a name of its own at this firewall's."""
        name = _BOUND_OPENS.get(module.__name__)
        bound = getattr(module, name, None) if name else None
        if bound is None or bound is self._open:
            return
        self._patched.append((module, name, bound))
        setattr(module, name, self._open)

    def _cover_globber(self, module):
        """Wrap the two os calls glob's string globber bound when its class was made, when it has one."""
        globber = getattr(module, "_StringGlobber", None)
        if globber is None:
            return
        for name in _GLOBBER_CALLS:
            bound = globber.__dict__.get(name)
            original = getattr(bound, "__func__", None)
            if original is None or getattr(original, "data_firewall", None) is self:
                continue
            wrapper = self._named_scandir(original) if name == "scandir" else self._one_path(original)
            wrapper.data_firewall = self
            self._patched.append((globber, name, bound))
            setattr(globber, name, staticmethod(wrapper))

    def install(self, *, eager=True):
        """Redirect, for this process, until ``uninstall``. Installing twice is a no-op.

        ``eager`` imports the SQLite and SQLCipher packages and glob now, and
        wraps their connects and the globber, as the test process does. A
        child installs with ``eager=False``: what is already loaded is
        wrapped now, the rest as its module is executed, and nothing is
        imported for the firewall's sake. Either way a module not yet
        executed is patched when it is.
        """
        if self._saved:
            return
        if self.mirror is None:
            import tempfile

            self.mirror = tempfile.mkdtemp(prefix="oo-data-mirror-")
        for place in PLACES:
            os.makedirs(os.path.join(self.mirror, *place.split("/")), exist_ok=True)
        if self._seed:
            self._seed_from_head()
        target = self.target
        original_open = builtins.open

        def redirected_open(file, *args, **kwargs):
            return original_open(target(file), *args, **kwargs)

        redirected_open.data_firewall = self
        self._open = redirected_open
        self._saved["open"] = original_open
        builtins.open = io.open = redirected_open
        for name in _ONE_PATH:
            original = getattr(os, name, None)
            if original is None:
                continue
            self._saved["os." + name] = original
            setattr(os, name, self._one_path(original, listing=name in _LISTINGS))
        for names, wrap in ((_TWO_PATHS, self._two_paths), (_NEW_LINK, self._new_link)):
            for name in names:
                original = getattr(os, name, None)
                if original is None:
                    continue
                self._saved["os." + name] = original
                setattr(os, name, wrap(original))
        if eager:
            import glob  # noqa: F401 - executed now, so wrapped now
            import sqlite3  # noqa: F401

            try:
                import sqlcipher3.dbapi2  # noqa: F401
            except Exception:  # noqa: BLE001 - no SQLCipher, nothing more to cover
                pass
        patches = {name: self._wrap_connect for name in _CONNECT_MODULES}
        patches["glob"] = self._cover_globber
        patches.update((name, self._cover_bound_open) for name in _BOUND_OPENS)
        waiting = {}
        for name, patch in patches.items():
            module = sys.modules.get(name)
            if module is not None:
                patch(module)
            else:
                waiting[name] = patch
        if waiting:
            self._finder = _AfterImport(waiting)
            sys.meta_path.insert(0, self._finder)
        if self.report is not None:
            self._mark()
        if not self._at_fork:
            self._at_fork = True
            os.register_at_fork(after_in_child=self._forked)

    def uninstall(self):
        """Put every function back; remove the mirror when this firewall made it."""
        if not self._saved:
            return
        if self._finder is not None:
            if self._finder in sys.meta_path:
                sys.meta_path.remove(self._finder)
            self._finder = None
        for owner, name, original in reversed(self._patched):
            setattr(owner, name, original)
        self._patched.clear()
        original_open = self._saved.pop("open")
        builtins.open = io.open = original_open
        for module_name, name in _BOUND_OPENS.items():
            module = sys.modules.get(module_name)
            if module is not None and getattr(module, name, None) is self._open:
                setattr(module, name, original_open)
        for key, original in list(self._saved.items()):
            setattr(os, key[len("os."):], original)
        self._saved.clear()
        if self._own_mirror and self.mirror:
            import shutil

            shutil.rmtree(self.mirror, ignore_errors=True)

    # -- child processes --------------------------------------------------------

    def cover_children(self, site=None):
        """Carry this firewall to the Python children started from now on; None, or why they are not covered.

        Call it once installed. The variables and ``PYTHONPATH`` are set
        whatever the answer, and ``uncover_children`` puts all three back.
        The site is ``tests/_firewall_site`` beside this module; a copy of
        this module that ships none, inside a process a session already
        covers, carries its children through the site already on
        ``PYTHONPATH``. The answer is measured: one child is started, and it
        must report this tree among the ones its firewall covers.
        """
        if self._children is not None:
            return self.children_uncovered
        site = _site(site)
        report = os.path.join(self.mirror, CHILDREN_REPORT)
        os.makedirs(report, exist_ok=True)
        self._children_report = report
        saved = {name: os.environ.get(name) for name in (ROOTS_VARIABLE, MIRRORS_VARIABLE, "PYTHONPATH")}
        self._children = saved
        os.environ[ROOTS_VARIABLE] = _added(saved[ROOTS_VARIABLE], self.root)
        os.environ[MIRRORS_VARIABLE] = _added(saved[MIRRORS_VARIABLE], self.mirror)
        os.environ["PYTHONPATH"] = os.pathsep.join([site, self.root] + ([saved["PYTHONPATH"]] if saved["PYTHONPATH"] else []))
        self._home = _package_home()
        self._home.data_firewall_homes.append(self.root)
        self._site = site
        _watch(self)
        if os.pathsep in self.root or os.pathsep in self.mirror:
            self.children_uncovered = f"the root or the mirror holds {os.pathsep!r}, which the variables cannot carry"
        elif not os.path.isfile(os.path.join(site, "usercustomize.py")):
            self.children_uncovered = f"no usercustomize in {site}, so nothing installs it there"
        else:
            self.children_uncovered = self._start_check()
        return self.children_uncovered

    def _start_check(self):
        """None when a child of this interpreter reports this tree covered; else what it did."""
        import subprocess

        try:
            child = subprocess.Popen(
                [sys.executable, "-c", _START_CHECK], cwd=self.mirror, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        except OSError as error:
            return f"a child of this interpreter could not be started to check it ({error})"
        try:
            out, err = child.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            child.kill()
            out, err = child.communicate()
        finally:
            self._forget(child.pid)
        if child.returncode != 0:
            said = (err.strip().splitlines() or ["(nothing on its standard error)"])[-1]
            return f"a child of this interpreter exited {child.returncode}: {said}"
        if self.root not in out.splitlines():
            return ("a child of this interpreter did not install it: its user site is disabled "
                    "(-s, PYTHONNOUSERSITE, a virtual environment) or another usercustomize comes first")
        return None

    def _forget(self, pid):
        """Remove the report files of the child ``cover_children`` started to check."""
        try:
            for name in os.listdir(self._children_report):
                if name.startswith(f"{pid}-"):
                    os.remove(os.path.join(self._children_report, name))
        except OSError:
            pass

    def uncover_children(self):
        """Put back the two variables and ``PYTHONPATH`` as ``cover_children`` found them."""
        if self._children is None:
            return
        for name, value in self._children.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if self._home is not None:
            homes = self._home.data_firewall_homes
            if self.root in homes:
                homes.remove(self.root)
            if not homes and self._home in sys.meta_path:
                sys.meta_path.remove(self._home)
            self._home = None
        if self in _LAUNCH_WATCHES:
            _LAUNCH_WATCHES.remove(self)
        self._children = None

    def children_seen(self):
        """What the covered processes reported, and the launches this process's firewall did not reach.

        ``installed``: processes whose firewall was installed (a marker
        each); ``paths`` and ``processes``: the paths they kept off and how
        many kept any; ``launches``: reason -> count of launches not reached.
        """
        seen = {"installed": 0, "paths": 0, "processes": 0, "launches": {}}
        try:
            names = sorted(os.listdir(self._children_report)) if self._children_report else []
        except OSError:
            names = []
        for name in names:
            kind = name.rpartition(".")[2]
            if kind == "covered":
                seen["installed"] += 1
                continue
            try:
                with open(os.path.join(self._children_report, name), encoding="utf-8", errors="surrogateescape") as lines:
                    found = [line.rstrip("\n") for line in lines if line.strip()]
            except OSError:
                continue
            if kind == "paths" and found:
                seen["paths"] += len(set(found))
                seen["processes"] += 1
            elif kind == "launches":
                for reason in found:
                    seen["launches"][reason] = seen["launches"].get(reason, 0) + 1
        for reason in self.launched:
            seen["launches"][reason] = seen["launches"].get(reason, 0) + 1
        return seen

    def summary(self):
        """One line for the end of the session: how much was kept off the real places."""
        paths = sum(len(found) for found in self.redirected.values())
        line = (
            f"data firewall: {paths} path(s) from {len(self.redirected)} test(s) or imports "
            "kept off the maintainer's data and served from the mirror"
        )
        if self._children_report is None:
            return line
        if self.children_uncovered:
            return f"{line}; child processes NOT covered: {self.children_uncovered}"
        seen = self.children_seen()
        line += (f"; child processes: {seen['paths']} path(s) kept off in {seen['processes']} process(es), "
                 f"{seen['installed']} installed the firewall")
        launches = seen["launches"]
        detail = ", ".join(f"{reason}: {count}" for reason, count in sorted(launches.items()))
        return f"{line}; {sum(launches.values())} launch(es) it does not reach" + (f" ({detail})" if detail else "")


def cover_this_child(roots, mirrors):
    """In a child process: one firewall per root, installed lazily on its mirror; the firewalls.

    Called by the site's ``usercustomize`` before any code of the child's
    own. Each firewall reports into its mirror, watches the launches this
    child makes, and adds its tree to the package's covered homes. Raises
    whatever stops an install: the caller refuses to run.
    """
    firewalls = []
    for root, mirror in zip(roots, mirrors):
        firewall = DataFirewall(root, mirror=mirror, seed=False, report=os.path.join(mirror, CHILDREN_REPORT))
        firewall.install(eager=False)
        firewall.current = "(child process)"
        firewalls.append(firewall)
    home = _package_home()
    for firewall in firewalls:
        home.data_firewall_homes.append(firewall.root)
        _watch(firewall)
    return firewalls


def _site(site):
    """The site that carries the firewall to a child: the one given, else this module's, else one already on the path."""
    if site is not None:
        return os.fspath(site)
    if os.path.isfile(os.path.join(SITE, "usercustomize.py")) or not os.environ.get(ROOTS_VARIABLE):
        return SITE
    for entry in (os.environ.get("PYTHONPATH") or "").split(os.pathsep):
        if os.path.basename(entry) == os.path.basename(SITE) and os.path.isfile(os.path.join(entry, "usercustomize.py")):
            return entry
    return SITE


def _added(listed, value):
    """``listed`` (an ``os.pathsep`` list, or None) with ``value`` at its end."""
    return value if not listed else listed + os.pathsep + value
