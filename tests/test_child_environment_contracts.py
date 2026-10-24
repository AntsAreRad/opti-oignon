#!/usr/bin/env python3
"""Contracts for the backend's children, started without the platform's secrets (block 178).

The launchers hand the backend the key file's passphrase and the master key in its environment. Up to block 177
every program the backend ran was handed both -- the bubblewrap probes and the compiler at import, the pool's
workers, the tools of the security pages --, a shell among them read the caller's ``BASH_ENV``, and any process of
the same user read both in ``/proc/<pid>/environ``. The entry now takes them out first
(``opti_oignon/process_secrets.py``), every start is handed a built environment, and a census guard
(``.github/scripts/child_env_census_guard.py``) holds every start of the package to it.

  * XE1 -- the census counts every process-start spelling it lists, through aliases, from-imports, getattr, the
    module cache, a star import, a function that returns a start, a start module reached through another module
    or through the package, ``posix`` for ``os``, a tuple, a class attribute, ``staticmethod``, ``partial``, a
    dispatch table, an event loop's ``subprocess_exec`` in any module (one taken as a value -- bound, wrapped,
    read through getattr -- included), multiprocessing's process classes and spawner, shared memory and the
    resource tracker, a C library's ``system`` through ctypes (read as an attribute or by its name), a browser
    built, the stdlib's ``_posixsubprocess.fork_exec`` called directly (``subprocess._fork_exec`` too), and nothing
    else: prose, a local function of a start's name, a uname field that runs nothing, a multiprocessing call that
    starts nothing, a C library's other function.
  * XE2 -- every direct start is handed a built environment: ``child_env`` imported from process_secrets (at
    module or function level, by a relative import too) and bound to nothing else in its module, its extra
    environment read as a mapping is, an untrusted tier's allowlist in the module that defines it once, a built
    one with a literal mapping merged by ``|``, or a name its function binds once to one and uses only as the
    census reads -- read, its length taken, handed as the environment of a start or a launcher --; no env, a
    local child_env, a builder shadowed or rebound, an allowlist called elsewhere or bound again in its module, a
    name bound twice, a literal, ``**kwargs``, a module-level name, a secret's name or a shell variable put back
    (by a key, an update, its keyword, setdefault, or child_env's extra), a secret's value under any name, a key
    from a variable, an alias, a helper the environment is handed to -- as an argument or by ``env=``, a method of
    another class that shares a launcher's name among them --, a function nested in the one that starts, a ``len``
    of the module's own, a merger handed on as a value, are findings; a merger's ``extra or {}`` and its retry are
    not; a merger's parameter rebound is read as bound; the real tree has none.
  * XE3 -- a start handed a copy of the process's environment is a finding, in each of its spellings, and so is a
    built environment, or a launcher's parameter, merged with anything that reads ``os.environ`` -- a key set
    from it, a copy bound to a name first --; the real tree has none.
  * XE4 -- a launcher's environment parameter is followed to every caller, in its module and in the others: each
    must fill it with a built one; a caller that leaves it to its default, a launcher handed on as a value or
    called nowhere, taken as a value or read through getattr in another module, is a finding; a module with an
    attribute or a function of the same name calls its own; a method launcher named ``run`` takes neither
    ``subprocess.run`` nor ``asyncio.run`` for its calls. A parameter merged into a built environment (a merger)
    is read at every caller of the package, its module's and the others', and a merger called nowhere is a
    finding. The real sandbox's ``_spawn_tracked`` is filled by both its callers.
  * XE5 -- every inheriting start is listed by module, function and spelling with its reason and its contract;
    an unlisted one, a listed one gone and a spelling swapped are findings; every listed contract names a test of
    a suite that names the module.
  * XE6 -- every process sink the egress census charges, in every spelling it charges, is a start this census
    counts, on the same line in the same function; a spelling this census forgot is a finding.
  * XE7 -- what cannot be read fails the census by name: no module, a file that is not UTF-8, a module that does
    not parse, a module deeper than the levels whose names the census follows (one six levels down is read), a
    directory that cannot be listed, a process_secrets that does not write its names or its shell variables, a
    proving suite that is not UTF-8.
  * XE8 -- a secret's name appears only where it may -- the first argument of one of the holder's readers, bound
    by its import alone, or the one module-level constant that names it --: a read or a write of it in the
    environment, a name handed to any other function, compared (``==``, ``in`` a tuple, a ``match``), a
    parameter's default, a keyword, a binding in a function, a constant bound to another or bound twice, are
    findings, in every spelling the census follows -- a literal, a constant (a tuple's too), another module's
    constant read or imported under another name, an item of the holder's names, through a copy of the
    environment, bound to a name or not, a module whose text never names the environment, a name in two literals
    or in bytes, a constant handed on through a third module --; one identifier bound to both names gives two
    findings and the census ends; the holder's own reads, a reader shadowed (by a definition or an import) or
    rebound aside, are not findings, nor a name said in a sentence (a log line, a status), read by keyword or beside
    a local named like the reader -- a log call's argument and a mapping's value are places like any other --; and
    a secret's value put into the process's environment (``os.environ``, ``os.putenv``) is one;
    the real tree has none.
  * XE9 -- ``take()`` moves both secrets into a holder of its own and takes what a shell runs code from out of
    the environment with them: ``secret()`` answers them, an empty one is no secret and leaves too, a secret's
    name put back once one is held is refused and a second take keeps the held value -- a value given again
    behind the check is not held over it --, a name that is no secret is refused; the holder zeroes the bytes it
    is built from, and memory it could not lock is said so; where the limit allows, both are locked, and a wipe
    keeps the lock of a holder beside it.
  * XE10 -- after ``take()`` the process's ``/proc/<pid>/environ`` holds no byte of either value, read by the
    process and by a same-user child, a command's name with a parenthesis in it included; before it, it holds
    both; every entry of a name is zeroed whole; a block read wrong is caught when read back, never taken for a
    wipe: the read-back looks at every region it zeroed and for the value itself.
  * XE11 -- ``child_env()`` drops the secrets and what a shell runs code from: ``bash -c`` children started with it
    run no ``BASH_ENV``, no exported function, no trace under an inherited ``SHELLOPTS``, no ``PS4`` once a script
    traces itself, and see no secret; the same children started without it run all four; an ``extra`` that names
    one is refused; ``os.environ`` is left as it was.
  * XE12 -- a real worker of the background pool's ``_spawn_executor`` started after ``take()`` holds no secret;
    one started before holds both (opti_oignon.background_pool).
  * XE13 -- after ``take()``, ``load_keyfile()`` opens a key file wrapped under the held passphrase and
    ``get_encryption_key()`` answers the held master key, the environment holding neither.
  * XE14 -- setup after ``take()`` wraps the key file it makes under the held passphrase, never in the legacy
    format, says so (the route's ``wrapped``), and so does the database command's key writer.
  * XE15 -- the refusals and the statuses read the holder: an environment key that is no key is refused by the
    databases' layer and by setup, the status says a key is set, the being's store calls a key that does not load
    unreadable, the knowledge base's migration refuses to write plaintext beside a held key.
  * XE16 -- the held secrets outlive a signal's wipe of every SecureBytes, are wiped at exit, their bytes zeroed,
    and once wiped they are a third state: ``secret()``, the key loads and the databases' key raise
    ``SecretWiped``, never answer "no key", and setup refuses rather than write an unprotected key file.
  * XE17 -- the real entry, imported as uvicorn imports it, its environment holding both secrets, a ``BASH_ENV``
    and an exported function, with recording stand-ins for bwrap, gcc and ld first on its PATH: no start holds a
    secret or a shell hook, the bwrap probe's echo runs in a cleared environment, and neither the environment nor
    ``/proc/self/environ`` holds a secret afterwards (opti_oignon.pqc_signatures).
  * XE18 -- the untrusted tiers stay allowlisted and hold nothing a shell runs, whatever the environment holds,
    and the sandbox's bubblewrap itself is handed ``child_env()``.
  * XE19 -- no process starts before ``take()`` has taken both names out: an audit hook installed first records
    every start Python audits during the entry's import (``subprocess.Popen``, ``os.system``, ``os.exec*``,
    ``os.spawn*``, ``os.posix_spawn*``, ``os.fork*``, ``pty.spawn``), each after the pops and none handed a
    secret (opti_oignon.secure_bytes, opti_oignon.conversation_wipe, opti_oignon.pqc_signatures).
  * XE20 -- the start a request makes later, ``uname -p`` from ``platform.processor()``, holds no secret after
    ``take()``; without it, it holds both (opti_oignon.auto_tuner).
  * XE21 -- a contract that leaves the holder holding a secret fails at its teardown, naming it; one that leaves
    nothing passes.
  * XE22 -- ``scripts/dev_backend.sh`` run as it is reads no ``BASH_ENV`` and imports no function, and still runs
    the server; run under a plain bash, both run.
  * XE23 -- while a secret could not be zeroed in the environment the backend was started with, a sandbox
    command without bubblewrap and a plugin started without it are refused by name; with bubblewrap, or once
    nothing is left unwiped, they run; a wipe that fails is recorded, never taken for a wipe; a plugin's extra
    environment that names a secret or a shell variable is refused.
  * XE24 -- while ``take()`` holds a secret, the process and its children dump no core (``RLIMIT_CORE`` 0, both
    limits); holding none, the limits are left as they were.
  * XE25 -- the launcher and the key file's gate zero what they take out of their environment: neither secret is
    left in their ``/proc/self/environ``.
  * XE26 -- the startup checklist says what became of the secrets: held -- and whether their memory could be
    locked, or too short for the start check to look for --, never wiped, back in the environment, unreadable, or
    not checked in a process that did not start through the entry; never critical.
  * XE27 -- once a secret is held, a process start that would hand it to a child is refused as it is made --
    through subprocess, ``os.system`` (its command too), ``os.posix_spawn`` and ``os.execve`` --: a held value in
    the environment handed to it, inside an entry, or in its arguments, a secret's name in its environment (as
    text or bytes), a held value as a variable's name, a shell variable in its environment; and Python code of
    this interpreter cannot put a secret's
    value (as a variable's value or name), a secret's name or a shell variable into this process's environment,
    which the starts Python does not audit inherit; nothing is started, and the log holds neither the value nor
    where it sat; a value under 8 bytes, or one a variable holds already, is not looked for, and the status names
    both; one of 8 is; with the module's names gone, as the interpreter ends, the check still lets every other
    event through and refuses a start or a put; a clean start runs; a process that holds nothing installs no
    check. XE2, XE4, XE5 and XE8 read the cases of rounds 4 and 5 (``_ROUND4``, ``_ROUND5``) too.

Local-only (the public distribution ships no tests). The census contracts load the guard through the shared
isolation window; ``take()`` runs only in fresh interpreters, which the data firewall covers, with secrets these
contracts make up; the test process's own environment holds no secret (SE13).
"""

import base64
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_GUARD = REPO / ".github" / "scripts" / "child_env_census_guard.py"
_EGRESS = REPO / ".github" / "scripts" / "egress_census_guard.py"
PASS, KEYNAME = "OPTI_KEYFILE_PASSPHRASE", "OPTI_ENCRYPTION_KEY"
NAMES = (PASS, KEYNAME)
# Secrets these contracts make up: a passphrase no key file of the maintainer's is wrapped under, and a well-formed
# master key that encrypts nothing of theirs.
PHRASE = "xe phrase only these contracts know"
RAW_KEY = bytes(range(32))
KEY = base64.urlsafe_b64encode(RAW_KEY).decode("ascii")
HAZARDS = ("BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "PS4")

BUDGET_S = {
    "test_xe1_the_census_counts_every_process_start_spelling_and_nothing_else": 4.0,
    "test_xe2_every_direct_start_is_handed_a_built_environment": 8.0,
    "test_xe3_a_start_handed_a_copy_of_the_process_environment_is_a_finding": 8.0,
    "test_xe4_a_launchers_environment_parameter_is_followed_to_its_callers": 8.0,
    "test_xe5_every_inheriting_start_is_listed_with_its_reason_and_its_contract": 8.0,
    "test_xe6_every_process_sink_of_the_egress_census_is_a_start_counted_here": 8.0,
    "test_xe7_what_cannot_be_read_fails_the_census_by_name": 8.0,
    "test_xe8_no_module_but_process_secrets_reads_a_secret_in_the_environment": 8.0,
    "test_xe9_take_moves_both_secrets_into_the_holder": 6.0,
    "test_xe10_after_take_the_environment_the_process_was_started_with_holds_no_secret": 6.0,
    "test_xe11_child_env_drops_the_secrets_and_what_a_shell_runs_code_from": 6.0,
    "test_xe12_a_pool_worker_started_after_take_holds_no_secret": 30.0,
    "test_xe13_the_key_loads_read_the_held_secrets": 12.0,
    "test_xe14_setup_after_take_wraps_the_key_file_under_the_held_passphrase": 20.0,
    "test_xe15_the_refusals_and_the_statuses_read_the_holder": 20.0,
    "test_xe16_a_wiped_holder_is_a_third_state": 8.0,
    "test_xe17_the_real_entry_starts_nothing_that_holds_a_secret": 40.0,
    "test_xe18_the_untrusted_tiers_stay_allowlisted_and_their_launcher_gets_child_env": 6.0,
    "test_xe19_no_process_starts_before_the_secrets_leave_the_environment": 40.0,
    "test_xe20_uname_run_for_a_request_holds_no_secret": 12.0,
    "test_xe21_a_contract_that_leaves_a_secret_held_fails_at_its_teardown": 4.0,
    "test_xe22_the_dev_backend_script_reads_no_hook_run_as_it_is": 8.0,
    "test_xe23_untrusted_code_beside_an_unwiped_secret_is_refused_by_name": 12.0,
    "test_xe24_no_core_dump_while_a_secret_is_held": 6.0,
    "test_xe25_the_launcher_and_the_gate_zero_what_they_take_out": 12.0,
    "test_xe26_the_startup_checklist_says_what_became_of_the_secrets": 4.0,
    "test_xe27_a_start_that_would_hand_a_child_a_secret_is_refused": 8.0,
}


# ---------------------------------------------------------------------------
# The census, loaded on the shared window.
# ---------------------------------------------------------------------------
def _load():
    loaded, restore = isolate(targets={"child_env_census_guard": _GUARD})
    return loaded["child_env_census_guard"], restore


_SECRETS_FIXTURE = (
    'SECRET_NAMES = ("OPTI_KEYFILE_PASSPHRASE", "OPTI_ENCRYPTION_KEY")\n'
    'SHELL_HAZARDS = ("BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "PS4")\n'
    'FUNCTION_PREFIX = "BASH_FUNC_"\n\n\n'
    "def child_env(extra=None):\n    return dict(extra or {})\n"
)


def _module(guard, rel, text):
    """One module of the census, the secrets' names those of the fixture."""
    names = guard.Names(frozenset(NAMES), frozenset(HAZARDS), "BASH_FUNC_", NAMES)
    return guard._Module(rel, textwrap.dedent(text), names, {})


def _estate(root, modules, suites=None):
    """A tree under ``root`` holding ``modules`` (rel -> text), process_secrets.py among them unless given."""
    modules = dict(modules)
    modules.setdefault("opti_oignon/process_secrets.py", _SECRETS_FIXTURE)
    for rel, text in modules.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            path.write_bytes(text)
        else:
            path.write_text(textwrap.dedent(text), encoding="utf-8")
    for rel, text in (suites or {}).items():
        path = root / "tests" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text) if isinstance(text, bytes) else path.write_text(text, encoding="utf-8")
    return root


def _verdicts(module):
    return [(start.qual, kind, builder) for start, kind, _why, builder in module.verdicts]


# Round 4's cases (attacker D): the contract that reads each, its modules, and what its findings hold --
# None for a legitimate spelling that must give none.
_ROUND4 = {
    'C1 dict value read back': ('xe8', {
        'opti_oignon/m.py': ('\n'
            'import os\n'
            '\n'
            '_VARS = {"key": "OPTI_ENCRYPTION_KEY", "passphrase": '
            '"OPTI_KEYFILE_PASSPHRASE"}\n'
            '\n'
            '\n'
            'def key_present():\n'
            '    return bool(os.environ.get(_VARS["key"]))\n'
            '\n'
            '\n'
            'def set_passphrase(value):\n'
            '    os.environ[_VARS["passphrase"]] = value\n'),
    }, 'binds OPTI_ENCRYPTION_KEY'),
    "C2 |= a secret's value": ('xe8', {
        'opti_oignon/m.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'def hand_to_library():\n'
            '    os.environ |= {"TOOL_KEY": secret("OPTI_ENCRYPTION_KEY")}\n'),
    }, "puts a secret's value into os.environ |="),
    'C3 module-level put': ('xe8', {
        'opti_oignon/m.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '_KEY = secret("OPTI_ENCRYPTION_KEY")\n'
            'if _KEY:\n'
            '    os.environ["CHROMA_KEY"] = _KEY\n'),
    }, "puts a secret's value into os.environ[...]"),
    'C3 tuple target': ('xe8', {
        'opti_oignon/m.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'def f():\n'
            '    os.environ["TOOL_KEY"], done = secret("OPTI_ENCRYPTION_KEY"), '
            'True\n'
            '    return done\n'),
    }, "puts a secret's value into os.environ[...]"),
    'C4 reader shadowed in an enclosing function': ('xe8', {
        'opti_oignon/m.py': ('\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'def outer():\n'
            '    from os import getenv as secret  # noqa: F811\n'
            '\n'
            '    def inner():\n'
            '        return secret("OPTI_ENCRYPTION_KEY")\n'
            '\n'
            '    return inner()\n'),
    }, 'reads OPTI_ENCRYPTION_KEY (os.getenv)'),
    'C5 reader shadowed by a parameter': ('xe8', {
        'opti_oignon/m.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'def lookup(secret=os.environ.get):  # noqa: F811\n'
            '    return secret("OPTI_ENCRYPTION_KEY")\n'),
    }, 'hands OPTI_ENCRYPTION_KEY (to secret()'),
    'C6 reader shadowed by a nested def': ('xe8', {
        'opti_oignon/m.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'def f():\n'
            '    def secret(name):  # noqa: F811\n'
            '        return os.environ.get(name)\n'
            '\n'
            '    return secret("OPTI_ENCRYPTION_KEY")\n'),
    }, 'hands OPTI_ENCRYPTION_KEY (to secret()'),
    'C6 reader rebound in a function': ('xe8', {
        'opti_oignon/m.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'def f():\n'
            '    secret = os.environ.get  # noqa: F811\n'
            '    return secret("OPTI_ENCRYPTION_KEY")\n'),
    }, 'hands OPTI_ENCRYPTION_KEY (to secret()'),
    'C7 FP dotted import': ('xe8', {
        'opti_oignon/m.py': ('\n'
            'import opti_oignon.process_secrets\n'
            '\n'
            '\n'
            'def key_set():\n'
            '    return '
            'bool(opti_oignon.process_secrets.secret("OPTI_ENCRYPTION_KEY"))\n'),
    }, None),
    'C8 FP import inside a top-level try': ('xe8', {
        'opti_oignon/m.py': ('\n'
            'try:\n'
            '    from opti_oignon.process_secrets import secret\n'
            'except ImportError:  # pragma: no cover\n'
            '    raise\n'
            '\n'
            '\n'
            'def key_set():\n'
            '    return bool(secret("OPTI_ENCRYPTION_KEY"))\n'),
    }, None),
    'C8b FP closure over a function-level import': ('xe8', {
        'opti_oignon/m.py': ('\n'
            'def key_reader():\n'
            '    from opti_oignon.process_secrets import secret\n'
            '\n'
            '    def read():\n'
            '        return secret("OPTI_ENCRYPTION_KEY")\n'
            '\n'
            '    return read\n'),
    }, None),
    'C9 SECRET_NAMES re-exported': ('xe8', {
        'opti_oignon/b.py': ('from opti_oignon.process_secrets import SECRET_NAMES as '
            'PLATFORM_SECRETS  # noqa: F401\n'),
        'opti_oignon/c.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.b import PLATFORM_SECRETS\n'
            '\n'
            '\n'
            'def f():\n'
            '    return os.environ.get(PLATFORM_SECRETS[1])\n'),
    }, 'reads OPTI_ENCRYPTION_KEY (os.environ.get)'),
    'C10a constant re-exported by a star': ('xe8', {
        'opti_oignon/a.py': ('KEY = "OPTI_ENCRYPTION_KEY"\n'),
        'opti_oignon/b.py': ('from opti_oignon.a import *  # noqa: F403\n'),
        'opti_oignon/c.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.b import KEY\n'
            '\n'
            '\n'
            'def f():\n'
            '    return os.environ.get(KEY)\n'),
    }, 'reads OPTI_ENCRYPTION_KEY (os.environ.get)'),
    'C10b constant re-exported in a try': ('xe8', {
        'opti_oignon/a.py': ('KEY = "OPTI_ENCRYPTION_KEY"\n'),
        'opti_oignon/b.py': ('try:\n'
            '    from opti_oignon.a import KEY  # noqa: F401\n'
            'except ImportError:  # pragma: no cover\n'
            '    raise\n'),
        'opti_oignon/c.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.b import KEY\n'
            '\n'
            '\n'
            'def f():\n'
            '    return os.environ.get(KEY)\n'),
    }, 'reads OPTI_ENCRYPTION_KEY (os.environ.get)'),
    'C10c constant re-exported under an alias': ('xe8', {
        'opti_oignon/a.py': ('KEY = "OPTI_ENCRYPTION_KEY"\n'),
        'opti_oignon/b.py': ('from opti_oignon.a import KEY as MASTER_KEY_VAR  # noqa: F401\n'),
        'opti_oignon/c.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.b import MASTER_KEY_VAR\n'
            '\n'
            '\n'
            'def f():\n'
            '    return os.environ.get(MASTER_KEY_VAR)\n'),
    }, 'reads OPTI_ENCRYPTION_KEY (os.environ.get)'),
    'C10d constant through two modules sorted backwards': ('xe8', {
        'opti_oignon/z.py': ('KEY = "OPTI_ENCRYPTION_KEY"\n'),
        'opti_oignon/y.py': ('from opti_oignon.z import KEY  # noqa: F401\n'),
        'opti_oignon/x.py': ('from opti_oignon.y import KEY  # noqa: F401\n'),
        'opti_oignon/c.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.x import KEY\n'
            '\n'
            '\n'
            'def f():\n'
            '    return os.environ.get(KEY)\n'),
    }, 'reads OPTI_ENCRYPTION_KEY (os.environ.get)'),
    'C13 a method named like a log call': ('xe8', {
        'opti_oignon/m.py': ('\n'
            'import os\n'
            '\n'
            '\n'
            'class Settings:\n'
            '    def info(self, name):\n'
            '        return os.environ.get(name)\n'
            '\n'
            '\n'
            'SETTINGS = Settings()\n'
            '\n'
            '\n'
            'def key_set():\n'
            '    return bool(SETTINGS.info("OPTI_ENCRYPTION_KEY"))\n'),
    }, 'hands OPTI_ENCRYPTION_KEY (to info()'),
    "C16 a held value as a variable's name": ('xe8', {
        'opti_oignon/m.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'def mark():\n'
            '    os.environ[secret("OPTI_ENCRYPTION_KEY")] = "1"\n'
            '    os.putenv(secret("OPTI_KEYFILE_PASSPHRASE"), "1")\n'),
    }, "puts a secret's value into os.putenv"),
    'C11a FP launcher method from a subclass': ('xe2', {
        'opti_oignon/m.py': ('\n'
            'import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'class Runner:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'class ToolRunner(Runner):\n'
            '    def version(self):\n'
            '        env = child_env()\n'
            '        self._spawn(["tool", "--version"], env=env)\n'
            '        return subprocess.run(["tool", "--help"], env=env)\n'),
    }, 'is handed env handed as env= to _spawn()'),
    'C11b FP launcher method through an instance': ('xe2', {
        'opti_oignon/m.py': ('\n'
            'import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'class Runner:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'def version():\n'
            '    env = child_env()\n'
            '    Runner()._spawn(["tool", "--version"], env=env)\n'
            '    return subprocess.run(["tool", "--help"], env=env)\n'),
    }, 'is handed env handed as env= to _spawn()'),
    'C14 subprocess._fork_exec': ('xe2', {
        'opti_oignon/m.py': ('\n'
            'import os\n'
            'import subprocess\n'
            '\n'
            '\n'
            'def f(argv):\n'
            '    r, w = os.pipe()\n'
            '    return subprocess._fork_exec(argv, [os.fsencode(argv[0])], True, '
            '(), None, None, -1, -1, -1, -1,\n'
            '                                 -1, -1, r, w, False, False, -1, None, '
            'None, None, -1, None)\n'),
    }, 'subprocess._fork_exec in f() is handed a copy'),
    'C14b from subprocess import _fork_exec': ('xe2', {
        'opti_oignon/m.py': ('\n'
            'import os\n'
            'from subprocess import _fork_exec\n'
            '\n'
            '\n'
            'def f(argv, env_list):\n'
            '    r, w = os.pipe()\n'
            '    return _fork_exec(argv, [os.fsencode(argv[0])], True, (), None, '
            'env_list, -1, -1, -1, -1,\n'
            '                      -1, -1, r, w, False, False, -1, None, None, '
            'None, -1, None)\n'),
    }, 'subprocess._fork_exec in f()'),
    'C15 extra or os.environ': ('xe2', {
        'opti_oignon/m.py': ('\n'
            'import os\n'
            'import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'def run_tool(argv, extra=None):\n'
            '    env = child_env()\n'
            '    env.update(extra or os.environ)\n'
            '    return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'def version():\n'
            '    return run_tool(["tool", "--version"])\n'),
    }, 'merges os.environ'),
    'C12 merger called only by its retry': ('xe4', {
        'opti_oignon/tools/__init__.py': ('from .runner import run_tool  # noqa: F401\n'),
        'opti_oignon/tools/runner.py': ('\n'
            'import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'def run_tool(argv, extra=None, retries=1):\n'
            '    env = child_env()\n'
            '    if extra:\n'
            '        env.update(extra)\n'
            '    done = subprocess.run(argv, env=env)\n'
            '    if done.returncode and retries:\n'
            '        return run_tool(argv, extra, retries - 1)\n'
            '    return done\n'),
        'opti_oignon/other.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.tools import run_tool\n'
            '\n'
            '\n'
            'def version():\n'
            '    return run_tool(["tool", "--version"], dict(os.environ))\n'),
    }, 'merges os.environ'),
    'C18 a merger handed on by a merger in another module': ('xe4', {
        'opti_oignon/a.py': ('\n'
            'import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'def run_tool(argv, extra=None):\n'
            '    env = child_env()\n'
            '    if extra:\n'
            '        env.update(extra)\n'
            '    return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'def version():\n'
            '    return run_tool(["tool", "--version"], {"LANG": "C"})\n'),
        'opti_oignon/b.py': ('\n'
            'from opti_oignon.a import run_tool\n'
            '\n'
            '\n'
            'def run_quiet(argv, extra=None):\n'
            '    return run_tool(argv + ["--quiet"], extra)\n'),
        'opti_oignon/c.py': ('\n'
            'import os\n'
            '\n'
            'from opti_oignon.b import run_quiet\n'
            '\n'
            '\n'
            'def f():\n'
            '    return run_quiet(["tool"], dict(os.environ))\n'),
    }, 'merges os.environ'),
    "C17 a subinterpreter's start": ('xe5', {
        'opti_oignon/m.py': ('\n'
            'import _interpreters\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'def run_tool():\n'
            '    code = ("import subprocess\\n"\n'
            '            f"subprocess.run([\'tool\'], env={{\'TOOL_KEY\': '
            '{secret(\'OPTI_ENCRYPTION_KEY\')!r}}})\\n")\n'
            '    return _interpreters.exec(_interpreters.create(), code)\n'),
    }, '_interpreters.create in run_tool, 0 listed'),
}


_ROUND5 = {
    "G1 a logger's name rebound in the function that calls it": ('xe8', {
        'opti_oignon/m.py': ('import logging\n'
            'import os\n'
            '\n'
            'logger = logging.getLogger(__name__)\n'
            '\n'
            '\n'
            'class Settings:\n'
            '    def info(self, name):\n'
            '        return os.environ.get(name)\n'
            '\n'
            '\n'
            'def key_set():\n'
            '    logger = Settings()\n'
            '    return bool(logger.info("OPTI_ENCRYPTION_KEY"))\n'),
    }, 'hands OPTI_ENCRYPTION_KEY (to info()'),
    "G2 an attribute named like another class's logger": ('xe8', {
        'opti_oignon/m.py': ('import logging\n'
            'import os\n'
            '\n'
            '\n'
            'class Audit:\n'
            '    def __init__(self):\n'
            '        self.log = logging.getLogger(__name__)\n'
            '\n'
            '\n'
            'class Reader:\n'
            '    def info(self, name):\n'
            '        return os.environ.get(name)\n'
            '\n'
            '\n'
            'class Settings:\n'
            '    def __init__(self):\n'
            '        self.log = Reader()\n'
            '\n'
            '    def key_set(self):\n'
            '        return bool(self.log.info("OPTI_ENCRYPTION_KEY"))\n'),
    }, 'hands OPTI_ENCRYPTION_KEY (to info()'),
    'L1 a launcher method through any receiver': ('xe2', {
        'opti_oignon/m.py': ('import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'class Runner:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'def version(tools):\n'
            '    env = child_env()\n'
            '    tools._spawn(["tool", "--version"], env=env)\n'
            '    return subprocess.run(["tool", "--help"], env=env)\n'),
        'opti_oignon/n.py': ('from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'class Tools:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        env.update(TOOL_KEY=secret("OPTI_ENCRYPTION_KEY"))\n'),
    }, 'is handed env handed as env= to _spawn()'),
    "L1b control: round 3's fixture, both classes in one module": ('xe2', {
        'opti_oignon/m.py': ('import os\n'
            'import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'class Runner:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'class Recorder:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        env.update(os.environ)\n'
            '        env["BASH_ENV"] = "/tmp/hook.sh"\n'
            '\n'
            '\n'
            'def f(runner, recorder):\n'
            '    env = child_env()\n'
            '    recorder._spawn(["x"], env=env)\n'
            '    subprocess.run(["bash", "-c", "true"], env=env)\n'
            '    return runner._spawn(["y"], env=child_env())\n'),
    }, 'is handed env handed as env= to _spawn()'),
    'L1b the other class in another module adds BASH_ENV': ('xe2', {
        'opti_oignon/m.py': ('import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'class Runner:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'def f(runner, recorder):\n'
            '    env = child_env()\n'
            '    recorder._spawn(["x"], env=env)\n'
            '    subprocess.run(["bash", "-c", "true"], env=env)\n'
            '    return runner._spawn(["y"], env=child_env())\n'),
        'opti_oignon/n.py': ('class Recorder:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        env["BASH_ENV"] = "/tmp/hook.sh"\n'),
    }, 'is handed env handed as env= to _spawn()'),
    'L2 a launcher method overridden in the subclass that calls it': ('xe2', {
        'opti_oignon/m.py': ('import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env, secret\n'
            '\n'
            '\n'
            'class Runner:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'class ToolRunner(Runner):\n'
            '    def _spawn(self, argv, env=None):\n'
            '        env.update(TOOL_KEY=secret("OPTI_ENCRYPTION_KEY"))\n'
            '\n'
            '    def version(self):\n'
            '        env = child_env()\n'
            '        self._spawn(["tool", "--version"], env=env)\n'
            '        return subprocess.run(["tool", "--help"], env=env)\n'),
    }, 'is handed env handed as env= to _spawn()'),
    'L3 a base of the same name from another module': ('xe2', {
        'opti_oignon/m.py': ('import subprocess\n'
            '\n'
            'from opti_oignon import n\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'class Runner:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'class Tool(n.Runner):\n'
            '    def version(self):\n'
            '        env = child_env()\n'
            '        self._spawn(["tool", "--version"], env=env)\n'
            '        return subprocess.run(["tool", "--help"], env=env)\n'),
        'opti_oignon/n.py': ('from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'class Runner:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        env.update(TOOL_KEY=secret("OPTI_ENCRYPTION_KEY"))\n'),
    }, 'is handed env handed as env= to _spawn()'),
    'L4 a launcher handing its environment on to a launcher of another module': ('xe4', {
        'opti_oignon/a.py': ('import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'def run(argv, env):\n'
            '    return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'def version():\n'
            '    return run(["tool", "--version"], child_env())\n'),
        'opti_oignon/b.py': ('from opti_oignon.a import run\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'def run_quiet(argv, env):\n'
            '    return run(argv + ["--quiet"], env)\n'
            '\n'
            '\n'
            'def version():\n'
            '    return run_quiet(["tool", "--version"], child_env())\n'),
        'opti_oignon/c.py': ('import os\n'
            '\n'
            'from opti_oignon.b import run_quiet\n'
            '\n'
            '\n'
            'def f():\n'
            '    return run_quiet(["tool"], dict(os.environ))\n'),
    }, 'the call of run_quiet() at line 7 hands env an expression that reads os.environ'),
    'L4 control: the copy handed to the first launcher itself': ('xe4', {
        'opti_oignon/a.py': ('import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'def run(argv, env):\n'
            '    return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'def version():\n'
            '    return run(["tool", "--version"], child_env())\n'),
        'opti_oignon/c.py': ('import os\n'
            '\n'
            'from opti_oignon.a import run\n'
            '\n'
            '\n'
            'def f():\n'
            '    return run(["tool"], dict(os.environ))\n'),
    }, 'the call of run() at line 7 hands env an expression that reads os.environ'),
    'L5 a merger called through its package, beside a caller of its own module': ('xe4', {
        'opti_oignon/other.py': ('import os\n'
            '\n'
            'from opti_oignon.tools import run_tool\n'
            '\n'
            '\n'
            'def f():\n'
            '    return run_tool(["tool"], dict(os.environ))\n'),
        'opti_oignon/tools/__init__.py': ('from .runner import run_tool  # noqa: F401\n'),
        'opti_oignon/tools/runner.py': ('import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'def run_tool(argv, extra=None):\n'
            '    env = child_env()\n'
            '    if extra:\n'
            '        env.update(extra)\n'
            '    return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'def version():\n'
            '    return run_tool(["tool", "--version"], {"LANG": "C"})\n'),
    }, 'merges os.environ'),
    'M10 a tuple target fed a name bound to a tuple': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'def f():\n'
            '    pair = (secret("OPTI_ENCRYPTION_KEY"), True)\n'
            '    os.environ["TOOL_KEY"], done = pair\n'
            '    return done\n'),
    }, "puts a secret's value into os.environ[...]"),
    'M11 an annotated module-level value put at module level': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '_KEY: str = secret("OPTI_ENCRYPTION_KEY")\n'
            'os.environ["CHROMA_KEY"] = _KEY\n'),
    }, "puts a secret's value into os.environ[...]"),
    'M16 a constant handed on in the except branch of a top-level try': ('xe8', {
        'opti_oignon/a.py': ('KEY = "OPTI_ENCRYPTION_KEY"\n'),
        'opti_oignon/b.py': ('try:\n'
            '    from opti_oignon.fast import KEY  # noqa: F401\n'
            'except ImportError:\n'
            '    from opti_oignon.a import KEY  # noqa: F401\n'),
        'opti_oignon/c.py': ('import os\n'
            '\n'
            'from opti_oignon.b import KEY\n'
            '\n'
            '\n'
            'def f():\n'
            '    return os.environ.get(KEY)\n'),
    }, 'reads OPTI_ENCRYPTION_KEY (os.environ.get)'),
    "M2 a launcher's method name called through self in an unrelated class": ('xe2', {
        'opti_oignon/m.py': ('import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env, secret\n'
            '\n'
            '\n'
            'class Runner:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'class Other:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        env.update(TOOL_KEY=secret("OPTI_ENCRYPTION_KEY"))\n'
            '\n'
            '    def version(self):\n'
            '        env = child_env()\n'
            '        self._spawn(["tool", "--version"], env=env)\n'
            '        return subprocess.run(["tool", "--help"], env=env)\n'),
    }, 'is handed env handed as env= to _spawn()'),
    'M4 os.environ.get handed the name in a spread mapping': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            '\n'
            'def key_present():\n'
            '    return bool(os.environ.get(**{"key": "OPTI_ENCRYPTION_KEY"}))\n'),
    }, 'reads OPTI_ENCRYPTION_KEY (os.environ.get)'),
    'N0 control: the plain read': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            '\n'
            'def key_present():\n'
            '    return bool(os.getenv("OPTI_ENCRYPTION_KEY"))\n'),
    }, 'reads OPTI_ENCRYPTION_KEY (os.getenv)'),
    'N1 dict() around the mapping, bound and read back': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            '_VARS = dict({"key": "OPTI_ENCRYPTION_KEY", "passphrase": "OPTI_KEYFILE_PASSPHRASE"})\n'
            '\n'
            '\n'
            'def key_present():\n'
            '    return bool(os.environ.get(_VARS["key"]))\n'),
    }, 'hands OPTI_ENCRYPTION_KEY (to dict()'),
    'N2 MappingProxyType around the mapping, bound and read back': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            'import types\n'
            '\n'
            '_VARS = types.MappingProxyType({"key": "OPTI_ENCRYPTION_KEY"})\n'
            '\n'
            '\n'
            'def key_present():\n'
            '    return bool(os.environ.get(_VARS["key"]))\n'),
    }, 'hands OPTI_ENCRYPTION_KEY (to MappingProxyType()'),
    'N3 a mapping handed back, read back by its caller': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            '\n'
            'def _variables():\n'
            '    return {"key": "OPTI_ENCRYPTION_KEY"}\n'
            '\n'
            '\n'
            'def key_present():\n'
            '    return bool(os.environ.get(_variables()["key"]))\n'),
    }, 'uses OPTI_ENCRYPTION_KEY (where only the holder'),
    'N4 a mapping handed on to a helper that reads the environment with it': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            '\n'
            'def _present(variables):\n'
            '    return bool(os.environ.get(variables["key"]))\n'
            '\n'
            '\n'
            'def key_present():\n'
            '    return _present({"key": "OPTI_ENCRYPTION_KEY"})\n'),
    }, 'hands OPTI_ENCRYPTION_KEY (to _present()'),
    'N5 os.getenv handed the name in a spread mapping': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            '\n'
            'def key_present():\n'
            '    return bool(os.getenv(**{"key": "OPTI_ENCRYPTION_KEY"}))\n'),
    }, 'reads OPTI_ENCRYPTION_KEY (os.getenv)'),
    'P0 control: module-level value put at module level': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '_KEY = secret("OPTI_ENCRYPTION_KEY")\n'
            'os.environ["CHROMA_KEY"] = _KEY\n'),
    }, "puts a secret's value into os.environ[...]"),
    "P1 += a secret's value into os.environ[...]": ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'def hand_to_library():\n'
            '    os.environ["TOOL_ARGS"] += " --key " + secret("OPTI_ENCRYPTION_KEY")\n'),
    }, "puts a secret's value into os.environ[...]"),
    'P2 module-level value bound in a top-level try, put at module level': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            'try:\n'
            '    _KEY = secret("OPTI_ENCRYPTION_KEY")\n'
            'except RuntimeError:\n'
            '    _KEY = None\n'
            'if _KEY:\n'
            '    os.environ["CHROMA_KEY"] = _KEY\n'),
    }, "puts a secret's value into os.environ[...]"),
    'P3 module-level value put inside a function': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '_KEY = secret("OPTI_ENCRYPTION_KEY")\n'
            '\n'
            '\n'
            'def configure():\n'
            '    os.environ["CHROMA_KEY"] = _KEY\n'),
    }, "puts a secret's value into os.environ[...]"),
    'R1 class names rebound into a cycle through second bases': ('xe2', {
        'opti_oignon/m.py': ('import subprocess\n'
            '\n'
            'from opti_oignon.process_secrets import child_env\n'
            '\n'
            '\n'
            'class Runner:\n'
            '    def _spawn(self, argv, env=None):\n'
            '        return subprocess.run(argv, env=env)\n'
            '\n'
            '\n'
            'class P:\n'
            '    pass\n'
            '\n'
            '\n'
            'class Q:\n'
            '    pass\n'
            '\n'
            '\n'
            'class B:\n'
            '    pass\n'
            '\n'
            '\n'
            'class A(P, B):\n'
            '    def version(self):\n'
            '        env = child_env()\n'
            '        self._spawn(["tool", "--version"], env=env)\n'
            '        return subprocess.run(["tool", "--help"], env=env)\n'
            '\n'
            '\n'
            'class B(Q, A):  # noqa: F811\n'
            '    pass\n'),
    }, 'is handed env handed as env= to _spawn()'),
    'S0 control: the reader at module level': ('xe8', {
        'opti_oignon/m.py': ('from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'class Settings:\n'
            '    key_present = bool(secret("OPTI_ENCRYPTION_KEY"))\n'),
    }, None),
    'S1 reader shadowed in a class body': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'class Settings:\n'
            '    secret = os.environ.get  # noqa: F811\n'
            '    key_present = bool(secret("OPTI_ENCRYPTION_KEY"))\n'),
    }, 'hands OPTI_ENCRYPTION_KEY (to secret()'),
    'S2 reader rebound by a function through global': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'def _read_the_environment():\n'
            '    global secret\n'
            '    secret = os.environ.get\n'
            '\n'
            '\n'
            'def key_present():\n'
            '    return bool(secret("OPTI_ENCRYPTION_KEY"))\n'),
    }, 'hands OPTI_ENCRYPTION_KEY (to secret()'),
    'S3 reader rebound by a nested function through nonlocal': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            '\n'
            'def key_present():\n'
            '    from opti_oignon.process_secrets import secret\n'
            '\n'
            '    def swap():\n'
            '        nonlocal secret\n'
            '        secret = os.environ.get\n'
            '\n'
            '    swap()\n'
            '    return bool(secret("OPTI_ENCRYPTION_KEY"))\n'),
    }, 'hands OPTI_ENCRYPTION_KEY (to secret()'),
    'S4 reader defined again in a class body': ('xe8', {
        'opti_oignon/m.py': ('import os\n'
            '\n'
            'from opti_oignon.process_secrets import secret\n'
            '\n'
            '\n'
            'class Settings:\n'
            '    def secret(name):  # noqa: F811\n'
            '        return os.environ.get(name)\n'
            '\n'
            '    key_present = bool(secret("OPTI_ENCRYPTION_KEY"))\n'),
    }, 'hands OPTI_ENCRYPTION_KEY (to secret()'),
}

# The cases round 5's integration adds for the code it wrote: each new branch has one only it decides.
_READS5 = "import os\n\nfrom opti_oignon.process_secrets import secret\n\n\n"
_STARTS5 = "import subprocess\n\nfrom opti_oignon.process_secrets import child_env\n\n\n"
_PUT5 = "puts a secret's value into os.environ[...]"
_READER5 = "hands OPTI_ENCRYPTION_KEY (to secret()"
_INTEGRATION5 = {
    "a value bound under global, put by another function": ("xe8", {
        "opti_oignon/m.py": _READS5 + "_KEY = None\n\n\ndef load():\n    global _KEY\n"
                                      "    _KEY = secret('OPTI_ENCRYPTION_KEY')\n\n\n"
                                      "def export():\n    os.environ['TOOL_KEY'] = _KEY\n"}, _PUT5),
    "a value bound in the function a put is nested in": ("xe8", {
        "opti_oignon/m.py": _READS5 + "def outer():\n    value = secret('OPTI_ENCRYPTION_KEY')\n\n"
                                      "    def inner():\n        os.environ['TOOL_KEY'] = value\n\n"
                                      "    inner()\n"}, _PUT5),
    "a parameter named like a module's secret value, put": ("xe8", {
        "opti_oignon/m.py": _READS5 + "_KEY = secret('OPTI_ENCRYPTION_KEY')\n\n\ndef export(_KEY):\n"
                                      "    os.environ['TOOL_NAME'] = _KEY\n"}, None),
    "a reader rebound by a comprehension": ("xe8", {
        "opti_oignon/m.py": _READS5 + "def f():\n"
                                      "    return [secret('OPTI_ENCRYPTION_KEY') for secret in (os.getenv,)]\n"},
        "reads OPTI_ENCRYPTION_KEY (os.getenv)"),
    "a reader rebound by a lambda's parameter": ("xe8", {
        "opti_oignon/m.py": _READS5 + "def f():\n"
                                      "    return (lambda secret: secret('OPTI_ENCRYPTION_KEY'))(os.getenv)\n"},
        _READER5),
    "a reader rebound by a match capture": ("xe8", {
        "opti_oignon/m.py": _READS5 + "def f(reader):\n    match reader:\n        case secret:\n"
                                      "            return secret('OPTI_ENCRYPTION_KEY')\n"}, _READER5),
    # A type parameter is Python 3.12's syntax: the census reads it where the interpreter does.
    **({"a reader rebound by a type parameter": ("xe8", {
        "opti_oignon/m.py": "from opti_oignon.process_secrets import secret\n\n\ndef f[secret]():\n"
                            "    return secret('OPTI_ENCRYPTION_KEY')\n"}, _READER5)}
       if sys.version_info >= (3, 12) else {}),
    "a launcher's method its module overrides, called through self in its own class": ("xe2", {
        "opti_oignon/m.py": _STARTS5 + "class Runner:\n    def _spawn(self, argv, env=None):\n"
                                       "        return subprocess.run(argv, env=env)\n\n"
                                       "    def version(self):\n        env = child_env()\n"
                                       "        self._spawn(['tool', '--version'], env=env)\n"
                                       "        return subprocess.run(['tool', '--help'], env=env)\n\n\n"
                                       "class Quiet(Runner):\n    def _spawn(self, argv, env=None):\n"
                                       "        env['BASH_ENV'] = '/tmp/hook.sh'\n"},
        "is handed env handed as env= to _spawn()"),
    "a constant of a module four levels down, read by another": ("xe8", {
        "opti_oignon/allium/ref/organs/vault.py": "KEY = 'OPTI_ENCRYPTION_KEY'\n",
        "opti_oignon/m.py": "import os\n\nfrom opti_oignon.allium.ref.organs.vault import KEY\n\n\ndef f():\n"
                            "    return os.environ.get(KEY)\n"}, "reads OPTI_ENCRYPTION_KEY (os.environ.get)"),
    "a start module reached through a module four levels down": ("xe2", {
        "opti_oignon/allium/ref/organs/tool.py": "import subprocess  # noqa: F401\n",
        "opti_oignon/m.py": "from opti_oignon.allium.ref.organs import tool\n\n\ndef f():\n"
                            "    return tool.subprocess.run(['true'])\n"},
        "subprocess.run in f() is handed no env=: the child is handed the whole environment"),
    # Since a call through a module that hands it on is read, round 4's C12 has a caller: this one has none.
    "a merger called by its own retry alone": ("xe4", {
        "opti_oignon/tools.py": _STARTS5 + "def run_tool(argv, extra=None, retries=1):\n    env = child_env()\n"
                                           "    if extra:\n        env.update(extra)\n"
                                           "    done = subprocess.run(argv, env=env)\n"
                                           "    if done.returncode and retries:\n"
                                           "        return run_tool(argv, extra, retries - 1)\n    return done\n"},
        "is called nowhere in the package"),
    "a launcher's method reached through a class its package hands on": ("xe4", {
        "opti_oignon/tools/__init__.py": "from .runner import Runner  # noqa: F401\n",
        "opti_oignon/tools/runner.py": _STARTS5 + "class Runner:\n    def _spawn(self, argv, env=None):\n"
                                                  "        return subprocess.run(argv, env=env)\n\n"
                                                  "    def version(self):\n"
                                                  "        return self._spawn(['tool', '--version'],\n"
                                                  "                           env=child_env())\n",
        "opti_oignon/other.py": "from opti_oignon.tools import Runner\n\n\ndef version():\n"
                                "    return Runner()._spawn(['tool', '--version'])\n"},
        "the call of _spawn() at line 5 leaves env to its default"),
}
_ROUNDS = {"round 4": _ROUND4, "round 5": _ROUND5, "round 5's integration": _INTEGRATION5}
_ROUND4_FINDERS = {"xe2": ("unbuilt_starts", "copied_environments"), "xe4": ("unbuilt_starts", "copied_environments"),
                   "xe5": ("unlisted_inheriting",), "xe8": ("secret_reads",)}


def _round4(guard, tmp_path, contract):
    """Each case of rounds 4 and 5 this contract reads: its findings hold the fragment it is written with, or none."""
    cases = [(label, name, entry) for label, table in _ROUNDS.items() for name, entry in table.items()]
    for index, (label, name, (owner, modules, fragment)) in enumerate(cases):
        if owner != contract:
            continue
        estate = guard.read_estate(_estate(tmp_path / f"cases_{contract}_{index}", modules))
        found = [line for question in _ROUND4_FINDERS[contract]
                 for line in getattr(guard, "find_" + question)(estate)]
        if fragment is None:
            assert found == [], (label, name, found)
        else:
            assert any(fragment in line for line in found), (label, name, fragment, found)


# ---------------------------------------------------------------------------
# XE1 -- the census counts every process-start spelling, and nothing else
# ---------------------------------------------------------------------------
_STARTS = {
    "run": ("import subprocess\n\ndef f():\n    subprocess.run(['true'])\n", [("direct", "subprocess.run")]),
    "Popen, module aliased": ("import subprocess as sp\n\ndef f():\n    sp.Popen(['true'])\n",
                              [("direct", "subprocess.Popen")]),
    "call, from-imported": ("from subprocess import call\n\ndef f():\n    call(['true'])\n",
                            [("direct", "subprocess.call")]),
    "check_call, renamed": ("from subprocess import check_call as cc\n\ndef f():\n    cc(['true'])\n",
                            [("direct", "subprocess.check_call")]),
    "check_output": ("import subprocess\n\ndef f():\n    return subprocess.check_output(['uname'])\n",
                     [("direct", "subprocess.check_output")]),
    "getoutput": ("import subprocess\n\ndef f():\n    return subprocess.getoutput('ls')\n",
                  [("inheriting", "subprocess.getoutput")]),
    "getstatusoutput": ("import subprocess\n\ndef f():\n    return subprocess.getstatusoutput('ls')\n",
                        [("inheriting", "subprocess.getstatusoutput")]),
    "system": ("import os\n\ndef f():\n    os.system('ls')\n", [("inheriting", "os.system")]),
    "popen": ("import os\n\ndef f():\n    return os.popen('ls')\n", [("inheriting", "os.popen")]),
    "execv": ("import os\n\ndef f():\n    os.execv('/bin/true', ['true'])\n", [("inheriting", "os.execv")]),
    "execve": ("import os\n\ndef f(env):\n    os.execve('/bin/true', ['true'], env)\n", [("direct", "os.execve")]),
    "execlpe": ("import os\n\ndef f(env):\n    os.execlpe('true', 'true', env)\n", [("direct", "os.execlpe")]),
    "spawnlp": ("import os\n\ndef f():\n    os.spawnlp(os.P_WAIT, 'true', 'true')\n", [("inheriting", "os.spawnlp")]),
    "spawnve": ("import os\n\ndef f(env):\n    os.spawnve(os.P_WAIT, '/bin/true', ['true'], env)\n",
                [("direct", "os.spawnve")]),
    "posix_spawn": ("import os\n\ndef f(env):\n    os.posix_spawn('/bin/true', ['true'], env)\n",
                    [("direct", "os.posix_spawn")]),
    "fork": ("import os\n\ndef f():\n    return os.fork()\n", [("inheriting", "os.fork")]),
    "forkpty": ("import os\n\ndef f():\n    return os.forkpty()\n", [("inheriting", "os.forkpty")]),
    "pty.spawn": ("import pty\n\ndef f():\n    pty.spawn(['sh'])\n", [("inheriting", "pty.spawn")]),
    "pty.fork": ("from pty import fork\n\ndef f():\n    return fork()\n", [("inheriting", "pty.fork")]),
    "create_subprocess_exec": ("import asyncio\n\nasync def f():\n    await asyncio.create_subprocess_exec('true')\n",
                               [("direct", "asyncio.create_subprocess_exec")]),
    "create_subprocess_shell": ("import asyncio\n\nasync def f():\n    await asyncio.create_subprocess_shell('ls')\n",
                                [("direct", "asyncio.create_subprocess_shell")]),
    "Process": ("import multiprocessing\n\ndef f():\n    multiprocessing.Process(target=print).start()\n",
                [("inheriting", "multiprocessing.Process")]),
    "Pool, from-imported": ("from multiprocessing import Pool\n\ndef f():\n    return Pool(2)\n",
                            [("inheriting", "multiprocessing.Pool")]),
    "Manager": ("import multiprocessing\n\ndef f():\n    return multiprocessing.Manager()\n",
                [("inheriting", "multiprocessing.Manager")]),
    "a spawn context's queue": (
        "import multiprocessing\n\ndef f():\n    return multiprocessing.get_context('spawn').SimpleQueue()\n",
        [("inheriting", "multiprocessing.get_context().SimpleQueue")]),
    "a context bound, then its lock": (
        "import multiprocessing\n\ndef f():\n    ctx = multiprocessing.get_context('spawn')\n    return ctx.Lock()\n",
        [("inheriting", "multiprocessing.get_context().Lock")]),
    "a context a function returns": (
        "import multiprocessing\n\ndef _ctx():\n    return multiprocessing.get_context('spawn')\n\n"
        "def f():\n    return _ctx().Process(target=print)\n",
        [("inheriting", "multiprocessing.get_context().Process")]),
    "ProcessPoolExecutor": ("from concurrent.futures import ProcessPoolExecutor\n\ndef f():\n"
                            "    return ProcessPoolExecutor(1)\n",
                            [("inheriting", "concurrent.futures.ProcessPoolExecutor")]),
    "ProcessPoolExecutor, dotted": ("import concurrent.futures\n\ndef f():\n"
                                    "    return concurrent.futures.ProcessPoolExecutor(1)\n",
                                    [("inheriting", "concurrent.futures.ProcessPoolExecutor")]),
    "webbrowser.open": ("import webbrowser\n\ndef f(u):\n    webbrowser.open(u)\n",
                        [("inheriting", "webbrowser.open")]),
    "webbrowser.open_new_tab": ("from webbrowser import open_new_tab\n\ndef f(u):\n    open_new_tab(u)\n",
                                [("inheriting", "webbrowser.open_new_tab")]),
    "find_library": ("import ctypes.util\n\ndef f():\n    return ctypes.util.find_library('c')\n",
                     [("inheriting", "ctypes.util.find_library")]),
    "find_library at import": ("from ctypes.util import find_library\n\nNAME = find_library('m')\n",
                               [("inheriting", "ctypes.util.find_library")]),
    "processor": ("import platform\n\ndef f():\n    return platform.processor()\n",
                  [("inheriting", "platform.processor")]),
    "platform.platform": ("import platform\n\ndef f():\n    return platform.platform()\n",
                          [("inheriting", "platform.platform")]),
    "architecture": ("import platform\n\ndef f():\n    return platform.architecture()\n",
                     [("inheriting", "platform.architecture")]),
    "uname().processor": ("import platform\n\ndef f():\n    return platform.uname().processor\n",
                          [("inheriting", "platform.uname()")]),
    "uname, unpacked": ("import platform\n\ndef f():\n    system, *_rest = platform.uname()\n    return system\n",
                        [("inheriting", "platform.uname()")]),
    "getnode": ("import uuid\n\ndef f():\n    return uuid.getnode()\n", [("inheriting", "uuid.getnode")]),
    "uuid1": ("from uuid import uuid1\n\ndef f():\n    return uuid1()\n", [("inheriting", "uuid.uuid1")]),
    "import oqs": ("def f():\n    import oqs\n    return oqs\n", [("inheriting", "import oqs")]),
    "from oqs": ("from oqs import Signature\n", [("inheriting", "import oqs")]),
    "import_module oqs": ("import importlib\n\ndef f():\n    return importlib.import_module('oqs')\n",
                          [("inheriting", "import oqs")]),
    "getattr": ("import subprocess\n\ndef f():\n    getattr(subprocess, 'run')(['true'])\n",
                [("direct", "subprocess.run")]),
    "the module cache": ("import sys\n\ndef f():\n    sys.modules['subprocess'].Popen(['true'])\n",
                         [("direct", "subprocess.Popen")]),
    "import_module subprocess": ("import importlib\n\ndef f():\n"
                                 "    importlib.import_module('subprocess').run(['true'])\n",
                                 [("direct", "subprocess.run")]),
    "star import": ("from subprocess import *\n\ndef f():\n    run(['true'])\n", [("direct", "subprocess.run")]),
    "a loop's subprocess_exec": ("import asyncio\n\nasync def f():\n    loop = asyncio.get_running_loop()\n"
                                 "    await loop.subprocess_exec(asyncio.SubprocessProtocol, 'true')\n",
                                 [("direct", "asyncio.loop.subprocess_exec")]),
    "a subclass of Popen": ("import subprocess\n\nclass Tracked(subprocess.Popen):\n    pass\n\n"
                            "def f():\n    return Tracked(['true'])\n", [("direct", "subprocess.Popen")]),
    "an attribute bound to run": ("import subprocess\n\nclass R:\n    def __init__(self):\n"
                                  "        self.runner = subprocess.run\n\n    def go(self):\n"
                                  "        self.runner(['true'])\n", [("direct", "subprocess.run")]),
    "a partial of run": ("import functools\nimport subprocess\n\ndef f():\n"
                         "    go = functools.partial(subprocess.run, ['true'])\n    go()\n",
                         [("direct", "subprocess.run")]),
    "a browser's open": ("import webbrowser\n\ndef f(u):\n    webbrowser.get().open(u)\n",
                          [("inheriting", "webbrowser.get().open")]),
    "joblib.Parallel": ("from joblib import Parallel, delayed\n\ndef f(items):\n"
                        "    return Parallel(n_jobs=2)(delayed(len)(item) for item in items)\n",
                        [("inheriting", "joblib.Parallel")]),
    "asyncio.subprocess's own": ("import asyncio.subprocess\nfrom asyncio.subprocess import create_subprocess_shell\n\n"
                                 "async def f():\n    await asyncio.subprocess.create_subprocess_exec('true')\n"
                                 "    await create_subprocess_shell('true')\n",
                                 [("direct", "asyncio.subprocess.create_subprocess_exec"),
                                  ("direct", "asyncio.subprocess.create_subprocess_shell")]),
    "posix": ("import posix\n\ndef f():\n    posix.system('true')\n    return posix.fork()\n",
              [("inheriting", "os.system"), ("inheriting", "os.fork")]),
    "shared memory and the tracker": (
        "from multiprocessing import resource_tracker\nfrom multiprocessing.shared_memory import SharedMemory\n\n"
        "def f():\n    block = SharedMemory(create=True, size=8)\n    resource_tracker.ensure_running()\n"
        "    return block\n",
        [("inheriting", "multiprocessing.shared_memory.SharedMemory"),
         ("inheriting", "multiprocessing.resource_tracker.ensure_running")]),
    "a start module reached through another": (
        "import ctypes.util\nimport webbrowser\n\ndef f():\n    ctypes.util.subprocess.run(['true'])\n"
        "    webbrowser.subprocess.Popen(['true'])\n    webbrowser.os.system('true')\n",
        [("direct", "subprocess.run"), ("direct", "subprocess.Popen"), ("inheriting", "os.system")]),
    "a start module reached through the package": (
        "from opti_oignon import sandbox_manager\n\ndef f():\n    sandbox_manager.subprocess.run(['true'])\n"
        "    sandbox_manager.os.system('true')\n",
        [("direct", "subprocess.run"), ("inheriting", "os.system")]),
    "the C library's system": ("import ctypes\n\ndef f():\n    ctypes.CDLL(None).system(b'true')\n",
                               [("inheriting", "ctypes.CDLL().system")]),
    "a browser built": ("import webbrowser\n\ndef f(url):\n    webbrowser.BackgroundBrowser('xdg-open').open(url)\n",
                        [("inheriting", "webbrowser.BackgroundBrowser().open")]),
    "starts bound by a tuple": ("import subprocess\n\nrun, popen = subprocess.run, subprocess.Popen\n\n"
                                "def f():\n    run(['true'])\n    popen(['true'])\n",
                                [("direct", "subprocess.run"), ("direct", "subprocess.Popen")]),
    "class attributes": ("import subprocess\n\nclass Tool:\n    runner = subprocess.run\n"
                         "    _popen = staticmethod(subprocess.Popen)\n\n    def go(self):\n"
                         "        self.runner(['true'])\n        Tool.runner(['true'])\n        self._popen(['true'])\n",
                         [("direct", "subprocess.run"), ("direct", "subprocess.run"), ("direct", "subprocess.Popen")]),
    "a dispatch table": ("import os\nimport subprocess\n\nRUNNERS = {'run': subprocess.run, 'system': os.system}\n\n"
                         "def f():\n    RUNNERS['run'](['true'])\n    RUNNERS['system']('true')\n",
                         [("direct", "subprocess.run"), ("direct", "subprocess.run")]),
    "multiprocessing's process classes and spawner": (
        "import multiprocessing.context\nimport multiprocessing.util\n\n"
        "def f():\n    return multiprocessing.context.SpawnProcess(target=print)\n\n"
        "def g(fds):\n    return multiprocessing.util.spawnv_passfds('/bin/true', ['true'], fds)\n",
        [("inheriting", "multiprocessing.context.SpawnProcess"), ("inheriting", "multiprocessing.util.spawnv_passfds")]),
    "a C library's function read by its name": ("import ctypes\n\ndef f():\n    ctypes.CDLL(None)['system'](b'true')\n",
                                                [("inheriting", "ctypes.CDLL().system")]),
    "a loop's start, asyncio imported or not": ("def f(loop):\n    return loop.subprocess_exec(None, 'true')\n",
                                                [("direct", "asyncio.loop.subprocess_exec")]),
    "a loop's start taken as a value": (
        "import functools\n\ndef f(loop):\n    run = loop.subprocess_exec\n    return run(None, 'true')\n\n"
        "def g(loop):\n    return functools.partial(loop.subprocess_shell, None)('true')\n\n"
        "def h(loop):\n    return getattr(loop, 'subprocess_exec')(None, 'true')\n",
        [("direct", "asyncio.loop.subprocess_exec, taken as a value"),
         ("direct", "asyncio.loop.subprocess_shell, taken as a value"),
         ("direct", "asyncio.loop.subprocess_exec, taken as a value")]),
    "os reached through a module of the package": ("from opti_oignon.sandbox_manager import os\n\ndef f():\n"
                                                   "    os.system('true')\n", [("inheriting", "os.system")]),
    "os reached through ctypes": ("from ctypes.util import os as _os\n\ndef f():\n    _os.execv('/bin/true', ['true'])\n",
                                  [("inheriting", "os.execv")]),
    "the stdlib's fork_exec": ("import _posixsubprocess\n\ndef f(argv, env_list):\n"
                               "    return _posixsubprocess.fork_exec(argv, [argv[0]], True, (), None, env_list)\n",
                               [("direct", "_posixsubprocess.fork_exec")]),
}
_NOT_STARTS = {
    "prose": '"""subprocess.run(["x"]) and os.system("y") and import oqs, in prose."""\n',
    "a string": "CODE = 'subprocess.Popen(argv); os.fork()'\n",
    "a local run": "def run(argv):\n    return argv\n\ndef f():\n    run(['x'])\n",
    "a method named Popen": "class S:\n    def Popen(self, argv):\n        return argv\n\ndef f(s):\n    s.Popen(['x'])\n",
    "a uname field": "import platform\n\ndef f():\n    return platform.uname().system, platform.machine()\n",
    "cpu_count": "import multiprocessing\n\ndef f():\n    return multiprocessing.cpu_count()\n",
    "a context alone": "import multiprocessing\n\nCTX = multiprocessing.get_context('spawn')\n",
    "an environment read": "import os\n\ndef f():\n    return os.environ.get('HOME')\n",
    "which": "import shutil\n\ndef f():\n    return shutil.which('bwrap')\n",
    "uuid4": "import uuid\n\ndef f():\n    return uuid.uuid4()\n",
    "the current process": "import multiprocessing\n\ndef f():\n    return multiprocessing.current_process().name\n",
    "a C library's other function read by its name": "import ctypes\n\ndef f():\n    return ctypes.CDLL(None)['strnlen'](b'x', 1)\n",
    "a partial of print": "import functools\n\ndef f():\n    functools.partial(print, 'x')()\n",
}


def test_xe1_the_census_counts_every_process_start_spelling_and_nothing_else():
    guard, restore = _load()
    try:
        for name, (text, expected) in _STARTS.items():
            module = _module(guard, "opti_oignon/m.py", text)
            assert [(start.kind, start.qual) for start in module.starts] == expected, name
        for name, text in _NOT_STARTS.items():
            assert _module(guard, "opti_oignon/m.py", text).starts == [], name
        # Where a start sits: its function, or the module itself.
        located = _module(guard, "opti_oignon/m.py", _STARTS["find_library at import"][0]).starts[0]
        assert (located.line, located.function) == (3, "<module>"), located
        located = _module(guard, "opti_oignon/m.py", _STARTS["run"][0]).starts[0]
        assert (located.line, located.function) == (4, "f"), located
    finally:
        restore()


# ---------------------------------------------------------------------------
# XE2 -- every direct start is handed a built environment
# ---------------------------------------------------------------------------
_IMPORTED = "import subprocess\nfrom opti_oignon.process_secrets import child_env\n\n"
_BUILT = {
    "child_env imported": ("opti_oignon/m.py", _IMPORTED + "def f():\n    subprocess.run(['true'], env=child_env())\n",
                           [("subprocess.run", None, "child_env")]),
    "child_env by a relative import": (
        "opti_oignon/sub/m.py",
        "import subprocess\nfrom .. import process_secrets\n\n"
        "def f():\n    subprocess.run(['true'], env=process_secrets.child_env())\n",
        [("subprocess.run", None, "child_env")]),
    "child_env imported in the function": (
        "opti_oignon/m.py",
        "import subprocess\n\ndef f():\n    from opti_oignon.process_secrets import child_env\n\n"
        "    subprocess.run(['true'], env=child_env())\n",
        [("subprocess.run", None, "child_env")]),
    "a name bound once": (
        "opti_oignon/m.py",
        _IMPORTED + "def f():\n    env = child_env()\n    subprocess.run(['a'], env=env)\n"
        "    subprocess.run(['b'], env=env)\n",
        [("subprocess.run", None, "child_env"), ("subprocess.run", None, "child_env")]),
    "an allowlist in its module": (
        "opti_oignon/sandbox_manager.py",
        "import subprocess\n\ndef _tempdir_env(workspace):\n    return {'PATH': '/bin', 'HOME': workspace}\n\n"
        "def f(workspace):\n    subprocess.Popen(['true'], env=_tempdir_env(workspace))\n",
        [("subprocess.Popen", None, "allowlist")]),
    "an allowlist changed, never rebound": (
        "opti_oignon/plugin_subprocess.py",
        "import subprocess\n\ndef _build_plugin_env(extra):\n    return dict(extra)\n\n"
        "def f(extra):\n    env = _build_plugin_env({})\n    env['HOME'] = '/tmp'\n    env.update(extra)\n"
        "    subprocess.Popen(['true'], env=env)\n",
        [("subprocess.Popen", None, "allowlist")]),
    "an environment's length logged": ("opti_oignon/m.py", "import logging\n" + _IMPORTED + "def f():\n    env = child_env()\n"
                                        "    logging.info('%d', len(env))\n    subprocess.run(['true'], env=env)\n",
                                        [("subprocess.run", None, "child_env")]),
    "a mapping merged with |": ("opti_oignon/m.py",
                                _IMPORTED + "def f():\n    subprocess.run(['true'], env=child_env() | {'A': 'b'})\n",
                                [("subprocess.run", None, "child_env")]),
    "execve's environment by keyword": ("opti_oignon/m.py", "import os\nfrom opti_oignon.process_secrets import child_env\n\n"
                                        "def f():\n    os.execve('/bin/true', ['true'], env=child_env())\n",
                                        [("os.execve", None, "child_env")]),
    "a merger's extra or {}": ("opti_oignon/m.py", _IMPORTED + "def run_tool(argv, extra=None):\n    env = child_env()\n"
                               "    env.update(extra or {})\n    return subprocess.run(argv, env=env)\n\n\n"
                               "def version():\n    return run_tool(['tool', '--version'], {'LANG': 'C'})\n",
                               [("subprocess.run", None, "child_env")]),
    "a merger's retry": ("opti_oignon/m.py", _IMPORTED + "def run_tool(argv, extra=None, retries=1):\n    env = child_env()\n"
                         "    if extra:\n        env.update(extra)\n    done = subprocess.run(argv, env=env)\n"
                         "    if done.returncode and retries:\n        return run_tool(argv, extra, retries - 1)\n"
                         "    return done\n\n\ndef version():\n    return run_tool(['tool', '--version'], {'LANG': 'C'})\n",
                         [("subprocess.run", None, "child_env")]),
    "child_env's extra read": ("opti_oignon/m.py",
                               _IMPORTED + "def f():\n    subprocess.run(['true'], env=child_env({'TOOL_HOME': '/x'}))\n",
                               [("subprocess.run", None, "child_env")]),
    "an environment handed by position": (
        "opti_oignon/m.py", "import os\nfrom opti_oignon.process_secrets import child_env\n\n"
        "def f():\n    os.execve('/bin/true', ['true'], child_env())\n",
        [("os.execve", None, "child_env")]),
}
_UNBUILT = {
    "no env": ("opti_oignon/m.py", _IMPORTED + "def f():\n    subprocess.run(['true'])\n", "no env="),
    "a child_env of its own": (
        "opti_oignon/m.py",
        "import subprocess\n\ndef child_env():\n    return {}\n\ndef f():\n    subprocess.run(['true'], env=child_env())\n",
        "not opti_oignon.process_secrets'"),
    "an allowlist called elsewhere": (
        "opti_oignon/m.py",
        "import subprocess\n\ndef _tempdir_env(workspace):\n    return {}\n\n"
        "def f(w):\n    subprocess.run(['true'], env=_tempdir_env(w))\n",
        "not the one opti_oignon/sandbox_manager.py defines once"),
    "a name bound twice": (
        "opti_oignon/m.py", _IMPORTED + "def f():\n    env = child_env()\n    env = {'PATH': '/bin'}\n"
        "    subprocess.run(['true'], env=env)\n", "bound 2 time(s)"),
    "a literal": ("opti_oignon/m.py", _IMPORTED + "def f():\n    subprocess.run(['true'], env={'PATH': '/bin'})\n",
                  "no builder made"),
    "**kwargs": ("opti_oignon/m.py", _IMPORTED + "def f(options):\n    subprocess.run(['true'], **options)\n",
                 "hidden in **kwargs"),
    "a module-level name": ("opti_oignon/m.py", _IMPORTED + "ENV = child_env()\n\n"
                            "def f():\n    subprocess.run(['true'], env=ENV)\n", "bound nowhere in f()"),
    "an execve with no environment built": (
        "opti_oignon/m.py", "import os\n\ndef f():\n    os.execve('/bin/true', ['true'], {'A': 'b'})\n",
        "no builder made"),
    "a secret put back": ("opti_oignon/m.py", _IMPORTED + "def f(value):\n    env = child_env()\n"
                          "    env['OPTI_ENCRYPTION_KEY'] = value\n    subprocess.run(['true'], env=env)\n",
                          "never carries it"),
    "a shell hook put back": ("opti_oignon/m.py", _IMPORTED + "def f():\n    env = child_env()\n"
                              "    env.update({'BASH_ENV': '/tmp/hook.sh'})\n    subprocess.run(['true'], env=env)\n",
                              "sets BASH_ENV"),
    "a key from a variable": ("opti_oignon/m.py", _IMPORTED + "def f(pairs):\n    env = child_env()\n"
                              "    for name, value in pairs:\n        env[name] = value\n"
                              "    subprocess.run(['true'], env=env)\n", "a key this census cannot read"),
    "an alias": ("opti_oignon/m.py", _IMPORTED + "def f():\n    env = child_env()\n    alias = env\n"
                 "    alias['X'] = 'y'\n    subprocess.run(['true'], env=env)\n", "cannot follow it"),
    "a helper it is handed to": ("opti_oignon/m.py", _IMPORTED + "def _merge(env):\n    env.clear()\n\n"
                                 "def f():\n    env = child_env()\n    _merge(env)\n"
                                 "    subprocess.run(['true'], env=env)\n", "cannot follow it"),
    "a builder shadowed": ("opti_oignon/m.py", _IMPORTED + "def child_env(extra=None):  # noqa: F811\n"
                           "    return {}\n\ndef f():\n    subprocess.run(['true'], env=child_env())\n",
                           "binds to something else as well"),
    "a builder rebound": ("opti_oignon/m.py", _IMPORTED + "def f(other):\n    child_env = other\n"
                          "    subprocess.run(['true'], env=child_env())\n", "binds to something else as well"),
    "an allowlist bound again": (
        "opti_oignon/sandbox_manager.py",
        "import subprocess\n\ndef _tempdir_env(workspace):\n    return {'PATH': '/bin'}\n\n_tempdir_env = dict\n\n"
        "def f(w):\n    subprocess.Popen(['true'], env=_tempdir_env(w))\n",
        "not the one opti_oignon/sandbox_manager.py defines once"),
    "handed by env= to a helper": ("opti_oignon/m.py", _IMPORTED + "def _prepare(env=None):\n    env.clear()\n\n"
                                   "def f():\n    env = child_env()\n    _prepare(env=env)\n    subprocess.run(['true'], env=env)\n",
                                   "handed as env= to _prepare()"),
    "changed by a closure": ("opti_oignon/m.py", _IMPORTED + "def f():\n    env = child_env()\n\n    def poison():\n"
                             "        env['X'] = 'y'\n\n    poison()\n    subprocess.run(['true'], env=env)\n",
                             "a function nested in f()"),
    "a nested function's default": ("opti_oignon/m.py", _IMPORTED + "def f():\n    env = child_env()\n\n    def poison(e=env):\n"
                                    "        e['X'] = 'y'\n\n    poison()\n    subprocess.run(['true'], env=env)\n",
                                    "a function nested in f()"),
    "a nested function's default of the same name": (
        "opti_oignon/m.py", _IMPORTED + "def f():\n    env = child_env()\n\n    def poison(env=env):\n"
        "        env['X'] = 'y'\n\n    poison()\n    subprocess.run(['true'], env=env)\n", "a function nested in f()"),
    "a shell variable merged by |": ("opti_oignon/m.py",
                                     _IMPORTED + "def f():\n    subprocess.run(['true'], env=child_env() | {'BASH_ENV': '/x'})\n",
                                     "sets BASH_ENV"),
    "copied by dict()": ("opti_oignon/m.py", _IMPORTED + "def f():\n    env = child_env()\n    other = dict(env)\n"
                         "    other['BASH_ENV'] = '/x'\n    subprocess.run(['true'], env=env)\n", "cannot follow it"),
    "a secret's value under another name": (
        "opti_oignon/m.py", "import subprocess\nfrom opti_oignon.process_secrets import child_env, secret\n\n"
        "def f():\n    env = child_env()\n    env['SQLCIPHER_KEY'] = secret('OPTI_ENCRYPTION_KEY')\n"
        "    subprocess.run(['true'], env=env)\n", "set to a secret's value"),
    "a secret's value in child_env's extra": (
        "opti_oignon/m.py", "import subprocess\nfrom opti_oignon.process_secrets import child_env, secret\n\n"
        "def f():\n    subprocess.run(['true'], env=child_env({'TOOL_PASSPHRASE': secret('OPTI_KEYFILE_PASSPHRASE')}))\n",
        "sets TOOL_PASSPHRASE to a secret's value"),
    "a secret's name in child_env's extra": (
        "opti_oignon/m.py", _IMPORTED + "def f():\n    subprocess.run(['true'], env=child_env({'OPTI_ENCRYPTION_KEY': 'x'}))\n",
        "never carries"),
    "a shell variable by setdefault": ("opti_oignon/m.py", _IMPORTED + "def f():\n    env = child_env()\n"
                                       "    env.setdefault('BASH_ENV', '/tmp/hook.sh')\n    subprocess.run(['true'], env=env)\n",
                                       "setdefault() (line 6) sets BASH_ENV"),
    "a shell variable by an update's keyword": ("opti_oignon/m.py", _IMPORTED + "def f():\n    env = child_env()\n"
                                                "    env.update(BASH_ENV='/tmp/hook.sh')\n    subprocess.run(['true'], env=env)\n",
                                                "sets BASH_ENV"),
    "a secret's value through a name": (
        "opti_oignon/m.py", "import subprocess\nfrom opti_oignon.process_secrets import child_env, secret\n\n"
        "def f():\n    value = secret('OPTI_ENCRYPTION_KEY')\n    env = child_env()\n    env['TOOL'] = value\n"
        "    subprocess.run(['true'], env=env)\n", "set to a secret's value"),
    "a len of its own": ("opti_oignon/m.py", _IMPORTED + "def len(env):  # noqa: A001\n    env.clear()\n    return 0\n\n\n"
                         "def f():\n    env = child_env()\n    len(env)\n    subprocess.run(['true'], env=env)\n",
                         "cannot follow it"),
    "child_env's extra by keyword": ("opti_oignon/m.py", _IMPORTED + "def f():\n"
                                     "    subprocess.run(['true'], env=child_env(extra={'BASH_ENV': '/x'}))\n", "sets BASH_ENV"),
    "a merger handed on as a value": ("opti_oignon/m.py", _IMPORTED + "def run_tool(argv, extra=None):\n    env = child_env()\n"
                                      "    if extra:\n        env.update(extra)\n    return subprocess.run(argv, env=env)\n\n\n"
                                      "HANDLER = run_tool\n", "handed on as a value"),
}


def test_xe2_every_direct_start_is_handed_a_built_environment(tmp_path):
    guard, restore = _load()
    try:
        _round4(guard, tmp_path, "xe2")
        for name, (rel, text, expected) in _BUILT.items():
            assert _verdicts(_module(guard, rel, text)) == expected, name
        for name, (rel, text, why) in _UNBUILT.items():
            module = _module(guard, rel, text)
            assert len(module.verdicts) == 1, name
            start, kind, said, _builder = module.verdicts[0]
            assert kind == "unbuilt" and why in said, (name, kind, said)
        shared = _module(guard, "opti_oignon/m.py", _IMPORTED.replace("import subprocess\n", "import os\nimport subprocess\n")
                         + "class Runner:\n    def _spawn(self, argv, env=None):\n        return subprocess.run(argv, env=env)\n\n\n"
                         "class Recorder:\n    def _spawn(self, argv, env=None):\n        env.update(os.environ)\n\n\n"
                         "def f(runner, recorder):\n    env = child_env()\n    recorder._spawn(['x'], env=env)\n"
                         "    subprocess.run(['true'], env=env)\n    return runner._spawn(['y'], env=child_env())\n")
        assert [kind for _s, kind, _w, _b in shared.verdicts] == [None, "unbuilt"], shared.verdicts
        assert "handed as env= to _spawn()" in shared.verdicts[1][2], \
            "a method of another class that shares a launcher's name is no launcher"
        rebound = _module(guard, "opti_oignon/m.py", _IMPORTED.replace("import subprocess\n", "import os\nimport subprocess\n")
                          + "def run_tool(argv, extra=None):\n    env = child_env()\n    extra = dict(os.environ)\n"
                          "    env.update(extra)\n    return subprocess.run(argv, env=env)\n\n\n"
                          "def version():\n    return run_tool(['tool'])\n")
        assert [kind for _s, kind, _w, _b in rebound.verdicts] == ["copy"], "a merger's parameter rebound is read as bound"
        estate = guard.read_estate(REPO)
        assert guard.find_unbuilt_starts(estate) == []
        built = [verdict for rel, module in guard.take_census(estate).items() if rel not in guard.EXEMPT
                 for verdict in module.verdicts]
        assert len(built) >= 18 and all(kind is None for _start, kind, _why, _builder in built), built
    finally:
        restore()


# ---------------------------------------------------------------------------
# XE3 -- a start handed a copy of the process's environment
# ---------------------------------------------------------------------------
_COPIES = {
    "None": "env=None",
    "os.environ": "env=os.environ",
    "dict(os.environ)": "env=dict(os.environ)",
    "dict(os.environ, ...)": "env=dict(os.environ, A='b')",
    "os.environ.copy()": "env=os.environ.copy()",
    "{**os.environ}": "env={**os.environ, 'A': 'b'}",
    "os.environ | {...}": "env=os.environ | {'A': 'b'}",
    "a comprehension": "env={k: v for k, v in os.environ.items()}",
}


def test_xe3_a_start_handed_a_copy_of_the_process_environment_is_a_finding():
    guard, restore = _load()
    try:
        for name, given in _COPIES.items():
            text = f"import os\nimport subprocess\n\ndef f():\n    subprocess.run(['true'], {given})\n"
            assert [kind for _s, kind, _w, _b in _module(guard, "opti_oignon/m.py", text).verdicts] == ["copy"], name
        bound = "import os\nimport subprocess\n\ndef f():\n    env = os.environ.copy()\n    subprocess.run(['true'], env=env)\n"
        assert [kind for _s, kind, _w, _b in _module(guard, "opti_oignon/m.py", bound).verdicts] == ["copy"]
        aliased = "import subprocess\nfrom os import environ\n\ndef f():\n    subprocess.run(['true'], env=dict(environ))\n"
        assert [kind for _s, kind, _w, _b in _module(guard, "opti_oignon/m.py", aliased).verdicts] == ["copy"]
        merged = (_IMPORTED.replace("import subprocess\n", "import os\nimport subprocess\n")
                  + "def f():\n    env = child_env()\n    env.update(os.environ)\n    subprocess.run(['true'], env=env)\n")
        start, kind, why, _builder = _module(guard, "opti_oignon/m.py", merged).verdicts[0]
        assert kind == "copy" and "env.update()" in why, why
        head = "import copy\nimport os\nimport subprocess\nfrom opti_oignon.process_secrets import child_env\n\n"
        for given in ("os.environ.items()", "copy.copy(os.environ)", "dict(os.environ.items())",
                      "{'A': os.environ['HOME']}"):
            text = head + f"def f():\n    env = child_env()\n    env.update({given})\n    subprocess.run(['t'], env=env)\n"
            assert [kind for _s, kind, _w, _b in _module(guard, "opti_oignon/m.py", text).verdicts] == ["copy"], given
        launcher = head + ("class M:\n    def _spawn(self, argv, *, env=None):\n        env.update(os.environ)\n"
                           "        return subprocess.Popen(argv, env=env)\n\n    def a(self):\n"
                           "        return self._spawn(['x'], env=child_env())\n")
        assert [kind for _s, kind, _w, _b in _module(guard, "opti_oignon/m.py", launcher).verdicts] == ["copy"]
        for body in ("    env = child_env()\n    env['A'] = os.environ['HOME']\n",
                     "    extra = dict(os.environ)\n    env = child_env()\n    env.update(extra)\n"):
            text = head + "def f():\n" + body + "    subprocess.run(['t'], env=env)\n"
            assert [kind for _s, kind, _w, _b in _module(guard, "opti_oignon/m.py", text).verdicts] == ["copy"], body
        estate = guard.read_estate(REPO)
        assert guard.find_copied_environments(estate) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# XE4 -- a launcher's environment parameter is followed to its callers
# ---------------------------------------------------------------------------
_LAUNCHER = (_IMPORTED + "class M:\n    def _spawn(self, argv, *, env=None):\n"
             "        return subprocess.Popen(argv, env=env)\n\n")


def test_xe4_a_launchers_environment_parameter_is_followed_to_its_callers(tmp_path):
    guard, restore = _load()
    try:
        _round4(guard, tmp_path, "xe4")
        filled = _LAUNCHER + ("    def a(self):\n        return self._spawn(['x'], env=child_env())\n\n"
                              "    def b(self):\n        env = child_env()\n        return self._spawn(['y'], env=env)\n")
        assert _verdicts(_module(guard, "opti_oignon/m.py", filled)) == [("subprocess.Popen", None, "launcher")]
        positional = (_IMPORTED + "def _spawn(argv, env):\n    subprocess.Popen(argv, env=env)\n\n"
                      "def a():\n    _spawn(['x'], child_env())\n")
        assert _verdicts(_module(guard, "opti_oignon/m.py", positional)) == [("subprocess.Popen", None, "launcher")]
        cases = {
            "left to its default": (_LAUNCHER + "    def a(self):\n        return self._spawn(['x'])\n",
                                    "unbuilt", "leaves env to its default"),
            "a copy handed on": (_LAUNCHER + "    def a(self):\n        import os\n"
                                 "        return self._spawn(['x'], env=dict(os.environ))\n", "copy", "reads os.environ"),
            "handed on as a value": (_LAUNCHER + "    def a(self):\n        return [self._spawn]\n",
                                     "unbuilt", "handed on as a value"),
            "called nowhere": (_LAUNCHER, "unbuilt", "called nowhere"),
            "the second caller": (_LAUNCHER + "    def a(self):\n        return self._spawn(['x'], env=child_env())\n\n"
                                  "    def b(self):\n        return self._spawn(['y'])\n",
                                  "unbuilt", "leaves env to its default"),
        }
        for name, (text, kind, why) in cases.items():
            _start, found, said, _builder = _module(guard, "opti_oignon/m.py", text).verdicts[0]
            assert found == kind and why in said, (name, found, said)
        elsewhere = _estate(tmp_path / "elsewhere", {
            "opti_oignon/home.py": _LAUNCHER + "    def a(self):\n        return self._spawn(['x'], env=child_env())\n",
            "opti_oignon/caller.py": "from opti_oignon.home import M\n\ndef f(manager: M):\n"
                                     "    return manager._spawn(['y'])\n",
            "opti_oignon/own.py": "from opti_oignon.home import M  # noqa: F401\n\n\nclass N:\n"
                                  "    def __init__(self, spawn):\n        self._spawn = spawn\n\n"
                                  "    def go(self):\n        self._spawn(print)\n"})
        assert guard.find_unbuilt_starts(guard.read_estate(elsewhere)) == [
            "opti_oignon/caller.py:4: the call of _spawn() at line 4 leaves env to its default"]
        values = _estate(tmp_path / "values", {
            "opti_oignon/home.py": _LAUNCHER + "    def a(self):\n        return self._spawn(['x'], env=child_env())\n",
            "opti_oignon/bound.py": "from opti_oignon.home import M\n\ndef f(manager: M):\n    spawn = manager._spawn\n"
                                    "    return spawn(['y'])\n",
            "opti_oignon/read.py": "from opti_oignon.home import M\n\ndef f(manager: M):\n"
                                   "    return getattr(manager, '_spawn')(['y'])\n"})
        assert guard.find_unbuilt_starts(guard.read_estate(values)) == [
            "opti_oignon/bound.py:4: _spawn() of opti_oignon/home.py taken as a value: its callers cannot be followed",
            "opti_oignon/read.py:4: _spawn() of opti_oignon/home.py read through getattr: its callers cannot be followed"]
        functions = _estate(tmp_path / "functions", {
            "opti_oignon/home.py": _IMPORTED + "def spawn(argv, env=None):\n    return subprocess.Popen(argv, env=env)\n\n"
                                   "def a():\n    return spawn(['x'], env=child_env())\n",
            "opti_oignon/caller.py": "from opti_oignon.home import spawn\n\ndef f():\n    return spawn(['y'])\n",
            "opti_oignon/other.py": "import shlex\n\ndef spawn(x):\n    return shlex.split(x)\n\ndef f():\n    return spawn('a b')\n"})
        assert guard.find_unbuilt_starts(guard.read_estate(functions)) == [
            "opti_oignon/caller.py:4: the call of spawn() at line 4 leaves env to its default"]
        named_run = _estate(tmp_path / "named_run", {
            "opti_oignon/tool.py": _IMPORTED + "class Tool:\n    def run(self, argv, *, env=None):\n"
                                   "        return subprocess.run(argv, env=env)\n\n    def version(self):\n"
                                   "        return self.run(['tool', '--version'], env=child_env())\n",
            "opti_oignon/app.py": "import asyncio\n\ndef f(adapter):\n    asyncio.run(adapter.main())\n    return adapter.run('x')\n",
            "opti_oignon/serve.py": "import uvicorn\nfrom opti_oignon.tool import Tool\n\ndef f():\n    uvicorn.run(Tool)\n"})
        assert guard.find_unbuilt_starts(guard.read_estate(named_run)) == [], \
            "a launcher named run: neither subprocess.run in its module nor asyncio.run elsewhere is one of its calls"
        merger = (_IMPORTED + "class Tool:\n    def start(self, argv, extra=None):\n        env = child_env()\n"
                  "        if extra:\n            env.update(extra)\n        return subprocess.run(argv, env=env)\n")
        mergers = _estate(tmp_path / "mergers", {
            "opti_oignon/tool.py": merger,
            "opti_oignon/caller.py": "import os\nfrom opti_oignon.tool import Tool\n\ndef f(tool: Tool):\n"
                                     "    tool.start(['x'], extra=dict(os.environ))\n"
                                     "    tool.start(['y'], extra={'BASH_ENV': '/x'})\n"
                                     "    return tool.start(['z'], extra={'TOOL_HOME': '/x'})\n"})
        assert guard.find_unbuilt_starts(guard.read_estate(mergers)) == [
            "opti_oignon/caller.py:5: env.update() (line 8) merges os.environ",
            "opti_oignon/caller.py:6: env.update() (line 8) sets BASH_ENV, which a child's environment never carries"]
        defaulted = _estate(tmp_path / "defaulted", {"opti_oignon/tool.py": merger.replace(
            "extra=None", "extra={'BASH_ENV': '/x'}").replace("        if extra:\n            env.update(extra)\n",
                                                           "        env.update(extra)\n") + (
            "\n\ndef f(tool: Tool):\n    return tool.start(['x'])\n")})
        assert guard.find_unbuilt_starts(guard.read_estate(defaulted)) == [
            "opti_oignon/tool.py:8: subprocess.run in start() is handed env.update() (line 7), by its default, sets "
            "BASH_ENV, which a child's environment never carries"]
        nowhere = _estate(tmp_path / "nowhere", {"opti_oignon/tool.py": merger})
        assert guard.find_unbuilt_starts(guard.read_estate(nowhere)) == [
            "opti_oignon/tool.py:5: env.update() (line 8): start() takes extra and is called nowhere in the package, so "
            "what it merges cannot be read"]
        census = guard.take_census(guard.read_estate(REPO))
        tracked = [(start.function, kind, builder) for start, kind, _why, builder
                   in census["opti_oignon/sandbox_manager.py"].verdicts if start.function == "_spawn_tracked"]
        assert tracked == [("_spawn_tracked", None, "launcher")], tracked
    finally:
        restore()


# ---------------------------------------------------------------------------
# XE5 -- every inheriting start is listed with its reason and its contract
# ---------------------------------------------------------------------------
def test_xe5_every_inheriting_start_is_listed_with_its_reason_and_its_contract(tmp_path):
    guard, restore = _load()
    try:
        _round4(guard, tmp_path, "xe5")
        estate = guard.read_estate(REPO)
        census = guard.take_census(estate)
        assert guard.find_unlisted_inheriting(estate, census) == []
        assert guard.find_stale_inheriting(estate, census) == []
        assert guard.find_inheriting_unproven(estate, census) == []
        holding = {rel for rel, module in census.items() if rel not in guard.EXEMPT
                   and any(start.kind == "inheriting" for start in module.starts)}
        assert holding == set(guard.INHERITING), sorted(holding ^ set(guard.INHERITING))
        listed = guard.INHERITING
        try:
            guard.INHERITING = {rel: entry for rel, entry in listed.items() if rel != "opti_oignon/auto_tuner.py"}
            assert guard.find_unlisted_inheriting(estate, census) == [
                "opti_oignon/auto_tuner.py: 1 inheriting start(s) platform.processor in get_hardware_fingerprint, "
                "0 listed"]
            swapped = listed["opti_oignon/auto_tuner.py"]._replace(sites={"get_hardware_fingerprint": ("uuid.getnode",)})
            guard.INHERITING = dict(listed, **{"opti_oignon/auto_tuner.py": swapped})
            assert guard.find_unlisted_inheriting(estate, census) == [
                "opti_oignon/auto_tuner.py: 1 inheriting start(s) platform.processor in get_hardware_fingerprint, "
                "0 listed"]
            assert guard.find_stale_inheriting(estate, census) == [
                "opti_oignon/auto_tuner.py: 1 inheriting start(s) uuid.getnode listed in get_hardware_fingerprint, "
                "the census finds 0; lower the entry"]
            guard.INHERITING = dict(listed, **{"opti_oignon/gone.py": guard.Inheriting(
                sites={"f": ("os.fork",)}, reason="a witness", contracts=("xe5",))})
            assert guard.find_stale_inheriting(estate, census) == [
                "opti_oignon/gone.py: listed as inheriting, but it is gone"]
            guard.INHERITING = dict(listed, **{"opti_oignon/auto_tuner.py": listed["opti_oignon/auto_tuner.py"]._replace(
                contracts=("xe999",))})
            assert guard.find_inheriting_unproven(estate, census) == [
                "opti_oignon/auto_tuner.py: contract xe999 names no test function"]
        finally:
            guard.INHERITING = listed
    finally:
        restore()


# ---------------------------------------------------------------------------
# XE6 -- twin of the egress census
# ---------------------------------------------------------------------------
_EGRESS_SPELLINGS = {
    "run": "import subprocess\n\ndef f():\n    subprocess.run(['curl', 'x'])\n",
    "system": "import os\n\ndef f():\n    os.system('curl x')\n",
    "popen": "import os\n\ndef f():\n    os.popen('curl x')\n",
    "execv": "import os\n\ndef f():\n    os.execv('/usr/bin/curl', ['curl', 'x'])\n",
    "spawnlp": "import os\n\ndef f():\n    os.spawnlp(os.P_WAIT, 'curl', 'curl', 'x')\n",
    "posix_spawn": "import os\n\ndef f(env):\n    os.posix_spawn('/usr/bin/curl', ['curl', 'x'], env)\n",
    "create_subprocess_exec": "import asyncio\n\nasync def f():\n    await asyncio.create_subprocess_exec('curl', 'x')\n",
    "getoutput": "import subprocess\n\ndef f():\n    subprocess.getoutput('curl x')\n",
    "pty.spawn": "import pty\n\ndef f():\n    pty.spawn(['sh'])\n",
}


def test_xe6_every_process_sink_of_the_egress_census_is_a_start_counted_here(tmp_path):
    guard, restore = _load()
    try:
        root = _estate(tmp_path / "tree", {f"opti_oignon/m_{name}.py": text for name, text in _EGRESS_SPELLINGS.items()})
        estate = guard.read_estate(root)
        egress = guard._egress().take_census(estate)
        charged = sum(1 for sites in egress.values() for site in sites if site.kind == "process")
        assert charged == len(_EGRESS_SPELLINGS), egress
        assert guard.find_twin_gaps(estate) == []
        real = guard.read_estate(REPO)
        assert guard.find_twin_gaps(real) == []
        assert sum(1 for sites in guard._egress().take_census(real).values() for site in sites
                   if site.kind == "process") >= 9
        positional = dict(guard._ENV_POSITIONAL)
        try:
            del guard._ENV_POSITIONAL["os.posix_spawn"]
            estate = guard.read_estate(root)
            assert guard.find_twin_gaps(estate) == [
                "opti_oignon/m_posix_spawn.py:4: a process sink of the egress census in f that this census does not "
                "count"]
        finally:
            guard._ENV_POSITIONAL.clear()
            guard._ENV_POSITIONAL.update(positional)
    finally:
        restore()


# ---------------------------------------------------------------------------
# XE7 -- what cannot be read fails the census by name
# ---------------------------------------------------------------------------
def test_xe7_what_cannot_be_read_fails_the_census_by_name(tmp_path):
    guard, restore = _load()
    try:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = guard.run(empty)
        assert result.code == 1 and "nothing was scanned" in result.lines[0], result.lines
        result = guard.run(_estate(tmp_path / "bytes", {"opti_oignon/m.py": b"x = '\xff'\n"}))
        assert result.code == 1 and "could not be read" in result.lines[0] and any(
            "opti_oignon/m.py" in line for line in result.lines), result.lines
        result = guard.run(_estate(tmp_path / "syntax", {"opti_oignon/m.py": "def f(:\n"}))
        assert result.code == 1 and "do not parse" in result.lines[0] and "opti_oignon/m.py" in result.lines[1]
        result = guard.run(_estate(tmp_path / "names", {"opti_oignon/process_secrets.py": "NAMES = ()\n",
                                                        "opti_oignon/m.py": "X = 1\n"}))
        assert result.code == 1 and "writes no SECRET_NAMES" in result.lines[0], result.lines
        result = guard.run(_estate(tmp_path / "hazards", {
            "opti_oignon/process_secrets.py": "SECRET_NAMES = ('A',)\nFUNCTION_PREFIX = 'BASH_FUNC_'\n",
            "opti_oignon/m.py": "X = 1\n"}))
        assert result.code == 1 and "writes no SHELL_HAZARDS" in result.lines[0], result.lines
        result = guard.run(_estate(tmp_path / "suite", {"opti_oignon/m.py": "X = 1\n"},
                                   suites={"test_x.py": b"def test_xe1_():\n    '\xff'\n"}))
        assert result.code == 1 and "could not be read" in result.lines[0] and "suite" in result.lines[0], result.lines
        deep = guard.run(_estate(tmp_path / "deep", {"opti_oignon/a/b/c/d/e/f/m.py": "X = 1\n"}))
        assert deep.code == 1 and "deeper than" in deep.lines[0], deep.lines
        assert deep.lines[1:] == ["  opti_oignon/a/b/c/d/e/f/m.py"], "a module too deep to follow is named"
        level = guard.run(_estate(tmp_path / "level", {"opti_oignon/a/b/c/d/e/m.py": "X = 1\n"}))
        assert not any("deeper than" in line for line in level.lines), level.lines
        locked = _estate(tmp_path / "locked", {"opti_oignon/m.py": "X = 1\n", "opti_oignon/inner/n.py": "Y = 2\n"})
        inner = locked / "opti_oignon" / "inner"
        inner.chmod(0)
        try:
            if os.access(inner, os.R_OK):
                pytest.skip("this process reads a directory whatever its mode")
            result = guard.run(locked)
        finally:
            inner.chmod(stat.S_IRWXU)
        assert result.code == 1 and any("cannot be listed" in line for line in result.lines), result.lines
    finally:
        restore()


# ---------------------------------------------------------------------------
# XE8 -- no module but process_secrets reads a secret's name in the environment
# ---------------------------------------------------------------------------
_SECRET_USES = {
    "get": ("import os\n\ndef f():\n    return os.environ.get('OPTI_ENCRYPTION_KEY')\n", "reads", KEYNAME),
    "subscript": ("import os\n\ndef f():\n    return os.environ['OPTI_KEYFILE_PASSPHRASE']\n", "reads", PASS),
    "getenv by a constant": ("import os\n\nKEY = 'OPTI_ENCRYPTION_KEY'\n\ndef f():\n    return os.getenv(KEY)\n",
                             "reads", KEYNAME),
    "in": ("import os\n\ndef f():\n    return 'OPTI_ENCRYPTION_KEY' in os.environ\n", "reads", KEYNAME),
    "pop": ("import os\n\n_PASS = 'OPTI_KEYFILE_PASSPHRASE'\n\ndef f():\n    os.environ.pop(_PASS, None)\n",
            "writes", PASS),
    "setdefault": ("import os\n\ndef f():\n    os.environ.setdefault('OPTI_ENCRYPTION_KEY', 'x')\n", "writes", KEYNAME),
    "assignment": ("import os\n\ndef f():\n    os.environ['OPTI_ENCRYPTION_KEY'] = 'x'\n", "writes", KEYNAME),
    "del": ("import os\n\ndef f():\n    del os.environ['OPTI_KEYFILE_PASSPHRASE']\n", "writes", PASS),
    "update with a dict": ("import os\n\ndef f():\n    os.environ.update({'OPTI_ENCRYPTION_KEY': 'x'})\n",
                           "writes", KEYNAME),
    "update with a keyword": ("import os\n\ndef f():\n    os.environ.update(OPTI_KEYFILE_PASSPHRASE='x')\n",
                              "writes", PASS),
    "putenv": ("import os\n\ndef f():\n    os.putenv('OPTI_ENCRYPTION_KEY', 'x')\n", "writes", KEYNAME),
    "unsetenv": ("import os\n\ndef f():\n    os.unsetenv('OPTI_KEYFILE_PASSPHRASE')\n", "writes", PASS),
    "environ imported": ("from os import environ\n\ndef f():\n    return environ.get('OPTI_ENCRYPTION_KEY')\n",
                         "reads", KEYNAME),
    "environ bound": ("import os\n\ndef f():\n    env = os.environ\n    return env.get('OPTI_KEYFILE_PASSPHRASE')\n",
                      "reads", PASS),
    "constants by a tuple": ("import os\n\nPASS, KEYNAME = 'OPTI_KEYFILE_PASSPHRASE', 'OPTI_ENCRYPTION_KEY'\n\n"
                             "def f():\n    return os.environ.get(KEYNAME)\n", "reads", KEYNAME),
    "an item of the holder's names": ("import os\n\nfrom opti_oignon.process_secrets import SECRET_NAMES\n\n"
                                      "def f():\n    return os.environ.get(SECRET_NAMES[0])\n", "reads", PASS),
    "a helper handed a name": ("import os\n\ndef _env(name):\n    return os.environ.get(name)\n\n"
                               "def key():\n    return _env('OPTI_ENCRYPTION_KEY')\n", "hands", KEYNAME),
    "a parameter's default": ("import os\n\ndef passphrase(name='OPTI_KEYFILE_PASSPHRASE'):\n"
                              "    return os.environ.get(name)\n", "defaults", PASS),
    "a copy's get": ("import os\n\ndef f():\n    return os.environ.copy().get('OPTI_ENCRYPTION_KEY')\n",
                     "reads", KEYNAME),
    "a dict of it, indexed": ("import os\n\ndef f():\n    return dict(os.environ)['OPTI_KEYFILE_PASSPHRASE']\n",
                              "reads", PASS),
    "a scan of its items": ("import os\n\ndef f():\n"
                            "    return next((v for k, v in os.environ.items() if k == 'OPTI_ENCRYPTION_KEY'), None)\n",
                            "compares", KEYNAME),
    "its keys": ("import os\n\ndef f():\n    return 'OPTI_KEYFILE_PASSPHRASE' in os.environ.keys()\n",
                 "reads", PASS),
    "a list of pairs": ("import os\n\ndef f(value):\n    os.environ.update([('OPTI_ENCRYPTION_KEY', value)])\n",
                        "writes", KEYNAME),
    "a copy bound, then indexed": ("import os\n\ndef f():\n    env = dict(os.environ)\n    return env['OPTI_ENCRYPTION_KEY']\n",
                                   "reads", KEYNAME),
    "a membership in a tuple": ("def f(k):\n    return k in ('OPTI_ENCRYPTION_KEY',)\n", "compares", KEYNAME),
    "a match": ("def f(k):\n    match k:\n        case 'OPTI_KEYFILE_PASSPHRASE':\n            return True\n    return False\n",
                "compares", PASS),
    "a reader shadowed": ("import os\nfrom opti_oignon.process_secrets import secret\n\n\ndef secret(name):  # noqa: F811\n"
                          "    return os.environ.get(name)\n\n\ndef f():\n    return secret('OPTI_ENCRYPTION_KEY')\n",
                          "hands", KEYNAME),
    "a reader rebound": ("import os\nfrom opti_oignon.process_secrets import secret\n\nsecret = os.environ.get  # noqa: F811\n\n\n"
                         "def f():\n    return secret('OPTI_KEYFILE_PASSPHRASE')\n", "hands", PASS),
    "a name as a keyword": ("import os\n\ndef f(value):\n    os.environ.update(dict(OPTI_ENCRYPTION_KEY=value))\n",
                            "hands", KEYNAME),
    "a constant bound to another": ("import os\n\nPASS = 'OPTI_KEYFILE_PASSPHRASE'\nNAME = PASS\n\n\n"
                                    "def f():\n    return os.environ.get(NAME)\n", "binds", PASS),
    "a name held in a local": ("from opti_oignon.process_secrets import secret\n\ndef f():\n"
                               "    name = 'OPTI_ENCRYPTION_KEY'\n    return bool(secret(name))\n", "binds", KEYNAME),
    "a name= to another call": ("def helper(name):\n    return name\n\n\ndef f():\n"
                                "    return helper(name='OPTI_ENCRYPTION_KEY')\n", "hands", KEYNAME),
    "a name as a returned mapping's key": ("def f():\n    return {'OPTI_ENCRYPTION_KEY': 'x'}\n", "uses", KEYNAME),
    "a name in two literals": ("import os\n\nKEY = 'OPTI_ENCRYPTION_' 'KEY'\n\n\ndef f():\n    return os.environ.get(KEY)\n",
                               "reads", KEYNAME),
    "a name in bytes": ("import os\n\ndef f():\n    return os.getenvb(b'OPTI_KEYFILE_PASSPHRASE')\n", "reads", PASS),
    "a reader shadowed by an import": ("from opti_oignon.process_secrets import secret\n\ndef f():\n"
                                       "    from os import getenv as secret  # noqa: F811\n"
                                       "    return secret('OPTI_ENCRYPTION_KEY')\n", "reads", KEYNAME),
}


def test_xe8_no_module_but_process_secrets_reads_a_secret_in_the_environment(tmp_path):
    guard, restore = _load()
    try:
        _round4(guard, tmp_path, "xe8")
        for name, (text, verb, secret) in _SECRET_USES.items():
            root = _estate(tmp_path / name.replace(" ", "_"), {"opti_oignon/m.py": text})
            found = guard.find_secret_reads(guard.read_estate(root))
            assert len(found) == 1 and f": {verb} {secret} (" in found[0], (name, found)
        across = _estate(tmp_path / "across", {
            "opti_oignon/encryption.py": "_ENV_KEY_NAME = 'OPTI_ENCRYPTION_KEY'\n",
            "opti_oignon/store.py": "import os\nfrom opti_oignon import encryption\n\n"
                                    "def f():\n    return os.environ.get(encryption._ENV_KEY_NAME)\n"})
        assert guard.find_secret_reads(guard.read_estate(across)) == [
            "opti_oignon/store.py:5: reads OPTI_ENCRYPTION_KEY (os.environ.get); the one read is "
            "opti_oignon.process_secrets.secret()"]
        aliased = _estate(tmp_path / "aliased", {
            "opti_oignon/encryption.py": "_ENV_KEY_NAME = 'OPTI_ENCRYPTION_KEY'\n",
            "opti_oignon/store.py": "import os\nfrom opti_oignon.encryption import _ENV_KEY_NAME as KEY_VAR\n\n"
                                    "def f():\n    return os.getenv(KEY_VAR)\n"})
        assert guard.find_secret_reads(guard.read_estate(aliased)) == [
            "opti_oignon/store.py:5: reads OPTI_ENCRYPTION_KEY (os.getenv); the one read is "
            "opti_oignon.process_secrets.secret()"]
        held = _estate(tmp_path / "held", {
            "opti_oignon/m.py": "from opti_oignon.process_secrets import secret, withdraw\n\n"
                                "def f():\n    withdraw('OPTI_KEYFILE_PASSPHRASE')\n"
                                "    return secret('OPTI_ENCRYPTION_KEY')\n"})
        assert guard.find_secret_reads(guard.read_estate(held)) == [], "the holder's own reads"
        two = _estate(tmp_path / "two", {
            "opti_oignon/m.py": "from opti_oignon.process_secrets import secret\n\n"
                                "def a():\n    name = 'OPTI_ENCRYPTION_KEY'\n    return bool(secret(name))\n\n"
                                "def b():\n    name = 'OPTI_KEYFILE_PASSPHRASE'\n    return bool(secret(name))\n",
            "opti_oignon/n.py": "import os\n\ndef f(name):\n    return os.environ.get(name)\n"})
        found = guard.find_secret_reads(guard.read_estate(two))
        assert [line.split(": ", 1)[1].split(" (")[0] for line in found] == [f"binds {KEYNAME}", f"binds {PASS}"], \
            "one identifier bound to both names: two findings where they are bound, nothing elsewhere, and an end"
        helper = _estate(tmp_path / "helper", {
            "opti_oignon/helper.py": "import os\n\ndef env_value(name):\n    return os.environ.get(name)\n",
            "opti_oignon/caller.py": "from opti_oignon.helper import env_value\n\ndef f():\n    return env_value('OPTI_ENCRYPTION_KEY')\n"})
        assert guard.find_secret_reads(guard.read_estate(helper)) == [
            "opti_oignon/caller.py:4: hands OPTI_ENCRYPTION_KEY (to env_value(), which may read the environment with it); "
            "the one read is opti_oignon.process_secrets.secret()"], "a module whose text never names the environment"
        rebound = _estate(tmp_path / "rebound", {"opti_oignon/m.py": "KEY = 'OPTI_ENCRYPTION_KEY'\nKEY = 'OPTI_ENCRYPTION_KEY'\n"})
        assert guard.find_secret_reads(guard.read_estate(rebound)) == [
            "opti_oignon/m.py:2: binds OPTI_ENCRYPTION_KEY (KEY again, where a constant that names a secret is bound once); "
            "the one read is opti_oignon.process_secrets.secret()"]
        quiet = _estate(tmp_path / "quiet", {
            "opti_oignon/process_secrets.py": _SECRETS_FIXTURE + "\nimport os\n\ndef secret(name):\n"
                                              "    return os.environ.get('OPTI_ENCRYPTION_KEY')\n",
            "opti_oignon/m.py": "import os\nimport logging\n\n_PASS = 'OPTI_KEYFILE_PASSPHRASE'\n\n\ndef f():\n"
                                "    logging.warning('set OPTI_KEYFILE_PASSPHRASE to open it')\n"
                                "    logging.warning(f'set {_PASS} to open it')\n"
                                "    return os.environ.get('HOME')\n",
            "opti_oignon/n.py": "import base64\nimport logging\nimport os\n\nfrom opti_oignon.process_secrets import secret\n\n"
                                "logger = logging.getLogger(__name__)\nKEY = 'OPTI_ENCRYPTION_KEY'\n\n\n"
                                "def f():\n    if not secret(KEY):\n        logger.warning(f'{KEY} is not set')\n"
                                "    return {'hint': f'set {KEY} to open it',\n"
                                "            'set': bool(secret(name='OPTI_KEYFILE_PASSPHRASE'))}\n\n\n"
                                "def enroll():\n    secret = base64.b32encode(os.urandom(20)).decode()\n    return secret\n",
            "opti_oignon/o.py": "from opti_oignon.process_secrets import secret\n\n\ndef a():\n"
                                "    from os import getenv as secret  # noqa: F811\n    return secret('HOME')\n\n\n"
                                "def b():\n    return secret('OPTI_ENCRYPTION_KEY')\n"})
        assert guard.find_secret_reads(guard.read_estate(quiet)) == [], \
            "a name said in a sentence (a log line, a status), read by keyword, beside a local named like the reader"
        spelled = _estate(tmp_path / "spelled", {
            "opti_oignon/m.py": "import logging\n\nfrom opti_oignon.process_secrets import secret\n\n"
                                "logger = logging.getLogger(__name__)\nKEY = 'OPTI_ENCRYPTION_KEY'\n\n\n"
                                "def f():\n    if not secret(KEY):\n        logger.warning('%s is not set', KEY)\n"
                                "    return {'variable': KEY}\n"})
        spelled_found = guard.find_secret_reads(guard.read_estate(spelled))
        verbs = [line.split(": ", 1)[1].split(" (")[0] for line in spelled_found]
        assert verbs == [f"hands {KEYNAME}", f"uses {KEYNAME}"], \
            "a log call's argument and a mapping's value are places like any other: a mapping is read back"
        handed = _estate(tmp_path / "handed", {
            "opti_oignon/a.py": "KEY = 'OPTI_ENCRYPTION_KEY'\n",
            "opti_oignon/b.py": "from opti_oignon.a import KEY  # noqa: F401\n",
            "opti_oignon/c.py": "import os\n\nfrom opti_oignon.b import KEY\n\n\ndef f():\n    return os.environ.get(KEY)\n"})
        assert guard.find_secret_reads(guard.read_estate(handed)) == [
            "opti_oignon/c.py:7: reads OPTI_ENCRYPTION_KEY (os.environ.get); the one read is "
            "opti_oignon.process_secrets.secret()"], "a constant handed on through a third module"
        put = _estate(tmp_path / "put", {
            "opti_oignon/m.py": "import os\n\nfrom opti_oignon.process_secrets import secret\n\n\ndef f():\n"
                                "    os.putenv('TOOL_KEY', secret('OPTI_ENCRYPTION_KEY'))\n"
                                "    os.environ['TOOL_PASSPHRASE'] = secret('OPTI_KEYFILE_PASSPHRASE')\n"
                                "    os.environ.update(TOOL_A=secret('OPTI_ENCRYPTION_KEY'))\n"
                                "    os.environ.setdefault('TOOL_B', secret('OPTI_KEYFILE_PASSPHRASE'))\n"
                                "    value = secret('OPTI_ENCRYPTION_KEY')\n"
                                "    os.environ['TOOL_C'] = value\n"
                                "    os.environ[secret('OPTI_ENCRYPTION_KEY')] = '1'\n"
                                "    os.putenv(secret('OPTI_KEYFILE_PASSPHRASE'), '1')\n"})
        tail = ", which every child inherits; a secret's value never goes into the environment"
        assert guard.find_secret_reads(guard.read_estate(put)) == [
            f"opti_oignon/m.py:7: puts a secret's value into os.putenv{tail}",
            f"opti_oignon/m.py:8: puts a secret's value into os.environ[...]{tail}",
            f"opti_oignon/m.py:9: puts a secret's value into os.environ.update{tail}",
            f"opti_oignon/m.py:10: puts a secret's value into os.environ.setdefault{tail}",
            f"opti_oignon/m.py:12: puts a secret's value into os.environ[...]{tail}",
            f"opti_oignon/m.py:13: puts a secret's value into os.environ[...]{tail}",
            f"opti_oignon/m.py:14: puts a secret's value into os.putenv{tail}"]
        through = _estate(tmp_path / "through", {
            "opti_oignon/encryption.py": "_ENV_KEY_NAME = 'OPTI_ENCRYPTION_KEY'\n",
            "opti_oignon/m.py": "import os\n\nfrom opti_oignon import encryption\n\n\ndef f():\n"
                                "    return os.environ.get(getattr(encryption, '_ENV_KEY_NAME'))\n"})
        assert guard.find_secret_reads(guard.read_estate(through)) == [
            "opti_oignon/m.py:7: reads OPTI_ENCRYPTION_KEY (os.environ.get); the one read is "
            "opti_oignon.process_secrets.secret()"], "a constant read through getattr, its identifier a string"
        assert guard.find_secret_reads(guard.read_estate(REPO)) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# Fresh interpreters: take() never runs in the test process.
# ---------------------------------------------------------------------------
_PREAMBLE = (
    "import json\nimport os\nimport sys\n\n"
    f"PHRASE, KEY = {PHRASE!r}, {KEY!r}\n"
    f"NAMES = ({PASS!r}, {KEYNAME!r})\n\n\n"
    "def proc_holds(data):\n"
    "    \"\"\"Whether a block still holds each secret: its value anywhere in it, or any byte that is not zero\n"
    "    where an entry of its name kept its value -- a wipe of one byte leaves the rest readable.\"\"\"\n"
    "    out = {}\n"
    "    for label, name, value in (('phrase', NAMES[0], PHRASE), ('key', NAMES[1], KEY)):\n"
    "        raw, marker, block = value.encode(), b'\\0' + name.encode() + b'=', b'\\0' + data\n"
    "        held, at = raw in data, block.find(marker)\n"
    "        while at >= 0:\n"
    "            held = held or any(block[at + len(marker):at + len(marker) + len(raw)])\n"
    "            at = block.find(marker, at + 1)\n"
    "        out[label] = held\n"
    "    return out\n\n\n"
    "def own_environ():\n"
    "    with open('/proc/self/environ', 'rb') as block:\n"
    "        return block.read()\n\n\n"
)
# What a child imports encryption beside: a security mode with no lockfile to keep.
_NO_MODE = (
    "import types\n"
    "sys.modules['opti_oignon.security_mode'] = types.ModuleType('opti_oignon.security_mode')\n"
)


def _fresh(tmp_path, name, body, *, given=None, path=None, prelude="", timeout=120):
    """``body`` run in a fresh interpreter, after ``prelude``, its environment this process's with ``given``
    added (and ``path`` first on its PATH): its last line, read as JSON. Nothing it prints holds a secret's
    value."""
    script = tmp_path / f"{name}.py"
    script.write_text(_PREAMBLE + prelude + textwrap.dedent(body), encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if key not in NAMES}
    env.update(given or {})
    if path is not None:
        env["PATH"] = f"{path}{os.pathsep}{env.get('PATH', '/usr/bin:/bin')}"
    done = subprocess.run([sys.executable, str(script)], env=env, cwd=tmp_path, capture_output=True, text=True,
                          timeout=timeout)
    assert done.returncode == 0, (name, done.stderr[-4000:])
    return json.loads(done.stdout.strip().splitlines()[-1])


def _recorder(folder, name, notes, *, out="", code=0):
    """A stand-in program, in Python: it notes whether its environment holds a secret or a shell hook."""
    script = folder / name
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "values = list(os.environ.values())\n"
        f"note = {{'prog': {name!r}, 'argv': sys.argv[1:3],\n"
        f"        'phrase': any({PHRASE!r} in value for value in values),\n"
        f"        'key': any({KEY!r} in value for value in values),\n"
        "        'bash_env': 'BASH_ENV' in os.environ,\n"
        "        'function': any(key.startswith('BASH_FUNC_') for key in os.environ)}\n"
        f"with open({str(notes)!r}, 'a', encoding='utf-8') as fh:\n"
        "    fh.write(json.dumps(note) + '\\n')\n"
        f"sys.stdout.write({out!r})\n"
        f"sys.exit({code})\n", encoding="utf-8")
    script.chmod(0o755)


def _notes(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _secrets_window():
    loaded, restore = isolate(targets={"opti_oignon.process_secrets": source("process_secrets.py")})
    return loaded["opti_oignon.process_secrets"], restore


# ---------------------------------------------------------------------------
# XE9 -- take() moves both secrets into the holder
# ---------------------------------------------------------------------------
def test_xe9_take_moves_both_secrets_into_the_holder(tmp_path):
    out = _fresh(tmp_path, "xe9", """
        from opti_oignon import process_secrets as ps

        before = [name in os.environ for name in NAMES]
        ps.take()
        out = {"before": before, "after": [name in os.environ for name in NAMES],
               "answered": [ps.secret(NAMES[0]) == PHRASE, ps.secret(NAMES[1]) == KEY],
               "held": ps.status()["held"], "unwiped": ps.status()["unwiped"],
               "holder": type(ps._held[NAMES[1]]).__name__, "unlocked": ps.status()["unlocked"],
               "hazards": sorted(name for name in os.environ if ps.withheld(name))}
        older = ps._held[NAMES[1]]
        try:
            os.environ[NAMES[1]] = "a newer value"
            put = "made"
        except PermissionError:
            put = "refused"
        ps.take()
        out["newer"] = [put, NAMES[1] in os.environ, older.is_wiped, ps.secret(NAMES[1]) == KEY,
                        ps.secret(NAMES[0]) == PHRASE]
        try:
            ps.secret("HOME")
            out["refused"] = False
        except ValueError:
            out["refused"] = True
        print(json.dumps(out))
    """, given={PASS: PHRASE, KEYNAME: KEY, "BASH_ENV": "/x", "PS4": "$(true)", "BASH_FUNC_ls%%": "() { :; }"})
    assert out["before"] == [True, True] and out["after"] == [False, False], out
    assert out["hazards"] == [], "what a shell runs code from leaves the environment with the secrets"
    assert isinstance(out["unlocked"], list) and set(out["unlocked"]) <= set(NAMES), out
    assert out["answered"] == [True, True] and out["held"] == sorted(NAMES) and out["unwiped"] == {}, out
    assert out["holder"] == "_Held", "a holder of its own, which no signal handler wipes"
    assert out["newer"] == ["refused", False, False, True, True], \
        "once a value is held, a secret's name cannot be put back in the environment, and the held value is kept"
    again = _fresh(tmp_path, "xe9_again", """
        import posix

        from opti_oignon import process_secrets as ps

        ps.take()
        posix.environ[os.fsencode(NAMES[1])] = b"a newer value given behind the check"
        ps.take()
        print(json.dumps({"kept": ps.secret(NAMES[1]) == KEY, "left": NAMES[1] in os.environ}))
    """, given={PASS: PHRASE, KEYNAME: KEY})
    assert again == {"kept": True, "left": False}, "a value given again, behind the check, is not held over the first"
    assert out["refused"] is True
    empty = _fresh(tmp_path, "xe9_empty", """
        from opti_oignon import process_secrets as ps

        ps.take()
        print(json.dumps({"left": [name in os.environ for name in NAMES], "held": ps.status()["held"],
                          "answer": ps.secret(NAMES[0])}))
    """, given={PASS: "", KEYNAME: KEY})
    assert empty == {"left": [False, False], "held": [KEYNAME], "answer": None}, empty
    locked = _fresh(tmp_path, "xe9_locked", """
        import resource

        from opti_oignon import process_secrets as ps

        source = bytearray(b"a value")
        holder = ps._Held(source)
        out = {"source": bytes(source) == bytes(len(source)), "kept": holder.as_bytes() == b"a value"}
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (0, 0))
        ps.take()
        out["unlocked"] = ps.status()["unlocked"]
        print(json.dumps(out))
    """, given={PASS: PHRASE, KEYNAME: KEY})
    assert locked["source"] and locked["kept"], "the holder zeroes the bytes it is built from and keeps its own"
    assert locked["unlocked"] == sorted(NAMES), "memory that could not be locked is said so, never taken for locked"
    kept = _fresh(tmp_path, "xe9_kept", """
        import resource

        from opti_oignon import process_secrets as ps


        def locked_kb():
            with open("/proc/self/status", encoding="ascii") as status:
                return next(int(line.split()[1]) for line in status if line.startswith("VmLck:"))


        soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
        ps.take()
        first = {"unlocked": ps.status()["unlocked"], "kb": locked_kb()}
        one, other = ps._Held(bytearray(b"a value held first")), ps._Held(bytearray(b"a value held beside it"))
        before = locked_kb()
        one.wipe()
        print(json.dumps({"room": soft == resource.RLIM_INFINITY or soft >= 65536, "first": first,
                          "wiped": {"before": before, "after": locked_kb(), "other": other.locked}}))
    """, given={PASS: PHRASE, KEYNAME: KEY})
    if kept["room"]:
        assert kept["first"]["unlocked"] == [] and kept["first"]["kb"] > 0, kept
        wiped = kept["wiped"]
        assert wiped["other"] and wiped["after"] >= wiped["before"] > 0, \
            "a wipe keeps its lock: the page may hold a live holder's value"


# ---------------------------------------------------------------------------
# XE10 -- after take(), /proc/<pid>/environ holds neither secret
# ---------------------------------------------------------------------------
def test_xe10_after_take_the_environment_the_process_was_started_with_holds_no_secret(tmp_path):
    reader = tmp_path / "xe10_reader.py"
    reader.write_text("import os, sys\n"
                      "with open(f'/proc/{os.getppid()}/environ', 'rb') as block:\n"
                      "    sys.stdout.buffer.write(block.read())\n", encoding="utf-8")
    out = _fresh(tmp_path, "xe10", f"""
        import subprocess
        from opti_oignon import process_secrets as ps

        before = proc_holds(own_environ())
        ps.take()
        after = proc_holds(own_environ())
        seen = subprocess.run([sys.executable, {str(reader)!r}], capture_output=True, env=ps.child_env(),
                              check=True).stdout
        print(json.dumps({{"before": before, "after": after, "child": proc_holds(seen), "read": len(seen) > 0,
                          "unwiped": ps.status()["unwiped"]}}))
    """, given={PASS: PHRASE, KEYNAME: KEY})
    assert out["before"] == {"phrase": True, "key": True}, out
    assert out["after"] == {"phrase": False, "key": False} and out["unwiped"] == {}, out
    assert out["read"] is True and out["child"] == {"phrase": False, "key": False}, out
    named = _fresh(tmp_path, "xe10_named", """
        import ctypes

        from opti_oignon import process_secrets as ps

        ctypes.CDLL(None).prctl(15, b"x) 1 (y", 0, 0, 0)
        with open("/proc/self/stat", "rb") as stat:
            named = b"x) 1 (y" in stat.read()
        ps.take()
        print(json.dumps({"named": named, "after": proc_holds(own_environ()), "unwiped": ps.status()["unwiped"]}))
    """, given={PASS: PHRASE, KEYNAME: KEY})
    assert named == {"named": True, "after": {"phrase": False, "key": False}, "unwiped": {}}, \
        "a command's name with a parenthesis in it: the block's bounds are read after the last one"
    entries = _fresh(tmp_path, "xe10_entries", """
        import ctypes

        from opti_oignon import process_secrets as ps

        raw = b"A=1\\0" + NAMES[1].encode() + b"=first\\0B=2\\0" + NAMES[1].encode() + b"=second value\\0"
        buffer = ctypes.create_string_buffer(raw, len(raw))
        start = ctypes.addressof(buffer)
        zeroed = ps._zero(NAMES[1], (start, start + len(raw)))
        print(json.dumps({"zeroed": zeroed, "block": buffer.raw.decode("latin-1")}))
    """)
    name = KEYNAME + "="
    first, second = len("A=1\0") + len(name), len("A=1\0") + len(name) + len("first\0B=2\0") + len(name)
    assert entries["zeroed"] == [[first, len("first")], [second, len("second value")]], entries
    assert entries["block"] == f"A=1\0{name}" + "\0" * 5 + f"\0B=2\0{name}" + "\0" * 12 + "\0", entries
    misread = _fresh(tmp_path, "xe10_misread", """
        from opti_oignon import process_secrets as ps

        real = ps._initial_block


        def one_byte():
            start, _end = real()
            return start, start + 1


        ps._initial_block = one_byte
        ps.take()
        print(json.dumps({"unwiped": ps.status()["unwiped"], "refused": ps.unwiped_refusal() is not None}))
    """, given={PASS: PHRASE, KEYNAME: KEY})
    assert misread["refused"] and set(misread["unwiped"]) == set(NAMES), misread
    shown = _fresh(tmp_path, "xe10_shown", """
        from opti_oignon import process_secrets as ps

        block = own_environ()
        first = block.index(0)
        print(json.dumps({"held": ps._still_shown(None, [(0, first)]), "zeros": ps._still_shown(None, [(first, 1)]),
                          "value": ps._still_shown(os.fsdecode(block[:first]), [])}))
    """)
    assert shown["held"] and "still reads" in shown["held"] and shown["zeros"] is None, shown
    assert shown["value"] and "still shows" in shown["value"], shown
    assert all("still shows" in why for why in misread["unwiped"].values()), \
        "the block is read back where /proc shows it: a wipe that missed is said, never taken for one"


# ---------------------------------------------------------------------------
# XE11 -- child_env() drops the secrets and what a shell runs code from
# ---------------------------------------------------------------------------
def test_xe11_child_env_drops_the_secrets_and_what_a_shell_runs_code_from(tmp_path, monkeypatch):
    bash = shutil.which("bash") or "/bin/bash"
    notes = tmp_path / "notes"
    notes.mkdir()
    hook = tmp_path / "hook.sh"
    hook.write_text(f"builtin echo hook >> {notes}/bash_env\n", encoding="utf-8")
    monkeypatch.setenv("BASH_ENV", str(hook))
    monkeypatch.setenv("BASH_FUNC_echo%%", f"() {{ builtin echo function >> {notes}/function; }}")
    monkeypatch.setenv("SHELLOPTS", "xtrace")
    monkeypatch.setenv("PS4", f"$(builtin echo trace >> {notes}/ps4)")
    monkeypatch.setenv("BASHOPTS", "xpg_echo")
    monkeypatch.setenv("ENV", str(hook))
    monkeypatch.setenv(PASS, PHRASE)
    monkeypatch.setenv(KEYNAME, KEY)

    def ran(env, script):
        for note in notes.iterdir():
            note.unlink()
        subprocess.run([bash, "-c", script], env=env, capture_output=True, timeout=30, check=False)
        return sorted(note.name for note in notes.iterdir())

    ps, restore = _secrets_window()
    try:
        built = ps.child_env()
        assert not (set(NAMES) | set(HAZARDS)) & set(built), sorted(set(built) & (set(NAMES) | set(HAZARDS)))
        assert not any(name.startswith("BASH_FUNC_") for name in built)
        assert PHRASE not in built.values() and KEY not in built.values()
        assert built.get("PATH") == os.environ["PATH"] and built.get("HOME") == os.environ.get("HOME")
        assert ran(built, "echo hi") == [] and ran(built, "set -x; true") == []
        whole = dict(os.environ)
        assert ran(whole, "echo hi") == ["bash_env", "function", "ps4"]
        untraced = {name: value for name, value in whole.items() if name != "SHELLOPTS"}
        assert ran(untraced, "set -x; true") == ["bash_env", "ps4"]
        for name in (PASS, KEYNAME, "BASH_ENV", "PS4", "BASH_FUNC_ls%%"):
            with pytest.raises(ValueError):
                ps.child_env({name: "x"})
        assert ps.child_env({"XE_EXTRA": "kept"})["XE_EXTRA"] == "kept"
        assert os.environ[KEYNAME] == KEY and os.environ["BASH_ENV"] == str(hook), "os.environ is left as it was"
    finally:
        restore()


# ---------------------------------------------------------------------------
# XE12 -- a pool worker started after take() holds no secret
# ---------------------------------------------------------------------------
def test_xe12_a_pool_worker_started_after_take_holds_no_secret(tmp_path):
    out = _fresh(tmp_path, "xe12", """
        from opti_oignon import background_pool
        from opti_oignon import process_secrets as ps


        def seen():
            with background_pool._spawn_executor(1, os.getpid, ()) as pool:
                return [pool.submit(os.getenv, name).result(timeout=120) == value
                        for name, value in zip(NAMES, (PHRASE, KEY))]


        if __name__ == "__main__":
            before = seen()
            ps.take()
            print(json.dumps({"before": before, "after": seen()}))
    """, given={PASS: PHRASE, KEYNAME: KEY}, timeout=180)
    assert out == {"before": [True, True], "after": [False, False]}, out


# ---------------------------------------------------------------------------
# XE13 -- the key loads read the held secrets
# ---------------------------------------------------------------------------
def test_xe13_the_key_loads_read_the_held_secrets(tmp_path):
    out = _fresh(tmp_path, "xe13_phrase", """
        from pathlib import Path

        from opti_oignon import process_secrets as ps

        ps.take()
        from opti_oignon import encryption as enc

        enc._DEFAULT_KEYFILE = Path(sys.argv[0]).with_name("xe13.keyfile")
        made = enc.generate_key()
        enc.save_keyfile(made, kdf_name="random", passphrase=PHRASE)
        loaded, _salt, _kdf = enc.load_keyfile()
        key = enc.get_encryption_key()
        print(json.dumps({"opened": loaded.as_bytes() == made, "key": key is not None and key.as_bytes() == made,
                          "environ": [name in os.environ for name in NAMES]}))
    """, given={PASS: PHRASE}, prelude=_NO_MODE)
    assert out == {"opened": True, "key": True, "environ": [False, False]}, out
    out = _fresh(tmp_path, "xe13_key", """
        import base64
        from pathlib import Path

        from opti_oignon import process_secrets as ps

        ps.take()
        from opti_oignon import encryption as enc

        enc._DEFAULT_KEYFILE = Path(sys.argv[0]).with_name("xe13.absent")
        key = enc.get_encryption_key()
        print(json.dumps({"key": key is not None and key.as_bytes() == base64.urlsafe_b64decode(KEY),
                          "environ": [name in os.environ for name in NAMES]}))
    """, given={KEYNAME: KEY}, prelude=_NO_MODE)
    assert out == {"key": True, "environ": [False, False]}, out


# ---------------------------------------------------------------------------
# XE14 -- setup after take() wraps the key file under the held passphrase
# ---------------------------------------------------------------------------
def test_xe14_setup_after_take_wraps_the_key_file_under_the_held_passphrase(tmp_path):
    out = _fresh(tmp_path, "xe14", """
        import types
        from pathlib import Path

        from opti_oignon import process_secrets as ps

        ps.take()
        from opti_oignon import encryption as enc

        here = Path(sys.argv[0]).parent


        def envelope(path):
            try:
                return json.loads(path.read_text(encoding="ascii")).get("version") == "envelope-v1"
            except ValueError:
                return False


        enc._DEFAULT_KEYFILE = here / "manager.keyfile"
        manager = enc.EncryptionManager(enabled=False)
        made = manager.setup_random_key()
        out = {"made": made, "wrapped": envelope(enc._DEFAULT_KEYFILE),
               "opens": enc.load_keyfile(passphrase=PHRASE)[0].as_bytes() == manager._key.as_bytes()}

        from opti_oignon import db_encryption as dbe

        enc._DEFAULT_KEYFILE = here / "command.keyfile"
        master = enc.generate_key()
        dbe._write_master(master, dbe._db_key(master))
        out["command"] = envelope(enc._DEFAULT_KEYFILE)

        from opti_oignon.api import routes_security as routes

        journal = types.ModuleType("opti_oignon.run_journal")
        journal.on_encryption_enabled = lambda: None
        sys.modules["opti_oignon.run_journal"] = journal
        routes._load_security_yaml = lambda: {}
        routes._save_security_yaml = lambda data: None
        routes._plaintext_stores = lambda: []
        enc._DEFAULT_KEYFILE = here / "route.keyfile"
        enc._encryption_manager = enc.EncryptionManager(enabled=False)
        answer = routes.setup_encryption(routes.EncryptionSetupRequest(mode="random"))
        out["route"] = [answer["key"], answer["wrapped"], envelope(enc._DEFAULT_KEYFILE)]
        print(json.dumps(out))
    """, given={PASS: PHRASE}, prelude=_NO_MODE)
    assert out == {"made": True, "wrapped": True, "opens": True, "command": True, "route": ["made", True, True]}, out


# ---------------------------------------------------------------------------
# XE15 -- the refusals and the statuses read the holder
# ---------------------------------------------------------------------------
def test_xe15_the_refusals_and_the_statuses_read_the_holder(tmp_path):
    out = _fresh(tmp_path, "xe15_no_key", """
        from pathlib import Path

        from opti_oignon import process_secrets as ps

        ps.take()
        from fastapi import HTTPException

        from opti_oignon import db_encryption as dbe
        from opti_oignon import encryption as enc
        from opti_oignon.allium import store
        from opti_oignon.api import routes_security as routes

        enc._DEFAULT_KEYFILE = Path(sys.argv[0]).with_name("xe15.absent")
        enc._encryption_manager = enc.EncryptionManager(enabled=False)
        out = {"refused": dbe._environment_key_refused() is not None,
               "status": enc.EncryptionManager(enabled=False).get_status()["env_key_set"],
               "loads": routes._environment_key_loads(),
               "anchor": store._default_anchor_secret()[0]}
        try:
            routes.setup_encryption(routes.EncryptionSetupRequest(mode="random"))
            out["setup"] = None
        except HTTPException as exc:
            out["setup"] = [exc.status_code, exc.detail == routes._ENVIRONMENT_KEY_REFUSAL]
        out["environ"] = [name in os.environ for name in NAMES]
        print(json.dumps(out))
    """, given={KEYNAME: "no key at all"}, prelude=_NO_MODE)
    assert out == {"refused": True, "status": True, "loads": False, "anchor": "unreadable", "setup": [409, True],
                   "environ": [False, False]}, out
    out = _fresh(tmp_path, "xe15_key", """
        from pathlib import Path

        from opti_oignon import process_secrets as ps

        ps.take()
        from opti_oignon import chroma_migration as cm
        from opti_oignon import db_encryption as dbe
        from opti_oignon import encryption as enc
        from opti_oignon.api import routes_security as routes

        enc._DEFAULT_KEYFILE = Path(sys.argv[0]).with_name("xe15.absent")
        dbe.SQLCIPHER_AVAILABLE = False
        try:
            cm._key()
            refused = None
        except dbe.EncryptionRefused as exc:
            refused = "SQLCipher is not installed and a key is set" in str(exc)
        print(json.dumps({"migration": refused, "loads": routes._environment_key_loads(),
                          "refused": dbe._environment_key_refused() is not None}))
    """, given={KEYNAME: KEY}, prelude=_NO_MODE)
    assert out == {"migration": True, "loads": True, "refused": False}, out


# ---------------------------------------------------------------------------
# XE16 -- a wiped holder is a third state
# ---------------------------------------------------------------------------
def test_xe16_a_wiped_holder_is_a_third_state(tmp_path):
    out = _fresh(tmp_path, "xe16", """
        from pathlib import Path

        from opti_oignon import process_secrets as ps

        ps.take()
        from opti_oignon import encryption as enc
        from opti_oignon import secure_bytes

        here = Path(sys.argv[0]).parent
        enc._DEFAULT_KEYFILE = here / "xe16.keyfile"
        enc.save_keyfile(enc.generate_key(), kdf_name="random", passphrase=PHRASE)
        secure_bytes._wipe_all_keys()
        survived = ps.secret(NAMES[0]) == PHRASE
        ps._wipe_held()
        outcomes = {}
        for label, read in (("secret", lambda: ps.secret(NAMES[0])), ("key", enc.get_encryption_key)):
            try:
                read()
                outcomes[label] = "answered"
            except ps.SecretWiped:
                outcomes[label] = "wiped"
        enc._DEFAULT_KEYFILE = here / "xe16.made"
        made = enc.EncryptionManager(enabled=False).setup_random_key()
        print(json.dumps({"survived": survived, "outcomes": outcomes, "made": made,
                          "written": enc._DEFAULT_KEYFILE.exists()}))
    """, given={PASS: PHRASE}, prelude=_NO_MODE)
    assert out["survived"] is True, "a signal's wipe of every SecureBytes leaves the held secrets to exit"
    assert out["outcomes"] == {"secret": "wiped", "key": "wiped"}, out
    assert (out["made"], out["written"]) == (False, False), "setup refuses rather than write a key file unwrapped"
    master = _fresh(tmp_path, "xe16_master", """
        from pathlib import Path

        from opti_oignon import process_secrets as ps

        ps.take()
        from opti_oignon import db_encryption as dbe
        from opti_oignon import encryption as enc

        enc._DEFAULT_KEYFILE = Path(sys.argv[0]).with_name("xe16.absent")
        ps._wipe_held()
        outcomes = {}
        for label, read in (("key", enc.get_encryption_key), ("databases", dbe._get_db_encryption_key)):
            try:
                outcomes[label] = "none" if read() is None else "answered"
            except ps.SecretWiped:
                outcomes[label] = "wiped"
        print(json.dumps(outcomes))
    """, given={KEYNAME: KEY}, prelude=_NO_MODE)
    assert master == {"key": "wiped", "databases": "wiped"}, \
        "a wiped master key is never read as no key: a store would open in plaintext"
    at_exit = _fresh(tmp_path, "xe16_exit", """
        import atexit

        from opti_oignon import process_secrets as ps

        held = []


        def report():
            print(json.dumps({"wiped": [holder.is_wiped for holder in held],
                              "zeros": [not any(holder._buf) for holder in held]}))


        atexit.register(report)
        ps.take()
        held.extend(ps._held.values())
    """, given={PASS: PHRASE, KEYNAME: KEY})
    assert at_exit == {"wiped": [True, True], "zeros": [True, True]}, \
        "the held secrets are wiped at exit, their bytes zeroed and not only flagged"


# ---------------------------------------------------------------------------
# XE17 -- the real entry starts nothing that holds a secret
# ---------------------------------------------------------------------------
def test_xe17_the_real_entry_starts_nothing_that_holds_a_secret(tmp_path):
    folder, notes = tmp_path / "bin", tmp_path / "starts.jsonl"
    folder.mkdir()
    _recorder(folder, "bwrap", notes, out="sandbox-functional-test-ok\n")
    _recorder(folder, "gcc", notes, code=1)
    _recorder(folder, "ld", notes, code=1)
    hook = tmp_path / "hook.sh"
    hook.write_text("builtin echo hook\n", encoding="utf-8")
    out = _fresh(tmp_path, "xe17", """
        import uvicorn.importer

        uvicorn.importer.import_from_string("opti_oignon.api.app:app")
        print(json.dumps({"environ": [name in os.environ for name in NAMES], "proc": proc_holds(own_environ())}))
    """, given={PASS: PHRASE, KEYNAME: KEY, "BASH_ENV": str(hook), "BASH_FUNC_echo%%": "() { builtin echo f; }"},
        path=folder, timeout=180)
    assert out == {"environ": [False, False], "proc": {"phrase": False, "key": False}}, out
    starts = _notes(notes)
    probes = [start for start in starts if start["prog"] == "bwrap"]
    assert len(probes) >= 2, starts
    assert not any(start["phrase"] or start["key"] for start in starts), starts
    assert not any(start["bash_env"] or start["function"] for start in starts), \
        "the entry takes the shell's variables out with the secrets: no start inherits them"
    assert [start["argv"][:1] for start in probes if start["argv"][:1] != ["--version"]] == [["--clearenv"]], \
        "the functional probe runs its echo in a cleared environment"


# ---------------------------------------------------------------------------
# XE18 -- the untrusted tiers stay allowlisted, and their launcher gets child_env()
# ---------------------------------------------------------------------------
def _sandbox_window(tmp_path, bulbe=False):
    """The sandbox manager over a byte-exact copy whose import-time singleton stays off, the holder beside it."""
    package = tmp_path / "copy"
    (package / "config").mkdir(parents=True)
    (package / "config" / "sandbox.yaml").write_text("enabled: false\n", encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = lambda path, *args, **kwargs: __import__("sqlite3").connect(path)
    config = types.ModuleType("opti_oignon.config")
    config.DATA_DIR, config.CONFIG_DIR, config.PROJECT_ROOT = tmp_path / "data", tmp_path / "config", tmp_path / "p"
    seccomp = types.ModuleType("opti_oignon.sandbox_seccomp")
    seccomp.build_filter_program = lambda arch=None: b"\x06\x00\x00\x00\x00\x00\xff\x7f"
    mode = types.ModuleType("opti_oignon.security_mode")
    mode.is_bulbe = lambda: bulbe
    loaded, restore = isolate(
        targets={"opti_oignon.process_secrets": source("process_secrets.py"), "opti_oignon.sandbox_manager": copy},
        seeded={"opti_oignon.db_utils": db_utils, "opti_oignon.config": config,
                "opti_oignon.sandbox_seccomp": seccomp, "opti_oignon.security_mode": mode})
    sm = loaded["opti_oignon.sandbox_manager"]
    if sm.sandbox_manager is not None:
        restore()
        raise AssertionError("the import-time manager must stay off in this window")
    return sm, loaded["opti_oignon.process_secrets"], restore


def _sandbox(sm, tmp_path, *, bwrap, backend):
    sm._detect_bwrap = lambda: (bwrap, "bubblewrap of the contract" if bwrap else "no bubblewrap here")
    config = sm.SandboxConfig(
        workspace_base=str(tmp_path / "workspaces"), audit_db_path=str(tmp_path / "audit.db"),
        reconcile_on_start=False, strict_mode=False, isolation_backend=backend, seccomp_enabled=False,
        seccomp_required=False, limits_enabled=False, require_degraded_confirmation=False)
    manager = sm.SandboxManager(config)
    launches = []

    def spawn(argv, **kwargs):
        launches.append(dict(kwargs, argv=list(argv)))
        return subprocess.CompletedProcess(argv, 0, b"ran\n", b"")

    manager._spawn_tracked = spawn
    return manager, launches


def _withheld(env):
    return sorted(name for name in env if name in NAMES or name in HAZARDS or name.startswith("BASH_FUNC_"))


def test_xe18_the_untrusted_tiers_stay_allowlisted_and_their_launcher_gets_child_env(tmp_path, monkeypatch):
    for name, value in ((PASS, PHRASE), (KEYNAME, KEY), ("BASH_ENV", "/x"), ("ENV", "/x"), ("SHELLOPTS", "xtrace"),
                        ("BASHOPTS", "xpg_echo"), ("PS4", "$(true)"), ("BASH_FUNC_ls%%", "() { :; }")):
        monkeypatch.setenv(name, value)
    import importlib

    plugins = importlib.import_module("opti_oignon.plugin_subprocess")
    worker = plugins._build_plugin_env({"OO_PLUGIN_NAME": "xe18"})
    assert _withheld(worker) == [] and set(worker) <= set(plugins._FORWARDED_ENV_VARS) | {"OO_PLUGIN_NAME"}, worker
    sm, ps, restore = _sandbox_window(tmp_path)
    try:
        assert sm._tempdir_env("/w") == {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/w", "TMPDIR": "/w",
                                         "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
        manager, launches = _sandbox(sm, tmp_path, bwrap=True, backend="auto")
        workspace = tmp_path / "ws"
        workspace.mkdir()
        manager._run_bwrap("true", str(workspace), 5)
        assert len(launches) == 1 and launches[0]["argv"][0] == "bwrap", launches
        handed = launches[0]["env"]
        assert _withheld(handed) == [] and handed == ps.child_env(), _withheld(handed)
        assert handed.get("PATH") == os.environ["PATH"], "the bubblewrap it runs keeps the user's session"
        assert "--clearenv" in launches[0]["argv"], "the command inside starts from a cleared environment"
    finally:
        restore()


# ---------------------------------------------------------------------------
# XE19 -- no process starts before the secrets leave the environment
# ---------------------------------------------------------------------------
def test_xe19_no_process_starts_before_the_secrets_leave_the_environment(tmp_path):
    events = _fresh(tmp_path, "xe19", """
        # Every start Python audits, and the environment each is handed: its own, or this process's.
        EVENTS = {"subprocess.Popen": (1, 3), "os.posix_spawn": (1, 2), "os.exec": (1, 2), "os.spawn": (2, 3),
                  "os.system": (0, None), "os.fork": (None, None), "os.forkpty": (None, None),
                  "pty.spawn": (0, None)}


        def hook(event, args):
            if event not in EVENTS:
                return
            where, env_at = EVENTS[event]
            env = args[env_at] if env_at is not None and len(args) > env_at else None
            values = [str(value) for value in (env.values() if hasattr(env, "values") else env or ())] \
                if env is not None else [str(value) for value in os.environ.values()]
            argv = args[where] if where is not None and len(args) > where else event
            first = argv[0] if isinstance(argv, (list, tuple)) and argv else argv
            started.append({"prog": os.path.basename(os.fsdecode(first)) if isinstance(first, (str, bytes)) else event,
                            "event": event, "popped": not any(name in os.environ for name in NAMES),
                            "held": any(PHRASE in value or KEY in value for value in values)})


        started = []
        sys.addaudithook(hook)
        import uvicorn.importer

        uvicorn.importer.import_from_string("opti_oignon.api.app:app")
        print(json.dumps(started))
    """, given={PASS: PHRASE, KEYNAME: KEY}, timeout=180)
    assert events and "ldconfig" in {event["prog"] for event in events}, events
    assert all(event["popped"] for event in events), events
    assert not any(event["held"] for event in events), events


# ---------------------------------------------------------------------------
# XE20 -- uname -p, run for a request, holds no secret
# ---------------------------------------------------------------------------
def test_xe20_uname_run_for_a_request_holds_no_secret(tmp_path):
    folder = tmp_path / "bin"
    folder.mkdir()
    seen = {}
    for take in (True, False):
        notes = tmp_path / f"uname_{take}.jsonl"
        _recorder(folder, "uname", notes, out="x86_64\n")
        _fresh(tmp_path, f"xe20_{take}", f"""
            from opti_oignon import process_secrets as ps

            if {take!r}:
                ps.take()
            from opti_oignon import auto_tuner

            auto_tuner.get_hardware_fingerprint()
            print(json.dumps(True))
        """, given={PASS: PHRASE, KEYNAME: KEY}, path=folder)
        seen[take] = [(note["phrase"], note["key"]) for note in _notes(notes)]
    assert seen == {True: [(False, False)], False: [(True, True)]}, seen


# ---------------------------------------------------------------------------
# XE21 -- a contract that leaves a secret held fails at its teardown
# ---------------------------------------------------------------------------
def _conftest():
    here = Path(__file__).resolve().with_name("conftest.py")
    for module in list(sys.modules.values()):
        if getattr(module, "__file__", None) and Path(module.__file__).resolve() == here:
            return module
    raise AssertionError("the suite's conftest is not loaded")


def test_xe21_a_contract_that_leaves_a_secret_held_fails_at_its_teardown():
    conftest = _conftest()
    ps = sys.modules["opti_oignon.process_secrets"]
    item = types.SimpleNamespace(stash=pytest.Stash(), nodeid="xe21 witness")
    current = conftest._FIREWALL.current

    def teardown(leave):
        setup = conftest.pytest_runtest_setup(item)
        next(setup)
        with pytest.raises(StopIteration):
            setup.send(None)
        if leave:
            ps._held[KEYNAME] = object()
        ending = conftest.pytest_runtest_teardown(item)
        next(ending)
        try:
            ending.send(None)
        except StopIteration:
            return None
        except AssertionError as exc:
            return str(exc)
        finally:
            ps._held.pop(KEYNAME, None)
        return "the teardown did not end"

    try:
        assert KEYNAME not in ps._held
        assert teardown(False) is None
        said = teardown(True)
        assert said is not None and "the platform's secrets left held by opti_oignon.process_secrets: " \
            "['OPTI_ENCRYPTION_KEY']" in said, said
    finally:
        conftest._FIREWALL.current = current


# ---------------------------------------------------------------------------
# XE22 -- the dev backend script reads no hook run as it is
# ---------------------------------------------------------------------------
def test_xe22_the_dev_backend_script_reads_no_hook_run_as_it_is(tmp_path):
    root, notes, folder = tmp_path / "root", tmp_path / "notes", tmp_path / "bin"
    for place in (root / "scripts", notes, folder):
        place.mkdir(parents=True)
    script = root / "scripts" / "dev_backend.sh"
    script.write_bytes((REPO / "scripts" / "dev_backend.sh").read_bytes())
    script.chmod(0o755)
    (folder / "python3").write_text(f"#!/bin/sh\necho server >> '{notes}/server'\n", encoding="ascii")
    (folder / "python3").chmod(0o755)
    hook = tmp_path / "hook.sh"
    hook.write_text(f"builtin echo hook >> '{notes}/hook'\n", encoding="ascii")
    env = {"PATH": f"{folder}:/usr/bin:/bin", "HOME": str(tmp_path), "LANG": "C.UTF-8", "BASH_ENV": str(hook),
           "BASH_FUNC_echo%%": f"() {{ builtin echo function >> '{notes}/function'; }}"}

    def ran(argv):
        for note in notes.iterdir():
            note.unlink()
        done = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60, check=False)
        assert done.returncode == 0, done.stderr
        return sorted(note.name for note in notes.iterdir())

    assert ran([str(script)]) == ["server"]
    assert ran([shutil.which("bash") or "/bin/bash", str(script)]) == ["function", "hook", "server"]


# ---------------------------------------------------------------------------
# XE23 -- untrusted code beside an unwiped secret is refused by name
# ---------------------------------------------------------------------------
_WHY = "a wipe these contracts made fail"


def test_xe23_untrusted_code_beside_an_unwiped_secret_is_refused_by_name(tmp_path, monkeypatch):
    sm, ps, restore = _sandbox_window(tmp_path)
    try:
        manager, launches = _sandbox(sm, tmp_path / "tempdir", bwrap=False, backend="tempdir")
        session = manager.create_sandbox("xe23")
        session = getattr(session, "session_id", session)
        ps._unwiped[KEYNAME] = _WHY
        refused = manager.execute_command(session, "echo hi")
        assert refused.blocked and "BLOCKED" in refused.block_reason and KEYNAME in refused.block_reason \
            and _WHY in refused.block_reason, refused
        assert launches == []
        bwrapped, wrapped_launches = _sandbox(sm, tmp_path / "bwrap", bwrap=True, backend="auto")
        inside = bwrapped.create_sandbox("xe23-bwrap")
        ran = bwrapped.execute_command(getattr(inside, "session_id", inside), "echo hi")
        assert not ran.blocked and len(wrapped_launches) == 1 and wrapped_launches[0]["argv"][0] == "bwrap", ran
        ps._unwiped.clear()
        ran = manager.execute_command(session, "echo hi")
        assert not ran.blocked and len(launches) == 1, ran
    finally:
        restore()
    import importlib

    plugins = importlib.import_module("opti_oignon.plugin_subprocess")
    isolation = importlib.import_module("opti_oignon.plugin_isolation")
    held = sys.modules["opti_oignon.process_secrets"]
    plugin = tmp_path / "plugins" / "xe23"
    plugin.mkdir(parents=True)
    (plugin / "entry_point.py").write_text("def pre_inference(context):\n    return context\n", encoding="utf-8")
    real_popen = subprocess.Popen

    def recorded(argv, **kwargs):
        if any(str(token).endswith("plugin_worker.py") for token in argv):
            raise OSError("launch recorded by the contract, nothing started")
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", recorded)

    def started(mode, extra=None):
        posture = isolation.PluginPosture(mode=mode, strict=False, reason=f"{mode} posture of the contract",
                                          seccomp_enabled=False, seccomp_required=False)
        manager = plugins.PluginSubprocessManager(log_dir=tmp_path / "logs", data_root=tmp_path / "data",
                                                  posture_provider=lambda: posture, startup_timeout=5.0)
        try:
            manager.start_plugin("xe23", plugin, "entry_point.py", env_extra=extra)
        except isolation.PluginIsolationError as exc:
            return f"refused: {exc}"
        except plugins.PluginSubprocessError as exc:
            return f"launched: {exc}"
        return "started"

    try:
        held._unwiped[KEYNAME] = _WHY
        said = started(isolation.MODE_DIRECT)
        assert said.startswith("refused") and "without bubblewrap" in said and KEYNAME in said and _WHY in said, said
        said = started(isolation.MODE_BWRAP)
        assert said.startswith("launched") and "recorded by the contract" in said, said
    finally:
        held._unwiped.pop(KEYNAME, None)
    said = started(isolation.MODE_DIRECT)
    assert said.startswith("launched") and "recorded by the contract" in said, said
    for name in ("BASH_ENV", KEYNAME, "BASH_FUNC_ls%%"):
        said = started(isolation.MODE_BWRAP, extra={"OO_EXTRA": "kept", name: "x"})
        assert said.startswith("refused") and f"never carries {name}" in said, said
    said = started(isolation.MODE_BWRAP, extra={"OO_EXTRA": "kept"})
    assert said.startswith("launched"), said
    out = _fresh(tmp_path, "xe23_wipe", """
        from opti_oignon import process_secrets as ps


        def unreadable():
            raise OSError("a block these contracts hid")


        ps._initial_block = unreadable
        ps.take()
        state = ps.status()
        print(json.dumps({"refusal": ps.unwiped_refusal(), "unwiped": state["unwiped"], "held": state["held"],
                          "environ": [name in os.environ for name in NAMES]}))
    """, given={PASS: PHRASE, KEYNAME: KEY})
    assert out["unwiped"] == {name: "OSError: a block these contracts hid" for name in NAMES}, out
    assert out["held"] == sorted(NAMES) and out["environ"] == [False, False], out
    assert all(name in out["refusal"] for name in NAMES) and "a block these contracts hid" in out["refusal"], out


# ---------------------------------------------------------------------------
# XE24 -- no core dump while a secret is held
# ---------------------------------------------------------------------------
def test_xe24_no_core_dump_while_a_secret_is_held(tmp_path):
    reader = tmp_path / "xe24_reader.py"
    reader.write_text("import json, resource\nprint(json.dumps(list(resource.getrlimit(resource.RLIMIT_CORE))))\n",
                      encoding="utf-8")
    body = f"""
        import resource
        import subprocess

        from opti_oignon import process_secrets as ps

        before = list(resource.getrlimit(resource.RLIMIT_CORE))
        ps.take()
        after = list(resource.getrlimit(resource.RLIMIT_CORE))
        child = json.loads(subprocess.run([sys.executable, {str(reader)!r}], capture_output=True, text=True,
                                          env=ps.child_env(), check=True).stdout)
        print(json.dumps({{"before": before, "after": after, "child": child, "held": ps.status()["held"]}}))
    """
    held = _fresh(tmp_path, "xe24_held", body, given={PASS: PHRASE, KEYNAME: KEY})
    assert held["held"] == sorted(NAMES) and held["after"] == [0, 0] and held["child"] == [0, 0], held
    none = _fresh(tmp_path, "xe24_none", body)
    assert none["held"] == [] and none["after"] == none["before"] and none["child"] == none["before"], none


# ---------------------------------------------------------------------------
# XE25 -- the launcher and the gate zero what they take out
# ---------------------------------------------------------------------------
def test_xe25_the_launcher_and_the_gate_zero_what_they_take_out(tmp_path):
    out = _fresh(tmp_path, "xe25_launcher", """
        from opti_oignon import ui


        class Stopped(Exception):
            pass


        def stop():
            raise Stopped()


        ui._find_project_root = stop
        try:
            ui.launch()
        except Stopped:
            pass
        print(json.dumps({"proc": proc_holds(own_environ()), "environ": [name in os.environ for name in NAMES]}))
    """, given={PASS: PHRASE, KEYNAME: KEY})
    assert out == {"proc": {"phrase": False, "key": False}, "environ": [False, False]}, out
    out = _fresh(tmp_path, "xe25_gate", """
        from opti_oignon import keyfile_gate

        before = proc_holds(own_environ())
        answer = keyfile_gate.main(["wrapped"])
        print(json.dumps({"before": before["phrase"], "proc": proc_holds(own_environ())["phrase"],
                          "environ": NAMES[0] in os.environ, "answered": answer in (0, 3, 4)}))
    """, given={PASS: PHRASE})
    assert out == {"before": True, "proc": False, "environ": False, "answered": True}, out


# ---------------------------------------------------------------------------
# XE26 -- the startup checklist says what became of the secrets
# ---------------------------------------------------------------------------
class _Holder:
    is_wiped = False
    locked = True
    common = False
    _buf = bytearray(32)


class _Unlocked(_Holder):
    locked = False


class _Short(_Holder):
    _buf = bytearray(5)


class _Common(_Holder):
    common = True


def test_xe26_the_startup_checklist_says_what_became_of_the_secrets(monkeypatch):
    loaded, restore = isolate(targets={"opti_oignon.startup_checks": source("startup_checks.py")})
    try:
        item = loaded["opti_oignon.startup_checks"]._check_process_secrets()
        assert (item.passed, item.severity) == (True, "info") and item.detail.startswith("Not checked"), item
    finally:
        restore()
    loaded, restore = isolate(targets={"opti_oignon.process_secrets": source("process_secrets.py"),
                                       "opti_oignon.startup_checks": source("startup_checks.py")})
    try:
        ps, checks = loaded["opti_oignon.process_secrets"], loaded["opti_oignon.startup_checks"]
        item = checks._check_process_secrets()
        assert (item.passed, item.severity, item.name) == (True, "info", "process_secrets"), item
        assert "neither" in item.detail, item
        ps._held[KEYNAME] = _Holder()
        item = checks._check_process_secrets()
        assert (item.passed, item.severity) == (True, "info") and item.detail.startswith("Held out of the"), item
        assert "could not be locked" not in item.detail, item
        ps._held[PASS] = _Unlocked()
        item = checks._check_process_secrets()
        assert item.passed and f"the memory holding {PASS} could not be locked" in item.detail, item
        assert "shorter than 8 bytes" not in item.detail, item
        ps._held[PASS] = _Short()
        item = checks._check_process_secrets()
        assert item.passed and f"{PASS} is shorter than 8 bytes" in item.detail, item
        ps._held[PASS] = _Common()
        item = checks._check_process_secrets()
        assert (item.passed, item.severity) == (False, "warning") and item.tips, item
        assert f"the value of {PASS} is held by this environment already" in item.detail, item
        del ps._held[PASS]
        ps._unwiped[KEYNAME] = _WHY
        item = checks._check_process_secrets()
        assert (item.passed, item.severity) == (False, "warning") and item.detail.startswith("Not wiped"), item
        assert KEYNAME in item.detail and _WHY in item.detail and "Held" not in item.detail and item.tips, item
        ps._unwiped.clear()
        monkeypatch.setenv(KEYNAME, KEY)
        item = checks._check_process_secrets()
        assert (item.passed, item.severity) == (False, "warning") and KEYNAME in item.detail, item
        monkeypatch.delenv(KEYNAME)

        def unreadable():
            raise RuntimeError("a state these contracts hid")

        ps.status = unreadable
        item = checks._check_process_secrets()
        assert (item.passed, item.severity) == (False, "warning") and "cannot be read" in item.detail, item
    finally:
        restore()


# ---------------------------------------------------------------------------
# XE27 -- once a secret is held, a process start that would hand it to a child is refused
# ---------------------------------------------------------------------------
_XE27 = """
    import io
    import logging
    import subprocess

    from opti_oignon import process_secrets as ps

    log = io.StringIO()
    logging.getLogger("opti_oignon.process_secrets").addHandler(logging.StreamHandler(log))
    ps.take()
    folder = sys.argv[0] + ".ran"
    os.mkdir(folder)
    TOUCH = next(os.path.join(d, "touch") for d in os.environ["PATH"].split(os.pathsep)
                 if os.access(os.path.join(d, "touch"), os.X_OK))


    def mark(n):
        return [TOUCH, os.path.join(folder, str(n))]


    def tried(start):
        try:
            start()
            return "ran"
        except PermissionError as exc:
            return "refused" if "a child never holds" in str(exc) else "other"


    def waited(pid):
        return os.waitpid(pid, 0)[1]


    out = {}
    ARGS
    out["started"] = sorted(os.listdir(folder))
    out["logged"] = log.getvalue()
    out["value logged"] = any(value in out["logged"] for value in (PHRASE, KEY, "abc1234", "TOOL_"))
    out["checking"] = bool(ps._checking)
    print(json.dumps(out))
"""


def test_xe27_a_start_that_would_hand_a_child_a_secret_is_refused(tmp_path):
    # os.system is run on purpose, with fixed arguments: the check must cover the shell's start as well.
    phrase = PHRASE.replace(" ", "-")
    held = _fresh(tmp_path, "xe27", _XE27.replace("ARGS", """
    clean = ps.child_env()
    phrase = PHRASE.replace(" ", "-")
    out[1] = tried(lambda: subprocess.run(mark(1), env=dict(clean, TOOL_KEY=KEY)))
    out[2] = tried(lambda: subprocess.run(mark(2), env=dict(clean, TOOL_ARGS="--key=" + KEY)))
    out[3] = tried(lambda: subprocess.run(mark(3), env=dict(clean, **{NAMES[0]: "x"})))
    out[4] = tried(lambda: subprocess.run(mark(4) + ["--phrase=" + phrase], env=clean))
    out[5] = tried(lambda: subprocess.run(mark(5), env={os.fsencode(NAMES[1]): b"x", b"PATH": b"/bin"}))
    out[6] = tried(lambda: os.system(" ".join(mark(6)) + " --phrase=" + phrase))
    out[7] = tried(lambda: waited(os.posix_spawn(TOUCH, mark(7), {"TOOL_KEY": KEY})))
    pid = os.fork()
    if pid == 0:
        try:
            os.execve(TOUCH, mark(8), {"TOOL_KEY": KEY})
        except PermissionError:
            os._exit(42)
        os._exit(1)
    out[8] = "refused" if waited(pid) >> 8 == 42 else "ran"
    out["put"] = [tried(lambda: os.environ.__setitem__("TOOL_PHRASE", phrase)),
                  tried(lambda: os.putenv("TOOL_KEY", KEY)),
                  tried(lambda: os.environ.__setitem__(NAMES[0], "x")),
                  tried(lambda: os.putenv(phrase, "1")),
                  tried(lambda: os.environ.__setitem__("BASH_ENV", "/tmp/hook.sh")),
                  "TOOL_PHRASE" in os.environ, NAMES[0] in os.environ, "BASH_ENV" in os.environ]
    out[12] = tried(lambda: subprocess.run(mark(12), env={KEY: "1", "PATH": "/bin"}))
    out[13] = tried(lambda: subprocess.run(mark(13), env=dict(clean, BASH_ENV="/tmp/hook.sh")))
    out[14] = tried(lambda: waited(os.posix_spawn(TOUCH, mark(14), {"PATH": "/bin", "BASH_FUNC_x%%": "() { :; }"})))
    out[15] = tried(lambda: subprocess.run(mark(15), env={"PATH": "/bin", "ENV": "/tmp/hook.sh"}))
    out[9] = tried(lambda: subprocess.run(mark(9), env=clean))
    out[10] = tried(lambda: os.system(" ".join(mark(10))))
    out[11] = tried(lambda: waited(os.posix_spawn(TOUCH, mark(11), clean)))
    """), given={PASS: phrase, KEYNAME: KEY})
    refused = {str(n): "refused" for n in (1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15)}
    assert {key: held[key] for key in refused} == refused, held
    assert held["put"] == ["refused"] * 5 + [False, False, False], \
        "a secret's value, as a name or a value, a secret's name or a shell variable, is never put into the environment"
    assert (held["9"], held["10"], held["11"]) == ("ran", "ran", "ran") and held["started"] == ["10", "11", "9"], held
    assert held["checking"] and not held["value logged"] and held["logged"].count("Refused (") == 16, held
    words = _fresh(tmp_path, "xe27_words", _XE27.replace("ARGS", """
    out[1] = tried(lambda: subprocess.run(mark(1) + ["abc1234"], env=ps.child_env()))
    out[2] = tried(lambda: subprocess.run(mark(2), env=ps.child_env()))
    out[3] = tried(lambda: subprocess.run(mark(3) + ["12345678"], env=ps.child_env()))
    out["status"] = {key: ps.status()[key] for key in ("short", "common", "looked_for_from")}
    """), given={PASS: "abc1234", KEYNAME: "12345678", "TOOL_WORD": "x-12345678-x"})
    assert (words["1"], words["2"], words["3"]) == ("ran", "ran", "ran"), words
    assert words["status"] == {"short": [PASS], "common": [KEYNAME], "looked_for_from": 8}, \
        "a value under 8 bytes, or one a variable holds already, is not looked for: it would refuse starts by chance"
    eight = _fresh(tmp_path, "xe27_eight", _XE27.replace("ARGS", """
    out[1] = tried(lambda: subprocess.run(mark(1) + ["12345678"], env=ps.child_env()))
    """), given={KEYNAME: "12345678"})
    assert eight["1"] == "refused", "a value of 8 bytes is looked for"
    ended = _fresh(tmp_path, "xe27_end", _XE27.replace("ARGS", """
    for name in [name for name in vars(ps) if not name.startswith("__")]:
        setattr(ps, name, None)
    with open(os.devnull, "rb"):
        out[1] = "opened"
    out[2] = tried(lambda: subprocess.run(mark(2), env={"PATH": "/bin"}))
    out[3] = tried(lambda: os.putenv("TOOL_PUT", "x"))
    """).replace('out["checking"] = bool(ps._checking)', 'out["checking"] = True'), given={PASS: phrase})
    assert (ended["1"], ended["2"], ended["3"]) == ("opened", "refused", "refused"), \
        "with the module's names gone, as the interpreter ends, the check still lets every other event through"
    empty = _fresh(tmp_path, "xe27_empty", _XE27.replace("ARGS", """
    out[1] = tried(lambda: subprocess.run(mark(1), env={"PATH": os.environ["PATH"], "TOOL": "anything"}))
    """))
    assert (empty["1"], empty["checking"]) == ("ran", False), "a process that holds no secret installs no check"
