#!/usr/bin/env python3
"""Contracts for ``oo garden``: the terminal's surface over the componion's service.

The garden runs in the terminal's own process. Its declarations live in
``cli/main.py``, its bodies in ``cli/garden.py``, imported when a command
runs; every body reaches the componion through one ``Garden`` of
``allium.service``, built by the ``allium_service`` factory of the root
object (the production one when none is given).

  * AF1 -- the terminal drives a whole first life under a fake clock and a
    temporary store: ``sow`` prints the card, every disclosure before its
    two questions, reads a name and a yes from stdin, and says what it
    sowed, the genome and the being named by the store; the trunk holds the
    genesis, no rhythm consented, then the name. ``show`` draws it;
    ``care water`` three days later writes one act and a settle keeps a
    state; the text tier draws nothing; ``keep verify`` finds every kept
    state in agreement. A sowing names only the channel's latest law, and
    refuses when it is retired, writing nothing; the production factory is
    never called when the test's is given. The two lines read from stdin are
    refused by name -- ``no``, an answer or a name the rules refuse, a line
    that is not UTF-8, no line, a line too long -- and nothing is sown; an
    unattended terminal is refused before the card is printed.
  * AF2 -- the cold footprints, measured then frozen, each in a fresh child
    process: importing the CLI loads its seven modules only; ``garden show``
    over a sown being, every seam by value and the data places behind the
    firewall, loads the garden's own closure and touches no data path; a
    garden switched off loads the settings, the service and the description
    and nothing of the store, the mode, the engine or the keys.
  * AF8 -- no text is read from the command line, and none is repeated: the
    ``garden`` subtree holds exactly its 28 commands, groups and hidden ones
    included, with no free-text parameter and every type a closed, quiet one
    (a scratch command with a text argument is found by the same walk);
    every command carries the settings that route extras and unknown options
    to its own refusal. A canary in every parameter slot the walk finds, as
    a positional extra or as an unknown option on every command, and as an
    unknown subtopic at every level, gives a usage exit and appears on
    neither stream, and no garden is built; typed on stdin, the same word
    becomes a name, and ``keep name`` writes nothing for an unattended
    terminal, a name the rules refuse or a line that is not UTF-8. ``share
    confirm`` refuses an unattended terminal and, attended, says that no
    consent request is open, writing nothing either way; a malformed code is
    a quiet usage error. Every attended verb of the service refuses a caller
    that is not an attended terminal. ``membrane.Transport(`` is called
    once, in ``service.transport``, and never by the terminal.
  * AF10 -- ``show`` draws the seed in text and ASCII from what it may read:
    the sample renderings, byte for byte, in both tiers, the text tier the
    ASCII one without the drawing; every row printable and within 78
    columns, the drawing within 32 with no ``@`` and no ``o``; real states
    read as awake, breathing, dormant by winter and dormant by drought; no
    count, chemistry, clock or genome of the state moves a byte, and the
    moisture does; no colour but the error prefix; the gallery prints the
    same renderings, under captions that pass the nets.
  * AF11 -- deep verification replays every kept state on the reference
    engine and compares the state served now: after a month of gestures
    every kept state agrees; a kept state forged with its hash and anchor
    rewritten is found, by day and law version, and nothing is replaced; a
    older kept state that could never be given to the engine (past its
    input limit) is a state that does not hold what it names; a stale one
    and one past the minute shown are counted, not refused; a being with
    none says so; a served state that disagrees with the replay is found;
    an engine that stops on a fault freezes the look on the stored state.
  * AF12 -- the laboratory shows the laws of its world, and ``keep laws``
    writes only what the diff confirmed: the laws in force, the proposal the
    same, then different; a wrong code writes nothing; the right one writes
    one law update, pending until its midnight and in force after it, cited
    by its event; a pin, a refused second pin, an unpin; an unattended
    terminal refused, an unpin too. Under a stable law a successor is
    available, and the next gesture carries it and says so. A new local
    offset is noted. Every look leaves the store's files and their bytes as
    they were and asks no terminal test; a gesture moves them.
  * AF13 -- the production wiring: the strong terminal test and its three
    conditions; the single-user reader over the auth settings and store,
    latching off once a second account exists, never asking for the auth
    module; ``Garden.production()`` passes the reader and nothing else to
    the one ``Store`` it builds; exactly one ``Store(`` call in the package,
    with that keyword alone. The reader, with the real keyed connector in a
    child, leaves an auth store's files and bytes as a stopped process left
    them: closed cleanly, frames never checkpointed, a hot journal (not
    single-user, and never rolled back).
  * AF14 -- the production look, behind the firewall's empty mirror, with
    the garden switched on: the soil is awaited, and the project modules,
    the data paths reached and the mirror's listing are the frozen ones. In
    a process that configures no logging, no platform log record reaches
    either stream while a garden command runs.

Local-only (the public distribution ships no tests). The platform and the
terminal load through the shared isolation window with the platform's
configuration, keys, mode, audit log and user modules proven unreachable;
every seam is injected (``tests/_allium_store_support.py``,
``tests/_allium_garden_support.py``). The footprint children run the
package as it is installed, in an empty directory, with no colour, API, key
or passphrase variable.
"""

import ast
import copy
import hashlib
import json
import re
import sqlite3
import sys
import threading
import types
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_garden_support as garden  # noqa: E402
import _allium_life_support as life_support  # noqa: E402
import _allium_store_support as support  # noqa: E402

BUDGET_S = {
    "test_af1_the_terminal_sows_shows_cares_for_and_verifies_a_seed_under_a_fake_clock": 2.0,
    "test_af2_the_cold_footprints_of_the_cli_and_the_garden_are_the_frozen_ones": 2.0,
    "test_af8_no_text_is_read_from_the_command_line_and_none_is_repeated": 2.0,
    "test_af10_show_draws_the_seed_from_what_it_may_read_in_text_and_ascii": 2.0,
    "test_af11_deep_verification_replays_every_kept_state_and_a_fault_freezes_the_look": 2.0,
    "test_af12_the_lab_shows_the_laws_and_keep_laws_writes_only_what_the_diff_confirmed": 2.0,
    "test_af13_the_production_wiring_passes_the_single_user_reader_and_nothing_else": 2.0,
    "test_af14_the_production_look_touches_only_the_frozen_modules_and_data_paths": 2.0,
}
PACKAGE = garden.REPO / "opti_oignon"
TESTS = garden.REPO / "tests"
DAY = 1440
DAY_S = DAY * 60

# The sow card of a fixture seed in an encrypted pot, the hemisphere left to its default and the band and the
# weather read from the settings file, then the answer to "Pip" and "yes" (the identity line follows).
CARD = (
    "This is a simulation of an onion. It does not feel anything; what it does\n"
    "follows the laws its laboratory names (oo garden lab laws). It will never ask\n"
    "you to come back.\n"
    "Prototype: this onion lives under a provisional world. It may have to be\n"
    "composted when the stable world arrives.\n"
    "It is a seed; in this version it does not grow further. Nothing about it\n"
    "leaves this machine.\n"
    "Hemisphere north (the default), daylight band long (from allium.yaml), weather\n"
    "garden (from allium.yaml). None of them changes after sowing.\n"
    "Learning your rhythm is not offered in this version: a seed sown now never\n"
    "learns it.\n"
    "There is no way yet to compost it or put it to rest. Turning the garden off\n"
    "hides it; its record stays on this machine.\n"
    "A name, or an empty line for none: 1 to 32 letters, digits, spaces, hyphens,\n"
    "apostrophes or periods. Every name given stays in its record for good.\n"
    "One seed; no preview and no second draw. Sow it? Answer yes or no:\n"
    "Prototype (provisional world): it may have to be composted later.\n"
    "Beetle: Sown. It is a seed, day 0.\n"
)
CONDITIONS = re.compile(r"^(Winter|Spring|Summer|Autumn) (in the garden|on a windowsill)\. "
                        r"(Sun down|Sun up|Sunrise|Sunset)\. (Dry|Damp|Wet) soil\. [A-Z][a-z ,]+\.$")
POT_RIM = "   ." + "-" * 24 + "."

# The cold footprints. The CLI's import and the switched-off garden were predicted before they were measured,
# and hold as written. The seamed show and the production look are frozen from their measurement (container,
# native artefact present, four runs alike): the show's project set less the native core, and its delta of 319
# modules over the child's setup, plus 5 %; the production look's project set, the data paths the firewall
# redirected (the planted one aside), and the mirror's listing after ``show``, ``lab`` and ``keep verify``.
CLI_MODULES = frozenset({"opti_oignon", "opti_oignon.__version__", "opti_oignon.cli", "opti_oignon.cli.client",
                         "opti_oignon.cli.config", "opti_oignon.cli.main", "opti_oignon.cli.output"})
CLI_CEILING = 400
DISABLED_MODULES = CLI_MODULES | {"opti_oignon.allium", "opti_oignon.allium.settings", "opti_oignon.allium.service",
                                  "opti_oignon.allium.describe", "opti_oignon.allium.wording",
                                  "opti_oignon.allium.ethics", "opti_oignon.cli.garden"}
DISABLED_ABSENT = ("opti_oignon.allium.mode", "opti_oignon.security_mode", "opti_oignon.allium.store",
                   "opti_oignon.allium.life", "opti_oignon.allium.engine", "opti_oignon.allium.ref",
                   "opti_oignon.db_utils", "opti_oignon.config", "opti_oignon.encryption", "opti_oignon.auth")
NATIVE_CORE = "opti_oignon.native.oo_core"
GARDEN_SHOW_MODULES = frozenset({
    "opti_oignon", "opti_oignon.__version__", "opti_oignon.allium", "opti_oignon.allium.anchors",
    "opti_oignon.allium.chain", "opti_oignon.allium.describe", "opti_oignon.allium.engine",
    "opti_oignon.allium.ethics", "opti_oignon.allium.evolution", "opti_oignon.allium.fx",
    "opti_oignon.allium.habitat", "opti_oignon.allium.lawfiles", "opti_oignon.allium.life",
    "opti_oignon.allium.membrane", "opti_oignon.allium.mode", "opti_oignon.allium.ref",
    "opti_oignon.allium.ref.civil", "opti_oignon.allium.ref.journal", "opti_oignon.allium.ref.lawdata",
    "opti_oignon.allium.ref.organs", "opti_oignon.allium.ref.organs.chem", "opti_oignon.allium.ref.organs.clock",
    "opti_oignon.allium.ref.organs.compile", "opti_oignon.allium.ref.organs.genome",
    "opti_oignon.allium.ref.organs.phon", "opti_oignon.allium.ref.organs.soil",
    "opti_oignon.allium.ref.organs.stage", "opti_oignon.allium.ref.organs.weather",
    "opti_oignon.allium.ref.protocol", "opti_oignon.allium.ref.world", "opti_oignon.allium.rng",
    "opti_oignon.allium.service", "opti_oignon.allium.settings", "opti_oignon.allium.store",
    "opti_oignon.allium.wire", "opti_oignon.allium.wording", "opti_oignon.cli", "opti_oignon.cli.client",
    "opti_oignon.cli.config", "opti_oignon.cli.garden", "opti_oignon.cli.main", "opti_oignon.cli.output",
    "opti_oignon.native"})
GARDEN_SHOW_DELTA = 319 * 105 // 100
PRODUCTION_LOOK_MODULES = frozenset({
    "opti_oignon", "opti_oignon.__version__", "opti_oignon.allium", "opti_oignon.allium.anchors",
    "opti_oignon.allium.describe", "opti_oignon.allium.ethics", "opti_oignon.allium.habitat",
    "opti_oignon.allium.life", "opti_oignon.allium.membrane", "opti_oignon.allium.mode",
    "opti_oignon.allium.service", "opti_oignon.allium.settings", "opti_oignon.allium.store",
    "opti_oignon.allium.wording", "opti_oignon.cli", "opti_oignon.cli.client", "opti_oignon.cli.config",
    "opti_oignon.cli.garden", "opti_oignon.cli.main", "opti_oignon.cli.output", "opti_oignon.config",
    "opti_oignon.db_encryption", "opti_oignon.db_utils", "opti_oignon.encryption", "opti_oignon.secure_bytes",
    "opti_oignon.security_mode", "opti_oignon.signed_audit_log"})
# The store's files are named by the owner tag of the single-user account, ``local``.
PRODUCTION_LOOK_PATHS = (
    "data/.keyfile", "data/.security_mode_lock", "data/audit_chain.db", "data/auth.db", "opti_oignon/data",
    "opti_oignon/data/allium", "opti_oignon/data/allium/94df8b67c73b0cf179fb8e88e34536a0.db",
    "opti_oignon/data/allium/94df8b67c73b0cf179fb8e88e34536a0.glass.db",
    "opti_oignon/data/allium/94df8b67c73b0cf179fb8e88e34536a0.sowing.db",
    "opti_oignon/data/allium/94df8b67c73b0cf179fb8e88e34536a0.sowing.glass.db",
    "opti_oignon/data/user_config.yaml")
PRODUCTION_LOOK_MIRROR = ("data", "opti_oignon", "opti_oignon/data")

# What every footprint child reports on its last stderr line.
_REPORT = '''
project = sorted(name for name in sys.modules if name == "opti_oignon" or name.startswith("opti_oignon."))
final = len(sys.modules)
'''

# A child that imports the CLI and nothing else.
CHILD_IMPORT = '''
import json, sys
import opti_oignon.cli.main
''' + _REPORT + '''
sys.stderr.write(json.dumps({"final": final, "project": project}) + "\\n")
'''

# The firewall with an empty mirror, and one planted data path stat-ed: its counter then reads one.
_FIREWALL = '''
import json, os, sqlite3, sys
from pathlib import Path
sys.path.insert(0, __TESTS__)
from _data_firewall import DataFirewall
firewall = DataFirewall(__REPO__, seed=False)
firewall.install()
try:
    os.stat(os.path.join(__REPO__, "data", "planted-" + __CANARY__))
except OSError:
    pass
planted = sum(len(found) for found in firewall.redirected.values())
'''

# A child that shows a being the parent sowed, every seam by value.
CHILD_SHOW = _FIREWALL + '''


class FileAudit:
    def __init__(self, path):
        self.path = path
        with open(path, encoding="ascii") as handle:
            self.entries = json.load(handle)
        self.calls = 0

    def _save(self):
        with open(self.path, "w", encoding="ascii") as handle:
            json.dump(self.entries, handle, sort_keys=True)

    def append_event(self, event_type, source="", action="", severity="INFO", details=None):
        self.calls += 1
        entry = {"action": action, "details": json.loads(json.dumps(details or {})), "event_type": event_type,
                 "id": len(self.entries) + 1, "severity": severity, "source": source}
        self.entries.append(entry)
        self._save()
        return entry["id"]

    def get_events(self, limit=50, offset=0, event_type=None, severity=None, after=None, before=None):
        self.calls += 1
        rows = [json.loads(json.dumps(entry)) for entry in reversed(self.entries)
                if event_type is None or entry["event_type"] == event_type]
        return rows[offset:offset + limit]

    def verify_chain(self):
        self.calls += 1
        return (True, None, len(self.entries))


def connect(path, check_same_thread=True, timeout=5.0):
    conn = sqlite3.connect(path, check_same_thread=check_same_thread, timeout=timeout)
    conn.execute("PRAGMA secure_delete = OFF")
    return conn


def identity(key, data):
    return bytes(data)


def factory():
    from opti_oignon.allium import service, store

    seams = {"single_user": lambda: True, "data_dir": Path(__DATA__), "persistence": __PERSISTENCE__,
             "connect": connect, "plain_connect": connect, "probe": lambda path, conn=None: "encrypted",
             "cipher_available": lambda: True,
             "anchor_secret": lambda: ("readable", bytes.fromhex(__KEY__), __KEY_ID__),
             "audit": FileAudit(__AUDIT__), "entropy": os.urandom, "cipher": (identity, identity),
             "clock": lambda: __WALL__, "mode": lambda: "daily", "tz": lambda now: 0, "laws": __LAWS__,
             "life": __LIFE__}
    return service.Garden(store_factory=lambda: store.Store(**seams), switch=lambda: "on", stopped=None,
                          attended=lambda: False, law="fixture")


setup = len(sys.modules)
from opti_oignon.cli import main
code = main.cli(["--no-color", "garden", "show"], standalone_mode=False, obj={"allium_service": factory})
''' + _REPORT + '''
from opti_oignon.allium import engine
native = engine.native_in_use()
counted = sum(len(found) for found in firewall.redirected.values())
firewall.uninstall()
sys.stderr.write(json.dumps({"code": code or 0, "counter": counted, "final": final, "native": native,
                             "planted": planted, "project": project, "setup": setup}) + "\\n")
'''

# A child that runs the production factory with the garden switched off.
CHILD_DISABLED = _FIREWALL + '''
from opti_oignon.allium import settings
settings.config_file = lambda: Path(__CONFIG__)
setup = len(sys.modules)
from opti_oignon.cli import main
code = main.cli(["--no-color", "garden", "show"], standalone_mode=False, obj={})
''' + _REPORT + '''
counted = sum(len(found) for found in firewall.redirected.values())
firewall.uninstall()
sys.stderr.write(json.dumps({"code": code or 0, "counter": counted, "final": final, "planted": planted,
                             "project": project, "setup": setup}) + "\\n")
'''

# A child that runs the production look, switched on, behind the empty mirror.
CHILD_PRODUCTION = _FIREWALL + '''
from opti_oignon.allium import settings
shipped = Path(__REPO__, "opti_oignon", "config", "allium.yaml").read_text(encoding="utf-8")
assert "enabled: false" in shipped
Path(__CONFIG__).write_text(shipped.replace("enabled: false", "enabled: true", 1), encoding="utf-8")
settings.config_file = lambda: Path(__CONFIG__)
from opti_oignon.cli import main
codes = [main.cli(["--no-color", "garden", *args], standalone_mode=False, obj={}) or 0
         for args in (["show"], ["lab"], ["keep", "verify"])]
''' + _REPORT + '''
paths = sorted(set().union(*firewall.redirected.values()))
listing = sorted(os.path.relpath(os.path.join(root, name), firewall.mirror)
                 for root, dirs, files in os.walk(firewall.mirror) for name in dirs + files)
firewall.uninstall()
sys.stderr.write(json.dumps({"codes": codes, "final": final, "listing": listing, "paths": paths,
                             "planted": planted, "project": project}) + "\\n")
'''


# A writer that leaves five auth stores as a process that stopped leaves them: one closed cleanly in WAL mode,
# three with their frames never checkpointed (the connection never closed), and one in rollback mode with a hot
# journal (a transaction cut in the middle). Its exit skips every close.
CHILD_AUTH_WRITER = '''
import os, sqlite3
root = __ROOT__
kept = []


def store(name):
    folder = os.path.join(root, name, "data")
    os.makedirs(folder)
    return sqlite3.connect(os.path.join(folder, "auth.db"), isolation_level=None)


clean = store("clean")
clean.execute("PRAGMA journal_mode=WAL")
clean.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)")
clean.execute("INSERT INTO users (username) VALUES ('one')")
clean.close()
for name, users in (("pending", 1), ("pending2", 2), ("control", 2)):
    conn = store(name)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)")
    for i in range(users):
        conn.execute("INSERT INTO users (username) VALUES (?)", ("user%d" % i,))
    kept.append(conn)
hot = store("hot")
hot.execute("PRAGMA journal_mode=DELETE")
hot.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)")
hot.execute("INSERT INTO users (username) VALUES ('one')")
hot.execute("PRAGMA cache_size=1")
hot.execute("BEGIN")
for i in range(3000):
    hot.execute("INSERT INTO users (username) VALUES (?)", ("x" * 40,))
kept.append(hot)
os._exit(0)
'''

# The production single-user reader, the real connector behind the firewall, over those stores: its answer, and
# whether each store's files and their bytes were left as they were; then an ordinary connection to the control
# store, which must be seen to change it.
CHILD_AUTH_READER = '''
import hashlib, json, os, sys
sys.path.insert(0, __TESTS__)
from _data_firewall import DataFirewall
firewall = DataFirewall(__REPO__, seed=False)
firewall.install()
from opti_oignon import db_utils
from opti_oignon.allium import service


def listing(folder):
    out = {}
    for name in sorted(os.listdir(folder)):
        with open(os.path.join(folder, name), "rb") as handle:
            out[name] = hashlib.sha256(handle.read()).hexdigest()
    return out


found = {}
for name in ("clean", "pending", "hot", "pending2"):
    folder = os.path.join(__ROOT__, name, "data")
    before = listing(folder)
    answer = service.platform_single_user(config=__CONFIG__, root=os.path.join(__ROOT__, name))
    found[name] = [answer, listing(folder) == before, sorted(before)]
folder = os.path.join(__ROOT__, "control", "data")
before = listing(folder)
conn = db_utils.safe_connect(os.path.join(folder, "auth.db"))
count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
conn.close()
found["control"] = [count, listing(folder) == before, sorted(before)]
firewall.uninstall()
sys.stderr.write(json.dumps(found) + "\\n")
'''


# A child with no log configuration at all, as ``oo`` runs: a record no handler takes reaches stderr through
# logging's last resort (the witness, before and after), and no garden command lets one through. Each command's
# two streams are captured apart: a switch file that cannot be read, then the garden on with a plaintext auth
# store in the mirror (the keyed connector warns when it opens one in clear), a look and a refused gesture.
CHILD_LOGS = _FIREWALL + '''
import io, logging
from opti_oignon.allium import settings
from opti_oignon.cli import main


def captured(call):
    out, err = io.StringIO(), io.StringIO()
    real = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        code = call()
    finally:
        sys.stdout, sys.stderr = real
    return [code, out.getvalue(), err.getvalue()]


def garden(*args):
    return captured(lambda: main.cli(["--no-color", "garden", *args], standalone_mode=False, obj={}) or 0)


def record():
    logging.getLogger("opti_oignon.allium.settings").warning("a record no handler takes")
    return 0


report = {"handlers": len(logging.getLogger().handlers), "before": captured(record)}
config = Path(__CONFIG__)
settings.config_file = lambda: config
config.write_text("enabled: [true\\n", encoding="utf-8")
report["unreadable"] = garden("show")
shipped = Path(__REPO__, "opti_oignon", "config", "allium.yaml").read_text(encoding="utf-8")
config.write_text(shipped.replace("enabled: false", "enabled: true", 1), encoding="utf-8")
auth = sqlite3.connect(os.path.join(__REPO__, "data", "auth.db"))
auth.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)")
auth.execute("INSERT INTO users (username) VALUES ('one')")
auth.commit()
auth.close()
report["show"] = garden("show")
report["care"] = garden("care", "water")
report["after"] = captured(record)
from opti_oignon import db_encryption
from opti_oignon.allium import service
db_encryption._plaintext_warned = False
report["plaintext"] = captured(lambda: service.platform_single_user())
report["mirror"] = "data/auth.db" in set().union(*firewall.redirected.values())
firewall.uninstall()
sys.stderr.write(json.dumps(report) + "\\n")
'''


def _code(template, **values):
    """A child's code with its placeholders filled by the Python literals of ``values``."""
    out = template
    for name, value in values.items():
        out = out.replace("__" + name.upper() + "__", repr(value))
    assert "__" not in out.replace("__name__", "").replace("__init__", ""), "every placeholder is filled"
    return out


# The sample felts of the renderings: the default felt with these fields changed.
FELTS = {
    'F1': {},
    'F2': {'name': 'Pip', 'day': 52, 'season': 0, 'light': 'down', 'life': 'dormant_winter', 'minute': 413, 'daylength': 492, 'sun': 0, 'local': '2025-11-30 06:53'},
    'F2u': {'name': 'Pip', 'day': 52, 'season': 0, 'light': 'down', 'life': 'dormant_winter', 'minute': 413, 'daylength': 492, 'sun': 0, 'layer': 'bulbe', 'labels': ('prototype', 'mode_unknown'), 'local': '2025-11-30 06:53'},
    'F3': {'day': 18, 'place': 'windowsill', 'light': 'down', 'soil': 'dry', 'life': 'dormant_dry', 'minute': 173, 'daylength': 560, 'sun': 0, 'layer': 'bulbe', 'labels': ('prototype', 'bulbe'), 'local': '2025-10-27 02:53'},
    'F4': {'name': 'Pip', 'day': 7, 'season': 1, 'light': 'rise', 'soil': 'damp', 'life': 'breathing', 'minute': 390, 'daylength': 700, 'sun': 30000, 'jar': True, 'labels': ('prototype', 'glass_jar'), 'local': '2025-10-16 06:30'},
    'F5': {'jar': True, 'labels': ('prototype', 'glass_jar', 'catching_up')},
    'frozen': {'name': 'Pip', 'day': 12, 'light': 'set', 'soil': 'damp', 'life': 'breathing', 'minute': 1000, 'daylength': 600, 'sun': 20000, 'labels': ('prototype', 'frozen'), 'local': '2025-10-21 16:40', 'offset': '+02:00'},
}
# Their exact rows, in both tiers.
GOLDENS = {
    ('F1', 'ascii'): (
        'Prototype (provisional world): it may have to be composted later.',
        '       (*)',
        '',
        '   .------------------------.',
        '    \\~~~~~~~~~~~~~~~~~~~~~~/',
        '     \\~~~~~~~~~~~~~~~~~~~~/',
        '      \\~~~~~~~(..)~~~~~~~/',
        '       \\~~~~~~~~~~~~~~~~/',
        "        `--------------'",
        '"  "  "  "  "  "  "  "  "  "  "',
        'Onion, seed, day 0.',
        'Autumn in the garden. Sun up. Wet soil. Awake.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
    ('F1', 'text'): (
        'Prototype (provisional world): it may have to be composted later.',
        'Onion, seed, day 0.',
        'Autumn in the garden. Sun up. Wet soil. Awake.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
    ('F2', 'ascii'): (
        'Prototype (provisional world): it may have to be composted later.',
        '                          ( )',
        '',
        '   .------------------------.',
        "    \\''''''''''''''''''''''/",
        '     \\~~~~~~~~~~~~~~~~~~~~/',
        '      \\~~~~~~~(--)~~~~~~~/',
        '       \\~~~~~~~~~~~~~~~~/',
        "        `--------------'",
        '"  "  "  "  "  "  "  "  "  "  "',
        'Pip, seed, day 52.',
        'Winter in the garden. Sun down. Wet soil. Went dormant when winter came.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
    ('F2', 'text'): (
        'Prototype (provisional world): it may have to be composted later.',
        'Pip, seed, day 52.',
        'Winter in the garden. Sun down. Wet soil. Went dormant when winter came.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
    ('F2u', 'ascii'): (
        'Prototype (provisional world): it may have to be composted later.',
        "The security mode cannot be read, so Bulbe's rules apply. Its life goes on.",
        '                          ( )',
        '',
        '   .========================.',
        "    \\''''''''''''''''''''''/",
        '     \\~~~~~~~~~~~~~~~~~~~~/',
        '      \\~~~~~~~(--)~~~~~~~/',
        '       \\~~~~~~~~~~~~~~~~/',
        "        `--------------'",
        '"  "  "  "  "  "  "  "  "  "  "',
        'Pip, seed, day 52.',
        'Winter in the garden. Sun down. Wet soil. Went dormant when winter came.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
    ('F2u', 'text'): (
        'Prototype (provisional world): it may have to be composted later.',
        "The security mode cannot be read, so Bulbe's rules apply. Its life goes on.",
        'Pip, seed, day 52.',
        'Winter in the garden. Sun down. Wet soil. Went dormant when winter came.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
    ('F3', 'ascii'): (
        'Prototype (provisional world): it may have to be composted later.',
        'Bulbe: this machine is in Bulbe mode. Its life goes on as in Daily.',
        '                          ( )',
        '',
        '   .========================.',
        '    \\ . . . . . . . . . . ./',
        '     \\. . . . . . . . . . /',
        '      \\ . . . (--). . . ./',
        '       \\. . . . . . . . /',
        "        `--------------'",
        '________________________________',
        'Onion, seed, day 18.',
        'Autumn on a windowsill. Sun down. Dry soil. Went dormant in a dry spell.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
    ('F3', 'text'): (
        'Prototype (provisional world): it may have to be composted later.',
        'Bulbe: this machine is in Bulbe mode. Its life goes on as in Daily.',
        'Onion, seed, day 18.',
        'Autumn on a windowsill. Sun down. Dry soil. Went dormant in a dry spell.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
    ('F4', 'ascii'): (
        'Prototype (provisional world): it may have to be composted later.',
        'This onion lives in a glass jar: its store is not encrypted.',
        '',
        '   (*)',
        '     .--------------------.',
        '     |                    |',
        '     |::::::::::::::::::::|',
        '     |::::::::(..)::::::::|',
        '     |::::::::::::::::::::|',
        "     '--------------------'",
        '"  "  "  "  "  "  "  "  "  "  "',
        'Pip, seed, day 7.',
        'Spring in the garden. Sunrise. Damp soil. Awake, breathing.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
    ('F4', 'text'): (
        'Prototype (provisional world): it may have to be composted later.',
        'This onion lives in a glass jar: its store is not encrypted.',
        'Pip, seed, day 7.',
        'Spring in the garden. Sunrise. Damp soil. Awake, breathing.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
    ('F5', 'ascii'): (
        'Prototype (provisional world): it may have to be composted later.',
        'This onion lives in a glass jar: its store is not encrypted.',
        'Shown as of 2025-10-09 08:53 (+00:00): its life after that is not computed',
        'yet.',
        '       (*)',
        '',
        '     .--------------------.',
        '     |                    |',
        '     |~~~~~~~~~~~~~~~~~~~~|',
        '     |~~~~~~~~(..)~~~~~~~~|',
        '     |~~~~~~~~~~~~~~~~~~~~|',
        "     '--------------------'",
        '"  "  "  "  "  "  "  "  "  "  "',
        'Onion, seed, day 0.',
        'Autumn in the garden. Sun up. Wet soil. Awake.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
    ('F5', 'text'): (
        'Prototype (provisional world): it may have to be composted later.',
        'This onion lives in a glass jar: its store is not encrypted.',
        'Shown as of 2025-10-09 08:53 (+00:00): its life after that is not computed',
        'yet.',
        'Onion, seed, day 0.',
        'Autumn in the garden. Sun up. Wet soil. Awake.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
    ('frozen', 'ascii'): (
        'Prototype (provisional world): it may have to be composted later.',
        'Shown as of 2025-10-21 16:40 (+02:00): the engine stopped on a fault after',
        'that. Nothing was replaced.',
        '',
        '                         (*)',
        '   .------------------------.',
        '    \\::::::::::::::::::::::/',
        '     \\::::::::::::::::::::/',
        '      \\:::::::(..):::::::/',
        '       \\::::::::::::::::/',
        "        `--------------'",
        '"  "  "  "  "  "  "  "  "  "  "',
        'Pip, seed, day 12.',
        'Autumn in the garden. Sunset. Damp soil. Awake, breathing.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
    ('frozen', 'text'): (
        'Prototype (provisional world): it may have to be composted later.',
        'Shown as of 2025-10-21 16:40 (+02:00): the engine stopped on a fault after',
        'that. Nothing was replaced.',
        'Pip, seed, day 12.',
        'Autumn in the garden. Sunset. Damp soil. Awake, breathing.',
        'This is a simulation of an onion. It does not feel anything; what it does',
        'follows the laws its laboratory names (oo garden lab laws). It will never ask',
        'you to come back.',
    ),
}

# The garden subtree: every command and group, the hidden ones included.
TREE = {
    ("garden",), ("garden", "show"), ("garden", "sow"),
    ("garden", "care"), ("garden", "care", "greet"), ("garden", "care", "water"), ("garden", "care", "warm"),
    ("garden", "care", "play"),
    ("garden", "lab"), ("garden", "lab", "laws"),
    ("garden", "keep"), ("garden", "keep", "verify"), ("garden", "keep", "name"), ("garden", "keep", "laws"),
    ("garden", "keep", "laws", "diff"), ("garden", "keep", "laws", "apply"), ("garden", "keep", "laws", "pin"),
    ("garden", "keep", "laws", "unpin"), ("garden", "keep", "resume"), ("garden", "keep", "finish"),
    ("garden", "keep", "celebrate"), ("garden", "keep", "bury"),
    ("garden", "lang"), ("garden", "lang", "talk"), ("garden", "lang", "teach"),
    ("garden", "tray"),
    ("garden", "share"), ("garden", "share", "confirm"),
}
HIDDEN = {("garden", "lang"), ("garden", "lang", "talk"), ("garden", "lang", "teach"), ("garden", "tray"),
          ("garden", "share"), ("garden", "share", "confirm"), ("garden", "keep", "celebrate"),
          ("garden", "keep", "bury")}
# Every parameter slot that takes a value: (path, parameter name).
SLOTS = {
    (("garden", "show"), "tier"), (("garden", "sow"), "hemisphere"), (("garden", "sow"), "band"),
    (("garden", "sow"), "weather"), (("garden", "keep", "resume"), "kept"), (("garden", "keep", "resume"), "discarded"),
    (("garden", "keep", "finish"), "tag"), (("garden", "keep", "laws", "apply"), "confirm"),
    (("garden", "share", "confirm"), "code"),
}
QUIET = ("_QuietChoice", "_Count", "_Hex", "_ConsentCode")
CODE = "K7QX-MPLR"
# Values each closed type accepts, tried in turn to fill the slots that are not the canary's.
FILLERS = ("ascii", "north", "long", "garden", "1", "0" * 8, "0" * 16, CODE)


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


@pytest.fixture
def p(monkeypatch, tmp_path):
    window, restore = garden.open_garden(monkeypatch, tmp_path)
    try:
        yield window
    finally:
        restore()


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------
def _walk(command, path=("garden",)):
    """Every command under ``command``, groups and hidden ones included: ``[(path, command)]``."""
    import click

    found = [(path, command)]
    if isinstance(command, click.MultiCommand):
        ctx = click.Context(command, info_name=path[-1])
        for name in command.list_commands(ctx):
            found += _walk(command.get_command(ctx, name), path + (name,))
    return found


def _type_findings(walked):
    """``(path, parameter)`` of every parameter whose type is free text or not one of the quiet ones."""
    import click

    findings = []
    for path, command in walked:
        for param in command.params:
            flag = isinstance(param, click.Option) and param.is_flag
            if param.type is click.STRING or not (flag or type(param.type).__name__ in QUIET):
                findings.append((path, param.name))
    return findings


def _settings_findings(walked):
    return [path for path, command in walked
            if command.context_settings.get("ignore_unknown_options") is not True
            or command.context_settings.get("allow_extra_args") is not True]


def _filler(param):
    """A value the parameter's closed type accepts."""
    for value in FILLERS:
        try:
            param.type.convert(value, param, None)
        except Exception:  # noqa: BLE001 - not a value of this type
            continue
        return value
    raise AssertionError(f"no filler for {param.name}")


def _arguments(command, canary_at=None, canary=None):
    """The command's positional arguments, each filled, the canary at index ``canary_at``."""
    import click

    out = []
    for index, param in enumerate(q for q in command.params if isinstance(q, click.Argument)):
        out.append(canary if index == canary_at else _filler(param))
    return out


def _calls(tree, name):
    """``(function or None, call)`` of every call to ``name`` in ``tree``, as a name or an attribute."""
    found = []

    def visit(node, owner):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else owner
            if isinstance(child, ast.Call):
                func = child.func
                if (isinstance(func, ast.Name) and func.id == name) or (
                        isinstance(func, ast.Attribute) and func.attr == name):
                    found.append((owner, child))
            visit(child, inner)

    visit(tree, None)
    return found


# ---------------------------------------------------------------------------
# AF8 -- no text in argv, and none repeated
# ---------------------------------------------------------------------------
def test_af8_no_text_is_read_from_the_command_line_and_none_is_repeated(p, tmp_path):
    import click

    # Static: the subtree, its types and its settings.
    group = p.main.cli.commands["garden"]
    walked = _walk(group)
    assert len(walked) == 28, [path for path, _command in walked]
    assert {path for path, _command in walked} == TREE
    assert {path for path, command in walked if command.hidden} == HIDDEN
    root = click.Context(group, info_name="garden")
    assert {name for name in group.list_commands(root) if not group.get_command(root, name).hidden} == \
        {"show", "sow", "care", "lab", "keep"}, "five subtopics in the help, three hidden"
    assert _type_findings(walked) == [], _type_findings(walked)
    assert _settings_findings(walked) == [], _settings_findings(walked)
    slots = {(path, param.name) for path, command in walked for param in command.params
             if not (isinstance(param, click.Option) and param.is_flag)}
    assert slots == SLOTS, slots
    lang = dict(walked)[("garden", "lang", "talk")]
    scratch = click.Group("lang", commands={"talk": click.Command(
        "talk", params=[click.Argument(["text"])], context_settings=dict(lang.context_settings))})
    assert _type_findings(_walk(scratch, ("garden", "lang"))) == [(("garden", "lang", "talk"), "text")], \
        "witness: a text argument is found by the walk"

    # Behaviour: a canary in every slot, as an extra, as an unknown option and as an unknown subtopic.
    given = support.seams(p, tmp_path.joinpath("garden"), suite="af8")
    factory = garden.garden(p, given, attended=True)
    canary = garden.canary(p, "af8")
    shapes = []
    commands = dict(walked)
    for path, name in sorted(SLOTS):
        command = commands[path]
        param = next(param for param in command.params if param.name == name)
        if isinstance(param, click.Argument):
            index = [q.name for q in command.params if isinstance(q, click.Argument)].index(name)
            shapes.append(list(path[1:]) + _arguments(command, index, canary))
        else:
            shapes.append(list(path[1:]) + _arguments(command) + [param.opts[0], canary])
    for path, command in walked:
        filled = _arguments(command)
        shapes.append(list(path[1:]) + filled + [canary])
        shapes.append(list(path[1:]) + filled + ["--x=" + canary])
    shapes.append(["show", "--", canary])
    assert len(shapes) == len(SLOTS) + 2 * 28 + 1
    for argv in shapes:
        result = garden.invoke(p, argv, factory=factory)
        assert result.exit_code == 2, (argv, result.exit_code, result.stdout, result.stderr, result.exception)
        assert canary not in result.stdout and canary not in result.stderr, (argv, result.stdout, result.stderr)
        assert result.stdout == "", (argv, result.stdout)
    assert factory.calls == 0, "no garden is built for a refused command line"
    refused = garden.invoke(p, ["keep", canary], factory=factory)
    assert garden.says(refused.stderr, p, "refuse.subtopic"), refused.stderr
    extra = garden.invoke(p, ["show", canary], factory=factory)
    assert garden.says(extra.stderr, p, "refuse.argv"), extra.stderr

    # Control: typed on stdin, the canary is a name, written as a name fact.
    target = support.store(p, given)
    try:
        support.sow(p, target)
    finally:
        target.close()
    named = garden.invoke(p, ["keep", "name"], input=canary + "\n", factory=factory)
    assert named.exit_code == 0, (named.stdout, named.stderr, named.exception)
    target = support.store(p, given)
    try:
        facts = [fact for _seq, _eid, fact in target.open("local").in_order()]
    finally:
        target.close()
    assert [fact["body"] for fact in facts if fact["kind"] == "name"] == [{"name": canary}]

    # keep name writes nothing for an unattended terminal, a name the rules refuse, or a line that is not UTF-8.
    factory.attended = False
    alone = garden.invoke(p, ["keep", "name"], input="Pip\n", factory=factory)
    assert (alone.exit_code, alone.stdout) == (1, "") and garden.says(alone.stderr, p, "refuse.attended"), \
        (alone.stdout, alone.stderr, alone.exception)
    factory.attended = True
    for stdin in ("Beetle\n", "beetle pip\n", "Pip  Pip\n", "-Pip\n", "\n", b"Pi\xffp\n"):
        refused = garden.invoke(p, ["keep", "name"], input=stdin, factory=factory)
        assert refused.exit_code == 2 and garden.says(refused.stderr, p, "refuse.name"), \
            (stdin, refused.stdout, refused.stderr, refused.exception)
    target = support.store(p, given)
    try:
        names = [fact["body"] for _seq, _eid, fact in target.open("local").in_order() if fact["kind"] == "name"]
    finally:
        target.close()
    assert names == [{"name": canary}], names

    # The terminal clause of share confirm: refused either way, nothing written.
    path = support.store_path(p, given)
    before = garden.sha256(path)
    factory.attended = False
    alone = garden.invoke(p, ["share", "confirm", CODE], factory=factory)
    assert alone.exit_code == 1 and garden.says(alone.stderr, p, "refuse.attended"), (alone.stderr, alone.exception)
    factory.attended = True
    attended = garden.invoke(p, ["share", "confirm", CODE.lower()], factory=factory)
    assert attended.exit_code == 1 and garden.says(attended.stderr, p, "refuse.share.none"), \
        (attended.stderr, attended.exception)
    assert garden.sha256(path) == before, "share confirm writes nothing"

    # Every attended verb refuses a caller that is not an attended terminal, whatever the terminal test says.
    web = p.service.transport("web", attended=True, principal="someone")
    unattended = p.service.transport("cli", attended=False)
    gardener = factory()
    outcomes = {}
    try:
        verbs = {
            "sow_card": lambda caller: gardener.sow_card(caller),
            "sow": lambda caller: gardener.sow(caller, name="Pip"),
            "name_card": lambda caller: gardener.name_card(caller),
            "name": lambda caller: gardener.name("Pip", caller),
            "laws_apply": lambda caller: gardener.laws_apply("0" * 16, caller),
            "laws_pin": lambda caller: gardener.laws_pin(caller),
            "laws_unpin": lambda caller: gardener.laws_unpin(caller),
            "resume": lambda caller: gardener.resume(1, 1, caller),
            "finish": lambda caller: gardener.finish("0" * 8, caller),
            "confirm_share": lambda caller: gardener.confirm_share(CODE, caller),
        }
        for name, call in verbs.items():
            for label, caller in (("web", web), ("unattended", unattended)):
                try:
                    call(caller)
                    outcomes[(name, label)] = "done"
                except p.service.ServiceRefused as exc:
                    outcomes[(name, label)] = exc.code
                except p.membrane.MembraneRefused as exc:
                    outcomes[(name, label)] = "membrane " + exc.code
    finally:
        gardener.close()
    expected = {key: "attended" for key in outcomes}
    expected[("confirm_share", "unattended")] = "membrane attended"
    assert len(outcomes) == 20 and outcomes == expected, outcomes
    assert factory.attended is True and garden.sha256(path) == before, "nothing written for any of them"
    calls = factory.calls
    for malformed in ("K7QX-MP0R", "K7QXMPLR", "K7QX-MPLRR", "K7Q-XMPLR"):
        result = garden.invoke(p, ["share", "confirm", malformed], factory=factory)
        assert result.exit_code == 2, (malformed, result.exit_code)
        assert malformed not in result.stdout + result.stderr, (malformed, result.stderr)
    assert factory.calls == calls, "a malformed code builds no garden"

    # Census: one Transport constructor, in service.transport; none in the terminal.
    service_calls = _calls(ast.parse((PACKAGE / "allium" / "service.py").read_text(encoding="ascii")), "Transport")
    assert [owner for owner, _call in service_calls] == ["transport"], service_calls
    for source in sorted((PACKAGE / "cli").glob("*.py")):
        assert _calls(ast.parse(source.read_text(encoding="ascii")), "Transport") == [], source.name
    planted = ast.parse("from opti_oignon.allium import membrane\ndef f():\n    return membrane.Transport('cli')\n")
    assert [owner for owner, _call in _calls(planted, "Transport")] == ["f"], "witness: the census counts a call"


# ---------------------------------------------------------------------------
# AF1 -- a first life, from the terminal
# ---------------------------------------------------------------------------
def _in_order(p, given):
    target = support.store(p, given)
    try:
        being = target.open("local")
        return being, [fact for _seq, _eid, fact in being.in_order()], being.view(to=0)
    finally:
        target.close()


def test_af1_the_terminal_sows_shows_cares_for_and_verifies_a_seed_under_a_fake_clock(p, tmp_path):
    say = p.wording.say
    production = []

    def counting(*args, **kwargs):
        production.append(1)
        raise AssertionError("the production factory is not called when the test's is given")

    real_production = p.service.Garden.__dict__["production"]
    p.service.Garden.production = staticmethod(counting)
    try:
        laws = copy.deepcopy(support.seams(p, tmp_path, suite="af1")["laws"])
        laws["seasons"] = {"default_band": "long"}
        given = support.seams(p, tmp_path.joinpath("garden"), suite="af1", laws=laws)
        factory = garden.garden(p, given, attended=True)

        # Sow: the card, its two questions, then what was sown.
        sown = garden.invoke(p, ["sow"], input="Pip\nyes\n", factory=factory)
        assert sown.exit_code == 0, (sown.stdout, sown.stderr, sown.exception)
        being, facts, zero = _in_order(p, given)
        genome, tag = zero.state["genome"][:12], being.being_tag[:8]
        assert sown.stdout == CARD + f"Genome {genome}. Being {tag}.\n", sown.stdout
        assert sown.stdout == garden.printed(
            p, say("doctrine"), say("label.prototype"), say("sow.sees"),
            say("sow.place", hemisphere="north", hemisphere_from="default", band="long", band_from="file",
                weather="garden", weather_from="file"),
            say("sow.rhythm"), say("sow.exit"), say("sow.ask.name"), say("sow.ask.confirm"),
            say("label.prototype.short"), say("sow.done"), say("sow.identity", genome=genome, tag=tag))
        assert sown.stderr == ""
        assert [fact["kind"] for fact in facts] == ["genesis", "name"], facts
        assert facts[0]["body"]["rhythm_consent"] is False and facts[1]["body"] == {"name": "Pip"}
        assert facts[0]["body"]["laws"]["name"] == "fixture" and facts[0]["body"]["hemisphere"] == "north"

        # Show: the name, a conditions line, a drawing.
        shown = garden.invoke(p, ["show"], factory=factory)
        rows = shown.stdout.splitlines()
        assert shown.exit_code == 0 and "Pip, seed, day 0." in rows, shown.stdout
        assert any(CONDITIONS.match(row) for row in rows) and POT_RIM in rows, shown.stdout

        # Three days later, a gesture: one act, acknowledged, and a state kept by the settle after it.
        given["clock"].advance_days(3)
        cared = garden.invoke(p, ["care", "water"], factory=factory)
        assert cared.exit_code == 0 and "Beetle: Water noted." in cared.stdout.splitlines(), cared.stdout
        being, facts, _zero = _in_order(p, given)
        assert [fact["body"] for fact in facts if fact["kind"] == "act"] == [{"act": "water"}], facts
        path = support.store_path(p, given)
        [(kept,)] = support.read(path, "SELECT COUNT(*) FROM checkpoints")
        assert kept >= 1, "the settle after the gesture kept a state"

        text = garden.invoke(p, ["show", "--tier", "text"], factory=factory)
        assert text.exit_code == 0 and "Pip, seed, day 3." in text.stdout.splitlines(), text.stdout
        assert POT_RIM not in text.stdout and "\\" not in text.stdout and "(..)" not in text.stdout

        # Deep verification: the chain, every kept state in agreement, the state served now.
        verified = garden.invoke(p, ["keep", "verify"], factory=factory)
        assert verified.exit_code == 0, (verified.stdout, verified.stderr, verified.exception)
        assert garden.shows(verified.stdout, p, "verify.chain", facts=len(facts), days=3), verified.stdout
        found = re.search(r"(\d+) of (\d+) kept states agree", garden.flat(verified.stdout))
        assert found and found.group(1) == found.group(2) and int(found.group(1)) >= 1, verified.stdout
        assert int(found.group(1)) == kept
        assert garden.shows(verified.stdout, p, "verify.kept", agreed=kept, kept=kept)
        assert garden.shows(verified.stdout, p, "verify.served", engine="reference"), verified.stdout

        # The channel: the latest law, never the fixture; retired, nothing is sown.
        assert p.service.sowing_law() == "v0_1" and "fixture" not in p.service.CHANNEL
        register = p.lawfiles.retired
        pair = {"name": "v0_1", "sha256": p.lawfiles.digest(p.lawfiles.law("v0_1"))}
        fresh = support.seams(p, tmp_path.joinpath("channel"), suite="af1", index=1)
        try:
            p.lawfiles.retired = lambda: [dict(pair)]
            with pytest.raises(p.service.ServiceRefused) as refusal:
                p.service.sowing_law()
            assert refusal.value.code == "channel"
            refused = garden.invoke(p, ["sow"], input="Pip\nyes\n",
                                    factory=garden.garden(p, fresh, attended=True, law=None))
        finally:
            p.lawfiles.retired = register
        assert refused.exit_code == 1 and garden.says(refused.stderr, p, "refuse.channel"), refused.stderr
        assert not support.store_path(p, fresh).exists() and not support.store_path(p, fresh, suffix=".glass.db").exists()
        assert p.service.sowing_law() == "v0_1", "witness: registered no longer, it is the channel's law again"

        # The two lines read from stdin, refused by name: nothing is sown in any refused case.
        card = garden.printed(p, *p.describe.card(p.service.Card(
            look=None, law="fixture", provisional=True, soil="encrypted",
            place=(("hemisphere", "north", "default"), ("band", "long", "file"), ("weather", "garden", "file")))))
        cases = (
            ("no", b"Pip\nno\n", 1, "sow.cancelled", "stdout"),
            ("an answer that is not yes or no", b"Pip\nmaybe\n", 2, "refuse.answer", "stderr"),
            ("a name the rules refuse", b"Beetle\nyes\n", 2, "refuse.name", "stderr"),
            ("a name that is not UTF-8", b"Pi\xffp\nyes\n", 2, "refuse.name", "stderr"),
            ("an answer that is not UTF-8", b"Pip\ny\xffs\n", 2, "refuse.answer", "stderr"),
            ("no line at all", b"", 2, "refuse.eof", "stderr"),
            ("the name and no answer", b"Pip\n", 2, "refuse.eof", "stderr"),
            ("a line of 256 characters without its end", b"x" * 300, 2, "refuse.line", "stderr"),
        )
        for index, (case, stdin, code, key, stream) in enumerate(cases, start=2):
            fresh = support.seams(p, tmp_path.joinpath(f"stdin-{index}"), suite="af1", index=index, laws=laws)
            result = garden.invoke(p, ["sow"], input=stdin, factory=garden.garden(p, fresh, attended=True))
            assert result.exit_code == code, (case, result.exit_code, result.stdout, result.stderr, result.exception)
            assert result.stdout.startswith(card), ("the card and its questions come first", case, result.stdout)
            said = result.stdout[len(card):] if stream == "stdout" else result.stderr
            assert garden.says(said, p, key), (case, said)
            assert not support.store_path(p, fresh).exists(), ("nothing is sown", case)
        refused_name = garden.invoke(p, ["sow"], input="Beetle\n", factory=garden.garden(
            p, support.seams(p, tmp_path.joinpath("stdin-early"), suite="af1", index=len(cases) + 2, laws=laws),
            attended=True))
        assert refused_name.exit_code == 2 and garden.says(refused_name.stderr, p, "refuse.name"), \
            "a refused name is refused before the confirmation is read"

        # An unattended terminal is refused the sowing before its card is printed or a line is read.
        alone = support.seams(p, tmp_path.joinpath("alone"), suite="af1", index=len(cases) + 3, laws=laws)
        lonely = garden.garden(p, alone, attended=False)
        result = garden.invoke(p, ["sow"], input="Pip\nyes\n", factory=lonely)
        assert (result.exit_code, result.stdout) == (1, ""), (result.stdout, result.stderr, result.exception)
        assert garden.says(result.stderr, p, "refuse.attended"), result.stderr
        assert lonely.attended_reads == 1 and not support.store_path(p, alone).exists()
    finally:
        p.service.Garden.production = real_production
    assert production == [], "the production factory was never called"


# ---------------------------------------------------------------------------
# AF2 -- the cold footprints, in fresh children
# ---------------------------------------------------------------------------
def _identity(key, data):
    return bytes(data)


def test_af2_the_cold_footprints_of_the_cli_and_the_garden_are_the_frozen_ones(p, tmp_path):
    say = p.wording.say
    canary = garden.canary(p, "af2")

    # The parent sows the being child (b) shows, with the same cipher stand-in and an audit log on file.
    home = tmp_path.joinpath("sown")
    home.mkdir()
    audit = home.joinpath("audit.json")
    given = support.seams(p, home, suite="af2", audit=garden.FileAudit(audit), cipher=(_identity, _identity))
    target = support.store(p, given)
    try:
        support.sow(p, target)
    finally:
        target.close()
    wall = support.WALL + 2 * DAY_S
    config = tmp_path.joinpath("off", "allium.yaml")
    config.parent.mkdir()
    config.write_text("enabled: false\n", encoding="ascii")
    common = {"tests": str(TESTS), "repo": str(garden.REPO), "canary": canary}
    children = {
        "import": garden.child(tmp_path, CHILD_IMPORT, cwd=tmp_path.joinpath("cwd-import")),
        "show": garden.child(tmp_path, _code(
            CHILD_SHOW, data=str(given["data_dir"]), persistence=given["persistence"], key=support.KEY.hex(),
            key_id=support.KEY_ID, audit=str(audit), wall=wall, laws=given["laws"], life=given["life"], **common),
            cwd=tmp_path.joinpath("cwd-show")),
        "off": garden.child(tmp_path, _code(CHILD_DISABLED, config=str(config), **common),
                            cwd=tmp_path.joinpath("cwd-off")),
    }
    done = {name: garden.finish(process) for name, process in children.items()}

    # (a) The CLI's import: its seven modules, a total under the ceiling, nothing written where it ran.
    code, _out, err, report = done["import"]
    assert code == 0 and report is not None, err[-800:]
    assert set(report["project"]) == CLI_MODULES, report["project"]
    assert 20 <= report["final"] <= CLI_CEILING, report["final"]
    assert list(tmp_path.joinpath("cwd-import").iterdir()) == []

    # (b) The seamed show: the being is drawn, the garden's own closure loaded, no data path reached.
    code, out, err, report = done["show"]
    assert code == 0 and report is not None, err[-800:]
    assert report["code"] == 0 and "Onion, seed, day 2." in out.splitlines() and POT_RIM in out.splitlines(), out
    assert (report["planted"], report["counter"]) == (1, 1), "the counter's witness, and nothing else redirected"
    project = set(report["project"])
    assert project - {NATIVE_CORE} == GARDEN_SHOW_MODULES, sorted(project ^ GARDEN_SHOW_MODULES)
    assert NATIVE_CORE not in project or report["native"] is True
    assert 0 < report["final"] - report["setup"] <= GARDEN_SHOW_DELTA, (report["final"], report["setup"])
    assert list(tmp_path.joinpath("cwd-show").iterdir()) == []

    # (c) Switched off: said, with the file; the settings, the service and the description, nothing more.
    code, out, err, report = done["off"]
    assert code == 0 and report is not None, err[-800:]
    assert report["code"] == 0 and out == garden.printed(
        p, say("status.disabled"), say("path.line", path=str(config)), say("status.disabled.kept"),
        say("doctrine")), out
    assert (report["planted"], report["counter"]) == (1, 1)
    assert set(report["project"]) == DISABLED_MODULES, sorted(set(report["project"]) ^ DISABLED_MODULES)
    for name in DISABLED_ABSENT:
        assert not [module for module in report["project"] if module == name or module.startswith(name + ".")], name
    assert list(tmp_path.joinpath("cwd-off").iterdir()) == []


# ---------------------------------------------------------------------------
# AF10 -- show, in text and ASCII, from what it may read
# ---------------------------------------------------------------------------
GALLERY_HEADER = "Sample forms from fixed values; no onion was read."
# The fields of a state ``show`` never reads: counts, the soil's water, the chemistry, the clock, the genome,
# and the reducer's bookkeeping. Each is moved; no byte moves with it.
UNREAD = (("organs", "stage", "dry"), ("organs", "stage", "rest"), ("organs", "stage", "since"),
          ("organs", "soil", "m"), ("bus", "circadian"), ("genome",), ("n",), ("through",), ("pending",),
          ("pinned",))


def _printable(text):
    return all(0x20 <= ord(char) <= 0x7E for char in text)


def _moved(value):
    """Another value of the same kind."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 4099
    if isinstance(value, str):
        return "0" * len(value) if value.strip("0") else "1" * len(value)
    if isinstance(value, list):
        return value + [0] if not value else []
    if isinstance(value, dict):
        return {} if value else {"moved": 1}
    return [1, "0" * 16, 0]


def _set(state, path, value):
    target = state
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _get(state, path):
    target = state
    for key in path:
        target = target[key]
    return target


def _find(rows, wanted, start):
    """The index at or after ``start`` where ``wanted`` begins in ``rows``, whole; -1 when it does not."""
    for i in range(start, len(rows) - len(wanted) + 1):
        if rows[i:i + len(wanted)] == wanted:
            return i
    return -1


def test_af10_show_draws_the_seed_from_what_it_may_read_in_text_and_ascii(p, tmp_path):
    say = p.wording.say
    d = p.describe

    # The renderings, byte for byte, in both tiers.
    for name, fields in FELTS.items():
        felt = garden.felt(p, **fields)
        ascii_lines, text_lines = d.show_form(felt, "ascii"), d.show_form(felt, "text")
        for tier, lines in (("ascii", ascii_lines), ("text", text_lines)):
            rows = d.wrap(lines)
            assert tuple(rows) == GOLDENS[(name, tier)], (name, tier, rows)
            for row in rows:
                assert _printable(row) and len(row) <= 78, (name, tier, row)
        art = [line.text for line in ascii_lines if line.key == "art"]
        assert art == list(d.art(felt)) and len(art) == 9, (name, art)
        for row in art:
            assert len(row) <= 32 and "@" not in row and "o" not in row, (name, row)
        assert [line for line in ascii_lines if line.key != "art"] == list(text_lines), name
        assert text_lines[-1] == say("doctrine"), name

    # Real fixture states: awake, then breathing; dormant by winter; dormant by drought.
    constants = p.lawfiles.law("fixture")["constants"]
    beings = {}
    stores = []
    try:
        for index, weather in enumerate(("garden", "windowsill")):
            given = support.seams(p, tmp_path.joinpath(weather), suite="af10", index=index)
            target = support.store(p, given)
            stores.append((target, given))
            beings[weather] = target.sow(transport=support.cli(p), law="fixture", tz_minutes=0,
                                         rhythm_consent=False, weather=weather)

        def felt_at(weather, t):
            view = beings[weather].view(to=t)
            return view, d.felt(view, name=None, weather=weather, soil="encrypted", layer="open",
                                labels=("prototype",), constants=constants)

        _view, zero = felt_at("garden", 0)
        breathing_view, breathing = felt_at("garden", 7)
        _view, winter = felt_at("garden", 26 * DAY + 600)
        _view, drought = felt_at("windowsill", 20 * DAY + 600)
    finally:
        for target, _given in stores:
            target.close()
    assert (zero.life, breathing.life) == ("awake", "breathing"), (zero, breathing)
    assert (zero.day, winter.day, drought.day) == (0, 26, 20)
    assert (winter.life, winter.season) == ("dormant_winter", 0), winter
    assert (drought.life, drought.soil, drought.place) == ("dormant_dry", "dry", "windowsill"), drought
    lines = [line.text for line in d.show_form(winter, "text")]
    assert lines[-2].endswith("Went dormant when winter came."), lines
    assert any("(--)" in row for row in d.art(winter)) and any("(..)" in row for row in d.art(breathing))

    # Nothing show may not read moves a byte; the moisture does.
    baseline = {tier: d.wrap(d.show_form(breathing, tier)) for tier in ("text", "ascii")}

    def rendered(state):
        view = breathing_view._replace(state=state)
        felt = d.felt(view, name=None, weather="garden", soil="encrypted", layer="open", labels=("prototype",),
                      constants=constants)
        return {tier: d.wrap(d.show_form(felt, tier)) for tier in ("text", "ascii")}

    paths = list(UNREAD) + [("organs", "chem", key) for key in breathing_view.state["organs"]["chem"]]
    paths += [("organs", "clock", key) for key in breathing_view.state["organs"]["clock"]]
    assert len(paths) >= 14
    for path in paths:
        state = copy.deepcopy(breathing_view.state)
        before = _get(state, path)
        _set(state, path, _moved(before))
        assert _get(state, path) != before, path
        assert rendered(state) == baseline, path
    state = copy.deepcopy(breathing_view.state)
    assert state["bus"]["moisture"] >= constants["stage"]["theta_dry"]
    state["bus"]["moisture"] = constants["stage"]["theta_dry"] - 1
    assert rendered(state) != baseline, "witness: the moisture across the dry threshold moves the bytes"

    # No colour but the error prefix.
    given = support.seams(p, tmp_path.joinpath("colour"), suite="af10", index=2)
    target = support.store(p, given)
    try:
        support.sow(p, target)
    finally:
        target.close()
    factory = garden.garden(p, given, attended=True)
    shown = garden.invoke(p, ["show"], factory=factory, color=True)
    assert shown.exit_code == 0 and "Onion, seed, day 0." in shown.stdout, (shown.stdout, shown.exception)
    assert chr(27) not in shown.stdout and chr(27) not in shown.stderr
    refused = garden.invoke(p, ["keep", "resume", "0", "0"], factory=factory, color=True)
    assert refused.exit_code == 1 and chr(27) in refused.stderr, "witness: the colour was on"
    assert chr(27) not in refused.stdout

    # The gallery prints the same renderings, from the same values, after its header.
    script = life_support.load_script("allium_garden_gallery.py", "allium_garden_gallery")
    for tier in ("ascii", "text"):
        rows = list(script.gallery(tier))
        assert rows[0] == GALLERY_HEADER, rows[:2]
        at = 1
        for name in ("F1", "F2", "F3", "F4", "F5", "F2u", "frozen"):
            wanted = list(GOLDENS[(name, tier)])
            found = _find(rows, wanted, at)
            assert found >= at, (tier, name)
            at = found + len(wanted)
        for row in rows:
            assert _printable(row), (tier, row)

    # Its captions are the garden's words too: each passes the nets, and the nets read them (witness).
    captions = [caption for caption, _fields in script.SAMPLES] + [caption for caption, _look in script.STATUSES]
    assert len(captions) == len(script.SAMPLES) + len(script.STATUSES) >= 20
    for caption in captions:
        assert p.ethics.check(caption) == (), (caption, p.ethics.check(caption))
    assert p.ethics.check("A windowsill never watered, Bulbe's rules: gone dormant in a dry spell."), \
        "witness: a caption that blames the person is refused"


# ---------------------------------------------------------------------------
# AF11 -- deep verification, and the frozen view
# ---------------------------------------------------------------------------
LOW = "0" * 16


def _rows(path):
    """``(t, laws, through, state_hash, blob)`` of every kept state, in minute order."""
    return support.read(path, "SELECT t, laws, through, state_hash, blob FROM checkpoints ORDER BY t")


def _reanchor(p, path):
    [(seq,)] = support.read(path, "SELECT MAX(seq) FROM links")
    support.rewrite_anchor(p, path, seq=seq, key_id=support.KEY_ID, anchor_key=support.KEY)


def _served_hash(p, given):
    target = support.store(p, given)
    try:
        return target.open("local").view().hash
    finally:
        target.close()


def _kept(stdout):
    found = re.search(r"(\d+) of (\d+) kept states agree", garden.flat(stdout))
    return (int(found.group(1)), int(found.group(2))) if found else None


def test_af11_deep_verification_replays_every_kept_state_and_a_fault_freezes_the_look(p, tmp_path):
    given = support.seams(p, tmp_path.joinpath("life"), suite="af11")
    clock = given["clock"]
    target = support.store(p, given)
    try:
        being = support.sow(p, target)
        gestures = []
        for step in range(10):
            clock.advance_days(3)
            gestures.append(being.append("act", {"act": ("water", "greet", "play")[step % 3]},
                                         transport=support.cli(p)))
            assert being.settle().done
        facts = len(being.in_order())
    finally:
        target.close()
    path = support.store_path(p, given)
    grown = path.read_bytes()
    rows = _rows(path)
    goal = (clock.wall - support.WALL) // 60
    assert rows and all(row[0] <= goal for row in rows) and goal // DAY == 30
    factory = garden.garden(p, given)

    # A month of gestures: every kept state agrees, and the state served now.
    verified = garden.invoke(p, ["keep", "verify"], factory=factory)
    assert verified.exit_code == 0, (verified.stdout, verified.stderr, verified.exception)
    assert garden.shows(verified.stdout, p, "verify.chain", facts=facts, days=30), verified.stdout
    assert _kept(verified.stdout) == (len(rows), len(rows)), verified.stdout
    assert garden.shows(verified.stdout, p, "verify.served", engine="reference")
    assert "stale" not in verified.stdout and "past the minute" not in verified.stdout
    assert path.read_bytes() == grown, "verification writes nothing"

    # A kept state forged whole -- its hash and the anchor rewritten: the view moves, and it is found.
    t, laws, _through, _hash, blob = rows[-1]
    state = p.wire.parse(zlib.decompress(bytes(blob)))
    state["organs"]["soil"]["m"] //= 2
    canonical = p.wire.emit(state)
    support.edit(path, lambda conn: conn.execute("UPDATE checkpoints SET blob = ?, state_hash = ? WHERE t = ?",
                                                 (zlib.compress(canonical), hashlib.sha256(canonical).hexdigest(), t)))
    _reanchor(p, path)
    served = _served_hash(p, given)
    path.write_bytes(grown)
    assert served != _served_hash(p, given), "witness: the forged state moves what is served"
    support.edit(path, lambda conn: conn.execute("UPDATE checkpoints SET blob = ?, state_hash = ? WHERE t = ?",
                                                 (zlib.compress(canonical), hashlib.sha256(canonical).hexdigest(), t)))
    _reanchor(p, path)
    forged = path.read_bytes()
    found = garden.invoke(p, ["keep", "verify"], factory=factory)
    assert found.exit_code == 1 and found.stdout == "", (found.stdout, found.stderr, found.exception)
    assert garden.says(found.stderr, p, "verify.diverged.kept", day=t // DAY, v=laws), found.stderr
    assert garden.says(found.stderr, p, "verify.diverged.after"), found.stderr
    assert path.read_bytes() == forged, "nothing was replaced"
    path.write_bytes(grown)

    # An older kept state forged past the engine's input limit, its hash and the anchor rewritten: the view does not
    # start from it, and deep verification names it as a state that does not hold what it names.
    first_t, first_laws = rows[0][0], rows[0][1]
    assert first_t < rows[-1][0], "presence: an older kept state than the one a view starts from"
    oversized = b"{" + b" " * (p.life.limits()[0] + 64) + b"}"
    support.edit(path, lambda conn: conn.execute("UPDATE checkpoints SET blob = ?, state_hash = ? WHERE t = ?",
                                                 (zlib.compress(oversized), hashlib.sha256(oversized).hexdigest(),
                                                  first_t)))
    _reanchor(p, path)
    bloated = path.read_bytes()
    shown = garden.invoke(p, ["show"], factory=factory)
    assert shown.exit_code == 0, (shown.stdout, shown.stderr)
    blob = garden.invoke(p, ["keep", "verify"], factory=factory)
    assert blob.exit_code == 1 and blob.stdout == "", (blob.stdout, blob.stderr, blob.exception)
    assert garden.says(blob.stderr, p, "verify.diverged.blob", day=first_t // DAY, v=first_laws), blob.stderr
    assert path.read_bytes() == bloated, "nothing was replaced"
    path.write_bytes(grown)

    # A stale kept state: a fact landed before the event it runs through. Counted, never a divergence.
    target = support.store(p, given)
    try:
        support.land(target.open("local"), LOW, 0, gestures[5].t, "act", {"act": "touch"})
    finally:
        target.close()
    stale = garden.invoke(p, ["keep", "verify"], factory=factory)
    assert stale.exit_code == 0, (stale.stdout, stale.stderr, stale.exception)
    counted = re.search(r"(\d+) kept states were stale", garden.flat(stale.stdout))
    assert counted and int(counted.group(1)) >= 1, stale.stdout
    assert garden.shows(stale.stdout, p, "verify.stale", stale=int(counted.group(1)))
    assert _kept(stale.stdout) == (len(rows) - int(counted.group(1)), len(rows)), stale.stdout
    assert not garden.says(stale.stderr, p, "verify.diverged.after")
    path.write_bytes(grown)

    # Kept states past the minute shown now: the clock set back, with no fact since.
    target = support.store(p, given)
    try:
        clock.advance_days(2)
        assert target.open("local").settle().done
    finally:
        target.close()
    later = [row[0] for row in _rows(path) if row[0] > goal]
    assert later, "presence: kept states past the minute the set-back clock shows"
    clock.advance_days(-2)
    ahead = garden.invoke(p, ["keep", "verify"], factory=factory)
    assert ahead.exit_code == 0 and garden.shows(ahead.stdout, p, "verify.ahead", ahead=len(later)), ahead.stdout
    path.write_bytes(grown)

    # A served state that disagrees with the replay: the engine's answer for the minute shown is altered.
    real = p.engine.call

    def altered(request):
        answer = real(request)
        asked = p.wire.parse(request)
        if asked.get("op") == "advance" and asked.get("to") == goal:
            parsed = p.wire.parse(answer)
            if "hash" in parsed:
                parsed["hash"] = ("0" if parsed["hash"][0] != "0" else "1") + parsed["hash"][1:]
                return p.wire.emit(parsed)
        return answer

    p.engine.call = altered
    try:
        served = garden.invoke(p, ["keep", "verify"], factory=factory)
    finally:
        p.engine.call = real
    assert served.exit_code == 1, (served.stdout, served.stderr, served.exception)
    assert garden.says(served.stderr, p, "verify.diverged.served", day=goal // DAY, v=0, engine="reference"), \
        served.stderr
    assert path.read_bytes() == grown

    # An engine that stops on a fault past the latest kept state: the look is frozen on that state.
    last = rows[-1][0]

    def panicking(request):
        asked = p.wire.parse(request)
        if asked.get("op") == "advance" and asked.get("to", 0) > last:
            return p.wire.emit({"detail": "panic", "refused": "engine_panic"})
        return real(request)

    target = support.store(p, given)
    try:
        stored = target.open("local").view(to=last)
    finally:
        target.close()
    civil, minute = stored.env["civil"], stored.env["minute"]
    when = f"{civil[0]:04d}-{civil[1]:02d}-{civil[2]:02d} {minute // 60:02d}:{minute % 60:02d} (+00:00)"
    p.engine.call = panicking
    try:
        frozen = garden.invoke(p, ["show"], factory=factory)
    finally:
        p.engine.call = real
    assert frozen.exit_code == 1, (frozen.stdout, frozen.stderr, frozen.exception)
    assert garden.shows(frozen.stdout, p, "label.frozen", when=when), frozen.stdout
    assert f"Onion, seed, day {last // DAY}." in frozen.stdout.splitlines(), frozen.stdout
    assert path.read_bytes() == grown, "nothing records the fault"

    # A being with no kept state says so.
    fresh = support.seams(p, tmp_path.joinpath("fresh"), suite="af11", index=1)
    target = support.store(p, fresh)
    try:
        support.sow(p, target)
    finally:
        target.close()
    fresh["clock"].advance_days(1)
    none = garden.invoke(p, ["keep", "verify"], factory=garden.garden(p, fresh))
    assert none.exit_code == 0 and garden.shows(none.stdout, p, "verify.none"), none.stdout
    assert garden.shows(none.stdout, p, "verify.served", engine="reference"), none.stdout


# ---------------------------------------------------------------------------
# AF12 -- the lab's laws, and keep laws
# ---------------------------------------------------------------------------
def _facts(p, given):
    target = support.store(p, given)
    try:
        return [(seq, fact) for seq, _eid, fact in target.open("local").in_order()]
    finally:
        target.close()


def _offset(minutes):
    sign = "+" if minutes >= 0 else "-"
    return f"{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"


def test_af12_the_lab_shows_the_laws_and_keep_laws_writes_only_what_the_diff_confirmed(p, tmp_path):
    say = p.wording.say
    given = support.seams(p, tmp_path.joinpath("fixture"), suite="af12")
    clock = given["clock"]
    target = support.store(p, given)
    try:
        support.sow(p, target)
    finally:
        target.close()
    facts = _facts(p, given)
    genesis = facts[0][1]["body"]["laws"]
    params = dict(genesis["params"])
    factory = garden.garden(p, given, attended=True)

    # The laws screen of a fresh fixture being, whole.
    screen = garden.invoke(p, ["lab", "laws"], factory=factory)
    assert screen.exit_code == 0, (screen.stdout, screen.stderr, screen.exception)
    assert screen.stdout == garden.printed(
        p, say("doctrine"), say("lab.record", events=len(facts)), say("label.prototype"), say("lab.title"),
        say("lab.born.provisional", law="fixture", v=0, digest=genesis["sha256"][:12]),
        say("lab.in_force", law="fixture", v=0, params=params), say("lab.unpinned"), say("lab.history.none"),
        say("lab.proposal.same")), screen.stdout

    # The proposal moves: the screen says so; the diff shows its row and its code.
    given["laws"]["soil"]["rain_gain"] = 49152
    screen = garden.invoke(p, ["lab", "laws"], factory=factory)
    assert garden.shows(screen.stdout, p, "lab.proposal.differs", names=("rain_gain",)), screen.stdout
    diff = garden.invoke(p, ["keep", "laws", "diff"], factory=factory)
    assert diff.exit_code == 0, (diff.stdout, diff.stderr, diff.exception)
    code = re.search(r"oo garden keep laws apply ([0-9a-f]{16})\.", garden.flat(diff.stdout))
    assert code, diff.stdout
    code = code.group(1)
    for key, slots in (("laws.diff.head", {"law": "fixture", "v": 0}),
                       ("laws.diff.row", {"param": "rain_gain", "before": params["rain_gain"], "after": 49152}),
                       ("laws.diff.confirm", {"confirm": code})):
        assert garden.shows(diff.stdout, p, key, **slots), (key, diff.stdout)
    target = support.store(p, given)
    try:
        assert target.open("local").laws_diff().confirm == code, "the code the store confirms"
    finally:
        target.close()

    # A wrong code writes nothing; the right one writes one law update, pending until its midnight.
    wrong = garden.invoke(p, ["keep", "laws", "apply", "0" * 16], factory=factory)
    assert wrong.exit_code == 1 and garden.says(wrong.stderr, p, "refuse.laws.confirm"), wrong.stderr
    assert [fact for _seq, fact in _facts(p, given) if fact["kind"] == "evolve"] == []
    applied = garden.invoke(p, ["keep", "laws", "apply", code.upper()], factory=factory)
    assert applied.exit_code == 0, (applied.stdout, applied.stderr, applied.exception)
    evolves = [(seq, fact) for seq, fact in _facts(p, given) if fact["kind"] == "evolve"]
    assert len(evolves) == 1, evolves
    seq, evolve = evolves[0]
    effective = evolve["body"]["effective_from"]
    assert evolve["body"]["params"]["rain_gain"] == 49152
    assert garden.shows(applied.stdout, p, "laws.applied", day=effective // DAY), applied.stdout
    screen = garden.invoke(p, ["lab", "laws"], factory=factory)
    assert garden.shows(screen.stdout, p, "lab.pending", law="fixture", v=0, day=effective // DAY), screen.stdout
    assert garden.shows(screen.stdout, p, "lab.history.update", day=evolve["t"] // DAY, seq=seq, law="fixture", v=0,
                        from_day=effective // DAY), screen.stdout
    assert f"(#{seq})" in screen.stdout

    # Past its midnight, in force, and pending no longer.
    clock.wall = support.WALL + (effective + 1) * 60
    screen = garden.invoke(p, ["lab", "laws"], factory=factory)
    assert garden.shows(screen.stdout, p, "lab.in_force", law="fixture", v=0,
                        params=dict(params, rain_gain=49152)), screen.stdout
    assert not garden.says(screen.stdout, p, "lab.pending", law="fixture", v=0, day=effective // DAY)

    # Pin, a refused second pin, unpin; an unattended terminal is refused.
    pinned = garden.invoke(p, ["keep", "laws", "pin"], factory=factory)
    assert pinned.exit_code == 0 and garden.shows(pinned.stdout, p, "laws.pinned"), (pinned.stdout, pinned.stderr)
    again = garden.invoke(p, ["keep", "laws", "pin"], factory=factory)
    assert again.exit_code == 1 and garden.says(again.stderr, p, "refuse.laws.pinned"), again.stderr
    assert garden.says(again.stderr, p, "label.prototype.short") or garden.says(again.stderr, p, "label.prototype"), \
        "the being's label follows a refusal about it"
    unpinned = garden.invoke(p, ["keep", "laws", "unpin"], factory=factory)
    assert unpinned.exit_code == 0 and garden.shows(unpinned.stdout, p, "laws.unpinned"), unpinned.stderr
    factory.attended = False
    before = _facts(p, given)
    for args in (["keep", "laws", "apply", code], ["keep", "laws", "pin"]):
        alone = garden.invoke(p, args, factory=factory)
        assert alone.exit_code == 1 and garden.says(alone.stderr, p, "refuse.attended"), (args, alone.stderr)
    alone = garden.invoke(p, ["keep", "laws", "unpin"], factory=factory)
    assert (alone.exit_code, alone.stdout) == (1, "") and garden.says(alone.stderr, p, "refuse.attended"), \
        (alone.stdout, alone.stderr)
    assert _facts(p, given) == before, "nothing written from an unattended terminal"
    factory.attended = True

    # A new local offset: the lab says the next gesture records it.
    given["tz"] = lambda now: 120
    screen = garden.invoke(p, ["lab", "laws"], factory=factory)
    assert garden.shows(screen.stdout, p, "lab.offset", recorded=_offset(0), now=_offset(120)), screen.stdout

    # Under a stable law, a successor is available, and the next gesture carries it.
    engine = types.SimpleNamespace(lawfiles=p.lawfiles, wire=p.wire)
    files = life_support.stable_pair(engine)
    with life_support.injected(engine, files):
        stable = support.seams(p, tmp_path.joinpath("stable"), suite="af12", index=1)
        target = support.store(p, stable)
        try:
            support.sow(p, target, law="fixture_s")
        finally:
            target.close()
        genesis = _facts(p, stable)[0][1]["body"]["laws"]
        keeper = garden.garden(p, stable, attended=True, law="fixture_s")
        screen = garden.invoke(p, ["lab", "laws"], factory=keeper)
        assert screen.exit_code == 0, (screen.stdout, screen.stderr, screen.exception)
        assert garden.shows(screen.stdout, p, "lab.born.stable", law="fixture_s", v=1,
                            digest=genesis["sha256"][:12]), screen.stdout
        assert garden.shows(screen.stdout, p, "lab.available", law="fixture_s2", v=2), screen.stdout
        assert not garden.says(screen.stdout, p, "label.prototype"), "a stable law carries no label"
        cared = garden.invoke(p, ["care", "greet"], factory=keeper)
        assert cared.exit_code == 0, (cared.stdout, cared.stderr, cared.exception)
        evolves = [fact for _seq, fact in _facts(p, stable) if fact["kind"] == "evolve"]
        assert len(evolves) == 1 and evolves[0]["body"]["to"]["name"] == "fixture_s2", evolves
        effective = evolves[0]["body"]["effective_from"]
        assert garden.shows(cared.stdout, p, "care.noted", act="greet"), cared.stdout
        assert garden.shows(cared.stdout, p, "laws.carried", law="fixture_s2", v=2, day=effective // DAY), cared.stdout
        screen = garden.invoke(p, ["lab", "laws"], factory=keeper)
        assert garden.shows(screen.stdout, p, "lab.pending", law="fixture_s2", v=2, day=effective // DAY), \
            screen.stdout

    # Every look leaves the store's files and their bytes as they were, and asks no terminal test; a gesture moves
    # them (witness). Two days pass first, so a look that settled the being would have a state to keep.
    clock.advance_days(2)
    data = Path(given["data_dir"])

    def listing():
        return {str(file.relative_to(data)): garden.sha256(file) for file in sorted(data.rglob("*")) if file.is_file()}

    kept = listing()
    assert kept, "presence: the being's store"
    asked = factory.attended_reads
    looks = (["show"], ["show", "--json"], ["show", "--tier", "text"], [], ["lab"], ["lab", "laws"],
             ["keep", "laws", "diff"], ["keep", "verify"])
    for args in looks:
        result = garden.invoke(p, args, factory=factory)
        assert result.exit_code == 0, (args, result.stdout, result.stderr, result.exception)
        assert listing() == kept, ("a look writes nothing", args)
    assert factory.attended_reads == asked, "no look asks the terminal test"
    cared = garden.invoke(p, ["care", "water"], factory=factory)
    assert cared.exit_code == 0 and listing() != kept, "witness: a gesture moves what the listing reads"
    assert factory.attended_reads == asked, "nor does a gesture"


# ---------------------------------------------------------------------------
# AF13 -- the production wiring
# ---------------------------------------------------------------------------
class _Os:
    """The three conditions of the strong terminal test, each answerable, each able to fail."""

    O_RDONLY = 0

    def __init__(self, tty=True, foreground=True, device=True, failing=None):
        self.tty = tty
        self.foreground = foreground
        self.device = device
        self.failing = failing
        self.opened = []

    def _step(self, name):
        if self.failing == name:
            raise OSError(name)

    def isatty(self, fd):
        self._step("isatty")
        return self.tty and fd == 0

    def tcgetpgrp(self, fd):
        self._step("tcgetpgrp")
        return 4242 if self.foreground else 4343

    def getpgrp(self):
        self._step("getpgrp")
        return 4242

    def open(self, path, flags):
        self._step("open")
        self.opened.append((path, flags))
        if not self.device:
            raise OSError("no controlling terminal")
        return 9

    def close(self, fd):
        self._step("close")


def _db_utils(connects, failing):
    stub = types.ModuleType("opti_oignon.db_utils")

    def safe_connect(db_path, **kwargs):
        connects.append(str(db_path))
        if failing[0]:
            raise RuntimeError("the keyed connection refuses")
        return sqlite3.connect(str(db_path))

    stub.safe_connect = safe_connect
    return stub


class _Imports:
    """Every import statement run while installed: ``(module, names)``, whatever the import finds or refuses."""

    def __init__(self):
        self.asked = []

    def __enter__(self):
        import builtins

        self.real = builtins.__import__
        asked, real = self.asked, self.real

        def recording(name, globals=None, locals=None, fromlist=(), level=0):
            asked.append((name, tuple(fromlist or ())))
            return real(name, globals, locals, fromlist, level)

        builtins.__import__ = recording
        return self

    def __exit__(self, *exc_info):
        import builtins

        if getattr(self, "real", None) is not None:
            builtins.__import__, self.real = self.real, None
        return False

    def named(self, word):
        return [entry for entry in self.asked if word in entry[0].split(".") or word in entry[1]]


def _store_calls(tree):
    """``[(call, positional count, keywords)]`` of every call to ``Store`` as a name or an attribute."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id == "Store") or (
                    isinstance(func, ast.Attribute) and func.attr == "Store"):
                found.append((node, len(node.args), sorted(keyword.arg for keyword in node.keywords)))
    return found


def test_af13_the_production_wiring_passes_the_single_user_reader_and_nothing_else(monkeypatch, tmp_path):
    # The auth stores a stopped process leaves, read by the production reader in a child, behind the firewall.
    stores = tmp_path.joinpath("auth-stores")
    writer = garden.child(tmp_path, _code(CHILD_AUTH_WRITER, root=str(stores)), cwd=tmp_path.joinpath("cwd-writer"))
    code, _out, err, _report = garden.finish(writer)
    assert code == 0, err[-600:]
    auth_config = tmp_path.joinpath("auth-single.yaml")
    auth_config.write_text("single_user_mode: true\ndb_path: data/auth.db\n", encoding="ascii")
    reader = garden.child(tmp_path, _code(CHILD_AUTH_READER, tests=str(TESTS), repo=str(garden.REPO), root=str(stores),
                                          config=str(auth_config)), cwd=tmp_path.joinpath("cwd-reader"))
    connects, failing = [], [False]
    blocked = tuple(name for name in support.BLOCKED if name != "opti_oignon.db_utils")
    assert "opti_oignon.auth" in blocked
    p, restore = garden.open_garden(monkeypatch, tmp_path, seeded={"opti_oignon.db_utils": _db_utils(connects, failing)},
                                    blocked=blocked)
    imports = _Imports()
    try:
        s = p.service

        # (a) The strong terminal test: all three conditions, or False.
        assert s.terminal_attended(_os=_Os()) is True
        stand_in = _Os()
        s.terminal_attended(_os=stand_in)
        assert stand_in.opened == [("/dev/tty", _Os.O_RDONLY)]
        for case in ({"tty": False}, {"foreground": False}, {"device": False}):
            assert s.terminal_attended(_os=_Os(**case)) is False, case
        for step in ("isatty", "tcgetpgrp", "getpgrp", "open", "close"):
            assert s.terminal_attended(_os=_Os(failing=step)) is False, step

        # (b) The single-user reader: the auth settings and store, read only, the latch one way.
        imports.__enter__()
        root = tmp_path.joinpath("root")
        root.mkdir()
        config = tmp_path.joinpath("auth.yaml")
        config.write_text("single_user_mode: false\ndb_path: data/auth.db\n", encoding="ascii")
        assert s.platform_single_user(config=config, root=root) is False and connects == []
        config.write_text("single_user_mode: true\ndb_path: data/auth.db\n", encoding="ascii")
        assert s.platform_single_user(config=config, root=root) is True and connects == []
        assert not root.joinpath("data").exists(), "no directory is created"
        assert s.platform_single_user(config=tmp_path.joinpath("absent.yaml"), root=root) is True and connects == []
        store = root.joinpath("data", "auth.db")
        store.parent.mkdir()
        conn = sqlite3.connect(str(store))
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)")
        conn.commit()
        conn.close()
        answers = []
        for users in (0, 1):
            support.edit(store, lambda db: db.execute("DELETE FROM users"))
            for i in range(users):
                support.edit(store, lambda db, i=i: db.execute("INSERT INTO users (username) VALUES (?)", (f"u{i}",)))
            answers.append(s.platform_single_user(config=config, root=root))
        assert answers == [True, True] and [Path(c).resolve() for c in connects] == [store.resolve()] * 2, connects
        failing[0] = True
        assert s.platform_single_user(config=config, root=root) is False, "a connect that raises is not single-user"
        failing[0] = False
        assert s.platform_single_user(config=config, root=root) is True, "and it latches nothing"
        support.edit(store, lambda db: db.execute("INSERT INTO users (username) VALUES ('second')"))
        assert s.platform_single_user(config=config, root=root) is False
        support.edit(store, lambda db: db.execute("DELETE FROM users WHERE username = 'second'"))
        assert support.read(store, "SELECT COUNT(*) FROM users") == [(1,)]
        assert s.platform_single_user(config=config, root=root) is False, "once off, off for the process"
        imports.__exit__()
        assert sys.modules.get("opti_oignon.auth") is None, "the auth module is never imported"
        assert imports.named("db_utils"), "witness: the recorder saw the reader's import of the keyed connector"
        assert imports.named("auth") == [], ("the reader asks for no auth module", imports.named("auth"))
        with _Imports() as planted:
            try:
                from opti_oignon import auth  # noqa: F401
            except ImportError:
                pass
        assert planted.named("auth"), "witness: an import of the auth module is recorded, even when refused"

        # (c) Garden.production(): the reader, and nothing else, to the one store it builds.
        made = []

        class Recording:
            def __init__(self, **kwargs):
                made.append(kwargs)
                self.closed = 0

            def close(self):
                self.closed += 1

        real = p.store.Store
        p.store.Store = Recording
        try:
            production = s.Garden.production()
            assert production.switch is p.settings.switch and production.attended is s.terminal_attended
            assert production.stopped is None and production.law is None
            first, second = production.store(), production.store()
            assert first is second and len(made) == 1, made
            assert list(made[0]) == ["single_user"] and made[0]["single_user"] is s.platform_single_user, made
            production.close()
            assert first.closed == 1
        finally:
            p.store.Store = real
    finally:
        imports.__exit__()
        restore()

    # A child whose stdin is not a terminal: the real test says no.
    process = garden.child(tmp_path, "from opti_oignon.allium import service\nprint(service.terminal_attended())\n",
                           cwd=tmp_path.joinpath("cwd-attended"))
    code, out, err, _report = garden.finish(process)
    assert (code, out) == (0, "False\n"), (code, out, err[-600:])

    # (d) The census: one production Store call, in the service, with the single-user reader alone.
    calls = []
    for source in sorted(PACKAGE.rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        if "Store" not in text:
            continue
        for _node, positional, keywords in _store_calls(ast.parse(text)):
            calls.append((source.relative_to(PACKAGE).as_posix(), positional, keywords))
    assert calls == [("allium/service.py", 0, ["single_user"])], calls
    planted = _store_calls(ast.parse("from opti_oignon.allium import store\nstore.Store(single_user=f, mode=g)\n"))
    assert [(positional, keywords) for _node, positional, keywords in planted] == [(0, ["mode", "single_user"])], \
        "witness: a call with another seam is counted and differs"

    # (e) The production reader writes nothing in the auth store, whatever state a stopped process left it in.
    code, _out, err, found = garden.finish(reader)
    assert code == 0 and found is not None, err[-1200:]
    assert found["clean"] == [True, True, ["auth.db"]], found["clean"]
    assert found["pending"][:2] == [True, True], ("frames never checkpointed are read, and stay", found["pending"])
    assert found["pending"][2] == ["auth.db", "auth.db-shm", "auth.db-wal"], "presence: the frames were pending"
    assert found["hot"][:2] == [False, True] and "auth.db-journal" in found["hot"][2], \
        ("a hot journal is never rolled back by a look: not single-user, and left as it is", found["hot"])
    assert found["pending2"][:2] == [False, True], ("two accounts in pending frames", found["pending2"])
    assert found["control"][0] == 2 and found["control"][1] is False, \
        ("witness: an ordinary connection checkpoints the frames, and the listing sees it", found["control"])


# ---------------------------------------------------------------------------
# AF14 -- the production look, behind the empty mirror
# ---------------------------------------------------------------------------
def test_af14_the_production_look_touches_only_the_frozen_modules_and_data_paths(p, tmp_path):
    say = p.wording.say
    config = tmp_path.joinpath("on", "allium.yaml")
    config.parent.mkdir()
    cwd = tmp_path.joinpath("cwd-production")
    process = garden.child(tmp_path, _code(CHILD_PRODUCTION, tests=str(TESTS), repo=str(garden.REPO),
                                           canary=garden.canary(p, "af14"), config=str(config)), cwd=cwd)
    quiet = tmp_path.joinpath("logs", "allium.yaml")
    quiet.parent.mkdir()
    logging_child = garden.child(tmp_path, _code(CHILD_LOGS, tests=str(TESTS), repo=str(garden.REPO),
                                                 canary=garden.canary(p, "af14"), config=str(quiet)),
                                 cwd=tmp_path.joinpath("cwd-logs"))
    code, out, err, report = garden.finish(process)
    assert code == 0 and report is not None, err[-1200:]
    assert report["codes"] == [2, 2, 1], report["codes"]
    assert garden.shows(out, p, "status.awaiting_soil"), out
    assert garden.says(err, p, "status.awaiting_soil"), err[-1200:]
    project = set(report["project"])
    assert "opti_oignon.auth" not in project
    assert project == PRODUCTION_LOOK_MODULES, sorted(project ^ PRODUCTION_LOOK_MODULES)
    assert report["planted"] == 1 and "data/planted-" + garden.canary(p, "af14") in report["paths"]
    paths = tuple(path for path in report["paths"] if not path.startswith("data/planted-"))
    assert paths == PRODUCTION_LOOK_PATHS, paths
    assert tuple(report["listing"]) == PRODUCTION_LOOK_MIRROR, report["listing"]
    assert list(cwd.iterdir()) == [], "nothing written where it ran"
    assert garden.shows(out, p, "doctrine"), out

    # The children import the tree under test, never a copy an install points at.
    probe = garden.child(tmp_path, 'import json, sys, opti_oignon\n'
                                   'sys.stderr.write(json.dumps({"package": opti_oignon.__file__}) + "\\n")\n',
                         cwd=tmp_path.joinpath("cwd-package"))
    code, _out, err, where = garden.finish(probe)
    assert code == 0 and where is not None, err[-1200:]
    assert Path(where["package"]).resolve().is_relative_to(Path(garden.REPO).resolve()), where

    # No log record reaches the terminal while a garden command runs; outside one, the same channel is open.
    code, _out, err, logs = garden.finish(logging_child)
    assert code == 0 and logs is not None, err[-1200:]
    assert logs["handlers"] == 0, "the child configures no logging, as oo does not"
    for moment in ("before", "after"):
        assert "a record no handler takes" in logs[moment][2], ("witness: the last resort prints a record", moment)
    assert "PLAINTEXT" in logs["plaintext"][2], "witness: the connector's warning reaches stderr outside a command"
    assert logs["mirror"] is True, "the auth store the commands read is the mirror's"
    assert logs["unreadable"] == [0, garden.printed(p, say("status.disabled.unreadable"), say("path.line", path=str(quiet)),
                                                   say("status.disabled.kept"), say("doctrine")), ""], logs["unreadable"]
    assert logs["show"] == [2, garden.printed(p, say("status.awaiting_soil"), say("doctrine")), ""], logs["show"]
    code, stdout, stderr = logs["care"]
    assert (code, stdout) == (1, "") and stderr.startswith("Error: "), logs["care"]
    assert garden.flat(stderr) == " ".join(say("status.awaiting_soil").text.split()), logs["care"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
