"""Contracts for the egress census guard: every network sink is gated, owed or exempt.

The platform keeps its outbound requests behind gates: the web gate, the
local rule and the peer rule. A gate protects only the calls that ask it, so
the census guard proves, on every commit, that each network sink in the
package sits in a gate home behind a gate in its own function, is exempt by
name with a reason, or is owed in a ledger of counts that may only shrink.

  * EC1 -- the census counts every sink spelling, and nothing else: every
    family it names, through the bindings it follows, and never prose, a
    local function that shares a name, a Unix socket, a program that cannot
    reach the network, or a client built with its telemetry off.
  * EC2 -- an estate with nothing to scan is refused, and a directory named
    ``data`` is never opened: the maintainer's content lives there.
  * EC3 -- each question answers its own finding: one fixture estate per
    finding, each seen by its own question only, and a clean estate seen by
    none.
  * EC4 -- the real tree: the census is green, every home is proven by a
    contract that names it, the ledger never grows past its birth, and every
    third-party library the package imports is classified.
  * EC5 -- the limits of the census are written in its docstring, and its
    green line carries every figure it stands on.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window.
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate  # noqa: E402

_GUARD = REPO / ".github" / "scripts" / "egress_census_guard.py"

BUDGET_S = {
    "test_ec1_the_census_counts_every_sink_spelling_and_nothing_else": 2.0,
    "test_ec2_an_empty_estate_is_refused_and_data_is_never_opened": 2.0,
    "test_ec3_each_question_answers_its_own_finding": 2.0,
    "test_ec4_the_real_tree_is_green_and_its_ledger_never_grows": 2.0,
    "test_ec5_the_limits_are_written_and_the_green_carries_its_denominator": 2.0,
}


def _load():
    loaded, restore = isolate(targets={"egress_census_guard": _GUARD})
    return loaded["egress_census_guard"], restore


# ---------------------------------------------------------------------------
# EC1 fixtures: name -> (text, kind, sites). One module per spelling.
# ---------------------------------------------------------------------------
_POSITIVES = {
    "urlopen": ("import urllib.request\n\ndef f(u):\n    return urllib.request.urlopen(u)\n", "urllib", 1),
    "urlopen renamed": ("from urllib.request import urlopen as fetch\n\ndef f(u):\n    return fetch(u)\n", "urllib", 1),
    "urlretrieve": ("from urllib import request\n\ndef f(u, p):\n    request.urlretrieve(u, p)\n", "urllib", 1),
    "build_opener open": (
        "from urllib.request import build_opener\n\ndef f(u):\n    return build_opener().open(u)\n", "urllib", 1),
    "module-level opener": (
        "import urllib.request\n\n_OPENER = urllib.request.build_opener()\n\n"
        "def f(u):\n    return _OPENER.open(u)\n", "urllib", 1),
    "OpenerDirector open": (
        "import urllib.request\n\ndef f(u):\n    o = urllib.request.OpenerDirector()\n    return o.open(u)\n",
        "urllib", 1),
    "opener as a parameter default": (
        "import urllib.request\n\ndef f(u, opener=urllib.request.build_opener()):\n    return opener.open(u)\n",
        "urllib", 1),
    "HTTPSConnection": ("import http.client\n\ndef f():\n    return http.client.HTTPSConnection('h')\n", "http.client", 1),
    "HTTPSConnection subclass": (
        "from http.client import HTTPSConnection\n\nclass Pinned(HTTPSConnection):\n    pass\n\n"
        "def f():\n    return Pinned('h')\n", "http.client", 1),
    "HTTPConnection request": (
        "import http.client\n\ndef f():\n    conn = http.client.HTTPConnection('h')\n    conn.request('GET', '/')\n",
        "http.client", 2),
    "create_connection": ("import socket\n\ndef f(h):\n    return socket.create_connection((h, 80))\n", "socket", 1),
    "getaddrinfo imported": ("from socket import getaddrinfo\n\ndef f(h):\n    return getaddrinfo(h, 80)\n", "socket", 1),
    "gethostbyname aliased": ("import socket as s\n\ndef f(h):\n    return s.gethostbyname(h)\n", "socket", 1),
    "gethostbyname_ex": ("import socket\n\ndef f(h):\n    return socket.gethostbyname_ex(h)\n", "socket", 1),
    "gethostbyaddr": ("import socket\n\ndef f(a):\n    return socket.gethostbyaddr(a)\n", "socket", 1),
    "getnameinfo": ("import socket\n\ndef f(a):\n    return socket.getnameinfo(a, 0)\n", "socket", 1),
    "getfqdn": ("import socket\n\ndef f():\n    return socket.getfqdn()\n", "socket", 1),
    "star import": ("from socket import *\n\ndef f(h):\n    return create_connection((h, 80))\n", "socket", 1),
    "default family connect": (
        "import socket\n\ndef f(h):\n    s = socket.socket()\n    s.connect((h, 80))\n", "socket", 1),
    "inet6 connect_ex": (
        "import socket\n\ndef f(h):\n    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)\n"
        "    return s.connect_ex((h, 80))\n", "socket", 1),
    "sendto": (
        "import socket\n\ndef f(h, b):\n    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "    s.sendto(b, (h, 53))\n", "socket", 1),
    "fromfd sendmsg": (
        "import socket\n\ndef f(fd, b):\n    s = socket.fromfd(fd, socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "    s.sendmsg([b])\n", "socket", 1),
    "socketpair": ("import socket\n\ndef f(h):\n    a, b = socket.socketpair()\n    a.connect(h)\n", "socket", 1),
    "wrap_socket": ("import ssl\n\ndef f(ctx, raw, h):\n    s = ctx.wrap_socket(raw)\n    s.connect((h, 443))\n",
                    "socket", 1),
    "variable family": ("import socket\n\ndef f(fam, h):\n    s = socket.socket(fam)\n    s.connect(h)\n", "socket", 1),
    "for target": (
        "import socket\n\ndef f(hosts):\n    for s in [socket.socket()]:\n        s.connect(hosts[0])\n", "socket", 1),
    "unix in one function, inet in another": (
        "import socket\n\ndef a(p):\n    s = socket.socket(socket.AF_UNIX)\n    s.connect(p)\n\n"
        "def b(h):\n    s = socket.socket(socket.AF_INET)\n    return h\n", "socket", 1),
    "open_connection": (
        "import asyncio\n\nasync def f(h):\n    return await asyncio.open_connection(h, 80)\n", "asyncio", 1),
    "loop create_connection": (
        "import asyncio\n\nasync def f(h):\n    loop = asyncio.get_running_loop()\n"
        "    return await loop.create_connection(lambda: None, h, 80)\n", "asyncio", 1),
    "loop getaddrinfo": ("import asyncio\n\nasync def f(loop, h):\n    return await loop.getaddrinfo(h, 80)\n",
                         "asyncio", 1),
    "create_datagram_endpoint": (
        "import asyncio\n\nasync def f(loop, h):\n"
        "    return await loop.create_datagram_endpoint(None, remote_addr=(h, 53))\n", "asyncio", 1),
    "sock_connect": ("import asyncio\n\nasync def f(loop, s, h):\n    await loop.sock_connect(s, (h, 80))\n",
                     "asyncio", 1),
    "multiprocessing tuple address": (
        "from multiprocessing.connection import Client\n\ndef f():\n    return Client(('h', 1))\n",
        "multiprocessing", 1),
    "multiprocessing variable address": (
        "import multiprocessing.connection as mc\n\ndef f(addr):\n    return mc.Client(addr)\n", "multiprocessing", 1),
    "SysLogHandler tuple": (
        "import logging.handlers\n\ndef f():\n    return logging.handlers.SysLogHandler(address=('h', 514))\n",
        "logging", 1),
    "SysLogHandler default": (
        "from logging.handlers import SysLogHandler\n\ndef f():\n    return SysLogHandler()\n", "logging", 1),
    "SocketHandler": ("from logging import handlers\n\ndef f():\n    return handlers.SocketHandler('h', 9020)\n",
                      "logging", 1),
    "DatagramHandler": ("import logging.handlers\n\ndef f():\n    return logging.handlers.DatagramHandler('h', 1)\n",
                        "logging", 1),
    "HTTPHandler": ("import logging.handlers\n\ndef f():\n    return logging.handlers.HTTPHandler('h', '/')\n",
                    "logging", 1),
    "SMTPHandler": (
        "import logging.handlers\n\ndef f():\n    return logging.handlers.SMTPHandler('h', 'a', ['b'], 's')\n",
        "logging", 1),
    "requests get": ("import requests\n\ndef f(u):\n    return requests.get(u)\n", "requests", 1),
    "what a request returns is not the client": (
        "import requests\n\ndef f(u):\n    r = requests.get(u)\n    return r.json()\n", "requests", 1),
    "getattr with a constant": ("import requests\n\ndef f(u):\n    return getattr(requests, 'get')(u)\n",
                                "requests", 1),
    "import_module": (
        "from importlib import import_module\n\ndef f(u):\n    return import_module('requests').post(u)\n",
        "requests", 1),
    "__import__": ("def f(u):\n    return __import__('requests').get(u)\n", "requests", 1),
    "sys.modules lookup": ("import sys\n\ndef f(u):\n    return sys.modules['requests'].get(u)\n", "requests", 1),
    "httpx with": ("import httpx\n\ndef f(u):\n    with httpx.Client() as c:\n        return c.get(u)\n", "httpx", 2),
    "httpx walrus": (
        "import httpx\n\ndef f(u):\n    if (c := httpx.Client()):\n        return c.get(u)\n", "httpx", 2),
    "a function that returns the client": (
        "import httpx\n\ndef _client():\n    return httpx.Client()\n\ndef f(u):\n    return _client().get(u)\n",
        "httpx", 2),
    "aiohttp": (
        "import aiohttp\n\nasync def f(u):\n    async with aiohttp.ClientSession() as s:\n"
        "        return await s.get(u)\n", "aiohttp", 2),
    "urllib3": ("import urllib3\n\ndef f(u):\n    return urllib3.request('GET', u)\n", "urllib3", 1),
    "primp": ("import primp\n\ndef f(u):\n    return primp.get(u)\n", "primp", 1),
    "pycurl": ("import pycurl\n\ndef f():\n    return pycurl.Curl()\n", "pycurl", 1),
    "curl_cffi": ("from curl_cffi import requests as cr\n\ndef f(u):\n    return cr.get(u)\n", "curl_cffi", 1),
    "websockets": ("import websockets\n\nasync def f(u):\n    return await websockets.connect(u)\n",
                   "websockets", 1),
    "websocket": ("import websocket\n\ndef f(u):\n    return websocket.create_connection(u)\n", "websocket", 1),
    "ddgs": ("from ddgs import DDGS\n\ndef f(q):\n    return DDGS().text(q)\n", "ddgs", 2),
    "duckduckgo_search": ("import duckduckgo_search\n\ndef f():\n    return duckduckgo_search.DDGS()\n",
                          "duckduckgo_search", 1),
    "huggingface_hub": (
        "from huggingface_hub import hf_hub_download\n\ndef f():\n    return hf_hub_download('r', 'f')\n",
        "huggingface_hub", 1),
    "qdrant_client": ("from qdrant_client import QdrantClient\n\ndef f():\n    return QdrantClient(url='u')\n",
                      "qdrant_client", 1),
    "weaviate": ("import weaviate\n\ndef f():\n    return weaviate.connect_to_local()\n", "weaviate", 1),
    "pinecone": ("from pinecone import Pinecone\n\ndef f():\n    return Pinecone(api_key='k')\n", "pinecone", 1),
    "chromadb HttpClient": ("import chromadb\n\ndef f():\n    return chromadb.HttpClient(host='h')\n",
                            "chromadb", 1),
    "veilid api_connector": ("import veilid\n\nasync def f(cb):\n    return await veilid.api_connector(cb)\n",
                             "veilid", 1),
    "smtplib": ("import smtplib\n\ndef f():\n    return smtplib.SMTP('h')\n", "smtplib", 1),
    "ftplib": ("import ftplib\n\ndef f():\n    return ftplib.FTP('h')\n", "ftplib", 1),
    "imaplib": ("import imaplib\n\ndef f():\n    return imaplib.IMAP4_SSL('h')\n", "imaplib", 1),
    "poplib": ("import poplib\n\ndef f():\n    return poplib.POP3('h')\n", "poplib", 1),
    "nntplib": ("import nntplib\n\ndef f():\n    return nntplib.NNTP('h')\n", "nntplib", 1),
    "telnetlib": ("import telnetlib\n\ndef f():\n    return telnetlib.Telnet('h')\n", "telnetlib", 1),
    "xmlrpc": ("import xmlrpc.client\n\ndef f(u):\n    return xmlrpc.client.ServerProxy(u)\n", "xmlrpc.client", 1),
    "webbrowser": ("import webbrowser\n\ndef f(u):\n    webbrowser.open_new_tab(u)\n", "webbrowser", 1),
    "ollama pull": ("import ollama\n\ndef f():\n    return ollama.pull('m')\n", "ollama", 1),
    "ollama through an injected receiver": (
        "import ollama\n\nclass M:\n    def __init__(self, injected=None):\n        self._o = injected or ollama\n\n"
        "    def get(self, m):\n        return self._o.pull(m)\n", "ollama", 1),
    "ollama web_search": (
        "import ollama\n\ndef f(q):\n    client = ollama.Client()\n    return client.web_search(q)\n", "ollama", 2),
    "ollama AsyncClient": ("from ollama import AsyncClient\n\nasync def f():\n    return AsyncClient()\n",
                           "ollama", 1),
    "model management endpoint literal": (
        "import requests\n\ndef f(base):\n    return base + '/api/pull'\n", "ollama", 1),
    "chromadb documents-only upsert": (
        "import chromadb\n\ndef f(coll, ids, docs):\n    coll.upsert(ids=ids, documents=docs)\n", "chromadb", 1),
    "chromadb splat": ("import chromadb\n\ndef f(coll, kw):\n    return coll.query(**kw)\n", "chromadb", 1),
    "chromadb query_texts constant": ("import chromadb\n\nKEY = 'query_texts'\n", "chromadb", 1),
    "chromadb query_texts keyword": (
        "import chromadb\n\ndef f(coll, q):\n    return coll.query(query_texts=[q])\n", "chromadb", 1),
    "chromadb embeddings None": (
        "import chromadb\n\ndef f(coll, ids, docs):\n    coll.add(ids=ids, documents=docs, embeddings=None)\n",
        "chromadb", 1),
    "chromadb conditional embeddings": (
        "import chromadb\n\ndef f(coll, ids, docs, e):\n"
        "    coll.update(ids=ids, documents=docs, embeddings=e if e else None)\n", "chromadb", 1),
    "chromadb telemetry on": ("import chromadb\n\ndef f(p):\n    return chromadb.PersistentClient(path=p)\n",
                              "chromadb", 1),
    "argv variable": ("import subprocess\n\ndef f(cmd):\n    subprocess.run(cmd)\n", "process", 1),
    "npm ci": ("import subprocess\n\ndef f():\n    subprocess.run(['npm', 'ci'])\n", "process", 1),
    "bash -c": ("import subprocess\n\ndef f(c):\n    subprocess.run(['bash', '-c', c])\n", "process", 1),
    "env assignment curl": (
        "import subprocess\n\ndef f(u):\n    subprocess.Popen(['env', 'X=1', 'curl', u])\n", "process", 1),
    "nice git": (
        "import subprocess\n\ndef f():\n    subprocess.check_call(['nice', '-n', '5', 'git', 'fetch'])\n", "process", 1),
    "timeout curl": ("import subprocess\n\ndef f():\n    subprocess.run(['timeout', '5', 'curl', 'u'])\n", "process", 1),
    "bwrap sh": (
        "import subprocess\n\ndef f():\n    subprocess.call(['bwrap', '--ro-bind', '/', '/', 'sh'])\n", "process", 1),
    "os.system chain": ("import os\n\ndef f():\n    os.system('true; curl x')\n", "process", 1),
    "getoutput": ("import subprocess\n\ndef f():\n    return subprocess.getoutput('git pull')\n", "process", 1),
    "posix_spawnp": (
        "import os\n\ndef f(p):\n    os.posix_spawnp('pip', ['pip', 'install', p], os.environ)\n", "process", 1),
    "sys.executable": (
        "import subprocess\nimport sys\n\ndef f():\n    subprocess.check_output([sys.executable, '-m', 'pip'])\n",
        "process", 1),
    "create_subprocess_exec": (
        "import asyncio\n\nasync def f():\n    await asyncio.create_subprocess_exec('ollama', 'pull', 'm')\n",
        "process", 1),
    "pty.spawn": ("import pty\n\ndef f():\n    pty.spawn(['ssh', 'h'])\n", "process", 1),
    "execvp": ("import os\n\ndef f():\n    os.execvp('wget', ['wget', 'u'])\n", "process", 1),
    "shell interpreter": (
        "import subprocess\n\ndef f():\n    subprocess.run('python3.12 -c pass', shell=True)\n", "process", 1),
}

# The families of what counts as a sink, each of which a positive above meets.
_FAMILIES = {
    "urllib", "http.client", "socket", "asyncio", "multiprocessing", "logging", "requests", "httpx",
    "aiohttp", "urllib3", "primp", "pycurl", "curl_cffi", "websockets", "websocket", "ddgs",
    "duckduckgo_search", "huggingface_hub", "qdrant_client", "weaviate", "pinecone", "chromadb",
    "veilid", "smtplib", "ftplib", "imaplib", "poplib", "nntplib", "telnetlib", "xmlrpc.client",
    "webbrowser", "ollama", "process",
}

_NEGATIVES = {
    "prose": (
        '"""Posts to /api/pull with query_texts, and calls urllib.request.urlopen."""\n'
        "# requests.get(u) and socket.create_connection((h, 80))\n"
        "import chromadb\nimport requests\n"
    ),
    "a local function named urlopen": "def urlopen(u):\n    return u\n\ndef f(u):\n    return urlopen(u)\n",
    # Docstrings exactly as the two string rules would count them, were
    # docstrings not prose: the census must pass them over.
    "docstrings that spell a string sink": (
        '"""query_texts"""\n'
        "import chromadb\nimport requests\n\n"
        "def f():\n"
        '    """Posts to /api/pull"""\n'
        "    return None\n"
    ),
    "a Unix socket": (
        "import socket\n\ndef f(p):\n    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n    s.connect(p)\n"
    ),
    "a program without a network": "import subprocess\n\ndef f():\n    subprocess.run(['nvidia-smi'])\n",
    "a wrapper around a program without a network": (
        "import subprocess\n\ndef f():\n    subprocess.run(['timeout', '5', 'ls'])\n"
    ),
    "documents in a module without chromadb": (
        "def f(resp, docs):\n    resp.update(documents=docs)\n    return DocumentsListResponse(documents=docs)\n"
    ),
    "chromadb with its telemetry off": (
        "import chromadb\nfrom chromadb.config import Settings\n\ndef f(p):\n"
        "    return chromadb.PersistentClient(path=p, settings=Settings(anonymized_telemetry=False))\n"
    ),
    "a syslog socket path": (
        "import logging.handlers\n\ndef f():\n    return logging.handlers.SysLogHandler(address='/dev/log')\n"
    ),
    "an exception class of a client library": (
        "import requests\n\nclass Refused(requests.RequestException):\n    pass\n\ndef f():\n    raise Refused('x')\n"
    ),
    "a text that does not parse": "def f(:\n    requests.get(u)\n",
}


# ---------------------------------------------------------------------------
# Fixture estates for EC2, EC3 and EC5.
# ---------------------------------------------------------------------------
_HOME = (
    "import requests\n\n"
    "def require_web(label):\n    return None\n\n"
    "def fetch(u):\n    require_web('fetch')\n    return requests.get(u)\n"
)
_HOME_GATED_ELSEWHERE = (
    "import requests\n\n"
    "def require_web(label):\n    return None\n\n"
    "def check():\n    require_web('check')\n\n"
    "def fetch(u):\n    return requests.get(u)\n"
)
_OWED = "import socket\n\ndef probe(h):\n    return socket.create_connection((h, 1))\n"
_OWED_GROWN = _OWED + "\nimport requests\n\ndef grow(u):\n    return requests.get(u)\n"
_EXEMPT = "import httpx\n\ndef talk(u):\n    return httpx.get(u)\n"
_PLUGIN = "import requests\n\ndef hook(u):\n    return requests.get(u)\n"
_MANIFEST = "name: p\npermissions:\n  - network_outbound\n"
_MANIFEST_WITHOUT = "name: p\n# network_outbound\npermissions:\n  - read_files\n"
_SUITE = (
    "def test_zz1_the_home_is_proven():\n"
    "    assert 'opti_oignon.gate_home' and 'opti_oignon.gone'\n"
)
_CARGO = '[package]\nname = "x"\nversion = "0.1.0"\n\n[dependencies]\ntokio = { version = "1", features = ["rt"] }\n'
_RUST = '// std::net is named in a comment\nfn main() {\n    let _s = "std::net";\n}\n'


def _clean_files():
    return {
        "opti_oignon/__init__.py": "",
        "opti_oignon/gate_home.py": _HOME,
        "opti_oignon/owed.py": _OWED,
        "opti_oignon/cli_talk.py": _EXEMPT,
        "opti_oignon/plugins/p/main.py": _PLUGIN,
        "opti_oignon/plugins/p/manifest.yaml": _MANIFEST,
        "tests/test_zz_contracts.py": _SUITE,
        "rust/x/Cargo.toml": _CARGO,
        "rust/x/src/main.rs": _RUST,
    }


def _home(guard, sinks=1, contracts=("zz1",)):
    return guard.Home(
        classes=("web",), gates=("require_web",), factories=(), callers_in=(), ungated={},
        sinks=sinks, contracts=contracts, reason="the fixture's home",
    )


def _clean_tables(guard):
    return {
        "HOMES": {"opti_oignon/gate_home.py": _home(guard)},
        "LEDGER": {"opti_oignon/owed.py": {"socket": 1}},
        "EXEMPT": {"opti_oignon/cli_talk.py": "the fixture's client talks to its own server"},
        "LIBRARIES": {
            "requests": ("network", "a client library", ()),
            "httpx": ("network", "a client library", ()),
        },
    }


def _write(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _set_tables(guard, tables):
    for name, value in tables.items():
        setattr(guard, name, value)


# ---------------------------------------------------------------------------
# EC1 -- the census counts every sink spelling and nothing else
# ---------------------------------------------------------------------------
def test_ec1_the_census_counts_every_sink_spelling_and_nothing_else():
    guard, restore = _load()
    try:
        # c1: one positive per family, every spelling counted, by its kind.
        assert len(_POSITIVES) >= 55, "at least fifty-five spellings are exercised"
        met = set()
        for name, (text, kind, sites) in _POSITIVES.items():
            assert guard.count_sinks(text) == sites, f"{name}: {sites} sink site(s) expected"
            kinds = {site.kind for site in guard.sink_sites(text)}
            assert kind in kinds, f"{name}: counted as {sorted(kinds)}, not as {kind!r}"
            met.add(kind)
        assert met == _FAMILIES, f"every family is met: missing {sorted(_FAMILIES - met)}"
        # c2: prose, a namesake, a Unix socket, an offline program and the rest count nothing.
        for name, text in _NEGATIVES.items():
            assert guard.count_sinks(text) == 0, f"{name}: nothing to count"
        # Positive witness in the same clause: the same Unix socket text, made internet, is one site.
        witness = _NEGATIVES["a Unix socket"].replace("socket.AF_UNIX", "socket.AF_INET")
        assert guard.count_sinks(witness) == 1, "the negative's probe can count"
        # And the docstrings' strings, written as values, are two sites: the
        # string rules are live on that text.
        spelled = _NEGATIVES["docstrings that spell a string sink"]
        as_values = spelled.replace('"""query_texts"""\n', 'KEY = "query_texts"\n').replace(
            '    """Posts to /api/pull"""\n', '    path = "/api/pull"\n')
        assert guard.count_sinks(as_values) == 2, "the same strings as values are counted"
    finally:
        restore()


# ---------------------------------------------------------------------------
# EC2 -- an empty estate is refused, and data is never opened
# ---------------------------------------------------------------------------
def test_ec2_an_empty_estate_is_refused_and_data_is_never_opened(tmp_path, capsys):
    guard, restore = _load()
    locked = None
    try:
        # c1: nothing to scan is a refusal, never a green.
        empty = tmp_path / "empty"
        empty.mkdir()
        assert guard.main(["egress_census_guard.py", str(empty)]) == 1, "an empty directory is refused"
        assert "nothing was scanned" in capsys.readouterr().out
        bare = _write(tmp_path / "bare", {"README.md": "no package here\n"})
        assert guard.main(["egress_census_guard.py", str(bare)]) == 1, "a root without the package is refused"
        assert "nothing was scanned" in capsys.readouterr().out
        # c2: a directory named data is pruned before listing, so its files are never opened.
        sink = "import requests\n\ndef f(u):\n    return requests.get(u)\n"
        root = _write(tmp_path / "estate", {
            "opti_oignon/__init__.py": "",
            "opti_oignon/data/p.py": sink,
            "opti_oignon/datax/p.py": sink,
        })
        locked = root / "opti_oignon" / "data" / "p.py"
        os.chmod(locked, 0)
        _set_tables(guard, {
            "HOMES": {},
            "LEDGER": {"opti_oignon/datax/p.py": {"requests": 1}},
            "EXEMPT": {},
            "LIBRARIES": {"requests": ("network", "a client library", ())},
        })
        opened = []
        real_read = guard._read_text

        def recording(path):
            opened.append(Path(path).as_posix())
            return real_read(path)

        guard._read_text = recording
        assert guard.main(["egress_census_guard.py", str(root)]) == 0, "the estate beside data is green"
        out = capsys.readouterr().out
        assert "2 module(s) scanned" in out, out
        assert "1 sink(s) owed in 1 module(s)" in out, "the witness under datax is counted"
        assert not [p for p in opened if "/opti_oignon/data/" in p], f"data was opened: {opened}"
        assert [p for p in opened if p.endswith("/opti_oignon/datax/p.py")], "the witness was read"
    finally:
        if locked is not None:
            os.chmod(locked, 0o644)
        restore()


# ---------------------------------------------------------------------------
# EC3 -- each question answers its own finding
# ---------------------------------------------------------------------------
_QUESTIONS = (
    "violations", "ungated_sinks", "count_drift", "ledger_growth", "stale_ledger_entries",
    "stale_exemptions", "stale_homes", "home_proofs_missing", "unpermitted_plugins",
    "unclassified_imports", "network_crates",
)


def _variant(guard, question):
    """The clean estate and tables, changed so that exactly one finding holds."""
    files, tables = _clean_files(), _clean_tables(guard)
    if question == "violations":
        files["opti_oignon/rogue.py"] = "import requests\n\ndef f(u):\n    return requests.post(u)\n"
    elif question == "ungated_sinks":
        files["opti_oignon/gate_home.py"] = _HOME_GATED_ELSEWHERE
    elif question == "count_drift":
        tables["HOMES"] = {"opti_oignon/gate_home.py": _home(guard, sinks=2)}
    elif question == "ledger_growth":
        files["opti_oignon/owed.py"] = _OWED_GROWN
    elif question == "stale_ledger_entries":
        tables["LEDGER"] = {"opti_oignon/owed.py": {"socket": 2}}
    elif question == "stale_exemptions":
        files["opti_oignon/quiet.py"] = "def f():\n    return 1\n"
        tables["EXEMPT"]["opti_oignon/quiet.py"] = "a module that no longer talks"
    elif question == "stale_homes":
        tables["HOMES"]["opti_oignon/gone.py"] = _home(guard)
    elif question == "home_proofs_missing":
        tables["HOMES"] = {"opti_oignon/gate_home.py": _home(guard, contracts=("zz9",))}
    elif question == "unpermitted_plugins":
        files["opti_oignon/plugins/p/manifest.yaml"] = _MANIFEST_WITHOUT
    elif question == "unclassified_imports":
        files["opti_oignon/extra.py"] = "import frobnicate\n"
    elif question == "network_crates":
        files["rust/x/src/net.rs"] = "use std::net::TcpStream;\n"
    return files, tables


def test_ec3_each_question_answers_its_own_finding(tmp_path):
    guard, restore = _load()
    try:
        _set_tables(guard, _clean_tables(guard))
        clean = guard.find_all(guard.read_estate(_write(tmp_path / "clean", _clean_files())))
        assert tuple(clean) == _QUESTIONS, "every question is asked, in its order"
        assert all(found == [] for found in clean.values()), f"the clean estate has no finding: {clean}"
        for question in _QUESTIONS:
            files, tables = _variant(guard, question)
            _set_tables(guard, tables)
            found = guard.find_all(guard.read_estate(_write(tmp_path / question, files)))
            answered = [name for name, items in found.items() if items]
            assert answered == [question], f"{question}: answered by {answered}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# EC4 -- the real tree
# ---------------------------------------------------------------------------
# The ledger as this guard found it when it was written: module -> {kind: count}.
BIRTH = {
    "opti_oignon/async_plugin_subprocess.py": {"process": 1},
    "opti_oignon/code_executor.py": {"process": 1},
    "opti_oignon/context_manager.py": {"process": 2},
    "opti_oignon/core_client.py": {"urllib": 1},
    "opti_oignon/dep_monitor.py": {"process": 1},
    "opti_oignon/memory/vector_store.py": {"chromadb": 1},
    "opti_oignon/model_lifecycle.py": {"ollama": 5, "requests": 2},
    "opti_oignon/network_hardening.py": {"socket": 1},
    "opti_oignon/plugin_subprocess.py": {"process": 1},
    "opti_oignon/rag_external.py": {"pinecone": 12, "qdrant_client": 1, "weaviate": 2},
    "opti_oignon/rag_store.py": {"chromadb": 4},
    "opti_oignon/sandbox_manager.py": {"process": 3},
    "opti_oignon/token_counter.py": {"urllib": 1},
    "opti_oignon/ui.py": {"process": 3, "urllib": 3, "webbrowser": 1},
    "opti_oignon/veilid/client.py": {"veilid": 2},
}
# The total sink sites the census found in the package once the front door and
# its three named paths were in place: the capacity the census must keep.
FLOOR = 86
# Every third-party top-level name the package imported when the guard was written.
_THIRD_PARTY = (
    "Crypto", "argon2", "bcrypt", "chromadb", "click", "cryptography", "ddgs", "docx",
    "duckduckgo_search", "fastapi", "fido2", "httpx", "joblib", "llama_cpp", "numpy", "ollama",
    "openpyxl", "oqs", "pinecone", "psutil", "pydantic", "pyotp", "pypdf", "pysqlcipher3",
    "pytesseract", "pywhispercpp", "qdrant_client", "qrcode", "requests", "sklearn", "sqlcipher3",
    "starlette", "tqdm", "uvicorn", "veilid", "weaviate", "websockets", "yaml",
)


def test_ec4_the_real_tree_is_green_and_its_ledger_never_grows():
    guard, restore = _load()
    try:
        result = guard.run(REPO)
        # c1: the census on the repository is green, with its green line.
        assert result.code == 0, "\n".join(result.lines)
        assert result.lines[-1].startswith("Egress census OK: "), result.lines[-1]
        # c2: the ratchet, as properties.
        assert guard.find_home_proofs_missing(result.estate, result.census) == []
        assert set(guard.LEDGER) <= set(BIRTH), f"owed modules beyond the birth: {sorted(set(guard.LEDGER) - set(BIRTH))}"
        for rel, kinds in guard.LEDGER.items():
            for kind, count in kinds.items():
                assert count <= BIRTH[rel].get(kind, 0), f"{rel}: {kind} {count} is above its birth"
        assert guard.EXEMPT and all(isinstance(r, str) and r.strip() for r in guard.EXEMPT.values())
        # c3: the capacity anchor.
        total = sum(len(sites) for sites in result.census.values())
        assert FLOOR > 0 and total >= FLOOR, f"{total} sink site(s), floor {FLOOR}"
        for rel in guard.LEDGER:
            assert len(result.census.get(rel, ())) >= 1, f"{rel} is owed and has no sink"
        # c4: every third-party name is classified.
        assert len(_THIRD_PARTY) == 38
        assert [n for n in _THIRD_PARTY if n not in guard.LIBRARIES] == []
        imported = guard.third_party_imports(result.estate)
        classified = [n for n in imported if n in guard.LIBRARIES]
        assert len(classified) >= 30, f"only {len(classified)} classified import(s) seen"
    finally:
        restore()


# ---------------------------------------------------------------------------
# EC5 -- the limits are written and the green carries its denominator
# ---------------------------------------------------------------------------
_LIMITS = (
    "vars", "__dict__", "globals", "attrgetter", "methodcaller", "exec", "eval",
    "received from another module", "built elsewhere", "LIBRARIES", "Ollama server", "pip",
    "npm", "git", "veilid-server", "control flow", "destination", "C extensions", "frontend",
    "android/", "opti_oignon/data/", "maintainer scripts",
)


def test_ec5_the_limits_are_written_and_the_green_carries_its_denominator(tmp_path, capsys):
    guard, restore = _load()
    try:
        # c1: the docstring names each limit of the census.
        doc = guard.__doc__ or ""
        start = doc.find("WHAT THE CENSUS DOES NOT SEE")
        assert start >= 0, "the limits have their own section"
        section = doc[start:]
        assert [limit for limit in _LIMITS if limit not in section] == []
        # c2: the green line names every figure as a number.
        _set_tables(guard, _clean_tables(guard))
        root = _write(tmp_path / "clean", _clean_files())
        assert guard.main(["egress_census_guard.py", str(root)]) == 0
        line = capsys.readouterr().out.strip().splitlines()[-1]
        assert line == (
            "Egress census OK: 5 module(s) scanned, 1 sink site(s) in 1 gate home(s), "
            "1 sink(s) owed in 1 module(s), 1 exempt by name, 1 bundled plugin(s) gated at the host, "
            "2 third-party import(s) classified, 1 crate manifest(s) read; the ledger may only shrink."
        ), line
        assert len(re.findall(r"\d+", line)) == 9, "nine figures, each a number"
    finally:
        restore()
