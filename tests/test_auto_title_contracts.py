"""Contracts for the automatic conversation titles (blocks 179 and 180): the store titles a conversation itself,
from the words the user typed in its first turns, inside the transaction that saves, deletes or lowers a user turn;
the local model may refine that title (rule b, off as shipped), held to the rule's own words.

  * TT1 -- rule (a) gives the design's golden titles: the plain words kept; a slash command, a path, an address, a
    key, a random string and every word holding a digit dropped; a turn that names a password titles nothing.
  * TT2 -- no secret's shape survives: an address, an e-mail, a path, a provider key, a JWT, a hex or base64 run, a
    card number typed whole or in groups, user:pass@host, a random-case string, an encryption prefix, a word holding
    a digit of any script or another numeral.
  * TT3 -- a title is at most max_chars code points, cut at a space, ends with the ellipsis, and never splits a
    combining sequence; an emoji is dropped and a zero-width character is a space; a script written without spaces
    titles, its run ending where a digit, a mark of punctuation or another script begins.
  * TT4 -- only typed words title: a document, refined, legacy or withdrawn turn and words typed beside a document
    give none; the next user turn titles when the first gives none, within max_user_turns.
  * TT5 -- typed_text is memory/auto_capture.typed_turns, on a case table.
  * TT6 -- the first user turn saved titles the conversation (encrypted at rest, source auto, its basis encrypted);
    an answer changes nothing; each save publishes once.
  * TT7 -- a title the user gave (manual) survives every mutation of the turns and a sync apply.
  * TT8 -- deleting the turn a title came from titles from the next one, or returns to the placeholder; the retry's
    re-add brings the same title back.
  * TT9 -- withdrawing a document of the turn a title came from titles the conversation again.
  * TT10 -- an older file gains the title's columns: a placeholder conversation is the default, titled when the backend
    starts (tend_titles) from a typed turn it holds; any other title is legacy, never replaced, and encrypted then,
    its basis too -- only the value the pass read, and nothing at all with field encryption off.
  * TT11 -- at rest every title but the placeholder and every basis is encrypted; a read is plaintext; a title still
    encrypted reads empty: keyless, a listing tries one; with the key, a title that does not decrypt, or whose
    decryption fails outright, is that title's alone. TT11b -- a sync apply re-encrypts the title (supersedes
    tests/test_conversation_apply.py::test_apply_materialises_conversation_and_reencrypts).
  * TT12 -- no title and no search word reaches the log, at DEBUG.
  * TT13 -- a search finds a conversation by its encrypted title, with or without the titles' configuration.
  * TT14 -- a sync payload carries the title's source; an apply keeps a manual title the user's, a placeholder the
    default, and freezes any other received title (legacy); a received title is one line within the manual bound,
    and an empty one -- its sender had no key -- leaves this device's title while a received turn holds its words.
  * TT21 -- a title the user types is folded to one line, by the store whatever its caller; empty, blank or past the
    bound is refused (422), and the route checks it before it renames or creates.
  * TT23 -- the wiring on the real package's shape: conversation.py imports conversation_titles under its real
    name; the chat socket and the CLI create a conversation with no title; the backend's start titles the
    placeholders; the fallback encryption prefixes and the placeholder are the real ones.
  * TT25 -- titles.yaml: the loader reads exactly the file's keys; an unknown key, a wrong type, a value out of range
    or a malformed password entry turns titles off, said by name, and the store keeps the placeholder.
  * TT26 -- the CLI session hands no title to the conversation it creates.
  * TT27 -- the verb gives a manual title back to the rule, in one publish; the route answers the summary.
  * TT28 -- a turn that cannot be read for a title is still saved, its title left as it was; a sealed turn, whatever
    its origin, lets nothing after it title; a keyless deletion or withdrawal touches the title only when it takes
    the title's own turn, and then puts the placeholder back, titled again as the backend starts with the key.
  * TT29 -- the chats index offers "Use automatic title" for a title the user gave, through the store, and shows
    the title the server kept after a rename or the verb, in the page and in the store's list.
  * TT30 -- the legacy history import stores its title and its turns encrypted, the title never replaced.
  * TT31 -- the Markdown export's heading is one line whatever the title.
  * TT32 -- uvicorn's access and WebSocket handshake lines say a request's path, never its query.
  * TT33 -- a turn that names a password, anywhere in its content, titles nothing, nor does any later turn; a title
    drawn before it stays; each name of titles.yaml names one as its pattern says, the listed forms do whatever the
    file holds, the patterns' language holds on names of its own, and words that only look like a name are none; a
    long text naming nothing is never cut, a turn added where it cannot change the title reads nothing, and the
    turns are read in the order they were saved.
  * TT34 -- titles.yaml is read once, under a lock: a second first reader waits for the reading.

Rule (b), block 180:

  * TT15 -- a refined title lands only over the rule's own -- automatic, from the same turn under the same print,
    its digest as read --, in one publish, and only as a title rule (a) keeps whole; a rename, a deletion, a changed
    print or digest write nothing; a turn added leaves it; the rule draws it again when its print changes or its
    own turn goes; a placeholder, a manual or legacy title and a sealed turn give no state to refine.
  * TT16 -- the model's answer is held to the rule's own words: kept_words are rule (a)'s words before its bound;
    an answer is its first line, a label taken off, refused when the rule would drop a word of it, when it names a
    password, when a word is neither typed nor a function word, when the bound would cut it; a run of a script
    written without spaces is given back whole or refused.
  * TT17 -- a call past its deadline is given up, nothing written, and no other call to that model starts while it
    lives; the queue holds max_queued jobs, one drain at a time; a job is pending until its deadline or its cancel,
    and one that waited past its bound in the queue never runs.
  * TT18 -- a backend that cannot say it answers on this machine is never called, whatever the mode: a LAN address,
    a host name, no endpoint or an unreadable one; in the process, localhost, a loopback or an unspecified address
    are; the governor is asked only for a call that may go.
  * TT19 -- nothing is dispatched with the refinement or titles off, the emergency stop engaged or absent, a title
    that is not the rule's own, no model named; each title is asked once in a process, a rename forgets it; a
    failing store is no job, never an error; titles.yaml may name the model.
  * TT20 -- the call is the background's: the shipped file and the defaults class titler as background with a
    floor; it is admitted as titler, its wait bounded and its cancel handed over, its ticket held on the thread that
    asks the backend, an admission no call holds handed back; a governor that cannot be had admits nothing.
  * TT22 -- GET /api/conversations/{id}/title says pending, with titles.yaml's retry_after_ms, while a job lives,
    and not after its deadline; the summary and the detail say it; a rename and a deletion cancel the job.
  * TT24 -- the interface watches a pending title at the server's pace only, one watcher a conversation, and puts
    the title the server answers in the list once nothing is pending.
  * TT35 -- the model is handed the words rule (a) keeps of the title's turn, cut to max_input_chars, under the
    system prompt of titles.yaml: no number, path, code, pasted part, answer or password-naming turn.
  * TT36 -- the wiring on the real package's shape: the refinement handed the conversation right before each done,
    the rename and the deletion cancelling it, the refiner importing nothing of the package but the titles module,
    and the public locality check the private one.
  * TT37 -- no word of a turn, of an answer or of a backend's error reaches the log: the outcomes are counted.
  * TT38 -- titles.yaml's refine section is read exactly, every key typed and bounded; local.yaml sets the
    machine's own switch and model, anything else said by name and setting nothing.

Local-only (the public distribution ships no tests). conversation.py and conversation_titles.py are loaded in the
shared isolation window: safe_connect a plain sqlite3 connection, DATA_DIR a temporary directory, the field key a
reversible marker (E:) as tests/test_conversation_apply.py does. Nothing reaches a model, the network or data/.
"""

import ast
import base64
import contextlib
import dataclasses
import hashlib
import json
import logging
import re
import sqlite3
import sys
import threading
import types
import unicodedata
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
_CONVERSATION = "opti_oignon.conversation"
_TITLES = "opti_oignon.conversation_titles"
_REFINER = "opti_oignon.title_refiner"
_TITLES_YAML = REPO / "opti_oignon" / "config" / "titles.yaml"

BUDGET_S = {
    "test_tt1_rule_a_gives_the_golden_titles": 4.0,
    "test_tt2_no_secret_shape_survives_in_a_title": 4.0,
    "test_tt3_a_title_is_bounded_in_code_points_and_cut_at_a_space": 4.0,
    "test_tt4_only_typed_words_title_a_conversation": 4.0,
    "test_tt5_typed_text_is_the_memorys_typed_turns": 4.0,
    "test_tt6_the_first_user_turn_saved_titles_the_conversation": 4.0,
    "test_tt7_a_title_the_user_gave_survives_every_mutation": 4.0,
    "test_tt8_deleting_the_turn_a_title_came_from_titles_again": 4.0,
    "test_tt9_withdrawing_the_document_of_the_titling_turn_titles_again": 4.0,
    "test_tt10_an_older_file_gains_the_columns_and_is_titled_once": 4.0,
    "test_tt11_titles_and_bases_are_encrypted_at_rest": 4.0,
    "test_tt11b_a_sync_apply_reencrypts_the_title": 4.0,
    "test_tt12_no_title_and_no_search_word_reaches_the_log": 4.0,
    "test_tt13_a_search_finds_a_conversation_by_its_encrypted_title": 4.0,
    "test_tt14_a_sync_apply_keeps_a_manual_title_and_freezes_any_other": 4.0,
    "test_tt21_a_typed_title_is_one_line_within_its_bound": 4.0,
    "test_tt23_the_wiring_on_the_real_package_shape": 4.0,
    "test_tt25_titles_yaml_is_read_exactly_or_turns_titles_off_by_name": 4.0,
    "test_tt26_the_cli_session_hands_no_title": 4.0,
    "test_tt27_the_verb_gives_a_manual_title_back_to_the_rule": 4.0,
    "test_tt28_a_turn_that_cannot_be_read_for_a_title_is_still_saved": 4.0,
    "test_tt29_the_chats_index_offers_the_automatic_title_for_a_manual_one": 4.0,
    "test_tt30_the_legacy_history_import_is_encrypted_and_never_retitled": 4.0,
    "test_tt31_the_markdown_export_says_its_title_on_one_line": 4.0,
    "test_tt32_uvicorns_lines_say_a_path_never_its_query": 4.0,
    "test_tt33_a_turn_that_names_a_password_titles_nothing_nor_any_later_turn": 4.0,
    "test_tt34_a_second_first_reader_of_titles_yaml_waits_for_the_reading": 4.0,
    "test_tt15_a_refined_title_lands_only_over_the_rules_own": 4.0,
    "test_tt16_the_models_answer_is_held_to_the_rules_own_words": 4.0,
    "test_tt17_a_call_past_its_deadline_is_given_up_and_the_queue_is_bounded": 8.0,
    "test_tt18_a_backend_that_cannot_say_it_answers_on_this_machine_is_never_called": 8.0,
    "test_tt19_nothing_is_dispatched_unless_the_title_is_the_rules_own_and_all_may_run": 6.0,
    "test_tt20_the_call_is_the_backgrounds_admitted_as_titler_its_ticket_held_where_it_is_asked": 6.0,
    "test_tt22_the_title_route_says_pending_while_a_job_lives_and_the_routes_cancel_it": 4.0,
    "test_tt24_the_interface_watches_a_pending_title_at_the_servers_pace": 4.0,
    "test_tt35_the_model_is_handed_the_words_the_rule_keeps_and_nothing_else": 4.0,
    "test_tt36_the_refinement_is_wired_on_the_real_package_shape": 4.0,
    "test_tt37_no_word_of_a_turn_or_of_an_answer_reaches_the_log": 4.0,
    "test_tt38_the_refine_section_is_read_exactly_and_a_machine_sets_its_own_switch": 4.0,
}


# ---------------------------------------------------------------------------
# The store in the isolation window
# ---------------------------------------------------------------------------
def _encryption(has_key):
    """A stand-in for opti_oignon.encryption: the names conversation.py imports, its manager's key as asked."""
    stand_in = types.ModuleType("opti_oignon.encryption")
    stand_in.decrypt_field = lambda value: value
    stand_in.encrypt_field = lambda value: value
    stand_in._ENCRYPTED_PREFIX_V2, stand_in._ENCRYPTED_PREFIX_V1 = "ENC2:", "ENC:"
    stand_in.get_encryption_manager = lambda: types.SimpleNamespace(has_key=has_key)
    return stand_in


def _load(tmp_path, *, auto_capture=False, opaque=False, encryption=None, refiner=False, extra=None):
    """conversation.py and conversation_titles.py in the window, the field key a reversible marker: E: then the text,
    or, ``opaque``, E: then its base64, which holds none of its words as a real ciphertext does. Like the real one,
    the marker passes a value already marked through, and the store reads it as it reads the real prefixes."""
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda path, **kw: sqlite3.connect(path, check_same_thread=kw.get("check_same_thread", False))
    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = tmp_path
    targets = {_TITLES: source("conversation_titles.py"), _CONVERSATION: source("conversation.py")}
    if refiner:
        targets[_REFINER] = source("title_refiner.py")
    packages = ()
    if auto_capture:
        targets["opti_oignon.memory.auto_capture"] = source("memory", "auto_capture.py")
        packages = ("opti_oignon.memory",)
    seeded = {"opti_oignon.db_utils": db, "opti_oignon.config": cfg}
    if encryption is not None:
        seeded["opti_oignon.encryption"] = encryption
    seeded.update(extra or {})
    loaded, restore = isolate(targets=targets, seeded=seeded, packages=packages)
    mod = loaded[_CONVERSATION]
    marked = ("E:",) + tuple(mod._ENCRYPTED_PREFIXES)
    # Like the real one, the marker hands a value that is no text back as it came.
    if opaque:
        mod._encrypt = lambda v: v if not isinstance(v, str) or v.startswith(marked) \
            else "E:" + base64.b64encode(v.encode()).decode("ascii")
        mod._decrypt = lambda v: base64.b64decode(v[2:]).decode() if isinstance(v, str) and v.startswith("E:") else v
    else:
        mod._encrypt = lambda v: v if not isinstance(v, str) or v.startswith(marked) else "E:" + v
        mod._decrypt = lambda v: v[2:] if isinstance(v, str) and v.startswith("E:") else v
    mod._ENCRYPTED_PREFIXES = marked
    return mod, loaded, restore


def _routes(manager, refiner=None):
    """api/routes_conversations.py in a window of its own, its deps the store under test, its schemas the real ones;
    ``refiner`` the title refinement it reaches, when given."""
    deps = types.ModuleType("opti_oignon.api.deps")
    deps.CONVERSATION_AVAILABLE = True
    deps.conversation_manager = manager
    targets = {"opti_oignon.api.schemas": source("api", "schemas.py"),
               "opti_oignon.api.routes_conversations": source("api", "routes_conversations.py")}
    seeded = {"opti_oignon.api.deps": deps}
    if refiner is not None:
        seeded["opti_oignon.title_refiner"] = refiner
    loaded, restore = isolate(targets=targets, seeded=seeded, packages=("opti_oignon.api",))
    return loaded["opti_oignon.api.routes_conversations"], loaded["opti_oignon.api.schemas"], restore


def _spy(mod):
    """Every publish the store makes, with the snapshot it would send."""
    calls = []

    def publish(conv_id, payload_fn=None, **kwargs):
        calls.append((conv_id, payload_fn() if payload_fn is not None else None))

    mod._sync_publish_conversation = publish
    return calls


def _row(path, conv_id):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return dict(conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone())
    finally:
        conn.close()


def _plain(path, conv_id, **columns):
    """Set ``columns`` of a conversation's row as an older process left them, behind the store."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("UPDATE conversations SET " + ", ".join(f"{name} = ?" for name in columns) + " WHERE id = ?",
                     (*columns.values(), conv_id))
        conn.commit()
    finally:
        conn.close()


def _typed(manager, conv_id, text):
    return manager.add_message(conv_id, "user", text, origin="typed")


def _config(titles, **derive):
    """The shipped configuration, with ``derive`` keys replaced."""
    data = yaml.safe_load(_TITLES_YAML.read_text(encoding="utf-8"))
    data["derive"].update(derive)
    return titles.read_config(data)


# ---------------------------------------------------------------------------
# TT1 -- the golden titles
# ---------------------------------------------------------------------------
_GOLDEN = [
    ("Comment calculer une ANOVA en R?", "Comment calculer une ANOVA en R"),
    ("/code fix the parser in src/a.py please", "fix the parser in please"),
    ("my api key is sk-abc123XYZ456 why 401", "my api key is why"),
    ("mail bob@example.com about 2026 budget", "mail about budget"),
    ("AKIAIOSFODNN7EXAMPLE leaked?", "leaked"),
    ("XkqTrvPzLmWn", None),
    ("card 4111111111111111 refused", "card refused"),
    ("```\nprint('only code')\n```", None),
    ("ENC2:Zm9vYmFy", None),
    ("# **How** do I _merge_ two `git` branches?", "How do I merge two branches"),
    # A turn that names a password titles nothing, wherever the secret sits -- before the name or after it --;
    # whatever its case, accents and plural; in an option, an identifier or code; written with lookalikes.
    ("my password is hunter2", None),
    ("is hunter2 a strong password?", None),
    ("mon mot de passe est soleil merci", None),
    ("mes mots de passe sont soleil et azerty", None),
    ("PASSWORD is hunter thanks", None),
    ("mysql -u root --password hunter2", None),
    ("DB_PASSWORD is hunter2", None),
    ("my dbPassword is soleil", None),
    ("the adminPwd is chat", None),
    ("p@ssw0rd is blue", None),
    ("my `passwd` is blue", None),
    ("fix it\n```\nexport PASSWORD=abc\n```\nthen deploy", None),
    ("minha senha " + chr(0xE9) + " gato", None),
    # A passphrase, a wallet's recovery phrase: words of letters alone, kept out by their names.
    ("my pass phrase is correct horse battery staple", None),
    ("my seed phrase is abandon ability able about above", None),
    # A listed name in its usual spellings, in an identifier, after a sigil, before a digit.
    ("the Wi-Fi password is hunter", None),
    ("rootpassword --plaintext soleil", None),
    ("in PHP $password = 'hunter' fails", None),
    ("PASSWORD1: hunter", None),
    # Words that name no password keep their title.
    ("How do I pass code to a function", "How do I pass code to a function"),
    ("how does pwd work in bash", "how does pwd work in bash"),
    ("PwC audit report", "PwC audit report"),
    ("Fix the slow query", "Fix the slow query"),
    ("What are the key differences between TCP and UDP", "What are the key differences between TCP and UDP"),
    ("Explain tokens in transformers", "Explain tokens in transformers"),
    ("Comment tailler un pin parasol", "Comment tailler un pin parasol"),
    ("Cl" + chr(0xE9) + " USB non reconnue sous Windows", "Cl" + chr(0xE9) + " USB non reconnue sous Windows"),
    # Code goes whole, its fence of backticks or tildes; a list's marker goes.
    ("please run `ls -la` today", "please run today"),
    ("fix it\n~~~\nls -la\n~~~\nthen deploy", "fix it then deploy"),
    ("1. Plan the garden", "Plan the garden"),
    # No digit and no other numeral, ever: a number, a version, a word holding digits goes whole.
    ("install python3 on x86 with 4k screens", "install on with screens"),
    ("Upgrade from Python 3.11 to 3.12", "Upgrade from Python to"),
    ("budget 2026 review", "budget review"),
    ("error 401 card 4111111111111111 refused", "error card refused"),
    ("4521 is my PIN", "is my PIN"),
    ("unlock code 4521 wifi soleil42", "unlock code wifi"),
    ("error " + chr(0xFF14) + chr(0xFF15) + " and x" + chr(0xB2) + " and " + chr(0x2167), "error and and"),
]


def test_tt1_rule_a_gives_the_golden_titles(tmp_path):
    _mod, loaded, restore = _load(tmp_path)
    try:
        titles = loaded[_TITLES]
        cfg = _config(titles).derive
        assert [titles.derive_title(text, cfg) for text, _title in _GOLDEN] == [title for _text, title in _GOLDEN]
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT2 -- no secret's shape survives
# ---------------------------------------------------------------------------
_SECRET_SHAPES = [
    "https://example.com/reset?token=abc", "bob@example.com", "/etc/shadow", "C:\\Users\\bob\\notes",
    "sk-proj-abcdEFGH12345678", "ghp_16C7e42F292c6912E7710c838347Ae178B4a", "AKIAIOSFODNN7EXAMPLE",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl", "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b",
    "c2VjcmV0IHBhc3N3b3JkIGhlcmU=", "4111111111111111", "admin:hunter2@db.local", "XkqTrvPzLmWnQ",
    "ENC2:Zm9vYmFyYmF6", "ENC:Zm9vYmFy", "1234567890",
    # Letters joined by a path's, a scheme's or an identifier's separator: no plain word.
    "etc/shadow", "user:hunter", "my_api_key",
    # A number typed in groups, a short number, a word holding digits: no digit ever stays.
    "4111 1111 1111 1111", "FR76 3000 6000 0112 3456 7890 189", "06 12 34 56 78", "123 45 6789",
    "4111-1111-1111-1111", "Xk42pqrs", "a1b2c3d4", "abc123", "longidentifier12", "hunter2", "soleil42", "4521",
    "ab12 cd34 ef56 gh78", "xy12 zt34 4521", "python3", "x86", "3.11",
    # The digits of other scripts, and other numerals.
    chr(0xFF14) + chr(0xFF11) + chr(0xFF11) + chr(0xFF11), "x" + chr(0xB2), chr(0x2167), chr(0x2460),
    chr(0x0664) + chr(0x0665),
]


def test_tt2_no_secret_shape_survives_in_a_title(tmp_path):
    _mod, loaded, restore = _load(tmp_path)
    try:
        titles = loaded[_TITLES]
        cfg = _config(titles).derive
        found = {shape: titles.derive_title(f"please check {shape} today", cfg) for shape in _SECRET_SHAPES}
        # Nothing of the shape stays: the title is the plain words around it.
        leaked = {shape: title for shape, title in found.items() if title != "please check today"}
        assert leaked == {}, leaked
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT3 -- bounded in code points, cut at a space, a combining sequence whole
# ---------------------------------------------------------------------------
def test_tt3_a_title_is_bounded_in_code_points_and_cut_at_a_space(tmp_path):
    _mod, loaded, restore = _load(tmp_path)
    try:
        titles = loaded[_TITLES]
        cfg = _config(titles).derive
        accented = "caf" + "e" + chr(0x301)
        words = " ".join([accented] * 30)
        title = titles.derive_title(words, cfg)
        assert len(title) <= cfg.max_chars and title.endswith(cfg.ellipsis), title
        kept = title[: -len(cfg.ellipsis)]
        composed = unicodedata.normalize("NFC", words)
        assert composed.startswith(kept) and composed[len(kept)] == " ", "cut at a space, never inside a word"
        assert all(word == "caf" + chr(0xE9) for word in kept.split(" ")), "NFC: the sequence composes, never splits"
        # An emoji is no word; a zero-width space separates two.
        emoji = chr(0x1F600)
        assert titles.derive_title(f"hello {emoji} world", cfg) == "hello world"
        assert titles.derive_title("hello" + chr(0x200B) + "world again", cfg) == "hello world again"
        narrow = _config(titles, max_chars=12).derive
        assert titles.derive_title("alpha beta gamma delta", narrow) == "alpha" + narrow.ellipsis
        # A word cut where no space is: never between a letter and the mark on it (q has no composed form).
        tight = _config(titles, max_chars=6).derive
        assert titles.derive_title("abq" + chr(0x301) + "rstu", tight) == "ab" + tight.ellipsis
        # A script written without spaces is one long word: bounded, never dropped.
        han = "".join(chr(0x4E00 + index) for index in range(80))
        bounded = titles.derive_title(han + chr(0x3002) + han, cfg)
        assert len(bounded) == cfg.max_chars and bounded.endswith(cfg.ellipsis) and han.startswith(bounded[:-3])
        # Where a script written without spaces meets a digit, a mark of punctuation or another script, its word
        # ends; a run of kana or of Myanmar letters is one word however long.
        hira = "".join(chr(code) for code in (0x3072, 0x3089, 0x304C, 0x306A))
        kata = "".join(chr(code) for code in (0x30AB, 0x30BF, 0x30AB, 0x30CA))
        kanji = "".join(chr(code) for code in (0x65E5, 0x672C, 0x8A9E))
        assert titles.derive_title(kata + chr(0x30FB) + kanji, cfg) == kata + " " + kanji
        assert titles.derive_title(hira + chr(0xFF0C) + kanji + "," + kata, cfg) == " ".join((hira, kanji, kata))
        assert titles.derive_title("Excel" + hira + "3" + kanji, cfg) == "Excel " + hira + " " + kanji
        assert titles.derive_title("hello" + chr(0xFF08) + "world" + chr(0xFF09) + " again", cfg) == \
            "hello world again"
        iterated = chr(0x4EBA) + chr(0x3005) + chr(0x306E) + chr(0x8A71)
        assert titles.derive_title(iterated, cfg) == iterated
        # Their marks of punctuation separate words of any script; a word of marks alone is no word.
        assert titles.derive_title("alpha" + chr(0x30FB) + "beta gamma", cfg) == "alpha beta gamma"
        assert titles.derive_title("hello" + chr(0x3002) + "world again", cfg) == "hello world again"
        assert titles.derive_title("hello " + chr(0x301) + " world again", cfg) == "hello world again"
        long_kana = hira * 8
        burmese = "".join(chr(code) for code in (0x1019, 0x103C, 0x1014, 0x103A, 0x1019, 0x102C)) * 6
        halfwidth = "".join(chr(code) for code in (0xFF76, 0xFF80, 0xFF76, 0xFF85)) * 8
        javanese = "".join(chr(code) for code in (0xA98F, 0xA9B8, 0xA9A0)) * 10
        # Tibetan's syllable marks join a word's syllables: the run is one word, its marks kept.
        tibetan = chr(0x0F0B).join(("".join(chr(code) for code in (0x0F56, 0x0F7C, 0x0F51)),
                                    "".join(chr(code) for code in (0x0F40, 0x0FB1, 0x0F72)),
                                    "".join(chr(code) for code in (0x0F66, 0x0F90, 0x0F51)))) * 3
        for run in (long_kana, burmese, halfwidth, javanese, tibetan):
            assert len(run) > cfg.max_token_chars and titles.derive_title(run, cfg) == run, run
        # A syllable mark that ends a run is stripped like a word's punctuation, never the word with it.
        assert titles.derive_title(tibetan + chr(0x0F0B), cfg) == tibetan
        # The title is drawn from the typed words' first max_read_chars characters, cut back to a space and before
        # an inline code span the cut would leave open.
        assert titles.derive_title("1 " * (cfg.max_read_chars // 2) + "Plan the garden", cfg) is None
        assert titles.derive_title("1 " * (cfg.max_read_chars // 2 - 10) + "Plan the garden", cfg) == "Plan the garden"
        assert titles.derive_title("1 " * (cfg.max_read_chars // 2 - 4) + "plan garden", cfg) == "plan"
        assert titles.derive_title("1 " * (cfg.max_read_chars // 2 - 10) + "plan `deploy the secretword now` later",
                                   cfg) == "plan"
        # The vertical and the compatibility forms of punctuation separate words.
        for mark in (chr(0xFE10), chr(0xFE31)):
            assert titles.derive_title("alpha" + mark + "beta gamma", cfg) == "alpha beta gamma"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT4 -- only typed words title
# ---------------------------------------------------------------------------
def test_tt4_only_typed_words_title_a_conversation(tmp_path):
    mod, loaded, restore = _load(tmp_path)
    try:
        titles = loaded[_TITLES]
        cfg = _config(titles)
        doc = "Quarterly figures for the northern region"
        pasted = "summarize this " + doc
        typed_then_doc = "summarize please\n\n" + doc
        cases = {
            "a document turn": [(doc, "document", [], [])],
            "a refined turn": ("Reworded question about budgets", "refined", [], []),
            "a legacy turn": ("Old question about budgets", "legacy", [], []),
            "words typed beside a document": (pasted, "typed", [[0, 15, "typed"], [15, len(pasted), "document"]],
                                              ["document"]),
            "a withdrawn turn": ("Typed question about budgets", "typed", [], ["withdrawn"]),
        }
        for name, turn in cases.items():
            turns = turn if isinstance(turn, list) else [turn]
            assert titles.first_typed_source(turns, cfg, defect=mod._origin_defect) is None, name
        separate = (typed_then_doc, "typed", [[0, 16, "typed"], [18, len(typed_then_doc), "document"]],
                    ["document"])
        assert titles.first_typed_source([separate], cfg, defect=mod._origin_defect)[0] == "summarize please"
        later = [("Old question", "legacy", [], []), ("Plan the garden layout", "typed", [], [])]
        assert titles.first_typed_source(later, cfg)[0] == "Plan the garden layout"
        beyond = [("x", "legacy", [], [])] * cfg.derive.max_user_turns + [("Plan the garden", "typed", [], [])]
        assert titles.first_typed_source(beyond, cfg) is None, "only the first max_user_turns turns are read"
        # In the store: a document turn leaves the placeholder, the typed turn after it titles.
        manager = mod.ConversationManager(db_path=tmp_path / "c.db", publish_to_sync=False)
        conv = manager.create_conversation()
        manager.add_message(conv.id, "user", doc, origin="document")
        assert manager.get_conversation(conv.id).title == "New conversation"
        _typed(manager, conv.id, "Plan the garden layout")
        assert manager.get_conversation(conv.id).title == "Plan the garden layout"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT5 -- typed_text is the memory's typed_turns
# ---------------------------------------------------------------------------
def test_tt5_typed_text_is_the_memorys_typed_turns(tmp_path):
    _mod, loaded, restore = _load(tmp_path, auto_capture=True)
    try:
        titles, capture = loaded[_TITLES], loaded["opti_oignon.memory.auto_capture"]
        text = "alpha beta gamma delta epsilon"
        table = [
            (text, "typed", []), (text, "refined", []), (text, "document", []), (text, "legacy", []),
            (text, "typed", [[0, 5, "typed"], [6, 10, "document"]]),
            (text, "typed", [[0, 5, "typed"], [5, 10, "document"]]),
            (text, "typed", [[0, 5, "typed"], [11, 16, "typed"], [17, 22, "document"]]),
            (text, "refined", [[0, 10, "refined"], [10, 16, "typed"]]),
            (text, "typed", [[0, 5, "typed"], [3, 9, "typed"]]),
            (text, "typed", [[0, 99, "typed"]]),
            ("  spaced  ", "typed", []),
        ]
        for content, origin, segments in table:
            turns = capture.typed_turns([{"role": "user", "content": content, "origin": origin,
                                          "segments": segments}])
            expected = turns[0]["content"] if turns else ""
            assert titles.typed_text(content, origin, segments) == expected, (origin, segments)
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT6 -- the first user turn titles, encrypted; one publish per save
# ---------------------------------------------------------------------------
def test_tt6_the_first_user_turn_saved_titles_the_conversation(tmp_path):
    mod, _loaded, restore = _load(tmp_path)
    try:
        calls = _spy(mod)
        db = tmp_path / "c.db"
        manager = mod.ConversationManager(db_path=db)
        conv = manager.create_conversation()
        assert (conv.title, conv.title_source) == ("New conversation", "default")
        _typed(manager, conv.id, "Comment calculer une ANOVA en R?")
        row = _row(db, conv.id)
        basis = hashlib.sha256(b"Comment calculer une ANOVA en R?").hexdigest()
        assert (row["title"], row["title_source"], row["title_basis"]) == \
            ("E:Comment calculer une ANOVA en R", "auto", "E:" + basis)
        manager.add_message(conv.id, "assistant", "Use aov().", origin="assistant", context=[], lineage=[])
        assert _row(db, conv.id)["title"] == row["title"], "an answer changes nothing"
        assert manager.get_conversation(conv.id).title == "Comment calculer une ANOVA en R"
        assert len(calls) == 3, "one publish per save, the title riding the turn's"
        assert calls[1][1]["conversation"]["title"] == "Comment calculer une ANOVA en R"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT7 -- a manual title survives
# ---------------------------------------------------------------------------
def test_tt7_a_title_the_user_gave_survives_every_mutation(tmp_path):
    mod, _loaded, restore = _load(tmp_path)
    try:
        _spy(mod)
        db = tmp_path / "c.db"
        manager = mod.ConversationManager(db_path=db)
        named = manager.create_conversation(title="My budget notes")
        assert named.title_source == "manual" and _row(db, named.id)["title"] == "E:My budget notes"
        doc = "Quarterly figures"
        content = "summarize please\n\n" + doc
        manager.add_message(named.id, "user", content, origin="typed",
                            segments=[[0, 16, "typed"], [18, len(content), "document"]])
        manager.add_message(named.id, "assistant", "Done.", origin="assistant", context=[], lineage=[])
        assert manager.delete_last_message(named.id) is True
        manager.withdraw_source("document:" + hashlib.sha256(doc.encode("utf-8")).hexdigest())
        renamed = manager.create_conversation()
        _typed(manager, renamed.id, "Plan the garden layout")
        assert manager.rename_conversation(renamed.id, "Garden") is True
        _typed(manager, renamed.id, "And the shed")
        snapshot = manager._sync_snapshot(manager._get_connection(), named.id)
        assert snapshot["conversation"]["title_source"] == "manual"
        assert manager.apply_synced_conversation(snapshot) is True
        assert [(manager.get_conversation(c).title, manager.get_conversation(c).title_source)
                for c in (named.id, renamed.id)] == [("My budget notes", "manual"), ("Garden", "manual")]
        # A rename in another process between the rule's reading and its writing: the writing's own condition keeps
        # the user's title.
        raced = manager.create_conversation()
        _typed(manager, raced.id, "Plan the garden layout")
        _plain(db, raced.id, title="New conversation", title_source="default", title_basis="", title_turn="")
        marker = mod._encrypt

        def racing(value):
            if value == "Plan the garden layout":
                other = sqlite3.connect(db, timeout=0.5)
                try:
                    other.execute("UPDATE conversations SET title = 'E:Renamed elsewhere', title_source = 'manual' "
                                  "WHERE id = ?", (raced.id,))
                    other.commit()
                finally:
                    other.close()
            return marker(value)

        mod._encrypt = racing
        try:
            manager.tend_titles()
        finally:
            mod._encrypt = marker
        assert (_row(db, raced.id)["title"], _row(db, raced.id)["title_source"]) == ("E:Renamed elsewhere", "manual")
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT8 -- the titling turn deleted
# ---------------------------------------------------------------------------
def test_tt8_deleting_the_turn_a_title_came_from_titles_again(tmp_path):
    mod, _loaded, restore = _load(tmp_path)
    try:
        manager = mod.ConversationManager(db_path=tmp_path / "c.db", publish_to_sync=False)
        conv = manager.create_conversation()
        _typed(manager, conv.id, "Plan the garden layout")
        manager.add_message(conv.id, "assistant", "Sure.", origin="assistant", context=[], lineage=[])
        # The retry: the answer, then the turn, deleted -- and the same turn saved again.
        assert manager.delete_last_message(conv.id, role="assistant") and manager.delete_last_message(conv.id)
        assert manager.get_conversation(conv.id).title_source == "default"
        _typed(manager, conv.id, "Plan the garden layout")
        assert manager.get_conversation(conv.id).title == "Plan the garden layout"
        other = manager.create_conversation()
        manager.add_message(other.id, "user", "Old words", origin="legacy")
        _typed(manager, other.id, "Choose the tiles")
        assert manager.get_conversation(other.id).title == "Choose the tiles"
        assert manager.delete_last_message(other.id)
        assert (manager.get_conversation(other.id).title, manager.get_conversation(other.id).title_source) == \
            ("New conversation", "default"), "no turn gives a title any more: the placeholder"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT9 -- the titling turn's document withdrawn
# ---------------------------------------------------------------------------
def test_tt9_withdrawing_the_document_of_the_titling_turn_titles_again(tmp_path):
    mod, _loaded, restore = _load(tmp_path)
    try:
        manager = mod.ConversationManager(db_path=tmp_path / "c.db", publish_to_sync=False)
        conv = manager.create_conversation()
        doc = "Quarterly figures for the northern region"
        content = "summarize please\n\n" + doc
        manager.add_message(conv.id, "user", content, origin="typed",
                            segments=[[0, 16, "typed"], [18, len(content), "document"]])
        _typed(manager, conv.id, "Then draw a chart")
        assert manager.get_conversation(conv.id).title == "summarize please"
        assert manager.withdraw_source("document:" + hashlib.sha256(doc.encode("utf-8")).hexdigest()) == 1
        assert manager.get_conversation(conv.id).title == "Then draw a chart", "the withdrawn turn titles no more"
        # The same words from another turn: the title follows its new turn, and that turn's deletion takes it.
        db = tmp_path / "c.db"
        twin = manager.create_conversation()
        other = "Annual budget lines"
        pasted = "summarize please\n\n" + other
        manager.add_message(twin.id, "user", pasted, origin="typed",
                            segments=[[0, 16, "typed"], [18, len(pasted), "document"]])
        _typed(manager, twin.id, "summarize please")
        first = _row(db, twin.id)["title_turn"]
        manager.withdraw_source("document:" + hashlib.sha256(other.encode("utf-8")).hexdigest())
        assert manager.get_conversation(twin.id).title == "summarize please"
        assert _row(db, twin.id)["title_turn"] != first, "the title follows its new turn"
        assert manager.delete_last_message(twin.id)
        assert manager.get_conversation(twin.id).title == "New conversation", "its new turn deleted, the title goes"
        # One document beside two turns, withdrawn: every turn it lowered counts, the title's own among them.
        both = manager.create_conversation()
        shared = "Shared figures"
        first_turn = "summarize please\n\n" + shared
        manager.add_message(both.id, "user", first_turn, origin="typed",
                            segments=[[0, 16, "typed"], [18, len(first_turn), "document"]])
        second_turn = "and plot it\n\n" + shared
        manager.add_message(both.id, "user", second_turn, origin="typed",
                            segments=[[0, 11, "typed"], [13, len(second_turn), "document"]])
        assert manager.get_conversation(both.id).title == "summarize please"
        manager.withdraw_source("document:" + hashlib.sha256(shared.encode("utf-8")).hexdigest())
        assert manager.get_conversation(both.id).title == "New conversation", "both turns lowered, the title goes"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT10 -- an older file
# ---------------------------------------------------------------------------
_BEFORE_TITLES = """
CREATE TABLE conversations (
    id TEXT PRIMARY KEY, title TEXT DEFAULT 'New conversation', created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, model TEXT, task_type TEXT, preset TEXT, metadata TEXT DEFAULT '{}'
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, timestamp TEXT NOT NULL, token_estimate INTEGER DEFAULT 0, model TEXT,
    metadata TEXT DEFAULT '{}', origin TEXT NOT NULL DEFAULT 'legacy', segments TEXT NOT NULL DEFAULT '[]',
    context TEXT NOT NULL DEFAULT '["legacy"]', lineage TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
"""


def test_tt10_an_older_file_gains_the_columns_and_is_titled_once(tmp_path):
    mod, loaded, restore = _load(tmp_path)
    try:
        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.executescript(_BEFORE_TITLES)
        for conv_id, title in (("typed", "New conversation"), ("empty", "New conversation"),
                               ("documents", "New conversation"), ("named", "Notes")):
            conn.execute("INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, 't', 't')",
                         (conv_id, title))
        for conv_id, origin in (("typed", "typed"), ("named", "typed"), ("documents", "document")):
            conn.execute("INSERT INTO messages (conversation_id, role, content, timestamp, origin, context) "
                         "VALUES (?, 'user', 'E:Plan the garden layout', 't1', ?, '[]')", (conv_id, origin))
        conn.commit()
        conn.close()
        manager = mod.ConversationManager(db_path=db, publish_to_sync=False)
        found = {c.id: (c.title, c.title_source) for c in manager.list_conversations()}
        assert found == {"typed": ("New conversation", "default"), "empty": ("New conversation", "default"),
                         "documents": ("New conversation", "default"), "named": ("Notes", "legacy")}, \
            "the migration labels the titles; it writes none"
        assert _row(db, "named")["title"] == "Notes", "an older title is plaintext until the backend tends it"
        assert manager.tend_titles() == 2, "one title derived, one plaintext title encrypted"
        assert manager.tend_titles() == 0, "and nothing the second time: a document-only turn titles nothing"
        found = {c.id: (c.title, c.title_source) for c in manager.list_conversations()}
        assert found == {"typed": ("Plan the garden layout", "auto"), "empty": ("New conversation", "default"),
                         "documents": ("New conversation", "default"), "named": ("Notes", "legacy")}, found
        assert _row(db, "named")["title"] == "E:Notes" and _row(db, "typed")["title"] == "E:Plan the garden layout"
        _typed(manager, "named", "Choose the tiles")
        assert manager.get_conversation("named").title == "Notes", "a legacy title is never replaced"
        conn = sqlite3.connect(db)
        try:
            columns = {row[1]: row for row in conn.execute("PRAGMA table_info(conversations)")}
        finally:
            conn.close()
        assert columns["title_source"][3] == 1 and columns["title_source"][4] == "'legacy'"
        assert columns["title_basis"][3] == 1 and columns["title_basis"][4] == "''"
        assert columns["title_turn"][3] == 1 and columns["title_turn"][4] == "''"
        # The pass writes only the value it read: a rename between its reading and its writing keeps its own.
        raced_db = tmp_path / "raced.db"
        raced = mod.ConversationManager(db_path=raced_db, publish_to_sync=False)
        draft = raced.create_conversation(title="Draft")
        _plain(raced_db, draft.id, title="Draft")
        marker = mod._encrypt

        def racing(value):
            if value == "Draft":
                other = sqlite3.connect(raced_db, timeout=0.5)
                try:
                    other.execute("UPDATE conversations SET title = 'E:Renamed elsewhere' WHERE id = ?", (draft.id,))
                    other.commit()
                finally:
                    other.close()
            return marker(value)

        mod._encrypt = racing
        try:
            assert raced.tend_titles() == 0
        finally:
            mod._encrypt = marker
        assert _row(raced_db, draft.id)["title"] == "E:Renamed elsewhere", "a write in between keeps its own"
        # A basis still in plaintext is encrypted too; with field encryption off, nothing at all is written.
        based_db = tmp_path / "based.db"
        based = mod.ConversationManager(db_path=based_db, publish_to_sync=False)
        notes = based.create_conversation(title="Notes")
        _plain(based_db, notes.id, title="Notes", title_basis="abc")
        mod._encrypt = lambda value: value
        try:
            assert based.tend_titles() == 0 and _row(based_db, notes.id)["title"] == "Notes", "encryption off"
        finally:
            mod._encrypt = marker
        assert based.tend_titles() == 2
        assert (_row(based_db, notes.id)["title"], _row(based_db, notes.id)["title_basis"]) == ("E:Notes", "E:abc")
        # A placeholder the rule settled -- every turn read, none titles -- is left be by the next start, and read
        # again once the settings change.
        settled_db = tmp_path / "settled.db"
        settled = mod.ConversationManager(db_path=settled_db, publish_to_sync=False)
        quiet = settled.create_conversation()
        _typed(settled, quiet.id, "```\nls -la\n```")
        assert _row(settled_db, quiet.id)["title_turn"].startswith("-")
        titles = loaded[_TITLES]
        reads = []
        rule = titles.first_typed_source
        titles.first_typed_source = lambda *args, **kwargs: reads.append(1) or rule(*args, **kwargs)
        kept = dict(mod._titles_read)
        try:
            assert settled.tend_titles() == 0 and reads == [], "a settled placeholder is not read again"
            mod._titles_read.update(done=True, config=_config(titles, max_chars=59))
            assert settled.tend_titles() == 0 and reads == [1], "other settings: it is read again"
            reads.clear()
            shipped = yaml.safe_load(_TITLES_YAML.read_text(encoding="utf-8"))["derive"]["password_names"]
            mod._titles_read.update(config=_config(titles, max_chars=59, password_names=shipped + ["zzyzx"]))
            assert settled.tend_titles() == 0 and reads == [1], "another name: it is read again"
            reads.clear()
            mod._RULE_PRINT[:] = ["another rule"]
            assert settled.tend_titles() == 0 and reads == [1], "another rule: it is read again"
        finally:
            titles.first_typed_source = rule
            mod._titles_read.update(kept)
            mod._RULE_PRINT[:] = []
        # A title drawn under a list that did not name what its turn names is drawn again, at the start and at the
        # next turn: the name the list has since learned takes the title it let through.
        older = [name for name in shipped if name != "*pswd*"]
        for path in ("start", "turn"):
            drawn_db = tmp_path / f"drawn_{path}.db"
            drawn = mod.ConversationManager(db_path=drawn_db, publish_to_sync=False)
            mod._titles_read.update(done=True, config=_config(titles, password_names=older))
            leaked = drawn.create_conversation()
            _typed(drawn, leaked.id, "my pswd is sunflower")
            assert drawn.get_conversation(leaked.id).title == "my pswd is sunflower"
            mod._titles_read.update(kept)
            if path == "start":
                drawn.tend_titles()
            else:
                _typed(drawn, leaked.id, "and the shed")
            assert drawn.get_conversation(leaked.id).title == "New conversation", path
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT11 -- encrypted at rest
# ---------------------------------------------------------------------------
def test_tt11_titles_and_bases_are_encrypted_at_rest(tmp_path):
    mod, _loaded, restore = _load(tmp_path)
    try:
        db = tmp_path / "c.db"
        manager = mod.ConversationManager(db_path=db, publish_to_sync=False)
        auto = manager.create_conversation()
        _typed(manager, auto.id, "Plan the garden layout")
        manual = manager.create_conversation(title="Mine")
        renamed = manager.create_conversation()
        manager.rename_conversation(renamed.id, "Renamed")
        placeholder = manager.create_conversation()
        rows = {c: _row(db, c) for c in (auto.id, manual.id, renamed.id, placeholder.id)}
        assert [rows[c]["title"] for c in (auto.id, manual.id, renamed.id, placeholder.id)] == \
            ["E:Plan the garden layout", "E:Mine", "E:Renamed", "New conversation"]
        assert rows[auto.id]["title_basis"].startswith("E:") and len(rows[auto.id]["title_basis"]) == 2 + 64
        conn = sqlite3.connect(db)
        try:
            conn.execute("UPDATE conversations SET title = 'ENC2:c3RpbGwgZW5jcnlwdGVk' WHERE id = ?", (manual.id,))
            conn.commit()
        finally:
            conn.close()
        assert manager.get_conversation(manual.id).title == "", "a title still encrypted is never served as such"
        assert manager.get_conversation(auto.id).title == "Plan the garden layout"
        # Keyless, a listing tries one sealed title, not every one: one failure said, not one per title.
        tried = []
        marker = mod._decrypt
        keyed = mod._key_loaded
        mod._decrypt = lambda value: tried.append(value) or value
        mod._key_loaded = lambda: False
        listed = {c.id: c.title for c in manager.list_conversations()}
        mod._decrypt, mod._key_loaded = marker, keyed
        assert listed[auto.id] == "" and len([v for v in tried if str(v).startswith(("E:", "ENC2:"))]) == 1, tried
        # With the key, a title that does not decrypt is that title's alone -- newest of all, it blanks no other --,
        # and so is one whose decryption fails outright: the listing, a read and a search see every other.
        _plain(db, manual.id, updated_at="9999")
        listed = {c.id: c.title for c in manager.list_conversations()}
        assert (listed[manual.id], listed[auto.id], listed[renamed.id]) == ("", "Plan the garden layout", "Renamed")
        # Whether the key is loaded is the encryption manager's to say, through the name the store imports.
        for has_key in (True, False):
            folder = tmp_path / f"key{has_key}"
            folder.mkdir()
            keyed_mod, _keyed, keyed_restore = _load(folder, encryption=_encryption(has_key))
            try:
                assert keyed_mod._key_loaded() is has_key
            finally:
                keyed_restore()

        def raising(value):
            if isinstance(value, str) and value.startswith("ENC2:"):
                raise ValueError("not a ciphertext")
            return marker(value)

        mod._decrypt = raising
        try:
            listed = {c.id: c.title for c in manager.list_conversations()}
            assert (listed[manual.id], listed[auto.id]) == ("", "Plan the garden layout"), listed
            assert manager.get_conversation(manual.id).title == ""
            assert [c.id for c in manager.search_conversations("garden")] == [auto.id]
        finally:
            mod._decrypt = marker
    finally:
        restore()


def test_tt11b_a_sync_apply_reencrypts_the_title(tmp_path):
    mod, _loaded, restore = _load(tmp_path)
    try:
        db = tmp_path / "c.db"
        manager = mod.ConversationManager(db_path=db)
        payload = {"user_id": "u", "conversation": {
            "id": "c1", "title": "Hello", "created_at": "t0", "updated_at": "t1", "model": "m", "task_type": None,
            "preset": None, "metadata": {"k": "v"},
            "messages": [{"role": r, "content": c, "timestamp": f"t{i}", "token_estimate": 1, "model": None,
                          "metadata": {}} for i, (r, c) in enumerate((("user", "hi"), ("assistant", "yo")))]}}
        assert manager.apply_synced_conversation(payload) is True
        conn = sqlite3.connect(db)
        try:
            title = conn.execute("SELECT title FROM conversations WHERE id = ?", ("c1",)).fetchone()
            rows = conn.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
                                ("c1",)).fetchall()
        finally:
            conn.close()
        assert title is not None and title[0] == "E:Hello"
        assert [r[0] for r in rows] == ["user", "assistant"]
        assert rows[0][1] == "E:hi" and rows[1][1] == "E:yo"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT12 -- no title and no search word in the log
# ---------------------------------------------------------------------------
def test_tt12_no_title_and_no_search_word_reaches_the_log(tmp_path, caplog):
    mod, _loaded, restore = _load(tmp_path)
    try:
        manager = mod.ConversationManager(db_path=tmp_path / "c.db", publish_to_sync=False)
        with caplog.at_level(logging.DEBUG):
            named = manager.create_conversation(title="Zanzibarite notes")
            manager.rename_conversation(named.id, "Quixotical plans")
            conv = manager.create_conversation()
            _typed(manager, conv.id, "Xylophonic gardening")
            manager.search_conversations("Pterodactylish")
            real = manager._get_connection

            def broken():
                connection = real()
                connection.close()
                return connection

            manager._get_connection = broken
            assert manager.search_conversations("Pterodactylish") == []
        text = caplog.text
        assert "Error searching" in text, "the failed search is said"
        assert not any(word in text for word in ("Zanzibarite", "Quixotical", "Xylophonic", "Pterodactylish")), text
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT13 -- search by an encrypted title
# ---------------------------------------------------------------------------
def test_tt13_a_search_finds_a_conversation_by_its_encrypted_title(tmp_path):
    mod, _loaded, restore = _load(tmp_path, opaque=True)
    try:
        db = tmp_path / "c.db"
        manager = mod.ConversationManager(db_path=db, publish_to_sync=False)
        conv = manager.create_conversation()
        _typed(manager, conv.id, "Comment calculer une ANOVA en R?")
        other = manager.create_conversation(title="Unrelated")
        assert "Unrelated" not in _row(db, other.id)["title"] and "ANOVA" not in _row(db, conv.id)["title"]
        assert [c.id for c in manager.search_conversations("anova")] == [conv.id], "by the title, decrypted"
        assert [c.id for c in manager.search_conversations("UNRELATED")] == [other.id], "folded: case aside"
        # More matches than the limit: the newest, newest first; a match older than the limit's worth of newer
        # conversations is still read, within scan_limit.
        manager2 = mod.ConversationManager(db_path=tmp_path / "many.db", publish_to_sync=False)
        oldest = manager2.create_conversation(title="Orchard ledger")
        ids = [manager2.create_conversation(title=f"Garden plan {index}").id for index in range(25)]
        newer = [manager2.create_conversation(title=f"Kitchen {index}").id for index in range(25)]
        conn = sqlite3.connect(tmp_path / "many.db")
        try:
            for rank, conv_id in enumerate([oldest.id] + ids + newer):
                conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                             (f"2026-01-01T00:{rank:04d}", conv_id))
            conn.commit()
        finally:
            conn.close()
        assert [c.id for c in manager2.search_conversations("garden plan", limit=20)] == ids[::-1][:20]
        assert [c.id for c in manager2.search_conversations("orchard", limit=20)] == [oldest.id]
        mod._titles_read.update(done=True, config=None)
        assert [c.id for c in manager.search_conversations("Unrelated")] == [other.id], "without the configuration too"
        assert [c.id for c in manager2.search_conversations("orchard", limit=20)] == [oldest.id], "every title read"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT14 -- sync: the source travels; an apply freezes what it cannot vouch for
# ---------------------------------------------------------------------------
def test_tt14_a_sync_apply_keeps_a_manual_title_and_freezes_any_other(tmp_path):
    mod, loaded, restore = _load(tmp_path)
    try:
        calls = _spy(mod)
        manager = mod.ConversationManager(db_path=tmp_path / "c.db")
        conv = manager.create_conversation()
        _typed(manager, conv.id, "Plan the garden layout")
        assert calls[-1][1]["conversation"]["title_source"] == "auto"
        receiver = mod.ConversationManager(db_path=tmp_path / "peer.db", publish_to_sync=False)
        cases = {"auto": "legacy", None: "legacy", "manual": "manual", "refined": "legacy", "weird": "legacy"}
        for index, (sent, kept) in enumerate(cases.items()):
            body = {"id": f"p{index}", "title": "Peer title", "created_at": "t", "updated_at": "t", "messages": []}
            if sent is not None:
                body["title_source"] = sent
            assert receiver.apply_synced_conversation({"conversation": body}) is True
            got = receiver.get_conversation(f"p{index}")
            assert (got.title, got.title_source) == ("Peer title", kept), sent
        assert receiver.apply_synced_conversation({"conversation": {
            "id": "p9", "title": "New conversation", "title_source": "auto", "messages": []}}) is True
        assert receiver.get_conversation("p9").title_source == "default"
        frozen = receiver.get_conversation("p0")
        _typed(receiver, frozen.id, "Words typed here later")
        assert receiver.get_conversation(frozen.id).title == "Peer title", "a frozen title stays"
        # A title its sender could not read -- sent empty, its key absent -- or one that reads as a ciphertext is
        # no title: this device's stays, frozen when it was automatic, while a received turn holds its words; with
        # none of them, the placeholder.
        mine = receiver.create_conversation(title="Mine here")
        derived = receiver.create_conversation()
        _typed(receiver, derived.id, "Plan the garden layout")
        gone = receiver.create_conversation()
        _typed(receiver, gone.id, "Plan the garden layout")
        kept_turn = [{"role": "user", "content": "Plan the garden layout", "timestamp": "t1", "token_estimate": 1,
                      "model": None, "metadata": {}, "origin": "typed", "segments": [], "context": [], "lineage": []}]
        for conv_id, title, sent, messages in ((mine.id, "", "manual", []), (mine.id, "ENC2:YWJj", "manual", []),
                                               (derived.id, "", "auto", kept_turn), (gone.id, "", "legacy", [])):
            assert receiver.apply_synced_conversation({"conversation": {
                "id": conv_id, "title": title, "title_source": sent, "messages": messages}}) is True
        got = receiver.get_conversation(mine.id)
        assert (got.title, got.title_source) == ("Mine here", "manual"), "a keyless peer erases no title"
        got = receiver.get_conversation(derived.id)
        assert (got.title, got.title_source) == ("Plan the garden layout", "legacy"), "its words still received"
        got = receiver.get_conversation(gone.id)
        assert (got.title, got.title_source) == ("New conversation", "default"), "none of its words received"
        # A sender without its key sends its turns still sealed: whether they hold the words cannot be told, and the
        # title stays, frozen. An answer that repeats the words holds nothing; words typed beside a pasted part hold
        # them; a withdrawn turn holds nothing.
        def turn(content, role="user", segments=(), context=()):
            return {"role": role, "content": content, "timestamp": "t1", "token_estimate": 1, "model": None,
                    "metadata": {}, "origin": "typed", "segments": list(segments), "context": list(context),
                    "lineage": []}

        beside = "Plan the garden layout\n\nQuarterly"
        for messages, kept_as in (([turn("ENC2:c2VhbGVk")], ("Plan the garden layout", "legacy")),
                                  ([turn("ENC2:c2VhbGVk", context=["withdrawn"])], ("New conversation", "default")),
                                  (None, ("New conversation", "default")),
                                  ([turn("Plan the garden layout", role="assistant")], ("New conversation", "default")),
                                  ([turn(beside, segments=[[0, 22, "typed"], [24, 33, "document"]])],
                                   ("Plan the garden layout", "legacy")),
                                  ([turn("Plan the garden layout", context=["withdrawn"])],
                                   ("New conversation", "default"))):
            local = receiver.create_conversation()
            _typed(receiver, local.id, "Plan the garden layout")
            body = {"id": local.id, "title": "", "title_source": "auto"}
            if messages is not None:
                body["messages"] = messages
            assert receiver.apply_synced_conversation({"conversation": body}) is True
            got = receiver.get_conversation(local.id)
            assert (got.title, got.title_source) == kept_as, messages
        # A title whose digest does not decrypt here cannot be told from the received turns: it stays, frozen.
        unread = receiver.create_conversation()
        _typed(receiver, unread.id, "Plan the garden layout")
        _plain(tmp_path / "peer.db", unread.id, title_basis="ENC2:YmFk")
        assert receiver.apply_synced_conversation({"conversation": {
            "id": unread.id, "title": "", "title_source": "auto", "messages": []}}) is True
        got = receiver.get_conversation(unread.id)
        assert (got.title, got.title_source) == ("Plan the garden layout", "legacy")
        # A received title is one line within the manual bound, cut there whole: never inside a combining sequence,
        # never left as blanks alone.
        bound = _config(loaded[_TITLES]).manual_max_chars
        sent = " Peer\nnotes " + "x" * (2 * bound)
        assert receiver.apply_synced_conversation({"conversation": {
            "id": "long", "title": sent, "title_source": "manual", "messages": []}}) is True
        assert receiver.get_conversation("long").title == ("Peer notes " + "x" * (2 * bound))[:bound]
        marked = "y" * (bound - 1) + "q" + chr(0x301) + "rest"
        assert receiver.apply_synced_conversation({"conversation": {
            "id": "marked", "title": marked, "title_source": "manual", "messages": []}}) is True
        assert receiver.get_conversation("marked").title == "y" * (bound - 1)
        assert receiver.apply_synced_conversation({"conversation": {
            "id": "blank", "title": chr(0x3164) * bound + " words", "title_source": "manual", "messages": []}}) is True
        assert receiver.get_conversation("blank").title_source == "default", "a cut that leaves blanks is no title"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT21 -- a typed title is one line within its bound
# ---------------------------------------------------------------------------
def _route_body(name):
    tree = ast.parse((REPO / "opti_oignon" / "api" / "routes_conversations.py").read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_tt21_a_typed_title_is_one_line_within_its_bound(tmp_path):
    from fastapi import HTTPException

    mod, loaded, restore = _load(tmp_path)
    try:
        db = tmp_path / "c.db"
        manager = mod.ConversationManager(db_path=db, publish_to_sync=False)
        bound = _config(loaded[_TITLES]).manual_max_chars
        blank = chr(0x2800) * 2
        assert [manager.fold_manual_title(t) for t in ("", "   ", "\n\t", "x" * (bound + 1), blank, chr(0x3164),
                                                       chr(0x301) * 2, "E:not a title")] == [None] * 8
        assert manager.fold_manual_title("  Budget\nnotes " + chr(0x2028) + " 2026 ") == "Budget notes 2026"
        assert manager.fold_manual_title("x" * bound) == "x" * bound
        conv = manager.create_conversation()
        routes, schemas, close = _routes(manager)
        try:
            answer = routes.rename_conversation(conv.id, schemas.ConversationRename(title="  Budget\n notes  "))
            assert (answer["title"], answer["title_source"]) == ("Budget notes", "manual")
            assert _row(db, conv.id)["title"] == "E:Budget notes"
            for refused in ("", "   ", blank, "x" * (bound + 1)):
                with pytest.raises(HTTPException) as raised:
                    routes.rename_conversation(conv.id, schemas.ConversationRename(title=refused))
                assert raised.value.status_code == 422
            assert _row(db, conv.id)["title"] == "E:Budget notes", "a refused rename writes nothing"
            made = routes.create_conversation(schemas.ConversationCreate(title=" Mine\nnotes "))
            assert (made["title"], made["title_source"]) == ("Mine notes", "manual")
            untitled = routes.create_conversation(schemas.ConversationCreate(title="   "))
            assert untitled["title_source"] == "default"
            with pytest.raises(HTTPException) as raised:
                routes.create_conversation(schemas.ConversationCreate(title=blank))
            assert raised.value.status_code == 422
        finally:
            close()
        placeholder = manager.create_conversation(title="New conversation")
        assert placeholder.title_source == "default", "the placeholder's words are no title"
        named = manager.create_conversation(title="  Budget\nnotes  ")
        assert _row(db, named.id)["title"] == "E:Budget notes", "the store folds a title it is created with"
        direct = manager.create_conversation()
        assert manager.rename_conversation(direct.id, "  a\nb  ") is True and _row(db, direct.id)["title"] == "E:a b", \
            "the store folds what a caller hands it, the route aside"
        assert manager.rename_conversation(conv.id, blank) is False and _row(db, conv.id)["title"] == "E:Budget notes"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT23 -- the wiring on the real package's shape
# ---------------------------------------------------------------------------
def _module(rel):
    return ast.parse((REPO / rel).read_text(encoding="utf-8"))


def test_tt23_the_wiring_on_the_real_package_shape(tmp_path):
    conversation = _module("opti_oignon/conversation.py")
    assert any(isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
               and [alias.name for alias in node.names] == ["conversation_titles"]
               for node in ast.walk(conversation)), "conversation.py imports conversation_titles by its real name"
    encryption = _module("opti_oignon/encryption.py")
    prefixes = {target.id: node.value.value for node in encryption.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)
                and target.id in ("_ENCRYPTED_PREFIX_V1", "_ENCRYPTED_PREFIX_V2")}
    fallback = next(node for node in ast.walk(conversation) if isinstance(node, ast.ExceptHandler)
                    and any(isinstance(s, ast.Assign) and getattr(s.targets[0], "id", "") == "_ENCRYPTED_PREFIXES"
                            for s in node.body))
    value = next(s.value for s in fallback.body if isinstance(s, ast.Assign))
    assert sorted(element.value for element in value.elts) == sorted(prefixes.values()) and len(prefixes) == 2
    # The key's presence is read through names the real encryption module has.
    manager_class = next(node for node in encryption.body if isinstance(node, ast.ClassDef)
                         and node.name == "EncryptionManager")
    assert any(isinstance(node, ast.FunctionDef) and node.name == "get_encryption_manager" for node in encryption.body)
    assert any(isinstance(node, ast.FunctionDef) and node.name == "has_key"
               and any(getattr(decorator, "id", "") == "property" for decorator in node.decorator_list)
               for node in manager_class.body), "has_key is a property the store reads without calling"
    mod, loaded, restore = _load(tmp_path)
    try:
        assert mod._PLACEHOLDER == loaded[_TITLES].PLACEHOLDER
    finally:
        restore()
    for rel in ("opti_oignon/api/routes_chat.py", "opti_oignon/cli/session.py"):
        tree = _module(rel)
        literals = [node for node in ast.walk(tree) if isinstance(node, ast.Constant)
                    and node.value == "New conversation"]
        assert literals == [], f"{rel} names the placeholder: the store gives it"
    chat = _module("opti_oignon/api/routes_chat.py")
    creates = [node for node in ast.walk(chat) if isinstance(node, ast.Call)
               and getattr(node.func, "attr", None) == "create_conversation"]
    assert creates and all(not node.args and not node.keywords for node in creates), "the socket names no title"
    app = _module("opti_oignon/api/app.py")
    lifespan = next(node for node in app.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan")
    calls = [node for node in ast.walk(lifespan) if isinstance(node, ast.Call)
             and getattr(node.func, "attr", None) == "tend_titles"]
    yields = [node for node in ast.walk(lifespan) if isinstance(node, ast.Yield)]
    assert calls and yields and calls[0].lineno < yields[0].lineno, "the backend titles the placeholders as it starts"


# ---------------------------------------------------------------------------
# TT25 -- titles.yaml
# ---------------------------------------------------------------------------
def test_tt25_titles_yaml_is_read_exactly_or_turns_titles_off_by_name(tmp_path, caplog):
    mod, loaded, restore = _load(tmp_path)
    try:
        titles = loaded[_TITLES]
        shipped = yaml.safe_load(_TITLES_YAML.read_text(encoding="utf-8"))
        assert set(shipped) == set(titles._SECTIONS) and set(shipped["derive"]) == set(titles._DERIVE)
        assert set(shipped["manual"]) == set(titles._MANUAL) and set(shipped["search"]) == set(titles._SEARCH)
        cfg = titles.read_config(shipped)
        assert cfg.enabled is True and cfg.derive.max_chars == shipped["derive"]["max_chars"]
        for broken, word in ((dict(shipped, extra=1), "extra"),
                             ({**shipped, "derive": {**shipped["derive"], "max_chars": True}}, "max_chars"),
                             ({k: v for k, v in shipped.items() if k != "search"}, "search")):
            with pytest.raises(titles.TitlesConfigError, match=word):
                titles.read_config(broken)
        # A value out of range, a malformed password entry: refused, by name.
        for section, shape in (("derive", titles._DERIVE), ("manual", titles._MANUAL), ("search", titles._SEARCH)):
            for key, kind in shape.items():
                if kind is int:
                    low = 0 if key in titles._MAY_BE_ZERO else 1
                    with pytest.raises(titles.TitlesConfigError, match=f"{key} is out of range"):
                        titles.read_config({**shipped, section: {**shipped[section], key: low - 1}})
        for key, value in (("password_names", ["--"]), ("password_names", ["pa*ss"]), ("password_names", ["*"]),
                           ("password_names", ["+"]), ("password_names", [""]), ("password_names", [5]),
                           ("password_lookalikes", {0: "o"}), ("password_lookalikes", {"a": "o"}),
                           ("password_lookalikes", {"0": "oo"})):
            with pytest.raises(titles.TitlesConfigError, match=key):
                titles.read_config({**shipped, "derive": {**shipped["derive"], key: value}})
        bad = tmp_path / "titles.yaml"
        bad.write_text("enabled: true\nderive: {max_chars: 60}\nsurprise: 1\n", encoding="utf-8")
        mod._TITLES_PATH = bad
        with caplog.at_level(logging.WARNING):
            assert mod._titles_config() is None
        assert "Automatic titles off" in caplog.text and "surprise" in caplog.text
        manager = mod.ConversationManager(db_path=tmp_path / "c.db", publish_to_sync=False)
        conv = manager.create_conversation()
        _typed(manager, conv.id, "Plan the garden layout")
        assert manager.get_conversation(conv.id).title == "New conversation", "titles off: the placeholder"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT26 -- the CLI hands no title
# ---------------------------------------------------------------------------
def test_tt26_the_cli_session_hands_no_title():
    tree = _module("opti_oignon/cli/session.py")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "attr", None) == "_new_conversation"]
    assert calls and all(isinstance(call.args[0], ast.Constant) and call.args[0].value is None for call in calls)


# ---------------------------------------------------------------------------
# TT27 -- the verb
# ---------------------------------------------------------------------------
def test_tt27_the_verb_gives_a_manual_title_back_to_the_rule(tmp_path):
    from fastapi import HTTPException

    mod, _loaded, restore = _load(tmp_path)
    try:
        calls = _spy(mod)
        manager = mod.ConversationManager(db_path=tmp_path / "c.db")
        conv = manager.create_conversation(title="Mine")
        _typed(manager, conv.id, "Plan the garden layout")
        before = len(calls)
        routes, _schemas, close = _routes(manager)
        try:
            assert routes._conv_to_summary(manager.get_conversation(conv.id))["title_source"] == "manual"
            answer = routes.use_automatic_title(conv.id)
            assert (answer["title"], answer["title_source"]) == ("Plan the garden layout", "auto"), \
                "the route answers the title the verb wrote"
            with pytest.raises(HTTPException) as raised:
                routes.use_automatic_title("absent")
            assert raised.value.status_code == 404
        finally:
            close()
        assert len(calls) == before + 1, "one publish"
        got = manager.get_conversation(conv.id)
        assert (got.title, got.title_source) == ("Plan the garden layout", "auto")
        assert manager.use_automatic_title("absent") is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT28 -- a turn that cannot be read for a title is still saved
# ---------------------------------------------------------------------------
def test_tt28_a_turn_that_cannot_be_read_for_a_title_is_still_saved(tmp_path, caplog):
    mod, loaded, restore = _load(tmp_path)
    try:
        titles = loaded[_TITLES]
        db = tmp_path / "c.db"
        manager = mod.ConversationManager(db_path=db, publish_to_sync=False)
        conv = manager.create_conversation()
        marker, sealing = mod._decrypt, mod._encrypt

        def failing(value):
            if isinstance(value, str) and "Unreadable" in value:
                raise ValueError("cannot decrypt")
            return marker(value)

        # A decryption that fails outright: the turn is saved, unknown to the rule, and the title stays.
        mod._decrypt = failing
        with caplog.at_level(logging.WARNING):
            saved = _typed(manager, conv.id, "Unreadable words here")
        assert saved is not None and saved.content == "Unreadable words here"
        assert "Unreadable" not in caplog.text
        mod._decrypt = marker
        assert manager.get_conversation(conv.id).title_source == "default"
        # A rule that fails: the turn is saved, the failure said by the conversation's id, never with its words.
        rule = titles.first_typed_source

        def broken(*args, **kwargs):
            raise RuntimeError("the rule broke")

        titles.first_typed_source = broken
        try:
            with caplog.at_level(logging.WARNING):
                saved = _typed(manager, conv.id, "Words of a failing rule")
        finally:
            titles.first_typed_source = rule
        assert saved is not None and "not derived" in caplog.text and "failing rule" not in caplog.text
        assert manager.get_conversation(conv.id).title_source == "default"
        # The real decrypt hands a ciphertext back when the key is absent or another one: a turn that does not
        # decrypt is unknown. A turn added, deleted or lowered keyless that is not the title's own leaves the title
        # as it was.
        titled = manager.create_conversation()
        _typed(manager, titled.id, "Plan the garden layout")
        manager.add_message(titled.id, "assistant", "Sure.", origin="assistant", context=[], lineage=[])
        mod._decrypt = lambda value: value
        mod._encrypt = lambda value: value
        caplog.clear()
        _typed(manager, titled.id, "Order the paint")
        assert _row(db, titled.id)["title"] == "E:Plan the garden layout", "added keyless: it stays"
        assert manager.delete_last_message(titled.id)
        assert _row(db, titled.id)["title"] == "E:Plan the garden layout", "another turn deleted keyless: it stays"
        doc = "Quarterly figures"
        content = "summarize please\n\n" + doc
        manager.add_message(titled.id, "user", content, origin="typed",
                            segments=[[0, 16, "typed"], [18, len(content), "document"]])
        manager.withdraw_source("document:" + hashlib.sha256(doc.encode("utf-8")).hexdigest())
        assert _row(db, titled.id)["title"] == "E:Plan the garden layout", "another turn lowered keyless: it stays"
        assert "not derived" not in caplog.text, "a turn that does not decrypt is unknown, not a failure"
        mod._decrypt, mod._encrypt = marker, sealing
        # The title's own turn deleted keyless, a sealed turn before it: the title may hold the gone turn's words,
        # and the placeholder comes back.
        own = manager.create_conversation()
        _typed(manager, own.id, "```\nls -la\n```")
        _typed(manager, own.id, "Plan the garden layout")
        assert _row(db, own.id)["title_turn"] != "" and manager.get_conversation(own.id).title_source == "auto"
        mod._decrypt = lambda value: value
        mod._encrypt = lambda value: value
        assert manager.delete_last_message(own.id)
        mod._decrypt, mod._encrypt = marker, sealing
        got = manager.get_conversation(own.id)
        assert (got.title, got.title_source) == ("New conversation", "default"), "its own turn deleted keyless"
        assert _row(db, own.id)["title_turn"] == ""
        # The title's own turn lowered keyless: the placeholder too.
        lowered = manager.create_conversation()
        survey = "Survey answers"
        pasted = "summarize please\n\n" + survey
        manager.add_message(lowered.id, "user", pasted, origin="typed",
                            segments=[[0, 16, "typed"], [18, len(pasted), "document"]])
        assert manager.get_conversation(lowered.id).title == "summarize please"
        mod._decrypt = lambda value: value
        mod._encrypt = lambda value: value
        manager.withdraw_source("document:" + hashlib.sha256(survey.encode("utf-8")).hexdigest())
        mod._decrypt, mod._encrypt = marker, sealing
        assert manager.get_conversation(lowered.id).title_source == "default", "its own turn lowered keyless"
        # A turn the store cannot decrypt may name a password, whatever is stored beside it: nothing after it titles
        # until it reads again, and the backend's start with the key titles the conversation then.
        judged = manager.create_conversation()
        manager.add_message(judged.id, "user", "Annual budget lines for the region", origin="document")
        mod._decrypt = lambda value: value if str(value).startswith("E:Annual") else marker(value)
        _typed(manager, judged.id, "Order the paint")
        mod._decrypt = marker
        assert manager.get_conversation(judged.id).title == "New conversation", "a sealed turn before it"
        assert manager.tend_titles() >= 2 and manager.get_conversation(judged.id).title == "Order the paint"
        assert manager.get_conversation(conv.id).title == "Unreadable words here", "read with the key, it titles"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT29 -- the chats index offers the automatic title for a manual one
# ---------------------------------------------------------------------------
def test_tt29_the_chats_index_offers_the_automatic_title_for_a_manual_one():
    front = REPO / "frontend" / "src"
    page = (front / "routes" / "(app)" / "(use)" / "chat" / "+page.svelte").read_text(encoding="utf-8")
    store = (front / "lib" / "stores" / "conversations.ts").read_text(encoding="utf-8")
    client = (front / "lib" / "api" / "conversations.ts").read_text(encoding="utf-8")
    assert "label: 'Use automatic title'" in page and "chat.title_source === 'manual'" in page
    assert "items={actionsFor(chat)}" in page and "automaticTitleConv(chat.id)" in page
    verb = page[page.index("async function autoTitle"):]
    verb = verb[: verb.index("\n\t}\n")]
    assert "summary.title" in verb and "summary.title_source" in verb, "the row takes the title the verb wrote"
    renamed = page[page.index("async function rename()"):]
    assert "const summary = await renameConv(" in renamed and "title: summary.title" in renamed, \
        "a rename shows the title the server kept"
    assert "export async function automaticTitleConv" in store and "api.automaticTitle(id)" in store
    listed = store[store.index("export async function renameConv"):]
    listed = listed[: listed.index("\n}\n")]
    assert "title: summary.title" in listed and "title_source: summary.title_source" in listed, \
        "the store's list takes the title the server kept"
    assert "`/api/conversations/${id}/title/auto`" in client and "apiPost<ConversationSummary>" in client


# ---------------------------------------------------------------------------
# TT30 -- the legacy history import
# ---------------------------------------------------------------------------
def test_tt30_the_legacy_history_import_is_encrypted_and_never_retitled(tmp_path):
    mod, _loaded, restore = _load(tmp_path)
    try:
        history = tmp_path / "history"
        history.mkdir()
        entry = {"id": "h1", "question": "Plan the garden layout", "response": "Start with the paths.",
                 "timestamp": "2026-01-01T00:00:00", "model": "m"}
        (history / "history_2026.json").write_text(json.dumps([entry]), encoding="utf-8")
        db = tmp_path / "c.db"
        manager = mod.ConversationManager(db_path=db, publish_to_sync=False)
        assert manager.migrate_json_history(history) == (1, 0)
        conn = sqlite3.connect(db)
        try:
            (conv_id, title, source), = conn.execute("SELECT id, title, title_source FROM conversations").fetchall()
            contents = [row[0] for row in conn.execute("SELECT content FROM messages ORDER BY id")]
        finally:
            conn.close()
        assert (title, source) == ("E:Plan the garden layout", "legacy")
        assert contents == ["E:Plan the garden layout", "E:Start with the paths."], "the turns at rest, encrypted"
        _typed(manager, conv_id, "Choose the tiles")
        assert manager.get_conversation(conv_id).title == "Plan the garden layout", "a legacy title stays"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT31 -- the Markdown export's heading
# ---------------------------------------------------------------------------
def test_tt31_the_markdown_export_says_its_title_on_one_line(tmp_path):
    mod, _loaded, restore = _load(tmp_path)
    try:
        db = tmp_path / "c.db"
        manager = mod.ConversationManager(db_path=db, publish_to_sync=False)
        conv = manager.create_conversation()
        _typed(manager, conv.id, "Plan the garden layout")
        conn = sqlite3.connect(db)
        try:
            conn.execute("UPDATE conversations SET title = 'Two\nlines', title_source = 'legacy' WHERE id = ?",
                         (conv.id,))
            conn.commit()
        finally:
            conn.close()
        assert manager.export_conversation_markdown(conv.id).split("\n")[0] == "# Two lines"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT32 -- the access log
# ---------------------------------------------------------------------------
def test_tt32_uvicorns_lines_say_a_path_never_its_query():
    app = _module("opti_oignon/api/app.py")
    wanted = {"_PathWithoutQuery", "_install_log_filters"}
    found = [node for node in app.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    assert {node.name for node in found} == wanted
    space = {"logging": logging}
    exec(compile(ast.Module(body=found, type_ignores=[]), "app.py", "exec"), space)
    loggers = {name: logging.getLogger(name) for name in ("uvicorn.access", "uvicorn.error")}
    kept = {name: (log.level, list(log.filters)) for name, log in loggers.items()}
    lines = []

    class Keep(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler = Keep()
    try:
        for log in loggers.values():
            log.setLevel(logging.INFO)
            log.addHandler(handler)
        space["_install_log_filters"]()
        space["_install_log_filters"]()
        assert [sum(isinstance(f, space["_PathWithoutQuery"]) for f in log.filters) for log in loggers.values()] == \
            [1, 1], "on both loggers, once each"
        access, error = loggers["uvicorn.access"], loggers["uvicorn.error"]
        access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5", "GET", "/api/conversations?q=Pterodactylish words",
                    "1.1", 200)
        access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5", "GET", "/api/health", "1.1", 200)
        error.info('%s - "WebSocket %s" [accepted]', "127.0.0.1:6", "/ws/chat?token=Pterodactylish")
        error.info('%s - "WebSocket %s" 403', "127.0.0.1:6", "/ws/chat?token=Pterodactylish")
        error.info('%s - "WebSocket %s" %d', "127.0.0.1:6", "/ws/chat?token=Pterodactylish", 500)
        error.info("Started server process [%d]", 42)
    finally:
        for name, log in loggers.items():
            log.removeHandler(handler)
            log.setLevel(kept[name][0])
            log.filters[:] = kept[name][1]
    assert lines == ['127.0.0.1:5 - "GET /api/conversations?(query) HTTP/1.1" 200',
                     '127.0.0.1:5 - "GET /api/health HTTP/1.1" 200',
                     '127.0.0.1:6 - "WebSocket /ws/chat?(query)" [accepted]',
                     '127.0.0.1:6 - "WebSocket /ws/chat?(query)" 403',
                     '127.0.0.1:6 - "WebSocket /ws/chat?(query)" 500',
                     "Started server process [42]"], lines
    lifespan = next(node for node in app.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan")
    calls = [node for node in ast.walk(lifespan) if isinstance(node, ast.Call)
             and getattr(node.func, "id", "") == "_install_log_filters"]
    yields = [node for node in ast.walk(lifespan) if isinstance(node, ast.Yield)]
    assert calls and calls[0].lineno < yields[0].lineno, "installed as the backend starts"


# ---------------------------------------------------------------------------
# TT33 -- a turn that names a password
# ---------------------------------------------------------------------------
def test_tt33_a_turn_that_names_a_password_titles_nothing_nor_any_later_turn(tmp_path):
    mod, loaded, restore = _load(tmp_path)
    try:
        titles = loaded[_TITLES]
        cfg = _config(titles).derive
        manager = mod.ConversationManager(db_path=tmp_path / "c.db", publish_to_sync=False)
        conv = manager.create_conversation()
        _typed(manager, conv.id, "is hunter a strong password?")
        assert manager.get_conversation(conv.id).title == "New conversation", "the words before the name may be it"
        _typed(manager, conv.id, "remember hunter for me")
        assert manager.get_conversation(conv.id).title == "New conversation", "and no later turn titles"
        titled = manager.create_conversation()
        _typed(manager, titled.id, "Plan the garden layout")
        _typed(manager, titled.id, "my password is hunter")
        assert manager.get_conversation(titled.id).title == "Plan the garden layout", "a title drawn before stays"
        # The password named anywhere in a turn's content -- typed beside a document, in a turn since withdrawn -- ends
        # the reading all the same: the next turn's words may be the secret.
        doc = "Router manual"
        beside = "the wifi password is in my next message " + doc
        for withdraw in (False, True):
            told = manager.create_conversation()
            manager.add_message(told.id, "user", beside, origin="typed",
                                segments=[[0, 40, "typed"], [40, len(beside), "document"]])
            if withdraw:
                manager.withdraw_source("document:" + hashlib.sha256(doc.encode("utf-8")).hexdigest())
            _typed(manager, told.id, "remember hunter for me")
            assert manager.get_conversation(told.id).title == "New conversation", withdraw
        # Each name titles.yaml lists names a password, its pattern followed; and, whatever the file holds, these
        # do: the usual spellings, identifiers, sigils, digits and symbols written for letters, inflected forms.
        shipped = yaml.safe_load(_TITLES_YAML.read_text(encoding="utf-8"))["derive"]

        def sample(word):
            if word.startswith("+"):
                return "x" + word[1:] + "x"
            if len(word) > 2 and word.startswith("*") and word.endswith("*"):
                return "x" + word[1:-1] + "x"
            return "x" + word[1:] if word.startswith("*") else word[:-1] + "x" if word.endswith("*") else word

        for entry in shipped["password_names"]:
            text = "note " + " ".join(sample(word) for word in entry.split()) + " here"
            assert titles.names_password(text, cfg) and titles.names_password(text.upper(), cfg), text

        def u(*codes):
            return "".join(chr(code) for code in codes)

        # Whatever the file holds, these name a password: each language's names, their usual forms, identifiers,
        # sigils, digits and symbols written for letters.
        for text in ("my dbPassword is blue", "set adminPwd now", "the --passwd flag", "export DB_PASSWORD=x",
                     "Mes Mots-De-Passe", "MotDePasse oublie", "mon mdp est chat", "mypassword", "Kenn-Wort hier",
                     "Kennw" + chr(0xF6) + "rter", "contrase" + chr(0xF1) + "as", "pass-word", "pswd here",
                     "psswd here",
                     "the Wi-Fi password is hunter", "rootpassword --plaintext soleil", "db_pwd is soleil",
                     "my seedphrase is abandon", "my seed phrase is abandon", "the recovery phrase", "phrase de passe",
                     "matkhau la meo", "katasandi saya kucing", "kata sandinya adalah kucing", "kata sandiku",
                     "set mySenhaHash now", "in PHP $password = 'hunter' fails", "no PHP $senha = 'gato' falha",
                     "PASSWORD1: hunter", "senha1: gato", "passw0rd", "s1fre", "k3nnwort", "p4ssword", "pa55word",
                     "kennwor7", "p@ssword", "pa$$word", "wachtwoord", "l" + chr(0xF6) + "senord", "passordet",
                     "adgangskode", "kodeord", "salasana", "salasanoja", "jelsz" + chr(0xF3), "jelszavak",
                     chr(0x015F) + "ifre", "heslo", "hesel je", "the haslo is kot", "has" + chr(0x0142) + "o",
                     "has" + "e" + chr(0x0142), u(0x041F, 0x0430, 0x0440, 0x043E, 0x043B, 0x044F),
                     u(0x03BA, 0x03C9, 0x03B4, 0x03B9, 0x03BA, 0x03CC) + " " + u(0x03C0, 0x03C1, 0x03CC, 0x03C3, 0x03B2,
                                                                                 0x03B1, 0x03C3, 0x03B7, 0x03C2),
                     u(0x05D4, 0x05E1, 0x05D9, 0x05E1, 0x05DE, 0x05D4),
                     u(0x05E1, 0x05D9, 0x05E1, 0x05DE, 0x05EA, 0x05D9),
                     u(0x05E1, 0x05D9, 0x05E1, 0x05DE, 0x05D0, 0x05D5, 0x05EA),
                     u(0x0643, 0x0644, 0x0645, 0x0629) + " " + u(0x0645, 0x0631, 0x0648, 0x0631, 0x064A),
                     u(0x0648, 0x0643, 0x0644, 0x0645, 0x0629) + " " + u(0x0627, 0x0644, 0x0633, 0x0631),
                     u(0x0643, 0x0644, 0x0645, 0x0627, 0x062A) + " "
                     + u(0x0627, 0x0644, 0x0645, 0x0631, 0x0648, 0x0631),
                     u(0x6211, 0x7684, 0x5BC6, 0x7801, 0x662F) + "hunter", u(0x5BC6, 0x78BC),
                     u(0x30D1, 0x30B9, 0x30EF, 0x30FC, 0x30C9), u(0xBE44, 0xBC00, 0xBC88, 0xD638),
                     u(0xD328, 0xC2A4, 0xC6CC, 0xB4DC), u(0xBE44, 0xBC88, 0xC774) + " test",
                     u(0x0E23, 0x0E2B, 0x0E31, 0x0E2A, 0x0E1C, 0x0E48, 0x0E32, 0x0E19),
                     u(0x092A, 0x093E, 0x0938, 0x0935, 0x0930, 0x094D, 0x0921)):
            assert titles.names_password(text, cfg), text
        # A format character inside a name hides nothing, nor does a typographic hyphen; the forms a narrower name
        # would lose are listed.
        for code in (0xAD, 0x200B, 0x200C, 0x200D, 0x2060, 0x200E, 0xFEFF):
            assert titles.names_password("my pass" + chr(code) + "word is hunter", cfg), hex(code)
        for text in ("Kenn" + chr(0x2010) + "Wort", "Kenn" + chr(0x2011) + "Wort", "Pass" + chr(0xAD) + "wort",
                     "w ha" + chr(0x015B) + "le jest kot", "haslom", "haslami", "haslach", "hovedpassord", "passord",
                     "pwd then adminpwd", "Kata SanDi saya kucing",
                     "engangspassordet", "wifipassordet", "e-postpassordet", "hesla", "heslem", "hesly"):
            assert titles.names_password(text, cfg), text
        for text in ("Heslington hall", "Heslop road", "the cache self test",
                     u(0x0643, 0x0644, 0x0645, 0x0627, 0x062A) + " " + u(0x062A, 0x0641, 0x0633, 0x0631) + " that"):
            assert not titles.names_password(text, cfg), text
        # Names that mean other things in common use are left out, and words that only look like a name are none.
        for text in ("What are the key differences between TCP and UDP", "Explain tokens in transformers",
                     "pass the salt please", "passport renewal", "compass heading", "upwards and onwards",
                     "Cl" + chr(0xE9) + " USB non reconnue", "a secret recipe", "Comment tailler un pin parasol",
                     "How do I pass code to a function", "pass word lists to the tokenizer", "how does pwd work",
                     "why is $PWD empty in my cron job", "PwC audit report", "PwD access to public buildings",
                     "CacheSlot eviction policy", "the matches/list endpoint is slow", "Fix the slow query",
                     "Is Diablo Immortal P2W", "a 10 pW laser", "the PW4000 engine", "P/W ratio", "P&W engines",
                     "P.W. Botha", "Haslemere station", "Bill Haslam", "hasLoaded", "has_lock", "POMDP", "MDPI",
                     "drawable-mdpi", "cmd.py", "cmdPrompt", "cmd_parser", "$PWD/build", "PATH=$PWD/bin", "/bin/pwd",
                     "ls/cd/pwd", "passOrder", "a mnemonic phrase for the planets", "in-flight wifi pass",
                     "the secret phrase in the film"):
            assert not titles.names_password(text, cfg), text
        # The patterns' language, whatever names are shipped: each mode against the runs or the segments.
        data = yaml.safe_load(_TITLES_YAML.read_text(encoding="utf-8"))
        data["derive"]["password_names"] = ["abc", "def*", "*ghi", "*jkl*", "+mno", "pqr stu", "~vwx~"]
        made = titles.read_config(data).derive
        modes = {"abc": True, "abcs": True, "abcd": False, "xAbc": True, "defx": True, "xdef": False, "xghi": True,
                 "xghis": True, "ghix": False, "xjklx": True, "jkl": True, "mno": False, "mnos": False, "xmno": True,
                 "$mno": False, "pqr stu": True, "pqr-stu": True, "XPqrStux": True, "xpqrstux": True,
                 "pqr x stu": False, "AbcD": False, "avwxb": True, "AvwXb": False, "x-pqr stu": True}
        assert {text: titles.names_password(text, made) for text in modes} == modes
        # A long text that names nothing is never cut; a turn added after the title's own, or past the turns the
        # rule reads, reads nothing.
        cuts = []
        real_runs = titles._runs
        titles._runs = lambda segment: cuts.append(segment) or real_runs(segment)
        try:
            assert not titles.names_password("the cache self test of the north region " * 20000, cfg) and cuts == []
            # Only the segments that hold a name's letters are cut again.
            assert titles.names_password("alpha budget " * 20000 + "the senha is gato", cfg) and len(cuts) <= 4
        finally:
            titles._runs = real_runs
        reads = []
        rule = titles.first_typed_source
        titles.first_typed_source = lambda *args, **kwargs: reads.append(1) or rule(*args, **kwargs)
        try:
            _typed(manager, titled.id, "Order the paint")
            assert reads == [], "a turn added after the title's own reads nothing"
            named = manager.create_conversation(title="Mine")
            _typed(manager, named.id, "Plan the garden layout")
            assert manager.delete_last_message(named.id)
            assert reads == [], "a turn added to, or deleted from, a conversation the user named reads nothing"
            many = manager.create_conversation()
            for code in ("ls", "cd", "du"):
                _typed(manager, many.id, "```\n" + code + "\n```")
            reads.clear()
            _typed(manager, many.id, "Plan the garden layout")
            assert reads == [] and manager.get_conversation(many.id).title == "New conversation", "past the reading"
            third = manager.create_conversation()
            for code in ("ls", "cd"):
                _typed(manager, third.id, "```\n" + code + "\n```")
            _typed(manager, third.id, "Plan the garden layout")
            assert manager.get_conversation(third.id).title == "Plan the garden layout", "the third turn still titles"
        finally:
            titles.first_typed_source = rule
        # The turns are read in the order they were saved, whatever the clock said when it stepped back.
        stepped = manager.create_conversation()
        _typed(manager, stepped.id, "my password is soleil")
        _typed(manager, stepped.id, "sunflower fields")
        db = tmp_path / "c.db"
        conn = sqlite3.connect(db)
        try:
            for content, stamp in (("E:my password is soleil", "2026-10-25T02:50:00"),
                                   ("E:sunflower fields", "2026-10-25T02:10:00")):
                conn.execute("UPDATE messages SET timestamp = ? WHERE conversation_id = ? AND content = ?",
                             (stamp, stepped.id, content))
            conn.execute("UPDATE conversations SET title_turn = '' WHERE id = ?", (stepped.id,))
            conn.commit()
        finally:
            conn.close()
        manager.tend_titles()
        assert manager.get_conversation(stepped.id).title == "New conversation", "a clock step reorders nothing"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT34 -- titles.yaml read once, under a lock
# ---------------------------------------------------------------------------
def test_tt34_a_second_first_reader_of_titles_yaml_waits_for_the_reading(tmp_path):
    mod, loaded, restore = _load(tmp_path)
    try:
        titles = loaded[_TITLES]
        started, release = threading.Event(), threading.Event()
        real = titles.read_config
        reads = []

        def slow(data):
            reads.append(data)
            started.set()
            release.wait(5)
            return real(data)

        titles.read_config = slow
        mod._titles_read.update(done=False, config=None)
        got = {}
        first = threading.Thread(target=lambda: got.update(first=mod._titles_config()))
        second = threading.Thread(target=lambda: got.update(second=mod._titles_config()))
        first.start()
        assert started.wait(5)
        second.start()
        second.join(0.3)
        assert second.is_alive(), "a second first reader waits for the reading rather than read titles off"
        release.set()
        first.join(5)
        second.join(5)
        assert got["first"] is not None and got["second"] is got["first"] and len(reads) == 1, "read once"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Rule (b): the rule's title refined by the local model (block 180)
# ---------------------------------------------------------------------------
_TURN = "Plan the garden layout for spring with raised beds"


def _locality():
    """inference_backend.py's own locality check compiled alone -- ``_is_on_this_machine``, its public name and
    ``ENDPOINT_IN_PROCESS`` as the file writes them --, with ``local_refusal`` a Daily stand-in that lets every request
    go: a refinement that consulted it would call off the machine."""
    tree = ast.parse(Path(source("inference_backend.py")).read_text(encoding="utf-8"))
    kept = [node for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in ("_is_on_this_machine", "is_on_this_machine"))
            or (isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "ENDPOINT_IN_PROCESS"
                                                     for t in node.targets))]
    module = types.ModuleType("opti_oignon.inference_backend")
    exec(compile(ast.Module(body=kept, type_ignores=[]), "inference_backend.py", "exec"), module.__dict__)
    module.local_refusal = lambda label, endpoint, where="": None
    return module


class _Governance:
    """The resource governor's stand-in: what it was asked, the admissions handed back, the ticket each thread holds."""

    def __init__(self, admitted=True, refuse_entry=False, on_admit=None):
        self.admitted, self.refuse_entry, self.on_admit = admitted, refuse_entry, on_admit
        self.asked, self.handed, self.local = [], [], threading.local()
        governance = self

        class _Governor:
            def admit_or_wait(self, model, requested_ctx=None, caller="benchmark", extra_models=None, digest=None,
                              wait_s=None, engine=None, cancel=None):
                governance.asked.append({"model": model, "requested_ctx": requested_ctx, "caller": caller,
                                         "wait_s": wait_s, "cancel": cancel})
                if governance.on_admit is not None:
                    governance.on_admit()
                return types.SimpleNamespace(admitted=governance.admitted, num_ctx=2048, keep_alive="1m",
                                             ticket_id=f"t{len(governance.asked)}", reason="busy")

            def end_pending_load(self, ticket_id):
                governance.handed.append(ticket_id)

        @contextlib.contextmanager
        def ticket_scope(decision):
            if governance.refuse_entry:
                raise RuntimeError("the ticket is refused on entry")
            governance.local.ticket = decision.ticket_id
            try:
                yield
            finally:
                governance.local.ticket = None

        governor = _Governor()
        self.module = types.ModuleType("opti_oignon.resource_governor")
        self.module.get_resource_governor = lambda: governor
        self.module.ticket_scope = ticket_scope


class _Backend:
    """A backend stand-in: says where its requests go (its own, never forwarded), records each call with the ticket
    its thread holds, and answers as told -- once its gate opens when one is given (``gates``: one per call, in
    order), after ``on_call`` when given."""

    forwards_requests = False

    def __init__(self, governance, endpoint="in-process", answer="Garden layout for spring", gate=None, error=None,
                 gates=(), on_call=None):
        self.governance, self._endpoint, self.answer, self.gate, self.error = governance, endpoint, answer, gate, error
        self.gates, self.on_call = list(gates), on_call
        self.calls = []

    def endpoint(self):
        if isinstance(self._endpoint, BaseException):
            raise self._endpoint
        return self._endpoint

    def generate(self, model, messages, options=None, keep_alive="30m", think=None, images=None):
        self.calls.append({"model": model, "messages": messages, "options": dict(options or {}),
                           "keep_alive": keep_alive, "think": think, "thread": threading.current_thread().name,
                           "ticket": getattr(self.governance.local, "ticket", None)})
        gate = self.gates[len(self.calls) - 1] if len(self.calls) <= len(self.gates) else self.gate
        if gate is not None:
            gate.wait(10)
        if self.on_call is not None:
            self.on_call()
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(content=self.answer)


def _refining(tmp_path, *, backend=None, governance=None, stop=False, backends=None, **refine):
    """The store, the titles module and title_refiner.py in the window, the refinement on (its timeout one second,
    ``refine`` keys replaced): the registry resolves model "m" to ``backend``, the governor is ``governance``, the
    emergency stop engaged when ``stop`` is True, unreadable when it is "raise", absent from the package when it is
    None (``state["stopped"]`` flips it mid-job); ``backends`` replaces the inference module's stand-in whole."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    governance = governance or _Governance()
    backend = backend if backend is not None else _Backend(governance)
    if backends is None:
        backends = _locality()
        backends.get_backend_registry = lambda: types.SimpleNamespace(resolve_backend={"m": backend}.get)
    seeded = {"opti_oignon.inference_backend": backends, "opti_oignon.resource_governor": governance.module}
    state = {"stopped": stop is True}
    if stop is not None:
        halt = types.ModuleType("opti_oignon.emergency_stop")

        def is_stopped():
            if stop == "raise":
                raise OSError("the stop cannot be read")
            return state["stopped"]

        halt.is_stopped = is_stopped
        seeded["opti_oignon.emergency_stop"] = halt
    mod, loaded, restore = _load(tmp_path, refiner=True, extra=seeded)
    shipped = _config(loaded[_TITLES])
    cfg = dataclasses.replace(shipped, refine=dataclasses.replace(
        shipped.refine, **{"enabled": True, "timeout_s": 1, **refine}))
    mod._titles_read.update(done=True, config=cfg)
    world = types.SimpleNamespace(mod=mod, refiner=loaded[_REFINER], titles=loaded[_TITLES], cfg=cfg,
                                  backend=backend, governance=governance, state=state, db=tmp_path / "c.db")
    return world, restore


def _drained(world):
    """A runner that drains the queue on the caller's thread: the outcome is read as maybe_refine returns."""
    return lambda manager, cfg: world.refiner._drain(manager, cfg)


def _titled(world, text=_TURN, *, manager=None):
    """A conversation answered with model "m" whose first turn typed ``text``: its title is the rule's."""
    manager = manager or world.mod.ConversationManager(db_path=world.db, publish_to_sync=False)
    conv = manager.create_conversation(model="m")
    _typed(manager, conv.id, text)
    manager.add_message(conv.id, "assistant", "Here is a plan.", origin="assistant", context=[], lineage=[])
    return manager, conv.id


def _seal(path, conv_id):
    """Every user turn of a conversation left as a turn of another key would be: encrypted, behind the store."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("UPDATE messages SET content = 'ENC2:c2VhbGVk' WHERE conversation_id = ? AND role = 'user'",
                     (conv_id,))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# TT15 -- a refined title lands only over the rule's own
# ---------------------------------------------------------------------------
def test_tt15_a_refined_title_lands_only_over_the_rules_own(tmp_path):
    world, restore = _refining(tmp_path)
    try:
        mod, db = world.mod, world.db
        calls = _spy(mod)
        manager = mod.ConversationManager(db_path=db)
        conv = manager.create_conversation(model="m")
        _typed(manager, conv.id, _TURN)
        turn, basis, title, words, model = manager.title_state(conv.id)
        assert (title, model, turn) == (_TURN, "m", _row(db, conv.id)["title_turn"])
        assert words == world.titles.kept_words(_TURN, world.cfg.derive) and basis == _row(db, conv.id)["title_basis"]
        published = len(calls)
        assert manager.commit_refined_title(conv.id, turn, basis, "Garden layout for spring") is True
        row = _row(db, conv.id)
        assert (row["title"], row["title_source"], row["title_turn"]) == ("E:Garden layout for spring", "refined", turn)
        assert len(calls) == published + 1, "one publish"
        assert manager.commit_refined_title(conv.id, turn, basis, "Raised beds") is False, "refined once"
        assert manager.title_state(conv.id) is None, "a refined title is not the rule's own"
        _typed(manager, conv.id, "And the watering")
        assert _row(db, conv.id)["title_source"] == "refined", "a turn added leaves a refined title"
        # Its words changed under it, its print changed, a rename, a deletion: nothing written, nothing published.
        for spoil in ("basis", "print", "rename", "delete"):
            other = manager.create_conversation(model="m")
            _typed(manager, other.id, _TURN)
            turn, basis, _title, _words, _model = manager.title_state(other.id)
            if spoil == "basis":
                _plain(db, other.id, title_basis="E:" + "0" * 64)
                assert manager.title_state(other.id) is None, "a title whose words changed under it is not refined"
            elif spoil == "print":
                _plain(db, other.id, title_turn=turn.split(":")[0] + ":" + "0" * 16)
                assert manager.title_state(other.id) is None, "a title drawn under another print is not refined"
            elif spoil == "rename":
                manager.rename_conversation(other.id, "Mine")
            else:
                manager.delete_conversation(other.id)
            published = len(calls)
            assert manager.commit_refined_title(other.id, turn, basis, "Garden layout") is False, spoil
            assert len(calls) == published, f"{spoil}: nothing published"
            if spoil != "delete":
                assert _row(db, other.id)["title_source"] != "refined", spoil
        # Only a title rule (a) keeps whole as it stands is written, whatever hands it.
        last = manager.create_conversation(model="m")
        _typed(manager, last.id, _TURN)
        turn, basis, _title, _words, _model = manager.title_state(last.id)
        for refused in ("Garden 2 layout", "the password garden", "Garden `x` layout", "", "garden " * 20, None):
            assert manager.commit_refined_title(last.id, turn, basis, refused) is False, refused
        assert _row(db, last.id)["title_source"] == "auto"
        # The rule draws a refined title again when its print changes (the start) or its own turn goes.
        assert manager.commit_refined_title(last.id, turn, basis, "Garden layout") is True
        _plain(db, last.id, title_turn=turn.split(":")[0] + ":" + "0" * 16)
        assert manager.tend_titles() >= 1
        assert (_row(db, last.id)["title_source"], manager.get_conversation(last.id).title) == ("auto", _TURN)
        turn, basis, _title, _words, _model = manager.title_state(last.id)
        assert manager.commit_refined_title(last.id, turn, basis, "Garden layout") is True
        assert manager.delete_last_message(last.id) is True
        assert (_row(db, last.id)["title_source"], manager.get_conversation(last.id).title) == ("default",
                                                                                                "New conversation")
        # No title of the rule's own, no state: a placeholder, a manual or legacy title, a turn that does not decrypt.
        plain = manager.create_conversation(model="m")
        assert manager.title_state(plain.id) is None
        named = manager.create_conversation(title="Mine", model="m")
        _typed(manager, named.id, _TURN)
        assert manager.title_state(named.id) is None
        older = manager.create_conversation(model="m")
        _typed(manager, older.id, _TURN)
        _plain(db, older.id, title_source="legacy")
        assert manager.title_state(older.id) is None
        sealed = manager.create_conversation(model="m")
        _typed(manager, sealed.id, _TURN)
        _seal(db, sealed.id)
        assert manager.title_state(sealed.id) is None, "a turn that does not decrypt starts nothing"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT16 -- the model's answer is held to the rule's own words
# ---------------------------------------------------------------------------
def test_tt16_the_models_answer_is_held_to_the_rules_own_words(tmp_path):
    mod, loaded, restore = _load(tmp_path)
    try:
        titles = loaded[_TITLES]
        cfg = _config(titles)
        derive, refine = cfg.derive, cfg.refine
        # kept_words are rule (a)'s words before the bound: its title is them, bounded.
        for text, _title in _GOLDEN:
            kept = titles.kept_words(text, derive)
            assert (kept is None) == (titles.derive_title(text, derive) is None), text
            if kept is not None:
                assert titles.derive_title(" ".join(kept), derive) == titles.derive_title(text, derive), text
        kept = titles.kept_words("Plan the garden layout for spring with 3 raised beds, see src/a.py", derive)
        assert kept == ["Plan", "the", "garden", "layout", "for", "spring", "with", "raised", "beds", "see"]
        accepted = {
            "Garden layout for spring": "Garden layout for spring",
            "Title: Raised beds for spring": "Raised beds for spring",
            "**Titre :** Garden layout": "Garden layout",
            '"Spring garden layout"': "Spring garden layout",
            "GARDEN LAYOUT\nA second line the model added": "GARDEN LAYOUT",
            "How to plan the garden": "How to plan the garden",
        }
        for answer, title in accepted.items():
            assert titles.accept_refined(answer, kept, derive, refine) == title, answer
        for refused in ("3 raised beds", "Garden layout 2026", "the password garden", "Garden `rm -rf` layout",
                        "1. Garden layout", "Jardin layout", "Summary: garden layout", "", "Title:", "   \n  ",
                        "garden " * 20, "src/a.py garden", "garden@layout.com", None):
            assert titles.accept_refined(refused, kept, derive, refine) is None, refused
        assert titles.accept_refined("Garden layout", [], derive, refine) is None, "nothing kept: nothing refined"
        # Composed and case folded, nothing else: an accent, a mark, a selector or a fullwidth form makes a word the
        # user did not type, even one that folds to a function word.
        wide = "".join(chr(0xFF00 + ord(char) - 0x20) for char in "garden")
        for added in ("Th" + chr(0xE9) + " garden layout", "O" + chr(0xF9) + " garden", "L" + chr(0xE0) + " garden",
                      "garden" + chr(0x337) * 6 + " layout", "garden" + chr(0xFE0F) + " layout", wide + " layout"):
            assert titles.accept_refined(added, kept, derive, refine) is None, added
        sin = titles.kept_words("Le p" + chr(0xE9) + "ch" + chr(0xE9) + " mignon", derive)
        assert titles.accept_refined("Le peche mignon", sin, derive, refine) is None, "an accent taken off: another word"
        upper = "LE P" + chr(0xC9) + "CH" + chr(0xC9) + " MIGNON"
        assert titles.accept_refined(upper, sin, derive, refine) == upper, "its case is the same word"
        # Lowercased, not case folded: a sharp s, a ligature, a long s or a symbol form is another word.
        for typed, answered in (("Busse", "Bu" + chr(0xDF) + "e"), ("Masse", "Ma" + chr(0xDF) + "e"),
                                ("Floss", "Flo" + chr(0xDF)), ("file", chr(0xFB01) + "le"), ("state", chr(0x17F) + "tate"),
                                (chr(0x3B8) + "eta", chr(0x3D1) + "eta"), (chr(0x3BC) + "eson", chr(0xB5) + "eson")):
            pair = titles.kept_words(typed + " garden layout", derive)
            assert titles.accept_refined(answered + " garden", pair, derive, refine) is None, (typed, answered)
            assert titles.accept_refined(typed.upper() + " garden", pair, derive, refine) is not None, typed
        # An answer is composed as the words it is compared with were: a decomposed accent is the typed word.
        composed = titles.kept_words("Cl" + chr(0xE9) + " USB non reconnue", derive)
        assert titles.accept_refined("Cle" + chr(0x301) + " USB", composed, derive, refine) == "Cl" + chr(0xE9) + " USB"
        # A script written without spaces: its run is one word, given back whole or refused.
        run = "".join(chr(code) for code in (0x82B1, 0x56ED, 0x5E03, 0x5C40, 0x8BA1, 0x5212))
        assert titles.kept_words(run, derive) == [run]
        assert titles.accept_refined(run, [run], derive, refine) == run
        assert titles.kept_words(run[:4], derive) == [run[:4]], "a part of the run is a word of its own"
        assert titles.accept_refined(run[:4], [run], derive, refine) is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT17 -- a call past its deadline is given up, and the queue is bounded
# ---------------------------------------------------------------------------
def test_tt17_a_call_past_its_deadline_is_given_up_and_the_queue_is_bounded(tmp_path):
    gate = threading.Event()
    governance = _Governance()
    backend = _Backend(governance, gate=gate)
    world, restore = _refining(tmp_path, backend=backend, governance=governance, max_queued=2, max_in_flight=1)
    try:
        refiner = world.refiner
        now = [100.0]
        refiner._clock = lambda: now[0]
        manager, first = _titled(world)
        assert refiner.maybe_refine(manager, first, runner=_drained(world)) is True
        assert refiner.counts().get("timeout") == 1 and len(backend.calls) == 1
        assert _row(world.db, first)["title_source"] == "auto", "nothing committed"
        # While the call given up lives, no other call to that model starts.
        _manager, second = _titled(world, "Prune the apple trees in winter", manager=manager)
        assert refiner.maybe_refine(manager, second, runner=_drained(world)) is True
        assert refiner.counts().get("hung") == 1 and len(backend.calls) == 1
        gate.set()
        for thread in [t for t in threading.enumerate() if t.name == "oo-titler-call"]:
            thread.join(5)
        _manager, third = _titled(world, "Repot the lemon tree in spring", manager=manager)
        backend.answer = "Lemon tree in spring"
        assert refiner.maybe_refine(manager, third, runner=_drained(world)) is True
        assert len(backend.calls) == 2 and refiner.counts().get("committed") == 1
        assert manager.get_conversation(third).title == "Lemon tree in spring"
        # The queue holds max_queued jobs and one more is dropped; one drain at a time.
        drains, convs = [], []
        for text in ("Sow the carrots", "Water the roses", "Mulch the beds", "Rake the leaves"):
            _manager, conv = _titled(world, text, manager=manager)
            convs.append(conv)
        queued = [refiner.maybe_refine(manager, conv, runner=lambda m, c: drains.append(c)) for conv in convs]
        assert queued == [True, True, False, False] and len(drains) == 1
        assert refiner.counts().get("dropped_queue_full") == 2
        assert [refiner.pending(conv) for conv in convs] == [True, True, False, False]
        assert refiner.cancel(convs[1]) is True and refiner.pending(convs[1]) is False
        # One job a conversation: a new title of a conversation whose job waits queues nothing more.
        assert manager.delete_last_message(convs[0]) is True and manager.delete_last_message(convs[0]) is True
        _typed(manager, convs[0], "Mulch the roses")
        assert refiner.maybe_refine(manager, convs[0], runner=lambda m, c: drains.append(c)) is False
        # A job is pending through each of its four waits -- the queue, the backend's resolution, the admission, the
        # answer --, not after its deadline; dequeued past its wait in the queue, it never runs.
        now[0] += 3 * world.cfg.refine.timeout_s
        assert refiner.pending(convs[0]) is True
        now[0] += world.cfg.refine.timeout_s + 1
        assert refiner.pending(convs[0]) is False
        refiner._drain(manager, world.cfg)
        assert refiner.counts().get("expired") == 1 and len(backend.calls) == 2
    finally:
        restore()
    # The attempts are bounded; a drain that cannot start takes its job back, its title asked again later.
    world, restore = _refining(tmp_path / "bounds", attempted_cap=2)
    try:
        refiner = world.refiner
        manager = world.mod.ConversationManager(db_path=world.db, publish_to_sync=False)
        convs = [_titled(world, text, manager=manager)[1] for text in ("Sow the carrots", "Water the roses",
                                                                         "Mulch the beds")]
        for conv in convs:
            assert refiner.maybe_refine(manager, conv, runner=_drained(world)) is True
        assert len(refiner._attempted) == 2, "attempted_cap holds"

        _manager, other = _titled(world, "Rake the leaves", manager=manager)
        _manager, beside = _titled(world, "Sweep the paths", manager=manager)

        def refused(manager, cfg):
            # A job queued while this drain fails to start finds no drain either.
            assert refiner.maybe_refine(manager, beside, runner=refused) is True
            raise RuntimeError("no thread")

        assert refiner.maybe_refine(manager, other, runner=refused) is False
        assert not refiner._jobs and not refiner._queue and refiner._draining[0] == 0
        assert refiner.counts().get("taken_back") == 2, "both taken back, none left queued without a drain"
        assert refiner.maybe_refine(manager, other, runner=_drained(world)) is True
        assert refiner.maybe_refine(manager, beside, runner=_drained(world)) is True
    finally:
        restore()
    # With a drain still running, a start that fails takes nothing back: that drain runs the queue.
    gate = threading.Event()
    governance = _Governance()
    backend = _Backend(governance, gate=gate)
    world, restore = _refining(tmp_path / "running", backend=backend, governance=governance, timeout_s=5,
                               max_in_flight=2)
    try:
        refiner = world.refiner
        manager, first = _titled(world)
        _manager, second = _titled(world, "Prune the apple trees in winter", manager=manager)
        assert refiner.maybe_refine(manager, first) is True
        for _ in range(500):
            if backend.calls:
                break
            threading.Event().wait(0.01)

        def refused(manager, cfg):
            raise RuntimeError("no thread")

        assert refiner.maybe_refine(manager, second, runner=refused) is False
        assert first in refiner._jobs and second in refiner._jobs and refiner.pending(first), "nothing taken back"
        gate.set()
        for thread in [t for t in threading.enumerate() if t.name in ("oo-titler", "oo-titler-call")]:
            thread.join(10)
        assert len(backend.calls) == 2 and not refiner._jobs
    finally:
        restore()
    # An older job's end keeps the newer job of its conversation: the deletion's cancel still reaches it.
    gates = [threading.Event(), threading.Event()]
    governance = _Governance()
    backend = _Backend(governance, gates=gates)
    world, restore = _refining(tmp_path / "newer", backend=backend, governance=governance, timeout_s=5)
    try:
        refiner = world.refiner
        manager, conv = _titled(world)

        def wait_calls(count):
            for _ in range(500):
                if len(backend.calls) >= count:
                    return
                threading.Event().wait(0.01)
            raise AssertionError(f"{count} call(s) never came")

        assert refiner.maybe_refine(manager, conv) is True
        wait_calls(1)
        assert refiner.cancel(conv) is True
        assert refiner.maybe_refine(manager, conv, runner=lambda m, c: None) is True, "the cancel forgot the title"
        gates[0].set()
        wait_calls(2)
        assert refiner.cancel(conv) is True, "the newer job is still the conversation's"
        gates[1].set()
        for thread in [t for t in threading.enumerate() if t.name in ("oo-titler", "oo-titler-call")]:
            thread.join(10)
        assert _row(world.db, conv)["title_source"] == "auto" and refiner.counts().get("cancelled") == 2
    finally:
        restore()
    # A resolution of the backend that outlives the wait is given up; the drain goes on, and no other starts for that
    # model while it lives.
    gate = threading.Event()
    governance = _Governance()
    backend = _Backend(governance)
    backends = _locality()
    backends.get_backend_registry = lambda: types.SimpleNamespace(
        resolve_backend=lambda model: backend if gate.wait(10) else None)
    world, restore = _refining(tmp_path / "resolve", backend=backend, governance=governance, backends=backends)
    try:
        refiner = world.refiner
        manager, first = _titled(world)
        _manager, second = _titled(world, "Prune the apple trees in winter", manager=manager)
        assert refiner.maybe_refine(manager, first, runner=_drained(world)) is True
        assert refiner.maybe_refine(manager, second, runner=_drained(world)) is True
        counts = refiner.counts()
        assert (counts.get("resolve_timeout"), counts.get("hung"), len(backend.calls)) == (1, 1, 0)
        assert "no_backend" not in counts, "a resolution given up is counted once"
        gate.set()
        for thread in [t for t in threading.enumerate() if t.name == "oo-titler-resolve"]:
            thread.join(10)
        _manager, third = _titled(world, "Repot the lemon tree in spring", manager=manager)
        assert refiner.maybe_refine(manager, third, runner=_drained(world)) is True and len(backend.calls) == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT18 -- a backend that cannot say it answers on this machine is never called
# ---------------------------------------------------------------------------
def test_tt18_a_backend_that_cannot_say_it_answers_on_this_machine_is_never_called(tmp_path):
    cases = (("http://10.0.0.5:11434", False), ("http://ollama.lan:11434", False), ("http://192.168.1.2:8080", False),
             (None, False), ("", False), (OSError("unreadable"), False),
             ("in-process", True), ("http://127.0.0.1:11434", True), ("http://localhost:11434", True),
             ("http://[::1]:11434", True), ("http://0.0.0.0:11434", True))
    for n, (endpoint, called) in enumerate(cases + (("forwards", False), ("undeclared", False))):
        governance = _Governance()
        backend = _Backend(governance, endpoint="http://127.0.0.1:9000" if endpoint in ("forwards", "undeclared")
                           else endpoint)
        if endpoint == "forwards":
            backend.forwards_requests = True
        elif endpoint == "undeclared":
            # A backend that does not say whether it forwards its requests: no such attribute at all.
            backend = types.SimpleNamespace(calls=[], endpoint=lambda: "http://127.0.0.1:9000",
                                            generate=lambda **kwargs: backend.calls.append(kwargs))
        world, restore = _refining(tmp_path / f"case{n}", backend=backend, governance=governance)
        try:
            manager, conv = _titled(world)
            world.refiner.maybe_refine(manager, conv, runner=_drained(world))
            assert (len(backend.calls) == 1) is called, endpoint
            assert bool(governance.asked) is called, f"{endpoint}: the governor is asked only for a call that may go"
            if not called:
                assert world.refiner.counts().get("refused_off_machine") == 1, endpoint
        finally:
            restore()


# ---------------------------------------------------------------------------
# TT19 -- nothing is dispatched unless the title is the rule's own and all may run
# ---------------------------------------------------------------------------
def test_tt19_nothing_is_dispatched_unless_the_title_is_the_rules_own_and_all_may_run(tmp_path):
    for n, (setting, why) in enumerate(((dict(enabled=False), "refinement off"), ({}, "titles off"),
                                         ({}, "stop engaged"), ({}, "stop absent"), ({}, "stop unreadable"))):
        stop = {"stop engaged": True, "stop absent": None, "stop unreadable": "raise"}.get(why, False)
        world, restore = _refining(tmp_path / f"off{n}", stop=stop, **setting)
        try:
            manager, conv = _titled(world)
            if why == "titles off":
                world.mod._titles_read.update(config=dataclasses.replace(world.cfg, enabled=False))
            assert world.refiner.maybe_refine(manager, conv, runner=_drained(world)) is False, why
            assert world.backend.calls == [] and world.refiner.pending(conv) is False, why
        finally:
            restore()
    world, restore = _refining(tmp_path / "on")
    try:
        refiner, db = world.refiner, world.db
        manager, conv = _titled(world)
        # A placeholder, a manual, legacy or refined title, a turn that does not decrypt: nothing to refine.
        plain = manager.create_conversation(model="m").id
        named = manager.create_conversation(title="Mine", model="m").id
        _typed(manager, named, _TURN)
        _manager, older = _titled(world, manager=manager)
        _plain(db, older, title_source="legacy")
        _manager, refined = _titled(world, manager=manager)
        _plain(db, refined, title_source="refined")
        _manager, sealed = _titled(world, manager=manager)
        _seal(db, sealed)
        for other in (plain, named, older, refined, sealed):
            assert refiner.maybe_refine(manager, other, runner=_drained(world)) is False, other
        assert world.backend.calls == []
        # No model named: the conversation has none and titles.yaml names none.
        modelless = manager.create_conversation().id
        _typed(manager, modelless, _TURN)
        assert refiner.maybe_refine(manager, modelless, runner=_drained(world)) is False
        assert refiner.counts().get("not_asked") == 1
        # Each title is asked once in a process; a rename forgets the conversation's.
        assert refiner.maybe_refine(manager, conv, runner=_drained(world)) is True and len(world.backend.calls) == 1
        manager.use_automatic_title(conv)
        assert refiner.maybe_refine(manager, conv, runner=_drained(world)) is False, "asked once"
        refiner.cancel(conv)
        assert refiner.maybe_refine(manager, conv, runner=_drained(world)) is True and len(world.backend.calls) == 2
        # Dispatch never raises: a store that fails is no job.
        broken = types.SimpleNamespace(titles_config=lambda: world.cfg, title_state=lambda conv_id: 1 / 0)
        assert refiner.maybe_refine(broken, conv) is False
    finally:
        restore()
    # Past dispatch: a stop engaged before the drain, during the admission or during the call, and a cancel during the
    # call -- no call, the admission handed back, nothing written.
    for n, moment in enumerate(("queued", "admission", "reread", "reread-cancel", "call", "cancel")):
        governance = _Governance()
        backend = _Backend(governance)
        world, restore = _refining(tmp_path / f"past{n}", backend=backend, governance=governance)
        try:
            manager, conv = _titled(world)
            if moment == "admission":
                governance.on_admit = lambda: world.state.update(stopped=True)
            elif moment == "call":
                backend.on_call = lambda: world.state.update(stopped=True)
            elif moment == "cancel":
                backend.on_call = lambda: world.refiner.cancel(conv)
            elif moment.startswith("reread"):
                real, reads = manager.title_state, []

                def reading(conv_id, real=real, reads=reads, moment=moment):
                    found = real(conv_id)
                    reads.append(conv_id)
                    if len(reads) > 1:
                        # The store's read once admitted: the stop or a cancel comes while it waits.
                        if moment == "reread":
                            world.state["stopped"] = True
                        else:
                            world.refiner.cancel(conv_id)
                    return found

                manager.title_state = reading
            idle = []
            assert world.refiner.maybe_refine(manager, conv, runner=lambda m, c: idle.append(c)) is True
            if moment == "queued":
                world.state["stopped"] = True
            world.refiner._drain(manager, world.cfg)
            assert len(backend.calls) == (1 if moment in ("call", "cancel") else 0), moment
            assert governance.handed == (["t1"] if moment in ("admission", "reread", "reread-cancel") else []), moment
            assert _row(world.db, conv)["title_source"] == "auto", f"{moment}: nothing written"
        finally:
            restore()
    # A named model is asked for a conversation that has none.
    governance = _Governance()
    backend = _Backend(governance)
    world, restore = _refining(tmp_path / "named", backend=backend, governance=governance, model="m")
    try:
        manager = world.mod.ConversationManager(db_path=world.db, publish_to_sync=False)
        conv = manager.create_conversation().id
        _typed(manager, conv, _TURN)
        assert world.refiner.maybe_refine(manager, conv, runner=_drained(world)) is True
        assert [call["model"] for call in backend.calls] == ["m"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT20 -- the call is the background's, admitted as titler, its ticket held where it is asked
# ---------------------------------------------------------------------------
def test_tt20_the_call_is_the_backgrounds_admitted_as_titler_its_ticket_held_where_it_is_asked(tmp_path):
    shipped = yaml.safe_load((REPO / "opti_oignon" / "config" / "resource_governor.yaml").read_text(encoding="utf-8"))
    assert shipped["classes"]["callers"]["titler"] == "background"
    assert shipped["ctx_floor"]["titler"] == min(shipped["ctx_ladder"])
    governor = _module("opti_oignon/resource_governor.py")
    defaults = next(ast.literal_eval(node.value) for node in governor.body if isinstance(node, ast.Assign)
                    and getattr(node.targets[0], "id", "") == "_DEFAULT_CALLER_CLASSES")
    assert defaults["titler"] == "background"
    floors = next(ast.literal_eval(node.value.keywords[0].value.body) for node in ast.walk(governor)
                  if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "ctx_floor")
    assert floors["titler"] == min(shipped["ctx_ladder"]), "the defaults give the titler its floor"
    world, restore = _refining(tmp_path / "admitted")
    try:
        manager, conv = _titled(world)
        assert world.refiner.maybe_refine(manager, conv, runner=_drained(world)) is True
        (asked,) = world.governance.asked
        assert {key: asked[key] for key in ("model", "requested_ctx", "caller", "wait_s")} == {
            "model": "m", "requested_ctx": world.cfg.refine.num_ctx, "caller": "titler",
            "wait_s": float(world.cfg.refine.timeout_s)}
        assert isinstance(asked["cancel"], threading.Event), "the job's cancel goes to the governor"
        (call,) = world.backend.calls
        assert (call["thread"], call["ticket"]) == ("oo-titler-call", "t1"), "the ticket held on the asking thread"
        assert (call["options"]["num_ctx"], call["keep_alive"]) == (2048, "1m"), "the admitted window and residency"
        assert world.governance.handed == []
    finally:
        restore()
    for n, (governance, why) in enumerate(((_Governance(admitted=False), "not admitted"),
                                           (_Governance(refuse_entry=True), "refused on entry"))):
        backend = _Backend(governance)
        world, restore = _refining(tmp_path / f"refused{n}", backend=backend, governance=governance)
        try:
            manager, conv = _titled(world)
            world.refiner.maybe_refine(manager, conv, runner=_drained(world))
            assert backend.calls == [], why
            assert governance.handed == ([] if why == "not admitted" else ["t1"]), f"{why}: an unused admission"
            assert _row(world.db, conv)["title_source"] == "auto", why
        finally:
            restore()
    # A governor that cannot be had admits nothing: no call at all.
    governance = _Governance()
    backend = _Backend(governance)
    governance.module.get_resource_governor = lambda: 1 / 0
    world, restore = _refining(tmp_path / "ungoverned", backend=backend, governance=governance)
    try:
        manager, conv = _titled(world)
        world.refiner.maybe_refine(manager, conv, runner=_drained(world))
        assert backend.calls == [] and world.refiner.counts().get("not_admitted") == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT22 -- the title route says pending while a job lives, and the routes cancel it
# ---------------------------------------------------------------------------
def test_tt22_the_title_route_says_pending_while_a_job_lives_and_the_routes_cancel_it(tmp_path):
    from fastapi import HTTPException

    world, restore = _refining(tmp_path, poll_after_ms=1234)
    try:
        refiner = world.refiner
        now = [100.0]
        refiner._clock = lambda: now[0]
        idle = []
        manager, conv = _titled(world)
        _manager, gone = _titled(world, "Prune the apple trees in winter", manager=manager)
        _manager, late = _titled(world, "Repot the lemon tree in spring", manager=manager)
        for each in (conv, gone, late):
            assert refiner.maybe_refine(manager, each, runner=lambda m, c: idle.append(c)) is True
        routes, schemas, close = _routes(manager, refiner=refiner)
        try:
            assert routes.get_conversation_title(conv) == {
                "title": _TURN, "title_source": "auto", "title_pending": True, "retry_after_ms": 1234}
            assert routes.get_conversation(conv)["title_pending"] is True
            job = refiner._jobs[conv]
            renamed = routes.rename_conversation(conv, schemas.ConversationRename(title="Mine"))
            assert renamed["title_pending"] is False and refiner.pending(conv) is False, "a rename cancels"
            assert job.cancelled.is_set(), "the job itself is told: a wait at the governor ends"
            assert routes.get_conversation_title(conv) == {
                "title": "Mine", "title_source": "manual", "title_pending": False, "retry_after_ms": None}
            job = refiner._jobs[gone]
            routes.delete_conversation(gone)
            assert refiner.pending(gone) is False and job.cancelled.is_set(), "a deletion cancels"
            with pytest.raises(HTTPException) as raised:
                routes.get_conversation_title(gone)
            assert raised.value.status_code == 404
            assert routes.get_conversation_title(late)["title_pending"] is True
            now[0] += 4 * world.cfg.refine.timeout_s + 1
            assert routes.get_conversation_title(late) == {
                "title": "Repot the lemon tree in spring", "title_source": "auto", "title_pending": False,
                "retry_after_ms": None}, "past its deadline a job is not pending"
        finally:
            close()
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT24 -- the interface watches a pending title at the server's pace
# ---------------------------------------------------------------------------
def _ts_function(text, name):
    start = text.index(f"function {name}(")
    return text[start:text.index("\n}\n", start)]


def test_tt24_the_interface_watches_a_pending_title_at_the_servers_pace():
    store = (REPO / "frontend" / "src" / "lib" / "stores" / "conversations.ts").read_text(encoding="utf-8")
    api = (REPO / "frontend" / "src" / "lib" / "api" / "conversations.ts").read_text(encoding="utf-8")
    load = _ts_function(store, "loadConversations")
    assert re.search(r"conversations\.set\(list\);\s*(?://[^\n]*\n\s*)?"
                     r"for \(const c of list\) if \(c\.title_pending\) watchTitle\(c\.id\);", load)
    watch = _ts_function(store, "watchTitle")
    assert "await api.getConversationTitle(id)" in watch
    assert "again = answer.retry_after_ms;" in watch and "setTimeout(ask, again);" in watch
    assert "answer.title_pending && typeof answer.retry_after_ms === 'number'" in watch
    assert not re.search(r"\d", watch), "no delay of the client's own: the server paces the watch"
    assert "if (watched.has(id)) return;" in watch and "watched.delete(id);" in watch
    assert "title_pending: false" in watch, "the title the server answers, once nothing is pending"
    assert re.search(r"export async function getConversationTitle\(id: string\): Promise<ConversationTitle> \{\s*"
                     r"return apiGet<ConversationTitle>\(`/api/conversations/\$\{id\}/title`\);", api)


# ---------------------------------------------------------------------------
# TT35 -- the model is handed the words the rule keeps, and nothing else
# ---------------------------------------------------------------------------
_TYPED = "Plan the garden layout v2 for spring, see /srv/plan.txt and `make beds`"
_DOCUMENT = "Confidential ledger of the northern farm"


def _beside(world, manager):
    """A conversation whose first turn typed _TYPED beside a pasted document, and an answer."""
    conv = manager.create_conversation(model="m").id
    content = _TYPED + "\n\n" + _DOCUMENT
    manager.add_message(conv, "user", content, origin="typed",
                        segments=[[0, len(_TYPED), "typed"], [len(_TYPED) + 2, len(content), "document"]])
    manager.add_message(conv, "assistant", "Your answer mentions irrigation pumps.", origin="assistant", context=[],
                        lineage=[])
    return conv


def test_tt35_the_model_is_handed_the_words_the_rule_keeps_and_nothing_else(tmp_path):
    world, restore = _refining(tmp_path / "cut", max_input_chars=40)
    try:
        manager = world.mod.ConversationManager(db_path=world.db, publish_to_sync=False)
        conv = _beside(world, manager)
        assert world.refiner.maybe_refine(manager, conv, runner=_drained(world)) is True
        handed, size = [], 0
        for word in world.titles.kept_words(_TYPED, world.cfg.derive):
            if size + len(word) + bool(handed) > 40:
                break
            handed.append(word)
            size += len(word) + (len(handed) > 1)
        assert world.backend.calls[0]["messages"][1] == {"role": "user", "content": " ".join(handed)}
        assert len(" ".join(handed)) <= 40 < len(" ".join(world.titles.kept_words(_TYPED, world.cfg.derive)))
    finally:
        restore()
    world, restore = _refining(tmp_path / "whole")
    try:
        manager = world.mod.ConversationManager(db_path=world.db, publish_to_sync=False)
        conv = _beside(world, manager)
        assert world.refiner.maybe_refine(manager, conv, runner=_drained(world)) is True
        (call,) = world.backend.calls
        system, user = call["messages"]
        assert system == {"role": "system", "content": world.cfg.refine.system_prompt}
        assert user == {"role": "user", "content": " ".join(world.titles.kept_words(_TYPED, world.cfg.derive))}
        assert len(user["content"]) < world.cfg.refine.max_input_chars, "nothing here is past the cut"
        for absent in ("v2", "srv", "txt", "make", "Confidential", "ledger", "northern", "irrigation", "answer"):
            assert absent not in user["content"], absent
        assert call["think"] is False and call["model"] == "m"
        assert call["options"] == {"temperature": world.cfg.refine.temperature, "seed": world.cfg.refine.seed,
                                   "num_predict": world.cfg.refine.num_predict, "num_ctx": 2048,
                                   "timeout": world.cfg.refine.timeout_s}
        # A turn that names a password titles nothing: nothing of it is ever handed.
        _manager, named = _titled(world, "my password is hunter so plan the garden", manager=manager)
        assert world.refiner.maybe_refine(manager, named, runner=_drained(world)) is False
        assert len(world.backend.calls) == 1
        # The title read again right before the call: its turn withdrawn, the conversation deleted by the store, its
        # turn deleted -- the words are never handed.
        idle = []
        withdrawn = _beside(world, manager)
        moved = _beside(world, manager)
        _typed(manager, moved, "Then draw a chart of the beds")
        deleted = _beside(world, manager)
        _manager, retried = _titled(world, "Prune the apple trees in winter", manager=manager)
        for each in (withdrawn, moved, deleted, retried):
            assert world.refiner.maybe_refine(manager, each, runner=lambda m, c: idle.append(c)) is True
        assert manager.withdraw_source("document:" + hashlib.sha256(_DOCUMENT.encode("utf-8")).hexdigest()) >= 1
        assert manager.title_state(moved)[2] == "Then draw a chart of the beds", "the title moved to the next turn"
        manager.delete_conversation(deleted)
        assert manager.delete_last_message(retried) is True and manager.delete_last_message(retried) is True
        world.refiner._drain(manager, world.cfg)
        assert len(world.backend.calls) == 1 and world.refiner.counts().get("stale") == 4
        assert len(world.governance.handed) == 4, "each admission no call holds is handed back"
        # A re-read that fails is no reading: the words are not handed.
        _manager, flaky = _titled(world, "Repot the lemon tree in spring", manager=manager)
        real = manager.title_state
        assert world.refiner.maybe_refine(manager, flaky, runner=lambda m, c: idle.append(c)) is True

        def failing(conv_id):
            raise sqlite3.OperationalError("the store cannot be read")

        manager.title_state = failing
        world.refiner._drain(manager, world.cfg)
        manager.title_state = real
        assert len(world.backend.calls) == 1 and world.refiner.counts().get("stale") == 5
    finally:
        restore()


# ---------------------------------------------------------------------------
# TT36 -- the refinement's wiring on the real package's shape
# ---------------------------------------------------------------------------
def _calls(node, name):
    return [call for call in ast.walk(node) if isinstance(call, ast.Call)
            and (getattr(call.func, "id", None) == name or getattr(call.func, "attr", None) == name)]


def test_tt36_the_refinement_is_wired_on_the_real_package_shape():
    chat = _module("opti_oignon/api/routes_chat.py")
    for name in ("_stream_turn", "_stream_chat_coding"):
        function = next(node for node in ast.walk(chat) if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
        sends = 0
        for block in [node for node in ast.walk(function) if isinstance(getattr(node, "body", None), list)]:
            for before, statement in zip(block.body, block.body[1:]):
                done = [call for call in _calls(statement, "_send_token")
                        if len(call.args) > 1 and getattr(call.args[1], "value", None) == "done"]
                if done and isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Await):
                    sends += 1
                    refine = _calls(before, "_refine_title")
                    assert isinstance(before, ast.Expr) and refine and [a.id for a in refine[0].args] == \
                        ["conversation_id"], f"{name}: the refinement is handed the conversation right before done"
        assert sends == 1, name
    hand = next(node for node in chat.body if isinstance(node, ast.FunctionDef) and node.name == "_refine_title")
    (dispatch,) = _calls(hand, "maybe_refine")
    assert [a.id for a in dispatch.args] == ["conversation_manager", "conversation_id"]
    assert dispatch.func.value.id == "_title_refiner"
    for route in ("delete_conversation", "rename_conversation"):
        body = _route_body(route)
        cancels = _calls(body, "_cancel_refinement")
        assert len(cancels) == 1 and [a.id for a in cancels[0].args] == ["conv_id"], route
    rename = _route_body("rename_conversation")
    lines = {name: _calls(rename, name)[0].lineno for name in ("_cancel_refinement", "rename_conversation")}
    assert lines["_cancel_refinement"] < lines["rename_conversation"], "cancelled before the user's title lands"
    refiner = _module("opti_oignon/title_refiner.py")
    imported = [node for node in refiner.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert {alias.name for node in imported if isinstance(node, ast.Import) for alias in node.names} <= {
        "collections", "importlib", "logging", "threading", "time"}
    assert [(node.module, node.level, [a.name for a in node.names]) for node in imported
            if isinstance(node, ast.ImportFrom)] == [("dataclasses", 0, ["dataclass", "field"]),
                                                      (None, 1, ["conversation_titles"])]
    backends = _module("opti_oignon/inference_backend.py")
    base = next(node for node in backends.body if isinstance(node, ast.ClassDef) and node.name == "InferenceBackend")
    assert any(isinstance(node, ast.AnnAssign) and node.target.id == "forwards_requests" and node.value.value is False
               for node in base.body), "a backend's requests stay where its endpoint says, unless it says otherwise"
    core = next(node for node in _module("opti_oignon/core_client.py").body
                if isinstance(node, ast.ClassDef) and node.name == "RemoteCoreBackend")
    assert any(isinstance(node, ast.Assign) and node.targets[0].id == "forwards_requests" and node.value.value is True
               for node in core.body), "the core daemon's backend says it forwards"
    public = next(node for node in backends.body if isinstance(node, ast.FunctionDef) and node.name ==
                  "is_on_this_machine")
    (returned,) = [node for node in ast.walk(public) if isinstance(node, ast.Return)]
    assert returned.value.func.id == "_is_on_this_machine" and returned.value.args[0].id == "url"


# ---------------------------------------------------------------------------
# TT37 -- no word of a turn or of an answer reaches the log
# ---------------------------------------------------------------------------
def test_tt37_no_word_of_a_turn_or_of_an_answer_reaches_the_log(tmp_path, caplog):
    cases = (("Garden layout for spring", None), ("Pumpkin secretword", None),
             ("ignored", RuntimeError("boom pumpkin secretword garden")))
    for n, (answer, error) in enumerate(cases):
        governance = _Governance()
        backend = _Backend(governance, answer=answer, error=error)
        world, restore = _refining(tmp_path / f"case{n}", backend=backend, governance=governance)
        try:
            with caplog.at_level(logging.DEBUG):
                manager, conv = _titled(world)
                world.refiner.maybe_refine(manager, conv, runner=_drained(world))
            assert len(backend.calls) == 1
        finally:
            restore()
    assert "title refinement: committed" in caplog.text and "title refinement: guard_refused" in caplog.text
    assert "title refinement: error" in caplog.text
    for word in ("garden", "layout", "spring", "raised", "pumpkin", "secretword", "boom", "plan"):
        assert word not in caplog.text.lower(), word


# ---------------------------------------------------------------------------
# TT38 -- the refine section is read exactly, and a machine sets its switch and model in local.yaml
# ---------------------------------------------------------------------------
def test_tt38_the_refine_section_is_read_exactly_and_a_machine_sets_its_own_switch(tmp_path, caplog):
    mod, loaded, restore = _load(tmp_path)
    try:
        titles = loaded[_TITLES]
        shipped = yaml.safe_load(_TITLES_YAML.read_text(encoding="utf-8"))
        assert set(shipped["refine"]) == set(titles._REFINE)
        refine = titles.read_config(shipped).refine
        assert refine.enabled is False, "off as shipped"
        assert refine.function_words >= {"the", "and", "de", "pour"} and refine.answer_labels == {"title", "titre"}
        for key, kind in titles._REFINE.items():
            if kind is int:
                low = 0 if key in titles._MAY_BE_ZERO else 1
                with pytest.raises(titles.TitlesConfigError, match=f"{key} is out of range"):
                    titles.read_config({**shipped, "refine": {**shipped["refine"], key: low - 1}})
        for key, value, word in (("temperature", 2.5, "temperature is out of range"),
                                 ("temperature", 1, "temperature is not a float"),
                                 ("keep_alive", "  ", "keep_alive is empty"), ("system_prompt", "", "system_prompt"),
                                 ("function_words", ["the", "5"], "function_words"),
                                 ("function_words", ["the", True], "function_words"),
                                 ("answer_labels", ["title:"], "answer_labels"), ("enabled", "yes", "enabled"),
                                 ("model", None, "model"), ("surprise", 1, "surprise")):
            with pytest.raises(titles.TitlesConfigError, match=word):
                titles.read_config({**shipped, "refine": {**shipped["refine"], key: value}})
        with pytest.raises(titles.TitlesConfigError, match="refine"):
            titles.read_config({k: v for k, v in shipped.items() if k != "refine"})
        # local.yaml: only a YAML boolean switches, only a name that is not blank names the model.
        base = titles.read_config(shipped).refine
        assert titles.local_refine({"refine": {"enabled": True, "model": " small "}}, base)[0] == \
            dataclasses.replace(base, enabled=True, model="small")
        assert titles.local_refine(None, base) == (base, [])
        for entry, reason in (({"refine": {"enabled": "yes"}}, "no boolean"), ({"refine": {"model": "  "}}, "no name"),
                              ({"refine": {"enabled": True, "speed": 2}}, "speed"), ({"other": 1}, "other"),
                              ([1], "no mapping for titles"), ({"refine": [1]}, "no mapping for titles.refine")):
            settings, reasons = titles.local_refine(entry, base)
            assert any(reason in said for said in reasons), entry
            assert settings.enabled is (entry == {"refine": {"enabled": True, "speed": 2}}), entry
        # The store lays the machine's file over titles.yaml, once, and says what it set nothing for.
        folder = tmp_path / "config"
        folder.mkdir()
        (folder / "titles.yaml").write_text(_TITLES_YAML.read_text(encoding="utf-8"), encoding="utf-8")
        mod._TITLES_PATH = folder / "titles.yaml"
        for local, enabled, said in (("titles: {refine: {enabled: true, model: small}}\n", True, None),
                                     ("titles: {refine: {enabled: 1}}\n", False, "holds no boolean"),
                                     ("titles: [\n", False, "does not parse"),
                                     ("- 1\n", False, "holds no mapping"),
                                     ("onion: {enabled: true}\n", False, None)):
            (folder / "local.yaml").write_text(local, encoding="utf-8")
            mod._titles_read.update(done=False, config=None)
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                cfg = mod._titles_config()
            assert cfg.refine.enabled is enabled, local
            assert (said is None and "local.yaml" not in caplog.text) or (said in caplog.text), local
            assert "small" not in caplog.text, "said by name, never with the file's content"
        assert cfg.enabled is True, "a machine's file never turns rule (a) off"
    finally:
        restore()
