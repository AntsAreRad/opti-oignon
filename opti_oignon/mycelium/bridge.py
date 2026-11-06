"""Words for the full text, from a local model: a French question put into English, any question put into the
words its code would use.

The full text and the symbol table match words, so a French question finds little in an English repository, and a
question in a person's words little in a code base's; the vectors read meaning in any language and take the
question as typed. Both are asked of one model (``bridge.model`` in ``config/mycelium.yaml``) through the inference
registry only, on this machine only (a backend that sends elsewhere, or cannot say where, is never asked, in any
mode; a backend on this machine that serves a model from elsewhere -- a proxy, a cloud model -- is judged by its own
address: naming such a model is sending there), admitted by the resource governor as interactive work and sent
under that admission's ticket, at the context and with the keep_alive the admission names (none of the bridge's
own: a resident model is never asked at another context, which would load it again), the question bounded. A
call's resolution and answer share the settings' timeout, given up past it whatever the backend does, and while a
call given up lives no other call from the Mycelium to that model starts (the chat's and the titles' are their
own). A model's reasoning (a ``<think>``
block) is never taken for its answer. A model away, held, refusing, failing, hung or answering nothing leaves the
question as typed (and no second list); ``silent`` -- the caller's, when it hands one -- says why. Each question's
words are kept, a bounded number of questions, so a question asked again asks nothing."""
import importlib
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass

# The registry and the governor are reached through the vectors' seams, by the module: one place for both.
from . import vectors

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

CALLER = "chat"
# The option a backend reads one call's deadline from (``inference_backend.TIMEOUT_OPTION``).
TIMEOUT_OPTION = "timeout"
# French words, without their accents (an accent is counted apart): two of them, or one and an accent, make a
# question French. See ``words`` for what a word is.
FRENCH = frozenset("""le la les un une des du de et est sont au aux ce cette ces qui que qu quoi quel quelle quels
quelles comment pourquoi quand ou il elle ils elles je tu nous vous pour par sur dans avec sans pas ne rien cela
fait peut mon ma mes notre leur si alors donc mais""".split())
APOSTROPHES = ("'", chr(0x2019))
WORD = re.compile(r"[^\W\d_]+")
TRANSLATE = ("You are a professional French (fr) to English (en) translator. Your goal is to accurately convey the "
             "meaning and nuances of the original French text while adhering to English grammar, vocabulary, and "
             "cultural sensitivities.\nProduce only the English translation, without any additional explanations or "
             "commentary. Keep code identifiers, file names and setting names exactly as written. Please translate "
             "the following French text into English:\n\n\n{text}")
REWRITE = ("A user asks the question below about a software project. Write the words a developer would type into "
           "a code search to find the code that answers it: names the functions, variables or settings would likely "
           "have, and the words of the error or log message the user would see. English, one line, at most 15 words, "
           "nothing else.\n\nQuestion: {text}")
# The most questions a bridge may keep, and the most seconds a call may wait: bounds on what the settings name. The
# most characters of a question a prompt holds.
MAX_CACHE = 4096
MAX_TIMEOUT_S = 600
MAX_QUESTION = 2000
# The calls given up at their timeout, by model, while their backend has not returned.
_hung = {}
_lock = threading.Lock()


class AskError(RuntimeError):
    """The model could not be asked, or gave nothing."""


def _plain(word):
    return "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")


def words(text):
    """The words of a question, for telling its language: letters only; a chunk holding an underscore or a digit is
    code, no word; an apostrophe splits ("qu'est"); a hyphen joins words only when each part is a French word
    ("est-ce", never "de-serialize"); an acronym is no word."""
    for chunk in text.split():
        if "_" in chunk or any(c.isdigit() for c in chunk):
            continue
        for apostrophe in APOSTROPHES[1:]:
            chunk = chunk.replace(apostrophe, APOSTROPHES[0])
        for piece in chunk.split(APOSTROPHES[0]):
            parts = WORD.findall(piece)
            if len(parts) > 1 and not all(_plain(part).lower() in FRENCH for part in parts):
                continue
            yield from (part for part in parts if not (len(part) > 1 and part.isupper()))


def is_french(text):
    words_ = list(words(text))
    hits = sum(1 for w in words_ if _plain(w).lower() in FRENCH)
    accents = sum(1 for w in words_ for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) == "Mn")
    return hits >= 2 or (hits >= 1 and accents >= 1)


def answer_of(text):
    """What a model answered, without its reasoning: what follows a ``</think>`` (the block closed, or opened by
    the chat template in the prompt itself); an open block with no close is no answer."""
    text = text.strip()
    end = text.lower().rfind("</think>")
    if end >= 0:
        return text[end + len("</think>"):].strip()
    return "" if text[:7].lower() == "<think>" else text


def _hanging(model):
    """Whether a call given up for ``model`` still lives; those that returned are forgotten."""
    with _lock:
        hanging = {thread for thread in _hung.get(model, ()) if thread.is_alive()}
        _hung[model] = hanging
    return bool(hanging)


class Unstarted(RuntimeError):
    """The thread of a call could not start: the call never happened."""


def _within(seconds, model, work):
    """``(finished, value, error)``: ``work()`` on a thread of its own, waited for ``seconds``; past them it is given
    up and kept among the model's hung calls. A thread that cannot start is an error (``Unstarted``)."""
    out = {}

    def run():
        try:
            out["value"] = work()
        except BaseException as exc:  # noqa: BLE001 - handed back below
            out["error"] = exc

    worker = threading.Thread(target=run, name="oo-mycelium-ask", daemon=True)
    try:
        worker.start()
    except RuntimeError as exc:
        return True, None, Unstarted(f"its thread could not start ({exc})")
    worker.join(seconds)
    if worker.is_alive():
        with _lock:
            _hung.setdefault(model, set()).add(worker)
        return False, None, None
    return True, out.get("value"), out.get("error")


def _hand_back(governor, decision):
    """An admission whose call never happened, handed back to the governor (its pending load ended)."""
    handed = getattr(governor, "end_pending_load", None)
    ticket = getattr(decision, "ticket_id", None)
    if decision is not None and callable(handed) and ticket:
        try:
            handed(ticket)
        except Exception:  # noqa: BLE001 - the pending load expires by itself
            pass


def _on_this_machine(backend):
    """Whether ``backend`` says its requests stay on this machine: in the process, or to a loopback or unspecified
    address (``inference_backend.is_on_this_machine``). A backend that forwards its requests, or cannot say, is
    elsewhere."""
    try:
        backends = importlib.import_module("opti_oignon.inference_backend")
        if getattr(backend, "forwards_requests", True) is not False:
            return False
        endpoint = backend.endpoint()
        if not isinstance(endpoint, str) or not endpoint:
            return False
        return endpoint == backends.ENDPOINT_IN_PROCESS or bool(backends.is_on_this_machine(endpoint))
    except Exception:  # noqa: BLE001 - what cannot be told is elsewhere
        return False


@dataclass(frozen=True)
class BridgeSettings:
    model: str
    translate: bool
    rewrite: bool
    timeout_s: float
    cache: int


def load_settings(path=None):
    """The bridge's settings from ``mycelium.yaml``, refused by name (ValueError) when malformed: a model that is
    no name, a flag that is not one, a timeout that is not a positive number of seconds (at most
    ``MAX_TIMEOUT_S``), a cache that is not a count (at most ``MAX_CACHE``)."""
    section = vectors.Section("bridge", path, error=ValueError)
    section.only("model", "translate", "rewrite", "timeout_s", "cache", where="bridge")
    raw, typed = section.raw, section.typed
    model = typed(raw.get("model"), (str,), "bridge.model")
    if not model.strip():
        raise ValueError("mycelium.yaml: bridge.model names no model")
    cache = typed(raw.get("cache", 256), (int,), "bridge.cache")
    if not 0 <= cache <= MAX_CACHE:
        raise ValueError(f"mycelium.yaml: bridge.cache is a count from 0 to {MAX_CACHE}, not {str(cache)[:40]}")
    return BridgeSettings(model, typed(raw.get("translate", True), (bool,), "bridge.translate"),
                          typed(raw.get("rewrite", True), (bool,), "bridge.rewrite"),
                          float(section.positive(raw.get("timeout_s", 30), (int, float), "bridge.timeout_s",
                                                 MAX_TIMEOUT_S)), cache)


class Asker:
    """One model, asked through the registry under the governor; ``ask`` gives its text or raises AskError."""

    def __init__(self, model, timeout_s, caller=CALLER):
        self.model, self.timeout_s, self.caller = model, timeout_s, caller

    def ask(self, prompt, num_predict, system=None, timeout_s=None):
        """The model's answer, within ``timeout_s`` (the asker's own when None) for the resolution and the call
        together."""
        model = self.model
        seconds = self.timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + seconds
        if _hanging(model):
            raise AskError(f"a call to {model} given up still lives: nothing more is asked of it until it returns")
        finished, backend, error = _within(seconds, model, lambda: vectors._registry().resolve_backend(model))
        if not finished:
            raise AskError(f"no backend for {model} within {seconds:.3g} s: given up")
        if error is not None:
            raise AskError(f"no backend serves {model}: {error}") from error
        if backend is None:
            raise AskError(f"no backend serves {model}")
        if not _on_this_machine(backend):
            raise AskError(f"{model} is not on this machine: nothing is sent to it")
        if _hanging(model):     # a call given up while this one was resolved
            raise AskError(f"a call to {model} given up still lives: nothing more is asked of it until it returns")
        left = deadline - time.monotonic()
        if left <= 0:
            raise AskError(f"no time left to ask {model} within {seconds:.3g} s")
        decision = None
        governor = vectors._governor()
        if governor is not None:
            try:
                decision = governor.admit(model, caller=self.caller)
            except Exception as exc:  # noqa: BLE001 - an admission that fails sends nothing
                raise AskError(f"the admission of {model} failed: {exc}") from exc
            if decision is not None and not getattr(decision, "admitted", True):
                reason = getattr(decision, "reason", "")
                if vectors._final(decision):
                    raise AskError(f"the governor cannot admit {model}, and no wait lifts it: {reason}")
                raise AskError(f"{model} held by the governor: {reason}")
        left = max(deadline - time.monotonic(), 0.001)
        options = {"temperature": 0.0, "seed": 0, "num_predict": num_predict, TIMEOUT_OPTION: left}
        admitted_ctx = getattr(decision, "num_ctx", None)
        if admitted_ctx:
            options["num_ctx"] = int(admitted_ctx)
        if getattr(backend, "name", "") == "llama_server":
            options["max_tokens"] = num_predict     # the bound a llama-server reads
        keep_alive = getattr(decision, "keep_alive", None)
        extra = {"keep_alive": keep_alive} if keep_alive else {}
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]

        def call():
            with vectors._holding(decision):
                return backend.generate(model, messages, options=options, think=False, **extra)

        finished, said, error = _within(left, model, call)
        if not finished:
            raise AskError(f"{model} gave no answer within {seconds:.3g} s: given up")
        if isinstance(error, Unstarted):
            _hand_back(governor, decision)
            raise AskError(f"the call to {model} could not start: {error}") from error
        if error is not None:
            raise AskError(f"{model} did not answer: {error}") from error
        text = answer_of(str(getattr(said, "content", "") or ""))
        if not text:
            raise AskError(f"{model} answered nothing")
        return text


class Bridge:
    def __init__(self, settings, asker=None):
        self.settings = settings
        self.asker = asker or Asker(settings.model, settings.timeout_s)
        self.silent = {}
        self._kept = OrderedDict()
        self._lock = threading.Lock()

    def _cached(self, kind, question, make, silent):
        key = (kind, question)
        with self._lock:
            if key in self._kept:
                self._kept.move_to_end(key)
                return self._kept[key]
        try:
            found = make()
        except AskError as exc:
            silent[kind] = str(exc)
            return None
        if self.settings.cache:
            with self._lock:
                self._kept[key] = found
                while len(self._kept) > self.settings.cache:
                    self._kept.popitem(last=False)
        return found

    def english(self, question, silent=None):
        """The question in English for the words-matching lists: as typed when it is not French, or when the
        model could not be asked (why in ``silent``; when none is handed, the bridge's own, which then holds this
        call's only)."""
        if silent is None:
            self.silent = silent = {}
        if not self.settings.translate or not is_french(question):
            return question
        asked = question[:MAX_QUESTION]
        found = self._cached("translate", asked, lambda: self.asker.ask(TRANSLATE.format(text=asked), 256), silent)
        return found or question

    def rewritten(self, question, silent=None):
        """The words the question's code would use (one line), or "" when there are none (why in ``silent``, as
        for ``english``)."""
        if silent is None:
            self.silent = silent = {}
        if not self.settings.rewrite:
            return ""
        asked = question[:MAX_QUESTION]
        found = self._cached("rewrite", asked,
                             lambda: self.asker.ask(REWRITE.format(text=asked), 48).split("\n")[0].strip(), silent)
        return found or ""
