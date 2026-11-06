"""The Mycelium's vectors: one encoder at a time, asked only through the inference registry.

The encoder and its card (the prefixes its model card asks for, the dimension of its vectors) come from
``config/mycelium.yaml``. A pass asks the resource governor before every batch, as background work (the caller
``index``), and holds the ticket it was given around the call, so the engine's own gate accounts that load instead
of admitting it again; a batch held by the governor stops the pass, what it embedded kept, and a refusal of the
governor that no wait lifts stops it as missing. A request the engine refuses as a request (HTTP 400, 413 or 422:
a text it will not take) is cut in two down to the text it refuses, which is counted and left without a vector;
any other failure -- a model the engine does not serve, an engine that fails or does not answer, an answer that
cannot be read -- stops the pass at that request, by name, as does a batch whose every text is refused or an
engine with no embedding endpoint; a busy engine (429, 503), or its own gate refusing as the governor would,
holds it. A vector of another dimension than the card's, or that is not numbers, is
refused by name; one that is not finite, or that points nowhere, leaves its text without a vector. Vectors are
kept as unit vectors in float16, normalized in double precision first.

A question is embedded the same way as interactive work (the caller ``chat``): held, refused or missing, the search
answers by its other stages and says why."""
import hashlib
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .store import MAX_VALUE, encoder_value

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

_CONFIG = Path(__file__).resolve().parent.parent / "config" / "mycelium.yaml"
CALLER = "index"
QUERY_CALLER = "chat"
HEADERS = ("full", "path", "none")
# The statuses by which an engine refuses a request for what it holds (a text too long for the model): only these
# are cut down to the text; any other failure stops the pass. A busy engine holds it, as the governor does.
REFUSED = (400, 413, 422)
BUSY = (429, 503)


class VectorError(RuntimeError):
    """The vectors cannot be had as asked."""


class Held(VectorError):
    """The resource governor held the request."""


class EncoderMissing(VectorError):
    """The engine does not serve the encoder, has no embedding endpoint, or refused a whole batch."""


@dataclass(frozen=True)
class Encoder:
    model: str
    dim: int
    query_prefix: str = ""
    text_prefix: str = ""

    def card(self):
        return asdict(self)

    @property
    def key(self):
        """The name its vectors are kept under: the model and its card, so a changed prefix embeds again."""
        import json

        return f"{self.model}#{text_sha(json.dumps(self.card(), sort_keys=True))[:12]}"


@dataclass(frozen=True)
class VectorSettings:
    enabled: bool
    encoder: Encoder
    header: str
    batch: int
    timeout_s: float


class Section:
    """A section of ``mycelium.yaml``, read strictly: a file that is not YAML, not a mapping, or a value of another
    type than its own (a quoted "false" is no flag, a flag no number) is refused by name, as ``error``."""

    def __init__(self, name, path=None, error=None):
        import yaml

        self.error = error or VectorError
        try:
            with open(path or _CONFIG, encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise self.error(f"mycelium.yaml: cannot be read as YAML ({type(exc).__name__})") from exc
        if loaded is None:
            loaded = {}
        raw = loaded.get(name) if isinstance(loaded, dict) else None
        if not isinstance(raw, dict):
            raise self.error(f"mycelium.yaml: a mapping with a {name} mapping is expected")
        self.raw = raw

    def typed(self, value, kinds, what):
        if isinstance(value, bool) and bool not in kinds or not isinstance(value, kinds):
            raise self.error(f"mycelium.yaml: {what} is not of type {' or '.join(k.__name__ for k in kinds)}")
        return value

    def positive(self, value, kinds, what, most=None):
        value = self.typed(value, kinds, what)
        try:
            fits = math.isfinite(value) and value > 0 and (most is None or value <= most)
        except OverflowError:       # an integer past a float's range
            fits = False
        if not fits:
            bound = "" if most is None else f" at most {most}"
            raise self.error(f"mycelium.yaml: {what} is a positive number{bound} a float holds, not "
                             f"{str(value)[:40]}")
        return value

    def text(self, value, what):
        return "" if value is None else self.typed(value, (str,), what)

    def only(self, *keys, raw=None, where=None):
        """Refuses by name a key the code does not read (a misspelled ``timout_s`` would fall back unseen)."""
        held = self.raw if raw is None else raw
        unread = sorted(str(key) for key in held if key not in keys)
        if unread:
            raise self.error(f"mycelium.yaml: {where or 'this section'} holds {', '.join(unread)}, which no code "
                             f"reads")


def load_settings(path=None):
    """The vectors' settings from ``mycelium.yaml``, refused by name when malformed (``Section``): a dimension,
    batch or timeout that is not a positive number, a header the indexer does not know, a card longer than a store
    keeps."""
    section = Section("vectors", path)
    section.only("enabled", "encoder", "header", "batch", "timeout_s", "encoders", where="vectors")
    raw, typed, positive = section.raw, section.typed, section.positive
    name = typed(raw.get("encoder"), (str,), "vectors.encoder")
    cards = typed(raw.get("encoders"), (dict,), "vectors.encoders")
    card = typed(cards.get(name), (dict,), f"the card of {name}")
    section.only("dim", "query_prefix", "text_prefix", raw=card, where=f"the card of {name}")
    encoder = Encoder(name, positive(card.get("dim"), (int,), f"the dim of {name}"),
                      section.text(card.get("query_prefix"), f"the query_prefix of {name}"),
                      section.text(card.get("text_prefix"), f"the text_prefix of {name}"))
    header = typed(raw.get("header", "full"), (str,), "vectors.header")
    if header not in HEADERS:
        raise VectorError(f"mycelium.yaml: vectors.header is one of {', '.join(HEADERS)}, not {header!r}")
    settings = VectorSettings(typed(raw.get("enabled", True), (bool,), "vectors.enabled"), encoder, header,
                              positive(raw.get("batch", 32), (int,), "vectors.batch"),
                              float(positive(raw.get("timeout_s", 120), (int, float), "vectors.timeout_s")))
    if len(encoder_value(encoder.key, encoder.card())) > MAX_VALUE:
        # The card is kept in the store's meta, whose values are bounded: refused here, never halfway through a pass.
        raise VectorError(f"mycelium.yaml: the card of {name} (its name and prefixes) is longer than a store keeps")
    return settings


def header(path, kind, symbol, module_doc):
    """The context a part is embedded with: where it is, what it is, what its module is for (a Python module's
    first sentence only)."""
    lines = [path, f"{kind.split('~')[0]} {symbol}"]
    if module_doc and path.endswith(".py"):
        lines.append(module_doc)
    return "\n".join(lines)


def embedded_text(mode, path, kind, symbol, module_doc, body):
    """What is embedded for a part: its lines, under the header ``mode`` asks for."""
    if mode == "full":
        return f"{header(path, kind, symbol, module_doc)}\n{body}"
    if mode == "path":
        return f"{path}\n{body}"
    if mode == "none":
        return body
    raise VectorError(f"a header is one of {', '.join(HEADERS)}, not {mode!r}")


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unit(vector):
    """The unit vector, in double precision: scaled by its largest component first, so a finite vector of large
    components never overflows into infinities or NaNs (a NaN or an infinity stays one; zeros stay zeros)."""
    import numpy as np

    v = np.asarray(vector, dtype=np.float64)
    largest = float(np.max(np.abs(v))) if v.size else 0.0
    if largest > 0:
        v = v / largest
    return v / (np.linalg.norm(v) or 1.0)


def to_blob(vector):
    """The unit vector in float16."""
    import numpy as np

    return unit(vector).astype(np.float16).tobytes()


def from_blobs(blobs, dim):
    import numpy as np

    if not blobs:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack([np.frombuffer(b, dtype=np.float16) for b in blobs]).astype(np.float32)


def _registry():
    from opti_oignon.inference_backend import get_backend_registry

    return get_backend_registry()


def _governor():
    """The resource governor, or None when there is none to ask (the request then goes as every funnel's does)."""
    try:
        from opti_oignon.resource_governor import get_resource_governor

        return get_resource_governor()
    except Exception:  # noqa: BLE001 - absence is an answer here
        return None


@contextmanager
def _holding(decision):
    if decision is None:
        yield
        return
    from opti_oignon.resource_governor import ticket_scope

    with ticket_scope(decision):
        yield


def _status(exc):
    """The HTTP status a failure carries (the engine client's ``status_code``, or its response's), or None."""
    for holder in (exc, getattr(exc, "response", None)):
        code = getattr(holder, "status_code", None)
        if isinstance(code, int) and not isinstance(code, bool) and code > 0:
            return code
    return None


def _final(decision):
    """Whether the governor refuses for good (no wait lifts it): its own reading of its decision."""
    from opti_oignon import resource_governor

    return resource_governor.refusal_is_final(decision)


def unreachable(exc):
    """Whether a failure says the engine was not reached (a timeout, a refused or lost connection), not that it
    refused a text."""
    return isinstance(exc, (TimeoutError, ConnectionError)) or any(
        word in type(exc).__name__ for word in ("Timeout", "Connect"))


def engine_name(backend, model):
    """The name the engine serving ``model`` knows it by, and the governor sees it loaded under: Ollama resolves a
    name with no tag to its ``:latest`` and shows every model it holds with its tag (the tag follows the last colon
    after the last slash, so a registry's port is no tag); another engine's names are its own. Asked under the
    settings' name, an encoder Ollama holds was no resident to the governor, its cost unknown, and the pass refused
    for good as background work."""
    if str(getattr(backend, "name", "")) != "ollama" or ":" in model.rsplit("/", 1)[-1]:
        return model
    return model + ":latest"


class Embedder:
    def __init__(self, encoder, timeout_s=120.0):
        self.encoder = encoder
        self.timeout_s = timeout_s

    def _backend(self):
        try:
            backend = _registry().resolve_backend(self.encoder.model)
        except Exception as exc:  # noqa: BLE001 - a registry that cannot answer serves nothing
            raise EncoderMissing(f"no backend serves {self.encoder.model}: {exc}") from exc
        if backend is None:
            raise EncoderMissing(f"no backend serves {self.encoder.model}")
        return backend

    def _admit(self, caller, name, engine):
        governor = _governor()
        if governor is None:
            return None
        try:
            decision = governor.admit(name, caller=caller, engine=engine)
        except Exception as exc:  # noqa: BLE001 - an admission that fails sends nothing, as the warm-up's
            raise Held(f"the admission of {self.encoder.model} failed: {exc}") from exc
        if decision is not None and not getattr(decision, "admitted", True):
            reason = getattr(decision, "reason", "")
            if _final(decision):
                raise EncoderMissing(f"the governor cannot admit {self.encoder.model}, and no wait lifts it: {reason}")
            raise Held(f"{self.encoder.model} held by the governor: {reason}")
        return decision

    def _ask(self, backend, texts, caller):
        # The governor and the engine are asked under one name, the engine's: the call is made under the very
        # ticket the admission gave, which the engine's gate matches by name. The governor is told the engine: its
        # cache of the registry's routes holds the settings' name, resolved by ``_backend``, not the engine's.
        name = engine_name(backend, self.encoder.model)
        decision = self._admit(caller, name, str(getattr(backend, "name", "")) or None)
        with _holding(decision):
            found = backend.embed_many(name, list(texts), timeout=self.timeout_s)
        if found is None:
            raise EncoderMissing(f"the backend of {self.encoder.model} has no embedding endpoint")
        return found

    def _checked(self, vector):
        """The unit vector the engine's vector points to, as floats; None when one is not finite or it points
        nowhere (no unit vector to keep); a vector that is not numbers, or of another dimension than the card's, is
        refused by name."""
        import numpy as np

        try:
            values = [float(x) for x in vector]
        except (TypeError, ValueError) as exc:
            raise EncoderMissing(f"{self.encoder.model} gave a vector that is not numbers") from exc
        if len(values) != self.encoder.dim:
            raise EncoderMissing(f"{self.encoder.model} gave {len(values)} dimensions, its card says "
                                 f"{self.encoder.dim}")
        # Kept as the store keeps it: a NaN or an infinity never makes a finite unit vector, nor do zeros.
        direction = unit(values)
        kept = direction.astype(np.float16)
        return direction.tolist() if np.isfinite(kept).all() and kept.any() else None

    def texts(self, texts):
        """One vector or None per text, in order, with the count of texts left without one; a batch of several
        texts none of which has a vector stops the pass (an engine that refuses everything is not refusing a
        text)."""
        backend = self._backend()
        out = self._cut(backend, [self.encoder.text_prefix + t for t in texts])
        if len(out) > 1 and all(v is None for v in out):
            raise EncoderMissing(f"{self.encoder.model} gave no vector for any text of a batch of {len(out)}")
        return out, sum(v is None for v in out)

    def _cut(self, backend, texts):
        try:
            return [self._checked(v) for v in self._ask(backend, texts, CALLER)]
        except VectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - only a refused request is cut down to its text
            status = _status(exc)
            if status not in REFUSED:
                # Cutting in two asks 2N - 1 times, each its timeout: only a text the engine refuses is worth it.
                raise self._stopped(exc, status) from exc
            if len(texts) == 1:
                return [None]
            half = len(texts) // 2
            return self._cut(backend, texts[:half]) + self._cut(backend, texts[half:])

    def _stopped(self, exc, status):
        """What stops the pass: Held when a wait may lift it (a busy engine, its gate refusing as the governor
        would), EncoderMissing otherwise."""
        decision = getattr(exc, "decision", None)
        if decision is not None and not getattr(decision, "admitted", True):
            if _final(decision):
                return EncoderMissing(f"the engine's gate cannot admit {self.encoder.model}, and no wait lifts it: "
                                      f"{exc}")
            return Held(f"the engine's gate holds {self.encoder.model}: {exc}")
        if status in BUSY:
            return Held(f"the engine is busy ({status}): {exc}")
        return EncoderMissing(self._failure(exc, status))

    def _failure(self, exc, status):
        model = self.encoder.model
        if status == 404:
            return f"the engine does not serve {model}: {exc}"
        if unreachable(exc):
            return f"{model} did not answer: {exc}"
        return f"{model} failed ({status or type(exc).__name__}): {exc}"

    def query(self, text):
        """The question's vector; Held or EncoderMissing say why there is none."""
        try:
            found = self._ask(self._backend(), [self.encoder.query_prefix + text], QUERY_CALLER)
            vector = self._checked(found[0]) if found else None
        except VectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - the chat never sees the engine's failure, only its silence
            raise EncoderMissing(f"{self.encoder.model} did not answer the question: {exc}") from exc
        if vector is None:
            raise EncoderMissing(f"{self.encoder.model} gave no finite vector for the question")
        return vector
