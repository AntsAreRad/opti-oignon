"""The first passages of a search judged by a local model: does this passage answer the question?

Each of the first ``depth`` passages is shown to the model as a document (its path, its lines, at most
``max_chars`` of its text), with the question, in the judge's instruction of the Qwen3-Reranker card; one token is
read as a word: "yes" (or "oui"), "no" (or "non"), anything else unknown. The passages it says answer come first,
then the unknown, then the others, each group in the order the fusion gave it; the passages beyond ``depth`` follow
as they came. Each judged passage carries its verdict, and ``found`` says whether the first one is a "yes": False
only when every passage judged says no (then no passage answers, and the search can say so rather than hand on the
nearest), None when one could not be judged or the search gives passages past the depth. The judges of a search
share the settings' timeout. A verdict is asked as interactive work through the registry
(``bridge.Asker``: on this machine, bounded, the question cut), ``parallel`` at once; one that cannot be had is
unknown; none at all leaves the order as it came, and ``silent`` says why. A passage is data shown to a judge: what
it says can move it to the head of the order, and make ``found`` say a passage answers -- a judge said so, not the
source."""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import vectors
from .bridge import MAX_QUESTION, MAX_TIMEOUT_S, Asker, AskError

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note '
          'that the answer can only be "yes" or "no".')
INSTRUCT = ("Given a question about a software repository, judge whether this passage of its code or documentation "
            "answers it or is a place the answer comes from")
ORDER = {"yes": 0, "unknown": 1, "no": 2}
# Bounds on what the settings name: the passages judged (a search's depth), the judges at once, a document's size.
MAX_DEPTH = 100
MAX_PARALLEL = 16
MAX_CHARS = 20000


@dataclass(frozen=True)
class RerankSettings:
    model: str
    depth: int
    parallel: int
    max_chars: int
    timeout_s: float


def load_settings(path=None):
    """The reranker's settings from ``mycelium.yaml``, refused by name (ValueError) when malformed: a model that is
    no name, a depth, parallel or max_chars that is not a positive count within its bound, a timeout that is not a
    positive number of seconds."""
    section = vectors.Section("rerank", path, error=ValueError)
    section.only("model", "depth", "parallel", "max_chars", "timeout_s", where="rerank")
    raw, typed, positive = section.raw, section.typed, section.positive
    model = typed(raw.get("model"), (str,), "rerank.model")
    if not model.strip():
        raise ValueError("mycelium.yaml: rerank.model names no model")
    return RerankSettings(model, positive(raw.get("depth", 20), (int,), "rerank.depth", MAX_DEPTH),
                          positive(raw.get("parallel", 4), (int,), "rerank.parallel", MAX_PARALLEL),
                          positive(raw.get("max_chars", 3000), (int,), "rerank.max_chars", MAX_CHARS),
                          float(positive(raw.get("timeout_s", 60), (int, float), "rerank.timeout_s", MAX_TIMEOUT_S)))


YES, NO = ("yes", "oui"), ("no", "non")


def verdict_of(text):
    """The verdict a judge's answer gives, read as its first word (letters only: "**Yes**" is yes, "Not" is no
    verdict)."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    first = words[0] if words else ""
    return "yes" if first in YES else "no" if first in NO else "unknown"


class Reranker:
    def __init__(self, settings, asker=None):
        self.settings = settings
        self.asker = asker or Asker(settings.model, settings.timeout_s)
        self.silent = {}

    def _judge(self, question, document, deadline):
        left = deadline - time.monotonic()
        if left <= 0:
            return "unknown", f"the reranker's {self.settings.timeout_s:.3g} s were spent"
        prompt = f"<Instruct>: {INSTRUCT}\n<Query>: {question[:MAX_QUESTION]}\n<Document>: {document}"
        try:
            return verdict_of(self.asker.ask(prompt, 1, system=SYSTEM, timeout_s=left)), None
        except AskError as exc:
            return "unknown", str(exc)

    def __call__(self, question, hits, read, silent=None):
        """``hits`` (cited, best first) reordered; ``read(hit)`` gives a hit's text, or None when it cannot; why a
        verdict is missing goes to ``silent`` (the reranker's own when none is handed)."""
        if silent is None:
            self.silent = silent = {}
        head, tail = list(hits[:self.settings.depth]), list(hits[self.settings.depth:])
        if not head:
            return list(hits)
        documents = []
        for hit in head:
            text = read(hit)
            documents.append(None if text is None else
                             f"{hit['path']}:{hit['first']}-{hit['last']}\n{text}"[:self.settings.max_chars])
        jobs = [(i, d) for i, d in enumerate(documents) if d is not None]
        if not jobs:
            silent["rerank"] = "no passage could be read from its source"
            return list(hits)
        # The judges share one timeout: a judge that would start past it is not asked.
        deadline = time.monotonic() + self.settings.timeout_s
        try:
            with ThreadPoolExecutor(max_workers=min(self.settings.parallel, len(jobs))) as pool:
                judged = list(pool.map(lambda job: self._judge(question, job[1], deadline), jobs))
        except RuntimeError as exc:     # the judges' threads could not start
            silent["rerank"] = f"the judges could not start ({exc})"
            return list(hits)
        verdicts = ["unknown"] * len(head)
        failures = []
        for (i, _d), (verdict, failure) in zip(jobs, judged):
            verdicts[i] = verdict
            if failure:
                failures.append(failure)
        if failures and len(failures) == len(jobs):
            silent["rerank"] = failures[0]
            return list(hits)
        if failures:
            silent["rerank"] = f"{len(failures)} of {len(jobs)} passages unjudged: {failures[0]}"
        elif all(verdicts[i] == "unknown" for i, _d in jobs):
            silent["rerank"] = f"the verdicts could not be read: the model answered neither yes nor no ({len(jobs)})"
        placed = sorted(range(len(head)), key=lambda i: (ORDER[verdicts[i]], i))
        return [{**head[i], "verdict": verdicts[i], "why": head[i]["why"] + [f"rerank:{verdicts[i]}"]}
                for i in placed] + tail

    @staticmethod
    def found(hits):
        """Whether a passage was judged to answer: True when the first was, False when every passage given was
        judged and says no, None otherwise -- nothing judged, one that could not be, or passages past the depth
        never judged (the answer may be among them)."""
        if not hits or "verdict" not in hits[0]:
            return None
        first = hits[0]["verdict"]
        if first == "yes":
            return True
        return False if all(hit.get("verdict") == "no" for hit in hits) else None
