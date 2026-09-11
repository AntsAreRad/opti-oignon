#!/usr/bin/env python3
"""Baseline measurement of the memory mechanism as it stands.

Before the memory block changes anything it records what the current
mechanism does on corpora it names: what a set of natural queries recalls
through the real retriever, how much of the salience floor survives, how many
facts the prompt budget drops and whether the archive search finds them
again, and how far the rule-based compressor shrinks a conversation. Every
figure is produced by the code as it stands over recorder stores, with no
vector signal -- the keyword path alone -- so the run is deterministic and
reproduces to the digit. ``BASELINE`` records it; a contract requires the
live measurement to equal the record, so the record can only change when the
mechanism does, on purpose.

Two things this deliberately is not. It is not a claim about real recall
quality: the vector layer is absent here, and the model-backed compressor
strategies are measured on the host, in a runbook, or not at all. And it is
not pipeline: nothing on the chat path imports it, and a contract on the tree
says so. The heavier siblings it drives are imported inside the one function
that needs them, so importing this module costs nothing.
"""

from dataclasses import dataclass, field

checkpoint_before_apply = True

LONG_CORPUS = "episodic-long"


@dataclass(frozen=True)
class Corpus:
    """A named, deterministic set of facts, natural queries and turns."""

    name: str
    facts: list = field(default_factory=list)     # (id, text, category, use_count)
    queries: list = field(default_factory=list)   # (query, expected fact id)
    turns: list = field(default_factory=list)     # {"role", "content"}


class _Record:
    __slots__ = ("id", "text", "category", "source", "user_id", "created_at", "updated_at", "active", "use_count")

    def __init__(self, rid, text, category, use_count, stamp):
        self.id = rid
        self.text = text
        self.category = category
        self.source = "baseline"
        self.user_id = "local"
        self.created_at = stamp
        self.updated_at = stamp
        self.active = True
        self.use_count = use_count


class _Canonical:
    """A recorder canonical store: reads served from memory, writes recorded."""

    def __init__(self, records):
        self.records = list(records)
        self.touched = []

    def resolve_user(self, user_id=None):
        return "local"

    def list(self, *, active_only=True, user_id=None, **_kw):
        return list(self.records)

    def count(self, *, active_only=True, user_id=None):
        return len(self.records)

    def touch(self, fact_id, *, user_id=None):
        self.touched.append(fact_id)
        return True


class _NoVector:
    """The vector layer absent: no embedder, no neighbours, keyword recall only."""

    def embed(self, text):
        return None

    def find_similar(self, embedding, *, user_id=None, top_k=5, threshold=None):
        return []


def _estimate_tokens(text):
    """The retriever's own fallback estimate, restated so the ratio is stable."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))


def _identity():
    facts = [
        ("i01", "The user is called Alice and lives in Berlin", "identity", 9),
        ("i02", "Alice prefers concise answers in English", "preference", 8),
        ("i03", "Alice works at Contoso as a data engineer", "identity", 7),
        ("i04", "The main project is called Harvest and ships in May", "project", 5),
        ("i05", "Alice drinks tea, never coffee", "preference", 4),
        ("i06", "Bob is the reviewer for the Harvest release", "contact", 3),
        ("i07", "The release budget is 1200 euros", "fact", 2),
        ("i08", "Alice keeps notes in Markdown files", "preference", 2),
        ("i09", "The demo must not use Docker", "project", 1),
        ("i10", "Carol handles the Oslo customer account", "contact", 0),
    ]
    queries = [
        ("where does alice live", "i01"),
        ("what language does alice want answers in", "i02"),
        ("who reviews the harvest release", "i06"),
        ("what is the budget for the release", "i07"),
        ("which format are the notes kept in", "i08"),
        ("who handles the oslo account", "i10"),
    ]
    turns = [
        {"role": "user", "content": "I am Alice, I live in Berlin and I work at Contoso as a data engineer. Please keep answers concise and in English, I read them between meetings and long replies get skipped."},
        {"role": "assistant", "content": "Understood: concise answers in English. Berlin and Contoso are noted. Is there a project you want me to keep in mind so that later questions land in the right context?"},
        {"role": "user", "content": "The project is called Harvest and it ships in May with a budget of 1200 euros. Bob reviews the release and the demo must not use Docker because the venue has no container runtime."},
        {"role": "assistant", "content": "Noted: Harvest ships in May, 1200 euros, Bob reviews, no Docker for the demo. I will keep those constraints in view when you ask about the release plan or the demo environment."},
    ]
    return Corpus("identity", facts, queries, turns)


def _episodic_long():
    facts = []
    stamps = 0
    for i in range(1, 41):
        stamps += 1
        # Long on purpose: the corpus exists to exceed the prompt budget, so
        # that the budget cut and the recovery path are exercised, not assumed.
        text = (
            f"On day {i} the team reviewed the migration of service number {i} "
            f"from the old cluster to the new one, measured the request latency "
            f"before and after on the same traffic replay, compared the memory "
            f"and the connection counts of both deployments under that replay, "
            f"recorded the cutover window agreed with the operations rota and "
            f"the rollback path that keeps the old cluster warm for a week, "
            f"listed the dashboards and the alerts that had to exist before the "
            f"switch, and agreed that service {i} stays on the new cluster unless "
            f"the error rate rises above the threshold fixed for that day, in "
            f"which case the rollback is executed without a further meeting."
        )
        facts.append((f"e{i:02d}", text, "project", max(0, 12 - i)))
    queries = [
        ("what happened with service number 3", "e03"),
        ("cutover window for service 7", "e07"),
        ("rollback path of service 12", "e12"),
        ("error rate threshold day 20", "e20"),
        ("migration of service number 25", "e25"),
        ("latency measured for service 31", "e31"),
        ("does service 38 stay on the new cluster", "e38"),
        ("traffic replay for service 40", "e40"),
    ]
    turns = []
    for i in range(1, 9):
        turns.append({"role": "user", "content": (
            f"Let us go over service {i}. We replayed the same traffic on both clusters, "
            f"the latency went from {90 + i} to {60 + i} milliseconds, the cutover window "
            f"is Tuesday night and the rollback path is the old cluster kept warm for a week. "
            f"Is there anything in that plan you would change before we commit to it?"
        )})
        turns.append({"role": "assistant", "content": (
            f"For service {i} the plan holds: the latency gain is measured on the same replay, "
            f"the cutover window is bounded and the rollback path is real. I would only add "
            f"an explicit error-rate threshold for the first day so the decision to stay is a "
            f"number rather than a feeling."
        )})
    return Corpus(LONG_CORPUS, facts, queries, turns)


CORPORA = {c.name: c for c in (_identity(), _episodic_long())}


def measure(name):
    """The measures for a named corpus, produced by the mechanism as it stands."""
    from ..conversation_compressor import ConversationCompressor
    from .retrieval import DEFAULT_TOP_N, MemoryRetriever

    corpus = CORPORA[name]
    records = [
        _Record(rid, text, category, use_count, f"2026-01-{i + 1:02d}")
        for i, (rid, text, category, use_count) in enumerate(corpus.facts)
    ]
    canonical = _Canonical(records)
    retriever = MemoryRetriever(canonical, _NoVector())

    hits = sum(
        1 for query, expected in corpus.queries
        if expected in {m.id for m in retriever.retrieve(query, top_n=DEFAULT_TOP_N)}
    )
    recall_rate = hits / len(corpus.queries)

    top_used = {r.id for r in sorted(records, key=lambda r: r.use_count, reverse=True)[:3]}
    floor = {m.id for m in retriever.recent_memories(top_n=DEFAULT_TOP_N)}
    floor_retention = len(top_used & floor) / len(top_used)

    composed = retriever.composed_memories(corpus.queries[0][0])
    fitted = retriever.fit_to_budget(composed)
    dropped = [m for m in composed[len(fitted):]]
    budget_dropped = len(dropped)

    found = sum(
        1 for r in records
        if r.id in {m.id for m in retriever.recover(r.text)}
    )
    recovery_rate = found / len(records)
    dropped_recovered = sum(
        1 for m in dropped
        if m.id in {x.id for x in retriever.recover(m.text)}
    )

    compressor = ConversationCompressor()
    summary, _label = compressor._compress_rule(corpus.turns)
    before = sum(_estimate_tokens(t["content"]) for t in corpus.turns)
    rule_ratio = _estimate_tokens(summary) / before if before else 0.0

    return {
        "facts": len(records),
        "queries": len(corpus.queries),
        "recall_rate": round(recall_rate, 4),
        "floor_retention": round(floor_retention, 4),
        "budget_dropped": budget_dropped,
        "dropped_recovered": dropped_recovered,
        "recovery_rate": round(recovery_rate, 4),
        "rule_ratio": round(rule_ratio, 4),
    }


# The record. Taken by running measure() on the mechanism as it stood on
# 2026-09-11, over recorder stores with the vector layer absent. A contract
# requires the live measurement to reproduce it; it changes only when the
# mechanism changes, on purpose, and never by editing to fit.
BASELINE = {
    "identity": {
        "facts": 10, "queries": 6,
        "recall_rate": 1.0, "floor_retention": 1.0,
        "budget_dropped": 0, "dropped_recovered": 0, "recovery_rate": 1.0,
        # The first finding: on short multi-sentence turns the rule strategy
        # does not compress at all -- it expands, by prefixing each kept
        # sentence with its role and keeping two sentences of three.
        "rule_ratio": 1.0542,
    },
    LONG_CORPUS: {
        "facts": 40, "queries": 8,
        "recall_rate": 1.0, "floor_retention": 1.0,
        # Two facts fall past the 512-token prompt budget; the archive search
        # finds both by their own text -- the dual-layer invariant, as a rate.
        "budget_dropped": 2, "dropped_recovered": 2, "recovery_rate": 1.0,
        # Barely a percent shorter: the rule strategy keeps almost everything.
        "rule_ratio": 0.9881,
    },
}
