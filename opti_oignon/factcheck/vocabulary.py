"""Every word the fact-check core can say, closed, and the verdict that holds them.

Five verdicts, three bases, and a closed list of reasons per verdict. The list
is complete for the whole design, including the reasons only a later reader
(frames, a calibrated judge, decomposition) emits, so that reader adds rules
and never a word. A verdict built with any other value, basis or reason, or
with a reason under the wrong verdict, raises: an unnamed outcome cannot be
constructed, let alone shown.

No value, basis or code is a word that claims truth: "supported" means
supported by a named passage, and nothing here says more.
"""

from dataclasses import dataclass, field

checkpoint_before_apply = True

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
CONFLICTING = "conflicting"
NOT_ENOUGH_EVIDENCE = "not_enough_evidence"
OUT_OF_SCOPE = "out_of_scope"

VERDICTS = (SUPPORTED, CONTRADICTED, CONFLICTING, NOT_ENOUGH_EVIDENCE, OUT_OF_SCOPE)

DETERMINISTIC = "deterministic"
MODEL_JUDGED = "model_judged"
NO_BASIS = "none"
BASES = (DETERMINISTIC, MODEL_JUDGED, NO_BASIS)

REASONS = {
    SUPPORTED: ("verbatim_sentence", "equivalent_restatement", "judged_entailment"),
    CONTRADICTED: ("value_disjoint", "negation_flip", "quantifier_opposed", "superseded_state",
                   "judged_contradiction"),
    CONFLICTING: ("sources_disagree", "superseded"),
    NOT_ENOUGH_EVIDENCE: (
        "no_sources", "no_admissible_source", "no_valid_source", "expired", "no_longer_held",
        "source_undated", "context_qualified", "context_incomplete", "attributed",
        "extraction_uncertain", "ingested_before_correction", "cites_retracted", "no_judge",
        "no_passage", "quote_not_found", "anchor_unmatched", "anchor_differs", "anchor_ambiguous",
        "anchor_incomparable", "anchor_binding_unverified", "role_order", "negation_differs",
        "negation_ambiguous", "quantifier_differs", "direction_differs", "evidence_hedged",
        "strength_exceeds", "conditional", "population_narrower", "judge_uncalibrated",
        "calibration_stale", "judge_abstained", "judge_failed", "judge_refused",
        "judged_contradiction_unanchored", "part_adds_anchor", "parts_incomplete",
        "parts_unfaithful",
    ),
    OUT_OF_SCOPE: ("empty", "too_long", "question", "code", "heading", "table_row", "image",
                   "markup_unparsed", "instruction", "not_standalone", "subject_unresolved"),
}

# Why a source, or one of its chunks or spans, is not read as evidence.
REFUSALS = ("author_model", "author_unknown", "author_model_quoted", "snippet", "no_consent",
            "retracted", "successor_missing", "chunk_changed", "chunk_not_nfc",
            "chunk_too_large", "over_budget")

# What the passage check answers for one quote in one chunk, when it locates nothing.
PASSAGE_REFUSALS = ("chunk_changed", "chunk_not_nfc", "quote_empty", "quote_too_short",
                    "quote_not_found")

FLAGS = ("validity_unknown", "supersession_coarse", "truncated")
WRAPPERS = ("hedged", "advice", "quoted")

ITEM_FLAGS = ("retracted", "corrected", "preprint", "synced", "cites_retracted")
DATED_ITEM_FLAGS = ("retracted", "corrected")
EXTRACTION_FLAGS = ("table", "ocr", "multi_column")

SOURCE_KINDS = ("note", "ledger", "core", "library", "web", "snippet")
FACT_LEVEL_KINDS = ("ledger", "core")
AUTHORS = ("user", "third_party", "model", "unknown")
TIERS = ("own_decision", "own_note", "library", "web")
OWN_TIERS = ("own_decision", "own_note")
CLAIM_KINDS = ("world", "own", "auto")
LANGS = ("en", "fr", "und")

REPLAY_OUTCOMES = ("reproduced", "source_changed", "rules_changed", "record_differs")
CANARY_OUTCOMES = ("pass", "fail", "not_run", "running")

# When several reasons apply, the text leads with the first of this order,
# then the rest in the order of REASONS.
LEADING_ORDER = (
    "anchor_differs", "no_longer_held", "context_qualified", "attributed", "population_narrower",
    "conditional", "strength_exceeds", "evidence_hedged", "negation_differs", "direction_differs",
    "anchor_binding_unverified", "role_order", "anchor_ambiguous", "extraction_uncertain",
    "source_undated", "judge_refused", "judge_uncalibrated", "calibration_stale", "no_judge",
    "judge_abstained", "no_passage", "no_valid_source", "no_admissible_source", "no_sources",
)

# The per-answer summary leads with the most severe verdict present.
SEVERITY = (CONTRADICTED, CONFLICTING, NOT_ENOUGH_EVIDENCE, OUT_OF_SCOPE, SUPPORTED)

# Words no template may hold (English, and French folded to ASCII), with their
# inflected forms. Quoted source text may hold them; a template never does.
FORBIDDEN_WORDS = ("true", "false", "verified", "correct", "accurate", "proven", "confirmed",
                   "fact", "vrai", "faux", "verifie", "exact", "confirme",
                   "truly", "truth", "truths", "falsely", "verify", "verifies", "verifying", "correctly",
                   "accurately", "prove", "proves", "proved", "confirm", "confirms", "facts", "factual",
                   "vraie", "vrais", "vraies", "fausse", "fausses", "verite", "verifiee",
                   "verifiees", "exacte", "exacts", "exactes", "confirmee", "confirmes", "confirmees",
                   "correcte", "fait", "faits", "prouve", "prouvee")


def all_codes():
    """Every value, basis and code of the vocabulary, as one set."""
    codes = set(VERDICTS) | set(BASES)
    for group in REASONS.values():
        codes |= set(group)
    for group in (REFUSALS, PASSAGE_REFUSALS, FLAGS, WRAPPERS, ITEM_FLAGS, EXTRACTION_FLAGS,
                  SOURCE_KINDS, AUTHORS, TIERS, CLAIM_KINDS, REPLAY_OUTCOMES, CANARY_OUTCOMES):
        codes |= set(group)
    return frozenset(codes)


def leading(reasons):
    """The reason a text leads with."""
    reasons = tuple(reasons)
    if not reasons:
        raise ValueError("no reason to lead with")
    for code in LEADING_ORDER:
        if code in reasons:
            return code
    order = [code for group in REASONS.values() for code in group]
    return min(reasons, key=lambda code: order.index(code) if code in order else len(order))


@dataclass(frozen=True)
class Verdict:
    """A verdict, its basis and its reasons; the text and the record ride along.

    Construction refuses any value, basis or reason outside the vocabulary, a
    reason under the wrong verdict, no reason at all, a basis on a verdict
    that has none, and no basis on one that needs it.
    """

    value: str
    basis: str
    reasons: tuple
    text: str = ""
    record: object = None
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if self.value not in VERDICTS:
            raise ValueError(f"unknown verdict {self.value!r}")
        if self.basis not in BASES:
            raise ValueError(f"unknown basis {self.basis!r}")
        if not self.reasons:
            raise ValueError(f"a {self.value} verdict needs a reason")
        allowed = REASONS[self.value]
        stray = [code for code in self.reasons if code not in allowed]
        if stray:
            raise ValueError(f"reasons {stray} do not belong to a {self.value} verdict")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError(f"a reason is repeated in {self.reasons}")
        needs_basis = self.value in (SUPPORTED, CONTRADICTED, CONFLICTING)
        if needs_basis and self.basis == NO_BASIS:
            raise ValueError(f"a {self.value} verdict needs a basis")
        if not needs_basis and self.basis != NO_BASIS:
            raise ValueError(f"a {self.value} verdict has no basis, not {self.basis!r}")

    @property
    def leading(self):
        return leading(self.reasons)


@dataclass(frozen=True)
class Refusal:
    """What a refused checker answers instead of a verdict: the failing items, named."""

    failing: tuple
    text: str
