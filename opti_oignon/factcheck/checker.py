"""The entry point: a fact checker that proves itself on its canary before it answers.

``FactChecker()`` reads ``config/factcheck.yaml`` (lazily, refused by name when
malformed), then runs the canary through the very check function that will
answer: if any planted error comes out supported, any positive control does
not, or any item comes out otherwise than planted, the checker is built
refused, and every call returns a refusal naming the failing items, never a
verdict and never a record. It runs again at every construction; nothing
caches it.

    checker = FactChecker()
    verdict = checker.check(claim, items, as_of="2026-09-26", read_on="2026-09-27")
    verdict.value, verdict.basis, verdict.reasons, verdict.text, verdict.record
    claims = checker.claims_from_text(answer_markdown, lang="en", origin={"author": "assistant"})

The creation time of a record is read here, outside the check, and stays
outside the record's digest.
"""

import dataclasses
import datetime
from pathlib import Path

from . import canary, decide, render, scope
from . import vocabulary as V

checkpoint_before_apply = True

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "factcheck.yaml"
LIMIT_KEYS = ("max_claim_chars", "min_quote_chars", "max_chunk_chars", "max_items", "max_chunks_per_item",
              "max_total_chars", "max_occurrences")


class FactCheckConfigError(ValueError):
    """The configuration file cannot be used as it is; the message names why."""


def load_config(path=None):
    """The configuration, read and checked; any other shape is refused by name."""
    import yaml

    path = Path(path) if path is not None else CONFIG_PATH
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FactCheckConfigError(f"{path}: cannot be read ({exc})") from exc
    if not isinstance(raw, dict):
        raise FactCheckConfigError(f"{path}: expected a mapping at the top")
    unknown = sorted(set(raw) - {"rules_version", "limits", "canary"})
    if unknown:
        raise FactCheckConfigError(f"{path}: unknown keys {unknown}")
    version = raw.get("rules_version")
    if isinstance(version, bool) or version != decide.RULES_VERSION:
        raise FactCheckConfigError(
            f"{path}: rules_version is {version!r}; this core applies rules version {decide.RULES_VERSION}")
    limits = raw.get("limits")
    if not isinstance(limits, dict) or set(limits) != set(LIMIT_KEYS):
        raise FactCheckConfigError(f"{path}: limits must hold exactly {list(LIMIT_KEYS)}")
    for name in LIMIT_KEYS:
        value = limits[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise FactCheckConfigError(f"{path}: limits.{name} must be a whole number of at least 1")
    section = raw.get("canary")
    if not isinstance(section, dict) or set(section) != {"refuse_on_failure"} or not isinstance(
            section["refuse_on_failure"], bool):
        raise FactCheckConfigError(f"{path}: canary must hold refuse_on_failure, true or false")
    return {"rules_version": version, "limits": {name: limits[name] for name in LIMIT_KEYS},
            "canary": {"refuse_on_failure": section["refuse_on_failure"]}}


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class FactChecker:
    """A checker whose canary ran, through its own check function, when it was built."""

    def __init__(self, judge=None, *, config=None, check=None, clock=None):
        if judge is not None:
            raise ValueError("this core has no judge seam; a judged path is not part of it")
        self.config = config if config is not None else load_config()
        self.check_function = check if check is not None else decide.check
        self._clock = clock or _utc_now
        self.canary = canary.run(self.check_function, config=self.config)
        self.refused = self.canary.outcome != "pass" and self.config["canary"]["refuse_on_failure"]

    def _refusal(self):
        return V.Refusal(failing=self.canary.failing, text=render.canary_refusal(self.canary.failing))

    def check(self, claim, items, *, as_of, read_on):
        """A verdict, or a refusal naming the canary's failing items."""
        if self.refused:
            return self._refusal()
        verdict = self.check_function(claim, items, as_of=as_of, read_on=read_on, config=self.config,
                                      canary=self.canary.summary())
        record = dict(verdict.record)
        record["created_at"] = self._clock()
        text = verdict.text
        if self.canary.outcome != "pass":
            text = render.canary_refusal(self.canary.failing) + " " + text
        return dataclasses.replace(verdict, record=record, text=text)

    def claims_from_text(self, text, *, lang="und", origin=None):
        """One claim per sentence of a markdown answer, with offsets into it; or the refusal."""
        if self.refused:
            return self._refusal()
        return scope.claims_from_text(text, lang=lang, origin=origin or {},
                                      max_claim_chars=self.config["limits"]["max_claim_chars"])

    def summarise(self, verdicts):
        return render.summarise(verdicts)
