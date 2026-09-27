"""How a verdict is shown: closed templates that never say more than the evidence.

Every text is built from the templates below, in English (the interface
translates). None of them holds a word of truth or verification; quoted source
text may. A supported or conflicting text names its source, its passage, the
source's own date (or "undated"), and the date it was read; a "not enough
evidence" text names every source searched, or says none was; a text says
"stated in", never that anything is so, and never that a model used a source.

The per-answer summary has no percentage and no badge: it leads with the most
severe verdict present, lists the claims in answer order, and counts what was
not checked, by reason.
"""

import re
import unicodedata

from . import vocabulary

checkpoint_before_apply = True

TEMPLATES = {
    "supported": 'Stated in {source} ({date}), read on {read_on}: "{quote}".',
    "supported_extracted": 'Stated in the extracted text of {source}{page} ({date}), read on {read_on}: '
                           '"{quote}".',
    "supported_own_only": 'Your notes say ({source}, {date}; read on {read_on}): "{quote}". Not checked '
                          'against any other source.',
    "also_stated": "Also stated in: {sources}.",
    "as_of": "Checked for {as_of}.",
    "later_replaced": 'Later replaced, on {date}, by {source}: "{quote}".',
    "superseded": 'Your decision {when_x} ({source_x}, {date_x}) says this: "{quote_x}"; a later '
                  'decision {when_y} ({source_y}, {date_y}) replaced it{via}: "{quote_y}". Read on '
                  '{read_on}.',
    "through": " (through {moments})",
    "out_of_scope": "Not checked: {reason}.",
    "searched": "Searched: {sources}.",
    "nothing_searched": "Nothing was searched.",
    "not_checked_against": "Not checked against: {refused}.",
    "not_counted": "Not counted for your own decision: {sources} (only your decisions and notes are).",
    "summary_head": "Most severe: {verdict}.",
    "summary_claim": '{number}. "{claim}": {verdict}. {text}',
    "summary_not_checked": "Not checked: {count} sentence(s) ({reasons}).",
    "canary_refused": "The self-test of planted errors failed on {names}; this checker gives no verdict "
                      "until it passes.",
}

REASON_TEXT = {
    "no_sources": "No source was handed in; nothing was searched.",
    "no_admissible_source": "No source handed in could be read as evidence.",
    "no_valid_source": "No source searched was valid on {as_of}.",
    "no_valid_source_stated": "Stated only in sources not valid on {as_of}: {sources}.",
    "no_admissible_source_quoted": "Stated only in passages that repeat a model's words, which are never "
                                   "evidence: {spans}.",
    "no_judge": "Not restated as a whole sentence in {sources}; this check reads nothing short of a "
                "whole sentence restated, and a paraphrase would need the entailment model, which it does "
                "not have.",
    "nearest": 'The nearest sentence, in {source} ({date}), reads: "{quote}"; where they differ, {runs}.',
    "nearest_run": 'the claim has "{claim}" and the source "{source}"',
    "nearest_more": "{count} other near sentence(s) are set beside the claim in the record.",
    "own_not_handed": "No decision or note of yours was handed in; only those count for your own decision.",
    "expired": '{source} ({date}) states it, but held only until {valid_until}: "{quote}".',
    "no_longer_held": 'Held until {held_until} ({source_x}); replaced{via} by {source_y} ({date_y}): '
                      '"{quote_y}".',
    "no_longer_held_dated": "The claim carries its own date, which this check does not read.",
    "source_undated": "Found only in undated sources ({sources}); a claim about the present needs a "
                      "dated one.",
    "context_qualified": '{source} ({date}) states it, but "{marker}" stands in its {where}, which '
                         'qualifies it.',
    "context_incomplete": "{source} ({date}) states it in a passage cut from its context ({cause}).",
    "attributed": '{source} ({date}) reports it as said by someone else ("{marker}", in its {where}); '
                  'it does not state it itself.',
    "extraction_uncertain": "{source} ({date}) states it in text extracted with {flags}; a number read "
                            "there may have been changed by the extraction.",
    "ingested_before_correction": "{source} ({date}) states it in a copy taken on {ingested}, before "
                                  "its correction of {corrected}.",
    "cites_retracted": "{source} ({date}) states it, and cites a retracted work.",
    "quote_not_found": "{source} ({date}) states it, but the host could not locate the passage "
                       "({refusal}).",
    "quote_too_short": "{source} ({date}) states it, but a passage shorter than {floor} characters is "
                       "never located.",
    "quote_truncated": "{source} ({date}) states it, but past the number of places one passage is "
                       "located in a chunk.",
    "conditional": '{source} ({date}) states it under a condition ("{marker}", in its {where}).',
    "evidence_hedged": '{source} ({date}) states it as a possibility or a forecast ("{marker}", in its '
                       '{where}).',
    "population_narrower": '{source} ({date}) states it for a narrower population ("{marker}", in its '
                           '{where}).',
}

CAUSES = {
    "list_item_without_lead_in": "a list item without its lead-in or heading",
    "continuation": "the continuation of a sentence cut before it",
    "chunk_start": "the first sentence of a chunk that may start inside a sentence",
    "chunk_end": "the last sentence of a chunk that may end inside a sentence",
    "uncertain_period": "after a period that may not end the sentence before it",
}

WHERE = {"heading": "heading", "lead_in": "lead-in", "before": "sentence before",
         "after": "sentence after", "quoted": "quotation", "sentence": "own text, struck through"}

PLAIN_REASONS = {
    "empty": "no words", "too_long": "too long", "question": "question", "code": "code",
    "heading": "heading", "table_row": "table row", "image": "image",
    "markup_unparsed": "markup not read", "instruction": "instruction",
    "not_standalone": "pronoun subject", "subject_unresolved": "unnamed subject",
}

SCOPE_TEXT = {
    "empty": "the sentence has no words",
    "too_long": "the sentence is longer than the limit",
    "question": "a question states nothing",
    "code": "code",
    "heading": "a heading",
    "table_row": "a table row",
    "image": "an image",
    "markup_unparsed": "markup this reader does not read",
    "instruction": "an instruction states nothing",
    "not_standalone": "its subject is a pronoun that refers to something else",
    "subject_unresolved": "its subject is not named",
}

VERDICT_WORDS = {"supported": "supported", "contradicted": "contradicted", "conflicting": "conflicting",
                 "not_enough_evidence": "not enough evidence", "out_of_scope": "out of scope"}

REFUSAL_TEXT = {
    "author_model": "written by a model", "author_unknown": "author unknown",
    "author_model_quoted": "repeats a model's words", "snippet": "a search excerpt",
    "no_consent": "no consent", "retracted": "retracted", "successor_missing": "its successor is missing",
    "chunk_changed": "its text changed since it was stored", "chunk_not_nfc": "its text is not NFC",
    "chunk_too_large": "a chunk over the size limit", "over_budget": "over the budget of this check",
}


def forbidden_in(text):
    """The forbidden words a text holds, in English or folded French, in order, once each."""
    folded = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)).lower()
    found = []
    for word in re.findall(r"[a-z]+", folded):
        if word in vocabulary.FORBIDDEN_WORDS and word not in found:
            found.append(word)
    return found


def label(info):
    """A source's name: its id, and its title when it has one."""
    return f'{info["source_id"]} "{info["title"]}"' if info.get("title") else info["source_id"]


def date(info):
    """A source's own date, or "undated", and its flags."""
    shown = info.get("date") or "undated"
    flags = info.get("flags") or {}
    if "preprint" in flags:
        shown += " (preprint)"
    if "corrected" in flags:
        shown += f" (corrected on {flags['corrected']})"
    if info.get("validity_unknown"):
        shown += " (validity unknown)"
    return shown


def _cited(info):
    """A cited source: its name, how it was read, its own date and its flags."""
    shown = f"the extracted text of {label(info)}" if info.get("extracted") else label(info)
    if info.get("extracted") and info.get("page") is not None:
        shown += f", p. {info['page']}"
    return f"{shown} ({date(info)})"


def supported(supports, *, read_on, as_of, world):
    first = supports[0]
    fields = {"source": label(first), "date": date(first), "read_on": read_on, "quote": first["quote"]}
    own_only = world and all(s["tier"] in vocabulary.OWN_TIERS for s in supports)
    if own_only:
        text = TEMPLATES["supported_own_only"].format(**fields)
    elif first.get("extracted"):
        page = first.get("page")
        text = TEMPLATES["supported_extracted"].format(page=f", p. {page}" if page is not None else "",
                                                       **fields)
    else:
        text = TEMPLATES["supported"].format(**fields)
    seen = {first["source_id"]}
    others = []
    for info in supports[1:]:
        if info["source_id"] not in seen:
            seen.add(info["source_id"])
            others.append(_cited(info))
    if others:
        text += " " + TEMPLATES["also_stated"].format(sources=", ".join(others))
    if as_of != read_on:
        text += " " + TEMPLATES["as_of"].format(as_of=as_of)
    replaced = {}
    for info in supports:
        if info.get("replaced"):
            replaced.setdefault(info["replaced"]["source_id"], info["replaced"])
    for source_id in sorted(replaced):
        entry = replaced[source_id]
        text += " " + TEMPLATES["later_replaced"].format(date=entry["date"], source=source_id,
                                                         quote=entry["quote"])
    return text


def _when(moment):
    return f'{moment["label"]} {moment["date"]}' if moment["date"] else moment["label"]


def _through(moments):
    """The decisions a replacement went through, each with its date; empty for a direct one."""
    if not moments:
        return ""
    return TEMPLATES["through"].format(moments=", ".join(f'{m["source_id"]}, {_when(m)}' for m in moments))


def superseded(x, y, *, d1, d2, read_on, as_of, via=()):
    text = TEMPLATES["superseded"].format(
        when_x=_when(d1), source_x=label(x), date_x=date(x), quote_x=x["quote"],
        when_y=_when(d2), source_y=label(y), date_y=date(y), quote_y=y["quote"], via=_through(via),
        read_on=read_on)
    if as_of != read_on:
        text += " " + TEMPLATES["as_of"].format(as_of=as_of)
    return text


def _held(entry):
    """When a source held, as far as its dates say."""
    start, end = entry.get("start"), entry.get("end")
    if entry.get("end_from") == "successor_earlier":
        return (f"{label(entry)} (from {start}, replaced by a decision dated {end}, before it: when it held "
                "is unknown)")
    if entry.get("end_from") == "successor_undated":
        return f"{label(entry)} (replaced by an undated decision: when it held is unknown)"
    if start and end:
        return f"{label(entry)} (from {start} until {end})"
    if start:
        return f"{label(entry)} (from {start})"
    if end:
        return f"{label(entry)} (until {end})"
    return label(entry)


def _refused(entries):
    return ", ".join(f'{e["source_id"]} ({_refusal_words(e)})' for e in entries)


def _refusal_words(entry):
    words = REFUSAL_TEXT.get(entry["admission"], entry["admission"])
    if entry["admission"] == "retracted":
        words += f' on {entry["flags"]["retracted"]}'
    return words


def not_enough(reasons, details, *, searched, refused, not_counted, as_of):
    """One sentence per reason, the leading one first, then what was and was not searched."""
    lines = []
    ids = ", ".join(searched) if searched else "no source"
    for reason in reasons:
        detail = dict(details.get(reason) or {})
        if reason == "no_judge":
            lines.append(REASON_TEXT["no_judge"].format(sources=ids))
            found = list(detail.get("nearest") or ())
            if found:
                near = found[0]
                runs = "; ".join(REASON_TEXT["nearest_run"].format(claim=d["claim"], source=d["source"])
                                 for d in near["differences"])
                lines.append(REASON_TEXT["nearest"].format(source=label(near), date=date(near),
                                                           quote=near["quote"], runs=runs))
            if len(found) > 1:
                lines.append(REASON_TEXT["nearest_more"].format(count=len(found) - 1))
        elif reason == "no_valid_source" and detail.get("stated_in"):
            lines.append(REASON_TEXT["no_valid_source_stated"].format(
                as_of=as_of, sources=", ".join(_held(entry) for entry in detail["stated_in"])))
        elif reason == "no_valid_source":
            lines.append(REASON_TEXT["no_valid_source"].format(as_of=as_of))
        elif reason == "no_admissible_source" and not refused and not_counted and not detail:
            lines.append(REASON_TEXT["own_not_handed"])
        elif reason == "no_admissible_source" and detail.get("model_quoted"):
            lines.append(REASON_TEXT["no_admissible_source_quoted"].format(spans=", ".join(
                f'{label(span)}, characters {span["start"]} to {span["end"]}' for span in detail["model_quoted"])))
        elif reason == "source_undated":
            lines.append(REASON_TEXT["source_undated"].format(sources=", ".join(detail.get("sources", []))))
        elif reason == "no_longer_held":
            lines.append(REASON_TEXT["no_longer_held"].format(
                held_until=detail["held_until"] or "an unknown date", source_x=label(detail["x"]),
                source_y=label(detail["y"]), date_y=date(detail["y"]), quote_y=detail["y"]["quote"],
                via=_through(detail.get("via", ()))))
            if detail.get("claim_carries_date"):
                lines.append(REASON_TEXT["no_longer_held_dated"])
        elif reason == "quote_not_found" and detail.get("refusal") in ("quote_too_short", "truncated"):
            source = detail.get("source") or {}
            name = "quote_too_short" if detail["refusal"] == "quote_too_short" else "quote_truncated"
            lines.append(REASON_TEXT[name].format(source=label(source) if source else "a source",
                                                  date=date(source) if source else "undated",
                                                  floor=detail.get("floor", "")))
        elif reason in ("no_sources", "no_admissible_source"):
            lines.append(REASON_TEXT[reason])
        elif reason in REASON_TEXT:
            source = detail.get("source") or {}
            where = WHERE.get(detail.get("where"), detail.get("where", ""))
            lines.append(REASON_TEXT[reason].format(
                source=label(source) if source else "a source", date=date(source) if source else "undated",
                marker=detail.get("marker", ""), where=where, flags=", ".join(detail.get("flags", [])),
                ingested=detail.get("ingested") or "an unknown date", corrected=detail.get("corrected", ""),
                valid_until=detail.get("valid_until", ""), quote=source.get("quote", ""),
                refusal=detail.get("refusal", ""), cause=CAUSES.get(detail.get("cause"), "")))
    if searched and "no_judge" not in reasons:
        lines.append(TEMPLATES["searched"].format(sources=ids))
    if not searched and "no_sources" not in reasons:
        lines.append(REASON_TEXT["no_sources"] if not refused and not not_counted
                     else TEMPLATES["nothing_searched"])
    if refused:
        lines.append(TEMPLATES["not_checked_against"].format(refused=_refused(refused)))
    if not_counted:
        lines.append(TEMPLATES["not_counted"].format(sources=", ".join(not_counted)))
    return " ".join(lines)


def out_of_scope(reason):
    return TEMPLATES["out_of_scope"].format(reason=SCOPE_TEXT[reason])


def canary_refusal(names):
    return TEMPLATES["canary_refused"].format(names=", ".join(names))


def summarise(verdicts):
    """The per-answer summary: most severe first, claims in answer order, what was not checked."""
    verdicts = list(verdicts)
    present = {v.value for v in verdicts}
    lines = []
    if not verdicts:
        return "Nothing to summarise."
    most = next(value for value in vocabulary.SEVERITY if value in present)
    lines.append(TEMPLATES["summary_head"].format(verdict=VERDICT_WORDS[most]))
    number = 0
    counts = {}
    for verdict in verdicts:
        if verdict.value == vocabulary.OUT_OF_SCOPE:
            counts[verdict.leading] = counts.get(verdict.leading, 0) + 1
            continue
        number += 1
        claim = verdict.record["claim"]["text"] if verdict.record else ""
        lines.append(TEMPLATES["summary_claim"].format(number=number, claim=claim,
                                                       verdict=VERDICT_WORDS[verdict.value], text=verdict.text))
    if counts:
        order = vocabulary.REASONS[vocabulary.OUT_OF_SCOPE]
        parts = ", ".join(f"{PLAIN_REASONS[r]} {counts[r]}" for r in order if r in counts)
        lines.append(TEMPLATES["summary_not_checked"].format(count=sum(counts.values()), reasons=parts))
    return "\n".join(lines)
