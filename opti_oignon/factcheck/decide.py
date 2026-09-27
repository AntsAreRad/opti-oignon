"""The check: one claim, the evidence handed in, a date, and a verdict with its record.

Order of evaluation, fixed:

1. the markdown pass and the scope of the claim; out of scope stops here, and a
   wrapper is stripped and recorded;
2. admission of every source, refusals recorded, budgets applied in sorted order;
3. validity at the date the claim is about, supersession derived;
4. a scan of every sentence of every admitted chunk: a sentence the claim
   restates whole, under fold v1 (final punctuation aside, its first letter
   lowered, the owner's pronouns rewritten when the owner's own claim meets
   the owner's own source), is a candidate; its passage is located by the host
   in the chunk whose hash it recomputes; its window is read (struck text, a
   qualifying marker, a frame of condition, forecast, population or negation
   set above it, an attribution, a context cut by a list, a chunk boundary or
   an uncertain period), then the time (a claim about the present needs a
   dated source), then provenance (a number from an uncertain extraction, a
   copy taken before a correction, a model-quoted range, a work that cites a
   retracted one); a chunk with no candidate keeps its nearest sentence, to
   be shown beside the claim;
5. aggregation: a candidate from a source valid at the date that passed every
   check supports the claim; otherwise a candidate from a replaced fact whose
   successor holds gives "conflicting, superseded" for a report of a past
   decision with no date of its own, and "no longer held" for anything else;
   otherwise "not enough evidence" with every reason found, the leading one
   first.

This core has no contradiction: a source that differs never contradicts here,
it is "not enough evidence", shown beside the claim with each differing run of
words named. It reaches no model; a judge is refused by name. Every set is
iterated sorted (source id, chunk hash, offset), so the order evidence is
handed in changes nothing, and no clock is read.
"""

import collections
import difflib
import unicodedata

from . import markup, passage, record, render, scope, sources
from . import vocabulary as V

checkpoint_before_apply = True

RULES_VERSION = 1
_UNCERTAIN_EXTRACTION = ("table", "ocr", "multi_column")
_FINAL = ".!?;"
# A sentence is shown as the nearest to the claim when it shares at least this
# share of the claim's words (as a fraction: numerator, denominator), counted
# with their repeats, under fold v1; the best of each chunk is kept, the
# earliest on a tie.
NEAREST_FLOOR = (1, 2)


def rules_document():
    """Everything that decides a verdict besides its inputs, for the rules digest."""
    return {
        "rules_version": RULES_VERSION,
        "unicode_version": unicodedata.unidata_version,
        "fold": passage.fold_rules(),
        "splitter": passage.splitter_rules(),
        "markup": markup.rules(),
        "lexicons": scope.rules(),
        "sources": sources.rules(),
        "vocabulary": {
            "verdicts": list(V.VERDICTS), "bases": list(V.BASES),
            "reasons": {k: list(v) for k, v in V.REASONS.items()},
            "leading": list(V.LEADING_ORDER), "severity": list(V.SEVERITY),
        },
        "templates": {"templates": dict(render.TEMPLATES), "reasons": dict(render.REASON_TEXT)},
        "nearest": {"floor": list(NEAREST_FLOOR), "unit": "words of fold v1, with their repeats"},
        "context": {"final_punctuation": _FINAL,
                    "chunk_boundaries": "a chunk starts at a block when its item has one chunk, its offset is 0, "
                                        "or its locator says starts_block; it ends at one when its item has one "
                                        "chunk or its locator says ends_block"},
    }


def rules_digest():
    return record.digest(rules_document())


def _persons(scoped, item):
    """(claim person, source person) for the owner rewrite, or (None, None)."""
    if scoped.kind != "own" or sources.tier(item) not in V.OWN_TIERS:
        return None, None
    return scoped.person, ("first" if item.author == "user" else None)


def restates(claim, sentence, *, owner=False):
    """Whether ``claim``, its wrapper stripped, restates ``sentence`` whole under fold v1.

    With ``owner``, the claim is read as the owner's own and ``sentence`` as
    written by the owner.
    """
    claim = scope.as_claim(claim)
    scoped = scope.analyse(claim, max_claim_chars=10 ** 9)
    person = scoped.person or ("second" if claim.origin.get("author") == "assistant" else "first")
    left = scope.compare_form(scoped.checked, person=person if owner else None, lang=claim.lang)
    right = scope.compare_form(sentence, person="first" if owner else None, lang=claim.lang)
    return left == right


class _Candidate:
    """A sentence of an admitted chunk that the claim restates, and what was found about it."""

    def __init__(self, item, chunk, view, sents, index, persons, markdown):
        self.item = item
        self.chunk = chunk
        self.view = view
        self.sentence = sents[index]
        self.index = index
        self.persons = persons
        self.markdown = markdown
        self.start, self.end = markup.to_original(view, self.sentence.start, self.sentence.end)
        self.text = chunk.text[self.start:self.end]
        self.window = None
        self.located = None
        self.blockers = []
        self.span_refused = False
        self.time_block = None

    @property
    def passage_ok(self):
        return self.located is not None and self.located.refusal is None and (
            (self.start, self.end) in self.located.occurrences)

    @property
    def passage_outcome(self):
        if self.located is None:
            return "not_located"
        if self.located.refusal:
            return self.located.refusal
        if self.passage_ok:
            return "located"
        return "truncated" if self.located.truncated else "quote_not_found"

    @property
    def clean(self):
        return self.passage_ok and not self.span_refused and not self.blockers


def _lead_in_range(view, entry):
    where, start, end = entry
    return list(markup.to_original(view, start, end)) if where == "plain" else [start, end]


def _window_positions(candidate):
    view, win = candidate.view, candidate.window

    def span(rng):
        if rng is None:
            return None
        return list(markup.to_original(view, rng[0], rng[1]))

    if win.heading is not None:
        start, end = markup.to_original(view, win.heading[0], win.heading[1])
        heading = {"start": start, "end": end, "from": "line"}
    elif win.heading_from == "locator":
        heading = {"text": win.heading_text, "from": "locator"}
    else:
        heading = None
    headings = [{"start": a, "end": b, "from": "line"}
                for a, b in (markup.to_original(view, s, e) for s, e in win.headings)]
    if win.locator_heading:
        headings.append({"text": win.locator_heading, "from": "locator"})
    lead_ins = [_lead_in_range(view, entry) for entry in win.lead_ins]
    return {"sentence": [candidate.start, candidate.end], "before": span(win.before),
            "after": span(win.after), "heading": heading, "headings": headings,
            "lead_in": lead_ins[0] if lead_ins else None, "lead_ins": lead_ins}


def _source_info(item, validity, *, quote="", chunk=None):
    info = {"source_id": item.source_id, "title": item.title, "date": sources.day(item.source_date),
            "flags": dict(item.flags), "validity_unknown": "validity_unknown" in validity.flags,
            "tier": sources.tier(item), "quote": quote, "extracted": False, "page": None}
    if chunk is not None and chunk.extraction is not None:
        info["extracted"] = True
        info["page"] = chunk.locator.get("page")
    return info


def _places(view, win):
    """(where, text) of every part of the window read for markers, in reading order."""
    places = [("heading", view.text[a:b]) for a, b in win.headings]
    if win.locator_heading:
        places.append(("heading", win.locator_heading))
    for where, a, b in win.lead_ins:
        places.append(("lead_in", view.text[a:b] if where == "plain" else view.original[a:b]))
    if win.before is not None:
        places.append(("before", view.text[win.before[0]:win.before[1]]))
    if win.after is not None:
        places.append(("after", view.text[win.after[0]:win.after[1]]))
    return places


def _ends_final(text):
    text = text.rstrip().rstrip("\"')]}" + chr(0xBB) + chr(0x201D) + chr(0x2019)).rstrip()
    return bool(text) and text[-1] in _FINAL


def _evaluate(candidate, scoped, validity, limits, located, claim_strength):
    item, chunk, view = candidate.item, candidate.chunk, candidate.view
    quote = view.text[candidate.sentence.start:candidate.sentence.end]
    # One passage check per chunk and folded quote: a chunk that repeats the
    # sentence is searched once, not once per repetition.
    key = (chunk.text, chunk.sha256, candidate.markdown, quote)
    if key not in located:
        folded = (chunk.text, chunk.sha256, candidate.markdown, passage.fold_quote(quote))
        if folded not in located:
            located[folded] = passage.locate(
                chunk.text, chunk.sha256, quote, min_quote_chars=limits["min_quote_chars"],
                max_occurrences=limits["max_occurrences"], markdown=candidate.markdown)
        located[key] = located[folded]
    candidate.located = located[key]
    candidate.span_refused = any(a < candidate.end and candidate.start < b for a, b in chunk.model_quoted)
    win = candidate.window
    info = _source_info(item, validity, quote=candidate.text, chunk=chunk)
    places = _places(view, win)
    blockers = candidate.blockers
    if view.struck_chars and any(p in view.struck_chars
                                 for p in range(candidate.sentence.start, candidate.sentence.end)):
        blockers.append(("context_qualified", {"marker": "~~", "where": "sentence", "source": info}))
    else:
        for where, text in places:
            markers = scope.qualifiers(text)
            if not markers and where in ("before", "after"):
                markers = [m for m in (scope.denial(text),) if m]
            if markers:
                blockers.append(("context_qualified", {"marker": markers[0], "where": where, "source": info}))
                break
    seen = {reason for reason, _ in blockers}
    for where, text in places:
        if where not in ("heading", "lead_in"):
            continue
        for reason, marker in scope.strength(text):
            if reason in seen or (reason != "context_qualified" and reason in claim_strength):
                continue
            seen.add(reason)
            blockers.append((reason, {"marker": marker, "where": where, "source": info}))
    attributed = None
    for where, text in places:
        markers = scope.attributions(text)
        if markers:
            attributed = {"marker": markers[0], "where": where, "source": info}
            break
    if attributed is None and win.quoted:
        attributed = {"marker": ">", "where": "quoted", "source": info}
    if attributed is not None:
        blockers.append(("attributed", attributed))
    has_heading = bool(win.headings) or bool(win.locator_heading)
    first = view.text[candidate.sentence.start:candidate.sentence.start + 1]
    whole_item = len(item.chunks) == 1
    starts_block = whole_item or chunk.offset_in_source == 0 or chunk.locator.get("starts_block") is True
    ends_block = whole_item or chunk.locator.get("ends_block") is True
    cut = None
    if win.list_item and not win.lead_ins and not has_heading:
        cut = {"where": "lead_in", "cause": "list_item_without_lead_in"}
    elif win.first_in_chunk and first.isalpha() and first.islower():
        cut = {"where": "before", "cause": "continuation"}
    elif win.first_in_chunk and not starts_block:
        cut = {"where": "before", "cause": "chunk_start"}
    elif win.last_in_chunk and not ends_block and not _ends_final(quote):
        cut = {"where": "after", "cause": "chunk_end"}
    elif win.cut_after:
        cut = {"where": "before", "cause": "uncertain_period", "marker": win.cut_after}
    if cut is not None:
        blockers.append(("context_incomplete", {"source": info, **cut}))
    if scoped.time_sensitive and item.source_date is None:
        blockers.append(("source_undated", {"sources": [item.source_id], "source": info}))
    flags = tuple((chunk.extraction or {}).get("flags") or ())
    uncertain = [f for f in flags if f in _UNCERTAIN_EXTRACTION]
    if uncertain and any(c.isdigit() for c in scoped.checked):
        blockers.append(("extraction_uncertain", {"flags": uncertain, "source": info}))
    corrected = item.flags.get("corrected")
    if corrected:
        ingested = sources.day(chunk.ingested_at)
        if ingested is None or ingested < sources.day(corrected):
            blockers.append(("ingested_before_correction", {
                "ingested": ingested, "corrected": sources.day(corrected), "source": info}))
    if scoped.kind == "world" and "cites_retracted" in item.flags:
        blockers.append(("cites_retracted", {"source": info}))


def _add(reasons, details, reason, detail=None):
    if reason not in reasons:
        reasons.append(reason)
        if detail is not None:
            details[reason] = detail


def _statement(item, admission):
    """The whole statement of a fact-level item: its first admitted chunk's sentences."""
    for chunk, result in admission.chunks:
        if result != "admitted":
            continue
        view = markup.verbatim(chunk.text)
        sents = passage.sentences(view)
        if not sents:
            continue
        start, end = markup.to_original(view, sents[0].start, sents[-1].end)
        return chunk, view, sents, start, end
    return None


def _differences(claim_form, source_form):
    """Each differing run of words between the claim and a source, as written after the fold."""
    left, right = claim_form.split(" "), source_form.split(" ")
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    return [{"claim": " ".join(left[a:b]), "source": " ".join(right[c:d])}
            for tag, a, b, c, d in matcher.get_opcodes() if tag != "equal"]


def _json(value):
    """A detail made JSON-able: tuples to lists, source infos kept as dicts."""
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _claim_record(claim, scoped, as_of, read_on):
    runs = None
    if scoped.view is not None:
        runs = []
        origin = scoped.view.origin
        for position, original in enumerate(origin):
            if runs and runs[-1][1] + runs[-1][2] == original and runs[-1][0] + runs[-1][2] == position:
                runs[-1][2] += 1
            else:
                runs.append([position, original, 1])
    return {
        "input": {"text": claim.text, "lang": claim.lang, "kind": claim.kind, "origin": dict(claim.origin),
                  "marks": list(claim.marks) if claim.marks is not None else None,
                  "start": claim.start, "end": claim.end},
        "text": claim.text,
        "sha256": passage.sha256(claim.text),
        "lang": claim.lang,
        "kind": scoped.kind,
        "as_of": as_of,
        "read_on": read_on,
        "origin": dict(claim.origin),
        "wrapper": scoped.wrapper,
        "marks": list(scoped.marks),
        "scope": {"in_scope": scoped.in_scope, "reason": scoped.reason},
        "checked": {"text": scoped.checked, "start": scoped.checked_start, "end": scoped.checked_end},
        "past_decision": scoped.past_decision,
        "carries_date": scoped.dated,
        "owner_person": scoped.person,
        "time_sensitive": scoped.time_sensitive,
        "markup": {"version": markup.MARKUP_VERSION, "map": runs},
        "split": {"version": passage.SPLITTER_VERSION, "start": claim.start, "end": claim.end},
    }


def _searched_entry(item, admission, validity, counted, span_refusals):
    return {
        "source_id": item.source_id, "kind": item.kind, "tier": sources.tier(item), "author": item.author,
        "title": item.title, "lang": item.lang, "recorded_at": item.recorded_at,
        "valid_from": item.valid_from, "valid_until": item.valid_until, "superseded_by": item.superseded_by,
        "source_date": item.source_date, "flags": dict(item.flags), "consent": item.consent,
        "admission": admission.result, "counted": counted,
        "validity": {"start": {"date": validity.start, "from": validity.start_from},
                     "end": {"date": validity.end, "from": validity.end_from},
                     "valid_at": validity.valid, "flags": list(validity.flags),
                     "successor": validity.successor},
        "chunks": [{"sha256": c.sha256, "text_sha256": passage.sha256(c.text), "admission": result,
                    "length": len(c.text), "ingested_at": c.ingested_at, "offset_in_source": c.offset_in_source,
                    "locator": dict(c.locator), "extraction": _json(c.extraction),
                    "model_quoted": [list(p) for p in c.model_quoted]}
                   for c, result in admission.chunks],
        "span_refusals": span_refusals,
    }


def _span(candidate, role):
    offset = candidate.chunk.offset_in_source
    return {
        "source_id": candidate.item.source_id, "tier": sources.tier(candidate.item),
        "chunk_sha256": candidate.chunk.sha256, "start": candidate.start, "end": candidate.end,
        "abs_start": offset + candidate.start if offset is not None else None,
        "abs_end": offset + candidate.end if offset is not None else None,
        "text_sha256": passage.sha256(candidate.text), "window": _window_positions(candidate),
        "role": role, "quote_origin": "host_sentence",
    }


def _strip_sources(detail):
    """A detail for the record: source infos reduced to their ids."""
    out = {}
    for name, value in detail.items():
        if isinstance(value, dict) and "source_id" in value and "quote" in value:
            out[name] = value["source_id"]
        else:
            out[name] = _json(value)
    return out


class _Nearest:
    """The sentence of one chunk nearest to the claim, when one reaches the floor."""

    def __init__(self, form):
        tokens = form.split(" ")
        self.form = form
        self.size = len(tokens)
        self.distinct = set(tokens)
        self.counts = collections.Counter(tokens)
        self.best = None
        self.compared = 0

    def offer(self, sentence, form):
        self.compared += 1
        tokens = form.split(" ")
        numerator, denominator = NEAREST_FLOOR
        if len(self.distinct.intersection(tokens)) * denominator < len(self.distinct) * numerator:
            return
        shared = sum((self.counts & collections.Counter(tokens)).values())
        if shared * denominator < self.size * numerator:
            return
        if self.best is None or shared > self.best[0]:
            self.best = (shared, sentence, form)


def check(claim, items, *, as_of, read_on, config, judge=None, canary=None):
    """A verdict for one claim over the evidence handed in, at ``as_of``, read on ``read_on``.

    ``as_of`` may not be later than ``read_on``: sources are read as they stand
    on the day of the check. The record carries the canary summary it is
    handed; called here directly, with none, it says the canary did not run.
    Only a ``FactChecker``, which runs the canary at its construction, hands
    one in.
    """
    if judge is not None:
        raise ValueError("this core has no judge seam; a judged path is not part of it")
    as_of = sources.day(as_of, "as_of", required=True)
    read_on = sources.day(read_on, "read_on", required=True)
    if as_of > read_on:
        raise ValueError(f"as_of {as_of} is after read_on {read_on}: sources are read as they stand on the "
                         "day of the check, which cannot speak for a later date")
    limits = config["limits"]
    claim = scope.as_claim(claim)
    items = list(items)
    for one in items:
        if not isinstance(one, sources.SourceItem):
            raise TypeError("evidence is a list of SourceItem")
    ids = [one.source_id for one in items]
    if len(set(ids)) != len(ids):
        raise ValueError("two sources carry the same id")
    items.sort(key=lambda one: one.source_id)
    by_id = {one.source_id: one for one in items}
    scoped = scope.analyse(claim, max_claim_chars=limits["max_claim_chars"])
    admissions = sources.admit(items, limits=limits)
    validities = sources.validities(items, by_id, as_of, time_sensitive=scoped.time_sensitive)

    def counted(one):
        if admissions[one.source_id].result != "admitted":
            return False
        return scoped.kind != "own" or sources.tier(one) in V.OWN_TIERS

    candidates = []
    nearest = {}
    located = {}
    span_refusals = {one.source_id: [] for one in items}
    if scoped.in_scope:
        claim_strength = {reason for reason, _ in scope.strength(scoped.checked)}
        forms = {}
        for one in items:
            if not counted(one):
                continue
            claim_person, source_person = _persons(scoped, one)
            if claim_person not in forms:
                forms[claim_person] = scope.compare_form(scoped.checked, person=claim_person, lang=claim.lang)
            target = forms[claim_person]
            markdown = one.kind == "note"
            for chunk, result in admissions[one.source_id].chunks:
                if result != "admitted":
                    continue
                view = markup.plain(chunk.text) if markdown else markup.verbatim(chunk.text)
                sents = passage.sentences(view)
                near = _Nearest(target)
                restated = False
                for index, sentence in enumerate(sents):
                    body = view.text[sentence.start:sentence.end]
                    form = scope.compare_form(body, person=source_person, lang=one.lang)
                    if form != target:
                        if not restated:
                            near.offer(sentence, form)
                        continue
                    restated = True
                    candidate = _Candidate(one, chunk, view, sents, index, (claim_person, source_person), markdown)
                    candidate.window = passage.window(view, sents, index,
                                                      locator_heading=chunk.locator.get("heading"))
                    _evaluate(candidate, scoped, validities[one.source_id], limits, located, claim_strength)
                    if candidate.span_refused:
                        span_refusals[one.source_id].append(
                            {"start": candidate.start, "end": candidate.end, "refusal": "author_model_quoted"})
                    candidates.append(candidate)
                if not restated:
                    nearest[(one.source_id, chunk.sha256)] = (one, chunk, view, near)

    verdict_value, basis, reasons, details = None, V.NO_BASIS, [], {}
    spans, side = [], None
    supports = [c for c in candidates if validities[c.item.source_id].valid and c.clean]
    overlay = None
    texts = {}
    if not scoped.in_scope:
        verdict_value, reasons = V.OUT_OF_SCOPE, [scoped.reason]
        texts["text"] = render.out_of_scope(scoped.reason)
    elif supports:
        verdict_value, basis, reasons = V.SUPPORTED, V.DETERMINISTIC, ["verbatim_sentence"]
        spans = [_span(c, "support") for c in supports]
        infos = []
        for c in supports:
            info = _source_info(c.item, validities[c.item.source_id], quote=c.text, chunk=c.chunk)
            successor = validities[c.item.source_id].successor
            if successor is not None:
                replaced_on, _ = sources.start_of(by_id[successor])
                statement = _statement(by_id[successor], admissions[successor])
                if replaced_on is not None and replaced_on <= read_on and statement is not None:
                    y_chunk, _, _, y_start, y_end = statement
                    info["replaced"] = {"date": replaced_on, "source_id": successor,
                                        "quote": y_chunk.text[y_start:y_end]}
            infos.append(info)
        texts["text"] = render.supported(infos, read_on=read_on, as_of=as_of, world=scoped.kind == "world")
    else:
        for candidate in candidates:
            validity = validities[candidate.item.source_id]
            if validity.valid or not candidate.clean or validity.successor is None:
                continue
            if validity.end_from == "successor_earlier":
                continue
            head = by_id[validity.successor]
            seen = {candidate.item.source_id}
            via = []
            while (not validities[head.source_id].valid and validities[head.source_id].successor
                   and head.source_id not in seen):
                seen.add(head.source_id)
                via.append(head.source_id)
                head = by_id[validities[head.source_id].successor]
            if validities[head.source_id].valid and counted(head):
                statement = _statement(head, admissions[head.source_id])
                if statement is not None:
                    overlay = (candidate, head, statement, tuple(via))
                    break
        if overlay is not None:
            candidate, head, (y_chunk, y_view, y_sents, y_start, y_end), via = overlay
            x_validity = validities[candidate.item.source_id]
            d1 = {"date": x_validity.start, "label": _date_label(x_validity.start_from),
                  "source_id": candidate.item.source_id}
            d2 = _moment(head)
            through = [_moment(by_id[source_id]) for source_id in via]
            x_info = _source_info(candidate.item, x_validity, quote=candidate.text, chunk=candidate.chunk)
            y_quote = y_chunk.text[y_start:y_end]
            y_info = _source_info(head, validities[head.source_id], quote=y_quote, chunk=y_chunk)
            y_persons = _persons(scoped, head)
            y_candidate = _Candidate(head, y_chunk, y_view, y_sents, 0, y_persons, False)
            y_candidate.start, y_candidate.end, y_candidate.text = y_start, y_end, y_quote
            y_candidate.window = passage.window(y_view, y_sents, 0,
                                                locator_heading=y_chunk.locator.get("heading"))
            spans = [_span(candidate, "superseded"), _span(y_candidate, "replacement")]
            if scoped.past_decision and not scoped.dated:
                verdict_value, basis, reasons = V.CONFLICTING, V.DETERMINISTIC, ["superseded"]
                details["superseded"] = {"d1": d1, "d2": d2, "replaced_by": head.source_id, "via": through}
                texts["text"] = render.superseded(x_info, y_info, d1=d1, d2=d2, via=through, read_on=read_on,
                                                  as_of=as_of)
            else:
                _add(reasons, details, "no_longer_held", {"held_until": x_validity.end, "x": x_info,
                                                          "y": y_info, "replaced_by": head.source_id,
                                                          "via": through, "d1": d1, "d2": d2,
                                                          "claim_carries_date": scoped.dated})
                candidate.time_block = {"reason": "no_longer_held", "held_until": x_validity.end}
            texts["y_form"] = scope.compare_form(y_quote, person=y_persons[1], lang=head.lang)
            texts["y_candidate"] = y_candidate
        if verdict_value is None:
            verdict_value = V.NOT_ENOUGH_EVIDENCE
            not_valid, model_quoted = [], []
            for candidate in candidates:
                validity = validities[candidate.item.source_id]
                if overlay is not None and candidate is overlay[0]:
                    continue
                if validity.valid:
                    if candidate.span_refused:
                        model_quoted.append({"source_id": candidate.item.source_id, "title": candidate.item.title,
                                             "chunk_sha256": candidate.chunk.sha256, "start": candidate.start,
                                             "end": candidate.end})
                        continue
                    spans.append(_span(candidate, "blocked"))
                    if not candidate.passage_ok:
                        _add(reasons, details, "quote_not_found", {
                            "refusal": candidate.passage_outcome, "floor": limits["min_quote_chars"],
                            "source": _source_info(candidate.item, validity, quote=candidate.text,
                                                   chunk=candidate.chunk)})
                    for reason, detail in candidate.blockers:
                        _add(reasons, details, reason, detail)
                elif (candidate.clean and validity.end_from == "valid_until" and validity.end is not None
                      and validity.end <= as_of and validity.successor is None):
                    spans.append(_span(candidate, "expired"))
                    _add(reasons, details, "expired", {
                        "valid_until": validity.end, "source": _source_info(
                            candidate.item, validity, quote=candidate.text, chunk=candidate.chunk)})
                    candidate.time_block = {"reason": "expired", "valid_until": validity.end}
                else:
                    spans.append(_span(candidate, "historical"))
                    held = {"start": validity.start, "start_from": validity.start_from, "end": validity.end,
                            "end_from": validity.end_from}
                    not_valid.append({"source_id": candidate.item.source_id, "title": candidate.item.title,
                                      **held})
                    candidate.time_block = {"reason": "no_valid_source", **held}
            usable = [one for one in items if counted(one)]
            # "No judge" says that no sentence restates the claim; when one does, the
            # check it failed is the reason, never that one.
            if not items:
                _add(reasons, details, "no_sources")
            elif not usable:
                _add(reasons, details, "no_admissible_source")
            elif not candidates and not any(validities[one.source_id].valid for one in usable):
                _add(reasons, details, "no_valid_source")
            elif not candidates:
                _add(reasons, details, "no_judge", {"nearest": _nearest_details(nearest)})
            if not_valid:
                _add(reasons, details, "no_valid_source", {"as_of": as_of, "stated_in": not_valid})
            if model_quoted:
                _add(reasons, details, "no_admissible_source", {"model_quoted": model_quoted})
            if "source_undated" in details:
                details["source_undated"]["sources"] = sorted(
                    {c.item.source_id for c in candidates if any(r == "source_undated" for r, _ in c.blockers)})

    verdict_value = verdict_value or V.NOT_ENOUGH_EVIDENCE
    leading = V.leading(reasons)
    ordered = [leading] + [r for r in reasons if r != leading]

    usable_ids = [one.source_id for one in items if counted(one)]
    refused = []
    not_counted = []
    for one in items:
        admission = admissions[one.source_id]
        entry = {"source_id": one.source_id, "admission": admission.result, "flags": dict(one.flags)}
        if admission.result != "admitted":
            refused.append(entry)
        elif not counted(one):
            not_counted.append(one.source_id)
    if verdict_value == V.NOT_ENOUGH_EVIDENCE:
        texts["text"] = render.not_enough(ordered, details, searched=usable_ids, refused=refused,
                                          not_counted=not_counted, as_of=as_of)

    if verdict_value != V.SUPPORTED:
        side = _side_by_side(claim, scoped, candidates, overlay, texts, nearest, refused, not_counted,
                             verdict_value)

    support_tiers = sorted({sources.tier(c.item) for c in supports}) if verdict_value == V.SUPPORTED else []
    record_details = {name: _strip_sources(detail) for name, detail in details.items()}
    spans.sort(key=lambda s: (s["source_id"], s["chunk_sha256"], s["start"], s["role"]))
    checks = [{
        "source_id": c.item.source_id, "chunk_sha256": c.chunk.sha256, "start": c.start, "end": c.end,
        "valid_at": validities[c.item.source_id].valid,
        "passage": c.passage_outcome,
        "occurrences": c.located.count if c.located is not None else 0,
        "truncated": bool(c.located is not None and c.located.truncated),
        "span_refused": c.span_refused,
        "blockers": [{"reason": r, **_strip_sources(d)} for r, d in c.blockers],
    } for c in candidates]
    config_copy = _json(config)
    body = {
        "record_version": record.RECORD_VERSION,
        "rules_digest": rules_digest(),
        "rules_version": config.get("rules_version", RULES_VERSION),
        "unicode_version": unicodedata.unidata_version,
        "config": config_copy,
        "config_digest": record.digest(config_copy),
        "canary": dict(canary) if canary is not None else {"digest": None, "n": 0, "outcome": "not_run"},
        "claim": _claim_record(claim, scoped, as_of, read_on),
        "parts": [],
        "sources_searched": [
            _searched_entry(one, admissions[one.source_id], validities[one.source_id], counted(one),
                            span_refusals[one.source_id]) for one in items],
        "spans": spans,
        "checks": checks,
        "judge": None,
        "judge_reason": "no_judge",
        "verdict": {"value": verdict_value, "basis": basis, "reasons": ordered, "leading": leading,
                    "details": record_details, "support_tiers": support_tiers, "contradict_tiers": []},
        "side_by_side": side,
        "text": texts["text"],
        "integrity": "digest_only",
    }
    sealed = record.seal(body)
    return V.Verdict(verdict_value, basis, tuple(ordered), text=texts["text"], record=sealed,
                     details=record_details)


def _nearest_details(nearest):
    """The nearest sentences found, the nearest first (then in source order): source, date, quote, differences."""
    ranked = []
    for position, key in enumerate(sorted(nearest)):
        one, chunk, view, near = nearest[key]
        if near.best is None:
            continue
        shared, sentence, form = near.best
        start, end = markup.to_original(view, sentence.start, sentence.end)
        ranked.append((-shared, position, {
            "source_id": one.source_id, "title": one.title, "date": sources.day(one.source_date),
            "quote": chunk.text[start:end], "differences": _differences(near.form, form)}))
    ranked.sort(key=lambda entry: entry[:2])
    return [entry for _, _, entry in ranked]


def _date_label(derived):
    return {"valid_from": "of", "recorded_at": "recorded on", "source_date": "dated"}.get(derived,
                                                                                          "of an unknown date")


def _moment(item):
    """When an item started, how that date was derived, and whose date it is."""
    start, derived = sources.start_of(item)
    return {"date": start, "label": _date_label(derived), "source_id": item.source_id}


def _side_by_side(claim, scoped, candidates, overlay, texts, nearest, refused, not_counted, verdict_value):
    """The claim as written beside each source passage examined, every differing run named.

    A chunk with no sentence restating the claim is shown by its nearest
    sentence, when one shares at least the floor of the claim's words, with
    each differing run of words; else by the count of sentences compared. A
    source not read as evidence (refused, or not counted for the owner's own
    decision) is named with why; its text is not examined.
    """
    start = claim.start if claim.start is not None else 0
    end = claim.end if claim.end is not None else len(claim.text)
    side = {"claim": {"text": claim.text, "start": start, "end": end, "checked": scoped.checked},
            "passages": [], "not_examined": []}
    if verdict_value == V.OUT_OF_SCOPE:
        return side
    passages = side["passages"]
    shown = set()
    for candidate in candidates:
        passages.append({
            "source_id": candidate.item.source_id, "tier": sources.tier(candidate.item),
            "chunk_sha256": candidate.chunk.sha256, "start": candidate.start, "end": candidate.end,
            "text": candidate.text, "role": "restates", "read": "wording", "differences": [],
            "blocked_by": [{"reason": r, **_strip_sources(d)} for r, d in candidate.blockers]
            + ([{"reason": "author_model_quoted"}] if candidate.span_refused else [])
            + ([{"reason": "quote_not_found", "refusal": candidate.passage_outcome}]
               if not candidate.passage_ok else [])
            + ([dict(candidate.time_block)] if candidate.time_block else []),
        })
    if overlay is not None:
        x, head, _, _ = overlay
        y = texts["y_candidate"]
        shown.add((head.source_id, y.chunk.sha256))
        form = scope.compare_form(scoped.checked, person=x.persons[0], lang=claim.lang)
        passages.append({
            "source_id": head.source_id, "tier": sources.tier(head), "chunk_sha256": y.chunk.sha256,
            "start": y.start, "end": y.end, "text": y.text, "role": "replaced_by", "read": "wording",
            "differences": _differences(form, texts["y_form"]), "blocked_by": [],
        })
    for key in sorted(nearest):
        if key in shown:
            continue
        one, chunk, view, near = nearest[key]
        if near.best is not None:
            shared, sentence, form = near.best
            s_start, s_end = markup.to_original(view, sentence.start, sentence.end)
            passages.append({
                "source_id": one.source_id, "tier": sources.tier(one), "chunk_sha256": chunk.sha256,
                "start": s_start, "end": s_end, "text": chunk.text[s_start:s_end], "role": "nearest",
                "read": "wording", "shared_words": shared, "claim_words": near.size,
                "sentences_compared": near.compared, "differences": _differences(near.form, form),
                "blocked_by": [],
            })
        else:
            passages.append({
                "source_id": one.source_id, "tier": sources.tier(one), "chunk_sha256": chunk.sha256,
                "start": 0, "end": len(chunk.text), "role": "searched", "read": "wording",
                "sentences_compared": near.compared, "differences": [], "blocked_by": [],
            })
    passages.sort(key=lambda p: (p["source_id"], p["chunk_sha256"], p["start"], p["role"]))
    side["not_examined"] = ([{"source_id": e["source_id"], "why": e["admission"]} for e in refused]
                            + [{"source_id": source_id, "why": "not_counted_for_own_decision"}
                               for source_id in not_counted])
    side["not_examined"].sort(key=lambda e: e["source_id"])
    return side
