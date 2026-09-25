#!/usr/bin/env python3
"""What a human is shown of a sensitive sync record before deciding on it.

A skill received from a paired device lands only once this device's human
lets it through. The gate names it -- the record id is the category and
name it lands under -- but never showed what it says: the live prompt
cuts every value to 60 characters and gives up after 30 seconds, and the
pending list carries provenance alone.

``describe`` answers what may be shown anywhere, in a prompt or a list:
the category and name the skill will land under, and the SHA-256 of its
text, the digest ``/adopt`` asks for once it has landed. ``review_text``
answers the text itself, for one record a human asked to read; no list
ever carries it. ``gate_arguments`` is what the live approval shows, the
description first so that a truncated summary still names the skill.

Pure and import-safe: the landing name comes from the skill registry's own
sanitiser, asked for at the call.
"""

import hashlib

checkpoint_before_apply = True

SKILL = "skill"


def _skill_fields(kind, payload):
    """``(category, name, markdown)`` of a skill payload, or None for anything else."""
    if kind != SKILL or not isinstance(payload, dict):
        return None
    skill = payload.get("skill")
    if not isinstance(skill, dict):
        return None
    category, name, markdown = skill.get("category"), skill.get("name"), skill.get("markdown")
    if not (isinstance(category, str) and category and isinstance(name, str) and name):
        return None
    return category, name, markdown if isinstance(markdown, str) else ""


def _landing(category, name):
    """The ``category/name`` a skill lands under, by the registry's sanitiser; None if unreachable."""
    try:
        from opti_oignon.agent.skills import _safe_segment
    except Exception:  # noqa: BLE001 - an unknown landing is not guessed
        return None
    return f"{_safe_segment(category, 'general')}/{_safe_segment(name, 'untitled-skill')}"


def describe(kind, payload):
    """For a skill, the name it lands under and the digest of its text; ``{}`` otherwise."""
    fields = _skill_fields(kind, payload)
    if fields is None:
        return {}
    category, name, markdown = fields
    described = {}
    landing = _landing(category, name)
    if landing is not None:
        described["skill"] = landing
    if markdown:
        described["digest"] = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    return described


def review_text(kind, payload):
    """The text a skill record would land with, for a human who asked; None for anything else."""
    fields = _skill_fields(kind, payload)
    if fields is None or not fields[2]:
        return None
    return fields[2]


def gate_arguments(kind, payload, *, peer_id, record_id, device):
    """What the live approval shows: the description, then the provenance."""
    return {**describe(kind, payload), "peer_id": peer_id, "kind": kind, "id": record_id, "device": device}
