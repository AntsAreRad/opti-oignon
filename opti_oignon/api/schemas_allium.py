#!/usr/bin/env python3
"""The models of the garden's routes: the componion's served status, and a refusal of a garden route.

A module of its own, beside the platform's ``schemas.py``, which it never
touches: every class here is named ``Allium*``, so no component of the
published OpenAPI document shares a name with another and none is renamed
after its module. The closed codes are the garden's own (its statuses, its
label codes, the codes of a being at the served minute), and the models'
fields are exactly the keys of ``describe.web_fields`` at every level: a
response model drops an undeclared key in silence, so the two are held equal
by contract rather than by care.

The componion is a simulation. ``AlliumBeing.light`` is where the sun is in
its sky, never a count of anything.
"""

from typing import Literal

from pydantic import BaseModel, Field

# The garden's statuses, in the order it derives them (``service.STATUSES``).
StatusCode = Literal["disabled", "stopped", "unavailable", "awaiting_soil", "ready", "sealed_bulbe", "missing",
                     "retired_prototype", "unreadable", "resting", "alive"]
# The label codes, in the order their lines open a form (``describe.LABEL_ORDER``).
LabelCode = Literal["prototype", "retired", "glass_jar", "catching_up", "frozen", "mode_unknown", "bulbe"]


class AlliumLine(BaseModel):
    """One served line: its catalogue key and its text, unwrapped; render it whole, never clipped."""

    key: str = Field(description="The catalogue key; the family before the first dot says what the line is.")
    text: str = Field(description="The line as the garden checked it, unwrapped.")


class AlliumAsOf(BaseModel):
    """The minute the served state is at, in the being's own local time."""

    local: str = Field(description="YYYY-MM-DD HH:MM, local to the being.")
    offset: str = Field(description="The local offset it was recorded with, +HH:MM.")


class AlliumBeing(BaseModel):
    """What the being is at the served minute, as closed codes; the words are in the lines."""

    day: int = Field(description="Its day of life, counted from its sowing.")
    life: Literal["awake", "breathing", "dormant_winter", "dormant_dry", "dormant"] = Field(
        description="Whether it is awake, breathing, or dormant, and why it went dormant.")
    light: Literal["down", "up", "rise", "set"] = Field(
        description="Where the sun is in its sky: down, up, rising or setting. Not a count of anything.")
    name: str | None = Field(description="The name given to it, or null.")
    place: Literal["garden", "windowsill"] = Field(description="Where it lives: a garden bed or a windowsill.")
    season: Literal["winter", "spring", "summer", "autumn"] = Field(
        description="The season of its world at the served minute.")
    soil: Literal["dry", "damp", "wet"] = Field(description="How wet its soil is, against its law's thresholds.")
    stage: Literal["seed"] = Field(description="Its stage of life; a seed in this version.")


class AlliumHabitat(BaseModel):
    """Its container and the layer it is drawn in under this machine's security mode."""

    container: Literal["pot", "jar"]
    layer: Literal["open", "bulbe", "sealed"]


class AlliumLaw(BaseModel):
    """The law its world was sown under."""

    name: str = Field(description="The name of the law.")
    provisional: bool = Field(description="Whether the law is provisional: a prototype's world.")
    v: int = Field(description="The version of the law.")


class AlliumStatus(BaseModel):
    """The componion as it is served now: a simulation, with the lines that say it."""

    status: StatusCode = Field(description="The status of the look, in the order the garden derives them.")
    labels: list[LabelCode] = Field(description="Label codes, in the order their lines open the form.")
    lines: list[AlliumLine] = Field(description="The text tier, unwrapped, the doctrine last; empty when off.")
    source: Literal["simulation"]
    as_of: AlliumAsOf | None
    being: AlliumBeing | None
    habitat: AlliumHabitat | None
    law: AlliumLaw | None


class AlliumRefusal(BaseModel):
    """A refusal of a garden route: one closed line, and its code."""

    detail: str
    refusal: Literal["host", "origin", "site", "sign_in", "auth_unavailable", "request", "fault"]
