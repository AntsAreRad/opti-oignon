#!/usr/bin/env python3
"""Contracts against the timing race in the browser specs' shared setup.

A fresh install raises a modal configuration dialog once the backend has
answered what it knows about the install. Every browser spec starts by
dismissing it. The helper used to wait a fixed budget for the dialog to
appear and, if it did not, return silently -- at which point the dialog would
open a moment later, on top of the page, and swallow the next click.

The failure that produces is the worst kind: it depends on how fast the
machine is, it lands on an unrelated locator, and it looks like a product
regression. Three runs out of three failed to reproduce the expected red at
one point, which is what a budget buys.

The fix is not a larger budget. A budget cannot tell "the dialog has not
appeared YET" from "the dialog will NEVER appear", and that ambiguity is the
race. The application marks the moment it has decided; the helper waits for
that mark and then looks once.

  * FR1 -- the helper carries no appearance budget and swallows no timeout.
  * FR2 -- the helper waits for the decision to be marked before looking.
  * FR3 -- the component marks the decision on every path out, including the
    path where it decides to show nothing and the path where the backend
    never answers.
  * FR4 -- the mark is written where a spec can see it, and the two files
    agree on its name. They are edited apart and bound by a string.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HELPER = REPO / "frontend" / "tests" / "e2e" / "support" / "app.ts"
COMPONENT = (REPO / "frontend" / "src" / "lib" / "components" / "ui"
             / "OnboardingOverlay.svelte")

# The attribute the application sets once it has decided.
MARK = "data-onboarding"
RESOLVED = "resolved"


def _helper():
    return HELPER.read_text(encoding="utf-8")


def _component():
    return COMPONENT.read_text(encoding="utf-8")


def test_fr1_the_helper_carries_no_appearance_budget():
    text = _helper()
    assert "APPEARANCE_BUDGET_MS" not in text, (
        "a budget cannot tell 'not yet' from 'never'; that ambiguity is the "
        "race, and a larger budget only makes it rarer"
    )
    assert not re.search(r"catch\(\s*\(\s*\)\s*=>\s*false\s*\)", text), (
        "swallowing the timeout is what let the helper return as though "
        "there were nothing to dismiss"
    )
    assert "timeout:" not in text, (
        "no bounded wait belongs in this helper any more"
    )


def test_fr2_the_helper_waits_for_the_decision():
    text = _helper()
    assert MARK in text and RESOLVED in text, (
        "the helper must wait for the application to say it has decided, "
        f"which it does by setting {MARK}={RESOLVED}"
    )
    marker = text.index(MARK)
    dialog = text.index("getByRole('dialog'")
    assert marker < dialog, (
        "the wait must come BEFORE the dialog is looked for, or the look "
        "still happens against an undecided page"
    )


def test_fr3_the_component_marks_every_path_out():
    text = _component()
    assert MARK.replace("data-", "") in text or MARK in text, (
        "the component must mark the decision"
    )
    # Not merely "a finally exists somewhere": this component has others.
    # The mark must be set INSIDE one, or it does not cover every path.
    assert re.search(r"finally\s*\{[^{}]*" + MARK, text, re.S), (
        "the mark has to be set on every path out -- the install is "
        "configured, it is not, or the backend never answered -- and only a "
        "finally around the resolution covers all three"
    )
    # Proven capable: the component really does have the paths in question.
    assert "user_initialized" in text
    assert "MAX_RETRIES" in text


def test_fr4_the_two_files_agree_on_the_name_of_the_mark():
    helper, component = _helper(), _component()
    for text, where in ((helper, "the helper"), (component, "the component")):
        assert MARK in text, f"{where} does not name {MARK}"
        assert RESOLVED in text, f"{where} does not name {RESOLVED}"
    assert f'{MARK}="{RESOLVED}"' in helper or f"{MARK}='{RESOLVED}'" in helper, (
        "the helper must select on the resolved value, not merely mention it"
    )
