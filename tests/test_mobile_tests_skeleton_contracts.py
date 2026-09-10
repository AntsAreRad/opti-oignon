#!/usr/bin/env python3
"""Contracts for the JVM-pure test skeleton on the phone side.

The wire envelopes are the desktop responder's literal contract. They are
plain Kotlin data classes with a strict codec, so their behaviour is decidable
on a JVM alone -- no device, no emulator, no native library. That is the whole
point of a `src/test` skeleton rather than `src/androidTest`: the cheapest
place these can be checked is a unit test, and until one exists they are
checked nowhere.

These contracts do not run the Kotlin tests; running them needs the Gradle
wrapper, which is owed. They pin that the skeleton exists, that it mirrors the
surface it claims to cover, and that its absence from CI is stated rather than
silent.

  * MT1 -- one test class per public type in Envelopes.kt, the list parsed
    from the source rather than written down here. A skeleton that mirrors a
    hand-copied list stops mirroring the moment the source moves.
  * MT2 -- each test class USES the type it covers, in code rather than
    in a name or a sentence. A class that only mentions its type in a
    method name exercises nothing.
  * MT3 -- every test method carries a real assertion, and none is a
    tautology. A test that passes by construction is worse than no test: it
    reports coverage that does not exist.
  * MT4 -- the runbook says these run on the host through the wrapper.
  * MT5 -- the workflow says CI skips them, by name. An execution that is
    absent for a good reason must say so where the reader looks.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_WIRE = REPO / "android" / "app" / "src" / "main" / "kotlin" / "org" / "optioignon" / "mobile" / "wire"
ENVELOPES = _WIRE / "Envelopes.kt"
SKELETON = (REPO / "android" / "app" / "src" / "test" / "kotlin" / "org"
            / "optioignon" / "mobile" / "wire" / "EnvelopesTest.kt")
RUNBOOK = REPO / "android" / "BUILD_RUNBOOK.md"
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"

# A tautology is an assertion whose truth does not depend on the code under
# test. These are the shapes that pass whatever the surface does.
_TAUTOLOGIES = (
    "assertTrue(true)", "assertFalse(false)", "assertEquals(1, 1)",
    "assertNotNull(this)",
)


def _public_types():
    """Top-level public types Envelopes.kt declares, parsed from the source."""
    source = ENVELOPES.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"^(?:@Serializable\s*\n)?"
                                 r"(?:object|data class) (\w+)",
                                 source, re.M)))


def _test_classes():
    """Test class names the skeleton declares."""
    source = SKELETON.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"^class (\w+)", source, re.M)))


def _class_bodies():
    """``{class name: body}`` for each test class, declaration line excluded.

    The declaration carries the class name, and the class name carries the
    covered type: including it would make MT2 true of every class whatever
    its body said. Its own blade is what found that.
    """
    source = SKELETON.read_text(encoding="utf-8")
    found = {}
    starts = [(m.group(1), m.end()) for m in re.finditer(r"^class (\w+)[^\n]*\n",
                                                         source, re.M)]
    heads = [m.start() for m in re.finditer(r"^class \w+", source, re.M)]
    for index, (name, start) in enumerate(starts):
        end = heads[index + 1] if index + 1 < len(heads) else len(source)
        found[name] = source[start:end]
    return found


def test_mt1_one_test_class_per_public_type():
    types = _public_types()
    assert len(types) >= 5, (
        f"the parser found {len(types)} public type(s) in Envelopes.kt; a "
        "parser that finds none would make every check below it vacuous"
    )
    expected = {name + "Test" for name in types}
    assert set(_test_classes()) == expected, (
        f"the skeleton must mirror the surface: {sorted(expected)}"
    )


def test_mt2_each_test_class_uses_the_type_it_covers():
    """The reference must be in code, not in a name or a sentence.

    A method called ``everyReasonIsDistinct`` mentions Reason and exercises
    nothing. Its own blade found that: the first two spellings of this
    contract were satisfied by the class declaration and then by the method
    name, and both would have passed over a class that touched no type at
    all. What counts is a member access, a construction or a type argument.
    """
    for name, body in sorted(_class_bodies().items()):
        covered = name[:-len("Test")]
        used = (covered + "." in body
                or covered + "(" in body
                or "<" + covered + ">" in body)
        assert used, (
            f"{name} never uses {covered} in code; a mirror that reflects "
            "nothing is a name with no test behind it"
        )


def test_mt3_every_test_method_asserts_something_real():
    source = SKELETON.read_text(encoding="utf-8")
    methods = re.findall(r"@Test\s*\n\s*fun (\w+)\(\)\s*\{(.*?)\n    \}",
                         source, re.S)
    assert len(methods) >= 5, (
        f"the parser found {len(methods)} test method(s); proven capable of "
        "finding more than none"
    )
    for name, body in methods:
        assertions = re.findall(r"\bassert\w*\(", body)
        assert assertions, f"{name} carries no assertion"
        for tautology in _TAUTOLOGIES:
            assert tautology not in body.replace(" ", ""), (
                f"{name} contains {tautology}, which passes whatever the "
                "surface does"
            )


def test_mt4_the_runbook_names_the_host_execution():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "app:test" in text, (
        "the runbook must name the command that runs these on the host"
    )
    assert "EnvelopesTest" in text, (
        "and name what it runs, so the reader can find it"
    )


def test_mt5_the_workflow_names_the_skip():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "EnvelopesTest" in text, (
        "CI does not run these and must say so by name; an execution absent "
        "without a word reads as an execution that passed"
    )
