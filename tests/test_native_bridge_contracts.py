#!/usr/bin/env python3
"""Contracts for the agreement between the bridge declarations and the stubs.

A Kotlin external function and its native entry point are bound by name at
load time, not at build time. Nothing in either tree fails when they drift:
the Kotlin side compiles against a declaration that has no implementation,
the native side compiles an implementation nothing declares, and the pair
only meets on a device, at the first call, as an UnsatisfiedLinkError with
no compiler having objected anywhere along the way.

That is the whole reason these contracts read both files as text. The pair
they check is decidable without a JVM, without the Android toolchain and
without a device, so it can be checked on every change rather than on the
one occasion someone has a phone in hand.

  * NB1 -- every declared external function has a native entry point.
  * NB2 -- every native entry point has a declaration, so a renamed
    function cannot leave its old body behind to be bound by a stale caller.
  * NB3 -- every stub returns its documented not-implemented sentinel and
    never a zero, because a zero is success in both return conventions.
  * NB4 -- both parsers return something on the real files. A parser that
    quietly returns nothing makes every later check vacuously true.
  * NB5 -- the two name sets are equal, and neither side is short.
  * NB6 -- an arity mismatch is reported.
  * NB7 -- a type outside the agreed table is REFUSED, never mapped by
    default. A default mapping is a guess wearing the clothes of a check.
  * NB8 -- a swapped native type is reported.
  * NB9 -- a return type that disagrees is reported.
  * NB10 -- a nullable return keeps the native type its non-null spelling
    would have; nullability is a Kotlin fact with no native counterpart.
  * NB11 -- silence-zero. An empty declaration file is refused, and two
    empty sides are refused explicitly rather than agreeing by default.
  * NB12 -- a name carrying an underscore is unmangled from its JNI form.
    No real declaration carries one today, so the path exists only under a
    synthetic fixture, which is exactly why it needs a contract.
  * NB13 -- the checker is named in the workflow, since nothing else would
    notice it falling out.

NB3 is the one worth stating plainly: the bridge documents that an integer
return is zero on success and negative on error, and that a byte array or
string return is null when unavailable. A stub that returned zero would
report that the node attached, the record opened and the write landed.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BRIDGE = REPO / "android" / "veilid-bridge" / "src" / "main"
DECLARATIONS = BRIDGE / "kotlin" / "org" / "optioignon" / "veilid" / "VeilidBridge.kt"
NATIVE = BRIDGE / "cpp" / "veilid_bridge.cpp"

# The JNI name of an instance method on a class, which is what a Kotlin
# object's members compile to.
_ENTRY_PREFIX = "Java_org_optioignon_veilid_VeilidBridge_"

# Returns that mean "not implemented", by declared return type. A value
# outside this mapping is a return type nobody has decided a sentinel for.
_SENTINELS = {
    "jlong": "-1",
    "jint": "-1",
    "jbyteArray": "nullptr",
    "jstring": "nullptr",
}


def _declared_functions():
    """Return the external function names the bridge declares."""
    source = DECLARATIONS.read_text(encoding="utf-8")
    return set(re.findall(r"external fun (\w+)\(", source))


def _native_entries():
    """Return ``{name: (return_type, body)}`` for each native entry point."""
    source = NATIVE.read_text(encoding="utf-8")
    pattern = re.compile(
        r"JNIEXPORT\s+(\w+)\s+JNICALL\s*\n"
        + re.escape(_ENTRY_PREFIX)
        + r"(\w+)\([^)]*\)\s*\{(.*?)\n\}",
        re.S,
    )
    return {
        name: (return_type, body)
        for return_type, name, body in pattern.findall(source)
    }


def test_nb1_every_declared_function_has_a_native_entry_point():
    declared = _declared_functions()
    assert declared, "no external function is declared"
    missing = sorted(declared - set(_native_entries()))
    assert not missing, (
        f"declared with no native entry point: {missing}; each binds at the "
        "first call on a device and nowhere earlier"
    )


def test_nb2_every_native_entry_point_has_a_declaration():
    orphaned = sorted(set(_native_entries()) - _declared_functions())
    assert not orphaned, (
        f"native entry points nothing declares: {orphaned}"
    )


def test_nb3_every_stub_returns_its_not_implemented_sentinel():
    for name, (return_type, body) in sorted(_native_entries().items()):
        sentinel = _SENTINELS.get(return_type)
        assert sentinel is not None, (
            f"{name} returns {return_type}, which has no agreed sentinel"
        )
        returned = re.findall(r"return\s+([^;]+);", body)
        assert returned, f"{name} has no return statement"
        for value in returned:
            resolved = value.strip()
            if resolved.startswith("OO_"):
                constant = re.search(
                    r"#define\s+" + re.escape(resolved) + r"\s+\(?([^)\n]+)\)?",
                    NATIVE.read_text(encoding="utf-8"),
                )
                assert constant, f"{name} returns undefined {resolved}"
                resolved = constant.group(1).strip()
            assert resolved == sentinel, (
                f"{name} returns {resolved!r} where the not-implemented "
                f"sentinel for {return_type} is {sentinel!r}; a zero would "
                "report success"
            )


# ---------------------------------------------------------------------------
# NB4-NB13 -- the static concordance checker
# ---------------------------------------------------------------------------
_CONCORDANCE = REPO / "scripts" / "native_bridge_concordance.py"
_WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"


_CHECKER = "_native_bridge_concordance_under_contract"


def _checker():
    """Load the checker through the shared window; returns (module, restore).

    NB1 to NB3 read both files as text and import nothing, so they needed no
    window. These contracts drive the checker itself, which is a module: it
    goes through the shared window like every other module a suite loads, or
    the isolation guard reports the suite as a violation and is right to.

    The restore is returned rather than dropped. A window left open outlives
    the contract that opened it and breaks whatever runs next, which is the
    failure this window exists to prevent rather than to cause.
    """
    loaded, restore = isolate(targets={_CHECKER: _CONCORDANCE})
    return loaded[_CHECKER], restore


def _kotlin(*declarations):
    """A minimal declaration file carrying the given signatures."""
    body = "\n".join("    external fun " + one for one in declarations)
    return (
        "package org.optioignon.veilid\n\nobject VeilidBridge {\n"
        + body + "\n}\n"
    )


def _native(*prototypes):
    """A minimal native file carrying the given entry points."""
    blocks = []
    for return_type, name, params in prototypes:
        tail = "".join(", " + one for one in params)
        blocks.append(
            "JNIEXPORT " + return_type + " JNICALL\n"
            + _ENTRY_PREFIX + name + "(JNIEnv *, jobject" + tail
            + ") {\n    return -1;\n}\n"
        )
    return "\n".join(blocks)


def test_nb4_both_parsers_report_something_on_the_real_files():
    checker, restore = _checker()
    try:
        declared = checker.kotlin_externals(DECLARATIONS.read_text(encoding="utf-8"))
        exported = checker.jni_exports(NATIVE.read_text(encoding="utf-8"))
        assert len(declared) >= 11, (
            f"the declaration parser found {len(declared)}; a parser that finds "
            "nothing makes every later check vacuously true"
        )
        assert len(exported) >= 11, f"the export parser found {len(exported)}"
        # Proven capable in the other direction too.
        assert checker.kotlin_externals("") == []
        assert checker.jni_exports("") == []
    finally:
        restore()


def test_nb5_the_two_name_sets_are_equal_and_neither_is_short():
    checker, restore = _checker()
    try:
        declared = checker.kotlin_externals(DECLARATIONS.read_text(encoding="utf-8"))
        exported = checker.jni_exports(NATIVE.read_text(encoding="utf-8"))
        assert {one[0] for one in declared} == {one[0] for one in exported}
        assert not checker.concordance(declared, exported), (
            "the two real files must agree today"
        )
    finally:
        restore()


def test_nb6_an_arity_mismatch_is_reported():
    checker, restore = _checker()
    try:
        kotlin = checker.kotlin_externals(_kotlin("f(handle: Long): Int"))
        native = checker.jni_exports(_native(("jint", "f", ["jlong", "jint"])))
        reasons = checker.concordance(kotlin, native)
        assert reasons, "an extra native parameter must be reported"
        assert any("arity" in one or "parameter" in one for one in reasons), reasons
    finally:
        restore()


def test_nb7_a_type_outside_the_table_is_refused():
    checker, restore = _checker()
    try:
        kotlin = checker.kotlin_externals(_kotlin("f(handle: Widget): Int"))
        native = checker.jni_exports(_native(("jint", "f", ["jobject"])))
        reasons = checker.concordance(kotlin, native)
        assert reasons, (
            "an unknown Kotlin type must be refused, never mapped by default"
        )
        assert any("Widget" in one for one in reasons), reasons
    finally:
        restore()


def test_nb8_a_swapped_native_type_is_reported():
    checker, restore = _checker()
    try:
        kotlin = checker.kotlin_externals(_kotlin("f(handle: Long, key: String): Int"))
        native = checker.jni_exports(_native(("jint", "f", ["jstring", "jlong"])))
        assert checker.concordance(kotlin, native), (
            "two parameters exchanged must not pass as agreement"
        )
    finally:
        restore()


def test_nb9_a_return_type_that_disagrees_is_reported():
    checker, restore = _checker()
    try:
        kotlin = checker.kotlin_externals(_kotlin("f(handle: Long): Int"))
        native = checker.jni_exports(_native(("jlong", "f", ["jlong"])))
        assert checker.concordance(kotlin, native), (
            "a return type that disagrees must be reported"
        )
    finally:
        restore()


def test_nb10_a_nullable_return_keeps_its_native_type():
    checker, restore = _checker()
    try:
        kotlin = checker.kotlin_externals(_kotlin("f(handle: Long): ByteArray?"))
        native = checker.jni_exports(_native(("jbyteArray", "f", ["jlong"])))
        assert kotlin[0][3] is True, "the parser must see the nullability"
        assert not checker.concordance(kotlin, native), (
            "nullability is a Kotlin fact; the native type is unchanged by it"
        )
    finally:
        restore()


def test_nb11_zero_declarations_is_refused_not_agreed():
    checker, restore = _checker()
    try:
        assert checker.concordance([], []), (
            "two empty sides agree about nothing and must be refused explicitly"
        )
        empty = checker.kotlin_externals(_kotlin())
        native = checker.jni_exports(_native(("jint", "f", ["jlong"])))
        assert checker.concordance(empty, native), (
            "a declaration file with nothing in it is a parse failure or a "
            "deletion; either way it is not concordance"
        )
    finally:
        restore()


def test_nb12_a_mangled_underscore_is_unmangled():
    checker, restore = _checker()
    try:
        exported = checker.jni_exports(
            _native(("jint", "node_1init", ["jlong"])),
        )
        assert [one[0] for one in exported] == ["node_init"], (
            "JNI writes a literal underscore as _1; a name split on the last "
            "underscore would read this as node and 1init: " + str(exported)
        )
    finally:
        restore()


def test_nb13_the_checker_is_named_in_the_workflow():
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "native_bridge_concordance.py" in text, (
        "the workflow contract globs .github/scripts only, so nothing else "
        "would notice this checker falling out of CI"
    )
