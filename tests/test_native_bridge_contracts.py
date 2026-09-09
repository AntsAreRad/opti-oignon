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

NB3 is the one worth stating plainly: the bridge documents that an integer
return is zero on success and negative on error, and that a byte array or
string return is null when unavailable. A stub that returned zero would
report that the node attached, the record opened and the write landed.
"""

import re
from pathlib import Path

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
