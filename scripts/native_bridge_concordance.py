#!/usr/bin/env python3
"""Static concordance of the bridge declarations with the native exports.

A Kotlin external function and its JNI entry point are bound by NAME, at load
time, on a device. Nothing earlier objects: the Kotlin side compiles against a
declaration with no implementation, the native side compiles an implementation
nothing declares, and the pair meets for the first time as an
UnsatisfiedLinkError in someone's hand. The types are worse than the names,
because a mismatched parameter binds successfully and then reads the wrong
bytes.

Both files are text, and the agreement between them is decidable without a
JVM, without the Android toolchain and without a device. So it is decided
here, on every change, rather than on the one occasion someone has a phone.

What this proves and what it does not: it proves the two TEXTS agree. Only the
machine can prove the binding holds -- that the library loads, that the symbols
resolve, that the compiler emits the receiver this module assumes. Those stay
owed.

Three refusals worth stating, because each is a way to agree about nothing:

  * A type outside the table is REFUSED, never mapped by default. A default
    mapping is a guess wearing the clothes of a check.
  * Zero declarations on either side is REFUSED. Two empty lists agree
    perfectly and prove nothing; a parser that silently returns nothing would
    otherwise make every check below it vacuously true.
  * The receiver is pinned to ``jobject``. The declarations live in a Kotlin
    ``object`` with no ``@JvmStatic``, so they compile to instance methods. A
    future ``@JvmStatic`` emits ``jclass`` instead, and it must change this
    contract rather than pass unnoticed.
"""

from pathlib import Path

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

_ROOT = Path(__file__).resolve().parents[1]
_BRIDGE = _ROOT / "android" / "veilid-bridge" / "src" / "main"
DECLARATIONS = _BRIDGE / "kotlin" / "org" / "optioignon" / "veilid" / "VeilidBridge.kt"
NATIVE = _BRIDGE / "cpp" / "veilid_bridge.cpp"

_ENTRY_PREFIX = "Java_org_optioignon_veilid_VeilidBridge_"

# The agreed correspondence. A Kotlin type absent from this table is refused,
# so widening the bridge is a deliberate edit here rather than a silent pass.
_TYPES = {
    "String": "jstring",
    "Long": "jlong",
    "Int": "jint",
    "ByteArray": "jbyteArray",
    "Boolean": "jboolean",
    "Unit": "void",
}

# The receiver every entry point carries before its declared parameters.
_RECEIVER = ("JNIEnv", "jobject")


def _unmangle(name):
    """The Java method name behind a JNI symbol suffix.

    JNI escapes what a C identifier cannot carry: ``_1`` is a literal
    underscore, ``_2`` a semicolon, ``_3`` a left bracket. No declaration in
    this bridge carries one today, which is precisely why this is written out
    rather than guessed -- splitting on the last underscore is correct only
    while every declared name happens to have none.
    """
    out = []
    index = 0
    while index < len(name):
        char = name[index]
        if char == "_" and index + 1 < len(name):
            escape = name[index + 1]
            if escape == "1":
                out.append("_")
                index += 2
                continue
            if escape == "2":
                out.append(";")
                index += 2
                continue
            if escape == "3":
                out.append("[")
                index += 2
                continue
        out.append(char)
        index += 1
    return "".join(out)


def kotlin_externals(text):
    """``[(name, [types], return_type, nullable)]`` for each declaration.

    Returns an empty list on text that declares nothing. The caller must treat
    that as a refusal rather than as agreement -- see ``concordance``.
    """
    found = []
    for line in text.splitlines():
        stripped = line.strip()
        marker = "external fun "
        if not stripped.startswith(marker):
            continue
        signature = stripped[len(marker):]
        if "(" not in signature or ")" not in signature:
            continue
        name = signature[:signature.index("(")].strip()
        inside = signature[signature.index("(") + 1:signature.rindex(")")]
        types = []
        for parameter in inside.split(","):
            if ":" not in parameter:
                continue
            types.append(parameter.split(":", 1)[1].strip())
        tail = signature[signature.rindex(")") + 1:].strip()
        return_type = tail[1:].strip() if tail.startswith(":") else "Unit"
        nullable = return_type.endswith("?")
        found.append((name, types, return_type.rstrip("?"), nullable))
    return found


def jni_exports(text):
    """``[(name, [jtypes], jreturn)]`` for each entry point, receiver removed.

    Parameter names are optional in a prototype and absent in the stubs, so
    each parameter is read as its leading type tokens with any name dropped.
    """
    found = []
    for index, line in enumerate(text.splitlines()):
        position = line.find(_ENTRY_PREFIX)
        if position < 0 or "(" not in line:
            continue
        return_type = _return_type_before(text.splitlines(), index, line, position)
        if return_type is None:
            continue
        symbol = line[position + len(_ENTRY_PREFIX):line.index("(", position)]
        inside = line[line.index("(", position) + 1:]
        inside = inside[:inside.rindex(")")] if ")" in inside else inside
        parameters = [_parameter_type(one) for one in inside.split(",") if one.strip()]
        if len(parameters) < len(_RECEIVER):
            continue
        found.append((_unmangle(symbol), parameters[len(_RECEIVER):], return_type))
    return found


def _return_type_before(lines, index, line, position):
    """The declared return type, on this line or the one above it."""
    head = line[:position].strip()
    if not head and index:
        head = lines[index - 1].strip()
    tokens = head.replace("JNIEXPORT", " ").replace("JNICALL", " ").split()
    return tokens[-1] if tokens else None


def _parameter_type(parameter):
    """A parameter's type, with any name and pointer spelling normalised."""
    cleaned = parameter.replace("*", " * ").strip()
    tokens = cleaned.split()
    if len(tokens) > 1 and tokens[-1] not in {"*"} and "*" not in tokens[-1]:
        # A trailing identifier is a parameter name, not part of the type.
        if tokens[-2] != "*":
            tokens = tokens[:-1]
    return " ".join(tokens).replace(" * ", " *").strip()


def concordance(declared, exported):
    """Reasons the two sides disagree; empty means concordant."""
    reasons = []
    if not declared or not exported:
        reasons.append(
            f"nothing to compare: {len(declared)} declaration(s) and "
            f"{len(exported)} entry point(s); two empty sides agree about "
            "nothing"
        )
        return reasons

    by_name = {one[0]: one for one in exported}
    for name, types, return_type, _nullable in declared:
        entry = by_name.get(name)
        if entry is None:
            reasons.append(f"{name}: declared with no native entry point")
            continue
        _, jtypes, jreturn = entry
        if len(jtypes) != len(types):
            reasons.append(
                f"{name}: arity, {len(types)} declared parameter(s) against "
                f"{len(jtypes)} native"
            )
            continue
        for position, (one, two) in enumerate(zip(types, jtypes), start=1):
            expected = _TYPES.get(one.rstrip("?"))
            if expected is None:
                reasons.append(
                    f"{name}: parameter {position} is {one}, which has no "
                    "agreed native type"
                )
            elif expected != two:
                reasons.append(
                    f"{name}: parameter {position} is {one}, which is "
                    f"{expected}, but the entry point takes {two}"
                )
        expected = _TYPES.get(return_type)
        if expected is None:
            reasons.append(
                f"{name}: returns {return_type}, which has no agreed native "
                "type"
            )
        elif expected != jreturn:
            reasons.append(
                f"{name}: returns {return_type}, which is {expected}, but the "
                f"entry point returns {jreturn}"
            )

    for name in sorted(set(by_name) - {one[0] for one in declared}):
        reasons.append(f"{name}: native entry point nothing declares")
    return reasons


def main(argv=None):
    """Compare the two real files; non-zero on any reason."""
    declared = kotlin_externals(DECLARATIONS.read_text(encoding="utf-8"))
    exported = jni_exports(NATIVE.read_text(encoding="utf-8"))
    reasons = concordance(declared, exported)
    if reasons:
        print("native bridge concordance: FAILED")
        for reason in reasons:
            print(f"  {reason}")
        return 1
    print(
        f"native bridge concordance: {len(declared)} declaration(s) and "
        f"{len(exported)} entry point(s) agree in name, arity and type"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
