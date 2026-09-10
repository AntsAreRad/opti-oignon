#!/usr/bin/env python3
"""Drive a real interlanguage round trip: Kotlin declares, C++ answers.

The bridge is two languages meeting at a name. A sibling script proves the
two texts agree about that name, its arity and its types, and it proves it
anywhere, on every change. What it cannot prove is that the binding holds:
that the library loads, that the symbol resolves, that the value crosses back
intact. Only a compiler and a running JVM establish that.

So this script builds the smallest thing that can fail for the right reason.
A Kotlin object declares one external function; a C++ file implements it and
returns the bridge's not-implemented sentinel; the two are compiled, linked
and run, and the value that comes back is compared with the value that was
sent. If the mangling is wrong the symbol does not resolve. If the receiver
is wrong the JVM refuses. If the types disagree the value arrives corrupted.
Each of those is a real failure of the real mechanism.

THE PART WORTH READING. This measurement cannot run everywhere, and a script
that cannot run its measurement has three honest outcomes and one dishonest
one. It may run and pass, run and fail, or refuse to run and say what it
lacks. What it must never do is report success having measured nothing --
which is the easy mistake, because from the outside the absence of a failure
is indistinguishable from a pass.

Hence three exit codes rather than two. OWED is not a pass and not a failure:
it is the script saying the number does not exist yet. Nothing downstream may
round it to either.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

OK = 0
FAILED = 1
OWED = 2

# The compiler the measurement is pinned to. A runbook that does not say
# which compiler produced a number has not recorded a measurement.
KOTLIN_VERSION = "2.0.20"

# The bridge returns -1 from a stub, never 0: zero is success in both return
# conventions, so a round trip asserting zero would pass on a stub that did
# nothing at all.
SENTINEL = -1

REQUIRED = ("kotlinc", "javac", "java", "g++")

_PACKAGE = "org.optioignon.veilid.roundtrip"
_CLASS = "RoundTrip"
_METHOD = "answer"

_KOTLIN = """package {package}

object {cls} {{
    external fun {method}(handle: Long): Int

    @JvmStatic
    fun main(args: Array<String>) {{
        System.loadLibrary("roundtrip")
        val value = {method}(1L)
        println("roundtrip=" + value)
    }}
}}
"""

_CPP = """#include <jni.h>

extern "C" JNIEXPORT jint JNICALL
Java_org_optioignon_veilid_roundtrip_{cls}_{method}(JNIEnv *, jobject, jlong) {{
    return {sentinel};
}}
"""


def missing_tools():
    """Which of the required tools this machine does not have."""
    return [name for name in REQUIRED if shutil.which(name) is None]


def kotlin_version_ok(text):
    """True when ``kotlinc -version`` output names the pinned version."""
    return KOTLIN_VERSION in (text or "")


def _java_home():
    """The JDK root, for the JNI headers g++ needs."""
    javac = shutil.which("javac")
    if javac is None:
        return None
    return Path(javac).resolve().parent.parent


def _run(command, cwd):
    return subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)


def _run_round_trip(work):
    """Build and run the harness; return (code, message)."""
    home = _java_home()
    if home is None:
        return OWED, "javac disappeared between the check and the build"

    kotlin = work / f"{_CLASS}.kt"
    kotlin.write_text(
        _KOTLIN.format(package=_PACKAGE, cls=_CLASS, method=_METHOD),
        encoding="utf-8",
    )
    native = work / "roundtrip.cpp"
    native.write_text(
        _CPP.format(cls=_CLASS, method=_METHOD, sentinel=SENTINEL),
        encoding="utf-8",
    )

    built = _run(["kotlinc", str(kotlin), "-include-runtime",
                  "-d", "roundtrip.jar"], work)
    if built.returncode != 0:
        return FAILED, "the Kotlin side did not compile:\n" + built.stderr

    compiled = _run(
        ["g++", "-shared", "-fPIC",
         f"-I{home}/include", f"-I{home}/include/linux",
         str(native), "-o", "libroundtrip.so"],
        work,
    )
    if compiled.returncode != 0:
        return FAILED, "the native side did not compile:\n" + compiled.stderr

    ran = _run(["java", f"-Djava.library.path={work}",
                "-jar", "roundtrip.jar"], work)
    if ran.returncode != 0:
        return FAILED, (
            "the round trip did not run; a symbol that does not resolve "
            "fails here and nowhere earlier:\n" + ran.stderr
        )

    expected = f"roundtrip={SENTINEL}"
    if expected not in ran.stdout:
        return FAILED, (
            f"expected {expected!r} to come back across the bridge, got:\n"
            + ran.stdout
        )
    return OK, f"the value crossed back intact: {expected}"


def main(argv=None):
    """Run the round trip, or say plainly that it is owed."""
    absent = missing_tools()
    if absent:
        print("native bridge round trip: OWED -- not run, and no number from "
              "it may be quoted")
        print(f"  missing: {', '.join(absent)}")
        print(f"  this measurement needs kotlinc {KOTLIN_VERSION}, a JDK and "
              "g++ on the machine that reports it")
        return OWED

    version = _run(["kotlinc", "-version"], Path.cwd())
    reported = version.stdout + version.stderr
    if not kotlin_version_ok(reported):
        print("native bridge round trip: OWED -- not run")
        print(f"  the measurement is pinned to kotlinc {KOTLIN_VERSION}; this "
              f"machine reports: {reported.strip() or 'nothing readable'}")
        print("  a number from another compiler is a different measurement")
        return OWED

    with tempfile.TemporaryDirectory() as name:
        code, message = _run_round_trip(Path(name))

    if code == OK:
        print("native bridge round trip: PASSED")
        print(f"  {message}")
        return OK
    if code == OWED:
        print("native bridge round trip: OWED -- not run")
        print(f"  {message}")
        return OWED
    print("native bridge round trip: FAILED")
    print(f"  {message}")
    return FAILED


if __name__ == "__main__":
    raise SystemExit(main())
