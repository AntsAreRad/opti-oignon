# Build runbook (host-side)

This skeleton cannot be built or validated in the environment it was generated
in. Everything below runs on a host with an Android toolchain. The steps are
ordered: stand up the toolchain, build the stub, then implement and validate the
native bridge.

## 1. Prerequisites

- **JDK 17** (the build targets Java 17).
- **Android SDK** command-line tools, with:
  - Platform `android-34` (compile/target SDK 34).
  - Build-tools matching the platform.
- **Android NDK** r26 or newer (for the JNI C++ in `veilid-bridge`).
- **CMake 3.22.1+** (the native build uses CMake).
- **veilid-core** built for Android ABIs (`arm64-v8a`, `x86_64`) -- needed only
  once you implement the JNI bodies (step 4).

Pin or verify the Gradle/AGP/Kotlin versions in `build.gradle.kts` against the
installed SDK before the first build; the pinned values are a starting point.

## 2. Generate the Gradle wrapper

No wrapper jar is committed. From `android/`:

```
gradle wrapper --gradle-version 8.9
```

(Use a Gradle version compatible with the Android Gradle Plugin in
`build.gradle.kts`.) Then use `./gradlew` for everything below.

## 3. Build the skeleton (stub native side)

```
./gradlew :veilid-bridge:assembleDebug
./gradlew :app:assembleDebug
```

This compiles the Kotlin, builds the JNI stub `.so`, and packages a debug APK.
At this stage the app runs but every Veilid call returns a sentinel: `appCall`
returns null, so the inference client reports `NoReply`. That is expected -- it
proves the wiring and the contract types compile and link, nothing more.

## 4. Implement the JNI bridge

Replace the stub bodies in `veilid-bridge/src/main/cpp/veilid_bridge.cpp` with
calls into veilid-core, and link veilid-core into the `optioignon_veilid` target
in `CMakeLists.txt`. The function signatures in `veilid_bridge.h` and the
`external fun` declarations in `VeilidBridge.kt` are the contract -- keep them in
lockstep (the JNI names mangle the class `org/optioignon/veilid/VeilidBridge`).

Map each to veilid-core:

- `nodeInit` / `nodeAttach` / `nodeDetach` / `nodeShutdown` -- node lifecycle.
- `routeAllocate` / `routeImport` -- private route setup with the paired desktop.
- `appCall` -- the core RPC: send the encoded request envelope over the route,
  return the reply bytes. Return null on transport failure.
- `recordOpen` / `recordGet` / `recordSet` / `recordClose` -- DHT records for the
  note/vault sync surface.

## 4b. Unit tests for the wire envelopes (host, no device)

The envelopes in `app/src/main/kotlin/org/optioignon/mobile/wire/Envelopes.kt`
are plain data classes with a strict codec, so their behaviour is decidable on
a JVM alone. `EnvelopesTest.kt` checks the round trips, the omitted null, the
disjointness of a success and a refusal, and that no two refusal reasons share
a string.

From `android/`, once the wrapper exists (section 2):

```bash
./gradlew :app:test
```

**CI does not run these.** The workflow has no JVM toolchain and no Gradle
wrapper, so `EnvelopesTest` is skipped there by name rather than silently: an
execution that is absent without a word reads as an execution that passed.
Until the wrapper is generated on this machine, that run is owed and no number
from it may be quoted.

## 4c. Interlanguage round trip (host, pinned compiler)

The concordance checker proves the declarations and the native exports agree
as text, and it proves it on every change. It cannot prove the binding holds:
that the library loads, that the symbol resolves, that the value crosses back
intact. Only a compiler and a running JVM establish that.

```bash
python3 scripts/native_bridge_roundtrip.py
```

It builds the smallest thing that can fail for the right reason -- one Kotlin
`external fun`, one C++ implementation returning the not-implemented sentinel
-- compiles both, runs them, and compares what comes back with what was sent.
A wrong mangling fails to resolve; a wrong receiver is refused by the JVM; a
wrong type arrives corrupted.

Three exit codes, and the third is the point:

| code | meaning |
|---|---|
| 0 | the round trip ran and the value crossed back intact |
| 1 | it ran and something is wrong |
| 2 | **owed** -- it did not run, and no number from it may be quoted |

The measurement is pinned to `kotlinc 2.0.20`. A different compiler is
refused rather than accepted quietly, because a number without the compiler
that produced it is not a measurement. Record which install produced it here
when you run it.

**CI does not run this**, for the same reason it does not run `EnvelopesTest`:
no Kotlin toolchain. Only the script's shape is checked there.

## 5. Validate the contract (the real test)

Runtime validation is host-side and is what finally confirms the contract:

1. Pair the phone with a desktop (complete mutual confirmation).
2. Establish the private route.
3. Run one remote chat turn via `RemoteInferenceClient.chat(...)` and confirm:
   - a normal prompt assembles across chunks and terminates on `done`;
   - a request with `rag` is refused `rag_not_granted` until the desktop grants
     the sub-grant;
   - a field outside the surface is refused `out_of_surface`;
   - a revoke at the desktop turns the next pull into `buffer_mismatch` and a
     fresh request into `remote_chat_disabled`;
   - with the desktop in Bulbe, the client sees `NoReply` (no refusal envelope).
4. Confirm note sync delivers only desktop-opted-in notes, and that the phone
   has no path that sets the opt-in.

Cross-check each observed reason against `MOBILE_SYNC_CONTRACT.md`. The desktop
side of every one of these is already covered by the backend test suite; this
step confirms the phone speaks the same contract over a real route.

## 6. What must exist before a two-device round is attempted

A two-device round over the live transport is not gated on enthusiasm. It is
gated on one artefact, named here so that its absence is visible rather than
argued about:

**A bound bridge**: `libveilid_bridge.so`, built for `arm64-v8a` and
`x86_64`, whose eleven entry points call into `veilid-core` rather than
returning the not-implemented sentinel.

It exists when all four of these hold, and not before:

| | check | where |
|---|---|---|
| 1 | `python3 scripts/native_bridge_concordance.py` returns 0 | anywhere, every change |
| 2 | `python3 scripts/native_bridge_roundtrip.py` returns 0 | this machine, pinned compiler |
| 3 | no entry point returns the not-implemented sentinel any more | this machine |
| 4 | `./gradlew :app:test` runs and passes | this machine, needs the wrapper |

Checks 1 and 2 prove the mechanism: the names agree, and a value crosses back
intact. They say nothing about whether the bridge does anything. Check 3 is
what separates a proven mechanism from a working bridge, and it is the one
that needs `veilid-core` built for the two ABIs.

Until check 3 passes, a failed two-device round cannot be told apart from an
unimplemented one, and no conclusion may be drawn from attempting it. That is
the whole reason this list exists: an attempt made too early produces a
result that looks like evidence and is not.

## Note on the live round

The live two-device round (edit on one device, sync, verify on the other) is
gated on this app existing. Bringing this client up against a running desktop is
the prerequisite; the round itself is host-driven once the client is live.
