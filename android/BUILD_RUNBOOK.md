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
- **veilid-core** built for Android ABIs (`arm64-v8a`, `x86_64`) — needed only
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
returns null, so the inference client reports `NoReply`. That is expected — it
proves the wiring and the contract types compile and link, nothing more.

## 4. Implement the JNI bridge

Replace the stub bodies in `veilid-bridge/src/main/cpp/veilid_bridge.cpp` with
calls into veilid-core, and link veilid-core into the `optioignon_veilid` target
in `CMakeLists.txt`. The function signatures in `veilid_bridge.h` and the
`external fun` declarations in `VeilidBridge.kt` are the contract — keep them in
lockstep (the JNI names mangle the class `org/optioignon/veilid/VeilidBridge`).

Map each to veilid-core:

- `nodeInit` / `nodeAttach` / `nodeDetach` / `nodeShutdown` — node lifecycle.
- `routeAllocate` / `routeImport` — private route setup with the paired desktop.
- `appCall` — the core RPC: send the encoded request envelope over the route,
  return the reply bytes. Return null on transport failure.
- `recordOpen` / `recordGet` / `recordSet` / `recordClose` — DHT records for the
  note/vault sync surface.

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

## Note on the live round

The live two-device round (edit on one device, sync, verify on the other) is
gated on this app existing. Bringing this client up against a running desktop is
the prerequisite; the round itself is host-driven once the client is live.
