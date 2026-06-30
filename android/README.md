# Opti-Oignon mobile client (skeleton)

This directory is the starting tree for the Android client (Kotlin + JNI +
Veilid). It is a **skeleton, validated host-side** — there is no Android
toolchain in the environment it was generated in, so nothing here has been
compiled or run. Treat every Kotlin/native file as a contract to validate on a
real machine, not as working code.

## What is real vs. what is a stub

- **Real (and the point of this tree):** `MOBILE_SYNC_CONTRACT.md` — the exact
  wire behaviour the desktop responder already enforces and proves, restated for
  the phone. The Kotlin envelopes, the pull-loop client, and the consumer-only
  notes stance encode that contract directly. These are derived from, and match,
  the desktop source.
- **Stub (host-side work):** the JNI bridge to veilid-core
  (`veilid-bridge/src/main/cpp/`) returns not-implemented sentinels; the UI
  (`MainActivity`) is a placeholder; gradle plugin/SDK/NDK versions are pinned
  but must be checked against the host.

## Layout

```
android/
  MOBILE_SYNC_CONTRACT.md      the implementer's contract (the spine)
  BUILD_RUNBOOK.md             host-side build + validation steps
  settings.gradle.kts          modules: :app, :veilid-bridge
  build.gradle.kts             root plugin versions
  app/                         the client
    wire/Envelopes.kt          wire data classes + reason set
    sync/RemoteInferenceClient.kt   the pull-loop contract
    sync/NotesSyncContract.kt  consumer-only notes (no opt-in setter by design)
  veilid-bridge/               JNI -> veilid-core
    VeilidBridge.kt            external fun signatures (the native contract)
    cpp/                       JNI stubs (replace host-side)
```

## Where to start

1. Read `MOBILE_SYNC_CONTRACT.md`.
2. Follow `BUILD_RUNBOOK.md` to stand up the toolchain and build the skeleton.
3. Implement the JNI bodies against veilid-core, then validate the contract
   against a running desktop over a real route.

The live two-device round remains gated on this app existing; until then, the
contract is validated by a desktop-to-desktop exercise over a real route, and
this client is brought up against that.
