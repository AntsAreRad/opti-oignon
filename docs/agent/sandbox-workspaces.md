# Sandbox Workspaces

A workspace (S209-S213) is a named, user-managed sandbox bound to a
conversation, replacing the per-run, auto-destroyed session for interactive
work. It is still the same disposable, host-isolated bubblewrap environment
the agent's tools have always run in -- the cycle adds lifecycle, copy-in,
a diff-gated way back out, and one explicitly opt-in capability. Nothing in
it weakens the original contract: inputs enter only by explicit copy-in,
results leave only behind the human approval gate, and there is no host
fallback.

## Containment

Every workspace launch is contained by kernel namespaces (net, pid, ipc,
uts, cgroup), a seccomp-BPF syscall denylist, resource caps (memory,
process count, file size, CPU seconds, no core dumps), a tmpfs size cap,
and a cleared environment. The denylist is defense in depth; the namespaces
are the boundary. If the seccomp filter cannot be built and it is required,
the launch is refused rather than run unfiltered, and if bwrap itself is
unavailable, strict mode refuses execution entirely -- the sandbox never
silently degrades to the host.

## The manager

The sandbox panel lists workspaces with their age, running state, and
approximate disk use. You can create, select, stop, and destroy them, and
bind one to the current conversation so every agent run in that
conversation uses it. Binding is explicit and atomic, one conversation per
workspace, scoped to the owning user. Idle workspaces expire on a TTL
(bound ones are exempt); a disk soft quota and a concurrency cap bound the
whole surface; destroy remains the disposable guarantee.

## Copy-in

Host data enters a workspace in exactly two ways, both explicit user
actions:

- Drag-and-drop upload: file bytes only; no host path is exposed to the
  sandbox or the model.
- Host browse and clone: a server-side explorer restricted to an allowlist
  of roots ($HOME plus configured roots, never /), symlink-safe and
  size-capped, cloning a chosen directory into the workspace.

Every copy-in records a baseline manifest (per-file hashes). The model
cannot reach into the host on its own; copy-in is never model-triggerable.

## Diff review and apply

Copy-out is approval-gated, refined into a diff model. The review compares
the workspace against the copy-in baseline and shows added, modified, and
deleted files by hash. You approve changes individually; deletions are
confirmed separately from writes, so a deletion can never ride in on a
bulk approval. The apply writer is symlink-safe (temp-plus-rename,
no-follow opens, refusal on any path that escapes the chosen root), and
the applied set is bound by hash to the exact diff you reviewed: what is
written to the host is exactly what you saw, or nothing. Auto-apply does
not exist.

## The settings strip

Each workspace has a settings strip showing its command timeout
(per-workspace override or the configured default) and the configured
containment caps, read-only. It also hosts the network toggle and, when
the network is on, the provision row.

## The optional network

The sandbox network is default-off, per-workspace, and turns on only by
your explicit action in the strip. There is no configuration key that
enables it, no tool surface that reaches it, and it is never
model-triggerable. It is Daily-only at a fail-secure binding-layer gate:
under Bulbe -- or any unset, unknown, or undeterminable mode -- enabling is
refused and the toggle is disabled with the refusal stated. Turning the
network off is permitted in any mode. Every toggle, refusal, and run is
audited, per-session and in the hash chain.

The exfiltration warning, plainly: the copy-out approval gate controls
what leaves the sandbox through files. A network is a second exit that the
approval gate does not cover -- code running in a network-enabled workspace
could transmit workspace contents outward without any diff to review. The
warning is sharpened when host files have been cloned in, because that is
when the workspace holds data you may not want to leave the machine. Turn
the network on only for workspaces whose contents you would be comfortable
sending, and turn it off when the provision is done.

The only shipped egress is the provision phase: the server builds a fixed,
hash-pinned command (pip install --require-hashes --only-binary=:all:)
into a workspace venv from a requirements file you point at inside the
workspace. Requirements must be exact-and-hash-pinned; any pip option line
(-r, -e, --index-url, and the rest) is refused per line, and nothing
installs when any line is refused. Task code never touches the network:
every task run keeps the network unshared, and the network is off again by
construction after the provision run. A proxy-allowlist mode is designed
and labelled in the spec but not wired; raw network sharing is permanently
out of scope.

## Honest limitations

bwrap is required: without it, strict mode refuses execution -- there is no
host fallback to misbehave. The running enforcement (the seccomp kills,
the cap limits, the live provision run with real DNS, the rendered UI
walks) assures only on a real host; the container proves the argv, the
gates, the refusals, and the audit. The host-side checklist is
consolidated in SANDBOX_CYCLE_LIVE_WALK.md, with the bwrap argv baseline
capture ordered first.
