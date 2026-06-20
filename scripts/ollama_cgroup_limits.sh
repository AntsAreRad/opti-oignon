#!/usr/bin/env bash
# Ollama OS-level resource limits -- HOST-BOUND reference recipe (R-03, S226).
#
# Usage:
#   scripts/ollama_cgroup_limits.sh
#
# This script PRINTS the two supported recipes for capping the Ollama
# service with OS cgroups (systemd MemoryMax) and for pinning the
# OLLAMA_* limit variables where the server actually reads them. It
# never applies anything itself.
#
# WARNINGS -- read before using either recipe:
#   - HOST-BOUND, reference only (RESOURCE_GOVERNOR_SPEC.md Section 6).
#     This script is never executed by the application and is
#     never simulated in tests; only its existence and these warnings are
#     container-checkable facts.
#   - Both recipes require systemd and root (sudo) on the HOST machine.
#   - A cgroup memory cap applies to the WHOLE Ollama service: when the
#     cap is hit the kernel may OOM-kill the server mid-request.
#     Admission-side accounting in Opti-Oignon (R-01) remains the
#     primary control; this is a hard OS backstop for the user who
#     explicitly wants one.
#   - Values below are EXAMPLES. Size them to your card and RAM before
#     copying anything.

set -euo pipefail

cat <<'RECIPE'
Recipe 1 -- transient scope (one-off, gone at stop):

    sudo systemd-run --scope -p MemoryMax=24G -p MemoryHigh=22G \
        ollama serve

  MemoryHigh throttles before MemoryMax kills; set MemoryHigh a notch
  below MemoryMax.

Recipe 2 -- persistent unit drop-in (the documented systemd install):

    sudo mkdir -p /etc/systemd/system/ollama.service.d
    sudo tee /etc/systemd/system/ollama.service.d/10-limits.conf <<'EOF'
    [Service]
    MemoryMax=24G
    MemoryHigh=22G
    # The OLLAMA_* limit variables only take effect where the SERVER
    # reads them -- which is here, not in the Opti-Oignon process.
    # Mirror config/resource_governor.yaml ollama_limits values:
    Environment=OLLAMA_MAX_LOADED_MODELS=2
    Environment=OLLAMA_NUM_PARALLEL=1
    Environment=OLLAMA_MAX_QUEUE=128
    EOF
    sudo systemctl daemon-reload
    sudo systemctl restart ollama

Verification:

    systemctl show ollama -p MemoryMax -p MemoryHigh
    systemctl show ollama -p Environment

Nothing was applied by this script. Copy a recipe deliberately, with
values sized to your host.
RECIPE
