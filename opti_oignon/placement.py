#!/usr/bin/env python3
"""A measured placement, expressed as a recipe rather than as a launch.

This repository never starts llama-server. The process is launched host-side
and consumed through ``inference_backend.LlamaServerBackend``; what belongs
here is the DESCRIPTION of how a model should be laid out across the memory
hierarchy, and the rule for turning that description into an argv. The same
plan must answer the same command every time, because a placement that cannot
be reproduced cannot be compared, and a placement that cannot be compared
cannot be measured.

A plan says only what an operator chose. Every field defaults to "say nothing",
and a field that says nothing contributes no argument at all -- so adding this
module changes no existing command until someone writes a plan.

On refusing rather than guessing: ``inference_backend._resolve_ggml_kv_type``
deliberately fails open on a cache type the installed library does not expose,
because a performance knob must never block a model load. That decision is
right there and is left alone. It does not transfer here. A recipe has no
library to interrogate and no load to protect; an unrecognised cache type in a
plan is a placement that was never going to be what its author asked for, so it
is refused by name at validation instead of being dropped into a default the
author did not choose.

What is NOT decided here, and is owed to the machine: whether any particular
placement is faster than any other. This module can prove that a plan says what
it means; only the host can say whether it was a good plan.
"""

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

checkpoint_before_apply = True

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "placement.yaml"

# The cache types llama-server accepts for --cache-type-k / --cache-type-v.
# A closed allowlist on purpose: the recipe is written before anything runs,
# so a typo here surfaces as a refusal now rather than as a server that fails
# to start later, or one that starts with a cache type nobody asked for.
KV_CACHE_TYPES = frozenset({
    "f32", "f16", "bf16",
    "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl",
})

# The context-extension methods llama-server accepts for --rope-scaling.
ROPE_SCALING_METHODS = frozenset({"none", "linear", "yarn"})


@dataclass
class PlacementPlan:
    """One model's placement, in llama-server's own terms.

    Every field is optional and means "say nothing" when left alone, so an
    empty plan resolves to an empty list of arguments.

    ``override_tensor`` carries an operator's tensor-routing expression (the
    ``-ot`` form, e.g. ``\\.ffn_.*_exps\\.=CPU`` to keep mixture experts in
    system RAM while attention and the KV cache stay on the card). It reaches
    the command verbatim: it is a regular expression in llama.cpp's own
    dialect, and rewriting or validating it here would only add a second
    dialect to be wrong in.
    """

    # Where the weights go.
    n_gpu_layers: int | None = None
    override_tensor: str | None = None
    n_cpu_moe: int | None = None

    # Where the KV cache goes, and in what precision.
    no_kv_offload: bool = False
    type_k: str | None = None
    type_v: str | None = None

    # Attention implementation.
    flash_attn: bool = False

    # Context extension.
    rope_scaling: str | None = None
    rope_scale: float | None = None
    yarn_orig_ctx: int | None = None

    def validate(self) -> list[str]:
        """Return every reason this plan cannot be honoured, or an empty list.

        Loud by design and exhaustive: a caller that fixes one refusal should
        not have to run again to discover the next one.
        """
        errors: list[str] = []

        for label, value in (("type_k", self.type_k), ("type_v", self.type_v)):
            if value is not None and str(value) not in KV_CACHE_TYPES:
                errors.append(
                    f"{label}: unknown KV cache type {value!r}; "
                    f"expected one of {', '.join(sorted(KV_CACHE_TYPES))}"
                )

        if (
            self.rope_scaling is not None
            and str(self.rope_scaling) not in ROPE_SCALING_METHODS
        ):
            errors.append(
                f"rope_scaling: unknown method {self.rope_scaling!r}; "
                f"expected one of {', '.join(sorted(ROPE_SCALING_METHODS))}"
            )

        if self.n_gpu_layers is not None and int(self.n_gpu_layers) < 0:
            errors.append(
                f"n_gpu_layers: {self.n_gpu_layers} is negative; use 0 for "
                "none and a positive count for the rest"
            )

        if self.n_cpu_moe is not None and int(self.n_cpu_moe) < 0:
            errors.append(f"n_cpu_moe: {self.n_cpu_moe} is negative")

        if self.rope_scale is not None and float(self.rope_scale) <= 0:
            errors.append(
                f"rope_scale: {self.rope_scale} must be greater than zero"
            )

        if self.yarn_orig_ctx is not None and int(self.yarn_orig_ctx) <= 0:
            errors.append(
                f"yarn_orig_ctx: {self.yarn_orig_ctx} must be greater than zero"
            )

        return errors

    def to_args(self) -> list[str]:
        """Resolve this plan into llama-server arguments.

        Order is fixed so that two runs of the same plan are comparable by
        string equality, not merely by set membership.
        """
        args: list[str] = []

        if self.n_gpu_layers is not None:
            args += ["-ngl", str(int(self.n_gpu_layers))]
        if self.override_tensor:
            args += ["-ot", str(self.override_tensor)]
        if self.n_cpu_moe is not None:
            args += ["--n-cpu-moe", str(int(self.n_cpu_moe))]

        if self.no_kv_offload:
            args.append("--no-kv-offload")
        if self.flash_attn:
            args.append("--flash-attn")
        if self.type_k:
            args += ["--cache-type-k", str(self.type_k)]
        if self.type_v:
            args += ["--cache-type-v", str(self.type_v)]

        if self.rope_scaling:
            args += ["--rope-scaling", str(self.rope_scaling)]
        if self.rope_scale is not None:
            args += ["--rope-scale", str(float(self.rope_scale))]
        if self.yarn_orig_ctx is not None:
            args += ["--yarn-orig-ctx", str(int(self.yarn_orig_ctx))]

        return args

    def to_dict(self) -> dict[str, Any]:
        """Serialise, keeping the fields that say something."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value not in (None, False):
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlacementPlan":
        """Build from a mapping, ignoring keys this module does not know.

        An unknown key is dropped with a log line rather than raising: a plan
        file may legitimately be newer than the code reading it, and refusing
        the whole placement over one unread key would be worse than saying so.
        An unknown VALUE for a key this module does own is a different matter
        and is refused by ``validate``.
        """
        known = {f.name for f in fields(cls)}
        data = dict(data or {})
        for key in set(data) - known:
            logger.warning(
                "placement: ignoring unknown key %r; this module knows %s",
                key, ", ".join(sorted(known)),
            )
            data.pop(key)
        return cls(**data)


def load_placement_plan(
    model_name: str,
    config_path: str | Path | None = None,
) -> PlacementPlan:
    """Return the placement plan for ``model_name``.

    The file carries a ``defaults`` mapping and a ``models`` mapping. A named
    model REFINES the defaults: a key it does not mention keeps the default
    value, so the common placement is written once and each model states only
    where it differs. A model with no entry is the defaults exactly.

    A missing or unreadable file is an empty plan, not an error: no placement
    configured means no placement arguments, which is the behaviour that held
    before this file existed.
    """
    import yaml

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.is_file():
        return PlacementPlan()

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("placement: cannot read %s: %s", path, exc)
        return PlacementPlan()

    defaults = raw.get("defaults") or {}
    models = raw.get("models") or {}
    merged = dict(defaults)
    merged.update(models.get(model_name) or {})
    return PlacementPlan.from_dict(merged)
