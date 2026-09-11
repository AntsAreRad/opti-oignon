#!/usr/bin/env python3
"""Print the llama-server command for a model's placement plan.

The repository never launches llama-server. This prints the recipe so the
maintainer can run it on the host, and run the identical one again later: the
plan resolves to the same argv every time, which is what makes two runs
comparable.

    python3 scripts/placement_recipe.py MODEL_PATH [--model-name NAME]
                                        [--config PATH] [--port N]
                                        [--host H] [--ctx N]

MODEL_PATH is the GGUF handed to -m. --model-name selects the entry in
placement.yaml and defaults to the file's stem, so the usual case needs no
second name.

Exit codes: 0 printed a command, 2 refused the plan and said why. A refusal is
the point rather than a nuisance -- an unrecognised cache type caught here is
one that does not become a server started with a precision nobody chose.

Nothing here is measured. The command's speed is a question for the host.
"""

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print the llama-server command for a placement plan.",
    )
    parser.add_argument("model_path", help="GGUF file passed to -m")
    parser.add_argument(
        "--model-name", default=None,
        help="entry to read from placement.yaml (default: the file's stem)",
    )
    parser.add_argument(
        "--config", default=None, help="placement.yaml to read",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--ctx", type=int, default=None, help="context size passed to -c",
    )
    args = parser.parse_args(argv)

    from opti_oignon.placement import load_placement_plan
    from opti_oignon.speculative_decoding import (
        SpeculativeConfig,
        build_llama_server_command,
    )

    name = args.model_name or Path(args.model_path).stem
    plan = load_placement_plan(name, config_path=args.config)

    # Placement only. The draft posture -- which target/draft pair, and the
    # flag quad that goes with it -- is a separate question with its own
    # configuration, and wiring it in here would mean this command quietly
    # changed when that file changed.
    speculative = SpeculativeConfig(enabled=False)

    try:
        cmd = build_llama_server_command(
            args.model_path,
            speculative,
            host=args.host,
            port=args.port,
            n_ctx=args.ctx,
            placement=plan,
        )
    except ValueError as exc:
        print(f"placement refused: {exc}", file=sys.stderr)
        return 2

    declared = plan.to_dict()
    if declared:
        print(f"# placement for {name}: "
              + ", ".join(f"{k}={v!r}" for k, v in sorted(declared.items())),
              file=sys.stderr)
    else:
        print(f"# no placement declared for {name}; "
              "the command below places nothing", file=sys.stderr)

    print(" ".join(cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
