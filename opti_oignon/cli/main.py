#!/usr/bin/env python3
"""
CLI entry point -- Opti-Oignon.

Click-based command-line interface that talks to a running Opti-Oignon
backend.  Installed as the ``oo`` console script.  Three commands run in
this process instead: ``oo chat``, the interactive session; ``oo core``,
the resident core daemon; and ``oo garden``, the componion, a simulated
onion whose record lives on this machine.

Usage examples::

    oo chat
    oo chat -m llama3 --conversation 3f2a...
    oo ask "Summarise this dataset"
    oo ask -m llama3 "Explain PCA"
    cat data.csv | oo ask --pipe "Analyse this"
    oo ask -f prompt.txt
    oo models
    oo status
    oo backup export backup.json
    oo backup import backup.json --strategy merge
    oo rag ingest paper.pdf --collection ecology
    oo rag query "What is BCI?" --collection ecology
    oo redteam run
    oo core serve
    oo core status
    oo garden
    oo garden sow
    oo garden lab laws
    oo config
    oo config set api_url http://remote:8001
"""

import json
import sys
from pathlib import Path

import click

from .client import CLIClientError, OOClient
from .config import (
    ANIMATION_RANGES,
    VALID_OUTPUT_FORMATS,
    CLIConfig,
    ConfigFileError,
    default_settings,
    load_config,
    parse_animation_ms,
    parse_switch,
    parse_timeout,
    read_file,
    write_file,
)
from .output import (
    Spinner,
    echo_error,
    echo_success,
    format_models_table,
    format_status,
    make_wait_line,
)


def _get_client(ctx: click.Context) -> OOClient:
    """Retrieve the shared OOClient from the Click context."""
    return ctx.obj["client"]


def _get_config(ctx: click.Context) -> CLIConfig:
    """Retrieve the CLIConfig from the Click context."""
    return ctx.obj["config"]


def _waiting(ctx: click.Context, message: str, call):
    """Answer ``call()`` from behind a spinner.

    A client error stops the spinner and erases its line before the error
    is written, so the message never lands inside the spinner's line.
    """
    try:
        with Spinner(message, enabled=_get_config(ctx).color):
            return call()
    except CLIClientError as exc:
        echo_error(str(exc))
        ctx.exit(1)


# =========================================================================
# Root group
# =========================================================================

@click.group()
@click.option("--api-url", envvar="OO_API_URL", default=None,
              help="Override the backend API URL.")
@click.option("--no-color", is_flag=True, default=False,
              help="Disable colour output.")
@click.version_option(package_name="opti-oignon")
@click.pass_context
def cli(ctx: click.Context, api_url: str | None, no_color: bool) -> None:
    """oo -- Opti-Oignon CLI companion.

    Talk to a running Opti-Oignon backend from your terminal.
    """
    cfg = load_config()
    if api_url:
        cfg.api_url = api_url.rstrip("/")
    if no_color:
        cfg.color = False
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["client"] = OOClient(config=cfg)


# =========================================================================
# oo ask
# =========================================================================

@cli.command()
@click.argument("prompt", required=False)
@click.option("-m", "--model", default=None, help="Target a specific model.")
@click.option("-f", "--file", "input_file", type=click.Path(exists=True),
              help="Read prompt from a file.")
@click.option("--pipe", is_flag=True, default=False,
              help="Read prompt from stdin (pipe mode).")
@click.option("--json-out", "json_out", is_flag=True, default=False,
              help="Output full response as JSON.")
@click.pass_context
def ask(ctx: click.Context, prompt: str | None, model: str | None,
        input_file: str | None, pipe: bool, json_out: bool) -> None:
    """Send a prompt to the LLM and stream the response."""
    client = _get_client(ctx)
    cfg = _get_config(ctx)

    # Resolve prompt source
    text = _resolve_prompt(prompt, input_file, pipe)
    if not text:
        echo_error("No prompt provided. Pass a string, use -f <file>, or --pipe.")
        ctx.exit(1)
        return

    effective_model = model or cfg.default_model

    if json_out:
        # Non-streaming: the spinner is on stderr, the JSON alone on stdout
        full = _waiting(ctx, "Generating", lambda: client.stream_chat(text, model=effective_model))
        click.echo(json.dumps({"model": effective_model or "router",
                                "prompt": text, "response": full}, indent=2))
        return

    # Streaming mode: print tokens as they arrive
    metadata_store: dict = {}

    def _on_token(tok: str) -> None:
        click.echo(tok, nl=False)

    def _on_metadata(meta: dict) -> None:
        metadata_store.update(meta)

    try:
        client.stream_chat(
            text,
            model=effective_model,
            on_token=_on_token,
            on_metadata=_on_metadata,
        )
        # Ensure trailing newline
        click.echo()
    except CLIClientError as exc:
        click.echo()  # newline after partial output
        echo_error(str(exc))
        ctx.exit(1)


def _resolve_prompt(prompt: str | None, input_file: str | None, pipe: bool) -> str:
    """Determine the final prompt string from the various input sources."""
    if pipe or (not sys.stdin.isatty() and not prompt and not input_file):
        return sys.stdin.read().strip()
    if input_file:
        return Path(input_file).read_text(encoding="utf-8").strip()
    return (prompt or "").strip()


# =========================================================================
# oo models
# =========================================================================

@cli.command()
@click.pass_context
def models(ctx: click.Context) -> None:
    """List available models with status information."""
    client = _get_client(ctx)
    cfg = _get_config(ctx)
    data = _waiting(ctx, "Fetching models", lambda: client.get("/api/models"))
    model_list = data.get("models", [])
    if not model_list:
        click.echo("No models found.")
        return
    output = format_models_table(model_list, color=cfg.color)
    click.echo(output)


# =========================================================================
# oo status
# =========================================================================

@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system health and backend status."""
    client = _get_client(ctx)
    cfg = _get_config(ctx)
    data = _waiting(ctx, "Checking status", lambda: client.get("/api/health/dashboard"))
    output = format_status(data, color=cfg.color)
    click.echo(output)


# =========================================================================
# oo chat
# =========================================================================

def _wait_kind(line: str) -> str:
    """Which wait a line starts: /close and /open have their own frames, anything else waits."""
    text = line.strip()
    if text.startswith("/"):
        name = text[1:].partition(" ")[0]
        if name == "close":
            return "closing"
        if name == "open":
            return "opening"
    return "waiting"


def _default_chat_session(model: str | None, conversation_id: str | None):
    """The in-process session: the registry configured from backends.yaml, then the session over it."""
    from opti_oignon.inference_backend import init_backends_from_config

    from .session import ChatSession

    init_backends_from_config()
    return ChatSession(model=model, conversation_id=conversation_id)


@cli.command()
@click.option("-m", "--model", default=None, help="Force a model instead of routing.")
@click.option("--conversation", "conversation_id", default=None,
              help="Continue an existing conversation by id.")
@click.pass_context
def chat(ctx: click.Context, model: str | None, conversation_id: str | None) -> None:
    """Chat in this process, one line per turn; /help lists the commands.

    Unlike ``oo ask`` this does not talk to the API server: the executor,
    the conversation store, the onion memory and the skill registry run
    here, and inference goes through the registry (the core daemon when
    core.yaml enables it). Every refusal is printed to stderr by name.
    """
    cfg = _get_config(ctx)
    factory = (ctx.obj or {}).get("chat_session") or _default_chat_session
    try:
        session = factory(model, conversation_id)
    except Exception as exc:  # noqa: BLE001 - a session that cannot start is said
        echo_error(f"the chat session could not start: {exc}", color=cfg.color)
        sys.exit(2)
    stdin = click.get_text_stream("stdin")
    interactive = stdin.isatty()
    # The wait animation: stderr only, erased before any output, None when
    # any switch says no. The seams are the contracts' way in.
    waiter = make_wait_line(cfg, **((ctx.obj or {}).get("wait_seams") or {}))
    click.echo("oo chat -- /help lists the commands, /quit ends the session", err=True)
    try:
        while True:
            if interactive:
                click.echo("> ", nl=False, err=True)
            line = stdin.readline()
            if not line:
                break
            streamed = ended = answered = False
            # A line cut off by the end of input arms nothing.
            armed = waiter is not None and line.endswith("\n") and bool(line.strip())
            kind = _wait_kind(line)
            origin = waiter.now() if armed else 0.0
            if armed:
                waiter.arm(kind, origin)
            try:
                for event in session.handle(line):
                    if armed:
                        # An empty token is the executor's keepalive: the
                        # request is with the model. Anything else is about to
                        # be shown, so the line is erased first.
                        if event.kind == "token" and event.text == "":
                            waiter.request()
                        else:
                            waiter.stop()
                    if event.kind == "token":
                        click.echo(event.text, nl=False)
                        streamed = True
                        answered = answered or bool(event.text)
                    elif event.kind == "thinking":
                        click.echo(event.text, nl=False, err=True)
                        answered = answered or bool(event.text)
                    elif event.kind == "refusal":
                        if streamed:
                            click.echo()
                            streamed = False
                        echo_error(event.text, color=cfg.color)
                    elif event.kind == "quit":
                        ended = True
                        if waiter is not None:
                            waiter.bye()
                    else:
                        click.echo(event.text)
                    # A status line before the answer: the wait goes on,
                    # counted from the same Enter.
                    if armed and not answered and event.kind not in ("token", "thinking", "quit"):
                        waiter.arm(kind, origin)
            finally:
                if waiter is not None:
                    waiter.stop()
            if streamed:
                click.echo()
            if ended:
                break
    except KeyboardInterrupt:
        if waiter is not None:
            waiter.stop()
        click.echo("", err=True)
        echo_error("interrupted", color=cfg.color)
        sys.exit(130)


# =========================================================================
# oo core (group)
# =========================================================================

@cli.group()
def core() -> None:
    """The resident core daemon: serve it, or ask whether it answers."""


@core.command("serve")
@click.option("--config", "config_path", default=None, help="Path to core.yaml (default: the package's)")
def core_serve(config_path: str | None) -> None:
    """Run the core daemon in the foreground on the loopback."""
    from opti_oignon.core_daemon import CoreError, serve

    try:
        serve(config_path)
    except CoreError as exc:
        echo_error(str(exc))
        sys.exit(2)


@core.command("status")
@click.option("--config", "config_path", default=None, help="Path to core.yaml (default: the package's)")
def core_status(config_path: str | None) -> None:
    """Ask the configured daemon for its health."""
    from opti_oignon.core_client import RemoteCoreBackend
    from opti_oignon.core_daemon import load_config

    config = load_config(config_path)
    remote = RemoteCoreBackend(config.base_url, token=config.token, timeout_s=min(config.timeout_s, 5.0))
    if remote.health_check():
        echo_success(f"core daemon answers at {config.base_url}")
    else:
        echo_error(f"no core daemon at {config.base_url} (enabled: {config.enabled})")
        sys.exit(1)


# =========================================================================
# oo garden (group): the componion, in this process
# =========================================================================
#
# Declarations only. Every body imports its runner from ``.garden`` when it
# runs, so importing the CLI imports nothing of the garden. Text is never
# read from the command line: every parameter type is closed and quiet (a
# refusal names the shape it expects, never the value given), and extras
# and unknown options reach the body, which refuses them without repeating
# them. A name and every answer are read from stdin.

_GARDEN = {"ignore_unknown_options": True, "allow_extra_args": True}
_COUNT_MAX = (1 << 53) - 1
_DIGITS = "0123456789"
_HEX_DIGITS = "0123456789abcdefABCDEF"
_BASE32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"


class _QuietGroup(click.Group):
    """A group whose unknown subtopic is refused without its words being repeated."""

    def resolve_command(self, ctx: click.Context, args: list[str]):
        name = args[0] if args else None
        command = self.get_command(ctx, name) if isinstance(name, str) else None
        if command is None:
            from .garden import refuse_subtopic

            refuse_subtopic(ctx)
        return name, command, args[1:]


class _QuietChoice(click.ParamType):
    """One word of a closed set, case-insensitive, lowered."""

    name = "word"

    def __init__(self, words: tuple[str, ...]) -> None:
        self.words = tuple(words)

    def get_metavar(self, param: click.Parameter) -> str:
        return "[" + "|".join(self.words) + "]"

    def convert(self, value, param, ctx):
        if isinstance(value, str) and value.isascii() and value.lower() in self.words:
            return value.lower()
        self.fail("expected one of: " + ", ".join(self.words), param, ctx)


class _Count(click.ParamType):
    """A whole number of 1 to 16 digits, at most 2^53-1."""

    name = "count"

    def convert(self, value, param, ctx):
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _COUNT_MAX:
            return value
        if isinstance(value, str) and 1 <= len(value) <= 16 and all(char in _DIGITS for char in value):
            number = int(value)
            if number <= _COUNT_MAX:
                return number
        self.fail("expected a whole number of 1 to 16 digits, at most 2^53-1", param, ctx)


class _Hex(click.ParamType):
    """Exactly ``length`` hexadecimal digits, case-insensitive, lowered."""

    name = "hex"

    def __init__(self, length: int) -> None:
        self.length = length

    def convert(self, value, param, ctx):
        if isinstance(value, str) and len(value) == self.length and all(char in _HEX_DIGITS for char in value):
            return value.lower()
        self.fail(f"expected {self.length} hexadecimal digits", param, ctx)


class _ConsentCode(click.ParamType):
    """Four RFC 4648 base32 symbols, a dash, four more; case-insensitive, raised."""

    name = "code"

    def convert(self, value, param, ctx):
        if isinstance(value, str) and value.isascii() and len(value) == 9 and value[4] == "-":
            code = value.upper()
            if all(char in _BASE32 for char in code[:4] + code[5:]):
                return code
        self.fail("expected four of A-Z or 2-7, a dash, and four more", param, ctx)


@cli.group(cls=_QuietGroup, invoke_without_command=True, context_settings=_GARDEN)
@click.pass_context
def garden(ctx: click.Context) -> None:
    """Your componion, a simulated onion, in this process. Without a subtopic: show."""
    if ctx.invoked_subcommand is None:
        from .garden import run_show

        run_show(ctx, tier="ascii", as_json=False)


@garden.command("show", context_settings=_GARDEN)
@click.option("--tier", type=_QuietChoice(("text", "ascii")), default="ascii", show_default=True,
              help="text: the lines alone; ascii: the lines and a drawing.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="One line of JSON: the lines, and closed codes for the rest.")
@click.pass_context
def garden_show(ctx: click.Context, tier: str, as_json: bool) -> None:
    """Show the onion as the simulation computes it now. Writes nothing in its record."""
    from .garden import run_show

    run_show(ctx, tier=tier, as_json=as_json)


@garden.command("sow", context_settings=_GARDEN)
@click.option("--hemisphere", type=_QuietChoice(("north", "south")), default=None,
              help="The hemisphere of its seasons (else allium.yaml, else north).")
@click.option("--band", type=_QuietChoice(("long", "medium", "short")), default=None,
              help="How much its daylight changes with the seasons (else allium.yaml, else long).")
@click.option("--weather", type=_QuietChoice(("garden", "windowsill")), default=None,
              help="Rain, or only the water you give (else allium.yaml, else garden).")
@click.pass_context
def garden_sow(ctx: click.Context, hemisphere: str | None, band: str | None, weather: str | None) -> None:
    """Sow the one seed of this garden, from an interactive terminal.

    The card is printed first; the name and the confirmation are then read
    from stdin, one line each.
    """
    from .garden import run_sow

    run_sow(ctx, hemisphere=hemisphere, band=band, weather=weather)


@garden.group("care", cls=_QuietGroup, context_settings=_GARDEN)
def garden_care() -> None:
    """A gesture: greet, water, warm or play. Each is noted in its record."""


@garden_care.command("greet", context_settings=_GARDEN)
@click.pass_context
def garden_care_greet(ctx: click.Context) -> None:
    """Greet the onion."""
    from .garden import run_care

    run_care(ctx, "greet")


@garden_care.command("water", context_settings=_GARDEN)
@click.pass_context
def garden_care_water(ctx: click.Context) -> None:
    """Water the onion."""
    from .garden import run_care

    run_care(ctx, "water")


@garden_care.command("warm", context_settings=_GARDEN)
@click.pass_context
def garden_care_warm(ctx: click.Context) -> None:
    """Warm the onion."""
    from .garden import run_care

    run_care(ctx, "warm")


@garden_care.command("play", context_settings=_GARDEN)
@click.pass_context
def garden_care_play(ctx: click.Context) -> None:
    """Play with the onion."""
    from .garden import run_care

    run_care(ctx, "play")


@garden.group("lab", cls=_QuietGroup, invoke_without_command=True, context_settings=_GARDEN)
@click.pass_context
def garden_lab(ctx: click.Context) -> None:
    """The laboratory: what its record holds. Subtopic: laws. Writes nothing in its record."""
    if ctx.invoked_subcommand is None:
        from .garden import run_lab

        run_lab(ctx)


@garden_lab.command("laws", context_settings=_GARDEN)
@click.pass_context
def garden_lab_laws(ctx: click.Context) -> None:
    """The laws of its world: in force, pinned, pending, written, and your proposal. Writes nothing in its record."""
    from .garden import run_lab_laws

    run_lab_laws(ctx)


@garden.group("keep", cls=_QuietGroup, context_settings=_GARDEN)
def garden_keep() -> None:
    """Keeping its record: verify, name, laws, resume, finish."""


@garden_keep.command("verify", context_settings=_GARDEN)
@click.pass_context
def garden_keep_verify(ctx: click.Context) -> None:
    """Verify its whole record, and replay every kept state on the reference engine. Writes nothing in its record."""
    from .garden import run_verify

    run_verify(ctx)


@garden_keep.command("name", context_settings=_GARDEN)
@click.pass_context
def garden_keep_name(ctx: click.Context) -> None:
    """Name the onion, from an interactive terminal: one line read from stdin, kept in its record for good."""
    from .garden import run_name

    run_name(ctx)


@garden_keep.group("laws", cls=_QuietGroup, context_settings=_GARDEN)
def garden_keep_laws() -> None:
    """Law updates from your allium.yaml proposal: diff, apply, pin, unpin."""


@garden_keep_laws.command("diff", context_settings=_GARDEN)
@click.pass_context
def garden_keep_laws_diff(ctx: click.Context) -> None:
    """What a law update from your proposal would change, and the code that writes it. Writes nothing in its record."""
    from .garden import run_laws_diff

    run_laws_diff(ctx)


@garden_keep_laws.command("apply", context_settings=_GARDEN)
@click.argument("confirm", type=_Hex(16))
@click.pass_context
def garden_keep_laws_apply(ctx: click.Context, confirm: str) -> None:
    """Write the law update the diff shows, with its code, from an interactive terminal."""
    from .garden import run_laws_apply

    run_laws_apply(ctx, confirm)


@garden_keep_laws.command("pin", context_settings=_GARDEN)
@click.pass_context
def garden_keep_laws_pin(ctx: click.Context) -> None:
    """Keep the laws in force: no law update applies until unpin. From an interactive terminal."""
    from .garden import run_laws_pin

    run_laws_pin(ctx)


@garden_keep_laws.command("unpin", context_settings=_GARDEN)
@click.pass_context
def garden_keep_laws_unpin(ctx: click.Context) -> None:
    """Lift the pin, from an interactive terminal."""
    from .garden import run_laws_unpin

    run_laws_unpin(ctx)


@garden_keep.command("resume", context_settings=_GARDEN)
@click.argument("kept", type=_Count())
@click.argument("discarded", type=_Count())
@click.pass_context
def garden_keep_resume(ctx: click.Context, kept: int, discarded: int) -> None:
    """Resume a record that failed its verification, with the two numbers the garden shows.

    From an interactive terminal. The later events it names are deleted for good.
    """
    from .garden import run_resume

    run_resume(ctx, kept, discarded)


@garden_keep.command("finish", context_settings=_GARDEN)
@click.argument("tag", type=_Hex(8))
@click.pass_context
def garden_keep_finish(ctx: click.Context, tag: str) -> None:
    """Finish an interrupted sowing, with the tag the garden shows, from an interactive terminal."""
    from .garden import run_finish

    run_finish(ctx, tag)


@garden_keep.command("celebrate", hidden=True, context_settings=_GARDEN)
@click.pass_context
def garden_keep_celebrate(ctx: click.Context) -> None:
    """Not in this version."""
    from .garden import run_later_refused

    run_later_refused(ctx)


@garden_keep.command("bury", hidden=True, context_settings=_GARDEN)
@click.pass_context
def garden_keep_bury(ctx: click.Context) -> None:
    """Not in this version."""
    from .garden import run_later_refused

    run_later_refused(ctx)


@garden.group("lang", cls=_QuietGroup, hidden=True, context_settings=_GARDEN)
def garden_lang() -> None:
    """Not in this version."""


@garden_lang.command("talk", hidden=True, context_settings=_GARDEN)
@click.pass_context
def garden_lang_talk(ctx: click.Context) -> None:
    """Not in this version."""
    from .garden import run_later_refused

    run_later_refused(ctx)


@garden_lang.command("teach", hidden=True, context_settings=_GARDEN)
@click.pass_context
def garden_lang_teach(ctx: click.Context) -> None:
    """Not in this version."""
    from .garden import run_later_refused

    run_later_refused(ctx)


@garden.command("tray", hidden=True, context_settings=_GARDEN)
@click.pass_context
def garden_tray(ctx: click.Context) -> None:
    """Not in this version."""
    from .garden import run_later

    run_later(ctx)


@garden.group("share", cls=_QuietGroup, hidden=True, invoke_without_command=True, context_settings=_GARDEN)
@click.pass_context
def garden_share(ctx: click.Context) -> None:
    """Not in this version."""
    if ctx.invoked_subcommand is None:
        from .garden import run_later

        run_later(ctx)


@garden_share.command("confirm", hidden=True, context_settings=_GARDEN)
@click.argument("code", type=_ConsentCode())
@click.pass_context
def garden_share_confirm(ctx: click.Context, code: str) -> None:
    """Not in this version: no consent request can be opened, and none is confirmed."""
    from .garden import run_share_confirm

    run_share_confirm(ctx, code)


# =========================================================================
# oo backup (group)
# =========================================================================

@cli.group()
def backup() -> None:
    """Backup and restore configuration."""
    pass


@backup.command("export")
@click.argument("output_path", required=False, default=None)
@click.pass_context
def backup_export(ctx: click.Context, output_path: str | None) -> None:
    """Export configuration backup to a JSON file."""
    client = _get_client(ctx)
    dest = output_path or "opti-oignon-backup.json"
    data = _waiting(ctx, "Exporting backup", lambda: client.get("/api/backup/export"))
    Path(dest).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    echo_success(f"Backup saved to {dest}")


@backup.command("import")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--strategy", type=click.Choice(["merge", "replace"]), default="merge",
              help="Import strategy: merge (keep existing) or replace (overwrite).")
@click.pass_context
def backup_import(ctx: click.Context, input_path: str, strategy: str) -> None:
    """Import configuration from a backup JSON file."""
    client = _get_client(ctx)
    try:
        raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        echo_error(f"Failed to read backup file: {exc}")
        ctx.exit(1)
        return

    result = _waiting(ctx, "Importing backup", lambda: client.post("/api/backup/import", json_body={
        "backup": raw, "strategy": strategy,
    }))
    changes = result.get("changes_applied", 0) if isinstance(result, dict) else "?"
    echo_success(f"Backup imported ({strategy}). Changes applied: {changes}")


# =========================================================================
# oo rag (group)
# =========================================================================

@cli.group()
def rag() -> None:
    """RAG knowledge base operations."""
    pass


@rag.command("ingest")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("-c", "--collection", default="default", help="Target collection name.")
@click.pass_context
def rag_ingest(ctx: click.Context, filepath: str, collection: str) -> None:
    """Ingest a file into the RAG knowledge base."""
    client = _get_client(ctx)
    result = _waiting(ctx, f"Ingesting {Path(filepath).name}", lambda: client.post_file(
        "/api/rag/ingest",
        filepath=filepath,
        extra_fields={"collection": collection},
    ))
    doc_id = result.get("doc_id", "?") if isinstance(result, dict) else "?"
    chunks = result.get("chunk_count", "?") if isinstance(result, dict) else "?"
    echo_success(f"Ingested {Path(filepath).name} -> doc_id={doc_id}, chunks={chunks}")


@rag.command("query")
@click.argument("question")
@click.option("-c", "--collection", default="default", help="Collection to query.")
@click.option("-n", "--results", "n_results", default=5, type=int, help="Number of results.")
@click.pass_context
def rag_query(ctx: click.Context, question: str, collection: str, n_results: int) -> None:
    """Query the RAG knowledge base."""
    client = _get_client(ctx)
    data = _waiting(ctx, "Querying knowledge base", lambda: client.post("/api/rag/query", json_body={
        "query": question, "collection": collection,
        "n_results": n_results,
    }))
    results = data.get("results", []) if isinstance(data, dict) else []
    if not results:
        click.echo("No results found.")
        return
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        source = r.get("source_file", "?")
        content = r.get("content", "")[:200]
        click.echo(f"\n--- Result {i} (score: {score:.3f}, source: {source}) ---")
        click.echo(content)
        if len(r.get("content", "")) > 200:
            click.echo("...")



# =========================================================================
# oo redteam (group)
# =========================================================================

@cli.group()
def redteam() -> None:
    """Red team audit operations."""
    pass


@redteam.command("run")
@click.option("--quick", is_flag=True, default=False,
              help="Quick mode: fewer attacks per category.")
@click.option("-c", "--category", "categories", multiple=True,
              help="Restrict to specific categories (repeatable).")
@click.option("-t", "--target", "targets", multiple=True,
              help="Restrict to specific targets (repeatable).")
@click.pass_context
def redteam_run(ctx: click.Context, quick: bool,
                categories: tuple, targets: tuple) -> None:
    """Launch a red team audit campaign."""
    client = _get_client(ctx)
    cfg = _get_config(ctx)

    body: dict = {}
    if quick:
        body["attacks_per_category"] = 2
    if categories:
        body["categories"] = list(categories)
    if targets:
        body["targets"] = list(targets)

    result = _waiting(ctx, "Launching campaign", lambda: client.post("/api/security/redteam/run", json_body=body))

    echo_success("Campaign started.")
    click.echo(f"  Categories: {result.get('config', {}).get('categories', '?')}")
    click.echo(f"  Targets:    {result.get('config', {}).get('targets', '?')}")
    click.echo(f"  Attacks/cat: {result.get('config', {}).get('attacks_per_category', '?')}")
    click.echo("\nPoll progress with: oo redteam status")

    # If quick mode, poll until complete
    if quick:
        import time as _time
        click.echo()
        with Spinner("Waiting for results", enabled=cfg.color):
            for _ in range(300):  # max ~5 min
                _time.sleep(1)
                try:
                    st = client.get("/api/security/redteam/status")
                except CLIClientError:
                    break
                if not st.get("running", False):
                    break

        try:
            results = client.get("/api/security/redteam/results")
            _print_redteam_summary(results, color=cfg.color)
        except CLIClientError as exc:
            echo_error(str(exc))
            ctx.exit(1)


@redteam.command("status")
@click.pass_context
def redteam_status_cmd(ctx: click.Context) -> None:
    """Check the progress of a running red team campaign."""
    client = _get_client(ctx)
    try:
        st = client.get("/api/security/redteam/status")
    except CLIClientError as exc:
        echo_error(str(exc))
        ctx.exit(1)
        return

    running = st.get("running", False)
    progress = st.get("progress")
    error = st.get("error")

    if running and progress:
        pct = progress.get("percent", 0)
        done = progress.get("completed_steps", 0)
        total = progress.get("total_steps", 0)
        cat = progress.get("current_category", "")
        click.echo(f"Running: {pct:.1f}% ({done}/{total})")
        click.echo(f"  Current category: {cat}")
        click.echo(f"  Errors so far:    {progress.get('errors', 0)}")
    elif running:
        click.echo("Campaign is running (no progress data yet).")
    elif error:
        echo_error(f"Last campaign failed: {error}")
    elif st.get("has_results"):
        echo_success("Campaign complete. Use 'oo redteam report' to view results.")
    else:
        click.echo("No campaign running or completed.")


@redteam.command("report")
@click.option("--format", "fmt", type=click.Choice(["json", "md"]),
              default="md", help="Report format.")
@click.option("--last", "use_last", is_flag=True, default=True,
              help="Use latest results (default).")
@click.option("--id", "report_id", default=None,
              help="Specific report ID.")
@click.pass_context
def redteam_report_cmd(ctx: click.Context, fmt: str, use_last: bool,
                       report_id: str | None) -> None:
    """Display or download a red team report."""
    client = _get_client(ctx)
    cfg = _get_config(ctx)

    if report_id:
        # Fetch specific report from stored reports
        try:
            result = client.get(f"/api/security/redteam/reports/{report_id}")
            if fmt == "json":
                click.echo(json.dumps(result, indent=2))
            else:
                _print_redteam_summary(result, color=cfg.color)
        except CLIClientError as exc:
            echo_error(str(exc))
            ctx.exit(1)
        return

    # Use latest results
    try:
        results = client.get("/api/security/redteam/results")
    except CLIClientError as exc:
        echo_error(str(exc))
        ctx.exit(1)
        return

    if fmt == "json":
        click.echo(json.dumps(results, indent=2))
    else:
        _print_redteam_summary(results, color=cfg.color)


@redteam.command("compare")
@click.argument("id1")
@click.argument("id2")
@click.pass_context
def redteam_compare(ctx: click.Context, id1: str, id2: str) -> None:
    """Compare two red team reports and highlight regressions."""
    client = _get_client(ctx)
    cfg = _get_config(ctx)

    try:
        result = client.get("/api/security/redteam/compare", id1=id1, id2=id2)
    except CLIClientError as exc:
        echo_error(str(exc))
        ctx.exit(1)
        return

    _print_redteam_comparison(result, color=cfg.color)


def _print_redteam_summary(results: dict, *, color: bool = True) -> None:
    """Print a human-readable red team summary to the terminal."""
    score = results.get("score", {})
    campaign = results.get("campaign", {})

    total = score.get("total", 0)
    bypasses = score.get("total_bypasses", 0)
    flags = score.get("total_flags", 0)
    blocks = score.get("total_blocks", 0)
    bypass_rate = score.get("overall_bypass_rate", 0)

    click.echo()
    click.echo("Red Team Audit Summary")
    click.echo("-" * 40)
    click.echo(f"  Total attacks:   {total}")
    click.echo(f"  Blocks:          {blocks}")
    click.echo(f"  Flags:           {flags}")
    click.echo(f"  Bypasses:        {bypasses}")
    click.echo(f"  Bypass rate:     {bypass_rate:.1%}")

    if campaign.get("duration_seconds"):
        click.echo(f"  Duration:        {campaign['duration_seconds']:.1f}s")
    if campaign.get("errors_count"):
        click.echo(f"  Errors:          {campaign['errors_count']}")

    # Per-category breakdown
    by_cat = score.get("by_category", {})
    if by_cat:
        click.echo()
        click.echo("By Category:")
        for name, bd in sorted(by_cat.items()):
            br = bd.get("bypass_rate", 0)
            marker = "  CRITICAL" if br > 0.5 else ""
            click.echo(
                f"  {name:25s}  "
                f"total={bd.get('total', 0):3d}  "
                f"bypass={br:.1%}"
                f"{marker}"
            )

    # Per-target breakdown
    by_tgt = score.get("by_target", {})
    if by_tgt:
        click.echo()
        click.echo("By Target:")
        for name, bd in sorted(by_tgt.items()):
            br = bd.get("bypass_rate", 0)
            click.echo(
                f"  {name:25s}  "
                f"total={bd.get('total', 0):3d}  "
                f"bypass={br:.1%}  "
                f"block={bd.get('block_rate', 0):.1%}"
            )

    # Highlight critical findings
    critical = [
        (name, bd) for name, bd in by_cat.items()
        if bd.get("bypass_rate", 0) > 0.3
    ]
    if critical:
        click.echo()
        click.echo("Critical findings (bypass rate > 30%):")
        for name, bd in critical:
            click.echo(f"  - {name}: {bd.get('bypass_rate', 0):.1%} bypass rate")

    click.echo()


def _print_redteam_comparison(result: dict, *, color: bool = True) -> None:
    """Print a red team report comparison to the terminal."""
    click.echo()
    click.echo("Red Team Comparison")
    click.echo("-" * 40)

    summary = result.get("summary", {})
    click.echo(
        f"  Bypass rate: {summary.get('bypass_rate_before', 0):.1%}"
        f" -> {summary.get('bypass_rate_after', 0):.1%}"
    )

    regressions = result.get("regressions", [])
    improvements = result.get("improvements", [])

    if regressions:
        click.echo()
        click.echo("Regressions:")
        for r in regressions:
            click.echo(
                f"  - {r.get('category', '?')}: "
                f"{r.get('bypass_rate_before', 0):.1%} -> "
                f"{r.get('bypass_rate_after', 0):.1%}"
            )

    if improvements:
        click.echo()
        click.echo("Improvements:")
        for imp in improvements:
            click.echo(
                f"  - {imp.get('category', '?')}: "
                f"{imp.get('bypass_rate_before', 0):.1%} -> "
                f"{imp.get('bypass_rate_after', 0):.1%}"
            )

    if not regressions and not improvements:
        click.echo("  No significant changes detected.")

    click.echo()


# =========================================================================
# oo config
# =========================================================================

@cli.group(name="config", invoke_without_command=True)
@click.pass_context
def config_cmd(ctx: click.Context) -> None:
    """Show or edit CLI configuration."""
    if ctx.invoked_subcommand is None:
        cfg = _get_config(ctx)
        click.echo("Current CLI configuration:")
        for key, val in cfg.to_dict().items():
            click.echo(f"  {key}: {val}")
        from .config import CONFIG_FILE
        click.echo(f"Config path: {CONFIG_FILE}")


@config_cmd.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx: click.Context, key: str, value: str) -> None:
    """Set a configuration value.

    Examples:
        oo config set api_url http://remote:8001
        oo config set default_model llama3
        oo config set output_format json
        oo config set animations false
        oo config set animation_interval_ms 200
    """
    allowed = {"api_url", "default_model", "output_format", "color", "timeout",
               "animations", *ANIMATION_RANGES}
    if key not in allowed:
        echo_error(f"Unknown config key '{key}'. Valid keys: {', '.join(sorted(allowed))}")
        ctx.exit(1)
        return
    if key in ("animations", "color"):
        parsed = parse_switch(value)
        problem = f"{key} must be true or false (also yes/no, on/off, 1/0)"
    elif key in ANIMATION_RANGES:
        parsed = parse_animation_ms(key, value)
        low, high, _ = ANIMATION_RANGES[key]
        problem = f"{key} must be a whole number of milliseconds between {low} and {high}"
    elif key == "timeout":
        parsed = parse_timeout(value)
        problem = "timeout must be a positive whole number of seconds"
    elif key == "output_format":
        parsed = value if value in VALID_OUTPUT_FORMATS else None
        problem = f"output_format must be one of {', '.join(VALID_OUTPUT_FORMATS)}"
    else:
        parsed, problem = value, ""
    if parsed is None:
        echo_error(problem)
        ctx.exit(1)
        return
    # The file, not the run: an override of this run (NO_COLOR, --no-color,
    # --api-url, OO_API_URL) is never written.
    try:
        settings = read_file()
    except ConfigFileError as exc:
        echo_error(f"{exc}; nothing was written (oo config reset starts it over)")
        ctx.exit(1)
        return
    settings[key] = parsed
    saved = write_file(settings)
    echo_success(f"{key} = {parsed} (saved to {saved})")


@config_cmd.command("reset")
@click.pass_context
def config_reset(ctx: click.Context) -> None:
    """Reset CLI configuration to defaults."""
    saved = write_file(default_settings())
    echo_success(f"Configuration reset to defaults (saved to {saved})")


# =========================================================================
# Entry point
# =========================================================================

def main() -> None:
    """Entry point for the ``oo`` console script."""
    cli()


if __name__ == "__main__":
    main()
