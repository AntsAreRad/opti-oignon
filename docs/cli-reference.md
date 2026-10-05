# CLI Reference

## Overview

The `oo` command-line tool is a companion for interacting with a
running Opti-Oignon backend from the terminal. Most commands communicate
via the HTTP API and never import heavy backend dependencies. Three run
in the calling process instead: `oo chat`, the interactive session;
`oo core`, the resident core daemon; and `oo garden`, the componion.

Install: `pip install -e ".[all]"` registers the `oo` console script.


## Global options

| Option | Env var | Description |
|--------|---------|-------------|
| `--api-url URL` | `OO_API_URL` | Override the backend API URL |
| `--no-color` | | Disable colored output |
| `--version` | | Show version and exit |


## Commands

### oo ask

Send a prompt to the backend and display the response.

```bash
oo ask "Summarize this dataset"
oo ask -m llama3 "Explain PCA"
cat data.csv | oo ask --pipe "Analyze this"
oo ask -f prompt.txt
```

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | Force a specific model |
| `-f, --file PATH` | Read prompt from file |
| `--pipe` | Read stdin as additional context |

### oo chat

An interactive session, one line per turn, run in this process: the
executor, the conversation store, the onion memory and the skill registry
are the local ones, and inference goes through the inference registry
configured from `backends.yaml` -- the core daemon when `core.yaml`
enables it, never a client of the session's own. It does not need the API
server.

```bash
oo chat
oo chat -m llama3
oo chat --conversation <conversation-id>
```

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | Force a specific model instead of routing |
| `--conversation ID` | Continue an existing conversation |

A line that does not start with `/` is a turn. A line that does is a
command, and every command is a user action:

| Command | Description |
|---------|-------------|
| `/open ID` | Find a persisted onion again and continue that conversation |
| `/close` | Evict the whole Flesh through the gate, save, and end the conversation |
| `/pin TEXT` | Pin a statement to the conversation's Core, as the user |
| `/recall KEY` | Show the verbatim span behind a receipt; the receipt stays open |
| `/resolve KEY` | Close a receipt, as the user: it leaves the digest and stays in the ledger |
| `/skill NAME ARGS` | Run `ARGS` as a turn with a published skill as the system suffix |
| `/adopt NAME [DIGEST]` | Show a skill received from a paired device with its digest; with the digest, adopt exactly those bytes |
| `/help` | List the commands |
| `/quit` | End the session |

The answer streams to stdout; every refusal goes to stderr with its
reason. The onion commands are refused while `onion.yaml` has the onion
switched off, and `/open` is refused when no persistence path is set or
the conversation was never saved. `/close` never overrides the gate: a
span it refuses stays verbatim in the Flesh and the refusal names the
probes that failed. `/skill` takes `name` or `category/name`; a draft, an
unknown name, or a name published in two categories is refused, and only
a published skill -- human-approved, and named by the user on purpose --
reaches the system prompt.

A skill received from a paired device is published once this device's
sync gate lets the record through, and that gate shows its provenance,
not its text. So `/skill` refuses the bytes of a received skill until they
are adopted here: `/adopt NAME` prints the text exactly as it is on disk
with its digest, and `/adopt NAME DIGEST` (twelve hex characters or more)
adopts those bytes and no others. A new version received later is refused
again until it is adopted; writing the whole skill on this device adopts
what it writes, while an edit of bytes never adopted leaves them
unadopted.

While you wait, one line of ASCII art on stderr shows it: an onion that
breathes from Enter until the first answer, sprouts once the executor
reports the request is with the model, sheds its peels during `/close` and
regrows during `/open`. It appears only after `animation_delay_ms`, is
erased before anything else is printed, never touches stdout -- so
`oo chat < in > out` gives the same `out` with or without it -- and writes
no escape sequence, so an interrupted session leaves the terminal as it
was. It is off when stderr is not a terminal, with `NO_COLOR` or
`--no-color`, with `TERM=dumb`, on a terminal of 32 columns or fewer, and
with `animations: false`. `/quit` answers with a one-line goodbye.

### oo models

List all models available in the connected Ollama instance.

```bash
oo models
```

Displays model name, size, quantization, and parameter count in a
formatted table.

### oo status

Show backend health and configuration summary.

```bash
oo status
```

Displays version, uptime, active model, security mode, and feature
availability flags.

### oo backup

Export and import conversation history and configuration.

```bash
oo backup export backup.json
oo backup export                    # auto-named with timestamp
oo backup import backup.json
oo backup import backup.json --strategy merge
```

| Subcommand | Description |
|------------|-------------|
| `export [OUTPUT]` | Export all data to JSON |
| `import INPUT` | Import data from JSON |

Import options:

| Option | Description |
|--------|-------------|
| `--strategy {replace,merge}` | How to handle conflicts (default: replace) |

### oo rag

Manage RAG collections from the terminal.

```bash
oo rag ingest paper.pdf --collection ecology
oo rag query "What is BCI?" --collection ecology
oo rag query "species diversity" --collection ecology --n-results 10
```

| Subcommand | Description |
|------------|-------------|
| `ingest FILE --collection NAME` | Ingest a file into a collection |
| `query QUESTION --collection NAME` | Query a collection |

Query options:

| Option | Description |
|--------|-------------|
| `--n-results N` | Number of results to return (default: 5) |

### oo redteam

Run and manage red team security audits.

```bash
oo redteam run
oo redteam run --quick
oo redteam run --categories injection,jailbreak --targets rag_sanitizer
oo redteam status
oo redteam report
oo redteam report --format json
oo redteam report --id <report-id>
oo redteam compare <id1> <id2>
```

| Subcommand | Description |
|------------|-------------|
| `run` | Launch a red team audit campaign |
| `status` | Show current campaign progress |
| `report` | Display the latest or a specific report |
| `compare ID1 ID2` | Compare two reports side by side |

Run options:

| Option | Description |
|--------|-------------|
| `--quick` | Reduced attack count for faster results |
| `--categories LIST` | Comma-separated attack categories |
| `--targets LIST` | Comma-separated target components |

Report options:

| Option | Description |
|--------|-------------|
| `--format {text,json}` | Output format (default: text) |
| `--id ID` | Specific report ID |
| `--last` | Use the most recent report |

### oo core

Serve the resident core daemon, or ask whether it answers. Both read
`core.yaml`; see the architecture page on the core and the packs.

```bash
oo core serve                      # run the daemon in the foreground
oo core status                     # exit 0 when it answers, 1 otherwise
oo core serve --config path/to/core.yaml
```

| Subcommand | Description |
|------------|-------------|
| `serve` | Run the core daemon on the loopback |
| `status` | Ask the configured daemon for its health |

### oo garden

The componion: a simulated onion whose record lives on this machine,
looked after in this process through the componion's service (see the
architecture page on the onion memory). It needs no API server. The
garden is off until `opti_oignon/config/allium.yaml` says
`enabled: true`; any other value, and a file that cannot be read, is off.
Switched off, every look and every write says so and names the file,
and nothing opens a store or reads a key, a clock or the security mode.

```bash
oo garden                          # the same as oo garden show
oo garden show --tier text         # the lines alone, without the drawing
oo garden show --json              # one line of JSON
oo garden sow --weather windowsill # the card, then a name and yes on stdin
oo garden care water
oo garden lab laws
oo garden keep verify
```

| Subtopic | Description |
|----------|-------------|
| `show [--tier text\|ascii] [--json]` | The onion as the simulation computes it now; writes nothing in its record |
| `sow [--hemisphere north\|south] [--band long\|medium\|short] [--weather garden\|windowsill]` | Sow the one seed of this garden |
| `care greet\|water\|warm\|play` | A gesture, noted in its record |
| `lab` | The doctrine, the record's event count and its labels; writes nothing in its record |
| `lab laws` | The laws of its world: born under, in force, pinned, pending, written, and your proposal; writes nothing in its record |
| `keep verify` | Verify the whole record and replay every kept state on the reference engine; writes nothing in its record |
| `keep name` | Name the onion; the name is read from stdin |
| `keep laws diff` | What a law update from your `allium.yaml` proposal would change, and its code |
| `keep laws apply CODE` | Write that law update, with the 16 hex digits the diff shows |
| `keep laws pin` / `unpin` | Keep the laws in force, or lift the pin |
| `keep resume KEPT DISCARDED` | Resume a record that failed its verification from its last verified event |
| `keep finish TAG` | Link an interrupted sowing into place |

A seed is sown only in an encrypted store (a readable master key and
SQLCipher). When no key is configured and `persistence.require_encryption`
is false, it can live in a glass jar instead, in Daily mode only; a jar is
labelled on every form. There is one seed per account, sown under the
newest law the engine carries for sowing; every law in this version is a
prototype, and every form says so. Before its question, `sow` prints a
card that says what the seed is, where its hemisphere, daylight band and
weather come from (the option, `allium.yaml`, or the default), that
learning your rhythm is not offered, and that there is no way yet to
compost it or put it to rest.

Text is never read from the command line. Every parameter is a closed
choice, a count or a fixed number of hex digits, and a refusal names the
shape it expects, never the value given; words left over, unknown
options and unknown subtopics are refused without being repeated. The
name and the confirmation of `sow`, and the name of `keep name`, are read
from stdin, one line each of at most 255 bytes of UTF-8, and every
question is printed on stdout before its line is read. A name is 1 to 32
letters, digits, spaces, hyphens, apostrophes or periods, starts with a
letter or a digit and not with the word Beetle, and stays in the record
for good. `sow`, `keep name`, `keep laws apply`, `pin`, `unpin`,
`keep resume`, `keep finish` and the hidden `share confirm` run only from
an interactive terminal in the foreground.

Forms go to stdout. A refusal, and the form of a store that cannot be
opened or an onion that cannot be computed now, go to stderr: its first
line after `Error:` says what was refused, and the labels of the onion
concerned follow it. A refusal that comes after a write says that the
write was done. `show --json` always prints on stdout, with the exit of
the form it projects. The exit is 0 when it was done or the state was
said, 1 when it could not be done, and 2 for a usage or set-up error
(text on the command line, a malformed answer or name, or no soil yet).
The garden's own lines are printable ASCII, wrapped at 78 columns (the
drawing, 32 columns by 9 rows, and a file's path are never broken); its
only colour is the `Error:` prefix, which follows `--no-color`,
`NO_COLOR` and `color: false`. Click prints its own usage lines for a
command line it cannot parse. `show --json` prints one line: the
text-tier lines with their keys, then the status, labels, habitat and
law, the onion's day, name, place, season, light, soil, life and stage,
the local time shown, and `"source": "simulation"`; nothing else of its
state is served. No log record of the platform reaches the terminal
while a garden command runs.

In Bulbe mode, or when the mode cannot be read, its life goes on and a
label says so; a glass jar stays sealed and nothing of it is shown.
`keep verify` names a kept state that disagrees with the replay by its day
and law version and replaces nothing. An engine that stops on a fault
shows the onion as of its last kept state, labelled, with exit 1.

`python3 scripts/allium_garden_gallery.py [--tier text|ascii]` prints
sample forms from fixed values, without a store, to see how a terminal
draws them. Hidden subtopics of later versions (`lang`, `tray`, `share`,
`keep celebrate`, `keep bury`) say that they are not in this version and
write nothing; `share confirm` looks at the onion first and is always
refused.

The terminal's account comes from the single-user answer, which is read
from the auth settings and, when the auth store exists, from one count of
its accounts. That store is opened so that nothing in it is written: read
only while a WAL file or a journal lies beside it (pending frames are read
and never checkpointed, and a hot journal is never rolled back -- the
answer is then "not single-user"), otherwise as usual with SQL writes
refused.

### oo config

View and modify CLI configuration.

```bash
oo config                          # show current config
oo config set api_url http://remote:8000
oo config set animations false
oo config set animation_interval_ms 200
oo config reset                    # reset to defaults
```

| Key | Default | Description |
|-----|---------|-------------|
| `api_url` | `http://localhost:8001` | Backend API URL |
| `default_model` | (none) | Model to use when `-m` is not given |
| `output_format` | `text` | `text`, `json` or `markdown` |
| `color` | `true` | Colour output; `NO_COLOR` and `--no-color` turn it off |
| `timeout` | `120` | HTTP request timeout in seconds |
| `animations` | `true` | The wait animation of `oo chat` |
| `animation_interval_ms` | `150` | Time between two frames, 50 to 1000 |
| `animation_delay_ms` | `400` | Wait before the first frame, 100 to 5000 |
| `animation_stop_ms` | `100` | Longest the chat waits to erase a frame, 10 to 1000 |

`oo config set` refuses by name a value it cannot read -- an animation
value outside its range, a switch that is not true or false (yes/no,
on/off and 1/0 are read too), a timeout that is not a positive whole
number of seconds, an output format it does not know -- and leaves the
file as it was; it refuses as well to edit a file that is not a mapping of
settings. It writes what the file already holds plus the key it was given:
`NO_COLOR`, `--no-color`, `--api-url` and `OO_API_URL` change a run, never
the file, and `oo config reset` writes the defaults whatever the
environment says. In the file, every key is read alone: a value that
cannot be read falls back to its own default and leaves the others as
they are, and an unreadable `animations` is off.

| Subcommand | Description |
|------------|-------------|
| (none) | Show current configuration |
| `set KEY VALUE` | Set a configuration value |
| `reset` | Reset all settings to defaults |

Configuration is stored in `~/.config/opti-oignon/cli.yaml`.
