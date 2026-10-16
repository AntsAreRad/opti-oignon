# Run Journal

The run journal is an opt-in record of your own agent runs, kept on this
machine for your own study. For each run you start with a task you typed,
it keeps your words and every tool call the model chose, in order, with what
the gates did with each call and the trust the run was in when the model
chose it. No model ever reads it. It is off until you turn it on.

## What is kept

One record per run, as one line of JSON (`"journal": 1`):

- the run's identifier -- the same one its proposals carry in the review
  queue -- its conversation, model, mode and network policy, its settings
  (verification, consulted skills, memory, round cap) and the tools it was
  offered;
- the words you typed, whole up to `max_value_chars` characters (a longer
  task is cut and said to be), and their SHA-256;
- when the run started (UTC), how long it took, why it stopped, its rounds,
  and the verifier's verdict;
- every call the model emitted, in order: its round and `t_ms`, the time
  from the run's start to when the call's round was done; the tool and the
  format it came in (native, or written in the text); its arguments as the
  model gave them -- each string cut at `max_value_chars`, the names cut
  listed, the SHA-256 taken over the whole; the gate's outcome (`ran`,
  `denied`, `refused`, `halted`, `unrunnable`, `error`, `not_dispatched`,
  or `other`) with its reason and whether you were asked; and its trust: the
  effect class and the label of each argument (typed, default, unendorsed)
  as the gate found them, the tools whose results the model had read when it
  chose the call, and the kinds of untrusted content it had been handed
  (memory, a tool's output, a skill).

A subtask or the verifier that made a call is a run of its own, `<id>-t1`,
`<id>-t2`, ... and `<id>-v`, naming the run that started it and its round;
each carries your typed words, which judge its calls. The loop does not
report a subtask's rounds or stop, so they are left empty.

## What is never kept

No tool's output, no subtask's answer, no memory block, no skills block, no
system prompt, nothing the model wrote but its calls, no final or verifier
answer, no spilled output, no proposal and no user identifier. A call's
arguments are the model's own, though, and can quote what it read: the
journal keeps them as they were.

Only agent runs whose task the caller vouches you typed are recorded -- the
agent panel's runs. Chat turns, evaluation and bench runs are not.

## Turning it on, pausing it, wiping it

The switch is `opti_oignon/config/run_journal.yaml`:

```yaml
enabled: true
retention_days: 30
```

It is on only when `enabled` is the YAML boolean `true`: a missing file or
key, the string `"true"` or `1` leave it off, and so does a file that cannot
be read (said once in the log). Off, a run creates and writes nothing.
Turning it off is a pause: what is kept stays until its retention ends, and a
run going when you turn it off is not written.

A record is kept less than `retention_days` days (1 to 30, 30 by default),
counted back from today in UTC. Each UTC day a run started on is one file,
`data/run_journal/YYYY-MM-DD.jsonl`, so a run is kept by the day it started:
a run whose day is already past the retention when it ends -- with a
retention of one day, a run that goes past midnight UTC -- is not written,
and is counted. The days past the retention are removed at boot, every hour
while the server runs, when a run starts (at most once an hour), and at each
write, deletion and wipe, whether the journal is on or off. A day dated
after tomorrow -- written while the clock ran ahead -- is removed too and
counted; the price is that a clock set back past the newest day removes
every day. A read never removes anything: it only leaves out the days a pass
would remove. A value outside its range is refused by name in the log and
the default kept. The other bounds -- `max_calls_per_run` (500),
`max_value_chars` (16384), `max_record_bytes` (262144) and `max_day_bytes`
(8388608) -- drop what passes them and count it: a run with a record over
`max_record_bytes` is not written at all.

To remove it at once: deleting a conversation, here or from a paired device,
removes that conversation's runs and their sub-runs; the user data wipe
(`DELETE /api/users/{id}/data`) and the full conversation wipe
(`POST /api/security/conversation-wipe/all?purge_disk=true`) remove every
file. A run still going when its conversation is deleted, or when the
journal is wiped, is not written when it ends. The lines the journal can
read are removed at once, with any line the loaded key cannot open; a sealed
line no key opens is never dropped for want of a key: the deletion waits in
the journal's ledger, written to disk at once -- the SHA-256 of the
conversation and the day, never its identifier; the newest day kept when
a clock that ran ahead filed later runs -- for the next pass that has the
key, reads leave that conversation's runs of that day and before out until
then, and the retention removes it in any case. A deletion that fails or
finds the journal busy waits the same way, under tomorrow. A run started
after a deletion is kept only when the deletion applied at once: a deletion
that waits for the key hides, then removes, that conversation's runs of
its day too. A day that cannot be rewritten for a deletion is removed whole
when it held one of that conversation's runs, and kept otherwise. The
journal holds one user's runs, so a wipe removes all of it; a wipe that
cannot says so: in `not_covered` in the user data wipe's answer, and under
`run_journal` in the full conversation wipe's. It is never exported and
never part of a backup; a fact deleted from memory or a source withdrawn
is not traced into it.

## Sealing

With encryption on in `config/security.yaml` -- read strictly: a file that
cannot say, empty or unparsable, is no leave to write in clear -- or while
the server holds an enabled key, each line is `JS1:` followed by the record
sealed with AES-256-GCM under the journal's own subkey of your master key
(HMAC-SHA256 of the label `oo-run-journal-v1`), with a fresh nonce for each
line. The key is the one that loads, or the one a passphrase set up in this
server holds. With encryption on and no key loaded, nothing is written. With
encryption off, a Daily run is written in plaintext and a Bulbe run not at
all. Records written in plaintext are sealed when encryption is turned on,
at setup or through the security configuration, or at the journal's next
pass, and a plaintext line a crash cut short is removed then; nothing is
ever unsealed.
Lines sealed under a key that no longer loads cannot be read: they are
counted, removed with their day, and removed by a wipe.

Each pass holds the journal's directory open and takes every file relative
to it, so a link put in the directory's place is never followed; the
directory must belong to the user the server runs as. It is created `0700`
and each file `0600`, whatever the umask, and both are held to those modes
at each write; no file is opened through a link or blocks on a pipe. A run's
records go out in one append, so a reader sees a run whole or not at all; a
tail cut short by a crash is skipped and counted. One process writes the
journal: running a second server process over the same data directory is
not supported. A reader only reads, and may be another process.

A journal that cannot record -- busy for more than half a second, a seal or
a write that fails, a bound passed -- drops what it held and counts why, and
says it once per reason in the log. It never changes a run's result, events
or tool results.

## Reading it

The journal is read on the machine, with the package:

```python
from opti_oignon import run_journal

for record in run_journal.iter_runs():          # every record, by start
    ...
for line in run_journal.trace_lines():          # the runs as trace lines
    ...
run_journal.status()                            # numbers only
```

`iter_runs` yields the records ordered by their start, one day at a time (a
run is written when it ends, so a long run can follow a shorter one in its
file); `kinds=("run",)` keeps the runs alone. `trace_lines` gives each run as
a line of the [injection bench's trace](injection-bench.md#the-trace),
format 1: a run alone, by the user, its calls with their trust, its start
in `started_at`. `status` says whether the journal is on, how it writes now
(sealed, plaintext, or why not), how many deletions wait for a key, and per
day the runs, sub-runs, calls and bytes kept, with the failures counted --
never a word of a record.
