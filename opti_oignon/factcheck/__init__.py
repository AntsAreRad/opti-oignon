"""A fact-check core that decides over evidence it is handed, and reaches nothing.

"Supported" means supported by this passage of this source, read on this date,
never "true". The core locates every passage itself, in a chunk whose hash it
recomputes; it supports only a whole source sentence restated verbatim under a
fixed fold, from a source the owner or a named third party wrote, valid at the
time the claim is about, in a context that does not qualify it. Everything else
is "not enough evidence" with a closed reason, or "out of scope" with its
reason. Every verdict is a record that replays.

The package imports the standard library alone at module level, reads its
YAML lazily, and imports nothing from the application; nothing in the
application imports it yet. ``docs/architecture/fact-check.md`` says what it
decides and why, what it will never say, and what it cannot see.

Modules: ``vocabulary`` (every closed code, and the verdict type), ``markup``
(the markdown subset), ``passage`` (fold v1, the passage check, sentences and
windows), ``scope`` (claims, wrappers, owner normalisation and lexicons),
``sources`` (items, chunks, admission, validity), ``decide`` (the check),
``record`` (canonical JSON, digest, replay), ``render`` (the templates),
``canary`` (the planted errors) and ``checker`` (the entry point).
"""

checkpoint_before_apply = True
