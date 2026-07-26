#!/usr/bin/env python3
"""Published-prose guard: the generated schema must match its digest.

Two independent things move the prose a client reads. One is an edit to a
string literal, which every prover here already treats as executable shape.
The other is an edit to a DOCSTRING -- and the framework publishes several
kinds of docstring verbatim into the generated schema: the one on a route
handler becomes the endpoint description, the one on a model class becomes
the schema description, and there is no reason to believe that list is
closed. A prover taught only the first kind waves the second one through and
the public surface moves in silence.

So this guard does not teach a prover what the framework publishes. It asks
the framework and records the answer. It builds a DIGEST of the generated
schema -- one line per published description, keyed by where the description
lives, valued by the md5 of its text -- and compares that against the digest
committed next to this script. Any movement of published prose then shows up
as a line of diff in a tracked file, in front of a reviewer, whether or not
any prover understood the construct that moved.

Three properties make the digest worth trusting:

  * It stores hashes, never the prose. This file must not become a second
    copy of the public surface, and a reviewer does not need the text to see
    that a description moved -- the key says which one.
  * The key carries the identifier. A renamed model or endpoint moves its
    line even when its prose is untouched, so a rename cannot hide either.
  * Framework-generated text is deliberately excluded. Response descriptions
    and endpoint summaries are manufactured from names, not written; they
    would add thousands of lines that can only move when a key already moved.

``build_digest`` is pure and import-safe: it takes a schema mapping and
returns text. Everything that imports the application lives in ``main``.

Usage:
    published_prose_guard.py            compare the schema to the digest
    published_prose_guard.py --write    record the current schema
"""

import contextlib
import hashlib
import io
import os
import sys

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

_HEADER = (
    "# Published descriptions of the generated schema.\n"
    "# One line per description: KIND KEY MD5. Sorted, hashes only.\n"
    "# Regenerate with: python .github/scripts/published_prose_guard.py --write\n"
)

_DIGEST_NAME = "published_prose.digest"


def _hash(text):
    """md5 of a description, over its UTF-8 bytes."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_digest(schema):
    """Text digest of every authored description in ``schema``.

    Four kinds are recorded, and they are exactly the four a source edit can
    move without moving a key:

      schema NAME            -- the docstring of a model class
      field  NAME.PROPERTY   -- the description argument of a model field
      op     OPERATION       -- the docstring of a route handler
      param  OPERATION.NAME  -- the description argument of a parameter

    Pure: no import of the application, no filesystem, no clock.
    """
    lines = []

    for name, model in sorted((schema.get("components") or {})
                              .get("schemas", {}).items()):
        if not isinstance(model, dict):
            continue
        text = model.get("description")
        if isinstance(text, str):
            lines.append(f"schema {name} {_hash(text)}")
        for prop, field in sorted((model.get("properties") or {}).items()):
            if not isinstance(field, dict):
                continue
            text = field.get("description")
            if isinstance(text, str):
                lines.append(f"field {name}.{prop} {_hash(text)}")

    for path, operations in sorted((schema.get("paths") or {}).items()):
        if not isinstance(operations, dict):
            continue
        for verb, operation in sorted(operations.items()):
            if not isinstance(operation, dict):
                continue
            key = operation.get("operationId") or f"{verb.upper()} {path}"
            text = operation.get("description")
            if isinstance(text, str):
                lines.append(f"op {key} {_hash(text)}")
            for parameter in (operation.get("parameters") or []):
                if not isinstance(parameter, dict):
                    continue
                text = parameter.get("description")
                if isinstance(text, str):
                    name = parameter.get("name", "?")
                    lines.append(f"param {key}.{name} {_hash(text)}")

    return _HEADER + "".join(f"{line}\n" for line in sorted(lines))


def entries(digest):
    """Mapping of KIND+KEY -> MD5 for a digest text, comments dropped."""
    found = {}
    for line in digest.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            found[parts[0]] = parts[1]
    return found


def compare(recorded, computed):
    """Movements from ``recorded`` to ``computed``.

    Returns three sorted lists of keys: appeared, disappeared, and rewritten.
    An empty triple means the published prose has not moved.
    """
    was, now = entries(recorded), entries(computed)
    appeared = sorted(set(now) - set(was))
    disappeared = sorted(set(was) - set(now))
    rewritten = sorted(k for k in set(was) & set(now) if was[k] != now[k])
    return appeared, disappeared, rewritten


def _load_schema():
    """Generated schema of the live application, startup chatter suppressed."""
    sys.path.insert(0, os.getcwd())
    noise = io.StringIO()
    with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
        from opti_oignon.api.app import app
        return app.openapi()


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    digest_path = os.path.join(here, _DIGEST_NAME)
    computed = build_digest(_load_schema())

    if "--write" in argv:
        with open(digest_path, "w", encoding="utf-8") as handle:
            handle.write(computed)
        total = len(entries(computed))
        print(f"published-prose guard: recorded {total} description(s)")
        return 0

    try:
        with open(digest_path, encoding="utf-8") as handle:
            recorded = handle.read()
    except FileNotFoundError:
        print(f"published-prose guard: {_DIGEST_NAME} is missing; "
              f"run with --write to record the current schema", file=sys.stderr)
        return 1

    appeared, disappeared, rewritten = compare(recorded, computed)
    if not (appeared or disappeared or rewritten):
        total = len(entries(computed))
        print(f"published-prose guard: {total} published description(s) "
              f"match the recorded digest")
        return 0

    print("published-prose guard: the published prose moved and the digest "
          "was not updated with it", file=sys.stderr)
    for key in rewritten:
        print(f"  rewritten     {key}", file=sys.stderr)
    for key in disappeared:
        print(f"  no longer published  {key}", file=sys.stderr)
    for key in appeared:
        print(f"  newly published      {key}", file=sys.stderr)
    print("\nIf the movement is intended, record it in the same change:\n"
          "  python .github/scripts/published_prose_guard.py --write",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
