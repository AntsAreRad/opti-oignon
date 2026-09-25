"""The componion's engine (the companion onion): the Python reference and the seam to its Rust twin.

The being is a pure function of its facts. Everything that is hashed travels
in one canonical encoding (``wire``), every quantity is an integer in fixed
point (``fx``), and every random draw is addressed by content (``rng``). The
reference in this package is the readable specification and the oracle; the
Rust crate ``rust/allium``, linked into the native core, must answer byte for
byte as it does, and ``engine`` uses it only when its handshake agrees.

This package imports nothing at module level beyond the standard library,
starts no thread, and opens no file and no database at import; the store
opens its own database when called. Nothing on the chat path imports it.
"""

checkpoint_before_apply = True
