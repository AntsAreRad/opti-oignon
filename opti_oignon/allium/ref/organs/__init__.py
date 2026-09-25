"""The reference organs of the companion's engine: pure functions over law values.

Every function here takes the law and the founder pool as values, never by
name; the protocol loads the embedded files and hands them over. The Rust
twin under ``rust/allium/src/organs/`` answers every operation that reaches
the wire with the same bytes.
"""

checkpoint_before_apply = True
