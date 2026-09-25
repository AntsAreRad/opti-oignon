"""The reference engine, module for module with ``rust/allium/src``.

Standard library only, integers only, no hashed iteration order: no loop over
a ``set`` or ``frozenset``, no ``hash()``, ``id()`` or ``popitem()``, because
Python salts string hashing per process and the reference is the oracle.
"""

checkpoint_before_apply = True
