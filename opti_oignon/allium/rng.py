"""Randomness addressed by content: every draw names its domain, nothing is global.

The reference. Its Rust twin (``rust/allium/src/rng.rs``) draws the same
64-bit words.

* ``key(seed, domain, parts)`` is the SHA-256 of ``"oo-allium-rng-v1"``, a
  zero byte, the ASCII domain, a zero byte, the 32-byte seed, then each part
  as an 8-byte big-endian word. It serves the rare decisions: coining a word,
  a meiosis, a sound law.
* ``Stream(seed, domain, index)`` is SplitMix64 seeded by the first eight
  bytes of ``key(seed, domain, (index,))``, for draws in bulk.
* ``noise(k, counter)`` mixes a counter into a key: noise addressed by
  position, never by a running state.

Adding draws in one domain changes no other domain, which is what lets the
laws evolve without rewriting a lineage. ``below(n)`` rejects the low
``2^64 mod n`` words so every residue is equally likely.
"""

import hashlib

checkpoint_before_apply = True

M64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB
PREFIX = b"oo-allium-rng-v1"
SEED_BYTES = 32


def _domain_bytes(domain):
    raw = domain.encode("ascii")
    if not raw or b"\x00" in raw or any(byte < 0x20 or byte > 0x7E for byte in raw):
        raise ValueError("a domain is printable ASCII, not empty")
    return raw


def key(seed, domain, parts=()):
    """The 32-byte key of a decision, from its seed, domain and integer parts."""
    if len(seed) != SEED_BYTES:
        raise ValueError("a seed is 32 bytes")
    digest = hashlib.sha256()
    digest.update(PREFIX + b"\x00" + _domain_bytes(domain) + b"\x00" + bytes(seed))
    for part in parts:
        if not 0 <= part <= M64:
            raise ValueError("a part is an unsigned 64-bit word")
        digest.update(part.to_bytes(8, "big"))
    return digest.digest()


def mix(z):
    """The SplitMix64 finaliser."""
    z = ((z ^ (z >> 30)) * _MIX_A) & M64
    z = ((z ^ (z >> 27)) * _MIX_B) & M64
    return z ^ (z >> 31)


def noise(k, counter):
    """A word addressed by position: ``mix(k ^ counter * GOLDEN)``."""
    return mix((k ^ ((counter * GOLDEN) & M64)) & M64)


def below_threshold(n):
    """The rejection threshold of ``below(n)``: ``2^64 mod n``."""
    if n < 1:
        raise ValueError("below takes n >= 1")
    return ((1 << 64) - n) % n


class Stream:
    """SplitMix64 over a domain's key: a reproducible sequence of 64-bit words."""

    __slots__ = ("state",)

    def __init__(self, seed, domain, index=0):
        self.state = int.from_bytes(key(seed, domain, (index,))[:8], "big")

    def next_u64(self):
        self.state = (self.state + GOLDEN) & M64
        return mix(self.state)

    def below(self, n):
        """A uniform integer in [0, n), without modulo bias."""
        threshold = below_threshold(n)
        while True:
            word = self.next_u64()
            if word >= threshold:
                return word % n

    def unit(self):
        """A Q16 level in [0, 65535]: the top sixteen bits of a word."""
        return self.next_u64() >> 48
