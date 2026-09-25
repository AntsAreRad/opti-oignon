//! Randomness addressed by content: the twin of `opti_oignon/allium/rng.py`.
//!
//! `key` is SHA-256 over a prefix, the domain, the seed and 8-byte big-endian
//! parts; `Stream` is SplitMix64 seeded from a key; `noise` mixes a position
//! into a key. Arithmetic modulo 2^64 is the one place wrapping is the
//! intended operation, and it is written as such.

use sha2::{Digest, Sha256};

pub const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_B: u64 = 0x94D0_49BB_1331_11EB;
pub const PREFIX: &[u8] = b"oo-allium-rng-v1";

pub fn key(seed: &[u8; 32], domain: &str, parts: &[u64]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(PREFIX);
    digest.update([0_u8]);
    digest.update(domain.as_bytes());
    digest.update([0_u8]);
    digest.update(seed);
    for part in parts {
        digest.update(part.to_be_bytes());
    }
    digest.finalize().into()
}

pub fn first_word(bytes: &[u8; 32]) -> u64 {
    let mut word = [0_u8; 8];
    word.copy_from_slice(bytes.get(..8).unwrap_or(&[0_u8; 8]));
    u64::from_be_bytes(word)
}

pub fn mix(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(MIX_A);
    let z = (z ^ (z >> 27)).wrapping_mul(MIX_B);
    z ^ (z >> 31)
}

pub fn noise(k: u64, counter: u64) -> u64 {
    mix(k ^ counter.wrapping_mul(GOLDEN))
}

/// The rejection threshold of `below(n)`: 2^64 mod n.
pub fn below_threshold(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    0_u64.wrapping_sub(n).checked_rem(n).unwrap_or(0)
}

pub struct Stream {
    state: u64,
}

impl Stream {
    pub fn new(seed: &[u8; 32], domain: &str, index: u64) -> Stream {
        Stream { state: first_word(&key(seed, domain, &[index])) }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN);
        mix(self.state)
    }

    /// A uniform integer in [0, n), without modulo bias; n >= 1.
    pub fn below(&mut self, n: u64) -> u64 {
        let threshold = below_threshold(n);
        loop {
            let word = self.next_u64();
            if word >= threshold {
                return word.checked_rem(n).unwrap_or(0);
            }
        }
    }

    /// A Q16 level: the top sixteen bits of a word.
    pub fn unit(&mut self) -> u64 {
        self.next_u64() >> 48
    }
}
