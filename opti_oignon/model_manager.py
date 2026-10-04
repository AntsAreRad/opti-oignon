#!/usr/bin/env python3
"""
GGUF MODEL MANAGER -- OPTI-OIGNON
======================================

Manages local GGUF model files: scanning directories, parsing
GGUF headers for metadata, downloading models from URLs, and
tracking storage usage.

The GGUF header parser is pure Python -- no external dependencies
required for model scanning and metadata extraction.

GGUF Format Reference (v3):
    - Magic: 0x47475546 ('GGUF')
    - Version: uint32
    - Tensor count: uint64
    - Metadata KV count: uint64
    - Metadata KV pairs (typed key-value)

Author: Leon
"""

import hashlib
import hmac
import http.client
import logging
import os
import re
import socket
import stat
import struct
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GGUF format constants
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x47475546  # 'GGUF' in little-endian
GGUF_MAGIC_BYTES = b"GGUF"

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# Struct format for each scalar type
_GGUF_SCALAR_FORMATS = {
    GGUF_TYPE_UINT8: ("<B", 1),
    GGUF_TYPE_INT8: ("<b", 1),
    GGUF_TYPE_UINT16: ("<H", 2),
    GGUF_TYPE_INT16: ("<h", 2),
    GGUF_TYPE_UINT32: ("<I", 4),
    GGUF_TYPE_INT32: ("<i", 4),
    GGUF_TYPE_FLOAT32: ("<f", 4),
    GGUF_TYPE_BOOL: ("<B", 1),
    GGUF_TYPE_UINT64: ("<Q", 8),
    GGUF_TYPE_INT64: ("<q", 8),
    GGUF_TYPE_FLOAT64: ("<d", 8),
}

# Well-known metadata keys
GGUF_KEY_ARCHITECTURE = "general.architecture"
GGUF_KEY_NAME = "general.name"
GGUF_KEY_AUTHOR = "general.author"
GGUF_KEY_DESCRIPTION = "general.description"
GGUF_KEY_FILE_TYPE = "general.file_type"
GGUF_KEY_QUANTIZATION = "general.quantization_version"
GGUF_KEY_CONTEXT_LENGTH = "{arch}.context_length"
GGUF_KEY_EMBEDDING_LENGTH = "{arch}.embedding_length"
GGUF_KEY_BLOCK_COUNT = "{arch}.block_count"
GGUF_KEY_HEAD_COUNT = "{arch}.attention.head_count"
GGUF_KEY_HEAD_COUNT_KV = "{arch}.attention.head_count_kv"
GGUF_KEY_VOCAB_SIZE = "{arch}.vocab_size"

# File type ID to quantization name mapping
_FILE_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS",
    21: "IQ3_XXS", 22: "IQ1_S", 23: "IQ4_NL",
    24: "IQ3_S", 25: "IQ2_S", 26: "IQ4_XS",
    27: "IQ1_M", 28: "BF16",
}


# ---------------------------------------------------------------------------
# GGUF header parser (pure Python)
# ---------------------------------------------------------------------------

class GGUFParseError(Exception):
    """Raised when GGUF file parsing fails."""
    pass


class GGUFMetadata:
    """Parsed GGUF file metadata."""

    def __init__(self):
        self.version: int = 0
        self.tensor_count: int = 0
        self.metadata_kv_count: int = 0
        self.metadata: dict[str, Any] = {}

        # Convenience fields extracted from metadata
        self.architecture: str | None = None
        self.model_name: str | None = None
        self.author: str | None = None
        self.description: str | None = None
        self.context_length: int | None = None
        self.embedding_length: int | None = None
        self.block_count: int | None = None
        self.head_count: int | None = None
        self.head_count_kv: int | None = None
        self.vocab_size: int | None = None
        self.file_type: int | None = None
        self.quantization_name: str | None = None
        self.parameter_count: int | None = None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "tensor_count": self.tensor_count,
            "architecture": self.architecture,
            "model_name": self.model_name,
            "author": self.author,
            "description": self.description,
            "context_length": self.context_length,
            "embedding_length": self.embedding_length,
            "block_count": self.block_count,
            "head_count": self.head_count,
            "head_count_kv": self.head_count_kv,
            "vocab_size": self.vocab_size,
            "file_type": self.file_type,
            "quantization_name": self.quantization_name,
            "parameter_count": self.parameter_count,
            "metadata_kv_count": self.metadata_kv_count,
        }


def parse_gguf_header(filepath: str | Path, max_kv_read: int = 200) -> GGUFMetadata:
    """Parse GGUF file header and extract metadata.

    This is a pure Python implementation that reads only the
    header portion of the file. It does NOT load tensors or
    model weights, making it fast even for very large files.

    Args:
        filepath: Path to the .gguf file.
        max_kv_read: Maximum number of metadata KV pairs to read.
                     Set to 0 for unlimited. Default 200 is enough
                     for all standard metadata while staying fast.

    Returns:
        GGUFMetadata with parsed header information.

    Raises:
        GGUFParseError: If the file is not a valid GGUF file.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise GGUFParseError(f"File not found: {filepath}")

    meta = GGUFMetadata()

    with open(filepath, "rb") as f:
        # Read magic bytes
        magic = f.read(4)
        if magic != GGUF_MAGIC_BYTES:
            raise GGUFParseError(
                f"Invalid GGUF magic bytes: {magic!r} "
                f"(expected {GGUF_MAGIC_BYTES!r})"
            )

        # Version (uint32)
        meta.version = _read_u32(f)
        if meta.version not in (1, 2, 3):
            raise GGUFParseError(
                f"Unsupported GGUF version: {meta.version}"
            )

        # Tensor count and metadata KV count
        if meta.version == 1:
            meta.tensor_count = _read_u32(f)
            meta.metadata_kv_count = _read_u32(f)
        else:
            meta.tensor_count = _read_u64(f)
            meta.metadata_kv_count = _read_u64(f)

        # Read metadata KV pairs
        kv_limit = meta.metadata_kv_count
        if max_kv_read > 0:
            kv_limit = min(kv_limit, max_kv_read)

        for _ in range(kv_limit):
            try:
                key = _read_string(f)
                value_type = _read_u32(f)
                value = _read_value(f, value_type)
                meta.metadata[key] = value
            except (struct.error, EOFError, GGUFParseError):
                # Reached end of readable metadata
                break

    # Extract convenience fields
    _extract_convenience_fields(meta)

    return meta


def _read_u32(f) -> int:
    """Read a little-endian uint32."""
    data = f.read(4)
    if len(data) < 4:
        raise GGUFParseError("Unexpected EOF reading uint32")
    return struct.unpack("<I", data)[0]


def _read_u64(f) -> int:
    """Read a little-endian uint64."""
    data = f.read(8)
    if len(data) < 8:
        raise GGUFParseError("Unexpected EOF reading uint64")
    return struct.unpack("<Q", data)[0]


def _read_i64(f) -> int:
    """Read a little-endian int64."""
    data = f.read(8)
    if len(data) < 8:
        raise GGUFParseError("Unexpected EOF reading int64")
    return struct.unpack("<q", data)[0]


def _read_string(f) -> str:
    """Read a GGUF string (uint64 length + bytes)."""
    length = _read_u64(f)
    if length > 1_000_000:  # Sanity check
        raise GGUFParseError(f"String length too large: {length}")
    data = f.read(length)
    if len(data) < length:
        raise GGUFParseError("Unexpected EOF reading string")
    return data.decode("utf-8", errors="replace")


def _read_value(f, value_type: int) -> Any:
    """Read a typed GGUF metadata value."""
    if value_type == GGUF_TYPE_STRING:
        return _read_string(f)

    if value_type == GGUF_TYPE_ARRAY:
        return _read_array(f)

    if value_type in _GGUF_SCALAR_FORMATS:
        fmt, size = _GGUF_SCALAR_FORMATS[value_type]
        data = f.read(size)
        if len(data) < size:
            raise GGUFParseError("Unexpected EOF reading scalar value")
        val = struct.unpack(fmt, data)[0]
        if value_type == GGUF_TYPE_BOOL:
            return bool(val)
        return val

    raise GGUFParseError(f"Unknown GGUF value type: {value_type}")


def _read_array(f) -> list:
    """Read a GGUF array value."""
    elem_type = _read_u32(f)
    length = _read_u64(f)

    if length > 10_000_000:  # Sanity check
        raise GGUFParseError(f"Array length too large: {length}")

    result = []
    for _ in range(length):
        result.append(_read_value(f, elem_type))
    return result


def _extract_convenience_fields(meta: GGUFMetadata) -> None:
    """Extract well-known fields from raw metadata into convenience attrs."""
    kv = meta.metadata

    meta.architecture = kv.get(GGUF_KEY_ARCHITECTURE)
    meta.model_name = kv.get(GGUF_KEY_NAME)
    meta.author = kv.get(GGUF_KEY_AUTHOR)
    meta.description = kv.get(GGUF_KEY_DESCRIPTION)
    meta.file_type = kv.get("general.file_type")

    if meta.file_type is not None:
        meta.quantization_name = _FILE_TYPE_NAMES.get(
            int(meta.file_type), f"unknown({meta.file_type})"
        )

    arch = meta.architecture or ""

    # Architecture-specific keys
    meta.context_length = kv.get(f"{arch}.context_length")
    meta.embedding_length = kv.get(f"{arch}.embedding_length")
    meta.block_count = kv.get(f"{arch}.block_count")
    meta.head_count = kv.get(f"{arch}.attention.head_count")
    meta.head_count_kv = kv.get(f"{arch}.attention.head_count_kv")
    meta.vocab_size = kv.get(f"{arch}.vocab_size")

    # Estimate parameter count from architecture
    if meta.embedding_length and meta.block_count:
        meta.parameter_count = _estimate_parameter_count(
            embedding_length=meta.embedding_length,
            block_count=meta.block_count,
            head_count=meta.head_count,
            head_count_kv=meta.head_count_kv,
            vocab_size=meta.vocab_size,
        )


def _estimate_parameter_count(
    embedding_length: int,
    block_count: int,
    head_count: int | None = None,
    head_count_kv: int | None = None,
    vocab_size: int | None = None,
) -> int:
    """Estimate total parameter count from model architecture.

    This is an approximation based on transformer architecture.
    The actual count may vary slightly depending on the model.
    """
    d = embedding_length
    n_layers = block_count
    v = vocab_size or 32000  # Common default

    # Embedding: vocab_size * embedding_length
    embed_params = v * d

    # Per transformer layer (approximate):
    # - Self-attention: 4 * d * d (Q, K, V, O projections)
    # - FFN: typically 3 * d * (4*d) for SwiGLU or 2 * d * (4*d) for GELU
    # Using a conservative 12 * d * d per layer
    layer_params = 12 * d * d

    # Final norm + output head
    final_params = d + v * d

    total = embed_params + (n_layers * layer_params) + final_params
    return total


# ---------------------------------------------------------------------------
# GGUF tensor table (pure Python, bounded)
#
# What each layer of a model weighs, read from the table of tensors that
# follows the metadata: a split model is placed layer by layer, and only the
# file says how large each layer is. The reader is held to bounds a damaged or
# hostile file cannot widen. Metadata values are stepped over, never kept,
# except the architecture name and the alignment; nothing is allocated from a
# declared length but a name already found short; and every refusal carries a
# reason from a closed set.
# ---------------------------------------------------------------------------

# ggml tensor types: id -> (name, elements per block, bytes per block), as
# ggml's type traits define them (checked against libggml 0.11.1). ggml
# retired ids 4 and 5, 31 to 33 and 36 to 38; an id not listed is refused.
GGML_TENSOR_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
}

# Every reason a tensor table is refused for; GGUFTableError.reason is one.
GGUF_TABLE_REFUSALS = frozenset(
    {
        "bad_magic",
        "unsupported_version",
        "too_many_tensors",
        "too_many_pairs",
        "name_too_long",
        "string_too_long",
        "array_too_long",
        "nesting_too_deep",
        "unknown_value_type",
        "bad_alignment",
        "bad_dimensions",
        "unknown_tensor_type",
        "partial_block",
        "misaligned_offset",
        "past_end_of_file",
        "duplicate_tensor",
        "truncated",
    }
)

# The reader's bounds. A tensor name shorter than 64 bytes and at most four
# dimensions are ggml's own limits (GGML_MAX_NAME, GGML_MAX_DIMS), and a key
# is at most 65535 bytes, as the format says. The rest bound the work a header
# can ask for, far above any model file: the tensors, the key/value pairs, the
# array elements declared over the whole header, and how deep arrays nest.
_TABLE_MAX_TENSORS = 1 << 17
_TABLE_MAX_PAIRS = 1 << 16
_TABLE_MAX_KEY = 65535
_TABLE_MAX_NAME = 63
_TABLE_MAX_ARCHITECTURE = 256
_TABLE_MAX_ELEMENTS = 1 << 24
_TABLE_MAX_NESTING = 4
_TABLE_MAX_DIMS = 4
_TABLE_DEFAULT_ALIGNMENT = 32
_TABLE_CHUNK = 1 << 20
_INT64_MAX = (1 << 63) - 1
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_KEY_ARCHITECTURE = GGUF_KEY_ARCHITECTURE.encode("ascii")
_KEY_ALIGNMENT = b"general.alignment"
_KEPT_KEY_LENGTHS = frozenset({len(_KEY_ARCHITECTURE), len(_KEY_ALIGNMENT)})
# A block's tensors: "blk.N." with N written as ggml writes it.
_BLOCK_NAME = re.compile(rb"blk\.(0|[1-9][0-9]{0,5})\.")


class GGUFTableError(GGUFParseError):
    """A tensor table refused; ``reason`` names why, from GGUF_TABLE_REFUSALS."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class GGUFTensorTable:
    """What a model file's tensors weigh, as its tensor table declares them.

    ``blocks`` maps each block index N (the tensors named ``blk.N.``) to its
    bytes, and ``other_bytes`` counts every tensor outside the blocks. The
    output head is ``output.weight``, else the token embedding the head is
    tied to (``output_tied``); its bytes are among ``other_bytes`` too. Bytes
    are exact: ggml's block layout of each tensor's type and shape.
    """

    version: int
    architecture: str | None
    alignment: int
    data_offset: int
    tensor_count: int
    blocks: Mapping[int, int]
    other_bytes: int
    output_bytes: int
    output_tied: bool
    total_bytes: int


class _TableReader:
    """Little-endian reads over a file in chunks, never past its size.

    ``elements`` is what the header may still declare in array elements.
    """

    def __init__(self, handle, size: int):
        self._handle = handle
        self.size = size
        self._buf = b""
        self._base = 0  # file offset of _buf[0]
        self._pos = 0  # read position within _buf
        self.elements = _TABLE_MAX_ELEMENTS

    def tell(self) -> int:
        return self._base + self._pos

    def _need(self, n: int) -> None:
        if self._pos + n <= len(self._buf):
            return
        at = self._base + self._pos
        if at + n > self.size:
            raise GGUFTableError("truncated", f"{n} bytes wanted at {at} of {self.size}")
        self._handle.seek(at)
        self._buf = self._handle.read(max(n, _TABLE_CHUNK))
        self._base, self._pos = at, 0
        if len(self._buf) < n:
            raise GGUFTableError("truncated", f"the file ended at {at + len(self._buf)}")

    def u32(self) -> int:
        self._need(4)
        value = _U32.unpack_from(self._buf, self._pos)[0]
        self._pos += 4
        return value

    def u64(self) -> int:
        self._need(8)
        value = _U64.unpack_from(self._buf, self._pos)[0]
        self._pos += 8
        return value

    def take(self, n: int) -> bytes:
        self._need(n)
        data = self._buf[self._pos:self._pos + n]
        self._pos += n
        return data

    def skip(self, n: int) -> None:
        at = self.tell() + n
        if at > self.size:
            raise GGUFTableError("truncated", f"{n} bytes skipped past the end at {self.tell()}")
        if self._pos + n <= len(self._buf):
            self._pos += n
        else:
            self._buf, self._base, self._pos = b"", at, 0

    def skip_strings(self, count: int) -> None:
        """Step over ``count`` strings, reading only their lengths."""
        buf, pos, end, unpack = self._buf, self._pos, len(self._buf), _U64.unpack_from
        for _ in range(count):
            if pos + 8 > end:
                self._pos = pos
                self._need(8)
                buf, pos, end = self._buf, self._pos, len(self._buf)
            n = unpack(buf, pos)[0]
            pos += 8
            if pos + n <= end:
                pos += n
            else:
                self._pos = pos
                self.skip(n)
                buf, pos, end = self._buf, self._pos, len(self._buf)
        self._pos = pos


def _scalar_bytes(kind: int) -> int:
    layout = _GGUF_SCALAR_FORMATS.get(kind)
    if layout is None:
        raise GGUFTableError("unknown_value_type", str(kind))
    return layout[1]


def _skip_value(reader: _TableReader, kind: int, depth: int = 0) -> None:
    """Step over one metadata value of type ``kind``, ``depth`` arrays deep."""
    if kind == GGUF_TYPE_STRING:
        reader.skip(reader.u64())
        return
    if kind != GGUF_TYPE_ARRAY:
        reader.skip(_scalar_bytes(kind))
        return
    if depth >= _TABLE_MAX_NESTING:
        raise GGUFTableError("nesting_too_deep", f"more than {_TABLE_MAX_NESTING} levels")
    element = reader.u32()
    count = reader.u64()
    size = 0 if element in (GGUF_TYPE_STRING, GGUF_TYPE_ARRAY) else _scalar_bytes(element)
    if count > reader.elements:
        raise GGUFTableError("array_too_long", f"{count} elements over the header's bound")
    reader.elements -= count
    if element == GGUF_TYPE_STRING:
        reader.skip_strings(count)
    elif element == GGUF_TYPE_ARRAY:
        for _ in range(count):
            _skip_value(reader, GGUF_TYPE_ARRAY, depth + 1)
    else:
        reader.skip(count * size)


def read_gguf_tensors(path: str | Path) -> GGUFTensorTable:
    """Read the tensor table of a GGUF file, version 2 or 3.

    Raises GGUFTableError, whose reason is one of GGUF_TABLE_REFUSALS, for a
    table that cannot be trusted; GGUFParseError for a path that is not a
    regular file; OSError when the file cannot be opened. A symbolic link is
    not followed, and opening never waits on a pipe.
    """
    flags = os.O_RDONLY
    for name in ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC", "O_BINARY"):
        flags |= getattr(os, name, 0)
    try:
        fd = os.open(path, flags)
    except (TypeError, ValueError) as exc:
        raise GGUFParseError(f"Not a file path: {path!r}") from exc
    try:
        # Before the descriptor becomes a file object, which refuses a
        # directory in a type of its own.
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise GGUFParseError(f"Not a regular file: {path}")
        handle = os.fdopen(fd, "rb", buffering=0)
    except BaseException:
        os.close(fd)
        raise
    with handle:
        return _read_table(_TableReader(handle, info.st_size))


def _read_table(reader: _TableReader) -> GGUFTensorTable:
    if reader.take(4) != GGUF_MAGIC_BYTES:
        raise GGUFTableError("bad_magic")
    version = reader.u32()
    if version not in (2, 3):
        raise GGUFTableError("unsupported_version", str(version))
    tensor_count = reader.u64()
    if tensor_count > _TABLE_MAX_TENSORS:
        raise GGUFTableError("too_many_tensors", str(tensor_count))
    pair_count = reader.u64()
    if pair_count > _TABLE_MAX_PAIRS:
        raise GGUFTableError("too_many_pairs", str(pair_count))

    architecture: str | None = None
    alignment = _TABLE_DEFAULT_ALIGNMENT
    for _ in range(pair_count):
        key_length = reader.u64()
        if key_length > _TABLE_MAX_KEY:
            raise GGUFTableError("name_too_long", f"a {key_length}-byte key")
        key = None
        if key_length in _KEPT_KEY_LENGTHS:
            key = reader.take(key_length)
        else:
            reader.skip(key_length)
        kind = reader.u32()
        if key == _KEY_ALIGNMENT:
            if kind != GGUF_TYPE_UINT32:
                raise GGUFTableError("bad_alignment", f"of type {kind}")
            alignment = reader.u32()
            if alignment == 0 or alignment & (alignment - 1):
                raise GGUFTableError("bad_alignment", str(alignment))
        elif key == _KEY_ARCHITECTURE and kind == GGUF_TYPE_STRING:
            length = reader.u64()
            if length > _TABLE_MAX_ARCHITECTURE:
                raise GGUFTableError("string_too_long", f"a {length}-byte architecture name")
            architecture = reader.take(length).decode("utf-8", errors="replace")
        else:
            _skip_value(reader, kind)

    seen: set[bytes] = set()
    placed: list[tuple[int, int]] = []
    blocks: dict[int, int] = {}
    other = 0
    output: int | None = None
    embedding: int | None = None
    for _ in range(tensor_count):
        name_length = reader.u64()
        if name_length > _TABLE_MAX_NAME:
            raise GGUFTableError("name_too_long", f"a {name_length}-byte tensor name")
        name = reader.take(name_length)
        if name in seen:
            raise GGUFTableError("duplicate_tensor", name.decode("utf-8", errors="replace"))
        seen.add(name)
        n_dims = reader.u32()
        if n_dims > _TABLE_MAX_DIMS:
            raise GGUFTableError("bad_dimensions", f"{n_dims} dimensions")
        dims = [reader.u64() for _ in range(n_dims)]
        elements = 1
        for dim in dims:
            elements *= dim
        if any(dim > _INT64_MAX for dim in dims) or elements > _INT64_MAX:
            raise GGUFTableError("bad_dimensions", str(dims))
        kind = reader.u32()
        layout = GGML_TENSOR_TYPES.get(kind)
        if layout is None:
            raise GGUFTableError("unknown_tensor_type", str(kind))
        _type_name, per_block, block_bytes = layout
        if (dims[0] if dims else 1) % per_block:
            raise GGUFTableError("partial_block", f"{dims} in blocks of {per_block}")
        nbytes = elements // per_block * block_bytes
        offset = reader.u64()
        if offset % alignment:
            raise GGUFTableError("misaligned_offset", f"{offset} against {alignment}")
        placed.append((offset, nbytes))
        block = _BLOCK_NAME.match(name)
        if block is not None:
            index = int(block.group(1))
            blocks[index] = blocks.get(index, 0) + nbytes
        else:
            other += nbytes
            if name == b"output.weight":
                output = nbytes
            elif name == b"token_embd.weight":
                embedding = nbytes

    # The data begins at the next multiple of the alignment. Each tensor must
    # lie inside the file, and together they cannot claim more bytes than the
    # file holds after that point.
    data_offset = -(-reader.tell() // alignment) * alignment
    total = 0
    for offset, nbytes in placed:
        if data_offset + offset + nbytes > reader.size:
            raise GGUFTableError("past_end_of_file", f"{nbytes} bytes at {offset}")
        total += nbytes
    if placed and total > reader.size - data_offset:
        raise GGUFTableError("past_end_of_file", f"{total} bytes claimed")

    if output is not None:
        head, tied = output, False
    elif embedding is not None:
        head, tied = embedding, True
    else:
        head, tied = 0, False
    return GGUFTensorTable(
        version=version,
        architecture=architecture,
        alignment=alignment,
        data_offset=data_offset,
        tensor_count=tensor_count,
        blocks=MappingProxyType(dict(sorted(blocks.items()))),
        other_bytes=other,
        output_bytes=head,
        output_tied=tied,
        total_bytes=total,
    )


# ---------------------------------------------------------------------------
# SSRF-safe download
#
# _validate_download_url only validated the original URL. urllib's
# default opener then followed HTTP redirects whose Location was never
# re-validated, and re-resolved DNS independently of validation, so a public
# host could 302 to a private IP and a name could flip its A record between
# validation and the connect (DNS rebinding / TOCTOU). urlopen_ssrf_safe
# follows redirects manually -- legitimate CDN redirects (HuggingFace ->
# S3/CloudFront) still work -- re-validating every hop and pinning the TCP
# connection to the IP that was just validated. TLS SNI and certificate
# verification still run against the original hostname; only the connect target
# is pinned, which closes both bypasses.
# ---------------------------------------------------------------------------

_REDIRECT_STATUSES = (301, 302, 303, 307, 308)
_MAX_REDIRECTS = 5


def _ip_is_blocked(ip_str: str) -> bool:
    """True if an IP is private/loopback/link-local/multicast/reserved."""
    import ipaddress

    ip = ipaddress.ip_address(ip_str)
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_validated_ips(hostname: str, port: int, *, resolver) -> list[str]:
    """Resolve a hostname and reject if any resolved IP is internal.

    Returns the list of resolved IPs (all confirmed routable). Rejecting when
    *any* resolved address is internal is intentionally strict: it removes the
    rebinding window where one of several A records points inside.

    An address is refused when it is private or internal, and also when the
    page fetch's own check (``web_search.address_refusal``) names a class for
    it: shared address space, site-local, this machine, a network on one of
    its links. The downloader and the page fetch refuse the same addresses.
    """
    from opti_oignon.web_search import address_refusal

    try:
        infos = resolver(hostname, port)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve hostname: {hostname}") from exc

    ips: list[str] = []
    for info in infos:
        ip_str = info[4][0]
        if _ip_is_blocked(ip_str):
            raise ValueError(
                f"Download URL resolves to private/internal IP "
                f"({hostname} -> {ip_str}). This may be an SSRF attempt."
            )
        found = address_refusal(ip_str)
        if found is not None:
            raise ValueError(
                f"Download URL resolves to a non-public address "
                f"({hostname} -> {ip_str}: {found}). This may be an SSRF attempt."
            )
        ips.append(ip_str)
    if not ips:
        raise ValueError(f"Cannot resolve hostname: {hostname}")
    return ips


def _validate_and_resolve(url: str, *, resolver) -> tuple[str, str, int, str]:
    """Validate a URL for SSRF and return (scheme, host, port, pinned_ip).

    pinned_ip is the validated address the connection must use, so the connect
    cannot be redirected to a private host by a flipped DNS record.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    scheme = parsed.scheme
    if scheme not in ("https", "http"):
        raise ValueError(
            f"Only HTTPS URLs allowed for model download, got: {scheme}"
        )
    host = parsed.hostname or ""
    if not host:
        raise ValueError("URL has no hostname")
    if scheme == "http" and host not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError(
            f"HTTP is only allowed for localhost. Use HTTPS for: {host}"
        )

    port = parsed.port or (443 if scheme == "https" else 80)
    ips = _resolve_validated_ips(host, port, resolver=resolver)
    return scheme, host, port, ips[0]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection whose TCP socket is pinned to a pre-validated IP.

    SNI and certificate verification still run against the original hostname
    (server_hostname=self.host); only the connect target is the validated IP,
    which defeats DNS rebinding between validation and the connect.
    """

    def __init__(self, host, *, pinned_ip, **kwargs):
        super().__init__(host, **kwargs)
        self._pinned_ip = pinned_ip

    def connect(self):
        sock = socket.create_connection(
            (self._pinned_ip, self.port), self.timeout
        )
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Plaintext HTTP connection pinned to a pre-validated IP (localhost dev)."""

    def __init__(self, host, *, pinned_ip, **kwargs):
        super().__init__(host, **kwargs)
        self._pinned_ip = pinned_ip

    def connect(self):
        self.sock = socket.create_connection(
            (self._pinned_ip, self.port), self.timeout
        )


class _SSRFSafeResponse:
    """Subset of HTTPResponse used by the downloader.

    Exposes .status, .headers and .read(); closing it on __exit__ also closes
    the underlying pinned connection.
    """

    def __init__(self, raw, conn):
        self._raw = raw
        self._conn = conn
        self.headers = raw.headers
        self.status = raw.status

    def read(self, amt: int = -1):
        return self._raw.read(amt)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self._raw.close()
        finally:
            if self._conn is not None:
                self._conn.close()


def _default_pinned_opener(url: str, pinned_ip: str, headers: dict, timeout: int):
    """Open a pinned connection for one hop and return the raw HTTPResponse."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    if parsed.scheme == "https":
        conn = _PinnedHTTPSConnection(
            host, pinned_ip=pinned_ip, port=port, timeout=timeout
        )
    else:
        conn = _PinnedHTTPConnection(
            host, pinned_ip=pinned_ip, port=port, timeout=timeout
        )
    conn.request("GET", path, headers=headers or {})
    raw = conn.getresponse()
    raw._conn_to_close = conn  # so the caller can close the socket
    return raw


def _close_raw(raw) -> None:
    conn = getattr(raw, "_conn_to_close", None)
    try:
        raw.close()
    except Exception:
        pass
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def urlopen_ssrf_safe(
    url: str,
    *,
    headers: dict | None = None,
    timeout: int = 30,
    max_redirects: int = _MAX_REDIRECTS,
    resolver=None,
    opener=None,
) -> "_SSRFSafeResponse":
    """Open a URL defending against SSRF via redirects and DNS rebinding.

    Redirects are followed manually so legitimate CDN redirects still work,
    re-validating and re-pinning every hop; the connection is pinned to the
    validated IP. resolver and opener are injectable for testing; the defaults
    use socket.getaddrinfo and a pinned HTTPS/HTTP connection.

    The web gate is asked through the front door before every hop is
    resolved; a refusal raises ``egress.EgressRefused``.
    """
    from urllib.parse import urljoin

    from opti_oignon.egress import require_web

    resolver = resolver or socket.getaddrinfo
    opener = opener or _default_pinned_opener
    headers = headers or {}

    current = url
    hops = 0
    while True:
        require_web("Model download")
        _scheme, _host, _port, pinned_ip = _validate_and_resolve(
            current, resolver=resolver
        )
        raw = opener(current, pinned_ip, headers, timeout)
        status = getattr(raw, "status", None)
        if status in _REDIRECT_STATUSES:
            location = raw.headers.get("Location") if raw.headers else None
            _close_raw(raw)
            if not location:
                raise ValueError("Redirect response without a Location header")
            hops += 1
            if hops > max_redirects:
                raise ValueError(f"Too many redirects (> {max_redirects})")
            current = urljoin(current, location)
            continue
        conn = getattr(raw, "_conn_to_close", None)
        return _SSRFSafeResponse(raw, conn)


# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------

def _sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Streaming sha256 over a file's bytes.

    Deliberately local and stdlib-only. This is the primitive that decides
    whether a substituted file is rejected, so it must not be reachable only
    through an optional import: a check that can be skipped because a module
    is missing is not a check.

    Streaming is not tidiness -- a GGUF routinely runs to tens of gigabytes,
    so reading it whole is not an option.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _record_provenance(target_path: Path, digest: str) -> dict:
    """Pin a freshly downloaded model in the provenance manifest.

    Reported rather than raised: the bytes are already on disk and verified
    against the expected digest if one was given, so failing the whole
    download over a manifest write would be the wrong trade. It is also safe,
    because an unenrolled model is refused by the load gate under enforcement
    -- an absent pin degrades to a refusal, never to a silent pass.
    """
    try:
        from opti_oignon.model_provenance import record_model

        recorded = record_model(target_path)
        return {"recorded": True, "scheme": recorded.get("scheme", "")}
    except Exception as exc:
        logger.warning(
            "Model downloaded but not pinned in the provenance manifest: %s",
            exc,
        )
        return {"recorded": False, "sha256": digest, "error": str(exc)}


class ModelManager:
    """Manages local GGUF model files.

    Provides scanning, metadata extraction, download management,
    and storage tracking for GGUF model files.
    """

    def __init__(
        self,
        model_dirs: list[str] | None = None,
        default_dir: str | None = None,
    ):
        # A configured directory written with ~ is the home directory.
        self._model_dirs = [Path(d).expanduser() for d in (model_dirs or [])]
        self._default_dir = Path(default_dir).expanduser() if default_dir else None
        self._metadata_cache: dict[str, GGUFMetadata] = {}
        self._lock = threading.Lock()
        self._active_downloads: dict[str, dict] = {}

    @property
    def model_dirs(self) -> list[Path]:
        """Return configured model directories."""
        return list(self._model_dirs)

    @property
    def default_dir(self) -> Path | None:
        """Return the default directory for downloaded models."""
        return self._default_dir

    def add_model_dir(self, path: str | Path) -> bool:
        """Add a model directory to scan; a leading ~ is the home directory."""
        p = Path(path).expanduser()
        if p in self._model_dirs:
            return False
        self._model_dirs.append(p)
        return True

    def scan_models(self, force_refresh: bool = False) -> list[dict]:
        """Scan all configured directories for GGUF files.

        Returns a list of model info dictionaries with metadata
        extracted from GGUF headers.
        """
        results = []
        seen = set()

        for d in self._model_dirs:
            if not d.is_dir():
                logger.debug("Model directory not found: %s", d)
                continue

            for gguf_path in sorted(d.glob("*.gguf")):
                if gguf_path.name in seen:
                    continue
                seen.add(gguf_path.name)

                info = self.get_model_info(str(gguf_path), force_refresh)
                if info:
                    results.append(info)

        return results

    def get_model_info(
        self, filepath: str, force_refresh: bool = False
    ) -> dict | None:
        """Get metadata for a specific GGUF model file.

        Results are cached by file path + mtime for efficiency.
        """
        filepath = str(filepath)
        path = Path(filepath)

        if not path.is_file():
            # Search in model dirs
            path = self._resolve_path(filepath)
            if path is None:
                return None

        # Cache key includes mtime to invalidate on file change
        try:
            mtime = path.stat().st_mtime
            file_size = path.stat().st_size
        except OSError:
            return None

        cache_key = f"{path}:{mtime}"

        if not force_refresh and cache_key in self._metadata_cache:
            meta = self._metadata_cache[cache_key]
        else:
            try:
                meta = parse_gguf_header(path)
                with self._lock:
                    self._metadata_cache[cache_key] = meta
            except GGUFParseError as exc:
                logger.warning("Failed to parse GGUF: %s -- %s", path, exc)
                return None
            except Exception as exc:
                logger.debug("Error reading GGUF header: %s -- %s", path, exc)
                return None

        return {
            "filename": path.name,
            "path": str(path),
            "file_size": file_size,
            "file_size_human": _format_size(file_size),
            "gguf_version": meta.version,
            "tensor_count": meta.tensor_count,
            "architecture": meta.architecture,
            "model_name": meta.model_name,
            "author": meta.author,
            "context_length": meta.context_length,
            "embedding_length": meta.embedding_length,
            "block_count": meta.block_count,
            "head_count": meta.head_count,
            "vocab_size": meta.vocab_size,
            "file_type": meta.file_type,
            "quantization_name": meta.quantization_name,
            "parameter_count": meta.parameter_count,
            "parameter_count_human": _format_params(meta.parameter_count),
        }

    def get_storage_usage(self) -> dict:
        """Calculate storage usage across all model directories.

        Returns:
            Dict with total size, per-directory breakdown, and model count.
        """
        total_size = 0
        model_count = 0
        dirs = []

        for d in self._model_dirs:
            if not d.is_dir():
                dirs.append({
                    "path": str(d),
                    "exists": False,
                    "size": 0,
                    "size_human": "0B",
                    "model_count": 0,
                })
                continue

            dir_size = 0
            dir_count = 0
            for gguf_path in d.glob("*.gguf"):
                try:
                    size = gguf_path.stat().st_size
                    dir_size += size
                    dir_count += 1
                except OSError:
                    continue

            total_size += dir_size
            model_count += dir_count
            dirs.append({
                "path": str(d),
                "exists": True,
                "size": dir_size,
                "size_human": _format_size(dir_size),
                "model_count": dir_count,
            })

        return {
            "total_size": total_size,
            "total_size_human": _format_size(total_size),
            "model_count": model_count,
            "directories": dirs,
        }

    @staticmethod
    def _validate_download_url(url: str) -> None:
        """Validate a download URL to prevent SSRF attacks.

        Blocks:
          - Non-HTTPS URLs (except localhost for dev)
          - Private/internal IP ranges (10.x, 172.16-31.x, 192.168.x, 127.x)
          - Link-local, multicast, and loopback addresses
          - URLs without a hostname

        Raises ValueError if the URL is suspicious. This is the early-reject
        gate for the initial URL additionally re-validates and
        pins every redirect hop inside urlopen_ssrf_safe, so redirect-following
        and DNS rebinding can no longer bypass this check.
        """
        _validate_and_resolve(url, resolver=socket.getaddrinfo)

    @staticmethod
    def _gguf_name(filename: str) -> str:
        """The name a download is saved under: its last path segment, ending in .gguf.

        ``../x.gguf`` and ``/tmp/x.gguf`` save as ``x.gguf``, ``a/b.gguf`` as
        ``b.gguf``, ``x.pth`` as ``x.pth.gguf``. A name that is empty, ``.``
        or ``..`` once reduced is refused with ValueError.
        """
        name = PurePosixPath(str(filename)).name
        if name in ("", ".", ".."):
            raise ValueError(f"Download filename {filename!r} names no file")
        return name if name.endswith(".gguf") else f"{name}.gguf"

    def _model_directory(self, target_dir: str | None) -> Path:
        """The directory a download is written to, inside a configured model directory.

        No ``target_dir`` means the default directory. A given one must
        resolve inside one of the model directories or the default one, each
        resolved too, so a link out of a model directory is refused with the
        rest. Returns the resolved directory; raises ValueError naming the
        configured directories otherwise.
        """
        if not target_dir:
            if self._default_dir is None:
                raise ValueError(
                    "No target directory specified and no default_dir configured"
                )
            return self._default_dir
        roots = [d for d in (*self._model_dirs, self._default_dir) if d is not None]
        wanted = Path(target_dir).expanduser().resolve()
        for root in roots:
            if wanted.is_relative_to(root.resolve()):
                return wanted
        listed = ", ".join(str(root) for root in roots) or "none is configured"
        raise ValueError(
            f"Download directory {target_dir} is not inside a model directory ({listed})"
        )

    def download_model(
        self,
        url: str,
        filename: str | None = None,
        target_dir: str | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        expected_sha256: str | None = None,
    ) -> dict:
        """Download a GGUF model from a URL.

        Audit fix: validates URL to prevent SSRF attacks.
        Only HTTPS URLs to public hosts are allowed.

        The SSRF guard proves WHERE the bytes came from. Only a digest proves
        WHAT they are. ``expected_sha256`` is the one check on this path that
        does not reduce to trust on first use: the caller supplies it out of
        band -- from the model card, not from the server that just served the
        file -- so a compromised or substituting mirror cannot satisfy it. It
        is verified BEFORE the partial file is promoted to a loadable .gguf.

        Args:
            url: Direct URL to the .gguf file.
            filename: Optional override for the saved filename.
            target_dir: Directory to save to (defaults to default_dir).
            progress_callback: Called with progress dicts:
                {"status": "downloading", "downloaded": N, "total": M, "percent": P}
            expected_sha256: Optional out-of-band digest. When supplied, a
                mismatch discards the download and no .gguf is ever created.

        Returns:
            Dict with download result info, including the computed sha256 and
            the outcome of enrolling it in the provenance manifest.

        Raises:
            EgressRefused: The web gate refused, before the first request, at a
                redirect or after a block; the partial file is removed.
            ValueError: The URL, the name or the directory is refused, before
                any request.
        """
        from opti_oignon.egress import EgressRefused, require_web

        # The web gate first, through the front door: nothing is resolved,
        # fetched or written when it refuses.
        require_web("Model download")

        # The write path: a .gguf named by its last segment, inside a model
        # directory. Checked before the URL is resolved.
        if filename is None:
            filename = url.split("?")[0].split("/")[-1]
        filename = self._gguf_name(filename)
        save_dir = self._model_directory(target_dir)

        # Audit fix: SSRF protection. urlopen_ssrf_safe (below)
        # re-validates and pins every redirect hop, so neither redirect
        # following nor DNS rebinding can reach an internal address. This
        # call is the early-reject gate for the initial URL.
        self._validate_download_url(url)

        save_dir.mkdir(parents=True, exist_ok=True)
        target_path = save_dir / filename

        if target_path.exists():
            return {
                "status": "exists",
                "path": str(target_path),
                "message": f"File already exists: {filename}",
            }

        # Track download
        download_id = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:8]
        self._active_downloads[download_id] = {
            "url": url,
            "filename": filename,
            "status": "starting",
            "downloaded": 0,
            "total": 0,
        }

        temp_path = target_path.with_suffix(".gguf.part")

        try:
            logger.info("Downloading GGUF model: %s -> %s", url, target_path)

            if progress_callback:
                progress_callback({
                    "status": "starting",
                    "downloaded": 0,
                    "total": 0,
                    "percent": 0,
                })

            with urlopen_ssrf_safe(
                url,
                headers={"User-Agent": "Opti-Oignon/2.0"},
                timeout=30,
            ) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                self._active_downloads[download_id]["total"] = total_size

                downloaded = 0
                block_size = 1024 * 1024  # 1MB blocks

                with open(temp_path, "wb") as out_file:
                    while True:
                        block = response.read(block_size)
                        if not block:
                            break
                        # The gate again after every block: a mode changed
                        # during the download stops it.
                        require_web("Model download")
                        out_file.write(block)
                        downloaded += len(block)

                        self._active_downloads[download_id].update({
                            "status": "downloading",
                            "downloaded": downloaded,
                        })

                        if progress_callback:
                            percent = (
                                (downloaded / total_size * 100)
                                if total_size > 0 else 0
                            )
                            progress_callback({
                                "status": "downloading",
                                "downloaded": downloaded,
                                "total": total_size,
                                "percent": round(percent, 1),
                            })

            # Integrity gate. Hashing happens on the .part file, so a
            # mismatch is discarded by the ValueError handler below and NO
            # loadable .gguf is ever materialised. Computed with no import
            # dependency at all: the check that refuses a substituted file
            # must not be capable of being skipped because an optional
            # module failed to load.
            digest = _sha256_file(temp_path)
            if expected_sha256:
                expected = str(expected_sha256).strip().lower()
                if not hmac.compare_digest(digest, expected):
                    raise ValueError(
                        "Downloaded file does not match the expected sha256 "
                        f"(expected {expected[:16]}..., got {digest[:16]}...)"
                    )

            # Rename temp to final
            temp_path.rename(target_path)

            # Clear metadata cache for this path
            self._metadata_cache.pop(str(target_path), None)

            # Enrol the digest so the load seam can verify it later. A failure
            # here is reported, never swallowed into a false success -- and it
            # is not fatal, because an unenrolled model is already refused by
            # the load gate under enforcement. The fail-secure property
            # composes: a missing pin cannot become a silent pass.
            provenance = _record_provenance(target_path, digest)

            result = {
                "status": "completed",
                "path": str(target_path),
                "filename": filename,
                "size": downloaded,
                "size_human": _format_size(downloaded),
                "sha256": digest,
                "provenance": provenance,
            }

            if progress_callback:
                progress_callback({"status": "completed", "percent": 100})

            logger.info("Download completed: %s (%s)", filename, _format_size(downloaded))
            return result

        except EgressRefused:
            # A refusal is raised, not reported: the route answers it as such.
            logger.warning("Download refused by the web gate: %s", url)
            if temp_path.exists():
                temp_path.unlink()
            raise
        except ValueError as exc:
            # SSRF rejection (private IP, redirect to private, rebinding) or
            # a malformed URL surfaced mid-download.
            logger.warning("Download blocked: %s - %s", url, exc)
            if temp_path.exists():
                temp_path.unlink()
            return {
                "status": "error",
                "message": f"Download blocked: {exc}",
                "url": url,
            }
        except (OSError, http.client.HTTPException) as exc:
            logger.error("Download failed: %s - %s", url, exc)
            if temp_path.exists():
                temp_path.unlink()
            return {
                "status": "error",
                "message": f"Download failed: {exc}",
                "url": url,
            }
        except Exception as exc:
            logger.error("Download error: %s -- %s", url, exc)
            if temp_path.exists():
                temp_path.unlink()
            return {
                "status": "error",
                "message": f"Unexpected error: {exc}",
                "url": url,
            }
        finally:
            self._active_downloads.pop(download_id, None)

    def get_active_downloads(self) -> list[dict]:
        """Return status of all active downloads."""
        return list(self._active_downloads.values())

    def delete_model(self, filepath: str) -> dict:
        """Delete a GGUF model file.

        Args:
            filepath: Path or filename of the model to delete.

        Returns:
            Result dict with status.
        """
        path = Path(filepath)
        if not path.is_file():
            path = self._resolve_path(filepath)
            if path is None:
                return {"status": "error", "message": f"Model not found: {filepath}"}

        try:
            size = path.stat().st_size
            path.unlink()
            # Clear cache
            for key in list(self._metadata_cache.keys()):
                if key.startswith(str(path)):
                    del self._metadata_cache[key]
            logger.info("Deleted model: %s (%s)", path.name, _format_size(size))
            return {
                "status": "deleted",
                "filename": path.name,
                "freed": size,
                "freed_human": _format_size(size),
            }
        except OSError as exc:
            return {"status": "error", "message": f"Delete failed: {exc}"}

    def clear_cache(self) -> int:
        """Clear the metadata cache. Returns number of entries cleared."""
        with self._lock:
            count = len(self._metadata_cache)
            self._metadata_cache.clear()
            return count

    # -- internal helpers --

    def _resolve_path(self, name: str) -> Path | None:
        """Resolve a model name to a full path."""
        for d in self._model_dirs:
            candidate = d / name
            if candidate.is_file():
                return candidate
            if not name.endswith(".gguf"):
                candidate = d / f"{name}.gguf"
                if candidate.is_file():
                    return candidate
        return None


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_model_manager_instance: ModelManager | None = None
_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """Return the global ModelManager singleton.

    Initial configuration is applied by init_model_manager().
    """
    global _model_manager_instance
    if _model_manager_instance is not None:
        return _model_manager_instance

    with _manager_lock:
        if _model_manager_instance is not None:
            return _model_manager_instance

        _model_manager_instance = ModelManager()
        return _model_manager_instance


def init_model_manager(config_path: str | None = None) -> ModelManager:
    """Initialize the model manager from backends.yaml configuration.

    Reads the llama_cpp.model_dirs setting and configures scanning
    directories accordingly.
    """
    manager = get_model_manager()

    # Load config
    cfg = _load_model_config(config_path)
    if not cfg:
        return manager

    llama_cfg = cfg.get("llama_cpp", {})
    model_dirs = llama_cfg.get("model_dirs", [])
    default_dir = llama_cfg.get("default_download_dir")

    for d in model_dirs:
        manager.add_model_dir(d)

    if default_dir:
        manager._default_dir = Path(default_dir).expanduser()

    return manager


def _load_model_config(config_path: str | None = None) -> dict:
    """Load backends.yaml for model directory configuration."""
    if config_path:
        p = Path(config_path)
    else:
        p = Path(__file__).parent / "config" / "backends.yaml"

    if not p.is_file():
        return {}

    try:
        import yaml
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _format_size(size_bytes: int | None) -> str:
    """Format a byte count into human-readable string."""
    if not size_bytes:
        return "0B"
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f}GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f}MB"
    if size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f}KB"
    return f"{size_bytes}B"


def _format_params(count: int | None) -> str | None:
    """Format a parameter count (e.g. 7_000_000_000 -> '7.0B')."""
    if count is None:
        return None
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)
