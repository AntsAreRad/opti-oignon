#!/usr/bin/env python3
"""
GGUF MODEL MANAGER -- OPTI-OIGNON S105
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
import http.client
import logging
import os
import socket
import struct
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

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
        self.architecture: Optional[str] = None
        self.model_name: Optional[str] = None
        self.author: Optional[str] = None
        self.description: Optional[str] = None
        self.context_length: Optional[int] = None
        self.embedding_length: Optional[int] = None
        self.block_count: Optional[int] = None
        self.head_count: Optional[int] = None
        self.head_count_kv: Optional[int] = None
        self.vocab_size: Optional[int] = None
        self.file_type: Optional[int] = None
        self.quantization_name: Optional[str] = None
        self.parameter_count: Optional[int] = None

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
    head_count: Optional[int] = None,
    head_count_kv: Optional[int] = None,
    vocab_size: Optional[int] = None,
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
# SSRF-safe download (MM-01, S185)
#
# _validate_download_url (S136) only validated the original URL. urllib's
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
    """
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
    headers: Optional[dict] = None,
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
    """
    from urllib.parse import urljoin

    resolver = resolver or socket.getaddrinfo
    opener = opener or _default_pinned_opener
    headers = headers or {}

    current = url
    hops = 0
    while True:
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

class ModelManager:
    """Manages local GGUF model files.

    Provides scanning, metadata extraction, download management,
    and storage tracking for GGUF model files.
    """

    def __init__(
        self,
        model_dirs: Optional[list[str]] = None,
        default_dir: Optional[str] = None,
    ):
        self._model_dirs = [Path(d) for d in (model_dirs or [])]
        self._default_dir = Path(default_dir) if default_dir else None
        self._metadata_cache: dict[str, GGUFMetadata] = {}
        self._lock = threading.Lock()
        self._active_downloads: dict[str, dict] = {}

    @property
    def model_dirs(self) -> list[Path]:
        """Return configured model directories."""
        return list(self._model_dirs)

    @property
    def default_dir(self) -> Optional[Path]:
        """Return the default directory for downloaded models."""
        return self._default_dir

    def add_model_dir(self, path: str | Path) -> bool:
        """Add a model directory to scan."""
        p = Path(path)
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
    ) -> Optional[dict]:
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
        """Validate a download URL to prevent SSRF attacks (S136 audit fix).

        Blocks:
          - Non-HTTPS URLs (except localhost for dev)
          - Private/internal IP ranges (10.x, 172.16-31.x, 192.168.x, 127.x)
          - Link-local, multicast, and loopback addresses
          - URLs without a hostname

        Raises ValueError if the URL is suspicious. This is the early-reject
        gate for the initial URL; MM-01 (S185) additionally re-validates and
        pins every redirect hop inside urlopen_ssrf_safe, so redirect-following
        and DNS rebinding can no longer bypass this check.
        """
        _validate_and_resolve(url, resolver=socket.getaddrinfo)

    def download_model(
        self,
        url: str,
        filename: Optional[str] = None,
        target_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Download a GGUF model from a URL.

        S136 audit fix: validates URL to prevent SSRF attacks.
        Only HTTPS URLs to public hosts are allowed.

        Args:
            url: Direct URL to the .gguf file.
            filename: Optional override for the saved filename.
            target_dir: Directory to save to (defaults to default_dir).
            progress_callback: Called with progress dicts:
                {"status": "downloading", "downloaded": N, "total": M, "percent": P}

        Returns:
            Dict with download result info.
        """
        # S136/MM-01 audit fix: SSRF protection. urlopen_ssrf_safe (below)
        # re-validates and pins every redirect hop, so neither redirect
        # following nor DNS rebinding can reach an internal address. This
        # call is the early-reject gate for the initial URL.
        self._validate_download_url(url)

        # Determine target path
        save_dir = Path(target_dir) if target_dir else self._default_dir
        if save_dir is None:
            raise ValueError(
                "No target directory specified and no default_dir configured"
            )
        save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            # Extract filename from URL
            url_path = url.split("?")[0]
            filename = url_path.split("/")[-1]
            if not filename.endswith(".gguf"):
                filename = f"{filename}.gguf"

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

            # Rename temp to final
            temp_path.rename(target_path)

            # Clear metadata cache for this path
            self._metadata_cache.pop(str(target_path), None)

            result = {
                "status": "completed",
                "path": str(target_path),
                "filename": filename,
                "size": downloaded,
                "size_human": _format_size(downloaded),
            }

            if progress_callback:
                progress_callback({"status": "completed", "percent": 100})

            logger.info("Download completed: %s (%s)", filename, _format_size(downloaded))
            return result

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

    def _resolve_path(self, name: str) -> Optional[Path]:
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

_model_manager_instance: Optional[ModelManager] = None
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


def init_model_manager(config_path: Optional[str] = None) -> ModelManager:
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
        manager._default_dir = Path(default_dir)

    return manager


def _load_model_config(config_path: Optional[str] = None) -> dict:
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

def _format_size(size_bytes: Optional[int]) -> str:
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


def _format_params(count: Optional[int]) -> Optional[str]:
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
