#!/usr/bin/env python3
"""
EMBEDDINGS - Ollama embedding interface
===============================================
Generate vector embeddings via Ollama.

Supported models:
- mxbai-embed-large (1024 dim, best quality)
- nomic-embed-text (768 dim, faster)

Auto-detects available embedding models if configured ones are missing.
Falls back to legacy /api/embeddings endpoint for older Ollama versions.
"""

import logging
import threading

import numpy as np
import requests
from tqdm import tqdm

from .config import EmbeddingConfig, get_config

logger = logging.getLogger(__name__)


class OllamaEmbeddings:
    """
    Client for generating embeddings via Ollama.

    Usage:
        embedder = OllamaEmbeddings()
        vectors = embedder.embed(["texte 1", "texte 2"])
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        """
        Initialize the Ollama embedding client.

        Args:
            config: Embedding configuration (optional)
        """
        self.config = config or get_config().embedding
        self.url = f"{self.config.ollama_url}/api/embed"
        self._model_verified = False
        self._use_legacy = False

    def _verify_model(self) -> bool:
        """Verify that the embedding model is available in Ollama.

        Resolution order:
        1. Check configured model (exact match or base-name match)
        2. Check configured fast_model
        3. Auto-discover any available embedding model
        Uses the exact full name from Ollama (including :tag) to avoid 400 errors.
        """
        if self._model_verified:
            return True

        try:
            list_url = f"{self.config.ollama_url}/api/tags"
            response = requests.get(list_url, timeout=10)
            response.raise_for_status()

            models = response.json().get("models", [])
            if not models:
                logger.error("No models available in Ollama")
                return False

            # Build lookup: base_name -> full_name (e.g. "mxbai-embed-large" -> "mxbai-embed-large:latest")
            full_names = {}
            for m in models:
                full = m.get("name", "")
                if not full:
                    continue
                base = full.split(":")[0]
                full_names[base] = full
                full_names[full] = full  # also map full name to itself

            # 1. Try configured model
            main_base = self.config.model.split(":")[0]
            if main_base in full_names:
                self.config.model = full_names[main_base]
                self._model_verified = True
                logger.info("Embedding model verified: %s", self.config.model)
                return True

            # 2. Try configured fast_model
            fast_base = self.config.fast_model.split(":")[0]
            if fast_base in full_names:
                logger.warning(
                    "Primary embedding model %s not found, using %s",
                    main_base, self.config.fast_model,
                )
                self.config.model = full_names[fast_base]
                self._model_verified = True
                return True

            # 3. Auto-discover any embedding model (name contains "embed")
            embed_keywords = ("embed", "nomic", "bge", "minilm", "e5-")
            for base_name, full_name in full_names.items():
                lower = base_name.lower()
                if any(kw in lower for kw in embed_keywords):
                    logger.warning(
                        "Auto-detected embedding model: %s (configured %s not found)",
                        full_name, main_base,
                    )
                    self.config.model = full_name
                    self._model_verified = True
                    return True

            available = list(full_names.keys())
            logger.error(
                "No embedding model found. Available models: %s. "
                "Install one with: ollama pull %s",
                available, main_base,
            )
            return False

        except requests.exceptions.ConnectionError:
            logger.error(
                "Cannot connect to Ollama (%s). Make sure Ollama is running.",
                self.config.ollama_url,
            )
            return False
        except Exception as e:
            logger.error("Model verification error: %s", e)
            return False

    def embed_single(self, text: str) -> list[float] | None:
        """
        Generate embedding for a single text.

        Args:
            text: The text to encode

        Returns:
            Embedding vector or None on error
        """
        if not self._verify_model():
            return None

        # RST-01: once switched to the legacy /api/embeddings endpoint, route
        # single embeds through the legacy path. The legacy endpoint expects the
        # "prompt" payload key and returns the singular "embedding" response key;
        # the /api/embed code path below sends "input" and parses the plural
        # "embeddings" key, so it would return None for a legacy 200 response
        # (the 400-based re-route does not fire because legacy responds 200).
        if self._use_legacy:
            return self._embed_single_legacy(text)

        try:
            payload = {
                "model": self.config.model,
                "input": text
            }

            response = requests.post(
                self.url,
                json=payload,
                timeout=self.config.timeout
            )

            # Handle 400 by trying legacy /api/embeddings endpoint
            if response.status_code == 400:
                return self._embed_single_legacy(text)

            response.raise_for_status()

            result = response.json()
            embeddings = result.get("embeddings", [])

            if embeddings:
                return embeddings[0]
            else:
                logger.warning("No embeddings returned")
                return None

        except requests.exceptions.Timeout:
            logger.error("Embedding timeout (>%ds)", self.config.timeout)
            return None
        except requests.exceptions.RequestException as e:
            logger.error("HTTP error during embedding: %s", e)
            return None
        except Exception as e:
            logger.error("Embedding error: %s", e)
            return None

    def _embed_single_legacy(self, text: str) -> list[float] | None:
        """Fallback: use legacy /api/embeddings endpoint (older Ollama versions)."""
        try:
            legacy_url = f"{self.config.ollama_url}/api/embeddings"
            payload = {
                "model": self.config.model,
                "prompt": text
            }
            response = requests.post(
                legacy_url,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            result = response.json()
            embedding = result.get("embedding", [])
            if embedding:
                # Switch to legacy endpoint for subsequent calls
                self.url = legacy_url
                self._use_legacy = True
                logger.info("Switched to legacy /api/embeddings endpoint")
                return embedding
            return None
        except Exception as e:
            logger.error("Legacy embedding endpoint also failed: %s", e)
            return None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """
        Generate embeddings for a batch of texts.

        Ollama supports native batching with the "input" parameter.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """
        if not self._verify_model():
            return [None] * len(texts)

        if not texts:
            return []

        # If we discovered we need legacy mode, use sequential
        if self._use_legacy:
            return self._embed_sequential(texts)

        try:
            payload = {
                "model": self.config.model,
                "input": texts
            }

            response = requests.post(
                self.url,
                json=payload,
                timeout=self.config.timeout * 2  # More time for batches
            )

            # Handle 400 by falling back to sequential (which tries legacy)
            if response.status_code == 400:
                logger.warning("Batch embed returned 400, falling back to sequential")
                return self._embed_sequential(texts)

            response.raise_for_status()

            result = response.json()
            embeddings = result.get("embeddings", [])

            if len(embeddings) != len(texts):
                logger.warning(
                    "Embedding count mismatch: got %d, expected %d -- "
                    "falling back to sequential to keep 1:1 alignment with inputs",
                    len(embeddings), len(texts),
                )
                # RST-02: a mismatched-length list, returned as-is, is zipped
                # against the inputs by the caller (rag_store._store_chunks),
                # which truncates to the shortest and can pair a chunk with the
                # wrong vector. Sequential embedding is length-preserving (one
                # slot per input, None on failure), so alignment is guaranteed.
                return self._embed_sequential(texts)

            return embeddings

        except requests.exceptions.Timeout:
            logger.warning("Batch timeout, falling back to sequential")
            return self._embed_sequential(texts)
        except Exception as e:
            logger.error("Batch embed error, falling back to sequential: %s", e)
            return self._embed_sequential(texts)

    def _embed_sequential(self, texts: list[str]) -> list[list[float] | None]:
        """Fallback: sequential embedding when batch fails."""
        results = []
        for text in texts:
            emb = self.embed_single(text)
            results.append(emb)
        return results

    def embed(
        self,
        texts: list[str],
        show_progress: bool = True,
        batch_size: int | None = None
    ) -> list[list[float] | None]:
        """
        Generate embeddings for a list of texts.

        Processes texts in batches for optimal performance.

        Args:
            texts: List of texts to encode
            show_progress: Show a progress bar
            batch_size: Batch size (default: config.batch_size)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if not self._verify_model():
            return [None] * len(texts)

        batch_size = batch_size or self.config.batch_size
        all_embeddings = []

        # Create batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        if show_progress:
            batches = tqdm(batches, desc="Embedding", unit="batch")

        for batch in batches:
            embeddings = self.embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def get_dimension(self) -> int:
        """Return the embedding dimension for the current model."""
        # Known dimensions per model
        dimensions = {
            "mxbai-embed-large": 1024,
            "nomic-embed-text": 768,
            "all-minilm": 384,
            "bge-m3": 1024,
        }

        model_name = self.config.model.split(":")[0]
        if model_name in dimensions:
            return dimensions[model_name]

        # Detect dynamically
        test_emb = self.embed_single("test")
        if test_emb:
            return len(test_emb)

        return self.config.dimension


class CachedEmbeddings:
    """
    Wrapper with cache to avoid recomputing embeddings.

    Uses a content hash to identify previously seen texts.
    """

    def __init__(self, embedder: OllamaEmbeddings | None = None):
        """
        Args:
            embedder: OllamaEmbeddings instance (creates a new one if not provided)
        """
        self.embedder = embedder or OllamaEmbeddings()
        self._cache = {}

    def _hash_text(self, text: str) -> str:
        """Generate a hash for the text."""
        import hashlib
        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()

    def embed(
        self,
        texts: list[str],
        show_progress: bool = True
    ) -> list[list[float] | None]:
        """
        Generate embeddings with caching.

        Args:
            texts: List of texts
            show_progress: Show progress bar

        Returns:
            List of embeddings
        """
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        # Check cache
        for i, text in enumerate(texts):
            text_hash = self._hash_text(text)
            if text_hash in self._cache:
                results[i] = self._cache[text_hash]
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        cache_hits = len(texts) - len(texts_to_embed)
        if cache_hits > 0:
            logger.info(f"Cache hits: {cache_hits}/{len(texts)}")

        # Compute new embeddings
        if texts_to_embed:
            new_embeddings = self.embedder.embed(texts_to_embed, show_progress)

            for i, (text, emb) in enumerate(zip(texts_to_embed, new_embeddings)):
                original_idx = indices_to_embed[i]
                results[original_idx] = emb

                # Store in cache
                if emb is not None:
                    text_hash = self._hash_text(text)
                    self._cache[text_hash] = emb

        return results

    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of cached embeddings."""
        return len(self._cache)


# =============================================================================
# BATCH EMBEDDING MANAGER
# =============================================================================

# Hardcoded, never overridable
checkpoint_before_apply = True

# Default batch size for grouping embedding requests
DEFAULT_EMBEDDING_BATCH_SIZE: int = 10


class BatchEmbeddingManager:
    """Groups embedding requests into batches for efficient Ollama calls.

    Instead of sending individual texts to Ollama one at a time, this
    manager accumulates texts and flushes them as a single batch call
    when the batch is full or when explicitly flushed.

    Thread-safe: multiple callers can ``add()`` concurrently; flushing
    is serialized internally.

    Parameters
    ----------
    embedder : OllamaEmbeddings
        The underlying embedder to use for batch calls.
    batch_size : int
        Number of texts to accumulate before auto-flushing (default: 10).

    Usage::

        mgr = BatchEmbeddingManager(embedder, batch_size=10)
        mgr.add("text A", callback=lambda emb: store(emb))
        mgr.add("text B", callback=lambda emb: store(emb))
        # ...when batch is full, callbacks fire automatically
        mgr.flush()  # force-flush any remaining
    """

    def __init__(
        self,
        embedder: OllamaEmbeddings,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._embedder = embedder
        self._batch_size = batch_size
        self._pending_texts: list[str] = []
        self._pending_callbacks: list[object] = []
        self._lock = threading.Lock()
        self._total_batches: int = 0
        self._total_texts: int = 0
        self._total_errors: int = 0

    @property
    def batch_size(self) -> int:
        """Configured batch size."""
        return self._batch_size

    @property
    def pending_count(self) -> int:
        """Number of texts waiting in the current batch."""
        with self._lock:
            return len(self._pending_texts)

    @property
    def stats(self) -> dict[str, int]:
        """Cumulative stats for this manager."""
        return {
            "total_batches": self._total_batches,
            "total_texts": self._total_texts,
            "total_errors": self._total_errors,
            "pending": self.pending_count,
            "batch_size": self._batch_size,
        }

    def add(
        self,
        text: str,
        callback: object = None,
    ) -> list[list[float] | None] | None:
        """Add a text to the pending batch.

        If the batch reaches ``batch_size``, it is automatically flushed
        and the resulting embeddings are returned.  The *callback*, if
        provided, is called with the individual embedding result for
        this text (``list[float] | None``).

        Parameters
        ----------
        text : str
            Text to embed.
        callback : callable, optional
            ``(embedding: list[float] | None) -> None``

        Returns
        -------
        list or None
            If the batch was flushed, returns all embeddings from this
            batch.  Otherwise returns None (text queued).
        """
        with self._lock:
            self._pending_texts.append(text)
            self._pending_callbacks.append(callback)
            if len(self._pending_texts) >= self._batch_size:
                return self._flush_locked()
        return None

    def flush(self) -> list[list[float] | None]:
        """Force-flush any pending texts and return their embeddings.

        Returns an empty list if nothing was pending.
        """
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> list[list[float] | None]:
        """Flush the current batch (caller must hold self._lock)."""
        if not self._pending_texts:
            return []

        texts = self._pending_texts[:]
        callbacks = self._pending_callbacks[:]
        self._pending_texts.clear()
        self._pending_callbacks.clear()

        self._total_batches += 1
        self._total_texts += len(texts)

        try:
            embeddings = self._embedder.embed_batch(texts)
        except Exception as exc:
            logger.error("Batch embedding flush failed: %s", exc)
            self._total_errors += 1
            embeddings = [None] * len(texts)

        # Fire individual callbacks
        for emb, cb in zip(embeddings, callbacks):
            if cb is not None and callable(cb):
                try:
                    cb(emb)
                except Exception as exc:
                    logger.debug("Embedding callback error: %s", exc)

        return embeddings

    def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float] | None]:
        """Embed a list of texts using optimal batching.

        Splits *texts* into groups of ``batch_size`` and calls
        ``embed_batch`` on each group.  Unlike the regular ``embed``
        method on ``OllamaEmbeddings``, this method does not show a
        progress bar and is designed for programmatic use.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        list
            Embedding vectors (or None for failures).
        """
        if not texts:
            return []

        all_embeddings: list[list[float] | None] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            self._total_batches += 1
            self._total_texts += len(batch)
            try:
                embs = self._embedder.embed_batch(batch)
                all_embeddings.extend(embs)
            except Exception as exc:
                logger.error("embed_many batch %d failed: %s", i // self._batch_size, exc)
                self._total_errors += 1
                all_embeddings.extend([None] * len(batch))

        return all_embeddings

    def reset_stats(self) -> None:
        """Reset cumulative counters."""
        self._total_batches = 0
        self._total_texts = 0
        self._total_errors = 0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_ollama_status() -> dict:
    """
    Check Ollama status and embedding model availability.

    Returns:
        Dictionary with status information
    """
    config = get_config().embedding
    status = {
        "ollama_running": False,
        "embedding_model_available": False,
        "model_name": config.model,
        "available_models": [],
        "error": None
    }

    try:
        # Check Ollama is responding
        response = requests.get(f"{config.ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        status["ollama_running"] = True

        # List available models
        models = response.json().get("models", [])
        status["available_models"] = [m.get("name") for m in models]

        # Check embedding model
        model_base = config.model.split(":")[0]
        for m in models:
            if model_base in m.get("name", ""):
                status["embedding_model_available"] = True
                break

    except requests.exceptions.ConnectionError:
        status["error"] = "Ollama not reachable. Run: ollama serve"
    except Exception as e:
        status["error"] = str(e)

    return status


def normalize_embeddings(embeddings: list[list[float]]) -> np.ndarray:
    """
    Normalize embeddings (L2 normalization).

    Args:
        embeddings: List of vectors

    Returns:
        Normalized numpy array
    """
    arr = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return arr / norms


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Ollama Embeddings Test ===\n")

    # Check status
    status = check_ollama_status()
    print(f"Ollama running: {status['ollama_running']}")
    print(f"Embedding model: {status['model_name']}")
    print(f"Model available: {status['embedding_model_available']}")

    if status['error']:
        print(f"Error: {status['error']}")
    else:
        print(f"Available models: {len(status['available_models'])}")

    if status['embedding_model_available']:
        print("\n--- Embedding test ---")
        embedder = OllamaEmbeddings()

        texts = [
            "R function to compute Shannon diversity index",
            "How to run a PCA with vegan in R",
            "Orthoptera diversity analysis"
        ]

        embeddings = embedder.embed(texts, show_progress=False)

        for text, emb in zip(texts, embeddings):
            if emb:
                print(f"'{text[:40]}...' -> dim={len(emb)}")
            else:
                print(f"'{text[:40]}...' -> ERROR")

        # Similarity test
        if all(embeddings):
            embeddings_norm = normalize_embeddings(embeddings)
            similarity = np.dot(embeddings_norm[0], embeddings_norm[1])
            print(f"\nSimilarity text 1-2: {similarity:.3f}")
