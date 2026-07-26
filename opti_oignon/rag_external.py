#!/usr/bin/env python3
"""
RAG EXTERNAL VECTOR STORE CONNECTORS.

Provides:
- Abstract connector interface (connect, query, disconnect, health)
- Qdrant connector (via qdrant-client, optional)
- Weaviate connector (via weaviate-client, optional)
- Pinecone connector (via pinecone-client, optional)
- ExternalVectorStoreManager: unified query across local + external stores
- All connectors gated by FEATURE_AVAILABLE flags
- Configurable via config/rag.yaml [external_stores] section

Usage::

    manager = ExternalVectorStoreManager()
    manager.register_connector("my-qdrant", QdrantConnector(
        url="http://localhost:6333", collection="papers"
    ))
    results = manager.query("What is Shannon diversity?", top_k=5)
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =========================================================================
# FEATURE FLAGS
# =========================================================================

QDRANT_AVAILABLE = False
try:
    from qdrant_client import QdrantClient as _QdrantClient
    from qdrant_client.models import Distance as _QdrantDistance
    from qdrant_client.models import VectorParams as _QdrantVectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    _QdrantClient = None
    _QdrantDistance = None
    _QdrantVectorParams = None

WEAVIATE_AVAILABLE = False
try:
    import weaviate as _weaviate_lib
    WEAVIATE_AVAILABLE = True
except ImportError:
    _weaviate_lib = None

PINECONE_AVAILABLE = False
try:
    from pinecone import Pinecone as _PineconeClient
    PINECONE_AVAILABLE = True
except ImportError:
    _PineconeClient = None

EXTERNAL_STORES_AVAILABLE = True  # Manager itself is always available


# =========================================================================
# DATA STRUCTURES
# =========================================================================

@dataclass
class ExternalSearchResult:
    """A single result from an external vector store."""
    content: str
    score: float
    source: str
    connector_name: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "score": round(self.score, 4),
            "source": self.source,
            "connector_name": self.connector_name,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
        }


@dataclass
class ConnectorStatus:
    """Health/status information for a connector."""
    name: str
    connector_type: str
    connected: bool
    document_count: int
    last_query_time_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "connector_type": self.connector_type,
            "connected": self.connected,
            "document_count": self.document_count,
            "last_query_time_ms": round(self.last_query_time_ms, 2),
            "error": self.error,
        }


# =========================================================================
# ABSTRACT CONNECTOR INTERFACE
# =========================================================================

class BaseVectorConnector(ABC):
    """
    Abstract interface for external vector store connectors.

    All connectors must implement connect, disconnect, query, and health.
    """

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._connected = False
        self._last_query_ms: float = 0.0
        self._last_error: str | None = None

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection cleanly."""
        ...

    @abstractmethod
    def query(
        self,
        query_text: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[ExternalSearchResult]:
        """
        Query the external store.

        Args:
            query_text: The search query (for text-based search).
            query_embedding: Pre-computed embedding vector (for vector search).
            top_k: Maximum number of results.
            filters: Optional metadata filters.

        Returns:
            List of ExternalSearchResult.
        """
        ...

    @abstractmethod
    def get_document_count(self) -> int:
        """Return the number of documents/vectors in the store."""
        ...

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def connector_type(self) -> str:
        return self.__class__.__name__

    def health(self) -> ConnectorStatus:
        """Return connector health status."""
        doc_count = 0
        try:
            if self._connected:
                doc_count = self.get_document_count()
        except Exception as exc:
            self._last_error = str(exc)

        return ConnectorStatus(
            name=self.name,
            connector_type=self.connector_type,
            connected=self._connected,
            document_count=doc_count,
            last_query_time_ms=self._last_query_ms,
            error=self._last_error,
        )


# =========================================================================
# QDRANT CONNECTOR
# =========================================================================

class QdrantConnector(BaseVectorConnector):
    """
    Connector for Qdrant vector database.

    Requires: pip install qdrant-client
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "default",
        api_key: str | None = None,
        name: str = "qdrant",
    ):
        super().__init__(name=name)
        self._url = url
        self._collection = collection
        self._api_key = api_key
        self._client: Any = None

    def connect(self) -> bool:
        if not QDRANT_AVAILABLE:
            self._last_error = "qdrant-client not installed"
            logger.warning("Qdrant connector: qdrant-client not installed")
            return False

        try:
            kwargs: dict[str, Any] = {"url": self._url}
            if self._api_key:
                kwargs["api_key"] = self._api_key

            self._client = _QdrantClient(**kwargs)
            # Verify connection by listing collections
            self._client.get_collections()
            self._connected = True
            self._last_error = None
            logger.info("Qdrant connected: %s", self._url)
            return True
        except Exception as exc:
            self._last_error = str(exc)
            self._connected = False
            logger.error("Qdrant connection failed: %s", exc)
            return False

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._connected = False

    def query(
        self,
        query_text: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[ExternalSearchResult]:
        if not self._connected or self._client is None:
            return []

        start = time.time()
        results: list[ExternalSearchResult] = []

        try:
            if query_embedding is None:
                logger.warning("Qdrant requires an embedding vector; no text search fallback")
                self._last_query_ms = (time.time() - start) * 1000
                return []

            search_result = self._client.search(
                collection_name=self._collection,
                query_vector=query_embedding,
                limit=top_k,
            )

            for hit in search_result:
                payload = hit.payload or {}
                results.append(ExternalSearchResult(
                    content=payload.get("content", payload.get("text", "")),
                    score=float(hit.score),
                    source=payload.get("source", "qdrant"),
                    connector_name=self.name,
                    chunk_id=str(hit.id),
                    metadata=payload,
                ))

            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)
            logger.error("Qdrant query failed: %s", exc)

        self._last_query_ms = (time.time() - start) * 1000
        return results

    def get_document_count(self) -> int:
        if not self._connected or self._client is None:
            return 0
        try:
            info = self._client.get_collection(self._collection)
            return info.points_count or 0
        except Exception:
            return 0


# =========================================================================
# WEAVIATE CONNECTOR
# =========================================================================

class WeaviateConnector(BaseVectorConnector):
    """
    Connector for Weaviate vector database.

    Requires: pip install weaviate-client
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        class_name: str = "Document",
        api_key: str | None = None,
        name: str = "weaviate",
    ):
        super().__init__(name=name)
        self._url = url
        self._class_name = class_name
        self._api_key = api_key
        self._client: Any = None

    def connect(self) -> bool:
        if not WEAVIATE_AVAILABLE:
            self._last_error = "weaviate-client not installed"
            logger.warning("Weaviate connector: weaviate-client not installed")
            return False

        try:
            auth = None
            if self._api_key:
                auth = _weaviate_lib.auth.AuthApiKey(api_key=self._api_key)

            self._client = _weaviate_lib.Client(
                url=self._url,
                auth_client_secret=auth,
            )
            # Verify connection
            self._client.schema.get()
            self._connected = True
            self._last_error = None
            logger.info("Weaviate connected: %s", self._url)
            return True
        except Exception as exc:
            self._last_error = str(exc)
            self._connected = False
            logger.error("Weaviate connection failed: %s", exc)
            return False

    def disconnect(self) -> None:
        self._client = None
        self._connected = False

    def query(
        self,
        query_text: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[ExternalSearchResult]:
        if not self._connected or self._client is None:
            return []

        start = time.time()
        results: list[ExternalSearchResult] = []

        try:
            query_builder = (
                self._client.query
                .get(self._class_name, ["content", "source"])
                .with_additional(["id", "certainty"])
                .with_limit(top_k)
            )

            if query_embedding is not None:
                query_builder = query_builder.with_near_vector(
                    {"vector": query_embedding}
                )
            elif query_text:
                query_builder = query_builder.with_near_text(
                    {"concepts": [query_text]}
                )

            response = query_builder.do()

            data = response.get("data", {}).get("Get", {}).get(self._class_name, [])
            for item in data:
                additional = item.get("_additional", {})
                results.append(ExternalSearchResult(
                    content=item.get("content", ""),
                    score=float(additional.get("certainty", 0.0)),
                    source=item.get("source", "weaviate"),
                    connector_name=self.name,
                    chunk_id=additional.get("id", uuid.uuid4().hex[:12]),
                    metadata={k: v for k, v in item.items() if k != "_additional"},
                ))

            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)
            logger.error("Weaviate query failed: %s", exc)

        self._last_query_ms = (time.time() - start) * 1000
        return results

    def get_document_count(self) -> int:
        if not self._connected or self._client is None:
            return 0
        try:
            result = (
                self._client.query
                .aggregate(self._class_name)
                .with_meta_count()
                .do()
            )
            agg = result.get("data", {}).get("Aggregate", {}).get(self._class_name, [{}])
            return agg[0].get("meta", {}).get("count", 0) if agg else 0
        except Exception:
            return 0


# =========================================================================
# PINECONE CONNECTOR
# =========================================================================

class PineconeConnector(BaseVectorConnector):
    """
    Connector for Pinecone vector database.

    Requires: pip install pinecone-client
    """

    def __init__(
        self,
        api_key: str = "",
        index_name: str = "default",
        environment: str = "",
        namespace: str = "",
        name: str = "pinecone",
    ):
        super().__init__(name=name)
        self._api_key = api_key
        self._index_name = index_name
        self._environment = environment
        self._namespace = namespace
        self._client: Any = None
        self._index: Any = None

    def connect(self) -> bool:
        if not PINECONE_AVAILABLE:
            self._last_error = "pinecone-client not installed"
            logger.warning("Pinecone connector: pinecone-client not installed")
            return False

        if not self._api_key:
            self._last_error = "Pinecone API key not configured"
            return False

        try:
            self._client = _PineconeClient(api_key=self._api_key)
            self._index = self._client.Index(self._index_name)
            # Verify by fetching stats
            self._index.describe_index_stats()
            self._connected = True
            self._last_error = None
            logger.info("Pinecone connected: index=%s", self._index_name)
            return True
        except Exception as exc:
            self._last_error = str(exc)
            self._connected = False
            logger.error("Pinecone connection failed: %s", exc)
            return False

    def disconnect(self) -> None:
        self._index = None
        self._client = None
        self._connected = False

    def query(
        self,
        query_text: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[ExternalSearchResult]:
        if not self._connected or self._index is None:
            return []

        start = time.time()
        results: list[ExternalSearchResult] = []

        try:
            if query_embedding is None:
                logger.warning("Pinecone requires an embedding vector; no text search fallback")
                self._last_query_ms = (time.time() - start) * 1000
                return []

            query_kwargs: dict[str, Any] = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True,
            }
            if self._namespace:
                query_kwargs["namespace"] = self._namespace
            if filters:
                query_kwargs["filter"] = filters

            response = self._index.query(**query_kwargs)

            for match in response.get("matches", []):
                metadata = match.get("metadata", {})
                results.append(ExternalSearchResult(
                    content=metadata.get("content", metadata.get("text", "")),
                    score=float(match.get("score", 0.0)),
                    source=metadata.get("source", "pinecone"),
                    connector_name=self.name,
                    chunk_id=match.get("id", uuid.uuid4().hex[:12]),
                    metadata=metadata,
                ))

            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)
            logger.error("Pinecone query failed: %s", exc)

        self._last_query_ms = (time.time() - start) * 1000
        return results

    def get_document_count(self) -> int:
        if not self._connected or self._index is None:
            return 0
        try:
            stats = self._index.describe_index_stats()
            return stats.get("total_vector_count", 0)
        except Exception:
            return 0


# =========================================================================
# EXTERNAL VECTOR STORE MANAGER
# =========================================================================

class ExternalVectorStoreManager:
    """
    Manages multiple external vector store connectors and provides
    a unified query interface across local (ChromaDB) + external stores.

    Usage::

        manager = ExternalVectorStoreManager()
        manager.register_connector("my-qdrant", QdrantConnector(...))
        manager.connect_all()
        results = manager.query("search terms", top_k=5)
    """

    def __init__(self):
        self._connectors: dict[str, BaseVectorConnector] = {}
        self._config: dict[str, Any] | None = None

    def register_connector(
        self,
        name: str,
        connector: BaseVectorConnector,
    ) -> None:
        """Register an external connector by name."""
        connector.name = name
        self._connectors[name] = connector
        logger.info("Registered external connector: %s (%s)", name, connector.connector_type)

    def unregister_connector(self, name: str) -> bool:
        """Unregister and disconnect a connector."""
        connector = self._connectors.pop(name, None)
        if connector is None:
            return False
        try:
            connector.disconnect()
        except Exception:
            pass
        return True

    def get_connector(self, name: str) -> BaseVectorConnector | None:
        """Get a connector by name."""
        return self._connectors.get(name)

    def list_connectors(self) -> list[ConnectorStatus]:
        """List all registered connectors with their status."""
        return [c.health() for c in self._connectors.values()]

    def connect_all(self) -> dict[str, bool]:
        """Connect all registered connectors. Returns name -> success map."""
        results: dict[str, bool] = {}
        for name, connector in self._connectors.items():
            results[name] = connector.connect()
        return results

    def disconnect_all(self) -> None:
        """Disconnect all connectors."""
        for connector in self._connectors.values():
            try:
                connector.disconnect()
            except Exception as exc:
                logger.debug("Disconnect error for %s: %s", connector.name, exc)

    def query(
        self,
        query_text: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        connector_names: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[ExternalSearchResult]:
        """
        Query across all connected external stores (or a subset).

        Args:
            query_text: Search query text.
            query_embedding: Pre-computed embedding vector.
            top_k: Max results per connector.
            connector_names: Subset of connectors to query (None = all).
            filters: Optional metadata filters.

        Returns:
            Merged list of results sorted by score descending.
        """
        all_results: list[ExternalSearchResult] = []
        targets = connector_names or list(self._connectors.keys())

        for name in targets:
            connector = self._connectors.get(name)
            if connector is None or not connector.connected:
                continue

            try:
                results = connector.query(
                    query_text=query_text,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    filters=filters,
                )
                all_results.extend(results)
            except Exception as exc:
                logger.error("External query failed for %s: %s", name, exc)

        # Sort by score descending and deduplicate by content hash
        all_results.sort(key=lambda r: r.score, reverse=True)
        return self._deduplicate(all_results, top_k)

    @staticmethod
    def _deduplicate(
        results: list[ExternalSearchResult],
        max_results: int,
    ) -> list[ExternalSearchResult]:
        """Deduplicate results by content similarity (exact match)."""
        seen: set[str] = set()
        deduped: list[ExternalSearchResult] = []

        for r in results:
            # Use first 200 chars of content as dedup key
            key = r.content[:200].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
            if len(deduped) >= max_results:
                break

        return deduped

    def _load_config(self) -> dict[str, Any]:
        """Load external stores config from rag.yaml."""
        if self._config is not None:
            return self._config

        defaults: dict[str, Any] = {
            "enabled": False,
            "connectors": [],
        }
        try:
            import yaml
            config_path = Path(__file__).parent / "config" / "rag.yaml"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                ext_cfg = cfg.get("external_stores", {})
                if isinstance(ext_cfg, dict):
                    defaults.update(ext_cfg)
        except Exception as exc:
            logger.debug("Could not load external_stores config: %s", exc)

        self._config = defaults
        return defaults

    def auto_configure(self) -> int:
        """
        Auto-configure connectors from rag.yaml settings.

        Returns the number of connectors registered.
        """
        cfg = self._load_config()
        if not cfg.get("enabled", False):
            return 0

        count = 0
        for conn_cfg in cfg.get("connectors", []):
            if not isinstance(conn_cfg, dict):
                continue

            conn_type = conn_cfg.get("type", "").lower()
            conn_name = conn_cfg.get("name", conn_type)

            try:
                if conn_type == "qdrant" and QDRANT_AVAILABLE:
                    connector = QdrantConnector(
                        url=conn_cfg.get("url", "http://localhost:6333"),
                        collection=conn_cfg.get("collection", "default"),
                        api_key=conn_cfg.get("api_key"),
                        name=conn_name,
                    )
                    self.register_connector(conn_name, connector)
                    count += 1

                elif conn_type == "weaviate" and WEAVIATE_AVAILABLE:
                    connector = WeaviateConnector(
                        url=conn_cfg.get("url", "http://localhost:8080"),
                        class_name=conn_cfg.get("class_name", "Document"),
                        api_key=conn_cfg.get("api_key"),
                        name=conn_name,
                    )
                    self.register_connector(conn_name, connector)
                    count += 1

                elif conn_type == "pinecone" and PINECONE_AVAILABLE:
                    connector = PineconeConnector(
                        api_key=conn_cfg.get("api_key", ""),
                        index_name=conn_cfg.get("index_name", "default"),
                        environment=conn_cfg.get("environment", ""),
                        namespace=conn_cfg.get("namespace", ""),
                        name=conn_name,
                    )
                    self.register_connector(conn_name, connector)
                    count += 1

                else:
                    logger.warning(
                        "Unknown or unavailable connector type: %s", conn_type
                    )
            except Exception as exc:
                logger.error(
                    "Failed to configure connector %s: %s", conn_name, exc
                )

        return count

    def get_available_backends(self) -> dict[str, bool]:
        """Return which external backends have their client library installed."""
        return {
            "qdrant": QDRANT_AVAILABLE,
            "weaviate": WEAVIATE_AVAILABLE,
            "pinecone": PINECONE_AVAILABLE,
        }


# =========================================================================
# MODULE-LEVEL SINGLETON
# =========================================================================

_external_manager: ExternalVectorStoreManager | None = None


def get_external_manager() -> ExternalVectorStoreManager:
    """Return the module-level ExternalVectorStoreManager singleton."""
    global _external_manager
    if _external_manager is None:
        _external_manager = ExternalVectorStoreManager()
    return _external_manager
