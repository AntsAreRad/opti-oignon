#!/usr/bin/env python3
"""
RAG v2 API routes.

POST   /api/rag/ingest             -- Upload and ingest document
POST   /api/rag/ingest/url         -- Ingest from URL
POST   /api/rag/ingest/batch       -- Batch ingest multiple files
POST   /api/rag/ingest/folder      -- Scan folder and ingest
GET    /api/rag/ingest/jobs        -- List ingestion jobs
GET    /api/rag/ingest/jobs/{id}   -- Single job status
DELETE /api/rag/ingest/jobs/{id}   -- Cancel/delete a job
GET    /api/rag/collections        -- List collections
POST   /api/rag/collections        -- Create collection
DELETE /api/rag/collections/{name} -- Delete collection
POST   /api/rag/query              -- Query with retrieval + citations
POST   /api/rag/query/stream       -- Query with chunked transfer encoding
GET    /api/rag/documents          -- List ingested documents (enhanced)
DELETE /api/rag/documents/{doc_id} -- Remove document + chunks
POST   /api/rag/injection-defense/sanitize-preview  -- Preview sanitized chunks
POST   /api/rag/injection-defense/approve           -- Approve/reject chunks
GET    /api/rag/injection-defense/audit              -- Query audit log
DELETE /api/rag/injection-defense/audit              -- Clear audit log
GET    /api/rag/injection-defense/config             -- Get defense config
"""

import logging
import os
import tempfile
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Chunked transfer for large RAG responses
try:
    from opti_oignon.chunked_response import DEFAULT_CHUNK_SIZE, chunked_json_generator
    CHUNKED_RESPONSE_AVAILABLE = True
except ImportError:
    CHUNKED_RESPONSE_AVAILABLE = False
    chunked_json_generator = None  # type: ignore[assignment]
    DEFAULT_CHUNK_SIZE = 4096

# Audit fix: require authentication for all endpoints
try:
    from .routes_auth import _get_current_user
    _auth_dep = [Depends(_get_current_user)]
except ImportError:
    _auth_dep = []

router = APIRouter(prefix="/api/rag", tags=["rag"], dependencies=_auth_dep)


# =========================================================================
# PYDANTIC SCHEMAS
# =========================================================================

# -- Collections --

class CollectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field("", max_length=500)


class CollectionResponse(BaseModel):
    name: str
    description: str
    document_count: int
    chunk_count: int
    created_at: float
    updated_at: float


class CollectionsListResponse(BaseModel):
    collections: list[CollectionResponse]
    total: int


class CollectionDeleteResponse(BaseModel):
    deleted: bool
    name: str


# -- Documents --

class DocumentResponse(BaseModel):
    doc_id: str
    collection_name: str
    source_file: str
    file_type: str
    chunk_count: int
    raw_text_length: int
    ingested_at: float
    metadata: dict[str, Any] = {}


class DocumentsListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class DocumentDeleteResponse(BaseModel):
    deleted: bool
    doc_id: str


# -- Ingestion --

class IngestResponse(BaseModel):
    doc_id: str
    collection_name: str
    source_file: str
    file_type: str
    chunk_count: int
    raw_text_length: int
    ingested_at: float


class IngestURLRequest(BaseModel):
    url: str = Field(..., min_length=8)
    collection: str = Field("default", max_length=100)
    metadata: dict[str, Any] = {}


# -- Batch Ingestion --

class IngestFolderRequest(BaseModel):
    directory: str = Field(..., min_length=1, description="Absolute path to the folder to scan")
    collection: str = Field("default", max_length=100)
    recursive: bool = Field(True, description="Recurse into subdirectories")


class IngestFileStatusResponse(BaseModel):
    file_id: str
    job_id: str
    filepath: str
    filename: str
    file_size: int
    status: str
    doc_id: str | None = None
    chunk_count: int = 0
    error_message: str | None = None
    started_at: float | None = None
    completed_at: float | None = None


class IngestJobResponse(BaseModel):
    job_id: str
    status: str
    collection: str
    source_type: str
    source_path: str | None = None
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    progress: float = 0.0
    created_at: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None
    files: list[IngestFileStatusResponse] = []


class IngestJobsListResponse(BaseModel):
    jobs: list[IngestJobResponse]
    total: int


class IngestJobDeleteResponse(BaseModel):
    deleted: bool
    job_id: str


# -- Query --

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str = Field("default", max_length=100)
    n_results: int = Field(5, ge=1, le=50)
    min_score: float = Field(0.3, ge=0.0, le=1.0)
    source_filter: str | None = None
    file_type_filter: str | None = None
    rerank: bool = True
    track_citations: bool = True
    # Optional chunk size for streamed responses
    chunk_size: int = Field(DEFAULT_CHUNK_SIZE, ge=64, le=65536)


class CitationResponse(BaseModel):
    citation_id: str
    query: str
    collection_name: str
    chunk_id: str
    parent_doc_id: str
    source_file: str
    section: str | None = None
    score: float
    timestamp: float


class RetrievalResultResponse(BaseModel):
    content: str
    score: float
    source_file: str
    file_type: str
    chunk_index: int
    total_chunks: int
    parent_doc_id: str
    collection_name: str
    section: str | None = None
    page: int | None = None


class QueryResponseSchema(BaseModel):
    query: str
    results: list[RetrievalResultResponse]
    citations: list[CitationResponse]
    total_results: int


# =========================================================================
# HELPERS
# =========================================================================

def _get_store():
    """Get the RAGVectorStore singleton, raising 503 if unavailable."""
    try:
        from opti_oignon.rag_store import RAG_STORE_AVAILABLE, RAGVectorStore  # noqa: F401
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="RAG store module is not available",
        )

    try:
        from opti_oignon.rag_store import get_rag_store
        return get_rag_store()
    except Exception as exc:
        logger.error("Failed to initialize RAG store: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"RAG store initialisation failed: {exc}",
        )


def _get_batch_engine():
    """Get the BatchIngestEngine singleton, raising 503 if unavailable."""
    try:
        from opti_oignon.rag.batch_ingest import get_batch_ingest_engine
        return get_batch_ingest_engine()
    except Exception as exc:
        logger.error("Failed to initialize batch ingest engine: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Batch ingest engine initialisation failed: {exc}",
        )


def _job_to_response(job) -> IngestJobResponse:
    """Convert an IngestJobRecord to an API response."""
    return IngestJobResponse(
        job_id=job.job_id,
        status=job.status,
        collection=job.collection,
        source_type=job.source_type,
        source_path=job.source_path,
        total_files=job.total_files,
        completed_files=job.completed_files,
        failed_files=job.failed_files,
        skipped_files=job.skipped_files,
        total_chunks=job.total_chunks,
        progress=round(job.progress, 4),
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        files=[
            IngestFileStatusResponse(
                file_id=f.file_id,
                job_id=f.job_id,
                filepath=f.filepath,
                filename=f.filename,
                file_size=f.file_size,
                status=f.status,
                doc_id=f.doc_id,
                chunk_count=f.chunk_count,
                error_message=f.error_message,
                started_at=f.started_at,
                completed_at=f.completed_at,
            )
            for f in job.files
        ],
    )


# =========================================================================
# COLLECTION ENDPOINTS
# =========================================================================

@router.get("/collections", response_model=CollectionsListResponse)
def list_collections() -> dict:
    """List all knowledge base collections with stats."""
    store = _get_store()
    collections = store.list_collections()
    items = [
        CollectionResponse(
            name=c.name,
            description=c.description,
            document_count=c.document_count,
            chunk_count=c.chunk_count,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in collections
    ]
    return CollectionsListResponse(collections=items, total=len(items))


@router.post("/collections", response_model=CollectionResponse, status_code=201)
def create_collection(request: CollectionCreateRequest) -> dict:
    """Create a new collection."""
    store = _get_store()
    info = store.create_collection(name=request.name, description=request.description)
    return CollectionResponse(
        name=info.name,
        description=info.description,
        document_count=info.document_count,
        chunk_count=info.chunk_count,
        created_at=info.created_at,
        updated_at=info.updated_at,
    )


@router.delete("/collections/{name}", response_model=CollectionDeleteResponse)
def delete_collection(name: str) -> dict:
    """Delete a collection and all its documents, chunks, and citations."""
    store = _get_store()
    deleted = store.delete_collection(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return CollectionDeleteResponse(deleted=True, name=name)


# =========================================================================
# INGESTION ENDPOINTS
# =========================================================================

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    collection: str = Form("default"),
) -> dict:
    """
    Upload and ingest a document (PDF, DOCX, XLSX, CSV, TXT, MD, etc.).

    The file is temporarily saved, chunked, embedded, and stored in
    the specified collection.

    BUG-11 S108: Improved validation and error handling.
    """
    store = _get_store()

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Sanitize collection name
    collection = (collection or "default").strip()
    if not collection:
        collection = "default"

    # Save to temp file
    suffix = os.path.splitext(file.filename)[1] or ".txt"
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {exc}",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = store.ingest_file(
            filepath=tmp_path,
            collection=collection,
            metadata={"original_filename": file.filename},
        )
        return IngestResponse(
            doc_id=doc.doc_id,
            collection_name=doc.collection_name,
            source_file=file.filename,
            file_type=doc.file_type,
            chunk_count=doc.chunk_count,
            raw_text_length=doc.raw_text_length,
            ingested_at=doc.ingested_at,
        )
    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/ingest/url", response_model=IngestResponse)
def ingest_url(request: IngestURLRequest) -> dict:
    """
    Ingest a web page by URL.

    Fetches the page, extracts readable text (stripping nav, ads,
    boilerplate), chunks, embeds, and stores.
    """
    store = _get_store()

    try:
        doc = store.ingest_url(
            url=request.url,
            collection=request.collection,
            metadata=request.metadata,
        )
        return IngestResponse(
            doc_id=doc.doc_id,
            collection_name=doc.collection_name,
            source_file=doc.source_file,
            file_type=doc.file_type,
            chunk_count=doc.chunk_count,
            raw_text_length=doc.raw_text_length,
            ingested_at=doc.ingested_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("URL ingestion failed for %s: %s", request.url, exc)
        raise HTTPException(status_code=500, detail=f"URL ingestion failed: {exc}")


# =========================================================================
# BATCH INGESTION ENDPOINTS
# =========================================================================

@router.post("/ingest/batch", response_model=IngestJobResponse, status_code=202)
async def ingest_batch(
    files: list[UploadFile] = File(...),
    collection: str = Form("default"),
) -> dict:
    """
    Upload and ingest multiple files in a single request.

    Files are saved to a temp directory, a background job is created,
    and ingestion proceeds asynchronously. Returns the job record
    immediately so the client can poll for progress.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Sanitize collection
    collection = (collection or "default").strip() or "default"

    # Save all files to temp directory
    tmp_dir = tempfile.mkdtemp(prefix="rag_batch_")
    saved_paths: list[str] = []

    for upload in files:
        if not upload.filename:
            continue

        suffix = os.path.splitext(upload.filename)[1] or ".txt"  # noqa: F841
        # Use a unique name to avoid collisions
        safe_name = f"{len(saved_paths):04d}_{upload.filename}"
        tmp_path = os.path.join(tmp_dir, safe_name)

        try:
            content = await upload.read()
            if len(content) == 0:
                logger.warning("Skipping empty file: %s", upload.filename)
                continue
            with open(tmp_path, "wb") as f:
                f.write(content)
            saved_paths.append(tmp_path)
        except Exception as exc:
            logger.error("Failed to save uploaded file %s: %s", upload.filename, exc)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid files to ingest")

    # Create and start the batch job
    engine = _get_batch_engine()
    job = engine.create_batch_job(filepaths=saved_paths, collection=collection)
    engine.start_job(job.job_id)

    # Refresh job state
    job = engine.get_job(job.job_id)
    return _job_to_response(job)


@router.post("/ingest/folder", response_model=IngestJobResponse, status_code=202)
def ingest_folder(request: IngestFolderRequest) -> dict:
    """
    Scan a local directory for supported files and ingest them.

    Creates a background job that processes all discovered files.
    Returns the job record immediately for progress polling.
    """
    from pathlib import Path

    # Validate directory path against traversal (SA-155-042)
    raw_dir = request.directory
    if ".." in raw_dir.replace("\\", "/").split("/"):
        raise HTTPException(
            status_code=400,
            detail="Directory path must not contain '..' components.",
        )

    directory = Path(raw_dir).resolve()

    # Reject symlinks that resolve outside their parent
    raw_path = Path(raw_dir)
    if raw_path.is_symlink():
        link_target = raw_path.resolve()
        if not str(link_target).startswith(str(raw_path.parent.resolve())):
            raise HTTPException(
                status_code=400,
                detail="Symlink targets outside its parent directory are not allowed.",
            )

    if not directory.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Directory not found: {request.directory}",
        )

    # Sanitize collection
    collection = (request.collection or "default").strip() or "default"

    engine = _get_batch_engine()
    try:
        job = engine.create_folder_job(
            directory=request.directory,
            collection=collection,
            recursive=request.recursive,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if job.total_files == 0:
        return _job_to_response(job)

    engine.start_job(job.job_id)

    # Refresh job state
    job = engine.get_job(job.job_id)
    return _job_to_response(job)


@router.get("/ingest/jobs", response_model=IngestJobsListResponse)
def list_ingest_jobs(
    status: str | None = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """List batch ingestion jobs with optional status filter."""
    engine = _get_batch_engine()
    jobs = engine.list_jobs(status=status, limit=limit, offset=offset)
    items = [_job_to_response(j) for j in jobs]
    # Jobs from list_jobs don't include files by default; fetch them
    for item, j in zip(items, jobs):
        # list_jobs returns jobs without files; only include file count
        item.files = []
    return IngestJobsListResponse(jobs=items, total=len(items))


@router.get("/ingest/jobs/{job_id}", response_model=IngestJobResponse)
def get_ingest_job(job_id: str) -> dict:
    """Get detailed status of a single ingestion job, including per-file progress."""
    engine = _get_batch_engine()
    job = engine.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _job_to_response(job)


@router.delete("/ingest/jobs/{job_id}", response_model=IngestJobDeleteResponse)
def delete_ingest_job(job_id: str) -> dict:
    """Cancel and delete a batch ingestion job."""
    engine = _get_batch_engine()

    # Try to cancel first if running
    engine.cancel_job(job_id)

    deleted = engine.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return IngestJobDeleteResponse(deleted=True, job_id=job_id)


# =========================================================================
# QUERY ENDPOINT
# =========================================================================

@router.post("/query", response_model=QueryResponseSchema)
def query_knowledge_base(request: QueryRequest) -> dict:
    """
    Query the knowledge base with retrieval and citation tracking.

    Returns the top-k most relevant chunks from the specified collection,
    along with citation records linking each result to its source.
    """
    store = _get_store()

    response = store.query(
        query_text=request.query,
        collection=request.collection,
        n_results=request.n_results,
        min_score=request.min_score,
        source_filter=request.source_filter,
        file_type_filter=request.file_type_filter,
        rerank=request.rerank,
        track_citations=request.track_citations,
    )

    results = [
        RetrievalResultResponse(
            content=r.content,
            score=r.score,
            source_file=r.source_file,
            file_type=r.file_type,
            chunk_index=r.chunk_index,
            total_chunks=r.total_chunks,
            parent_doc_id=r.parent_doc_id,
            collection_name=r.collection_name,
            section=r.section,
            page=r.page,
        )
        for r in response.results
    ]

    citations = [
        CitationResponse(
            citation_id=c.citation_id,
            query=c.query,
            collection_name=c.collection_name,
            chunk_id=c.chunk_id,
            parent_doc_id=c.parent_doc_id,
            source_file=c.source_file,
            section=c.section,
            score=c.score,
            timestamp=c.timestamp,
        )
        for c in response.citations
    ]

    return QueryResponseSchema(
        query=response.query,
        results=results,
        citations=citations,
        total_results=response.total_results,
    )


# =========================================================================
# STREAMING QUERY ENDPOINT
# =========================================================================

@router.post("/query/stream")
def query_knowledge_base_stream(request: QueryRequest):
    """Query the knowledge base with chunked transfer encoding (S159).

    Returns the same JSON payload as ``/query`` but delivered via
    chunked transfer encoding for large responses.  The ``chunk_size``
    field in the request controls the byte size of each chunk
    (default 4096, range 64-65536).

    If the chunked response module is unavailable, falls back to a
    regular JSON response transparently.
    """
    store = _get_store()

    response = store.query(
        query_text=request.query,
        collection=request.collection,
        n_results=request.n_results,
        min_score=request.min_score,
        source_filter=request.source_filter,
        file_type_filter=request.file_type_filter,
        rerank=request.rerank,
        track_citations=request.track_citations,
    )

    results = [
        RetrievalResultResponse(
            content=r.content,
            score=r.score,
            source_file=r.source_file,
            file_type=r.file_type,
            chunk_index=r.chunk_index,
            total_chunks=r.total_chunks,
            parent_doc_id=r.parent_doc_id,
            collection_name=r.collection_name,
            section=r.section,
            page=r.page,
        )
        for r in response.results
    ]

    citations = [
        CitationResponse(
            citation_id=c.citation_id,
            query=c.query,
            collection_name=c.collection_name,
            chunk_id=c.chunk_id,
            parent_doc_id=c.parent_doc_id,
            source_file=c.source_file,
            section=c.section,
            score=c.score,
            timestamp=c.timestamp,
        )
        for c in response.citations
    ]

    payload = QueryResponseSchema(
        query=response.query,
        results=results,
        citations=citations,
        total_results=response.total_results,
    ).model_dump()

    # Use chunked transfer if available
    if CHUNKED_RESPONSE_AVAILABLE and chunked_json_generator is not None:
        chunk_sz = request.chunk_size or DEFAULT_CHUNK_SIZE
        return StreamingResponse(
            chunked_json_generator(payload, chunk_size=chunk_sz),
            media_type="application/json",
            headers={"Transfer-Encoding": "chunked"},
        )

    # Fallback: regular JSON
    from fastapi.responses import JSONResponse
    return JSONResponse(content=payload)


# =========================================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# =========================================================================

@router.get("/documents", response_model=DocumentsListResponse)
def list_documents(
    collection: str | None = Query(None, description="Filter by collection name"),
    search: str | None = Query(None, description="Search by filename (case-insensitive substring match)"),
    file_type: str | None = Query(None, description="Filter by file type (e.g. pdf, docx, markdown)"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """
    List ingested documents with optional filters (S119 enhanced).

    Supports filtering by collection, searching by filename, and
    filtering by file type. Results are paginated.
    """
    store = _get_store()
    # Fetch a larger window to allow client-side filtering
    fetch_limit = limit + offset + 500 if (search or file_type) else limit
    docs = store.list_documents(collection=collection, limit=fetch_limit, offset=0 if (search or file_type) else offset)

    # Apply search filter
    if search:
        search_lower = search.lower()
        docs = [d for d in docs if search_lower in d.source_file.lower()]

    # Apply file_type filter
    if file_type:
        ft_lower = file_type.lower()
        docs = [d for d in docs if d.file_type.lower() == ft_lower]

    # Apply offset/limit after filtering
    if search or file_type:
        total_filtered = len(docs)
        docs = docs[offset:offset + limit]
    else:
        total_filtered = len(docs)

    items = [
        DocumentResponse(
            doc_id=d.doc_id,
            collection_name=d.collection_name,
            source_file=d.source_file,
            file_type=d.file_type,
            chunk_count=d.chunk_count,
            raw_text_length=d.raw_text_length,
            ingested_at=d.ingested_at,
            metadata=d.metadata,
        )
        for d in docs
    ]
    return DocumentsListResponse(documents=items, total=total_filtered)


@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
def delete_document(doc_id: str) -> dict:
    """Delete a document and its chunks from the vector store."""
    store = _get_store()
    deleted = store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return DocumentDeleteResponse(deleted=True, doc_id=doc_id)


# =========================================================================
# RAG INJECTION DEFENSE ENDPOINTS
# =========================================================================

def _get_sanitizer():
    """Get or create the RAG sanitizer singleton."""
    try:
        from opti_oignon.rag_sanitizer import get_rag_sanitizer
        return get_rag_sanitizer()
    except Exception as exc:
        logger.warning("RAG sanitizer unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="RAG injection defense not available")


class SanitizePreviewRequest(BaseModel):
    """Request to preview sanitized RAG chunks."""
    query: str = Field(..., description="User query to retrieve chunks for")
    collection: str = Field("", description="Collection name")
    n_results: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    min_score: float = Field(0.3, ge=0.0, le=1.0, description="Minimum relevance score")


class ChunkApprovalRequest(BaseModel):
    """Request to approve or reject specific chunks."""
    chunk_ids: list[str] = Field(..., description="List of chunk IDs to approve")
    action: str = Field("approve", description="'approve' or 'reject'")


class AuditQueryRequest(BaseModel):
    """Query parameters for audit log."""
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)
    min_score: float | None = Field(None, ge=0.0, le=1.0)
    collection: str | None = None


@router.post("/injection-defense/sanitize-preview")
def sanitize_preview(request: SanitizePreviewRequest) -> dict:
    """Preview retrieved chunks after sanitization pipeline (S144).

    Returns sanitized chunks with injection scores, flags, and blocking
    decisions. If preview mode is enabled, chunks flagged as suspicious
    will require manual approval before injection into the prompt.
    """
    sanitizer = _get_sanitizer()

    # Retrieve chunks via the RAG store
    try:
        store = _get_store()
        raw_results = store.query(
            request.query,
            collection_name=request.collection or None,
            n_results=request.n_results,
            min_score=request.min_score,
        )
    except Exception as exc:
        logger.warning("RAG query failed: %s", exc)
        raw_results = {"results": []}

    # Convert to chunk dicts
    results = raw_results.get("results", [])
    chunk_dicts = []
    for r in results:
        chunk_dicts.append({
            "text": r.get("content", r.get("text", "")),
            "chunk_id": r.get("chunk_id", r.get("id", "")),
            "source": r.get("source", r.get("source_file", "")),
            "collection": request.collection,
        })

    # Run sanitization
    san_result = sanitizer.sanitize_chunks(chunk_dicts, collection=request.collection)
    return san_result.to_dict()


@router.post("/injection-defense/approve")
def approve_chunks(request: ChunkApprovalRequest) -> dict:
    """Approve or reject chunks after preview (S144).

    This is a stateless endpoint — the caller must track chunk IDs
    from the sanitize-preview response and pass approved/rejected
    IDs when building the final prompt.
    """
    if request.action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'")
    return {
        "action": request.action,
        "chunk_ids": request.chunk_ids,
        "count": len(request.chunk_ids),
    }


@router.get("/injection-defense/audit")
def query_audit_log(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    min_score: float | None = Query(None, ge=0.0, le=1.0),
    collection: str | None = Query(None),
) -> dict:
    """Query the injection audit log (S144).

    Returns flagged injection attempts with metadata, pattern matches,
    and chunk text (if configured).
    """
    sanitizer = _get_sanitizer()
    entries = sanitizer.query_audit(
        limit=limit, offset=offset,
        min_score=min_score, collection=collection,
    )
    total = sanitizer.get_audit_log().count()
    return {"entries": entries, "total": total, "limit": limit, "offset": offset}


@router.delete("/injection-defense/audit")
def clear_audit_log() -> dict:
    """Clear all injection audit log entries (S144)."""
    sanitizer = _get_sanitizer()
    deleted = sanitizer.get_audit_log().clear()
    return {"deleted": deleted}


@router.get("/injection-defense/config")
def get_injection_defense_config() -> dict:
    """Get the current injection defense configuration (S144)."""
    sanitizer = _get_sanitizer()
    config = sanitizer.config
    return {
        "enabled": config.get("enabled", False),
        "separation": config.get("separation", {}),
        "scoring": config.get("scoring", {}),
        "trust_levels": config.get("trust_levels", {}),
        "preview": config.get("preview", {}),
        "audit": config.get("audit", {}),
    }
