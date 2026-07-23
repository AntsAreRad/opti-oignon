/**
 * RAG v2 Knowledge Base API client.
 *
 * Collection management, document ingestion (file + URL),
 * query with retrieval and citation tracking.
 */

import { apiGet, apiPost, apiDelete, getAccessToken } from './client';
import type {
	RAGCollection,
	RAGCollectionsListResponse,
	RAGCollectionCreateRequest,
	RAGCollectionDeleteResponse,
	RAGDocument,
	RAGDocumentsListResponse,
	RAGDocumentDeleteResponse,
	RAGIngestResponse,
	RAGIngestURLRequest,
	RAGQueryRequest,
	RAGQueryResponse,
	RAGIngestJob,
	RAGIngestJobsListResponse,
	RAGIngestJobDeleteResponse,
	RAGFolderScanRequest,
} from '$lib/types';

// =========================================================================
// COLLECTIONS
// =========================================================================

/** List all collections with document/chunk counts. */
export async function listCollections(): Promise<RAGCollectionsListResponse> {
	return (await apiGet('/api/rag/collections')) as RAGCollectionsListResponse;
}

/** Create a new collection. */
export async function createCollection(
	req: RAGCollectionCreateRequest
): Promise<RAGCollection> {
	return (await apiPost('/api/rag/collections', req)) as RAGCollection;
}

/** Delete a collection and all its data. */
export async function deleteCollection(
	name: string
): Promise<RAGCollectionDeleteResponse> {
	return (await apiDelete(`/api/rag/collections/${encodeURIComponent(name)}`)) as RAGCollectionDeleteResponse;
}

// =========================================================================
// INGESTION
// =========================================================================

/** Upload and ingest a file into a collection. */
export async function ingestFile(
	file: File,
	collection: string = 'default'
): Promise<RAGIngestResponse> {
	const formData = new FormData();
	formData.append('file', file);
	formData.append('collection', collection || 'default');

	// BUG-11: Include auth headers (raw fetch was missing them)
	const headers: Record<string, string> = {
		'Accept': 'application/json',
	};
	const token = getAccessToken();
	if (token) {
		headers['Authorization'] = `Bearer ${token}`;
	}

	const response = await fetch('/api/rag/ingest', {
		method: 'POST',
		headers,
		body: formData,
		credentials: 'include',
	});

	if (!response.ok) {
		let detail = response.statusText;
		try {
			const body = await response.json();
			// FastAPI 422 returns {detail: [{loc, msg, type}, ...]}
			if (Array.isArray(body.detail)) {
				detail = body.detail.map((d: any) => `${d.loc?.join('.')}: ${d.msg}`).join('; ');
			} else {
				detail = body.detail || detail;
			}
		} catch { /* keep statusText */ }
		throw new Error(`Ingestion failed (${response.status}): ${detail}`);
	}

	return response.json() as Promise<RAGIngestResponse>;
}

/** Ingest a web page by URL. */
export async function ingestURL(
	req: RAGIngestURLRequest
): Promise<RAGIngestResponse> {
	return (await apiPost('/api/rag/ingest/url', req)) as RAGIngestResponse;
}

// =========================================================================
// QUERY
// =========================================================================

/** Query the knowledge base with retrieval and citation tracking. */
export async function queryKnowledgeBase(
	req: RAGQueryRequest
): Promise<RAGQueryResponse> {
	return (await apiPost('/api/rag/query', req)) as RAGQueryResponse;
}

// =========================================================================
// DOCUMENTS
// =========================================================================

/** List ingested documents, optionally filtered by collection, search, file type. */
export async function listDocuments(params?: {
	collection?: string;
	search?: string;
	file_type?: string;
	limit?: number;
	offset?: number;
}): Promise<RAGDocumentsListResponse> {
	const query: Record<string, string> = {};
	if (params?.collection) query.collection = params.collection;
	if (params?.search) query.search = params.search;
	if (params?.file_type) query.file_type = params.file_type;
	if (params?.limit !== undefined) query.limit = String(params.limit);
	if (params?.offset !== undefined) query.offset = String(params.offset);
	return (await apiGet('/api/rag/documents', query)) as RAGDocumentsListResponse;
}

/** Delete a document and its chunks. */
export async function deleteDocument(
	docId: string
): Promise<RAGDocumentDeleteResponse> {
	return (await apiDelete(`/api/rag/documents/${encodeURIComponent(docId)}`)) as RAGDocumentDeleteResponse;
}

// =========================================================================
// BATCH INGESTION
// =========================================================================

/** Upload multiple files for batch ingestion. Returns a job (202). */
export async function ingestBatch(
	files: File[],
	collection: string = 'default'
): Promise<RAGIngestJob> {
	const formData = new FormData();
	for (const file of files) {
		formData.append('files', file);
	}
	formData.append('collection', collection || 'default');

	const headers: Record<string, string> = {
		'Accept': 'application/json',
	};
	const token = getAccessToken();
	if (token) {
		headers['Authorization'] = `Bearer ${token}`;
	}

	const response = await fetch('/api/rag/ingest/batch', {
		method: 'POST',
		headers,
		body: formData,
		credentials: 'include',
	});

	if (!response.ok) {
		let detail = response.statusText;
		try {
			const body = await response.json();
			if (Array.isArray(body.detail)) {
				detail = body.detail.map((d: any) => `${d.loc?.join('.')}: ${d.msg}`).join('; ');
			} else {
				detail = body.detail || detail;
			}
		} catch { /* keep statusText */ }
		throw new Error(`Batch ingestion failed (${response.status}): ${detail}`);
	}

	return response.json() as Promise<RAGIngestJob>;
}

/** Scan a local folder and ingest files. Returns a job (202). */
export async function ingestFolder(
	req: RAGFolderScanRequest
): Promise<RAGIngestJob> {
	return (await apiPost('/api/rag/ingest/folder', req)) as RAGIngestJob;
}

/** List batch ingestion jobs, optionally filtered by status. */
export async function listIngestJobs(params?: {
	status?: string;
}): Promise<RAGIngestJobsListResponse> {
	const query: Record<string, string> = {};
	if (params?.status) query.status = params.status;
	return (await apiGet('/api/rag/ingest/jobs', query)) as RAGIngestJobsListResponse;
}

/** Get details of a single ingestion job (with per-file status). */
export async function getIngestJob(
	jobId: string
): Promise<RAGIngestJob> {
	return (await apiGet(`/api/rag/ingest/jobs/${encodeURIComponent(jobId)}`)) as RAGIngestJob;
}

/** Cancel and delete a batch ingestion job. */
export async function deleteIngestJob(
	jobId: string
): Promise<RAGIngestJobDeleteResponse> {
	return (await apiDelete(`/api/rag/ingest/jobs/${encodeURIComponent(jobId)}`)) as RAGIngestJobDeleteResponse;
}
