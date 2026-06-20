/**
 * Typed API functions for Project endpoints (S57 + S58 + S59).
 *
 * Covers project CRUD, file management, output management,
 * conversation linking, and project context (indexation, triggers, RAG).
 */

import { apiGet, apiPost, apiPut, apiDelete, ApiError } from './client';
import type {
	ProjectInfo,
	ProjectDetailInfo,
	ProjectFileInfo,
	ProjectOutputInfo,
	ProjectContextPreview,
	ProjectFileSummary,
} from '$lib/types';

const API_BASE = import.meta.env.VITE_API_URL ?? '';

function buildUrl(path: string): string {
	return new URL(`${API_BASE}${path}`, window.location.origin).toString();
}

// =========================================================================
// PROJECT CRUD
// =========================================================================

/** List all projects. */
export async function listProjects(): Promise<ProjectInfo[]> {
	return apiGet<ProjectInfo[]>('/api/projects');
}

/** Get full project detail with files, outputs, conversations, stats. */
export async function getProject(projectId: string): Promise<ProjectDetailInfo> {
	return apiGet<ProjectDetailInfo>(`/api/projects/${projectId}`);
}

/** Create a new project. */
export async function createProject(data: {
	name: string;
	description?: string;
	system_instructions?: string;
	settings?: Record<string, unknown>;
}): Promise<ProjectInfo> {
	return apiPost<ProjectInfo>('/api/projects', data);
}

/** Update an existing project. */
export async function updateProject(
	projectId: string,
	data: {
		name?: string;
		description?: string;
		system_instructions?: string;
		settings?: Record<string, unknown>;
	}
): Promise<ProjectInfo> {
	return apiPut<ProjectInfo>(`/api/projects/${projectId}`, data);
}

/** Delete a project and all associated data. */
export async function deleteProject(projectId: string): Promise<{ status: string; deleted: string }> {
	return apiDelete<{ status: string; deleted: string }>(`/api/projects/${projectId}`);
}

// =========================================================================
// FILE MANAGEMENT
// =========================================================================

/** Upload a file to a project via multipart/form-data. */
export async function uploadProjectFile(projectId: string, file: File): Promise<ProjectFileInfo> {
	const formData = new FormData();
	formData.append('file', file);

	try {
		const response = await fetch(buildUrl(`/api/projects/${projectId}/files`), {
			method: 'POST',
			body: formData,
		});

		if (!response.ok) {
			let detail = response.statusText;
			try {
				const body = await response.json();
				detail = body.detail || detail;
			} catch {
				// No JSON body
			}
			throw new ApiError(response.status, `Upload failed: ${detail}`, detail);
		}

		return (await response.json()) as ProjectFileInfo;
	} catch (err) {
		if (err instanceof ApiError) throw err;
		throw new ApiError(0, 'Upload connection failed', 'Unable to reach the API server');
	}
}

/** List all files for a project. */
export async function listProjectFiles(projectId: string): Promise<ProjectFileInfo[]> {
	return apiGet<ProjectFileInfo[]>(`/api/projects/${projectId}/files`);
}

/** Get a specific file's metadata. */
export async function getProjectFile(projectId: string, fileId: string): Promise<ProjectFileInfo> {
	return apiGet<ProjectFileInfo>(`/api/projects/${projectId}/files/${fileId}`);
}

/** Delete a project file. */
export async function deleteProjectFile(
	projectId: string,
	fileId: string
): Promise<{ status: string; deleted: string }> {
	return apiDelete<{ status: string; deleted: string }>(`/api/projects/${projectId}/files/${fileId}`);
}

// =========================================================================
// OUTPUT MANAGEMENT
// =========================================================================

/** Create a project output entry. */
export async function createProjectOutput(
	projectId: string,
	data: {
		filename: string;
		output_type?: string;
		description?: string;
		source_conversation_id?: string;
	}
): Promise<ProjectOutputInfo> {
	return apiPost<ProjectOutputInfo>(`/api/projects/${projectId}/outputs`, data);
}

/** List all outputs for a project. */
export async function listProjectOutputs(projectId: string): Promise<ProjectOutputInfo[]> {
	return apiGet<ProjectOutputInfo[]>(`/api/projects/${projectId}/outputs`);
}

/** Delete a project output. */
export async function deleteProjectOutput(
	projectId: string,
	outputId: string
): Promise<{ status: string; deleted: string }> {
	return apiDelete<{ status: string; deleted: string }>(`/api/projects/${projectId}/outputs/${outputId}`);
}

// =========================================================================
// CONVERSATION LINKING
// =========================================================================

/** Link a conversation to a project. */
export async function linkConversation(
	projectId: string,
	conversationId: string
): Promise<{ status: string; project_id: string; conversation_id: string }> {
	return apiPost<{ status: string; project_id: string; conversation_id: string }>(
		`/api/projects/${projectId}/conversations/${conversationId}`
	);
}

/** Unlink a conversation from a project. */
export async function unlinkConversation(
	projectId: string,
	conversationId: string
): Promise<{ status: string }> {
	return apiDelete<{ status: string }>(
		`/api/projects/${projectId}/conversations/${conversationId}`
	);
}

/** List all conversations linked to a project. */
export async function listProjectConversations(
	projectId: string
): Promise<{ conversation_id: string; linked_at: string }[]> {
	return apiGet<{ conversation_id: string; linked_at: string }[]>(
		`/api/projects/${projectId}/conversations`
	);
}

// =========================================================================
// S58: CONTEXT (INDEXATION + TRIGGERS + RAG)
// =========================================================================

/** Index a single file into ChromaDB. */
export async function indexProjectFile(
	projectId: string,
	fileId: string
): Promise<{ status: string; chunks: number; summary: string; key_terms: string[] }> {
	return apiPost<{ status: string; chunks: number; summary: string; key_terms: string[] }>(
		`/api/projects/${projectId}/files/${fileId}/index`
	);
}

/** Reindex all project files. */
export async function reindexProject(
	projectId: string
): Promise<{ status: string; indexed: number; failed: number; results: Record<string, unknown>[] }> {
	return apiPost<{ status: string; indexed: number; failed: number; results: Record<string, unknown>[] }>(
		`/api/projects/${projectId}/reindex`
	);
}

/** Preview context injection for a query. */
export async function getContextPreview(
	projectId: string,
	query: string
): Promise<ProjectContextPreview> {
	return apiGet<ProjectContextPreview>(`/api/projects/${projectId}/context`, { query });
}

/** Get file summary and key terms. */
export async function getFileSummary(
	projectId: string,
	fileId: string
): Promise<ProjectFileSummary> {
	return apiGet<ProjectFileSummary>(`/api/projects/${projectId}/files/${fileId}/summary`);
}
