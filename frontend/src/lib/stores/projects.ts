/**
 * Svelte stores for project state management (S59).
 *
 * Provides reactive stores and action functions for project CRUD,
 * file management, conversation linking, and context operations.
 * Follows the same pattern as conversations.ts.
 */

import { writable, derived, get } from 'svelte/store';
import type {
	ProjectInfo,
	ProjectDetailInfo,
	ProjectFileInfo,
	ProjectOutputInfo,
	ProjectContextPreview,
	ProjectFileSummary,
} from '$lib/types';
import * as api from '$lib/api/projects';

// =========================================================================
// STORES
// =========================================================================

/** List of all projects. */
export const projects = writable<ProjectInfo[]>([]);

/** Currently selected project ID. */
export const activeProjectId = writable<string | null>(null);

/** Full detail of the active project (loaded on demand). */
export const activeProjectDetail = writable<ProjectDetailInfo | null>(null);

/** Loading state for project list. */
export const projectsLoading = writable<boolean>(false);

/** Loading state for project detail. */
export const detailLoading = writable<boolean>(false);

/** Current error message. */
export const projectError = writable<string | null>(null);

/** Project linked to the current conversation (set externally). */
export const conversationProjectId = writable<string | null>(null);

/** Derived: active project summary from the list. */
export const activeProject = derived(
	[projects, activeProjectId],
	([$projects, $activeId]) => {
		if (!$activeId) return null;
		return $projects.find((p) => p.id === $activeId) ?? null;
	}
);

/** Derived: project linked to the current conversation. */
export const conversationProject = derived(
	[projects, conversationProjectId],
	([$projects, $projId]) => {
		if (!$projId) return null;
		return $projects.find((p) => p.id === $projId) ?? null;
	}
);

// =========================================================================
// PROJECT CRUD ACTIONS
// =========================================================================

/** Load the list of all projects from the API. */
export async function loadProjects(): Promise<void> {
	projectsLoading.set(true);
	projectError.set(null);
	try {
		const list = await api.listProjects();
		projects.set(list);
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to load projects';
		projectError.set(msg);
		projects.set([]);
	} finally {
		projectsLoading.set(false);
	}
}

/** Select a project and load its full detail. */
export async function selectProject(projectId: string): Promise<void> {
	activeProjectId.set(projectId);
	detailLoading.set(true);
	projectError.set(null);
	try {
		const detail = await api.getProject(projectId);
		activeProjectDetail.set(detail);
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to load project';
		projectError.set(msg);
		activeProjectDetail.set(null);
	} finally {
		detailLoading.set(false);
	}
}

/** Create a new project. Returns the created project. */
export async function createProject(data: {
	name: string;
	description?: string;
	system_instructions?: string;
	settings?: Record<string, unknown>;
}): Promise<ProjectInfo> {
	projectError.set(null);
	try {
		const project = await api.createProject(data);
		projects.update((list) => [project, ...list]);
		return project;
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to create project';
		projectError.set(msg);
		throw err;
	}
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
): Promise<void> {
	projectError.set(null);
	try {
		const updated = await api.updateProject(projectId, data);
		// Update in list
		projects.update((list) =>
			list.map((p) => (p.id === projectId ? { ...p, ...updated } : p))
		);
		// Update detail if viewing
		const detail = get(activeProjectDetail);
		if (detail && detail.id === projectId) {
			activeProjectDetail.update((d) =>
				d ? { ...d, ...updated } : d
			);
		}
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to update project';
		projectError.set(msg);
		throw err;
	}
}

/** Delete a project. */
export async function deleteProject(projectId: string): Promise<void> {
	projectError.set(null);
	try {
		await api.deleteProject(projectId);
		projects.update((list) => list.filter((p) => p.id !== projectId));
		// Clear active if it was the deleted one
		const currentId = get(activeProjectId);
		if (currentId === projectId) {
			activeProjectId.set(null);
			activeProjectDetail.set(null);
		}
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to delete project';
		projectError.set(msg);
		throw err;
	}
}

// =========================================================================
// FILE ACTIONS
// =========================================================================

/** Upload a file to the active project. Returns the file info. */
export async function uploadFile(projectId: string, file: File): Promise<ProjectFileInfo> {
	projectError.set(null);
	try {
		const pf = await api.uploadProjectFile(projectId, file);
		// Update detail if viewing this project
		const detail = get(activeProjectDetail);
		if (detail && detail.id === projectId) {
			activeProjectDetail.update((d) =>
				d ? { ...d, files: [...d.files, pf] } : d
			);
		}
		return pf;
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to upload file';
		projectError.set(msg);
		throw err;
	}
}

/** Delete a file from a project. */
export async function deleteFile(projectId: string, fileId: string): Promise<void> {
	projectError.set(null);
	try {
		await api.deleteProjectFile(projectId, fileId);
		const detail = get(activeProjectDetail);
		if (detail && detail.id === projectId) {
			activeProjectDetail.update((d) =>
				d ? { ...d, files: d.files.filter((f) => f.id !== fileId) } : d
			);
		}
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to delete file';
		projectError.set(msg);
		throw err;
	}
}

/** Index a single file. Updates its indexed status in the store. */
export async function indexFile(
	projectId: string,
	fileId: string
): Promise<{ chunks: number; summary: string; key_terms: string[] }> {
	projectError.set(null);
	try {
		const result = await api.indexProjectFile(projectId, fileId);
		// Update file in detail
		const detail = get(activeProjectDetail);
		if (detail && detail.id === projectId) {
			activeProjectDetail.update((d) =>
				d
					? {
							...d,
							files: d.files.map((f) =>
								f.id === fileId
									? {
											...f,
											indexed: true,
											chunk_count: result.chunks,
											summary: result.summary,
											key_terms: result.key_terms,
										}
									: f
							),
						}
					: d
			);
		}
		return result;
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to index file';
		projectError.set(msg);
		throw err;
	}
}

/** Reindex all files in a project. */
export async function reindexAll(projectId: string): Promise<{ indexed: number; failed: number }> {
	projectError.set(null);
	try {
		const result = await api.reindexProject(projectId);
		// Reload full detail to get updated index status
		await selectProject(projectId);
		return { indexed: result.indexed, failed: result.failed };
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to reindex';
		projectError.set(msg);
		throw err;
	}
}

/** Get file summary and key terms. */
export async function getFileSummary(
	projectId: string,
	fileId: string
): Promise<ProjectFileSummary> {
	return api.getFileSummary(projectId, fileId);
}

// =========================================================================
// CONVERSATION LINKING ACTIONS
// =========================================================================

/** Link a conversation to a project. */
export async function linkConversation(
	projectId: string,
	conversationId: string
): Promise<void> {
	projectError.set(null);
	try {
		await api.linkConversation(projectId, conversationId);
		conversationProjectId.set(projectId);
		// Update detail if viewing
		const detail = get(activeProjectDetail);
		if (detail && detail.id === projectId) {
			activeProjectDetail.update((d) =>
				d
					? {
							...d,
							conversations: [
								...d.conversations,
								{ conversation_id: conversationId, linked_at: new Date().toISOString() },
							],
						}
					: d
			);
		}
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to link conversation';
		projectError.set(msg);
		throw err;
	}
}

/** Unlink a conversation from a project. */
export async function unlinkConversation(
	projectId: string,
	conversationId: string
): Promise<void> {
	projectError.set(null);
	try {
		await api.unlinkConversation(projectId, conversationId);
		conversationProjectId.set(null);
		// Update detail if viewing
		const detail = get(activeProjectDetail);
		if (detail && detail.id === projectId) {
			activeProjectDetail.update((d) =>
				d
					? {
							...d,
							conversations: d.conversations.filter(
								(c) => c.conversation_id !== conversationId
							),
						}
					: d
			);
		}
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to unlink conversation';
		projectError.set(msg);
		throw err;
	}
}

// =========================================================================
// CONTEXT PREVIEW
// =========================================================================

/** Get context injection preview for a query. */
export async function getContextPreview(
	projectId: string,
	query: string
): Promise<ProjectContextPreview> {
	return api.getContextPreview(projectId, query);
}

// =========================================================================
// HELPERS
// =========================================================================

/** Clear the active project selection. */
export function clearActiveProject(): void {
	activeProjectId.set(null);
	activeProjectDetail.set(null);
}

/** Format file size to a human-readable string. */
export function formatFileSize(bytes: number): string {
	if (bytes < 1024) return `${bytes} B`;
	if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
	return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
