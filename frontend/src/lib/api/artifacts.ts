/**
 * Typed API functions for artifact endpoints.
 *
 * Handles listing, viewing, downloading, deleting and exporting
 * artifacts associated with conversations.
 */

import { apiGet, apiDelete } from './client';
import type { ArtifactInfo, ArtifactContent, ArtifactExport } from '$lib/types';

const API_BASE = import.meta.env.VITE_API_URL ?? '';

function buildUrl(path: string): string {
	return new URL(`${API_BASE}${path}`, window.location.origin).toString();
}

/** Liste les artifacts d'une conversation. */
export async function listArtifacts(conversationId: string): Promise<ArtifactInfo[]> {
	return apiGet<ArtifactInfo[]>(`/api/conversations/${conversationId}/artifacts`);
}

/** Retrieve metadata + content of an artifact. */
export async function getArtifact(id: string): Promise<ArtifactContent> {
	return apiGet<ArtifactContent>(`/api/artifacts/${id}`);
}

/** Retrieve the raw content of an artifact (text). */
export async function getArtifactContent(id: string): Promise<string> {
	const response = await fetch(buildUrl(`/api/artifacts/${id}/content`), {
		method: 'GET',
		headers: { 'Accept': 'text/plain' }
	});
	if (!response.ok) {
		throw new Error(`Failed to fetch artifact content: ${response.status}`);
	}
	return response.text();
}

/** Delete an artifact. */
export async function deleteArtifact(id: string): Promise<void> {
	await apiDelete<void>(`/api/artifacts/${id}`);
}

/** Telecharge un artifact comme fichier blob. */
export async function downloadArtifact(id: string): Promise<Blob> {
	const response = await fetch(buildUrl(`/api/artifacts/${id}/download`), {
		method: 'GET'
	});
	if (!response.ok) {
		throw new Error(`Failed to download artifact: ${response.status}`);
	}
	return response.blob();
}

/** Export all artifacts from a conversation (JSON array). */
export async function exportArtifacts(conversationId: string): Promise<ArtifactExport[]> {
	return apiGet<ArtifactExport[]>(`/api/conversations/${conversationId}/artifacts/export`);
}
