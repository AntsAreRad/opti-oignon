/**
 * API client for conversation export endpoints.
 *
 * Permet d'exporter une conversation en markdown, JSON ou HTML.
 * GET /api/conversations/{id}/export?format=markdown|json|html
 */

import { apiGet } from './client';
import type { ExportResponse } from '$lib/types';

export type ExportFormat = 'markdown' | 'json' | 'html';

/**
 * Export a conversation in the specified format.
 */
export async function exportConversation(
	id: string,
	format: ExportFormat
): Promise<ExportResponse> {
	return apiGet<ExportResponse>(`/api/conversations/${encodeURIComponent(id)}/export`, {
		format
	});
}

/**
 * Download the exported content as a file.
 * Create a Blob, generate a temporary URL and trigger download.
 */
export function downloadExport(content: string, filename: string, format: ExportFormat): void {
	const mimeTypes: Record<ExportFormat, string> = {
		markdown: 'text/markdown',
		json: 'application/json',
		html: 'text/html'
	};

	const blob = new Blob([content], { type: mimeTypes[format] });
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	a.click();
	URL.revokeObjectURL(url);
}
