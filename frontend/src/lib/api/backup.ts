/**
 * API client for Backup / Restore endpoints --.
 *
 * Covers:
 *   GET  /api/backup/sections  -- list available sections
 *   GET  /api/backup/export    -- download backup JSON
 *   POST /api/backup/preview   -- preview import diff
 *   POST /api/backup/import    -- apply backup
 */

import { apiGet, apiPost } from './client';
import type {
	BackupSectionsResponse,
	BackupData,
	BackupPreviewResponse,
	BackupImportResponse,
} from '../types';

/** List available backup sections with item counts. */
export async function listBackupSections(): Promise<BackupSectionsResponse> {
	return apiGet<BackupSectionsResponse>('/api/backup/sections');
}

/**
 * Export a backup as JSON.
 *
 * @param sections - Optional comma-separated section names. Omit for all.
 * @returns The backup data object.
 */
export async function exportBackup(sections?: string): Promise<BackupData> {
	const params: Record<string, string> = {};
	if (sections) {
		params.sections = sections;
	}
	return apiGet<BackupData>('/api/backup/export', params);
}

/**
 * Trigger backup download as a .oo-backup.json file.
 *
 * @param sections - Optional comma-separated section names.
 */
export async function downloadBackup(sections?: string): Promise<void> {
	const data = await exportBackup(sections);
	const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	const ts = new Date().toISOString().slice(0, 10);
	a.download = `opti-oignon-backup-${ts}.oo-backup.json`;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	URL.revokeObjectURL(url);
}

/**
 * Preview what an import would change without applying.
 *
 * @param backup - The backup JSON object.
 * @param strategy - 'merge' or 'replace'.
 */
export async function previewImport(
	backup: BackupData,
	strategy: string = 'merge',
): Promise<BackupPreviewResponse> {
	return apiPost<BackupPreviewResponse>('/api/backup/preview', { backup, strategy });
}

/**
 * Import a backup file with the given strategy.
 *
 * @param backup - The backup JSON object.
 * @param strategy - 'merge' or 'replace'.
 */
export async function importBackup(
	backup: BackupData,
	strategy: string = 'merge',
): Promise<BackupImportResponse> {
	return apiPost<BackupImportResponse>('/api/backup/import', { backup, strategy });
}
