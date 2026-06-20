/**
 * Theme API client for Opti-Oignon.
 *
 * S152: Endpoints for loading/saving user theme configuration,
 * listing presets (built-in + custom), CRUD custom presets,
 * and import/export.
 */

import { apiGet, apiPost, apiDelete } from './client';

// -- Types --

export interface ThemePreset {
	id: string;
	name: string;
	description: string;
	accent_hue: number;
	accent_saturation: number;
	secondary_hue: number;
	secondary_saturation: number;
	accent_lightness_offset: number;
	secondary_lightness_offset: number;
	accent_warmth: number;
	secondary_warmth: number;
	builtin: boolean;
}

export interface ThemeConfig {
	accent_hue: number;
	accent_saturation: number;
	secondary_hue: number;
	secondary_saturation: number;
	accent_lightness_offset: number;
	secondary_lightness_offset: number;
	accent_warmth: number;
	secondary_warmth: number;
	mode: string;
	preset_id: string | null;
	variables: Record<string, string>;
}

export interface ThemeSaveRequest {
	accent_hue: number;
	accent_saturation?: number;
	secondary_hue?: number;
	secondary_saturation?: number;
	accent_lightness_offset?: number;
	secondary_lightness_offset?: number;
	accent_warmth?: number;
	secondary_warmth?: number;
	mode?: string;
	preset_id?: string | null;
}

export interface CustomPresetCreateRequest {
	name: string;
	description?: string;
	accent_hue: number;
	accent_saturation?: number;
	secondary_hue?: number;
	secondary_saturation?: number;
	accent_lightness_offset?: number;
	secondary_lightness_offset?: number;
	accent_warmth?: number;
	secondary_warmth?: number;
}

// -- Theme config --

export async function getThemeConfig(): Promise<ThemeConfig> {
	return apiGet<ThemeConfig>('/api/settings/theme');
}

export async function saveThemeConfig(req: ThemeSaveRequest): Promise<ThemeConfig> {
	return apiPost<ThemeConfig>('/api/settings/theme', req);
}

// -- Presets --

export async function getThemePresets(): Promise<ThemePreset[]> {
	const resp = await apiGet<{ presets: ThemePreset[] }>('/api/settings/theme/presets');
	return resp.presets;
}

export async function createCustomPreset(req: CustomPresetCreateRequest): Promise<ThemePreset> {
	return apiPost<ThemePreset>('/api/settings/theme/presets/custom', req);
}

export async function deleteCustomPreset(presetId: string): Promise<void> {
	await apiDelete<{ deleted: string }>(`/api/settings/theme/presets/custom/${presetId}`);
}

// -- Import / Export --

export async function exportCustomPresets(): Promise<string> {
	const resp = await apiGet<{ presets_json: string }>('/api/settings/theme/presets/export');
	return resp.presets_json;
}

export async function importCustomPresets(presets: unknown[]): Promise<ThemePreset[]> {
	const resp = await apiPost<{ presets: ThemePreset[] }>(
		'/api/settings/theme/presets/import',
		{ presets }
	);
	return resp.presets;
}
