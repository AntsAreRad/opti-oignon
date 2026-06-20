/**
 * Typed API functions for settings endpoints.
 *
 * Provides get/set/reload operations for application configuration.
 */

import { apiGet, apiPut, apiPost } from './client';
import type { SettingsResponse, SettingValue } from '$lib/types';

/** Retrieve global configuration (models, presets, user). */
export async function getSettings(): Promise<SettingsResponse> {
	return apiGet<SettingsResponse>('/api/settings');
}

/** Retrieve the value of an individual parameter. */
export async function getSetting(key: string): Promise<SettingValue> {
	return apiGet<SettingValue>(`/api/settings/${encodeURIComponent(key)}`);
}

/** Met a jour la valeur d'un parametre individuel. */
export async function updateSetting(key: string, value: unknown): Promise<SettingValue> {
	return apiPut<SettingValue>(`/api/settings/${encodeURIComponent(key)}`, { value });
}

/** Recharge la configuration depuis le disque. */
export async function reloadSettings(): Promise<{ status: string }> {
	return apiPost<{ status: string }>('/api/settings/reload');
}
