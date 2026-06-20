/**
 * Plugins API client (S101).
 *
 * List, install, enable, disable, uninstall plugins.
 * Get/update plugin configuration.
 */

import { apiGet, apiPost, apiPut, apiDelete } from './client';
import type {
	PluginInfo,
	PluginListResponse,
	PluginInstallRequest,
	PluginInstallResponse,
	PluginStateChangeResponse,
	PluginUninstallResponse,
	PluginConfigResponse,
	PluginUpdateConfigResponse,
} from '$lib/types';

const BASE = '/api/plugins';

/** List all installed plugins, optionally filtered by state. */
export async function listPlugins(
	state?: 'installed' | 'enabled' | 'disabled'
): Promise<PluginListResponse> {
	const params: Record<string, string> = {};
	if (state) params.state = state;
	return (await apiGet(`${BASE}`, params)) as PluginListResponse;
}

/** Install a plugin from a directory path. */
export async function installPlugin(
	sourceDir: string,
	autoEnable: boolean = false
): Promise<PluginInstallResponse> {
	return (await apiPost(`${BASE}/install`, {
		source_dir: sourceDir,
		auto_enable: autoEnable,
	})) as PluginInstallResponse;
}

/** Enable an installed plugin. */
export async function enablePlugin(name: string): Promise<PluginStateChangeResponse> {
	return (await apiPost(`${BASE}/${encodeURIComponent(name)}/enable`)) as PluginStateChangeResponse;
}

/** Disable an enabled plugin. */
export async function disablePlugin(name: string): Promise<PluginStateChangeResponse> {
	return (await apiPost(`${BASE}/${encodeURIComponent(name)}/disable`)) as PluginStateChangeResponse;
}

/** Uninstall a plugin completely. */
export async function uninstallPlugin(name: string): Promise<PluginUninstallResponse> {
	return (await apiDelete(`${BASE}/${encodeURIComponent(name)}`)) as PluginUninstallResponse;
}

/** Get a plugin's current configuration and schema. */
export async function getPluginConfig(name: string): Promise<PluginConfigResponse> {
	return (await apiGet(`${BASE}/${encodeURIComponent(name)}/config`)) as PluginConfigResponse;
}

/** Update a plugin's configuration. */
export async function updatePluginConfig(
	name: string,
	config: Record<string, unknown>
): Promise<PluginUpdateConfigResponse> {
	return (await apiPut(`${BASE}/${encodeURIComponent(name)}/config`, {
		config,
	})) as PluginUpdateConfigResponse;
}
