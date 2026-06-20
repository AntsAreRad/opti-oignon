/**
 * Keyboard shortcuts API client for Opti-Oignon.
 *
 * S153: Endpoints for loading/saving custom keyboard shortcut bindings.
 */

import { apiGet, apiPut } from './client';

// -- Types --

export interface ShortcutBinding {
	action: string;
	key: string;
	ctrl: boolean;
	shift: boolean;
	alt: boolean;
	meta: boolean;
	description: string;
	category: string;
}

export interface BrowserConflict {
	action: string;
	combo: string;
	browser_function: string;
}

export interface KeyboardShortcutsResponse {
	shortcuts: Record<string, ShortcutBinding>;
	custom_overrides: Record<string, Partial<ShortcutBinding>>;
	browser_conflicts: BrowserConflict[];
}

export interface KeyboardShortcutsUpdateResponse {
	success: boolean;
	shortcuts: Record<string, ShortcutBinding>;
	custom_overrides: Record<string, Partial<ShortcutBinding>>;
	browser_conflicts: BrowserConflict[];
	warnings: string[];
}

// -- API calls --

export async function getKeyboardShortcuts(): Promise<KeyboardShortcutsResponse> {
	return apiGet<KeyboardShortcutsResponse>('/api/settings/keyboard_shortcuts');
}

export async function updateKeyboardShortcuts(
	customBindings: Record<string, Partial<ShortcutBinding>>
): Promise<KeyboardShortcutsUpdateResponse> {
	return apiPut<KeyboardShortcutsUpdateResponse>('/api/settings/keyboard_shortcuts', {
		custom_bindings: customBindings
	});
}

export async function resetAllShortcuts(): Promise<KeyboardShortcutsUpdateResponse> {
	return apiPut<KeyboardShortcutsUpdateResponse>('/api/settings/keyboard_shortcuts', {
		custom_bindings: {}
	});
}
