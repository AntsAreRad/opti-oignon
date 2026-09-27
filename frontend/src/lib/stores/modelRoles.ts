/**
 * The model roles: which model each task type runs on (primary, fast and
 * quality), the models installed, how the last read went, and each role's
 * last failed save.
 *
 * Model assignment (Workshop > Models and inference) renders from this
 * store, so each of its states can be drawn: a read under way, a read that
 * failed and why, a read that found no role, and a save the server refused,
 * shown under its own role. A failed read is never shown as an empty list.
 */

import { writable } from 'svelte/store';
import type { ModelRoleInfo } from '$lib/types';
import { getConfigRoles, updateRoleAssignment } from '$lib/api/benchmark';
import { parseApiError } from '$lib/api/errorHandler';

/** How the last read of the roles went. */
export interface RolesRead {
	state: 'idle' | 'loading' | 'ok' | 'error';
	/** Why the read failed. */
	reason?: string;
}

/** The roles, as last read. */
export const roles = writable<ModelRoleInfo[]>([]);
/** The models installed, as the last read of the roles gave them. */
export const installedModels = writable<string[]>([]);
/** How the last read went. */
export const rolesRead = writable<RolesRead>({ state: 'idle' });
/** Each role's last failed save, by role; a successful save clears it. */
export const saveErrors = writable<Record<string, string>>({});

/**
 * Reads the roles and the models installed. A read that follows a save
 * (`quietly`) keeps the list on screen while it runs.
 */
export async function loadRoles(quietly = false): Promise<void> {
	if (!quietly) rolesRead.set({ state: 'loading' });
	try {
		const data = await getConfigRoles();
		roles.set(data.roles ?? []);
		installedModels.set(data.installed_models ?? []);
		rolesRead.set({ state: 'ok' });
	} catch (error) {
		rolesRead.set({ state: 'error', reason: parseApiError(error).message });
	}
}

/** Forgets a role's last failed save. */
export function forgetSaveError(role: string): void {
	saveErrors.update((errors) => {
		if (!(role in errors)) return errors;
		const next = { ...errors };
		delete next[role];
		return next;
	});
}

/**
 * Saves one role's models. Says whether it was saved; a failure is kept
 * under its role, never thrown.
 */
export async function saveRole(
	role: string,
	assignment: { primary?: string; fast?: string; quality?: string }
): Promise<boolean> {
	try {
		await updateRoleAssignment(role, assignment);
		forgetSaveError(role);
		await loadRoles(true);
		return true;
	} catch (error) {
		const reason = parseApiError(error).message;
		saveErrors.update((errors) => ({ ...errors, [role]: reason }));
		return false;
	}
}
