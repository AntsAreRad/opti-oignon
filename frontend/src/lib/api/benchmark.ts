/**
 * The model roles of the configuration: which model each task type runs on.
 *
 * The benchmarks page runs the quality evaluation, whose client is
 * lib/api/benchmarkV2.ts. The older suite engine's routes (its runs, its
 * suites, its live progress) stay on the server, and the interface no
 * longer calls them; what remains here is the role assignment, which the
 * roles store (lib/stores/modelRoles.ts) reads and writes.
 */

import { apiGet, apiPut } from './client';
import type { ModelRoleInfo } from '$lib/types';

/** The roles and the models installed. */
export function getConfigRoles(): Promise<{ roles: ModelRoleInfo[]; installed_models: string[] }> {
	return apiGet('/api/benchmark/models/config/roles');
}

/** Saves one role's primary, fast and quality models. */
export function updateRoleAssignment(
	role: string,
	assignment: { primary?: string; fast?: string; quality?: string }
): Promise<{ status: string }> {
	return apiPut(`/api/benchmark/models/config/roles/${role}`, assignment);
}
