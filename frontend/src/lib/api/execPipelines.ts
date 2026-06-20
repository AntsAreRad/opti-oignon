/**
 * API client for execution pipeline management (S53).
 */

import { apiGet, apiPost, apiPut, apiDelete } from './client';
import type {
	ExecPipelineInfo,
	ExecPipelineCreate,
	ExecPipelineUpdate,
	ExecStepTypeInfo,
} from '$lib/types';

const BASE = '/api/execution-pipelines';

export async function listExecPipelines(opts?: {
	builtin_only?: boolean;
	custom_only?: boolean;
}): Promise<ExecPipelineInfo[]> {
	const params = new URLSearchParams();
	if (opts?.builtin_only) params.set('builtin_only', 'true');
	if (opts?.custom_only) params.set('custom_only', 'true');
	const qs = params.toString();
	return apiGet<ExecPipelineInfo[]>(BASE + (qs ? '?' + qs : ''));
}

export async function getExecPipeline(id: string): Promise<ExecPipelineInfo> {
	return apiGet<ExecPipelineInfo>(BASE + '/' + encodeURIComponent(id));
}

export async function listStepTypes(): Promise<ExecStepTypeInfo[]> {
	return apiGet<ExecStepTypeInfo[]>(BASE + '/step-types');
}

export async function createExecPipeline(config: ExecPipelineCreate): Promise<ExecPipelineInfo> {
	return apiPost<ExecPipelineInfo>(BASE, config);
}

export async function updateExecPipeline(
	id: string,
	config: ExecPipelineUpdate
): Promise<ExecPipelineInfo> {
	return apiPut<ExecPipelineInfo>(BASE + '/' + encodeURIComponent(id), config);
}

export async function deleteExecPipeline(id: string): Promise<void> {
	return apiDelete<void>(BASE + '/' + encodeURIComponent(id));
}

export async function duplicateExecPipeline(
	id: string,
	newId: string
): Promise<ExecPipelineInfo> {
	return apiPost<ExecPipelineInfo>(BASE + '/' + encodeURIComponent(id) + '/duplicate', {
		new_id: newId
	});
}
