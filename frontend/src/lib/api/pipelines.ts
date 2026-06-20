/**
 * API client for pipeline management endpoints.
 *
 * Fonctions CRUD pour les pipelines multi-agents: lister, creer,
 * modifier, supprimer, dupliquer, exporter.
 */

import { apiGet, apiPost, apiPut, apiDelete } from './client';
import type {
	PipelineInfo,
	PipelineCreate,
	PipelineUpdate,
	PipelineStats,
	PipelineExportData
} from '$lib/types';

/**
 * Liste tous les pipelines.
 * GET /api/pipelines
 */
export async function listPipelines(opts?: {
	builtin_only?: boolean;
	custom_only?: boolean;
}): Promise<PipelineInfo[]> {
	const params = new URLSearchParams();
	if (opts?.builtin_only) params.set('builtin_only', 'true');
	if (opts?.custom_only) params.set('custom_only', 'true');
	const qs = params.toString();
	return apiGet<PipelineInfo[]>(`/api/pipelines${qs ? `?${qs}` : ''}`);
}

/**
 * Retrieve a pipeline by ID.
 * GET /api/pipelines/{id}
 */
export async function getPipeline(id: string): Promise<PipelineInfo> {
	return apiGet<PipelineInfo>(`/api/pipelines/${encodeURIComponent(id)}`);
}

/**
 * Create a new pipeline.
 * POST /api/pipelines
 */
export async function createPipeline(config: PipelineCreate): Promise<PipelineInfo> {
	return apiPost<PipelineInfo>('/api/pipelines', config);
}

/**
 * Met a jour un pipeline existant.
 * PUT /api/pipelines/{id}
 */
export async function updatePipeline(id: string, config: PipelineUpdate): Promise<PipelineInfo> {
	return apiPut<PipelineInfo>(`/api/pipelines/${encodeURIComponent(id)}`, config);
}

/**
 * Delete a pipeline.
 * DELETE /api/pipelines/{id}
 */
export async function deletePipeline(id: string): Promise<void> {
	return apiDelete<void>(`/api/pipelines/${encodeURIComponent(id)}`);
}

/**
 * Duplicate a pipeline.
 * POST /api/pipelines/{id}/duplicate
 */
export async function duplicatePipeline(id: string, newId: string): Promise<PipelineInfo> {
	return apiPost<PipelineInfo>(`/api/pipelines/${encodeURIComponent(id)}/duplicate`, {
		new_id: newId
	});
}

/**
 * Statistiques des pipelines.
 * GET /api/pipelines/stats
 */
export async function getPipelineStats(): Promise<PipelineStats> {
	return apiGet<PipelineStats>('/api/pipelines/stats');
}

/**
 * Export pipelines as YAML.
 * POST /api/pipelines/export
 */
export async function exportPipelines(customOnly: boolean = false): Promise<PipelineExportData> {
	return apiPost<PipelineExportData>('/api/pipelines/export', { custom_only: customOnly });
}

/**
 * Liste les agents disponibles pour les pipelines.
 * GET /api/pipelines/agents
 */
export async function listAgents(): Promise<string[]> {
	return apiGet<string[]>('/api/pipelines/agents');
}

/**
 * Liste les templates de pipeline disponibles.
 * GET /api/pipelines/templates
 */
export async function listTemplates(): Promise<string[]> {
	return apiGet<string[]>('/api/pipelines/templates');
}
