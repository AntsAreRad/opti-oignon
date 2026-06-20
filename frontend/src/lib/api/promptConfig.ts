/**
 * API client for Prompt Optimization (S65).
 *
 * Provides typed access to token budget, template, and config endpoints.
 */

import { apiGet, apiPut, apiPost, apiDelete } from './client';

// -- Types --

export interface TokenBudget {
	model: string;
	total_window: number;
	system_tokens: number;
	project_tokens: number;
	history_tokens: number;
	user_tokens: number;
	reserve_tokens: number;
	total_input_tokens: number;
	total_allocated: number;
	utilization: number;
}

export interface PromptTemplate {
	task_type: string;
	system_prompt: string;
	temperature_override: number | null;
	stop_sequences: string[];
	source: string;
}

export interface TemplateSummary {
	task_type: string;
	has_temperature_override: boolean;
	temperature_override: number | null;
	source: string;
	prompt_length: number;
}

export interface CacheStats {
	entries: number;
	max_entries: number;
	ttl_seconds: number;
	models: string[];
}

export interface PromptConfig {
	enabled: boolean;
	budget: Record<string, unknown>;
	templates: Record<string, unknown>;
}

// -- Budget endpoints --

export function getBudget(
	model: string,
	projectActive = false,
	conversationLength = 0
): Promise<TokenBudget> {
	const params: Record<string, string> = {
		project_active: String(projectActive),
		conversation_length: String(conversationLength)
	};
	return apiGet<TokenBudget>(`/api/prompt/budget/${model}`, params);
}

export function getContextWindow(model: string): Promise<{ model: string; context_window: number }> {
	return apiGet(`/api/prompt/budget/window/${model}`);
}

export function getCacheStats(): Promise<CacheStats> {
	return apiGet<CacheStats>('/api/prompt/budget/cache/stats');
}

export function clearCache(): Promise<{ cleared: number }> {
	return apiPost('/api/prompt/budget/cache/clear');
}

// -- Template endpoints --

export function listTemplates(): Promise<TemplateSummary[]> {
	return apiGet<TemplateSummary[]>('/api/prompt/templates');
}

export function getTemplate(taskType: string, projectId?: string): Promise<PromptTemplate> {
	const params: Record<string, string> = {};
	if (projectId) params.project_id = projectId;
	return apiGet<PromptTemplate>(`/api/prompt/templates/${taskType}`, params);
}

export function setTemplateOverride(
	taskType: string,
	body: { system_prompt: string; temperature_override?: number | null; stop_sequences?: string[] }
): Promise<PromptTemplate> {
	return apiPut<PromptTemplate>(`/api/prompt/templates/${taskType}`, body);
}

export function clearTemplateOverride(taskType: string): Promise<{ cleared: string }> {
	return apiDelete(`/api/prompt/templates/${taskType}/override`);
}

export function clearAllOverrides(): Promise<{ cleared: number }> {
	return apiDelete('/api/prompt/templates/overrides/all');
}

// -- Config endpoints --

export function getPromptConfig(): Promise<PromptConfig> {
	return apiGet<PromptConfig>('/api/prompt/config');
}

export function reloadPromptConfig(): Promise<{ status: string }> {
	return apiPost('/api/prompt/config/reload');
}
