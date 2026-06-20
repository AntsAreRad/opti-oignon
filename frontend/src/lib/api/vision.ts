/**
 * Vision configuration API client (S94).
 *
 * Manages vision model selection, auto-detection, and config persistence.
 */

import { apiGet, apiPut, apiPost } from './client';

export interface VisionConfig {
	vision_model: string;
	effective_model: string | null;
	detection_strategy: string;
	auto_detect_patterns: string[];
	vision_families: string[];
	known_vision_models: string[];
	describe_prompt: string;
	available_vision_models: string[];
}

export interface VisionConfigUpdate {
	vision_model?: string;
	describe_prompt?: string;
	known_vision_models?: string[];
}

export interface VisionModelInfo {
	name: string;
	is_selected: boolean;
	detection_method: string;
}

/** Fetch current vision configuration. */
export async function getVisionConfig(): Promise<VisionConfig> {
	return (await apiGet('/api/vision/config')) as VisionConfig;
}

/** Update vision model selection or prompt. */
export async function updateVisionConfig(update: VisionConfigUpdate): Promise<VisionConfig> {
	return (await apiPut('/api/vision/config', update)) as VisionConfig;
}

/** List all detected vision-capable models. */
export async function listVisionModels(): Promise<VisionModelInfo[]> {
	return (await apiGet('/api/vision/models')) as VisionModelInfo[];
}

/** Clear capability probe cache to force re-detection. */
export async function clearVisionCache(): Promise<void> {
	await apiPost('/api/vision/clear-cache', {});
}
