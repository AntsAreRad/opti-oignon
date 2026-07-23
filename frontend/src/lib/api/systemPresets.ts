/**
 * Typed API functions for system preset endpoints.
 *
 * Infrastructure-level presets (Minimal/Balanced/Power) that
 * configure multiple YAML config files at once.
 */

import { apiGet, apiPost } from './client';
import type {
	SystemPresetListResponse,
	SystemPresetDetectResponse,
	SystemPresetApplyResponse,
	OnboardingStateResponse,
} from '$lib/types';

/** List all available system presets. */
export async function listSystemPresets(): Promise<SystemPresetListResponse> {
	return apiGet<SystemPresetListResponse>('/api/system-presets/list');
}

/** Auto-detect installed Ollama models and get a recommendation. */
export async function detectAndRecommend(): Promise<SystemPresetDetectResponse> {
	return apiGet<SystemPresetDetectResponse>('/api/system-presets/detect');
}

/** Apply a system preset to all config files. */
export async function applySystemPreset(presetId: string): Promise<SystemPresetApplyResponse> {
	return apiPost<SystemPresetApplyResponse>(
		`/api/system-presets/apply/${encodeURIComponent(presetId)}`
	);
}

/** Get current onboarding state. */
export async function getOnboardingState(): Promise<OnboardingStateResponse> {
	return apiGet<OnboardingStateResponse>('/api/system-presets/onboarding');
}

/** Reset onboarding state to re-trigger overlay. */
export async function resetOnboarding(): Promise<{ reset: boolean }> {
	return apiPost<{ reset: boolean }>('/api/system-presets/onboarding/reset');
}
