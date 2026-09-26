/**
 * Svelte stores for the composer's choices (model, preset, temperature and
 * the per-message options), and the model and preset lists they choose from.
 *
 * The chat store reads them when a message is sent: getChatOptions() turns
 * them into request fields through lib/chat/requestFields.ts, which builds
 * only fields of the server's ChatRequest.
 *
 * The quick sandbox and coding agent stores mirror server-wide defaults: they
 * are written only with the state the server confirmed (the composer's
 * server switches adopt it), never with the state a button asked for.
 */

import { writable, get } from 'svelte/store';
import type { ModelInfo, PresetInfo } from '$lib/types';
import { listModels, getEffectiveModel } from '$lib/api/models';
import { listPresets } from '$lib/api/presets';
import { chatOptions, type ChatOptions } from '$lib/chat/requestFields';

// -- The user's choices --

/** Manually selected model (null = automatic). */
export const selectedModel = writable<string | null>(null);

/** Manually selected preset (null = automatic). */
export const selectedPreset = writable<string | null>(null);

/** Temperature (null = the preset's or the model's). */
export const temperature = writable<number | null>(null);

/** Whether presets are detected automatically. */
export const usePresets = writable<boolean>(true);

/** Thinking forced on for the next messages. */
export const thinkingEnabled = writable<boolean>(false);

/** Web search forced on for the next messages. */
export const webSearchEnabled = writable<boolean>(false);

/** The quick sandbox's server-wide default, as the server confirmed it. */
export const quickSandboxEnabled = writable<boolean>(false);

/** The chat coding agent's server-wide default, as the server confirmed it. */
export const chatCodingEnabled = writable<boolean>(false);

/** Selected execution pipeline. null = auto. */
export const selectedExecPipeline = writable<string | null>(null);

// -- Lists --

/** The available models (loaded once). */
export const availableModels = writable<ModelInfo[]>([]);

/** The available presets (loaded once). */
export const availablePresets = writable<PresetInfo[]>([]);

/** The model in effect now. */
export const effectiveModel = writable<string>('');

/** Source of the effective model (auto_router, preset, forced, etc.). */
export const effectiveModelSource = writable<string>('');

/** The lists are loading. */
export const optionsLoading = writable<boolean>(false);

// -- Actions --

/** Load models and presets from the API. Called once on mount. */
export async function loadOptions(): Promise<void> {
	optionsLoading.set(true);
	try {
		const [modelsResp, presets, effective] = await Promise.allSettled([
			listModels(),
			listPresets(),
			getEffectiveModel(),
		]);

		if (modelsResp.status === 'fulfilled') {
			availableModels.set(modelsResp.value.models);
		}
		if (presets.status === 'fulfilled') {
			availablePresets.set(presets.value);
		}
		if (effective.status === 'fulfilled') {
			effectiveModel.set(effective.value.model);
			effectiveModelSource.set(effective.value.source);
		}
	} finally {
		optionsLoading.set(false);
	}
}

/**
 * Reset the user's choices. The server-wide defaults are the server's and
 * are not touched here.
 */
export function resetOptions(): void {
	selectedModel.set(null);
	selectedPreset.set(null);
	temperature.set(null);
	usePresets.set(true);
	thinkingEnabled.set(false);
	webSearchEnabled.set(false);
	selectedExecPipeline.set(null);
}

/** The options the next message sends: request fields, each only when chosen. */
export function getChatOptions(): ChatOptions {
	return chatOptions({
		model: get(selectedModel),
		preset: get(selectedPreset),
		temperature: get(temperature),
		usePresets: get(usePresets),
		think: get(thinkingEnabled),
		webSearch: get(webSearchEnabled),
		quickSandbox: get(quickSandboxEnabled),
		chatCoding: get(chatCodingEnabled),
		execPipeline: get(selectedExecPipeline),
	});
}
