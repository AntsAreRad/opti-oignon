/**
 * Svelte stores pour les options de chat (modele, preset, temperature).
 *
 * Ces valeurs sont lues par le chat store lors de l'envoi de messages
 * et transmises via ChatRequest.
 */

import { writable, get } from 'svelte/store';
import type { ModelInfo, PresetInfo } from '$lib/types';
import { listModels, getEffectiveModel } from '$lib/api/models';
import { listPresets } from '$lib/api/presets';

// -- Selections utilisateur --

/** Manually selected model (null = automatic). */
export const selectedModel = writable<string | null>(null);

/** Manually selected preset (null = automatic). */
export const selectedPreset = writable<string | null>(null);

/** Temperature (null = defaut du preset/modele). */
export const temperature = writable<number | null>(null);

/** Activer la detection automatique de preset. */
export const usePresets = writable<boolean>(true);

/** Activer le mode thinking/raisonnement. */
export const thinkingEnabled = writable<boolean>(false);

/** Activer la recherche web. */
export const webSearchEnabled = writable<boolean>(false);

/** Semantic cache toggle. */
export const cacheEnabled = writable<boolean>(false);

/** Cascading inference toggle. */
export const cascadingEnabled = writable<boolean>(false);

/** Speculative generation toggle. */
export const speculativeEnabled = writable<boolean>(false);

/** Prompt enhancement/optimization toggle. */
export const promptEnhanceEnabled = writable<boolean>(false);

/** Humanizer post-processing toggle. */
export const humanizeEnabled = writable<boolean>(false);

/** Quick sandbox mode toggle. */
export const quickSandboxEnabled = writable<boolean>(false);

/** Chat coding agent toggle. */
export const chatCodingEnabled = writable<boolean>(false);

/** Selected execution pipeline. null = auto. */
export const selectedExecPipeline = writable<string | null>(null);

// -- Caches --

/** Liste des modeles disponibles (chargee une fois). */
export const availableModels = writable<ModelInfo[]>([]);

/** Liste des presets disponibles (chargee une fois). */
export const availablePresets = writable<PresetInfo[]>([]);

/** Modele effectif courant. */
export const effectiveModel = writable<string>('');

/** Source of the effective model (auto_router, preset, forced, etc.). */
export const effectiveModelSource = writable<string>('');

/** Indicateur de chargement. */
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

/** Reset all user selections. */
export function resetOptions(): void {
	selectedModel.set(null);
	selectedPreset.set(null);
	temperature.set(null);
	usePresets.set(true);
	thinkingEnabled.set(false);
	webSearchEnabled.set(false);
	cacheEnabled.set(false);
	cascadingEnabled.set(false);
	speculativeEnabled.set(false);
	promptEnhanceEnabled.set(false);
	humanizeEnabled.set(false);
	quickSandboxEnabled.set(false);
	chatCodingEnabled.set(false);
	selectedExecPipeline.set(null);
}

/** Return current options for ChatRequest. */
export function getChatOptions(): {
	model?: string;
	preset?: string;
	temperature?: number;
	usePresets?: boolean;
	think?: boolean;
	web_search?: boolean;
	no_cache?: boolean;
	cascading?: boolean;
	speculative?: boolean;
	prompt_enhance?: boolean;
	humanize?: boolean;
	exec_pipeline?: string;
	quick_sandbox?: boolean;
	chat_coding?: boolean;
} {
	const opts: {
		model?: string;
		preset?: string;
		temperature?: number;
		usePresets?: boolean;
		think?: boolean;
		web_search?: boolean;
		no_cache?: boolean;
		cascading?: boolean;
		speculative?: boolean;
		prompt_enhance?: boolean;
		humanize?: boolean;
		exec_pipeline?: string;
		quick_sandbox?: boolean;
		chat_coding?: boolean;
	} = {};

	const model = get(selectedModel);
	if (model) opts.model = model;

	const preset = get(selectedPreset);
	if (preset) opts.preset = preset;

	const temp = get(temperature);
	if (temp !== null) opts.temperature = temp;

	const up = get(usePresets);
	if (!up) opts.usePresets = false;

	const think = get(thinkingEnabled);
	if (think) opts.think = true;

	const ws = get(webSearchEnabled);
	if (ws) opts.web_search = true;

	const ce = get(cacheEnabled);
	if (!ce) opts.no_cache = true;

	// Mutual exclusion: speculative takes priority over cascading
	const spec = get(speculativeEnabled);
	const casc = get(cascadingEnabled);
	if (spec) {
		opts.speculative = true;
	} else if (casc) {
		opts.cascading = true;
	}

	const ep = get(selectedExecPipeline);
	if (ep) opts.exec_pipeline = ep;

	const pe = get(promptEnhanceEnabled);
	if (pe) opts.prompt_enhance = true;

	const hz = get(humanizeEnabled);
	if (hz) opts.humanize = true;

	const qs = get(quickSandboxEnabled);
	if (qs) opts.quick_sandbox = true;

	const cc = get(chatCodingEnabled);
	if (cc) opts.chat_coding = true;

	return opts;
}
