/**
 * Svelte stores pour l'etat du streaming de chat.
 *
 * Gere l'etat de streaming, le contenu en cours, et les actions
 * d'envoi, retry et annulation.
 */

import { writable, get } from 'svelte/store';
import { streamChat, retryChat, cancelGeneration } from '$lib/api/chat';
import type { ChatConnection } from '$lib/api/chat';
import { messages, loadConversations } from '$lib/stores/conversations';
import { getMessages } from '$lib/api/conversations';
import type { ChatStreamCallbacks } from '$lib/types';

// -- Stores de streaming --

/** True pendant qu'une generation est en cours. */
export const isStreaming = writable<boolean>(false);

/** Contenu accumule du streaming en cours. */
export const streamingContent = writable<string>('');

/** Contenu thinking accumule du streaming en cours. */
export const streamingThinking = writable<string>('');

/** Modele utilise pour la generation en cours. */
export const streamingModel = writable<string | null>(null);

/** Erreur du dernier streaming. */
export const streamingError = writable<string | null>(null);

/** Metadata de recherche du dernier message (resultats inline). */
export const lastSearchMetadata = writable<Record<string, unknown> | null>(null);

/** Map de search metadata par message ID pour l'historique. */
export const searchMetadataMap = writable<Map<string, Record<string, unknown>>>(new Map());

/** Vision delegation state during streaming. */
export const streamingVisionDelegation = writable<Record<string, unknown> | null>(null);

/** Intermediate status message during streaming (e.g. "Searching...", "Thinking..."). */
export const streamingStatus = writable<string | null>(null);

/** Sandbox metadata from the last done message (session_id, files). */
export const lastSandboxMeta = writable<{
	sandbox_active: boolean;
	sandbox_session_id: string;
	sandbox_files: unknown[];
	sandbox_files_created: string[];
} | null>(null);

/** Coding agent metadata from the last done message. */
export const lastCodingMeta = writable<{
	chat_coding: boolean;
	coding_result: Record<string, unknown>;
	sandbox_session_id: string;
	sandbox_files: unknown[];
	sandbox_files_created: string[];
	turn_count: number;
} | null>(null);

/** Live coding agent events accumulated during streaming. */
export interface CodingEventEntry {
	eventType: string;
	content: string;
	data: Record<string, unknown>;
	timestamp: number;
}
export const streamingCodingEvents = writable<CodingEventEntry[]>([]);

/** Whether the current streaming is a coding agent turn. */
export const isCodingStream = writable<boolean>(false);

// -- Etat interne --

let activeConnection: ChatConnection | null = null;

/**
 * Envoie un message et streame la reponse.
 *
 * Ajoute immediatement le message user au store, puis accumule
 * les tokens de l'assistant au fur et a mesure.
 */
export async function sendMessage(
	conversationId: string,
	message: string,
	options?: {
		model?: string;
		preset?: string;
		temperature?: number;
		usePresets?: boolean;
		think?: boolean;
		web_search?: boolean;
		images?: string[];
		quick_sandbox?: boolean;
		chat_coding?: boolean;
		exec_pipeline?: string;
	}
): Promise<void> {
	if (get(isStreaming)) return;

	// Reset etat
	isStreaming.set(true);
	streamingContent.set('');
	streamingThinking.set('');
	streamingModel.set(null);
	streamingError.set(null);
	lastSearchMetadata.set(null);
	streamingVisionDelegation.set(null);
	streamingStatus.set(null);
	lastSandboxMeta.set(null);
	lastCodingMeta.set(null);
	streamingCodingEvents.set([]);
	isCodingStream.set(false);

	// Ajouter le message user localement
	const userMsg = {
		id: null,
		role: 'user',
		content: message,
		timestamp: new Date().toISOString(),
		model: null,
		token_estimate: 0,
	};
	messages.update((msgs) => [...msgs, userMsg]);

	// Construire la requete
	const request: Record<string, unknown> = {
		conversation_id: conversationId,
		message,
		model: options?.model,
		preset: options?.preset,
		temperature: options?.temperature,
		use_presets: options?.usePresets,
		think: options?.think,
		web_search: options?.web_search,
		quick_sandbox: options?.quick_sandbox,
		chat_coding: options?.chat_coding,
		// (PIP-06): the field existed in getChatOptions since but
		// was dropped here; the backend ChatRequest now carries it.
		exec_pipeline: options?.exec_pipeline,
	};
	if (options?.images && options.images.length > 0) {
		request.images = options.images;
	}

	const callbacks: ChatStreamCallbacks = {
		onToken: (content: string) => {
			streamingContent.update((prev) => prev + content);
		},
		onThinking: (content: string) => {
			// Accumuler le contenu de reflexion separement
			streamingThinking.update((prev) => prev + content);
		},
		onDone: (response) => {
			// Ajouter le message assistant final au store
			// Inclure le thinking content si present
			const thinkingText = get(streamingThinking);
			// Include vision delegation info if present
			const visionDel = get(streamingVisionDelegation);
			const assistantMsg: Record<string, unknown> = {
				id: response.message_id,
				role: 'assistant',
				content: response.content,
				timestamp: new Date().toISOString(),
				model: response.model,
				token_estimate: response.tokens,
				thinking: thinkingText || (response as Record<string, unknown>).thinking as string || undefined,
			};
			if (visionDel?.vision_model) {
				assistantMsg.vision_delegation = visionDel;
			}
			// Attach sandbox metadata to the message if sandbox was used
			const resp = response as Record<string, unknown>;
			if (resp.sandbox_active) {
				const sMeta = {
					sandbox_active: true,
					sandbox_session_id: String(resp.sandbox_session_id || ''),
					sandbox_files: (resp.sandbox_files as unknown[]) || [],
					sandbox_files_created: (resp.sandbox_files_created as string[]) || [],
				};
				assistantMsg.sandbox_meta = sMeta;
				lastSandboxMeta.set(sMeta);
			}
			// Attach coding agent metadata if coding agent was used
			if (resp.chat_coding) {
				const cMeta = {
					chat_coding: true,
					coding_result: (resp.coding_result as Record<string, unknown>) || {},
					sandbox_session_id: String(resp.sandbox_session_id || ''),
					sandbox_files: (resp.sandbox_files as unknown[]) || [],
					sandbox_files_created: (resp.sandbox_files_created as string[]) || [],
					turn_count: Number(resp.turn_count || 0),
				};
				assistantMsg.coding_meta = cMeta;
				lastCodingMeta.set(cMeta);
				// Also set sandbox meta (coding agent uses sandbox)
				if (resp.sandbox_active) {
					lastSandboxMeta.set({
						sandbox_active: true,
						sandbox_session_id: String(resp.sandbox_session_id || ''),
						sandbox_files: (resp.sandbox_files as unknown[]) || [],
						sandbox_files_created: (resp.sandbox_files_created as string[]) || [],
					});
				}
			}
			messages.update((msgs) => [...msgs, assistantMsg]);

			// Sauvegarder les search metadata pour ce message
			const searchMeta = get(lastSearchMetadata);
			if (searchMeta && response.message_id != null) {
				searchMetadataMap.update((map) => {
					const newMap = new Map(map);
					newMap.set(String(response.message_id), searchMeta);
					return newMap;
				});
			}

			// Reset streaming
			isStreaming.set(false);
			streamingContent.set('');
			streamingThinking.set('');
			streamingModel.set(null);
			streamingVisionDelegation.set(null);
			streamingStatus.set(null);
			streamingCodingEvents.set([]);
			isCodingStream.set(false);
			activeConnection = null;

			// Rafraichir la liste (pour mettre a jour le titre et message_count)
			loadConversations();
		},
		onError: (error: string) => {
			streamingError.set(error);
			isStreaming.set(false);
			streamingContent.set('');
			streamingThinking.set('');
			streamingModel.set(null);
			streamingVisionDelegation.set(null);
			streamingStatus.set(null);
			streamingCodingEvents.set([]);
			isCodingStream.set(false);
			activeConnection = null;
		},
		onMetadata: (metadata) => {
			if (metadata.model) {
				streamingModel.set(metadata.model as string);
			}
			// Capture search results metadata
			if (metadata.search_results || metadata.search) {
				lastSearchMetadata.set(metadata);
			}
			// Capture vision delegation from done metadata
			if (metadata.vision_delegation) {
				streamingVisionDelegation.set(metadata.vision_delegation as Record<string, unknown>);
			}
			// Detect coding agent mode from initial metadata
			if (metadata.chat_coding) {
				isCodingStream.set(true);
			}
		},
		// Vision delegation status updates
		onVisionDelegation: (info) => {
			streamingVisionDelegation.set(info);
		},
		// Intermediate status for StreamingIndicator
		onStatus: (message) => {
			streamingStatus.set(message || null);
		},
		// Live coding agent events during streaming
		onCodingEvent: (eventType, data) => {
			streamingCodingEvents.update((events) => [
				...events,
				{
					eventType,
					content: (data.event_content as string) || '',
					data,
					timestamp: Date.now(),
				},
			]);
		},
	};

	activeConnection = streamChat(request, callbacks);
}

/**
 * Regenerate the last assistant response.
 *
 * Delete user+assistant messages locally, then
 * relance le streaming via /api/chat/retry.
 */
export async function retryLastMessage(conversationId: string): Promise<void> {
	if (get(isStreaming)) return;

	isStreaming.set(true);
	streamingContent.set('');
	streamingThinking.set('');
	streamingModel.set(null);
	streamingError.set(null);
	streamingStatus.set(null);

	// Supprimer les derniers messages localement (assistant puis user)
	// Le backend s'occupe de la suppression en DB
	messages.update((msgs) => {
		const copy = [...msgs];
		// Supprimer le dernier assistant
		for (let i = copy.length - 1; i >= 0; i--) {
			if (copy[i].role === 'assistant') {
				copy.splice(i, 1);
				break;
			}
		}
		return copy;
	});

	const callbacks: ChatStreamCallbacks = {
		onToken: (content: string) => {
			streamingContent.update((prev) => prev + content);
		},
		onThinking: (content: string) => {
			streamingThinking.update((prev) => prev + content);
		},
		onDone: async (response) => {
			// Recharger les messages depuis l'API (le backend gere l'etat)
			try {
				const freshMessages = await getMessages(conversationId);
				messages.set(freshMessages);
			} catch {
				// Fallback: ajouter le message assistant localement
				const thinkingText = get(streamingThinking);
				const assistantMsg = {
					id: response.message_id,
					role: 'assistant',
					content: response.content,
					timestamp: new Date().toISOString(),
					model: response.model,
					token_estimate: response.tokens,
					thinking: thinkingText || undefined,
				};
				messages.update((msgs) => [...msgs, assistantMsg]);
			}

			isStreaming.set(false);
			streamingContent.set('');
			streamingThinking.set('');
			streamingModel.set(null);
			streamingStatus.set(null);
			activeConnection = null;
		},
		onError: (error: string) => {
			streamingError.set(error);
			isStreaming.set(false);
			streamingContent.set('');
			streamingThinking.set('');
			streamingModel.set(null);
			streamingStatus.set(null);
			activeConnection = null;
		},
		onMetadata: (metadata) => {
			if (metadata.model) {
				streamingModel.set(metadata.model as string);
			}
		},
		// Intermediate status for StreamingIndicator
		onStatus: (message) => {
			streamingStatus.set(message || null);
		},
	};

	activeConnection = retryChat(conversationId, callbacks);
}

/**
 * Annule la generation en cours.
 * Envoie un POST /api/chat/cancel et ferme le WebSocket.
 */
export async function cancelCurrentGeneration(conversationId: string): Promise<void> {
	if (!get(isStreaming)) return;

	// Fermer le WebSocket cote client
	if (activeConnection) {
		activeConnection.cancel();
		activeConnection = null;
	}

	// Demander l'annulation cote serveur
	try {
		await cancelGeneration(conversationId);
	} catch {
		// L'annulation peut echouer si la generation est deja terminee
	}

	// Finaliser le contenu partiel comme message
	const partial = get(streamingContent);
	if (partial) {
		const model = get(streamingModel);
		const thinkingText = get(streamingThinking);
		const partialMsg = {
			id: null,
			role: 'assistant',
			content: partial + '\n\n[Generation cancelled]',
			timestamp: new Date().toISOString(),
			model: model,
			token_estimate: 0,
			thinking: thinkingText || undefined,
		};
		messages.update((msgs) => [...msgs, partialMsg]);
	}

	isStreaming.set(false);
	streamingContent.set('');
	streamingThinking.set('');
	streamingModel.set(null);
}
