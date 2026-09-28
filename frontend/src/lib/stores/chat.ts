/**
 * Svelte stores pour l'etat du streaming de chat.
 *
 * Gere l'etat de streaming, le contenu en cours, et les actions
 * d'envoi, retry et annulation.
 *
 * One reset puts every live store of a reply back at rest; sending,
 * retrying, done, an error and a Stop all call it. The loader's state is
 * not emptied there: the reducer ends it itself (done, an error, a lost
 * stream), so the status region can still say how the reply ended and a run
 * cut short stays drawn; the next reply's start replaces it. A reply that
 * ends keeps what was written: a Stop marks it `stopped`, adding nothing to
 * its text, and an error keeps the partial reply beside the error. A reply
 * whose `done` carries the steps' record keeps it, for its summary line.
 */

import { writable, get } from 'svelte/store';
import { streamChat, retryChat, cancelGeneration } from '$lib/api/chat';
import type { ChatConnection } from '$lib/api/chat';
import { messages, loadConversations } from '$lib/stores/conversations';
import { getMessages } from '$lib/api/conversations';
import type {
	ChatResponse,
	ChatStreamCallbacks,
	ChatToken,
	MessageItem,
	ReasoningMetaInfo,
	ReasoningStepInfo,
	ToolCallInfo,
	VerificationInfo,
} from '$lib/types';
import { chatRequest, type ChatOptions } from '$lib/chat/requestFields';
import {
	lose,
	observe,
	reconnecting,
	start,
	stop as stopLoader,
	type LoaderState,
} from '$lib/chat/progress';
import { statusLine } from '$lib/chat/loaderWords';

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

/** The loader's state for the reply being streamed (lib/chat/progress.ts). */
export const streamingLoader = writable<LoaderState | null>(null);

/** The conversation the reply being streamed (or the last one) belongs to. */
export const streamingConversation = writable<string | null>(null);

/** What the stream reported about the reply, attached to it when it is done. */
export const streamingVerifications = writable<VerificationInfo[]>([]);
export const streamingToolCalls = writable<ToolCallInfo[]>([]);
export const streamingReasoningSteps = writable<ReasoningStepInfo[]>([]);
export const streamingReasoningMeta = writable<ReasoningMetaInfo | null>(null);

// -- Etat interne --

let activeConnection: ChatConnection | null = null;

/** Every live store of a reply back at rest: the one reset every path calls. */
function resetStreaming(): void {
	isStreaming.set(false);
	streamingContent.set('');
	streamingThinking.set('');
	streamingModel.set(null);
	streamingVisionDelegation.set(null);
	streamingStatus.set(null);
	streamingCodingEvents.set([]);
	isCodingStream.set(false);
	streamingVerifications.set([]);
	streamingToolCalls.set([]);
	streamingReasoningSteps.set([]);
	streamingReasoningMeta.set(null);
	activeConnection = null;
}

/** A new reply: every live store at rest, the last reply's error gone, the loader started. */
function beginStream(conversationId: string): void {
	resetStreaming();
	streamingError.set(null);
	lastSearchMetadata.set(null);
	streamingConversation.set(conversationId);
	streamingLoader.set(start(Date.now()));
	isStreaming.set(true);
}

/** Whether the reader asked for a Stop of the reply being streamed. */
function stopAsked(): boolean {
	return get(streamingLoader)?.stopRequested === true;
}

/** How the reply ended, as its own fields: stopped, and the steps' record `done` carried. */
function endOf(response: ChatResponse): Partial<MessageItem> {
	const end: Partial<MessageItem> = {};
	if (response.cancelled === true || stopAsked()) end.stopped = true;
	if (Array.isArray(response.steps) && response.steps.length > 0) {
		end.steps = response.steps;
		end.duration_ms = response.duration_ms;
	}
	return end;
}

/** An error ended the reply: what was written stays, beside the error. */
function keepPartialReply(): void {
	const partial = get(streamingContent);
	if (!partial) return;
	const kept: MessageItem = {
		id: null,
		role: 'assistant',
		content: partial,
		timestamp: new Date().toISOString(),
		model: get(streamingModel),
		token_estimate: 0,
		thinking: get(streamingThinking) || undefined,
		...streamReports(),
	};
	if (stopAsked()) kept.stopped = true;
	messages.update((msgs) => [...msgs, kept]);
}

function feedLoader(frame: ChatToken): void {
	streamingLoader.update((state) => state && observe(state, frame, Date.now(), statusLine));
}

function reconnectLoader(attempt: number, max: number): void {
	streamingLoader.update((state) => state && reconnecting(state, attempt, max, Date.now()));
}

function loseLoader(): void {
	streamingLoader.update((state) => state && lose(state, Date.now()));
}

function keepVerification(info: VerificationInfo): void {
	streamingVerifications.update((list) => [...list, info]);
}

function keepToolCall(info: ToolCallInfo): void {
	streamingToolCalls.update((list) => [...list, info]);
}

function keepReasoningStep(info: ReasoningStepInfo): void {
	streamingReasoningSteps.update((list) => [...list, info]);
}

function keepReasoningMeta(info: ReasoningMetaInfo): void {
	streamingReasoningMeta.set(info);
}

function keepCodingEvent(eventType: string, data: Record<string, unknown>): void {
	streamingCodingEvents.update((events) => [
		...events,
		{
			eventType,
			content: (data.event_content as string) || '',
			data,
			timestamp: Date.now(),
		},
	]);
}

/** What the stream reported, as the finished reply's own fields. */
function streamReports(): Partial<MessageItem> {
	const reports: Partial<MessageItem> = {};
	const verifications = get(streamingVerifications);
	const toolCalls = get(streamingToolCalls);
	const reasoningSteps = get(streamingReasoningSteps);
	const reasoningMeta = get(streamingReasoningMeta);
	if (verifications.length > 0) reports.verification = verifications;
	if (toolCalls.length > 0) reports.tool_calls = toolCalls;
	if (reasoningSteps.length > 0) reports.reasoning_steps = reasoningSteps;
	if (reasoningMeta) reports.reasoning_meta = reasoningMeta;
	return reports;
}

/**
 * Envoie un message et streame la reponse.
 *
 * Ajoute immediatement le message user au store, puis accumule
 * les tokens de l'assistant au fur et a mesure.
 */
export async function sendMessage(
	conversationId: string,
	message: string,
	options: ChatOptions = {}
): Promise<void> {
	if (get(isStreaming)) return;

	// A new reply, and a new session: the last one's sandbox and coding records go.
	lastSandboxMeta.set(null);
	lastCodingMeta.set(null);
	beginStream(conversationId);

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

	// The request: the conversation, the message, and every option that is a
	// ChatRequest field (the builder drops any other key).
	const request = chatRequest(conversationId, message, options);

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
				...streamReports(),
				...endOf(response),
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

			resetStreaming();

			// Rafraichir la liste (pour mettre a jour le titre et message_count)
			loadConversations();
		},
		onError: (error: string) => {
			keepPartialReply();
			streamingError.set(error);
			resetStreaming();
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
		// A server status as sent; the loader says it in the table's words, through onFrame
		onStatus: (message) => {
			streamingStatus.set(message || null);
		},
		// Live coding agent events during streaming
		onCodingEvent: keepCodingEvent,
		onVerification: keepVerification,
		onToolCall: keepToolCall,
		onReasoningStep: keepReasoningStep,
		onReasoningDone: keepReasoningMeta,
		onFrame: feedLoader,
		onReconnecting: reconnectLoader,
		onLost: loseLoader,
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

	// The sandbox and coding records stay: a retry answers in the same session.
	beginStream(conversationId);

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
			// How it ended, kept before the reset; the saved message does not carry it.
			const end = endOf(response);
			const reports = streamReports();
			const thinkingText = get(streamingThinking);
			resetStreaming();
			// Recharger les messages depuis l'API (le backend gere l'etat)
			try {
				const freshMessages = await getMessages(conversationId);
				let last = -1;
				freshMessages.forEach((msg, i) => {
					if (msg.role === 'assistant') last = i;
				});
				messages.set(
					last < 0 ? freshMessages : freshMessages.map((msg, i) => (i === last ? { ...msg, ...end } : msg))
				);
			} catch {
				// Fallback: ajouter le message assistant localement
				const assistantMsg = {
					id: response.message_id,
					role: 'assistant',
					content: response.content,
					timestamp: new Date().toISOString(),
					model: response.model,
					token_estimate: response.tokens,
					thinking: thinkingText || undefined,
					...reports,
					...end,
				};
				messages.update((msgs) => [...msgs, assistantMsg]);
			}
		},
		onError: (error: string) => {
			keepPartialReply();
			streamingError.set(error);
			resetStreaming();
		},
		onMetadata: (metadata) => {
			if (metadata.model) {
				streamingModel.set(metadata.model as string);
			}
		},
		onVisionDelegation: (info) => {
			streamingVisionDelegation.set(info);
		},
		// A server status as sent; the loader says it in the table's words, through onFrame
		onStatus: (message) => {
			streamingStatus.set(message || null);
		},
		onCodingEvent: keepCodingEvent,
		onVerification: keepVerification,
		onToolCall: keepToolCall,
		onReasoningStep: keepReasoningStep,
		onReasoningDone: keepReasoningMeta,
		onFrame: feedLoader,
		onReconnecting: reconnectLoader,
		onLost: loseLoader,
	};

	activeConnection = retryChat(conversationId, callbacks);
}

/**
 * Stops the reply being streamed.
 *
 * The cancel goes first (POST /api/chat/cancel) and the socket keeps being
 * read: the server closes its open steps and sends done, which finishes the
 * reply through the stream's own callbacks. Only a stream that ends without
 * an answer leaves the partial reply to be kept here.
 */
export async function cancelCurrentGeneration(conversationId: string): Promise<void> {
	if (!get(isStreaming)) return;

	streamingLoader.update((state) => state && stopLoader(state, Date.now()));
	const connection = activeConnection;
	const request = () => cancelGeneration(conversationId);
	if (connection) {
		const ended = await connection.stop(request);
		if (ended === 'done' || ended === 'error') return;
	} else {
		try {
			await request();
		} catch {
			/* the reply may be over already */
		}
	}
	// A second Stop waited on the same end: the first one kept the reply.
	if (!get(isStreaming)) return;

	// Nobody answered the Stop: the text written so far is the reply, marked stopped.
	const partial = get(streamingContent);
	if (partial) {
		const partialMsg = {
			id: null,
			role: 'assistant',
			content: partial,
			timestamp: new Date().toISOString(),
			model: get(streamingModel),
			token_estimate: 0,
			thinking: get(streamingThinking) || undefined,
			...streamReports(),
			stopped: true,
		};
		messages.update((msgs) => [...msgs, partialMsg]);
	}
	resetStreaming();
}
