/**
 * Service WebSocket pour le streaming de chat.
 *
 * Gere les connexions WebSocket ephemeres (une par message),
 * l'annulation, et le retry.
 * S94: Auto-reconnection with exponential backoff on connection loss.
 */

import { wsUrl } from './client';
import { apiPost } from './client';
import type {
	ChatRequest,
	ChatToken,
	ChatResponse,
	ChatStreamCallbacks,
	ChatRetryRequest,
	ToolCallInfo,
} from '$lib/types';

export interface ChatConnection {
	cancel: () => void;
	socket: WebSocket;
}

// S94: Reconnection config
const WS_MAX_RETRIES = 3;
const WS_BASE_DELAY_MS = 500;
const WS_MAX_DELAY_MS = 4000;

/** Compute exponential backoff delay with jitter. */
function backoffDelay(attempt: number): number {
	const base = Math.min(WS_BASE_DELAY_MS * Math.pow(2, attempt), WS_MAX_DELAY_MS);
	const jitter = Math.random() * base * 0.3;
	return base + jitter;
}

/**
 * Ouvre un WebSocket vers /api/chat/stream, envoie la requete,
 * et dispatch les tokens via les callbacks.
 * S94: Auto-reconnects on unexpected connection loss (up to WS_MAX_RETRIES).
 */
export function streamChat(
	request: ChatRequest,
	callbacks: ChatStreamCallbacks
): ChatConnection {
	let closed = false;
	let retryCount = 0;
	let hasReceivedData = false;
	let currentSocket: WebSocket | null = null;

	function cleanup() {
		closed = true;
		if (currentSocket && (currentSocket.readyState === WebSocket.OPEN || currentSocket.readyState === WebSocket.CONNECTING)) {
			try { currentSocket.close(); } catch { /* ignore */ }
		}
	}

	function connect() {
		const socket = new WebSocket(wsUrl('/api/chat/stream'));
		currentSocket = socket;

		socket.onopen = () => {
			try {
				socket.send(JSON.stringify(request));
			} catch {
				callbacks.onError('Failed to send chat request');
				cleanup();
			}
		};

		socket.onmessage = (event: MessageEvent) => {
			if (closed) return;
			hasReceivedData = true;
			retryCount = 0;
			try {
				const data: ChatToken = JSON.parse(event.data);
				switch (data.type) {
					case 'token':
						callbacks.onToken(data.content);
						break;
					case 'thinking':
						callbacks.onThinking?.(data.content);
						break;
					case 'done': {
						const response: ChatResponse = {
							conversation_id: (data.metadata?.conversation_id as string) ?? request.conversation_id ?? '',
							message_id: (data.metadata?.message_id as number) ?? null,
							content: data.content,
							model: (data.metadata?.model as string) ?? '',
							tokens: (data.metadata?.tokens as number) ?? 0,
							duration_ms: (data.metadata?.duration_ms as number) ?? 0,
							// S117: Quick sandbox metadata
							sandbox_active: (data.metadata?.sandbox_active as boolean) ?? false,
							sandbox_session_id: (data.metadata?.sandbox_session_id as string) ?? undefined,
							sandbox_files: (data.metadata?.sandbox_files as unknown[]) ?? undefined,
							sandbox_files_created: (data.metadata?.sandbox_files_created as string[]) ?? undefined,
							// S118: Chat coding agent metadata
							chat_coding: (data.metadata?.chat_coding as boolean) ?? false,
							coding_result: (data.metadata?.coding_result as Record<string, unknown>) ?? undefined,
							turn_count: (data.metadata?.turn_count as number) ?? undefined,
						};
						callbacks.onDone(response);
						cleanup();
						break;
					}
					case 'error':
						callbacks.onError(data.content);
						cleanup();
						break;
					case 'metadata':
						callbacks.onMetadata?.(data.metadata ?? {});
						break;
					case 'verification':
						if (data.metadata && callbacks.onVerification) {
							callbacks.onVerification(data.metadata as any);
						}
						break;
					case 'tool_call':
						if (data.metadata && callbacks.onToolCall) {
							callbacks.onToolCall(data.metadata as ToolCallInfo);
						}
						break;
					case 'vision_delegation':
						callbacks.onVisionDelegation?.(data.metadata ?? {});
						break;
					case 'status':
						// S109: Intermediate status feedback
						callbacks.onStatus?.((data.metadata?.message as string) ?? '');
						break;
					// S118: Coding agent events
					case 'coding_plan':
					case 'coding_step':
					case 'coding_test':
					case 'coding_fix':
					case 'coding_done':
					case 'coding_status':
						callbacks.onCodingEvent?.(data.type, data.metadata ?? {});
						break;
					case 'coding_error':
						callbacks.onError(data.content || 'Coding agent error');
						cleanup();
						break;
				}
			} catch {
				callbacks.onError('Failed to parse server message');
				cleanup();
			}
		};

		socket.onerror = () => {
			if (closed) return;
			// S94: Attempt reconnect if we haven't received data yet
			if (!hasReceivedData && retryCount < WS_MAX_RETRIES) {
				retryCount++;
				const delay = backoffDelay(retryCount);
				callbacks.onMetadata?.({ reconnecting: true, attempt: retryCount });
				setTimeout(() => {
					if (!closed) connect();
				}, delay);
			} else {
				callbacks.onError('WebSocket connection error');
				cleanup();
			}
		};

		socket.onclose = (event: CloseEvent) => {
			if (closed) return;
			if (event.code === 1000) {
				// Normal closure
				closed = true;
				return;
			}
			// S94: Attempt reconnect on unexpected close before data received
			if (!hasReceivedData && retryCount < WS_MAX_RETRIES) {
				retryCount++;
				const delay = backoffDelay(retryCount);
				callbacks.onMetadata?.({ reconnecting: true, attempt: retryCount });
				setTimeout(() => {
					if (!closed) connect();
				}, delay);
			} else {
				callbacks.onError(`Connection lost (code: ${event.code}). Please retry.`);
				closed = true;
			}
		};
	}

	connect();

	return {
		cancel: () => {
			cleanup();
		},
		get socket() {
			return currentSocket!;
		},
	};
}

/**
 * Ouvre un WebSocket vers /api/chat/retry pour regenerer la derniere reponse.
 * S94: Auto-reconnects on unexpected connection loss.
 */
export function retryChat(
	conversationId: string,
	callbacks: ChatStreamCallbacks
): ChatConnection {
	let closed = false;
	let retryCount = 0;
	let hasReceivedData = false;
	let currentSocket: WebSocket | null = null;

	function cleanup() {
		closed = true;
		if (currentSocket && (currentSocket.readyState === WebSocket.OPEN || currentSocket.readyState === WebSocket.CONNECTING)) {
			try { currentSocket.close(); } catch { /* ignore */ }
		}
	}

	function connect() {
		const socket = new WebSocket(wsUrl('/api/chat/retry'));
		currentSocket = socket;

		socket.onopen = () => {
			try {
				const retryReq: ChatRetryRequest = { conversation_id: conversationId };
				socket.send(JSON.stringify(retryReq));
			} catch {
				callbacks.onError('Failed to send retry request');
				cleanup();
			}
		};

		socket.onmessage = (event: MessageEvent) => {
			if (closed) return;
			hasReceivedData = true;
			retryCount = 0;
			try {
				const data: ChatToken = JSON.parse(event.data);
				switch (data.type) {
					case 'token':
						callbacks.onToken(data.content);
						break;
					case 'thinking':
						callbacks.onThinking?.(data.content);
						break;
					case 'done': {
						const response: ChatResponse = {
							conversation_id: (data.metadata?.conversation_id as string) ?? conversationId,
							message_id: (data.metadata?.message_id as number) ?? null,
							content: data.content,
							model: (data.metadata?.model as string) ?? '',
							tokens: (data.metadata?.tokens as number) ?? 0,
							duration_ms: (data.metadata?.duration_ms as number) ?? 0,
							// S117: Quick sandbox metadata
							sandbox_active: (data.metadata?.sandbox_active as boolean) ?? false,
							sandbox_session_id: (data.metadata?.sandbox_session_id as string) ?? undefined,
							sandbox_files: (data.metadata?.sandbox_files as unknown[]) ?? undefined,
							sandbox_files_created: (data.metadata?.sandbox_files_created as string[]) ?? undefined,
						};
						callbacks.onDone(response);
						cleanup();
						break;
					}
					case 'error':
						callbacks.onError(data.content);
						cleanup();
						break;
					case 'metadata':
						callbacks.onMetadata?.(data.metadata ?? {});
						break;
					case 'verification':
						if (data.metadata && callbacks.onVerification) {
							callbacks.onVerification(data.metadata as any);
						}
						break;
					case 'tool_call':
						if (data.metadata && callbacks.onToolCall) {
							callbacks.onToolCall(data.metadata as ToolCallInfo);
						}
						break;
					case 'vision_delegation':
						callbacks.onVisionDelegation?.(data.metadata ?? {});
						break;
					case 'status':
						// S109: Intermediate status feedback
						callbacks.onStatus?.((data.metadata?.message as string) ?? '');
						break;
				}
			} catch {
				callbacks.onError('Failed to parse server message');
				cleanup();
			}
		};

		socket.onerror = () => {
			if (closed) return;
			if (!hasReceivedData && retryCount < WS_MAX_RETRIES) {
				retryCount++;
				const delay = backoffDelay(retryCount);
				callbacks.onMetadata?.({ reconnecting: true, attempt: retryCount });
				setTimeout(() => {
					if (!closed) connect();
				}, delay);
			} else {
				callbacks.onError('WebSocket connection error');
				cleanup();
			}
		};

		socket.onclose = (event: CloseEvent) => {
			if (closed) return;
			if (event.code === 1000) {
				closed = true;
				return;
			}
			if (!hasReceivedData && retryCount < WS_MAX_RETRIES) {
				retryCount++;
				const delay = backoffDelay(retryCount);
				callbacks.onMetadata?.({ reconnecting: true, attempt: retryCount });
				setTimeout(() => {
					if (!closed) connect();
				}, delay);
			} else {
				callbacks.onError(`Connection lost (code: ${event.code}). Please retry.`);
				closed = true;
			}
		};
	}

	connect();

	return {
		cancel: () => {
			cleanup();
		},
		get socket() {
			return currentSocket!;
		},
	};
}

/**
 * Annule la generation en cours pour une conversation via POST.
 */
export async function cancelGeneration(conversationId: string): Promise<void> {
	await apiPost('/api/chat/cancel', { conversation_id: conversationId });
}


// -- Consensus (S50) --

import type { ConsensusResult, ConsensusConfig } from '$lib/types';
import { apiGet } from './client';

/**
 * Execute un consensus multi-modele via POST.
 */
export async function runConsensus(params: {
	message: string;
	models?: string[];
	strategy?: string;
	system_prompt?: string;
	temperature?: number;
}): Promise<ConsensusResult> {
	const response = await apiPost('/api/chat/consensus', params);
	return response as ConsensusResult;
}

/**
 * Retrieve the consensus configuration.
 */
export async function getConsensusConfig(): Promise<ConsensusConfig> {
	const response = await apiGet('/api/chat/consensus/config');
	return response as ConsensusConfig;
}
