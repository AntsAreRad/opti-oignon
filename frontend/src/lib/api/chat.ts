/**
 * The chat stream's client: the browser's WebSocket handed to the stream
 * reader (`lib/chat/stream.ts`), one ephemeral connection per message.
 *
 * Both paths read the same frames the same way: `/api/chat/stream` sends a
 * message, `/api/chat/retry` regenerates the last reply.
 */

import { wsUrl } from './client';
import { apiPost } from './client';
import type { ChatRequest, ChatStreamCallbacks, ChatRetryRequest } from '$lib/types';
import { openStream, type ChatConnection, type SocketLike, type StreamDeps } from '$lib/chat/stream';

export type { ChatConnection };

/** The browser's socket, timers and random source for one endpoint. */
function browserDeps(path: string): StreamDeps {
	return {
		open: () => new WebSocket(wsUrl(path)) as unknown as SocketLike,
		later: (run, ms) => setTimeout(run, ms),
		cancelLater: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
		random: Math.random,
	};
}

/** Streams the reply to one message through /api/chat/stream. */
export function streamChat(
	request: ChatRequest,
	callbacks: ChatStreamCallbacks
): ChatConnection {
	return openStream(
		{ payload: request, conversationId: request.conversation_id ?? '', sendError: 'Failed to send chat request' },
		callbacks,
		browserDeps('/api/chat/stream'),
	);
}

/** Regenerates the last reply through /api/chat/retry, streamed the same way. */
export function retryChat(
	conversationId: string,
	callbacks: ChatStreamCallbacks
): ChatConnection {
	const retryReq: ChatRetryRequest = { conversation_id: conversationId };
	return openStream(
		{ payload: retryReq, conversationId, sendError: 'Failed to send retry request' },
		callbacks,
		browserDeps('/api/chat/retry'),
	);
}

/**
 * Annule la generation en cours pour une conversation via POST.
 */
export async function cancelGeneration(conversationId: string): Promise<void> {
	await apiPost('/api/chat/cancel', { conversation_id: conversationId });
}


// -- Consensus --

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
