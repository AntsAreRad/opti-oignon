/**
 * One chat WebSocket, read to its end.
 *
 * Every frame goes to `onFrame` first (the loader reads the whole stream
 * there), then to its own callback; `done` and `error` end the stream. A
 * connection lost before the first frame is retried with a backoff. A Stop
 * sends the cancel first and keeps reading: the server closes its open steps
 * and sends `done`, and only then does the socket close, or after
 * `STOP_WAIT_MS` without an answer. A stream that ends without `done` or
 * `error` calls `onLost`.
 *
 * It holds no dependency: the socket, the timers and the random source are
 * handed in, so it runs under Node over a fake socket. `lib/api/chat.ts`
 * hands it the browser's.
 */

import type {
	ChatResponse,
	ChatStreamCallbacks,
	ChatToken,
	PipelineStepFrame,
	ReasoningMetaInfo,
	ReasoningStepInfo,
	ToolCallInfo,
	VerificationInfo,
} from '$lib/types';

export const MAX_RETRIES = 3;
export const BASE_DELAY_MS = 500;
export const MAX_DELAY_MS = 4000;
/** How long a Stop waits for the server's closing frames before it closes the socket. */
export const STOP_WAIT_MS = 10000;

const CONNECTING = 0;
const OPEN = 1;
const NORMAL_CLOSURE = 1000;

export interface SocketLike {
	readonly readyState: number;
	send(data: string): void;
	close(): void;
	onopen: ((event?: unknown) => void) | null;
	onmessage: ((event: { data: string }) => void) | null;
	onerror: ((event?: unknown) => void) | null;
	onclose: ((event: { code: number }) => void) | null;
}

export interface StreamDeps {
	open: () => SocketLike;
	later: (run: () => void, ms: number) => unknown;
	cancelLater: (handle: unknown) => void;
	random: () => number;
}

export interface StreamOptions {
	payload: unknown;
	conversationId: string;
	/** The error said when the request cannot be sent. */
	sendError: string;
}

/** How a stream ended: by its frames, by a lost connection, or by a Stop nobody answered. */
export type StreamEnd = 'done' | 'error' | 'lost' | 'unanswered';

export interface ChatConnection {
	/** Closes at once and reads nothing more. */
	cancel: () => void;
	/** Sends `request` (the cancel), then reads until the stream ends. */
	stop: (request: () => Promise<void>) => Promise<StreamEnd>;
	readonly socket: SocketLike | null;
}

/** Exponential backoff with jitter. */
export function backoffDelay(attempt: number, random: () => number): number {
	const base = Math.min(BASE_DELAY_MS * Math.pow(2, attempt), MAX_DELAY_MS);
	return base + random() * base * 0.3;
}

/** The reply a `done` frame describes. */
export function responseOf(data: ChatToken, conversationId: string): ChatResponse {
	const meta = data.metadata ?? {};
	return {
		conversation_id: (meta.conversation_id as string) ?? conversationId,
		message_id: (meta.message_id as number) ?? null,
		content: data.content ?? '',
		model: (meta.model as string) ?? '',
		tokens: (meta.tokens as number) ?? 0,
		duration_ms: (meta.duration_ms as number) ?? 0,
		cancelled: meta.cancelled === true,
		steps: Array.isArray(meta.steps) ? (meta.steps as PipelineStepFrame[]) : undefined,
		// Quick sandbox metadata
		sandbox_active: (meta.sandbox_active as boolean) ?? false,
		sandbox_session_id: (meta.sandbox_session_id as string) ?? undefined,
		sandbox_files: (meta.sandbox_files as unknown[]) ?? undefined,
		sandbox_files_created: (meta.sandbox_files_created as string[]) ?? undefined,
		// Chat coding agent metadata
		chat_coding: (meta.chat_coding as boolean) ?? false,
		coding_result: (meta.coding_result as Record<string, unknown>) ?? undefined,
		turn_count: (meta.turn_count as number) ?? undefined,
	};
}

/** Hands one frame to its callbacks; says whether it ends the stream. */
export function dispatch(
	data: ChatToken,
	callbacks: ChatStreamCallbacks,
	conversationId: string,
): 'done' | 'error' | null {
	callbacks.onFrame?.(data);
	const meta = data.metadata;
	switch (data.type) {
		case 'token':
			callbacks.onToken(data.content);
			return null;
		case 'thinking':
			callbacks.onThinking?.(data.content);
			return null;
		case 'done':
			callbacks.onDone(responseOf(data, conversationId));
			return 'done';
		case 'error':
			callbacks.onError(data.content);
			return 'error';
		case 'coding_error':
			callbacks.onError(data.content || 'Coding agent error');
			return 'error';
		case 'metadata':
			callbacks.onMetadata?.(meta ?? {});
			return null;
		case 'verification':
			if (meta) callbacks.onVerification?.(meta as unknown as VerificationInfo);
			return null;
		case 'tool_call':
			if (meta) callbacks.onToolCall?.(meta as unknown as ToolCallInfo);
			return null;
		case 'reasoning_step':
			if (meta) callbacks.onReasoningStep?.(meta as unknown as ReasoningStepInfo);
			return null;
		case 'reasoning_done':
			if (meta) callbacks.onReasoningDone?.(meta as unknown as ReasoningMetaInfo);
			return null;
		case 'vision_delegation':
			callbacks.onVisionDelegation?.(meta ?? {});
			return null;
		case 'status':
			callbacks.onStatus?.((meta?.message as string) ?? '');
			return null;
		case 'coding_plan':
		case 'coding_step':
		case 'coding_test':
		case 'coding_fix':
		case 'coding_done':
		case 'coding_status':
			callbacks.onCodingEvent?.(data.type, meta ?? {});
			return null;
		case 'pipeline_step':
		case 'ping':
		case 'tool_call_pending':
		case 'tool_call_resolved':
			// Read by the loader through onFrame.
			return null;
		default:
			// A frame type this client does not know is ignored.
			return null;
	}
}

/** Opens the stream and reads it to its end. */
export function openStream(
	options: StreamOptions,
	callbacks: ChatStreamCallbacks,
	deps: StreamDeps,
): ChatConnection {
	let closed = false;
	let ended: StreamEnd | null = null;
	let retryCount = 0;
	let hasReceivedData = false;
	let current: SocketLike | null = null;
	let stopping: Promise<StreamEnd> | null = null;
	let settle: ((end: StreamEnd) => void) | null = null;
	let waitHandle: unknown = null;

	function finish(end: StreamEnd) {
		if (closed) return;
		closed = true;
		ended = end;
		if (waitHandle !== null) {
			deps.cancelLater(waitHandle);
			waitHandle = null;
		}
		if (current && (current.readyState === OPEN || current.readyState === CONNECTING)) {
			try {
				current.close();
			} catch {
				/* already closing */
			}
		}
		if (settle) {
			const resolve = settle;
			settle = null;
			resolve(end);
		}
	}

	function lost(message: string) {
		callbacks.onLost?.();
		// A Stop's own end is not an error: the reader asked for it.
		if (!stopping) callbacks.onError(message);
		finish('lost');
	}

	function retry(): boolean {
		if (hasReceivedData || stopping || retryCount >= MAX_RETRIES) return false;
		retryCount++;
		callbacks.onReconnecting?.(retryCount, MAX_RETRIES);
		deps.later(() => {
			if (!closed) connect();
		}, backoffDelay(retryCount, deps.random));
		return true;
	}

	function connect() {
		const socket = deps.open();
		current = socket;

		socket.onopen = () => {
			try {
				socket.send(JSON.stringify(options.payload));
			} catch {
				callbacks.onError(options.sendError);
				finish('error');
			}
		};

		socket.onmessage = (event) => {
			if (closed) return;
			hasReceivedData = true;
			retryCount = 0;
			try {
				const end = dispatch(JSON.parse(event.data) as ChatToken, callbacks, options.conversationId);
				if (end) finish(end);
			} catch {
				callbacks.onError('Failed to parse server message');
				finish('error');
			}
		};

		socket.onerror = () => {
			if (closed) return;
			if (!retry()) lost('WebSocket connection error');
		};

		socket.onclose = (event) => {
			if (closed) return;
			if (event.code === NORMAL_CLOSURE) {
				lost('Connection closed before the reply ended.');
				return;
			}
			if (!retry()) lost(`Connection lost (code: ${event.code}). Please retry.`);
		};
	}

	function stop(request: () => Promise<void>): Promise<StreamEnd> {
		if (stopping) return stopping;
		if (closed) return Promise.resolve(ended ?? 'lost');
		stopping = new Promise<StreamEnd>((resolve) => {
			settle = resolve;
		});
		waitHandle = deps.later(() => {
			waitHandle = null;
			if (closed) return;
			callbacks.onLost?.();
			finish('unanswered');
		}, STOP_WAIT_MS);
		try {
			request().catch(() => {
				/* the reply may be over already; its own frames still end the stop */
			});
		} catch {
			/* same: the stream's frames or the wait end the stop */
		}
		return stopping;
	}

	connect();

	return {
		cancel: () => finish('lost'),
		stop,
		get socket() {
			return current;
		},
	};
}
