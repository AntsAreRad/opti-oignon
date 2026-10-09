/**
 * The fields a chat request carries, and the two builders that fill them.
 *
 * Every field listed here is a field of the server's ChatRequest
 * (opti_oignon/api/schemas.py); the server drops any other key without a
 * word, so an option the composer shows and the request cannot carry would
 * promise an effect that never happens. The options builder turns the
 * composer's selections into those fields only, and the request builder
 * keeps only those fields, whatever it is handed.
 *
 * Pure and dependency-free: the chat stores call it, and it runs under Node
 * as it is.
 */

/** The ChatRequest fields the web chat fills. */
export const REQUEST_FIELDS = [
	'conversation_id',
	'message',
	'model',
	'preset',
	'temperature',
	'use_presets',
	'think',
	'web_search',
	'images',
	'documents',
	'pasted',
	'quick_sandbox',
	'chat_coding',
	'exec_pipeline',
] as const;

export type RequestField = (typeof REQUEST_FIELDS)[number];

/** What the composer has chosen for the next message. */
export interface ChatSelections {
	/** A model forced by hand; null lets the router choose. */
	model: string | null;
	/** A preset forced by hand; null lets the router choose. */
	preset: string | null;
	/** A temperature; null keeps the preset's or the model's. */
	temperature: number | null;
	/** Whether presets are detected automatically. */
	usePresets: boolean;
	/** Thinking forced on; off leaves the choice to the server. */
	think: boolean;
	/** Web search forced on; off leaves the choice to the server. */
	webSearch: boolean;
	/** The quick sandbox forced on; off leaves the server's default. */
	quickSandbox: boolean;
	/** The coding agent forced on; off leaves the server's default. */
	chatCoding: boolean;
	/** An execution pipeline to run; null is a plain chat. */
	execPipeline: string | null;
}

/** The options one message sends: request fields, each only when chosen. */
export interface ChatOptions {
	model?: string;
	preset?: string;
	temperature?: number;
	use_presets?: boolean;
	think?: boolean;
	web_search?: boolean;
	quick_sandbox?: boolean;
	chat_coding?: boolean;
	exec_pipeline?: string;
	images?: string[];
	/** The files attached to the message, each sent beside the typed words. */
	documents?: { filename: string; content: string }[];
	/** The ranges of the message the user pasted or dropped, [start, end) in code points. */
	pasted?: [number, number][];
}

/** The options the selections send: a field only when it carries a choice. */
export function chatOptions(selections: ChatSelections): ChatOptions {
	const options: ChatOptions = {};
	if (selections.model) options.model = selections.model;
	if (selections.preset) options.preset = selections.preset;
	if (typeof selections.temperature === 'number') options.temperature = selections.temperature;
	if (selections.usePresets === false) options.use_presets = false;
	if (selections.think) options.think = true;
	if (selections.webSearch) options.web_search = true;
	if (selections.quickSandbox) options.quick_sandbox = true;
	if (selections.chatCoding) options.chat_coding = true;
	if (selections.execPipeline) options.exec_pipeline = selections.execPipeline;
	return options;
}

/** A chat request: the conversation, the message, and the options it carries. */
export interface ChatRequestBody extends ChatOptions {
	conversation_id: string;
	message: string;
}

const CARRIED: ReadonlySet<string> = new Set(REQUEST_FIELDS);

/**
 * The request for one message: the conversation and the message from the
 * arguments, then every option that is a request field and holds a value.
 * Any other key is dropped; an empty image, document or pasted-range list is
 * not sent.
 */
export function chatRequest(
	conversationId: string,
	message: string,
	options: ChatOptions | Readonly<Record<string, unknown>> = {}
): ChatRequestBody {
	const request: Record<string, unknown> = { conversation_id: conversationId, message };
	for (const [key, value] of Object.entries(options)) {
		if (key === 'conversation_id' || key === 'message') continue;
		if (!CARRIED.has(key) || value === undefined || value === null) continue;
		if ((key === 'images' || key === 'documents' || key === 'pasted') && (!Array.isArray(value) || value.length === 0)) {
			continue;
		}
		request[key] = value;
	}
	return request as unknown as ChatRequestBody;
}
