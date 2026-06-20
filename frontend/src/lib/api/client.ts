/**
 * Base fetch wrapper for the Opti-Oignon API.
 *
 * Provides typed HTTP methods with consistent error handling.
 * API_BASE is configurable via VITE_API_URL env variable.
 * In dev mode, Vite proxy handles /api -> localhost:8001.
 *
 * S87: Actionable error messages distinguishing network errors
 * (backend down, timeout) from API errors (400/404/500).
 */

const API_BASE = import.meta.env.VITE_API_URL ?? '';

/**
 * S98: Module-level access token for Authorization header injection.
 * Set by the auth store on login/refresh, cleared on logout.
 */
let _accessToken: string | null = null;

/** Set the current access token (called by auth store). */
export function setAccessToken(token: string | null): void {
	_accessToken = token;
}

/** Get the current access token. */
export function getAccessToken(): string | null {
	return _accessToken;
}

/** Build auth headers if a token is set. */
function authHeaders(): Record<string, string> {
	if (_accessToken) {
		return { 'Authorization': `Bearer ${_accessToken}` };
	}
	return {};
}

/**
 * S125 hardening: Read CSRF token from cookie and return as header.
 * The oo_csrf_token cookie is set by the backend (non-httpOnly)
 * and must be echoed back as X-CSRF-Token for state-changing requests.
 */
function csrfHeader(): Record<string, string> {
	try {
		const match = document.cookie.match(/(?:^|;\s*)oo_csrf_token=([^;]+)/);
		if (match && match[1]) {
			return { 'X-CSRF-Token': decodeURIComponent(match[1]) };
		}
	} catch {
		// SSR or cookie unavailable
	}
	return {};
}

export class ApiError extends Error {
	status: number;
	detail: string;
	/** True if the error is a network/connection issue, false for HTTP status errors. */
	isNetworkError: boolean;

	constructor(status: number, message: string, detail: string = '', isNetworkError: boolean = false) {
		super(message);
		this.name = 'ApiError';
		this.status = status;
		this.detail = detail || message;
		this.isNetworkError = isNetworkError;
	}
}

/**
 * S87: Build a user-friendly, actionable error message from an HTTP status code.
 */
function actionableMessage(status: number, serverDetail: string, path: string): string {
	const resource = path.split('/').filter(Boolean).slice(1, 3).join('/') || 'resource';

	switch (status) {
		case 400:
			return `Bad request for ${resource}: ${serverDetail}. Check your input and try again.`;
		case 401:
			return `Authentication required for ${resource}. Check your API configuration.`;
		case 403:
			return `Access denied for ${resource}. You may not have permission for this action.`;
		case 404:
			return `${capitalize(resource)} not found. It may have been deleted or the URL is incorrect.`;
		case 409:
			return `Conflict updating ${resource}: ${serverDetail}. Try refreshing first.`;
		case 422:
			return `Invalid data for ${resource}: ${serverDetail}. Check your input values.`;
		case 429:
			return `Too many requests. Please wait a moment and try again.`;
		case 500:
			return `Server error processing ${resource}. The backend encountered an internal error.`;
		case 502:
			return `Bad gateway. The backend may be restarting. Try again in a few seconds.`;
		case 503:
			return `Service unavailable. The backend is temporarily overloaded or down.`;
		default:
			if (status >= 500) {
				return `Server error (${status}) for ${resource}: ${serverDetail}`;
			}
			return `Request failed (${status}) for ${resource}: ${serverDetail}`;
	}
}

/**
 * S87: Build a user-friendly message for network-level failures (no HTTP response).
 */
function networkErrorMessage(err: unknown): string {
	if (err instanceof TypeError) {
		const msg = err.message.toLowerCase();
		if (msg.includes('failed to fetch') || msg.includes('network')) {
			return 'Unable to reach the backend. Is the server running? Check that the API is started (launch.sh).';
		}
		if (msg.includes('abort') || msg.includes('timeout')) {
			return 'Request timed out. The backend may be overloaded or unresponsive.';
		}
	}
	if (err instanceof DOMException && err.name === 'AbortError') {
		return 'Request was cancelled or timed out.';
	}
	return 'Connection failed. Unable to reach the API server. Is the backend running?';
}

function capitalize(s: string): string {
	return s.charAt(0).toUpperCase() + s.slice(1);
}

/**
 * Public auth endpoints whose own 401 is a meaningful business response
 * (wrong password, unauthenticated status check) and must NOT trigger a
 * redirect to the login page. These mirror the backend auth middleware
 * allowlist.
 */
const AUTH_PATHS_NO_REDIRECT = [
	'/api/auth/login',
	'/api/auth/register',
	'/api/auth/refresh',
	'/api/auth/status',
	'/api/auth/2fa-challenge',
	'/api/auth/me',
];

/** Guards against firing multiple navigations when several requests 401 at once. */
let _redirectingToLogin = false;

/**
 * On a 401 from a protected endpoint, send the user to the login page instead
 * of surfacing a raw "API error 401" and leaving the UI in a broken state.
 *
 * This is the recovery path when authentication becomes required mid-session,
 * for example switching to Bulbe mode, which enforces auth even for a
 * single-user install. Requests to the public auth endpoints are exempt: their
 * 401 is a response the caller must handle, not a reason to navigate away. A
 * full-page navigation is used deliberately so the auth stores re-initialise
 * from a clean state.
 */
function maybeRedirectToLogin(status: number, path: string): void {
	if (status !== 401) return;
	if (typeof window === 'undefined') return;
	if (_redirectingToLogin) return;
	if (AUTH_PATHS_NO_REDIRECT.some((p) => path.startsWith(p))) return;
	const current = window.location.pathname;
	if (current.startsWith('/login') || current.startsWith('/register')) return;
	_redirectingToLogin = true;
	window.location.assign('/login');
}

async function handleResponse<T>(response: Response, path: string): Promise<T> {
	if (!response.ok) {
		let serverDetail = response.statusText;
		try {
			const body = await response.json();
			serverDetail = body.detail || body.message || serverDetail;
		} catch {
			// No JSON body - keep statusText
		}
		maybeRedirectToLogin(response.status, path);
		const detail = actionableMessage(response.status, serverDetail, path);
		throw new ApiError(response.status, `API error ${response.status}`, detail, false);
	}

	// 204 No Content
	if (response.status === 204) {
		return undefined as T;
	}

	return response.json() as Promise<T>;
}

function buildUrl(path: string, params?: Record<string, string>): string {
	const url = new URL(`${API_BASE}${path}`, window.location.origin);
	if (params) {
		for (const [key, value] of Object.entries(params)) {
			if (value !== undefined && value !== null && value !== '') {
				url.searchParams.set(key, value);
			}
		}
	}
	return url.toString();
}

function handleNetworkError(err: unknown, path: string): never {
	if (err instanceof ApiError) throw err;
	const detail = networkErrorMessage(err);
	throw new ApiError(0, 'Connection failed', detail, true);
}

export async function apiGet<T>(path: string, params?: Record<string, string>): Promise<T> {
	try {
		const response = await fetch(buildUrl(path, params), {
			method: 'GET',
			headers: { 'Accept': 'application/json', ...authHeaders() },
			credentials: 'include',
		});
		return handleResponse<T>(response, path);
	} catch (err) {
		return handleNetworkError(err, path);
	}
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
	try {
		const response = await fetch(buildUrl(path), {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Accept': 'application/json',
				...authHeaders(),
				...csrfHeader(),
			},
			body: body !== undefined ? JSON.stringify(body) : undefined,
			credentials: 'include',
		});
		return handleResponse<T>(response, path);
	} catch (err) {
		return handleNetworkError(err, path);
	}
}

export async function apiPut<T>(path: string, body?: unknown): Promise<T> {
	try {
		const response = await fetch(buildUrl(path), {
			method: 'PUT',
			headers: {
				'Content-Type': 'application/json',
				'Accept': 'application/json',
				...authHeaders(),
				...csrfHeader(),
			},
			body: body !== undefined ? JSON.stringify(body) : undefined,
			credentials: 'include',
		});
		return handleResponse<T>(response, path);
	} catch (err) {
		return handleNetworkError(err, path);
	}
}

export async function apiPatch<T>(path: string, body?: unknown): Promise<T> {
	try {
		const response = await fetch(buildUrl(path), {
			method: 'PATCH',
			headers: {
				'Content-Type': 'application/json',
				'Accept': 'application/json',
				...authHeaders(),
				...csrfHeader(),
			},
			body: body !== undefined ? JSON.stringify(body) : undefined,
			credentials: 'include',
		});
		return handleResponse<T>(response, path);
	} catch (err) {
		return handleNetworkError(err, path);
	}
}

export async function apiDelete<T>(path: string): Promise<T> {
	try {
		const response = await fetch(buildUrl(path), {
			method: 'DELETE',
			headers: { 'Accept': 'application/json', ...authHeaders(), ...csrfHeader() },
			credentials: 'include',
		});
		return handleResponse<T>(response, path);
	} catch (err) {
		return handleNetworkError(err, path);
	}
}

/**
 * S211: multipart upload helper.
 *
 * Sends a FormData body (drag-and-drop file uploads). Content-Type is
 * deliberately NOT set so the browser writes the multipart boundary itself;
 * auth, CSRF, credentials and error handling match the JSON helpers above.
 */
export async function apiUpload<T>(path: string, formData: FormData): Promise<T> {
	try {
		const response = await fetch(buildUrl(path), {
			method: 'POST',
			headers: { 'Accept': 'application/json', ...authHeaders(), ...csrfHeader() },
			body: formData,
			credentials: 'include',
		});
		return handleResponse<T>(response, path);
	} catch (err) {
		return handleNetworkError(err, path);
	}
}

/**
 * S171: Formalized `fetchApi` shim.
 *
 * Several security API modules (securityMode, toolCallApproval,
 * pluginAllowlist, searchKillSwitch) call a `fetchApi(path, options?)` helper
 * with a `fetch`-like signature. It was imported from this module but never
 * exported, so those calls resolved to `undefined` at runtime and `npm run
 * check` flagged the missing export. This shim restores the contract by
 * routing through the typed apiGet / apiPost / apiPut / apiPatch / apiDelete
 * helpers, inheriting their auth header, CSRF, and error handling.
 *
 * The `body` option follows the callers' convention of an already-stringified
 * JSON payload; it is parsed back to an object before being handed to the
 * method helpers (which serialize it themselves), avoiding double-encoding.
 */
export interface FetchApiOptions {
	method?: string;
	body?: string;
}

export async function fetchApi<T>(path: string, options?: FetchApiOptions): Promise<T> {
	const method = (options?.method ?? 'GET').toUpperCase();
	let parsedBody: unknown;
	if (options?.body !== undefined) {
		try {
			parsedBody = JSON.parse(options.body);
		} catch {
			parsedBody = options.body;
		}
	}
	switch (method) {
		case 'POST':
			return apiPost<T>(path, parsedBody);
		case 'PUT':
			return apiPut<T>(path, parsedBody);
		case 'PATCH':
			return apiPatch<T>(path, parsedBody);
		case 'DELETE':
			return apiDelete<T>(path);
		case 'GET':
		default:
			return apiGet<T>(path);
	}
}

/**
 * Build a WebSocket URL for a given path.
 */
export function wsUrl(path: string): string {
	const base = API_BASE || window.location.origin;
	return base.replace(/^http/, 'ws') + path;
}

/**
 * S171: WebSocket client with automatic reconnect and exponential backoff.
 *
 * Wraps the native WebSocket for long-lived status/progress streams. On an
 * unexpected close it reconnects with exponentially increasing delay starting
 * at 1s and capped at 30s (1, 2, 4, 8, 16, 30, 30, ...), with optional jitter
 * to avoid thundering-herd reconnects. A clean local close() stops reconnects.
 *
 * Usage:
 *   const ws = new ReconnectingWebSocket(wsUrl('/api/.../progress'));
 *   ws.onmessage = (e) => { ... };
 *   ws.onopen = () => { ... };
 *   ws.close(); // stops reconnecting
 */
export const WS_BACKOFF_MIN_MS = 1000;
export const WS_BACKOFF_MAX_MS = 30000;

export class ReconnectingWebSocket {
	private url: string;
	private protocols?: string | string[];
	private socket: WebSocket | null = null;
	private attempts = 0;
	private timer: ReturnType<typeof setTimeout> | null = null;
	private closedByClient = false;
	private readonly jitter: boolean;

	onopen: ((ev: Event) => void) | null = null;
	onmessage: ((ev: MessageEvent) => void) | null = null;
	onerror: ((ev: Event) => void) | null = null;
	/** Fired only on the final, client-initiated close. */
	onclose: ((ev: CloseEvent) => void) | null = null;
	/** Fired before each scheduled reconnect attempt, with the delay (ms). */
	onreconnect: ((delayMs: number) => void) | null = null;

	constructor(url: string, protocols?: string | string[], jitter = true) {
		this.url = url;
		this.protocols = protocols;
		this.jitter = jitter;
		this.connect();
	}

	/** Current backoff delay for the next attempt (exponential, capped). */
	private nextDelay(): number {
		const exp = Math.min(WS_BACKOFF_MAX_MS, WS_BACKOFF_MIN_MS * 2 ** this.attempts);
		if (!this.jitter) return exp;
		// Full jitter in [exp/2, exp] to spread reconnects.
		return Math.round(exp / 2 + Math.random() * (exp / 2));
	}

	private connect(): void {
		try {
			this.socket = this.protocols
				? new WebSocket(this.url, this.protocols)
				: new WebSocket(this.url);
		} catch (err) {
			this.scheduleReconnect();
			return;
		}

		this.socket.onopen = (ev) => {
			this.attempts = 0;
			if (this.onopen) this.onopen(ev);
		};
		this.socket.onmessage = (ev) => {
			if (this.onmessage) this.onmessage(ev);
		};
		this.socket.onerror = (ev) => {
			if (this.onerror) this.onerror(ev);
		};
		this.socket.onclose = (ev) => {
			if (this.closedByClient) {
				if (this.onclose) this.onclose(ev);
				return;
			}
			this.scheduleReconnect();
		};
	}

	private scheduleReconnect(): void {
		if (this.closedByClient) return;
		const delay = this.nextDelay();
		this.attempts += 1;
		if (this.onreconnect) this.onreconnect(delay);
		this.timer = setTimeout(() => this.connect(), delay);
	}

	/** Send data if the socket is open; returns false otherwise. */
	send(data: string | ArrayBufferLike | Blob | ArrayBufferView): boolean {
		if (this.socket && this.socket.readyState === WebSocket.OPEN) {
			this.socket.send(data);
			return true;
		}
		return false;
	}

	/** Close permanently and stop reconnecting. */
	close(code?: number, reason?: string): void {
		this.closedByClient = true;
		if (this.timer !== null) {
			clearTimeout(this.timer);
			this.timer = null;
		}
		if (this.socket) {
			try {
				this.socket.close(code, reason);
			} catch {
				// already closing/closed
			}
		}
	}

	/** Expose the underlying readyState (CONNECTING/OPEN/CLOSING/CLOSED). */
	get readyState(): number {
		return this.socket ? this.socket.readyState : WebSocket.CLOSED;
	}
}
