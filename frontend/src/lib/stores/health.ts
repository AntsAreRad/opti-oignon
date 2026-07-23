/**
 * Backend health polling store.
 *
 * Polls /api/health at a configurable interval and exposes:
 * - backendStatus: 'connected' | 'degraded' | 'disconnected'
 * - backendVersion: string (e.g. "2.0.1")
 * - backendModules: module availability map
 * - backendError: last error message or null
 */

import { writable, derived, get } from 'svelte/store';

export type BackendStatusValue = 'connected' | 'degraded' | 'disconnected';

interface HealthData {
	status: string;
	version: string;
	modules: Record<string, boolean>;
}

/** Raw health response data (null = never polled or disconnected). */
export const healthData = writable<HealthData | null>(null);

/** Last error message from polling, null if healthy. */
export const backendError = writable<string | null>(null);

/** Timestamp of last successful poll. */
export const lastHealthCheck = writable<number>(0);

/** Derived: backend version string. */
export const backendVersion = derived(healthData, ($d) => $d?.version ?? '');

/** Derived: module availability map. */
export const backendModules = derived(healthData, ($d) => $d?.modules ?? {});

/**
 * Derived: overall backend status.
 * - 'connected': healthy response, most modules available
 * - 'degraded': healthy response but many modules unavailable
 * - 'disconnected': cannot reach backend
 */
export const backendStatus = derived(
	[healthData, backendError],
	([$data, $error]): BackendStatusValue => {
		if ($data === null || $error !== null) return 'disconnected';
		const modules = $data.modules ?? {};
		const total = Object.keys(modules).length;
		if (total === 0) return 'connected';
		const available = Object.values(modules).filter(Boolean).length;
		// If less than 50% of modules are available, consider degraded
		if (available / total < 0.5) return 'degraded';
		return 'connected';
	}
);

const API_BASE = import.meta.env.VITE_API_URL ?? '';
const POLL_INTERVAL_MS = 15_000;
const POLL_TIMEOUT_MS = 5_000;

let pollTimer: ReturnType<typeof setInterval> | null = null;

/** Perform a single health check. */
async function pollHealth(): Promise<void> {
	try {
		const controller = new AbortController();
		const timeout = setTimeout(() => controller.abort(), POLL_TIMEOUT_MS);

		const url = `${API_BASE}/api/health`;
		const response = await fetch(url, {
			method: 'GET',
			headers: { 'Accept': 'application/json' },
			signal: controller.signal,
		});
		clearTimeout(timeout);

		if (!response.ok) {
			backendError.set(`HTTP ${response.status}`);
			healthData.set(null);
			return;
		}

		const data: HealthData = await response.json();
		healthData.set(data);
		backendError.set(null);
		lastHealthCheck.set(Date.now());
	} catch (err) {
		backendError.set(
			err instanceof DOMException && err.name === 'AbortError'
				? 'Backend timeout'
				: 'Backend unreachable'
		);
		healthData.set(null);
	}
}

/** Start periodic health polling. Safe to call multiple times. */
export function startHealthPolling(): void {
	if (pollTimer !== null) return;
	// Immediate first poll
	pollHealth();
	pollTimer = setInterval(pollHealth, POLL_INTERVAL_MS);
}

/** Stop health polling. */
export function stopHealthPolling(): void {
	if (pollTimer !== null) {
		clearInterval(pollTimer);
		pollTimer = null;
	}
}

/** Force an immediate health check (e.g. after reconnect). */
export function checkHealthNow(): void {
	pollHealth();
}
