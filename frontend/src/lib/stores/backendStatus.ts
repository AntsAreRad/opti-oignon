/**
 * The inference backend's state, as the status card shows it.
 *
 * Read from /api/backends once a minute, and only while the page is
 * visible: a hidden tab asks nothing, and showing it again reads at once.
 * Each read lists the backends (which costs the server a listing of its
 * models), so the minute is deliberate.
 *
 * Two failures stay apart. `reachable` is false when the API itself did not
 * answer (the server is down, or the network between); it is true when the
 * server answered, and then `backend` is the active backend with its own
 * health, so a server that answers while its backend is down reads as that
 * backend unavailable. `reachable` is null before the first answer.
 */

import { writable } from 'svelte/store';
import { apiGet } from '$lib/api/client';

export interface ActiveBackend {
	name: string;
	display_name: string;
	healthy: boolean;
	active: boolean;
	model_count: number | null;
}

export interface BackendState {
	/** Null before the first answer, false when the API itself failed. */
	reachable: boolean | null;
	/** The active backend, or null when none is active. */
	backend: ActiveBackend | null;
}

interface BackendList {
	backends: ActiveBackend[];
	active_backend: string | null;
}

const READ_EVERY_MS = 60_000;

let timer: ReturnType<typeof setTimeout> | null = null;
let watching = false;

function visible(): boolean {
	return typeof document !== 'undefined' && document.visibilityState === 'visible';
}

function stopTimer(): void {
	if (timer !== null) {
		clearTimeout(timer);
		timer = null;
	}
}

function scheduleNextRead(): void {
	stopTimer();
	if (!watching || !visible()) return;
	timer = setTimeout(() => {
		timer = null;
		void refreshBackendStatus();
	}, READ_EVERY_MS);
}

function onVisibilityChange(): void {
	if (document.hidden) stopTimer();
	else void refreshBackendStatus();
}

/** The backend's state. Subscribing in a browser starts the reads; the last unsubscribe stops them. */
export const backendStatus = writable<BackendState>({ reachable: null, backend: null }, () => {
	if (typeof document === 'undefined') return undefined;
	watching = true;
	document.addEventListener('visibilitychange', onVisibilityChange);
	if (visible()) void refreshBackendStatus();
	return () => {
		watching = false;
		document.removeEventListener('visibilitychange', onVisibilityChange);
		stopTimer();
	};
});

/** Reads the backends now. */
export async function refreshBackendStatus(): Promise<void> {
	try {
		const list = await apiGet<BackendList>('/api/backends');
		const backends = Array.isArray(list.backends) ? list.backends : [];
		const active =
			backends.find((backend) => backend.name === list.active_backend) ??
			backends.find((backend) => backend.active) ??
			null;
		backendStatus.set({ reachable: true, backend: active });
	} catch {
		backendStatus.update((state) => ({ ...state, reachable: false }));
	} finally {
		scheduleNextRead();
	}
}
