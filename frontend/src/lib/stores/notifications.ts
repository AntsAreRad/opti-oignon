/**
 * Toast notification store + persistent notification history.
 *
 * Manages a notification stack with configurable auto-dismiss,
 * plus a persistent history of the last N notifications accessible
 * via the NotificationCenter panel.
 *
 * Max visible toasts (oldest auto-dismissed), error duration 8s,
 * dismissing state for exit animations, programmatic dismissById.
 */

import { writable, derived, get } from 'svelte/store';

export type ToastType = 'info' | 'success' | 'error' | 'warning';

/** Optional retry/undo action rendered inside a toast. */
export interface ToastAction {
	label: string;
	run: () => Promise<void>;
}

export interface ToastItem {
	id: string;
	type: ToastType;
	message: string;
	duration: number;
	/** True when the toast is animating out. */
	dismissing?: boolean;
	/** Optional bold first line. */
	title?: string;
	/** Optional action button (e.g. retry); dismisses on success. */
	action?: ToastAction;
	/** Whether the toast can be dismissed by the user (default true). */
	dismissible?: boolean;
}

export interface NotificationItem {
	id: string;
	type: ToastType;
	message: string;
	timestamp: number;
	read: boolean;
}

/** Active toast stack (ephemeral, auto-dismiss). */
export const toasts = writable<ToastItem[]>([]);

/** Persistent notification history (last N items). */
export const notificationHistory = writable<NotificationItem[]>([]);

/** Maximum number of notifications to keep in history. */
const MAX_HISTORY = 50;

/** Maximum number of visible toasts; oldest auto-dismissed when exceeded. */
const MAX_VISIBLE_TOASTS = 3;

/** Exit animation duration in ms (must match CSS). */
const EXIT_ANIMATION_MS = 200;

let counter = 0;

/** Track active auto-dismiss timers for programmatic cancellation. */
const _timers: Map<string, ReturnType<typeof setTimeout>> = new Map();

/** Extra options for richer toasts (title, retry action, dismissible). */
export interface ToastOptions {
	title?: string;
	action?: ToastAction;
	dismissible?: boolean;
}

/** Show a toast notification and add to history. Returns the toast ID. */
export function addToast(
	message: string,
	type: ToastType = 'info',
	duration: number = 4000,
	options: ToastOptions = {}
): string {
	const id = `toast-${++counter}-${Date.now()}`;
	const toast: ToastItem = {
		id,
		type,
		message,
		duration,
		dismissing: false,
		title: options.title,
		action: options.action,
		dismissible: options.dismissible ?? true,
	};

	toasts.update((list) => {
		const updated = [...list, toast];
		// Evict oldest non-dismissing toasts if we exceed the limit
		while (updated.filter((t) => !t.dismissing).length > MAX_VISIBLE_TOASTS) {
			const oldest = updated.find((t) => !t.dismissing);
			if (oldest) {
				// Remove immediately (no exit animation for evicted toasts)
				const idx = updated.indexOf(oldest);
				updated.splice(idx, 1);
				// Cancel its timer
				const timer = _timers.get(oldest.id);
				if (timer) {
					clearTimeout(timer);
					_timers.delete(oldest.id);
				}
			} else {
				break;
			}
		}
		return updated;
	});

	// Also add to persistent history
	const notification: NotificationItem = {
		id,
		type,
		message,
		timestamp: Date.now(),
		read: false,
	};
	notificationHistory.update((list) => {
		const updated = [notification, ...list];
		return updated.slice(0, MAX_HISTORY);
	});

	if (duration > 0) {
		const timer = setTimeout(() => dismissToast(id), duration);
		_timers.set(id, timer);
	}

	return id;
}

/**
 * Dismiss a toast with exit animation.
 * Sets dismissing=true, then removes after animation completes.
 */
export function dismissToast(id: string): void {
	// Cancel any pending auto-dismiss timer
	const timer = _timers.get(id);
	if (timer) {
		clearTimeout(timer);
		_timers.delete(id);
	}

	// Mark as dismissing for exit animation
	toasts.update((list) =>
		list.map((t) => (t.id === id ? { ...t, dismissing: true } : t))
	);

	// Remove after exit animation
	setTimeout(() => {
		toasts.update((list) => list.filter((t) => t.id !== id));
	}, EXIT_ANIMATION_MS);
}

/** Remove a toast by ID immediately (no animation, backward compat). */
export function removeToast(id: string): void {
	const timer = _timers.get(id);
	if (timer) {
		clearTimeout(timer);
		_timers.delete(id);
	}
	toasts.update((list) => list.filter((t) => t.id !== id));
}

/** Mark a notification as read by ID. */
export function markNotificationRead(id: string): void {
	notificationHistory.update((list) =>
		list.map((n) => (n.id === id ? { ...n, read: true } : n))
	);
}

/** Mark all notifications as read. */
export function markAllRead(): void {
	notificationHistory.update((list) =>
		list.map((n) => ({ ...n, read: true }))
	);
}

/** Clear all notification history. */
export function clearNotificationHistory(): void {
	notificationHistory.set([]);
}

/** Derived: count of unread notifications. */
export const unreadCount = derived(notificationHistory, ($history) =>
	$history.filter((n) => !n.read).length
);

/** Convenience shortcuts for each type. */
export function toastSuccess(message: string, duration?: number): string {
	return addToast(message, 'success', duration ?? 3000);
}

export function toastError(message: string, duration?: number): string {
	return addToast(message, 'error', duration ?? 8000);
}

export function toastWarning(message: string, duration?: number): string {
	return addToast(message, 'warning', duration ?? 5000);
}

export function toastInfo(message: string, duration?: number): string {
	return addToast(message, 'info', duration ?? 4000);
}
