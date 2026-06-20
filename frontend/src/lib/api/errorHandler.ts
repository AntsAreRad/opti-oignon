/**
 * errorHandler.ts — Standardized API error display (S135).
 *
 * Parses errors from the API client layer (ApiError) and maps them
 * to user-friendly toast messages. Detects specific conditions like
 * rate limiting (429), network offline, and feature unavailability (501).
 *
 * Usage:
 *   import { handleApiError } from '$lib/api/errorHandler';
 *   try { await someApiCall(); } catch (e) { handleApiError(e); }
 *   // or with a custom context:
 *   try { ... } catch (e) { handleApiError(e, 'saving preset'); }
 */

import { ApiError } from './client';
import { addToast } from '$lib/stores/notifications';

/** Parsed error information for display or programmatic handling. */
export interface ParsedApiError {
	/** Short human-readable message for the user. */
	message: string;
	/** HTTP status code, or 0 for network errors. */
	status: number;
	/** True if the device appears to be offline. */
	isOffline: boolean;
	/** True if the server returned 429 Too Many Requests. */
	isRateLimited: boolean;
	/** True if the feature returned 501 Not Implemented. */
	isFeatureUnavailable: boolean;
	/** True if the error is a network-level failure (no HTTP response). */
	isNetworkError: boolean;
	/** True when retrying may succeed: offline, network failure, 429, or 5xx
	 *  (spec 6.9). 501 feature-unavailable is NOT retriable. */
	retriable: boolean;
}

/**
 * Parse any caught error into a structured ParsedApiError.
 * Handles ApiError, standard Error, and unknown throw values.
 */
export function parseApiError(err: unknown, context?: string): ParsedApiError {
	const ctx = context ? ` while ${context}` : '';

	// Detect browser offline state
	if (typeof navigator !== 'undefined' && !navigator.onLine) {
		return {
			message: `You appear to be offline${ctx}. Check your network connection.`,
			status: 0,
			isOffline: true,
			isRateLimited: false,
			isFeatureUnavailable: false,
			isNetworkError: true,
			retriable: true,
		};
	}

	if (err instanceof ApiError) {
		const base: ParsedApiError = {
			message: err.detail || err.message,
			status: err.status,
			isOffline: false,
			isRateLimited: err.status === 429,
			isFeatureUnavailable: err.status === 501,
			isNetworkError: err.isNetworkError,
			retriable: err.status === 429 || err.isNetworkError || err.status >= 500,
		};

		// Specific status overrides
		if (err.status === 429) {
			base.message = `Rate limit exceeded${ctx}. Please wait a moment and try again.`;
		} else if (err.status === 501) {
			base.message = `This feature is not available${ctx}. The required backend module may not be installed.`;
		} else if (err.isNetworkError) {
			base.message = err.detail || `Connection failed${ctx}. Is the backend running?`;
		}

		return base;
	}

	if (err instanceof Error) {
		return {
			message: err.message || `An unexpected error occurred${ctx}.`,
			status: 0,
			isOffline: false,
			isRateLimited: false,
			isFeatureUnavailable: false,
			isNetworkError: false,
			retriable: false,
		};
	}

	return {
		message: `An unexpected error occurred${ctx}.`,
		status: 0,
		isOffline: false,
		isRateLimited: false,
		isFeatureUnavailable: false,
		isNetworkError: false,
		retriable: false,
	};
}

/**
 * Parse and display an API error as a toast notification.
 * Uses warning level for rate limits and feature unavailability,
 * error level for everything else.
 *
 * @param err - The caught error.
 * @param context - Optional verb phrase, e.g. "loading models".
 * @param retry - Optional async retry; attached as a toast action when the
 *   error is retriable (spec 6.9). Prefer an inline <InlineError onRetry>
 *   anchored to the failing control when one is available.
 * @returns The parsed error for further handling if needed.
 */
export function handleApiError(
	err: unknown,
	context?: string,
	retry?: () => Promise<void>
): ParsedApiError {
	const parsed = parseApiError(err, context);

	const action = retry && parsed.retriable ? { label: 'Retry', run: retry } : undefined;

	if (parsed.isRateLimited || parsed.isFeatureUnavailable) {
		addToast(parsed.message, 'warning', 5000, { action });
	} else {
		addToast(parsed.message, 'error', 8000, { action });
	}

	return parsed;
}

/**
 * Utility: wrap an async function with standardized error handling.
 * Returns the result or undefined if an error occurred (error is toasted).
 *
 * Usage:
 *   const data = await withErrorHandling(() => fetchSomething(), 'loading data');
 */
export async function withErrorHandling<T>(
	fn: () => Promise<T>,
	context?: string
): Promise<T | undefined> {
	try {
		return await fn();
	} catch (err) {
		handleApiError(err, context);
		return undefined;
	}
}
