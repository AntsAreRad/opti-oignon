/**
 * Authentication store (S98, S125).
 *
 * Manages current user state, token persistence, automatic token refresh,
 * and login/logout flows.
 *
 * S125: When cookie_mode is enabled (default), JWT tokens are stored in
 * httpOnly cookies set by the backend. localStorage is no longer used
 * for token storage, eliminating XSS token theft. The Authorization
 * header is still set as a fallback for backward compatibility.
 *
 * In single-user mode, the store remains in an "authenticated" state
 * with a synthetic local user, requiring no login.
 */

import { writable, derived, get } from 'svelte/store';
import { setAccessToken } from '$lib/api/client';
import * as authApi from '$lib/api/auth';
import type { AuthUser, AuthTokens, AuthStatus, UserSettings } from '$lib/types';

// ---------------------------------------------------------------------------
// Stores
// ---------------------------------------------------------------------------

/** Current authenticated user (null if not logged in). */
export const currentUser = writable<AuthUser | null>(null);

/** Auth system status. */
export const authStatus = writable<AuthStatus | null>(null);

/** Per-user settings. */
export const userSettings = writable<UserSettings | null>(null);

/** Whether the auth system is loading (initial check). */
export const authLoading = writable<boolean>(true);

/** Whether the user is authenticated. */
export const isAuthenticated = derived(currentUser, ($u) => $u !== null);

/** Whether the system is in single-user mode. */
export const isSingleUserMode = derived(authStatus, ($s) => $s?.single_user_mode ?? true);

// ---------------------------------------------------------------------------
// S125: Cookie mode flag
// ---------------------------------------------------------------------------

/** Whether the backend uses httpOnly cookie auth (S125). */
let _cookieMode = true;

// ---------------------------------------------------------------------------
// Token persistence (localStorage keys — legacy / non-cookie mode only)
// ---------------------------------------------------------------------------

const STORAGE_ACCESS_TOKEN = 'oo-access-token';
const STORAGE_REFRESH_TOKEN = 'oo-refresh-token';

function storeTokens(tokens: AuthTokens): void {
	// S125: In cookie mode the backend sets httpOnly cookies and the
	// Authorization Bearer header is NOT used. Bulbe mode enforces cookie-only
	// auth and rejects any Bearer header with 403, so keeping an in-memory
	// access token would break every request under Bulbe. Only retain the token
	// (and localStorage) in legacy non-cookie mode.
	if (!_cookieMode) {
		setAccessToken(tokens.access_token);
		try {
			localStorage.setItem(STORAGE_ACCESS_TOKEN, tokens.access_token);
			localStorage.setItem(STORAGE_REFRESH_TOKEN, tokens.refresh_token);
		} catch {
			// localStorage may be unavailable (SSR, privacy mode)
		}
	}
}

function clearTokens(): void {
	setAccessToken(null);

	if (!_cookieMode) {
		try {
			localStorage.removeItem(STORAGE_ACCESS_TOKEN);
			localStorage.removeItem(STORAGE_REFRESH_TOKEN);
		} catch {
			// Ignore
		}
	}
}

function getStoredAccessToken(): string | null {
	if (_cookieMode) return null;
	try {
		return localStorage.getItem(STORAGE_ACCESS_TOKEN);
	} catch {
		return null;
	}
}

function getStoredRefreshToken(): string | null {
	if (_cookieMode) return null;
	try {
		return localStorage.getItem(STORAGE_REFRESH_TOKEN);
	} catch {
		return null;
	}
}

/**
 * S125: Remove legacy localStorage tokens when migrating to cookie mode.
 * Called once on init if cookie_mode is detected and old tokens exist.
 */
function _migrateLegacyTokens(): void {
	try {
		if (
			localStorage.getItem(STORAGE_ACCESS_TOKEN) ||
			localStorage.getItem(STORAGE_REFRESH_TOKEN)
		) {
			localStorage.removeItem(STORAGE_ACCESS_TOKEN);
			localStorage.removeItem(STORAGE_REFRESH_TOKEN);
		}
	} catch {
		// Ignore
	}
}

// ---------------------------------------------------------------------------
// Token refresh timer
// ---------------------------------------------------------------------------

let refreshTimer: ReturnType<typeof setTimeout> | null = null;

function scheduleRefresh(expiresIn: number): void {
	if (refreshTimer) clearTimeout(refreshTimer);
	// Refresh 60 seconds before expiry (minimum 10s)
	const delay = Math.max((expiresIn - 60) * 1000, 10_000);
	refreshTimer = setTimeout(async () => {
		await doRefreshToken();
	}, delay);
}

function cancelRefresh(): void {
	if (refreshTimer) {
		clearTimeout(refreshTimer);
		refreshTimer = null;
	}
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/** Initialize auth state on app startup. */
export async function initAuth(): Promise<void> {
	authLoading.set(true);
	try {
		// Fetch system status
		const status = await authApi.getAuthStatus();
		authStatus.set(status);

		// S125: Detect cookie mode from backend status
		_cookieMode = status.cookie_mode ?? true;

		// S125: Clean up legacy localStorage tokens when switching to cookie mode
		if (_cookieMode) {
			_migrateLegacyTokens();
		}

		if (status.single_user_mode) {
			// Single-user mode normally bypasses authentication. Bulbe mode is
			// the exception (S136): it enforces auth even for a single-user
			// install. Probe a protected endpoint to tell the two apart — in
			// Daily the backend accepts the request (synthetic local user),
			// under Bulbe it rejects with 401. Installing a synthetic user when
			// auth is actually enforced would leave the login screen
			// unreachable, so only do it when the probe succeeds.
			let authEnforced = false;
			try {
				await authApi.getMe();
			} catch {
				authEnforced = true;
			}
			if (!authEnforced) {
				// Daily single-user: auth is bypassed, use a synthetic local user
				currentUser.set({
					user_id: 'local',
					username: 'local',
					email: '',
					role: 'admin',
					created_at: 0,
					updated_at: 0,
					metadata: {},
				});
				// Load settings for local user
				await loadUserSettings();
				return;
			}
			// Bulbe single-user: a real login is required; leave currentUser null
			// so the login screen is reachable.
			currentUser.set(null);
			return;
		}

		// Multi-user mode: try to restore session
		if (_cookieMode) {
			// S125: In cookie mode, just call /me — the httpOnly cookie
			// is sent automatically via credentials: 'include'
			try {
				const user = await authApi.getMe();
				currentUser.set(user);
				await loadUserSettings();
				// Schedule refresh in 30min (we don't know exact cookie expiry)
				scheduleRefresh(1800);
				return;
			} catch {
				// No valid cookie session, try refresh via cookie
				const refreshed = await doRefreshToken();
				if (refreshed) return;
			}
		} else {
			// Legacy mode: restore from localStorage
			const accessToken = getStoredAccessToken();
			if (accessToken) {
				setAccessToken(accessToken);
				try {
					const user = await authApi.getMe();
					currentUser.set(user);
					await loadUserSettings();
					scheduleRefresh(1800);
					return;
				} catch {
					const refreshed = await doRefreshToken();
					if (refreshed) return;
				}
			}
		}

		// No valid session
		currentUser.set(null);
	} catch {
		// Backend unreachable, assume single-user mode
		authStatus.set({
			available: false,
			single_user_mode: true,
			registration_enabled: false,
			user_count: 0,
		});
		currentUser.set({
			user_id: 'local',
			username: 'local',
			email: '',
			role: 'admin',
			created_at: 0,
			updated_at: 0,
			metadata: {},
		});
	} finally {
		authLoading.set(false);
	}
}

/** Register a new account. */
export async function doRegister(
	username: string,
	password: string,
	email = ''
): Promise<AuthUser> {
	const tokens = await authApi.register({ username, password, email });
	storeTokens(tokens);
	scheduleRefresh(tokens.expires_in);

	const user = await authApi.getMe();
	currentUser.set(user);
	await loadUserSettings();
	return user;
}

/** Log in with username and password. */
export async function doLogin(username: string, password: string): Promise<AuthUser> {
	const tokens = await authApi.login({ username, password });
	storeTokens(tokens);
	scheduleRefresh(tokens.expires_in);

	const user = await authApi.getMe();
	currentUser.set(user);
	await loadUserSettings();
	return user;
}

/** Refresh the access token using the stored refresh token. */
async function doRefreshToken(): Promise<boolean> {
	try {
		if (_cookieMode) {
			// S125: In cookie mode, refresh token is in httpOnly cookie.
			// Send empty string — backend reads cookie automatically.
			const tokens = await authApi.refreshToken('');
			storeTokens(tokens);
			scheduleRefresh(tokens.expires_in);
		} else {
			const refreshTok = getStoredRefreshToken();
			if (!refreshTok) return false;
			const tokens = await authApi.refreshToken(refreshTok);
			storeTokens(tokens);
			scheduleRefresh(tokens.expires_in);
		}

		const user = await authApi.getMe();
		currentUser.set(user);
		return true;
	} catch {
		clearTokens();
		currentUser.set(null);
		return false;
	}
}

/** Log out the current user. */
export async function doLogout(): Promise<void> {
	cancelRefresh();

	if (_cookieMode) {
		// S125: In cookie mode, backend reads refresh cookie and clears both.
		try {
			await authApi.logout('');
		} catch {
			// Best effort
		}
	} else {
		const refreshTok = getStoredRefreshToken();
		if (refreshTok) {
			try {
				await authApi.logout(refreshTok);
			} catch {
				// Best effort
			}
		}
	}

	clearTokens();
	currentUser.set(null);
	userSettings.set(null);
}

/** Load per-user settings. */
async function loadUserSettings(): Promise<void> {
	try {
		const settings = await authApi.getUserSettings();
		userSettings.set(settings);
	} catch {
		// Settings module may not be available
	}
}

/** Update per-user settings. */
export async function updateSettings(
	updates: Partial<UserSettings>
): Promise<UserSettings | null> {
	try {
		const settings = await authApi.updateUserSettings(updates);
		userSettings.set(settings);
		return settings;
	} catch {
		return null;
	}
}

/** Change current user password. */
export async function doChangePassword(
	currentPassword: string,
	newPassword: string
): Promise<boolean> {
	try {
		await authApi.changePassword({
			current_password: currentPassword,
			new_password: newPassword,
		});
		// After password change, all sessions are invalidated; re-login
		await doLogout();
		return true;
	} catch {
		return false;
	}
}

/** Check if user needs to log in (multi-user mode, no session). */
export function needsLogin(): boolean {
	const status = get(authStatus);
	const user = get(currentUser);
	if (!status || status.single_user_mode) return false;
	return user === null;
}
