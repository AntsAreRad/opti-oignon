/**
 * Authentication store.
 *
 * Manages current user state, token persistence, automatic token refresh,
 * and login/logout flows.
 *
 * When cookie_mode is enabled (default), JWT tokens are stored in
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
import { httpRefusal } from '$lib/switches/serverSwitch';
import type {
	AuthUser,
	AuthTokens,
	AuthStatus,
	UserSettings,
	TwoFactorAnswer,
	TwoFactorChallenge,
	TwoFactorEnrolment,
	TwoFactorEnrolmentSecret,
} from '$lib/types';

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
// Cookie mode flag
// ---------------------------------------------------------------------------

/** Whether the backend uses httpOnly cookie auth. */
let _cookieMode = true;

// ---------------------------------------------------------------------------
// Token persistence (localStorage keys -- legacy / non-cookie mode only)
// ---------------------------------------------------------------------------

const STORAGE_ACCESS_TOKEN = 'oo-access-token';
const STORAGE_REFRESH_TOKEN = 'oo-refresh-token';

function storeTokens(tokens: AuthTokens): void {
	// In cookie mode the backend sets httpOnly cookies and the
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
 * Remove legacy localStorage tokens when migrating to cookie mode.
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
		let status: AuthStatus;
		try {
			status = await authApi.getAuthStatus();
		} catch (err) {
			if (httpRefusal(err)?.status !== 403) throw err;
			// The backend answered and refused (403: Bulbe serves this machine's clients only): no synthetic user,
			// no single-user mode assumed -- the sign-in page is shown, and says the refusal when it meets it.
			// Any other failure -- a proxy's 500 or 502 while the API starts, a network error -- is the fallback's.
			authStatus.set({
				available: true,
				single_user_mode: false,
				registration_enabled: false,
				user_count: 0,
			});
			currentUser.set(null);
			return;
		}
		authStatus.set(status);

		// Detect cookie mode from backend status
		_cookieMode = status.cookie_mode ?? true;

		// Clean up legacy localStorage tokens when switching to cookie mode
		if (_cookieMode) {
			_migrateLegacyTokens();
		}

		if (status.single_user_mode) {
			// Single-user mode normally bypasses authentication. Bulbe mode is
			// the exception: it enforces auth even for a single-user
			// install. Probe a protected endpoint to tell the two apart -- in
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
			// In cookie mode, just call /me -- the httpOnly cookie
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

/** What the registration gives: the user, or -- in Bulbe -- the enrolment of the account's first second factor. */
export type RegisterOutcome = { user: AuthUser } | { enrolment: TwoFactorEnrolment };

/** Register a new account: the user signed in, or -- in Bulbe -- the enrolment its first session waits for. */
export async function doRegister(
	username: string,
	password: string,
	email = ''
): Promise<RegisterOutcome> {
	const answer = await authApi.register({ username, password, email });
	if ('requires_2fa_enrolment' in answer && answer.requires_2fa_enrolment) {
		return { enrolment: answer };
	}
	return { user: await finishLogin(answer as AuthTokens) };
}

/** What the login's first step gives: the user, the challenge its second step answers, or -- in Bulbe, for an
 * account with no second factor -- the enrolment of a first one. */
export type LoginOutcome =
	| { user: AuthUser }
	| { challenge: TwoFactorChallenge }
	| { enrolment: TwoFactorEnrolment };

/** Log in with username and password: the user, or -- for an account with a second factor -- its challenge, or the
 * enrolment Bulbe asks of an account with none. */
export async function doLogin(username: string, password: string): Promise<LoginOutcome> {
	const answer = await authApi.login({ username, password });
	if ('requires_2fa_enrolment' in answer && answer.requires_2fa_enrolment) {
		return { enrolment: answer };
	}
	if ('requires_2fa' in answer && answer.requires_2fa) {
		return { challenge: answer };
	}
	return { user: await finishLogin(answer as AuthTokens) };
}

/** The login's second step: the code or the security key's assertion the challenge asked for. */
export async function doLoginSecondStep(answer: TwoFactorAnswer): Promise<AuthUser> {
	return finishLogin(await authApi.login2fa(answer));
}

/** Begin the TOTP an enrolment asks for: the secret and the QR code to show. */
export async function beginEnrolment(challengeId: string): Promise<TwoFactorEnrolmentSecret> {
	return authApi.enrolBegin({ challenge_id: challengeId });
}

/** Confirm it with one of its codes: the session's tokens and the recovery codes to show this once -- nothing
 * stored and no user set yet: a signed-in user on the login page is sent on at once by the root layout, which would
 * leave the codes unseen. The session begins with `finishEnrolment`, once they were shown. */
export async function completeEnrolment(
	challengeId: string,
	code: string
): Promise<{ tokens: AuthTokens; recoveryCodes: string[] }> {
	const answer = await authApi.enrolComplete({ challenge_id: challengeId, code });
	return { tokens: answer, recoveryCodes: answer.recovery_codes ?? [] };
}

/** Begin the session an enrolment gave, after its recovery codes were shown. */
export async function finishEnrolment(tokens: AuthTokens): Promise<AuthUser> {
	return finishLogin(tokens);
}

async function finishLogin(tokens: AuthTokens): Promise<AuthUser> {
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
			// In cookie mode, refresh token is in httpOnly cookie.
			// Send empty string -- backend reads cookie automatically.
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
		// In cookie mode, backend reads refresh cookie and clears both.
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
