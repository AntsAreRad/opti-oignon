/**
 * Authentication API client.
 *
 * Handles register, login, logout, token refresh, profile,
 * user settings, shared projects, and admin operations.
 */

import { apiGet, apiPost, apiPut, apiDelete } from './client';
import type {
	AuthUser,
	AuthTokens,
	AuthStatus,
	TwoFactorAnswer,
	TwoFactorChallenge,
	TwoFactorEnrolment,
	TwoFactorEnrolmentSecret,
	EnrolledTokens,
	UserSettings,
	RegisterRequest,
	CreateUserRequest,
	LoginRequest,
	ProfileUpdateRequest,
	PasswordChangeRequest,
	SettingsUpdateRequest,
	ShareProjectRequest,
	ProjectMember,
	AuditLogEntry,
} from '$lib/types';

const BASE = '/api/auth';

// -- Public endpoints (no auth required) ------------------------------------

/** Get auth system status. */
export async function getAuthStatus(): Promise<AuthStatus> {
	return apiGet<AuthStatus>(`${BASE}/status`);
}

/** Register a new account and get tokens -- in Bulbe, the enrolment of its first second factor instead. */
export async function register(req: RegisterRequest): Promise<AuthTokens | TwoFactorEnrolment> {
	return apiPost<AuthTokens | TwoFactorEnrolment>(`${BASE}/register`, req);
}

/** Log in: tokens, or -- for an account with a second factor -- the challenge its second step answers, or -- in
 * Bulbe, for an account with none -- the enrolment of a first one. */
export async function login(req: LoginRequest): Promise<AuthTokens | TwoFactorChallenge | TwoFactorEnrolment> {
	return apiPost<AuthTokens | TwoFactorChallenge | TwoFactorEnrolment>(`${BASE}/login`, req);
}

/** The login's second step: tokens once the factor is right. */
export async function login2fa(req: TwoFactorAnswer): Promise<AuthTokens> {
	return apiPost<AuthTokens>(`${BASE}/login/2fa`, req);
}

/** Begin the TOTP an enrolment challenge asks for: its secret, its URI, its QR code. */
export async function enrolBegin(req: { challenge_id: string }): Promise<TwoFactorEnrolmentSecret> {
	return apiPost<TwoFactorEnrolmentSecret>(`${BASE}/login/2fa/enrol/begin`, req);
}

/** Confirm it with one of its codes: the first session, and the recovery codes shown once. */
export async function enrolComplete(req: { challenge_id: string; code: string }): Promise<EnrolledTokens> {
	return apiPost<EnrolledTokens>(`${BASE}/login/2fa/enrol/complete`, req);
}

/** Refresh token pair. */
export async function refreshToken(refresh_token: string): Promise<AuthTokens> {
	return apiPost<AuthTokens>(`${BASE}/refresh`, { refresh_token });
}

/** Log out (invalidate session). */
export async function logout(refresh_token: string): Promise<{ logged_out: boolean }> {
	return apiPost<{ logged_out: boolean }>(`${BASE}/logout`, { refresh_token });
}

// -- Authenticated endpoints ------------------------------------------------

/** Get current user profile. */
export async function getMe(): Promise<AuthUser> {
	return apiGet<AuthUser>(`${BASE}/me`);
}

/** Update current user profile. */
export async function updateMe(req: ProfileUpdateRequest): Promise<AuthUser> {
	return apiPut<AuthUser>(`${BASE}/me`, req);
}

/** Change current user password. */
export async function changePassword(req: PasswordChangeRequest): Promise<{ changed: boolean }> {
	return apiPut<{ changed: boolean }>(`${BASE}/me/password`, req);
}

/** Get per-user settings. */
export async function getUserSettings(): Promise<UserSettings> {
	return apiGet<UserSettings>(`${BASE}/settings`);
}

/** Update per-user settings. */
export async function updateUserSettings(req: SettingsUpdateRequest): Promise<UserSettings> {
	return apiPut<UserSettings>(`${BASE}/settings`, req);
}

// -- Admin endpoints --------------------------------------------------------

/** List all users (admin only). */
export async function listUsers(limit = 100, offset = 0): Promise<AuthUser[]> {
	return apiGet<AuthUser[]>(`${BASE}/users`, {
		limit: String(limit),
		offset: String(offset),
	});
}

/** Add an account (the platform's owner, with the owner's own password). */
export async function createUser(req: CreateUserRequest): Promise<AuthUser> {
	return apiPost<AuthUser>(`${BASE}/users`, req);
}

/** Delete a user (admin only). */
export async function deleteUser(userId: string): Promise<{ deleted: boolean; user_id: string }> {
	return apiDelete<{ deleted: boolean; user_id: string }>(`${BASE}/users/${userId}`);
}

// -- Shared projects --------------------------------------------------------

/** Share a project with another user. */
export async function shareProject(req: ShareProjectRequest): Promise<{
	project_id: string;
	user_id: string;
	role: string;
	invite_token: string;
}> {
	return apiPost(`${BASE}/projects/share`, req);
}

/** List members of a shared project. */
export async function listProjectMembers(projectId: string): Promise<{
	project_id: string;
	members: ProjectMember[];
}> {
	return apiGet(`${BASE}/projects/${projectId}/members`);
}

/** Remove a user from a shared project. */
export async function removeProjectMember(
	projectId: string,
	userId: string
): Promise<{ removed: boolean }> {
	return apiDelete(`${BASE}/projects/${projectId}/members/${userId}`);
}

// -- Audit log --------------------------------------------------------------

/** Get audit log entries (admin only). */
export async function getAuditLog(params?: {
	user_id?: string;
	project_id?: string;
	limit?: number;
}): Promise<{ entries: AuditLogEntry[]; count: number }> {
	const qp: Record<string, string> = {};
	if (params?.user_id) qp.user_id = params.user_id;
	if (params?.project_id) qp.project_id = params.project_id;
	if (params?.limit) qp.limit = String(params.limit);
	return apiGet(`${BASE}/audit`, qp);
}
