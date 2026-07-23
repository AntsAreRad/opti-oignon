/**
 * Security Mode API client for Opti-Oignon.
 *
 * Provides typed access to the Daily/Bulbe dual-mode security system.
 */

import { fetchApi } from './client';

export interface ModePolicy {
  web_search_allowed: boolean;
  db_encryption_required: boolean;
  two_fa_required: boolean;
  plugin_allowlist_required: boolean;
  sandbox_bwrap_required: boolean;
  session_timeout: number;
  backup_encryption_required: boolean;
  cookie_samesite: string;
  tool_call_approval_required: boolean;
  rate_limit_max_attempts: number;
  rate_limit_window: number;
  bearer_auth_allowed: boolean;
}

export interface PendingDowngrade {
  pending: boolean;
  request_id?: string;
  requested_at?: number;
  cooldown_remaining?: number;
  cooldown_complete?: boolean;
  expires_at?: number;
  attempts?: number;
}

export interface SecurityModeStatus {
  mode: string;
  available: boolean;
  timestamp?: number;
  lockfile_exists?: boolean;
  sources_agree?: boolean;
  hmac_valid?: boolean;
  policy?: ModePolicy;
  pending_downgrade?: PendingDowngrade | null;
}

export interface ModeChangeResult {
  success: boolean;
  mode?: string;
  message?: string;
  changed?: boolean;
  timestamp?: number;
  error?: string;
}

export interface DowngradeRequestResult {
  success: boolean;
  pending?: boolean;
  request_id?: string;
  requested_at?: number;
  cooldown_seconds?: number;
  expires_at?: number;
  message?: string;
  error?: string;
}

/** Fetch current security mode and policy. */
export async function getSecurityMode(): Promise<SecurityModeStatus> {
  return fetchApi('/api/security/mode');
}

/** Escalate to Bulbe mode (immediate, no ceremony). */
export async function escalateToBulbe(): Promise<ModeChangeResult> {
  return fetchApi('/api/security/mode', {
    method: 'POST',
    body: JSON.stringify({ mode: 'bulbe' }),
  });
}

/** Start the downgrade ceremony (Bulbe -> Daily). */
export async function requestDowngrade(): Promise<DowngradeRequestResult> {
  return fetchApi('/api/security/mode/request-downgrade', {
    method: 'POST',
  });
}

/** Check downgrade request status. */
export async function getDowngradeStatus(): Promise<PendingDowngrade> {
  return fetchApi('/api/security/mode/downgrade-status');
}

/** Fetch the visual code for DOM display. */
export async function getVisualCode(): Promise<{ visual_code: string }> {
  return fetchApi('/api/security/mode/visual-code');
}

/** Confirm the downgrade ceremony. */
export async function confirmDowngrade(params: {
  request_id: string;
  visual_code: string;
  password: string;
  two_fa_code?: string | null;
}): Promise<ModeChangeResult> {
  return fetchApi('/api/security/mode/confirm-downgrade', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

/** Cancel a pending downgrade request. */
export async function cancelDowngrade(): Promise<{ success: boolean; message: string }> {
  return fetchApi('/api/security/mode/cancel-downgrade', {
    method: 'POST',
  });
}
