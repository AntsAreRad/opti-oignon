/**
 * Search Kill Switch API client for Opti-Oignon (S128).
 *
 * Provides typed access to all /api/security/search-killswitch/* endpoints.
 * Supports: status, engage, re-enable ceremony, domain allowlist management.
 */

import { fetchApi } from './client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DomainAllowlistStatus {
  enabled: boolean;
  domain_count: number;
  domains: string[];
}

export interface KillSwitchStatus {
  available: boolean;
  search_enabled: boolean;
  killed_at?: number | null;
  killed_by?: string | null;
  kill_reason?: string | null;
  circuit_breaker_tripped?: boolean;
  injection_count?: number;
  reenable_pending?: boolean;
  domain_allowlist?: DomainAllowlistStatus;
}

export interface KillResult {
  success: boolean;
  search_enabled?: boolean;
  killed_at?: number;
  killed_by?: string;
  reason?: string;
  message?: string;
}

export interface ReenableRequestResult {
  success: boolean;
  pending?: boolean;
  request_id?: string;
  cooldown_seconds?: number;
  message?: string;
  error?: string;
}

export interface ReenableConfirmResult {
  success: boolean;
  search_enabled?: boolean;
  message?: string;
  error?: string;
}

export interface DomainAllowlistUpdateResult {
  success: boolean;
  enabled: boolean;
  domains: string[];
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

/** Fetch current kill switch status. */
export async function getKillSwitchStatus(): Promise<KillSwitchStatus> {
  return fetchApi('/api/security/search-killswitch');
}

/** Engage the kill switch (one-click, no ceremony). */
export async function engageKillSwitch(reason: string = 'manual'): Promise<KillResult> {
  return fetchApi('/api/security/search-killswitch/kill', {
    method: 'POST',
    body: JSON.stringify({ reason }),
  });
}

/** Start the re-enable ceremony. */
export async function requestReenable(): Promise<ReenableRequestResult> {
  return fetchApi('/api/security/search-killswitch/request-reenable', {
    method: 'POST',
  });
}

/** Fetch the visual code for DOM display. */
export async function getReenableCode(): Promise<{ visual_code: string }> {
  return fetchApi('/api/security/search-killswitch/reenable-code');
}

/** Confirm the re-enable ceremony. */
export async function confirmReenable(params: {
  request_id: string;
  visual_code: string;
  password: string;
  two_fa_code?: string | null;
}): Promise<ReenableConfirmResult> {
  return fetchApi('/api/security/search-killswitch/confirm-reenable', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

/** Cancel a pending re-enable request. */
export async function cancelReenable(): Promise<{ success: boolean }> {
  return fetchApi('/api/security/search-killswitch/cancel-reenable', {
    method: 'POST',
  });
}

/** Update the server-enforced domain allowlist. */
export async function updateDomainAllowlist(
  enabled: boolean,
  domains: string[],
): Promise<DomainAllowlistUpdateResult> {
  return fetchApi('/api/security/search-killswitch/domain-allowlist', {
    method: 'PUT',
    body: JSON.stringify({ enabled, domains }),
  });
}
