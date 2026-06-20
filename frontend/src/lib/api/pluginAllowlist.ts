/**
 * Plugin Allowlist API client for Opti-Oignon (S128).
 *
 * Provides typed access to all /api/security/plugin-allowlist/* endpoints.
 * Supports: status, batch prepare/approve, per-plugin/batch revoke, verify.
 */

import { fetchApi } from './client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AllowlistEntry {
  plugin_id: string;
  code_hash: string;
  approved_by: string;
  approved_at: number;
  batch_id: string;
  permissions: string[];
  signature: string;
}

export interface BatchManifest {
  batch_id: string;
  plugins: BatchPlugin[];
  batch_hash: string;
}

export interface BatchPlugin {
  plugin_id: string;
  code_hash: string;
  permissions: string[];
  plugin_dir: string;
}

export interface AllowlistStatus {
  available: boolean;
  total_entries: number;
  batches: Record<string, number>;
  pending_batch: BatchManifest | null;
  entries: AllowlistEntry[];
}

export interface BatchApproveResult {
  success: boolean;
  entries_added?: number;
  batch_id?: string;
  message?: string;
  error?: string;
}

export interface RevokeResult {
  success: boolean;
  plugin_id?: string;
  batch_id?: string;
  revoked?: number;
}

export interface VerifyResult {
  allowed: boolean;
  reason?: string;
  entry?: AllowlistEntry;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

/** Fetch current plugin allowlist status (entries, batches, pending). */
export async function getAllowlistStatus(): Promise<AllowlistStatus> {
  return fetchApi('/api/security/plugin-allowlist');
}

/** Prepare a batch of plugins for approval ceremony. */
export async function prepareBatch(
  plugins: Array<{ plugin_id: string; plugin_dir: string; permissions: string[] }>,
): Promise<BatchManifest> {
  return fetchApi('/api/security/plugin-allowlist/prepare', {
    method: 'POST',
    body: JSON.stringify({ plugins }),
  });
}

/** Approve a prepared batch after ceremony verification. */
export async function approveBatch(params: {
  batch_id: string;
  visual_code: string;
  password: string;
  two_fa_code?: string | null;
}): Promise<BatchApproveResult> {
  return fetchApi('/api/security/plugin-allowlist/approve', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

/** Revoke a single plugin by plugin_id. */
export async function revokePlugin(pluginId: string): Promise<RevokeResult> {
  return fetchApi('/api/security/plugin-allowlist/revoke', {
    method: 'POST',
    body: JSON.stringify({ plugin_id: pluginId }),
  });
}

/** Revoke all plugins from a batch by batch_id. */
export async function revokeBatch(batchId: string): Promise<RevokeResult> {
  return fetchApi('/api/security/plugin-allowlist/revoke', {
    method: 'POST',
    body: JSON.stringify({ batch_id: batchId }),
  });
}

/** Verify a specific plugin against the allowlist. */
export async function verifyPlugin(pluginId: string): Promise<VerifyResult> {
  return fetchApi(`/api/security/plugin-allowlist/verify/${encodeURIComponent(pluginId)}`);
}
