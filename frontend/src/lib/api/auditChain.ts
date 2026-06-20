/**
 * API client for Hash-Chain Signed Audit Log -- S130.
 *
 * Covers:
 *   GET  /api/security/audit-chain/status  -- chain length, integrity
 *   GET  /api/security/audit-chain/events  -- paginated event query
 *   POST /api/security/audit-chain/verify  -- full chain verification
 *   GET  /api/security/audit-chain/export  -- CSV export
 */

import { apiGet, apiPost } from './client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AuditChainLastEntry {
	id: number;
	timestamp: number;
	event_type: string;
	entry_hash: string;
}

export interface AuditChainStatus {
	total_entries: number;
	last_entry: AuditChainLastEntry | null;
	chain_valid: boolean;
	first_broken_index: number | null;
}

export interface AuditEvent {
	id: number;
	timestamp: number;
	event_type: string;
	source: string;
	action: string;
	severity: string;
	details: Record<string, unknown>;
	prev_hash: string;
	entry_hash: string;
}

export interface AuditEventsResponse {
	events: AuditEvent[];
	count: number;
	offset: number;
}

export interface AuditChainVerifyResult {
	chain_valid: boolean;
	first_broken_index: number | null;
	total_entries: number;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

/** Get chain status: length, last entry, integrity. */
export async function getAuditChainStatus(): Promise<AuditChainStatus> {
	return apiGet<AuditChainStatus>('/api/security/audit-chain/status');
}

/** Get paginated audit events with optional filters. */
export async function getAuditChainEvents(params: {
	limit?: number;
	offset?: number;
	event_type?: string;
	severity?: string;
	after?: number;
	before?: number;
} = {}): Promise<AuditEventsResponse> {
	const query = new URLSearchParams();
	if (params.limit !== undefined) query.set('limit', String(params.limit));
	if (params.offset !== undefined) query.set('offset', String(params.offset));
	if (params.event_type) query.set('event_type', params.event_type);
	if (params.severity) query.set('severity', params.severity);
	if (params.after !== undefined) query.set('after', String(params.after));
	if (params.before !== undefined) query.set('before', String(params.before));

	const qs = query.toString();
	const path = '/api/security/audit-chain/events' + (qs ? '?' + qs : '');
	return apiGet<AuditEventsResponse>(path);
}

/** Run full chain verification. */
export async function verifyAuditChain(): Promise<AuditChainVerifyResult> {
	return apiPost<AuditChainVerifyResult>('/api/security/audit-chain/verify');
}

/** Export audit chain as CSV (returns raw text). */
export async function exportAuditChainCsv(): Promise<string> {
	const resp = await fetch('/api/security/audit-chain/export', {
		credentials: 'include',
	});
	if (!resp.ok) {
		throw new Error(`Export failed: ${resp.status}`);
	}
	return resp.text();
}
