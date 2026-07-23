/**
 * Tool Call Approval API client for Opti-Oignon.
 *
 * Provides typed access to all /api/security/tool-approval/* endpoints.
 * Used by ToolCallApproval.svelte to poll for pending approvals and
 * submit allow/deny decisions.
 */

import { fetchApi } from './client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PendingApproval {
  approval_id: string;
  conversation_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  arguments_summary: string;
  risk_level: 'low' | 'medium' | 'high';
  status: 'pending' | 'approved' | 'denied' | 'timeout';
  created_at: number;
  resolved_at: number | null;
  resolved_by: string | null;
  timeout_remaining: number;
}

export interface PendingResponse {
  available: boolean;
  pending: PendingApproval[];
  count: number;
}

export interface ApprovalResult {
  success: boolean;
  approval_id: string;
  status: string;
}

export interface AuditEntry {
  approval_id: string;
  tool_name: string;
  status: string;
  resolved_by: string;
  timestamp: number;
  conversation_id: string;
}

export interface AuditResponse {
  available: boolean;
  entries: AuditEntry[];
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

/** Fetch all pending tool call approval requests. */
export async function getPendingApprovals(): Promise<PendingResponse> {
  return fetchApi('/api/security/tool-approval/pending');
}

/** Approve a pending tool call. */
export async function approveToolCall(approvalId: string): Promise<ApprovalResult> {
  return fetchApi(`/api/security/tool-approval/${encodeURIComponent(approvalId)}/approve`, {
    method: 'POST',
  });
}

/** Deny a pending tool call. */
export async function denyToolCall(approvalId: string): Promise<ApprovalResult> {
  return fetchApi(`/api/security/tool-approval/${encodeURIComponent(approvalId)}/deny`, {
    method: 'POST',
  });
}

/** Fetch the tool call approval audit log. */
export async function getApprovalAudit(limit: number = 50): Promise<AuditResponse> {
  return fetchApi(`/api/security/tool-approval/audit?limit=${limit}`);
}
