/**
 * API client for Sandbox management.
 *
 * Provides typed functions for all sandbox endpoints:
 * create, inject, list files, execute, destroy, audit,
 * preview, download, approve, copy-out, reject.
 */

import { apiGet, apiPost, apiDelete, apiUpload } from './client';
import type {
	SandboxStatusResponse,
	SandboxCreateRequest,
	SandboxCreateResponse,
	SandboxInjectRequest,
	SandboxInjectResponse,
	SandboxFilesResponse,
	SandboxExecuteRequest,
	SandboxExecuteResponse,
	SandboxDestroyResponse,
	SandboxSessionInfo,
	SandboxAuditResponse,
	SandboxConfirmDegradedResponse,
	SandboxPreviewResponse,
	SandboxApproveRequest,
	SandboxApproveResponse,
	SandboxCopyOutResponse,
	SandboxRejectResponse,
	SandboxApprovalInfoResponse,
	SandboxApprovalAuditResponse,
	SandboxStopResponse,
	SandboxBindRequest,
	SandboxBindingResponse,
	SandboxUploadResponse,
	HostBrowseResponse,
	SandboxCloneRequest,
	SandboxCloneResponse,
	SandboxDiffResponse,
	SandboxConfirmDeletionsResponse,
	SandboxApplyRequest,
	SandboxApplyResponse,
	SandboxNetworkToggleResponse,
	SandboxProvisionRequest,
	SandboxProvisionResponse
} from '../types';

/** Get sandbox system status. */
export async function getSandboxStatus(): Promise<SandboxStatusResponse> {
	return apiGet('/api/sandbox/status');
}

/** Confirm degraded mode (required when bwrap unavailable). */
export async function confirmDegradedMode(): Promise<SandboxConfirmDegradedResponse> {
	return apiPost('/api/sandbox/confirm-degraded');
}

/** Create a new sandbox session. */
export async function createSandbox(
	request: SandboxCreateRequest
): Promise<SandboxCreateResponse> {
	return apiPost('/api/sandbox/create', request);
}

/** Inject files into a sandbox. */
export async function injectFiles(
	request: SandboxInjectRequest
): Promise<SandboxInjectResponse> {
	return apiPost('/api/sandbox/inject', request);
}

/** List files in a sandbox workspace (with approval status). */
export async function listSandboxFiles(sessionId: string): Promise<SandboxFilesResponse> {
	return apiGet(`/api/sandbox/files/${sessionId}`);
}

/** Execute a tool in a sandbox. */
export async function executeSandboxTool(
	request: SandboxExecuteRequest
): Promise<SandboxExecuteResponse> {
	return apiPost('/api/sandbox/execute', request);
}

/** Destroy a sandbox session. */
export async function destroySandbox(sessionId: string): Promise<SandboxDestroyResponse> {
	return apiDelete(`/api/sandbox/${sessionId}`);
}

/** List all sandbox sessions. */
export async function listSessions(): Promise<SandboxSessionInfo[]> {
	return apiGet('/api/sandbox/sessions');
}

/** Get audit log entries. */
export async function getAuditLog(
	sessionId?: string,
	limit: number = 100
): Promise<SandboxAuditResponse> {
	const params = new URLSearchParams();
	if (sessionId) params.set('session_id', sessionId);
	if (limit !== 100) params.set('limit', String(limit));
	const query = params.toString();
	return apiGet(`/api/sandbox/audit${query ? '?' + query : ''}`);
}

// --: Copy-out + human approval endpoints --

/** Preview a file from the sandbox (text capped at 64KB, binary hex preview). */
export async function previewSandboxFile(
	sessionId: string,
	path: string
): Promise<SandboxPreviewResponse> {
	return apiGet(`/api/sandbox/preview/${sessionId}/${path}`);
}

/** Download an approved file from the sandbox (binary). */
export function getSandboxDownloadUrl(sessionId: string, path: string): string {
	return `/api/sandbox/download/${sessionId}/${path}`;
}

/** Approve specific files for copy-out. */
export async function approveSandboxFiles(
	sessionId: string,
	request: SandboxApproveRequest
): Promise<SandboxApproveResponse> {
	return apiPost(`/api/sandbox/${sessionId}/approve`, request);
}

/** Copy approved files from sandbox to host. */
export async function copyOutSandboxFiles(
	sessionId: string,
	request: SandboxApproveRequest
): Promise<SandboxCopyOutResponse> {
	return apiPost(`/api/sandbox/${sessionId}/copy-out`, request);
}

/** Reject all files in a sandbox (prevent copy-out). */
export async function rejectSandboxFiles(
	sessionId: string
): Promise<SandboxRejectResponse> {
	return apiPost(`/api/sandbox/${sessionId}/reject`);
}

/** Get approval state summary. */
export async function getApprovalInfo(
	sessionId: string
): Promise<SandboxApprovalInfoResponse> {
	return apiGet(`/api/sandbox/${sessionId}/approval`);
}

/** Get approval audit trail. */
export async function getApprovalAudit(
	sessionId: string
): Promise<SandboxApprovalAuditResponse> {
	return apiGet(`/api/sandbox/${sessionId}/approval-audit`);
}

// -- (Bloc 1): workspace lifecycle + conversation binding --

/** SIGKILL the workspace's running command; the workspace persists.
 * stopped=false means nothing was running (honest no-op). */
export async function stopSandboxCommand(sessionId: string): Promise<SandboxStopResponse> {
	return apiPost(`/api/sandbox/${sessionId}/stop`);
}

/** Bind a conversation to a workspace (rebind allowed; 409 when the
 * workspace is held by another conversation, 403 on owner mismatch). */
export async function bindConversation(
	request: SandboxBindRequest
): Promise<SandboxBindingResponse> {
	return apiPost('/api/sandbox/bind', request);
}

/** Release a conversation's workspace binding (no-op when unbound). */
export async function unbindConversation(
	conversationId: string
): Promise<SandboxBindingResponse> {
	return apiDelete(`/api/sandbox/bind/${conversationId}`);
}

/** The workspace currently bound to a conversation, if any. */
export async function getConversationBinding(
	conversationId: string
): Promise<SandboxBindingResponse> {
	return apiGet(`/api/sandbox/bind/${conversationId}`);
}

// -- (Bloc 2): copy-in (drag-and-drop, host browse, host clone) --
// All three are EXPLICIT user actions through the manager UI; the model can
// trigger none of them. The agent only ever sees /workspace.

/** Upload files into a workspace via multipart (spec 5.1). The request is
 * refused whole with 413 over any cap (count, per-file bytes, the workspace
 * quota); invalid names and collisions are refused per file, never
 * overwritten. */
export async function uploadFiles(
	sessionId: string,
	files: File[],
	destSubdir = ''
): Promise<SandboxUploadResponse> {
	const form = new FormData();
	for (const file of files) {
		form.append('files', file, file.name);
	}
	form.append('dest_subdir', destSubdir);
	return apiUpload(`/api/sandbox/${sessionId}/upload`, form);
}

/** List an allowlisted host directory (spec 5.2a). No path lists the
 * configured share roots; outside the roots the server answers 403 before
 * any existence check. */
export async function browseHost(path?: string): Promise<HostBrowseResponse> {
	return apiGet('/api/sandbox/host/browse', path ? { path } : undefined);
}

/** Clone an allowlisted host directory into a workspace (spec 5.2b):
 * symlink-safe, size/file-count capped (413), collision-refusing (409);
 * records the section 6.1 baseline manifest. */
export async function cloneDirectory(
	sessionId: string,
	request: SandboxCloneRequest
): Promise<SandboxCloneResponse> {
	return apiPost(`/api/sandbox/${sessionId}/clone`, request);
}

// ---- (Bloc 3): diff-gated write-back ------------------------------------
// Review and apply are EXPLICIT user actions through the manager UI; the
// model can trigger neither the diff-approve chain nor the apply.

/** The live workspace classified against the baseline manifest (spec 6.1):
 * hash-driven added/modified/deleted; no baseline means everything is
 * "added" and there is no implicit write-back target. The returned
 * diff_hash must be echoed by applyChanges; over the diff bound the server
 * refuses (413) rather than truncate. */
export async function getDiff(sessionId: string): Promise<SandboxDiffResponse> {
	return apiGet(`/api/sandbox/${sessionId}/diff`);
}

/** Explicitly confirm deletions for apply (spec 6.2): distinct from
 * approval, never bundled into approve-all; paths not classified deleted
 * by the current diff are refused per path in the 200 body. */
export async function confirmDeletions(
	sessionId: string,
	paths: string[]
): Promise<SandboxConfirmDeletionsResponse> {
	return apiPost(`/api/sandbox/${sessionId}/confirm-deletions`, { paths });
}

/** Apply approved changes back to the host (spec 6.2). diff_hash must be
 * the digest of the reviewed diff (409 on any workspace drift since the
 * review); target_dir is required only for upload-only workspaces and must
 * resolve under the host share-root allowlist (403 outside, before any
 * existence check). Per-file results are honest: applied, deleted,
 * refused. */
export async function applyChanges(
	sessionId: string,
	request: SandboxApplyRequest
): Promise<SandboxApplyResponse> {
	return apiPost(`/api/sandbox/${sessionId}/apply`, request);
}

/** (Bloc 4): flip the per-workspace network flag -- an explicit user
 * action. Enabling answers 403 under Bulbe (the binding-layer gate; an
 * unset or unknown mode is treated as Bulbe); disabling works in any mode.
 * Both directions are audited. */
export async function setNetwork(
	sessionId: string,
	enabled: boolean
): Promise<SandboxNetworkToggleResponse> {
	return apiPost(`/api/sandbox/${sessionId}/network`, { enabled });
}

/** (Bloc 4): run the provision phase -- the one scoped egress: a
 * hash-pinned requirements set installed with --require-hashes
 * --only-binary=:all: into a workspace venv. 403 under Bulbe, 409 when the
 * workspace network flag is off, 400 on a set that is not exact-and-pinned
 * (per-line refusals; nothing installs on a partial validation). Refusals
 * from the run itself (bwrap absent, validator) come back blocked in the
 * 200 body, honestly. */
export async function provisionWorkspace(
	sessionId: string,
	request: SandboxProvisionRequest
): Promise<SandboxProvisionResponse> {
	return apiPost(`/api/sandbox/${sessionId}/provision`, request);
}
