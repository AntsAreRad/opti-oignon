/**
 * Conversation branching API client.
 *
 * Manages branch creation, navigation, comparison, and merging
 * for conversation exploration workflows.
 */

import { apiGet, apiPost, apiPut, apiDelete } from './client';
import type {
	Branch,
	BranchForkRequest,
	BranchUpdateRequest,
	BranchCompareRequest,
	BranchMergeRequest,
	BranchMergeResponse,
	BranchTreeNode,
	BranchComparison,
	BranchMessage,
	BranchMessagesResponse,
} from '$lib/types';

const BASE = '/api/branches';

/** Fork a conversation at a specific message. */
export async function forkBranch(req: BranchForkRequest): Promise<Branch> {
	return apiPost<Branch>(`${BASE}/fork`, req);
}

/** List all branches for a conversation. */
export async function listBranches(conversationId: string): Promise<Branch[]> {
	return apiGet<Branch[]>(`${BASE}/${conversationId}`);
}

/** Get details for a single branch. */
export async function getBranchDetail(branchId: string): Promise<Branch> {
	return apiGet<Branch>(`${BASE}/detail/${branchId}`);
}

/** Get messages for a branch (shared history + branch-specific by default). */
export async function getBranchMessages(
	branchId: string,
	full: boolean = true
): Promise<BranchMessagesResponse> {
	const params: Record<string, string> = {};
	if (!full) params.full = 'false';
	return apiGet<BranchMessagesResponse>(`${BASE}/${branchId}/messages`, params);
}

/** Add a message to a branch. */
export async function addBranchMessage(
	branchId: string,
	conversationId: string,
	role: string,
	content: string,
	model?: string,
	metadata?: Record<string, unknown>
): Promise<BranchMessage> {
	return apiPost<BranchMessage>(`${BASE}/${branchId}/messages`, {
		conversation_id: conversationId,
		role,
		content,
		model: model ?? null,
		metadata: metadata ?? null,
	});
}

/** Rename, recolor, or update metadata for a branch. */
export async function updateBranch(
	branchId: string,
	req: BranchUpdateRequest
): Promise<Branch> {
	return apiPut<Branch>(`${BASE}/${branchId}`, req);
}

/** Delete a branch and all its messages. */
export async function deleteBranch(
	branchId: string
): Promise<{ deleted: boolean; branch_id: string }> {
	return apiDelete<{ deleted: boolean; branch_id: string }>(`${BASE}/${branchId}`);
}

/** Get the branch tree structure for a conversation. */
export async function getBranchTree(conversationId: string): Promise<BranchTreeNode> {
	return apiGet<BranchTreeNode>(`${BASE}/${conversationId}/tree`);
}

/** Compare two branches side-by-side. */
export async function compareBranches(req: BranchCompareRequest): Promise<BranchComparison> {
	return apiPost<BranchComparison>(`${BASE}/compare`, req);
}

/** Merge messages from one branch into another. */
export async function mergeBranches(req: BranchMergeRequest): Promise<BranchMergeResponse> {
	return apiPost<BranchMergeResponse>(`${BASE}/merge`, req);
}
