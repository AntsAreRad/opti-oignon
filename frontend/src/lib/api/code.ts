/**
 * Typed API functions for code execution endpoints.
 *
 * Handles sandboxed code execution, code block extraction
 * and workdir management.
 */

import { apiPost } from './client';
import type { CodeExecuteResponse, CodeBlockInfo } from '$lib/types';

/** Execute du code dans le sandbox. */
export async function executeCode(params: {
	code: string;
	language: string;
	timeout?: number;
	conv_id?: string;
}): Promise<CodeExecuteResponse> {
	return apiPost<CodeExecuteResponse>('/api/code/execute', params);
}

/** Extrait les blocs de code d'un texte (reponse LLM). */
export async function extractCodeBlocks(text: string): Promise<CodeBlockInfo[]> {
	const result = await apiPost<{ blocks: CodeBlockInfo[] }>('/api/code/blocks', { text });
	return result.blocks;
}

/** Reset the sandbox working directory. */
export async function resetWorkdir(): Promise<void> {
	await apiPost<void>('/api/code/reset-workdir');
}
