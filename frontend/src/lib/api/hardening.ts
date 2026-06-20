/**
 * API client for Hardening endpoints -- S131.
 *
 * Covers:
 *   POST /api/security/conversation-wipe/all           -- wipe all conversations
 *   POST /api/security/conversation-wipe/{id}          -- wipe single conversation
 *   GET  /api/security/hardening/status                -- combined hardening status
 *   GET  /api/security/hardening/network               -- network details
 */

import { apiGet, apiPost } from './client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface WipeResult {
	conversation_id: string;
	buffers_wiped: number;
	fields_zeroed: number;
	success: boolean;
	memset_available: boolean;
	timestamp: number;
}

export interface WipeAllResult {
	conversations_wiped: number;
	total_buffers: number;
	total_fields_zeroed: number;
}

export interface ConversationWipeStatus {
	available: boolean;
	auto_wipe_on_close: boolean;
	bulbe_wipe_per_turn: boolean;
	active_conversations: number;
	total_registered_buffers: number;
	memset_available: boolean;
}

export interface SwapStatus {
	available?: boolean;
	swap_enabled?: boolean;
	encrypted?: boolean;
	safe?: boolean;
	devices?: Array<{ device: string; type: string; encrypted: boolean }>;
	error?: string;
	platform_supported?: boolean;
}

export interface OllamaLogStatus {
	available: boolean;
	log_level?: string;
	sanitization_enabled?: boolean;
	recommendations?: Record<string, string>;
}

export interface DnsStatus {
	encrypted: boolean;
	protocol: string;
	resolver: string;
	details: string;
}

export interface ProxyStatus {
	configured: boolean;
	proxy_url: string;
	reachable: boolean;
	error: string;
}

export interface PortDetail {
	port: number;
	address: string;
	process: string;
	expected: boolean;
}

export interface NetworkStatus {
	available: boolean;
	dns?: DnsStatus;
	proxy?: ProxyStatus;
	ports?: {
		total: number;
		unexpected: number;
		details: PortDetail[];
	};
	warnings?: string[];
}

export interface HardeningStatus {
	conversation_wipe: ConversationWipeStatus;
	swap: SwapStatus;
	ollama_log: OllamaLogStatus;
	network: NetworkStatus;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

/** Wipe a single conversation from RAM. */
export async function wipeConversation(conversationId: string): Promise<WipeResult> {
	return apiPost<WipeResult>(`/api/security/conversation-wipe/${encodeURIComponent(conversationId)}`);
}

/** Emergency wipe: zero all conversation buffers. */
export async function wipeAllConversations(): Promise<WipeAllResult> {
	return apiPost<WipeAllResult>('/api/security/conversation-wipe/all');
}

/** Get combined hardening status. */
export async function getHardeningStatus(): Promise<HardeningStatus> {
	return apiGet<HardeningStatus>('/api/security/hardening/status');
}

/** Get detailed network hardening status. */
export async function getNetworkStatus(): Promise<NetworkStatus> {
	return apiGet<NetworkStatus>('/api/security/hardening/network');
}
