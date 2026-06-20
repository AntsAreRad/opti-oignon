/**
 * Plugin Marketplace API client (S102).
 *
 * Browse, search, install from URL, reviews, template generation.
 */

import { apiGet, apiPost } from './client';
import type {
	MarketplaceListResponse,
	RemoteInstallResponse,
	ReviewListResponse,
	AddReviewResponse,
	TemplateResponse,
} from '$lib/types';

const BASE = '/api/plugins';

/** Browse available plugins from the marketplace index. */
export async function browseMarketplace(opts?: {
	sortBy?: string;
	limit?: number;
	offset?: number;
	refresh?: boolean;
}): Promise<MarketplaceListResponse> {
	const params: Record<string, string> = {};
	if (opts?.sortBy) params.sort_by = opts.sortBy;
	if (opts?.limit) params.limit = String(opts.limit);
	if (opts?.offset) params.offset = String(opts.offset);
	if (opts?.refresh) params.refresh = 'true';
	return (await apiGet(`${BASE}/marketplace`, params)) as MarketplaceListResponse;
}

/** Search the marketplace by keyword, tag, author, or hook. */
export async function searchMarketplace(opts: {
	keyword?: string;
	tag?: string;
	author?: string;
	hook?: string;
	sortBy?: string;
	limit?: number;
}): Promise<MarketplaceListResponse> {
	const params: Record<string, string> = {};
	if (opts.keyword) params.keyword = opts.keyword;
	if (opts.tag) params.tag = opts.tag;
	if (opts.author) params.author = opts.author;
	if (opts.hook) params.hook = opts.hook;
	if (opts.sortBy) params.sort_by = opts.sortBy;
	if (opts.limit) params.limit = String(opts.limit);
	return (await apiGet(`${BASE}/marketplace/search`, params)) as MarketplaceListResponse;
}

/** Install a plugin from a remote URL. */
export async function installFromUrl(
	url: string,
	expectedSha256: string = '',
	autoEnable: boolean = false
): Promise<RemoteInstallResponse> {
	return (await apiPost(`${BASE}/marketplace/install`, {
		url,
		expected_sha256: expectedSha256,
		auto_enable: autoEnable,
	})) as RemoteInstallResponse;
}

/** Get reviews for a plugin. */
export async function getPluginReviews(
	name: string,
	opts?: { sortBy?: string; limit?: number; offset?: number }
): Promise<ReviewListResponse> {
	const params: Record<string, string> = {};
	if (opts?.sortBy) params.sort_by = opts.sortBy;
	if (opts?.limit) params.limit = String(opts.limit);
	if (opts?.offset) params.offset = String(opts.offset);
	return (await apiGet(
		`${BASE}/${encodeURIComponent(name)}/reviews`,
		params
	)) as ReviewListResponse;
}

/**
 * Add a review for a plugin.
 *
 * REV-2 (S219): the author is bound server-side to the authenticated
 * identity; the client no longer sends an author field.
 */
export async function addPluginReview(
	name: string,
	rating: number,
	opts?: { title?: string; text?: string }
): Promise<AddReviewResponse> {
	return (await apiPost(`${BASE}/${encodeURIComponent(name)}/reviews`, {
		rating,
		title: opts?.title || '',
		text: opts?.text || '',
	})) as AddReviewResponse;
}

/** Generate a plugin scaffold from template. */
export async function generatePluginTemplate(opts: {
	name: string;
	author?: string;
	description?: string;
	version?: string;
	hooks?: string[];
	permissions?: string[];
}): Promise<TemplateResponse> {
	return (await apiPost(`${BASE}/marketplace/template`, {
		name: opts.name,
		author: opts.author || 'Your Name',
		description: opts.description || 'A custom Opti-Oignon plugin.',
		version: opts.version || '1.0.0',
		hooks: opts.hooks || ['post_inference'],
		permissions: opts.permissions || [],
	})) as TemplateResponse;
}
