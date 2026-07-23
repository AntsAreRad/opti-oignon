/**
 * Fine-tuning data export and variant tracking API client.
 *
 * Manages training data export, quality scoring, variant registration,
 * and A/B comparison between base and fine-tuned models.
 */

import { apiGet, apiPost, apiDelete } from './client';
import type {
	FineTuneExportRequest,
	FineTuneExportResponse,
	FineTunePreviewResponse,
	FineTuneQualityResponse,
	FineTuneVariant,
	FineTuneVariantCreate,
	FineTuneVariantListResponse,
	FineTuneCompareRequest,
	FineTuneCompareResponse,
} from '$lib/types';

/** Export conversations as training data. */
export async function exportTrainingData(
	req: FineTuneExportRequest
): Promise<FineTuneExportResponse> {
	return (await apiPost('/api/fine-tune/export', req)) as FineTuneExportResponse;
}

/** Preview export with filters (sample data + quality scores). */
export async function previewExport(params?: {
	format?: string;
	model?: string;
	min_quality?: number;
	min_turns?: number;
	max_preview?: number;
}): Promise<FineTunePreviewResponse> {
	const query: Record<string, string> = {};
	if (params?.format) query.format = params.format;
	if (params?.model) query.model = params.model;
	if (params?.min_quality !== undefined) query.min_quality = String(params.min_quality);
	if (params?.min_turns !== undefined) query.min_turns = String(params.min_turns);
	if (params?.max_preview !== undefined) query.max_preview = String(params.max_preview);
	return (await apiGet('/api/fine-tune/export/preview', query)) as FineTunePreviewResponse;
}

/** Get quality scores for conversations. */
export async function getQualityScores(params?: {
	conversation_ids?: string;
	limit?: number;
}): Promise<FineTuneQualityResponse> {
	const query: Record<string, string> = {};
	if (params?.conversation_ids) query.conversation_ids = params.conversation_ids;
	if (params?.limit !== undefined) query.limit = String(params.limit);
	return (await apiGet('/api/fine-tune/quality', query)) as FineTuneQualityResponse;
}

/** List registered fine-tuned variants. */
export async function listVariants(params?: {
	base_model?: string;
	status?: string;
	limit?: number;
}): Promise<FineTuneVariantListResponse> {
	const query: Record<string, string> = {};
	if (params?.base_model) query.base_model = params.base_model;
	if (params?.status) query.status = params.status;
	if (params?.limit !== undefined) query.limit = String(params.limit);
	return (await apiGet('/api/fine-tune/variants', query)) as FineTuneVariantListResponse;
}

/** Register a new fine-tuned model variant. */
export async function registerVariant(
	req: FineTuneVariantCreate
): Promise<FineTuneVariant> {
	return (await apiPost('/api/fine-tune/variants', req)) as FineTuneVariant;
}

/** Unregister a fine-tuned variant. */
export async function unregisterVariant(
	variantId: string
): Promise<{ deleted: boolean; variant_id: string }> {
	return (await apiDelete(`/api/fine-tune/variants/${variantId}`)) as {
		deleted: boolean;
		variant_id: string;
	};
}

/** Run an A/B comparison between base and fine-tuned models. */
export async function runComparison(
	req: FineTuneCompareRequest
): Promise<FineTuneCompareResponse> {
	return (await apiPost('/api/fine-tune/compare', req)) as FineTuneCompareResponse;
}

/** Get a comparison result by ID. */
export async function getComparison(
	comparisonId: string
): Promise<FineTuneCompareResponse> {
	return (await apiGet(`/api/fine-tune/compare/${comparisonId}`)) as FineTuneCompareResponse;
}
