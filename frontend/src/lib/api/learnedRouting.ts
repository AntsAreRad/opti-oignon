/**
 * API client for Learned Router.
 *
 * Provides typed access to training, config, classification,
 * and A/B metrics endpoints.
 */

import { apiGet, apiPut, apiPost } from './client';

// ============================================================================
// Types
// ============================================================================

export interface LearnedRouterStatus {
	available: boolean;
	trained: boolean;
	enabled: boolean;
	sklearn_available: boolean;
	sample_count: number;
	samples_since_retrain: number;
	min_training_samples: number;
	class_distribution: Record<string, number>;
	last_training: TrainingResult | null;
	model_type: string;
	confidence_threshold: number;
	auto_retrain_interval: number;
}

export interface TrainingResult {
	accuracy: number;
	n_samples: number;
	n_classes: number;
	trained_at: number;
	model_type: string;
	cv_folds: number;
	success: boolean;
	error: string;
}

export interface LearnedRouterConfig {
	enabled: boolean;
	model_type: string;
	confidence_threshold: number;
	min_training_samples: number;
	auto_retrain_interval: number;
	feature_max_features: number;
	feature_ngram_range: number[];
	logistic_max_iter: number;
	logistic_C: number;
	random_forest_n_estimators: number;
	random_forest_max_depth: number | null;
	max_stored_samples: number;
	cv_folds: number;
}

export interface LearnedRouterConfigUpdate {
	enabled?: boolean;
	model_type?: 'logistic' | 'random_forest';
	confidence_threshold?: number;
	min_training_samples?: number;
	auto_retrain_interval?: number;
	feature_max_features?: number;
	max_stored_samples?: number;
	cv_folds?: number;
}

export interface ClassifyRequest {
	query: string;
	yaml_task_type?: string;
}

export interface ClassifyResponse {
	ml_prediction: {
		task_type: string;
		confidence: number;
		model_type: string;
		fallback_used: boolean;
		top_classes: Array<{ task_type: string; confidence: number }>;
	};
	yaml_task_type: string;
	final_task_type: string;
	routing_source: 'learned' | 'yaml';
	confidence: number;
}

export interface HistogramBucket {
	bucket_min: number;
	bucket_max: number;
	count: number;
}

export interface TopDisagreement {
	ml_task_type: string;
	yaml_task_type: string;
	count: number;
}

export interface ABMetrics {
	total_decisions: number;
	learned_count: number;
	yaml_count: number;
	learned_ratio: number;
	avg_ml_confidence: number;
	avg_ml_confidence_learned: number;
	avg_ml_confidence_yaml: number;
	class_agreement_rate: number;
	top_disagreements: TopDisagreement[];
	decisions_by_source: Record<string, number>;
	window_hours: number;
	confidence_histogram: HistogramBucket[];
}

// ============================================================================
// Endpoints
// ============================================================================

export function getLearnedRouterStatus(): Promise<LearnedRouterStatus> {
	return apiGet<LearnedRouterStatus>('/api/routing/learned/status');
}

export function triggerTraining(): Promise<TrainingResult> {
	return apiPost<TrainingResult>('/api/routing/learned/train');
}

export function getLearnedRouterConfig(): Promise<LearnedRouterConfig> {
	return apiGet<LearnedRouterConfig>('/api/routing/learned/config');
}

export function updateLearnedRouterConfig(
	updates: LearnedRouterConfigUpdate
): Promise<{ success: boolean; updated: string[] }> {
	return apiPut('/api/routing/learned/config', updates);
}

export function classifyQuery(request: ClassifyRequest): Promise<ClassifyResponse> {
	return apiPost<ClassifyResponse>('/api/routing/learned/classify', request);
}

export function getABMetrics(windowHours = 24): Promise<ABMetrics> {
	return apiGet<ABMetrics>(`/api/routing/learned/metrics?window_hours=${windowHours}`);
}
