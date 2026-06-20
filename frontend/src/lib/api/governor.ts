/**
 * API client for the Resource Governor (S227, Resource Governor cycle Bloc 4).
 *
 * Defines the contract the status card (GovernorPanel.svelte) consumes from the
 * backend route (opti_oignon/api/routes_governor.py): the status surface
 * (snapshot with provenance, capacity, learned ceiling, pressure, queue depth
 * and the external-Ollama advisory), the bounded recent-decisions ring, a
 * per-model eviction, and the config read/write surface.
 *
 * The governor is mode-free: every endpoint behaves identically in Daily and
 * Bulbe (it is a local resource control with no egress, no secrets, and no
 * state mutation on user content). Eviction is fail-open at the engine: a
 * false result is not an error -- Ollama's own LRU then carries the pressure.
 */

import { apiGet, apiPost } from './client';

const BASE = '/api/governor';

/** One loaded model in the snapshot (mirrors LoadedModelView.to_dict). */
export interface LoadedModelView {
	name: string;
	size_vram_bytes: number;
	[key: string]: unknown;
}

/** The measurement snapshot (mirrors ResourceSnapshot.to_dict). */
export interface ResourceSnapshot {
	taken_at: number;
	ttl_s: number;
	loaded: LoadedModelView[];
	backend_resident: LoadedModelView[];
	capacity_gb: number | null;
	capacity_source: string;
	vram_in_use_gb: number;
	vram_available_gb: number | null;
	vram_status: string;
	ram_available_mb: number;
	/** Honest provenance: which read paths contributed. */
	sources: string[];
}

/** The R-02 pressure signal (mirrors pressure_state). */
export interface PressureState {
	level: string;
	ratio: number | null;
	effective_capacity_gb: number | null;
	in_use_gb: number;
	soft_threshold: number;
	hard_threshold: number;
	refusal_rate: number;
	refusals_in_window: number;
	decisions_in_window: number;
	refusal_window_s: number;
	keep_alive_overridden: boolean;
}

/** The external-Ollama advisory (mirrors ollama_limits_advisory). */
export interface OllamaLimitsAdvisory {
	status: string;
	[key: string]: unknown;
}

/** The /status body. */
export interface GovernorStatus {
	enabled: boolean;
	snapshot: ResourceSnapshot;
	learned_ceiling_gb: number | null;
	pressure: PressureState;
	queue_depth: number;
	ollama_limits: OllamaLimitsAdvisory;
}

/** One recorded admission decision (mirrors AdaptStore.recent_decisions). */
export interface AdmissionRecord {
	id: number;
	ts: number;
	caller: string;
	model: string;
	requested_ctx: number | null;
	admitted_ctx: number | null;
	decision: string;
	reason: string;
}

/** The /admissions body. */
export interface AdmissionsView {
	admissions: AdmissionRecord[];
	count: number;
	limit: number;
	ring_size: number;
}

/** The /evict result (fail-open: evicted false is not an error). */
export interface EvictResult {
	evicted: boolean;
	model: string;
	note: string;
}

/** The GET /config body. */
export interface GovernorConfigView {
	config: Record<string, unknown>;
	writable_keys: string[];
	read_only_keys: Record<string, string>;
}

/** The POST /config result. */
export interface ConfigWriteResult {
	applied: Record<string, { old: unknown; new: unknown }>;
	persisted: boolean;
	effective: string;
	notes: string[];
}

export function getGovernorStatus(): Promise<GovernorStatus> {
	return apiGet<GovernorStatus>(`${BASE}/status`);
}

export function getGovernorAdmissions(limit = 20): Promise<AdmissionsView> {
	return apiGet<AdmissionsView>(`${BASE}/admissions`, { limit: String(limit) });
}

export function evictGovernorModel(model: string): Promise<EvictResult> {
	return apiPost<EvictResult>(`${BASE}/evict`, { model });
}

export function getGovernorConfig(): Promise<GovernorConfigView> {
	return apiGet<GovernorConfigView>(`${BASE}/config`);
}

export function setGovernorConfig(
	changes: Record<string, unknown>
): Promise<ConfigWriteResult> {
	return apiPost<ConfigWriteResult>(`${BASE}/config`, changes);
}
