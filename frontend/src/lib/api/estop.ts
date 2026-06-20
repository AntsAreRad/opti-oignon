/**
 * Emergency stop API (S215).
 *
 *   GET  /api/security/estop         -- status (flag + last stop/resume)
 *   POST /api/security/estop         -- engage (optional drop_to_bulbe)
 *   POST /api/security/estop/resume  -- resume (no ceremony; auth required)
 *
 * An availability/safety control, distinct from the search kill switch
 * (which re-enables through a ceremony). The stop is fail-tolerant per
 * step and fail-secure on the flag; the response reports per-step
 * outcomes honestly, including failed_steps.
 */
import { apiGet, apiPost } from './client';

export interface EmergencyStepOutcome {
	step: string;
	ok: boolean;
	detail?: Record<string, unknown>;
	error?: string;
}

export interface EmergencyActionResult {
	stopped: boolean;
	already_stopped?: boolean;
	was_stopped?: boolean;
	since?: number | null;
	drop_to_bulbe?: boolean;
	steps: EmergencyStepOutcome[];
	failed_steps: string[];
}

export interface EmergencyStopStatus {
	available: boolean;
	stopped: boolean;
	since?: number | null;
	by?: string;
	last_stop?: EmergencyActionResult | null;
	last_resume?: EmergencyActionResult | null;
}

export function getEmergencyStopStatus(): Promise<EmergencyStopStatus> {
	return apiGet<EmergencyStopStatus>('/api/security/estop');
}

export function engageEmergencyStop(
	dropToBulbe = false
): Promise<EmergencyActionResult> {
	return apiPost<EmergencyActionResult>('/api/security/estop', {
		drop_to_bulbe: dropToBulbe
	});
}

export function resumeFromEmergencyStop(
	warmupModel: string | null = null
): Promise<EmergencyActionResult> {
	return apiPost<EmergencyActionResult>('/api/security/estop/resume', {
		warmup_model: warmupModel
	});
}
