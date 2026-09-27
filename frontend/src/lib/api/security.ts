/**
 * The security grade and the recent security events, read through the API
 * client: the status card's badge and the Security page read the grade
 * here, and nowhere else.
 *
 * The grade (`GET /api/security/status`) is a letter, a score out of a
 * maximum, and the checks it is made of, each with its points and a
 * detail. The events (`GET /api/security/audit`) gather sign-in activity
 * (logins, failed logins, registrations, password changes), sandbox
 * blocks, sign-in lockouts and detected search injections; they are not
 * the hash-chained audit log, which the audit chain's own panel reads.
 */

import { apiGet } from './client';

/** One check the grade is made of. */
export interface SecurityCheck {
	name: string;
	points: number;
	max_points: number;
	passed: boolean;
	detail: string;
}

/** The grade: its letter, its score out of the maximum, and its checks. */
export interface SecurityStatus {
	grade: string;
	score: number;
	max_score: number;
	checks: SecurityCheck[];
}

/** One recent security event. */
export interface SecurityEvent {
	source: string;
	event_type: string;
	action: string;
	severity: string;
	timestamp: number;
	details?: Record<string, unknown>;
}

/** Reads the grade. */
export function getSecurityStatus(): Promise<SecurityStatus> {
	return apiGet<SecurityStatus>('/api/security/status');
}

/** Reads the `limit` most recent security events. */
export function getSecurityEvents(limit: number): Promise<{ events: SecurityEvent[] }> {
	return apiGet<{ events: SecurityEvent[] }>('/api/security/audit', { limit: String(limit) });
}
