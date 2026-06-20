/**
 * Shared formatting and chart helpers for the benchmark sections (S169).
 * Extracted verbatim from BenchmarkV2Panel during its decomposition so the
 * per-section components share one implementation. Pure functions only; no
 * state, no side effects. Behaviour is unchanged.
 */

import type { BenchmarkV2ModelScore, BenchmarkV2TrendPoint } from '$lib/types';

export function scoreColor(score: number): string {
	if (score >= 0.8) return 'var(--oo-acc-400)';
	if (score >= 0.6) return 'var(--oo-fg-secondary)';
	if (score >= 0.4) return 'var(--oo-fg-tertiary)';
	return 'var(--oo-error)';
}

export function pct(v: number): string {
	return `${(v * 100).toFixed(1)}%`;
}

export function formatDuration(ms: number): string {
	if (ms < 1000) return `${ms.toFixed(0)}ms`;
	return `${(ms / 1000).toFixed(1)}s`;
}

export function formatDate(ts: number): string {
	if (!ts) return '-';
	return new Date(ts * 1000).toLocaleString();
}

// Radar chart SVG helper
export function radarPoints(scores: BenchmarkV2ModelScore, radius: number): string {
	const dims = [scores.accuracy_avg, scores.code_avg, scores.structure_avg, scores.speed_avg];
	const n = dims.length;
	const cx = radius;
	const cy = radius;
	return dims
		.map((v, i) => {
			const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
			const r = v * (radius - 10);
			const x = cx + r * Math.cos(angle);
			const y = cy + r * Math.sin(angle);
			return `${x},${y}`;
		})
		.join(' ');
}

export const radarLabels = ['Accuracy', 'Code', 'Structure', 'Speed'];
export const radarColors = ['var(--oo-acc-400)', 'var(--oo-radar-blue)', 'var(--oo-radar-sand)', 'var(--oo-radar-green)'];

export function radarLabelPos(index: number, radius: number): { x: number; y: number } {
	const n = 4;
	const angle = (Math.PI * 2 * index) / n - Math.PI / 2;
	const r = radius + 2;
	return { x: radius + r * Math.cos(angle), y: radius + r * Math.sin(angle) };
}

// Trend sparkline SVG helper (S89)
export function trendPath(points: BenchmarkV2TrendPoint[], w: number, h: number): string {
	if (points.length < 2) return '';
	const vals = points.map((p) => p.composite);
	const minV = Math.min(...vals);
	const maxV = Math.max(...vals);
	const range = maxV - minV || 1;
	const step = w / (points.length - 1);
	return vals
		.map((v, i) => {
			const x = i * step;
			const y = h - ((v - minV) / range) * (h - 8) - 4;
			return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
		})
		.join(' ');
}

// Role display names for recommendations
export function roleLabel(role: string): string {
	const map: Record<string, string> = { fast: 'Best Fast', quality: 'Best Quality', code: 'Best Code', value: 'Best Value' };
	return map[role] || role;
}

// Winner bar helper for H2H
export function winnerClass(winner: string, modelA: string): string {
	if (winner === 'tie') return 'tie';
	return winner === modelA ? 'a-wins' : 'b-wins';
}

export function formatCooldown(seconds: number): string {
	const m = Math.floor(seconds / 60);
	const s = Math.floor(seconds % 60);
	return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export function formatEventTime(ts: number): string {
	if (!ts) return '';
	return new Date(ts * 1000).toLocaleTimeString();
}
