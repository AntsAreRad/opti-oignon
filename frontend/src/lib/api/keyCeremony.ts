/**
 * API client for Encryption Key Ceremony and PQC status -- S129.
 *
 * Covers:
 *   GET  /api/security/encryption       -- encryption status
 *   POST /api/security/encryption/setup  -- setup encryption (passphrase or random)
 *   GET  /api/security/pqc/status        -- PQC signature availability
 *   POST /api/security/pqc/generate-keys -- generate PQC keypair
 *   DELETE /api/security/pqc/keys        -- delete PQC keypair
 */

import { apiGet, apiPost, apiDelete } from './client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface EncryptionStatus {
	enabled: boolean;
	config_enabled: boolean;
	has_key: boolean;
	algorithm: string;
	kdf: string;
	crypto_backend: string;
	argon2_available: boolean;
	keyfile_exists: boolean;
	env_key_set: boolean;
	keyfile_path: string;
	format_version: number;
	secure_bytes_active?: boolean;
	key_mlocked?: boolean;
}

export interface EncryptionSetupResult {
	setup: boolean;
	detail: string;
	status: EncryptionStatus;
}

export interface PqcStatus {
	available: boolean;
	algorithm: string;
	config_enabled: boolean;
	effective_enabled: boolean;
	keypair_exists: boolean;
	keypair_path: string;
	key_algorithm?: string;
	public_key_size?: number;
	private_key_size?: number;
}

export interface PqcKeyGenResult {
	success: boolean;
	public_key_size: number;
	private_key_size: number;
	status: PqcStatus;
}

export interface PqcKeyDeleteResult {
	success: boolean;
	deleted: boolean;
	status: PqcStatus;
}

// ---------------------------------------------------------------------------
// Encryption endpoints
// ---------------------------------------------------------------------------

/** Get current encryption status. */
export async function getEncryptionStatus(): Promise<EncryptionStatus> {
	return apiGet<EncryptionStatus>('/api/security/encryption');
}

/** Setup encryption with a passphrase-derived key. */
export async function setupEncryptionPassphrase(passphrase: string): Promise<EncryptionSetupResult> {
	return apiPost<EncryptionSetupResult>('/api/security/encryption/setup', {
		mode: 'passphrase',
		passphrase,
	});
}

/** Setup encryption with a randomly generated key. */
export async function setupEncryptionRandom(): Promise<EncryptionSetupResult> {
	return apiPost<EncryptionSetupResult>('/api/security/encryption/setup', {
		mode: 'random',
	});
}

// ---------------------------------------------------------------------------
// PQC endpoints
// ---------------------------------------------------------------------------

/** Get PQC signature availability and key status. */
export async function getPqcStatus(): Promise<PqcStatus> {
	return apiGet<PqcStatus>('/api/security/pqc/status');
}

/** Generate a new PQC keypair for backup signing. */
export async function generatePqcKeys(): Promise<PqcKeyGenResult> {
	return apiPost<PqcKeyGenResult>('/api/security/pqc/generate-keys');
}

/** Delete the PQC keypair from disk. */
export async function deletePqcKeys(): Promise<PqcKeyDeleteResult> {
	return apiDelete<PqcKeyDeleteResult>('/api/security/pqc/keys');
}

// ---------------------------------------------------------------------------
// Client-side passphrase strength scoring (zxcvbn-style, no server call)
// ---------------------------------------------------------------------------

export type StrengthLevel = 'weak' | 'fair' | 'good' | 'strong' | 'very_strong';

export interface StrengthResult {
	score: number;          // 0-4
	level: StrengthLevel;
	label: string;
	color: string;          // CSS variable reference
	percent: number;        // 0-100 for progress bar
	feedback: string;
}

const STRENGTH_MAP: Record<number, { level: StrengthLevel; label: string; color: string }> = {
	0: { level: 'weak',        label: 'Weak',        color: 'var(--oo-error)' },
	1: { level: 'fair',        label: 'Fair',        color: 'var(--oo-warning)' },
	2: { level: 'good',        label: 'Good',        color: 'var(--oo-tobacco)' },
	3: { level: 'strong',      label: 'Strong',      color: 'var(--oo-sage)' },
	4: { level: 'very_strong', label: 'Very Strong', color: 'var(--oo-sage)' },
};

/**
 * Estimate passphrase strength purely client-side.
 *
 * Simplified zxcvbn-like scoring based on:
 *   - Length (primary factor)
 *   - Character class diversity (upper, lower, digits, symbols)
 *   - Repeated character penalty
 *   - Common pattern penalty (123, abc, qwerty substrings)
 */
export function scorePassphrase(passphrase: string): StrengthResult {
	if (!passphrase) {
		return { score: 0, ...STRENGTH_MAP[0], percent: 0, feedback: 'Enter a passphrase' };
	}

	let score = 0;
	const len = passphrase.length;

	// Length scoring (most important factor)
	if (len >= 20) score += 2;
	else if (len >= 14) score += 1.5;
	else if (len >= 10) score += 1;
	else if (len >= 8) score += 0.5;

	// Character class diversity
	const hasLower = /[a-z]/.test(passphrase);
	const hasUpper = /[A-Z]/.test(passphrase);
	const hasDigit = /[0-9]/.test(passphrase);
	const hasSymbol = /[^a-zA-Z0-9]/.test(passphrase);
	const classes = [hasLower, hasUpper, hasDigit, hasSymbol].filter(Boolean).length;
	if (classes >= 4) score += 1.5;
	else if (classes >= 3) score += 1;
	else if (classes >= 2) score += 0.5;

	// Bonus for very long passphrases (multi-word)
	if (len >= 25 && passphrase.includes(' ')) score += 0.5;

	// Penalties
	const uniqueChars = new Set(passphrase.toLowerCase()).size;
	const uniqueRatio = uniqueChars / len;
	if (uniqueRatio < 0.4) score -= 1;  // Lots of repeated chars

	// Common patterns penalty
	const lower = passphrase.toLowerCase();
	const commonPatterns = [
		'123456', 'password', 'qwerty', 'abc123', 'letmein',
		'admin', '111111', 'iloveyou', 'welcome', 'monkey',
		'master', 'dragon', 'login', 'princess', 'football',
	];
	for (const pat of commonPatterns) {
		if (lower.includes(pat)) {
			score -= 1.5;
			break;
		}
	}

	// Sequential chars penalty
	if (/(.)\1{3,}/.test(passphrase)) score -= 0.5;
	if (/(?:abc|bcd|cde|def|efg|123|234|345|456|567|678|789)/i.test(passphrase)) score -= 0.5;

	// Clamp to 0-4
	const finalScore = Math.max(0, Math.min(4, Math.round(score)));
	const info = STRENGTH_MAP[finalScore];

	let feedback = '';
	if (finalScore <= 1) {
		if (len < 10) feedback = 'Try a longer passphrase (14+ characters recommended)';
		else if (classes < 3) feedback = 'Add uppercase, numbers, or symbols';
		else feedback = 'Avoid common words and patterns';
	} else if (finalScore === 2) {
		feedback = 'Decent, but a longer passphrase would be stronger';
	} else {
		feedback = '';
	}

	return {
		score: finalScore,
		...info,
		percent: (finalScore / 4) * 100,
		feedback,
	};
}
