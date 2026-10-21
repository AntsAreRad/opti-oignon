/**
 * What the key setup of the Security screen says: where it starts, what a
 * success says, what a refusal says. Each line follows what the server
 * answered and promises nothing it did not do.
 *
 * Dependency-free, so the contracts run it under Node (KP1, KP2).
 */

/** The encryption status fields the setup reads. */
export interface KeySetupStatus {
	enabled?: boolean;
	has_key?: boolean;
	env_key_set?: boolean;
	keyfile_exists?: boolean;
	keyfile_path?: string;
}

/** The answer of POST /api/security/encryption/setup. */
export interface KeySetupAnswer {
	setup: boolean;
	detail?: string;
	/** "existing": the key already here was enabled; "made": a key was made. Absent from an older server. */
	key?: string;
	/** How many of the platform's databases are still kept in plaintext. */
	plaintext_stores?: number;
	/** Whether the key file setup made is wrapped under a passphrase; null when it made none. */
	wrapped?: boolean | null;
	status?: KeySetupStatus | null;
}

/** An HTTP refusal as the API client's reader gives it, or null when the error was none. */
export interface KeySetupRefusal {
	status: number;
	detail: string;
}

/**
 * Where the setup starts. "existing": a key is here -- one loads, a key file
 * is there, or OPTI_ENCRYPTION_KEY is set -- and enabling takes it, making
 * none; when the server cannot open it, its refusal says why. "new": no key
 * anywhere; one is made. While encryption is off the server loads no key,
 * so a key file is the only sign of the one the databases were encrypted
 * with.
 */
export type KeySetupStart = 'existing' | 'new';

export function keySetupStart(status: KeySetupStatus | null | undefined): KeySetupStart {
	if (status?.has_key || status?.env_key_set || status?.keyfile_exists) return 'existing';
	return 'new';
}

/** What the panel says at each start. */
export const KEY_SETUP_TEXT: Record<KeySetupStart, string> = {
	existing:
		'A key is already here: a key file, or OPTI_ENCRYPTION_KEY. Enabling takes it: it makes no key and asks ' +
		'for no passphrase. If the server cannot open it, it says why.',
	new:
		'Choose how to make the encryption key. Either way it is saved in the key file: keep a copy of it off this ' +
		'machine, since without it your data does not open. Under a passphrase the key file is wrapped: a copy of ' +
		'it opens your data only with that passphrase, which the server needs as OPTI_KEYFILE_PASSPHRASE at each ' +
		'start.'
};

/** What a setup that succeeded says: the key the server enabled or made, as its answer names it. */
export function keySetupSucceeded(mode: 'passphrase' | 'random', answer: KeySetupAnswer): string {
	const plain = answer.plaintext_stores ?? 0;
	const left =
		plain > 0
			? ` ${plain} database(s) of the platform are still kept in plaintext and do not open under a key: stop ` +
				'the platform and run python3 -m opti_oignon.db_encryption encrypt.'
			: '';
	if (answer.key === 'existing') {
		return (
			'Encryption enabled with the key already here: no key was made.' +
			left +
			(mode === 'passphrase' ? ' The passphrase you typed was not used.' : '')
		);
	}
	if (answer.key !== 'made') {
		return (answer.detail ? `Encryption enabled: ${answer.detail}.` : 'Encryption enabled.') + left;
	}
	const path = answer.status?.keyfile_path;
	const where = path ? ` (${path})` : '';
	if (mode === 'passphrase') {
		return (
			`A key was made and saved in the key file${where}, wrapped under your passphrase. The server opens it ` +
			'only with OPTI_KEYFILE_PASSPHRASE set to that passphrase at each start. Keep a copy of the key file ' +
			'too: the passphrase alone does not open your data.'
		);
	}
	if (answer.wrapped) {
		return (
			`A random key was made and saved in the key file${where}, wrapped under the server's ` +
			'OPTI_KEYFILE_PASSPHRASE. Keep a copy of the key file and of that passphrase: without both your data ' +
			'does not open.'
		);
	}
	return (
		`A random key was made and saved in the key file${where}. Keep a copy of it off this machine: without it ` +
		'your data does not open.'
	);
}

/**
 * What a setup that failed says: the server's own reason when it refused,
 * else what the error says, never a bare status.
 */
export function keySetupFailed(refusal: KeySetupRefusal | null, error: unknown): string {
	if (refusal?.detail) return refusal.detail;
	if (error && typeof error === 'object') {
		const failure = error as { detail?: unknown; message?: unknown };
		if (typeof failure.detail === 'string' && failure.detail) return failure.detail;
		if (typeof failure.message === 'string' && failure.message) return failure.message;
	}
	return 'Setup failed';
}
