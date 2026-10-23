/**
 * WebAuthn transport between the server's JSON and the browser's credential API.
 *
 * The server sends options as python-fido2 shapes them: `{ publicKey: {...} }` with every binary field already in
 * base64url. The browser's API wants those fields as ArrayBuffers, and its answers come back as ArrayBuffers the
 * server reads as base64url again. Dependency-free, so that the contracts run it under Node.
 */

/** A base64url text as the bytes it encodes. */
export function base64urlToBuffer(base64url: string): ArrayBuffer {
	let base64 = base64url.replace(/-/g, '+').replace(/_/g, '/');
	while (base64.length % 4 !== 0) base64 += '=';
	const binary = atob(base64);
	const bytes = new Uint8Array(binary.length);
	for (let i = 0; i < binary.length; i++) {
		bytes[i] = binary.charCodeAt(i);
	}
	return bytes.buffer;
}

/** Bytes as base64url text, without padding. */
export function bufferToBase64url(buffer: ArrayBuffer): string {
	const bytes = new Uint8Array(buffer);
	let binary = '';
	for (const byte of bytes) {
		binary += String.fromCharCode(byte);
	}
	return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

type Json = Record<string, any>;

/** The `publicKey` member of the server's options, or the options themselves when they carry no such member. */
function unwrap(serverOptions: Json): Json {
	const inner = serverOptions && typeof serverOptions.publicKey === 'object' ? serverOptions.publicKey : serverOptions;
	return { ...(inner || {}) };
}

function descriptors(list: unknown): unknown {
	if (!Array.isArray(list)) return list;
	return list.map((d: Json) => ({ ...d, id: typeof d.id === 'string' ? base64urlToBuffer(d.id) : d.id }));
}

/** The options `navigator.credentials.create` takes, from what the server's registration begin answered. */
export function creationOptions(serverOptions: Json): PublicKeyCredentialCreationOptions {
	const opts = unwrap(serverOptions);
	if (typeof opts.challenge === 'string') opts.challenge = base64urlToBuffer(opts.challenge);
	if (opts.user && typeof opts.user.id === 'string') opts.user = { ...opts.user, id: base64urlToBuffer(opts.user.id) };
	if (opts.excludeCredentials) opts.excludeCredentials = descriptors(opts.excludeCredentials);
	return opts as PublicKeyCredentialCreationOptions;
}

/** The options `navigator.credentials.get` takes, from the options the login's first step answered. */
export function requestOptions(serverOptions: Json): PublicKeyCredentialRequestOptions {
	const opts = unwrap(serverOptions);
	if (typeof opts.challenge === 'string') opts.challenge = base64urlToBuffer(opts.challenge);
	if (opts.allowCredentials) opts.allowCredentials = descriptors(opts.allowCredentials);
	return opts as PublicKeyCredentialRequestOptions;
}

/** A new key's attestation, as the server's registration complete reads it. */
export function serializeAttestation(cred: PublicKeyCredential): Json {
	const response = cred.response as AuthenticatorAttestationResponse;
	return {
		id: cred.id,
		rawId: bufferToBase64url(cred.rawId),
		type: cred.type,
		response: {
			clientDataJSON: bufferToBase64url(response.clientDataJSON),
			attestationObject: bufferToBase64url(response.attestationObject)
		}
	};
}

/** A key's assertion, as the login's second step reads it. */
export function serializeAssertion(cred: PublicKeyCredential): Json {
	const response = cred.response as AuthenticatorAssertionResponse;
	return {
		id: cred.id,
		rawId: bufferToBase64url(cred.rawId),
		type: cred.type,
		response: {
			clientDataJSON: bufferToBase64url(response.clientDataJSON),
			authenticatorData: bufferToBase64url(response.authenticatorData),
			signature: bufferToBase64url(response.signature),
			userHandle: response.userHandle ? bufferToBase64url(response.userHandle) : null
		}
	};
}
