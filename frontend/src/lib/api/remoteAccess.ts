/**
 * API client for Remote Access endpoints --.
 *
 * Covers:
 *   GET  /api/security/remote-access/status              -- status
 *   POST /api/security/remote-access/enable               -- enable ceremony
 *   POST /api/security/remote-access/disable              -- disable
 *   POST /api/security/remote-access/generate-client-cert -- generate cert
 *   POST /api/security/remote-access/revoke-client-cert   -- revoke cert
 *   GET  /api/security/remote-access/client-certs         -- list certs
 */

import { apiGet, apiPost } from './client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ClientCertInfo {
	device_name: string;
	fingerprint: string;
	created_at: number;
	expires_at: number;
	revoked: boolean;
	serial_number?: string;
}

export interface TLSStatus {
	enabled: boolean;
	ca_exists: boolean;
	server_cert_exists: boolean;
	ca_fingerprint: string;
	server_cert_expiry: string;
	days_until_expiry: number;
	client_certs: ClientCertInfo[];
	warning: string;
}

export interface RemoteAccessStatus {
	remote_access_allowed: boolean;
	tls: TLSStatus;
}

export interface EnableResult {
	success: boolean;
	message?: string;
	error?: string;
}

export interface GenerateCertResult {
	success: boolean;
	device_name?: string;
	fingerprint?: string;
	p12_path?: string;
	expires_at?: string;
	message?: string;
	error?: string;
}

export interface RevokeResult {
	success: boolean;
	message?: string;
	error?: string;
	already_revoked?: boolean;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export async function getRemoteAccessStatus(): Promise<RemoteAccessStatus> {
	return apiGet<RemoteAccessStatus>('/api/security/remote-access/status');
}

export async function enableRemoteAccess(passphrase: string): Promise<EnableResult> {
	return apiPost<EnableResult>('/api/security/remote-access/enable', {
		passphrase,
		confirm: true,
	});
}

export async function disableRemoteAccess(): Promise<EnableResult> {
	return apiPost<EnableResult>('/api/security/remote-access/disable', {});
}

export async function generateClientCert(
	deviceName: string,
	passphrase: string,
): Promise<GenerateCertResult> {
	return apiPost<GenerateCertResult>(
		'/api/security/remote-access/generate-client-cert',
		{ device_name: deviceName, passphrase },
	);
}

export async function revokeClientCert(deviceName: string): Promise<RevokeResult> {
	return apiPost<RevokeResult>('/api/security/remote-access/revoke-client-cert', {
		device_name: deviceName,
	});
}

export async function listClientCerts(): Promise<{ client_certs: ClientCertInfo[] }> {
	return apiGet<{ client_certs: ClientCertInfo[] }>(
		'/api/security/remote-access/client-certs',
	);
}
