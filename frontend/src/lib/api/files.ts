/**
 * Typed API functions for file upload endpoints.
 *
 * Gere l'upload de fichiers texte via multipart/form-data
 * avec validation cote serveur (extension, taille).
 */

import { ApiError } from './client';
import type { FileUploadResponse, ImageUploadResponse } from '$lib/types';

const API_BASE = import.meta.env.VITE_API_URL ?? '';

function buildUrl(path: string): string {
	return new URL(`${API_BASE}${path}`, window.location.origin).toString();
}

/** Extensions autorisees (miroir de routes_files.py). */
export const ALLOWED_EXTENSIONS = new Set([
	'.r', '.R', '.py', '.sh', '.md', '.txt', '.json', '.yaml', '.yml',
	'.csv', '.tsv', '.xml', '.html', '.css', '.js', '.ts', '.jsx', '.tsx',
	'.c', '.cpp', '.h', '.java', '.go', '.rs', '.lua', '.rb', '.pl',
	'.toml', '.ini', '.cfg', '.conf', '.log', '.tex', '.bib', '.nf',
]);

/** Extensions d'image autorisees. */
export const ALLOWED_IMAGE_EXTENSIONS = new Set([
	'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp',
]);

/** Taille maximale en octets (500 KB). */
export const MAX_FILE_SIZE = 500_000;

/** Taille maximale image en octets (10 MB). */
export const MAX_IMAGE_SIZE = 10_000_000;

/**
 * Check if a file is valid before upload.
 * Return null if OK, an error message otherwise.
 */
export function validateFile(file: File): string | null {
	const dotIdx = file.name.lastIndexOf('.');
	if (dotIdx === -1) {
		return 'File has no extension';
	}
	const ext = file.name.substring(dotIdx);
	if (!ALLOWED_EXTENSIONS.has(ext) && !ALLOWED_EXTENSIONS.has(ext.toUpperCase())) {
		return `Unsupported extension: ${ext}`;
	}
	if (file.size > MAX_FILE_SIZE) {
		const sizeKB = (file.size / 1024).toFixed(0);
		return `File too large: ${sizeKB}KB (max ${MAX_FILE_SIZE / 1024}KB)`;
	}
	return null;
}

/** Upload un fichier via multipart/form-data. */
export async function uploadFile(file: File): Promise<FileUploadResponse> {
	const formData = new FormData();
	formData.append('file', file);

	try {
		const response = await fetch(buildUrl('/api/files/upload'), {
			method: 'POST',
			body: formData,
		});

		if (!response.ok) {
			let detail = response.statusText;
			try {
				const body = await response.json();
				detail = body.detail || detail;
			} catch {
				// Pas de corps JSON
			}
			throw new ApiError(response.status, `Upload failed: ${detail}`, detail);
		}

		return (await response.json()) as FileUploadResponse;
	} catch (err) {
		if (err instanceof ApiError) throw err;
		throw new ApiError(0, 'Upload connection failed', 'Unable to reach the API server');
	}
}


// ---------------------------------------------------------------------------
// Image upload
// ---------------------------------------------------------------------------

/**
 * Validate an image file before upload.
 * Returns null if OK, a descriptive error message otherwise.
 * Improved error messages with actionable guidance.
 */
export function validateImageFile(file: File): string | null {
	if (!file.name || file.size === 0) {
		return 'Empty or invalid file. Please select a valid image.';
	}
	const dotIdx = file.name.lastIndexOf('.');
	if (dotIdx === -1) {
		return 'No file extension detected. Supported formats: PNG, JPG, GIF, WebP, BMP.';
	}
	const ext = file.name.substring(dotIdx).toLowerCase();
	if (!ALLOWED_IMAGE_EXTENSIONS.has(ext)) {
		const supported = Array.from(ALLOWED_IMAGE_EXTENSIONS).map(e => e.replace('.', '').toUpperCase()).join(', ');
		return `Unsupported format "${ext}". Supported: ${supported}.`;
	}
	if (file.size > MAX_IMAGE_SIZE) {
		const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
		const maxMB = (MAX_IMAGE_SIZE / (1024 * 1024)).toFixed(0);
		return `Image too large (${sizeMB} MB). Maximum allowed: ${maxMB} MB. Try compressing or resizing the image.`;
	}
	return null;
}

/**
 * Check if a file is an image based on its MIME type or extension.
 */
export function isImageFile(file: File): boolean {
	if (file.type.startsWith('image/')) return true;
	const dotIdx = file.name.lastIndexOf('.');
	if (dotIdx === -1) return false;
	const ext = file.name.substring(dotIdx).toLowerCase();
	return ALLOWED_IMAGE_EXTENSIONS.has(ext);
}

/** Upload une image via multipart/form-data. */
export async function uploadImage(file: File): Promise<ImageUploadResponse> {
	const formData = new FormData();
	formData.append('file', file);

	try {
		const response = await fetch(buildUrl('/api/files/upload/image'), {
			method: 'POST',
			body: formData,
		});

		if (!response.ok) {
			let detail = response.statusText;
			try {
				const body = await response.json();
				detail = body.detail || detail;
			} catch {
				// Pas de corps JSON
			}
			throw new ApiError(response.status, `Image upload failed: ${detail}`, detail);
		}

		return (await response.json()) as ImageUploadResponse;
	} catch (err) {
		if (err instanceof ApiError) throw err;
		throw new ApiError(0, 'Image upload connection failed', 'Unable to reach the API server');
	}
}

/**
 * Convert a local image to base64 directly in the browser
 * (sans upload serveur). Utile pour un flux rapide.
 */
export function imageToBase64(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => {
			const result = reader.result as string;
			// Retirer le prefixe "data:image/...;base64,"
			const base64 = result.split(',')[1] || result;
			resolve(base64);
		};
		reader.onerror = () => reject(new Error('Failed to read image file'));
		reader.readAsDataURL(file);
	});
}
