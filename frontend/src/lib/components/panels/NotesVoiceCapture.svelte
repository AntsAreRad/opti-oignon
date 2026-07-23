<!--
  NotesVoiceCapture.svelte (Notes feature N.5 UI half)
  In-browser voice capture for the active note: MediaRecorder records the
  microphone, the recording uploads as an encrypted audio attachment over the
  upload route (sealed server-side under a per-attachment subkey; nothing
  plaintext touches disk), and each voice note offers the opt-in
  transcription as preview-then-approve: the first run returns the transcript
  for review without persisting, the explicit approval writes it back. A
  structured refusal (the fail-secure sandbox gate, the absent opt-in
  transcribe extra) is shown with its reason, never silently dropped.
  Playback decrypts in memory through a short-lived object URL, revoked on
  swap and destroy. Design-system tokens only (--oo-*); lucide-svelte icons
  through Icon.
-->
<script lang="ts">
	import { onDestroy } from 'svelte';
	import { Button, Card, Icon, EmptyState, InlineError } from '$lib/ds';
	import {
		audioAttachments,
		mediaLoading,
		mediaError,
		loadAttachments,
		uploadNoteAttachment,
		removeAttachment,
		transcribeAttachment
	} from '$lib/stores/attachments';
	import { fetchAttachmentBlob, type AttachmentRecord } from '$lib/api/attachments';
	import type { TranscriptionResult } from '$lib/api/transcription';
	import { toastSuccess, toastError } from '$lib/stores/notifications';

	/** The note whose voice notes this control captures and lists. */
	export let noteId: string;

	// Load the note's attachments (idempotent in the store, so the gallery's
	// identical call is a no-op for the same note).
	$: if (noteId) {
		void loadAttachments(noteId);
	}

	// -- Capture --

	const captureSupported =
		typeof navigator !== 'undefined' &&
		!!navigator.mediaDevices &&
		typeof MediaRecorder !== 'undefined';

	let recorder: MediaRecorder | null = null;
	let chunks: Blob[] = [];
	let recording = false;
	let uploading = false;

	async function startRecording(): Promise<void> {
		if (!captureSupported || recording) return;
		try {
			const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
			chunks = [];
			recorder = new MediaRecorder(stream);
			recorder.ondataavailable = (e: BlobEvent) => {
				if (e.data.size > 0) chunks.push(e.data);
			};
			recorder.onstop = () => {
				stream.getTracks().forEach((t) => t.stop());
				void uploadRecording();
			};
			recorder.start();
			recording = true;
		} catch {
			toastError('Microphone unavailable or permission denied');
		}
	}

	function stopRecording(): void {
		if (!recorder || !recording) return;
		recording = false;
		recorder.stop();
		recorder = null;
	}

	async function uploadRecording(): Promise<void> {
		if (chunks.length === 0) return;
		const blob = new Blob(chunks, { type: 'audio/webm' });
		chunks = [];
		uploading = true;
		try {
			await uploadNoteAttachment(noteId, 'audio', blob, 'voice-note.webm');
			toastSuccess('Voice note uploaded');
		} catch {
			toastError('Failed to upload voice note');
		} finally {
			uploading = false;
		}
	}

	// -- Playback (in-memory object URL, revoked on swap / destroy) --

	let playingId: string | null = null;
	let playUrl: string | null = null;

	function revokePlayUrl(): void {
		if (playUrl) {
			URL.revokeObjectURL(playUrl);
			playUrl = null;
		}
		playingId = null;
	}

	async function togglePlay(item: AttachmentRecord): Promise<void> {
		if (playingId === item.id) {
			revokePlayUrl();
			return;
		}
		revokePlayUrl();
		try {
			const blob = await fetchAttachmentBlob(item.id);
			playUrl = URL.createObjectURL(blob);
			playingId = item.id;
		} catch {
			toastError('Failed to load audio');
		}
	}

	// -- Transcription: preview-then-approve --

	let busyId: string | null = null;
	let previews: Record<string, TranscriptionResult> = {};

	async function transcribe(item: AttachmentRecord): Promise<void> {
		busyId = item.id;
		try {
			const result = await transcribeAttachment(item.id, false);
			previews = { ...previews, [item.id]: result };
			if (result.refused) {
				toastError(result.reason || 'Transcription refused');
			}
		} catch {
			toastError('Transcription request failed');
		} finally {
			busyId = null;
		}
	}

	async function approveTranscript(item: AttachmentRecord): Promise<void> {
		busyId = item.id;
		try {
			const result = await transcribeAttachment(item.id, true);
			previews = { ...previews, [item.id]: result };
			if (result.refused) {
				toastError(result.reason || 'Transcription refused');
			} else if (result.ok && result.written_back) {
				toastSuccess('Transcript saved');
			}
		} catch {
			toastError('Transcription request failed');
		} finally {
			busyId = null;
		}
	}

	function discardPreview(id: string): void {
		const { [id]: _gone, ...rest } = previews;
		previews = rest;
	}

	async function remove(item: AttachmentRecord): Promise<void> {
		busyId = item.id;
		try {
			if (playingId === item.id) revokePlayUrl();
			await removeAttachment(item.id);
			discardPreview(item.id);
		} catch {
			toastError('Failed to delete voice note');
		} finally {
			busyId = null;
		}
	}

	function shortStamp(iso: string): string {
		return iso ? iso.replace('T', ' ').slice(0, 16) : '';
	}

	onDestroy(() => {
		if (recorder && recording) {
			try {
				recorder.stop();
			} catch {
				// Already stopped.
			}
		}
		revokePlayUrl();
	});
</script>

<Card padding="md">
	<div class="vc-head">
		<div class="vc-title">
			<Icon name="mic" size="sm" />
			<h3>Voice notes</h3>
		</div>
		{#if !captureSupported}
			<span class="vc-unsupported">Recording is not supported in this browser</span>
		{:else if recording}
			<Button variant="danger" size="sm" iconLeft="square" on:click={stopRecording}>
				Stop
			</Button>
		{:else}
			<Button
				variant="secondary"
				size="sm"
				iconLeft="mic"
				loading={uploading}
				on:click={startRecording}
			>
				Record
			</Button>
		{/if}
	</div>

	{#if $mediaError}
		<InlineError
			message={$mediaError}
			onRetry={() => loadAttachments(noteId, true)}
			retrying={$mediaLoading}
		/>
	{:else if $audioAttachments.length === 0}
		{#if !$mediaLoading}
			<EmptyState
				icon="mic"
				title="No voice notes yet"
				description="Record one to attach it to this note."
			/>
		{/if}
	{:else}
		<ul class="vc-list">
			{#each $audioAttachments as item (item.id)}
				<li class="vc-item">
					<div class="vc-row">
						<span class="vc-meta">
							<Icon name="mic" size="sm" />
							<span class="vc-stamp">{shortStamp(item.created_at)}</span>
							<span class="vc-mime">{item.mime || 'audio'}</span>
						</span>
						<span class="vc-actions">
							<Button
								variant="ghost"
								size="sm"
								iconLeft={playingId === item.id ? 'square' : 'play'}
								on:click={() => togglePlay(item)}
							>
								{playingId === item.id ? 'Stop' : 'Play'}
							</Button>
							<Button
								variant="ghost"
								size="sm"
								iconLeft="eye"
								loading={busyId === item.id}
								on:click={() => transcribe(item)}
							>
								Transcribe
							</Button>
							<Button
								variant="ghost"
								size="sm"
								iconLeft="trash-2"
								loading={busyId === item.id}
								on:click={() => remove(item)}
							>
								Delete
							</Button>
						</span>
					</div>

					{#if playingId === item.id && playUrl}
						<audio class="vc-audio" controls autoplay src={playUrl}></audio>
					{/if}

					{#if previews[item.id] && previews[item.id].refused}
						<p class="vc-refused">
							Transcription refused: {previews[item.id].reason || 'unavailable'}
						</p>
					{:else if previews[item.id] && previews[item.id].ok && !previews[item.id].written_back}
						<div class="vc-preview">
							<p class="vc-preview-label">Transcript preview (not saved)</p>
							<p class="vc-text">{previews[item.id].transcript_text}</p>
							<div class="vc-preview-actions">
								<Button
									variant="primary"
									size="sm"
									iconLeft="check"
									loading={busyId === item.id}
									on:click={() => approveTranscript(item)}
								>
									Approve and save
								</Button>
								<Button variant="ghost" size="sm" on:click={() => discardPreview(item.id)}>
									Discard
								</Button>
							</div>
						</div>
					{:else if item.transcript_text}
						<p class="vc-text vc-saved">{item.transcript_text}</p>
					{/if}
				</li>
			{/each}
		</ul>
	{/if}
</Card>

<style>
	.vc-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-3);
		margin-bottom: var(--oo-space-3);
	}

	.vc-title {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
	}

	.vc-title h3 {
		margin: 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-primary);
	}

	.vc-unsupported {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
	}

	.vc-list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}

	.vc-item {
		border-bottom: 1px solid var(--oo-bd-default);
		padding-bottom: var(--oo-space-2);
	}

	.vc-item:last-child {
		border-bottom: none;
		padding-bottom: 0;
	}

	.vc-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-2);
		flex-wrap: wrap;
	}

	.vc-meta {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-xs);
	}

	.vc-mime {
		color: var(--oo-fg-faint);
	}

	.vc-actions {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-1);
	}

	.vc-audio {
		width: 100%;
		margin-top: var(--oo-space-2);
	}

	.vc-refused {
		margin: var(--oo-space-2) 0 0;
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
	}

	.vc-preview {
		margin-top: var(--oo-space-2);
		padding: var(--oo-space-2);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
		background: var(--oo-bg-elevated);
	}

	.vc-preview-label {
		margin: 0 0 var(--oo-space-1);
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-tertiary);
	}

	.vc-text {
		margin: 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-primary);
		white-space: pre-wrap;
	}

	.vc-saved {
		margin-top: var(--oo-space-2);
		color: var(--oo-fg-secondary);
	}

	.vc-preview-actions {
		display: flex;
		gap: var(--oo-space-2);
		margin-top: var(--oo-space-2);
	}
</style>
