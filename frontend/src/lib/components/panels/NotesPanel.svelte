<!--
  NotesPanel.svelte (Notes feature N.2 core UI)
  The Notes page: a master-detail surface over the per-user /api/notes store,
  built on the lib/ds primitives (Card, Button, Input, Icon, EmptyState,
  InlineError, Modal). The left pane lists notes with search and tag chips; the
  right pane edits the selected note (title, body, tags, pinned) and persists
  whole note state through the store. The body is plain text / markdown for now,
  carried opaque (the Yjs text CRDT is N.8). The N.3 selection-action panel is
  wired in separately. Design-system tokens only (--oo-*); lucide-svelte icons
  through Icon.
-->
<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { Button, Card, Input, Icon, EmptyState, InlineError, Modal } from '$lib/ds';
	import NoteActionPanel from './NoteActionPanel.svelte';
	import NotesVoiceCapture from './NotesVoiceCapture.svelte';
	import NotesMediaGallery from './NotesMediaGallery.svelte';
	import NotesDrawingCanvas from './NotesDrawingCanvas.svelte';
	import {
		notes,
		activeNote,
		filteredNotes,
		loading,
		error,
		search,
		loadNotes,
		selectNote,
		createNote,
		saveNote,
		toggleMobileAllowed,
		removeNote
	} from '$lib/stores/notes';
	import { encodeNoteBody, decodeNoteBody, type NoteRecord } from '$lib/api/notes';
	import { appendNoteUpdate } from '$lib/api/noteUpdates';
	import { toastSuccess, toastError } from '$lib/stores/notifications';

	// Editor form state, loaded from the active note when its id changes.
	let loadedId: string | null = null;
	let editTitle = '';
	let editBody = '';
	let editTags = '';
	let editPinned = false;
	let saving = false;
	// N.9: the phone-sync opt-in is NOT part of the dirty/save form.
	// It is an immediate trust control in CONFIRMED posture: the click only
	// REQUESTS the flip (the checkbox never self-toggles), the rendered
	// state comes from the server-returned record alone (the store replaces
	// the row from the PATCH response), and the control is disabled while
	// the request is in flight.
	let mobileBusy = false;

	async function flipMobileAllowed(): Promise<void> {
		if (!$activeNote || mobileBusy) return;
		mobileBusy = true;
		try {
			const updated = await toggleMobileAllowed($activeNote);
			toastSuccess(
				updated.mobile_allowed
					? 'Note will sync to phone-class devices'
					: 'Note will no longer sync to phone-class devices'
			);
		} catch {
			toastError('Failed to update phone sync');
		} finally {
			mobileBusy = false;
		}
	}

	// The body textarea element, exposed for the N.3 selection-action panel.
	let bodyEl: HTMLTextAreaElement | null = null;
	let selStart = 0;
	let selEnd = 0;
	let selection = '';

	function updateSelection(): void {
		if (!bodyEl) return;
		selStart = bodyEl.selectionStart ?? 0;
		selEnd = bodyEl.selectionEnd ?? 0;
		selection = editBody.slice(selStart, selEnd);
	}

	function clearSelection(): void {
		selStart = 0;
		selEnd = 0;
		selection = '';
	}

	// Replace the current selection in the body with the action result.
	async function insertIntoBody(text: string): Promise<void> {
		const start = selStart;
		const end = selEnd;
		editBody = editBody.slice(0, start) + text + editBody.slice(end);
		const caret = start + text.length;
		await tick();
		if (bodyEl) {
			bodyEl.focus();
			bodyEl.setSelectionRange(caret, caret);
		}
		selStart = caret;
		selEnd = caret;
		selection = '';
	}

	// Append the action result to the end of the body.
	function appendToBody(text: string): void {
		const sep = editBody.trim().length ? '\n\n' : '';
		editBody = editBody + sep + text;
	}

	// Delete-confirmation target.
	let deleteTarget: NoteRecord | null = null;
	let deleting = false;

	function loadEditorFrom(note: NoteRecord): void {
		loadedId = note.id;
		editTitle = note.title;
		editBody = decodeNoteBody(note.body_crdt_b64);
		editTags = note.tags.join(', ');
		editPinned = note.pinned;
		clearSelection();
	}

	// Load the editor when a different note becomes active; clear it when none is.
	$: if ($activeNote && $activeNote.id !== loadedId) {
		loadEditorFrom($activeNote);
	}
	$: if (!$activeNote && loadedId !== null) {
		loadedId = null;
		editTitle = '';
		editBody = '';
		editTags = '';
		editPinned = false;
		clearSelection();
	}

	function parseTags(raw: string): string[] {
		const seen = new Set<string>();
		const out: string[] = [];
		for (const part of raw.split(',')) {
			const t = part.trim();
			if (t && !seen.has(t)) {
				seen.add(t);
				out.push(t);
			}
		}
		return out;
	}

	$: dirty =
		$activeNote !== null &&
		(editTitle !== $activeNote.title ||
			editBody !== decodeNoteBody($activeNote.body_crdt_b64) ||
			editTags !== $activeNote.tags.join(', ') ||
			editPinned !== $activeNote.pinned);

	async function newNote(): Promise<void> {
		try {
			await createNote({ title: 'Untitled note', body_crdt_b64: '', tags: [], pinned: false });
		} catch {
			toastError('Failed to create note');
		}
	}

	async function save(): Promise<void> {
		if (!$activeNote || saving) return;
		const title = editTitle.trim();
		if (!title) {
			toastError('Note title cannot be empty');
			return;
		}
		saving = true;
		try {
			await saveNote($activeNote.id, {
				title,
				body_crdt_b64: encodeNoteBody(editBody),
				tags: parseTags(editTags),
				pinned: editPinned
			});
			loadedId = $activeNote.id;
			toastSuccess('Note saved');
		} catch {
			toastError('Failed to save note');
		} finally {
			saving = false;
		}
	}

	// N.8: the collaborative-edit confirmed-posture seam
	// (NOTES_CRDT_SPEC.md section 5). An incremental Yjs update renders in the
	// editor ONLY after the local backend acknowledges its append -- there is
	// no optimistic ghost state the store has not seen. When the append cannot
	// reach the backend the update is held in an explicit offline queue,
	// surfaced below the editor and replayed on reconnect; a refusal (a dead or
	// unknown parent, a failed gate -- section 5) surfaces a toast while the
	// display is kept at server truth, never advanced to a state the backend
	// rejected. The eventual remote echo is a no-op by Yjs idempotence. The
	// Y.Doc that produces the opaque update blobs is the host-assured half (the
	// live editor walk in NOTES_EDITOR_E2E_S265.md); this seam is the wiring it
	// drives.
	let editorBusy = false;
	let offlineQueue: string[] = [];

	function isOfflineError(err: unknown): boolean {
		// A lost connection (offline-queueable) vs a backend refusal (kept at
		// server truth): a fetch network failure surfaces as a TypeError, and
		// an explicitly offline navigator is offline too.
		if (typeof navigator !== 'undefined' && navigator.onLine === false) {
			return true;
		}
		return err instanceof TypeError;
	}

	async function commitEditorUpdate(blobB64: string): Promise<boolean> {
		if (!$activeNote || editorBusy) return false;
		editorBusy = true;
		try {
			// Render follows ack: await the backend, THEN treat the update as
			// applied -- nothing renders that the backend has not recorded.
			await appendNoteUpdate($activeNote.id, blobB64);
			return true;
		} catch (err) {
			if (isOfflineError(err)) {
				offlineQueue = [...offlineQueue, blobB64];
				toastError('Edit queued offline; it will sync on reconnect');
			} else {
				toastError('Edit refused; the display is kept at server truth');
			}
			return false;
		} finally {
			editorBusy = false;
		}
	}

	async function flushOfflineQueue(): Promise<void> {
		if (!$activeNote || editorBusy || offlineQueue.length === 0) return;
		const pending = offlineQueue;
		offlineQueue = [];
		for (const blobB64 of pending) {
			const ok = await commitEditorUpdate(blobB64);
			if (!ok) break;
		}
	}

	function askDelete(note: NoteRecord): void {
		deleteTarget = note;
	}

	function cancelDelete(): void {
		deleteTarget = null;
	}

	async function confirmDelete(): Promise<void> {
		if (!deleteTarget || deleting) return;
		deleting = true;
		try {
			await removeNote(deleteTarget.id);
			toastSuccess('Note deleted');
			deleteTarget = null;
		} catch {
			toastError('Failed to delete note');
		} finally {
			deleting = false;
		}
	}

	onMount(loadNotes);
</script>

<section class="notes-panel">
	<div class="notes-layout">
		<aside class="notes-list-pane">
			<header class="notes-header">
				<div class="notes-title">
					<Icon name="file-text" size="md" />
					<h2>Notes</h2>
				</div>
				<Button variant="primary" size="sm" iconLeft="plus" on:click={newNote}>New</Button>
			</header>

			<Input
				label="Search notes"
				hideLabel
				iconLeft="search"
				placeholder="Search notes..."
				bind:value={$search}
			/>

			{#if $error}
				<InlineError message={$error} onRetry={loadNotes} retrying={$loading} />
			{:else if $loading && $notes.length === 0}
				<p class="notes-status">Loading notes...</p>
			{:else if $filteredNotes.length === 0}
				<EmptyState
					icon="file-text"
					title={$search ? 'No matching notes' : 'No notes yet'}
					description={$search ? 'Try a different search.' : 'Create a note to get started.'}
				/>
			{:else}
				<ul class="notes-list">
					{#each $filteredNotes as note (note.id)}
						<li>
							<Card
								variant={$activeNote && $activeNote.id === note.id ? 'raised' : 'flat'}
								padding="sm"
							>
								<button
									type="button"
									class="note-item"
									class:note-item-active={$activeNote && $activeNote.id === note.id}
									on:click={() => selectNote(note.id)}
								>
									<span class="note-item-head">
										{#if note.pinned}
											<Icon name="pin" size="sm" />
										{/if}
										{#if note.mobile_allowed}
											<Icon name="smartphone" size="sm" />
										{/if}
										<span class="note-item-title">{note.title || 'Untitled note'}</span>
									</span>
									{#if note.tags.length > 0}
										<span class="note-item-tags">
											{#each note.tags as tag}
												<span class="note-tag">{tag}</span>
											{/each}
										</span>
									{/if}
								</button>
							</Card>
						</li>
					{/each}
				</ul>
			{/if}
		</aside>

		<section class="notes-editor-pane">
			{#if $activeNote}
				<div class="notes-editor">
					<Input label="Title" required bind:value={editTitle} placeholder="Note title" />

					<div class="notes-body-field">
						<label class="notes-body-label" for="notes-body">Body</label>
						<textarea
							id="notes-body"
							class="notes-body"
							rows="14"
							bind:this={bodyEl}
							bind:value={editBody}
							on:select={updateSelection}
							on:keyup={updateSelection}
							on:mouseup={updateSelection}
							on:input={updateSelection}
							on:focus={updateSelection}
							placeholder="Write your note..."
						></textarea>
					</div>

					{#if offlineQueue.length > 0}
						<div class="notes-offline-queue" role="status">
							<span class="notes-offline-queue-label">
								{offlineQueue.length} edit(s) queued offline; the editor stays at server truth until they sync
							</span>
							<Button variant="ghost" disabled={editorBusy} on:click={flushOfflineQueue}>
								Retry sync
							</Button>
						</div>
					{/if}

					<NoteActionPanel
						selection={selection}
						onInsert={insertIntoBody}
						onAppend={appendToBody}
					/>

					<Input
						label="Tags"
						hint="Comma-separated"
						iconLeft="tag"
						bind:value={editTags}
						placeholder="idea, todo, recipe"
					/>

					<div class="notes-media">
						<NotesVoiceCapture noteId={$activeNote.id} />
						<NotesMediaGallery noteId={$activeNote.id} />
						<NotesDrawingCanvas noteId={$activeNote.id} />
					</div>

					<label class="notes-pin-toggle">
						<input type="checkbox" bind:checked={editPinned} />
						<Icon name="pin" size="sm" />
						<span>Pinned</span>
					</label>

					<label class="notes-mobile-toggle">
						<input
							type="checkbox"
							checked={$activeNote.mobile_allowed}
							disabled={mobileBusy}
							on:click|preventDefault={flipMobileAllowed}
						/>
						<Icon name="smartphone" size="sm" />
						<span>Allow on phone</span>
						<span class="notes-mobile-hint">
							{mobileBusy
								? 'Confirming...'
								: 'Served to phone-class devices only while enabled'}
						</span>
					</label>

					<div class="notes-editor-actions">
						<Button
							variant="primary"
							iconLeft="save"
							loading={saving}
							disabled={!dirty}
							on:click={save}
						>
							Save
						</Button>
						<Button
							variant="danger"
							iconLeft="trash-2"
							on:click={() => $activeNote && askDelete($activeNote)}
						>
							Delete
						</Button>
						{#if dirty}
							<span class="notes-dirty">Unsaved changes</span>
						{/if}
					</div>
				</div>
			{:else}
				<EmptyState
					icon="file-text"
					title="No note selected"
					description="Select a note from the list, or create a new one."
				/>
			{/if}
		</section>
	</div>
</section>

<Modal open={deleteTarget !== null} title="Delete note" size="md" onClose={cancelDelete}>
	<p class="notes-delete-text">
		Delete "{deleteTarget?.title || 'Untitled note'}"? This cannot be undone.
	</p>
	<svelte:fragment slot="footer">
		<Button variant="ghost" on:click={cancelDelete}>Cancel</Button>
		<Button variant="danger" loading={deleting} on:click={confirmDelete}>Delete</Button>
	</svelte:fragment>
</Modal>

<style>
	.notes-panel {
		height: 100%;
		min-height: 0;
		padding: var(--oo-space-4);
	}

	.notes-layout {
		display: grid;
		grid-template-columns: minmax(220px, 320px) 1fr;
		gap: var(--oo-space-4);
		height: 100%;
		min-height: 0;
	}

	.notes-list-pane {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
		min-height: 0;
		overflow: hidden;
	}

	.notes-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.notes-title {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-primary);
	}

	.notes-title h2 {
		margin: 0;
		font-size: var(--oo-text-lg);
		font-weight: 600;
	}

	.notes-status {
		margin: 0;
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
	}

	.notes-list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		overflow-y: auto;
		min-height: 0;
	}

	.note-item {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
		width: 100%;
		text-align: left;
		background: transparent;
		border: none;
		padding: 0;
		cursor: pointer;
		color: var(--oo-fg-secondary);
	}

	.note-item-active {
		color: var(--oo-fg-primary);
	}

	.note-item-head {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-tertiary);
	}

	.note-item-title {
		font-size: var(--oo-text-sm);
		font-weight: 600;
		color: var(--oo-fg-primary);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.note-item-tags {
		display: flex;
		flex-wrap: wrap;
		gap: var(--oo-space-1);
	}

	.note-tag {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-faint);
		background: var(--oo-bg-elevated);
		border-radius: var(--oo-radius-full);
		padding: 0 var(--oo-space-2);
	}

	.notes-editor-pane {
		min-height: 0;
		overflow-y: auto;
	}

	.notes-editor {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}

	.notes-body-field {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
	}

	.notes-offline-queue {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-2);
		padding: var(--oo-space-2) var(--oo-space-3);
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
	}

	.notes-offline-queue-label {
		color: var(--oo-fg-muted);
	}

	.notes-body-label {
		font-size: var(--oo-text-sm);
		font-weight: 500;
		color: var(--oo-fg-secondary);
	}

	.notes-body {
		width: 100%;
		resize: vertical;
		font: inherit;
		color: var(--oo-fg-primary);
		background: var(--oo-bg-base);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
		padding: var(--oo-space-3);
		line-height: 1.5;
	}

	.notes-body:focus {
		outline: none;
		border-color: var(--oo-acc-500);
	}

	.notes-media {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}

	.notes-pin-toggle {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
		cursor: pointer;
	}

	.notes-mobile-toggle {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
		cursor: pointer;
	}

	.notes-mobile-hint {
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-xs);
	}

	.notes-editor-actions {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
	}

	.notes-dirty {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
	}

	.notes-delete-text {
		margin: 0;
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
	}

	@media (max-width: 720px) {
		.notes-layout {
			grid-template-columns: 1fr;
		}
	}
</style>
