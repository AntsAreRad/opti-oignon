<!--
  YourDataPanel.svelte -- the user's own data: export it first, see what a wipe removes and leaves, then wipe.

  1. The export, offered before anything else: downloaded, or skipped by a deliberate choice.
  2. The preview: what the wipe removes and what stays. In single-user mode the notes and the projects can be
     named for removal; the preview then counts them and gives the phrase to type, and says what the paired devices
     and the export do with them.
  3. The typed confirmations: the wipe's own words, and the preview's phrase when content is named. The server
     checks the phrase again against what it counts at that moment, and refuses when the counts moved.
  4. The answer, failures first.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import Button from '$lib/ds/Button.svelte';
	import Checkbox from '$lib/ds/Checkbox.svelte';
	import Input from '$lib/ds/Input.svelte';
	import InlineError from '$lib/ds/InlineError.svelte';
	import { currentUser } from '$lib/stores/auth';
	import { parseApiError } from '$lib/api/errorHandler';
	import { downloadUserData, previewWipe, wipeUserData, type WipePreview, type WipeResult } from '$lib/api/userData';
	import { WIPE_WORDS, answerLines, canWipe, counted, type WipeAnswerLines } from '$lib/settings/yourData';

	$: userId = $currentUser?.user_id ?? 'local';

	let exportOffered = false;
	let exporting = false;
	let exportError = '';

	let removeNotes = false;
	let removeProjects = false;
	let preview: WipePreview | null = null;
	let previewing = false;
	let previewError = '';
	// Each preview asked is numbered: only the last one asked is shown, whichever answers first.
	let previewAsked = 0;

	let typedWipe = '';
	let typedPhrase = '';
	let wiping = false;
	let wipeError = '';
	let result: WipeResult | null = null;
	let lines: WipeAnswerLines | null = null;

	$: named = wordsNamed(removeNotes, removeProjects);
	$: notesCounted = preview?.counts?.notes;
	$: projectsCounted = preview?.counts?.projects;
	$: allowed = canWipe({
		exportOffered,
		typedWipe,
		named,
		phrase: preview?.phrase,
		typedPhrase,
		busy: previewing || wiping
	});

	onMount(loadPreview);

	function wordsNamed(notes: boolean, projects: boolean): string[] {
		return [...(notes ? ['notes'] : []), ...(projects ? ['projects'] : [])];
	}

	async function downloadFirst() {
		exporting = true;
		exportError = '';
		try {
			await downloadUserData(userId);
			exportOffered = true;
		} catch (err: unknown) {
			exportError = parseApiError(err, 'exporting your data').message;
		} finally {
			exporting = false;
		}
	}

	async function loadPreview() {
		const asked = ++previewAsked;
		previewing = true;
		previewError = '';
		typedPhrase = '';
		try {
			const answer = await previewWipe(userId, wordsNamed(removeNotes, removeProjects));
			if (asked === previewAsked) preview = answer;
		} catch (err: unknown) {
			if (asked === previewAsked) previewError = parseApiError(err, 'reading what a wipe removes').message;
		} finally {
			if (asked === previewAsked) previewing = false;
		}
	}

	function toggleNotes(event: CustomEvent<boolean>) {
		removeNotes = event.detail;
		loadPreview();
	}

	function toggleProjects(event: CustomEvent<boolean>) {
		removeProjects = event.detail;
		loadPreview();
	}

	async function wipe() {
		if (!allowed || preview === null) return;
		wiping = true;
		wipeError = '';
		try {
			result = await wipeUserData(userId, named, named.length > 0 ? preview.phrase : undefined);
			const counts: Record<string, number> = {
				conversations: result.conversations,
				facts: result.memories,
				pending_writes: result.pending_writes,
				plugin_settings: result.plugin_configs,
				plugin_reviews: result.plugin_reviews,
				...result.content_removed
			};
			lines = answerLines(result.not_covered, preview.left, counts);
			typedWipe = '';
			typedPhrase = '';
		} catch (err: unknown) {
			const parsed = parseApiError(err, 'wiping your data');
			wipeError = parsed.status === 409 ? `${parsed.message} Preview again before wiping.` : parsed.message;
			if (parsed.status === 409) loadPreview();
		} finally {
			wiping = false;
		}
	}
</script>

<div class="oo-your-data">
	<section class="oo-your-data-step" aria-labelledby="oo-your-data-export">
		<h3 id="oo-your-data-export">1. Keep a copy first</h3>
		<p>
			The export holds your conversations, notes, projects, facts and settings as JSON; a project's files and
			outputs are downloaded from the project. A wipe cannot be undone.
		</p>
		<div class="oo-your-data-actions">
			<Button variant="primary" loading={exporting} on:click={downloadFirst}>Download my data</Button>
			<Button variant="link" disabled={exportOffered} on:click={() => (exportOffered = true)}>
				Continue without a copy
			</Button>
		</div>
		<InlineError message={exportError} />
	</section>

	<section class="oo-your-data-step" aria-labelledby="oo-your-data-preview">
		<h3 id="oo-your-data-preview">2. What a wipe removes</h3>
		{#if preview?.on_request?.notes}
			<Checkbox
				label="Also delete my notes"
				checked={removeNotes}
				disabled={previewing || wiping}
				description={notesCounted
					? `${counted(notesCounted.notes, 'note')} and ${counted(notesCounted.attachments, 'attachment')}` +
						(notesCounted.disputed === null
							? ', and the notes hidden here could not be counted'
							: notesCounted.disputed > 0
								? `, and ${notesCounted.disputed} hidden here that a paired device still shows`
								: '')
					: 'Your notes, their attachments and their history'}
				on:change={toggleNotes}
			/>
		{/if}
		{#if preview?.on_request?.projects}
			<Checkbox
				label="Also delete my projects"
				checked={removeProjects}
				disabled={previewing || wiping}
				description={projectsCounted
					? `${counted(projectsCounted.projects, 'project')}, ${counted(projectsCounted.files, 'file')} and ` +
						counted(projectsCounted.outputs, 'output')
					: 'Your projects, their files, outputs and index'}
				on:change={toggleProjects}
			/>
		{/if}
		{#if named.length > 0 && preview?.notice}
			<p class="oo-your-data-notice">{preview.notice}</p>
		{/if}
		<InlineError message={previewError} onRetry={loadPreview} retrying={previewing} />
		{#if preview}
			<details>
				<summary>Removed ({preview.removed.length})</summary>
				<ul>{#each preview.removed as label}<li>{label}</li>{/each}</ul>
			</details>
			<details>
				<summary>Left in place ({preview.left.length + preview.retained_by_design.length})</summary>
				<ul>
					{#each preview.left as label}<li>{label}</li>{/each}
					{#each preview.retained_by_design as label}<li>{label} (kept by design)</li>{/each}
				</ul>
			</details>
		{/if}
	</section>

	<section class="oo-your-data-step" aria-labelledby="oo-your-data-confirm">
		<h3 id="oo-your-data-confirm">3. Confirm</h3>
		{#if !exportOffered}
			<p>Download your data, or choose to continue without a copy, before wiping.</p>
		{/if}
		<Input label={`Type "${WIPE_WORDS}" to confirm`} bind:value={typedWipe} disabled={!exportOffered || wiping} />
		{#if named.length > 0 && preview?.phrase}
			<Input
				label={`Type "${preview.phrase}" to delete your ${named.join(' and ')}`}
				bind:value={typedPhrase}
				disabled={!exportOffered || wiping}
			/>
		{/if}
		<div class="oo-your-data-actions">
			<Button variant="danger" disabled={!allowed} loading={wiping} on:click={wipe}>Delete my data</Button>
		</div>
		<InlineError message={wipeError} />
	</section>

	{#if lines}
		<section class="oo-your-data-step" aria-labelledby="oo-your-data-answer" aria-live="polite">
			<h3 id="oo-your-data-answer">What the wipe did</h3>
			{#if lines.failed.length > 0}
				<p class="oo-your-data-failed">Not removed, or not overwritten:</p>
				<ul>{#each lines.failed as line}<li>{line}</li>{/each}</ul>
			{/if}
			<ul>{#each lines.removed as line}<li>{line}</li>{/each}</ul>
			<details>
				<summary>Left in place ({lines.left.length})</summary>
				<ul>{#each lines.left as line}<li>{line}</li>{/each}</ul>
			</details>
		</section>
	{/if}
</div>

<style>
	.oo-your-data {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
	}

	.oo-your-data-step {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}

	.oo-your-data-actions {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
	}

	.oo-your-data-notice,
	.oo-your-data-failed {
		font-weight: 600;
	}
</style>
