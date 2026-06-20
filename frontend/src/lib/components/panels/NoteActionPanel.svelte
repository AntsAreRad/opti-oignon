<!--
  NoteActionPanel.svelte (S248, Notes feature N.3 selection-action UI)
  The selection-action surface for the Notes editor. Given a text selection from
  the note body, it runs one local action -- fact-check, develop, summarize,
  rewrite, make-checklist -- or the Daily-only fact-check-with-web, over
  POST /api/notes/actions/run (the runNoteAction client). The selection is
  wrapped as untrusted context by the backend (note_actions); this panel never
  interprets it. The structured result is shown alongside: ok carries the model
  text, which the user can insert (replacing the selection) or append (to the
  note body); a web action outside Daily returns a structured refusal, shown as a
  notice, never a silent local downgrade; any other failure shows its reason. The
  model is the user's selected/effective model. Design-system tokens only
  (--oo-*); lucide-svelte icons through Icon.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { get } from 'svelte/store';
	import { Button, Card, Icon, InlineError } from '$lib/ds';
	import {
		NOTE_ACTIONS,
		runNoteAction,
		type NoteActionKind,
		type NoteActionResult
	} from '$lib/api/noteActions';
	import { selectedModel, effectiveModel, loadOptions } from '$lib/stores/chatOptions';

	/** The currently selected text from the note body. */
	export let selection = '';
	/** Replace the current selection in the note body with the given text. */
	export let onInsert: (text: string) => void = () => {};
	/** Append the given text to the note body. */
	export let onAppend: (text: string) => void = () => {};

	let runningKind: NoteActionKind | null = null;
	let result: NoteActionResult | null = null;
	let localError: string | null = null;
	let lastSelection = '';

	// The model to run with: a forced selection wins, else the effective model.
	$: model = $selectedModel || $effectiveModel || '';
	$: hasSelection = selection.trim().length > 0;

	// Drop a stale result when the selection actually changes.
	$: if (selection !== lastSelection) {
		lastSelection = selection;
		result = null;
		localError = null;
	}

	onMount(() => {
		// Resolve the effective model the way the chat surface does, so the notes
		// page has a model to run with even before chat is opened.
		if (!get(effectiveModel)) {
			loadOptions();
		}
	});

	async function run(kind: NoteActionKind): Promise<void> {
		if (!hasSelection || runningKind) return;
		runningKind = kind;
		result = null;
		localError = null;
		try {
			result = await runNoteAction({ action: kind, selection, model });
		} catch (err: unknown) {
			localError = err instanceof Error ? err.message : 'Action failed';
		} finally {
			runningKind = null;
		}
	}

	function insert(): void {
		if (result?.ok && result.text) onInsert(result.text);
	}

	function append(): void {
		if (result?.ok && result.text) onAppend(result.text);
	}
</script>

<div class="note-actions">
	<div class="note-actions-head">
		<Icon name="lightbulb" size="sm" />
		<span class="note-actions-title">Selection actions</span>
	</div>

	{#if !hasSelection}
		<p class="note-actions-hint">
			Select text in the note to fact-check, develop, summarize, rewrite, or make a checklist.
		</p>
	{:else}
		<div class="note-actions-buttons">
			{#each NOTE_ACTIONS as action (action.kind)}
				<Button
					variant="secondary"
					size="sm"
					loading={runningKind === action.kind}
					disabled={runningKind !== null}
					on:click={() => run(action.kind)}
				>
					{action.label}
				</Button>
			{/each}
		</div>
	{/if}

	{#if localError}
		<InlineError message={localError} />
	{:else if result}
		{#if result.refused}
			<Card variant="flat" padding="sm">
				<div class="note-actions-notice">
					<Icon name="shield-alert" size="sm" />
					<span>{result.reason || 'This action needs Daily mode and was refused.'}</span>
				</div>
			</Card>
		{:else if result.ok}
			<Card variant="flat" padding="sm">
				<div class="note-actions-result">
					<pre class="note-actions-text">{result.text}</pre>
					<div class="note-actions-result-buttons">
						<Button variant="primary" size="sm" iconLeft="replace" on:click={insert}>
							Insert
						</Button>
						<Button variant="secondary" size="sm" iconLeft="plus" on:click={append}>
							Append
						</Button>
					</div>
				</div>
			</Card>
		{:else}
			<Card variant="flat" padding="sm">
				<div class="note-actions-notice">
					<Icon name="alert-triangle" size="sm" />
					<span>{result.reason || 'The action could not be completed.'}</span>
				</div>
			</Card>
		{/if}
	{/if}
</div>

<style>
	.note-actions {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		border-top: 1px solid var(--oo-bd-subtle);
		padding-top: var(--oo-space-3);
	}

	.note-actions-head {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-tertiary);
	}

	.note-actions-title {
		font-size: var(--oo-text-sm);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.note-actions-hint {
		margin: 0;
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
	}

	.note-actions-buttons {
		display: flex;
		flex-wrap: wrap;
		gap: var(--oo-space-2);
	}

	.note-actions-result {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}

	.note-actions-text {
		margin: 0;
		white-space: pre-wrap;
		word-break: break-word;
		font: inherit;
		color: var(--oo-fg-primary);
		max-height: 18rem;
		overflow-y: auto;
	}

	.note-actions-result-buttons {
		display: flex;
		gap: var(--oo-space-2);
	}

	.note-actions-notice {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
	}
</style>
