<!--
  MemoryPanel.svelte
  Side panel for persistent memory management.
  Liste les faits, ajout, suppression, extraction depuis la conversation active.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { activeConversationId } from '$lib/stores/conversations';
	import { closePanel } from '$lib/stores/panels';
	import { toastError, toastSuccess } from '$lib/stores/notifications';
	import {
		listFacts,
		createFact,
		deleteFact,
		clearAllFacts,
		extractFacts,
		listTrash,
		restoreFact,
		eraseFact,
		emptyTrash,
	} from '$lib/api/memory';
	import type { MemoryFact, MemoryTrashedFact } from '$lib/types';
	import { Button, ConfirmDialog, TextButton } from '$lib/ds';
	import StopAllButton from '$lib/components/layout/StopAllButton.svelte';
	import { isPhone } from '$lib/stores/ui';
	import PendingWritesReview from './PendingWritesReview.svelte';

	let facts: MemoryFact[] = [];
	// The trash: a deleted fact waits there, restorable, until it is erased for good after trashDays.
	let trash: MemoryTrashedFact[] = [];
	let trashDays = 30;
	let showTrash = false;
	// What the erase dialog asks about: one fact, or the whole trash (null: the dialog is closed).
	let erasing: MemoryTrashedFact | 'all' | null = null;
	let erasingBusy = false;
	// The review section: the writes waiting for the user, reread after an extraction.
	let review: PendingWritesReview | undefined;
	let loading = false;
	let newFactText = '';
	let newFactCategory = 'context';
	let adding = false;
	let extracting = false;
	let clearConfirm = false;
	let deleteConfirmId: string | null = null;

	const CATEGORIES = ['context', 'preference', 'technical', 'project', 'personal'];

	async function loadFacts() {
		loading = true;
		try {
			facts = await listFacts();
		} catch {
			facts = [];
		} finally {
			loading = false;
		}
	}

	async function handleAddFact() {
		if (!newFactText.trim() || adding) return;

		adding = true;
		try {
			const fact = await createFact({
				fact: newFactText.trim(),
				category: newFactCategory,
				source_conversation_id: $activeConversationId ?? '',
			});
			facts = [fact, ...facts];
			newFactText = '';
			toastSuccess('Fact added');
		} catch {
			toastError('Failed to add fact');
		} finally {
			adding = false;
		}
	}

	async function loadTrash() {
		try {
			const result = await listTrash();
			trash = result.facts;
			trashDays = result.days;
		} catch {
			trash = [];
		}
	}

	async function handleDelete(factId: string) {
		try {
			await deleteFact(factId);
			facts = facts.filter(f => f.id !== factId);
			deleteConfirmId = null;
			toastSuccess(`Moved to the trash: erased for good in ${trashDays} days`);
			await loadTrash();
		} catch {
			toastError('Failed to delete fact');
		}
	}

	async function handleClearAll() {
		try {
			const result = await clearAllFacts();
			facts = [];
			clearConfirm = false;
			toastSuccess(`Moved ${result.count} facts to the trash: erased for good in ${trashDays} days`);
			await loadTrash();
		} catch {
			toastError('Failed to clear facts');
		}
	}

	async function handleRestore(factId: string) {
		try {
			await restoreFact(factId);
			trash = trash.filter(f => f.id !== factId);
			toastSuccess('Fact restored');
			await loadFacts();
		} catch {
			toastError('Failed to restore fact');
		}
	}

	async function handleErase() {
		const target = erasing;
		if (target === null) return;
		erasingBusy = true;
		try {
			if (target === 'all') {
				const result = await emptyTrash();
				trash = [];
				toastSuccess(`Erased ${result.erased} facts for good`);
			} else {
				await eraseFact(target.id);
				trash = trash.filter(f => f.id !== target.id);
				toastSuccess('Fact erased for good');
			}
			erasing = null;
		} catch {
			toastError(target === 'all' ? 'Failed to empty the trash' : 'Failed to erase fact');
		} finally {
			erasingBusy = false;
		}
	}

	function eraseLabel(dateStr: string): string {
		if (!dateStr) return 'erased soon';
		return `erased on ${new Date(dateStr).toLocaleDateString()}`;
	}

	async function handleExtract() {
		if (!$activeConversationId || extracting) return;

		extracting = true;
		try {
			const result = await extractFacts($activeConversationId);
			const proposed = result.facts_proposed ?? 0;
			toastSuccess(
				proposed > 0
					? `Extracted ${result.facts_added} facts; ${proposed} to review`
					: `Extracted ${result.facts_added} facts`,
			);
			await loadFacts();
			await review?.reload();
		} catch {
			toastError('Failed to extract facts');
		} finally {
			extracting = false;
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleAddFact();
		}
	}

	function categoryColor(cat: string): string {
		const colors: Record<string, string> = {
			context: 'bg-[var(--oo-info)]/15 text-[var(--oo-info)]',
			preference: 'bg-[var(--oo-cat-purple)]/15 text-[var(--oo-cat-purple)]',
			technical: 'bg-[var(--oo-success)]/15 text-[var(--oo-success)]',
			project: 'bg-[var(--oo-cat-orange)]/15 text-[var(--oo-cat-orange)]',
			personal: 'bg-[var(--oo-cat-pink)]/15 text-[var(--oo-cat-pink)]',
		};
		return colors[cat] || 'bg-surface-700 text-surface-400';
	}

	function timeAgo(dateStr: string): string {
		if (!dateStr) return '';
		const diff = Date.now() - new Date(dateStr).getTime();
		const mins = Math.floor(diff / 60000);
		if (mins < 1) return 'just now';
		if (mins < 60) return `${mins}m ago`;
		const hours = Math.floor(mins / 60);
		if (hours < 24) return `${hours}h ago`;
		const days = Math.floor(hours / 24);
		return `${days}d ago`;
	}

	onMount(() => {
		loadFacts();
		loadTrash();
	});
</script>

<div class="h-full flex flex-col bg-surface-900">
	<!-- Header -->
	<div class="flex items-center justify-between px-3 py-2 shrink-0" style="border-bottom: 1px solid var(--oo-bd-subtle);">
		<div class="flex items-center gap-2">
			<svg class="w-4 h-4 text-accent-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
			</svg>
			<h2 class="text-sm font-medium text-surface-200">Memory</h2>
			<span class="text-xs text-surface-500 tabular-nums">{facts.length}</span>
		</div>
		<button
			on:click={closePanel}
			class="p-1 rounded text-surface-500 hover:text-surface-300 hover:bg-surface-800 transition-colors"
			title="Close panel"
		aria-label="Close memory panel"
		>
			<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M6 18L18 6M6 6l12 12" />
			</svg>
		</button>
	</div>

	<!-- Add fact form -->
	<div class="px-3 py-2 border-b border-surface-800/50 shrink-0">
		<div class="flex gap-1.5 mb-1.5">
			<input
				type="text"
				bind:value={newFactText}
				on:keydown={handleKeydown}
				placeholder="Add a fact..."
				disabled={adding}
				class="flex-1 min-w-0 bg-surface-800 text-surface-200 text-xs px-2.5 py-1.5
					rounded-md border border-surface-700 outline-none
					focus:border-accent-600/50 placeholder:text-surface-600
					disabled:opacity-50"
			/>
			<button
				on:click={handleAddFact}
				disabled={!newFactText.trim() || adding}
				class="px-2 py-1.5 rounded-md text-xs font-medium bg-accent-600/20 text-accent-400
					hover:bg-accent-600/30 transition-colors
					disabled:opacity-30 disabled:cursor-not-allowed"
			>
				{adding ? '...' : 'Add'}
			</button>
		</div>
		<div class="flex items-center gap-1.5">
			<select
				bind:value={newFactCategory}
				class="bg-surface-800 text-surface-400 text-[10px] px-1.5 py-0.5
					rounded border border-surface-700 outline-none"
			>
				{#each CATEGORIES as cat}
					<option value={cat}>{cat}</option>
				{/each}
			</select>

			{#if $activeConversationId}
				<button
					on:click={handleExtract}
					disabled={extracting}
					class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px]
						text-surface-400 hover:text-surface-200 hover:bg-surface-800 transition-colors
						disabled:opacity-50"
					title="Extract facts from current conversation"
				>
					{#if extracting}
						<svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
							<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
							<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
						</svg>
					{:else}
						<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
							<path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
						</svg>
					{/if}
					Extract
				</button>
			{/if}

			{#if facts.length > 0}
				<div class="ml-auto">
					{#if clearConfirm}
						<span class="inline-flex items-center gap-1 text-[10px]">
							<span class="text-[var(--oo-error)]">Move all to the trash?</span>
							<button
								on:click={handleClearAll}
								class="text-[var(--oo-error)] hover:text-[var(--oo-error)] font-medium"
							>Yes</button>
							<button
								on:click={() => (clearConfirm = false)}
								class="text-surface-400 hover:text-surface-200"
							>No</button>
						</span>
					{:else}
						<button
							on:click={() => (clearConfirm = true)}
							class="text-[10px] text-surface-500 hover:text-[var(--oo-error)] transition-colors"
						>Clear all</button>
					{/if}
				</div>
			{/if}
		</div>
	</div>

	<!-- Facts list, after the writes waiting for review -->
	<div class="flex-1 overflow-y-auto">
		<PendingWritesReview bind:this={review} on:decided={loadFacts} />
		{#if loading}
			<div class="flex items-center justify-center py-8">
				<svg class="w-5 h-5 animate-spin text-surface-500" fill="none" viewBox="0 0 24 24">
					<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
					<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
				</svg>
			</div>
		{:else if facts.length === 0}
			<div class="text-center py-8 px-4">
				<p class="text-xs text-surface-500">No memory facts yet.</p>
				<p class="text-[10px] text-surface-600 mt-1">
					Add facts manually or extract them from a conversation.
				</p>
			</div>
		{:else}
			<div class="divide-y divide-surface-800/50">
				{#each facts as fact (fact.id)}
					<div class="px-3 py-2 hover:bg-surface-800/30 transition-colors group">
						<div class="flex items-start justify-between gap-2">
							<p class="text-xs text-surface-300 leading-relaxed flex-1 min-w-0">
								{fact.fact}
							</p>
							<!-- Delete button -->
							{#if deleteConfirmId === fact.id}
								<div class="flex items-center gap-1 shrink-0">
									<button
										on:click={() => handleDelete(fact.id)}
										class="text-[10px] text-[var(--oo-error)] hover:text-[var(--oo-error)] font-medium"
									>Trash</button>
									<button
										on:click={() => (deleteConfirmId = null)}
										class="text-[10px] text-surface-500 hover:text-surface-300"
									>No</button>
								</div>
							{:else}
								<button
									on:click={() => (deleteConfirmId = fact.id)}
									class="p-0.5 rounded text-surface-600 hover:text-[var(--oo-error)] transition-colors
										opacity-0 group-hover:opacity-100"
									title="Move to the trash"
								>
									<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
										<path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
									</svg>
								</button>
							{/if}
						</div>
						<div class="flex items-center gap-2 mt-1">
							<span class="inline-block px-1.5 py-0 rounded text-[10px] {categoryColor(fact.category)}">
								{fact.category}
							</span>
							{#if fact.confidence < 1.0}
								<span class="text-[10px] text-surface-600">
									{(fact.confidence * 100).toFixed(0)}%
								</span>
							{/if}
							{#if fact.created_at}
								<span class="text-[10px] text-surface-600">{timeAgo(fact.created_at)}</span>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		{/if}

		<!-- The trash: deleted facts, restorable until they are erased for good -->
		{#if trash.length > 0}
			<div class="mt-2 px-3 pb-3">
				<div class="flex items-center justify-between gap-2">
					<TextButton expanded={showTrash} controls="memory-trash" on:click={() => (showTrash = !showTrash)}>
						{showTrash ? 'Hide' : 'Show'} the trash ({trash.length})
					</TextButton>
					{#if showTrash}
						<Button variant="ghost" size="sm" on:click={() => (erasing = 'all')}>Empty the trash</Button>
					{/if}
				</div>
				{#if showTrash}
					<div id="memory-trash" class="mt-1 space-y-2">
						<p class="text-xs text-surface-500">
							A deleted fact is erased for good after {trashDays} days.
						</p>
						{#each trash as fact (fact.id)}
							<div class="space-y-1">
								<p class="text-xs text-surface-500 leading-relaxed">{fact.fact}</p>
								<div class="flex items-center gap-2">
									<span class="text-xs text-surface-600">{eraseLabel(fact.erase_at)}</span>
									<span class="ml-auto inline-flex gap-1">
										<Button variant="ghost" size="sm" on:click={() => handleRestore(fact.id)}>Restore</Button>
										<Button variant="danger" size="sm" on:click={() => (erasing = fact)}>Erase</Button>
									</span>
								</div>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		{/if}
	</div>
</div>

<ConfirmDialog
	open={erasing !== null}
	title={erasing === 'all' ? 'Empty the trash?' : 'Erase this fact for good?'}
	message={erasing === 'all'
		? `The ${trash.length} facts in the trash are erased for good: they cannot be restored.`
		: 'It is erased for good: it cannot be restored.'}
	confirmLabel="Erase"
	cancelLabel="Keep"
	danger
	busy={erasingBusy}
	onConfirm={handleErase}
	onCancel={() => (erasing = null)}
>
	<svelte:fragment slot="actions">
		<StopAllButton placement="dialog-head" large={$isPhone} />
	</svelte:fragment>
</ConfirmDialog>
