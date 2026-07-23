<!--
  ConversationItem.svelte
  Renders one conversation in the sidebar. Click to select, double-click
  to rename, hover for export / rename / delete actions. Uses the ds
  Tooltip (full title) and Icon primitives.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { ConversationSummary } from '$lib/types';
	import Tooltip from '$lib/ds/Tooltip.svelte';
	import Icon from '$lib/ds/Icon.svelte';

	export let conversation: ConversationSummary;
	export let isActive: boolean = false;

	const dispatch = createEventDispatcher<{
		select: { id: string };
		rename: { id: string; title: string };
		delete: { id: string };
		export: { id: string; title: string };
	}>();

	let editing = false;
	let editTitle = '';
	let confirmDelete = false;
	let inputEl: HTMLInputElement;

	function startEdit() {
		editing = true;
		editTitle = conversation.title;
		// Focus after render.
		setTimeout(() => inputEl?.focus(), 10);
	}

	function confirmEdit() {
		const trimmed = editTitle.trim();
		if (trimmed && trimmed !== conversation.title) {
			dispatch('rename', { id: conversation.id, title: trimmed });
		}
		editing = false;
	}

	function cancelEdit() {
		editing = false;
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') confirmEdit();
		if (e.key === 'Escape') cancelEdit();
	}

	function handleDelete() {
		if (confirmDelete) {
			dispatch('delete', { id: conversation.id });
			confirmDelete = false;
		} else {
			confirmDelete = true;
			// Reset after 3s.
			setTimeout(() => {
				confirmDelete = false;
			}, 3000);
		}
	}

	function formatDate(dateStr: string | null): string {
		if (!dateStr) return '';
		try {
			const d = new Date(dateStr);
			const now = new Date();
			const diff = now.getTime() - d.getTime();
			const mins = Math.floor(diff / 60000);
			if (mins < 1) return 'now';
			if (mins < 60) return `${mins}m`;
			const hours = Math.floor(mins / 60);
			if (hours < 24) return `${hours}h`;
			const days = Math.floor(hours / 24);
			if (days < 7) return `${days}d`;
			return d.toLocaleDateString('en', { month: 'short', day: 'numeric' });
		} catch {
			return '';
		}
	}
</script>

<div
	class="group relative flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer
		transition-colors duration-100"
	style="{isActive
		? 'background-color: var(--oo-msg-user-bg); color: var(--oo-fg-primary);'
		: 'color: var(--oo-fg-tertiary);'}"
	role="button"
	tabindex="0"
	on:click={() => dispatch('select', { id: conversation.id })}
	on:dblclick|stopPropagation={startEdit}
	on:keydown={(e) => e.key === 'Enter' && dispatch('select', { id: conversation.id })}
>
	{#if editing}
		<input
			bind:this={inputEl}
			bind:value={editTitle}
			on:blur={confirmEdit}
			on:keydown={handleKeydown}
			class="flex-1 text-sm px-2 py-1 rounded outline-none"
			style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary);
				border: 1px solid var(--oo-acc-500);"
			on:click|stopPropagation
		/>
	{:else}
		<div class="flex-1 min-w-0">
			<Tooltip content={conversation.title} placement="right">
				<div class="text-sm truncate w-full">{conversation.title}</div>
			</Tooltip>
			<div class="text-xs flex items-center gap-2 mt-0.5" style="color: var(--oo-fg-muted);">
				{#if conversation.message_count > 0}
					<span>{conversation.message_count} msgs</span>
				{/if}
				{#if conversation.updated_at}
					<span>{formatDate(conversation.updated_at)}</span>
				{/if}
			</div>
		</div>

		<!-- Actions revealed on hover -->
		<div class="hidden group-hover:flex items-center gap-1 shrink-0">
			<button
				on:click|stopPropagation={() =>
					dispatch('export', { id: conversation.id, title: conversation.title })}
				class="p-1 rounded transition-colors"
				style="color: var(--oo-fg-muted);"
				title="Export"
				aria-label="Export conversation"
			>
				<Icon name="download" size="sm" />
			</button>
			<button
				on:click|stopPropagation={startEdit}
				class="p-1 rounded transition-colors"
				style="color: var(--oo-fg-muted);"
				title="Rename"
				aria-label="Rename conversation"
			>
				<Icon name="pencil" size="sm" />
			</button>
			<button
				on:click|stopPropagation={handleDelete}
				class="p-1 rounded transition-colors"
				style="{confirmDelete
					? 'color: var(--oo-status-error); background-color: var(--oo-error-bg);'
					: 'color: var(--oo-fg-muted);'}"
				title={confirmDelete ? 'Click again to confirm' : 'Delete'}
				aria-label={confirmDelete ? 'Confirm delete conversation' : 'Delete conversation'}
			>
				<Icon name="trash-2" size="sm" />
			</button>
		</div>
	{/if}
</div>
