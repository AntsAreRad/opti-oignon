<!--
  ConversationList.svelte (refactored S167)
  Conversation list with search, date grouping (Today / Yesterday /
  Previous 7 days / Older), empty-state guidance and robust delete
  handling. Uses the ds Icon primitive for the search affordance.
  S87: empty-state guidance card, redirect to /chat after deleting the
  last conversation.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		conversations,
		activeConversationId,
		loading,
		error,
		loadConversations,
		renameConv,
		deleteConv
	} from '$lib/stores/conversations';
	import type { ConversationSummary } from '$lib/types';
	import Icon from '$lib/ds/Icon.svelte';
	import ConversationItem from './ConversationItem.svelte';
	import ConversationSkeleton from './ConversationSkeleton.svelte';

	export let onSelect: (id: string) => void = () => {};
	export let onExport: (id: string, title: string) => void = () => {};
	export let onNewConversation: () => void = () => {};

	let searchQuery = '';

	$: filteredConversations = searchQuery.trim()
		? $conversations.filter((c) => c.title.toLowerCase().includes(searchQuery.toLowerCase()))
		: $conversations;

	// Group conversations into date buckets, preserving store order within each.
	type Group = { key: string; label: string; items: ConversationSummary[] };

	function bucketLabel(dateStr: string | null): { key: string; label: string } {
		if (!dateStr) return { key: 'older', label: 'Older' };
		const d = new Date(dateStr).getTime();
		if (Number.isNaN(d)) return { key: 'older', label: 'Older' };
		const startOfToday = new Date();
		startOfToday.setHours(0, 0, 0, 0);
		const dayMs = 86400000;
		const todayStart = startOfToday.getTime();
		if (d >= todayStart) return { key: 'today', label: 'Today' };
		if (d >= todayStart - dayMs) return { key: 'yesterday', label: 'Yesterday' };
		if (d >= todayStart - 7 * dayMs) return { key: 'week', label: 'Previous 7 days' };
		return { key: 'older', label: 'Older' };
	}

	const GROUP_ORDER = ['today', 'yesterday', 'week', 'older'];

	$: groups = (() => {
		const map = new Map<string, Group>();
		for (const c of filteredConversations) {
			const { key, label } = bucketLabel(c.updated_at ?? c.created_at);
			if (!map.has(key)) map.set(key, { key, label, items: [] });
			map.get(key)!.items.push(c);
		}
		return GROUP_ORDER.filter((k) => map.has(k)).map((k) => map.get(k)!);
	})();

	onMount(() => {
		loadConversations();
	});

	async function handleRename(e: CustomEvent<{ id: string; title: string }>) {
		await renameConv(e.detail.id, e.detail.title);
	}

	async function handleDelete(e: CustomEvent<{ id: string }>) {
		const wasActive = $activeConversationId === e.detail.id;
		await deleteConv(e.detail.id);
		// S87: After deleting, redirect if it was the active conversation
		// or if no conversations remain.
		if (wasActive || $conversations.length === 0) {
			window.location.href = '/chat';
		}
	}

	function handleSelect(e: CustomEvent<{ id: string }>) {
		onSelect(e.detail.id);
	}

	function handleExport(e: CustomEvent<{ id: string; title: string }>) {
		onExport(e.detail.id, e.detail.title);
	}
</script>

<div class="flex flex-col h-full">
	<!-- Search -->
	<div class="px-3 pb-2">
		<div class="relative">
			<span
				class="absolute left-2.5 top-1/2 -translate-y-1/2 flex items-center"
				style="color: var(--oo-fg-muted);"
				aria-hidden="true"
			>
				<Icon name="search" size="sm" />
			</span>
			<input
				bind:value={searchQuery}
				placeholder="Search... (Ctrl+K)"
				aria-label="Search conversations"
				class="w-full text-xs pl-8 pr-3 py-1.5 rounded-md outline-none"
				style="background-color: var(--oo-input-bg); color: var(--oo-fg-secondary);
					border: 1px solid var(--oo-input-bd);"
			/>
		</div>
	</div>

	<!-- List -->
	<div class="flex-1 overflow-y-auto px-2 space-y-0.5">
		{#if $loading}
			<ConversationSkeleton count={5} />
		{:else if $error}
			<div class="px-3 py-4 text-center">
				<p class="text-sm" style="color: var(--oo-status-error);">{$error}</p>
				<button
					on:click={loadConversations}
					class="mt-2 text-xs hover:underline"
					style="color: var(--oo-acc-400);"
				>
					Retry
				</button>
			</div>
		{:else if filteredConversations.length === 0}
			<!-- S87: Empty state guidance card -->
			{#if searchQuery}
				<div class="px-3 py-8 text-center text-sm" style="color: var(--oo-fg-muted);">
					No matching conversations
				</div>
			{:else}
				<div class="px-3 py-6 text-center">
					<div
						class="mx-auto w-12 h-12 rounded-xl flex items-center justify-center mb-3"
						style="background-color: var(--oo-msg-user-bg); border: 1px solid var(--oo-msg-user-bd);"
					>
						<span style="color: var(--oo-acc-400);"><Icon name="message-square" size="lg" /></span>
					</div>
					<p class="text-sm font-medium mb-1" style="color: var(--oo-fg-primary);">
						Start your first conversation
					</p>
					<p class="text-xs mb-4" style="color: var(--oo-fg-muted);">
						Ask a question, write code, or explore your local models.
					</p>
					<button
						on:click={onNewConversation}
						class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
						style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
					>
						<Icon name="plus" size="sm" />
						New conversation
					</button>
				</div>
			{/if}
		{:else}
			{#each groups as group (group.key)}
				<div class="oo-conv-group-label">{group.label}</div>
				{#each group.items as conv (conv.id)}
					<ConversationItem
						conversation={conv}
						isActive={$activeConversationId === conv.id}
						on:select={handleSelect}
						on:rename={handleRename}
						on:delete={handleDelete}
						on:export={handleExport}
					/>
				{/each}
			{/each}
		{/if}
	</div>
</div>

<style>
	.oo-conv-group-label {
		padding: var(--oo-space-2) var(--oo-space-3) var(--oo-space-1);
		font-size: var(--oo-text-2xs);
		text-transform: uppercase;
		letter-spacing: var(--oo-tracking-wide);
		color: var(--oo-fg-muted);
	}
</style>
