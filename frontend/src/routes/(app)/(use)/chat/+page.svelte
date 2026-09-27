<!--
  The chats index: every conversation, newest first, grouped by the day it
  last changed.

  The search is the server's: its words go in `q` (the address carries
  them, so the command palette's "Every chat matching" entry lands here),
  the server reads titles and messages and returns at most the search
  limit, and the page says when that limit was reached rather than showing
  a partial list as the whole.
  Without words, the listing comes a page at a time, from an offset.
  lib/chat/chatsIndex.ts builds each request and sorts what comes back.

  Every row keeps its actions in sight: a menu button, always drawn, opens
  Rename, Export and Delete. Rename and Delete ask in a dialog, which shows
  why the server refused, if it does; Export opens the shell's export
  dialog. A rename or a delete goes through the conversations store, so the
  sidebar's recent chats follow.
-->
<script lang="ts">
	import { afterNavigate, goto } from '$app/navigation';
	import { page } from '$app/stores';
	import Button from '$lib/ds/Button.svelte';
	import ConfirmDialog from '$lib/ds/ConfirmDialog.svelte';
	import EmptyState from '$lib/ds/EmptyState.svelte';
	import IconButton from '$lib/ds/IconButton.svelte';
	import InlineError from '$lib/ds/InlineError.svelte';
	import Input from '$lib/ds/Input.svelte';
	import Menu from '$lib/ds/Menu.svelte';
	import type { MenuItem } from '$lib/ds/types';
	import { listConversations } from '$lib/api/conversations';
	import {
		SEARCH_LIMIT,
		dayGroups,
		indexParams,
		limitReached,
		messagesOf,
		morePages,
		whenOf
	} from '$lib/chat/chatsIndex';
	import { deleteConv, renameConv } from '$lib/stores/conversations';
	import { openExportDialog } from '$lib/stores/exportDialog';
	import { isPhone } from '$lib/stores/ui';
	import { toastError } from '$lib/stores/notifications';
	import type { ConversationSummary } from '$lib/types';

	const ACTIONS: MenuItem[] = [
		{ id: 'rename', label: 'Rename', icon: 'pencil' },
		{ id: 'export', label: 'Export', icon: 'download' },
		{ id: 'delete', label: 'Delete', icon: 'trash', danger: true }
	];

	/** The words the listing was asked for: the address's `q`. */
	let query = '';
	/** The words in the field. */
	let words = '';
	let chats: ConversationSummary[] = [];
	/** How many conversations the last search returned: its limit is judged on it. */
	let returned = 0;
	/** The last page of the plain listing read, from 0. */
	let pageRead = 0;
	/** Conversations deleted here since the first page. */
	let removed = 0;
	let hasMore = false;
	let loading = true;
	let loadingMore = false;
	let failed: string | null = null;
	let now = new Date();
	/** Each request's ticket: an answer to an older one is dropped. */
	let ticket = 0;

	$: groups = dayGroups(chats, now);

	function titleOf(chat: ConversationSummary): string {
		return chat.title?.trim() || 'Untitled chat';
	}

	async function load(nextQuery: string) {
		const mine = ++ticket;
		query = nextQuery;
		loading = true;
		failed = null;
		try {
			const list = await listConversations(indexParams(nextQuery, 0));
			if (mine !== ticket) return;
			chats = list;
			returned = list.length;
			pageRead = 0;
			removed = 0;
			hasMore = morePages(nextQuery, list.length);
			now = new Date();
		} catch (err: unknown) {
			if (mine !== ticket) return;
			failed = err instanceof Error ? err.message : 'The chats could not be read';
			chats = [];
			returned = 0;
			hasMore = false;
		} finally {
			if (mine === ticket) loading = false;
		}
	}

	async function more() {
		const mine = ticket;
		loadingMore = true;
		try {
			const list = await listConversations(indexParams('', pageRead + 1, removed));
			if (mine !== ticket) return;
			const shown = new Set(chats.map((chat) => chat.id));
			chats = [...chats, ...list.filter((chat) => !shown.has(chat.id))];
			pageRead += 1;
			hasMore = morePages('', list.length);
		} catch {
			if (mine === ticket) toastError('More chats could not be read');
		} finally {
			loadingMore = false;
		}
	}

	function go(nextWords: string) {
		const url = new URL($page.url);
		const trimmed = nextWords.trim();
		if (trimmed) url.searchParams.set('q', trimmed);
		else url.searchParams.delete('q');
		goto(`${url.pathname}${url.search}`, { replaceState: true, keepFocus: true, noScroll: true });
	}

	function search() {
		go(words);
	}

	function clearSearch() {
		words = '';
		go('');
	}

	// The address decides what is listed: on arrival, and each time its words change.
	afterNavigate(() => {
		const next = ($page.url.searchParams.get('q') ?? '').trim();
		words = next;
		if (ticket === 0 || next !== query) load(next);
	});

	// -- Rename and delete, each asked in its dialog. ----------------------------
	let renaming: ConversationSummary | null = null;
	let deleting: ConversationSummary | null = null;
	let newTitle = '';
	let busy = false;
	let dialogError: string | null = null;

	function act(action: string, chat: ConversationSummary) {
		dialogError = null;
		if (action === 'rename') {
			newTitle = chat.title ?? '';
			renaming = chat;
		} else if (action === 'delete') {
			deleting = chat;
		} else if (action === 'export') {
			openExportDialog(chat.id, titleOf(chat));
		}
	}

	function closeDialogs() {
		if (busy) return;
		renaming = null;
		deleting = null;
		dialogError = null;
	}

	async function rename() {
		const chat = renaming;
		const title = newTitle.trim();
		if (!chat) return;
		if (!title) {
			dialogError = 'A chat needs a title.';
			return;
		}
		if (title === chat.title) {
			closeDialogs();
			return;
		}
		busy = true;
		dialogError = null;
		try {
			await renameConv(chat.id, title);
			chats = chats.map((c) => (c.id === chat.id ? { ...c, title } : c));
			busy = false;
			closeDialogs();
		} catch (err: unknown) {
			dialogError = err instanceof Error ? err.message : 'The chat could not be renamed';
			busy = false;
		}
	}

	async function remove() {
		const chat = deleting;
		if (!chat) return;
		busy = true;
		dialogError = null;
		try {
			await deleteConv(chat.id);
			chats = chats.filter((c) => c.id !== chat.id);
			removed += 1;
			busy = false;
			closeDialogs();
		} catch (err: unknown) {
			dialogError = err instanceof Error ? err.message : 'The chat could not be deleted';
			busy = false;
		}
	}
</script>

<svelte:head>
	<title>Chats</title>
</svelte:head>

<div class="oo-chats-scroll">
	<div class="oo-chats">
		<header class="oo-chats-head">
			<h1 class="oo-chats-title">Chats</h1>
			<form class="oo-chats-search" role="search" on:submit|preventDefault={search}>
				<Input
					label="Search chats"
					hideLabel
					placeholder="Search titles and messages"
					iconLeft="search"
					size={$isPhone ? 'lg' : 'md'}
					bind:value={words}
				/>
				{#if words || query}
					<IconButton icon="x" label="Clear the search" size={$isPhone ? 'lg' : 'md'} on:click={clearSearch} />
				{/if}
			</form>
		</header>

		{#if loading}
			<p class="oo-chats-note" role="status">Reading the chats...</p>
		{:else if failed}
			<div class="oo-chats-failed">
				<InlineError message={`The chats could not be read: ${failed}`} />
				<Button variant="secondary" shape="pill" on:click={() => load(query)}>Try again</Button>
			</div>
		{:else if chats.length === 0}
			{#if query}
				<EmptyState
					icon="search"
					title={`No chat matches "${query}"`}
					description="The search reads the titles and the messages of every chat."
				/>
			{:else}
				<EmptyState
					icon="chat"
					title="No chats yet"
					description="Start one with New chat, in the sidebar."
				/>
			{/if}
		{:else}
			{#if limitReached(query, returned)}
				<p class="oo-chats-limit" role="status">
					The search shows its first {SEARCH_LIMIT} matches for "{query}", the newest. Add a
					word to find an older one.
				</p>
			{:else if query}
				<p class="oo-chats-note" role="status">
					{chats.length}
					{chats.length === 1 ? 'chat matches' : 'chats match'} "{query}"
				</p>
			{/if}

			{#each groups as group (group.key)}
				<section class="oo-chats-day" aria-labelledby={`oo-chats-${group.key}`}>
					<h2 class="oo-chats-day-title" id={`oo-chats-${group.key}`}>{group.label}</h2>
					<ul class="oo-chats-list" role="list">
						{#each group.items as chat (chat.id)}
							<li class="oo-chat-row">
								<a class="oo-chat-link" href={`/chat/${chat.id}`}>
									<span class="oo-chat-title">{titleOf(chat)}</span>
									<span class="oo-chat-meta">
										<span>{messagesOf(chat.message_count)}</span>
										<span>{whenOf(chat, now)}</span>
									</span>
								</a>
								<Menu
									icon="more"
									label={`Actions for ${titleOf(chat)}`}
									size={$isPhone ? 'lg' : 'md'}
									items={ACTIONS}
									on:select={(event) => act(event.detail, chat)}
								/>
							</li>
						{/each}
					</ul>
				</section>
			{/each}

			{#if hasMore}
				<div class="oo-chats-more">
					<Button variant="secondary" shape="pill" loading={loadingMore} on:click={more}>
						Show more chats
					</Button>
				</div>
			{/if}
		{/if}
	</div>
</div>

<ConfirmDialog
	open={renaming !== null}
	title="Rename this chat"
	confirmLabel="Rename"
	{busy}
	error={dialogError}
	onConfirm={rename}
	onCancel={closeDialogs}
>
	<Input label="Title" bind:value={newTitle} autofocus />
</ConfirmDialog>

<ConfirmDialog
	open={deleting !== null}
	title="Delete this chat?"
	message={deleting
		? `"${titleOf(deleting)}" and its messages are deleted for good. This cannot be undone.`
		: ''}
	confirmLabel="Delete"
	danger
	{busy}
	error={dialogError}
	onConfirm={remove}
	onCancel={closeDialogs}
/>

<style>
	.oo-chats-scroll {
		box-sizing: border-box;
		height: 100%;
		overflow-y: auto;
		overscroll-behavior: contain;
	}
	.oo-chats {
		box-sizing: border-box;
		width: 100%;
		max-width: 800px;
		margin: 0 auto;
		padding: var(--oo-space-8) var(--oo-space-5) var(--oo-space-9);
	}

	.oo-chats-head {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--oo-space-3) var(--oo-space-5);
		margin-bottom: var(--oo-space-6);
	}
	.oo-chats-title {
		flex: 1;
		min-width: 8rem;
		margin: 0;
		color: var(--oo-fg-primary);
		font-family: var(--oo-font-serif);
		font-size: var(--oo-text-3xl);
		font-weight: 400;
		line-height: var(--oo-leading-tight);
	}
	.oo-chats-search {
		display: flex;
		flex: 1;
		align-items: center;
		gap: var(--oo-space-2);
		min-width: 14rem;
		max-width: 26rem;
	}
	.oo-chats-search > :global(.oo-field) {
		flex: 1;
	}

	.oo-chats-note,
	.oo-chats-limit {
		margin: 0 0 var(--oo-space-4);
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
		line-height: var(--oo-leading-normal);
	}
	.oo-chats-limit {
		color: var(--oo-fg-secondary);
	}

	.oo-chats-failed {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--oo-space-3);
	}

	.oo-chats-day + .oo-chats-day {
		margin-top: var(--oo-space-6);
	}
	.oo-chats-day-title {
		margin: 0 0 var(--oo-space-2);
		padding: 0 var(--oo-space-4);
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
		font-weight: 500;
	}

	.oo-chats-list {
		display: flex;
		flex-direction: column;
		gap: 2px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.oo-chat-row {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		padding-right: var(--oo-space-2);
		border: 1px solid transparent;
		border-radius: var(--oo-radius-xl);
	}
	.oo-chat-row:hover,
	.oo-chat-row:focus-within {
		background-color: var(--oo-bg-hover);
	}

	.oo-chat-link {
		display: flex;
		flex: 1;
		flex-direction: column;
		justify-content: center;
		gap: 2px;
		min-width: 0;
		min-height: 60px;
		padding: var(--oo-space-2) var(--oo-space-4);
		border-radius: var(--oo-radius-xl);
		color: var(--oo-fg-primary);
		text-decoration: none;
	}
	.oo-chat-title {
		overflow: hidden;
		font-family: var(--oo-font-serif);
		font-size: var(--oo-text-lg);
		line-height: var(--oo-leading-snug);
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.oo-chat-meta {
		display: flex;
		flex-wrap: wrap;
		gap: 0 var(--oo-space-3);
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
		font-variant-numeric: tabular-nums;
	}

	.oo-chats-more {
		display: flex;
		justify-content: center;
		margin-top: var(--oo-space-6);
	}

	@media (max-width: 767px) {
		.oo-chats {
			padding: var(--oo-space-5) var(--oo-space-4) var(--oo-space-8);
		}
		.oo-chat-link {
			min-height: 64px;
		}
	}
</style>
