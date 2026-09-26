<!--
  Sidebar.svelte
  The sidebar of both spaces, on the page's ground. From the top: the
  onion mark and the name, New chat, Search, the destinations of the space
  on screen (from the destination table, lib/nav/destinations.ts), the six
  most recent chats, then at the foot Preferences, the space switch and the
  status card with Stop all.

  Which entry is current is decided by lib/nav/active.ts and nothing else:
  the page itself is active, the destination a page sits under is its
  section (in a conversation, Chats is the section and the conversation's
  recent row is active); the active entry is marked by its weight and its
  tint, never by colour alone.

  Collapsed on a desktop, it is a 72 px rail of icons that keeps Stop all,
  and the count of the tool calls waiting on an approval while any wait; it
  is never unmounted. On a phone it is the drawer's content, with no
  collapse control of its own, and every link is a 44 px target.
-->
<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import Button from '$lib/ds/Button.svelte';
	import IconButton from '$lib/ds/IconButton.svelte';
	import Icon from '$lib/ds/Icon.svelte';
	import Input from '$lib/ds/Input.svelte';
	import StatusCard from './StatusCard.svelte';
	import SpaceSwitch from './SpaceSwitch.svelte';
	import StopAllButton from './StopAllButton.svelte';
	import ApprovalsPill from './ApprovalsPill.svelte';
	import { DESTINATIONS, spaceHome, visibleDestinations } from '$lib/nav/destinations';
	import { activeState } from '$lib/nav/active';
	import { spaceOf } from '$lib/nav/space';
	import { conversations, createNewConversation, loadConversations } from '$lib/stores/conversations';
	import { sidebarOpen } from '$lib/stores/ui';
	import { toastError } from '$lib/stores/notifications';

	/** On a desktop: the 72 px rail. */
	export let collapsed = false;
	/** The drawer's content on a phone. */
	export let phone = false;

	const RECENT = 6;
	// The componion's own switch arrives with its page; until then its entry
	// is not ready, and the visible list never shows it.
	const visibility = { componion: true };

	let searchWords = '';

	$: pathname = $page.url?.pathname ?? '/';
	$: space = spaceOf(pathname) ?? 'use';
	$: shown = visibleDestinations(DESTINATIONS, visibility);
	$: hrefs = shown.map((d) => d.href);
	$: preferences = shown.find((d) => d.id === 'preferences');
	$: entries = shown.filter((d) => d.space === space && d.id !== 'preferences');
	$: recent = $conversations.slice(0, RECENT);
	$: home = spaceHome('use', DESTINATIONS);
	$: homeLabel = DESTINATIONS.find((d) => d.href === home)?.label ?? 'Home';
	$: preferencesState = preferences ? stateOf(preferences.href, pathname, hrefs) : 'none';

	function stateOf(href: string, path: string, all: string[]) {
		return activeState(path, href, all);
	}

	function afterNavigate() {
		if (phone) sidebarOpen.set(false);
	}

	async function newChat() {
		try {
			const id = await createNewConversation();
			await goto(`/chat/${id}`);
			afterNavigate();
		} catch {
			toastError('Failed to create a conversation');
		}
	}

	function search() {
		const words = searchWords.trim();
		goto(words ? `/chat?q=${encodeURIComponent(words)}` : '/chat');
		afterNavigate();
	}

	async function openSearch() {
		sidebarOpen.set(true);
		await tick();
		const field = document.querySelector('[data-oo-search] input');
		if (field instanceof HTMLInputElement) field.focus();
	}

	onMount(() => {
		loadConversations();
	});
</script>

<div class="oo-sidebar" data-collapsed={collapsed ? 'true' : 'false'} data-phone={phone ? 'true' : 'false'}>
	<div class="oo-side-scroll">
		<div class="oo-side-brand">
			<a class="oo-brand" href={home} aria-label={`Opti-Oignon, ${homeLabel}`} on:click={afterNavigate}>
				<span class="oo-brand-mark"><Icon name="onion" size="lg" /></span>
				{#if !collapsed}<span class="oo-brand-name">Opti-Oignon</span>{/if}
			</a>
			{#if !phone && !collapsed}
				<IconButton
					icon="panel-left"
					label="Collapse the sidebar"
					expanded={true}
					on:click={() => sidebarOpen.set(false)}
				/>
			{/if}
		</div>

		{#if collapsed}
			<div class="oo-rail-actions">
				<IconButton icon="panel-right" label="Expand the sidebar" expanded={false} on:click={() => sidebarOpen.set(true)} />
				<IconButton icon="plus" label="New chat" variant="primary" on:click={newChat} />
				<IconButton icon="search" label="Search chats" on:click={openSearch} />
			</div>
		{:else}
			<div class="oo-side-actions">
				<Button variant="secondary" shape="pill" size="lg" block iconLeft="plus" on:click={newChat}>
					New chat
				</Button>
				<form
					class="oo-side-search"
					role="search"
					aria-label="Search chats from the sidebar"
					data-oo-search
					on:submit|preventDefault={search}
				>
					<Input
						label="Search chats"
						hideLabel
						placeholder="Search chats"
						iconLeft="search"
						size={phone ? 'lg' : 'md'}
						bind:value={searchWords}
					/>
				</form>
			</div>
		{/if}

		<nav class="oo-side-nav" aria-label={space === 'workshop' ? 'Workshop' : 'Use'}>
			<ul class="oo-side-list" role="list">
				{#each entries as d (d.id)}
					{@const state = stateOf(d.href, pathname, hrefs)}
					<li>
						<a
							class="oo-nav-link"
							href={d.href}
							aria-current={state === 'active' ? 'page' : undefined}
							data-state={state}
							aria-label={collapsed ? d.label : undefined}
							on:click={afterNavigate}
						>
							<Icon name={d.icon} size="md" />
							{#if !collapsed}<span>{d.label}</span>{/if}
						</a>
					</li>
				{/each}
			</ul>
		</nav>

		{#if !collapsed && space === 'use' && recent.length > 0}
			<section class="oo-side-recent" aria-labelledby="oo-side-recent-title">
				<h2 class="oo-side-heading" id="oo-side-recent-title">Recent</h2>
				<ul class="oo-side-list" role="list">
					{#each recent as chat (chat.id)}
						{@const state = stateOf(`/chat/${chat.id}`, pathname, hrefs)}
						<li>
							<a
								class="oo-recent-link"
								href={`/chat/${chat.id}`}
								aria-current={state === 'active' ? 'page' : undefined}
								on:click={afterNavigate}
							>
								{chat.title || 'Untitled chat'}
							</a>
						</li>
					{/each}
				</ul>
			</section>
		{/if}

	</div>

	<div class="oo-side-foot">
		{#if preferences}
			<a
				class="oo-nav-link"
				href={preferences.href}
				aria-current={preferencesState === 'active' ? 'page' : undefined}
				data-state={preferencesState}
				aria-label={collapsed ? preferences.label : undefined}
				on:click={afterNavigate}
			>
				<Icon name={preferences.icon} size="md" />
				{#if !collapsed}<span>{preferences.label}</span>{/if}
			</a>
		{/if}
		{#if collapsed}
			<SpaceSwitch compact />
			<ApprovalsPill compact />
			<StopAllButton placement="rail" />
		{:else}
			<SpaceSwitch large={phone} />
			<StatusCard large={phone} />
		{/if}
	</div>
</div>

<style>
	/* The foot (Preferences, the switch, the status card with Stop all) never
	   scrolls away: the part above it scrolls on its own. */
	.oo-sidebar {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
		box-sizing: border-box;
		width: 272px;
		height: 100%;
		padding: var(--oo-space-6) var(--oo-space-3) var(--oo-space-4) var(--oo-space-4);
		overflow: hidden;
		background-color: var(--oo-sidebar-bg);
		color: var(--oo-fg-primary);
	}
	.oo-side-scroll {
		display: flex;
		flex: 1;
		flex-direction: column;
		gap: var(--oo-space-4);
		min-height: 0;
		overflow-y: auto;
		overscroll-behavior: contain;
	}
	.oo-sidebar[data-collapsed='true'] .oo-side-scroll {
		align-items: center;
	}
	.oo-sidebar[data-collapsed='true'] {
		width: 72px;
		align-items: center;
		padding: var(--oo-space-5) var(--oo-space-1) var(--oo-space-4);
	}
	.oo-sidebar[data-phone='true'] {
		width: min(300px, 86vw);
		padding-bottom: calc(var(--oo-space-4) + env(safe-area-inset-bottom, 0px));
	}

	.oo-side-brand {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-2);
		min-height: 40px;
	}
	.oo-brand {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-2);
		padding: 0 var(--oo-space-2);
		border-radius: var(--oo-radius-full);
		color: var(--oo-fg-primary);
		text-decoration: none;
	}
	.oo-sidebar[data-phone='true'] .oo-brand {
		min-height: 44px;
	}
	.oo-brand-mark {
		display: inline-flex;
		color: var(--oo-acc-ink);
	}
	.oo-brand-name {
		font-family: var(--oo-font-serif);
		font-size: 19px;
		font-weight: 600;
		line-height: 24px;
	}

	.oo-rail-actions,
	.oo-side-actions {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}
	.oo-rail-actions {
		align-items: center;
	}

	.oo-side-list {
		display: flex;
		flex-direction: column;
		gap: 2px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.oo-nav-link {
		display: flex;
		align-items: center;
		gap: var(--oo-space-3);
		min-height: 44px;
		padding: 0 var(--oo-space-4) 0 var(--oo-space-3);
		border: 1px solid transparent;
		border-radius: var(--oo-radius-full);
		color: var(--oo-fg-primary);
		font-size: var(--oo-text-md);
		text-decoration: none;
	}
	.oo-nav-link :global(svg) {
		flex-shrink: 0;
		color: var(--oo-fg-muted);
	}
	.oo-nav-link:hover {
		background-color: var(--oo-bg-hover);
	}
	.oo-nav-link[aria-current='page'] {
		border: 1px solid var(--oo-edge);
		background-color: var(--oo-bg-tint-1);
		font-weight: 600;
	}
	.oo-nav-link[data-state='section'] {
		font-weight: 600;
	}
	.oo-nav-link[aria-current='page'] :global(svg),
	.oo-nav-link[data-state='section'] :global(svg) {
		color: var(--oo-acc-ink);
		font-weight: 600;
	}
	.oo-sidebar[data-collapsed='true'] .oo-nav-link {
		justify-content: center;
		width: 44px;
		padding: 0;
	}

	.oo-side-heading {
		margin: 0 0 var(--oo-space-1);
		padding: 0 var(--oo-space-4);
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
		font-weight: 500;
	}
	.oo-recent-link {
		display: block;
		overflow: hidden;
		min-height: 36px;
		padding: var(--oo-space-2) var(--oo-space-4);
		border: 1px solid transparent;
		border-radius: var(--oo-radius-full);
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
		line-height: var(--oo-leading-snug);
		text-decoration: none;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.oo-recent-link:hover {
		background-color: var(--oo-bg-hover);
	}
	.oo-recent-link[aria-current='page'] {
		border: 1px solid var(--oo-edge);
		background-color: var(--oo-bg-tint-1);
		color: var(--oo-fg-primary);
		font-weight: 600;
	}
	.oo-sidebar[data-phone='true'] .oo-recent-link {
		min-height: 44px;
		padding-top: var(--oo-space-3);
		padding-bottom: var(--oo-space-3);
	}

	.oo-side-foot {
		display: flex;
		flex-shrink: 0;
		flex-direction: column;
		gap: var(--oo-space-3);
	}
	.oo-sidebar[data-collapsed='true'] .oo-side-foot {
		align-items: center;
	}
</style>
