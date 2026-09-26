<!--
  AppShell.svelte
  The one frame of both spaces, mounted once, by the layout every page of
  Use and Workshop sits under; the page comes through the default slot, the
  only one it has. The sidebar sits on the page's ground and the page on a
  sheet beside it.

  On a desktop the sidebar is never unmounted: expanded, or collapsed to
  its 72 px rail, which keeps Stop all. On a phone (under 768 px) a header
  outside the drawer holds the drawer's opener and Stop all, so the stop is
  one tap away while the drawer is shut; the sidebar opens as the drawer,
  a modal dialog with its own close control and its own Stop all: while it
  is open the header and the page behind it are inert, focus moves into it,
  and it goes back to the opener when the drawer shuts.

  At every width the shell keeps clear of the safe areas (a notch, rounded
  corners, the home indicator): the sidebar and the sheet read the insets
  of their edges, and on a phone the header and the drawer do.

  The shell also remembers the last route of each space, for the space
  switch.
-->
<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { afterNavigate } from '$app/navigation';
	import { page } from '$app/stores';
	import IconButton from '$lib/ds/IconButton.svelte';
	import Sidebar from './Sidebar.svelte';
	import PhoneHeader from './PhoneHeader.svelte';
	import { DESTINATIONS, visibleDestinations } from '$lib/nav/destinations';
	import { destinationFor } from '$lib/nav/active';
	import { rememberRoute } from '$lib/nav/space';
	import { lastRoutes } from '$lib/stores/lastRoutes';
	import { isPhone, sidebarOpen, toggleSidebar } from '$lib/stores/ui';

	const DRAWER = 'oo-drawer';
	const PHONE_QUERY = '(max-width: 767.98px)';

	$: pathname = $page.url?.pathname ?? '/';
	$: route = pathname + ($page.url?.search ?? '');
	$: lastRoutes.update((last) => rememberRoute(last, route));
	$: title = destinationFor(pathname, visibleDestinations(DESTINATIONS, { componion: true }))?.label ?? 'Opti-Oignon';

	let drawerPanel: HTMLElement | undefined;
	let wasOpen = false;

	function closeDrawer() {
		sidebarOpen.set(false);
	}

	/** Focus goes to the drawer's first control when it opens. */
	async function focusIntoDrawer() {
		await tick();
		drawerPanel?.querySelector<HTMLElement>('button, [href], input')?.focus();
	}

	/** And back to the opener that controls it when it shuts. */
	async function focusBackToOpener() {
		await tick();
		document.querySelector<HTMLElement>(`[aria-controls="${DRAWER}"]`)?.focus();
	}

	$: drawerOpen = $isPhone && $sidebarOpen;
	$: if (typeof document !== 'undefined' && drawerOpen !== wasOpen) {
		wasOpen = drawerOpen;
		if (drawerOpen) void focusIntoDrawer();
		else if ($isPhone) void focusBackToOpener();
	}

	function closeOnEscape(event: KeyboardEvent) {
		if ($isPhone && $sidebarOpen && event.key === 'Escape') closeDrawer();
	}

	afterNavigate(() => {
		if ($isPhone) closeDrawer();
	});

	onMount(() => {
		const query = window.matchMedia(PHONE_QUERY);
		const follow = (phone: boolean) => {
			isPhone.set(phone);
			// A phone starts with its drawer shut; a desktop with its sidebar expanded.
			sidebarOpen.set(!phone);
		};
		follow(query.matches);
		const onChange = (event: MediaQueryListEvent) => follow(event.matches);
		query.addEventListener('change', onChange);
		return () => query.removeEventListener('change', onChange);
	});
</script>

<svelte:window on:keydown={closeOnEscape} />

<div class="oo-shell" data-phone={$isPhone ? 'true' : 'false'}>
	{#if $isPhone}
		<div class="oo-phone-top" inert={drawerOpen || undefined}>
			<PhoneHeader {title} open={$sidebarOpen} drawer={DRAWER} onToggle={toggleSidebar} />
		</div>
		<div id={DRAWER} class="oo-drawer" hidden={!$sidebarOpen}>
			{#if $sidebarOpen}
				<button type="button" class="oo-drawer-scrim" tabindex="-1" aria-label="Close navigation" on:click={closeDrawer}></button>
				<div class="oo-drawer-panel" role="dialog" aria-modal="true" aria-label="Navigation" bind:this={drawerPanel}>
					<div class="oo-drawer-close">
						<IconButton icon="x" size="lg" label="Close navigation" on:click={closeDrawer} />
					</div>
					<Sidebar phone />
				</div>
			{/if}
		</div>
	{:else}
		<div class="oo-shell-side">
			<Sidebar collapsed={!$sidebarOpen} />
		</div>
	{/if}

	<div class="oo-sheet" inert={drawerOpen || undefined}>
		<main id="main-content" class="oo-main">
			<slot />
		</main>
	</div>
</div>

<style>
	.oo-shell {
		display: flex;
		height: 100vh;
		height: 100dvh;
		overflow: hidden;
		background-color: var(--oo-bg-base);
	}
	.oo-shell[data-phone='true'] {
		flex-direction: column;
	}

	/* The sidebar keeps clear of the top, bottom and left insets: a phone on
	   its side is 768 px wide and more, and draws this branch. */
	.oo-shell-side {
		display: flex;
		flex-shrink: 0;
		box-sizing: border-box;
		height: 100%;
		padding: env(safe-area-inset-top, 0px) 0 env(safe-area-inset-bottom, 0px) env(safe-area-inset-left, 0px);
	}

	/* The page: a sheet on the surface beside the sidebar, clear of the
	   top, right and bottom insets. */
	.oo-sheet {
		display: flex;
		flex: 1;
		flex-direction: column;
		min-width: 0;
		min-height: 0;
		margin: calc(var(--oo-space-3) + env(safe-area-inset-top, 0px)) calc(var(--oo-space-3) + env(safe-area-inset-right, 0px))
			calc(var(--oo-space-3) + env(safe-area-inset-bottom, 0px)) 0;
		overflow: hidden;
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-4xl);
		background-color: var(--oo-bg-surface);
		box-shadow: var(--oo-shadow-sm);
	}
	.oo-shell[data-phone='true'] .oo-sheet {
		margin: 0;
		border-radius: var(--oo-radius-2xl) var(--oo-radius-2xl) 0 0;
		box-shadow: none;
		padding-bottom: env(safe-area-inset-bottom, 0px);
	}

	.oo-main {
		position: relative;
		display: flex;
		flex: 1;
		flex-direction: column;
		min-width: 0;
		min-height: 0;
		overflow: hidden;
	}

	/* The phone's drawer: the sidebar over the page, a veil behind it. */
	.oo-drawer {
		position: fixed;
		inset: 0;
		z-index: var(--oo-z-modal);
	}
	.oo-drawer[hidden] {
		display: none;
	}
	.oo-drawer-scrim {
		position: absolute;
		inset: 0;
		width: 100%;
		padding: 0;
		border: 0;
		background-color: var(--oo-scrim);
		cursor: pointer;
	}
	.oo-drawer-panel {
		position: absolute;
		top: 0;
		bottom: 0;
		left: 0;
		display: flex;
		flex-direction: column;
		padding-top: env(safe-area-inset-top, 0px);
		padding-left: env(safe-area-inset-left, 0px);
		background-color: var(--oo-bg-base);
		box-shadow: var(--oo-shadow-lg);
	}
	.oo-drawer-close {
		display: flex;
		flex-shrink: 0;
		justify-content: flex-end;
		padding: var(--oo-space-2) var(--oo-space-2) 0;
	}
	.oo-drawer-panel > :global(.oo-sidebar) {
		flex: 1;
		min-height: 0;
	}
	.oo-phone-top {
		display: contents;
	}
</style>
