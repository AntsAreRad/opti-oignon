<!--
  AppShell.svelte
  Main layout: collapsible sidebar + content area + right panel.
  Mobile: sidebar as overlay with swipe-to-close, panel as overlay.
  Desktop: sidebar fixed, panel on the right.
  Enhanced mobile responsive -- swipe gesture, touch targets, dvh, safe-area.
  Slots: header, subheader, panel-toggle, default (main area), panel (right panel).
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { sidebarOpen, toggleSidebar } from '$lib/stores/ui';
	import { panelWidth, isPanelOpen, closePanel, setPanelWidth, PANEL_MIN_WIDTH, PANEL_MAX_WIDTH } from '$lib/stores/panels';
	import SidePanel from '$lib/ds/SidePanel.svelte';
	import Sidebar from './Sidebar.svelte';
	import Header from './Header.svelte';
	import StatusFooter from './StatusFooter.svelte';

	export let onSelect: (id: string) => void = () => {};
	export let onCreate: () => void = () => {};
	export let onExport: (id: string, title: string) => void = () => {};

	let isMobile = false;

	// Swipe-to-close state for sidebar
	let sidebarEl: HTMLDivElement;
	let swipeTouchStartX = 0;
	let swipeTouchCurrentX = 0;
	let isSwiping = false;
	const SWIPE_THRESHOLD = 60;

	function checkMobile() {
		const wasMobile = isMobile;
		isMobile = typeof window !== 'undefined' && window.innerWidth < 768;
		// Close sidebar when switching to mobile if open
		if (!wasMobile && isMobile && $sidebarOpen) {
			sidebarOpen.set(false);
		}
	}

	// Swipe-to-close touch handlers for sidebar overlay
	function handleSidebarTouchStart(event: TouchEvent) {
		if (!isMobile || !$sidebarOpen) return;
		const touch = event.touches[0];
		swipeTouchStartX = touch.clientX;
		swipeTouchCurrentX = touch.clientX;
		isSwiping = true;
	}

	function handleSidebarTouchMove(event: TouchEvent) {
		if (!isSwiping) return;
		const touch = event.touches[0];
		swipeTouchCurrentX = touch.clientX;
		// Only track leftward swipes (to dismiss sidebar)
		const delta = swipeTouchStartX - swipeTouchCurrentX;
		if (delta > 0 && sidebarEl) {
			// Apply real-time transform for visual feedback
			const offset = Math.min(delta, 280);
			sidebarEl.style.transform = `translateX(-${offset}px)`;
		}
	}

	function handleSidebarTouchEnd() {
		if (!isSwiping) return;
		isSwiping = false;
		const delta = swipeTouchStartX - swipeTouchCurrentX;
		if (sidebarEl) {
			sidebarEl.style.transform = '';
		}
		// If swiped left beyond threshold, close sidebar
		if (delta > SWIPE_THRESHOLD) {
			sidebarOpen.set(false);
		}
	}

	onMount(() => {
		checkMobile();
		// Start with sidebar closed on mobile
		if (isMobile) {
			sidebarOpen.set(false);
		}
		if (typeof window !== 'undefined') {
			window.addEventListener('resize', checkMobile);
		}
	});

	onDestroy(() => {
		if (typeof window !== 'undefined') {
			window.removeEventListener('resize', checkMobile);
		}
	});
</script>

<div class="h-viewport flex overflow-hidden" style="background-color: var(--oo-bg-base);">
	<!-- Route change announcements for screen readers -->
	<div class="sr-only" aria-live="polite" aria-atomic="true" id="oo-route-announcer"></div>

	<!-- Overlay backdrop (sidebar, mobile only) with fade transition -->
	{#if $sidebarOpen && isMobile}
		<button
			class="fixed inset-0 z-20 md:hidden sidebar-mobile-backdrop"
			style="background-color: rgba(0, 0, 0, 0.5);"
			on:click={toggleSidebar}
			aria-label="Close sidebar"
		/>
	{/if}

	<!-- Overlay mobile (panel) -->
	{#if $isPanelOpen && isMobile}
		<button
			class="fixed inset-0 bg-[var(--oo-scrim)] z-40 md:hidden"
			on:click={closePanel}
			aria-label="Close panel"
		/>
	{/if}

	<!-- Sidebar with swipe-to-close on mobile -->
	<nav
		aria-label="Sidebar navigation"
		bind:this={sidebarEl}
		class="shrink-0 h-full z-30 sidebar-transition sidebar-mobile-enter
			fixed md:relative
			{$sidebarOpen ? 'w-[280px] translate-x-0' : 'w-0 -translate-x-full md:w-0'}"
		on:touchstart={handleSidebarTouchStart}
		on:touchmove={handleSidebarTouchMove}
		on:touchend={handleSidebarTouchEnd}
	>
		{#if $sidebarOpen}
			<div class="w-[280px] h-full animate-sidebar-slide safe-area-pad safe-area-pad-top">
				<Sidebar {onSelect} {onCreate} {onExport} />
			</div>
		{/if}
	</nav>

	<!-- Main content + panel wrapper -->
	<div class="flex-1 flex flex-col min-w-0 h-full">
		<!-- Top bar with safe-area padding -->
		<header class="flex items-center gap-3 px-3 sm:px-4 h-12 shrink-0 safe-area-pad"
			style="border-bottom: 1px solid var(--oo-bd-subtle); background-color: var(--oo-header-bg);"
		>
			<!-- Touch-friendly hamburger button (44px target on mobile) -->
			<button
				on:click={toggleSidebar}
				class="p-1.5 rounded-md shrink-0
					{isMobile ? 'touch-target' : ''}"
				style="color: var(--oo-fg-tertiary);"
				title="Toggle sidebar"
				aria-label="Toggle sidebar"
				aria-expanded={$sidebarOpen}
			>
				<svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					{#if $sidebarOpen && !isMobile}
						<path d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
					{:else}
						<path d="M4 6h16M4 12h16M4 18h16" />
					{/if}
				</svg>
			</button>
			<slot name="header" />

			<!-- Consolidated header status cluster -->
			<div class="ml-auto shrink-0 flex items-center gap-2">
				<Header />
				<slot name="panel-toggle" />
			</div>
		</header>

		<!-- Sub-header (presets, options, etc.) -->
		<slot name="subheader" />

		<!-- Content + panel split -->
		<div class="flex-1 min-h-0 flex overflow-hidden">
			<!-- Main content area -->
			<main id="main-content" class="flex-1 min-w-0 overflow-hidden relative">
				<slot />
			</main>

			<!-- Right panel: beside the page on a desktop, over its edge on a phone -->
			{#if $isPanelOpen}
				<SidePanel
					label="Side panel"
					class="panel-transition animate-panel-slide"
					width={$panelWidth}
					min={PANEL_MIN_WIDTH}
					max={PANEL_MAX_WIDTH}
					overlay={isMobile}
					on:resize={(event) => setPanelWidth(event.detail)}
				>
					<slot name="panel" />
				</SidePanel>
			{/if}
		</div>

		<!-- Optional thin status footer (spec 8.5) -->
		<StatusFooter />
	</div>
</div>
