<!--
  Sidebar.svelte (refactored S167)
  Two-part sidebar (spec 8.3): App Nav at the top (primary sections) and a
  route-dependent Section Context below (via SectionContextList). Footer
  keeps the build tag, security badge and a quick light/dark toggle.
  S132: touch-friendly nav links (min-height 44px), touch-scroll body,
  safe-area footer padding, touch-target controls.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import SectionContextList from '$lib/components/sidebar/SectionContextList.svelte';
	import SecurityBadge from '$lib/components/sidebar/SecurityBadge.svelte';
	import Icon from '$lib/ds/Icon.svelte';
	import { darkMode } from '$lib/stores/ui';
	import { setPalette } from '$lib/stores/preferences';

	export let onSelect: (id: string) => void = () => {};
	export let onCreate: () => void = () => {};
	export let onExport: (id: string, title: string) => void = () => {};

	$: currentPath = $page.url?.pathname ?? '/chat';

	// S87: Dynamic version from health endpoint.
	let appVersion = '';

	onMount(async () => {
		try {
			const resp = await fetch('/api/health');
			if (resp.ok) {
				const data = await resp.json();
				appVersion = data.version || '';
			}
		} catch {}
	});

	// Palette-aware quick toggle: switch to the light/dark default palette.
	// (The full palette picker lives in the header ThemeSwitcher.)
	function quickToggleTheme() {
		setPalette($darkMode ? 'parchment' : 'anthracite');
	}

	const navLinks: { href: string; label: string; icon: string }[] = [
		{ href: '/chat', label: 'Chat', icon: 'message-square' },
		{ href: '/projects', label: 'Projects', icon: 'folder' },
		{ href: '/notes', label: 'Notes', icon: 'file-text' },
		{ href: '/verify', label: 'Verify', icon: 'shield-check' },
		{ href: '/settings', label: 'Settings', icon: 'settings' },
		{ href: '/benchmark', label: 'Benchmark', icon: 'bar-chart-3' },
		{ href: '/health', label: 'System Status', icon: 'heart-pulse' }
	];
</script>

<aside
	class="flex flex-col h-full"
	style="background-color: var(--oo-sidebar-bg); border-right: 1px solid var(--oo-bd-subtle);"
>
	<!-- Header -->
	<div
		class="flex items-center gap-2 px-4 py-3"
		style="border-bottom: 1px solid var(--oo-bd-subtle);"
	>
		<img
			src="/bousier-oignon.png"
			alt="Opti-Oignon"
			class="w-12 h-12 object-contain rounded oo-logo-adaptive"
		/>
		<span
			class="text-lg font-semibold tracking-tight"
			style="color: var(--oo-fg-primary); letter-spacing: var(--oo-tracking-tight);"
			>Opti-Oignon</span
		>
		{#if appVersion}
			<span class="text-xs font-mono" style="color: var(--oo-fg-muted);">v{appVersion}</span>
		{/if}
	</div>

	<!-- App Nav: S132 touch-friendly min-height -->
	<nav class="px-2 pt-2 pb-1 space-y-0.5">
		{#each navLinks as link}
			<a
				href={link.href}
				class="flex items-center gap-2.5 px-3 py-2 rounded-md text-sm transition-colors"
				aria-current={currentPath.startsWith(link.href) ? 'page' : undefined}
				style="min-height: 44px;
					{currentPath.startsWith(link.href)
					? 'background-color: var(--oo-tobacco-bg); color: var(--oo-tobacco);'
					: 'color: var(--oo-fg-tertiary);'}"
			>
				<Icon name={link.icon} size="sm" />
				{link.label}
			</a>
		{/each}
	</nav>

	<hr class="mx-3" style="border-color: var(--oo-bd-subtle);" />

	<!-- Section Context: route-dependent body, S132 touch-scroll -->
	<div class="flex-1 min-h-0 pt-2 flex flex-col touch-scroll">
		<SectionContextList {onSelect} {onCreate} {onExport} />
	</div>

	<!-- S132: Footer with safe-area bottom padding for notched phones -->
	<div
		class="px-4 py-2 flex items-center justify-between safe-area-bottom"
		style="border-top: 1px solid var(--oo-bd-subtle);"
	>
		<div class="flex items-center gap-2">
			<span class="text-xs" style="color: var(--oo-fg-faint);">FastAPI + SvelteKit</span>
			<SecurityBadge />
		</div>
		<button
			on:click={quickToggleTheme}
			class="p-1.5 rounded-md transition-colors touch-target"
			style="color: var(--oo-fg-muted);"
			title={$darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
			aria-label={$darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
		>
			{#if $darkMode}
				<Icon name="sun" size="sm" />
			{:else}
				<Icon name="moon" size="sm" />
			{/if}
		</button>
	</div>
</aside>
