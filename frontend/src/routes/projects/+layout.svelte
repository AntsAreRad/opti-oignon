<!--
  Projects route layout (S169).
  Wraps /projects and /projects/[id] in AppShell so the shared sidebar
  (with the Projects section context) and header cluster apply. The header
  label reflects the active project name on the detail route, "Projects"
  on the list. ProjectDetail is promoted to its own /projects/[id] route;
  navigation between list and detail is real routing, not internal state.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import AppShell from '$lib/components/layout/AppShell.svelte';
	import ErrorBoundary from '$lib/components/ui/ErrorBoundary.svelte';
	import { activeProjectDetail } from '$lib/stores/projects';

	function handleSelect(id: string) {
		goto(`/chat/${id}`);
	}

	function handleCreate() {
		goto('/chat');
	}

	$: detailId = $page.params.id ?? '';
	$: headerTitle = detailId ? ($activeProjectDetail?.name ?? 'Project') : 'Projects';
</script>

<AppShell onSelect={handleSelect} onCreate={handleCreate}>
	<svelte:fragment slot="header">
		<div class="flex items-center gap-2 flex-1 min-w-0">
			{#if detailId}
				<a
					href="/projects"
					class="flex items-center gap-1.5 text-xs shrink-0"
					style="color: var(--oo-fg-tertiary);"
				>
					<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path d="M15 19l-7-7 7-7" />
					</svg>
					Projects
				</a>
				<span class="w-px h-4 shrink-0" style="background-color: var(--oo-bd-default);"></span>
			{/if}
			<h1 class="text-sm font-medium truncate" style="color: var(--oo-fg-secondary);">{headerTitle}</h1>
		</div>
	</svelte:fragment>

	<ErrorBoundary fallbackMessage="Projects failed to load">
		<slot />
	</ErrorBoundary>
</AppShell>
