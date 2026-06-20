<!--
  Settings route layout.
  Wraps settings pages in AppShell with sidebar navigation.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import AppShell from '$lib/components/layout/AppShell.svelte';
	import ErrorBoundary from '$lib/components/ui/ErrorBoundary.svelte';

	function handleSelect(id: string) {
		goto(`/chat/${id}`);
	}

	function handleCreate() {
		goto('/chat');
	}
</script>

<AppShell onSelect={handleSelect} onCreate={handleCreate}>
	<svelte:fragment slot="header">
		<div class="flex items-center gap-2 flex-1 min-w-0">
			<h1 class="text-sm font-medium text-surface-300">Settings</h1>
		</div>
	</svelte:fragment>

	<ErrorBoundary fallbackMessage="Settings failed to load">
		<slot />
	</ErrorBoundary>
</AppShell>

