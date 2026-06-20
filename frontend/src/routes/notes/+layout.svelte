<!--
  Notes route layout (S248).
  Wraps /notes in AppShell so the shared sidebar and header cluster apply,
  mirroring the Settings and Projects route layouts. Selecting a conversation or
  creating one routes to chat, consistent with the other non-chat sections.
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
			<h1 class="text-sm font-medium" style="color: var(--oo-fg-secondary);">Notes</h1>
		</div>
	</svelte:fragment>

	<ErrorBoundary fallbackMessage="Notes failed to load">
		<slot />
	</ErrorBoundary>
</AppShell>
