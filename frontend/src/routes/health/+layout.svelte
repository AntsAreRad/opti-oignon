<!--
  Health route layout.
  Sidebar plus shared navigation bar.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import AppShell from '$lib/components/layout/AppShell.svelte';

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
			<h1 class="text-sm font-medium" style="color: var(--oo-fg-secondary);">System Status</h1>
		</div>
	</svelte:fragment>

	<slot />
</AppShell>

