<!--
  Verify route layout.
  Wraps /verify in AppShell so the shared sidebar and header cluster apply,
  mirroring the Notes, Settings and Projects route layouts. This restores the
  in-app navigation (and the back path) that the standalone /claims,
  /verify-citations and /verify-answer pages lacked.
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
			<h1 class="text-sm font-medium" style="color: var(--oo-fg-secondary);">Verify</h1>
		</div>
	</svelte:fragment>

	<ErrorBoundary fallbackMessage="Verification failed to load">
		<slot />
	</ErrorBoundary>
</AppShell>
