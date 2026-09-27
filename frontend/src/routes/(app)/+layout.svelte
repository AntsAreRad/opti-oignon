<!--
  The layout every page of both spaces sits under: the one shell, mounted
  once, so moving between pages and spaces never mounts it again. It also
  holds what belongs to every page: the approvals drawer (the agent waits
  on it wherever the reader is), the export dialog, opened through its
  store from anywhere, and the command palette, which Ctrl+K and the
  sidebar's Search entry open through its store.
-->
<script lang="ts">
	import AppShell from '$lib/components/layout/AppShell.svelte';
	import ErrorBoundary from '$lib/components/ui/ErrorBoundary.svelte';
	import ToolCallApprovalDrawer from '$lib/components/chat/ToolCallApprovalDrawer.svelte';
	import ExportDialog from '$lib/components/chat/ExportDialog.svelte';
	import CommandPalette from '$lib/components/palette/CommandPalette.svelte';
	import { exportDialog, closeExportDialog } from '$lib/stores/exportDialog';
</script>

<AppShell>
	<ErrorBoundary fallbackMessage="This page failed to load">
		<slot />
	</ErrorBoundary>
</AppShell>

{#if $exportDialog.open && $exportDialog.id}
	<ExportDialog
		conversationId={$exportDialog.id}
		conversationTitle={$exportDialog.title}
		on:close={closeExportDialog}
	/>
{/if}

<ToolCallApprovalDrawer />

<CommandPalette />
