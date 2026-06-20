<!--
  Chat layout: sidebar + main content + side panels.
  Manages navigation between conversations.
  Integrates model/preset selectors in the header.
  Panels: artifacts, code, memory, pipelines in the panel slot.
  ExportDialog: modal for exporting conversations.
  Keyboard shortcuts: Escape (closes sidebar/panel on mobile).
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import { onMount, onDestroy } from 'svelte';
	import AppShell from '$lib/components/layout/AppShell.svelte';
	import ModelSelector from '$lib/components/chat/ModelSelector.svelte';
	import PresetSelector from '$lib/components/chat/PresetSelector.svelte';
	import PanelToggle from '$lib/components/panels/PanelToggle.svelte';
	import ArtifactPanel from '$lib/components/panels/ArtifactPanel.svelte';
	import CodePanel from '$lib/components/panels/CodePanel.svelte';
	import MemoryPanel from '$lib/components/panels/MemoryPanel.svelte';
	import PipelinePanel from '$lib/components/panels/PipelinePanel.svelte';
	import ExecPipelinePanel from '$lib/components/panels/ExecPipelinePanel.svelte';
	import PluginsQuickPanel from '$lib/components/panels/PluginsQuickPanel.svelte';
	import AgentPanel from '$lib/components/panels/AgentPanel.svelte';
	import SandboxPanel from '$lib/components/panels/SandboxPanel.svelte';
	import ContextPanel from '$lib/components/chat/ContextPanel.svelte';
	import ExportDialog from '$lib/components/chat/ExportDialog.svelte';
	import ToolCallApprovalDrawer from '$lib/components/chat/ToolCallApprovalDrawer.svelte';
	import ErrorBoundary from '$lib/components/ui/ErrorBoundary.svelte';
	import ContextBar from '$lib/components/chat/ContextBar.svelte';
	import {
		activeConversationId,
		activeConversation,
		selectConversation,
		createNewConversation,
		loadConversations
	} from '$lib/stores/conversations';
	import { loadOptions } from '$lib/stores/chatOptions';
	import { sidebarOpen } from '$lib/stores/ui';
	import { activePanel, closePanel } from '$lib/stores/panels';
	import { toastError } from '$lib/stores/notifications';

	// Sync route -> store
	$: routeId = $page.params?.id ?? null;
	$: if (routeId && routeId !== $activeConversationId) {
		selectConversation(routeId);
	}

	// Detect mobile (<768px)
	let isMobile = false;
	function checkMobile() {
		isMobile = typeof window !== 'undefined' && window.innerWidth < 768;
	}

	function handleSelect(id: string) {
		goto(`/chat/${id}`);
		if (isMobile) sidebarOpen.set(false);
	}

	async function handleCreate() {
		try {
			const id = await createNewConversation();
			goto(`/chat/${id}`);
			if (isMobile) sidebarOpen.set(false);
		} catch {
			toastError('Failed to create conversation');
		}
	}

	// -- Export dialog state --
	let showExport = false;
	let exportConvId = '';
	let exportConvTitle = '';

	function handleExportFromSidebar(id: string, title: string) {
		exportConvId = id;
		exportConvTitle = title;
		showExport = true;
	}

	// Local chat keyboard shortcuts (Escape to close panel/sidebar)
	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			if (showExport) return;
			if ($activePanel !== 'none') {
				closePanel();
			} else if (isMobile && $sidebarOpen) {
				sidebarOpen.set(false);
			}
		}
	}

	onMount(() => {
		loadConversations();
		loadOptions();
		checkMobile();
		if (typeof window !== 'undefined') {
			window.addEventListener('resize', checkMobile);
			window.addEventListener('opti-export-conversation', handleGlobalExport);
		}
	});

	onDestroy(() => {
		if (typeof window !== 'undefined') {
			window.removeEventListener('resize', checkMobile);
			window.removeEventListener('opti-export-conversation', handleGlobalExport);
		}
	});

	function handleGlobalExport(e: Event) {
		const detail = (e as CustomEvent).detail;
		if (detail?.id) {
			exportConvId = detail.id;
			exportConvTitle = detail.title || 'conversation';
			showExport = true;
		}
	}
</script>

<svelte:window on:keydown={handleKeydown} />

<AppShell onSelect={handleSelect} onCreate={handleCreate} onExport={handleExportFromSidebar}>
	<svelte:fragment slot="header">
		<div class="flex items-center gap-2 flex-1 min-w-0">
			{#if $activeConversation}
				<h1 class="text-sm font-medium text-surface-300 truncate shrink min-w-0">
					{$activeConversation.title}
				</h1>
			{:else}
				<h1 class="text-sm font-medium text-surface-400 shrink-0">Opti-Oignon</h1>
			{/if}

			<!-- Separator -->
			<div class="w-px h-4 shrink-0 hidden sm:block" style="background-color: var(--oo-bd-default);" />

			<!-- Model selector -->
			<div class="shrink-0 hidden sm:block">
				<ModelSelector />
			</div>

			<!-- Export button (header) -->
			{#if $activeConversationId}
				<button
					on:click={() => { exportConvId = $activeConversationId || ''; exportConvTitle = $activeConversation?.title || 'conversation'; showExport = true; }}
					class="p-1.5 rounded-md text-surface-500 hover:text-accent-400 hover:bg-surface-800 transition-colors shrink-0 hidden sm:block"
					title="Export conversation (Ctrl+Shift+E)"
					aria-label="Export conversation"
				>
					<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
					</svg>
				</button>
			{/if}
		</div>
	</svelte:fragment>

	<!-- Panel toggle buttons -->
	<svelte:fragment slot="panel-toggle">
		<PanelToggle />
	</svelte:fragment>

	<!-- Preset bar -->
	<svelte:fragment slot="subheader">
		<div class="px-4 py-1.5 overflow-x-auto" style="border-bottom: 1px solid var(--oo-bd-subtle);">
			<div class="max-w-2xl mx-auto">
				<PresetSelector />
			</div>
		</div>
		<ContextBar />
	</svelte:fragment>

	<ErrorBoundary fallbackMessage="Chat failed to render">
		<slot />
	</ErrorBoundary>

	<!-- Right panel content -->
	<svelte:fragment slot="panel">
		{#if $activePanel === 'artifacts'}
			<ArtifactPanel />
		{:else if $activePanel === 'code'}
			<CodePanel />
		{:else if $activePanel === 'memory'}
			<MemoryPanel />
		{:else if $activePanel === 'pipelines'}
			<PipelinePanel />
		{:else if $activePanel === 'exec-pipelines'}
			<ExecPipelinePanel />
		{:else if $activePanel === 'context'}
			<ContextPanel />
		{:else if $activePanel === 'plugins'}
			<PluginsQuickPanel />
		{:else if $activePanel === 'agent'}
			<AgentPanel />
		{:else if $activePanel === 'sandbox'}
			<SandboxPanel />
		{/if}
	</svelte:fragment>
</AppShell>

<!-- Export dialog modal -->
{#if showExport && exportConvId}
	<ExportDialog
		conversationId={exportConvId}
		conversationTitle={exportConvTitle}
		on:close={() => (showExport = false)}
	/>
{/if}

<!-- Pending tool-call approvals (drawer-right) -->
<ToolCallApprovalDrawer />

