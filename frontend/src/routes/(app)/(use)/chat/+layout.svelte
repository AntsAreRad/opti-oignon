<!--
  The chat frame: the conversation's title with the model, export and the
  panel toggles; the preset bar and the context bar; the conversation; and
  the side panels (artifacts, code, memory, pipelines, context, plugins,
  agent, sandbox) in the side panel primitive, beside the conversation on a
  desktop and over its edge on a phone. The session's status line sits at
  its foot. The shell around it (the sidebar, Stop all, the approvals
  drawer, the export dialog) belongs to both spaces.

  All of it frames a conversation: the chats index (/chat itself) is drawn
  alone, with its own heading, and no conversation's bars around it.

  On a phone a side panel stands over this frame, below the shell's header
  (so the stop stays in sight), and holds a named control that closes it:
  five of the nine panels have none of their own.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import IconButton from '$lib/ds/IconButton.svelte';
	import SidePanel from '$lib/ds/SidePanel.svelte';
	import ModelSelector from '$lib/components/chat/ModelSelector.svelte';
	import PresetSelector from '$lib/components/chat/PresetSelector.svelte';
	import ContextBar from '$lib/components/chat/ContextBar.svelte';
	import ContextPanel from '$lib/components/chat/ContextPanel.svelte';
	import PanelToggle from '$lib/components/panels/PanelToggle.svelte';
	import ArtifactPanel from '$lib/components/panels/ArtifactPanel.svelte';
	import CodePanel from '$lib/components/panels/CodePanel.svelte';
	import MemoryPanel from '$lib/components/panels/MemoryPanel.svelte';
	import PipelinePanel from '$lib/components/panels/PipelinePanel.svelte';
	import ExecPipelinePanel from '$lib/components/panels/ExecPipelinePanel.svelte';
	import PluginsQuickPanel from '$lib/components/panels/PluginsQuickPanel.svelte';
	import AgentPanel from '$lib/components/panels/AgentPanel.svelte';
	import SandboxPanel from '$lib/components/panels/SandboxPanel.svelte';
	import StatusFooter from '$lib/components/layout/StatusFooter.svelte';
	import {
		activeConversationId,
		activeConversation,
		selectConversation,
		loadConversations
	} from '$lib/stores/conversations';
	import { loadOptions } from '$lib/stores/chatOptions';
	import {
		activePanel,
		closePanel,
		isPanelOpen,
		panelWidth,
		setPanelWidth,
		PANEL_MIN_WIDTH,
		PANEL_MAX_WIDTH
	} from '$lib/stores/panels';
	import { isPhone } from '$lib/stores/ui';
	import { openExportDialog } from '$lib/stores/exportDialog';

	// The route names the conversation; the store follows it.
	$: routeId = $page.params?.id ?? null;
	$: if (routeId && routeId !== $activeConversationId) {
		selectConversation(routeId);
	}

	function exportConversation() {
		openExportDialog($activeConversationId ?? '', $activeConversation?.title ?? 'conversation');
	}

	// Escape closes an open side panel.
	function closeOnEscape(event: KeyboardEvent) {
		if (event.key === 'Escape' && $activePanel !== 'none') closePanel();
	}

	onMount(() => {
		loadConversations();
		loadOptions();
	});
</script>

<svelte:window on:keydown={closeOnEscape} />

<div class="oo-chat-frame">
	{#if routeId}
		<div class="oo-chat-head">
			<h1 class="oo-chat-title">{$activeConversation?.title ?? 'Chats'}</h1>
			<div class="oo-chat-model">
				<ModelSelector />
			</div>
			{#if $activeConversationId}
				<IconButton icon="download" label="Export the conversation" on:click={exportConversation} />
			{/if}
			<PanelToggle />
		</div>

		<div class="oo-chat-presets">
			<PresetSelector />
		</div>
		<ContextBar on:openProject={(e) => goto('/projects/' + encodeURIComponent(e.detail))} />
	{/if}

	<div class="oo-chat-split">
		<div class="oo-chat-page">
			<slot />
		</div>
		{#if routeId && $isPanelOpen}
			<SidePanel
				label="Side panel"
				width={$panelWidth}
				min={PANEL_MIN_WIDTH}
				max={PANEL_MAX_WIDTH}
				overlay={$isPhone}
				on:resize={(event) => setPanelWidth(event.detail)}
			>
				{#if $isPhone}
					<div class="oo-chat-panel-close">
						<IconButton icon="x" size="lg" label="Close panel" on:click={closePanel} />
					</div>
				{/if}
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
			</SidePanel>
		{/if}
	</div>

	{#if routeId}
		<StatusFooter />
	{/if}
</div>

<style>
	/* The frame a side panel stands over on a phone. */
	.oo-chat-frame {
		position: relative;
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
	}

	.oo-chat-head {
		display: flex;
		flex-shrink: 0;
		align-items: center;
		gap: var(--oo-space-3);
		min-height: 52px;
		padding: var(--oo-space-2) var(--oo-space-4) var(--oo-space-2) var(--oo-space-6);
	}
	.oo-chat-title {
		flex: 1;
		min-width: 0;
		margin: 0;
		overflow: hidden;
		color: var(--oo-fg-primary);
		font-family: var(--oo-font-serif);
		font-size: var(--oo-text-lg);
		font-weight: 600;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.oo-chat-model {
		flex-shrink: 0;
	}

	.oo-chat-panel-close {
		display: flex;
		justify-content: flex-end;
		padding: var(--oo-space-2) var(--oo-space-2) 0;
	}

	.oo-chat-presets {
		flex-shrink: 0;
		padding: 0 var(--oo-space-4) var(--oo-space-2);
		overflow-x: auto;
	}

	.oo-chat-split {
		display: flex;
		flex: 1;
		min-height: 0;
		overflow: hidden;
	}
	.oo-chat-page {
		position: relative;
		flex: 1;
		min-width: 0;
		overflow: hidden;
	}

	@media (max-width: 639.98px) {
		.oo-chat-model {
			display: none;
		}
		.oo-chat-head {
			padding-left: var(--oo-space-4);
		}
	}
</style>
