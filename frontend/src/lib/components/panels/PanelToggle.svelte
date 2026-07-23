<!--
  PanelToggle.svelte
  Boutons de toggle pour les panneaux lateraux (artifacts, code, memory).
  Affiche dans le header, avec indicateur d'etat actif.
  Warm accent active states.
-->
<script lang="ts">
	import { activePanel, togglePanel } from '$lib/stores/panels';
</script>

<div class="flex items-center gap-1">
	<!-- Artifact panel toggle -->
	<button
		on:click={() => togglePanel('artifacts')}
		class="p-1.5 rounded-md transition-colors"
		style="{$activePanel === 'artifacts'
			? 'color: var(--oo-acc-400); background-color: var(--oo-warning-bg);'
			: 'color: var(--oo-fg-tertiary);'}"
		title="Toggle artifacts panel"
		aria-label="Toggle artifacts panel"
		aria-pressed={$activePanel === 'artifacts'}
	>
		<!-- Document/artifact icon -->
		<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
			<path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
		</svg>
	</button>

	<!-- Code panel toggle -->
	<button
		on:click={() => togglePanel('code')}
		class="p-1.5 rounded-md transition-colors"
		style="{$activePanel === 'code'
			? 'color: var(--oo-acc-400); background-color: var(--oo-warning-bg);'
			: 'color: var(--oo-fg-tertiary);'}"
		title="Toggle code execution panel"
		aria-label="Toggle code execution panel"
		aria-pressed={$activePanel === 'code'}
	>
		<!-- Code/terminal icon -->
		<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
			<path d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
		</svg>
	</button>

	<!-- Memory panel toggle -->
	<button
		on:click={() => togglePanel('memory')}
		class="p-1.5 rounded-md transition-colors"
		style="{$activePanel === 'memory'
			? 'color: var(--oo-acc-400); background-color: var(--oo-warning-bg);'
			: 'color: var(--oo-fg-tertiary);'}"
		title="Toggle memory panel"
		aria-label="Toggle memory panel"
		aria-pressed={$activePanel === 'memory'}
	>
		<!-- Brain/memory icon -->
		<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
			<path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
		</svg>
	</button>

	<!-- Pipeline panel toggle -->
	<button
		on:click={() => togglePanel('pipelines')}
		class="p-1.5 rounded-md transition-colors"
		style="{$activePanel === 'pipelines'
			? 'color: var(--oo-acc-400); background-color: var(--oo-warning-bg);'
			: 'color: var(--oo-fg-tertiary);'}"
		title="Toggle pipelines panel"
		aria-label="Toggle pipelines panel"
		aria-pressed={$activePanel === 'pipelines'}
	>
		<!-- Pipeline/workflow icon -->
		<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
			<path d="M4 6h16M4 12h8m-8 6h16M15 11l2 1-2 1m0-2v2" />
		</svg>
	</button>

	<!-- Context panel toggle -->
	<button
		on:click={() => togglePanel('context')}
		class="p-1.5 rounded-md transition-colors"
		style="{$activePanel === 'context'
			? 'color: var(--oo-acc-400); background-color: var(--oo-warning-bg);'
			: 'color: var(--oo-fg-tertiary);'}"
		title="Toggle context health panel"
		aria-label="Toggle context health panel"
		aria-pressed={$activePanel === 'context'}
	>
		<!-- Bar chart / metrics icon -->
		<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
			<path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
		</svg>
	</button>

	<!-- Exec Pipeline panel toggle -->
	<button
		on:click={() => togglePanel('exec-pipelines')}
		class="p-1.5 rounded-md transition-colors"
		style="{$activePanel === 'exec-pipelines'
			? 'color: var(--oo-acc-400); background-color: var(--oo-warning-bg);'
			: 'color: var(--oo-fg-tertiary);'}"
		title="Execution Pipelines"
		aria-label="Toggle execution pipelines panel"
	>
		<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8">
			<path d="M13 10V3L4 14h7v7l9-11h-7z" />
		</svg>
	</button>

	<!-- Plugins panel toggle -->
	<button
		on:click={() => togglePanel('plugins')}
		class="p-1.5 rounded-md transition-colors"
		style="{$activePanel === 'plugins'
			? 'color: var(--oo-acc-400); background-color: var(--oo-warning-bg);'
			: 'color: var(--oo-fg-tertiary);'}"
		title="Toggle plugins panel"
		aria-label="Toggle plugins panel"
		aria-pressed={$activePanel === 'plugins'}
	>
		<!-- Puzzle piece / plugin icon -->
		<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
			<path d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
		</svg>
	</button>

	<!-- Agent panel toggle (FRD-03: chat-side runtime home) -->
	<button
		on:click={() => togglePanel('agent')}
		class="p-1.5 rounded-md transition-colors"
		style="{$activePanel === 'agent'
			? 'color: var(--oo-acc-400); background-color: var(--oo-warning-bg);'
			: 'color: var(--oo-fg-tertiary);'}"
		title="Toggle agent run panel"
		aria-label="Toggle agent run panel"
		aria-pressed={$activePanel === 'agent'}
	>
		<!-- Bot / agent icon -->
		<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
			<path d="M12 8V4H8m8 4h2a2 2 0 012 2v6a2 2 0 01-2 2H6a2 2 0 01-2-2v-6a2 2 0 012-2h2m8 0H8m1 4h.01M15 12h.01" />
		</svg>
	</button>

	<!-- Sandbox workspace panel toggle (Sandbox Workspace cycle Bloc 1) -->
	<button
		on:click={() => togglePanel('sandbox')}
		class="p-1.5 rounded-md transition-colors"
		style="{$activePanel === 'sandbox'
			? 'color: var(--oo-acc-400); background-color: var(--oo-warning-bg);'
			: 'color: var(--oo-fg-tertiary);'}"
		title="Toggle sandbox workspaces panel"
		aria-label="Toggle sandbox workspaces panel"
		aria-pressed={$activePanel === 'sandbox'}
	>
		<!-- Box / workspace icon -->
		<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
			<path d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
		</svg>
	</button>
</div>
