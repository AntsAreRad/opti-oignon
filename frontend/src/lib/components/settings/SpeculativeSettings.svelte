<!--
  SpeculativeSettings.svelte
  Merge host for the two speculative-execution panels (spec 11.6 MERGE).

  Opti-Oignon has two distinct speculative mechanisms that were previously two
  separate settings entries:
    - Generation: a draft model proposes tokens that a verify model
      accepts or rejects, with a convergence threshold. Backend-agnostic.
      Mutually exclusive with cascading. -> SpeculativePanel.
    - Decoding: llama.cpp native speculative decoding with a draft
      model, draft_max/draft_min and GPU layer controls, gated on the
      llama.cpp backend. -> SpeculativeDecodingPanel.

  This component keeps both panels fully functional (each owns its own state
  and API) but presents them as one "Speculative" group with a segmented
  control, so the distinction is explicit and the two no longer look like
  duplicate settings.
-->
<script lang="ts">
	import Tabs from '$lib/ds/Tabs.svelte';
	import SpeculativePanel from '$lib/components/panels/SpeculativePanel.svelte';
	import SpeculativeDecodingPanel from '$lib/components/settings/SpeculativeDecodingPanel.svelte';
	import type { TabItem } from '$lib/ds/types';

	const tabs: TabItem[] = [
		{ id: 'generation', label: 'Generation (draft / verify)' },
		{ id: 'decoding', label: 'Decoding (llama.cpp)' }
	];

	let active = 'generation';
</script>

<div class="oo-spec">
	<p class="oo-spec-intro">
		Two independent ways to speculate ahead. Generation uses a draft model
		verified by a larger model and works with any backend. Decoding is
		llama.cpp's native speculative decoding and requires the llama.cpp
		backend. Enable at most one approach at a time.
	</p>

	<Tabs bind:value={active} {tabs} variant="pill" />

	<div class="oo-spec-panel">
		{#if active === 'generation'}
			<SpeculativePanel />
		{:else if active === 'decoding'}
			<SpeculativeDecodingPanel />
		{/if}
	</div>
</div>

<style>
	.oo-spec {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}

	.oo-spec-intro {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
		line-height: var(--oo-leading-relaxed);
		margin: 0;
	}

	.oo-spec-panel {
		margin-top: var(--oo-space-1);
	}
</style>
