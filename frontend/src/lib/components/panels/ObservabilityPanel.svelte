<!--
  ObservabilityPanel.svelte
  The inference pipeline, in one group of the Observability page: an
  overview, then the dashboards the catalog embeds in this group
  (lib/settings/catalog.ts, embeddedGroups('observability')): telemetry,
  its history, the profiler and performance. Each is a tab of one row of
  the design system's tabs, labelled with its group's title, drawn from a
  map keyed by the catalog's panel names. A tab whose feature key is off in
  the health feature map says the feature is unavailable
  (lib/observability/state.ts decides, tab by tab).

  The overview says, in words (lib/observability/state.ts), whether
  telemetry is collecting, how much the profiler has seen and how much
  history is kept, each with the way to its tab. A model picked in the
  profiler opens the history filtered by it, with a way to clear the
  filter: clearing it draws the history afresh, unfiltered.

  A control here that changes the tab (an overview's "Open", a model
  picked, the filter cleared) is gone once it has, so it hands focus to
  the selected tab: focus never falls back to the page. The tab and the
  feature map are props, set by the component itself once mounted, so each
  tab can be drawn on the server.

  The group's title is drawn by the group; this panel draws none. The
  overview sits on the sunken ground, bounded by the edge, never on the
  group's own tone.
-->
<script lang="ts">
	import { onMount, type ComponentType } from 'svelte';
	import { Tabs, TextButton } from '$lib/ds';
	import type { TabItem } from '$lib/ds';
	import PanelHeader from '$lib/ds/PanelHeader.svelte';
	import FeatureUnavailable from '$lib/components/ui/FeatureUnavailable.svelte';
	import TelemetryDashboard from './TelemetryDashboard.svelte';
	import TelemetryHistoryPanel from './TelemetryHistoryPanel.svelte';
	import ProfilerDashboard from './ProfilerDashboard.svelte';
	import PerformanceDashboard from './PerformanceDashboard.svelte';
	import { embeddedGroups, type SettingsGroupEntry } from '$lib/settings/catalog';
	import { getFeatureMap } from '$lib/api/featureCheck';
	import { getHistoryStats, getTelemetryStats, type TelemetryHistoryStats, type TelemetryStats } from '$lib/api/telemetry';
	import { getProfilerSummary, type ProfilerSummaryResponse } from '$lib/api/profiler';
	import { historyState, profilerState, tabUnavailable, telemetryState } from '$lib/observability/state';

	/** The dashboards, by the catalog's panel names. */
	const PANELS: Record<string, ComponentType> = {
		TelemetryDashboard,
		TelemetryHistoryPanel,
		ProfilerDashboard,
		PerformanceDashboard
	};

	const OVERVIEW = 'overview';
	const EMBEDDED: SettingsGroupEntry[] = embeddedGroups('observability');
	const tabs: TabItem[] = [
		{ id: OVERVIEW, label: 'Overview' },
		...EMBEDDED.map((group) => ({ id: group.id, label: group.title }))
	];
	const byPanel = (panel: string) => EMBEDDED.find((group) => group.panel === panel);
	const telemetryGroup = byPanel('TelemetryDashboard');
	const historyGroup = byPanel('TelemetryHistoryPanel');
	const profilerGroup = byPanel('ProfilerDashboard');

	/** The tab shown. */
	export let active = OVERVIEW;
	/** The health feature map, read once mounted. */
	export let featureMap: Record<string, boolean> = {};

	$: group = EMBEDDED.find((entry) => entry.id === active);
	$: shut = !!group && tabUnavailable(group.feature, featureMap);

	let tabsEl: Tabs | undefined;

	let loading = true;
	let telemetry: TelemetryStats | null = null;
	let profiler: ProfilerSummaryResponse | null = null;
	let history: TelemetryHistoryStats | null = null;

	/** The model picked in the profiler, which filters the history. */
	let linkedModel = '';

	$: blocks = [
		{ entry: telemetryGroup, state: telemetryState(telemetry) },
		{ entry: profilerGroup, state: profilerState(profiler) },
		{ entry: historyGroup, state: historyState(history) }
	].filter((block): block is { entry: SettingsGroupEntry; state: string } => !!block.entry);

	async function loadOverview() {
		loading = true;
		const [stats, summary, kept] = await Promise.all([
			getTelemetryStats().catch(() => null),
			getProfilerSummary().catch(() => null),
			getHistoryStats().catch(() => null)
		]);
		telemetry = stats;
		profiler = summary;
		history = kept;
		loading = false;
	}

	function openTab(id: string) {
		active = id;
		tabsEl?.focusSelected();
	}

	function pickModel(event: CustomEvent<string>) {
		linkedModel = event.detail;
		if (linkedModel && historyGroup) {
			active = historyGroup.id;
			tabsEl?.focusSelected();
		}
	}

	function clearLinkedModel() {
		linkedModel = '';
		tabsEl?.focusSelected();
	}

	onMount(async () => {
		loadOverview();
		try {
			featureMap = await getFeatureMap();
		} catch {
			// The health module may be unavailable; the tabs still render.
		}
	});
</script>

<div class="oo-obs">
	<Tabs bind:this={tabsEl} bind:value={active} {tabs} variant="underline" size="sm">
		{#if active === OVERVIEW}
			<div class="oo-obs-overview">
				{#each blocks as block (block.entry.id)}
					<div class="oo-obs-block">
						<PanelHeader title={block.entry.title} level={3} />
						<div class="oo-obs-block-body">
							<p class="oo-obs-state">{loading ? 'Reading' : block.state}</p>
							{#if block.entry === telemetryGroup && telemetry}
								<p class="oo-obs-figures">
									{telemetry.total_requests.toLocaleString()} requests,
									{telemetry.total_tokens.toLocaleString()} tokens
								</p>
							{:else if block.entry === profilerGroup && profiler && profiler.models.length > 0}
								<p class="oo-obs-figures">
									{profiler.models.length}
									{profiler.models.length === 1 ? 'model' : 'models'} seen
								</p>
							{:else if block.entry === historyGroup && history && history.available}
								<p class="oo-obs-figures">Kept for {history.retention_days} days</p>
							{/if}
							<TextButton on:click={() => openTab(block.entry.id)}>Open {block.entry.title}</TextButton>
						</div>
					</div>
				{/each}
			</div>
		{:else if group && shut}
			<FeatureUnavailable featureName={group.title} />
		{:else if group && group === historyGroup}
			{#if linkedModel}
				<p class="oo-obs-filter">
					<span>Filtered by model: <strong>{linkedModel}</strong></span>
					<TextButton on:click={clearLinkedModel}>Clear filter</TextButton>
				</p>
			{/if}
			{#key linkedModel}
				<TelemetryHistoryPanel initialModelFilter={linkedModel} />
			{/key}
		{:else if group && group === profilerGroup}
			<ProfilerDashboard on:selectModel={pickModel} />
		{:else if group && PANELS[group.panel]}
			<svelte:component this={PANELS[group.panel]} />
		{/if}
	</Tabs>
</div>

<style>
	.oo-obs {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
		min-width: 0;
	}

	.oo-obs-overview {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(14rem, 100%), 1fr));
		gap: var(--oo-space-3);
	}

	/* The overview sits on the sunken ground, bounded by the edge. */
	.oo-obs-block {
		display: flex;
		flex-direction: column;
		background-color: var(--oo-bg-subtle);
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-md);
		--oo-panel-header-radius: var(--oo-radius-md);
	}
	.oo-obs-block-body {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--oo-space-2);
		padding: 0 var(--oo-space-5) var(--oo-space-4);
	}
	.oo-obs-block-body :global(.oo-text-btn) {
		margin-left: calc(-1 * var(--oo-space-3));
	}

	.oo-obs-state {
		margin: 0;
		color: var(--oo-fg-primary);
		font-size: var(--oo-text-sm);
	}
	.oo-obs-figures {
		margin: 0;
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-xs);
		font-variant-numeric: tabular-nums;
	}

	.oo-obs-filter {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--oo-space-2) var(--oo-space-3);
		margin: 0 0 var(--oo-space-3);
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
	}
</style>
