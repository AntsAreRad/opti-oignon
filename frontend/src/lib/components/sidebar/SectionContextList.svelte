<!--
  SectionContextList.svelte
  Drives the lower "Section Context" half of the sidebar (spec 8.3),
  switching its contents by route: chat shows the new-conversation action
  and the conversation list; settings shows the settings sections; the
  other sections show their own context entry. Replaces the previous
  one-size-fits-all sidebar body.
-->
<script lang="ts">
	import { page } from '$app/stores';
	import Icon from '$lib/ds/Icon.svelte';
	import ConversationList from '$lib/components/sidebar/ConversationList.svelte';
	import NewConversationButton from '$lib/components/sidebar/NewConversationButton.svelte';
	import { projects, loadProjects } from '$lib/stores/projects';
	import { getHistory } from '$lib/api/benchmarkV2';
	import type { ProjectInfo, BenchmarkV2HistoryEntry } from '$lib/types';

	export let onSelect: (id: string) => void = () => {};
	export let onCreate: () => void = () => {};
	export let onExport: (id: string, title: string) => void = () => {};

	$: path = $page.url?.pathname ?? '/chat';
	$: section = path.startsWith('/projects')
		? 'projects'
		: path.startsWith('/settings')
			? 'settings'
			: path.startsWith('/benchmark')
				? 'benchmark'
				: path.startsWith('/health')
					? 'health'
					: 'chat';

	// Legacy ?tab= ids fold into the nine-section model so an old sidebar
	// link still highlights the right section.
	const LEGACY_TAB_TO_SECTION: Record<string, string> = {
		quick: 'conversation',
		presets: 'conversation',
		prompt: 'conversation',
		models: 'models',
		analytics: 'performance',
		performance: 'performance',
		'fine-tune': 'data',
		knowledge: 'knowledge',
		plugins: 'plugins',
		backup: 'data',
		security: 'account',
		advanced: 'performance'
	};

	// The nine consolidated settings sections (spec 5.5).
	const settingsSections: { id: string; label: string; icon: string }[] = [
		{ id: 'appearance', label: 'Appearance', icon: 'palette' },
		{ id: 'account', label: 'Account & Security', icon: 'shield-check' },
		{ id: 'conversation', label: 'Conversation & Chat', icon: 'messages-square' },
		{ id: 'models', label: 'Models & Inference', icon: 'cpu' },
		{ id: 'knowledge', label: 'Knowledge (RAG)', icon: 'book-open' },
		{ id: 'plugins', label: 'Plugins & Extensions', icon: 'plug' },
		{ id: 'performance', label: 'Performance & Telemetry', icon: 'activity' },
		{ id: 'network', label: 'Network & Privacy', icon: 'globe' },
		{ id: 'data', label: 'Backup & Data', icon: 'database' }
	];

	$: rawSection = $page.url?.searchParams.get('section') ?? $page.url?.searchParams.get('tab');
	$: currentTab = rawSection
		? settingsSections.some((s) => s.id === rawSection)
			? rawSection
			: (LEGACY_TAB_TO_SECTION[rawSection] ?? 'appearance')
		: 'appearance';

	// --- Projects section context: search + Starred/All/Archived ---
	let projectsLoaded = false;
	let projectQuery = '';

	// Load the project list the first time the Projects section is shown.
	$: if (section === 'projects' && !projectsLoaded) {
		projectsLoaded = true;
		loadProjects();
	}

	$: activeProjectId = $page.params.id ?? '';

	function isStarred(p: ProjectInfo): boolean {
		return Boolean((p.settings || {}).starred);
	}
	function isArchived(p: ProjectInfo): boolean {
		return Boolean((p.settings || {}).archived);
	}

	$: projectMatches = $projects.filter((p) =>
		projectQuery.trim() ? p.name.toLowerCase().includes(projectQuery.trim().toLowerCase()) : true
	);
	$: starredProjects = projectMatches.filter((p) => isStarred(p) && !isArchived(p));
	$: activeProjects = projectMatches.filter((p) => !isStarred(p) && !isArchived(p));
	$: archivedProjects = projectMatches.filter((p) => isArchived(p));

	// --- Benchmark section context: search + runs grouped by recency ---
	let benchmarkRunsLoaded = false;
	let benchmarkRuns: BenchmarkV2HistoryEntry[] = [];
	let runQuery = '';

	// Load recent runs the first time the Benchmark section is shown.
	$: if (section === 'benchmark' && !benchmarkRunsLoaded) {
		benchmarkRunsLoaded = true;
		loadBenchmarkRuns();
	}

	async function loadBenchmarkRuns() {
		try {
			const data = await getHistory(50);
			benchmarkRuns = data.runs;
		} catch {
			benchmarkRuns = [];
		}
	}

	const DAY_SEC = 86400;
	$: activeRunId = $page.url?.searchParams.get('run') ?? '';
	$: todayStartSec = (() => {
		const d = new Date();
		d.setHours(0, 0, 0, 0);
		return Math.floor(d.getTime() / 1000);
	})();
	$: weekStartSec = todayStartSec - 6 * DAY_SEC;
	$: runMatches = benchmarkRuns.filter((r) =>
		runQuery.trim()
			? `${r.run_id} ${r.profile} ${r.models.join(' ')}`.toLowerCase().includes(runQuery.trim().toLowerCase())
			: true
	);
	$: runsToday = runMatches.filter((r) => r.started_at >= todayStartSec);
	$: runsThisWeek = runMatches.filter((r) => r.started_at < todayStartSec && r.started_at >= weekStartSec);
	$: runsAllTime = runMatches.filter((r) => r.started_at < weekStartSec);
</script>

{#if section === 'chat'}
	<div class="oo-sc-chat">
		<div class="oo-sc-action">
			<NewConversationButton on:create={() => onCreate()} />
		</div>
		<div class="oo-sc-list">
			<ConversationList {onSelect} {onExport} onNewConversation={onCreate} />
		</div>
	</div>
{:else if section === 'settings'}
	<nav class="oo-sc-nav" aria-label="Settings sections">
		{#each settingsSections as s (s.id)}
			<a
				class="oo-sc-link"
				class:oo-sc-link-active={currentTab === s.id}
				href={`/settings?tab=${s.id}`}
				aria-current={currentTab === s.id ? 'page' : undefined}
			>
				<Icon name={s.icon} size="sm" />
				<span>{s.label}</span>
			</a>
		{/each}
	</nav>
{:else if section === 'projects'}
	<div class="oo-sc-chat">
		<div class="oo-sc-action oo-sc-action-stack">
			<a class="oo-sc-new" href="/projects?new=1">
				<Icon name="plus" size="sm" />
				<span>New project</span>
			</a>
			<div class="oo-sc-search">
				<Icon name="search" size="sm" />
				<input
					type="text"
					placeholder="Filter projects"
					aria-label="Filter projects"
					bind:value={projectQuery}
				/>
			</div>
		</div>
		<nav class="oo-sc-nav oo-sc-list" aria-label="Projects">
			<a
				class="oo-sc-link"
				class:oo-sc-link-active={!activeProjectId}
				href="/projects"
				aria-current={!activeProjectId ? 'page' : undefined}
			>
				<Icon name="folder" size="sm" />
				<span>All projects</span>
			</a>

			{#if starredProjects.length > 0}
				<p class="oo-sc-group">Starred</p>
				{#each starredProjects as p (p.id)}
					<a
						class="oo-sc-link"
						class:oo-sc-link-active={activeProjectId === p.id}
						href={`/projects/${p.id}`}
						aria-current={activeProjectId === p.id ? 'page' : undefined}
					>
						<Icon name="star" size="sm" />
						<span class="truncate">{p.name}</span>
					</a>
				{/each}
			{/if}

			<p class="oo-sc-group">All</p>
			{#if activeProjects.length === 0 && starredProjects.length === 0}
				<p class="oo-sc-empty">No projects</p>
			{:else}
				{#each activeProjects as p (p.id)}
					<a
						class="oo-sc-link"
						class:oo-sc-link-active={activeProjectId === p.id}
						href={`/projects/${p.id}`}
						aria-current={activeProjectId === p.id ? 'page' : undefined}
					>
						<Icon name="folder" size="sm" />
						<span class="truncate">{p.name}</span>
					</a>
				{/each}
			{/if}

			{#if archivedProjects.length > 0}
				<p class="oo-sc-group">Archived ({archivedProjects.length})</p>
				{#each archivedProjects as p (p.id)}
					<a
						class="oo-sc-link oo-sc-link-muted"
						class:oo-sc-link-active={activeProjectId === p.id}
						href={`/projects/${p.id}`}
						aria-current={activeProjectId === p.id ? 'page' : undefined}
					>
						<Icon name="archive" size="sm" />
						<span class="truncate">{p.name}</span>
					</a>
				{/each}
			{/if}
		</nav>
	</div>
{:else if section === 'benchmark'}
	<div class="oo-sc-chat">
		<div class="oo-sc-action oo-sc-action-stack">
			<a class="oo-sc-new" href="/benchmark">
				<Icon name="plus" size="sm" />
				<span>New run</span>
			</a>
			<div class="oo-sc-search">
				<Icon name="search" size="sm" />
				<input
					type="text"
					placeholder="Filter runs"
					aria-label="Filter runs"
					bind:value={runQuery}
				/>
			</div>
		</div>
		<nav class="oo-sc-nav oo-sc-list" aria-label="Benchmark runs">
			<a
				class="oo-sc-link"
				class:oo-sc-link-active={!activeRunId}
				href="/benchmark"
				aria-current={!activeRunId ? 'page' : undefined}
			>
				<Icon name="gauge" size="sm" />
				<span>Dashboard</span>
			</a>

			{#if runsToday.length > 0}
				<p class="oo-sc-group">Today</p>
				{#each runsToday as r (r.run_id)}
					<a
						class="oo-sc-link"
						class:oo-sc-link-active={activeRunId === r.run_id}
						href={`/benchmark?run=${r.run_id}`}
						aria-current={activeRunId === r.run_id ? 'page' : undefined}
					>
						<Icon name="activity" size="sm" />
						<span class="truncate">{r.profile} ({r.models.length})</span>
					</a>
				{/each}
			{/if}

			{#if runsThisWeek.length > 0}
				<p class="oo-sc-group">This week</p>
				{#each runsThisWeek as r (r.run_id)}
					<a
						class="oo-sc-link"
						class:oo-sc-link-active={activeRunId === r.run_id}
						href={`/benchmark?run=${r.run_id}`}
						aria-current={activeRunId === r.run_id ? 'page' : undefined}
					>
						<Icon name="activity" size="sm" />
						<span class="truncate">{r.profile} ({r.models.length})</span>
					</a>
				{/each}
			{/if}

			{#if runsAllTime.length > 0}
				<p class="oo-sc-group">All time</p>
				{#each runsAllTime as r (r.run_id)}
					<a
						class="oo-sc-link"
						class:oo-sc-link-active={activeRunId === r.run_id}
						href={`/benchmark?run=${r.run_id}`}
						aria-current={activeRunId === r.run_id ? 'page' : undefined}
					>
						<Icon name="activity" size="sm" />
						<span class="truncate">{r.profile} ({r.models.length})</span>
					</a>
				{/each}
			{/if}

			{#if benchmarkRunsLoaded && runMatches.length === 0}
				<p class="oo-sc-empty">No runs yet</p>
			{/if}
		</nav>
	</div>
{:else if section === 'health'}
	<nav class="oo-sc-nav" aria-label="System Status">
		<a class="oo-sc-link" href="/health">
			<Icon name="heart-pulse" size="sm" />
			<span>System Status</span>
		</a>
	</nav>
{/if}

<style>
	.oo-sc-chat {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
	}
	.oo-sc-action {
		padding: var(--oo-space-2) var(--oo-space-3) var(--oo-space-1);
	}
	.oo-sc-list {
		flex: 1;
		min-height: 0;
	}
	.oo-sc-nav {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: var(--oo-space-2);
		overflow-y: auto;
	}
	.oo-sc-link {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		min-height: 40px;
		padding: var(--oo-space-2) var(--oo-space-3);
		border-radius: var(--oo-radius-md);
		color: var(--oo-fg-tertiary);
		font-size: var(--oo-text-sm);
		transition:
			background-color 0.12s ease,
			color 0.12s ease;
	}
	.oo-sc-link:hover {
		background-color: var(--oo-bg-surface);
		color: var(--oo-fg-secondary);
	}
	.oo-sc-link-active {
		background-color: var(--oo-accent-bg);
		color: var(--oo-accent);
	}
	.oo-sc-action-stack {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}
	.oo-sc-new {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		min-height: 36px;
		padding: var(--oo-space-2) var(--oo-space-3);
		border-radius: var(--oo-radius-md);
		border: 1px dashed var(--oo-bd-default);
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
		transition: background-color 0.12s ease, color 0.12s ease;
	}
	.oo-sc-new:hover {
		background-color: var(--oo-bg-surface);
		color: var(--oo-fg-primary);
	}
	.oo-sc-search {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		padding: 0 var(--oo-space-3);
		border-radius: var(--oo-radius-md);
		border: 1px solid var(--oo-bd-default);
		color: var(--oo-fg-muted);
	}
	.oo-sc-search input {
		flex: 1;
		min-width: 0;
		background: transparent;
		border: none;
		outline: none;
		padding: var(--oo-space-2) 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-primary);
	}
	.oo-sc-search input::placeholder {
		color: var(--oo-fg-muted);
	}
	.oo-sc-list {
		flex: 1;
		min-height: 0;
	}
	.oo-sc-group {
		margin: var(--oo-space-2) var(--oo-space-3) 2px;
		font-size: var(--oo-text-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--oo-fg-faint);
	}
	.oo-sc-empty {
		padding: var(--oo-space-2) var(--oo-space-3);
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-faint);
	}
	.oo-sc-link-muted {
		color: var(--oo-fg-muted);
	}
</style>
