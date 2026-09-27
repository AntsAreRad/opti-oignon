<!--
  SettingsHub.svelte
  The groups of the settings catalog, on the page that holds them.

  Preferences shows the reader's own settings, section by section; each
  Workshop settings page (/workshop/<section>) shows the groups placed on
  it. Where each group lives is the catalog's (lib/settings/catalog.ts):
  its space and section, or retired (never shown), or embedded in the group
  whose panel renders it.

  The search reads the whole catalog, every group's title, description and
  synonyms, the name of the page that holds it and of the old settings
  section it sat in, so a search from Preferences finds a Workshop group,
  and each result links to the page that holds it (lib/settings/search.ts
  decides both). `?q=` lands on a search, `?g=` on one group. Enter opens
  the first result and Escape clears the search; the words go to the
  address as `?q=` once the reader pauses (replacing the entry, so Back
  leaves the page), and after every navigation they are read back from it,
  so another page's results never stay on screen.

  The network page also shows the server's reachability, and the security
  page opens on the grade: its checks, the session mode and the recent
  security events.

  On a Workshop page the groups fold (lib/settings/disclosure.ts decides):
  one group is open at a time, and only the open group loads its panel,
  or a group whose panel was used on this visit, kept mounted and hidden
  until the page is left. The address names the open group (`?g=`): on
  arrival the group it names opens, focused and scrolled into view (an
  embedded group opens its host); with none named, a page's only group
  opens. A toggle writes the address in place, and an address the hub
  wrote itself is never read as an arrival: the hub remembers the address
  it asked for, and a navigation to any other is one. Preferences' groups
  do not fold. While the search shows its results, the page's groups stay
  where they are, hidden, so a draft in a group outlives a search. The loaders
  below are keyed by the catalog's panel names; an embedded group's panel
  is loaded by its host's. The groups a section's introduction renders
  inline are rendered by that introduction, each on its own page. Feature
  availability gates a panel through the health feature map.

  Preferences also holds what the old header held: the palette switcher
  (Appearance), the account menu (Account) and the notification history.
-->
<script lang="ts">
	import { onMount, setContext, tick, type ComponentType } from 'svelte';
	import { writable } from 'svelte/store';
	import { page } from '$app/stores';
	import { goto, afterNavigate } from '$app/navigation';
	import IconButton from '$lib/ds/IconButton.svelte';
	import Icon from '$lib/ds/Icon.svelte';
	import Input from '$lib/ds/Input.svelte';
	import SkeletonLoader from '$lib/components/ui/SkeletonLoader.svelte';
	import FeatureUnavailable from '$lib/components/ui/FeatureUnavailable.svelte';
	import ThemeSwitcher from '$lib/components/ui/ThemeSwitcher.svelte';
	import UserMenu from '$lib/components/ui/UserMenu.svelte';
	import NotificationCenter from '$lib/components/ui/NotificationCenter.svelte';
	import SettingsGroup from '$lib/components/settings/SettingsGroup.svelte';
	import AppearanceSection from '$lib/components/settings/sections/AppearanceSection.svelte';
	import ConversationDefaults from '$lib/components/settings/sections/ConversationDefaults.svelte';
	import AccountAuthMode from '$lib/components/settings/sections/AccountAuthMode.svelte';
	import NetworkReachability from '$lib/components/settings/NetworkReachability.svelte';
	import SecurityGrade from '$lib/components/settings/SecurityGrade.svelte';
	import { getFeatureMap } from '$lib/api/featureCheck';
	import { scrollBehavior } from '$lib/motion';
	import { DESTINATIONS } from '$lib/nav/destinations';
	import { searchSettings, settingsIndex } from '$lib/settings/search';
	import {
		GROUPS_CONTEXT,
		addressFor,
		openOnArrival,
		toggled,
		withEdited,
		type GroupsContext
	} from '$lib/settings/disclosure';
	import {
		SETTINGS_SECTIONS,
		INLINE_GROUPS,
		PREFERENCES_SECTIONS,
		type GroupPlacement,
		type SectionIntro
	} from '$lib/settings/catalog';

	/** The space of the page: Preferences (use) or a Workshop settings page. */
	export let space: 'use' | 'workshop';
	/** The Workshop settings page, by the last segment of its URL. */
	export let section: string | null = null;

	type Loaded = Promise<{ default: ComponentType }>;

	// -- Lazy panel loaders, keyed by the catalog's panel names. ---------------
	const loaders: Record<string, () => Loaded> = {
		PresetManager: () => import('$lib/components/settings/PresetManager.svelte'),
		MemoriesPanel: () => import('$lib/components/panels/MemoriesPanel.svelte'),
		PromptConfigPanel: () => import('$lib/components/panels/PromptConfigPanel.svelte'),
		CompressionSettings: () => import('$lib/components/panels/CompressionSettings.svelte'),
		ContextOptimizerPanel: () => import('$lib/components/settings/ContextOptimizerPanel.svelte'),
		HumanizerPanel: () => import('$lib/components/panels/HumanizerPanel.svelte'),
		ModelHealthWidget: () => import('$lib/components/settings/ModelHealthWidget.svelte'),
		ModelProfilePanel: () => import('$lib/components/panels/ModelProfilePanel.svelte'),
		ModelAssignment: () => import('$lib/components/panels/ModelAssignment.svelte'),
		LearnedRouterPanel: () => import('$lib/components/panels/LearnedRouterPanel.svelte'),
		CascadingPanel: () => import('$lib/components/panels/CascadingPanel.svelte'),
		SpeculativeSettings: () => import('$lib/components/settings/SpeculativeSettings.svelte'),
		VisionModelSelector: () => import('$lib/components/settings/VisionModelSelector.svelte'),
		KnowledgeBasePanel: () => import('$lib/components/settings/KnowledgeBasePanel.svelte'),
		RAGDashboardPanel: () => import('$lib/components/settings/RAGDashboardPanel.svelte'),
		PluginsPanel: () => import('$lib/components/settings/PluginsPanel.svelte'),
		SkillsPanel: () => import('$lib/components/panels/SkillsPanel.svelte'),
		PluginMarketplace: () => import('$lib/components/settings/PluginMarketplace.svelte'),
		PluginAllowlistPanel: () => import('$lib/components/settings/PluginAllowlistPanel.svelte'),
		CacheStatsPanel: () => import('$lib/components/panels/CacheStatsPanel.svelte'),
		GovernorPanel: () => import('$lib/components/panels/GovernorPanel.svelte'),
		ObservabilityPanel: () => import('$lib/components/panels/ObservabilityPanel.svelte'),
		PerformanceTunerPanel: () => import('$lib/components/settings/PerformanceTunerPanel.svelte'),
		AnalyticsDashboard: () => import('$lib/components/panels/AnalyticsDashboard.svelte'),
		ProxySettingsPanel: () => import('$lib/components/panels/ProxySettingsPanel.svelte'),
		SyncPanel: () => import('$lib/components/panels/SyncPanel.svelte'),
		RemoteAccessPanel: () => import('$lib/components/settings/RemoteAccessPanel.svelte'),
		SearchKillSwitchPanel: () => import('$lib/components/settings/SearchKillSwitchPanel.svelte'),
		SecurityModePanel: () => import('$lib/components/settings/SecurityModePanel.svelte'),
		TOTPSetup: () => import('$lib/components/settings/TOTPSetup.svelte'),
		WebAuthnSetup: () => import('$lib/components/settings/WebAuthnSetup.svelte'),
		RecoveryCodesPanel: () => import('$lib/components/settings/RecoveryCodesPanel.svelte'),
		AppPasswordsPanel: () => import('$lib/components/settings/AppPasswordsPanel.svelte'),
		HardeningPanel: () => import('$lib/components/settings/HardeningPanel.svelte'),
		KeyCeremonyPanel: () => import('$lib/components/settings/KeyCeremonyPanel.svelte'),
		AuditChainPanel: () => import('$lib/components/settings/AuditChainPanel.svelte'),
		BackupRestorePanel: () => import('$lib/components/settings/BackupRestorePanel.svelte'),
		FineTunePanel: () => import('$lib/components/settings/FineTunePanel.svelte')
	};

	const resolved: Record<string, ComponentType> = {};
	async function loadPanel(key: string): Promise<ComponentType> {
		if (resolved[key]) return resolved[key];
		const loader = loaders[key];
		if (!loader) throw new Error(`No panel is named ${key}`);
		const module = await loader();
		resolved[key] = module.default;
		return module.default;
	}

	// -- The catalog's groups, each with where it lives and what renders it. --
	interface Group extends GroupPlacement {
		id: string;
		title: string;
		description: string;
		synonyms: string[];
		/** A lazy panel, by its loader's name. */
		panel?: string;
		/** The introduction that renders the group inline. */
		intro?: SectionIntro;
		feature?: string;
	}

	const GROUPS: Group[] = [
		...SETTINGS_SECTIONS.flatMap((s) =>
			s.groups.map((g) => ({ ...g, synonyms: g.synonyms ?? [] }))
		),
		...INLINE_GROUPS.map((g) => ({
			...g,
			intro: SETTINGS_SECTIONS.find((s) => s.id === g.sectionId)?.intro
		}))
	];

	const WORKSHOP_PREFIX = '/workshop/';
	const TITLE_ID = 'oo-hub-title';

	/** Each embedded group's host, so a link to it opens the group that shows it. */
	const HOST_OF: Record<string, string> = Object.fromEntries(
		GROUPS.filter((g) => g.embeddedIn).map((g) => [g.id, g.embeddedIn as string])
	);

	function groupsOf(inSpace: 'use' | 'workshop', inSection: string | null): Group[] {
		return GROUPS.filter((g) => !g.retired && !g.embeddedIn && g.space === inSpace && g.section === inSection);
	}

	/** The introductions a section renders, each with the groups it renders there. */
	function introsOf(groups: Group[]): { intro: SectionIntro; ids: string[] }[] {
		const out: { intro: SectionIntro; ids: string[] }[] = [];
		for (const group of groups) {
			if (!group.intro || group.panel) continue;
			const known = out.find((entry) => entry.intro === group.intro);
			if (known) known.ids.push(group.id);
			else out.push({ intro: group.intro, ids: [group.id] });
		}
		return out;
	}

	$: pageSections =
		space === 'use'
			? PREFERENCES_SECTIONS.map((p) => ({ id: p.id, label: p.label, groups: groupsOf('use', p.id) }))
			: [
					{
						id: section ?? '',
						label: DESTINATIONS.find((d) => d.href === WORKSHOP_PREFIX + section)?.label ?? 'Workshop',
						groups: groupsOf('workshop', section)
					}
				];
	$: title = space === 'use' ? 'Preferences' : pageSections[0].label;
	$: pageGroupIds = pageSections.flatMap((s) => s.groups.map((g) => g.id));

	// -- The groups' context: their level, whether they fold, which is open. --
	const open = writable<string | null>(null);
	const edited = writable<ReadonlySet<string>>(new Set());
	/** The address the hub last asked for itself: reaching it is no arrival. */
	let wrote: string | null = null;
	let hubEl: HTMLElement | undefined;

	function markEdited(id: string) {
		edited.update((set) => withEdited(set, id));
	}

	/** Opens a group, or closes the open one, and writes the address in place. */
	async function toggle(id: string) {
		const next = toggled($open, id);
		open.set(next);
		const target = addressFor($page.url, next);
		if (target !== `${$page.url.pathname}${$page.url.search}`) {
			wrote = target;
			goto(target, { replaceState: true, keepFocus: true, noScroll: true });
		}
		await tick();
		keepRowInView(id);
	}

	/** A group above the toggled one closed: its row comes back into view, at once. */
	function keepRowInView(id: string) {
		if (typeof document === 'undefined') return;
		const row = document.getElementById(`oo-set-${id}`);
		if (!row || !hubEl) return;
		const box = hubEl.getBoundingClientRect();
		const at = row.getBoundingClientRect();
		// No behaviour given: the row comes back at once, never smoothly.
		if (at.top < box.top || at.top > box.bottom) row.scrollIntoView({ block: 'nearest' });
	}

	setContext(GROUPS_CONTEXT, {
		level: space === 'workshop' ? 2 : 3,
		collapsible: space === 'workshop',
		open,
		edited,
		toggle,
		markEdited
	} as GroupsContext);

	// -- Search over the whole catalog, both spaces. ----------------------------
	const INDEX = settingsIndex(GROUPS, DESTINATIONS, PREFERENCES_SECTIONS, SETTINGS_SECTIONS);
	const SYNC_AFTER_MS = 250;

	let words = '';
	let syncTimer: ReturnType<typeof setTimeout> | undefined;
	/** A result is being opened: the address is the navigation's, not the search's. */
	let leaving = false;
	$: query = words.trim();
	$: results = searchSettings(INDEX, words);

	/** Writes the words to the address as ?q=, replacing the entry. */
	function syncQuery() {
		syncTimer = undefined;
		const url = new URL($page.url);
		const q = words.trim();
		if ((url.searchParams.get('q') ?? '') === q) return;
		if (q) url.searchParams.set('q', q);
		else url.searchParams.delete('q');
		const target = `${url.pathname}${url.search}`;
		wrote = target;
		goto(target, { replaceState: true, keepFocus: true, noScroll: true });
	}

	/** Once the reader pauses, the words go to the address. */
	function syncLater(_words: string) {
		if (typeof window === 'undefined') return;
		if (syncTimer) clearTimeout(syncTimer);
		if (leaving) return;
		syncTimer = setTimeout(syncQuery, SYNC_AFTER_MS);
	}
	$: syncLater(words);

	function clearSearch() {
		if (syncTimer) clearTimeout(syncTimer);
		words = '';
		syncQuery();
	}

	/** Enter opens the first result; Escape clears the search. */
	function onSearchKey(event: KeyboardEvent) {
		if (event.key === 'Enter' && results.length > 0) {
			event.preventDefault();
			leaving = true;
			goto(results[0].href);
			words = '';
		} else if (event.key === 'Escape' && words) {
			event.preventDefault();
			clearSearch();
		}
	}

	// -- Feature gates. -----------------------------------------------------------
	let featureMap: Record<string, boolean> = {};
	function featureOk(group: Group): boolean {
		return !group.feature || featureMap[group.feature] !== false;
	}

	/** The opened group comes into view, its title focused without a scroll of its own. */
	async function revealGroup(id: string) {
		if (typeof document === 'undefined') return;
		await tick();
		requestAnimationFrame(() => {
			const element = document.getElementById(`oo-set-${id}`);
			if (!element) return;
			const button = element.querySelector<HTMLElement>('button[aria-expanded][aria-controls]');
			button?.focus({ preventScroll: true });
			element.scrollIntoView({ behavior: scrollBehavior(), block: 'start' });
		});
	}

	/** A page reached: the group the address names opens, and comes into view. */
	function arrive(url: URL | null) {
		const target = openOnArrival(url?.searchParams.get('g') ?? null, pageGroupIds, HOST_OF);
		if (space === 'workshop') open.set(target);
		if (target && url?.searchParams.has('g')) revealGroup(target);
	}

	afterNavigate(({ from, to }) => {
		// The words follow the address: another page, or a link with other
		// words, replaces them; the address the search itself wrote does not.
		const q = to?.url.searchParams.get('q') ?? '';
		leaving = false;
		if (!from || from.url.pathname !== to?.url.pathname || q !== words.trim()) words = q;
		const own = wrote !== null && !!to && `${to.url.pathname}${to.url.search}` === wrote;
		wrote = null;
		if (own) return;
		if (!from || from.url.pathname !== to?.url.pathname) edited.set(new Set());
		arrive(to?.url ?? null);
	});

	onMount(async () => {
		try {
			featureMap = await getFeatureMap();
		} catch {
			// The health module may be unavailable; the panels still render.
		}
	});
</script>

<div class="oo-hub" data-space={space} bind:this={hubEl}>
	<header class="oo-hub-head">
		<h1 class="oo-hub-title" id={TITLE_ID}>{title}</h1>
		<div class="oo-hub-search">
			<Input
				label="Search every setting"
				hideLabel
				placeholder="Search every setting (theme, cascading, TOTP, chunk size...)"
				iconLeft="search"
				bind:value={words}
				on:keydown={onSearchKey}
			/>
			{#if words}
				<IconButton icon="x" label="Clear the search" on:click={clearSearch} />
			{/if}
		</div>
		{#if space === 'use'}
			<NotificationCenter />
		{/if}
	</header>

	{#if space === 'workshop' && section === 'network'}
		<NetworkReachability />
	{:else if space === 'workshop' && section === 'security'}
		<SecurityGrade />
	{/if}

	{#if query}
		<section class="oo-hub-results" aria-label="Search results">
			<p class="oo-hub-count" role="status">
				{results.length}
				{results.length === 1 ? 'setting matches' : 'settings match'} "{words.trim()}"
			</p>
			{#if results.length > 0}
				<ul class="oo-hub-list">
					{#each results as hit (hit.id)}
						<li>
							<a
								class="oo-hub-hit"
								href={hit.href}
								on:click={() => {
									leaving = true;
									words = '';
								}}
							>
								<span class="oo-hub-hit-main">
									<span class="oo-hub-hit-title">{hit.title}</span>
									<span class="oo-hub-hit-desc">{hit.description}</span>
								</span>
								<span class="oo-hub-hit-where">
									{hit.where}
									<Icon name="chevron-right" size="sm" />
								</span>
							</a>
						</li>
					{/each}
				</ul>
			{/if}
		</section>
	{/if}

	<!-- The page's groups, hidden while the search shows its results: an
	     open or used group keeps its panel, and its draft, through a search. -->
	<div class="oo-hub-pages" hidden={!!query}>
		{#each pageSections as s (s.id)}
			<section
				class="oo-hub-section"
				aria-labelledby={space === 'workshop' ? TITLE_ID : `oo-hub-${s.id}`}
			>
				{#if space === 'use'}
					<div class="oo-hub-section-head">
						<h2 class="oo-hub-section-title" id={`oo-hub-${s.id}`}>{s.label}</h2>
						{#if s.id === 'appearance'}
							<ThemeSwitcher />
						{:else if s.id === 'account'}
							<UserMenu />
						{/if}
					</div>
				{/if}
				{#each introsOf(s.groups) as block (block.intro)}
					{#if block.intro === 'appearance'}
						<AppearanceSection groups={block.ids} />
					{:else if block.intro === 'conversation'}
						<ConversationDefaults groups={block.ids} />
					{:else if block.intro === 'account'}
						<AccountAuthMode />
					{/if}
				{/each}
				<div class="oo-hub-groups">
					{#each s.groups.filter((g) => g.panel) as group (group.id)}
						<SettingsGroup id={group.id} title={group.title} description={group.description}>
							{#if !featureOk(group)}
								<FeatureUnavailable featureName={group.title} />
							{:else}
								{#await loadPanel(group.panel ?? '')}
									<SkeletonLoader />
								{:then Panel}
									<svelte:component this={Panel} />
								{:catch}
									<p class="oo-hub-error">This panel failed to load.</p>
								{/await}
							{/if}
						</SettingsGroup>
					{/each}
				</div>
			</section>
		{/each}
	</div>
</div>

<style>
	.oo-hub {
		box-sizing: border-box;
		height: 100%;
		overflow-y: auto;
		padding: var(--oo-space-6) var(--oo-space-5) var(--oo-space-9);
	}

	.oo-hub-head,
	.oo-hub-results,
	.oo-hub-section {
		max-width: 880px;
		margin: 0 auto;
	}

	.oo-hub-head {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--oo-space-3) var(--oo-space-4);
		margin-bottom: var(--oo-space-6);
	}

	.oo-hub-title {
		flex: 1;
		min-width: 12rem;
		margin: 0;
		color: var(--oo-fg-primary);
		font-family: var(--oo-font-serif);
		font-size: var(--oo-text-3xl);
		font-weight: 400;
		line-height: var(--oo-leading-tight);
	}

	.oo-hub-search {
		display: flex;
		flex: 1;
		align-items: center;
		gap: var(--oo-space-2);
		min-width: 14rem;
		max-width: 26rem;
	}
	.oo-hub-search > :global(.oo-field) {
		flex: 1;
	}

	.oo-hub-section + .oo-hub-section {
		margin-top: var(--oo-space-8);
	}
	.oo-hub-section-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-3);
		margin-bottom: var(--oo-space-4);
	}
	.oo-hub-section-title {
		margin: 0;
		color: var(--oo-fg-primary);
		font-family: var(--oo-font-serif);
		font-size: var(--oo-text-xl);
		font-weight: 400;
	}

	.oo-hub-groups {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
		margin-top: var(--oo-space-4);
	}
	/* A Workshop page's groups are quiet rows, set apart by space alone. */
	.oo-hub[data-space='workshop'] .oo-hub-groups {
		gap: var(--oo-space-3);
		margin-top: var(--oo-space-3);
	}

	.oo-hub-error {
		color: var(--oo-error);
		font-size: var(--oo-text-sm);
	}

	.oo-hub-count {
		margin: 0 0 var(--oo-space-3);
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
	}
	.oo-hub-list {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.oo-hub-hit {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-4);
		min-height: 44px;
		padding: var(--oo-space-3) var(--oo-space-4);
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-lg);
		background-color: var(--oo-bg-overlay);
		color: var(--oo-fg-primary);
		text-decoration: none;
	}
	.oo-hub-hit:hover {
		background-color: color-mix(in srgb, var(--oo-fg-primary) 4%, var(--oo-bg-overlay));
		border: 1px solid var(--oo-edge);
	}
	.oo-hub-hit-main {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	.oo-hub-hit-title {
		font-size: var(--oo-text-sm);
		font-weight: 500;
	}
	.oo-hub-hit-desc {
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-xs);
	}
	.oo-hub-hit-where {
		display: inline-flex;
		flex-shrink: 0;
		align-items: center;
		gap: var(--oo-space-1);
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-xs);
		white-space: nowrap;
	}
</style>
