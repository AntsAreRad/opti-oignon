<!--
  Settings hub (consolidation).

  The 12-tab settings page is replaced by a single search-driven hub with 9
  functional sections (spec 5.5 / 9.2): Appearance, Account & Security,
  Conversation & Chat, Models & Inference, Knowledge (RAG), Plugins &
  Extensions, Performance & Telemetry, Network & Privacy, Backup & Data.

  - Deep links: `?section=<id>` (and `?tab=<id>` for the sidebar /
    legacy compatibility), `?q=<query>` to land on a search, `&g=<groupId>`
    to scroll to one group. Legacy tab ids redirect to their new section.
  - Search indexes every group's label + description + synonyms and returns
    the matching groups in place; opening a result deep-links to its section.
  - Section content is lazy-loaded panel-by-panel (pattern preserved:
    dynamic import(), a resolved-constructor _cache, loadComponent, and
    SkeletonLoader while a panel resolves). Feature availability gates panels
    via the /api/health featureMap.
  - The in-page section nav is a WAI-ARIA tablist with arrow-key navigation
    (handleTabKeydown); the sidebar SectionContextList mirrors the same links.

  Every old tab's content lives somewhere in the new structure (migration
  completeness): the legacy "Quick" content is ConversationDefaults; the
  legacy "Security" panels are surfaced individually under Account & Security;
  cache / proxy / humanizer leave the old "Advanced" grab-bag for their proper
  homes; and the previously buried PluginAllowlist, AuditChain,
  TelemetryHistory and RAGDashboard panels become top-level groups.
-->
<script lang="ts">
	import { onMount, type SvelteComponent } from 'svelte';
	import { page } from '$app/stores';
	import { goto, afterNavigate } from '$app/navigation';
	import Icon from '$lib/ds/Icon.svelte';
	import Input from '$lib/ds/Input.svelte';
	import SkeletonLoader from '$lib/components/ui/SkeletonLoader.svelte';
	import FeatureUnavailable from '$lib/components/ui/FeatureUnavailable.svelte';
	import SettingsGroup from '$lib/components/settings/SettingsGroup.svelte';
	import AppearanceSection from '$lib/components/settings/sections/AppearanceSection.svelte';
	import ConversationDefaults from '$lib/components/settings/sections/ConversationDefaults.svelte';
	import AccountAuthMode from '$lib/components/settings/sections/AccountAuthMode.svelte';
	import { getFeatureMap } from '$lib/api/featureCheck';

	// -- Lazy panel loaders (dynamic-import pattern). --------------------------
	// Each panel loads only when its section is first viewed; resolved
	// constructors are cached so re-visiting a section is instant.
	const loaders: Record<string, () => Promise<{ default: typeof SvelteComponent }>> = {
		// Conversation & Chat
		PresetManager: () => import('$lib/components/settings/PresetManager.svelte'),
		MemoriesPanel: () => import('$lib/components/panels/MemoriesPanel.svelte'),
		PromptConfigPanel: () => import('$lib/components/panels/PromptConfigPanel.svelte'),
		CompressionSettings: () => import('$lib/components/panels/CompressionSettings.svelte'),
		ContextOptimizerPanel: () => import('$lib/components/settings/ContextOptimizerPanel.svelte'),
		HumanizerPanel: () => import('$lib/components/panels/HumanizerPanel.svelte'),
		// Models & Inference
		ModelHealthWidget: () => import('$lib/components/settings/ModelHealthWidget.svelte'),
		ModelProfilePanel: () => import('$lib/components/panels/ModelProfilePanel.svelte'),
		ModelAssignment: () => import('$lib/components/panels/ModelAssignment.svelte'),
		LearnedRouterPanel: () => import('$lib/components/panels/LearnedRouterPanel.svelte'),
		CascadingPanel: () => import('$lib/components/panels/CascadingPanel.svelte'),
		SpeculativeSettings: () => import('$lib/components/settings/SpeculativeSettings.svelte'),
		VisionModelSelector: () => import('$lib/components/settings/VisionModelSelector.svelte'),
		// Knowledge (RAG)
		KnowledgeBasePanel: () => import('$lib/components/settings/KnowledgeBasePanel.svelte'),
		RAGDashboardPanel: () => import('$lib/components/settings/RAGDashboardPanel.svelte'),
		// Plugins & Extensions
		PluginsPanel: () => import('$lib/components/settings/PluginsPanel.svelte'),
		SkillsPanel: () => import('$lib/components/panels/SkillsPanel.svelte'),
		PluginMarketplace: () => import('$lib/components/settings/PluginMarketplace.svelte'),
		PluginAllowlistPanel: () => import('$lib/components/settings/PluginAllowlistPanel.svelte'),
		// Performance & Telemetry
		CacheStatsPanel: () => import('$lib/components/panels/CacheStatsPanel.svelte'),
		GovernorPanel: () => import('$lib/components/panels/GovernorPanel.svelte'),
		ObservabilityPanel: () => import('$lib/components/panels/ObservabilityPanel.svelte'),
		TelemetryDashboard: () => import('$lib/components/panels/TelemetryDashboard.svelte'),
		TelemetryHistoryPanel: () => import('$lib/components/panels/TelemetryHistoryPanel.svelte'),
		ProfilerDashboard: () => import('$lib/components/panels/ProfilerDashboard.svelte'),
		PerformanceTunerPanel: () => import('$lib/components/settings/PerformanceTunerPanel.svelte'),
		PerformanceDashboard: () => import('$lib/components/panels/PerformanceDashboard.svelte'),
		AnalyticsDashboard: () => import('$lib/components/panels/AnalyticsDashboard.svelte'),
		// Network & Privacy
		ProxySettingsPanel: () => import('$lib/components/panels/ProxySettingsPanel.svelte'),
		SyncPanel: () => import('$lib/components/panels/SyncPanel.svelte'),
		RemoteAccessPanel: () => import('$lib/components/settings/RemoteAccessPanel.svelte'),
		SearchKillSwitchPanel: () => import('$lib/components/settings/SearchKillSwitchPanel.svelte'),
		// Account & Security
		SecurityModePanel: () => import('$lib/components/settings/SecurityModePanel.svelte'),
		TOTPSetup: () => import('$lib/components/settings/TOTPSetup.svelte'),
		WebAuthnSetup: () => import('$lib/components/settings/WebAuthnSetup.svelte'),
		RecoveryCodesPanel: () => import('$lib/components/settings/RecoveryCodesPanel.svelte'),
		AppPasswordsPanel: () => import('$lib/components/settings/AppPasswordsPanel.svelte'),
		HardeningPanel: () => import('$lib/components/settings/HardeningPanel.svelte'),
		KeyCeremonyPanel: () => import('$lib/components/settings/KeyCeremonyPanel.svelte'),
		AuditChainPanel: () => import('$lib/components/settings/AuditChainPanel.svelte'),
		// Backup & Data
		BackupRestorePanel: () => import('$lib/components/settings/BackupRestorePanel.svelte'),
		FineTunePanel: () => import('$lib/components/settings/FineTunePanel.svelte')
	};

	const _cache: Record<string, typeof SvelteComponent> = {};
	async function loadComponent(key: string): Promise<typeof SvelteComponent> {
		if (_cache[key]) return _cache[key];
		const mod = await loaders[key]();
		_cache[key] = mod.default;
		return mod.default;
	}

	// -- Section + group registry. --------------------------------------------
	interface Group {
		id: string;
		title: string;
		description: string;
		synonyms?: string[];
		/** Lazy panel key (into `loaders`). Omitted for inline-rendered groups. */
		panel?: string;
		/** Optional featureMap key; the panel is gated when the feature is off. */
		feature?: string;
	}
	interface Section {
		id: string;
		label: string;
		icon: string;
		description: string;
		/** Inline component rendered before this section's lazy-panel groups. */
		intro?: 'appearance' | 'conversation' | 'account';
		groups: Group[];
	}

	const SECTIONS: Section[] = [
		{
			id: 'appearance',
			label: 'Appearance',
			icon: 'palette',
			description: 'Theme, density, typography and motion.',
			intro: 'appearance',
			groups: []
		},
		{
			id: 'account',
			label: 'Account & Security',
			icon: 'shield-check',
			description: 'Authentication, security mode, recovery and the audit trail.',
			intro: 'account',
			groups: [
				{ id: 'security-mode', title: 'Security mode', description: 'Daily / Bulbe posture and the downgrade ceremony.', synonyms: ['daily', 'bulbe', 'offline', 'isolation', 'downgrade'], panel: 'SecurityModePanel' },
				{ id: 'totp', title: 'Two-factor (TOTP)', description: 'Time-based one-time-password authenticator setup.', synonyms: ['2fa', 'otp', 'authenticator', 'mfa'], panel: 'TOTPSetup' },
				{ id: 'webauthn', title: 'Two-factor (WebAuthn)', description: 'Hardware security key and passkey registration.', synonyms: ['2fa', 'passkey', 'fido', 'security key', 'mfa'], panel: 'WebAuthnSetup' },
				{ id: 'recovery-codes', title: 'Recovery codes', description: 'One-time backup codes for account recovery.', synonyms: ['backup codes', 'lockout'], panel: 'RecoveryCodesPanel' },
				{ id: 'app-passwords', title: 'App passwords', description: 'Scoped credentials for programmatic access.', synonyms: ['api', 'token', 'cli'], panel: 'AppPasswordsPanel' },
				{ id: 'hardening', title: 'Hardening', description: 'Security headers and runtime hardening switches.', synonyms: ['csp', 'headers', 'samesite'], panel: 'HardeningPanel' },
				{ id: 'key-ceremony', title: 'Key ceremony', description: 'Encryption key generation and rotation ceremony.', synonyms: ['encryption', 'rotation', 'sqlcipher'], panel: 'KeyCeremonyPanel' },
				{ id: 'audit-chain', title: 'Audit chain', description: 'Tamper-evident hash-chained audit log viewer.', synonyms: ['audit', 'log', 'events', 'tamper'], panel: 'AuditChainPanel' }
			]
		},
		{
			id: 'conversation',
			label: 'Conversation & Chat',
			icon: 'messages-square',
			description: 'Defaults for new conversations, presets, prompt and output behaviour.',
			intro: 'conversation',
			groups: [
				{ id: 'task-presets', title: 'Task presets', description: 'Saved task presets for new conversations.', synonyms: ['preset', 'template', 'quick'], panel: 'PresetManager' },
				{ id: 'memories', title: 'Memories', description: 'Two-tier agent memory: browse, edit, soft-delete and restore by category.', synonyms: ['memory', 'remember', 'canonical', 'archive'], panel: 'MemoriesPanel' },
				{ id: 'prompt-config', title: 'Prompt optimization', description: 'System prompt and prompt-enhancement configuration.', synonyms: ['prompt enhance', 'system prompt'], panel: 'PromptConfigPanel' },
				{ id: 'compression', title: 'Conversation compression', description: 'Summary compression with a fully searchable archive.', synonyms: ['summary', 'context window', 'tokens'], panel: 'CompressionSettings' },
				{ id: 'context-optimizer', title: 'Context optimizer', description: 'Trim and prioritize context sent to the model.', synonyms: ['context', 'window', 'truncation'], panel: 'ContextOptimizerPanel' },
				{ id: 'output-humanizer', title: 'Output formatting', description: 'Post-process LLM output with the Humanizer to soften model style.', synonyms: ['humanize', 'humanizer', 'style', 'tone'], panel: 'HumanizerPanel' }
			]
		},
		{
			id: 'models',
			label: 'Models & Inference',
			icon: 'cpu',
			description: 'Model assignment, routing, cascading, speculative decoding and vision.',
			groups: [
				{ id: 'model-health', title: 'Model health', description: 'Per-model availability and warmup monitor.', synonyms: ['warmup', 'status', 'lifecycle'], panel: 'ModelHealthWidget' },
				{ id: 'model-profiles', title: 'Model profiles', description: 'Per-profile model parameters and assignment.', synonyms: ['profile', 'parameters'], panel: 'ModelProfilePanel' },
				{ id: 'model-assignment', title: 'Model assignment', description: 'Map task types to specific models.', synonyms: ['assign', 'task type', 'mapping'], panel: 'ModelAssignment' },
				{ id: 'routing', title: 'Smart routing', description: 'Learned router that picks a model per request.', synonyms: ['router', 'routing strategy', 'learned'], panel: 'LearnedRouterPanel' },
				{ id: 'cascading', title: 'Cascading', description: 'Escalate from small to large models on demand.', synonyms: ['cascade', 'escalation'], panel: 'CascadingPanel' },
				{ id: 'speculative', title: 'Speculative execution', description: 'Draft / verify generation (S70) and llama.cpp native decoding.', synonyms: ['speculative', 'draft', 'verify', 'convergence', 'llama.cpp', 'draft model', 'vram', 'decoding', 'generation'], panel: 'SpeculativeSettings' },
				{ id: 'vision', title: 'Vision model', description: 'Model used for image-bearing requests.', synonyms: ['image', 'multimodal', 'vlm'], panel: 'VisionModelSelector' }
			]
		},
		{
			id: 'knowledge',
			label: 'Knowledge (RAG)',
			icon: 'book-open',
			description: 'Knowledge base, collections, ingestion and retrieval dashboards.',
			groups: [
				{ id: 'knowledge-base', title: 'Knowledge base', description: 'Documents, collections and ingestion configuration.', synonyms: ['rag', 'documents', 'collections', 'chunk size', 'retrieval'], panel: 'KnowledgeBasePanel', feature: 'rag' },
				{ id: 'rag-dashboard', title: 'RAG dashboard', description: 'Retrieval metrics and index health.', synonyms: ['rag', 'retrieval', 'index', 'metrics'], panel: 'RAGDashboardPanel', feature: 'rag' }
			]
		},
		{
			id: 'plugins',
			label: 'Plugins & Extensions',
			icon: 'plug',
			description: 'Installed plugins, the marketplace and the permission allowlist.',
			groups: [
				{ id: 'installed-plugins', title: 'Installed plugins', description: 'Manage installed plugins and pipeline hooks.', synonyms: ['extensions', 'tools', 'hooks'], panel: 'PluginsPanel', feature: 'plugins' },
				{ id: 'plugin-marketplace', title: 'Marketplace', description: 'Discover and install new plugins.', synonyms: ['install', 'catalog', 'discover'], panel: 'PluginMarketplace', feature: 'plugins' },
				{ id: 'plugin-allowlist', title: 'Permission allowlist', description: 'Per-plugin permission allowlist.', synonyms: ['permissions', 'allowlist', 'security'], panel: 'PluginAllowlistPanel', feature: 'plugins' },
				{ id: 'skills', title: 'Agent skills', description: 'Browse the SKILL.md registry: published skills and agent-proposed drafts, approval-gated publishing.', synonyms: ['skill', 'teacher', 'draft', 'odysseus'], panel: 'SkillsPanel' }
			]
		},
		{
			id: 'performance',
			label: 'Performance & Telemetry',
			icon: 'activity',
			description: 'Cache, observability, telemetry, profiler and analytics.',
			groups: [
				{ id: 'cache', title: 'Cache', description: 'Response cache statistics and controls.', synonyms: ['cache stats', 'hit rate'], panel: 'CacheStatsPanel' },
				{ id: 'resource-governor', title: 'Resource governor', description: 'VRAM capacity, in-use, pressure and recent admission decisions.', synonyms: ['governor', 'vram', 'capacity', 'pressure', 'admission', 'eviction'], panel: 'GovernorPanel' },
				{ id: 'observability', title: 'Observability (Observe)', description: 'Live observability of the inference pipeline.', synonyms: ['observe', 'tracing', 'spans'], panel: 'ObservabilityPanel', feature: 'observability' },
				{ id: 'telemetry', title: 'Telemetry', description: 'Aggregated telemetry dashboard.', synonyms: ['metrics', 'usage'], panel: 'TelemetryDashboard', feature: 'telemetry' },
				{ id: 'telemetry-history', title: 'Telemetry history', description: 'Historical telemetry detail over time.', synonyms: ['history', 'trend', 'metrics'], panel: 'TelemetryHistoryPanel', feature: 'telemetry' },
				{ id: 'profiler', title: 'Profiler', description: 'Per-request inference profiler.', synonyms: ['profile', 'latency', 'timing'], panel: 'ProfilerDashboard' },
				{ id: 'performance-tuner', title: 'Performance tuner', description: 'Throughput and concurrency tuning.', synonyms: ['tuning', 'concurrency', 'throughput'], panel: 'PerformanceTunerPanel' },
				{ id: 'performance-dashboard', title: 'Performance dashboard', description: 'High-level performance overview.', synonyms: ['overview', 'metrics'], panel: 'PerformanceDashboard' },
				{ id: 'analytics', title: 'Analytics', description: 'Feedback and performance analytics.', synonyms: ['feedback', 'analytics', 'ratings'], panel: 'AnalyticsDashboard', feature: 'analytics' }
			]
		},
		{
			id: 'network',
			label: 'Network & Privacy',
			icon: 'globe',
			description: 'Proxy, web search, remote access and the search kill switch.',
			groups: [
				{ id: 'proxy', title: 'Proxy & web search', description: 'Outbound proxy, Tor mode and web search defaults.', synonyms: ['proxy', 'tor', 'web search', 'network'], panel: 'ProxySettingsPanel' },
				{ id: 'remote-access', title: 'Remote access', description: 'Remote access exposure and binding.', synonyms: ['remote', 'expose', 'bind', 'network'], panel: 'RemoteAccessPanel' },
				{ id: 'search-kill-switch', title: 'Search kill switch', description: 'Hard switch to disable all outbound search.', synonyms: ['kill switch', 'disable search', 'privacy'], panel: 'SearchKillSwitchPanel' },
				{ id: 'device-sync', title: 'Device sync', description: 'Pair your own devices over Veilid, manage peers and watch sync status.', synonyms: ['veilid', 'sync', 'pairing', 'peers', 'p2p'], panel: 'SyncPanel' }
			]
		},
		{
			id: 'data',
			label: 'Backup & Data',
			icon: 'database',
			description: 'Backup and restore, fine-tune export and data management.',
			groups: [
				{ id: 'backup-restore', title: 'Backup & restore', description: 'Export and import configuration and data.', synonyms: ['backup', 'restore', 'export', 'import'], panel: 'BackupRestorePanel' },
				{ id: 'fine-tune', title: 'Fine-Tune export', description: 'Export training data, track variants and A/B compare.', synonyms: ['fine-tune', 'export data', 'variants', 'a/b'], panel: 'FineTunePanel' }
			]
		}
	];

	// Inline (intro-component) groups, indexed for search even though they are
	// not lazy panels. Their anchors match the SettingsGroup ids the intro
	// components render, so deep-links scroll correctly.
	const INLINE_INDEX: { sectionId: string; id: string; title: string; description: string; synonyms: string[] }[] = [
		{ sectionId: 'appearance', id: 'appearance-theme', title: 'Theme', description: 'Active palette and light/dark mode.', synonyms: ['dark mode', 'light mode', 'palette', 'colors', 'anthracite', 'parchment', 'slate', 'linen', 'high contrast'] },
		{ sectionId: 'appearance', id: 'appearance-density', title: 'Density', description: 'Compact, comfortable or spacious spacing.', synonyms: ['spacing', 'compact', 'comfortable', 'spacious'] },
		{ sectionId: 'conversation', id: 'conversation-system-preset', title: 'System preset', description: 'One-click hardware-tier infrastructure preset.', synonyms: ['quick', 'hardware', 'tier', 'minimal', 'balanced', 'power'] },
		{ sectionId: 'conversation', id: 'conversation-defaults', title: 'Defaults for new conversations', description: 'Default model, temperature, code execution, memory injection.', synonyms: ['quick', 'default model', 'temperature', 'code execution', 'memory injection'] },
		{ sectionId: 'account', id: 'account-auth-mode', title: 'Authentication mode', description: 'Single-user or multi-user authentication.', synonyms: ['login', 'single user', 'multi user', 'auth'] }
	];

	// Flat search index over every group (lazy + inline).
	interface Hit {
		sectionId: string;
		sectionLabel: string;
		id: string;
		title: string;
		description: string;
		haystack: string;
	}
	const searchIndex: Hit[] = [
		...SECTIONS.flatMap((s) =>
			s.groups.map((g) => ({
				sectionId: s.id,
				sectionLabel: s.label,
				id: g.id,
				title: g.title,
				description: g.description,
				haystack: [g.title, g.description, ...(g.synonyms ?? []), s.label].join(' ').toLowerCase()
			}))
		),
		...INLINE_INDEX.map((g) => {
			const sec = SECTIONS.find((s) => s.id === g.sectionId);
			return {
				sectionId: g.sectionId,
				sectionLabel: sec ? sec.label : g.sectionId,
				id: g.id,
				title: g.title,
				description: g.description,
				haystack: [g.title, g.description, ...g.synonyms, sec ? sec.label : ''].join(' ').toLowerCase()
			};
		})
	];

	// -- Legacy ?tab= ids -> new section id (backwards compatibility). ---------
	const LEGACY_TAB_TO_SECTION: Record<string, string> = {
		'quick': 'conversation',
		'presets': 'conversation',
		'prompt': 'conversation',
		'models': 'models',
		'analytics': 'performance',
		'performance': 'performance',
		'fine-tune': 'data',
		'knowledge': 'knowledge',
		'plugins': 'plugins',
		'backup': 'data',
		'security': 'account',
		'advanced': 'performance'
	};

	function isSectionId(id: string): boolean {
		return SECTIONS.some((s) => s.id === id);
	}

	function resolveSection(raw: string | null | undefined): string {
		if (!raw) return SECTIONS[0].id;
		if (isSectionId(raw)) return raw;
		if (raw in LEGACY_TAB_TO_SECTION) return LEGACY_TAB_TO_SECTION[raw];
		return SECTIONS[0].id;
	}

	// -- URL-driven state (deferred in-page ?tab sync). ------------------------
	$: rawTab = $page.url.searchParams.get('section') ?? $page.url.searchParams.get('tab');
	$: activeSection = resolveSection(rawTab);
	$: activeSectionDef = SECTIONS.find((s) => s.id === activeSection) ?? SECTIONS[0];
	$: deepGroup = $page.url.searchParams.get('g');
	$: urlQuery = $page.url.searchParams.get('q') ?? '';

	// Redirect a legacy tab id to its new section so old links resolve.
	$: if (rawTab && !isSectionId(rawTab) && rawTab in LEGACY_TAB_TO_SECTION) {
		goto(`/settings?section=${LEGACY_TAB_TO_SECTION[rawTab]}`, {
			replaceState: true,
			keepFocus: true,
			noScroll: true
		});
	}

	let searchInput = '';
	let mounted = false;
	let appVersion = '';

	$: trimmedQuery = searchInput.trim().toLowerCase();
	$: results = trimmedQuery
		? searchIndex.filter((h) => trimmedQuery.split(/\s+/).every((tok) => h.haystack.includes(tok)))
		: [];

	// featureMap availability. Whole-section gates plus a per-group gate.
	let featureMap: Record<string, boolean> = {};
	$: ragAvailable = featureMap['rag'] !== false;
	$: pluginsAvailable = featureMap['plugins'] !== false;
	$: analyticsAvailable = featureMap['analytics'] !== false;

	function featureOk(g: Group): boolean {
		if (!g.feature) return true;
		return featureMap[g.feature] !== false;
	}

	// -- Navigation helpers (URL is the source of truth; native back works). ---
	function selectSection(id: string) {
		searchInput = '';
		goto(`/settings?section=${id}`, { keepFocus: true, noScroll: true });
	}

	// Debounced ?q= sync so the search is shareable without spamming history;
	// live results come from the reactive `results` derived from `searchInput`.
	let searchSyncTimer: ReturnType<typeof setTimeout> | undefined;
	function syncQueryToUrl() {
		const q = searchInput.trim();
		goto(q ? `/settings?q=${encodeURIComponent(q)}` : '/settings', {
			replaceState: true,
			keepFocus: true,
			noScroll: true
		});
	}
	function onSearchInput() {
		if (searchSyncTimer) clearTimeout(searchSyncTimer);
		searchSyncTimer = setTimeout(syncQueryToUrl, 250);
	}

	function clearSearch() {
		if (searchSyncTimer) clearTimeout(searchSyncTimer);
		searchInput = '';
		goto('/settings', { replaceState: true, keepFocus: true, noScroll: true });
	}

	// Enter opens the top result; Escape clears the search (spec 5.6).
	function handleSearchKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && results.length > 0) {
			e.preventDefault();
			openResult(results[0]);
		} else if (e.key === 'Escape' && searchInput) {
			e.preventDefault();
			clearSearch();
		}
	}

	function openResult(hit: Hit) {
		if (searchSyncTimer) clearTimeout(searchSyncTimer);
		searchInput = '';
		goto(`/settings?section=${hit.sectionId}&g=${hit.id}`);
	}

	function scrollToGroup(groupId: string | null | undefined) {
		if (!groupId || typeof document === 'undefined') return;
		requestAnimationFrame(() => {
			const el = document.getElementById(`oo-set-${groupId}`);
			if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
		});
	}

	// Arrow-key navigation across the section tablist (spec 8.8).
	function handleTabKeydown(e: KeyboardEvent) {
		const keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'];
		if (!keys.includes(e.key)) return;
		e.preventDefault();
		const idx = SECTIONS.findIndex((s) => s.id === activeSection);
		let next = idx;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (idx + 1) % SECTIONS.length;
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (idx - 1 + SECTIONS.length) % SECTIONS.length;
		else if (e.key === 'Home') next = 0;
		else if (e.key === 'End') next = SECTIONS.length - 1;
		selectSection(SECTIONS[next].id);
		const container = (e.currentTarget as HTMLElement) ?? null;
		if (container) {
			const tabs = container.querySelectorAll<HTMLElement>('[role="tab"]');
			tabs[next]?.focus();
		}
	}

	afterNavigate(() => {
		if (deepGroup) scrollToGroup(deepGroup);
	});

	onMount(async () => {
		searchInput = urlQuery;
		mounted = true;
		try {
			featureMap = await getFeatureMap();
		} catch {
			// Health module may be unavailable; sections still render.
		}
		try {
			const resp = await fetch('/api/health');
			if (resp.ok) {
				const data = await resp.json();
				appVersion = data.version ?? '';
			}
		} catch {
			// Version is decorative; ignore failures.
		}
		if (deepGroup) scrollToGroup(deepGroup);
	});
</script>

<div class="oo-settings">
	<header class="oo-settings-head">
		<div class="oo-settings-title">
			<h1>Settings</h1>
			<p class="oo-settings-crumb">
				{activeSectionDef.label}
				{#if appVersion}
					<span class="oo-settings-version">v{appVersion}</span>
				{/if}
			</p>
		</div>
		<div class="oo-settings-search">
			<Input
				label="Search settings"
				hideLabel
				placeholder="Search settings (theme, cascading, TOTP, chunk size...)"
				iconLeft="search"
				bind:value={searchInput}
				on:input={onSearchInput}
				on:keydown={handleSearchKeydown}
			/>
			{#if searchInput}
				<button type="button" class="oo-search-clear" on:click={clearSearch} aria-label="Clear search">
					<Icon name="x" size="sm" />
				</button>
			{/if}
		</div>
	</header>

	<!-- In-page section navigation (WAI-ARIA tablist); mirrors the sidebar. -->
	<div
		class="oo-settings-tabs"
		role="tablist"
		aria-label="Settings sections"
		on:keydown={handleTabKeydown}
	>
		{#each SECTIONS as s (s.id)}
			<button
				type="button"
				class="oo-settings-tab"
				class:oo-settings-tab-active={!trimmedQuery && activeSection === s.id}
				role="tab"
				aria-selected={!trimmedQuery && activeSection === s.id}
				tabindex={activeSection === s.id ? 0 : -1}
				title={s.description}
				on:click={() => selectSection(s.id)}
			>
				<Icon name={s.icon} size="sm" />
				<span>{s.label}</span>
			</button>
		{/each}
	</div>

	<div class="oo-settings-body">
		{#if trimmedQuery}
			<!-- Search results: matching groups, deep-linking into their section. -->
			<div class="oo-search-results">
				<p class="oo-search-count">
					{results.length}
					{results.length === 1 ? 'result' : 'results'} for "{searchInput.trim()}"
				</p>
				{#if results.length === 0}
					<p class="oo-search-empty">No settings match that search.</p>
				{:else}
					<ul class="oo-search-list">
						{#each results as hit (hit.sectionId + '/' + hit.id)}
							<li>
								<button type="button" class="oo-search-hit" on:click={() => openResult(hit)}>
									<span class="oo-search-hit-main">
										<span class="oo-search-hit-title">{hit.title}</span>
										<span class="oo-search-hit-desc">{hit.description}</span>
									</span>
									<span class="oo-search-hit-sec">
										{hit.sectionLabel}
										<Icon name="chevron-right" size="sm" />
									</span>
								</button>
							</li>
						{/each}
					</ul>
				{/if}
			</div>
		{:else}
			<!-- Active section content. -->
			<div class="oo-section-head">
				<h2>{activeSectionDef.label}</h2>
				<p>{activeSectionDef.description}</p>
			</div>

			{#if activeSectionDef.intro === 'appearance'}
				<AppearanceSection />
			{:else if activeSectionDef.intro === 'conversation'}
				<ConversationDefaults />
			{:else if activeSectionDef.intro === 'account'}
				<AccountAuthMode />
			{/if}

			<div class="oo-section-groups">
				{#each activeSectionDef.groups as group (group.id)}
					<SettingsGroup id={group.id} title={group.title} description={group.description}>
						{#if !featureOk(group)}
							<FeatureUnavailable featureName={group.title} />
						{:else}
							{#await loadComponent(group.panel ?? '')}
								<SkeletonLoader />
							{:then PanelComponent}
								<svelte:component this={PanelComponent} />
							{:catch}
								<p class="oo-section-error">This panel failed to load.</p>
							{/await}
						{/if}
					</SettingsGroup>
				{/each}
			</div>
		{/if}
	</div>
</div>

<style>
	.oo-settings {
		height: 100%;
		overflow-y: auto;
	}

	.oo-settings-head {
		display: flex;
		align-items: flex-end;
		justify-content: space-between;
		gap: var(--oo-space-4);
		flex-wrap: wrap;
		max-width: 880px;
		margin: 0 auto;
		padding: var(--oo-space-6) var(--oo-space-4) var(--oo-space-3);
	}

	.oo-settings-title h1 {
		font-size: var(--oo-text-xl);
		font-weight: 600;
		color: var(--oo-fg-primary);
		margin: 0;
	}

	.oo-settings-crumb {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
		margin: 2px 0 0;
	}

	.oo-settings-version {
		display: inline-block;
		margin-left: var(--oo-space-2);
		padding: 0 var(--oo-space-2);
		border-radius: var(--oo-radius-full);
		font-family: var(--oo-font-mono);
		font-size: var(--oo-text-2xs);
		background-color: var(--oo-msg-user-bg);
		color: var(--oo-fg-on-accent);
	}

	.oo-settings-search {
		position: relative;
		flex: 1;
		min-width: 220px;
		max-width: 420px;
	}

	.oo-search-clear {
		position: absolute;
		top: 50%;
		right: var(--oo-space-2);
		transform: translateY(-50%);
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 22px;
		height: 22px;
		border-radius: var(--oo-radius-full);
		color: var(--oo-fg-muted);
		cursor: pointer;
	}

	.oo-search-clear:hover {
		color: var(--oo-fg-primary);
		background-color: var(--oo-bg-elevated);
	}

	.oo-settings-tabs {
		display: flex;
		gap: var(--oo-space-1);
		overflow-x: auto;
		max-width: 880px;
		margin: 0 auto;
		padding: 0 var(--oo-space-4) var(--oo-space-2);
		border-bottom: 1px solid var(--oo-bd-default);
	}

	.oo-settings-tab {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-2);
		padding: var(--oo-space-2) var(--oo-space-3);
		border-radius: var(--oo-radius-md) var(--oo-radius-md) 0 0;
		font-size: var(--oo-text-sm);
		font-weight: 500;
		white-space: nowrap;
		color: var(--oo-fg-tertiary);
		border-bottom: 2px solid transparent;
		cursor: pointer;
		transition:
			color 0.12s ease,
			background-color 0.12s ease,
			border-color 0.12s ease;
	}

	.oo-settings-tab:hover {
		color: var(--oo-fg-secondary);
		background-color: var(--oo-bg-surface);
	}

	.oo-settings-tab-active {
		color: var(--oo-accent);
		border-bottom-color: var(--oo-accent);
	}

	.oo-settings-body {
		max-width: 880px;
		margin: 0 auto;
		padding: var(--oo-space-5) var(--oo-space-4) var(--oo-space-9);
	}

	.oo-section-head {
		margin-bottom: var(--oo-space-4);
	}

	.oo-section-head h2 {
		font-size: var(--oo-text-lg);
		font-weight: 600;
		color: var(--oo-fg-primary);
		margin: 0;
	}

	.oo-section-head p {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
		margin: var(--oo-space-1) 0 0;
	}

	.oo-section-groups {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
		margin-top: var(--oo-space-4);
	}

	.oo-section-error {
		font-size: var(--oo-text-sm);
		color: var(--oo-error);
	}

	.oo-search-results {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}

	.oo-search-count {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
	}

	.oo-search-empty {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}

	.oo-search-list {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		list-style: none;
		margin: 0;
		padding: 0;
	}

	.oo-search-hit {
		width: 100%;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-3);
		padding: var(--oo-space-3) var(--oo-space-4);
		border-radius: var(--oo-radius-md);
		border: 1px solid var(--oo-bd-subtle);
		background-color: var(--oo-bg-surface);
		cursor: pointer;
		text-align: left;
		transition:
			border-color 0.12s ease,
			background-color 0.12s ease;
	}

	.oo-search-hit:hover {
		border-color: var(--oo-accent);
		background-color: var(--oo-accent-bg);
	}

	.oo-search-hit-main {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}

	.oo-search-hit-title {
		font-size: var(--oo-text-sm);
		font-weight: 500;
		color: var(--oo-fg-primary);
	}

	.oo-search-hit-desc {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
	}

	.oo-search-hit-sec {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-1);
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-tertiary);
		white-space: nowrap;
		flex-shrink: 0;
	}
</style>
