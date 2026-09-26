/**
 * The settings catalog: every section of the settings page, every group it
 * renders, and the map of the page's old tab ids.
 *
 * Each section lists the groups its panels render, each panel loaded when
 * its section is first shown (the settings page holds the loaders, keyed by
 * the panel names here). The groups a section's introduction renders inline
 * are listed apart, under the section whose introduction renders them, so
 * the settings search finds them too. The settings page and the sidebar's
 * section list both read this file; no other file declares a section list
 * or the old tab map.
 *
 * Pure data and two pure functions, with no dependency.
 */

export type SectionIntro = 'appearance' | 'conversation' | 'account';

export interface SettingsGroupEntry {
	id: string;
	title: string;
	description: string;
	synonyms?: string[];
	/** The lazy panel's key in the settings page's loaders. */
	panel: string;
	/** A feature-map key; the panel is gated when that feature is off. */
	feature?: string;
}

export interface SettingsSection {
	id: string;
	label: string;
	icon: string;
	description: string;
	/** The component rendered before the section's panel groups. */
	intro?: SectionIntro;
	groups: SettingsGroupEntry[];
}

/** A group a section's introduction renders inline, indexed for search. */
export interface InlineGroup {
	sectionId: string;
	id: string;
	title: string;
	description: string;
	synonyms: string[];
}

export const SETTINGS_SECTIONS: SettingsSection[] = [
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
			{ id: 'speculative', title: 'Speculative execution', description: 'Draft / verify generation and llama.cpp native decoding.', synonyms: ['speculative', 'draft', 'verify', 'convergence', 'llama.cpp', 'draft model', 'vram', 'decoding', 'generation'], panel: 'SpeculativeSettings' },
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

// The groups the section introductions render inline, in the order they
// render. Their ids are the ids of the SettingsGroup each introduction
// renders, so a deep link scrolls to them.
export const INLINE_GROUPS: InlineGroup[] = [
	{ sectionId: 'appearance', id: 'appearance-theme', title: 'Theme', description: 'Active palette and light/dark mode.', synonyms: ['dark mode', 'light mode', 'palette', 'colors', 'anthracite', 'parchment', 'slate', 'linen', 'high contrast'] },
	{ sectionId: 'appearance', id: 'appearance-density', title: 'Density', description: 'Compact, comfortable or spacious spacing.', synonyms: ['spacing', 'compact', 'comfortable', 'spacious'] },
	{ sectionId: 'appearance', id: 'appearance-typography', title: 'Text size', description: 'Scales every text size across the app. Composes with density.', synonyms: ['font size', 'text size', 'typography', 'zoom', 'scale'] },
	{ sectionId: 'appearance', id: 'appearance-motion', title: 'Motion', description: 'How much the interface animates.', synonyms: ['animation', 'reduce motion', 'reduced motion', 'transitions'] },
	{ sectionId: 'appearance', id: 'appearance-advanced', title: 'Advanced', description: 'Fine-tune accent colors and keyboard shortcuts.', synonyms: ['accent', 'colors', 'keyboard shortcuts', 'shortcuts'] },
	{ sectionId: 'conversation', id: 'conversation-system-preset', title: 'System preset', description: 'One-click hardware-tier infrastructure preset.', synonyms: ['quick', 'hardware', 'tier', 'minimal', 'balanced', 'power'] },
	{ sectionId: 'conversation', id: 'conversation-defaults', title: 'Defaults for new conversations', description: 'Default model, temperature, code execution, memory injection.', synonyms: ['quick', 'default model', 'temperature', 'code execution', 'memory injection'] },
	{ sectionId: 'conversation', id: 'conversation-config-maintenance', title: 'Configuration', description: 'Reload configuration from disk or re-run the first-time setup.', synonyms: ['reload', 'config', 'setup', 'first run', 'onboarding'] },
	{ sectionId: 'account', id: 'account-auth-mode', title: 'Authentication mode', description: 'Single-user or multi-user authentication.', synonyms: ['login', 'single user', 'multi user', 'auth'] }
];

// Old `?tab=` ids of the settings page, each to the section that took its
// content, so an old link still lands where it meant to.
export const LEGACY_TAB_TO_SECTION: Readonly<Record<string, string>> = {
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

/** True when `id` names a settings section. */
export function isSectionId(id: string): boolean {
	return SETTINGS_SECTIONS.some((section) => section.id === id);
}

/**
 * The section a `?section=` or `?tab=` value opens: a section id as it
 * is, an old tab id through the map, anything else the first section.
 */
export function resolveSection(raw: string | null | undefined): string {
	if (!raw) return SETTINGS_SECTIONS[0].id;
	if (isSectionId(raw)) return raw;
	if (Object.prototype.hasOwnProperty.call(LEGACY_TAB_TO_SECTION, raw)) return LEGACY_TAB_TO_SECTION[raw];
	return SETTINGS_SECTIONS[0].id;
}
