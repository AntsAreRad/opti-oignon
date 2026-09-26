/**
 * The settings catalog: every group of settings, in the sections the old
 * settings page showed them under, and the map of that page's old tab ids.
 *
 * Each group also says where it lives in the two spaces: `space` and
 * `section` (a Preferences section, or the Workshop settings page named by
 * the last segment of its URL), or `retired` with the reason it left the
 * interface, or `embeddedIn` the group whose panel renders it. Exactly one
 * of the three. The old settings links are resolved against these places
 * (lib/nav/legacy.ts).
 *
 * Each section lists the groups its panels render, each panel loaded when
 * the page holding it is first shown (the settings hub,
 * components/settings/SettingsHub.svelte, holds the loaders, keyed by the
 * panel names here). The groups a section's introduction renders inline are
 * listed apart, under the section whose introduction renders them, so the
 * settings search finds them too. The hub reads this file, and the old
 * addresses are resolved against it; no other file declares a section list
 * or the old tab map.
 *
 * Pure data, with no dependency.
 */

export type SectionIntro = 'appearance' | 'conversation' | 'account';

/**
 * Where a group lives: a section of one space, or retired with its reason,
 * or embedded in the group whose panel renders it. Exactly one of the three.
 */
export interface GroupPlacement {
	/** The space whose page shows the group. */
	space?: 'use' | 'workshop';
	/** A Preferences section, or a Workshop settings page. */
	section?: string;
	/** Why the group left the interface: it has no page. */
	retired?: string;
	/** The group whose panel renders this one. */
	embeddedIn?: string;
}

export interface SettingsGroupEntry extends GroupPlacement {
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
	description: string;
	/** The component rendered before the section's panel groups. */
	intro?: SectionIntro;
	groups: SettingsGroupEntry[];
}

/** A group a section's introduction renders inline, indexed for search. */
export interface InlineGroup extends GroupPlacement {
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
		description: 'Theme, density, typography and motion.',
		intro: 'appearance',
		groups: []
	},
	{
		id: 'account',
		label: 'Account & Security',
		description: 'Authentication, security mode, recovery and the audit trail.',
		intro: 'account',
		groups: [
			{ id: 'security-mode', title: 'Security mode', description: 'Daily / Bulbe posture and the downgrade ceremony.', synonyms: ['daily', 'bulbe', 'offline', 'isolation', 'downgrade'], panel: 'SecurityModePanel', space: 'workshop', section: 'security' },
			{ id: 'totp', title: 'Two-factor (TOTP)', description: 'Time-based one-time-password authenticator setup.', synonyms: ['2fa', 'otp', 'authenticator', 'mfa'], panel: 'TOTPSetup', space: 'use', section: 'account' },
			{ id: 'webauthn', title: 'Two-factor (WebAuthn)', description: 'Hardware security key and passkey registration.', synonyms: ['2fa', 'passkey', 'fido', 'security key', 'mfa'], panel: 'WebAuthnSetup', space: 'use', section: 'account' },
			{ id: 'recovery-codes', title: 'Recovery codes', description: 'One-time backup codes for account recovery.', synonyms: ['backup codes', 'lockout'], panel: 'RecoveryCodesPanel', space: 'use', section: 'account' },
			{ id: 'app-passwords', title: 'App passwords', description: 'Scoped credentials for programmatic access.', synonyms: ['api', 'token', 'cli'], panel: 'AppPasswordsPanel', space: 'use', section: 'account' },
			{ id: 'hardening', title: 'Hardening', description: 'Security headers and runtime hardening switches.', synonyms: ['csp', 'headers', 'samesite'], panel: 'HardeningPanel', space: 'workshop', section: 'security' },
			{ id: 'key-ceremony', title: 'Key ceremony', description: 'Encryption key generation and rotation ceremony.', synonyms: ['encryption', 'rotation', 'sqlcipher'], panel: 'KeyCeremonyPanel', space: 'workshop', section: 'security' },
			{ id: 'audit-chain', title: 'Audit chain', description: 'Tamper-evident hash-chained audit log viewer.', synonyms: ['audit', 'log', 'events', 'tamper'], panel: 'AuditChainPanel', space: 'workshop', section: 'security' }
		]
	},
	{
		id: 'conversation',
		label: 'Conversation & Chat',
		description: 'Defaults for new conversations, presets, prompt and output behaviour.',
		intro: 'conversation',
		groups: [
			{ id: 'task-presets', title: 'Task presets', description: 'Saved task presets for new conversations.', synonyms: ['preset', 'template', 'quick'], panel: 'PresetManager', space: 'use', section: 'chats' },
			{ id: 'memories', title: 'Memories', description: 'Two-tier agent memory: browse, edit, soft-delete and restore by category.', synonyms: ['memory', 'remember', 'canonical', 'archive'], panel: 'MemoriesPanel', space: 'use', section: 'memory' },
			{ id: 'prompt-config', title: 'Prompt optimization', description: 'System prompt and prompt-enhancement configuration.', synonyms: ['prompt enhance', 'system prompt'], panel: 'PromptConfigPanel', space: 'workshop', section: 'models' },
			{ id: 'compression', title: 'Conversation compression', description: 'Summary compression with a fully searchable archive.', synonyms: ['summary', 'context window', 'tokens'], panel: 'CompressionSettings', space: 'workshop', section: 'models' },
			{ id: 'context-optimizer', title: 'Context optimizer', description: 'Trim and prioritize context sent to the model.', synonyms: ['context', 'window', 'truncation'], panel: 'ContextOptimizerPanel', space: 'workshop', section: 'models' },
			{ id: 'output-humanizer', title: 'Output formatting', description: 'Post-process LLM output with the Humanizer to soften model style.', synonyms: ['humanize', 'humanizer', 'style', 'tone'], panel: 'HumanizerPanel', space: 'workshop', section: 'models' }
		]
	},
	{
		id: 'models',
		label: 'Models & Inference',
		description: 'Model assignment, routing, cascading, speculative decoding and vision.',
		groups: [
			{ id: 'model-health', title: 'Model health', description: 'Per-model availability and warmup monitor.', synonyms: ['warmup', 'status', 'lifecycle'], panel: 'ModelHealthWidget', space: 'workshop', section: 'models' },
			{ id: 'model-profiles', title: 'Model profiles', description: 'Per-profile model parameters and assignment.', synonyms: ['profile', 'parameters'], panel: 'ModelProfilePanel', space: 'workshop', section: 'models' },
			{ id: 'model-assignment', title: 'Model assignment', description: 'Map task types to specific models.', synonyms: ['assign', 'task type', 'mapping'], panel: 'ModelAssignment', space: 'workshop', section: 'models' },
			{ id: 'routing', title: 'Smart routing', description: 'Learned router that picks a model per request.', synonyms: ['router', 'routing strategy', 'learned'], panel: 'LearnedRouterPanel', space: 'workshop', section: 'models' },
			{ id: 'cascading', title: 'Cascading', description: 'Escalate from small to large models on demand.', synonyms: ['cascade', 'escalation'], panel: 'CascadingPanel', space: 'workshop', section: 'models' },
			{ id: 'speculative', title: 'Speculative execution', description: 'Draft / verify generation and llama.cpp native decoding.', synonyms: ['speculative', 'draft', 'verify', 'convergence', 'llama.cpp', 'draft model', 'vram', 'decoding', 'generation'], panel: 'SpeculativeSettings', space: 'workshop', section: 'models' },
			{ id: 'vision', title: 'Vision model', description: 'Model used for image-bearing requests.', synonyms: ['image', 'multimodal', 'vlm'], panel: 'VisionModelSelector', space: 'workshop', section: 'models' }
		]
	},
	{
		id: 'knowledge',
		label: 'Knowledge (RAG)',
		description: 'Knowledge base, collections, ingestion and retrieval dashboards.',
		groups: [
			{ id: 'knowledge-base', title: 'Knowledge base', description: 'Documents, collections and ingestion configuration.', synonyms: ['rag', 'documents', 'collections', 'chunk size', 'retrieval'], panel: 'KnowledgeBasePanel', feature: 'rag', space: 'workshop', section: 'knowledge' },
			{ id: 'rag-dashboard', title: 'RAG dashboard', description: 'Retrieval metrics and index health.', synonyms: ['rag', 'retrieval', 'index', 'metrics'], panel: 'RAGDashboardPanel', feature: 'rag', space: 'workshop', section: 'knowledge' }
		]
	},
	{
		id: 'plugins',
		label: 'Plugins & Extensions',
		description: 'Installed plugins, the marketplace and the permission allowlist.',
		groups: [
			{ id: 'installed-plugins', title: 'Installed plugins', description: 'Manage installed plugins and pipeline hooks.', synonyms: ['extensions', 'tools', 'hooks'], panel: 'PluginsPanel', feature: 'plugins', space: 'workshop', section: 'extensions' },
			{ id: 'plugin-marketplace', title: 'Marketplace', description: 'Discover and install new plugins.', synonyms: ['install', 'catalog', 'discover'], panel: 'PluginMarketplace', feature: 'plugins', space: 'workshop', section: 'extensions' },
			{ id: 'plugin-allowlist', title: 'Permission allowlist', description: 'Per-plugin permission allowlist.', synonyms: ['permissions', 'allowlist', 'security'], panel: 'PluginAllowlistPanel', feature: 'plugins', space: 'workshop', section: 'extensions' },
			{ id: 'skills', title: 'Agent skills', description: 'Browse the SKILL.md registry: published skills and agent-proposed drafts, approval-gated publishing.', synonyms: ['skill', 'teacher', 'draft', 'odysseus'], panel: 'SkillsPanel', space: 'workshop', section: 'extensions' }
		]
	},
	{
		id: 'performance',
		label: 'Performance & Telemetry',
		description: 'Cache, observability, telemetry, profiler and analytics.',
		groups: [
			{ id: 'cache', title: 'Cache', description: 'Response cache statistics and controls.', synonyms: ['cache stats', 'hit rate'], panel: 'CacheStatsPanel', space: 'workshop', section: 'observability' },
			{ id: 'resource-governor', title: 'Resource governor', description: 'VRAM capacity, in-use, pressure and recent admission decisions.', synonyms: ['governor', 'vram', 'capacity', 'pressure', 'admission', 'eviction'], panel: 'GovernorPanel', space: 'workshop', section: 'models' },
			{ id: 'observability', title: 'Observability (Observe)', description: 'Live observability of the inference pipeline.', synonyms: ['observe', 'tracing', 'spans'], panel: 'ObservabilityPanel', feature: 'observability', space: 'workshop', section: 'observability' },
			{ id: 'telemetry', title: 'Telemetry', description: 'Aggregated telemetry dashboard.', synonyms: ['metrics', 'usage'], panel: 'TelemetryDashboard', feature: 'telemetry', space: 'workshop', section: 'observability' },
			{ id: 'telemetry-history', title: 'Telemetry history', description: 'Historical telemetry detail over time.', synonyms: ['history', 'trend', 'metrics'], panel: 'TelemetryHistoryPanel', feature: 'telemetry', space: 'workshop', section: 'observability' },
			{ id: 'profiler', title: 'Profiler', description: 'Per-request inference profiler.', synonyms: ['profile', 'latency', 'timing'], panel: 'ProfilerDashboard', space: 'workshop', section: 'observability' },
			{ id: 'performance-tuner', title: 'Performance tuner', description: 'Throughput and concurrency tuning.', synonyms: ['tuning', 'concurrency', 'throughput'], panel: 'PerformanceTunerPanel', space: 'workshop', section: 'models' },
			{ id: 'performance-dashboard', title: 'Performance dashboard', description: 'High-level performance overview.', synonyms: ['overview', 'metrics'], panel: 'PerformanceDashboard', space: 'workshop', section: 'observability' },
			{ id: 'analytics', title: 'Analytics', description: 'Feedback and performance analytics.', synonyms: ['feedback', 'analytics', 'ratings'], panel: 'AnalyticsDashboard', feature: 'analytics', space: 'workshop', section: 'observability' }
		]
	},
	{
		id: 'network',
		label: 'Network & Privacy',
		description: 'Proxy, web search, remote access and the search kill switch.',
		groups: [
			{ id: 'proxy', title: 'Proxy & web search', description: 'Outbound proxy, Tor mode and web search defaults.', synonyms: ['proxy', 'tor', 'web search', 'network'], panel: 'ProxySettingsPanel', space: 'workshop', section: 'network' },
			{ id: 'remote-access', title: 'Remote access', description: 'Remote access exposure and binding.', synonyms: ['remote', 'expose', 'bind', 'network'], panel: 'RemoteAccessPanel', space: 'workshop', section: 'network' },
			{ id: 'search-kill-switch', title: 'Search kill switch', description: 'Hard switch to disable all outbound search.', synonyms: ['kill switch', 'disable search', 'privacy'], panel: 'SearchKillSwitchPanel', space: 'workshop', section: 'network' },
			{ id: 'device-sync', title: 'Device sync', description: 'Pair your own devices over Veilid, manage peers and watch sync status.', synonyms: ['veilid', 'sync', 'pairing', 'peers', 'p2p'], panel: 'SyncPanel', space: 'workshop', section: 'network' }
		]
	},
	{
		id: 'data',
		label: 'Backup & Data',
		description: 'Backup and restore, fine-tune export and data management.',
		groups: [
			{ id: 'backup-restore', title: 'Backup & restore', description: 'Export and import configuration and data.', synonyms: ['backup', 'restore', 'export', 'import'], panel: 'BackupRestorePanel', space: 'workshop', section: 'backup' },
			{ id: 'fine-tune', title: 'Fine-Tune export', description: 'Export training data, track variants and A/B compare.', synonyms: ['fine-tune', 'export data', 'variants', 'a/b'], panel: 'FineTunePanel', space: 'workshop', section: 'backup' }
		]
	}
];

/** The sections of Preferences, in the order the page shows them. */
export const PREFERENCES_SECTIONS: { id: string; label: string }[] = [
	{ id: 'appearance', label: 'Appearance' },
	{ id: 'keyboard', label: 'Keyboard' },
	{ id: 'account', label: 'Account' },
	{ id: 'chats', label: 'Chats' },
	{ id: 'memory', label: 'Memory' }
];

// The groups the section introductions render inline, in the order they
// render. Their ids are the ids of the SettingsGroup each introduction
// renders, so a deep link scrolls to them.
export const INLINE_GROUPS: InlineGroup[] = [
	{ sectionId: 'appearance', id: 'appearance-theme', title: 'Theme', description: 'Match system, day, night or high contrast.', synonyms: ['dark mode', 'light mode', 'palette', 'colors', 'day', 'night', 'high contrast', 'match system'], space: 'use', section: 'appearance' },
	{ sectionId: 'appearance', id: 'appearance-density', title: 'Density', description: 'Compact, comfortable or spacious spacing.', synonyms: ['spacing', 'compact', 'comfortable', 'spacious'], space: 'use', section: 'appearance' },
	{ sectionId: 'appearance', id: 'appearance-typography', title: 'Text size', description: 'Scales every text size across the app. Composes with density.', synonyms: ['font size', 'text size', 'typography', 'zoom', 'scale'], space: 'use', section: 'appearance' },
	{ sectionId: 'appearance', id: 'appearance-motion', title: 'Motion', description: 'How much the interface animates.', synonyms: ['animation', 'reduce motion', 'reduced motion', 'transitions'], space: 'use', section: 'appearance' },
	{ sectionId: 'appearance', id: 'appearance-advanced', title: 'Keyboard shortcuts', description: 'See and change the keyboard shortcuts.', synonyms: ['keyboard shortcuts', 'shortcuts', 'advanced'], space: 'use', section: 'keyboard' },
	{ sectionId: 'conversation', id: 'conversation-system-preset', title: 'System preset', description: 'One-click hardware-tier infrastructure preset.', synonyms: ['quick', 'hardware', 'tier', 'minimal', 'balanced', 'power'], space: 'workshop', section: 'models' },
	{ sectionId: 'conversation', id: 'conversation-defaults', title: 'Defaults for new conversations', description: 'Default model, temperature, code execution, memory injection.', synonyms: ['quick', 'default model', 'temperature', 'code execution', 'memory injection'], retired: 'Nothing in the application reads these defaults' },
	{ sectionId: 'conversation', id: 'conversation-config-maintenance', title: 'Configuration', description: 'Reload configuration from disk or re-run the first-time setup.', synonyms: ['reload', 'config', 'setup', 'first run', 'onboarding'], space: 'workshop', section: 'backup' },
	{ sectionId: 'account', id: 'account-auth-mode', title: 'Authentication mode', description: 'Single-user or multi-user authentication.', synonyms: ['login', 'single user', 'multi user', 'auth'], space: 'workshop', section: 'security' }
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

