<!--
  SecurityPanel.svelte
  Security status panel for settings page.
  Shows: security grade, individual checks, encryption status,
  JWT cookie mode, security mode, kill switch, plugin allowlist,
  encryption key ceremony, PQC signatures, 2FA, recent audit events,
  hash-chain audit log, system hardening.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import WebAuthnSetup from './WebAuthnSetup.svelte';
	import TOTPSetup from './TOTPSetup.svelte';
	import RecoveryCodesPanel from './RecoveryCodesPanel.svelte';
	import AppPasswordsPanel from './AppPasswordsPanel.svelte';
	import SecurityModePanel from './SecurityModePanel.svelte';
	import SearchKillSwitchPanel from './SearchKillSwitchPanel.svelte';
	import PluginAllowlistPanel from './PluginAllowlistPanel.svelte';
	import KeyCeremonyPanel from './KeyCeremonyPanel.svelte';
	import AuditChainPanel from './AuditChainPanel.svelte';
	import HardeningPanel from './HardeningPanel.svelte';
	import RemoteAccessPanel from './RemoteAccessPanel.svelte';

	interface SecurityCheck {
		name: string;
		points: number;
		max_points: number;
		passed: boolean;
		detail: string;
	}

	interface AuditEvent {
		source: string;
		event_type: string;
		action: string;
		severity: string;
		timestamp: number;
		details: Record<string, unknown>;
	}

	interface EncryptionStatus {
		enabled: boolean;
		has_key: boolean;
		keyfile_exists: boolean;
		env_key_set: boolean;
	}

	let grade = '?';
	let score = 0;
	let maxScore = 100;
	let checks: SecurityCheck[] = [];
	let auditEvents: AuditEvent[] = [];
	let encryptionStatus: EncryptionStatus | null = null;
	let cookieMode = true;
	let loading = true;
	let error = '';

	// Active security subsection
	let activeSection: 'overview' | 'mode' | 'killswitch' | 'plugins' | 'encryption' | 'auditlog' | 'hardening' | 'remote' = 'overview';

	onMount(async () => {
		await Promise.all([loadSecurityStatus(), loadAuditEvents(), loadEncryptionStatus()]);
		loading = false;
	});

	async function loadSecurityStatus() {
		try {
			const resp = await fetch('/api/security/status', { credentials: 'include' });
			if (resp.ok) {
				const data = await resp.json();
				grade = data.grade || '?';
				score = data.score || 0;
				maxScore = data.max_score || 100;
				checks = data.checks || [];
			}
		} catch (e) {
			error = 'Failed to load security status';
		}
	}

	async function loadAuditEvents() {
		try {
			const resp = await fetch('/api/security/audit?limit=10', { credentials: 'include' });
			if (resp.ok) {
				const data = await resp.json();
				auditEvents = data.events || [];
			}
		} catch { /* best-effort: ignore if endpoint unavailable */ }
	}

	async function loadEncryptionStatus() {
		try {
			const resp = await fetch('/api/security/encryption', { credentials: 'include' });
			if (resp.ok) {
				encryptionStatus = await resp.json();
			}
		} catch { /* best-effort: ignore if endpoint unavailable */ }
		try {
			const resp = await fetch('/api/auth/status', { credentials: 'include' });
			if (resp.ok) {
				const data = await resp.json();
				cookieMode = data.cookie_mode ?? true;
			}
		} catch { /* best-effort: ignore if endpoint unavailable */ }
	}

	function gradeColor(g: string): string {
		if (g.startsWith('A')) return 'var(--oo-sage)';
		if (g.startsWith('B')) return 'var(--oo-tobacco)';
		if (g === 'C') return 'var(--oo-fg-warning)';
		return 'var(--oo-fg-error)';
	}

	function severityColor(sev: string): string {
		if (sev === 'critical') return 'var(--oo-fg-error)';
		if (sev === 'warning') return 'var(--oo-fg-warning)';
		return 'var(--oo-fg-muted)';
	}

	function formatTimestamp(ts: number): string {
		if (!ts) return 'N/A';
		return new Date(ts * 1000).toLocaleString();
	}

	function humanizeName(name: string): string {
		return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
	}
</script>

<div class="space-y-6">
	<!-- Section Navigation -->
	<div class="flex gap-1 rounded-lg p-1" style="background-color: var(--oo-bg-subtle);">
		{#each [
			{ id: 'overview', label: 'Overview' },
			{ id: 'mode', label: 'Security Mode' },
			{ id: 'killswitch', label: 'Kill Switch' },
			{ id: 'plugins', label: 'Plugin Allowlist' },
			{ id: 'encryption', label: 'Encryption' },
			{ id: 'auditlog', label: 'Audit Log' },
		{ id: 'hardening', label: 'Hardening' },
		{ id: 'remote', label: 'Remote Access' },
		] as tab}
			<button
				class="flex-1 px-3 py-1.5 rounded text-xs font-medium transition-colors"
				style="
					background-color: {activeSection === tab.id ? 'var(--oo-card-bg)' : 'transparent'};
					color: {activeSection === tab.id ? 'var(--oo-fg-primary)' : 'var(--oo-fg-muted)'};
					{activeSection === tab.id ? 'box-shadow: 0 1px 2px rgba(0,0,0,0.05);' : ''}
				"
				on:click={() => activeSection = tab.id}
			>
				{tab.label}
			</button>
		{/each}
	</div>

	<!-- Security Mode Section -->
	{#if activeSection === 'mode'}
		<SecurityModePanel />

	<!-- Kill Switch Section -->
	{:else if activeSection === 'killswitch'}
		<SearchKillSwitchPanel />

	<!-- Plugin Allowlist Section -->
	{:else if activeSection === 'plugins'}
		<PluginAllowlistPanel />

	<!-- Encryption Key Ceremony Section -->
	{:else if activeSection === 'encryption'}
		<KeyCeremonyPanel />

	<!-- Audit Chain Log Section -->
	{:else if activeSection === 'auditlog'}
		<AuditChainPanel />

	<!-- Hardening Section -->
	{:else if activeSection === 'hardening'}
		<HardeningPanel />

	<!-- Remote Access Section -->
	{:else if activeSection === 'remote'}
		<RemoteAccessPanel />

	<!-- Overview Section (original content) -->
	{:else}

	<!-- Security Grade -->
	<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
		<div class="flex items-center justify-between mb-3">
			<h3 class="text-base font-semibold" style="color: var(--oo-fg-primary);">Security Posture</h3>
			{#if !loading}
				<div class="flex items-center gap-2">
					<span class="text-3xl font-bold" style="color: {gradeColor(grade)};">{grade}</span>
					<span class="text-sm" style="color: var(--oo-fg-muted);">{score}/{maxScore}</span>
				</div>
			{/if}
		</div>

		{#if loading}
			<p class="text-sm" style="color: var(--oo-fg-muted);">Loading security status...</p>
		{:else if error}
			<p class="text-sm" style="color: var(--oo-fg-error);">{error}</p>
		{:else}
			<!-- Progress bar -->
			<div class="w-full h-2 rounded-full mb-4" style="background-color: var(--oo-bg-tertiary);">
				<div
					class="h-2 rounded-full transition-all duration-500"
					style="width: {(score / maxScore) * 100}%; background-color: {gradeColor(grade)};"
				></div>
			</div>

			<!-- Checks grid -->
			<div class="space-y-2">
				{#each checks as check}
					<div class="flex items-center justify-between py-1.5 text-sm" style="border-bottom: 1px solid var(--oo-bd-subtle);">
						<div class="flex items-center gap-2">
							<span class="w-4 text-center">
								{#if check.passed}
									<span style="color: var(--oo-sage);">&#10003;</span>
								{:else}
									<span style="color: var(--oo-fg-error);">&#10007;</span>
								{/if}
							</span>
							<span style="color: var(--oo-fg-secondary);">{humanizeName(check.name)}</span>
						</div>
						<span class="text-xs font-mono" style="color: var(--oo-fg-muted);">
							{check.points}/{check.max_points}
						</span>
					</div>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Feature Status Cards -->
	<div class="grid grid-cols-1 gap-3" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
		<!-- JWT Cookie Mode -->
		<div class="rounded-lg p-3" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
			<div class="flex items-center gap-2 mb-1">
				<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" style="color: var(--oo-tobacco);">
					<path stroke-linecap="round" stroke-linejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
				</svg>
				<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">JWT Cookies</span>
			</div>
			<p class="text-xs" style="color: {cookieMode ? 'var(--oo-sage)' : 'var(--oo-fg-warning)'};">
				{cookieMode ? 'httpOnly cookies active' : 'Using localStorage (less secure)'}
			</p>
		</div>

		<!-- Encryption -->
		<div class="rounded-lg p-3" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
			<div class="flex items-center gap-2 mb-1">
				<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" style="color: var(--oo-tobacco);">
					<path stroke-linecap="round" stroke-linejoin="round" d="M15.75 5.25a3 3 0 013 3m3 0a6 6 0 01-7.029 5.912c-.563-.097-1.159.026-1.563.43L10.5 17.25H8.25v2.25H6v2.25H2.25v-2.818c0-.597.237-1.17.659-1.591l6.499-6.499c.404-.404.527-1 .43-1.563A6 6 0 1121.75 8.25z" />
				</svg>
				<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">Encryption</span>
			</div>
			{#if encryptionStatus}
				<p class="text-xs" style="color: {encryptionStatus.enabled ? 'var(--oo-sage)' : 'var(--oo-fg-muted)'};">
					{encryptionStatus.enabled ? 'Data-at-rest encryption active' : 'Not configured'}
				</p>
			{:else}
				<p class="text-xs" style="color: var(--oo-fg-muted);">Loading...</p>
			{/if}
		</div>
	</div>

	<!-- Two-Factor Authentication -->
	<div class="rounded-lg p-4 space-y-5" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
		<h3 class="text-base font-semibold" style="color: var(--oo-fg-primary);">Two-Factor Authentication</h3>
		<WebAuthnSetup />
		<hr style="border-color: var(--oo-bd-subtle);" />
		<TOTPSetup />
		<hr style="border-color: var(--oo-bd-subtle);" />
		<RecoveryCodesPanel />
		<hr style="border-color: var(--oo-bd-subtle);" />
		<AppPasswordsPanel />
	</div>

	<!-- Audit Events -->
	<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
		<h3 class="text-base font-semibold mb-3" style="color: var(--oo-fg-primary);">Recent Security Events</h3>

		{#if auditEvents.length === 0}
			<p class="text-sm" style="color: var(--oo-fg-muted);">No recent security events</p>
		{:else}
			<div class="space-y-2 max-h-64 overflow-y-auto">
				{#each auditEvents as event}
					<div class="flex items-start gap-2 py-1.5 text-xs" style="border-bottom: 1px solid var(--oo-bd-subtle);">
						<span class="shrink-0 w-1.5 h-1.5 mt-1 rounded-full" style="background-color: {severityColor(event.severity)};"></span>
						<div class="min-w-0 flex-1">
							<div class="flex items-center gap-2">
								<span class="font-medium" style="color: var(--oo-fg-secondary);">{event.action}</span>
								<span class="px-1 py-0.5 rounded text-xs" style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-muted);">{event.source}</span>
							</div>
							{#if event.timestamp}
								<span style="color: var(--oo-fg-faint);">{formatTimestamp(event.timestamp)}</span>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>

	{/if}
</div>
