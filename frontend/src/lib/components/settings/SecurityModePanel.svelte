<!--
  SecurityModePanel.svelte
  Daily/Bulbe dual-mode security toggle with ceremony-gated degradation.

  Features:
  - Mode indicator with concentric rings animation for Bulbe
  - One-click escalation to Bulbe
  - Multi-factor downgrade ceremony with cooldown timer
  - Policy comparison table
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		getSecurityMode,
		escalateToBulbe,
		requestDowngrade,
		getDowngradeStatus,
		getVisualCode,
		confirmDowngrade,
		cancelDowngrade,
	} from '$lib/api/securityMode';
	import type { SecurityModeStatus, PendingDowngrade, ModePolicy } from '$lib/api/securityMode';
	import { securityModeStatus } from '$lib/stores/securityMode';

	let status: SecurityModeStatus = { mode: 'daily', available: false };
	let loading = true;
	let error = '';
	let actionLoading = false;

	// Downgrade ceremony state
	let downgradeStep: 'idle' | 'pending' | 'confirm' = 'idle';
	let pendingDowngrade: PendingDowngrade | null = null;
	let cooldownRemaining = 0;
	let visualCode = '';
	let confirmCode = '';
	let confirmPassword = '';
	let confirmTwoFa = '';
	let pendingRequestId = '';
	let confirmError = '';

	// Policy display toggle
	let showPolicy = false;

	let cooldownInterval: ReturnType<typeof setInterval> | null = null;

	onMount(async () => {
		await loadStatus();
		loading = false;
	});

	onDestroy(() => {
		if (cooldownInterval) clearInterval(cooldownInterval);
	});

	async function loadStatus() {
		try {
			status = await getSecurityMode();
			securityModeStatus.set(status);
			if (status.pending_downgrade?.pending) {
				downgradeStep = 'pending';
				pendingDowngrade = status.pending_downgrade;
				cooldownRemaining = pendingDowngrade.cooldown_remaining ?? 0;
				startCooldownTimer();
				await fetchVisualCode();
			}
		} catch (e: any) {
			error = e.message || 'Failed to load security mode';
		}
	}

	function startCooldownTimer() {
		if (cooldownInterval) clearInterval(cooldownInterval);
		// Poll the server as the source of truth instead of trusting a local
		// countdown. A local timer drifts when the tab is backgrounded (browsers
		// throttle setInterval), which previously let the request expire before
		// the single check ran, leaving the UI stuck. Polling re-syncs the
		// remaining time and reliably catches cooldown completion.
		const poll = async () => {
			try {
				const ds = await getDowngradeStatus();
				if (!ds.pending) {
					if (cooldownInterval) clearInterval(cooldownInterval);
					if (downgradeStep === 'pending') {
						resetDowngradeState();
						error = 'Downgrade request expired. Please start again.';
					}
					return;
				}
				pendingDowngrade = ds;
				if (ds.cooldown_complete) {
					// Cooldown done: reveal the confirmation form. The template
					// renders that form from inside the 'pending' block when
					// cooldownRemaining reaches 0 (its {:else} branch). There is
					// no separate 'confirm' render block, so we must stay in
					// 'pending' and zero the counter -- switching step would blank
					// the panel ("nothing after the countdown").
					cooldownRemaining = 0;
					if (cooldownInterval) clearInterval(cooldownInterval);
				} else {
					cooldownRemaining = Math.ceil(ds.cooldown_remaining ?? 0);
				}
			} catch (_) {
				// Transient error: keep polling.
			}
		};
		void poll();
		cooldownInterval = setInterval(poll, 2000);
	}

	async function fetchVisualCode() {
		try {
			const resp = await getVisualCode();
			visualCode = resp.visual_code;
		} catch (_) {
			visualCode = '';
		}
	}

	async function handleEscalate() {
		actionLoading = true;
		error = '';
		try {
			const result = await escalateToBulbe();
			if (result.success) {
				await loadStatus();
			} else {
				error = result.message || 'Escalation failed';
			}
		} catch (e: any) {
			error = e.message || 'Escalation failed';
		} finally {
			actionLoading = false;
		}
	}

	async function handleRequestDowngrade() {
		actionLoading = true;
		error = '';
		try {
			const result = await requestDowngrade();
			if (result.success && result.pending) {
				pendingRequestId = result.request_id || '';
				cooldownRemaining = result.cooldown_seconds || 300;
				downgradeStep = 'pending';
				startCooldownTimer();
				await fetchVisualCode();
			} else if (result.success && !result.pending) {
				await loadStatus();
			} else {
				error = result.message || 'Downgrade request failed';
			}
		} catch (e: any) {
			error = e.message || 'Downgrade request failed';
		} finally {
			actionLoading = false;
		}
	}

	async function handleConfirmDowngrade() {
		actionLoading = true;
		confirmError = '';
		try {
			const result = await confirmDowngrade({
				request_id: pendingRequestId,
				visual_code: confirmCode,
				password: confirmPassword,
				two_fa_code: confirmTwoFa || null,
			});
			if (result.success) {
				resetDowngradeState();
				await loadStatus();
			} else {
				confirmError = result.message || 'Confirmation failed';
			}
		} catch (e: any) {
			confirmError = e.message || 'Confirmation failed';
		} finally {
			actionLoading = false;
		}
	}

	async function handleCancelDowngrade() {
		try {
			await cancelDowngrade();
		} catch (_) { /* ignore */ }
		resetDowngradeState();
		await loadStatus();
	}

	function resetDowngradeState() {
		downgradeStep = 'idle';
		pendingDowngrade = null;
		cooldownRemaining = 0;
		visualCode = '';
		confirmCode = '';
		confirmPassword = '';
		confirmTwoFa = '';
		pendingRequestId = '';
		confirmError = '';
		if (cooldownInterval) clearInterval(cooldownInterval);
	}

	function formatTime(seconds: number): string {
		const m = Math.floor(seconds / 60);
		const s = Math.floor(seconds % 60);
		return `${m}:${s.toString().padStart(2, '0')}`;
	}

	$: isBulbe = status.mode === 'bulbe';
	$: policy = status.policy;
</script>

{#if loading}
	<div class="flex items-center gap-2 p-4" style="color: var(--oo-fg-muted);">
		Loading security mode...
	</div>
{:else if !status.available}
	<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
		<p class="text-sm" style="color: var(--oo-fg-muted);">Security mode system not available.</p>
	</div>
{:else}
	<!-- Mode Indicator -->
	<div class="rounded-lg p-5" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
		<div class="flex items-center justify-between mb-4">
			<h3 class="text-base font-semibold" style="color: var(--oo-fg-primary);">
				Security Mode
			</h3>
			<button
				class="text-xs px-2 py-1 rounded"
				style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
				on:click={() => showPolicy = !showPolicy}
			>
				{showPolicy ? 'Hide' : 'Show'} policy
			</button>
		</div>

		<!-- Mode Badge with Concentric Rings -->
		<div class="flex items-center gap-4 mb-4">
			<div class="mode-indicator" class:bulbe={isBulbe}>
				<div class="mode-ring ring-outer" class:active={isBulbe}></div>
				<div class="mode-ring ring-middle" class:active={isBulbe}></div>
				<div class="mode-ring ring-inner" class:active={isBulbe}></div>
				<div class="mode-core" class:bulbe={isBulbe}>
					{#if isBulbe}B{:else}D{/if}
				</div>
			</div>
			<div>
				<span class="text-lg font-bold" style="color: {isBulbe ? 'var(--oo-fg-error)' : 'var(--oo-sage)'};">
					{isBulbe ? 'Bulbe' : 'Daily'}
				</span>
				<p class="text-xs mt-0.5" style="color: var(--oo-fg-muted);">
					{#if isBulbe}
						Maximum security active. Every layer is enforced.
					{:else}
						Standard security. Strong baseline, frictionless use.
					{/if}
				</p>
			</div>
		</div>

		<!-- Integrity Status -->
		{#if status.lockfile_exists}
			<div class="flex gap-3 text-xs mb-4" style="color: var(--oo-fg-muted);">
				<span>
					Sources:
					<span style="color: {status.sources_agree ? 'var(--oo-sage)' : 'var(--oo-fg-error)'};">
						{status.sources_agree ? 'aligned' : 'MISMATCH'}
					</span>
				</span>
				<span>
					HMAC:
					<span style="color: {status.hmac_valid ? 'var(--oo-sage)' : 'var(--oo-fg-error)'};">
						{status.hmac_valid ? 'valid' : 'INVALID'}
					</span>
				</span>
			</div>
		{/if}

		<!-- Error -->
		{#if error}
			<div class="rounded p-2 mb-3 text-sm" style="background-color: var(--oo-bg-error); color: var(--oo-fg-error);">
				{error}
			</div>
		{/if}

		<!-- Action Buttons -->
		{#if downgradeStep === 'idle'}
			<div class="flex gap-2">
				{#if !isBulbe}
					<button
						class="px-4 py-2 rounded text-sm font-medium transition-colors"
						style="background-color: var(--oo-fg-error); color: white;"
						on:click={handleEscalate}
						disabled={actionLoading}
					>
						{actionLoading ? 'Escalating...' : 'Escalate to Bulbe'}
					</button>
				{:else}
					<button
						class="px-4 py-2 rounded text-sm font-medium transition-colors"
						style="background-color: var(--oo-bg-subtle); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-subtle);"
						on:click={handleRequestDowngrade}
						disabled={actionLoading}
					>
						{actionLoading ? 'Requesting...' : 'Request Downgrade to Daily'}
					</button>
				{/if}
			</div>
		{/if}

		<!-- Downgrade Pending Banner -->
		{#if downgradeStep === 'pending'}
			<div class="rounded-lg p-4 mt-3" style="background-color: var(--oo-bg-warning); border: 1px solid var(--oo-fg-warning);">
				<div class="flex items-center justify-between mb-2">
					<span class="text-sm font-semibold" style="color: var(--oo-fg-warning);">
						Security downgrade pending
					</span>
					<span class="text-sm font-mono" style="color: var(--oo-fg-warning);">
						{formatTime(cooldownRemaining)}
					</span>
				</div>

				{#if cooldownRemaining > 0}
					<p class="text-xs mb-2" style="color: var(--oo-fg-muted);">
						Cooldown active. You can cancel anytime.
					</p>
					<!-- Visual code (DOM-only, human-readable) -->
					{#if visualCode}
						<!-- data-security-code for DOM injection pattern -->
						<template data-security-code={visualCode}></template>
						<div class="visual-code-display mt-2 mb-2 text-center select-none">
							<span class="text-2xl font-mono tracking-widest" style="
								color: var(--oo-fg-primary);
								background: var(--oo-bg-subtle);
								padding: 0.5rem 1rem;
								border-radius: 0.5rem;
								letter-spacing: 0.3em;
								font-family: 'Courier New', monospace;
							">
								{visualCode}
							</span>
						</div>
						<p class="text-xs text-center" style="color: var(--oo-fg-muted);">
							You will need this code to confirm the downgrade.
						</p>
					{/if}
				{:else}
					<p class="text-xs mb-3" style="color: var(--oo-fg-muted);">
						Cooldown complete. Enter the confirmation code and your credentials.
					</p>
					{#if visualCode}
						<div class="text-center select-none mb-3">
							<span class="text-2xl font-mono tracking-widest" style="color: var(--oo-fg-primary); background: var(--oo-bg-subtle); padding: 0.5rem 1rem; border-radius: 0.5rem; letter-spacing: 0.3em; font-family: 'Courier New', monospace;">
								{visualCode}
							</span>
						</div>
					{/if}
					<!-- Confirmation form -->
					<div class="space-y-2">
						<input
							type="text"
							class="w-full px-3 py-2 rounded text-sm"
							style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
							placeholder="6-digit confirmation code"
							aria-label="Six-digit confirmation code"
							maxlength="6"
							bind:value={confirmCode}
						/>
						<input
							type="password"
							class="w-full px-3 py-2 rounded text-sm"
							style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
							placeholder="Current password"
							aria-label="Current password"
							bind:value={confirmPassword}
						/>
						<input
							type="text"
							class="w-full px-3 py-2 rounded text-sm"
							style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
							placeholder="2FA code (if enabled)"
							aria-label="Two-factor authentication code"
							bind:value={confirmTwoFa}
						/>
						{#if confirmError}
							<p class="text-xs" style="color: var(--oo-fg-error);">{confirmError}</p>
						{/if}
						<div class="flex gap-2">
							<button
								class="px-4 py-2 rounded text-sm font-medium"
								style="background-color: var(--oo-fg-warning); color: white;"
								on:click={handleConfirmDowngrade}
								disabled={actionLoading || !confirmCode || !confirmPassword}
							>
								{actionLoading ? 'Confirming...' : 'Confirm Downgrade'}
							</button>
							<button
								class="px-4 py-2 rounded text-sm"
								style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
								on:click={handleCancelDowngrade}
							>
								Cancel
							</button>
						</div>
					</div>
				{/if}

				{#if cooldownRemaining > 0}
					<button
						class="mt-2 text-xs underline"
						style="color: var(--oo-fg-muted);"
						on:click={handleCancelDowngrade}
					>
						Cancel downgrade
					</button>
				{/if}
			</div>
		{/if}

		<!-- Policy Table -->
		{#if showPolicy && policy}
			<div class="mt-4 rounded" style="border: 1px solid var(--oo-bd-subtle); overflow: hidden;">
				<table class="w-full text-xs">
					<thead>
						<tr style="background-color: var(--oo-bg-subtle);">
							<th class="text-left p-2 font-medium" style="color: var(--oo-fg-secondary);">Feature</th>
							<th class="text-center p-2 font-medium" style="color: var(--oo-fg-secondary);">Status</th>
						</tr>
					</thead>
					<tbody>
						{#each [
							['Web search', policy.web_search_allowed ? 'Allowed' : 'Disabled'],
							['DB encryption', policy.db_encryption_required ? 'Required' : 'Optional'],
							['2FA', policy.two_fa_required ? 'Required' : 'Optional'],
							['Plugin allowlist', policy.plugin_allowlist_required ? 'Required' : 'Off'],
							['Sandbox bwrap', policy.sandbox_bwrap_required ? 'Required' : 'Optional'],
							['Session timeout', `${Math.floor(policy.session_timeout / 60)} min`],
							['Backup encryption', policy.backup_encryption_required ? 'Required' : 'Optional'],
							['Cookie SameSite', policy.cookie_samesite],
							['Tool call approval', policy.tool_call_approval_required ? 'Required' : 'Off'],
							['Rate limit', `${policy.rate_limit_max_attempts}/${Math.floor(policy.rate_limit_window / 60)}min`],
							['Bearer auth', policy.bearer_auth_allowed ? 'Allowed' : 'Disabled'],
						] as [label, value]}
							<tr style="border-top: 1px solid var(--oo-bd-subtle);">
								<td class="p-2" style="color: var(--oo-fg-primary);">{label}</td>
								<td class="p-2 text-center" style="color: var(--oo-fg-muted);">{value}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</div>
{/if}

<style>
	.mode-indicator {
		position: relative;
		width: 56px;
		height: 56px;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.mode-ring {
		position: absolute;
		border-radius: 50%;
		border: 2px solid transparent;
		transition: all 0.5s ease;
	}

	.ring-outer {
		width: 56px;
		height: 56px;
		top: 0;
		left: 0;
	}

	.ring-middle {
		width: 44px;
		height: 44px;
		top: 6px;
		left: 6px;
	}

	.ring-inner {
		width: 32px;
		height: 32px;
		top: 12px;
		left: 12px;
	}

	.ring-outer.active {
		border-color: var(--oo-fg-error);
		animation: pulse-ring 2s ease-in-out infinite;
	}

	.ring-middle.active {
		border-color: var(--oo-fg-error);
		opacity: 0.7;
		animation: pulse-ring 2s ease-in-out infinite 0.3s;
	}

	.ring-inner.active {
		border-color: var(--oo-fg-error);
		opacity: 0.5;
		animation: pulse-ring 2s ease-in-out infinite 0.6s;
	}

	.mode-core {
		position: relative;
		z-index: 1;
		width: 24px;
		height: 24px;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 0.75rem;
		font-weight: 700;
		background-color: var(--oo-bg-subtle);
		color: var(--oo-sage);
		border: 2px solid var(--oo-sage);
		transition: all 0.3s ease;
	}

	.mode-core.bulbe {
		background-color: var(--oo-fg-error);
		color: white;
		border-color: var(--oo-fg-error);
	}

	@keyframes pulse-ring {
		0%, 100% { transform: scale(1); opacity: 0.5; }
		50% { transform: scale(1.05); opacity: 1; }
	}

	.visual-code-display {
		user-select: none;
		-webkit-user-select: none;
	}
</style>
