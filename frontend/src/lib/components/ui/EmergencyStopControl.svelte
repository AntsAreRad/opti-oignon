<!--
  EmergencyStopControl.svelte (S215)
  Emergency-stop panic control in the header status cluster: makes the
  machine quiet immediately (cancel generations and runs, unload models,
  destroy sandboxes, stop the Veilid node) and offers resume in place.
  Two-step confirmation (click, then an explicit choice in the popover)
  with the two arbitrated actions: "Stop compute" (primary) and
  "Stop compute + Bulbe" (the no-ceremony escalation direction).
  The stopped state is a persistent pill visible app-wide (the header
  cluster renders on every route). An availability control, distinct from
  the search kill switch: resume needs no ceremony (auth still required).
  Uses --oo-* CSS variables exclusively.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		getEmergencyStopStatus,
		engageEmergencyStop,
		resumeFromEmergencyStop,
		type EmergencyStopStatus
	} from '$lib/api/estop';

	let available = false;
	let stopped = false;
	let confirming = false;
	let busy = false;
	let errorText = '';
	let announce = '';
	let pollTimer: ReturnType<typeof setInterval> | null = null;

	async function refresh() {
		try {
			const status: EmergencyStopStatus = await getEmergencyStopStatus();
			available = !!status.available;
			stopped = !!status.stopped;
		} catch {
			// Status endpoint unreachable: keep the last known state.
		}
	}

	async function engage(dropToBulbe: boolean) {
		if (busy) return;
		busy = true;
		errorText = '';
		try {
			const result = await engageEmergencyStop(dropToBulbe);
			stopped = !!result.stopped;
			confirming = false;
			announce = 'Emergency stop engaged';
			if (result.failed_steps && result.failed_steps.length > 0) {
				errorText = `Stopped with step errors: ${result.failed_steps.join(', ')}`;
			}
		} catch {
			errorText = 'Emergency stop request failed';
		} finally {
			busy = false;
		}
	}

	async function resume() {
		if (busy) return;
		busy = true;
		errorText = '';
		try {
			const result = await resumeFromEmergencyStop();
			stopped = !!result.stopped;
			announce = 'Resumed';
			if (result.failed_steps && result.failed_steps.length > 0) {
				errorText = `Resumed with step errors: ${result.failed_steps.join(', ')}`;
			}
		} catch {
			errorText = 'Resume request failed';
		} finally {
			busy = false;
		}
	}

	function toggleConfirm() {
		confirming = !confirming;
		errorText = '';
	}

	function handleClickOutside(event: MouseEvent) {
		const target = event.target as HTMLElement;
		if (confirming && !target.closest('.estop-wrapper')) {
			confirming = false;
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (confirming && event.key === 'Escape') {
			confirming = false;
		}
	}

	onMount(() => {
		refresh();
		pollTimer = setInterval(refresh, 10_000);
		document.addEventListener('click', handleClickOutside, true);
		document.addEventListener('keydown', handleKeydown, true);
	});

	onDestroy(() => {
		if (pollTimer) clearInterval(pollTimer);
		document.removeEventListener('click', handleClickOutside, true);
		document.removeEventListener('keydown', handleKeydown, true);
	});
</script>

{#if available}
	<div class="estop-wrapper">
		<span class="estop-announce" aria-live="polite">{announce}</span>
		{#if stopped}
			<span class="estop-stopped-pill" role="status">
				<svg
					class="estop-icon"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					stroke-linejoin="round"
					aria-hidden="true"
				>
					<rect x="6" y="6" width="12" height="12" rx="1" />
				</svg>
				Stopped
				<button
					class="estop-resume-btn"
					on:click={resume}
					disabled={busy}
					aria-label="Resume from emergency stop"
				>
					Resume
				</button>
			</span>
		{:else}
			<button
				class="estop-btn"
				on:click={toggleConfirm}
				disabled={busy}
				title="Emergency stop"
				aria-label="Emergency stop"
				aria-expanded={confirming}
			>
				<svg
					class="estop-icon"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					stroke-linejoin="round"
					aria-hidden="true"
				>
					<circle cx="12" cy="12" r="9" />
					<rect x="9" y="9" width="6" height="6" rx="0.5" />
				</svg>
			</button>
			{#if confirming}
				<div class="estop-popover" role="dialog" aria-label="Confirm emergency stop">
					<p class="estop-popover-title">Emergency stop</p>
					<p class="estop-popover-text">
						Cancels generations and runs, unloads models, destroys
						sandboxes and stops the sync node. Resume needs no ceremony.
					</p>
					<button
						class="estop-action-primary"
						on:click={() => engage(false)}
						disabled={busy}
					>
						Stop compute
					</button>
					<button
						class="estop-action-secondary"
						on:click={() => engage(true)}
						disabled={busy}
					>
						Stop compute + Bulbe
					</button>
					<button class="estop-action-cancel" on:click={toggleConfirm} disabled={busy}>
						Cancel
					</button>
				</div>
			{/if}
		{/if}
		{#if errorText}
			<span class="estop-error" role="alert">{errorText}</span>
		{/if}
	</div>
{/if}

<style>
	.estop-wrapper {
		position: relative;
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-1);
	}
	.estop-announce {
		position: absolute;
		width: 1px;
		height: 1px;
		overflow: hidden;
		clip: rect(0 0 0 0);
		white-space: nowrap;
	}
	.estop-icon {
		width: 16px;
		height: 16px;
	}
	.estop-btn {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		padding: var(--oo-space-1);
		border: 1px solid var(--oo-bd-subtle);
		border-radius: 6px;
		background: transparent;
		color: var(--oo-error);
		cursor: pointer;
	}
	.estop-btn:hover {
		background: var(--oo-error-bg);
	}
	.estop-btn:disabled {
		opacity: 0.6;
		cursor: default;
	}
	.estop-stopped-pill {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-1);
		padding: var(--oo-space-1) var(--oo-space-2);
		border: 1px solid var(--oo-error);
		border-radius: 999px;
		background: var(--oo-error-bg);
		color: var(--oo-error);
		font-size: var(--oo-text-2xs);
		letter-spacing: var(--oo-tracking-wide);
		text-transform: uppercase;
	}
	.estop-resume-btn {
		padding: 0 var(--oo-space-2);
		border: 1px solid var(--oo-error);
		border-radius: 999px;
		background: var(--oo-error);
		color: var(--oo-fg-on-semantic);
		font-size: var(--oo-text-2xs);
		cursor: pointer;
	}
	.estop-resume-btn:disabled {
		opacity: 0.6;
		cursor: default;
	}
	.estop-popover {
		position: absolute;
		top: calc(100% + var(--oo-space-2));
		right: 0;
		z-index: 50;
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		width: 240px;
		padding: var(--oo-space-3);
		border: 1px solid var(--oo-bd-subtle);
		border-radius: 8px;
		background: var(--oo-bg-elevated);
		color: var(--oo-fg-primary);
	}
	.estop-popover-title {
		margin: 0;
		font-size: var(--oo-text-2xs);
		letter-spacing: var(--oo-tracking-wide);
		text-transform: uppercase;
		color: var(--oo-fg-secondary);
	}
	.estop-popover-text {
		margin: 0;
		font-size: var(--oo-text-2xs);
		color: var(--oo-fg-tertiary);
	}
	.estop-action-primary,
	.estop-action-secondary,
	.estop-action-cancel {
		padding: var(--oo-space-1) var(--oo-space-2);
		border-radius: 6px;
		font-size: var(--oo-text-2xs);
		cursor: pointer;
	}
	.estop-action-primary {
		border: 1px solid var(--oo-error);
		background: var(--oo-error);
		color: var(--oo-fg-on-semantic);
	}
	.estop-action-secondary {
		border: 1px solid var(--oo-error);
		background: transparent;
		color: var(--oo-error);
	}
	.estop-action-cancel {
		border: 1px solid var(--oo-bd-subtle);
		background: transparent;
		color: var(--oo-fg-secondary);
	}
	.estop-action-primary:disabled,
	.estop-action-secondary:disabled,
	.estop-action-cancel:disabled {
		opacity: 0.6;
		cursor: default;
	}
	.estop-error {
		font-size: var(--oo-text-2xs);
		color: var(--oo-error);
	}
</style>
