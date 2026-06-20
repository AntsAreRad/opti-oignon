<!--
  Toast.svelte (lib/ds) -- toast container (spec 10.6), refactor of the
  previous ui/Toast.svelte. Subscribes to the notifications store, adds
  optional title and a retry action button (dismisses on success), and
  uses role="status" (success/info) or role="alert" (warning/error).
  Toasts are archived in the notifications store (NotificationCenter).
  Stacked bottom-right on desktop, full-width bottom on mobile.
-->
<script lang="ts">
	import { toasts, dismissToast } from '$lib/stores/notifications';
	import type { ToastItem, ToastType } from '$lib/stores/notifications';
	import Icon from './Icon.svelte';

	const ICON: Record<ToastType, string> = {
		success: 'check',
		error: 'x',
		warning: 'alert-triangle',
		info: 'info'
	};

	function roleFor(type: ToastType): 'status' | 'alert' {
		return type === 'warning' || type === 'error' ? 'alert' : 'status';
	}

	let running: Record<string, boolean> = {};

	async function runAction(toast: ToastItem) {
		if (!toast.action || running[toast.id]) return;
		running = { ...running, [toast.id]: true };
		try {
			await toast.action.run();
			dismissToast(toast.id);
		} catch {
			// Keep the toast visible so the user can retry.
		} finally {
			running = { ...running, [toast.id]: false };
		}
	}
</script>

{#if $toasts.length > 0}
	<div
		class="oo-toasts"
		aria-live="polite"
		aria-relevant="additions removals"
	>
		{#each $toasts as toast (toast.id)}
			<div
				class="oo-toast {toast.dismissing ? 'oo-toast-exit' : 'oo-toast-enter'}"
				data-type={toast.type}
				role={roleFor(toast.type)}
			>
				<span class="oo-toast-icon" aria-hidden="true">
					<Icon name={ICON[toast.type]} size="sm" />
				</span>
				<div class="oo-toast-body">
					{#if toast.title}
						<p class="oo-toast-title">{toast.title}</p>
					{/if}
					<p class="oo-toast-message">{toast.message}</p>
				</div>
				{#if toast.action}
					<button
						type="button"
						class="oo-toast-action"
						disabled={running[toast.id]}
						on:click={() => runAction(toast)}
					>
						{toast.action.label}
					</button>
				{/if}
				{#if toast.dismissible !== false}
					<button
						type="button"
						class="oo-toast-dismiss"
						aria-label="Dismiss notification"
						on:click={() => dismissToast(toast.id)}
					>
						<Icon name="x" size="sm" />
					</button>
				{/if}
			</div>
		{/each}
	</div>
{/if}

<style>
	.oo-toasts {
		position: fixed;
		bottom: var(--oo-space-5);
		right: var(--oo-space-5);
		z-index: var(--oo-z-toast);
		display: flex;
		flex-direction: column-reverse;
		gap: var(--oo-space-3);
		max-width: 24rem;
		width: 100%;
		pointer-events: none;
	}
	.oo-toast {
		display: flex;
		align-items: flex-start;
		gap: var(--oo-space-3);
		padding: var(--oo-space-3) var(--oo-space-4);
		border-radius: var(--oo-radius-lg);
		border: 1px solid var(--oo-bd-default);
		background-color: var(--oo-bg-elevated);
		box-shadow: var(--oo-shadow-lg);
		pointer-events: auto;
	}
	.oo-toast[data-type='success'] {
		border-color: var(--oo-success-bd);
	}
	.oo-toast[data-type='error'] {
		border-color: var(--oo-error-bd);
	}
	.oo-toast[data-type='warning'] {
		border-color: var(--oo-warning-bd);
	}
	.oo-toast[data-type='info'] {
		border-color: var(--oo-info-bd);
	}
	.oo-toast-icon {
		flex-shrink: 0;
		margin-top: 1px;
	}
	.oo-toast[data-type='success'] .oo-toast-icon {
		color: var(--oo-success);
	}
	.oo-toast[data-type='error'] .oo-toast-icon {
		color: var(--oo-error);
	}
	.oo-toast[data-type='warning'] .oo-toast-icon {
		color: var(--oo-warning);
	}
	.oo-toast[data-type='info'] .oo-toast-icon {
		color: var(--oo-info);
	}
	.oo-toast-body {
		flex: 1;
		min-width: 0;
	}
	.oo-toast-title {
		margin: 0 0 2px;
		font-size: var(--oo-text-sm);
		font-weight: 600;
		color: var(--oo-fg-primary);
	}
	.oo-toast-message {
		margin: 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
		word-break: break-word;
	}
	.oo-toast-action {
		flex-shrink: 0;
		align-self: center;
		border: 1px solid var(--oo-bd-strong);
		border-radius: var(--oo-radius-sm);
		background: transparent;
		color: var(--oo-acc-500);
		font-size: var(--oo-text-xs);
		font-weight: 500;
		padding: var(--oo-space-1) var(--oo-space-3);
		cursor: pointer;
	}
	.oo-toast-action:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}
	.oo-toast-dismiss {
		flex-shrink: 0;
		display: inline-flex;
		border: none;
		background: transparent;
		color: var(--oo-fg-muted);
		cursor: pointer;
		padding: 2px;
		opacity: 0.7;
	}
	.oo-toast-dismiss:hover {
		opacity: 1;
		color: var(--oo-fg-primary);
	}

	@keyframes oo-toast-in {
		from {
			opacity: 0;
			transform: translateY(0.75rem) scale(0.97);
		}
		to {
			opacity: 1;
			transform: translateY(0) scale(1);
		}
	}
	@keyframes oo-toast-out {
		from {
			opacity: 1;
		}
		to {
			opacity: 0;
			transform: translateY(0.5rem) scale(0.97);
		}
	}
	.oo-toast-enter {
		animation: oo-toast-in var(--oo-motion-normal) var(--oo-ease-default) forwards;
	}
	.oo-toast-exit {
		animation: oo-toast-out var(--oo-motion-normal) var(--oo-ease-default) forwards;
		pointer-events: none;
	}

	@media (max-width: 640px) {
		.oo-toasts {
			left: var(--oo-space-4);
			right: var(--oo-space-4);
			bottom: var(--oo-space-4);
			max-width: none;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.oo-toast-enter,
		.oo-toast-exit {
			animation: none;
		}
		.oo-toast-exit {
			opacity: 0;
		}
	}
</style>
