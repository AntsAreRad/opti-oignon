<!--
  NotificationCenter.svelte
  Bell icon with unread count badge, in Preferences.
  Dropdown panel showing notification history with timestamps.
  Mark as read, mark all read, clear history actions.
  Uses --oo-* CSS variables exclusively.

  Its panel is open while its store says so (lib/stores/notificationCenter.ts):
  the bell opens and shuts it, and so does the "Show notifications" command,
  from a key or from the command palette. Leaving the page shuts it.

  Drawn to the surface rules: the panel on the second surface with its
  edge and a shadow token, space rather than a line under its head, a wash
  under the pointer; an unread entry says so by its weight (and in words to
  a screen reader), not by a fill. Each kind of entry takes its ink from the
  status tokens.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		notificationHistory,
		unreadCount,
		markNotificationRead,
		markAllRead,
		clearNotificationHistory,
	} from '$lib/stores/notifications';
	import type { ToastType } from '$lib/stores/notifications';
	import { notificationCenter, closeNotifications, toggleNotifications } from '$lib/stores/notificationCenter';

	// Opened, from the bell or from the command, the history is read.
	$: if ($notificationCenter) markAllRead();

	function handleClickOutside(event: MouseEvent) {
		const target = event.target as HTMLElement;
		if ($notificationCenter && !target.closest('.notif-center-wrapper')) {
			closeNotifications();
		}
	}

	function typeIcon(type: ToastType): string {
		switch (type) {
			case 'success': return 'M5 13l4 4L19 7';
			case 'error': return 'M6 18L18 6M6 6l12 12';
			case 'warning': return 'M12 9v4m0 4h.01M12 2l10 18H2L12 2z';
			default: return 'M13 16h-1v-4h-1m1-4h.01';
		}
	}

	function formatTime(timestamp: number): string {
		const now = Date.now();
		const diff = now - timestamp;
		if (diff < 60_000) return 'just now';
		if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
		if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
		const d = new Date(timestamp);
		return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
	}

	onMount(() => {
		document.addEventListener('click', handleClickOutside, true);
	});

	onDestroy(() => {
		closeNotifications();
		if (typeof document === 'undefined') return;
		document.removeEventListener('click', handleClickOutside, true);
	});
</script>

<div class="notif-center-wrapper">
	<button
		class="notif-bell-btn"
		aria-expanded={$notificationCenter}
		on:click={toggleNotifications}
		title="Notifications{$unreadCount > 0 ? ` (${$unreadCount} unread)` : ''}"
		aria-label="Notifications{$unreadCount > 0 ? `, ${$unreadCount} unread` : ''}"
	>
		<!-- Bell icon -->
		<svg class="notif-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
			<path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 01-3.46 0" />
		</svg>
		<!-- Unread badge -->
		{#if $unreadCount > 0}
			<span class="notif-badge">
				{$unreadCount > 9 ? '9+' : $unreadCount}
			</span>
		{/if}
	</button>

	<!-- Dropdown panel -->
	{#if $notificationCenter}
		<div class="notif-panel">
			<div class="notif-panel-header">
				<span class="notif-panel-title">Notifications</span>
				<div class="notif-panel-actions">
					{#if $notificationHistory.length > 0}
						<button class="notif-action-btn" on:click={clearNotificationHistory} title="Clear all">
							Clear
						</button>
					{/if}
				</div>
			</div>

			<div class="notif-panel-list">
				{#if $notificationHistory.length === 0}
					<div class="notif-empty">No notifications yet</div>
				{:else}
					{#each $notificationHistory as notif (notif.id)}
						<div class="notif-item" class:notif-unread={!notif.read}>
							<svg class="notif-item-icon" data-type={notif.type}
								fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
								<path d={typeIcon(notif.type)} />
							</svg>
							<div class="notif-item-content">
								<span class="notif-item-msg">
									{notif.message}{#if !notif.read}<span class="oo-sr-only">{' (unread)'}</span>{/if}
								</span>
								<span class="notif-item-time">{formatTime(notif.timestamp)}</span>
							</div>
						</div>
					{/each}
				{/if}
			</div>
		</div>
	{/if}
</div>

<style>
	.notif-center-wrapper {
		position: relative;
		display: inline-flex;
		align-items: center;
	}

	.notif-bell-btn {
		position: relative;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		min-width: 36px;
		min-height: 36px;
		padding: var(--oo-space-2);
		border-radius: var(--oo-radius-full);
		border: none;
		background: transparent;
		cursor: pointer;
		color: var(--oo-fg-secondary);
		transition: background-color var(--oo-motion-fast) var(--oo-ease-default);
	}

	.notif-bell-btn:hover {
		background-color: var(--oo-bg-hover);
		color: var(--oo-fg-primary);
	}

	.notif-icon {
		width: 18px;
		height: 18px;
	}

	.notif-badge {
		position: absolute;
		top: 0;
		right: 0;
		min-width: 16px;
		height: 16px;
		padding: 0 4px;
		border-radius: var(--oo-radius-full);
		background-color: var(--oo-error);
		color: var(--oo-fg-on-semantic);
		font-size: var(--oo-text-2xs);
		font-weight: 700;
		display: flex;
		align-items: center;
		justify-content: center;
		line-height: 1;
	}

	.notif-panel {
		position: absolute;
		top: calc(100% + 6px);
		right: 0;
		z-index: var(--oo-z-overlay);
		width: min(20rem, calc(100vw - 32px));
		max-height: 400px;
		border-radius: var(--oo-radius-lg);
		background-color: var(--oo-bg-overlay);
		border: 1px solid var(--oo-edge);
		box-shadow: var(--oo-shadow-md);
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.notif-panel-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: var(--oo-space-3) var(--oo-space-4) var(--oo-space-2);
	}

	.notif-panel-title {
		font-size: var(--oo-text-sm);
		font-weight: 600;
		color: var(--oo-fg-primary);
	}

	.notif-panel-actions {
		display: flex;
		gap: var(--oo-space-2);
	}

	.notif-action-btn {
		min-height: 28px;
		padding: 0 var(--oo-space-3);
		border-radius: var(--oo-radius-full);
		border: none;
		background: transparent;
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-xs);
		cursor: pointer;
	}

	.notif-action-btn:hover {
		background-color: var(--oo-bg-hover);
		color: var(--oo-fg-primary);
	}

	.notif-panel-list {
		flex: 1;
		overflow-y: auto;
		padding: 0 var(--oo-space-1) var(--oo-space-2);
	}

	.notif-empty {
		padding: var(--oo-space-6) var(--oo-space-4);
		text-align: center;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}

	.notif-item {
		display: flex;
		align-items: flex-start;
		gap: var(--oo-space-3);
		padding: var(--oo-space-2) var(--oo-space-3);
		border-radius: var(--oo-radius-md);
	}

	.notif-item:hover {
		background-color: var(--oo-bg-hover);
	}

	.notif-unread .notif-item-msg {
		color: var(--oo-fg-primary);
		font-weight: 600;
	}

	.notif-item-icon {
		width: 14px;
		height: 14px;
		flex-shrink: 0;
		margin-top: 2px;
		color: var(--oo-fg-muted);
	}
	.notif-item-icon[data-type='success'] {
		color: var(--oo-success);
	}
	.notif-item-icon[data-type='error'] {
		color: var(--oo-error);
	}
	.notif-item-icon[data-type='warning'] {
		color: var(--oo-warning);
	}

	.notif-item-content {
		flex: 1;
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.notif-item-msg {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
		word-break: break-word;
		line-height: var(--oo-leading-snug);
	}

	.notif-item-time {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
	}
</style>
