<!--
  NotificationCenter.svelte (S107)
  Bell icon with unread count badge in the navbar.
  Dropdown panel showing notification history with timestamps.
  Mark as read, mark all read, clear history actions.
  Uses --oo-* CSS variables exclusively.
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

	let expanded = false;

	function toggle() {
		expanded = !expanded;
		// Mark all as read when opening
		if (expanded) {
			markAllRead();
		}
	}

	function handleClickOutside(event: MouseEvent) {
		const target = event.target as HTMLElement;
		if (expanded && !target.closest('.notif-center-wrapper')) {
			expanded = false;
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

	function typeColor(type: ToastType): string {
		switch (type) {
			case 'success': return 'var(--oo-success)';
			case 'error': return 'var(--oo-error)';
			case 'warning': return 'var(--oo-warning)';
			default: return 'var(--oo-fg-muted)';
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
		document.removeEventListener('click', handleClickOutside, true);
	});
</script>

<div class="notif-center-wrapper">
	<button
		class="notif-bell-btn"
		on:click={toggle}
		title="Notifications{$unreadCount > 0 ? ` (${$unreadCount} unread)` : ''}"
		aria-label="Notifications{$unreadCount > 0 ? `, ${$unreadCount} unread` : ''}"
	>
		<!-- Bell icon -->
		<svg class="notif-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
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
	{#if expanded}
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
							<svg class="notif-item-icon" style="color: {typeColor(notif.type)};"
								fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path d={typeIcon(notif.type)} />
							</svg>
							<div class="notif-item-content">
								<span class="notif-item-msg">{notif.message}</span>
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
		padding: 5px;
		border-radius: 6px;
		border: none;
		background: transparent;
		cursor: pointer;
		color: var(--oo-fg-tertiary);
		transition: background-color 0.15s ease, color 0.15s ease;
	}

	.notif-bell-btn:hover {
		background-color: var(--oo-bg-elevated);
		color: var(--oo-fg-secondary);
	}

	.notif-icon {
		width: 16px;
		height: 16px;
	}

	.notif-badge {
		position: absolute;
		top: -2px;
		right: -4px;
		min-width: 16px;
		height: 16px;
		padding: 0 4px;
		border-radius: 8px;
		background-color: var(--oo-error);
		color: var(--oo-fg-on-semantic);
		font-size: 0.625rem;
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
		z-index: 50;
		width: 320px;
		max-height: 400px;
		border-radius: 10px;
		background-color: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-subtle);
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.notif-panel-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 10px 14px;
		border-bottom: 1px solid var(--oo-bd-subtle);
	}

	.notif-panel-title {
		font-size: 0.8125rem;
		font-weight: 600;
		color: var(--oo-fg-primary);
	}

	.notif-panel-actions {
		display: flex;
		gap: 6px;
	}

	.notif-action-btn {
		padding: 2px 8px;
		border-radius: 4px;
		border: none;
		background: transparent;
		color: var(--oo-fg-muted);
		font-size: 0.6875rem;
		cursor: pointer;
		transition: background-color 0.15s ease, color 0.15s ease;
	}

	.notif-action-btn:hover {
		background-color: var(--oo-bg-overlay);
		color: var(--oo-fg-secondary);
	}

	.notif-panel-list {
		flex: 1;
		overflow-y: auto;
		padding: 4px 0;
	}

	.notif-empty {
		padding: 24px 14px;
		text-align: center;
		font-size: 0.8125rem;
		color: var(--oo-fg-faint);
	}

	.notif-item {
		display: flex;
		align-items: flex-start;
		gap: 10px;
		padding: 8px 14px;
		transition: background-color 0.1s ease;
	}

	.notif-item:hover {
		background-color: var(--oo-bg-overlay);
	}

	.notif-unread {
		background-color: var(--oo-bg-overlay);
	}

	.notif-item-icon {
		width: 14px;
		height: 14px;
		flex-shrink: 0;
		margin-top: 2px;
	}

	.notif-item-content {
		flex: 1;
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.notif-item-msg {
		font-size: 0.8125rem;
		color: var(--oo-fg-secondary);
		word-break: break-word;
		line-height: 1.35;
	}

	.notif-item-time {
		font-size: 0.6875rem;
		color: var(--oo-fg-faint);
	}
</style>
