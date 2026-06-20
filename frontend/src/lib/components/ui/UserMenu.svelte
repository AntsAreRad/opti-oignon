<!--
  UserMenu.svelte (S98)
  Displays current user avatar/initial and username in the header.
  Dropdown menu with profile, settings, and logout options.
  Hidden in single-user mode.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import {
		currentUser,
		isSingleUserMode,
		doLogout,
	} from '$lib/stores/auth';
	import { toastError } from '$lib/stores/notifications';

	let open = false;

	$: user = $currentUser;
	$: hidden = $isSingleUserMode;
	$: initial = user?.username?.charAt(0).toUpperCase() ?? '?';

	function toggle() {
		open = !open;
	}

	function close() {
		open = false;
	}

	async function handleLogout() {
		close();
		try {
			await doLogout();
			goto('/login');
		} catch {
			toastError('Logout failed');
		}
	}

	function handleSettings() {
		close();
		goto('/settings');
	}

	function handleClickOutside(e: MouseEvent) {
		const target = e.target as HTMLElement;
		if (!target.closest('.user-menu')) {
			close();
		}
	}
</script>

<svelte:window on:click={handleClickOutside} />

{#if !hidden && user}
	<div class="user-menu">
		<button
			class="user-trigger"
			on:click|stopPropagation={toggle}
			aria-label="User menu for {user.username}"
			aria-expanded={open}
		>
			<span class="user-avatar">{initial}</span>
			<span class="user-name">{user.username}</span>
			<svg class="chevron" class:flipped={open} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
				<path d="M4.5 6l3.5 3.5L11.5 6" stroke="currentColor" stroke-width="1.5" fill="none" />
			</svg>
		</button>

		{#if open}
			<div class="user-dropdown" role="menu">
				<div class="user-info">
					<span class="user-info-name">{user.username}</span>
					{#if user.email}
						<span class="user-info-email">{user.email}</span>
					{/if}
					<span class="user-info-role">{user.role}</span>
				</div>
				<div class="dropdown-divider"></div>
				<button class="dropdown-item" on:click={handleSettings} role="menuitem">
					<svg viewBox="0 0 20 20" fill="currentColor" class="dropdown-icon" aria-hidden="true">
						<path fill-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd" />
					</svg>
					Settings
				</button>
				<button class="dropdown-item dropdown-item--danger" on:click={handleLogout} role="menuitem">
					<svg viewBox="0 0 20 20" fill="currentColor" class="dropdown-icon" aria-hidden="true">
						<path fill-rule="evenodd" d="M3 3a1 1 0 00-1 1v12a1 1 0 001 1h5a1 1 0 100-2H4V5h4a1 1 0 100-2H3zm12.293 3.293a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 01-1.414-1.414L16.586 11H8a1 1 0 110-2h8.586l-1.293-1.293a1 1 0 010-1.414z" clip-rule="evenodd" />
					</svg>
					Sign out
				</button>
			</div>
		{/if}
	</div>
{/if}

<style>
	.user-menu {
		position: relative;
		display: flex;
		align-items: center;
	}

	.user-trigger {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0.3rem 0.5rem;
		background: var(--oo-btn-ghost-bg);
		border: 1px solid transparent;
		border-radius: var(--oo-radius-sm);
		color: var(--oo-fg-secondary);
		cursor: pointer;
		font-size: 0.8rem;
		transition: background-color 0.15s ease, border-color 0.15s ease;
	}

	.user-trigger:hover {
		background: var(--oo-btn-ghost-hover);
		border-color: var(--oo-bg-elevated);
	}

	.user-avatar {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 24px;
		height: 24px;
		border-radius: 50%;
		background-color: var(--oo-accent-primary);
		color: var(--oo-fg-on-accent);
		font-size: 0.75rem;
		font-weight: 600;
		flex-shrink: 0;
	}

	.user-name {
		max-width: 100px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.chevron {
		width: 14px;
		height: 14px;
		transition: transform 0.15s ease;
		flex-shrink: 0;
	}

	.chevron.flipped {
		transform: rotate(180deg);
	}

	.user-dropdown {
		position: absolute;
		top: calc(100% + 4px);
		right: 0;
		min-width: 200px;
		background-color: var(--oo-bg-surface);
		border: 1px solid var(--oo-bg-elevated);
		border-radius: var(--oo-radius-md);
		box-shadow: var(--oo-shadow-lg);
		z-index: 100;
		overflow: hidden;
	}

	.user-info {
		padding: 0.65rem 0.85rem;
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
	}

	.user-info-name {
		font-weight: 600;
		color: var(--oo-fg-primary);
		font-size: 0.85rem;
	}

	.user-info-email {
		color: var(--oo-fg-tertiary);
		font-size: 0.75rem;
	}

	.user-info-role {
		color: var(--oo-fg-muted);
		font-size: 0.7rem;
		text-transform: capitalize;
	}

	.dropdown-divider {
		height: 1px;
		background-color: var(--oo-bg-elevated);
		margin: 0;
	}

	.dropdown-item {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		width: 100%;
		padding: 0.55rem 0.85rem;
		background: none;
		border: none;
		color: var(--oo-fg-secondary);
		font-size: 0.8rem;
		cursor: pointer;
		text-align: left;
		transition: background-color 0.1s ease;
	}

	.dropdown-item:hover {
		background-color: var(--oo-bg-elevated);
		color: var(--oo-fg-primary);
	}

	.dropdown-item--danger:hover {
		color: var(--oo-error);
	}

	.dropdown-icon {
		width: 16px;
		height: 16px;
		flex-shrink: 0;
	}
</style>
