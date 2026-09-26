<!--
  UserMenu.svelte
  Displays current user avatar/initial and username, in Preferences >
  Account. Dropdown menu with the user's details, the Workshop's security
  page, and sign-out. Hidden in single-user mode.

  Drawn to the surface rules: the menu on the second surface with its
  edge, space rather than a line between the details and the actions, a
  wash under the pointer, the role in sentence case.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import {
		currentUser,
		isSingleUserMode,
		doLogout,
	} from '$lib/stores/auth';
	import { toastError } from '$lib/stores/notifications';
	import Icon from '$lib/ds/Icon.svelte';

	let open = false;

	$: user = $currentUser;
	$: hidden = $isSingleUserMode;
	$: initial = user?.username?.charAt(0).toUpperCase() ?? '?';
	$: role = user?.role ? user.role.charAt(0).toUpperCase() + user.role.slice(1) : '';

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

	function handleSecurity() {
		close();
		goto('/workshop/security');
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
			<span class="chevron" class:flipped={open}><Icon name="chevron-down" size="sm" /></span>
		</button>

		{#if open}
			<div class="user-dropdown" role="menu">
				<div class="user-info">
					<span class="user-info-name">{user.username}</span>
					{#if user.email}
						<span class="user-info-email">{user.email}</span>
					{/if}
					{#if role}<span class="user-info-role">{role}</span>{/if}
				</div>
				<button class="dropdown-item" on:click={handleSecurity} role="menuitem">
					<Icon name="shield-check" size="sm" />
					Security
				</button>
				<button class="dropdown-item dropdown-item--danger" on:click={handleLogout} role="menuitem">
					<Icon name="arrow-right" size="sm" />
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
		gap: var(--oo-space-2);
		min-height: 36px;
		padding: var(--oo-space-1) var(--oo-space-3) var(--oo-space-1) var(--oo-space-1);
		background: transparent;
		border: 1px solid transparent;
		border-radius: var(--oo-radius-full);
		color: var(--oo-fg-secondary);
		cursor: pointer;
		font-size: var(--oo-text-sm);
		transition: background-color var(--oo-motion-fast) var(--oo-ease-default);
	}

	.user-trigger:hover {
		background-color: var(--oo-bg-hover);
		color: var(--oo-fg-primary);
	}

	.user-avatar {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 26px;
		height: 26px;
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-full);
		background-color: var(--oo-acc-fill);
		color: var(--oo-fg-on-accent);
		font-size: var(--oo-text-xs);
		font-weight: 600;
		flex-shrink: 0;
	}

	.user-name {
		max-width: 10rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.chevron {
		display: inline-flex;
		transition: transform var(--oo-motion-fast) var(--oo-ease-default);
		flex-shrink: 0;
	}

	.chevron.flipped {
		transform: rotate(180deg);
	}

	.user-dropdown {
		position: absolute;
		top: calc(100% + 6px);
		right: 0;
		display: flex;
		flex-direction: column;
		gap: 2px;
		min-width: 13rem;
		padding: var(--oo-space-1);
		background-color: var(--oo-bg-overlay);
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-md);
		box-shadow: var(--oo-shadow-md);
		z-index: var(--oo-z-overlay);
	}

	.user-info {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: var(--oo-space-2) var(--oo-space-3) var(--oo-space-3);
	}

	.user-info-name {
		font-weight: 600;
		color: var(--oo-fg-primary);
		font-size: var(--oo-text-sm);
	}

	.user-info-email,
	.user-info-role {
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-xs);
	}

	.dropdown-item {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		width: 100%;
		min-height: 36px;
		padding: var(--oo-space-2) var(--oo-space-3);
		background: transparent;
		border: 0;
		border-radius: var(--oo-radius-sm);
		color: var(--oo-fg-primary);
		font-size: var(--oo-text-sm);
		cursor: pointer;
		text-align: left;
	}

	.dropdown-item:hover {
		background-color: var(--oo-bg-hover);
	}

	.dropdown-item--danger {
		color: var(--oo-error);
	}

	@media (prefers-reduced-motion: reduce) {
		.chevron {
			transition: none;
		}
	}
</style>
