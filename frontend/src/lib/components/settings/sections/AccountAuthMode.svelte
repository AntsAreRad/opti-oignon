<!--
  AccountAuthMode.svelte
  Account & Security > Authentication mode group of the /settings hub.

  Surfaces the single-user / multi-user choice that was previously made once
  in the onboarding overlay and then echoed inside the legacy settings page
  (s109). Now editable from its resolved settings location (spec 5.7:
  onboarding choices appear in their settings location, editable). Applies
  immediately with a toast.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import SettingsGroup from '$lib/components/settings/SettingsGroup.svelte';
	import Switch from '$lib/ds/Switch.svelte';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import { apiGet, apiPut } from '$lib/api/client';

	let singleUserMode = true;
	let authAvailable = false;
	let togglingAuth = false;

	onMount(async () => {
		try {
			const data = await apiGet<{ available?: boolean; single_user_mode?: boolean }>(
				'/api/auth/status'
			);
			authAvailable = data.available ?? false;
			singleUserMode = data.single_user_mode ?? true;
		} catch {
			// Auth module may be unavailable.
		}
	});

	async function toggleAuthMode() {
		togglingAuth = true;
		const newMode = !singleUserMode;
		try {
			// Route through the API client so the httpOnly cookie and CSRF token
			// are sent. A raw fetch omits both and is rejected under Bulbe.
			await apiPut('/api/auth/mode', { single_user_mode: newMode });
			singleUserMode = newMode;
			toastSuccess(newMode ? 'Authentication disabled' : 'Authentication enabled');
		} catch {
			toastError('Failed to change auth mode');
		} finally {
			togglingAuth = false;
		}
	}
</script>

<SettingsGroup
	id="account-auth-mode"
	title="Authentication mode"
	description="Single-user mode skips the login screen. Multi-user mode requires sign-in and enables the two-factor methods below."
>
	{#if authAvailable}
		<Switch
			label="Require authentication (multi-user)"
			description="When on, the app shows a login screen and enforces credentials."
			checked={!singleUserMode}
			disabled={togglingAuth}
			on:change={toggleAuthMode}
		/>
	{:else}
		<p class="oo-auth-unavailable">
			The authentication module is not installed. Install the optional
			<code>auth</code> dependency group to enable login, TOTP and WebAuthn.
		</p>
	{/if}
</SettingsGroup>

<style>
	.oo-auth-unavailable {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
		line-height: var(--oo-leading-snug);
	}

	.oo-auth-unavailable code {
		font-family: var(--oo-font-mono);
		font-size: var(--oo-text-xs);
		padding: 1px 4px;
		border-radius: var(--oo-radius-sm);
		background-color: var(--oo-bg-elevated);
		color: var(--oo-fg-secondary);
	}
</style>
