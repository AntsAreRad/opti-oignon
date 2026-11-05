<!--
  Register page (refactored onto ds primitives).
  Shown when multi-user mode is active and registration is enabled: the
  first account of an install with none, or any where the configuration
  opens registration in Daily mode. Past it, the owner adds accounts in the
  Workshop (Security > Authentication mode).
  Redirects to /chat after a successful registration.

  Theme parity (spec 12.3): like login, this page renders outside AppShell
  but inherits the palette from data-oo-theme (set by the root layout's
  initPreferences on every route). All controls use ds primitives, so the
  page matches the app in all five palettes.

  In Bulbe the new account enrols a second factor before its first session
  (EnrolSecondFactor); an enrolment that ends sends to the login, which asks
  it again.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { doRegister, authStatus } from '$lib/stores/auth';
	import type { TwoFactorEnrolment } from '$lib/types';
	import Card from '$lib/ds/Card.svelte';
	import Input from '$lib/ds/Input.svelte';
	import Button from '$lib/ds/Button.svelte';
	import EnrolSecondFactor from '$lib/components/auth/EnrolSecondFactor.svelte';

	let username = '';
	let email = '';
	let password = '';
	let confirmPassword = '';
	let loading = false;
	let errorMsg = '';

	// The enrolment of the new account's first factor, once the registration asked one.
	let enrolment: TwoFactorEnrolment | null = null;

	$: registrationDisabled = $authStatus && !$authStatus.registration_enabled;

	function validate(): string | null {
		if (!username.trim() || username.trim().length < 2) {
			return 'Username must be at least 2 characters.';
		}
		if (password.length < 8) {
			return 'Password must be at least 8 characters.';
		}
		if (password !== confirmPassword) {
			return 'Passwords do not match.';
		}
		return null;
	}

	async function handleRegister() {
		const err = validate();
		if (err) {
			errorMsg = err;
			return;
		}

		loading = true;
		errorMsg = '';

		try {
			const outcome = await doRegister(username.trim(), password, email.trim());
			if ('enrolment' in outcome) {
				enrolment = outcome.enrolment;
				password = '';
				confirmPassword = '';
				return;
			}
			goto('/chat');
		} catch (e: unknown) {
			const msg = e instanceof Error ? e.message : 'Registration failed';
			errorMsg = msg.includes('400')
				? 'Registration failed. Username may already be taken.'
				: msg;
		} finally {
			loading = false;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') handleRegister();
	}
</script>

<main id="main-content" class="oo-auth-page">
	<div class="oo-auth-card">
		<Card variant="raised" padding="lg">
			<div class="oo-auth-logo">
				<img src="/bousier-oignon.png" alt="Opti-Oignon" class="oo-auth-icon oo-logo-adaptive" />
				<h1 class="oo-auth-title">Opti-Oignon</h1>
			</div>

			<p class="oo-auth-subtitle">
				{enrolment ? 'Set up a second factor to sign in' : 'Create your account'}
			</p>

			{#if enrolment}
				<EnrolSecondFactor {enrolment} on:done={() => goto('/chat')} on:restart={() => goto('/login')} />
			{:else if registrationDisabled}
				<div class="oo-auth-error" role="alert">
					Registration is closed: the platform's owner adds accounts, in Workshop > Security > Authentication mode.
				</div>
			{:else}
				{#if errorMsg}
					<div class="oo-auth-error" role="alert">{errorMsg}</div>
				{/if}

				<div class="oo-auth-form">
					<Input
						label="Username"
						bind:value={username}
						placeholder="Choose a username"
						autocomplete="username"
						disabled={loading}
						on:keydown={handleKeydown}
					/>
					<Input
						type="email"
						label="Email (optional)"
						bind:value={email}
						placeholder="your@email.com"
						autocomplete="email"
						disabled={loading}
						on:keydown={handleKeydown}
					/>
					<Input
						type="password"
						label="Password"
						bind:value={password}
						placeholder="At least 8 characters"
						autocomplete="new-password"
						disabled={loading}
						on:keydown={handleKeydown}
					/>
					<Input
						type="password"
						label="Confirm password"
						bind:value={confirmPassword}
						placeholder="Repeat your password"
						autocomplete="new-password"
						disabled={loading}
						on:keydown={handleKeydown}
					/>
					<Button variant="primary" block loading={loading} on:click={handleRegister}>
						Create account
					</Button>
				</div>
			{/if}

			<p class="oo-auth-footer">
				Already have an account?
				<a href="/login" class="oo-auth-link">Sign in</a>
			</p>
		</Card>
	</div>
</main>

<style>
	.oo-auth-page {
		display: flex;
		align-items: center;
		justify-content: center;
		min-height: 100vh;
		background-color: var(--oo-bg-base);
		padding: var(--oo-space-4);
	}

	.oo-auth-card {
		width: 100%;
		max-width: 400px;
	}

	.oo-auth-logo {
		display: flex;
		flex-direction: column;
		align-items: center;
		margin-bottom: var(--oo-space-2);
	}

	.oo-auth-icon {
		width: 80px;
		height: 80px;
		margin-bottom: var(--oo-space-3);
		object-fit: contain;
		border-radius: var(--oo-radius-md);
	}

	.oo-auth-title {
		font-size: var(--oo-text-2xl);
		font-weight: 600;
		color: var(--oo-fg-primary);
		margin: 0;
	}

	.oo-auth-subtitle {
		text-align: center;
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
		margin: var(--oo-space-2) 0 var(--oo-space-5);
	}

	.oo-auth-error {
		background-color: var(--oo-error-bg);
		border: 1px solid var(--oo-error-bd);
		color: var(--oo-error);
		border-radius: var(--oo-radius-sm);
		padding: var(--oo-space-2) var(--oo-space-3);
		font-size: var(--oo-text-xs);
		margin-bottom: var(--oo-space-4);
	}

	.oo-auth-form {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}

	.oo-auth-footer {
		text-align: center;
		margin: var(--oo-space-5) 0 0;
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-tertiary);
	}

	.oo-auth-link {
		color: var(--oo-accent);
		text-decoration: none;
		font-weight: 500;
	}

	.oo-auth-link:hover {
		text-decoration: underline;
	}
</style>
