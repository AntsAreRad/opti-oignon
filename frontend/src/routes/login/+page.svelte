<!--
  Login page (S98, refactored onto ds primitives in S168).
  Shown when multi-user mode is active and the user is not authenticated.
  Redirects to /chat after a successful login.

  Theme parity (spec 12.3): the auth pages render outside AppShell, but the
  root layout calls initPreferences() on every route, so the selected palette
  is applied via data-oo-theme on <html> here too. Card / Input / Button and
  the --oo-bg-base background all read palette tokens, so login matches the
  app in all five palettes with no per-page theme code.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { doLogin } from '$lib/stores/auth';
	import Card from '$lib/ds/Card.svelte';
	import Input from '$lib/ds/Input.svelte';
	import Button from '$lib/ds/Button.svelte';

	let username = '';
	let password = '';
	let loading = false;
	let errorMsg = '';

	async function handleLogin() {
		if (!username.trim() || !password) {
			errorMsg = 'Please enter both username and password.';
			return;
		}

		loading = true;
		errorMsg = '';

		try {
			await doLogin(username.trim(), password);
			goto('/chat');
		} catch (err: unknown) {
			const msg = err instanceof Error ? err.message : 'Login failed';
			// S125: Detect rate limiting from 429 responses
			if (msg.includes('429') || msg.toLowerCase().includes('too many')) {
				errorMsg = 'Too many login attempts. Please wait a moment and try again.';
			} else if (msg.includes('401')) {
				errorMsg = 'Invalid username or password.';
			} else {
				errorMsg = msg;
			}
		} finally {
			loading = false;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') handleLogin();
	}
</script>

<div class="oo-auth-page">
	<div class="oo-auth-card">
		<Card variant="raised" padding="lg">
			<div class="oo-auth-logo">
				<img src="/bousier-oignon.png" alt="Opti-Oignon" class="oo-auth-icon oo-logo-adaptive" />
				<h1 class="oo-auth-title">Opti-Oignon</h1>
			</div>

			<p class="oo-auth-subtitle">Sign in to your account</p>

			{#if errorMsg}
				<div class="oo-auth-error" role="alert">{errorMsg}</div>
			{/if}

			<div class="oo-auth-form">
				<Input
					label="Username"
					bind:value={username}
					placeholder="Enter your username"
					autocomplete="username"
					disabled={loading}
					on:keydown={handleKeydown}
				/>
				<Input
					type="password"
					label="Password"
					bind:value={password}
					placeholder="Enter your password"
					autocomplete="current-password"
					disabled={loading}
					on:keydown={handleKeydown}
				/>
				<Button variant="primary" block loading={loading} on:click={handleLogin}>
					Sign in
				</Button>
			</div>

			<p class="oo-auth-footer">
				Don't have an account?
				<a href="/register" class="oo-auth-link">Create one</a>
			</p>
		</Card>
	</div>
</div>

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
