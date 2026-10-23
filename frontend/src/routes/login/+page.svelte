<!--
  Login page (refactored onto ds primitives).
  Shown when multi-user mode is active and the user is not authenticated.
  Redirects to /chat after a successful login.

  Theme parity (spec 12.3): the auth pages render outside AppShell, but the
  root layout calls initPreferences() on every route, so the selected palette
  is applied via data-oo-theme on <html> here too. Card / Input / Button and
  the --oo-bg-base background all read palette tokens, so login matches the
  app in all five palettes with no per-page theme code.

  Two steps: the password, then -- for an account with a second factor -- a
  code (TOTP or recovery) or the security key the first step offered.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { doLogin, doLoginSecondStep } from '$lib/stores/auth';
	import { requestOptions, serializeAssertion } from '$lib/webauthn';
	import { httpRefusal } from '$lib/switches/serverSwitch';
	import type { TwoFactorChallenge } from '$lib/types';
	import Card from '$lib/ds/Card.svelte';
	import Input from '$lib/ds/Input.svelte';
	import Button from '$lib/ds/Button.svelte';

	let username = '';
	let password = '';
	let loading = false;
	let errorMsg = '';

	// The second step, once the first answered a challenge.
	let challenge: TwoFactorChallenge | null = null;
	let code = '';

	$: offersCode = !!challenge && (challenge.methods.includes('totp') || challenge.methods.includes('recovery'));
	$: offersKey = !!challenge && challenge.methods.includes('webauthn') && !!challenge.webauthn_options;

	async function handleLogin() {
		if (!username.trim() || !password) {
			errorMsg = 'Please enter both username and password.';
			return;
		}

		loading = true;
		errorMsg = '';

		try {
			const outcome = await doLogin(username.trim(), password);
			if ('challenge' in outcome) {
				challenge = outcome.challenge;
				password = '';
				return;
			}
			goto('/chat');
		} catch (err: unknown) {
			const msg = err instanceof Error ? err.message : 'Login failed';
			const status = httpRefusal(err)?.status;
			// Detect rate limiting from 429 responses
			if (status === 429 || msg.includes('429') || msg.toLowerCase().includes('too many')) {
				errorMsg = 'Too many login attempts. Please wait a moment and try again.';
			} else if (status === 401 || msg.includes('401')) {
				errorMsg = 'Invalid username or password.';
			} else {
				errorMsg = reason(err, msg);
			}
		} finally {
			loading = false;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') handleLogin();
	}

	/** The server's own reason -- a wrong code and the tries left, a challenge to start over --, or the error's text. */
	function reason(err: unknown, fallback: string): string {
		const refusal = httpRefusal(err);
		if (refusal && refusal.detail) return refusal.detail;
		return err instanceof Error && err.message ? err.message : fallback;
	}

	/** A refusal that ends the challenge -- expired, unknown, locked -- sends back to the first step. */
	function endsChallenge(message: string): boolean {
		return message.toLowerCase().includes('log in again');
	}

	async function answer(send: () => Promise<unknown>) {
		if (!challenge) return;
		loading = true;
		errorMsg = '';
		try {
			await send();
			goto('/chat');
		} catch (err: unknown) {
			const message = reason(err, 'The second step failed.');
			if (endsChallenge(message)) {
				challenge = null;
				code = '';
			}
			errorMsg = message;
		} finally {
			loading = false;
		}
	}

	function submitCode() {
		const given = code.trim();
		if (!challenge || !given) {
			errorMsg = 'Enter the code from your authenticator app, or a recovery code.';
			return;
		}
		const challengeId = challenge.challenge_id;
		code = '';
		answer(() => doLoginSecondStep({ challenge_id: challengeId, method: 'auto', code: given }));
	}

	function useKey() {
		if (!challenge || !challenge.webauthn_options) return;
		const { challenge_id: challengeId, webauthn_options: options } = challenge;
		answer(async () => {
			if (!window.PublicKeyCredential) {
				throw new Error('This browser cannot use a security key here.');
			}
			const credential = await navigator.credentials.get({ publicKey: requestOptions(options) });
			if (!credential) throw new Error('The security key gave no answer.');
			return doLoginSecondStep({
				challenge_id: challengeId,
				method: 'webauthn',
				webauthn_response: serializeAssertion(credential as PublicKeyCredential)
			});
		});
	}

	function startOver() {
		challenge = null;
		code = '';
		errorMsg = '';
	}

	function handleCodeKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') submitCode();
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
				{challenge ? 'Confirm it is you: a second factor' : 'Sign in to your account'}
			</p>

			{#if errorMsg}
				<div class="oo-auth-error" role="alert">{errorMsg}</div>
			{/if}

			{#if challenge}
			<div class="oo-auth-form">
				{#if offersCode}
					<Input
						label="Code"
						bind:value={code}
						placeholder="From your authenticator app, or a recovery code"
						autocomplete="one-time-code"
						disabled={loading}
						on:keydown={handleCodeKeydown}
					/>
					<Button variant="primary" block loading={loading} on:click={submitCode}>
						Verify
					</Button>
				{/if}
				{#if offersKey}
					<Button variant={offersCode ? 'secondary' : 'primary'} block loading={loading} on:click={useKey}>
						Use your security key
					</Button>
				{/if}
				<Button variant="link" size="sm" disabled={loading} on:click={startOver}>
					Start over
				</Button>
			</div>
			{:else}
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
			{/if}

			<p class="oo-auth-footer">
				Don't have an account?
				<a href="/register" class="oo-auth-link">Create one</a>
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
