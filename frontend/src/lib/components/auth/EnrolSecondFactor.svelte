<!--
  EnrolSecondFactor.svelte
  The enrolment of a first second factor where the login (or the registration) asks one of an account with none, in
  Bulbe: the TOTP is begun when the step shows -- its QR code, and its secret to type --, confirmed with one of its
  codes, and the recovery codes are shown this once; the session begins only when they were kept (`done`), or the
  root layout would leave the page before they were seen. A refusal whose words say
  "log in again" -- the enrolment expired, locked or ended -- dispatches `restart` with the server's reason.
-->
<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { beginEnrolment, completeEnrolment, finishEnrolment } from '$lib/stores/auth';
	import { httpRefusal } from '$lib/switches/serverSwitch';
	import type { AuthTokens, TwoFactorEnrolment, TwoFactorEnrolmentSecret } from '$lib/types';
	import Input from '$lib/ds/Input.svelte';
	import Button from '$lib/ds/Button.svelte';
	import InlineError from '$lib/ds/InlineError.svelte';

	/** The enrolment challenge the first step gave. */
	export let enrolment: TwoFactorEnrolment;

	const dispatch = createEventDispatcher<{ done: null; restart: { reason: string } }>();

	let begun: TwoFactorEnrolmentSecret | null = null;
	let code = '';
	let loading = false;
	let errorMsg = '';
	/** The recovery codes, once the factor is confirmed: shown this once. */
	let recoveryCodes: string[] | null = null;
	/** The session the confirmation gave, begun once the codes were kept. */
	let session: AuthTokens | null = null;

	/** The server's own reason -- a wrong code and the tries left, an enrolment ended --, or the error's text. */
	function reason(err: unknown, fallback: string): string {
		const refusal = httpRefusal(err);
		if (refusal && refusal.detail) return refusal.detail;
		return err instanceof Error && err.message ? err.message : fallback;
	}

	/** A refusal that ends the enrolment sends back to the password; any other is shown here. */
	function refused(err: unknown, fallback: string) {
		const message = reason(err, fallback);
		if (message.toLowerCase().includes('log in again')) {
			dispatch('restart', { reason: message });
			return;
		}
		errorMsg = message;
	}

	async function begin() {
		loading = true;
		errorMsg = '';
		try {
			begun = await beginEnrolment(enrolment.challenge_id);
		} catch (err: unknown) {
			refused(err, 'The second factor could not be set up.');
		} finally {
			loading = false;
		}
	}

	async function confirmCode() {
		const given = code.trim();
		if (!given) {
			errorMsg = 'Enter the code your authenticator app shows.';
			return;
		}
		code = '';
		loading = true;
		errorMsg = '';
		try {
			const confirmed = await completeEnrolment(enrolment.challenge_id, given);
			session = confirmed.tokens;
			recoveryCodes = confirmed.recoveryCodes;
		} catch (err: unknown) {
			refused(err, 'The code was not accepted.');
		} finally {
			loading = false;
		}
	}

	/** The codes were kept: the session begins, then the page goes on. */
	async function proceed() {
		if (!session) return;
		loading = true;
		errorMsg = '';
		try {
			await finishEnrolment(session);
			dispatch('done', null);
		} catch (err: unknown) {
			refused(err, 'The session could not begin.');
		} finally {
			loading = false;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') confirmCode();
	}

	onMount(begin);
</script>

<div class="oo-enrol">
	<InlineError message={errorMsg} />

	{#if recoveryCodes}
		{#if recoveryCodes.length}
			<p class="oo-enrol-text">
				Your recovery codes: each one works once, in place of a code from your app. Keep them somewhere safe;
				they are shown only now.
			</p>
			<pre class="oo-enrol-codes">{recoveryCodes.join('\n')}</pre>
		{:else}
			<p class="oo-enrol-text">No recovery codes could be made now: make them in the security settings.</p>
		{/if}
		<Button variant="primary" block loading={loading} on:click={proceed}>
			{recoveryCodes.length ? 'I have kept them: continue' : 'Continue'}
		</Button>
	{:else if begun}
		<p class="oo-enrol-text">
			This instance asks every account a second factor. Scan the code with your authenticator app, or type the
			secret into it, then enter the code it shows. This secret replaces any shown before for this account:
			remove an earlier entry for it from your app.
		</p>
		{#if begun.qr_code_base64}
			<div class="oo-enrol-qr">
				<img src={begun.qr_code_base64} alt="The authenticator secret as a QR code" />
			</div>
		{/if}
		<code class="oo-enrol-secret">{begun.secret}</code>
		<Input
			label="Code"
			bind:value={code}
			placeholder="From your authenticator app"
			autocomplete="one-time-code"
			disabled={loading}
			on:keydown={handleKeydown}
		/>
		<Button variant="primary" block loading={loading} on:click={confirmCode}>
			Confirm
		</Button>
	{:else}
		<Button variant="secondary" block loading={loading} on:click={begin}>
			Set up the second factor
		</Button>
	{/if}
</div>

<style>
	.oo-enrol {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}

	.oo-enrol-text {
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
		margin: 0;
	}

	.oo-enrol-qr {
		display: flex;
		justify-content: center;
		padding: var(--oo-space-3);
		border-radius: var(--oo-radius-sm);
		background-color: var(--oo-qr-bg);
	}

	.oo-enrol-secret,
	.oo-enrol-codes {
		font-family: var(--oo-font-mono);
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-primary);
		background-color: var(--oo-bg-subtle);
		border-radius: var(--oo-radius-sm);
		padding: var(--oo-space-2) var(--oo-space-3);
		word-break: break-all;
		margin: 0;
	}
</style>
