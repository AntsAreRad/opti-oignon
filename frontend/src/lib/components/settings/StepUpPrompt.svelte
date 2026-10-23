<!--
  StepUpPrompt.svelte
  The password of the session's own account, asked before a second factor of that account is added or removed.
  Shown inline where the action was asked; dispatches `confirm` with the password, or `cancel`. The password is
  cleared once sent: the panel shows the server's answer (wrong password, a wait) in `error`.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import Input from '$lib/ds/Input.svelte';
	import Button from '$lib/ds/Button.svelte';

	/** The action's label on the confirm button. */
	export let action = 'Confirm';
	/** Whether the action is running: the field and the buttons wait. */
	export let busy = false;
	/** The server's reason when the last try was refused. */
	export let error = '';

	const dispatch = createEventDispatcher<{ confirm: { password: string }; cancel: null }>();
	let password = '';

	function send() {
		if (!password || busy) return;
		const given = password;
		password = '';
		dispatch('confirm', { password: given });
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter') send();
	}
</script>

<div class="oo-step-up" role="group" aria-label="Confirm with your password">
	<Input
		type="password"
		label="Your account's password"
		bind:value={password}
		autocomplete="current-password"
		size="sm"
		disabled={busy}
		error={error || undefined}
		on:keydown={handleKeydown}
	/>
	<div class="oo-step-up-actions">
		<Button variant="primary" size="sm" loading={busy} disabled={!password || busy} on:click={send}>
			{action}
		</Button>
		<Button variant="ghost" size="sm" disabled={busy} on:click={() => dispatch('cancel', null)}>
			Cancel
		</Button>
	</div>
</div>

<style>
	.oo-step-up {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}

	.oo-step-up-actions {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
	}
</style>
