<!--
  ConfirmDialog.svelte (lib/ds) -- asks before an action that cannot be
  undone, and shows why it failed if it does.

  Built on the ds Modal: a native modal dialog (the rest of the page is inert
  while it is open), focus moved into it on open and restored to its opener
  on close. The caller runs the action from onConfirm and closes the dialog
  itself once the action succeeds; a failure goes back in through `error`,
  shown in an alert inside the dialog, which stays open. While `busy`, the
  confirming button shows the action running and cannot be pressed again,
  the cancelling button and the close button wait, and Escape and the
  backdrop do not close it. The body is a form that confirms on submit, so
  a dialog that holds one field (given `autofocus`, where focus lands on
  open) confirms with Enter.
-->
<script lang="ts">
	import Modal from './Modal.svelte';
	import Button from './Button.svelte';
	import InlineError from './InlineError.svelte';

	/** Whether the dialog is shown. */
	export let open = false;
	/** The question, as the dialog's title. */
	export let title: string;
	/** What the action does, in a sentence under the title. */
	export let message = '';
	/** The confirming button's label: the action's verb. */
	export let confirmLabel = 'Confirm';
	/** The cancelling button's label. */
	export let cancelLabel = 'Cancel';
	/** The action destroys something: the confirming button reads as danger. */
	export let danger = false;
	/** The action is running. */
	export let busy = false;
	/** Why the action failed; null while it has not. */
	export let error: string | null = null;
	/** Runs the action. */
	export let onConfirm: () => void;
	/** Closes the dialog without acting. */
	export let onCancel: () => void;

	function accept() {
		if (busy) return;
		onConfirm();
	}

	function dismiss() {
		onCancel();
	}
</script>

<Modal
	{open}
	{title}
	size="sm"
	onClose={dismiss}
	closeOnEsc={!busy}
	closeOnBackdrop={!busy}
	closable={!busy}
>
	<form on:submit|preventDefault={accept}>
		{#if message}
			<p class="oo-confirm-message">{message}</p>
		{/if}
		<slot />
		{#if error}
			<div class="oo-confirm-error">
				<InlineError message={error} />
			</div>
		{/if}
	</form>
	<svelte:fragment slot="footer">
		<Button variant="secondary" disabled={busy} on:click={dismiss}>{cancelLabel}</Button>
		<Button variant={danger ? 'danger' : 'primary'} loading={busy} on:click={accept}>{confirmLabel}</Button>
	</svelte:fragment>
</Modal>

<style>
	.oo-confirm-message {
		margin: 0;
		font-size: var(--oo-text-sm);
		line-height: var(--oo-leading-normal);
		color: var(--oo-fg-secondary);
	}

	.oo-confirm-error {
		margin-top: var(--oo-space-3);
	}
</style>
