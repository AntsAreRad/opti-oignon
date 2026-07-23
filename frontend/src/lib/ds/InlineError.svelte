<!--
  InlineError.svelte (lib/ds, spec 6.9, Goal 2).
  Inline-first error surface anchored next to the failing control. Pairs with
  errorHandler.ts: pass a ParsedApiError's message, and (when retriable) wire
  onRetry to re-run the action. Global/async failures still go through
  handleApiError(err, ctx, retry) which surfaces a toast with the same Retry.

  Accessibility: role="alert" announces the message; expose `id` so the
  related field can reference it via aria-describedby (resolves the A8 root
  cause for form errors). The icon is decorative.
-->
<script lang="ts">
	import Button from './Button.svelte';
	import Icon from './Icon.svelte';

	/** Error text. When null/empty the component renders nothing. */
	export let message: string | null | undefined = null;
	/** Stable id so a related input can set aria-describedby={id}. */
	export let id: string | undefined = undefined;
	/** Optional retry handler; shows a Retry button when provided. */
	export let onRetry: (() => void) | undefined = undefined;
	/** Disable/spin the Retry button while a retry is in flight. */
	export let retrying = false;
</script>

{#if message}
	<div class="oo-inline-error" role="alert" {id}>
		<span class="oo-inline-error-icon" aria-hidden="true">
			<Icon name="alert-triangle" size="sm" />
		</span>
		<span class="oo-inline-error-msg">{message}</span>
		{#if onRetry}
			<span class="oo-inline-error-action">
				<Button variant="ghost" size="sm" loading={retrying} on:click={() => onRetry?.()}>
					Retry
				</Button>
			</span>
		{/if}
	</div>
{/if}

<style>
	.oo-inline-error {
		display: flex;
		align-items: flex-start;
		gap: var(--oo-space-2);
		padding: var(--oo-space-2) var(--oo-space-3);
		border-radius: var(--oo-radius-md);
		background-color: var(--oo-error-bg);
		border: 1px solid var(--oo-error-bd);
		color: var(--oo-error);
		font-size: var(--oo-text-xs);
		line-height: var(--oo-leading-snug);
	}
	.oo-inline-error-icon {
		display: inline-flex;
		flex-shrink: 0;
		margin-top: 1px;
	}
	.oo-inline-error-msg {
		flex: 1;
		min-width: 0;
		color: var(--oo-fg-secondary);
	}
	.oo-inline-error-action {
		flex-shrink: 0;
	}
</style>
