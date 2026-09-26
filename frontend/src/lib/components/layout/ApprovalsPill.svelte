<!--
  ApprovalsPill.svelte
  The tool calls waiting on an approval, as a pill that opens the approvals
  drawer (mounted once, by the shell). It renders only while approvals are
  pending, wherever the shell stands: in the status card, in the collapsed
  rail and in the phone header; the drawer writes the count
  (lib/stores/approvals.ts).

  `compact` draws the count beside the shield alone, its words kept for a
  screen reader (the rail's width, a phone header shared with the stop);
  `large` makes it a 44 px target, for a touch screen.
-->
<script lang="ts">
	import Button from '$lib/ds/Button.svelte';
	import { approvalsOpen, openApprovals, pendingApprovals } from '$lib/stores/approvals';

	/** The count and the shield alone, the words kept for a screen reader. */
	export let compact = false;
	/** A 44 px target, for a touch screen. */
	export let large = false;

	$: words = `pending approval${$pendingApprovals === 1 ? '' : 's'}`;
</script>

{#if $pendingApprovals > 0}
	<span class="oo-approvals" data-compact={compact ? 'true' : 'false'}>
		<Button
			variant="secondary"
			size={large ? 'lg' : compact ? 'md' : 'sm'}
			shape="pill"
			iconLeft={compact ? 'shield-check' : undefined}
			haspopup="dialog"
			expanded={$approvalsOpen}
			on:click={openApprovals}
		>
			{#if compact}
				{$pendingApprovals}<span class="oo-sr-only">{` ${words}`}</span>
			{:else}
				{$pendingApprovals} {words}
			{/if}
		</Button>
	</span>
{/if}

<style>
	/* The warning ink on the second surface: the pill asks for attention and
	   keeps its text readable on every palette. */
	.oo-approvals :global(.oo-btn) {
		color: var(--oo-warning);
		font-weight: 600;
	}
	.oo-approvals {
		flex-shrink: 0;
	}
</style>
