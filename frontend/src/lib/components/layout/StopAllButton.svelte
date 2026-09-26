<!--
  StopAllButton.svelte
  The emergency stop, and the only stop control: the shell mounts it where
  it stays one tap away at every width (the status card, the collapsed
  rail, the phone header), and the approvals drawer, a modal dialog that
  makes the rest of the page inert, holds one of its own. It makes the
  machine quiet: it cancels generations and runs, unloads models, destroys
  sandboxes and stops the sync node. Resuming needs no ceremony (sign-in is
  still required).

  It always renders, in one of three states read from the estop store
  (lib/stores/estop.ts, the one poller): available, and it opens its
  confirmation; unknown (nothing read yet, or the status unreachable),
  enabled all the same, and a request that fails says so; unavailable (the
  server says it cannot stop), disabled, with the reason written beside
  it. Stopped, it is the pill that says so, with Resume.

  The confirmation keeps the two actions (stop, or stop and switch to
  Bulbe), a polite live region for what changed, and the steps that failed.
  It is fixed to the viewport and placed from the button while it is open
  (floating-ui, as a menu is), so no clipping box cuts it: not the 72 px
  rail, not the card, not a dialog's panel. Several copies may be on
  screen; only the one the reader acted on announces what changed, or the
  error.

  Placement: `card` beside the backend's name, `rail` stacked in the 72 px
  rail, `header` in the phone header, `dialog` in a dialog's foot. On a
  touch screen (the phone header, or `large`) every button is a 44 px
  target, the confirmation's actions and Resume included.
-->
<script lang="ts">
	import { onDestroy, tick } from 'svelte';
	import { autoUpdate, computePosition, flip, offset, shift, type Placement } from '@floating-ui/dom';
	import Button from '$lib/ds/Button.svelte';
	import Icon from '$lib/ds/Icon.svelte';
	import { estop, engageStop, resumeStop } from '$lib/stores/estop';

	/** Where the control sits, which sets its shape and where its confirmation opens. */
	export let placement: 'card' | 'rail' | 'header' | 'dialog' = 'card';
	/** A 44 px target, for a touch screen; the phone header's always is. */
	export let large = false;

	const OPENS: Record<typeof placement, Placement> = {
		card: 'top-end',
		rail: 'right-end',
		header: 'bottom-end',
		dialog: 'top-end'
	};

	let confirming = false;
	/** Whether the reader acted on this copy: it alone speaks. */
	let acted = false;
	let wrapper: HTMLElement | undefined;
	let anchor: HTMLElement | undefined;
	let sheet: HTMLElement | undefined;
	let stopFollowing: (() => void) | undefined;

	$: unavailable = $estop.available === false;
	$: touch = placement === 'header' || large;
	$: actionSize = (touch ? 'lg' : 'sm') as 'lg' | 'sm';
	$: triggerSize = (touch ? 'lg' : 'md') as 'lg' | 'md';

	function place() {
		if (!anchor || !sheet) return;
		computePosition(anchor, sheet, {
			placement: OPENS[placement],
			strategy: 'fixed',
			middleware: [offset(8), flip(), shift({ padding: 8 })]
		}).then(({ x, y }) => {
			if (!sheet) return;
			sheet.style.left = `${x}px`;
			sheet.style.top = `${y}px`;
		});
	}

	async function follow(open: boolean) {
		stopFollowing?.();
		stopFollowing = undefined;
		if (!open) return;
		await tick();
		if (anchor && sheet) stopFollowing = autoUpdate(anchor, sheet, place);
	}

	$: if (typeof window !== 'undefined') void follow(confirming);

	onDestroy(() => stopFollowing?.());

	function toggleConfirm() {
		confirming = !confirming;
	}

	async function engage(dropToBulbe: boolean) {
		acted = true;
		const answered = await engageStop(dropToBulbe);
		if (answered) confirming = false;
	}

	async function resume() {
		acted = true;
		await resumeStop();
	}

	function closeOnOutside(event: MouseEvent) {
		if (!confirming || !wrapper) return;
		if (event.target instanceof Node && !wrapper.contains(event.target)) confirming = false;
	}

	function closeOnEscape(event: KeyboardEvent) {
		if (confirming && event.key === 'Escape') confirming = false;
	}
</script>

<svelte:document on:click|capture={closeOnOutside} on:keydown|capture={closeOnEscape} />

<div class="oo-stop" data-placement={placement} bind:this={wrapper}>
	<span class="oo-sr-only" aria-live="polite">{acted ? $estop.announce : ''}</span>
	{#if $estop.stopped}
		<span class="oo-stop-pill">
			<Icon name="stop" size="sm" />
			<span>Stopped</span>
			<Button variant="ghost" size={actionSize} shape="pill" disabled={$estop.busy} on:click={resume}>
				Resume<span class="oo-sr-only">{' from the emergency stop'}</span>
			</Button>
		</span>
	{:else}
		<span class="oo-stop-control" bind:this={anchor}>
			<Button
				variant="secondary"
				size={triggerSize}
				shape="pill"
				iconLeft="stop"
				disabled={unavailable || $estop.busy}
				haspopup="dialog"
				expanded={confirming}
				on:click={toggleConfirm}
			>
				Stop all<span class="oo-sr-only">{' (emergency stop for generation, agents and tools)'}</span>
			</Button>
		</span>
		{#if unavailable}
			<span class="oo-stop-reason">Emergency stop unavailable on this server</span>
		{/if}
		{#if confirming}
			<div class="oo-stop-confirm" role="dialog" aria-label="Confirm the emergency stop" bind:this={sheet}>
				<p class="oo-stop-title">Stop everything now?</p>
				<p class="oo-stop-text">
					Cancels generations and runs, unloads models, destroys sandboxes and stops the sync
					node. Resuming needs no ceremony.
				</p>
				<Button variant="danger" size={actionSize} block disabled={$estop.busy} on:click={() => engage(false)}>
					Stop compute
				</Button>
				<Button variant="secondary" size={actionSize} block disabled={$estop.busy} on:click={() => engage(true)}>
					Stop compute and switch to Bulbe
				</Button>
				<Button variant="ghost" size={actionSize} block disabled={$estop.busy} on:click={toggleConfirm}>
					Cancel
				</Button>
			</div>
		{/if}
	{/if}
	{#if acted && $estop.error}
		<span class="oo-stop-error" role="alert">{$estop.error}</span>
	{/if}
</div>

<style>
	.oo-stop {
		position: relative;
		display: inline-flex;
		flex-direction: column;
		align-items: flex-end;
		gap: var(--oo-space-1);
		min-width: 0;
	}
	.oo-stop[data-placement='rail'] {
		align-items: center;
	}
	/* In the phone header the stop keeps its width: the title gives way. */
	.oo-stop[data-placement='header'] {
		flex-shrink: 0;
	}

	/* The button: a pill on the sunken ground, in the stop ink. */
	.oo-stop-control :global(.oo-btn) {
		background-color: var(--oo-bg-subtle);
		color: var(--oo-fg-stop);
		border: 1px solid var(--oo-edge);
	}
	/* Under the pointer it lifts to the second surface, keeping its ink:
	   4.5:1 or more in every palette. */
	.oo-stop-control :global(.oo-btn:hover:not(:disabled)) {
		background-color: var(--oo-bg-overlay);
		color: var(--oo-fg-stop);
		border: 1px solid var(--oo-edge);
	}
	/* In the rail the label sits under the octagon, so the rail's width holds it. */
	.oo-stop[data-placement='rail'] .oo-stop-control :global(.oo-btn) {
		flex-direction: column;
		gap: var(--oo-space-1);
		width: 60px;
		padding: var(--oo-space-3) var(--oo-space-1);
		border-radius: var(--oo-radius-lg);
		font-size: var(--oo-text-2xs);
		white-space: normal;
		line-height: var(--oo-leading-tight);
	}

	.oo-stop-reason {
		font-size: var(--oo-text-2xs);
		color: var(--oo-fg-muted);
		text-align: right;
	}
	.oo-stop[data-placement='rail'] .oo-stop-reason {
		text-align: center;
	}

	.oo-stop-pill {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-2);
		padding: var(--oo-space-1) var(--oo-space-1) var(--oo-space-1) var(--oo-space-3);
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-full);
		background-color: var(--oo-bg-subtle);
		color: var(--oo-fg-stop);
		font-size: var(--oo-text-sm);
		font-weight: 600;
	}
	.oo-stop[data-placement='rail'] .oo-stop-pill {
		flex-direction: column;
		padding: var(--oo-space-2) var(--oo-space-1);
		border-radius: var(--oo-radius-lg);
	}

	/* The confirmation: a small sheet on the second surface, fixed to the
	   viewport and placed at run time: up from the card at the foot of the
	   sidebar, to the right of the rail, down from the phone header, up from
	   a dialog's foot, and kept inside the viewport. */
	.oo-stop-confirm {
		position: fixed;
		top: 0;
		left: 0;
		z-index: var(--oo-z-overlay);
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		box-sizing: border-box;
		width: 260px;
		max-width: calc(100vw - 16px);
		padding: var(--oo-space-4);
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-lg);
		background-color: var(--oo-bg-overlay);
		color: var(--oo-fg-primary);
		box-shadow: var(--oo-shadow-md);
	}

	.oo-stop-title {
		margin: 0;
		font-size: var(--oo-text-sm);
		font-weight: 600;
		color: var(--oo-fg-primary);
	}
	.oo-stop-text {
		margin: 0 0 var(--oo-space-1);
		font-size: var(--oo-text-xs);
		line-height: var(--oo-leading-snug);
		color: var(--oo-fg-secondary);
	}

	.oo-stop-error {
		font-size: var(--oo-text-2xs);
		color: var(--oo-error);
		text-align: right;
	}
</style>
