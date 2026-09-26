<!--
  ToggleChip.svelte (lib/ds) -- an option that is on or off, as a pill.

  It says its state as aria-pressed. Pressed, it takes the accent tint and
  draws a check in place of its icon, so the state is a shape as well as a
  colour, and under the pointer it keeps the tint with the hover wash over
  it; not pressed, it has no ground and shows its icon, if it has one, in
  the quieter text. Its one size is the desktop's 36 px; the phone's 44 px
  target is given with the phone's layout. A note is a visible word beside the label (such as
  "auto", for an option the app decides on its own), part of the chip's
  name. Clicking it flips `pressed` and dispatches `change` with the new
  state; bind `pressed` to follow it.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import Icon from './Icon.svelte';
	import type { IconName } from './types';

	/** The option's name, always shown. */
	export let label: string;
	/** Whether the option is on. */
	export let pressed = false;
	/** A visible word beside the label. */
	export let note: string | undefined = undefined;
	/** The icon shown while the option is off. */
	export let icon: IconName | undefined = undefined;
	export let disabled = false;

	const dispatch = createEventDispatcher<{ change: boolean }>();

	function toggle() {
		if (disabled) return;
		pressed = !pressed;
		dispatch('change', pressed);
	}
</script>

<button type="button" class="oo-chip" aria-pressed={pressed} {disabled} on:click={toggle}>
	{#if pressed}
		<span class="oo-chip-check" aria-hidden="true"><Icon name="check" size="sm" /></span>
	{:else if icon}
		<span class="oo-chip-icon" aria-hidden="true"><Icon name={icon} size="sm" /></span>
	{/if}
	<span class="oo-chip-label">{label}</span>
	{#if note}
		<span class="oo-chip-note">{note}</span>
	{/if}
</button>

<style>
	.oo-chip {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-2);
		flex-shrink: 0;
		height: 36px;
		min-height: 36px;
		padding: 0 var(--oo-space-5) 0 var(--oo-space-4);
		border: 1px solid transparent;
		border-radius: var(--oo-radius-full);
		background-color: transparent;
		color: var(--oo-fg-secondary);
		font: inherit;
		font-family: var(--oo-font-sans);
		font-size: var(--oo-text-base);
		line-height: 1;
		white-space: nowrap;
		cursor: pointer;
		transition:
			background-color var(--oo-motion-fast) var(--oo-ease-default),
			color var(--oo-motion-fast) var(--oo-ease-default);
	}
	.oo-chip[aria-pressed='false']:hover:not(:disabled) {
		background-color: var(--oo-bg-hover);
		color: var(--oo-fg-primary);
	}
	.oo-chip[aria-pressed='true'] {
		background-color: var(--oo-bg-tint-1);
		border-color: var(--oo-edge);
		color: var(--oo-fg-primary);
		font-weight: 500;
	}
	.oo-chip[aria-pressed='true']:hover:not(:disabled) {
		background-color: color-mix(in srgb, var(--oo-fg-primary) 4%, var(--oo-bg-tint-1));
	}
	.oo-chip-check,
	.oo-chip-icon {
		display: inline-flex;
	}
	.oo-chip-check {
		color: var(--oo-acc-ink);
	}
	.oo-chip-note {
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
		font-weight: 400;
	}
	.oo-chip:disabled {
		opacity: 0.55;
		cursor: not-allowed;
	}
	@media (prefers-reduced-motion: reduce) {
		.oo-chip {
			transition: none;
		}
	}
</style>
