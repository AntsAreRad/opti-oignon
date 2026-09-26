<!--
  IconButton.svelte (lib/ds) -- a round button that shows an icon alone.

  Its label is its accessible name, and it is required: an icon button
  without one refuses to render, with an error naming the accessible name,
  because a screen reader would announce it as "button" and nothing else.
  The refusal holds in every build: a label is written at each call site,
  so a missing one fails the first render, in development and in the
  contracts, rather than shipping a nameless control.
  The icon is drawn once and hidden from assistive technology.

  Sizes: md is a 36 px box, the desktop size; lg is 44 px, the phone's
  touch target. Variants: ghost (no ground until hovered) and primary (the
  accent fill, as the send button draws it). A button that opens something
  says so through expanded, haspopup and controls. A toggle is a
  ToggleChip or a pressed Button, which draw their check; an icon button
  has no pressed state.
-->
<script lang="ts">
	import Icon from './Icon.svelte';
	import type { IconButtonSize, IconButtonVariant, IconName, PopupKind } from './types';

	/** The icon drawn. */
	export let icon: IconName;
	/** The accessible name. Required. */
	export let label: string;
	export let size: IconButtonSize = 'md';
	export let variant: IconButtonVariant = 'ghost';
	export let type: 'button' | 'submit' = 'button';
	export let disabled = false;
	/** Whether what this button opens is open (aria-expanded); omitted when undefined. */
	export let expanded: boolean | undefined = undefined;
	/** What this button opens (aria-haspopup); omitted when undefined. */
	export let haspopup: PopupKind | undefined = undefined;
	/** The id of the element this button shows or controls (aria-controls). */
	export let controls: string | undefined = undefined;

	$: if (typeof label !== 'string' || label.trim() === '') {
		throw new Error(`IconButton "${icon}" needs an accessible name: give it a label`);
	}
</script>

<button
	{type}
	class="oo-icon-btn"
	data-size={size}
	data-variant={variant}
	aria-label={label}
	aria-expanded={expanded}
	aria-haspopup={haspopup}
	aria-controls={controls}
	{disabled}
	on:click
	on:keydown
	on:focus
	on:blur
>
	<Icon name={icon} size="md" />
</button>

<style>
	.oo-icon-btn {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		padding: 0;
		border: 1px solid transparent;
		border-radius: var(--oo-radius-full);
		background-color: transparent;
		color: var(--oo-fg-secondary);
		cursor: pointer;
		transition:
			background-color var(--oo-motion-fast) var(--oo-ease-default),
			color var(--oo-motion-fast) var(--oo-ease-default);
	}
	/* The box sets its own minimum too, so no wider rule stretches it out of square. */
	.oo-icon-btn[data-size='md'] {
		width: 36px;
		height: 36px;
		min-height: 36px;
	}
	.oo-icon-btn[data-size='lg'] {
		width: 44px;
		height: 44px;
		min-height: 44px;
	}
	.oo-icon-btn:hover:not(:disabled) {
		background-color: var(--oo-bg-hover);
		color: var(--oo-fg-primary);
	}
	.oo-icon-btn[aria-expanded='true'] {
		color: var(--oo-fg-primary);
	}
	.oo-icon-btn[data-variant='primary'] {
		background-color: var(--oo-acc-fill);
		color: var(--oo-fg-on-accent);
		border-color: var(--oo-edge);
	}
	.oo-icon-btn[data-variant='primary']:hover:not(:disabled) {
		background-color: var(--oo-acc-fill-hover);
		color: var(--oo-fg-on-accent);
	}
	.oo-icon-btn:disabled {
		opacity: 0.55;
		cursor: not-allowed;
	}
	@media (prefers-reduced-motion: reduce) {
		.oo-icon-btn {
			transition: none;
		}
	}
</style>
