<!--
  Button.svelte (lib/ds) -- the button every surface draws.
  Variants: primary, secondary, ghost, danger, link. Sizes: sm, md, lg.
  Shapes: rect (the default), pill, round (a circle, for an icon alone: a
  label in it is clipped). Renders a <button>, or an <a> when href is set.
  An icon alone needs its accessible name: iconOnly without a non-blank
  ariaLabel refuses to render, as the icon button does. The focus ring
  comes from the global :focus-visible.

  A button that toggles, opens a menu or shows a region says so: pressed,
  expanded, haspopup and controls become aria-pressed, aria-expanded,
  aria-haspopup and aria-controls, each omitted when not given; a link
  carries no state, and takes none of them. A pressed button draws a check,
  beside its label or at the corner of its icon, in the accent ink on the
  tint wherever it sits on a quiet ground, so its state never rests on a
  colour alone; a pressed quiet button keeps its tint under the pointer,
  with the hover wash over it.

  A toned ground (the accent fill, the secondary surface, the pressed tint)
  carries the edge token as its border: transparent where tone separates
  it, drawn in high contrast.
-->
<script lang="ts">
	import Icon from './Icon.svelte';
	import type { ButtonShape, ButtonVariant, IconName, PopupKind, Size } from './types';

	export let variant: ButtonVariant = 'secondary';
	export let size: Size = 'md';
	export let shape: ButtonShape = 'rect';
	export let iconLeft: IconName | undefined = undefined;
	export let iconRight: IconName | undefined = undefined;
	export let iconOnly: IconName | undefined = undefined;
	export let loading = false;
	export let disabled = false;
	export let type: 'button' | 'submit' | 'reset' = 'button';
	export let href: string | undefined = undefined;
	export let ariaLabel: string | undefined = undefined;
	/** A toggle button's state (aria-pressed); omitted when undefined. */
	export let pressed: boolean | undefined = undefined;
	/** Whether what this button opens is open (aria-expanded); omitted when undefined. */
	export let expanded: boolean | undefined = undefined;
	/** What this button opens (aria-haspopup); omitted when undefined. */
	export let haspopup: PopupKind | undefined = undefined;
	/** The id of the element this button shows or controls (aria-controls). */
	export let controls: string | undefined = undefined;
	/** Stretch to fill the container width. */
	export let block = false;

	const ICON_SIZE: Record<Size, 'sm' | 'md'> = { sm: 'sm', md: 'sm', lg: 'md' };
	// The corner check is drawn at 10 px: a stroke of 3 on its 24 unit box
	// keeps its line over a pixel wide.
	const CORNER_STROKE = 3;

	$: isDisabled = disabled || loading;
	$: computedLabel = ariaLabel ?? undefined;
	$: if (iconOnly && (typeof ariaLabel !== 'string' || ariaLabel.trim() === '')) {
		throw new Error(`Button with the icon "${iconOnly}" alone needs an accessible name: give it an ariaLabel`);
	}
</script>

{#if href}
	<a
		{href}
		class="oo-btn"
		data-variant={variant}
		data-size={size}
		data-shape={shape}
		data-icon-only={iconOnly ? 'true' : undefined}
		class:oo-btn-block={block}
		aria-label={computedLabel}
		aria-disabled={isDisabled ? 'true' : undefined}
		tabindex={isDisabled ? -1 : undefined}
		on:click
		on:keydown
		on:focus
		on:blur
		on:mouseenter
		on:mouseleave
	>
		{#if iconOnly}
			<Icon name={iconOnly} size={ICON_SIZE[size]} />
		{:else}
			{#if iconLeft}<Icon name={iconLeft} size={ICON_SIZE[size]} />{/if}
			<span class="oo-btn-label"><slot /></span>
			{#if iconRight}<Icon name={iconRight} size={ICON_SIZE[size]} />{/if}
		{/if}
	</a>
{:else}
	<button
		{type}
		class="oo-btn"
		data-variant={variant}
		data-size={size}
		data-shape={shape}
		data-icon-only={iconOnly ? 'true' : undefined}
		class:oo-btn-block={block}
		disabled={isDisabled}
		aria-busy={loading}
		aria-label={computedLabel}
		aria-pressed={pressed}
		aria-expanded={expanded}
		aria-haspopup={haspopup}
		aria-controls={controls}
		on:click
		on:keydown
		on:focus
		on:blur
		on:mouseenter
		on:mouseleave
	>
		{#if loading}
			<span class="oo-btn-spinner" aria-hidden="true"></span>
		{/if}
		{#if pressed}
			<span class="oo-btn-check" class:oo-btn-check-corner={!!iconOnly} aria-hidden="true">
				{#if iconOnly}
					<Icon name="check" size={ICON_SIZE[size]} strokeWidth={CORNER_STROKE} />
				{:else}
					<Icon name="check" size={ICON_SIZE[size]} />
				{/if}
			</span>
		{/if}
		{#if iconOnly}
			{#if !loading}<Icon name={iconOnly} size={ICON_SIZE[size]} />{/if}
		{:else}
			{#if iconLeft && !loading}<Icon name={iconLeft} size={ICON_SIZE[size]} />{/if}
			<span class="oo-btn-label"><slot /></span>
			{#if iconRight}<Icon name={iconRight} size={ICON_SIZE[size]} />{/if}
		{/if}
	</button>
{/if}

<style>
	.oo-btn {
		position: relative;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: var(--oo-space-2);
		border: 1px solid transparent;
		border-radius: var(--oo-radius-md);
		font-family: var(--oo-font-sans);
		font-weight: 500;
		line-height: 1;
		cursor: pointer;
		text-decoration: none;
		white-space: nowrap;
		transition:
			background-color var(--oo-motion-fast) var(--oo-ease-default),
			border-color var(--oo-motion-fast) var(--oo-ease-default),
			color var(--oo-motion-fast) var(--oo-ease-default);
	}

	.oo-btn-block {
		width: 100%;
	}

	/* Sizes */
	.oo-btn[data-size='sm'] {
		font-size: var(--oo-text-xs);
		padding: var(--oo-space-2) var(--oo-space-3);
		min-height: 28px;
	}
	.oo-btn[data-size='md'] {
		font-size: var(--oo-text-sm);
		padding: var(--oo-space-3) var(--oo-space-4);
		min-height: 36px;
	}
	.oo-btn[data-size='lg'] {
		font-size: var(--oo-text-base);
		padding: var(--oo-space-4) var(--oo-space-5);
		min-height: 44px;
	}

	/* Shapes. A pill keeps its padding; a circle, like an icon alone, is
	   square and holds no padding. */
	.oo-btn[data-shape='pill'] {
		border-radius: var(--oo-radius-full);
	}
	.oo-btn[data-shape='round'] {
		border-radius: var(--oo-radius-full);
		aspect-ratio: 1 / 1;
		padding: 0;
	}

	/* Icon-only: square */
	.oo-btn[data-icon-only='true'] {
		padding: 0;
		aspect-ratio: 1 / 1;
	}
	.oo-btn[data-icon-only='true'][data-size='sm'],
	.oo-btn[data-shape='round'][data-size='sm'] {
		width: 28px;
	}
	.oo-btn[data-icon-only='true'][data-size='md'],
	.oo-btn[data-shape='round'][data-size='md'] {
		width: 36px;
	}
	.oo-btn[data-icon-only='true'][data-size='lg'],
	.oo-btn[data-shape='round'][data-size='lg'] {
		width: 44px;
	}

	/* Variants */
	.oo-btn[data-variant='primary'] {
		background-color: var(--oo-acc-fill);
		color: var(--oo-fg-on-accent);
		border-color: var(--oo-edge);
	}
	.oo-btn[data-variant='primary']:hover:not(:disabled):not([aria-disabled='true']) {
		background-color: var(--oo-acc-fill-hover);
		color: var(--oo-fg-on-accent);
	}

	.oo-btn[data-variant='secondary'] {
		background-color: var(--oo-btn-secondary-bg);
		color: var(--oo-fg-primary);
		border-color: var(--oo-edge);
	}
	.oo-btn[data-variant='secondary']:hover:not(:disabled):not([aria-disabled='true']) {
		background-color: var(--oo-btn-secondary-hover);
	}

	.oo-btn[data-variant='ghost'] {
		background-color: transparent;
		color: var(--oo-fg-secondary);
		border-color: transparent;
	}
	.oo-btn[data-variant='ghost']:hover:not(:disabled):not([aria-disabled='true']) {
		background-color: var(--oo-bg-hover);
		color: var(--oo-fg-primary);
	}

	.oo-btn[data-variant='danger'] {
		background-color: var(--oo-error);
		color: var(--oo-fg-on-semantic);
		border-color: var(--oo-error);
	}
	.oo-btn[data-variant='danger']:hover:not(:disabled):not([aria-disabled='true']) {
		filter: brightness(0.93);
	}

	.oo-btn[data-variant='link'] {
		background-color: transparent;
		color: var(--oo-acc-ink);
		border-color: transparent;
		padding-left: var(--oo-space-1);
		padding-right: var(--oo-space-1);
		min-height: auto;
		text-decoration: underline;
		text-underline-offset: 2px;
	}
	.oo-btn[data-variant='link']:hover:not(:disabled):not([aria-disabled='true']) {
		color: var(--oo-fg-primary);
	}

	/* A pressed quiet button takes the accent tint, with its check; under the
	   pointer it keeps the tint, with the hover wash over it. */
	.oo-btn[data-variant='secondary'][aria-pressed='true'],
	.oo-btn[data-variant='ghost'][aria-pressed='true'] {
		background-color: var(--oo-bg-tint-1);
		color: var(--oo-fg-primary);
		border-color: var(--oo-edge);
	}
	.oo-btn[data-variant='secondary'][aria-pressed='true']:hover:not(:disabled):not([aria-disabled='true']),
	.oo-btn[data-variant='ghost'][aria-pressed='true']:hover:not(:disabled):not([aria-disabled='true']) {
		background-color: color-mix(in srgb, var(--oo-fg-primary) 4%, var(--oo-bg-tint-1));
	}

	.oo-btn-check {
		display: inline-flex;
		color: var(--oo-acc-ink);
	}
	/* On a filled button the check beside the label takes the button's own
	   ink; the corner badge has its own ground, the tint, and keeps the
	   accent ink. */
	.oo-btn[data-variant='primary'] .oo-btn-check:not(.oo-btn-check-corner),
	.oo-btn[data-variant='danger'] .oo-btn-check:not(.oo-btn-check-corner) {
		color: inherit;
	}
	/* Beside an icon alone, the check sits at the corner, inside the button's
	   box, so nothing that clips the button clips it. */
	.oo-btn-check-corner {
		position: absolute;
		top: 0;
		right: 0;
		width: 12px;
		height: 12px;
		border-radius: var(--oo-radius-full);
		background-color: var(--oo-bg-tint-1);
		border: 1px solid var(--oo-edge);
	}
	.oo-btn-check-corner :global(svg) {
		width: 10px;
		height: 10px;
		margin: auto;
	}

	.oo-btn:disabled,
	.oo-btn[aria-disabled='true'] {
		opacity: 0.55;
		cursor: not-allowed;
		pointer-events: none;
	}

	.oo-btn-spinner {
		width: 0.9em;
		height: 0.9em;
		border: 2px solid currentColor;
		border-top-color: transparent;
		border-radius: var(--oo-radius-full);
		animation: oo-btn-spin var(--oo-motion-slow) linear infinite;
	}

	@keyframes oo-btn-spin {
		to {
			transform: rotate(360deg);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.oo-btn-spinner {
			animation-duration: 1.2s;
		}
	}
</style>
