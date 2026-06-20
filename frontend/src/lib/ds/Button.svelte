<!--
  Button.svelte (lib/ds) -- the canonical button primitive (spec 10.2).
  Variants: primary, secondary, ghost, danger, link. Sizes: sm, md, lg.
  Renders a semantic <button>, or an <a> when href is set. iconOnly
  requires ariaLabel. Focus ring comes from the global :focus-visible.
-->
<script lang="ts">
	import Icon from './Icon.svelte';
	import type { ButtonVariant, IconName, Size } from './types';

	export let variant: ButtonVariant = 'secondary';
	export let size: Size = 'md';
	export let iconLeft: IconName | undefined = undefined;
	export let iconRight: IconName | undefined = undefined;
	export let iconOnly: IconName | undefined = undefined;
	export let loading = false;
	export let disabled = false;
	export let type: 'button' | 'submit' | 'reset' = 'button';
	export let href: string | undefined = undefined;
	export let ariaLabel: string | undefined = undefined;
	/** Stretch to fill the container width. */
	export let block = false;

	const ICON_SIZE: Record<Size, 'sm' | 'md'> = { sm: 'sm', md: 'sm', lg: 'md' };

	$: isDisabled = disabled || loading;
	$: computedLabel = ariaLabel ?? undefined;
</script>

{#if href}
	<a
		{href}
		class="oo-btn"
		data-variant={variant}
		data-size={size}
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
		data-icon-only={iconOnly ? 'true' : undefined}
		class:oo-btn-block={block}
		disabled={isDisabled}
		aria-busy={loading}
		aria-label={computedLabel}
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
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: var(--oo-space-2);
		border: 1px solid transparent;
		border-radius: var(--oo-radius-md);
		font-family: var(--oo-font-sans);
		font-weight: 500;
		line-height: 1;
		letter-spacing: var(--oo-tracking-wide);
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

	/* Icon-only: square */
	.oo-btn[data-icon-only='true'] {
		padding: 0;
		aspect-ratio: 1 / 1;
	}
	.oo-btn[data-icon-only='true'][data-size='sm'] {
		width: 28px;
	}
	.oo-btn[data-icon-only='true'][data-size='md'] {
		width: 36px;
	}
	.oo-btn[data-icon-only='true'][data-size='lg'] {
		width: 44px;
	}

	/* Variants */
	.oo-btn[data-variant='primary'] {
		background-color: var(--oo-acc-500);
		color: var(--oo-fg-on-accent);
		border-color: var(--oo-acc-500);
	}
	.oo-btn[data-variant='primary']:hover:not(:disabled):not([aria-disabled='true']) {
		background-color: var(--oo-acc-600);
		border-color: var(--oo-acc-600);
	}

	.oo-btn[data-variant='secondary'] {
		background-color: var(--oo-bg-elevated);
		color: var(--oo-fg-primary);
		border-color: var(--oo-bd-default);
	}
	.oo-btn[data-variant='secondary']:hover:not(:disabled):not([aria-disabled='true']) {
		background-color: var(--oo-bg-hover);
		border-color: var(--oo-bd-strong);
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
		color: var(--oo-fg-on-accent);
		border-color: var(--oo-error);
	}
	.oo-btn[data-variant='danger']:hover:not(:disabled):not([aria-disabled='true']) {
		filter: brightness(0.93);
	}

	.oo-btn[data-variant='link'] {
		background-color: transparent;
		color: var(--oo-acc-500);
		border-color: transparent;
		padding-left: var(--oo-space-1);
		padding-right: var(--oo-space-1);
		min-height: auto;
		text-decoration: underline;
		text-underline-offset: 2px;
	}
	.oo-btn[data-variant='link']:hover:not(:disabled):not([aria-disabled='true']) {
		color: var(--oo-acc-400);
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
