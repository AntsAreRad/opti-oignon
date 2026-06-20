<!--
  EmptyState.svelte (lib/ds) -- S170.
  The single empty-state surface for lists and panels (spec 12.5, Goal 2).
  Centred icon + title + optional description + an optional actions slot.
  Token-only; the icon is decorative (aria-hidden). Sizes: sm (inline panel)
  and md (full route/list). Use one EmptyState per list-rendering surface
  instead of ad-hoc markup so every empty surface looks and reads the same.
-->
<script lang="ts">
	import Icon from './Icon.svelte';
	import type { IconName } from './types';

	/** Decorative lucide icon shown above the title. */
	export let icon: IconName | undefined = undefined;
	/** Primary line (what is empty). Always provided. */
	export let title: string;
	/** Optional secondary line (how to populate it). */
	export let description: string | undefined = undefined;
	/** sm = compact (inside a panel), md = full (route-level list). */
	export let size: 'sm' | 'md' = 'md';
</script>

<div class="oo-empty" data-size={size}>
	{#if icon}
		<span class="oo-empty-icon" aria-hidden="true">
			<Icon name={icon} size={size === 'sm' ? 'md' : 'lg'} />
		</span>
	{/if}
	<p class="oo-empty-title">{title}</p>
	{#if description}
		<p class="oo-empty-desc">{description}</p>
	{/if}
	{#if $$slots.default}
		<div class="oo-empty-actions">
			<slot />
		</div>
	{/if}
</div>

<style>
	.oo-empty {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		text-align: center;
		color: var(--oo-fg-muted);
	}
	.oo-empty[data-size='sm'] {
		padding: var(--oo-space-6) var(--oo-space-4);
		gap: var(--oo-space-2);
	}
	.oo-empty[data-size='md'] {
		padding: var(--oo-space-9) var(--oo-space-5);
		gap: var(--oo-space-3);
	}

	.oo-empty-icon {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		color: var(--oo-fg-tertiary);
		margin-bottom: var(--oo-space-1);
	}
	.oo-empty[data-size='md'] .oo-empty-icon {
		width: 56px;
		height: 56px;
		border-radius: var(--oo-radius-full);
		background-color: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-subtle);
	}

	.oo-empty-title {
		margin: 0;
		color: var(--oo-fg-secondary);
		font-weight: 500;
	}
	.oo-empty[data-size='sm'] .oo-empty-title {
		font-size: var(--oo-text-sm);
	}
	.oo-empty[data-size='md'] .oo-empty-title {
		font-size: var(--oo-text-base);
	}

	.oo-empty-desc {
		margin: 0;
		max-width: 42ch;
		font-size: var(--oo-text-xs);
		line-height: var(--oo-leading-normal);
		color: var(--oo-fg-muted);
	}

	.oo-empty-actions {
		margin-top: var(--oo-space-3);
		display: flex;
		gap: var(--oo-space-2);
		flex-wrap: wrap;
		justify-content: center;
	}
</style>
