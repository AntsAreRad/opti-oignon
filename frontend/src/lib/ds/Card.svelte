<!--
  Card.svelte (lib/ds) -- surface container (spec 10.4).
  Variants: flat (border only), raised (border + shadow-sm).
  Padding sm/md/lg maps to --oo-space-3 / -5 / -7.
-->
<script lang="ts">
	export let variant: 'flat' | 'raised' = 'flat';
	export let padding: 'none' | 'sm' | 'md' | 'lg' = 'md';
	/** Optional element tag (e.g. 'section', 'article'). */
	export let as: string = 'div';
	let className = '';
	export { className as class };
</script>

<svelte:element
	this={as}
	class="oo-card {className}"
	data-variant={variant}
	data-padding={padding}
	on:click
	on:keydown
>
	<slot />
</svelte:element>

<style>
	.oo-card {
		background-color: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-lg);
		color: var(--oo-fg-primary);
	}
	.oo-card[data-variant='raised'] {
		box-shadow: var(--oo-shadow-sm);
	}
	.oo-card[data-padding='none'] {
		padding: 0;
	}
	.oo-card[data-padding='sm'] {
		padding: var(--oo-space-3);
	}
	.oo-card[data-padding='md'] {
		padding: var(--oo-space-5);
	}
	.oo-card[data-padding='lg'] {
		padding: var(--oo-space-7);
	}
</style>
