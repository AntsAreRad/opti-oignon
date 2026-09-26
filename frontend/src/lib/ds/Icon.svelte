<!--
  Icon.svelte (lib/ds) -- draws an icon by name, decorative (aria-hidden):
  the control that holds it carries the name. The interface's own set
  (icons.ts) is drawn first, stroked in the text colour at 1.5; a name the
  set has not drawn yet falls back to the older icon package, at the same
  stroke. A name is looked up in kebab-case, so PascalCase and snake_case
  resolve too. Sizes: sm = 16, md = 20, lg = 24.
-->
<script lang="ts">
	import { icons } from 'lucide-svelte';
	import { iconKey, inlineIcon } from './icons';
	import type { IconName } from './types';

	export let name: IconName;
	export let size: 'sm' | 'md' | 'lg' = 'md';
	export let strokeWidth = 1.5;
	/** Extra class names forwarded to the underlying SVG. */
	let className = '';
	export { className as class };

	const PX: Record<'sm' | 'md' | 'lg', number> = { sm: 16, md: 20, lg: 24 };

	function toPascal(raw: string): string {
		return raw
			.split(/[-_\s]+/)
			.filter(Boolean)
			.map((part) => part.charAt(0).toUpperCase() + part.slice(1))
			.join('');
	}

	$: shapes = inlineIcon(name);
	$: key = /^[A-Z]/.test(name) ? name : toPascal(name);
	$: Cmp = shapes || !icons ? undefined : (icons as Record<string, unknown>)[key];
	// An unresolved name renders nothing; in development it says so, so a
	// typo is not an invisible icon.
	$: if (import.meta.env.DEV && name && !shapes && !Cmp) {
		console.warn(`[ds/Icon] unresolved icon name "${name}" (fallback key "${key}")`);
	}
</script>

{#if shapes}
	<svg
		class="oo-icon {className}"
		xmlns="http://www.w3.org/2000/svg"
		width={PX[size]}
		height={PX[size]}
		viewBox="0 0 24 24"
		fill="none"
		stroke="currentColor"
		stroke-width={strokeWidth}
		stroke-linecap="round"
		stroke-linejoin="round"
		aria-hidden="true"
		focusable="false"
		data-oo-icon={iconKey(name)}
	>
		{#each shapes as shape}
			{#if typeof shape === 'string'}
				<path d={shape} />
			{:else}
				<path d={shape.d} fill="currentColor" stroke="none" />
			{/if}
		{/each}
	</svg>
{:else if Cmp}
	<svelte:component
		this={Cmp}
		size={PX[size]}
		{strokeWidth}
		class={className}
		aria-hidden="true"
		focusable="false"
	/>
{/if}

<style>
	.oo-icon {
		flex-shrink: 0;
	}
</style>
