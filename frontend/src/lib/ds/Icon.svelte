<!--
  Icon.svelte (lib/ds) -- wrapper around lucide-svelte.
  Standardizes the 231 inline SVG declarations and the inconsistent
  viewBox sizes behind a single name-based API. Color inherits
  currentColor. Sizes: sm = 16, md = 20, lg = 24 (spec 10.11).
-->
<script lang="ts">
	import { icons } from 'lucide-svelte';
	import type { IconName } from './types';

	export let name: IconName;
	export let size: 'sm' | 'md' | 'lg' = 'md';
	export let strokeWidth = 2;
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

	$: key = /^[A-Z]/.test(name) ? name : toPascal(name);
	$: Cmp = icons ? (icons as Record<string, unknown>)[key] : undefined;
	// DS-05 (S217): an unresolved name used to render nothing silently, so a
	// typo yielded an invisible icon. Warn in dev; production stays silent
	// and the render contract (no element on unresolved) is unchanged.
	$: if (import.meta.env.DEV && name && !Cmp) {
		console.warn(`[ds/Icon] unresolved icon name "${name}" (lucide key "${key}")`);
	}
</script>

{#if Cmp}
	<svelte:component
		this={Cmp}
		size={PX[size]}
		{strokeWidth}
		class={className}
		aria-hidden="true"
		focusable="false"
	/>
{/if}
