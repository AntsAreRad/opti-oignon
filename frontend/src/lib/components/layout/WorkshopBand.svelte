<!--
  WorkshopBand.svelte
  The band over every Workshop page: it says, in the cool tint, that these
  pages act on the whole machine rather than on one person's work. The
  Workshop layout draws it once, above the page.
-->
<script lang="ts">
	import Icon from '$lib/ds/Icon.svelte';
</script>

<div class="oo-band">
	<Icon name="chip" size="sm" />
	<span class="oo-band-name">Workshop</span>
	<span class="oo-band-text">Settings and tools that act on the whole machine.</span>
</div>

<style>
	.oo-band {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--oo-space-2) var(--oo-space-3);
		padding: var(--oo-space-3) var(--oo-space-5);
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-lg);
		background-color: var(--oo-bg-tint-3);
		color: var(--oo-fg-primary);
		font-size: var(--oo-text-sm);
	}
	.oo-band-name {
		font-weight: 600;
	}
	.oo-band-text {
		color: var(--oo-fg-secondary);
	}
</style>
