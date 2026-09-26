<!--
  The Workshop space: the operator pages, under /workshop. It marks its
  space on its root, is always drawn compact whatever the reader's density,
  and shows its band over every page.
-->
<script lang="ts">
	import WorkshopBand from '$lib/components/layout/WorkshopBand.svelte';
</script>

<div class="oo-workshop oo-density-compact" data-oo-space="workshop">
	<div class="oo-workshop-band">
		<WorkshopBand />
	</div>
	<div class="oo-workshop-page">
		<slot />
	</div>
</div>

<style>
	.oo-workshop {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
	}
	.oo-workshop-band {
		flex-shrink: 0;
		padding: var(--oo-space-4) var(--oo-space-4) 0;
	}
	/* A page that brings no scroll of its own scrolls here, under the band. */
	.oo-workshop-page {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
	}
</style>
