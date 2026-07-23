<!--
  ScrollToBottomFab.svelte
  Floating action button that appears when scrolled up from bottom.
  Smooth-scrolls to the latest message on click.
  48x48px touch target with tobacco accent color.
-->
<script lang="ts">
	export let visible: boolean = false;
	export let onClick: () => void = () => {};
</script>

<button
	class="scroll-fab {visible ? '' : 'hidden'}"
	on:click={onClick}
	title="Scroll to bottom"
	aria-label="Scroll to latest message"
	aria-hidden={!visible}
>
	<svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
		<path stroke-linecap="round" stroke-linejoin="round" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
	</svg>
</button>
