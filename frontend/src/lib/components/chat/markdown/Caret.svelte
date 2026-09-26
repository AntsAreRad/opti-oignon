<script lang="ts">
	/**
	 * Caret.svelte -- the streaming caret: a 2 px bar drawn after the last
	 * character of a reply that is still arriving. It blinks through the app's
	 * animate-cursor-blink class, which the reduced-motion rule stills, and is
	 * hidden from screen readers.
	 */
</script>

<span class="oo-md-caret animate-cursor-blink" aria-hidden="true"></span>

<style>
	.oo-md-caret {
		display: inline-block;
		width: 2px;
		height: 1.1em;
		margin-left: 1px;
		vertical-align: text-bottom;
		background-color: var(--oo-acc-mark);
	}
</style>
