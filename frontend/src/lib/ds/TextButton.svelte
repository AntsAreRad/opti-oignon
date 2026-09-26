<script lang="ts">
	/**
	 * TextButton.svelte (lib/ds) -- a quiet text button: the label in the
	 * surrounding text's colour, no border and no fill until it is hovered or
	 * focused. It forwards the two attributes a disclosure needs: expanded
	 * (aria-expanded, omitted when undefined) and controls (aria-controls, the
	 * id of the region it shows or hides). Deliberately small; the primitives'
	 * full button set grows from here.
	 */
	/** Whether the region this button shows or hides is shown. */
	export let expanded: boolean | undefined = undefined;
	/** The id of the region this button shows or hides. */
	export let controls: string | undefined = undefined;
</script>

<button
	type="button"
	class="oo-text-btn"
	aria-expanded={expanded}
	aria-controls={controls}
	on:click><slot /></button>

<style>
	.oo-text-btn {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-1);
		min-height: 28px;
		padding: 0 var(--oo-space-3);
		border: 0;
		border-radius: var(--oo-radius-md);
		background: transparent;
		color: inherit;
		font: inherit;
		font-family: var(--oo-font-sans);
		font-size: var(--oo-text-sm);
		line-height: 1.2;
		cursor: pointer;
		transition: background-color var(--oo-motion-fast) var(--oo-ease-default);
	}

	.oo-text-btn:hover,
	.oo-text-btn:focus-visible {
		background-color: var(--oo-bg-hover);
	}
</style>
