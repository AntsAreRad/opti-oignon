<!--
  Checkbox.svelte (lib/ds) -- one native checkbox with its visible label.

  The input is the browser's own, so its keyboard, its state and what a
  screen reader says of it are native; the label wraps it, so a press on
  the text toggles it, and the label alone is its name. The mixed state
  (some of a group chosen) is the input's own indeterminate property, which
  a click clears. It is drawn in the accent ink (accent-color). An optional
  description sits under the label, outside it, and is the input's
  description (aria-describedby), so it is said once, after the name. Bind
  `checked` and `indeterminate`; `change` carries the new state.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	/** The visible label. */
	export let label: string;
	export let checked = false;
	/** The mixed state: neither checked nor unchecked. */
	export let indeterminate = false;
	export let description: string | undefined = undefined;
	export let disabled = false;

	const dispatch = createEventDispatcher<{ change: boolean }>();
	const uid = `oo-check-${Math.random().toString(36).slice(2, 9)}`;
</script>

<div class="oo-check" class:oo-check-disabled={disabled}>
	<label class="oo-check-row">
		<input
			type="checkbox"
			class="oo-check-box"
			bind:checked
			bind:indeterminate
			{disabled}
			aria-describedby={description ? `${uid}-desc` : undefined}
			on:change={() => dispatch('change', checked)}
		/>
		<span class="oo-check-label">{label}</span>
	</label>
	{#if description}
		<span id={`${uid}-desc`} class="oo-check-desc">{description}</span>
	{/if}
</div>

<style>
	.oo-check {
		display: inline-flex;
		flex-direction: column;
		gap: var(--oo-space-1);
		color: var(--oo-fg-primary);
		font-family: var(--oo-font-sans);
		font-size: var(--oo-text-sm);
		line-height: var(--oo-leading-snug);
	}
	.oo-check-row {
		display: inline-flex;
		align-items: flex-start;
		gap: var(--oo-space-3);
		cursor: pointer;
	}
	.oo-check-box {
		flex-shrink: 0;
		width: 18px;
		height: 18px;
		margin: 1px 0 0;
		accent-color: var(--oo-acc-ink);
		cursor: pointer;
	}
	.oo-check-desc {
		padding-left: calc(18px + var(--oo-space-3));
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-xs);
		line-height: var(--oo-leading-normal);
	}
	.oo-check-disabled {
		opacity: 0.55;
	}
	.oo-check-disabled .oo-check-row,
	.oo-check-disabled .oo-check-box {
		cursor: not-allowed;
	}
</style>
