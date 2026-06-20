<!--
  Switch.svelte (lib/ds) -- binary toggle for settings (spec 10.8).
  Renders <button role="switch" aria-checked>. Label is associated;
  optional description rendered below. Two-way bound via `checked`.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	export let checked = false;
	export let label: string;
	export let description: string | undefined = undefined;
	export let size: 'sm' | 'md' = 'md';
	export let disabled = false;

	const dispatch = createEventDispatcher<{ change: boolean }>();
	const uid = `oo-switch-${Math.random().toString(36).slice(2, 9)}`;

	function toggle() {
		if (disabled) return;
		checked = !checked;
		dispatch('change', checked);
	}
</script>

<div class="oo-switch-row" data-size={size}>
	<button
		type="button"
		role="switch"
		aria-checked={checked}
		aria-labelledby={`${uid}-label`}
		aria-describedby={description ? `${uid}-desc` : undefined}
		class="oo-switch"
		data-size={size}
		{disabled}
		on:click={toggle}
	>
		<span class="oo-switch-knob" aria-hidden="true"></span>
	</button>
	<div class="oo-switch-text">
		<span id={`${uid}-label`} class="oo-switch-label">{label}</span>
		{#if description}
			<span id={`${uid}-desc`} class="oo-switch-desc">{description}</span>
		{/if}
	</div>
</div>

<style>
	.oo-switch-row {
		display: flex;
		align-items: flex-start;
		gap: var(--oo-space-3);
	}
	.oo-switch {
		position: relative;
		flex-shrink: 0;
		border: 1px solid var(--oo-bd-strong);
		border-radius: var(--oo-radius-full);
		background-color: var(--oo-bg-overlay);
		cursor: pointer;
		padding: 0;
		transition:
			background-color var(--oo-motion-fast) var(--oo-ease-default),
			border-color var(--oo-motion-fast) var(--oo-ease-default);
	}
	.oo-switch[data-size='md'] {
		width: 40px;
		height: 22px;
	}
	.oo-switch[data-size='sm'] {
		width: 32px;
		height: 18px;
	}
	.oo-switch[aria-checked='true'] {
		background-color: var(--oo-acc-500);
		border-color: var(--oo-acc-500);
	}
	.oo-switch:disabled {
		opacity: 0.55;
		cursor: not-allowed;
	}
	.oo-switch-knob {
		position: absolute;
		top: 50%;
		left: 2px;
		transform: translateY(-50%);
		border-radius: var(--oo-radius-full);
		background-color: var(--oo-toggle-knob);
		transition: left var(--oo-motion-fast) var(--oo-ease-default);
	}
	.oo-switch[data-size='md'] .oo-switch-knob {
		width: 16px;
		height: 16px;
	}
	.oo-switch[data-size='sm'] .oo-switch-knob {
		width: 12px;
		height: 12px;
	}
	.oo-switch[aria-checked='true'][data-size='md'] .oo-switch-knob {
		left: 20px;
	}
	.oo-switch[aria-checked='true'][data-size='sm'] .oo-switch-knob {
		left: 16px;
	}
	.oo-switch-text {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
	}
	.oo-switch-label {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-primary);
	}
	.oo-switch-desc {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-secondary);
		line-height: var(--oo-leading-normal);
	}
	@media (prefers-reduced-motion: reduce) {
		.oo-switch,
		.oo-switch-knob {
			transition: none;
		}
	}
</style>
