<!--
  Tabs.svelte (lib/ds) -- WAI-ARIA tabs (spec 10.9).
  tablist / tab / tabpanel semantics with roving tabindex and arrow-key
  navigation (Home/End supported). The active panel content is provided
  via the default slot; the consumer switches content on `value`.
-->
<script lang="ts">
	import { createEventDispatcher, tick } from 'svelte';
	import Icon from './Icon.svelte';
	import type { Size, TabItem } from './types';

	export let value: string;
	export let tabs: TabItem[] = [];
	export let orientation: 'horizontal' | 'vertical' = 'horizontal';
	export let variant: 'underline' | 'pill' = 'underline';
	export let size: Size = 'md';

	const dispatch = createEventDispatcher<{ change: string }>();
	const uid = `oo-tabs-${Math.random().toString(36).slice(2, 9)}`;
	let tabEls: HTMLButtonElement[] = [];

	function select(id: string) {
		if (id === value) return;
		value = id;
		dispatch('change', id);
	}

	async function focusIndex(index: number) {
		const count = tabs.length;
		if (count === 0) return;
		const wrapped = ((index % count) + count) % count;
		select(tabs[wrapped].id);
		await tick();
		tabEls[wrapped]?.focus();
	}

	function onKeydown(event: KeyboardEvent, index: number) {
		const next = orientation === 'horizontal' ? 'ArrowRight' : 'ArrowDown';
		const prev = orientation === 'horizontal' ? 'ArrowLeft' : 'ArrowUp';
		if (event.key === next) {
			event.preventDefault();
			focusIndex(index + 1);
		} else if (event.key === prev) {
			event.preventDefault();
			focusIndex(index - 1);
		} else if (event.key === 'Home') {
			event.preventDefault();
			focusIndex(0);
		} else if (event.key === 'End') {
			event.preventDefault();
			focusIndex(tabs.length - 1);
		}
	}
</script>

<div class="oo-tabs" data-orientation={orientation}>
	<div
		role="tablist"
		aria-orientation={orientation}
		class="oo-tablist"
		data-variant={variant}
		data-size={size}
	>
		{#each tabs as tab, i (tab.id)}
			<button
				type="button"
				role="tab"
				id={`${uid}-tab-${tab.id}`}
				aria-selected={value === tab.id}
				aria-controls={`${uid}-panel`}
				tabindex={value === tab.id ? 0 : -1}
				class="oo-tab"
				data-variant={variant}
				data-size={size}
				bind:this={tabEls[i]}
				on:click={() => select(tab.id)}
				on:keydown={(e) => onKeydown(e, i)}
			>
				{#if tab.icon}<Icon name={tab.icon} size="sm" />{/if}
				<span>{tab.label}</span>
			</button>
		{/each}
	</div>
	<div
		role="tabpanel"
		id={`${uid}-panel`}
		aria-labelledby={`${uid}-tab-${value}`}
		tabindex="0"
		class="oo-tabpanel"
	>
		<slot />
	</div>
</div>

<style>
	.oo-tabs[data-orientation='vertical'] {
		display: flex;
		gap: var(--oo-space-5);
	}
	.oo-tablist {
		display: flex;
		gap: var(--oo-space-1);
	}
	.oo-tabs[data-orientation='vertical'] .oo-tablist {
		flex-direction: column;
	}
	.oo-tab {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-2);
		border: 1px solid transparent;
		background: transparent;
		color: var(--oo-fg-secondary);
		cursor: pointer;
		font-family: var(--oo-font-sans);
		white-space: nowrap;
		transition: color var(--oo-motion-fast) var(--oo-ease-default);
	}
	.oo-tab[data-size='sm'] {
		font-size: var(--oo-text-xs);
		padding: var(--oo-space-2) var(--oo-space-3);
	}
	.oo-tab[data-size='md'] {
		font-size: var(--oo-text-sm);
		padding: var(--oo-space-3) var(--oo-space-4);
	}
	.oo-tab[data-size='lg'] {
		font-size: var(--oo-text-base);
		padding: var(--oo-space-3) var(--oo-space-5);
	}
	.oo-tab:hover {
		color: var(--oo-fg-primary);
	}

	/* Underline variant: the selected tab is underlined in the mark token
	   and set at weight 600; the list draws no line. */
	.oo-tab[data-variant='underline'] {
		border-radius: var(--oo-radius-sm);
	}
	.oo-tab[data-variant='underline'][aria-selected='true'] {
		color: var(--oo-fg-primary);
		font-weight: 600;
		text-decoration: underline;
		text-decoration-color: var(--oo-acc-mark);
		text-decoration-thickness: 2px;
		text-underline-offset: 6px;
	}

	/* Pill variant: the selected tab takes the accent fill, its edge and its weight. */
	.oo-tab[data-variant='pill'] {
		border-radius: var(--oo-radius-full);
	}
	.oo-tab[data-variant='pill'][aria-selected='true'] {
		color: var(--oo-fg-on-accent);
		background-color: var(--oo-acc-fill);
		border-color: var(--oo-edge);
		font-weight: 600;
	}

	/* The panel takes focus (tabindex 0), so it keeps the global focus ring. */
	.oo-tabpanel {
		margin-top: var(--oo-space-4);
	}
	.oo-tabs[data-orientation='vertical'] .oo-tabpanel {
		margin-top: 0;
		flex: 1;
	}
	/* Forced colours drop the fill: the selected pill takes the system's
	   selection colours, beside its weight. */
	@media (forced-colors: active) {
		.oo-tab[data-variant='pill'][aria-selected='true'] {
			background-color: SelectedItem;
			color: SelectedItemText;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.oo-tab {
			transition: none;
		}
	}
</style>
