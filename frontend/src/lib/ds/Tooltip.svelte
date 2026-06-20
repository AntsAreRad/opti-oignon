<!--
  Tooltip.svelte (lib/ds) -- hover/focus popover (spec 10.10).
  Positioned with @floating-ui/dom (offset + flip + shift). Wraps a
  trigger in the default slot; content via the `content` prop or the
  named `content` slot. role="tooltip" wired via aria-describedby.
-->
<script lang="ts">
	import { onDestroy } from 'svelte';
	import { computePosition, flip, shift, offset, autoUpdate } from '@floating-ui/dom';
	import type { TooltipPlacement } from './types';

	export let content = '';
	export let placement: TooltipPlacement = 'top';
	export let delay = 200;
	export let disabled = false;

	const uid = `oo-tip-${Math.random().toString(36).slice(2, 9)}`;
	let triggerEl: HTMLElement;
	let tipEl: HTMLElement;
	let visible = false;
	let openTimer: ReturnType<typeof setTimeout> | undefined;
	let cleanup: (() => void) | undefined;

	function position() {
		if (!triggerEl || !tipEl) return;
		computePosition(triggerEl, tipEl, {
			placement,
			middleware: [offset(8), flip(), shift({ padding: 8 })]
		}).then(({ x, y }) => {
			if (!tipEl) return;
			tipEl.style.left = `${x}px`;
			tipEl.style.top = `${y}px`;
		});
	}

	function show() {
		if (disabled) return;
		clearTimeout(openTimer);
		openTimer = setTimeout(() => {
			visible = true;
			queueMicrotask(() => {
				if (triggerEl && tipEl) {
					cleanup = autoUpdate(triggerEl, tipEl, position);
				}
			});
		}, delay);
	}

	function hide() {
		clearTimeout(openTimer);
		visible = false;
		cleanup?.();
		cleanup = undefined;
	}

	function onKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') hide();
	}

	onDestroy(() => {
		clearTimeout(openTimer);
		cleanup?.();
	});
</script>

<span
	class="oo-tip-trigger"
	bind:this={triggerEl}
	aria-describedby={visible ? uid : undefined}
	on:mouseenter={show}
	on:mouseleave={hide}
	on:focusin={show}
	on:focusout={hide}
	on:keydown={onKeydown}
>
	<slot />
</span>

{#if visible && !disabled}
	<div id={uid} role="tooltip" class="oo-tip" bind:this={tipEl}>
		{#if $$slots.content}
			<slot name="content" />
		{:else}
			{content}
		{/if}
	</div>
{/if}

<style>
	.oo-tip-trigger {
		display: inline-flex;
	}
	.oo-tip {
		position: absolute;
		top: 0;
		left: 0;
		z-index: var(--oo-z-tooltip);
		max-width: 16rem;
		padding: var(--oo-space-2) var(--oo-space-3);
		background-color: var(--oo-bg-overlay);
		color: var(--oo-fg-primary);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-sm);
		box-shadow: var(--oo-shadow-md);
		font-size: var(--oo-text-xs);
		line-height: var(--oo-leading-normal);
		pointer-events: none;
		width: max-content;
	}
</style>
