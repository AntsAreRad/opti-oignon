<!--
  SidePanel.svelte (lib/ds) -- a region beside the page: an inspector, an
  artifact, a tool's output.

  It is a labelled complementary landmark, never a dialog: the page beside
  it stays reachable, focus is not trapped and nothing else is hidden from
  assistive technology. On a phone (`overlay`) it stands over the right or
  left edge of the page instead of beside it: over the frame of the page
  that opens it (its nearest positioned ancestor), never fixed to the
  viewport, so the shell's header and its stop stay above it. That page
  draws whatever sits behind the panel, and a control that closes it. What
  a phone makes of a panel over the page (a sheet, a dialog) is decided
  with the phone's own layout, and checked there on the machine.

  Its inner edge is a resize handle: a focusable vertical separator that
  says the width in pixels, its bounds and the region it sizes. Its keys go
  through resizeKey (sidePanelSize.ts), a drag through dragWidth and the
  pointer's release through releaseWidth, so a width is always a whole pixel
  between the bounds; the arrows move the edge by `step`, Home and End take
  the panel to its narrowest and widest, and a press with no drag steps it
  through three widths, so a pointer resizes it without dragging. The
  handle's mark shows on hover, on focus and while dragging, and in the
  system's highlight under forced colours. A new width is written to
  `width` (bind it) and dispatched as `resize`; it reaches the element as
  run-time geometry, through an action, as a menu's position does.

  The panel is set apart from the page by tone: it sits on the surface,
  with the edge token as its border.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { clampWidth, dragWidth, releaseWidth, resizeKey } from './sidePanelSize';
	import type { PanelBounds } from './sidePanelSize';

	/** The region's name, for the landmark and its handle. */
	export let label: string;
	/** The width in px, on a desktop. */
	export let width = 400;
	export let min = 280;
	export let max = 640;
	/** How far one arrow key moves the edge, in px. */
	export let step = 16;
	/** The side of the page the panel stands on. */
	export let side: 'left' | 'right' = 'right';
	export let resizable = true;
	/** On a phone: over the page's edge, at its own width, with no handle. */
	export let overlay = false;
	let className = '';
	export { className as class };

	const dispatch = createEventDispatcher<{ resize: number }>();
	const uid = `oo-panel-${Math.random().toString(36).slice(2, 9)}`;

	let dragging = false;
	let startX = 0;
	let startWidth = 0;

	let bounds: PanelBounds;
	$: bounds = { min, max, step, side };
	$: size = clampWidth(width, bounds);

	function set(next: number) {
		if (next === width) return;
		width = next;
		dispatch('resize', next);
	}

	function onKey(event: KeyboardEvent) {
		const next = resizeKey(event.key, size, bounds);
		if (next === null) return;
		event.preventDefault();
		set(next);
	}

	function onPointerDown(event: PointerEvent) {
		if (event.button !== 0) return;
		event.preventDefault();
		dragging = true;
		startX = event.clientX;
		startWidth = size;
		(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
	}

	function onPointerMove(event: PointerEvent) {
		if (!dragging) return;
		set(dragWidth(startWidth, startX, event.clientX, bounds));
	}

	function onPointerEnd(event: PointerEvent) {
		if (!dragging) return;
		dragging = false;
		const handle = event.currentTarget as HTMLElement;
		if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
		if (event.type === 'pointerup') set(releaseWidth(startWidth, startX, event.clientX, bounds));
	}

	/** Gives the panel its width on a desktop; an overlay takes its own. */
	function sized(node: HTMLElement, px: number | null) {
		const apply = (value: number | null) => {
			node.style.width = value === null ? '' : `${value}px`;
		};
		apply(px);
		return { update: apply };
	}
</script>

<!-- svelte-ignore a11y-no-redundant-roles -->
<aside
	id={uid}
	role="complementary"
	aria-label={label}
	class="oo-side-panel {className}"
	data-side={side}
	data-overlay={overlay ? 'true' : undefined}
	class:oo-side-panel-dragging={dragging}
	use:sized={overlay ? null : size}
>
	{#if resizable && !overlay}
		<!-- svelte-ignore a11y-no-noninteractive-element-interactions a11y-no-noninteractive-tabindex -->
		<div
			role="separator"
			class="oo-side-panel-handle"
			aria-orientation="vertical"
			aria-label={`Resize ${label}`}
			aria-controls={uid}
			aria-valuenow={size}
			aria-valuetext={`${size} pixels`}
			aria-valuemin={min}
			aria-valuemax={max}
			tabindex="0"
			on:keydown={onKey}
			on:pointerdown={onPointerDown}
			on:pointermove={onPointerMove}
			on:pointerup={onPointerEnd}
			on:pointercancel={onPointerEnd}
		></div>
	{/if}
	<slot />
</aside>

<style>
	.oo-side-panel {
		position: relative;
		flex-shrink: 0;
		height: 100%;
		min-width: 0;
		background-color: var(--oo-bg-surface);
		border: 1px solid var(--oo-edge);
		color: var(--oo-fg-primary);
	}
	.oo-side-panel[data-overlay='true'] {
		position: absolute;
		top: 0;
		bottom: 0;
		right: 0;
		z-index: var(--oo-z-overlay);
		width: 100%;
		max-width: 90vw;
		box-shadow: var(--oo-shadow-lg);
	}
	.oo-side-panel[data-overlay='true'][data-side='left'] {
		right: auto;
		left: 0;
	}
	@media (min-width: 640px) {
		.oo-side-panel[data-overlay='true'] {
			max-width: 400px;
		}
	}

	.oo-side-panel-handle {
		position: absolute;
		top: 0;
		bottom: 0;
		left: -5px;
		z-index: 1;
		width: 9px;
		cursor: col-resize;
		touch-action: none;
	}
	.oo-side-panel[data-side='left'] .oo-side-panel-handle {
		left: auto;
		right: -5px;
	}
	.oo-side-panel-handle::after {
		content: '';
		position: absolute;
		top: 0;
		bottom: 0;
		left: 3px;
		width: 3px;
		border-radius: var(--oo-radius-full);
		background-color: transparent;
		transition: background-color var(--oo-motion-fast) var(--oo-ease-default);
	}
	.oo-side-panel-handle:hover::after,
	.oo-side-panel-handle:focus-visible::after,
	.oo-side-panel-dragging .oo-side-panel-handle::after {
		background-color: var(--oo-acc-mark);
	}
	.oo-side-panel-dragging {
		user-select: none;
	}
	@media (forced-colors: active) {
		.oo-side-panel-handle:hover::after,
		.oo-side-panel-handle:focus-visible::after,
		.oo-side-panel-dragging .oo-side-panel-handle::after {
			background-color: Highlight;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.oo-side-panel-handle::after {
			transition: none;
		}
	}
</style>
