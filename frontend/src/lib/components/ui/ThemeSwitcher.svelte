<!--
  ThemeSwitcher.svelte (S167)
  Header palette quick-switcher (spec 8.4). One click to move between the
  5 curated palettes without opening Settings, plus the 3 density modes.
  Wired to the preferences store, which sets `data-oo-theme` +
  `html.oo-density-*` and persists the choice.

  Accessibility: a button (aria-haspopup, aria-expanded) opens a
  role="menu" with role="menuitemradio" options (single selection per
  group). Full keyboard support: Arrow/Home/End move focus (roving
  tabindex), Enter/Space select, Escape closes and restores focus.
-->
<script lang="ts">
	import { onDestroy } from 'svelte';
	import Icon from '$lib/ds/Icon.svelte';
	import {
		palette,
		density,
		setPalette,
		setDensity,
		PALETTES,
		PALETTE_LABELS,
		PALETTE_SWATCH,
		DENSITIES,
		DENSITY_LABELS,
		type ThemePalette,
		type Density
	} from '$lib/stores/preferences';

	let open = false;
	let triggerEl: HTMLButtonElement;
	let menuEl: HTMLDivElement;
	let itemEls: HTMLButtonElement[] = [];
	let activeIndex = 0;

	// Flat list of focusable items in DOM order: palettes then densities.
	$: items = [
		...PALETTES.map((value) => ({ kind: 'palette' as const, value })),
		...DENSITIES.map((value) => ({ kind: 'density' as const, value }))
	];

	function selectedIndex(): number {
		const i = PALETTES.indexOf($palette);
		return i >= 0 ? i : 0;
	}

	function openMenu() {
		open = true;
		activeIndex = selectedIndex();
		queueMicrotask(() => focusItem(activeIndex));
	}

	function closeMenu(restoreFocus = true) {
		open = false;
		if (restoreFocus && triggerEl) triggerEl.focus();
	}

	function toggle() {
		open ? closeMenu(false) : openMenu();
	}

	function focusItem(index: number) {
		const max = items.length - 1;
		activeIndex = index < 0 ? max : index > max ? 0 : index;
		itemEls[activeIndex]?.focus();
	}

	function choose(index: number) {
		const item = items[index];
		if (!item) return;
		if (item.kind === 'palette') setPalette(item.value as ThemePalette);
		else setDensity(item.value as Density);
		closeMenu();
	}

	function onTriggerKeydown(event: KeyboardEvent) {
		if (event.key === 'ArrowDown' || event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			openMenu();
		}
	}

	function onMenuKeydown(event: KeyboardEvent) {
		switch (event.key) {
			case 'ArrowDown':
				event.preventDefault();
				focusItem(activeIndex + 1);
				break;
			case 'ArrowUp':
				event.preventDefault();
				focusItem(activeIndex - 1);
				break;
			case 'Home':
				event.preventDefault();
				focusItem(0);
				break;
			case 'End':
				event.preventDefault();
				focusItem(items.length - 1);
				break;
			case 'Enter':
			case ' ':
				event.preventDefault();
				choose(activeIndex);
				break;
			case 'Escape':
				event.preventDefault();
				closeMenu();
				break;
			case 'Tab':
				closeMenu(false);
				break;
		}
	}

	function onWindowClick(event: MouseEvent) {
		const target = event.target as Node;
		if (open && !menuEl?.contains(target) && !triggerEl?.contains(target)) {
			closeMenu(false);
		}
	}

	onDestroy(() => {
		// Listener is added conditionally below; ensure cleanup if still open.
		if (typeof window !== 'undefined') window.removeEventListener('click', onWindowClick, true);
	});

	$: if (typeof window !== 'undefined') {
		window.removeEventListener('click', onWindowClick, true);
		if (open) window.addEventListener('click', onWindowClick, true);
	}
</script>

<div class="oo-theme-switcher">
	<button
		bind:this={triggerEl}
		type="button"
		class="oo-ts-trigger"
		aria-haspopup="true"
		aria-expanded={open}
		aria-label="Theme: {PALETTE_LABELS[$palette]}, density: {DENSITY_LABELS[$density]}"
		title="Theme and density"
		on:click={toggle}
		on:keydown={onTriggerKeydown}
	>
		<Icon name="palette" size="sm" />
	</button>

	{#if open}
		<div
			bind:this={menuEl}
			class="oo-ts-menu"
			role="menu"
			aria-label="Theme and density"
			tabindex="-1"
			on:keydown={onMenuKeydown}
		>
			<p class="oo-ts-group-label" id="oo-ts-palette-label">Palette</p>
			{#each PALETTES as p, i (p)}
				<button
					bind:this={itemEls[i]}
					type="button"
					class="oo-ts-item"
					role="menuitemradio"
					aria-checked={$palette === p}
					tabindex={activeIndex === i ? 0 : -1}
					on:click={() => choose(i)}
				>
					<span
						class="oo-ts-swatch"
						style="background-color: {PALETTE_SWATCH[p].base}; border-color: {PALETTE_SWATCH[p]
							.surface};"
						aria-hidden="true"
					>
						<span class="oo-ts-swatch-dot" style="background-color: {PALETTE_SWATCH[p].fg};"></span>
					</span>
					<span class="oo-ts-item-label">{PALETTE_LABELS[p]}</span>
					{#if $palette === p}
						<span class="oo-ts-check"><Icon name="check" size="sm" /></span>
					{/if}
				</button>
			{/each}

			<div class="oo-ts-sep" role="separator"></div>

			<p class="oo-ts-group-label" id="oo-ts-density-label">Density</p>
			{#each DENSITIES as d, j (d)}
				<button
					bind:this={itemEls[PALETTES.length + j]}
					type="button"
					class="oo-ts-item"
					role="menuitemradio"
					aria-checked={$density === d}
					tabindex={activeIndex === PALETTES.length + j ? 0 : -1}
					on:click={() => choose(PALETTES.length + j)}
				>
					<span class="oo-ts-density-glyph oo-ts-density-{d}" aria-hidden="true"></span>
					<span class="oo-ts-item-label">{DENSITY_LABELS[d]}</span>
					{#if $density === d}
						<span class="oo-ts-check"><Icon name="check" size="sm" /></span>
					{/if}
				</button>
			{/each}
		</div>
	{/if}
</div>

<style>
	.oo-theme-switcher {
		position: relative;
		display: inline-flex;
	}

	.oo-ts-trigger {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		padding: 5px;
		border-radius: var(--oo-radius-md);
		border: none;
		background: transparent;
		cursor: pointer;
		color: var(--oo-fg-tertiary);
		transition:
			background-color var(--oo-motion-fast) var(--oo-ease-default),
			color var(--oo-motion-fast) var(--oo-ease-default);
	}
	.oo-ts-trigger:hover {
		background-color: var(--oo-bg-elevated);
		color: var(--oo-fg-secondary);
	}

	.oo-ts-menu {
		position: absolute;
		top: calc(100% + 6px);
		right: 0;
		z-index: var(--oo-z-overlay);
		min-width: 200px;
		padding: var(--oo-space-1);
		border-radius: var(--oo-radius-lg);
		background-color: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-subtle);
		box-shadow: var(--oo-shadow-lg);
		display: flex;
		flex-direction: column;
		gap: 2px;
		transform-origin: top right;
		animation: oo-ts-menu-in var(--oo-motion-fast) var(--oo-ease-emphasized);
	}

	@keyframes oo-ts-menu-in {
		from {
			opacity: 0;
			transform: scale(0.97) translateY(-2px);
		}
		to {
			opacity: 1;
			transform: scale(1) translateY(0);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.oo-ts-menu {
			animation: none;
		}
	}

	.oo-ts-group-label {
		margin: 0;
		padding: var(--oo-space-1) var(--oo-space-2);
		font-size: var(--oo-text-2xs);
		text-transform: uppercase;
		letter-spacing: var(--oo-tracking-wide);
		color: var(--oo-fg-tertiary);
	}

	.oo-ts-item {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		width: 100%;
		padding: var(--oo-space-2);
		border: none;
		border-radius: var(--oo-radius-md);
		background: transparent;
		cursor: pointer;
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
		text-align: left;
		transition:
			background-color var(--oo-motion-fast) var(--oo-ease-default),
			color var(--oo-motion-fast) var(--oo-ease-default);
	}
	.oo-ts-item:hover {
		background-color: var(--oo-bg-surface);
		color: var(--oo-fg-primary);
	}
	.oo-ts-item[aria-checked='true'] {
		color: var(--oo-fg-primary);
	}

	.oo-ts-swatch {
		position: relative;
		width: 18px;
		height: 18px;
		border-radius: var(--oo-radius-md);
		border: 1px solid;
		flex-shrink: 0;
		display: inline-flex;
		align-items: center;
		justify-content: center;
	}
	.oo-ts-swatch-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
	}

	.oo-ts-density-glyph {
		width: 18px;
		height: 18px;
		flex-shrink: 0;
		border-radius: var(--oo-radius-sm);
		background-image: linear-gradient(currentColor 0 0);
		background-repeat: no-repeat;
		background-position: center;
		color: var(--oo-fg-tertiary);
	}
	/* Density glyphs differ in line spacing density. */
	.oo-ts-density-compact {
		background-image: linear-gradient(
			currentColor 0 0,
			currentColor 0 0,
			currentColor 0 0
		);
		background-size:
			12px 1.5px,
			12px 1.5px,
			12px 1.5px;
		background-position:
			center 5px,
			center 9px,
			center 13px;
	}
	.oo-ts-density-comfortable {
		background-image: linear-gradient(currentColor 0 0, currentColor 0 0);
		background-size:
			12px 1.5px,
			12px 1.5px;
		background-position:
			center 6px,
			center 12px;
	}
	.oo-ts-density-spacious {
		background-image: linear-gradient(currentColor 0 0);
		background-size: 12px 1.5px;
		background-position: center 9px;
	}

	.oo-ts-item-label {
		flex: 1;
	}

	.oo-ts-check {
		display: inline-flex;
		color: var(--oo-acc-500);
	}

	.oo-ts-sep {
		height: 1px;
		margin: var(--oo-space-1) 0;
		background-color: var(--oo-bd-subtle);
	}
</style>
