<!--
  Menu.svelte (lib/ds) -- a button that opens a short list of actions.

  The trigger is a Button (a label, with a chevron) or, when an icon is
  given, an IconButton named by the label; it says it opens a menu and
  whether the menu is open. The label is required: a missing or blank one
  refuses to render, as the icon button does. The list is a role=menu of
  role=menuitem buttons, placed beside the trigger by floating-ui, on the
  second surface with the shadow token.

  Keys go through menuKeys.ts, the one place that decides them: on the
  closed trigger, triggerKey opens the menu on the first item (the down
  arrow) or the last (the up arrow), and Enter and Space open it as a
  click does; in the list, and on the trigger while the menu is open,
  menuKey moves and wraps with the arrows, jumps with Home and End, closes
  on Escape and returns focus to the trigger, and closes on Tab and lets
  focus move on. Enter and Space activate an item natively. An item that
  cannot act is disabled and skipped by the arrows; when none can take
  focus, the list itself takes it, so Escape still closes the menu. The
  pointer over an item focuses it, so the arrows go on from there.
  Choosing an item closes the menu, returns focus to the trigger and
  dispatches `select` with its id; a press outside closes the menu where it
  is.
-->
<script lang="ts">
	import { createEventDispatcher, onDestroy, tick } from 'svelte';
	import { autoUpdate, computePosition, flip, offset, shift } from '@floating-ui/dom';
	import Button from './Button.svelte';
	import IconButton from './IconButton.svelte';
	import Icon from './Icon.svelte';
	import { menuKey, triggerKey } from './menuKeys';
	import type { ButtonVariant, IconName, MenuItem, MenuPlacement } from './types';

	/** The trigger's name: its visible text, or an icon trigger's accessible name. */
	export let label: string;
	export let items: MenuItem[] = [];
	/** Draw the trigger as an icon alone. */
	export let icon: IconName | undefined = undefined;
	/** The labelled trigger's variant. */
	export let variant: ButtonVariant = 'ghost';
	export let placement: MenuPlacement = 'bottom-end';

	const dispatch = createEventDispatcher<{ select: string }>();
	const uid = `oo-menu-${Math.random().toString(36).slice(2, 9)}`;

	let open = false;
	let active = -1;
	let anchor: HTMLElement;
	let list: HTMLElement;
	let entries: HTMLButtonElement[] = [];
	let stopFollowing: (() => void) | undefined;

	$: disabled = items.map((item) => Boolean(item.disabled));
	$: if (typeof label !== 'string' || label.trim() === '') {
		throw new Error('A menu needs an accessible name: give it a label');
	}

	function place() {
		if (!anchor || !list) return;
		computePosition(anchor, list, {
			placement,
			strategy: 'fixed',
			middleware: [offset(6), flip(), shift({ padding: 8 })]
		}).then(({ x, y }) => {
			if (!list) return;
			list.style.left = `${x}px`;
			list.style.top = `${y}px`;
		});
	}

	async function show(start: number) {
		open = true;
		active = start;
		await tick();
		if (anchor && list) stopFollowing = autoUpdate(anchor, list, place);
		if (active >= 0) entries[active]?.focus();
		else list?.focus();
	}

	function hide(returnFocus: boolean) {
		open = false;
		active = -1;
		stopFollowing?.();
		stopFollowing = undefined;
		if (returnFocus) anchor?.querySelector<HTMLElement>('button')?.focus();
	}

	function onTriggerClick() {
		if (open) hide(false);
		else show(disabled.indexOf(false));
	}

	function onTriggerKey(event: KeyboardEvent) {
		if (open) {
			onListKey(event);
			return;
		}
		const start = triggerKey(event.key, items.length, disabled);
		if (start !== null) {
			event.preventDefault();
			show(start);
		}
	}

	function onListKey(event: KeyboardEvent) {
		const moved = menuKey(event.key, active, items.length, disabled);
		if (moved.handled) event.preventDefault();
		if (moved.close) {
			hide(moved.handled);
		} else if (moved.handled) {
			active = moved.active;
			entries[active]?.focus();
		}
	}

	function hover(index: number) {
		if (disabled[index]) return;
		active = index;
		entries[index]?.focus();
	}

	function choose(item: MenuItem) {
		if (item.disabled) return;
		hide(true);
		dispatch('select', item.id);
	}

	function onWindowPointer(event: PointerEvent) {
		if (!open) return;
		const target = event.target as Node | null;
		if (target && (anchor?.contains(target) || list?.contains(target))) return;
		hide(false);
	}

	onDestroy(() => stopFollowing?.());
</script>

<svelte:window on:pointerdown={onWindowPointer} />

<span class="oo-menu-anchor" bind:this={anchor}>
	{#if icon}
		<IconButton
			{icon}
			{label}
			haspopup="menu"
			expanded={open}
			controls={open ? uid : undefined}
			on:click={onTriggerClick}
			on:keydown={onTriggerKey}
		/>
	{:else}
		<Button
			{variant}
			iconRight="chevron-down"
			haspopup="menu"
			expanded={open}
			controls={open ? uid : undefined}
			on:click={onTriggerClick}
			on:keydown={onTriggerKey}>{label}</Button
		>
	{/if}
</span>
{#if open}
	<div
		id={uid}
		role="menu"
		tabindex="-1"
		aria-label={label}
		class="oo-menu"
		bind:this={list}
		on:keydown={onListKey}
	>
		{#each items as item, index (item.id)}
			<button
				type="button"
				role="menuitem"
				tabindex="-1"
				class="oo-menu-item"
				data-danger={item.danger ? 'true' : undefined}
				disabled={item.disabled}
				bind:this={entries[index]}
				on:click={() => choose(item)}
				on:pointerenter={() => hover(index)}
			>
				{#if item.icon}<Icon name={item.icon} size="sm" />{/if}
				<span class="oo-menu-label">{item.label}</span>
			</button>
		{/each}
	</div>
{/if}

<style>
	.oo-menu-anchor {
		display: inline-flex;
	}
	.oo-menu {
		position: fixed;
		top: 0;
		left: 0;
		z-index: var(--oo-z-overlay);
		display: flex;
		flex-direction: column;
		gap: 2px;
		min-width: 12rem;
		max-width: 20rem;
		margin: 0;
		padding: var(--oo-space-1);
		background-color: var(--oo-bg-overlay);
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-md);
		box-shadow: var(--oo-shadow-md);
	}
	.oo-menu-item {
		display: flex;
		align-items: center;
		gap: var(--oo-space-3);
		width: 100%;
		min-height: 36px;
		padding: var(--oo-space-2) var(--oo-space-4);
		border: 0;
		border-radius: var(--oo-radius-sm);
		background-color: transparent;
		color: var(--oo-fg-primary);
		font: inherit;
		font-family: var(--oo-font-sans);
		font-size: var(--oo-text-sm);
		text-align: start;
		cursor: pointer;
	}
	.oo-menu-item:hover:not(:disabled),
	.oo-menu-item:focus-visible {
		background-color: var(--oo-bg-hover);
	}
	.oo-menu-item[data-danger='true'] {
		color: var(--oo-error);
	}
	.oo-menu-item:disabled {
		opacity: 0.55;
		cursor: not-allowed;
	}
	.oo-menu-label {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
</style>
