<!--
  KeyboardShortcuts.svelte
  The keyboard's shortcuts, and the list of them. Mounted once, by the root
  layout, so the keys work on every page.

  Every shortcut is a command of the registry (lib/palette/commands.ts):
  the handler starts from the registry's bound commands, applies over them
  the keys the reader chose (kept by the server, and sent again by the
  shortcut settings when they change), and runs every command through the
  runner (lib/palette/run.ts), which refuses a command where it cannot run
  before any handler does: behind an open palette or list of shortcuts, a
  key runs only what closes them. It holds no handler of its own, sends no
  event of its own, and reads nothing of the page's markup. The keys it
  runs are written to their store (lib/stores/shortcutKeys.ts), where the
  command palette and the sidebar read the keys they show.

  A key the application binds with a modifier is the application's, run or
  refused. A plain key (? and Escape) is left to the field being typed in,
  and, refused, keeps its default action: an Escape that closes no dialog
  of ours still closes the native dialog it was pressed in.

  The list of shortcuts is a ds Modal, open while its store says so
  (lib/stores/shortcutsHelp.ts). Open, it holds Stop all in its head, like
  every modal dialog of the shell: the page under it is inert.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { Modal } from '$lib/ds';
	import StopAllButton from '$lib/components/layout/StopAllButton.svelte';
	import {
		COMMANDS,
		bindingLabel,
		chosenBindings,
		defaultShortcuts,
		type ChosenKeys,
		type DefaultShortcut
	} from '$lib/palette/commands';
	import { runCommand } from '$lib/palette/run';
	import { shortcutsHelp, closeShortcutsHelp } from '$lib/stores/shortcutsHelp';
	import { shortcutKeys } from '$lib/stores/shortcutKeys';
	import { isPhone } from '$lib/stores/ui';

	/** The reader's own keys, by action, as the server keeps them. */
	let chosen: Record<string, ChosenKeys> = {};

	$: bindings = chosenBindings(COMMANDS, chosen);
	$: shortcutKeys.set(bindings);
	$: shortcuts = defaultShortcuts(COMMANDS, bindings);
	$: paletteKeys = shortcuts.find((s) => s.action === 'search_conversations');

	/** The reader's own keys, from the server; the registry's stand when it cannot answer. */
	async function loadChosenKeys() {
		try {
			const { getKeyboardShortcuts } = await import('$lib/api/shortcuts');
			const response = await getKeyboardShortcuts();
			if (response?.custom_overrides && Object.keys(response.custom_overrides).length > 0) {
				chosen = response.custom_overrides;
			}
		} catch {
			// The registry's keys stand.
		}
	}

	/** The shortcut settings send the keys again when the reader changes them. */
	function onKeysChanged(e: Event) {
		const detail = (e as CustomEvent).detail;
		if (detail?.custom_overrides) chosen = detail.custom_overrides;
	}

	function matchesShortcut(e: KeyboardEvent, s: DefaultShortcut): boolean {
		// A browser's autofill sends a keydown with no key.
		if (typeof e.key !== 'string') return false;
		// '?' is typed with Shift on most layouts: it matches whatever Shift says.
		if (s.key === '?') {
			return e.key === '?' && !e.ctrlKey && !e.metaKey && !e.altKey;
		}
		if (s.ctrl !== (e.ctrlKey || e.metaKey)) return false;
		if (s.shift !== e.shiftKey) return false;
		if (s.alt !== e.altKey) return false;
		// The server and the shortcut settings keep keys in lower case.
		return e.key.toLowerCase() === s.key.toLowerCase();
	}

	function onKeydown(e: KeyboardEvent) {
		const target = e.target as HTMLElement | null;
		const typing =
			!!target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);
		for (const s of shortcuts) {
			if (!matchesShortcut(e, s)) continue;
			if (!s.ctrl && !s.shift && !s.alt && typing) continue;
			const refused = runCommand(s.action);
			if (refused === null || s.ctrl || s.alt) e.preventDefault();
			return;
		}
	}

	onMount(() => {
		document.addEventListener('keydown', onKeydown);
		window.addEventListener('opti-shortcuts-updated', onKeysChanged);
		loadChosenKeys();
	});

	onDestroy(() => {
		if (typeof document === 'undefined') return;
		document.removeEventListener('keydown', onKeydown);
		window.removeEventListener('opti-shortcuts-updated', onKeysChanged);
	});
</script>

<Modal open={$shortcutsHelp} variant="center" size="sm" title="Keyboard shortcuts" onClose={closeShortcutsHelp}>
	<svelte:fragment slot="actions">
		{#if $shortcutsHelp}
			<StopAllButton placement="dialog-head" large={$isPhone} />
		{/if}
	</svelte:fragment>
	<ul class="oo-keys" role="list">
		{#each shortcuts as shortcut (shortcut.action)}
			<li class="oo-keys-row">
				<span class="oo-keys-what">{shortcut.description}</span>
				<kbd class="oo-kbd">{bindingLabel(shortcut)}</kbd>
			</li>
		{/each}
	</ul>
	<svelte:fragment slot="footer">
		<p class="oo-keys-hint">
			{#if paletteKeys}
				Every command is in the palette too: <kbd class="oo-kbd">{bindingLabel(paletteKeys)}</kbd>.
			{/if}
			Press <kbd class="oo-kbd">?</kbd> or <kbd class="oo-kbd">Esc</kbd> to close.
		</p>
	</svelte:fragment>
</Modal>

<style>
	.oo-keys {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.oo-keys-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-4);
		padding: var(--oo-space-2) 0;
	}

	.oo-keys-what {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
	}

	/* A key: a small rounded cap on the sunken ground. */
	.oo-kbd {
		display: inline-flex;
		align-items: center;
		flex-shrink: 0;
		padding: 1px var(--oo-space-2);
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-sm);
		background-color: var(--oo-bg-subtle);
		color: var(--oo-fg-secondary);
		font-family: var(--oo-font-mono);
		font-size: var(--oo-text-xs);
		white-space: nowrap;
	}

	.oo-keys-hint {
		margin: 0;
		font-size: var(--oo-text-xs);
		line-height: var(--oo-leading-snug);
		color: var(--oo-fg-muted);
	}
</style>
