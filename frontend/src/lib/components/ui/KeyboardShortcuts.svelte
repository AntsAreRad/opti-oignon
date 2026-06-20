<!--
  KeyboardShortcuts.svelte (S107, refactored S153, S166)
  Global keyboard shortcut handler + help overlay.
  Mounted in root layout to cover the entire application.
  Supports custom bindings loaded from backend.
  S166: the help overlay now uses the shared <Modal> primitive (native
  dialog focus trap, Escape and backdrop handling).
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import type { KeyboardShortcut } from '$lib/types';
	import { Modal } from '$lib/ds';

	export let onNewConversation: (() => void) | null = null;
	export let onExportConversation: (() => void) | null = null;
	export let onGoToSettings: (() => void) | null = null;
	export let onToggleSearch: (() => void) | null = null;
	export let onToggleTheme: (() => void) | null = null;
	export let onToggleSidebar: (() => void) | null = null;

	let showHelp = false;

	// Default shortcut definitions
	const defaultShortcuts: KeyboardShortcut[] = [
		{ key: 'n', ctrl: true, description: 'New conversation', action: 'new_chat' },
		{ key: 'Enter', ctrl: true, description: 'Send message', action: 'send_message' },
		{ key: 'b', ctrl: true, description: 'Toggle sidebar', action: 'toggle_sidebar' },
		{ key: 'k', ctrl: true, description: 'Focus context-list search', action: 'search_conversations' },
		{ key: ',', ctrl: true, description: 'Open settings', action: 'open_settings' },
		{ key: 't', ctrl: true, shift: true, description: 'Toggle theme', action: 'toggle_theme' },
		{ key: 'e', ctrl: true, shift: true, description: 'Export conversation', action: 'export_conversation' },
		{ key: '?', description: 'Show keyboard shortcuts', action: 'show_shortcuts' },
		{ key: 'Escape', description: 'Close dialog / panel', action: 'close_dialog' }
	];

	// Active shortcuts (can be overridden by custom bindings)
	let shortcuts: KeyboardShortcut[] = [...defaultShortcuts];

	// Custom overrides loaded from backend (action -> partial binding)
	let customOverrides: Record<string, Partial<KeyboardShortcut>> = {};

	/**
	 * Apply custom overrides to the default shortcuts.
	 */
	function applyOverrides() {
		shortcuts = defaultShortcuts.map((s) => {
			const override = customOverrides[s.action];
			if (!override) return { ...s };
			return {
				...s,
				key: override.key ?? s.key,
				ctrl: override.ctrl ?? s.ctrl,
				shift: override.shift ?? s.shift,
				alt: override.alt ?? s.alt
			};
		});
	}

	/**
	 * Load custom bindings from backend.
	 */
	async function loadCustomBindings() {
		try {
			const { getKeyboardShortcuts } = await import('$lib/api/shortcuts');
			const response = await getKeyboardShortcuts();
			if (response?.custom_overrides && Object.keys(response.custom_overrides).length > 0) {
				customOverrides = response.custom_overrides;
				applyOverrides();
			}
		} catch {
			// Silently fall back to defaults if backend unavailable
		}
	}

	/**
	 * Listen for custom binding updates from ShortcutSettings.
	 */
	function handleBindingsUpdated(e: Event) {
		const detail = (e as CustomEvent).detail;
		if (detail?.custom_overrides) {
			customOverrides = detail.custom_overrides;
			applyOverrides();
		}
	}

	// Action dispatcher map
	const actionHandlers: Record<string, (() => void) | null> = {};
	$: {
		actionHandlers['new_chat'] = onNewConversation;
		actionHandlers['export_conversation'] = onExportConversation;
		actionHandlers['open_settings'] = onGoToSettings;
		actionHandlers['search_conversations'] = onToggleSearch;
		actionHandlers['toggle_theme'] = onToggleTheme;
		actionHandlers['toggle_sidebar'] = onToggleSidebar;
	}

	function matchesShortcut(e: KeyboardEvent, s: KeyboardShortcut): boolean {
		// KS-02 (S197): '?' is produced WITH Shift on most layouts, so match
		// it shift-agnostically and before the modifier gate.
		if (s.key === '?') {
			return e.key === '?' && !e.ctrlKey && !e.metaKey && !e.altKey;
		}

		const ctrl = s.ctrl || false;
		const shift = s.shift || false;
		const alt = s.alt || false;

		if (ctrl !== (e.ctrlKey || e.metaKey)) return false;
		if (shift !== e.shiftKey) return false;
		if (alt !== e.altKey) return false;

		// KS-03 (S197): custom overrides arrive lowercased (ShortcutSettings
		// and the backend both canonicalize), so compare case-insensitively
		// for every key length; this also covers Enter/Escape.
		return e.key.toLowerCase() === s.key.toLowerCase();
	}

	function handleKeydown(e: KeyboardEvent) {
		const target = e.target as HTMLElement;
		const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable;

		// Find matching shortcut
		for (const s of shortcuts) {
			if (!matchesShortcut(e, s)) continue;

			// For non-modifier shortcuts (? and Escape), skip if in input field
			if (!s.ctrl && !s.shift && !s.alt && isInput) continue;

			// KS-01 (S197): preventDefault ONLY on paths that actually handle
			// the key. An unhandled Escape must keep its default action --
			// that default is what fires the native <dialog> cancel event the
			// ds Modal relies on to close.

			// Toggle the help overlay (Modal handles focus + restore)
			if (s.action === 'show_shortcuts') {
				e.preventDefault();
				showHelp = !showHelp;
				return;
			}

			// Escape: close the help overlay if open; otherwise leave the
			// event untouched for the native dialog / focused consumer.
			if (s.action === 'close_dialog') {
				if (showHelp) {
					e.preventDefault();
					closeHelp();
				}
				return;
			}

			// Send message: dispatch global event
			if (s.action === 'send_message') {
				e.preventDefault();
				window.dispatchEvent(new CustomEvent('opti-send-message'));
				return;
			}

			// Dispatch to handler
			const handler = actionHandlers[s.action];
			if (handler) {
				e.preventDefault();
				handler();
			}
			return;
		}
	}

	function closeHelp() {
		showHelp = false;
	}

	function formatShortcut(shortcut: KeyboardShortcut): string {
		const parts: string[] = [];
		if (shortcut.ctrl) parts.push('Ctrl');
		if (shortcut.shift) parts.push('Shift');
		if (shortcut.alt) parts.push('Alt');

		const keyNames: Record<string, string> = {
			',': ',',
			'Escape': 'Esc',
			'?': '?',
			'Enter': 'Enter'
		};
		parts.push(keyNames[shortcut.key] || shortcut.key.toUpperCase());
		return parts.join(' + ');
	}

	onMount(() => {
		document.addEventListener('keydown', handleKeydown);
		window.addEventListener('opti-shortcuts-updated', handleBindingsUpdated);
		loadCustomBindings();
	});

	onDestroy(() => {
		document.removeEventListener('keydown', handleKeydown);
		window.removeEventListener('opti-shortcuts-updated', handleBindingsUpdated);
	});
</script>

<Modal open={showHelp} variant="center" size="sm" title="Keyboard Shortcuts" onClose={closeHelp}>
	<div class="help-list">
		{#each shortcuts as shortcut}
			<div class="help-row">
				<span class="help-desc">{shortcut.description}</span>
				<kbd class="help-kbd">{formatShortcut(shortcut)}</kbd>
			</div>
		{/each}
	</div>
	<svelte:fragment slot="footer">
		<span class="help-hint">
			Press <kbd class="help-kbd-inline">?</kbd>
			or <kbd class="help-kbd-inline">Esc</kbd>
			to close
		</span>
	</svelte:fragment>
</Modal>

<style>
	.help-list {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}

	.help-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-4);
		padding: var(--oo-space-2) 0;
	}

	.help-desc {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
	}

	.help-kbd {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-1);
		padding: 2px var(--oo-space-2);
		border-radius: var(--oo-radius-sm);
		background-color: var(--oo-bg-overlay);
		border: 1px solid var(--oo-bd-subtle);
		font-size: var(--oo-text-2xs);
		color: var(--oo-fg-muted);
		font-family: var(--oo-font-mono);
	}

	.help-hint {
		font-size: var(--oo-text-2xs);
		color: var(--oo-fg-muted);
	}

	.help-kbd-inline {
		display: inline;
		padding: 2px var(--oo-space-1);
		border-radius: var(--oo-radius-sm);
		background-color: var(--oo-bg-overlay);
		border: 1px solid var(--oo-bd-subtle);
		font-size: var(--oo-text-2xs);
		color: var(--oo-fg-muted);
		font-family: var(--oo-font-mono);
	}
</style>
