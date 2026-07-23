<!--
  ShortcutSettings.svelte
  Settings panel for keyboard shortcut customization.
  Displays current bindings in a table, supports click-to-rebind,
  individual and global reset to defaults.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import type { ShortcutBinding, BrowserConflict } from '$lib/api/shortcuts';

	let shortcuts: Record<string, ShortcutBinding> = {};
	let customOverrides: Record<string, Partial<ShortcutBinding>> = {};
	let browserConflicts: BrowserConflict[] = [];
	let loading = true;
	let saving = false;
	let error = '';
	let successMsg = '';

	// Rebinding state
	let rebindingAction: string | null = null;
	let capturedCombo = '';

	// Sorted list of shortcut entries for display
	$: sortedShortcuts = Object.values(shortcuts).sort((a, b) => {
		const catOrder = ['navigation', 'chat', 'ui', 'help', 'general'];
		const ai = catOrder.indexOf(a.category);
		const bi = catOrder.indexOf(b.category);
		if (ai !== bi) return ai - bi;
		return a.description.localeCompare(b.description);
	});

	$: conflictMap = new Map(browserConflicts.map((c) => [c.action, c.browser_function]));

	async function loadShortcuts() {
		loading = true;
		error = '';
		try {
			const { getKeyboardShortcuts } = await import('$lib/api/shortcuts');
			const response = await getKeyboardShortcuts();
			shortcuts = response.shortcuts;
			customOverrides = response.custom_overrides || {};
			browserConflicts = response.browser_conflicts || [];
		} catch (e) {
			error = 'Failed to load keyboard shortcuts';
		} finally {
			loading = false;
		}
	}

	async function saveBindings() {
		saving = true;
		error = '';
		successMsg = '';
		try {
			const { updateKeyboardShortcuts } = await import('$lib/api/shortcuts');
			const response = await updateKeyboardShortcuts(customOverrides);
			if (response.success) {
				shortcuts = response.shortcuts;
				customOverrides = response.custom_overrides || {};
				browserConflicts = response.browser_conflicts || [];
				successMsg = 'Shortcuts saved';
				// Notify KeyboardShortcuts.svelte
				window.dispatchEvent(
					new CustomEvent('opti-shortcuts-updated', {
						detail: { custom_overrides: customOverrides }
					})
				);
				setTimeout(() => (successMsg = ''), 2000);
			}
		} catch (e) {
			error = 'Failed to save shortcuts';
		} finally {
			saving = false;
		}
	}

	async function resetAll() {
		saving = true;
		error = '';
		successMsg = '';
		try {
			const { resetAllShortcuts } = await import('$lib/api/shortcuts');
			const response = await resetAllShortcuts();
			if (response.success) {
				shortcuts = response.shortcuts;
				customOverrides = {};
				browserConflicts = response.browser_conflicts || [];
				successMsg = 'Reset to defaults';
				window.dispatchEvent(
					new CustomEvent('opti-shortcuts-updated', {
						detail: { custom_overrides: {} }
					})
				);
				setTimeout(() => (successMsg = ''), 2000);
			}
		} catch (e) {
			error = 'Failed to reset shortcuts';
		} finally {
			saving = false;
		}
	}

	function resetSingle(action: string) {
		// Remove from custom overrides
		const updated = { ...customOverrides };
		delete updated[action];
		customOverrides = updated;
		// Save immediately
		saveBindings();
	}

	function startRebind(action: string) {
		rebindingAction = action;
		capturedCombo = '';
	}

	function cancelRebind() {
		rebindingAction = null;
		capturedCombo = '';
	}

	function handleRebindKeydown(e: KeyboardEvent) {
		if (!rebindingAction) return;

		e.preventDefault();
		e.stopPropagation();

		// Escape cancels rebinding
		if (e.key === 'Escape') {
			cancelRebind();
			return;
		}

		// Ignore lone modifier keys
		if (['Control', 'Shift', 'Alt', 'Meta'].includes(e.key)) return;

		const key = e.key.length === 1 ? e.key.toLowerCase() : e.key.toLowerCase();
		const ctrl = e.ctrlKey || e.metaKey;
		const shift = e.shiftKey;
		const alt = e.altKey;

		// Build display combo
		const parts: string[] = [];
		if (ctrl) parts.push('Ctrl');
		if (shift) parts.push('Shift');
		if (alt) parts.push('Alt');
		parts.push(e.key.length === 1 ? e.key.toUpperCase() : e.key);
		capturedCombo = parts.join(' + ');

		// Apply override
		customOverrides = {
			...customOverrides,
			[rebindingAction]: { key, ctrl, shift, alt }
		};

		// Auto-save after small delay
		rebindingAction = null;
		capturedCombo = '';
		saveBindings();
	}

	function formatBinding(s: ShortcutBinding): string {
		const parts: string[] = [];
		if (s.ctrl) parts.push('Ctrl');
		if (s.shift) parts.push('Shift');
		if (s.alt) parts.push('Alt');
		if (s.meta) parts.push('Meta');
		const keyNames: Record<string, string> = {
			',': ',',
			escape: 'Esc',
			'?': '?',
			enter: 'Enter'
		};
		const keyDisplay = keyNames[s.key] || (s.key.length === 1 ? s.key.toUpperCase() : s.key);
		parts.push(keyDisplay);
		return parts.join(' + ');
	}

	function isCustomized(action: string): boolean {
		return action in customOverrides;
	}

	onMount(() => {
		loadShortcuts();
		window.addEventListener('keydown', handleRebindKeydown, true);
		return () => {
			window.removeEventListener('keydown', handleRebindKeydown, true);
		};
	});
</script>

<div class="shortcut-settings">
	<div class="settings-header">
		<h3 class="settings-title">Keyboard Shortcuts</h3>
		<div class="header-actions">
			{#if Object.keys(customOverrides).length > 0}
				<button
					class="reset-all-btn"
					on:click={resetAll}
					disabled={saving}
					aria-label="Reset all shortcuts to defaults"
				>
					Reset all
				</button>
			{/if}
		</div>
	</div>

	{#if error}
		<div class="msg msg-error" role="alert">{error}</div>
	{/if}
	{#if successMsg}
		<div class="msg msg-success" role="status">{successMsg}</div>
	{/if}

	{#if loading}
		<div class="loading-msg" role="status" aria-label="Loading shortcuts">Loading shortcuts...</div>
	{:else}
		<div class="hint-text">
			Click a binding to rebind it. Press Escape to cancel.
		</div>

		<table class="shortcuts-table" role="grid" aria-label="Keyboard shortcuts bindings">
			<thead>
				<tr>
					<th>Action</th>
					<th>Shortcut</th>
					<th class="col-actions">Actions</th>
				</tr>
			</thead>
			<tbody>
				{#each sortedShortcuts as shortcut (shortcut.action)}
					<tr class:customized={isCustomized(shortcut.action)}>
						<td class="action-cell">
							<span class="action-desc">{shortcut.description}</span>
							<span class="action-category">{shortcut.category}</span>
						</td>
						<td class="binding-cell">
							{#if rebindingAction === shortcut.action}
								<span class="rebinding-indicator" role="status">
									{capturedCombo || 'Press a key combo...'}
								</span>
							{:else}
								<button
									class="binding-btn"
									class:has-conflict={conflictMap.has(shortcut.action)}
									on:click={() => startRebind(shortcut.action)}
									title="Click to rebind"
									aria-label="Rebind {shortcut.description}, currently {formatBinding(shortcut)}"
								>
									<kbd class="kbd-display">{formatBinding(shortcut)}</kbd>
								</button>
							{/if}
							{#if conflictMap.has(shortcut.action)}
								<span class="conflict-warning" title="Conflicts with {conflictMap.get(shortcut.action)}">
									conflicts with browser
								</span>
							{/if}
						</td>
						<td class="col-actions">
							{#if isCustomized(shortcut.action)}
								<button
									class="reset-btn"
									on:click={() => resetSingle(shortcut.action)}
									disabled={saving}
									title="Reset to default"
									aria-label="Reset {shortcut.description} to default"
								>
									Reset
								</button>
							{/if}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	{/if}
</div>

<style>
	.shortcut-settings {
		padding: 1rem;
	}

	.settings-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 0.75rem;
	}

	.settings-title {
		font-size: 1rem;
		font-weight: 600;
		color: var(--oo-fg-primary);
		margin: 0;
	}

	.header-actions {
		display: flex;
		gap: 0.5rem;
	}

	.reset-all-btn {
		padding: 0.25rem 0.75rem;
		border-radius: 6px;
		border: 1px solid var(--oo-bd-subtle);
		background: transparent;
		color: var(--oo-fg-secondary);
		font-size: 0.75rem;
		cursor: pointer;
		transition: background-color 0.15s ease, color 0.15s ease;
	}

	.reset-all-btn:hover {
		background-color: var(--oo-bg-overlay);
		color: var(--oo-fg-primary);
	}

	.reset-all-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.msg {
		padding: 0.5rem 0.75rem;
		border-radius: 6px;
		font-size: 0.813rem;
		margin-bottom: 0.75rem;
	}

	.msg-error {
		background-color: var(--oo-bg-overlay);
		color: var(--oo-danger);
		border: 1px solid var(--oo-danger);
	}

	.msg-success {
		background-color: var(--oo-bg-overlay);
		color: var(--oo-success);
		border: 1px solid var(--oo-success);
	}

	.loading-msg {
		padding: 2rem;
		text-align: center;
		color: var(--oo-fg-muted);
		font-size: 0.875rem;
	}

	.hint-text {
		font-size: 0.75rem;
		color: var(--oo-fg-faint);
		margin-bottom: 0.75rem;
	}

	.shortcuts-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.813rem;
	}

	.shortcuts-table th {
		text-align: left;
		padding: 0.5rem 0.75rem;
		color: var(--oo-fg-muted);
		font-weight: 500;
		font-size: 0.75rem;
		border-bottom: 1px solid var(--oo-bd-subtle);
	}

	.shortcuts-table td {
		padding: 0.5rem 0.75rem;
		border-bottom: 1px solid var(--oo-bd-subtle);
		vertical-align: middle;
	}

	.shortcuts-table tr.customized td {
		background-color: var(--oo-bg-overlay);
	}

	.col-actions {
		width: 5rem;
		text-align: center;
	}

	.action-cell {
		display: flex;
		flex-direction: column;
		gap: 0.125rem;
	}

	.action-desc {
		color: var(--oo-fg-primary);
	}

	.action-category {
		font-size: 0.688rem;
		color: var(--oo-fg-faint);
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	.binding-cell {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		align-items: flex-start;
	}

	.binding-btn {
		padding: 0.125rem 0.25rem;
		border: 1px solid transparent;
		border-radius: 4px;
		background: transparent;
		cursor: pointer;
		transition: border-color 0.15s ease;
	}

	.binding-btn:hover {
		border-color: var(--oo-bd-subtle);
	}

	.binding-btn:focus-visible {
		outline: 2px solid var(--oo-acc-500);
		outline-offset: 2px;
	}

	.binding-btn.has-conflict .kbd-display {
		color: var(--oo-warning);
	}

	.kbd-display {
		display: inline-flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.125rem 0.5rem;
		border-radius: 4px;
		background-color: var(--oo-bg-overlay);
		border: 1px solid var(--oo-bd-subtle);
		font-size: 0.75rem;
		color: var(--oo-fg-muted);
		font-family: var(--oo-font-mono);
	}

	.rebinding-indicator {
		padding: 0.25rem 0.5rem;
		border-radius: 4px;
		background-color: var(--oo-acc-500);
		color: var(--oo-fg-on-accent);
		font-size: 0.75rem;
		font-family: var(--oo-font-mono);
		animation: pulse-border 1s ease-in-out infinite;
	}

	@keyframes pulse-border {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.7; }
	}

	.conflict-warning {
		font-size: 0.688rem;
		color: var(--oo-warning);
	}

	.reset-btn {
		padding: 0.125rem 0.5rem;
		border-radius: 4px;
		border: 1px solid var(--oo-bd-subtle);
		background: transparent;
		color: var(--oo-fg-muted);
		font-size: 0.688rem;
		cursor: pointer;
		transition: background-color 0.15s ease;
	}

	.reset-btn:hover {
		background-color: var(--oo-bg-overlay);
		color: var(--oo-fg-primary);
	}

	.reset-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
</style>
