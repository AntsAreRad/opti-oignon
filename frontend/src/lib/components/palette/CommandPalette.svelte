<!--
  CommandPalette.svelte
  The command palette: one dialog, the ds Modal, mounted once by the layout
  of both spaces and open while its store says so (lib/stores/palette.ts).
  Ctrl+K opens it through the registry's command, the sidebar's Search entry
  through the store; nothing on the way reads the page's markup. Unmounted
  (the reader left both spaces), it shuts its store, so it never opens by
  itself on the next page that holds it.

  It reaches every place and every action of the interface from the
  keyboard. Its options come from where each kind lives, through the
  palette's sources (lib/palette/sources.ts): the visible destinations of
  the table, every command of the registry with the keys that run it for
  this reader (lib/stores/shortcutKeys.ts), the Stop all entry, every
  settings group that has a page (the settings search's index, both
  spaces), and the chats: the recent ones while nothing is typed, then the
  server's answer to the words (lib/palette/conversationSource.ts), which
  the palette asks as the words change and closes when it closes, and a
  last entry to the chats index when the server found more than the group
  shows. It ranks them with lib/palette/rank.ts and nothing of its own.

  A command that cannot run where the reader is stays listed, disabled, with
  its reason in words beside it. The palette judges its commands from
  inside (lib/palette/run.ts, contextAt, asked by the palette), runs the one
  chosen in that same context once it has closed, so what it shows disabled
  is exactly what the runner refuses. Opening a disabled one says its reason
  in the status line. The Stop all entry runs nothing: it opens the stop
  control's confirmation in the dialog's head, and the reader's own press
  stops.

  The field is a combobox that controls a listbox: focus stays in the
  field, the active option is named by aria-activedescendant and drawn with
  a ring in the focus ink; the options sit in groups, each labelled by a
  visible heading. Which option is active, the id each carries and the
  status line come from lib/palette/listing.ts: the option the reader moved
  to stays active while it is listed, however the list is ranked again;
  otherwise the first option that can run is. The arrows move, Home and End
  go to the first and the last option, Enter opens, Esc closes and focus
  returns to the opener. The status line says how many results are listed
  once the list settles.

  Stop all sits in the dialog's head in every state and at every width, so
  the stop stays one tap or click away while the palette covers the page,
  above the field, where a phone's keyboard never covers it. On a phone
  every option is a 44 px target, the field's type is large enough that a
  phone does not zoom into it, and a disabled command's reason wraps under
  its name instead of being cut.
-->
<script lang="ts">
	import { onDestroy, tick } from 'svelte';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import { Modal, Input, Icon } from '$lib/ds';
	import StopAllButton from '$lib/components/layout/StopAllButton.svelte';
	import { palette, closePalette } from '$lib/stores/palette';
	import { shortcutsHelp } from '$lib/stores/shortcutsHelp';
	import { shortcutKeys } from '$lib/stores/shortcutKeys';
	import { conversations } from '$lib/stores/conversations';
	import { isStreaming } from '$lib/stores/chat';
	import { isPhone } from '$lib/stores/ui';
	import { listConversations } from '$lib/api/conversations';
	import { DESTINATIONS, visibleDestinations } from '$lib/nav/destinations';
	import { SETTINGS_SECTIONS, INLINE_GROUPS, PREFERENCES_SECTIONS } from '$lib/settings/catalog';
	import { settingsIndex } from '$lib/settings/search';
	import { GROUP_LIMIT, rankPalette, type PaletteOption, type RankedGroup } from '$lib/palette/rank';
	import { COMMANDS, commandOptions } from '$lib/palette/commands';
	import {
		destinationOptions,
		settingOptions,
		chatOptions,
		stopOption,
		moreChatsOption
	} from '$lib/palette/sources';
	import {
		createConversationSource,
		followPalette,
		type ConversationState
	} from '$lib/palette/conversationSource';
	import {
		activeFor,
		stepActive,
		optionKey,
		optionDomId,
		createStatusLine,
		type Move
	} from '$lib/palette/listing';
	import { contextAt, runCommand } from '$lib/palette/run';
	import type { ConversationSummary } from '$lib/types';

	// The componion's own switch arrives with its page; until then its entry
	// is not ready, and the visible list never names it.
	const visibility = { componion: true };

	const uid = `oo-palette-${Math.random().toString(36).slice(2, 9)}`;
	const listId = `${uid}-list`;

	const DESTINATION_OPTIONS = destinationOptions(visibleDestinations(DESTINATIONS, visibility));
	const GROUPS = [...SETTINGS_SECTIONS.flatMap((s) => s.groups), ...INLINE_GROUPS];
	const SETTING_OPTIONS = settingOptions(
		settingsIndex(GROUPS, DESTINATIONS, PREFERENCES_SECTIONS, SETTINGS_SECTIONS),
		GROUPS
	);
	const chatsHref = DESTINATIONS.find((d) => d.id === 'chats')?.href ?? '';

	/** The server's answer to the words, and what it is still asking. */
	let found: ConversationState<ConversationSummary> = { query: '', hits: [], asking: null, error: null };
	const chatSource = createConversationSource(
		(query, signal) => listConversations({ ...query, signal }),
		(state) => (found = state)
	);
	// Open, each change of the words is asked once; closed, the search in
	// flight is aborted and its answer dropped.
	const follow = followPalette(
		(words) => chatSource.search(words),
		() => chatSource.close()
	);

	let words = '';
	let wasOpen = false;
	/** The option the reader moved to since the words changed, by its identity. */
	let pick: string | null = null;
	/** Why the option the reader tried to open cannot run. */
	let notice: string | null = null;
	let ranked = '';
	/** The Stop all control in the dialog's head, and why it cannot be asked for now. */
	let stopControl: StopAllButton;
	let stopRefusal: string | null = null;

	// Opened, the field takes the words the store opens it with.
	$: if ($palette.open !== wasOpen) {
		wasOpen = $palette.open;
		words = wasOpen ? $palette.words : '';
		pick = null;
		notice = null;
	}

	$: query = words.trim();
	// New words: the pick is dropped, and the best match is active again.
	$: if (query !== ranked) {
		ranked = query;
		pick = null;
		notice = null;
	}
	$: if (browser) follow($palette.open, words);

	// Judged from inside the palette: a command chosen here runs once the
	// palette has closed, in this same context.
	$: context = contextAt({
		pathname: $page.url?.pathname ?? '/',
		streaming: $isStreaming,
		palette: $palette.open,
		help: $shortcutsHelp,
		from: 'palette'
	});
	$: commands = commandOptions(COMMANDS, context, $shortcutKeys);
	$: stop = stopOption(stopRefusal);
	// No words: the recent chats. Words: the server's chats, kept whatever
	// their titles once it has answered these words; until then the last
	// answer stays only where its titles meet them.
	$: answered = found.query === query;
	$: chats =
		query === ''
			? chatOptions($conversations, chatsHref)
			: chatOptions(found.hits, chatsHref, answered);
	$: more = answered ? moreChatsOption(query, chatsHref, found.hits.length, GROUP_LIMIT) : null;
	$: listing = rankPalette(
		[...DESTINATION_OPTIONS, stop, ...commands, ...SETTING_OPTIONS, ...chats, ...(more ? [more] : [])],
		query
	);
	$: rows = numbered(listing);
	$: flat = rows.flatMap((group) => group.items.map((item) => item.option));
	$: activeKey = activeFor(flat, pick);
	$: activeOption = flat.find((option) => optionKey(option) === activeKey);
	$: activeId = activeOption ? optionDomId(uid, activeOption) : undefined;

	// The status line, read out once the list has stood still a moment; on
	// the server, at once.
	let spoken = '';
	const statusLine = createStatusLine((line) => (spoken = line), { immediate: !browser });
	$: statusLine.update({
		query,
		count: flat.length,
		asking: found.asking !== null,
		error: found.error,
		notice
	});

	onDestroy(() => {
		statusLine.close();
		closePalette();
	});

	/** The groups as listed, each option with its id in the page. */
	function numbered(groups: RankedGroup<PaletteOption>[]) {
		return groups.map((group) => ({
			id: group.id,
			label: group.label,
			items: group.items.map((option) => ({ option, key: optionKey(option), domId: optionDomId(uid, option) }))
		}));
	}

	/**
	 * Opens a place, runs a command, or opens the stop's confirmation; a
	 * disabled option says why in the status line and does nothing.
	 */
	async function choose(option: PaletteOption | undefined) {
		if (!option) return;
		if (option.disabled) {
			notice = option.reason ? `${option.label}: ${option.reason}` : `${option.label} cannot run here`;
			return;
		}
		if (option.stop) {
			await stopControl?.askToConfirm();
			return;
		}
		const where = context;
		closePalette();
		await tick();
		if (option.command) runCommand(option.command, where);
		else if (option.href) await goto(option.href);
	}

	function move(to: Move) {
		pick = stepActive(flat, activeKey, to);
		notice = null;
	}

	function onKey(event: KeyboardEvent) {
		// The dialog closes itself on Esc; nothing behind it hears the key.
		if (event.key === 'Escape') {
			event.stopPropagation();
			return;
		}
		if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
		if (flat.length === 0) return;
		if (event.key === 'ArrowDown') move('next');
		else if (event.key === 'ArrowUp') move('previous');
		else if (event.key === 'Home') move('first');
		else if (event.key === 'End') move('last');
		else if (event.key === 'Enter') void choose(activeOption);
		else return;
		event.preventDefault();
	}

	/** Keeps the active option in view as the arrows move it. */
	function reveal(node: HTMLElement, shown: boolean) {
		const follow = (on: boolean) => {
			if (on) node.scrollIntoView({ block: 'nearest' });
		};
		follow(shown);
		return { update: follow };
	}
</script>

<Modal open={$palette.open} variant="center" size="lg" title="Search" onClose={closePalette}>
	<svelte:fragment slot="actions">
		<StopAllButton bind:this={stopControl} bind:refusal={stopRefusal} placement="dialog-head" large={$isPhone} />
	</svelte:fragment>

	<div class="oo-palette" data-phone={$isPhone ? 'true' : 'false'}>
		<Input
			label="Search pages, commands, settings and chats"
			hideLabel
			placeholder="Go to a page, run a command, find a chat"
			iconLeft="search"
			size="lg"
			autofocus
			combobox={{ controls: listId, expanded: flat.length > 0, active: activeId }}
			bind:value={words}
			on:keydown={onKey}
		/>

		<div id={listId} class="oo-palette-list" role="listbox" aria-label="Results" hidden={rows.length === 0}>
			{#each rows as group (group.id)}
				<div class="oo-palette-group" role="group" aria-labelledby={`${uid}-group-${group.id}`}>
					<div id={`${uid}-group-${group.id}`} class="oo-palette-heading">{group.label}</div>
					{#each group.items as { option, key, domId } (key)}
						<div
							id={domId}
							class="oo-palette-option"
							class:active={key === activeKey}
							role="option"
							tabindex="-1"
							aria-selected={key === activeKey}
							aria-disabled={option.disabled ? 'true' : undefined}
							use:reveal={key === activeKey}
							on:mousedown|preventDefault
							on:mousemove={() => (pick = key)}
							on:click={() => choose(option)}
							on:keydown={onKey}
						>
							<span class="oo-palette-icon"><Icon name={option.icon ?? 'arrow-right'} size="sm" /></span>
							<span class="oo-palette-label">{option.label}</span>
							{#if option.disabled && option.reason}
								<span class="oo-palette-detail oo-palette-reason">{option.reason}</span>
							{:else if option.detail && option.group === 'commands' && option.command}
								<kbd class="oo-palette-keys">{option.detail}</kbd>
							{:else if option.detail}
								<span class="oo-palette-detail">{option.detail}</span>
							{/if}
						</div>
					{/each}
				</div>
			{/each}
		</div>

		<p class="oo-palette-note" role="status">{spoken}</p>

		{#if !$isPhone}
			<p class="oo-palette-hint">
				<kbd class="oo-palette-keys">&uarr;</kbd><kbd class="oo-palette-keys">&darr;</kbd> to move,
				<kbd class="oo-palette-keys">Enter</kbd> to open,
				<kbd class="oo-palette-keys">Esc</kbd> to close
			</p>
		{/if}
	</div>
</Modal>

<style>
	.oo-palette {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
	}

	/* The list scrolls on its own below the field, which stays in view. */
	.oo-palette-list {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-5);
		max-height: min(56vh, 28rem);
		overflow-y: auto;
		overscroll-behavior: contain;
	}
	.oo-palette-list[hidden] {
		display: none;
	}

	.oo-palette-group {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.oo-palette-heading {
		padding: 0 var(--oo-space-3) var(--oo-space-1);
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-xs);
		font-weight: 500;
	}

	/* An option: a soft rounded row. The active one takes the hover wash and
	   a ring in the focus ink, since focus stays in the field. */
	.oo-palette-option {
		display: flex;
		align-items: center;
		gap: var(--oo-space-3);
		min-height: 40px;
		padding: var(--oo-space-2) var(--oo-space-3);
		border-radius: var(--oo-radius-lg);
		color: var(--oo-fg-primary);
		font-size: var(--oo-text-md);
		cursor: pointer;
	}
	.oo-palette[data-phone='true'] .oo-palette-option {
		flex-wrap: wrap;
		min-height: 44px;
	}
	/* On a phone the field's type is 16 px at least, so the phone does not
	   zoom into the page when the field takes focus. */
	.oo-palette[data-phone='true'] :global(.oo-field-control) {
		font-size: max(16px, var(--oo-text-md));
	}
	.oo-palette-option.active {
		background-color: var(--oo-bg-hover);
		outline: 2px solid var(--oo-focus-ink);
		outline-offset: -2px;
	}
	.oo-palette-option[aria-disabled='true'] {
		color: var(--oo-fg-muted);
		cursor: default;
	}

	.oo-palette-icon {
		display: inline-flex;
		flex-shrink: 0;
		color: var(--oo-fg-muted);
	}
	.oo-palette-label {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.oo-palette-detail {
		flex-shrink: 0;
		max-width: 50%;
		overflow: hidden;
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	/* On a phone a disabled command's reason is read whole: it wraps under
	   the command's name, the width of the row, instead of being cut. */
	.oo-palette[data-phone='true'] .oo-palette-reason {
		flex-basis: 100%;
		max-width: none;
		padding-left: calc(var(--oo-space-3) + 16px);
		white-space: normal;
	}

	/* A key: a small rounded cap on the sunken ground. */
	.oo-palette-keys {
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

	.oo-palette-note {
		margin: 0;
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
	}

	.oo-palette-hint {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--oo-space-1);
		margin: 0;
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-xs);
	}
</style>
