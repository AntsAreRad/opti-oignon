<!--
  SettingsGroup.svelte
  One group of settings, on the page that holds it, titled once.

  The title, the description and the optional "Reset to default" are drawn
  by the design system's PanelHeader; the panel below is the group's own.
  The group sits on the ds Card, its tone set apart from the page's, and
  groups are separated by space, never by lines. Its anchor
  (`#oo-set-<id>`) is what a link to the group scrolls to.

  The settings hub hands the group its context (lib/settings/disclosure.ts):
  the level of its title (2 on a Workshop page, 3 in Preferences, 3 with no
  hub around it) and whether it folds. A group that folds is a quiet row:
  its title is a disclosure, one group of the page is open at a time, and
  only the open group renders its panel, so a closed group's panel never
  loads nor reads anything. A group whose panel was used on this visit (a
  field typed in, a choice clicked, a key pressed on a control, a file
  dropped: lib/settings/disclosure.ts says which) keeps its panel mounted
  and hidden once closed, so its draft is there when it reopens; the group
  listens in the capture phase, so a control that stops the event or
  answers with its own still counts. Leaving the page drops what was not
  saved. The region the title controls is always there, hidden while the
  group is closed.

  The frame is a section with no name of its own: the title's heading gives
  the page its outline, and a page of groups is not a page of landmarks.
-->
<script lang="ts">
	import { getContext } from 'svelte';
	import { readable } from 'svelte/store';
	import Card from '$lib/ds/Card.svelte';
	import Button from '$lib/ds/Button.svelte';
	import PanelHeader from '$lib/ds/PanelHeader.svelte';
	import {
		DRAFT_EVENTS,
		GROUPS_CONTEXT,
		mounted,
		startsDraft,
		type GroupsContext
	} from '$lib/settings/disclosure';

	export let id: string;
	export let title: string;
	export let description: string | undefined = undefined;
	/** Optional reset-to-default handler; when set, a reset button is shown. */
	export let onReset: (() => void) | undefined = undefined;
	export let resetLabel = 'Reset to default';

	const context = getContext<GroupsContext | undefined>(GROUPS_CONTEXT);
	const level = context?.level ?? 3;
	const collapsible = context?.collapsible ?? false;
	const open = context?.open ?? readable<string | null>(null);
	const edited = context?.edited ?? readable<ReadonlySet<string>>(new Set());

	const anchor = `oo-set-${id}`;
	const headingId = `${anchor}-title`;
	const bodyId = `${anchor}-body`;

	$: expanded = $open === id;
	$: shown = !collapsible || mounted(id, $open, $edited);

	/** The panel was used: whatever it holds must outlive a close. */
	function onEdit() {
		if (collapsible && !$edited.has(id)) context?.markEdited(id);
	}

	/** Watches the panel, in the capture phase, for what may start a draft. */
	function watchDrafts(node: HTMLElement) {
		const seen = (event: Event) => {
			if (startsDraft(event.type, event instanceof KeyboardEvent ? event.key : undefined)) onEdit();
		};
		for (const type of DRAFT_EVENTS) node.addEventListener(type, seen, true);
		return {
			destroy() {
				for (const type of DRAFT_EVENTS) node.removeEventListener(type, seen, true);
			}
		};
	}
</script>

<section id={anchor} class="oo-set-group" data-collapsible={collapsible}>
	<Card padding={collapsible ? 'none' : 'lg'}>
		{#if onReset}
			<PanelHeader
				{title}
				{level}
				{description}
				{headingId}
				expanded={collapsible ? expanded : undefined}
				controls={collapsible ? bodyId : undefined}
				on:toggle={() => context?.toggle(id)}
			>
				<svelte:fragment slot="actions">
					<Button variant="ghost" size="sm" iconLeft="retry" on:click={() => onReset && onReset()}>
						{resetLabel}
					</Button>
				</svelte:fragment>
			</PanelHeader>
		{:else}
			<PanelHeader
				{title}
				{level}
				{description}
				{headingId}
				expanded={collapsible ? expanded : undefined}
				controls={collapsible ? bodyId : undefined}
				on:toggle={() => context?.toggle(id)}
			/>
		{/if}

		<div id={bodyId} class="oo-set-body" hidden={collapsible && !expanded} use:watchDrafts>
			{#if shown}
				<slot />
			{/if}
		</div>
	</Card>
</section>

<style>
	.oo-set-group {
		--oo-panel-header-radius: var(--oo-radius-lg);
		scroll-margin-top: var(--oo-space-6);
	}

	/* A plain group's title sits on the card's own padding. */
	.oo-set-group[data-collapsible='false'] :global(.oo-panel-header) {
		min-height: 0;
		padding: 0 0 var(--oo-space-4);
	}

	.oo-set-body {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}
	.oo-set-group[data-collapsible='true'] .oo-set-body {
		padding: 0 var(--oo-space-5) var(--oo-space-5);
	}
	.oo-set-body[hidden] {
		display: none;
	}
</style>
