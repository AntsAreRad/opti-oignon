<!--
  PanelHeader.svelte (lib/ds) -- the title of a panel or a group, drawn once.

  The title is a heading of the level given (2, 3 or 4; 3 by default). It is
  required: a blank title refuses to render, as an icon button without a
  name does, because a group with no title is a group no one can find.

  Given `expanded`, the header is a disclosure: the title becomes a button
  inside the heading, which says whether the region it controls
  (`controls`) is shown and dispatches `toggle` when pressed. Its name is
  the title alone: the description stands outside the heading and the
  button, as the button's description, so a screen reader's list of
  headings reads titles only. The whole row is the target all the same:
  the button's hit area is stretched over the row, so a tap on the
  description or on the empty row toggles too, while the actions slot,
  beside the title, sits above that layer and keeps its own controls.

  The title is set in the serif at weight 400, in sentence case; the
  description in muted sans. The row is at least 44 px tall. Nothing draws
  a line: the header lies on its group's tone, and a disclosure takes a
  faint wash under the pointer. The chevron turns a quarter when open, and
  does not move under reduced motion.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import Icon from './Icon.svelte';

	/** The title. Required. */
	export let title: string;
	/** The heading's level: 2, 3 or 4. */
	export let level: 2 | 3 | 4 = 3;
	/** A sentence under the title. */
	export let description: string | undefined = undefined;
	/** The heading's id, for a region it names. */
	export let headingId: string | undefined = undefined;
	/** Undefined for a plain heading; whether the region is shown for a disclosure. */
	export let expanded: boolean | undefined = undefined;
	/** The id of the region the disclosure shows or hides. */
	export let controls: string | undefined = undefined;

	const dispatch = createEventDispatcher<{ toggle: void }>();
	const descriptionId = `oo-panel-header-${Math.random().toString(36).slice(2, 9)}-desc`;

	$: if (typeof title !== 'string' || title.trim() === '') {
		throw new Error('PanelHeader needs a title');
	}
	$: tag = level === 2 ? 'h2' : level === 4 ? 'h4' : 'h3';
	$: disclosure = expanded !== undefined;
</script>

<div
	class="oo-panel-header"
	data-level={level}
	data-disclosure={disclosure}
	data-expanded={disclosure ? expanded : undefined}
>
	<div class="oo-panel-header-text">
		<svelte:element this={tag} id={headingId} class="oo-panel-header-title">
			{#if disclosure}
				<button
					type="button"
					class="oo-panel-header-toggle"
					aria-expanded={expanded}
					aria-controls={controls}
					aria-describedby={description ? descriptionId : undefined}
					on:click={() => dispatch('toggle')}
				>
					<span class="oo-panel-header-chevron"><Icon name="chevron-right" size="sm" /></span>
					<span class="oo-panel-header-label">{title}</span>
				</button>
			{:else}
				{title}
			{/if}
		</svelte:element>
		{#if description}
			<p id={descriptionId} class="oo-panel-header-desc">{description}</p>
		{/if}
	</div>
	{#if $$slots.actions}
		<div class="oo-panel-header-actions"><slot name="actions" /></div>
	{/if}
</div>

<style>
	.oo-panel-header {
		position: relative;
		display: flex;
		align-items: center;
		gap: var(--oo-space-3);
		box-sizing: border-box;
		min-height: 44px;
		padding: var(--oo-space-4) var(--oo-space-5);
		border-radius: var(--oo-panel-header-radius, var(--oo-radius-md));
	}
	.oo-panel-header[data-expanded='true'] {
		border-bottom-left-radius: 0;
		border-bottom-right-radius: 0;
	}
	.oo-panel-header[data-disclosure='true']:hover {
		background-color: var(--oo-bg-hover);
	}

	.oo-panel-header-text {
		display: flex;
		flex: 1;
		flex-direction: column;
		gap: var(--oo-space-1);
		min-width: 0;
	}

	.oo-panel-header-title {
		margin: 0;
		color: var(--oo-fg-primary);
		font-family: var(--oo-font-serif);
		font-size: var(--oo-text-base);
		font-weight: 400;
		line-height: var(--oo-leading-snug);
	}
	.oo-panel-header[data-level='2'] .oo-panel-header-title {
		font-size: var(--oo-text-lg);
	}
	.oo-panel-header[data-level='4'] .oo-panel-header-title {
		font-size: var(--oo-text-sm);
	}

	.oo-panel-header-toggle {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-2);
		padding: 0;
		border: 0;
		background: transparent;
		color: inherit;
		font: inherit;
		text-align: left;
		cursor: pointer;
	}
	/* The whole row answers the pointer: the hit area covers the header,
	   rounded as the header is, so it never reaches past a rounded card. */
	.oo-panel-header-toggle::after {
		content: '';
		position: absolute;
		inset: 0;
		border-radius: var(--oo-panel-header-radius, var(--oo-radius-md));
	}

	.oo-panel-header-chevron {
		display: inline-flex;
		flex-shrink: 0;
		color: var(--oo-fg-muted);
		transition: transform var(--oo-motion-fast) var(--oo-ease-default);
	}
	.oo-panel-header-toggle[aria-expanded='true'] .oo-panel-header-chevron {
		transform: rotate(90deg);
	}

	.oo-panel-header-desc {
		margin: 0;
		color: var(--oo-fg-muted);
		font-family: var(--oo-font-sans);
		font-size: var(--oo-text-sm);
		line-height: var(--oo-leading-snug);
	}

	/* Above the stretched hit area, so its controls keep their own. */
	.oo-panel-header-actions {
		position: relative;
		z-index: 1;
		display: flex;
		flex-shrink: 0;
		align-items: center;
		gap: var(--oo-space-2);
	}

	@media (prefers-reduced-motion: reduce) {
		.oo-panel-header-chevron {
			transition: none;
		}
	}
</style>
