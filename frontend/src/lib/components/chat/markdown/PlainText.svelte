<script context="module" lang="ts">
	let shownTexts = 0;
</script>

<script lang="ts">
	/**
	 * PlainText.svelte -- text shown as it is written: a user's message, or a
	 * reply that was not lexed (its lexing would take too long, or marked
	 * could not lex it). Line breaks and indentation are kept. A long text
	 * shows its first whole lines and a toggle for the rest
	 * (lib/markdown/collapse.ts); the hidden lines are not rendered.
	 */
	import { collapseLines } from '$lib/markdown/collapse';
	import TextButton from '$lib/ds/TextButton.svelte';
	import Caret from './Caret.svelte';

	/** The text; anything but a string shows as nothing. */
	export let text: string;
	/** Lines above which the text collapses; 0 never collapses. */
	export let collapseAbove = 0;
	/** Lines shown while collapsed. */
	export let collapseKeep = 20;
	/** True to draw the streaming caret after the last character. */
	export let caret = false;

	shownTexts += 1;
	const regionId = `oo-plain-${shownTexts}`;

	let expanded = false;

	$: view = collapseLines(typeof text === 'string' ? text : '', {
		threshold: collapseAbove,
		keep: collapseKeep,
		expanded
	});
	$: toggleLabel = expanded
		? 'Show less'
		: `Show the rest (${view.hidden} more ${view.hidden === 1 ? 'line' : 'lines'})`;
</script>

<div id={regionId}><p class="oo-md-plain">{view.shown}{#if caret}<Caret />{/if}</p></div>
{#if view.collapsible}
	<TextButton {expanded} controls={regionId} on:click={() => (expanded = !expanded)}
		>{toggleLabel}</TextButton
	>
{/if}

<style>
	.oo-md-plain {
		margin: 0;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
</style>
