<!--
  RunSummary.svelte (components/chat) -- the reply's footer line after its
  run: the run's name, its steps and their ends counted, and the duration
  the server measured; or, after a Stop, where the run stopped. It is read
  only from the record `done` carries (`steps`): a reply without that record
  has no line, and nothing is ever rebuilt from the reply's text.

  The line is a disclosure: it opens the run as it ended, the plants still
  and the steps with their durations. A reply stopped without a run says
  Stopped alone.
-->
<script context="module" lang="ts">
	let made = 0;
</script>

<script lang="ts">
	import Icon from '$lib/ds/Icon.svelte';
	import TextButton from '$lib/ds/TextButton.svelte';
	import StepRow from './StepRow.svelte';
	import StepList from './StepList.svelte';
	import { observe, start, summary } from '$lib/chat/progress';
	import { WORDS, summaryText } from '$lib/chat/loaderWords';
	import type { PipelineStepFrame } from '$lib/types';

	/** The steps' record the reply's `done` carried, if any. */
	export let steps: PipelineStepFrame[] | null | undefined = undefined;
	/** The reply's duration, as the server measured it. */
	export let durationMs: number | null | undefined = undefined;
	/** Whether a Stop ended the reply. */
	export let stopped = false;

	made += 1;
	const id = `oo-run-summary-${made}`;
	let open = false;

	// The record, read the way the stream read it: a `done` that carries it.
	$: record =
		steps && steps.length > 0
			? observe(start(0), { type: 'done', content: '', metadata: { steps, duration_ms: durationMs, cancelled: stopped } }, 0)
			: null;
	$: told = record ? summary(record) : null;
	// The line breaks only after a comma: each part after the run's name holds
	// together, the last one with the chevron.
	$: parts = told ? summaryText(told).split(/(?<=,) /) : [];
	$: run = record ? record.runs.find((each) => each.parent === null && each.steps.length > 0) ?? null : null;
</script>

{#if told && run}
	<div class="oo-run-summary">
		<TextButton expanded={open} controls={id} on:click={() => (open = !open)}>
			<span class="oo-run-summary-line"
				>{#each parts as part, i}{#if i > 0}{' '}{/if}<span class:oo-run-summary-part={i > 0}
						>{part}{#if i === parts.length - 1}<span class="oo-run-summary-mark"
								><Icon name={open ? 'chevron-up' : 'chevron-down'} size="sm" /></span
							>{/if}</span
					>{/each}</span
			>
		</TextButton>
		<div class="oo-run-summary-run" {id} hidden={!open}>
			{#if open}
				<StepRow {run} now={0} live={false} />
				<StepList {run} />
			{/if}
		</div>
	</div>
{:else if stopped}
	<p class="oo-run-summary-stopped">{WORDS.stopped}</p>
{/if}

<style>
	.oo-run-summary {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--oo-space-3);
		margin-top: var(--oo-space-2);
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}

	/* The line starts where the reply's text starts, and wraps as text does,
	   the chevron after its last word. */
	.oo-run-summary > :global(.oo-text-btn) {
		margin-left: calc(-1 * var(--oo-space-3));
		text-align: left;
		font-variant-numeric: tabular-nums;
	}

	.oo-run-summary-line {
		line-height: 1.5;
	}

	.oo-run-summary-part {
		white-space: nowrap;
	}

	.oo-run-summary-mark {
		display: inline-block;
		margin-left: var(--oo-space-1);
		vertical-align: middle;
	}

	.oo-run-summary-run {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		column-gap: var(--oo-space-8);
		row-gap: var(--oo-space-4);
		padding-bottom: var(--oo-space-2);
	}

	.oo-run-summary-run[hidden] {
		display: none;
	}

	.oo-run-summary-stopped {
		margin: var(--oo-space-2) 0 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}

	/* On a phone the line is a full target. */
	@media (max-width: 639px) {
		.oo-run-summary > :global(.oo-text-btn) {
			min-height: 44px;
		}
	}
</style>
