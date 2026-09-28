<!--
  StepList.svelte (components/chat) -- a run's steps as a list of lines:
  the number, the step's name as the server sent it, and its state in the
  word table's words. A running step says its sub-progress when the server
  counts one, and its seconds from the fifth (counted in the tab, hidden
  from a screen reader); a finished one says the duration the server
  measured; a failed one says Failed in the stop ink, then the server's
  reason. The words carry every state: the plants beside them only repeat
  it.
-->
<script lang="ts">
	import {
		WORDS,
		durationWords,
		fill,
		liveSeconds,
		progressWords,
		stepWords,
	} from '$lib/chat/loaderWords';
	import type { Run, Step } from '$lib/chat/progress';

	/** The run whose steps are listed. */
	export let run: Run;
	/** The time the list is drawn at, for the live seconds; null draws none. */
	export let now: number | null = null;
	/** Whether live seconds count now (they stop at the end and in a silence). */
	export let counting = false;
	/** What the running step says in place of Running (a phase or a late status), or null. */
	export let phase: string | null = null;
	/** The index of the step drawn as the current one, or null. */
	export let current: number | null = null;

	/** When the step started running, as the tab saw it; null if it never did. */
	function startedAt(step: Step): number | null {
		const first = step.given.find((given) => given.look.startsWith('running:'));
		return first ? first.at : null;
	}

	function secondsOf(step: Step, at: number | null, on: boolean): string | null {
		const since = startedAt(step);
		return on && at !== null && step.state === 'running' && since !== null ? liveSeconds(since, at) : null;
	}
</script>

<ol class="oo-step-list" role="list">
	{#each run.steps as step (step.index)}
		{@const running = step.state === 'running'}
		{@const sub = progressWords(step)}
		{@const seconds = secondsOf(step, now, counting)}
		<li
			class="oo-step-line"
			class:oo-step-line-current={step.index === current}
			class:oo-step-line-waiting={step.state === 'pending' || step.state === 'not_run'}
		>
			<span class="oo-step-n">{step.index + 1}</span>
			<span class="oo-step-text">
				<span class="oo-step-label">{step.label}</span>
				<span class="oo-step-state">
					{#if running}
						<span class="oo-step-phase">{phase ?? WORDS.state_running}</span>{#if sub}{fill(WORDS.and, {
								words: sub,
							})}{/if}{#if seconds}<span class="oo-live-seconds" aria-hidden="true">{seconds}</span>{/if}
					{:else if step.state === 'failed'}
						<span class="oo-step-failed">{WORDS.state_failed}</span>{#if step.reason}{fill(WORDS.reason, {
								reason: step.reason,
							})}{/if}{#if step.durationMs !== null}{durationWords(step.durationMs)}{/if}
					{:else}
						{stepWords(step)}{#if step.durationMs !== null}{durationWords(step.durationMs)}{/if}
					{/if}
				</span>
			</span>
		</li>
	{/each}
</ol>

<style>
	.oo-step-list {
		display: flex;
		flex: 1 1 16rem;
		flex-direction: column;
		gap: var(--oo-space-1);
		min-width: 0;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.oo-step-line {
		display: grid;
		grid-template-columns: 2ch minmax(0, 1fr);
		column-gap: var(--oo-space-3);
		font-size: var(--oo-text-sm);
		line-height: 1.55;
		color: var(--oo-fg-muted);
	}

	.oo-step-n {
		font-variant-numeric: tabular-nums;
	}

	.oo-step-text {
		min-width: 0;
		overflow-wrap: anywhere;
	}

	.oo-step-label {
		margin-right: var(--oo-space-3);
		color: var(--oo-fg-primary);
	}

	.oo-step-line-current .oo-step-label {
		font-weight: 600;
	}

	.oo-step-line-waiting .oo-step-label {
		color: var(--oo-fg-muted);
	}

	.oo-step-state {
		font-variant-numeric: tabular-nums;
	}

	.oo-step-line-current .oo-step-phase {
		color: var(--oo-fg-primary);
	}

	.oo-step-failed {
		font-weight: 500;
		color: var(--oo-fg-stop);
	}
</style>
