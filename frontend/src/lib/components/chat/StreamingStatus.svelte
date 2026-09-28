<!--
  StreamingStatus.svelte (components/chat) -- what the stream of the reply
  being written has said so far, where the reply is written.

  Before any run, one waiting line: its words, and the onion beside them
  until the first token. A run the server executes shows its card instead:
  the run's name and the step it is at, its plants on one soil line
  (StepRow) and its step lines (StepList), held at the top of the thread
  while the run is open. After done the card folds into the reply's footer
  (RunSummary); a run the stream left cut short (the connection lost, an
  error) stays drawn as it ended. A late status (a silence, a Stop) is said
  on the running step's line, or under the card.

  Everything shown is the reducer's (lib/chat/progress.ts) in the word
  table's words (lib/chat/loaderWords.ts): nothing here grows a plant, reads
  step text or invents a state. One status region speaks for the whole
  stream: the reducer's transitions through its announcer, the last of a
  burst after calm, one a second, never the seconds.

  One clock: a timeout that sets the next one, for the live seconds, the
  plants' pace, the announcer and the silence. It stops at the end, and
  while the server is silent (the seconds are not counted then; the next
  frame starts it again, and the seconds shown are measured, not reset). A
  time handed in (`now`) draws that moment and starts no clock.
-->
<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import PixelStrip from '$lib/components/pixel/PixelStrip.svelte';
	import StepRow from './StepRow.svelte';
	import StepList from './StepList.svelte';
	import { ONION_INKS, ONION_STRIP } from '$lib/pixel/onionFrames';
	import {
		ANNOUNCE_GAP_MS,
		COALESCE_MS,
		SENDING_AFTER_MS,
		SILENCE_MS,
		announce,
		announcer,
		due,
		lineOf,
		moving,
		silent,
		summary,
		tick,
	} from '$lib/chat/progress';
	import type { Announcer, LoaderState, Run, Step } from '$lib/chat/progress';
	import { WORDS, fill, liveSeconds, runName, said, summaryText, words } from '$lib/chat/loaderWords';

	/** The reducer's state for the reply being written, or null. */
	export let loader: LoaderState | null = null;
	/** A time to draw at; null runs the component's own clock. */
	export let now: number | null = null;
	/** Reduced motion, when the caller knows it; null reads the preference. */
	export let reduced: boolean | null = null;
	/** Another view draws this stream: nothing is drawn here, the region still speaks. */
	export let quiet = false;

	const OPEN_STATES = ['pending', 'running'];

	let clock = Date.now();
	let mounted = false;
	let timer: ReturnType<typeof setTimeout> | null = null;
	let queue: Announcer = announcer();
	let spoken = '';
	let heard: LoaderState | null = null;
	let silenceSaid: number | null = null;
	let rowNext: number | null = null;
	let width = 600;

	function topRun(state: LoaderState | null): Run | null {
		return state ? state.runs.find((run) => run.parent === null && run.steps.length > 0) ?? null : null;
	}

	/** The step the card is at: the running one, else the last that started, else the first. */
	function currentStep(run: Run): Step {
		const running = run.steps.find((step) => step.state === 'running');
		if (running) return running;
		const started = run.steps.filter((step) => step.state !== 'pending');
		return started.length > 0 ? started[started.length - 1] : run.steps[0];
	}

	function startedAt(step: Step | null): number | null {
		const first = step ? step.given.find((given) => given.look.startsWith('running:')) : undefined;
		return first ? first.at : null;
	}

	$: t = now ?? clock;
	$: run = topRun(loader);
	$: runOpen = run !== null && run.steps.some((step) => OPEN_STATES.includes(step.state));
	$: showCard =
		!quiet && loader !== null && run !== null && (loader.end === null || loader.end === 'lost' || loader.end === 'error');
	$: hushed = loader !== null && silent(loader, t);
	$: counting = loader !== null && loader.end === null && !hushed;
	$: still = loader === null || !moving(loader, t);
	// A run's plants move while the server answers and nothing holds them: the
	// reducer's rule, less its first-word clause (the onion's), so the last
	// step's flowering plays while the reply is written.
	$: growing = loader !== null && loader.end === null && !loader.stopRequested && !hushed;
	$: line = loader ? lineOf(loader, t) : null;
	$: current = run ? currentStep(run) : null;
	$: stepSince = current && current.state === 'running' ? startedAt(current) : null;
	// A status said on the running step's line: one that came while it ran, or the silence.
	$: phase =
		loader && line && stepSince !== null && (hushed || loader.lineSince >= stepSince) ? words(line) : null;
	$: late = showCard && loader && line && phase === null && (hushed || loader.stopRequested) ? words(line) : null;
	$: lost = showCard && loader && loader.end === 'lost' ? summaryWords(loader) : null;
	$: waiting = !quiet && loader !== null && run === null && line !== null ? words(line) : null;
	$: onion = waiting !== null && loader !== null && !loader.writing;
	$: lineSeconds = waiting !== null && loader !== null && counting ? liveSeconds(loader.lineSince, t) : null;
	$: runSeconds = showCard && run !== null && counting ? liveSeconds(run.since, t) : null;
	$: at =
		run && current
			? fill(run.total !== null ? WORDS.at_step : WORDS.at_step_open, {
					i: current.index + 1,
					n: run.total ?? '',
					label: current.label,
				})
			: null;

	function summaryWords(state: LoaderState): string | null {
		const told = summary(state);
		return told ? summaryText(told) : null;
	}

	// The region: what each transition says, queued; the announcer lets it through.
	$: hear(loader);

	function hear(state: LoaderState | null): void {
		if (state === heard) return;
		if (state && (heard === null || state.startedAt !== heard.startedAt)) {
			queue = announcer();
			spoken = '';
			silenceSaid = null;
		}
		heard = state;
		if (state) {
			const at = now ?? Date.now();
			for (const told of state.said) queue = announce(queue, said(told), at);
		}
		schedule();
	}

	function speak(at: number): void {
		const out = due(queue, at);
		queue = out.announcer;
		if (out.text !== null) spoken = out.text;
	}

	/** The live seconds shown, each from when it counts. */
	function counters(): number[] {
		const from: number[] = [];
		if (loader && waiting !== null) from.push(loader.lineSince);
		if (run && showCard) from.push(run.since);
		if (stepSince !== null) from.push(stepSince);
		return from;
	}

	/** When anything shown next changes by itself, or null when nothing will. */
	function nextWake(at: number, rowAt: number | null): number | null {
		const times: number[] = [];
		if (queue.text !== null) {
			times.push(Math.max(queue.at + COALESCE_MS, queue.lastSpoken === null ? 0 : queue.lastSpoken + ANNOUNCE_GAP_MS));
		}
		if (loader && loader.end === null && !silent(loader, at)) {
			times.push(loader.lastFrameAt + SILENCE_MS);
			if (loader.startedAt + SENDING_AFTER_MS > at) times.push(loader.startedAt + SENDING_AFTER_MS);
			for (const since of counters()) times.push(since + 1000 * (Math.floor((at - since) / 1000) + 1));
		}
		if (rowAt !== null && !quiet) times.push(rowAt);
		return times.length > 0 ? Math.min(...times) : null;
	}

	function wake(): void {
		timer = null;
		clock = Date.now();
		if (loader && loader.end === null && loader.lastFrameAt !== silenceSaid) {
			const ticked = tick(loader, clock);
			if (ticked.said.length > 0) {
				silenceSaid = loader.lastFrameAt;
				for (const told of ticked.said) queue = announce(queue, said(told), clock);
			}
		}
		speak(clock);
		schedule();
	}

	/** The one clock: the next wake, and only that one. */
	function schedule(rowAt: number | null = rowNext): void {
		if (!mounted || now !== null) return;
		if (timer !== null) clearTimeout(timer);
		timer = null;
		const from = Date.now();
		const at = nextWake(from, rowAt);
		if (at !== null) timer = setTimeout(wake, Math.max(0, at - from));
	}

	$: if (mounted) schedule(rowNext);

	onMount(() => {
		mounted = true;
		clock = Date.now();
		schedule();
	});

	onDestroy(() => {
		mounted = false;
		if (timer !== null) clearTimeout(timer);
		timer = null;
	});
</script>

<div class="oo-stream" bind:clientWidth={width}>
	{#if showCard && run && loader}
		<div class="oo-run-card" class:oo-run-open={runOpen}>
			<div class="oo-run-head">
				<p class="oo-run-name">{runName(run.kind, run.name)}</p>
				{#if at}
					<p class="oo-run-at">
						{at}{#if runSeconds}<span class="oo-live-seconds" aria-hidden="true">{runSeconds}</span>{/if}
					</p>
				{/if}
			</div>
			<div class="oo-run-body">
				<StepRow
					{run}
					now={t}
					motion={growing}
					current={current ? current.index : null}
					column={width}
					{reduced}
					bind:next={rowNext}
				/>
				<StepList {run} now={t} {counting} {phase} current={current ? current.index : null} />
			</div>
			{#if lost}
				<p class="oo-run-late">{lost}</p>
			{:else if late}
				<p class="oo-run-late">{late}</p>
			{/if}
		</div>
	{:else if waiting !== null}
		<p class="oo-waiting">
			{#if onion}
				<PixelStrip strip={ONION_STRIP} inks={ONION_INKS} scale={2} moving={!still} {reduced} />
			{/if}
			<span class="oo-waiting-words"
				>{waiting}{#if lineSeconds}<span class="oo-live-seconds" aria-hidden="true">{lineSeconds}</span>{/if}</span
			>
		</p>
	{/if}
	<p class="sr-only" role="status">{spoken}</p>
</div>

<style>
	.oo-stream {
		display: flex;
		flex-direction: column;
	}

	/* The card stands on the thread's own ground, so text scrolled under it
	   while it is held at the top stays hidden; its edge is drawn where the
	   palette draws edges. */
	.oo-run-card {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
		padding: var(--oo-space-3) var(--oo-space-3) var(--oo-space-4);
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-lg);
		background-color: var(--oo-bg-surface);
	}

	.oo-run-card.oo-run-open {
		position: sticky;
		top: 0;
		z-index: 1;
	}

	.oo-run-head {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		column-gap: var(--oo-space-5);
		row-gap: var(--oo-space-1);
	}

	.oo-run-name {
		margin: 0;
		font-family: var(--oo-font-sans);
		font-size: var(--oo-text-base);
		font-weight: 600;
		line-height: 1.4;
		color: var(--oo-fg-primary);
	}

	.oo-run-at {
		margin: 0 0 0 auto;
		font-size: var(--oo-text-sm);
		line-height: 1.4;
		color: var(--oo-fg-muted);
		font-variant-numeric: tabular-nums;
	}

	.oo-run-body {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		column-gap: var(--oo-space-7);
		row-gap: var(--oo-space-4);
	}

	.oo-run-late {
		margin: 0;
		font-size: var(--oo-text-sm);
		line-height: 1.5;
		color: var(--oo-fg-muted);
	}

	.oo-waiting {
		display: flex;
		align-items: center;
		gap: var(--oo-space-3);
		min-height: 40px;
		margin: 0;
		padding: 0 var(--oo-space-3);
		font-size: var(--oo-text-sm);
		line-height: 1.4;
		color: var(--oo-fg-muted);
	}

	.oo-waiting-words {
		font-variant-numeric: tabular-nums;
	}

	/* On a phone the step it is at goes under the run's name. */
	@media (max-width: 639px) {
		.oo-run-head {
			flex-direction: column;
		}

		.oo-run-at {
			margin-left: 0;
		}
	}

	@media (min-width: 640px) {
		.oo-run-card {
			padding-inline: var(--oo-space-5);
		}

		.oo-waiting {
			padding-inline: var(--oo-space-5);
		}
	}
</style>
