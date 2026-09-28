<!--
  StepRow.svelte (components/chat) -- a run's plants: one per step, standing
  on one soil line, the step numbers under them (the current one bold, a
  failed one in the stop ink). Decorative: the step list beside it says
  every state in words.

  Each plant shows what the pacing (lib/chat/pacing.ts) schedules from the
  looks its step was given (lib/chat/progress.ts): every look held at least
  400 ms, a crowded queue jumping to its last, an end never kept waiting. One
  thing moves at a time: while a finished plant flowers, the next plant's
  root waits for the flowering to end; otherwise the one running plant
  sways, and none when several run at once or when the loader holds still.
  Played (`live`), the row says when its drawing next changes, for the
  caller's clock; frozen, every plant shows its last look, still.

  The row fits its column: 3x when every plant fits, else 2x, else a window
  around the current step with the plants left out counted in words.
-->
<script lang="ts">
	import PixelStrip from '$lib/components/pixel/PixelStrip.svelte';
	import { PLANT_INKS, SOIL_GAP, fitRow, plantStrip } from '$lib/pixel/plantFrames';
	import type { Strip } from '$lib/pixel/plantFrames';
	import { nextAt, schedule, shownAt } from '$lib/chat/pacing';
	import type { Shown } from '$lib/chat/pacing';
	import { WORDS, fill } from '$lib/chat/loaderWords';
	import { lookOf } from '$lib/chat/progress';
	import type { Run, Stage, Step, StepState } from '$lib/chat/progress';

	/** The run whose steps the row draws. */
	export let run: Run;
	/** The time the row is drawn at. */
	export let now: number;
	/** Played: paced, with its one moving thing; false: each plant's last look, still. */
	export let live = true;
	/** Whether the loader's moving thing may move now (the server answers, nothing holds it). */
	export let motion = true;
	/** The index of the step drawn as the current one, or null. */
	export let current: number | null = null;
	/** The column the row fits into, in CSS pixels. */
	export let column = 600;
	/** The device pixel ratio an art pixel is rounded to. */
	export let dpr = 1;
	/** Reduced motion, when the caller knows it; null reads the preference. */
	export let reduced: boolean | null = null;
	/** When the row's drawing next changes by itself, or null (for the caller's clock). */
	export let next: number | null = null;

	const OPEN: readonly StepState[] = ['pending', 'running'];
	const FLOWER = plantStrip('done', 'bulb');
	/** A flowering plays its lead frames once. */
	const FLOWERING_MS = FLOWER.lead * FLOWER.leadMs;
	/** Before its first look a plant is bare soil: the row's line, nothing on it yet. */
	const BARE = 'skipped:soil';

	// One strip per look, so a plant that keeps its look keeps its strip (and its motion).
	const strips = new Map<string, Strip>();

	function stripOf(look: string): Strip {
		let strip = strips.get(look);
		if (!strip) {
			const [state, stage] = look.split(':');
			strip = plantStrip(state as StepState, stage as Stage);
			strips.set(look, strip);
		}
		return strip;
	}

	function isGrowing(look: string): boolean {
		return look.startsWith('running:');
	}

	function isFinished(look: string): boolean {
		return look.startsWith('done:');
	}

	/** A plant's root waits while another plant flowers: one moving thing. */
	function waitForFlowering(plans: Shown[][]): Shown[][] {
		const flowerings = plans
			.flatMap((plan) => plan.filter((entry) => isFinished(entry.look)).map((entry) => entry.at))
			.sort((a, b) => a - b);
		return plans.map((plan) =>
			plan.flatMap((entry, i) => {
				let at = entry.at;
				if (isGrowing(entry.look) && !(i > 0 && isGrowing(plan[i - 1].look))) {
					for (const start of flowerings) if (at >= start && at < start + FLOWERING_MS) at = start + FLOWERING_MS;
				}
				const following = i + 1 < plan.length ? plan[i + 1].at : Infinity;
				return at < following ? [{ at, look: entry.look }] : [];
			}),
		);
	}

	/** When the look shown at `at` began, or null before the first. */
	function shownSince(plan: Shown[], at: number): number | null {
		let since: number | null = null;
		for (const entry of plan) {
			if (entry.at > at) break;
			since = entry.at;
		}
		return since;
	}

	function lastGiven(step: Step): number {
		return step.given.length > 0 ? step.given[step.given.length - 1].at : run.since;
	}

	$: open = run.steps.some((step) => OPEN.includes(step.state));
	$: ended = open ? null : Math.max(run.since, ...run.steps.map(lastGiven));
	$: plans = live
		? waitForFlowering(run.steps.map((step) => schedule(step.given, run.since, ended)))
		: run.steps.map((step) => [{ at: -Infinity, look: lookOf(step) }]);
	$: shown = plans.map((plan) => shownAt(plan, now) ?? BARE);
	$: since = plans.map((plan) => shownSince(plan, now));
	$: flowering = shown.map((look, i) => {
		const began = since[i];
		return live && isFinished(look) && began !== null && now < began + FLOWERING_MS;
	});
	$: mover = moverOf(flowering, shown, motion && live);
	$: next = live ? soonest(plans, flowering, since) : null;

	/** The one plant that moves: a flowering first, else the one running plant. */
	function moverOf(blooming: boolean[], looks: string[], may: boolean): number | null {
		if (!may) return null;
		const flower = blooming.lastIndexOf(true);
		if (flower >= 0) return flower;
		const growing = looks.flatMap((look, i) => (isGrowing(look) ? [i] : []));
		return growing.length === 1 ? growing[0] : null;
	}

	function soonest(all: Shown[][], blooming: boolean[], began: (number | null)[]): number | null {
		const times: number[] = [];
		all.forEach((plan, i) => {
			const change = nextAt(plan, now);
			if (change !== null) times.push(change);
			const start = began[i];
			if (blooming[i] && start !== null) times.push(start + FLOWERING_MS);
		});
		return times.length > 0 ? Math.min(...times) : null;
	}

	$: at = Math.max(0, run.steps.findIndex((step) => step.index === current));
	$: fit = fitRow(run.steps.length, at, column, dpr);
	$: inView = run.steps.map((step, i) => ({ step, i })).slice(fit.first, fit.first + fit.count);
</script>

<div class="oo-step-row" aria-hidden="true">
	{#if fit.earlier > 0}
		<span class="oo-step-row-more">{fill(WORDS.row_earlier, { n: fit.earlier })}</span>
	{/if}
	{#each inView as { step, i }, k (step.index)}
		{#if k > 0}
			<span class="oo-step-gap">
				<PixelStrip strip={SOIL_GAP} inks={PLANT_INKS} scale={fit.scale} {dpr} moving={false} {reduced} />
			</span>
		{/if}
		<span class="oo-step-plant">
			<PixelStrip
				strip={stripOf(shown[i])}
				inks={PLANT_INKS}
				scale={fit.scale}
				{dpr}
				moving={mover === i}
				{reduced}
			/>
			<span
				class="oo-step-number"
				class:oo-step-number-current={step.index === current}
				class:oo-step-number-failed={shown[i].startsWith('failed:')}>{step.index + 1}</span
			>
		</span>
	{/each}
	{#if fit.later > 0}
		<span class="oo-step-row-more">{fill(WORDS.row_later, { n: fit.later })}</span>
	{/if}
</div>

<style>
	.oo-step-row {
		display: flex;
		flex-shrink: 0;
		align-items: flex-start;
	}

	.oo-step-plant,
	.oo-step-gap {
		display: flex;
		flex-direction: column;
		align-items: center;
	}

	.oo-step-number {
		margin-top: var(--oo-space-2);
		font-size: var(--oo-text-xs);
		line-height: 1;
		color: var(--oo-fg-muted);
		font-variant-numeric: tabular-nums;
	}

	.oo-step-number-current {
		font-weight: 600;
		color: var(--oo-fg-primary);
	}

	.oo-step-number-failed {
		font-weight: 600;
		color: var(--oo-fg-stop);
	}

	.oo-step-row-more {
		align-self: center;
		padding: 0 var(--oo-space-2);
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
		white-space: nowrap;
	}
</style>
