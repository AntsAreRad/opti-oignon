<!--
  SecurityGrade.svelte
  The head of the Workshop's Security page: the grade the status card's
  badge shows, and what it is made of.

  The letter, named as the grade, the score and how many checks passed
  are always shown; the checks' own list, below, does not repeat them. A
  disclosure, closed at first ("Checks, sessions and recent events"), opens
  each check with its points and its detail (SecurityChecks), how sessions
  are kept (read from the auth store), and the ten most recent security
  events: sign-in activity (logins, failed logins, registrations, password
  changes), sandbox blocks, sign-in lockouts and detected search
  injections, each with its source, its time and its severity in a word.
  The events are read the first time the disclosure opens. A grade that
  cannot be read says so, with the reason and a retry; the failure and its
  retry stay on screen while the grade is read again, so the retry keeps
  focus, and a grade read at last takes it. Events that cannot be read say
  so in place.

  It reads through lib/api/security.ts. Its props are its state, set by the
  component itself once mounted, so each state can be drawn on the server.
-->
<script lang="ts">
	import { onMount, tick } from 'svelte';
	import Button from '$lib/ds/Button.svelte';
	import PanelHeader from '$lib/ds/PanelHeader.svelte';
	import SecurityChecks from './SecurityChecks.svelte';
	import {
		getSecurityEvents,
		getSecurityStatus,
		type SecurityEvent,
		type SecurityStatus
	} from '$lib/api/security';
	import { parseApiError } from '$lib/api/errorHandler';
	import { authStatus } from '$lib/stores/auth';

	/** The grade, once read. */
	export let status: SecurityStatus | null = null;
	/** Why the grade could not be read, or null. */
	export let failure: string | null = null;
	/** The recent security events, once read. */
	export let events: SecurityEvent[] | null = null;
	/** Why the events could not be read, or null. */
	export let eventsFailure: string | null = null;
	/** Whether the detail is open. */
	export let expanded = false;

	const detailId = 'oo-sec-grade-detail';
	let loading = false;
	let headline: HTMLElement | undefined;

	$: checks = status?.checks ?? [];
	$: passed = checks.filter((check) => check.passed).length;
	$: cookies = $authStatus?.cookie_mode;

	/** Reads the grade; a failure stays shown until the answer replaces it. */
	async function loadStatus() {
		loading = true;
		try {
			status = await getSecurityStatus();
			failure = null;
		} catch (error) {
			failure = parseApiError(error).message;
		} finally {
			loading = false;
		}
	}

	/** Reads the grade again; once it is read, focus goes to it. */
	async function retry() {
		if (loading) return;
		await loadStatus();
		if (!failure) {
			await tick();
			headline?.focus();
		}
	}

	async function loadEvents() {
		eventsFailure = null;
		try {
			events = (await getSecurityEvents(10)).events ?? [];
		} catch (error) {
			events = null;
			eventsFailure = parseApiError(error).message;
		}
	}

	function toggleDetail() {
		expanded = !expanded;
		if (expanded && events === null) loadEvents();
	}

	function when(timestamp: number): string {
		return timestamp ? new Date(timestamp * 1000).toLocaleString() : 'Time unknown';
	}

	onMount(() => {
		if (!status && !failure) loadStatus();
		if (expanded && events === null) loadEvents();
	});
</script>

<section class="oo-sec-grade" aria-label="Security grade">
	{#if failure}
		<div class="oo-sec-failure" role="alert">
			<p>Could not read the security grade: {failure}</p>
			<Button variant="secondary" size="sm" iconLeft="retry" on:click={retry}>Retry</Button>
		</div>
	{:else if !status}
		<p class="oo-sec-muted">{loading ? 'Reading the security grade' : 'The security grade has not been read yet'}</p>
	{:else}
		<p class="oo-sec-headline" tabindex="-1" bind:this={headline}>
			<span class="oo-sec-letter"><span class="oo-sr-only">Grade </span>{status.grade}</span>
			<span>Score {status.score} of {status.max_score}</span>
			<span>{passed} of {checks.length} checks passed</span>
		</p>
		<PanelHeader
			title="Checks, sessions and recent events"
			level={2}
			{expanded}
			controls={detailId}
			on:toggle={toggleDetail}
		/>
		<div id={detailId} class="oo-sec-detail" hidden={!expanded}>
			{#if expanded}
				<SecurityChecks {status} summary={false} />

				<h3 class="oo-sec-subtitle">Sessions</h3>
				<p class="oo-sec-line">
					{#if cookies === true}
						Sessions use httpOnly cookies
					{:else if cookies === false}
						Sessions use browser storage, which is less safe
					{:else}
						How sessions are kept is not known yet
					{/if}
				</p>

				<h3 class="oo-sec-subtitle">Recent security events</h3>
				{#if eventsFailure}
					<p class="oo-sec-line" role="alert">Could not read the recent security events: {eventsFailure}</p>
				{:else if events === null}
					<p class="oo-sec-muted">Reading the recent security events</p>
				{:else if events.length === 0}
					<p class="oo-sec-muted">No recent security events</p>
				{:else}
					<ul class="oo-sec-events">
						{#each events as event, index (index)}
							<li class="oo-sec-event">
								<span class="oo-sec-action">{event.action}</span>
								<span class="oo-sec-meta">
									<span>{event.source}</span>
									<span>{when(event.timestamp)}</span>
									<span>Severity: {event.severity}</span>
								</span>
							</li>
						{/each}
					</ul>
				{/if}
			{/if}
		</div>
	{/if}
</section>

<style>
	.oo-sec-grade {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
		max-width: 880px;
		margin: 0 auto var(--oo-space-6);
		padding: var(--oo-space-5);
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-lg);
		background-color: var(--oo-bg-subtle);
		--oo-panel-header-radius: var(--oo-radius-md);
	}
	.oo-sec-grade :global(.oo-panel-header) {
		margin: 0 calc(-1 * var(--oo-space-3));
		padding: var(--oo-space-3);
	}

	.oo-sec-headline {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: var(--oo-space-2) var(--oo-space-4);
		margin: 0;
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
	}
	.oo-sec-letter {
		color: var(--oo-fg-primary);
		font-family: var(--oo-font-serif);
		font-size: var(--oo-text-3xl);
		line-height: 1;
	}

	.oo-sec-detail {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}
	.oo-sec-detail[hidden] {
		display: none;
	}

	.oo-sec-subtitle {
		margin: var(--oo-space-3) 0 0;
		color: var(--oo-fg-primary);
		font-family: var(--oo-font-serif);
		font-size: var(--oo-text-base);
		font-weight: 400;
	}
	.oo-sec-line,
	.oo-sec-muted {
		margin: 0;
		font-size: var(--oo-text-sm);
	}
	.oo-sec-line {
		color: var(--oo-fg-secondary);
	}
	.oo-sec-muted {
		color: var(--oo-fg-muted);
	}

	.oo-sec-failure {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--oo-space-3);
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
	}
	.oo-sec-failure p {
		margin: 0;
	}

	.oo-sec-events {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.oo-sec-event {
		display: flex;
		flex-direction: column;
		font-size: var(--oo-text-sm);
	}
	.oo-sec-action {
		color: var(--oo-fg-primary);
	}
	.oo-sec-meta {
		display: flex;
		flex-wrap: wrap;
		gap: var(--oo-space-1) var(--oo-space-3);
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-xs);
	}
</style>
