<!--
  SecurityChecks.svelte
  What the security grade is made of: its letter, its score out of the
  maximum, how many of its checks passed, and each check with its points
  and its detail. Whether a check passed is said in a word, "Passed" or
  "Not passed", beside a mark; never by a colour alone.

  Presentational: it shows the status it is given (lib/api/security.ts).
-->
<script lang="ts">
	import Icon from '$lib/ds/Icon.svelte';
	import type { SecurityStatus } from '$lib/api/security';

	export let status: SecurityStatus;
	/** Whether to lead with the letter, the score and the count (a page that shows them already says no). */
	export let summary = true;

	$: checks = status.checks ?? [];
	$: passed = checks.filter((check) => check.passed).length;

	/** A check's name as the server spells it (snake case), in sentence case. */
	function named(name: string): string {
		const words = name.replace(/_/g, ' ').trim();
		return words.charAt(0).toUpperCase() + words.slice(1);
	}
</script>

<div class="oo-sec-checks">
	{#if summary}
		<p class="oo-sec-summary">
			<span class="oo-sec-letter">{status.grade}</span>
			<span>Score {status.score} of {status.max_score}</span>
			<span>{passed} of {checks.length} checks passed</span>
		</p>
	{/if}
	{#if checks.length > 0}
		<ul class="oo-sec-list">
			{#each checks as check (check.name)}
				<li class="oo-sec-check">
					<span class="oo-sec-mark" aria-hidden="true">
						<Icon name={check.passed ? 'check' : 'x'} size="sm" />
					</span>
					<span class="oo-sec-what">
						<span class="oo-sec-name">{named(check.name)}</span>
						{#if check.detail}
							<span class="oo-sec-detail">{check.detail}</span>
						{/if}
					</span>
					<span class="oo-sec-points">{check.points} of {check.max_points}</span>
					<span class="oo-sec-word">{check.passed ? 'Passed' : 'Not passed'}</span>
				</li>
			{/each}
		</ul>
	{/if}
</div>

<style>
	.oo-sec-checks {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}

	.oo-sec-summary {
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
		font-size: var(--oo-text-2xl);
		line-height: 1;
	}

	.oo-sec-list {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.oo-sec-check {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) auto auto;
		align-items: baseline;
		gap: var(--oo-space-3);
		font-size: var(--oo-text-sm);
	}
	.oo-sec-mark {
		display: inline-flex;
		align-self: center;
		color: var(--oo-fg-secondary);
	}
	.oo-sec-what {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	.oo-sec-name {
		color: var(--oo-fg-primary);
	}
	.oo-sec-detail {
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-xs);
	}
	.oo-sec-points {
		color: var(--oo-fg-muted);
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}
	.oo-sec-word {
		color: var(--oo-fg-secondary);
		white-space: nowrap;
	}

	/* On a phone the points and the word go under the check's name. */
	@media (max-width: 480px) {
		.oo-sec-check {
			grid-template-columns: auto minmax(0, 1fr);
			row-gap: var(--oo-space-1);
		}
		.oo-sec-points,
		.oo-sec-word {
			grid-column: 2;
		}
	}
</style>
