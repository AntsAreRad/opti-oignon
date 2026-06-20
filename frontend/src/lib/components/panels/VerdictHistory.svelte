<!--
  VerdictHistory.svelte (S279, the verdict-history affordance)
  A session-local list of the most recent verification verdicts, read from the
  verdictHistory store. It is a read-only display: the verdicts come only from
  results the server returned (a verifier surface records them), and the only
  control is Clear, which empties the session history. Newest first, capped to
  the store's MAX_VERDICT_HISTORY. There is no mode gate -- the surface runs
  identically in Daily and Bulbe. Design-system tokens only (--oo-*);
  lucide-svelte icons through Icon.
-->
<script lang="ts">
	import { Card, Icon, Button, EmptyState } from '$lib/ds';
	import { verdictHistory, clearVerdicts } from '$lib/stores/verdictHistory';

	// Read-only display mapping; the verdict itself is the server's. Unknown
	// verdicts fall back to the uncertain tone, never to a supported tone.
	const TONE: Record<string, string> = {
		supported: 'ok',
		unsupported: 'bad',
		uncertain: 'warn'
	};
	const GLYPH: Record<string, string> = {
		supported: 'shield-check',
		unsupported: 'shield-alert',
		uncertain: 'help-circle'
	};

	function toneOf(verdict: string): string {
		return TONE[verdict] ?? 'warn';
	}

	function glyphOf(verdict: string): string {
		return GLYPH[verdict] ?? 'help-circle';
	}

	function timeOf(ms: number): string {
		try {
			return new Date(ms).toLocaleTimeString();
		} catch {
			return '';
		}
	}
</script>

<Card variant="flat" padding="sm" class="verdict-history">
	<div class="vh-head">
		<Icon name="history" size="sm" />
		<span class="vh-title">Recent verdicts</span>
		{#if $verdictHistory.length > 0}
			<div class="vh-clear">
				<Button variant="ghost" size="sm" on:click={clearVerdicts}>Clear</Button>
			</div>
		{/if}
	</div>

	{#if $verdictHistory.length === 0}
		<EmptyState
			icon="history"
			title="No recent verdicts"
			description="Verdicts you run this session show up here."
		/>
	{:else}
		<ul class="vh-list">
			{#each $verdictHistory as entry (entry.id)}
				<li class="vh-item" data-tone={toneOf(entry.verdict)}>
					<Icon name={glyphOf(entry.verdict)} size="sm" />
					<span class="vh-verdict">{entry.verdict}</span>
					<span class="vh-surface">{entry.surface}</span>
					<span class="vh-summary">{entry.summary}</span>
					<span class="vh-time">{timeOf(entry.at)}</span>
				</li>
			{/each}
		</ul>
	{/if}
</Card>

<style>
	.vh-head {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-tertiary);
		margin-bottom: var(--oo-space-2);
	}

	.vh-title {
		font-size: var(--oo-text-sm);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.vh-clear {
		margin-left: auto;
	}

	.vh-list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
	}

	.vh-item {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		padding: var(--oo-space-1) 0;
	}

	.vh-verdict {
		font-size: var(--oo-text-sm);
		font-weight: 600;
		text-transform: capitalize;
	}

	.vh-item[data-tone='ok'] .vh-verdict {
		color: var(--oo-fg-success);
	}

	.vh-item[data-tone='bad'] .vh-verdict {
		color: var(--oo-fg-danger);
	}

	.vh-item[data-tone='warn'] .vh-verdict {
		color: var(--oo-fg-warning);
	}

	.vh-surface {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-tertiary);
		text-transform: capitalize;
		padding: 0 var(--oo-space-1);
		border: 1px solid var(--oo-border-subtle);
		border-radius: var(--oo-radius-sm);
		background: var(--oo-bg-subtle);
	}

	.vh-summary {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
	}

	.vh-time {
		margin-left: auto;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}
</style>
