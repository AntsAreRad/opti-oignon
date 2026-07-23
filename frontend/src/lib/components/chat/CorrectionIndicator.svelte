<!--
  CorrectionIndicator.svelte
  Lightweight indicator showing whether self-correction was applied.
  
  Shows a discreet badge with the number of iterations,
  le score de conformite avant/apres, et l'amelioration de qualite.
  Click to expand the details.
-->
<script lang="ts">
	export let correction: {
		was_corrected: boolean;
		iterations_performed: number;
		compliance_before: number;
		compliance_after: number;
		quality_before: number;
		quality_after: number;
		total_duration_ms: number;
	} | null = null;

	let expanded = false;

	function toggle() {
		expanded = !expanded;
	}

	$: wasCorrected = correction?.was_corrected ?? false;
	$: iterations = correction?.iterations_performed ?? 0;
	$: complianceBefore = correction?.compliance_before ?? 1.0;
	$: complianceAfter = correction?.compliance_after ?? 1.0;
	$: qualityBefore = correction?.quality_before ?? 1.0;
	$: qualityAfter = correction?.quality_after ?? 1.0;
	$: durationMs = correction?.total_duration_ms ?? 0;

	$: complianceImproved = complianceAfter > complianceBefore;
	$: qualityImproved = qualityAfter > qualityBefore;
	$: hasImprovement = complianceImproved || qualityImproved;

	function formatScore(score: number): string {
		return `${Math.round(score * 100)}%`;
	}

	function formatDuration(ms: number): string {
		if (ms < 1000) return `${ms}ms`;
		return `${(ms / 1000).toFixed(1)}s`;
	}
</script>

{#if correction && wasCorrected}
	<button
		class="correction-indicator"
		class:expanded
		on:click={toggle}
		title="Self-correction was applied"
	>
		<!-- Compact line -->
		<div class="correction-compact">
			<span class="correction-icon">&#x2714;</span>
			<span class="correction-label">Corrected</span>
			{#if iterations > 0}
				<span class="correction-iterations">{iterations} iter.</span>
			{/if}
			{#if hasImprovement}
				<span class="correction-improved">improved</span>
			{/if}
			<span class="correction-chevron">{expanded ? '\u25B4' : '\u25BE'}</span>
		</div>

		<!-- Expandable details -->
		{#if expanded}
			<div class="correction-details">
				<div class="correction-detail-row">
					<span class="detail-label">Compliance:</span>
					<span class="detail-value">
						{formatScore(complianceBefore)}
						{#if complianceImproved}
							<span class="arrow-up">&rarr;</span>
							<span class="score-improved">{formatScore(complianceAfter)}</span>
						{/if}
					</span>
				</div>
				<div class="correction-detail-row">
					<span class="detail-label">Quality:</span>
					<span class="detail-value">
						{formatScore(qualityBefore)}
						{#if qualityImproved}
							<span class="arrow-up">&rarr;</span>
							<span class="score-improved">{formatScore(qualityAfter)}</span>
						{/if}
					</span>
				</div>
				<div class="correction-detail-row">
					<span class="detail-label">Duration:</span>
					<span class="detail-value">{formatDuration(durationMs)}</span>
				</div>
			</div>
		{/if}
	</button>
{:else if correction && !wasCorrected}
	<span class="correction-pass" title="Response passed quality checks">
		<span class="pass-icon">&#x2713;</span>
		<span class="pass-label">Verified</span>
	</span>
{/if}

<style>
	.correction-indicator {
		display: inline-flex;
		flex-direction: column;
		gap: 4px;
		background: var(--oo-bg-elevated);
		border: 1px solid var(--oo-success-bd);
		border-radius: 8px;
		padding: 4px 10px;
		font-size: 0.75rem;
		color: var(--oo-fg-tertiary);
		cursor: pointer;
		transition: background 0.15s ease, border-color 0.15s ease;
		text-align: left;
		max-width: 100%;
	}

	.correction-indicator:hover {
		background: var(--oo-bd-default);
		border-color: var(--oo-success);
	}

	.correction-indicator.expanded {
		border-color: var(--oo-success);
	}

	.correction-compact {
		display: flex;
		align-items: center;
		gap: 6px;
		flex-wrap: wrap;
	}

	.correction-icon {
		color: var(--oo-success);
		font-size: 0.8rem;
	}

	.correction-label {
		font-weight: 600;
		color: var(--oo-success);
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.correction-iterations {
		font-family: 'JetBrains Mono', 'Fira Code', monospace;
		font-size: 0.65rem;
		color: var(--oo-fg-muted);
		background: var(--oo-bg-surface);
		border-radius: 3px;
		padding: 1px 4px;
	}

	.correction-improved {
		background: var(--oo-success-bg);
		color: var(--oo-success);
		border-radius: 4px;
		padding: 1px 5px;
		font-size: 0.6rem;
		font-weight: 500;
	}

	.correction-chevron {
		font-size: 0.625rem;
		color: var(--oo-fg-muted);
		margin-left: auto;
	}

	.correction-details {
		display: flex;
		flex-direction: column;
		gap: 3px;
		border-top: 1px solid var(--oo-bd-default);
		padding-top: 4px;
		margin-top: 2px;
	}

	.correction-detail-row {
		display: flex;
		gap: 8px;
		font-size: 0.675rem;
	}

	.detail-label {
		color: var(--oo-fg-muted);
		flex-shrink: 0;
	}

	.detail-value {
		color: var(--oo-fg-secondary);
		font-family: 'JetBrains Mono', 'Fira Code', monospace;
	}

	.arrow-up {
		color: var(--oo-success);
		margin: 0 2px;
	}

	.score-improved {
		color: var(--oo-success);
		font-weight: 600;
	}

	.correction-pass {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: 0.7rem;
		color: var(--oo-fg-muted);
		padding: 2px 6px;
	}

	.pass-icon {
		color: var(--oo-fg-muted);
		font-size: 0.7rem;
	}

	.pass-label {
		font-size: 0.65rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}
</style>
