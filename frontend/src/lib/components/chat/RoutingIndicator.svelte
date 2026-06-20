<!--
  RoutingIndicator.svelte
  S46: Non-intrusive indicator showing why a model was selected.
  
  Display the model name, reason for selection, and pipeline used.
  Click to expand the details (alternatives, score, profile).
-->
<script lang="ts">
	export let routingReason: {
		model: string;
		display_name: string;
		task_type: string;
		pipeline: string;
		reason: string;
		score: number;
		alternatives: string[];
		profile_used: boolean;
		feedback_adjusted: boolean;
		failover: boolean;
		original_model: string;
	} | null = null;

	export let model: string = '';
	export let taskType: string = '';
	export let pipeline: string = '';

	let expanded = false;

	function toggle() {
		expanded = !expanded;
	}

	// Display name: profile display_name or raw model name
	$: displayName = routingReason?.display_name || model || 'Unknown';
	// Reason: from the profile or a generic fallback
	$: reason = routingReason?.reason || (taskType ? `Task: ${taskType}` : '');
	// Formatted score
	$: scoreText = routingReason?.score
		? `${Math.round(routingReason.score * 100)}%`
		: '';
	// Profile indicator
	$: profileBadge = routingReason?.profile_used ?? false;
	// S62: Feedback-adjusted indicator
	$: feedbackBadge = routingReason?.feedback_adjusted ?? false;
	// S63: Failover indicator
	$: failoverBadge = routingReason?.failover ?? false;
	$: originalModel = routingReason?.original_model ?? '';
	// Displayed pipeline
	$: pipelineText = routingReason?.pipeline || pipeline || '';
	// Alternatives
	$: alternatives = routingReason?.alternatives || [];
</script>

{#if model || routingReason}
	<button
		class="routing-indicator"
		class:expanded
		on:click={toggle}
		title="Click for routing details"
	>
		<!-- Compact line -->
		<div class="routing-compact">
			<span class="routing-model">{displayName}</span>
			{#if profileBadge}
				<span class="routing-badge">profile</span>
			{/if}
			{#if feedbackBadge}
				<span class="routing-badge routing-badge-feedback">feedback-adjusted</span>
			{/if}
			{#if failoverBadge}
				<span class="routing-badge routing-badge-failover" title={originalModel ? `Original: ${originalModel}` : 'Model substitution due to health'}>failover</span>
			{/if}
			{#if reason}
				<span class="routing-reason">{reason}</span>
			{/if}
			{#if pipelineText}
				<span class="routing-pipeline">{pipelineText}</span>
			{/if}
			<span class="routing-chevron">{expanded ? '\u25B4' : '\u25BE'}</span>
		</div>

		<!-- Expandable details -->
		{#if expanded}
			<div class="routing-details">
				{#if scoreText}
					<div class="routing-detail-row">
						<span class="detail-label">Match score:</span>
						<span class="detail-value">{scoreText}</span>
					</div>
				{/if}
				{#if taskType || routingReason?.task_type}
					<div class="routing-detail-row">
						<span class="detail-label">Task type:</span>
						<span class="detail-value">{routingReason?.task_type || taskType}</span>
					</div>
				{/if}
				{#if alternatives.length > 0}
					<div class="routing-detail-row">
						<span class="detail-label">Alternatives:</span>
						<span class="detail-value">{alternatives.join(', ')}</span>
					</div>
				{/if}
				{#if failoverBadge && originalModel}
					<div class="routing-detail-row">
						<span class="detail-label">Original model:</span>
						<span class="detail-value">{originalModel}</span>
					</div>
				{/if}
			</div>
		{/if}
	</button>
{/if}

<style>
	.routing-indicator {
		display: inline-flex;
		flex-direction: column;
		gap: 4px;
		background: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-default);
		border-radius: 8px;
		padding: 4px 10px;
		font-size: 0.75rem;
		color: var(--oo-fg-tertiary);
		cursor: pointer;
		transition: background 0.15s ease, border-color 0.15s ease;
		text-align: left;
		max-width: 100%;
	}

	.routing-indicator:hover {
		background: var(--oo-bd-default);
		border-color: var(--oo-bd-strong);
	}

	.routing-indicator.expanded {
		border-color: var(--oo-acc-600);
	}

	.routing-compact {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}

	.routing-model {
		font-family: 'JetBrains Mono', 'Fira Code', monospace;
		font-weight: 600;
		color: var(--oo-fg-primary);
		font-size: 0.75rem;
	}

	.routing-badge {
		background: var(--oo-acc-800);
		color: var(--oo-acc-300);
		border-radius: 4px;
		padding: 1px 5px;
		font-size: 0.625rem;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.routing-badge-feedback {
		background: var(--oo-success-bg);
		color: var(--oo-success);
	}

	.routing-badge-failover {
		background: var(--oo-warning-bg);
		color: var(--oo-warning);
	}

	.routing-reason {
		color: var(--oo-fg-tertiary);
		font-size: 0.7rem;
	}

	.routing-pipeline {
		font-family: 'JetBrains Mono', 'Fira Code', monospace;
		font-size: 0.65rem;
		color: var(--oo-fg-muted);
		background: var(--oo-bg-surface);
		border-radius: 3px;
		padding: 1px 4px;
	}

	.routing-chevron {
		font-size: 0.625rem;
		color: var(--oo-fg-muted);
		margin-left: auto;
	}

	.routing-details {
		display: flex;
		flex-direction: column;
		gap: 3px;
		border-top: 1px solid var(--oo-bd-default);
		padding-top: 4px;
		margin-top: 2px;
	}

	.routing-detail-row {
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
		word-break: break-all;
	}
</style>
