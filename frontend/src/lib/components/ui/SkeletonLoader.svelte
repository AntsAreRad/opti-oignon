<!--
  SkeletonLoader.svelte
  Reusable skeleton placeholder shown during component/data loading.
  Variants: card, list, text, inline.
  CSS-only shimmer animation, themed via --oo-* variables.

  Usage:
    <SkeletonLoader variant="card" />
    <SkeletonLoader variant="list" rows={5} />
    <SkeletonLoader variant="text" rows={3} />
    <SkeletonLoader variant="inline" width="120px" />
-->
<script lang="ts">
	/** Skeleton layout variant. */
	export let variant: 'card' | 'list' | 'text' | 'inline' = 'card';

	/** Number of rows for list/text variants. */
	export let rows: number = 3;

	/** Custom width for inline variant. */
	export let width: string = '100%';

	/** Custom height for inline variant. */
	export let height: string = '1rem';

	/** Accessible label for screen readers. */
	export let label: string = 'Loading content';
</script>

<div class="skeleton-wrapper" role="status" aria-label={label} aria-busy="true">
	<span class="sr-only">{label}</span>

	{#if variant === 'card'}
		<div class="skeleton-card">
			<div class="skeleton-line skeleton-shimmer" style="height: 1.25rem; width: 60%;"></div>
			<div class="skeleton-line skeleton-shimmer" style="height: 0.875rem; width: 90%; margin-top: 0.75rem;"></div>
			<div class="skeleton-line skeleton-shimmer" style="height: 0.875rem; width: 75%; margin-top: 0.5rem;"></div>
			<div class="skeleton-line skeleton-shimmer" style="height: 2rem; width: 40%; margin-top: 1rem; border-radius: 0.5rem;"></div>
		</div>
	{:else if variant === 'list'}
		<div class="skeleton-list">
			{#each Array(rows) as _, i}
				<div class="skeleton-list-item">
					<div class="skeleton-circle skeleton-shimmer"></div>
					<div class="skeleton-list-text">
						<div class="skeleton-line skeleton-shimmer" style="height: 0.875rem; width: {75 + (i % 3) * 8}%;"></div>
						<div class="skeleton-line skeleton-shimmer" style="height: 0.625rem; width: {50 + (i % 2) * 20}%; margin-top: 0.375rem;"></div>
					</div>
				</div>
			{/each}
		</div>
	{:else if variant === 'text'}
		<div class="skeleton-text">
			{#each Array(rows) as _, i}
				<div
					class="skeleton-line skeleton-shimmer"
					style="height: 0.875rem; width: {i === rows - 1 ? '60%' : (90 - (i % 3) * 10) + '%'}; margin-top: {i === 0 ? '0' : '0.5rem'};"
				></div>
			{/each}
		</div>
	{:else if variant === 'inline'}
		<div
			class="skeleton-line skeleton-shimmer"
			style="height: {height}; width: {width}; display: inline-block; vertical-align: middle;"
		></div>
	{/if}
</div>

<style>
	.skeleton-wrapper {
		width: 100%;
	}

	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border-width: 0;
	}

	.skeleton-card {
		padding: 1rem;
		border-radius: 0.75rem;
		background-color: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-subtle);
	}

	.skeleton-line {
		border-radius: 0.375rem;
		background-color: var(--oo-bg-overlay);
	}

	.skeleton-shimmer {
		position: relative;
		overflow: hidden;
	}

	.skeleton-shimmer::after {
		content: '';
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		background: linear-gradient(
			90deg,
			transparent 0%,
			var(--oo-bd-subtle) 50%,
			transparent 100%
		);
		animation: shimmer 1.5s ease-in-out infinite;
	}

	@keyframes shimmer {
		0% {
			transform: translateX(-100%);
		}
		100% {
			transform: translateX(100%);
		}
	}

	.skeleton-list {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.skeleton-list-item {
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}

	.skeleton-circle {
		width: 2rem;
		height: 2rem;
		border-radius: 50%;
		background-color: var(--oo-bg-overlay);
		flex-shrink: 0;
	}

	.skeleton-list-text {
		flex: 1;
		min-width: 0;
	}

	.skeleton-text {
		display: flex;
		flex-direction: column;
	}
</style>
