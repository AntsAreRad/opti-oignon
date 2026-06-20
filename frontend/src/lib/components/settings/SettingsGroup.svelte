<!--
  SettingsGroup.svelte (S168)
  Standard wrapper for one settings group inside a /settings section.

  Wraps a labelled block (title + optional description) on the ds Card
  primitive, exposes a deep-link anchor (`#oo-set-<id>`), and provides the
  standardized per-group "Reset to default" affordance (spec 5.9). The reset
  control is shown only when an `onReset` handler is supplied; immediate-apply
  semantics (toast on change) live in the controls the group hosts.

  Atomic groups (a set of fields that only make sense applied together) and
  multi-step ceremonies pass `variant="ceremony"` so they are visually
  distinguishable from immediate-apply controls.
-->
<script lang="ts">
	import Card from '$lib/ds/Card.svelte';
	import Button from '$lib/ds/Button.svelte';

	export let id: string;
	export let title: string;
	export let description: string | undefined = undefined;
	/** Optional reset-to-default handler; when set, a reset button is shown. */
	export let onReset: (() => void) | undefined = undefined;
	export let resetLabel = 'Reset to default';
	/** 'standard' immediate-apply group, or 'ceremony' for atomic / multi-step. */
	export let variant: 'standard' | 'ceremony' = 'standard';

	const anchor = `oo-set-${id}`;
</script>

<section id={anchor} class="oo-set-group" class:oo-set-ceremony={variant === 'ceremony'}>
	<Card padding="lg">
		<header class="oo-set-head">
			<div class="oo-set-head-text">
				<h3 class="oo-set-title">
					{title}
					{#if variant === 'ceremony'}
						<span class="oo-set-badge">ceremony</span>
					{/if}
				</h3>
				{#if description}
					<p class="oo-set-desc">{description}</p>
				{/if}
			</div>
			{#if onReset}
				<Button variant="ghost" size="sm" iconLeft="rotate-ccw" on:click={() => onReset && onReset()}>
					{resetLabel}
				</Button>
			{/if}
		</header>

		<div class="oo-set-body">
			<slot />
		</div>
	</Card>
</section>

<style>
	.oo-set-group {
		scroll-margin-top: var(--oo-space-6);
	}

	.oo-set-head {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: var(--oo-space-3);
		margin-bottom: var(--oo-space-3);
	}

	.oo-set-head-text {
		min-width: 0;
	}

	.oo-set-title {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		font-size: var(--oo-text-base);
		font-weight: 600;
		color: var(--oo-fg-primary);
		margin: 0;
	}

	.oo-set-badge {
		font-size: var(--oo-text-2xs);
		text-transform: uppercase;
		letter-spacing: var(--oo-tracking-wide);
		padding: 1px var(--oo-space-2);
		border-radius: var(--oo-radius-full);
		background-color: var(--oo-warning-bg);
		color: var(--oo-warning);
		border: 1px solid var(--oo-warning-bd);
	}

	.oo-set-desc {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
		margin: var(--oo-space-1) 0 0;
		line-height: var(--oo-leading-snug);
	}

	.oo-set-body {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}

	.oo-set-ceremony :global(.oo-card) {
		border-color: var(--oo-warning-bd);
	}
</style>
