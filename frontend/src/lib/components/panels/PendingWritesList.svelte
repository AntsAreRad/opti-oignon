<!--
  PendingWritesList.svelte -- the writes waiting for the user's review.

  Each proposal says what accepting it would do, shows each of its words
  with where they came from (typed by you, or not), what the agent had read
  before proposing it and, for a change, what it changes. One checkbox per
  proposal and one for all; the two buttons act on the selection only and
  nothing here is saved until the user accepts it. The list holds no state:
  the review container (PendingWritesReview) does, and it is rendered empty
  of markup when nothing waits.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { Button, Checkbox } from '$lib/ds';
	import {
		ORIGIN_LABELS,
		allSelected,
		argumentRows,
		describeWrite,
		readBefore,
		sourceLabel,
		targetLine,
		type PendingWrite,
	} from '$lib/pendingWrites';

	/** The proposals waiting, in the order shown. */
	export let items: PendingWrite[] = [];
	/** The ids of the proposals the user selected. */
	export let selected: string[] = [];
	/** A decision is on its way: every control waits. */
	export let busy = false;

	const dispatch = createEventDispatcher<{ toggle: string; toggleAll: null; accept: null; decline: null }>();

	$: every = allSelected(items, selected);
	$: some = selected.length > 0 && !every;
</script>

{#if items.length > 0}
	<section class="oo-pending" aria-labelledby="oo-pending-title">
		<header class="oo-pending-head">
			<h3 id="oo-pending-title" class="oo-pending-title">
				To review <span class="oo-pending-count">{items.length}</span>
			</h3>
			<Checkbox
				label="Select all"
				checked={every}
				indeterminate={some}
				disabled={busy}
				on:change={() => dispatch('toggleAll', null)}
			/>
		</header>
		<p class="oo-pending-hint">Nothing here is saved until you accept it.</p>
		<ul class="oo-pending-list">
			{#each items as item (item.id)}
				<li class="oo-pending-item" data-pending-id={item.id}>
					<Checkbox
						label={describeWrite(item)}
						description={sourceLabel(item)}
						checked={selected.includes(item.id)}
						disabled={busy}
						on:change={() => dispatch('toggle', item.id)}
					/>
					<dl class="oo-pending-args">
						{#each argumentRows(item) as row}
							<div class="oo-pending-arg">
								<dt>{row.name}</dt>
								<dd>
									<span class="oo-pending-value">{row.value}</span>
									<span class="oo-pending-origin" data-origin={row.origin}>{ORIGIN_LABELS[row.origin]}</span>
								</dd>
							</div>
						{/each}
					</dl>
					{#if targetLine(item)}
						<p class="oo-pending-note">{targetLine(item)}</p>
					{/if}
					{#if readBefore(item)}
						<p class="oo-pending-note">{readBefore(item)}</p>
					{/if}
				</li>
			{/each}
		</ul>
		<div class="oo-pending-actions">
			<Button
				variant="primary"
				size="sm"
				disabled={busy || selected.length === 0}
				on:click={() => dispatch('accept', null)}
			>Accept selected</Button>
			<Button
				variant="secondary"
				size="sm"
				disabled={busy || selected.length === 0}
				on:click={() => dispatch('decline', null)}
			>Decline selected</Button>
		</div>
	</section>
{/if}

<style>
	.oo-pending {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		margin: var(--oo-space-2) var(--oo-space-3);
		padding: var(--oo-space-3);
		border-radius: var(--oo-radius-lg);
		background: var(--oo-bg-tint-1);
	}

	.oo-pending-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-2);
	}

	.oo-pending-title {
		margin: 0;
		font-size: var(--oo-text-sm);
		font-weight: 600;
		color: var(--oo-fg-primary);
	}

	.oo-pending-count {
		margin-left: var(--oo-space-1);
		font-weight: 400;
		color: var(--oo-fg-muted);
		font-variant-numeric: tabular-nums;
	}

	.oo-pending-hint {
		margin: 0;
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
	}

	.oo-pending-list {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.oo-pending-item {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
		padding: var(--oo-space-2);
		border-radius: var(--oo-radius-md);
		background: var(--oo-bg-surface);
	}

	.oo-pending-args {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
		margin: 0;
	}

	.oo-pending-arg {
		display: grid;
		grid-template-columns: auto 1fr;
		gap: var(--oo-space-2);
		font-size: var(--oo-text-xs);
	}

	.oo-pending-arg dt {
		color: var(--oo-fg-muted);
	}

	.oo-pending-arg dd {
		margin: 0;
		min-width: 0;
		color: var(--oo-fg-secondary);
		overflow-wrap: anywhere;
	}

	.oo-pending-origin {
		display: inline-block;
		margin-left: var(--oo-space-1);
		padding: 0 var(--oo-space-1);
		border-radius: var(--oo-radius-sm);
		font-size: var(--oo-text-2xs);
		color: var(--oo-fg-muted);
		background: var(--oo-bg-subtle);
	}

	/* The status ink alone marks an origin: washes belong to the primitives. */
	.oo-pending-origin[data-origin='typed'] {
		color: var(--oo-fg-success);
	}

	.oo-pending-origin[data-origin='not-typed'] {
		color: var(--oo-fg-warning);
	}

	.oo-pending-note {
		margin: 0;
		font-size: var(--oo-text-2xs);
		color: var(--oo-fg-tertiary);
	}

	.oo-pending-actions {
		display: flex;
		gap: var(--oo-space-2);
		justify-content: flex-end;
	}
</style>
