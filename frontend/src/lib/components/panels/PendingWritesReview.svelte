<!--
  PendingWritesReview.svelte -- the review section of the Memory and Notes panels.

  Loads the writes waiting for the user (every store's, or one store's),
  keeps the selection, and sends one batch decision for the selected
  proposals. After a decision it says how many were saved or declined,
  keeps any whose write failed, and tells its panel (decided) so the panel
  can reload what changed. The rendering is PendingWritesList's.
-->
<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { toastError, toastSuccess } from '$lib/stores/notifications';
	import { acceptPendingWrites, declinePendingWrites, listPendingWrites } from '$lib/api/pendingWrites';
	import {
		decisionIds,
		remaining,
		toggle,
		toggleAll,
		type PendingStore,
		type PendingWrite,
	} from '$lib/pendingWrites';
	import PendingWritesList from './PendingWritesList.svelte';

	/** One store's proposals, or every store's when unset. */
	export let store: PendingStore | undefined = undefined;

	const dispatch = createEventDispatcher<{ decided: null }>();

	let items: PendingWrite[] = [];
	let selected: string[] = [];
	let busy = false;

	/** Read the waiting proposals again; a selection of ones no longer shown is dropped. */
	export async function reload(): Promise<void> {
		try {
			items = await listPendingWrites(store);
		} catch {
			items = [];
		}
		selected = decisionIds(items, selected);
	}

	async function decide(kind: 'accept' | 'decline'): Promise<void> {
		const ids = decisionIds(items, selected);
		if (ids.length === 0 || busy) return;
		busy = true;
		try {
			const { results } = kind === 'accept' ? await acceptPendingWrites(ids) : await declinePendingWrites(ids);
			items = remaining(items, results);
			selected = [];
			const done = results.filter((r) => r.applied || r.declined).length;
			const failed = results.filter((r) => (r.reason ?? '').startsWith('failed')).length;
			const gone = results.filter((r) => r.reason === 'target not found').length;
			if (done > 0) toastSuccess(kind === 'accept' ? `Saved ${done}` : `Declined ${done}`);
			if (failed > 0) toastError(`${failed} could not be saved; they are still waiting`);
			if (gone > 0) toastError(`${gone} no longer had anything to change; nothing was saved`);
			dispatch('decided', null);
		} catch {
			toastError(kind === 'accept' ? 'Failed to accept' : 'Failed to decline');
		} finally {
			busy = false;
		}
	}

	onMount(reload);
</script>

<PendingWritesList
	{items}
	{selected}
	{busy}
	on:toggle={(event) => (selected = toggle(selected, event.detail))}
	on:toggleAll={() => (selected = toggleAll(items, selected))}
	on:accept={() => decide('accept')}
	on:decline={() => decide('decline')}
/>
