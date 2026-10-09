<!--
  PendingWritesReview.svelte -- the review section of the Memory, Notes and Skills panels.

  Loads the writes waiting for the user (every store's, or one store's),
  keeps the selection, and sends one batch decision for the selected
  proposals, an acceptance with the digest of what each one showed. After
  a decision it says how many were saved or declined, keeps any whose write
  failed or whose digest no longer named it, and tells its panel (decided)
  so the panel can reload what changed. The rendering is
  PendingWritesList's.
-->
<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { toastError, toastSuccess } from '$lib/stores/notifications';
	import { acceptPendingWrites, declinePendingWrites, listPendingWrites } from '$lib/api/pendingWrites';
	import {
		decisionDigests,
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
			const { results } =
				kind === 'accept'
					? await acceptPendingWrites(ids, decisionDigests(items, ids))
					: await declinePendingWrites(ids);
			items = remaining(items, results);
			selected = [];
			const done = results.filter((r) => r.applied || r.declined).length;
			const failed = results.filter((r) => (r.reason ?? '').startsWith('failed')).length;
			const gone = results.filter((r) => r.reason === 'target not found').length;
			const changed = results.filter((r) => r.reason === 'target changed' || r.reason === 'digest mismatch').length;
			const unnamed = results.filter((r) => (r.reason ?? '').startsWith('the digest')).length;
			if (done > 0) toastSuccess(kind === 'accept' ? `Saved ${done}` : `Declined ${done}`);
			if (failed > 0) toastError(`${failed} could not be saved; they are still waiting`);
			if (gone > 0) toastError(`${gone} no longer had anything to change; nothing was saved`);
			if (changed > 0) toastError(`${changed} changed since you were shown it; nothing was saved`);
			if (unnamed > 0) toastError(`${unnamed} no longer matched what you were shown; reload to read it again`);
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
