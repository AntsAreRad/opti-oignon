<!--
  NewConversationButton.svelte
  Creates a new conversation. Built on the ds Button primitive.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import Button from '$lib/ds/Button.svelte';

	const dispatch = createEventDispatcher<{ create: void }>();

	let creating = false;

	async function handleClick() {
		if (creating) return;
		creating = true;
		dispatch('create');
		// Reset after a short delay.
		setTimeout(() => {
			creating = false;
		}, 500);
	}
</script>

<Button variant="primary" block iconLeft="plus" loading={creating} on:click={handleClick}>
	New conversation
</Button>
